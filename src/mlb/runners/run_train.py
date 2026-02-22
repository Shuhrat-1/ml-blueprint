from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import DataLoader

from mlb.core.artifacts import create_run_dir, save_json, save_yaml

# --- imports: адаптируй пути, если у тебя они в других модулях ---
from mlb.core.config import load_config, to_dict
from mlb.core.logging import setup_logger
from mlb.core.seed import set_seed
from mlb.data.io import load_dataframe
from mlb.data.schema import Schema, resolve_columns
from mlb.data.split import split_dataframe
from mlb.eval.plots import save_plots
from mlb.models.sklearn.train import save_sklearn_artifacts, train_sklearn
from mlb.models.torch.dataset import (
    TabularDataset,
    build_cat_vocabs,
    cat_vocab_sizes_from_vocabs,
    compute_num_stats,
)
from mlb.models.torch.export import save_torch_bundle

# torch engine (если у тебя другие имена — поправь тут)
from mlb.models.torch.torch_train import train_torch_tabular


@dataclass(frozen=True)
class TrainRunResult:
    run_dir: Path
    metrics: dict[str, float]
    engine: str


def run_train(*, config_path: str | Path) -> TrainRunResult:
    """
    Runner for `mlb train`.
    Contains all business logic that used to live in CLI.
    """
    cfg = load_config(str(config_path))

    artifacts = create_run_dir(name=f"{cfg.run.name}_train")
    logger = setup_logger(log_file=artifacts.logs_path)
    logger.info(f"Run dir: {artifacts.run_dir}")

    set_seed(cfg.run.seed, deterministic_torch=cfg.run.deterministic_torch)

    # --- load + schema + split (общая часть)
    df = load_dataframe(cfg.data.path, cfg.data.format)
    schema = Schema(
        target=cfg.data.target,
        id_cols=cfg.data.id_cols,
        datetime_cols=cfg.data.datetime_cols,
        numeric_cols=cfg.data.numeric_cols,
        categorical_cols=cfg.data.categorical_cols,
    )
    schema = resolve_columns(df, schema)

    split = split_dataframe(
        df=df,
        target=schema.target,
        method=cfg.split.method,
        test_size=cfg.split.test_size,
        val_size=cfg.split.val_size,
        stratify=cfg.split.stratify if cfg.run.task == "classification" else False,
        time_col=cfg.split.time_col,
        random_state=cfg.split.random_state,
    )

    # --- ENGINE ROUTING
    if cfg.run.engine == "sklearn":
        result = train_sklearn(
            train_df=split.train,
            val_df=split.val,
            test_df=split.test,
            target=schema.target,
            numeric_cols=schema.numeric_cols,
            categorical_cols=schema.categorical_cols,
            task=cfg.run.task,
            model_name=cfg.model.name,
            model_params=cfg.model.params,
        )

        # ✅ schema_resolved.yaml = feature contract (НОВЫЙ формат)
        contract: dict[str, Any] = {
            "schema_version": 1,
            "features": {
                "numeric": schema.numeric_cols,
                "categorical": schema.categorical_cols,
                "text": [],  # пока пусто
                "datetime": schema.datetime_cols,
            },
            "feature_order": [*schema.numeric_cols, *schema.categorical_cols, *schema.datetime_cols],
            "target": schema.target,
            "time_col": cfg.split.time_col,
            "id_cols": schema.id_cols,
        }

        save_yaml(artifacts.run_dir / "schema_resolved.yaml", contract)

        # (опционально) если хочешь сохранить legacy schema отдельно:
        save_yaml(artifacts.run_dir / "schema_legacy.yaml", schema.model_dump())

        save_sklearn_artifacts(artifacts.run_dir, result.pipeline)
        save_json(artifacts.metrics_path, result.metrics)
        save_yaml(artifacts.config_path, to_dict(cfg))

        # plots
        X_test = split.test[[*schema.numeric_cols, *schema.categorical_cols]]
        y_test = split.test[schema.target]
        y_pred = result.pipeline.predict(X_test)
        y_proba = None
        if cfg.run.task == "classification" and hasattr(result.pipeline, "predict_proba"):
            y_proba = result.pipeline.predict_proba(X_test)

        save_plots(
            run_dir=artifacts.run_dir,
            task=cfg.run.task,
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
        )

        logger.info(f"Metrics: {result.metrics}")
        logger.info("Train completed (sklearn).")
        return TrainRunResult(run_dir=artifacts.run_dir, metrics=result.metrics, engine="sklearn")

    if cfg.run.engine == "torch":
        # ----------------------------
        # 1) Build torch preprocessing artifacts (TRAIN ONLY)
        # ----------------------------
        vocabs = build_cat_vocabs(split.train, schema.categorical_cols)

        # numeric mean/std on TRAIN ONLY
        num_mean, num_std = compute_num_stats(split.train, schema.numeric_cols)
        num_mean = np.array(num_mean, dtype=np.float32)
        num_std = np.array(num_std, dtype=np.float32)

        # ✅ Single source of truth for embedding sizes
        cat_vocab_sizes = cat_vocab_sizes_from_vocabs(
            vocabs=vocabs,
            categorical_cols=schema.categorical_cols,
            strict=True,
        )

        # ----------------------------
        # 2) Datasets
        # ----------------------------
        train_ds = TabularDataset(
            df=split.train,
            target=schema.target,
            numeric_cols=schema.numeric_cols,
            categorical_cols=schema.categorical_cols,
            vocabs=vocabs,
            num_mean=num_mean,
            num_std=num_std,
            task=cfg.run.task,
        )

        val_ds = None
        if split.val is not None and len(split.val) > 0:
            val_ds = TabularDataset(
                df=split.val,
                target=schema.target,
                numeric_cols=schema.numeric_cols,
                categorical_cols=schema.categorical_cols,
                vocabs=vocabs,
                num_mean=num_mean,
                num_std=num_std,
                task=cfg.run.task,
            )

        test_ds = TabularDataset(
            df=split.test,
            target=schema.target,
            numeric_cols=schema.numeric_cols,
            categorical_cols=schema.categorical_cols,
            vocabs=vocabs,
            num_mean=num_mean,
            num_std=num_std,
            task=cfg.run.task,
        )

        # ----------------------------
        # 3) DataLoaders (config-driven)
        # ----------------------------
        torch_cfg = cfg.torch
        batch_size = int(getattr(torch_cfg, "batch_size", 512))
        num_workers = int(getattr(torch_cfg, "num_workers", 0))

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = None if val_ds is None else DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # ----------------------------
        # 4) Train (engine-level API)
        # ----------------------------
        result = train_torch_tabular(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            task=cfg.run.task,
            n_num=len(schema.numeric_cols),
            cat_vocab_sizes=cat_vocab_sizes,
            hidden_dims=list(getattr(torch_cfg, "hidden_dims", [256, 128])),
            dropout=float(getattr(torch_cfg, "dropout", 0.1)),
            lr=float(getattr(torch_cfg, "lr", 1e-3)),
            weight_decay=float(getattr(torch_cfg, "weight_decay", 0.0)),
            max_epochs=int(getattr(torch_cfg, "max_epochs", 10)),
            early_stopping=bool(getattr(torch_cfg, "early_stopping", True)),
            patience=int(getattr(torch_cfg, "patience", 3)),
            min_delta=float(getattr(torch_cfg, "min_delta", 0.0)),
            scheduler=str(getattr(torch_cfg, "scheduler", "none")),
            step_size=int(getattr(torch_cfg, "step_size", 5)),
            gamma=float(getattr(torch_cfg, "gamma", 0.5)),
            run_dir=artifacts.run_dir,
            device=getattr(torch_cfg, "device", None),
        )

        # ----------------------------
        # 5) Save contract + bundle
        # ----------------------------
        contract: dict[str, Any] = {
            "schema_version": 1,
            "features": {
                "numeric": schema.numeric_cols,
                "categorical": schema.categorical_cols,
                "text": [],
                "datetime": schema.datetime_cols,
            },
            "feature_order": [*schema.numeric_cols, *schema.categorical_cols, *schema.datetime_cols],
            "target": schema.target,
            "time_col": cfg.split.time_col,
            "id_cols": schema.id_cols,
        }
        save_yaml(artifacts.run_dir / "schema_resolved.yaml", contract)
        save_yaml(artifacts.run_dir / "schema_legacy.yaml", schema.model_dump())
        save_yaml(artifacts.config_path, to_dict(cfg))
        save_json(artifacts.metrics_path, result.metrics)

        # torch bundle files: model_state.pt + vocabs + num_stats + torch_bundle.yaml
        vocabs_json = {k: v.mapping for k, v in vocabs.items()}
        save_torch_bundle(
            run_dir=artifacts.run_dir,
            model=result.model,
            vocabs=vocabs_json,
            num_mean=num_mean,
            num_std=num_std,
            schema=contract,   # IMPORTANT: contract is the bundle schema for signature checks
            config=to_dict(cfg),
        )

        logger.info(f"Metrics: {result.metrics}")
        logger.info("Train completed (torch).")
        return TrainRunResult(run_dir=artifacts.run_dir, metrics=result.metrics, engine="torch")

    raise ValueError(f"Unknown engine: {cfg.run.engine}")