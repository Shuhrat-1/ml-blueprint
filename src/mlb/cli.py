import argparse

import torch
from torch.utils.data import DataLoader

from mlb import __version__
from mlb.core.artifacts import create_run_dir, save_json, save_yaml
from mlb.core.config import load_config, to_dict
from mlb.core.logging import setup_logger
from mlb.core.seed import set_seed
from mlb.data.io import load_dataframe
from mlb.data.schema import Schema, resolve_columns
from mlb.data.split import split_dataframe
from mlb.eval.plots import save_plots
from mlb.models.sklearn.train import save_sklearn_artifacts, train_sklearn
from mlb.models.torch.dataset import TabularDataset, build_cat_vocabs, compute_num_stats
from mlb.models.torch.torch_train import train_torch_tabular


def main() -> None:
    parser = argparse.ArgumentParser(prog="mlb", description="ML Blueprint CLI")
    parser.add_argument("--version", action="store_true", help="Print version and exit")

    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Validate config + create run dir + log (Day3 MVP)")
    p_run.add_argument("--config", required=True, help="Path to YAML config")
    p_run.add_argument("--name", default=None, help="Optional run name override")
    p_run.add_argument(
        "--no-timestamp", action="store_true", help="Disable timestamp in run dir name"
    )
    p_split = sub.add_parser("split", help="Load data + resolve schema + create splits (Day4 MVP)")
    p_split.add_argument("--config", required=True, help="Path to YAML config")
    p_train = sub.add_parser("train", help="Train model (Day5 MVP: sklearn)")
    p_train.add_argument("--config", required=True, help="Path to YAML config")

    args = parser.parse_args()

    if args.version:
        print(__version__)
        return

    if args.cmd == "run":
        cfg = load_config(args.config)

        run_name = args.name or cfg.run.name
        artifacts = create_run_dir(name=run_name, with_timestamp=not args.no_timestamp)

        logger = setup_logger(log_file=artifacts.logs_path)
        logger.info(f"Run dir: {artifacts.run_dir}")

        # reproducibility
        set_seed(cfg.run.seed, deterministic_torch=cfg.run.deterministic_torch)
        logger.info(f"Seed set to {cfg.run.seed}")

        # save resolved config
        save_yaml(artifacts.config_path, to_dict(cfg))
        logger.info(f"Saved config: {artifacts.config_path}")

        logger.info(f"Engine={cfg.run.engine} Task={cfg.run.task}")
        logger.info("Day3 MVP run completed.")
        return

    if args.cmd == "split":
        cfg = load_config(args.config)

        artifacts = create_run_dir(name=f"{cfg.run.name}_split")
        logger = setup_logger(log_file=artifacts.logs_path)
        logger.info(f"Run dir: {artifacts.run_dir}")

        set_seed(cfg.run.seed, deterministic_torch=cfg.run.deterministic_torch)

        # load data
        df = load_dataframe(cfg.data.path, cfg.data.format)
        logger.info(f"Loaded data: shape={df.shape}")

        schema = Schema(
            target=cfg.data.target,
            id_cols=cfg.data.id_cols,
            datetime_cols=cfg.data.datetime_cols,
            numeric_cols=cfg.data.numeric_cols,
            categorical_cols=cfg.data.categorical_cols,
        )
        schema = resolve_columns(df, schema)
        logger.info(
            f"Schema resolved: num={len(schema.numeric_cols)} cat={len(schema.categorical_cols)}"
        )

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
        logger.info(
            f"Splits: train={split.train.shape} val={split.val.shape} test={split.test.shape}"
        )

        # save
        save_yaml(artifacts.config_path, to_dict(cfg))
        save_yaml(artifacts.run_dir / "schema_resolved.yaml", schema.model_dump())

        split_dir = artifacts.run_dir / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)

        split.train.to_parquet(split_dir / "train.parquet", index=False)
        split.val.to_parquet(split_dir / "val.parquet", index=False)
        split.test.to_parquet(split_dir / "test.parquet", index=False)
        logger.info(f"Saved splits to: {split_dir}")

        logger.info("Day4 MVP split completed.")
        return

    if args.cmd == "train":
        cfg = load_config(args.config)

        artifacts = create_run_dir(name=f"{cfg.run.name}_train")
        logger = setup_logger(log_file=artifacts.logs_path)
        logger.info(f"Run dir: {artifacts.run_dir}")

        set_seed(cfg.run.seed, deterministic_torch=cfg.run.deterministic_torch)

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

            save_sklearn_artifacts(artifacts.run_dir, result.pipeline)
            save_json(artifacts.metrics_path, result.metrics)
            save_yaml(artifacts.config_path, to_dict(cfg))
            save_yaml(artifacts.run_dir / "schema_resolved.yaml", schema.model_dump())

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
            logger.info("Day5 MVP train completed.")
            return

        if cfg.run.engine == "torch":
            # build stats/vocabs from TRAIN only
            num_mean, num_std = compute_num_stats(split.train, schema.numeric_cols)
            vocabs = build_cat_vocabs(split.train, schema.categorical_cols)
            cat_vocab_sizes = [vocabs[c].size for c in schema.categorical_cols]

            train_ds = TabularDataset(
                split.train,
                target=schema.target,
                numeric_cols=schema.numeric_cols,
                categorical_cols=schema.categorical_cols,
                vocabs=vocabs,
                task=cfg.run.task,
                num_mean=num_mean,
                num_std=num_std,
            )
            val_ds = TabularDataset(
                split.val,
                target=schema.target,
                numeric_cols=schema.numeric_cols,
                categorical_cols=schema.categorical_cols,
                vocabs=vocabs,
                task=cfg.run.task,
                num_mean=num_mean,
                num_std=num_std,
            )
            test_ds = TabularDataset(
                split.test,
                target=schema.target,
                numeric_cols=schema.numeric_cols,
                categorical_cols=schema.categorical_cols,
                vocabs=vocabs,
                task=cfg.run.task,
                num_mean=num_mean,
                num_std=num_std,
            )

            train_loader = DataLoader(train_ds, batch_size=cfg.torch.batch_size, shuffle=True)
            val_loader = (
                DataLoader(val_ds, batch_size=cfg.torch.batch_size, shuffle=False)
                if len(val_ds)
                else None
            )
            test_loader = DataLoader(test_ds, batch_size=cfg.torch.batch_size, shuffle=False)

            result = train_torch_tabular(
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                task=cfg.run.task,
                n_num=len(schema.numeric_cols),
                cat_vocab_sizes=cat_vocab_sizes,
                hidden_dims=cfg.torch.hidden_dims,
                dropout=cfg.torch.dropout,
                lr=cfg.torch.lr,
                weight_decay=cfg.torch.weight_decay,
                max_epochs=cfg.torch.max_epochs,
                early_stopping=cfg.torch.early_stopping,
                patience=cfg.torch.patience,
                min_delta=cfg.torch.min_delta,
                scheduler=cfg.torch.scheduler,
                step_size=cfg.torch.step_size,
                gamma=cfg.torch.gamma,
                run_dir=artifacts.run_dir,
            )

            # save model + training artifacts
            torch.save(result.model.state_dict(), artifacts.run_dir / "model_state.pt")
            save_json(artifacts.metrics_path, result.metrics)
            save_yaml(artifacts.config_path, to_dict(cfg))
            save_yaml(artifacts.run_dir / "schema_resolved.yaml", schema.model_dump())
            save_json(
                artifacts.run_dir / "torch_vocabs.json", {k: v.mapping for k, v in vocabs.items()}
            )
            save_json(
                artifacts.run_dir / "torch_num_stats.json",
                {"mean": num_mean.tolist(), "std": num_std.tolist()},
            )

            logger.info(f"Metrics: {result.metrics}")
            logger.info("Day6 Torch train completed.")
            return

        raise ValueError(f"Unknown engine: {cfg.run.engine}")

    parser.print_help()
