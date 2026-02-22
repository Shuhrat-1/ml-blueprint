import argparse
import json

import joblib
import numpy as np
import pandas as pd

from mlb import __version__
from mlb.core.artifacts import create_run_dir, save_yaml
from mlb.core.config import load_config, to_dict
from mlb.core.logging import setup_logger
from mlb.core.seed import set_seed
from mlb.data.align import align_features, align_frame
from mlb.data.io import load_dataframe
from mlb.data.schema import Schema, resolve_columns
from mlb.data.split import split_dataframe
from mlb.models.torch.infer import predict_torch_tabular
from mlb.runners.run_train import run_train


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

    p_pred = sub.add_parser("predict", help="Predict from a saved run dir (sklearn/torch)")
    p_pred.add_argument("--run-dir", help="Path to artifacts run directory")
    p_pred.add_argument("--latest", action="store_true", help="Use latest run directory")
    p_pred.add_argument("--input", required=True, help="Path to input csv/parquet")
    p_pred.add_argument("--output", default=None, help="Optional output path")

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
        res = run_train(config_path=args.config)
        # res может быть dataclass или просто dict — делаем вывод максимально совместимым
        run_dir = getattr(res, "run_dir", None) or getattr(res, "artifacts", None) or None
        metrics = getattr(res, "metrics", None)

        print(f"[INFO] Train completed. run_dir={run_dir}")
        if metrics is not None:
            print(f"[INFO] Metrics: {metrics}")
        return


    if args.cmd == "predict":
        from pathlib import Path

        import yaml

        from mlb.core.paths import Paths

        if not args.run_dir and not args.latest:
            raise ValueError("Either --run-dir or --latest must be provided.")

        if args.run_dir and args.latest:
            raise ValueError("Use either --run-dir or --latest, not both.")

        if args.latest:
            runs_root = (Paths.from_env().artifacts_dir / "runs").resolve()
            if not runs_root.exists():
                raise FileNotFoundError(f"No runs directory found: {runs_root}")

            # pick latest directory that actually looks like a run (has config_resolved.yaml)
            candidates = []
            for p in runs_root.iterdir():
                if p.is_dir() and (p / "config_resolved.yaml").exists():
                    candidates.append(p)

            if not candidates:
                raise FileNotFoundError(f"No valid run directories found in: {runs_root}")

            run_dir = max(candidates, key=lambda p: p.stat().st_mtime).resolve()
        else:
            run_dir = Path(args.run_dir).expanduser().resolve()

        if not run_dir.exists():
            raise FileNotFoundError(f"run_dir not found: {run_dir}")

        cfg_path = run_dir / "config_resolved.yaml"
        if not cfg_path.exists():
            raise FileNotFoundError(f"Missing config_resolved.yaml in run_dir: {run_dir}")

        cfg_dict = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        engine = cfg_dict["run"]["engine"]
        task = cfg_dict["run"]["task"]

        # load input
        inp = Path(args.input).expanduser().resolve()
        fmt = "parquet" if inp.suffix.lower() == ".parquet" else "csv"
        df = pd.read_parquet(inp) if fmt == "parquet" else pd.read_csv(inp)

        schema_path = run_dir / "schema_resolved.yaml"
        schema = (
            yaml.safe_load(schema_path.read_text(encoding="utf-8"))
            if schema_path.exists()
            else None
        )

        out_path = (
            Path(args.output).expanduser().resolve()
            if args.output
            else (run_dir / "predictions.csv")
        )

        if engine == "sklearn":
            pipe_path = run_dir / "pipeline.joblib"
            pipe = joblib.load(pipe_path)

            # features from schema if available; else use all except target
            if not schema:
                raise FileNotFoundError("schema_resolved.yaml required for predict (feature contract).")

            numeric_cols = schema.get("numeric_cols", [])
            categorical_cols = schema.get("categorical_cols", [])
            datetime_cols = schema.get("datetime_cols", [])

            X, rep = align_frame(df, schema, mode="predict")
            pred = pipe.predict(X)
            # можно логировать rep.added/rep.dropped если хочешь

            out = pd.DataFrame({"pred": pred})
            if task == "classification" and hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(X)
                if proba.shape[1] == 2:
                    out["proba_1"] = proba[:, 1]

            out.to_csv(out_path, index=False)
            print(f"Saved predictions to: {out_path}")
            return

        if engine == "torch":
            if not schema:
                raise FileNotFoundError("schema_resolved.yaml required for torch predict.")

            # load torch artifacts
            vocabs = json.loads((run_dir / "torch_vocabs.json").read_text(encoding="utf-8"))
            num_stats = json.loads((run_dir / "torch_num_stats.json").read_text(encoding="utf-8"))
            num_mean = np.array(num_stats["mean"], dtype=np.float32)
            num_std = np.array(num_stats["std"], dtype=np.float32)

            state_path = run_dir / "model_state.pt"

            # ---- Resolve feature lists (supports both new and legacy schema_resolved.yaml) ----
            features = schema.get("features") or {}
            numeric_cols = features.get("numeric") or schema.get("numeric_cols") or []
            categorical_cols = features.get("categorical") or schema.get("categorical_cols") or []
            target = schema.get("target")

            # ---- Align ONLY features used by torch model (num + cat) ----
            # torch currently does not consume datetime/text; keep it explicit
            df_aligned, rep = align_features(
                df,
                numeric_cols=list(numeric_cols),
                categorical_cols=list(categorical_cols),
                datetime_cols=[],
                strict=False,
            )

            # (optional but useful) log alignment report
            # print(f"[INFO] Align added={getattr(rep, 'added_missing_cols', getattr(rep, 'added', []))} "
            #       f"dropped={getattr(rep, 'dropped_extra_cols', getattr(rep, 'dropped', []))}")

            # ---- Model hyperparams from resolved config ----
            torch_cfg = cfg_dict.get("torch", {}) or {}
            hidden_dims = torch_cfg.get("hidden_dims", [256, 128])
            dropout = torch_cfg.get("dropout", 0.1)
            batch_size = torch_cfg.get("batch_size", 512)

            out = predict_torch_tabular(
                df=df_aligned,  # IMPORTANT: use aligned df
                target=target,  # if missing in input, infer() will create dummy
                numeric_cols=list(numeric_cols),
                categorical_cols=list(categorical_cols),
                vocabs_json=vocabs,
                num_mean=num_mean,
                num_std=num_std,
                state_path=state_path,
                task=task,
                hidden_dims=hidden_dims,
                dropout=dropout,
                batch_size=batch_size,
            )

            out.to_csv(out_path, index=False)
            print(f"Saved predictions to: {out_path}")
            return

        raise ValueError(f"Unknown engine in run dir config: {engine}")
    parser.print_help()
