import argparse

from mlb import __version__
from mlb.core.artifacts import create_run_dir, save_yaml
from mlb.core.config import load_config, to_dict
from mlb.core.logging import setup_logger
from mlb.core.seed import set_seed


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

        from mlb.data.io import load_dataframe
        from mlb.data.schema import Schema, resolve_columns
        from mlb.data.split import split_dataframe

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
    
    parser.print_help()
