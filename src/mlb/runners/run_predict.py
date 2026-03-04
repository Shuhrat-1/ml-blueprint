from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from mlb.core.paths import Paths
from mlb.data.align import align_features, align_frame
from mlb.models.torch.infer import predict_torch_tabular


@dataclass(frozen=True)
class PredictRunResult:
    run_dir: Path
    out_path: Path
    engine: str
    task: str


def _resolve_run_dir(*, run_dir: str | None, latest: bool) -> Path:
    if not run_dir and not latest:
        raise ValueError("Either --run-dir or --latest must be provided.")
    if run_dir and latest:
        raise ValueError("Use either --run-dir or --latest, not both.")

    if latest:
        runs_root = (Paths.from_env().artifacts_dir / "runs").resolve()
        if not runs_root.exists():
            raise FileNotFoundError(f"No runs directory found: {runs_root}")

        candidates: list[Path] = []
        for p in runs_root.iterdir():
            if p.is_dir() and (p / "config_resolved.yaml").exists():
                candidates.append(p)

        if not candidates:
            raise FileNotFoundError(f"No valid run directories found in: {runs_root}")

        return max(candidates, key=lambda p: p.stat().st_mtime).resolve()

    return Path(run_dir).expanduser().resolve()


def _load_input_df(inp: Path) -> pd.DataFrame:
    fmt = "parquet" if inp.suffix.lower() == ".parquet" else "csv"
    return pd.read_parquet(inp) if fmt == "parquet" else pd.read_csv(inp)


def run_predict(
    *,
    run_dir: str | None,
    latest: bool,
    input_path: str | Path,
    output_path: str | Path | None,
) -> PredictRunResult:
    """
    Runner for `mlb predict`.
    Mirrors previous CLI behavior (Day 9 state).
    """
    rd = _resolve_run_dir(run_dir=run_dir, latest=latest)
    if not rd.exists():
        raise FileNotFoundError(f"run_dir not found: {rd}")

    cfg_path = rd / "config_resolved.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config_resolved.yaml in run_dir: {rd}")

    cfg_dict = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    engine = cfg_dict["run"]["engine"]
    task = cfg_dict["run"]["task"]

    # load input
    inp = Path(input_path).expanduser().resolve()
    df = _load_input_df(inp)

    schema_path = rd / "schema_resolved.yaml"
    schema = yaml.safe_load(schema_path.read_text(encoding="utf-8")) if schema_path.exists() else None

    out_path = (
        Path(output_path).expanduser().resolve()
        if output_path
        else (rd / "predictions.csv")
    )

    if engine == "sklearn":
        pipe_path = rd / "pipeline.joblib"
        pipe = joblib.load(pipe_path)

        if not schema:
            raise FileNotFoundError("schema_resolved.yaml required for predict (feature contract).")

        # contract-based align (supports both legacy and new schema formats)
        X, _rep = align_frame(df, schema, mode="predict")

        pred = pipe.predict(X)

        out = pd.DataFrame({"pred": pred})
        if task == "classification" and hasattr(pipe, "predict_proba"):
            proba = pipe.predict_proba(X)
            if proba.shape[1] == 2:
                out["proba_1"] = proba[:, 1]

        out.to_csv(out_path, index=False)
        return PredictRunResult(run_dir=rd, out_path=out_path, engine=engine, task=task)

    if engine == "torch":
        if not schema:
            raise FileNotFoundError("schema_resolved.yaml required for torch predict.")

        vocabs = json.loads((rd / "torch_vocabs.json").read_text(encoding="utf-8"))
        num_stats = json.loads((rd / "torch_num_stats.json").read_text(encoding="utf-8"))
        num_mean = np.array(num_stats["mean"], dtype=np.float32)
        num_std = np.array(num_stats["std"], dtype=np.float32)

        state_path = rd / "model_state.pt"

        # Resolve feature lists (supports both new and legacy schema_resolved.yaml)
        features = schema.get("features") or {}
        numeric_cols = features.get("numeric") or schema.get("numeric_cols") or []
        categorical_cols = features.get("categorical") or schema.get("categorical_cols") or []
        target = schema.get("target")

        # Align ONLY features used by torch model (num + cat)
        df_aligned, _rep = align_features(
            df,
            numeric_cols=list(numeric_cols),
            categorical_cols=list(categorical_cols),
            datetime_cols=[],
            strict=False,
        )

        torch_cfg = cfg_dict.get("torch", {}) or {}
        hidden_dims = torch_cfg.get("hidden_dims", [256, 128])
        dropout = torch_cfg.get("dropout", 0.1)
        batch_size = torch_cfg.get("batch_size", 512)

        out = predict_torch_tabular(
            df=df_aligned,
            target=target,
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
        return PredictRunResult(run_dir=rd, out_path=out_path, engine=engine, task=task)

    raise ValueError(f"Unknown engine in run dir config: {engine}")