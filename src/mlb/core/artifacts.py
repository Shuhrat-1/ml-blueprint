from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .paths import Paths


@dataclass(frozen=True)
class RunArtifacts:
    run_dir: Path
    config_path: Path
    metrics_path: Path
    logs_path: Path

    @property
    def plots_dir(self) -> Path:
        p = self.run_dir / "plots"
        p.mkdir(parents=True, exist_ok=True)
        return p


def _safe_name(name: str) -> str:
    # keep it simple & filesystem-friendly
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name).strip("_")


def create_run_dir(
    paths: Paths | None = None,
    name: str = "run",
    prefix: str = "",
    with_timestamp: bool = True,
) -> RunArtifacts:
    paths = paths or Paths.from_env()
    paths.ensure()

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") if with_timestamp else ""
    parts = [p for p in [prefix, stamp, _safe_name(name)] if p]
    dir_name = "_".join(parts) if parts else "run"

    run_dir = (paths.artifacts_dir / "runs" / dir_name).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)

    config_path = run_dir / "config_resolved.yaml"
    metrics_path = run_dir / "metrics.json"
    logs_path = run_dir / "logs.jsonl"

    return RunArtifacts(
        run_dir=run_dir,
        config_path=config_path,
        metrics_path=metrics_path,
        logs_path=logs_path,
    )


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def save_yaml(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)
