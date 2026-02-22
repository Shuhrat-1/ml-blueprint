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
    base_dir_name = "_".join(parts) if parts else "run"

    runs_root = (paths.artifacts_dir / "runs").resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    # Ensure uniqueness even if started within the same second
    run_dir = (runs_root / base_dir_name).resolve()
    counter = 1
    while run_dir.exists():
        run_dir = (runs_root / f"{base_dir_name}_{counter}").resolve()
        counter += 1

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


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load YAML file into dict.

    Notes:
    - Returns {} if file is empty.
    - Raises FileNotFoundError if path does not exist.
    - Raises ValueError with readable message if YAML is invalid or not a mapping.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"YAML file not found: {p}")

    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return {}

    try:
        obj = yaml.safe_load(text)
    except Exception as e:  # pragma: no cover
        raise ValueError(f"Failed to parse YAML: {p} ({e})") from e

    if obj is None:
        return {}

    if not isinstance(obj, dict):
        raise ValueError(f"YAML root must be a mapping/dict: {p} (got {type(obj).__name__})")

    return obj


def _yaml_sanitize(obj: Any) -> Any:
    """
    Convert objects that PyYAML SafeDumper can't represent into plain Python types.

    Handles:
    - Path -> str
    - numpy scalars -> .item()
    - string-like subclasses (e.g., torch.torch_version.TorchVersion) -> str(obj)
    - dict/list/tuple recursion
    """
    if obj is None:
        return None

    # Paths
    if isinstance(obj, Path):
        return str(obj)

    # Recurse containers
    if isinstance(obj, dict):
        return {str(_yaml_sanitize(k)): _yaml_sanitize(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        return [_yaml_sanitize(x) for x in obj]

    # Numpy scalars (if numpy installed/used)
    # Avoid hard dependency import; use duck-typing
    if hasattr(obj, "item") and callable(obj.item):
        try:
            v = obj.item()
            # If still not plain after item(), continue sanitizing
            return _yaml_sanitize(v)
        except Exception:
            pass

    # Scalars (cast string subclasses to real str)
    if isinstance(obj, str):
        return str(obj)
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)

    # Fallback: stringify unknown objects (keeps YAML dump stable)
    return str(obj)


def save_yaml(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    safe_obj = _yaml_sanitize(obj)

    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(safe_obj, f, sort_keys=False)


