from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    """
    Centralized project paths.
    - root: repository root (detected by walking up to pyproject.toml)
    - data_dir: default <root>/data (override via MLB_DATA_DIR)
    - artifacts_dir: default <root>/artifacts (override via MLB_ARTIFACTS_DIR)
    """

    root: Path
    data_dir: Path
    artifacts_dir: Path

    @staticmethod
    def detect_root(start: Path | None = None) -> Path:
        cur = (start or Path.cwd()).resolve()
        for p in [cur, *cur.parents]:
            if (p / "pyproject.toml").exists():
                return p
        # Fallback: current dir
        return cur

    @classmethod
    def from_env(cls, start: Path | None = None) -> Paths:
        root = cls.detect_root(start)
        data_dir = Path(os.environ.get("MLB_DATA_DIR", root / "data")).resolve()
        artifacts_dir = Path(os.environ.get("MLB_ARTIFACTS_DIR", root / "artifacts")).resolve()
        return cls(root=root, data_dir=data_dir, artifacts_dir=artifacts_dir)

    def ensure(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
