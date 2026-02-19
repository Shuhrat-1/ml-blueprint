from pathlib import Path

from mlb.core.paths import Paths


def test_detect_root_finds_pyproject() -> None:
    root = Paths.detect_root(Path.cwd())
    assert (root / "pyproject.toml").exists()


def test_paths_from_env_defaults() -> None:
    p = Paths.from_env()
    assert p.root.exists()
    assert str(p.data_dir).endswith("data")
    assert str(p.artifacts_dir).endswith("artifacts")
