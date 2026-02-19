from pathlib import Path

import pytest

from mlb.core.config import load_config


def test_load_config_ok(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
run:
  name: "t"
  engine: "sklearn"
  task: "classification"
  seed: 1
  deterministic_torch: false
data:
  path: "x.csv"
  format: "csv"
  target: "y"
  id_cols: []
  datetime_cols: []
  numeric_cols: []
  categorical_cols: []
split:
  method: "random"
  test_size: 0.2
  val_size: 0.0
  stratify: true
  random_state: 42
""".strip(),
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)
    assert cfg.run.name == "t"
    assert cfg.data.target == "y"


def test_extra_keys_forbidden(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
run:
  name: "t"
  engine: "sklearn"
  task: "classification"
  seed: 1
  deterministic_torch: false
  extra: "nope"
data:
  path: "x.csv"
  format: "csv"
  target: "y"
  id_cols: []
  datetime_cols: []
  numeric_cols: []
  categorical_cols: []
""".strip(),
        encoding="utf-8",
    )

    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        load_config(cfg_path)
