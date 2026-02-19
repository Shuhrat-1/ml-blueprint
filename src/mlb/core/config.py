from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., description="Path to dataset (csv/parquet)")
    format: Literal["csv", "parquet"] = "csv"
    target: str = Field(..., description="Target column name")
    id_cols: list[str] = Field(default_factory=list, description="ID columns to exclude")
    datetime_cols: list[str] = Field(default_factory=list, description="Datetime columns")
    numeric_cols: list[str] = Field(default_factory=list, description="Numeric feature columns")
    categorical_cols: list[str] = Field(
        default_factory=list, description="Categorical feature columns"
    )


class SplitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["random", "time"] = "random"
    test_size: float = Field(0.2, ge=0.0, le=0.9)
    val_size: float = Field(0.0, ge=0.0, le=0.9)
    stratify: bool = True
    time_col: str | None = None
    random_state: int = 42


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "run"
    engine: Literal["sklearn", "torch"] = "sklearn"
    task: Literal["classification", "regression"] = "classification"
    seed: int = 42
    deterministic_torch: bool = False


class AppConfig(BaseModel):
    """
    Top-level config.
    Keep this stable: other modules depend on it.
    """

    model_config = ConfigDict(extra="forbid")

    run: RunConfig
    data: DataConfig
    split: SplitConfig = SplitConfig()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("Config YAML must define a mapping at the top level.")
    return data


def load_config(path: str | Path) -> AppConfig:
    p = Path(path).expanduser().resolve()
    raw = load_yaml(p)
    return AppConfig.model_validate(raw)


def to_dict(cfg: BaseModel) -> dict:
    # stable serialization for saving config_resolved.yaml
    return cfg.model_dump(mode="python")
