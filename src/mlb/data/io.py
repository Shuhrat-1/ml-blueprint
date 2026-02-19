from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd


def load_dataframe(path: str | Path, fmt: Literal["csv", "parquet"] = "csv") -> pd.DataFrame:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Data file not found: {p}")

    if fmt == "csv":
        return pd.read_csv(p)
    if fmt == "parquet":
        return pd.read_parquet(p)

    raise ValueError(f"Unsupported format: {fmt}")