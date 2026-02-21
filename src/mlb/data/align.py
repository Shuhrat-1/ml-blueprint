# src/mlb/data/align.py
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

MISSING_CAT = "__MISSING__"


@dataclass
class AlignmentReport:
    n_rows: int
    dropped_extra_cols: list[str]
    added_missing_cols: list[str]
    reordered: bool
    coerced_numeric_cols: list[str]
    coerced_datetime_cols: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ensure_cols(df: pd.DataFrame, cols: list[str], fill_value) -> list[str]:
    added = []
    for c in cols:
        if c not in df.columns:
            df[c] = fill_value
            added.append(c)
    return added


def align_frame(
    df: pd.DataFrame,
    contract: dict[str, Any],
    *,
    mode: str,  # "train" | "predict"
) -> tuple[pd.DataFrame, AlignmentReport]:
    """
    Align df to the feature contract:
    - drop extra columns
    - add missing columns with defaults
    - coerce types
    - enforce final column order (feature_order)
    """
    features = contract["features"]
    feature_order = contract["feature_order"]

    numeric = features.get("numeric", [])
    categorical = features.get("categorical", [])
    text = features.get("text", [])
    datetime_cols = features.get("datetime", [])

    # Columns that are allowed to exist besides features (id/time/target)
    allowed_non_features = set()
    for k in ("target", "time_col"):
        v = contract.get(k)
        if v:
            allowed_non_features.add(v)
    allowed_non_features.update(contract.get("id_cols", []))

    extra_cols = sorted([c for c in df.columns if c not in set(feature_order) and c not in allowed_non_features])

    # Drop extras (keep only allowed_non_features + features)
    keep = list(allowed_non_features) + list(feature_order)
    keep = [c for c in keep if c in df.columns]  # preserve existing ones
    df2 = df.loc[:, keep].copy()

    # Add missing feature columns
    added = []
    added += _ensure_cols(df2, numeric, pd.NA)
    added += _ensure_cols(df2, categorical, MISSING_CAT)
    added += _ensure_cols(df2, text, "")
    added += _ensure_cols(df2, datetime_cols, pd.NaT)

    coerced_num, coerced_dt = [], []

    # Coerce numeric
    for c in numeric:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
            coerced_num.append(c)

    # Coerce datetime
    for c in datetime_cols:
        if c in df2.columns:
            df2[c] = pd.to_datetime(df2[c], errors="coerce")
            coerced_dt.append(c)

    # Coerce categorical/text to string (stable)
    for c in categorical + text:
        if c in df2.columns:
            df2[c] = df2[c].astype("string")

    # Enforce final order
    before_cols = list(df2.columns)
    # remove non-feature cols for X
    X = df2.loc[:, feature_order].copy()
    reordered = before_cols != list(df2.columns) or list(X.columns) != feature_order

    report = AlignmentReport(
        n_rows=len(df2),
        dropped_extra_cols=extra_cols,
        added_missing_cols=sorted(set(added)),
        reordered=reordered,
        coerced_numeric_cols=coerced_num,
        coerced_datetime_cols=coerced_dt,
        notes=[],
    )

    # Minimal safety checks
    if mode == "predict" and len(X.columns) != len(feature_order):
        report.notes.append("Feature order mismatch after alignment (unexpected).")

    return X, report