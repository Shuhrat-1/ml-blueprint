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


def _normalize_contract(contract: dict) -> dict:
    """
    Backward compatible normalization:
    - New format:
        {"features": {"numeric": [...], "categorical": [...], "text": [...], "datetime": [...]},
         "feature_order": [...]}
    - Legacy schema_resolved.yaml format:
        {"numeric_cols": [...], "categorical_cols": [...], "text_cols": [...], "datetime_cols": [...], ...}
      optionally may contain:
        {"feature_order": [...]} or {"features": {...}} partially.

    Returns a dict guaranteed to have:
      - "features" with keys: numeric/categorical/text/datetime
      - "feature_order"
    Preserves other keys (target/id_cols/time_col etc.) as-is in the same dict.
    """
    if not isinstance(contract, dict):
        raise TypeError(f"contract must be a dict, got {type(contract).__name__}")

    # If already new format, just ensure all keys exist
    if "features" in contract and isinstance(contract.get("features"), dict):
        feats = contract["features"]
        numeric = feats.get("numeric", []) or []
        categorical = feats.get("categorical", []) or []
        text = feats.get("text", []) or []
        datetime_cols = feats.get("datetime", []) or []

        feature_order = contract.get("feature_order")
        if not feature_order:
            feature_order = [*numeric, *categorical, *text, *datetime_cols]

        out = dict(contract)
        out["features"] = {
            "numeric": list(numeric),
            "categorical": list(categorical),
            "text": list(text),
            "datetime": list(datetime_cols),
        }
        out["feature_order"] = list(feature_order)
        return out

    # Legacy format
    numeric = contract.get("numeric_cols", []) or []
    categorical = contract.get("categorical_cols", []) or []
    text = contract.get("text_cols", []) or []
    datetime_cols = contract.get("datetime_cols", []) or []

    feature_order = contract.get("feature_order")
    if not feature_order:
        feature_order = [*numeric, *categorical, *text, *datetime_cols]

    out = dict(contract)
    out["features"] = {
        "numeric": list(numeric),
        "categorical": list(categorical),
        "text": list(text),
        "datetime": list(datetime_cols),
    }
    out["feature_order"] = list(feature_order)
    return out


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
    contract = _normalize_contract(contract)
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


def align_features(
    df: pd.DataFrame,
    *,
    numeric_cols: list[str],
    categorical_cols: list[str],
    datetime_cols: list[str] | None = None,
    text_cols: list[str] | None = None,
    strict: bool = False,
) -> tuple[pd.DataFrame, Any]:
    """
    Backward-compatible wrapper around align_frame().

    Keeps old API used by CLI/train/predict while the framework moves to
    contract-based align_frame(df, contract, ...).

    Returns:
      X_aligned (only feature columns, in fixed order)
      report (AlignmentReport from align_frame; treated as 'Any' for compatibility)
    """
    datetime_cols = datetime_cols or []
    text_cols = text_cols or []

    contract = {
        "features": {
            "numeric": numeric_cols,
            "categorical": categorical_cols,
            "text": text_cols,
            "datetime": datetime_cols,
        },
        "feature_order": [*numeric_cols, *categorical_cols, *text_cols, *datetime_cols],
    }

    # align_frame is the contract-first API
    X, rep = align_frame(df, contract, mode="predict")

    if strict and getattr(rep, "added_missing_cols", []):
        raise ValueError(f"Missing required columns: {rep.added_missing_cols}")

    return X, rep