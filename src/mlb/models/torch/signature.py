from __future__ import annotations

import hashlib
import json
from typing import Any


def _stable_json(obj: Any) -> str:
    """
    Stable JSON serialization for hashing.
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


def compute_feature_signature(
    *,
    schema_resolved: dict[str, Any],
    vocabs: dict[str, CatVocab] | None = None,
    torch_vocabs: dict[str, CatVocab] | None = None,
    num_stats: dict[str, Any] | None = None,
    torch_num_stats: dict[str, Any] | None = None,
) -> str:
    """
    Compute a stable signature for Torch bundle compatibility.

    Backward/forward compatible API:
    - accepts `vocabs=` or `torch_vocabs=` (either one)
    - accepts `num_stats=` or `torch_num_stats=` (either one)

    Signature includes:
    - schema_resolved: features + feature_order (+ legacy cols lists)
    - vocab sizes per categorical column
    - num_stats keys (not raw float values)
    """
    # Normalize aliases
    if vocabs is None:
        vocabs = torch_vocabs or {}
    if num_stats is None:
        num_stats = torch_num_stats or {}

    features_part = {
        "schema_version": schema_resolved.get("schema_version"),
        "target": schema_resolved.get("target"),
        "id_cols": schema_resolved.get("id_cols", []),
        "time_col": schema_resolved.get("time_col"),
        "features": schema_resolved.get("features", {}),
        "feature_order": schema_resolved.get("feature_order", []),
        # legacy/alt formats (safe to include)
        "numeric_cols": schema_resolved.get("numeric_cols", []),
        "categorical_cols": schema_resolved.get("categorical_cols", []),
        "datetime_cols": schema_resolved.get("datetime_cols", []),
    }

    vocab_sizes = {col: vocabs[col].size for col in sorted(vocabs.keys())}

    payload = {
        "features_part": features_part,
        "vocab_sizes": vocab_sizes,
        "num_stats_keys": sorted(list(num_stats.keys())),
    }

    return hashlib.sha256(_stable_json(payload).encode("utf-8")).hexdigest()