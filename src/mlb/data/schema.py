from __future__ import annotations

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    target: str
    id_cols: list[str] = Field(default_factory=list)
    datetime_cols: list[str] = Field(default_factory=list)
    numeric_cols: list[str] = Field(default_factory=list)
    categorical_cols: list[str] = Field(default_factory=list)

    def feature_cols(self) -> list[str]:
        excluded = set(self.id_cols + self.datetime_cols + [self.target])
        feats = [*self.numeric_cols, *self.categorical_cols]
        return [c for c in feats if c not in excluded]


def resolve_columns(df: pd.DataFrame, schema: Schema) -> Schema:
    """
    If numeric_cols/categorical_cols empty -> infer.
    Keeps provided cols as-is but filters out missing columns with clear errors.
    """
    missing = [
        c for c in [schema.target, *schema.id_cols, *schema.datetime_cols] if c not in df.columns
    ]
    if missing:
        raise ValueError(f"Schema refers to missing columns: {missing}")

    if schema.numeric_cols or schema.categorical_cols:
        # validate provided feature columns exist
        feats = schema.numeric_cols + schema.categorical_cols
        missing_feats = [c for c in feats if c not in df.columns]
        if missing_feats:
            raise ValueError(f"Schema feature columns missing in data: {missing_feats}")
        return schema

    # infer features (MVP)
    excluded = set(schema.id_cols + schema.datetime_cols + [schema.target])
    candidates = [c for c in df.columns if c not in excluded]

    numeric = []
    categorical = []
    for c in candidates:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
        else:
            categorical.append(c)

    return Schema(
        target=schema.target,
        id_cols=schema.id_cols,
        datetime_cols=schema.datetime_cols,
        numeric_cols=numeric,
        categorical_cols=categorical,
    )
