from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class PreprocessArtifacts:
    preprocessor: ColumnTransformer
    feature_cols: list[str]


def build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> PreprocessArtifacts:
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("cat", cat_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    feature_cols = [*numeric_cols, *categorical_cols]
    return PreprocessArtifacts(preprocessor=preprocessor, feature_cols=feature_cols)


def split_xy(df: pd.DataFrame, target: str, feature_cols: list[str]):
    X = df[feature_cols]
    y = df[target]
    return X, y
