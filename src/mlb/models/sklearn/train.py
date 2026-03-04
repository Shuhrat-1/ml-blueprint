from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from mlb.eval.metrics import compute_metrics
from mlb.models.sklearn.pipeline import build_preprocessor, split_xy


@dataclass(frozen=True)
class TrainResult:
    model: Any
    pipeline: Pipeline
    metrics: dict[str, float]


def _default_params(task: Literal["classification", "regression"], name: str) -> dict:
    """
    Default hyperparameters for sklearn estimators.
    Config params override these defaults.
    """
    if name == "logreg":
        if task != "classification":
            raise ValueError("logreg is only valid for classification.")
        return {
            "max_iter": 2000,
        }

    return {}


def _make_estimator(task: Literal["classification", "regression"], name: str, params: dict) -> Any:
    defaults = _default_params(task, name)

    # config params override defaults
    merged = {**defaults, **params}

    if name == "logreg":
        return LogisticRegression(**merged)

    if name == "rf":
        if task == "classification":
            return RandomForestClassifier(**params)
        return RandomForestRegressor(**params)

    # gradient boosting
    if name in ("gb", "gbr", "gbrt"):
        if task == "classification":
            return GradientBoostingClassifier(**params)
        return GradientBoostingRegressor(**params)

    raise ValueError(f"Unknown model name: {name}")


def train_sklearn(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    target: str,
    numeric_cols: list[str],
    categorical_cols: list[str],
    task: Literal["classification", "regression"],
    model_name: str,
    model_params: dict,
) -> TrainResult:
    prep = build_preprocessor(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    estimator = _make_estimator(task=task, name=model_name, params=model_params)

    pipe = Pipeline(
        steps=[
            ("preprocess", prep.preprocessor),
            ("model", estimator),
        ]
    )

    X_train, y_train = split_xy(train_df, target=target, feature_cols=prep.feature_cols)
    pipe.fit(X_train, y_train)

    # evaluate on test (MVP). Later: use val for tuning/early stopping.
    X_test, y_test = split_xy(test_df, target=target, feature_cols=prep.feature_cols)

    preds = pipe.predict(X_test)
    proba = None
    if task == "classification" and hasattr(pipe, "predict_proba"):
        proba = pipe.predict_proba(X_test)

    metrics = compute_metrics(task=task, y_true=y_test, y_pred=preds, y_proba=proba)
    return TrainResult(model=estimator, pipeline=pipe, metrics=metrics)


def save_sklearn_artifacts(run_dir: Path, pipeline: Pipeline) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, run_dir / "pipeline.joblib")
