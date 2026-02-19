from __future__ import annotations

from typing import Any, Literal

from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression


def make_estimator(
    task: Literal["classification", "regression"],
    name: str,
    params: dict,
) -> Any:
    if name == "logreg":
        if task != "classification":
            raise ValueError("logreg is only valid for classification.")
        return LogisticRegression(max_iter=2000, **params)

    if name == "rf":
        return RandomForestClassifier(**params) if task == "classification" else RandomForestRegressor(**params)

    if name in ("gb", "gbr", "gbrt"):
        return (
            GradientBoostingClassifier(**params)
            if task == "classification"
            else GradientBoostingRegressor(**params)
        )

    raise ValueError(f"Unknown model name: {name}")