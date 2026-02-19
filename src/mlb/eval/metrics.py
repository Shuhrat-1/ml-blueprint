from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def compute_metrics(
    *,
    task: Literal["classification", "regression"],
    y_true,
    y_pred,
    y_proba=None,
) -> dict[str, float]:
    if task == "classification":
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        out: dict[str, float] = {
            "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        }

        # f1 only if at least 2 classes in predictions
        try:
            out["f1"] = float(
                f1_score(
                    y_true_arr,
                    y_pred_arr,
                    average="binary" if len(set(y_true_arr)) == 2 else "macro",
                )
            )
        except Exception:
            pass

        unique_classes = np.unique(y_true_arr)

        # ROC & log_loss require >= 2 classes
        if len(unique_classes) >= 2 and y_proba is not None:
            try:
                if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                    out["roc_auc"] = float(roc_auc_score(y_true_arr, y_proba[:, 1]))
                    out["log_loss"] = float(log_loss(y_true_arr, y_proba))
                else:
                    out["roc_auc_ovr"] = float(
                        roc_auc_score(y_true_arr, y_proba, multi_class="ovr")
                    )
                    out["log_loss"] = float(log_loss(y_true_arr, y_proba))
            except Exception:
                pass

        return out

    # regression
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    rmse = mean_squared_error(y_true_arr, y_pred_arr, squared=False)
    return {
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
    }
