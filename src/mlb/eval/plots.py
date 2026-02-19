from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import RocCurveDisplay


def save_plots(
    *,
    run_dir: Path,
    task: Literal["classification", "regression"],
    y_true,
    y_pred,
    y_proba=None,
) -> None:
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if task == "classification" and y_proba is not None:
        # binary only for MVP ROC curve
        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
            RocCurveDisplay.from_predictions(y_true, y_proba[:, 1])
            plt.title("ROC Curve")
            plt.savefig(plots_dir / "roc_curve.png", bbox_inches="tight")
            plt.close()

    if task == "regression":
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)

        plt.figure()
        plt.scatter(y_true_arr, y_pred_arr)
        plt.title("Predicted vs True")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.savefig(plots_dir / "pred_vs_true.png", bbox_inches="tight")
        plt.close()
