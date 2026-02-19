from __future__ import annotations

from typing import Any

import numpy as np


def torch_metrics_classification(logits: Any, y_true: Any) -> dict[str, float]:
    # logits: [N, C]
    import torch

    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    preds = probs.argmax(axis=1)
    y = y_true.detach().cpu().numpy()

    acc = float((preds == y).mean())
    return {"accuracy": acc}


def torch_metrics_regression(y_pred: Any, y_true: Any) -> dict[str, float]:

    yp = y_pred.detach().cpu().numpy().reshape(-1)
    yt = y_true.detach().cpu().numpy().reshape(-1)

    rmse = float(np.sqrt(np.mean((yp - yt) ** 2)))
    mae = float(np.mean(np.abs(yp - yt)))
    return {"rmse": rmse, "mae": mae}
