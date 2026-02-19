from __future__ import annotations

import torch.nn as nn


def make_loss(task: str) -> nn.Module:
    if task == "classification":
        return nn.CrossEntropyLoss()
    return nn.MSELoss()
