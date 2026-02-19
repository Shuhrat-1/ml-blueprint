from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class EarlyStopping:
    patience: int = 5
    min_delta: float = 0.0

    best: float | None = None
    bad_epochs: int = 0

    def step(self, current: float) -> bool:
        """
        Returns True if should stop.
        Minimizing metric (loss).
        """
        if self.best is None or current < self.best - self.min_delta:
            self.best = current
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_loss: float) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_loss": best_loss,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict(),
        },
        path,
    )
