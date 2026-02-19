from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader

from mlb.models.torch.callbacks import EarlyStopping, save_checkpoint
from mlb.models.torch.losses import make_loss
from mlb.models.torch.tabular_model import TabularMLP
from mlb.models.torch.torch_metrics import torch_metrics_classification, torch_metrics_regression


@dataclass(frozen=True)
class TorchTrainResult:
    model: TabularMLP
    metrics: dict[str, float]
    best_val_loss: float


def _make_scheduler(optimizer, kind: str, step_size: int, gamma: float):
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if kind == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=gamma, patience=2
        )
    return None


def train_torch_tabular(
    *,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    test_loader: DataLoader,
    task: Literal["classification", "regression"],
    n_num: int,
    cat_vocab_sizes: list[int],
    hidden_dims: list[int],
    dropout: float,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    early_stopping: bool,
    patience: int,
    min_delta: float,
    scheduler: str,
    step_size: int,
    gamma: float,
    run_dir: Path,
    device: str | None = None,
) -> TorchTrainResult:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # out dim
    if task == "classification":
        # infer number of classes from train dataset targets (assumes labels are 0..C-1)
        ys = []
        for _, _, y in train_loader:
            ys.append(y)
        n_classes = int(torch.cat(ys).max().item()) + 1
        out_dim = n_classes
    else:
        out_dim = 1

    model = TabularMLP(
        n_num=n_num,
        cat_vocab_sizes=cat_vocab_sizes,
        hidden_dims=hidden_dims,
        dropout=dropout,
        out_dim=out_dim,
    ).to(device)

    loss_fn = make_loss(task)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = _make_scheduler(optimizer, scheduler, step_size, gamma)

    stopper = EarlyStopping(patience=patience, min_delta=min_delta) if early_stopping else None
    best_val = float("inf")
    ckpt_path = run_dir / "model_best.pt"

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []

        for x_num, x_cat, y in train_loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            if task == "classification":
                logits = model(x_num, x_cat)  # [B, C]
                loss = loss_fn(logits, y)
            else:
                pred = model(x_num, x_cat).squeeze(1)  # [B]
                loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().cpu().item())

        # validation
        val_loss = None
        if val_loader is not None and len(val_loader.dataset) > 0:
            model.eval()
            losses = []
            with torch.no_grad():
                for x_num, x_cat, y in val_loader:
                    x_num = x_num.to(device)
                    x_cat = x_cat.to(device)
                    y = y.to(device)

                    if task == "classification":
                        logits = model(x_num, x_cat)
                        loss = loss_fn(logits, y)
                    else:
                        pred = model(x_num, x_cat).squeeze(1)
                        loss = loss_fn(pred, y)
                    losses.append(loss.detach().cpu().item())
            val_loss = float(np.mean(losses))

            # scheduler step
            if sched is not None:
                if scheduler == "plateau":
                    sched.step(val_loss)
                else:
                    sched.step()

            # checkpoint best
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(ckpt_path, model, optimizer, epoch=epoch, best_loss=best_val)

            # early stopping
            if stopper is not None and stopper.step(val_loss):
                break
        else:
            # no val loader: step scheduler by epoch if step scheduler
            if sched is not None and scheduler == "step":
                sched.step()

    # load best checkpoint if exists
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state"])
        best_val = float(state.get("best_loss", best_val))

    # test metrics
    model.eval()
    all_logits = []
    all_y = []
    all_pred = []
    with torch.no_grad():
        for x_num, x_cat, y in test_loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)

            if task == "classification":
                logits = model(x_num, x_cat)
                all_logits.append(logits)
                all_y.append(y)
            else:
                pred = model(x_num, x_cat).squeeze(1)
                all_pred.append(pred)
                all_y.append(y)

    if task == "classification":
        logits = torch.cat(all_logits, dim=0)
        y_true = torch.cat(all_y, dim=0)
        metrics = torch_metrics_classification(logits, y_true)
    else:
        y_pred = torch.cat(all_pred, dim=0)
        y_true = torch.cat(all_y, dim=0)
        metrics = torch_metrics_regression(y_pred, y_true)

    return TorchTrainResult(model=model, metrics=metrics, best_val_loss=best_val)
