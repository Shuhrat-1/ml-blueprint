from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mlb.models.torch.dataset import CatVocab, TabularDataset
from mlb.models.torch.tabular_model import TabularMLP


def _load_vocabs(vocabs_json: dict[str, dict[str, int]]) -> dict[str, CatVocab]:
    return {k: CatVocab(mapping=v) for k, v in vocabs_json.items()}


def load_model_for_infer(
    *,
    state_path: Path,
    n_num: int,
    hidden_dims: list[int],
    dropout: float,
    device: str | None = None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    state = torch.load(state_path, map_location=device)

    # infer embeddings shapes from checkpoint
    emb_keys = sorted([k for k in state.keys() if k.startswith("embeddings.") and k.endswith(".weight")])
    cat_vocab_sizes = []
    cat_embed_dims = []
    for k in emb_keys:
        w = state[k]
        cat_vocab_sizes.append(int(w.shape[0]))
        cat_embed_dims.append(int(w.shape[1]))

    # infer out_dim from final layer weight
    # last Linear is mlp.<last>.weight
    last_linear_w = None
    for k in sorted(state.keys()):
        if k.startswith("mlp.") and k.endswith(".weight"):
            last_linear_w = state[k]
    if last_linear_w is None:
        raise ValueError("Cannot infer out_dim from checkpoint.")
    out_dim = int(last_linear_w.shape[0])

    model = TabularMLP(
        n_num=n_num,
        cat_vocab_sizes=cat_vocab_sizes,
        cat_embed_dims=cat_embed_dims,
        hidden_dims=hidden_dims,
        dropout=dropout,
        out_dim=out_dim,
    ).to(device)

    model.load_state_dict(state)
    model.eval()
    return model, device, cat_vocab_sizes


def predict_torch_tabular(
    *,
    df: pd.DataFrame,
    target: str | None,
    numeric_cols: list[str],
    categorical_cols: list[str],
    vocabs_json: dict[str, dict[str, int]],
    num_mean: np.ndarray,
    num_std: np.ndarray,
    state_path: Path,
    task: Literal["classification", "regression"],
    hidden_dims: list[int],
    dropout: float,
    batch_size: int = 512,
) -> pd.DataFrame:
    vocabs = _load_vocabs(vocabs_json)

    model, device, cat_vocab_sizes_ckpt = load_model_for_infer(
        state_path=state_path,
        n_num=len(numeric_cols),
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    cat_vocab_sizes_from_json = [vocabs[c].size for c in categorical_cols]
    if cat_vocab_sizes_from_json != cat_vocab_sizes_ckpt:
        raise ValueError(
            "Vocab sizes from vocabs.json do not match checkpoint embeddings. "
            f"from_json={cat_vocab_sizes_from_json}, from_ckpt={cat_vocab_sizes_ckpt}. "
            "Likely wrong run_dir or schema/columns mismatch."
    )
    # create dataset; if target missing we create dummy column
    work = df.copy()
    if target is None or target not in work.columns:
        work["_dummy_target"] = 0
        target_col = "_dummy_target"
    else:
        target_col = target

    ds = TabularDataset(
        work,
        target=target_col,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        vocabs=vocabs,
        task="classification" if task == "classification" else "regression",
        num_mean=num_mean,
        num_std=num_std,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    preds = []
    prob1 = []

    with torch.no_grad():
        for x_num, x_cat, _ in loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)

            if task == "classification":
                logits = model(x_num, x_cat)
                probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
                pred = probs.argmax(axis=1)
                preds.append(pred)
                # binary convenience
                if probs.shape[1] == 2:
                    prob1.append(probs[:, 1])
            else:
                yhat = model(x_num, x_cat).squeeze(1).detach().cpu().numpy()
                preds.append(yhat)

    pred_arr = np.concatenate(preds, axis=0)
    out = pd.DataFrame({"pred": pred_arr})

    if task == "classification" and prob1:
        out["proba_1"] = np.concatenate(prob1, axis=0)

    return out
