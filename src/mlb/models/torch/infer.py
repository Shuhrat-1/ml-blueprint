from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from mlb.core.artifacts import load_yaml
from mlb.models.torch.dataset import (
    MISSING_ID,
    UNK_ID,
    CatVocab,
    TabularDataset,
    cat_vocab_sizes_from_vocabs,
)
from mlb.models.torch.signature import compute_feature_signature
from mlb.models.torch.tabular_model import TabularMLP


def _load_vocabs(vocabs_json: dict[str, dict[str, int]]) -> dict[str, CatVocab]:
    return {k: CatVocab(mapping=v) for k, v in vocabs_json.items()}


def _assert_bundle_compat(
    *,
    meta: dict,
    vocabs: dict[str, CatVocab],
    num_stats: dict,
    categorical_cols: list[str],
    cat_vocab_sizes_ckpt: list[int],
) -> None:
    if meta.get("bundle_version") != 1:
        raise ValueError(f"Unsupported torch bundle_version={meta.get('bundle_version')} (expected 1)")

    # special ids policy
    special_ids = meta.get("special_ids", {})
    if special_ids.get("unk") != UNK_ID or special_ids.get("missing") != MISSING_ID:
        raise ValueError(
            f"torch bundle special_ids mismatch. Expected {{unk:{UNK_ID}, missing:{MISSING_ID}}}, got {special_ids}"
        )

    # signature check
    schema_resolved = meta.get("schema", {})
    expected_sig = meta.get("feature_signature")
    if not expected_sig:
        raise ValueError("torch bundle missing feature_signature (retrain with Day 9 bundle format).")

    actual_sig = compute_feature_signature(
        schema_resolved=schema_resolved,
        vocabs=vocabs,
        num_stats=num_stats,
    )
    if actual_sig != expected_sig:
        raise ValueError(
            "Torch bundle feature_signature mismatch. "
            "This usually means schema/vocabs/num_stats do not match the trained run_dir."
        )

    # vocab sizes vs checkpoint embedding sizes (order matters!)
    cat_vocab_sizes_from_json = cat_vocab_sizes_from_vocabs(
    vocabs=vocabs,
    categorical_cols=categorical_cols,
    strict=True,
)
    

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

    # Load bundle meta (torch_bundle.yaml) from same directory as state_path
    meta_path = state_path.parent / "torch_bundle.yaml"
    meta = load_yaml(meta_path)

    model, device, cat_vocab_sizes_ckpt = load_model_for_infer(
        state_path=state_path,
        n_num=len(numeric_cols),
        hidden_dims=hidden_dims,
        dropout=dropout,
    )
    num_stats_dict = {"mean": num_mean.tolist(), "std": num_std.tolist()}
    _assert_bundle_compat(
        meta=meta,
        vocabs=vocabs,
        num_stats=num_stats_dict,
        categorical_cols=categorical_cols,
        cat_vocab_sizes_ckpt=cat_vocab_sizes_ckpt,
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
