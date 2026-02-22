from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

UNK_ID = 0
MISSING_ID = 1

@dataclass(frozen=True)
class CatVocab:
    """
    Maps category string -> integer id.

    Policy:
    - 0 = UNK (unknown category)
    - 1 = MISSING (None/NaN/empty)
    - known categories start from 2
    """

    mapping: dict[str, int]

    @property
    def size(self) -> int:
        # +2 for UNK and MISSING ids
        return len(self.mapping) + 2

    def encode(self, value: Any) -> int:
        # Missing
        if value is None:
            return MISSING_ID
        if isinstance(value, float) and np.isnan(value):
            return MISSING_ID

        s = str(value)
        if s == "" or s.lower() == "nan":
            return MISSING_ID

        # Unknown -> UNK
        return self.mapping.get(s, UNK_ID)


def build_cat_vocabs(df: pd.DataFrame, cat_cols: list[str]) -> dict[str, CatVocab]:
    """
    Build vocabs with stable reserved ids:
    - UNK_ID=0, MISSING_ID=1
    - known categories start at 2
    """
    vocabs: dict[str, CatVocab] = {}
    for c in cat_cols:
        # Keep original values; missing handled by encode()
        vals = df[c].astype("string")
        # Most frequent first; stable enough for MVP
        uniq = vals.dropna().value_counts().index.tolist()

        # start from 2
        mapping = {str(v): i + 2 for i, v in enumerate(uniq)}
        vocabs[c] = CatVocab(mapping=mapping)

    return vocabs


def cat_vocab_sizes_from_vocabs(
    *,
    vocabs: dict[str, CatVocab],
    categorical_cols: list[str],
    strict: bool = True,
) -> list[int]:
    """
    Single source of truth for embedding vocab sizes.

    Uses CatVocab.size which already encodes the policy:
    - UNK_ID = 0
    - MISSING_ID = 1
    - known categories start at 2
    Therefore: size = len(mapping) + 2

    strict=True -> raise if a categorical col is missing in vocabs
    strict=False -> treat missing vocab as size=2 (only UNK+MISSING)
    """
    sizes: list[int] = []
    for c in categorical_cols:
        v = vocabs.get(c)
        if v is None:
            if strict:
                raise KeyError(f"Missing vocab for categorical col '{c}'. Available: {sorted(vocabs.keys())}")
            sizes.append(2)  # UNK + MISSING
        else:
            sizes.append(int(v.size))
    return sizes


@dataclass(frozen=True)
class TabularTensors:
    x_num: torch.Tensor  # float32 [N, n_num]
    x_cat: torch.Tensor  # int64 [N, n_cat]
    y: torch.Tensor      # float32 [N] or int64 [N]


class TabularDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        target: str,
        numeric_cols: list[str],
        categorical_cols: list[str],
        vocabs: dict[str, CatVocab],
        task: str,
        num_mean: np.ndarray,
        num_std: np.ndarray,
    ) -> None:
        self.target = target
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.vocabs = vocabs
        self.task = task

        # numeric -> standardize
        if numeric_cols:
            x_num_np = df[numeric_cols].to_numpy(dtype=np.float32)
            x_num_np = (x_num_np - num_mean) / num_std
        else:
            x_num_np = np.zeros((len(df), 0), dtype=np.float32)

        # categorical -> ids
        if categorical_cols:
            x_cat_np = np.zeros((len(df), len(categorical_cols)), dtype=np.int64)
            for j, c in enumerate(categorical_cols):
                vocab = vocabs[c]
                col = df[c].to_numpy()
                x_cat_np[:, j] = np.vectorize(vocab.encode)(col)
        else:
            x_cat_np = np.zeros((len(df), 0), dtype=np.int64)

        # target
        if task == "classification":
            y_np = df[target].to_numpy(dtype=np.int64)
        else:
            y_np = df[target].to_numpy(dtype=np.float32)

        self.tensors = TabularTensors(
            x_num=torch.from_numpy(x_num_np),
            x_cat=torch.from_numpy(x_cat_np),
            y=torch.from_numpy(y_np),
        )

    def __len__(self) -> int:
        return self.tensors.y.shape[0]

    def __getitem__(self, idx: int):
        return (
            self.tensors.x_num[idx],
            self.tensors.x_cat[idx],
            self.tensors.y[idx],
        )


def compute_num_stats(df: pd.DataFrame, numeric_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    if not numeric_cols:
        return np.zeros((0,), dtype=np.float32), np.ones((0,), dtype=np.float32)

    x = df[numeric_cols].to_numpy(dtype=np.float32)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean.astype(np.float32), std.astype(np.float32)