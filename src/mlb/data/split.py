from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class SplitResult:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def split_dataframe(
    df: pd.DataFrame,
    target: str,
    method: Literal["random", "time"] = "random",
    test_size: float = 0.2,
    val_size: float = 0.0,
    stratify: bool = True,
    time_col: str | None = None,
    random_state: int = 42,
) -> SplitResult:
    if target not in df.columns:
        raise ValueError(f"Target column not in df: {target}")

    if not (0.0 <= test_size < 1.0) or not (0.0 <= val_size < 1.0):
        raise ValueError("test_size and val_size must be in [0, 1).")
    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size must be < 1.0")

    if method == "time":
        if not time_col:
            raise ValueError("time_col is required for time split.")
        if time_col not in df.columns:
            raise ValueError(f"time_col not in df: {time_col}")
        df_sorted = df.sort_values(time_col)
        n = len(df_sorted)
        n_test = int(round(n * test_size))
        n_val = int(round(n * val_size))
        n_train = n - n_test - n_val
        if n_train <= 0:
            raise ValueError("Not enough data for train split with given sizes.")
        train = df_sorted.iloc[:n_train]
        val = df_sorted.iloc[n_train : n_train + n_val] if n_val > 0 else df_sorted.iloc[0:0]
        test = df_sorted.iloc[n_train + n_val :]
        return SplitResult(train=train, val=val, test=test)

    # random split
    y = df[target]
    strat = y if stratify else None

    # 1) split off test
    try:
        train_full, test = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=strat
        )
    except ValueError:
        if stratify:
            train_full, test = train_test_split(
                df, test_size=test_size, random_state=random_state, stratify=None
            )
        else:
            raise

    # 2) split train_full into train/val if requested
    if val_size > 0:
        # val fraction relative to remaining after test split
        val_rel = val_size / (1.0 - test_size)

        y_train_full = train_full[target]
        strat2 = y_train_full if stratify else None

        try:
            train, val = train_test_split(
                train_full, test_size=val_rel, random_state=random_state, stratify=strat2
            )
        except ValueError:
            if stratify:
                train, val = train_test_split(
                    train_full, test_size=val_rel, random_state=random_state, stratify=None
                )
            else:
                raise
    else:
        train = train_full
        val = train_full.iloc[0:0]

    return SplitResult(train=train, val=val, test=test)
