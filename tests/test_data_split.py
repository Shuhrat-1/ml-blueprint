import pandas as pd
import pytest

from mlb.data.split import split_dataframe


def test_split_random_shapes() -> None:
    df = pd.DataFrame({"x": range(100), "y": [0, 1] * 50})
    res = split_dataframe(df, target="y", method="random", test_size=0.2, val_size=0.1, stratify=True)
    assert len(res.train) + len(res.val) + len(res.test) == 100
    assert len(res.test) > 0


def test_split_time_requires_time_col() -> None:
    df = pd.DataFrame({"t": range(10), "y": [0, 1] * 5})
    with pytest.raises(ValueError):
        split_dataframe(df, target="y", method="time", test_size=0.2, val_size=0.0, time_col=None)


def test_split_stratify_tiny_fallback() -> None:
    df = pd.DataFrame({"x": range(5), "y": [0, 1, 0, 1, 0]})
    res = split_dataframe(df, target="y", method="random", test_size=0.2, val_size=0.0, stratify=True)
    assert len(res.train) + len(res.test) == 5