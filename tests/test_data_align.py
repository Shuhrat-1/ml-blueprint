import pandas as pd

from mlb.data.align import align_frame


def test_align_adds_missing_and_orders() -> None:
    df = pd.DataFrame({"income": [10, 20], "city": ["A", "B"]})

    contract = {
        "features": {"numeric": ["age", "income"], "categorical": ["city", "segment"], "text": [], "datetime": []},
        "feature_order": ["age", "income", "city", "segment"],
        "target": "target",
        "id_cols": [],
    }

    X, rep = align_frame(df, contract, mode="predict")
    assert list(X.columns) == ["age", "income", "city", "segment"]
    assert "age" in rep.added_missing_cols
    assert "segment" in rep.added_missing_cols


def test_align_drops_extra_cols() -> None:
    df = pd.DataFrame({"age": [1], "income": [2], "city": ["A"], "extra": [999]})

    contract = {
        "features": {"numeric": ["age", "income"], "categorical": ["city"], "text": [], "datetime": []},
        "feature_order": ["age", "income", "city"],
        "target": "target",
        "id_cols": [],
    }

    X, rep = align_frame(df, contract, mode="predict")
    assert "extra" in rep.dropped_extra_cols
    assert list(X.columns) == ["age", "income", "city"]