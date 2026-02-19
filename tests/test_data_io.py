import pandas as pd

from mlb.data.io import load_dataframe


def test_load_dataframe_csv(tmp_path) -> None:
    p = tmp_path / "x.csv"
    pd.DataFrame({"a": [1, 2]}).to_csv(p, index=False)

    df = load_dataframe(p, "csv")
    assert df.shape == (2, 1)