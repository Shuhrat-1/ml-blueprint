import pandas as pd
from mlb.models.sklearn.train import train_sklearn

from mlb.data.schema import Schema, resolve_columns
from mlb.data.split import split_dataframe


def test_train_sklearn_smoke() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8],
            "age": [25, 30, 40, 35, 28, 45, 50, 33],
            "income": [50, 60, 80, 75, 52, 90, 100, 65],
            "city": ["A", "B", "A", "C", "B", "C", "A", "B"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    schema = resolve_columns(
        df,
        Schema(
            target="target",
            id_cols=["id"],
            numeric_cols=["age", "income"],
            categorical_cols=["city"],
        ),
    )

    split = split_dataframe(df, target="target", test_size=0.25, val_size=0.0, stratify=True)
    res = train_sklearn(
        train_df=split.train,
        val_df=split.val,
        test_df=split.test,
        target="target",
        numeric_cols=schema.numeric_cols,
        categorical_cols=schema.categorical_cols,
        task="classification",
        model_name="gb",
        model_params={"n_estimators": 10},
    )

    assert "accuracy" in res.metrics