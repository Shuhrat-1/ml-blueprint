import pandas as pd
from torch.utils.data import DataLoader

from mlb.data.schema import Schema, resolve_columns
from mlb.data.split import split_dataframe
from mlb.models.torch.dataset import TabularDataset, build_cat_vocabs, compute_num_stats
from mlb.models.torch.torch_train import train_torch_tabular


def test_torch_train_smoke() -> None:
    df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "age": [25, 30, 40, 35, 28, 45, 50, 33, 29, 41],
            "income": [50, 60, 80, 75, 52, 90, 100, 65, 55, 82],
            "city": ["A", "B", "A", "C", "B", "C", "A", "B", "C", "A"],
            "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
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
    split = split_dataframe(df, target="target", test_size=0.2, val_size=0.2, stratify=False)

    num_mean, num_std = compute_num_stats(split.train, schema.numeric_cols)
    vocabs = build_cat_vocabs(split.train, schema.categorical_cols)

    train_ds = TabularDataset(
        split.train,
        target="target",
        numeric_cols=schema.numeric_cols,
        categorical_cols=schema.categorical_cols,
        vocabs=vocabs,
        task="classification",
        num_mean=num_mean,
        num_std=num_std,
    )
    val_ds = TabularDataset(
        split.val,
        target="target",
        numeric_cols=schema.numeric_cols,
        categorical_cols=schema.categorical_cols,
        vocabs=vocabs,
        task="classification",
        num_mean=num_mean,
        num_std=num_std,
    )
    test_ds = TabularDataset(
        split.test,
        target="target",
        numeric_cols=schema.numeric_cols,
        categorical_cols=schema.categorical_cols,
        vocabs=vocabs,
        task="classification",
        num_mean=num_mean,
        num_std=num_std,
    )

    res = train_torch_tabular(
        train_loader=DataLoader(train_ds, batch_size=4, shuffle=True),
        val_loader=DataLoader(val_ds, batch_size=4, shuffle=False),
        test_loader=DataLoader(test_ds, batch_size=4, shuffle=False),
        task="classification",
        n_num=len(schema.numeric_cols),
        cat_vocab_sizes=[vocabs[c].size for c in schema.categorical_cols],
        hidden_dims=[32],
        dropout=0.1,
        lr=1e-2,
        weight_decay=0.0,
        max_epochs=5,
        early_stopping=True,
        patience=2,
        min_delta=0.0,
        scheduler="plateau",
        step_size=10,
        gamma=0.5,
        run_dir=pd.Path(".") if False else __import__("pathlib").Path("."),  # simple Path
    )

    assert "accuracy" in res.metrics
