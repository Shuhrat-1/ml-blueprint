from pathlib import Path

import numpy as np
import pandas as pd

from mlb.models.torch.export import save_torch_bundle
from mlb.models.torch.infer import predict_torch_tabular
from mlb.models.torch.tabular_model import TabularMLP


def test_torch_predict_smoke(tmp_path: Path) -> None:
    # minimal model with 1 cat col, 2 num cols, binary out_dim=2
    model = TabularMLP(n_num=2, cat_vocab_sizes=[5], hidden_dims=[8], dropout=0.0, out_dim=2)

    run_dir = tmp_path / "run"
    schema = {
        "target": "target",
        "numeric_cols": ["age", "income"],
        "categorical_cols": ["city"],
    }
    cfg = {
        "run": {"engine": "torch", "task": "classification"},
        "torch": {"hidden_dims": [8], "dropout": 0.0, "batch_size": 8},
    }

    vocabs = {"city": {"Lisbon": 1, "Porto": 2, "Braga": 3, "Faro": 4}}
    num_mean = np.array([0.0, 0.0], dtype=np.float32)
    num_std = np.array([1.0, 1.0], dtype=np.float32)

    save_torch_bundle(
        run_dir=run_dir,
        model=model,
        vocabs=vocabs,
        num_mean=num_mean,
        num_std=num_std,
        schema=schema,
        config=cfg,
    )

    df = pd.DataFrame({"age": [20, 30], "income": [50000, 60000], "city": ["Lisbon", "Porto"]})
    out = predict_torch_tabular(
        df=df,
        target=None,
        numeric_cols=["age", "income"],
        categorical_cols=["city"],
        vocabs_json=vocabs,
        num_mean=num_mean,
        num_std=num_std,
        state_path=run_dir / "model_state.pt",
        task="classification",
        hidden_dims=[8],
        dropout=0.0,
        batch_size=8,
    )
    assert "pred" in out.columns
