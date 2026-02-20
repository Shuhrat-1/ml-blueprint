from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mlb.core.artifacts import save_json, save_yaml


@dataclass(frozen=True)
class TorchBundle:
    state_path: Path
    vocabs_path: Path
    num_stats_path: Path
    meta_path: Path


def save_torch_bundle(
    *,
    run_dir: Path,
    model,
    vocabs: dict[str, dict[str, int]],
    num_mean: np.ndarray,
    num_std: np.ndarray,
    schema: dict[str, Any],
    config: dict[str, Any],
) -> TorchBundle:
    run_dir.mkdir(parents=True, exist_ok=True)

    state_path = run_dir / "model_state.pt"
    vocabs_path = run_dir / "torch_vocabs.json"
    num_stats_path = run_dir / "torch_num_stats.json"
    meta_path = run_dir / "torch_bundle.yaml"

    torch.save(model.state_dict(), state_path)
    save_json(vocabs_path, vocabs)
    save_json(num_stats_path, {"mean": num_mean.tolist(), "std": num_std.tolist()})
    save_yaml(
        meta_path,
        {
            "schema": schema,
            "config": config,
            "files": {
                "state": str(state_path.name),
                "vocabs": str(vocabs_path.name),
                "num_stats": str(num_stats_path.name),
            },
        },
    )

    return TorchBundle(
        state_path=state_path,
        vocabs_path=vocabs_path,
        num_stats_path=num_stats_path,
        meta_path=meta_path,
    )
