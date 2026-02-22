from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mlb.core.artifacts import save_json, save_yaml
from mlb.core.versioning import get_runtime_info
from mlb.models.torch.dataset import MISSING_ID, UNK_ID, CatVocab
from mlb.models.torch.signature import compute_feature_signature


@dataclass(frozen=True)
class TorchBundle:
    state_path: Path
    vocabs_path: Path
    num_stats_path: Path
    meta_path: Path

BUNDLE_VERSION = 1

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
    runtime = get_runtime_info()
    created_utc = datetime.now(timezone.utc).isoformat()

    # Reconstruct CatVocab objects to compute vocab sizes consistently
    vocabs_obj = {k: CatVocab(mapping=v) for k, v in vocabs.items()}
    num_stats_dict = {"mean": num_mean.tolist(), "std": num_std.tolist()}

    feature_signature = compute_feature_signature(
        schema_resolved=schema,
        vocabs=vocabs_obj,
        num_stats=num_stats_dict,
    )

    torch.save(model.state_dict(), state_path)
    save_json(vocabs_path, vocabs)
    save_json(num_stats_path, {"mean": num_mean.tolist(), "std": num_std.tolist()})
    save_yaml(
        meta_path,
        {
            # Day 9 additions
            "bundle_version": BUNDLE_VERSION,
            "created_utc": created_utc,
            "runtime": runtime.to_dict(),
            "torch_version": torch.__version__, # Даже с санитизацией полезно не таскать “нечистые типы”. "torch_version": str(torch.__version__),
            "special_ids": {"unk": UNK_ID, "missing": MISSING_ID},
            "feature_signature": feature_signature,
            # Keep old fields for compatibility
            "schema": schema,
            "config": config,
            "files": {
                "state": str(state_path.name),
                "vocabs": str(vocabs_path.name),
                "num_stats": str(num_stats_path.name),
                "meta": str(meta_path.name),
            },
        },
    )

    return TorchBundle(
        state_path=state_path,
        vocabs_path=vocabs_path,
        num_stats_path=num_stats_path,
        meta_path=meta_path,
    )