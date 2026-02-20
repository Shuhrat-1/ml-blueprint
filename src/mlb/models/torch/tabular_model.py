from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn


def _embed_dim(vocab_size: int) -> int:
    # common heuristic
    return int(min(64, round(1.6 * (vocab_size**0.56))))


@dataclass(frozen=True)
class ModelInfo:
    num_features: int
    cat_features: int
    cat_vocab_sizes: list[int]
    cat_embed_dims: list[int]


class TabularMLP(nn.Module):
    def __init__(
        self,
        *,
        n_num: int,
        cat_vocab_sizes: list[int],
        hidden_dims: list[int],
        dropout: float,
        out_dim: int,
        cat_embed_dims: list[int] | None = None,
    ) -> None:
        super().__init__()

        self.embeddings = nn.ModuleList()
        cat_out_dim = 0

        if cat_embed_dims is not None and len(cat_embed_dims) != len(cat_vocab_sizes):
            raise ValueError("cat_embed_dims length must match cat_vocab_sizes length.")

        for i, vs in enumerate(cat_vocab_sizes):
            ed = cat_embed_dims[i] if cat_embed_dims is not None else _embed_dim(vs)
            self.embeddings.append(nn.Embedding(num_embeddings=vs, embedding_dim=ed))
            cat_out_dim += ed

        in_dim = n_num + cat_out_dim

        layers: list[nn.Module] = []
        cur = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(cur, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            cur = h
        layers.append(nn.Linear(cur, out_dim))
        self.mlp = nn.Sequential(*layers)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        if x_cat.numel() > 0:
            embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
            x = torch.cat([x_num, *embs], dim=1) if x_num.numel() > 0 else torch.cat(embs, dim=1)
        else:
            x = x_num
        return self.mlp(x)
