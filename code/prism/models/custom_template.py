"""Template for adding experiment-ready model families.

Usage example:
    python train_repr_baselines.py \
        --model-module prism.models.custom_template \
        --model-profile wide_shared_encoder

A custom family must return a model for every supported role:
    simple_mlp, acp_predictor, two_stage_mlp, oracle_mlp
"""

from __future__ import annotations

import torch.nn as nn

from prism.models.registry import ModelRequest, register_model_family
from prism.models.tabular import Encoder, TargetMLP


class WideSharedMLPRegressor(nn.Module):
    """Example custom MLP family with a wider latent head.

    It keeps the same pattern as the default comparison models:
        input -> encoder -> head -> output
    """

    def __init__(self, input_dim: int, output_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim, dropout=dropout)
        self.head = TargetMLP(input_dim=latent_dim, output_dim=output_dim, dropout=dropout)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.head(self.encode(x))


@register_model_family("wide_shared_encoder")
def build_wide_shared_encoder(request: ModelRequest) -> nn.Module:
    role_dropout = {
        "simple_mlp": 0.12,
        "acp_predictor": 0.08,
        "two_stage_mlp": 0.12,
        "oracle_mlp": 0.12,
    }[request.role]
    return WideSharedMLPRegressor(
        input_dim=request.input_dim,
        output_dim=request.output_dim,
        latent_dim=request.latent_dim * 2,
        dropout=role_dropout if request.dropout is None else request.dropout,
    )
