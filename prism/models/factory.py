"""Factory helpers that build model components by name.

These keep training/analysis code free of hard-coded class names:

    - ``build_encoder``     : backbone name  -> encoder instance.
    - ``build_target_head`` : head-type name -> Stage-2 head instance.
    - ``build_role_model``  : role name      -> model via the family registry
                              (see :mod:`prism.models.registry`).
"""

from typing import Optional

import torch.nn as nn

from prism.models.encoders import CNNEncoder, Encoder, FTTransformerEncoder
from prism.models.heads import LinearHead, ThinHead
from prism.models.regressors import TargetMLP

__all__ = ["build_encoder", "build_target_head", "build_role_model"]


def build_target_head(
    head_type: str, input_dim: int, output_dim: int, dropout: float = 0.1
) -> nn.Module:
    h = head_type.lower().strip()
    if h == "mlp":
        return TargetMLP(input_dim=input_dim, output_dim=output_dim, dropout=dropout)
    if h == "thin":
        return ThinHead(input_dim=input_dim, output_dim=output_dim, dropout=min(dropout, 0.05))
    if h == "linear":
        return LinearHead(input_dim=input_dim, output_dim=output_dim)
    raise ValueError(f"Unknown stage2 head '{head_type}'. Choices: mlp, thin, linear")


def build_role_model(
    role: str,
    input_dim: int,
    output_dim: int,
    profile: str = "shared_encoder",
    dropout: Optional[float] = None,
    latent_dim: int = 32,
) -> nn.Module:
    """Compatibility wrapper around the model-family registry."""
    from prism.models.registry import build_registered_model

    return build_registered_model(
        family=profile,
        role=role,
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=latent_dim,
        dropout=dropout,
    )


def build_encoder(
    backbone: str,
    input_dim: int,
    latent_dim: int,
    cfg=None,
) -> nn.Module:
    """
    Factory function: backbone 이름으로 encoder 인스턴스 생성.

    Parameters
    ----------
    backbone   : "mlp" | "cnn" | "ftt"
    input_dim  : 입력 feature 수 (예: 8)
    latent_dim : 출력 embedding 차원 (예: 128)
    cfg        : config 모듈 (None이면 config.py 자동 import)
    """
    if cfg is None:
        from prism import config as cfg  # type: ignore[assignment]
    b = backbone.lower().strip()
    if b == "mlp":
        return Encoder(input_dim=input_dim, latent_dim=latent_dim)
    elif b == "cnn":
        return CNNEncoder(input_dim=input_dim, latent_dim=latent_dim)
    elif b in ("ftt", "ft_transformer", "fttransformer"):
        return FTTransformerEncoder(
            num_continuous=input_dim,
            latent_dim=latent_dim,
            dim=getattr(cfg, "FT_DIM", 64),
            depth=getattr(cfg, "FT_DEPTH", 6),
            heads=getattr(cfg, "FT_HEADS", 8),
            attn_dropout=getattr(cfg, "FT_ATTN_DROPOUT", 0.1),
            ff_dropout=getattr(cfg, "FT_FF_DROPOUT", 0.1),
        )
    else:
        raise ValueError(f"Unknown backbone '{backbone}'. Choices: mlp, cnn, ftt")
