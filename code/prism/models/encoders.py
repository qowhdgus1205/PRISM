"""Encoders: map raw engine features ``X`` to a latent vector ``z``.

All encoders share the same contract ``forward(x) -> z`` so they are
drop-in replacements for one another in the SimCLR pretraining pipeline:

    - ``Encoder``               : 3-layer MLP backbone (default).
    - ``CNNEncoder``            : 1-D CNN over the feature axis.
    - ``FTTransformerEncoder``  : FT-Transformer tabular backbone.
    - ``EGRConditionedEncoder`` : FiLM-conditioned MLP for EGR-robust OOD.

Use :func:`prism.models.factory.build_encoder` to construct one by name.
"""

import torch
import torch.nn as nn

from prism import config as _cfg
from prism.models.layers import init_linear

try:
    from tab_transformer_pytorch import FTTransformer as _FTTransformer
    _HAS_TAB_TRANSFORMER = True
except ImportError:
    _HAS_TAB_TRANSFORMER = False

__all__ = [
    "Encoder",
    "CNNEncoder",
    "FTTransformerEncoder",
    "EGRConditionedEncoder",
]


class Encoder(nn.Module):
    """
    3-layer MLP encoder: R^D -> R^{latent_dim}

    Changes vs original:
    - 8→512 급팽창 제거: 8→64→256→latent_dim 으로 점진적 확장
    - 각 레이어 후 BatchNorm + Dropout 추가 (contrastive 학습 안정화)
    - 마지막 레이어에 LayerNorm 적용 (latent 벡터 스케일 정규화)
    """

    def __init__(self, input_dim: int = 8, latent_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        mid_dim = max(64, input_dim * 8)   # 8 → 64
        wide_dim = max(256, mid_dim * 4)   # 64 → 256

        # USE_LAYER_NORM_ENCODER flag로 BN/LN 선택
        # LN: 각 샘플 독립 정규화 → contrastive BN leakage 방지
        # BN: 배치 통계 공유 → 소규모 배치에서 불안정할 수 있음
        use_ln = getattr(_cfg, "USE_LAYER_NORM_ENCODER", True)
        norm1 = nn.LayerNorm(mid_dim)   if use_ln else nn.BatchNorm1d(mid_dim)
        norm2 = nn.LayerNorm(wide_dim)  if use_ln else nn.BatchNorm1d(wide_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, mid_dim),
            norm1,
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(mid_dim, wide_dim),
            norm2,
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(wide_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        for m in self.encoder:
            if isinstance(m, nn.Linear):
                init_linear(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class CNNEncoder(nn.Module):
    """
    1-D CNN encoder: 8 tabular features를 길이-8 시퀀스로 처리.

    Architecture:
        (B, 8) → unsqueeze → (B, 1, 8)
               → Conv1d(1→16, k=3, p=1) + ReLU   # (B, 16, 8)
               → Conv1d(16→32, k=3, p=1) + ReLU  # (B, 32, 8)
               → GlobalAvgPool(dim=2)              # (B, 32)
               → Linear(32, latent_dim) + LayerNorm
    """

    def __init__(self, input_dim: int = 8, latent_dim: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(32, latent_dim),
            nn.LayerNorm(latent_dim),
        )
        for m in self.proj:
            if isinstance(m, nn.Linear):
                init_linear(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)       # (B, 1, input_dim)
        x = self.conv(x)         # (B, 32, input_dim)
        x = x.mean(dim=2)        # (B, 32)  global avg pool
        return self.proj(x)      # (B, latent_dim)


class FTTransformerEncoder(nn.Module):
    """
    FT-Transformer backbone for SimCLR pipeline.
    CLS token output = latent vector h (latent_dim-dim).
    Drop-in replacement for Encoder: forward(x) → h.
    """

    def __init__(
        self,
        num_continuous: int,
        latent_dim: int = 128,
        dim: int = 64,
        depth: int = 6,
        heads: int = 8,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
    ):
        if not _HAS_TAB_TRANSFORMER:
            raise ImportError(
                "FTTransformerEncoder requires tab-transformer-pytorch. "
                "Install: pip install tab-transformer-pytorch"
            )
        super().__init__()
        self.ft = _FTTransformer(
            categories=(),
            num_continuous=num_continuous,
            dim=dim,
            dim_out=latent_dim,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
        )

    def forward(self, x_cont: torch.Tensor) -> torch.Tensor:
        B = x_cont.shape[0]
        x_categ = torch.empty((B, 0), dtype=torch.long, device=x_cont.device)
        return self.ft(x_categ, x_cont)


class EGRConditionedEncoder(nn.Module):
    """
    FiLM-conditioned encoder for EGR-robust OOD generalization.

    Same backbone as Encoder (8→64→256→latent_dim) but applies
    Feature-wise Linear Modulation (FiLM) using EGR as the conditioning signal.
    FiLM layers learn scale/shift adjustments per EGR value, helping the model
    extrapolate to high-EGR conditions beyond the training distribution.

    Reference: Perez et al. 2018 "FiLM: Visual Reasoning with a General
               Conditioning Layer"
    """

    def __init__(self, input_dim: int = 8, latent_dim: int = 32,
                 egr_idx: int = 2, dropout: float = 0.1):
        super().__init__()
        self.egr_idx = egr_idx
        mid_dim  = max(64, input_dim * 8)
        wide_dim = max(256, mid_dim * 4)

        self.fc1 = nn.Linear(input_dim, mid_dim)
        self.fc2 = nn.Linear(mid_dim, wide_dim)
        self.fc3 = nn.Linear(wide_dim, latent_dim)
        self.norm_final = nn.LayerNorm(latent_dim)
        self.act  = nn.SiLU()
        self.drop = nn.Dropout(dropout)

        # FiLM conditioning: EGR (scalar) → (gamma, beta) for each hidden layer
        # Small init so FiLM starts as near-identity, backbone dominates early
        film_hidden = 32
        self.film1 = nn.Sequential(
            nn.Linear(1, film_hidden), nn.SiLU(),
            nn.Linear(film_hidden, mid_dim * 2),
        )
        self.film2 = nn.Sequential(
            nn.Linear(1, film_hidden), nn.SiLU(),
            nn.Linear(film_hidden, wide_dim * 2),
        )
        # Near-zero init for residual FiLM (γ≈0, β≈0 → behaves like standard encoder initially)
        nn.init.zeros_(self.film1[-1].weight); nn.init.zeros_(self.film1[-1].bias)
        nn.init.zeros_(self.film2[-1].weight); nn.init.zeros_(self.film2[-1].bias)

        for m in [self.fc1, self.fc2, self.fc3]:
            init_linear(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        egr = x[:, self.egr_idx:self.egr_idx + 1]  # (B, 1)

        h = self.fc1(x)
        film1 = self.film1(egr)
        gamma1, beta1 = film1.chunk(2, dim=-1)
        h = h * (1.0 + gamma1) + beta1   # residual FiLM: scale around 1
        h = self.act(h)
        h = self.drop(h)

        h = self.fc2(h)
        film2 = self.film2(egr)
        gamma2, beta2 = film2.chunk(2, dim=-1)
        h = h * (1.0 + gamma2) + beta2
        h = self.act(h)
        h = self.drop(h)

        h = self.fc3(h)
        h = self.norm_final(h)
        return h
