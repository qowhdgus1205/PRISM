"""Heads: small networks that map a latent ``z`` (or features) to an output.

    - ``ProjectionHead`` : latent → contrastive projection space (SimCLR).
    - ``ACPHead``        : latent → ACP regression (auxiliary supervision).
    - ``LinearHead``     : minimal linear probe of latent quality.
    - ``ThinHead``       : shallow nonlinear probe (between linear and TargetMLP).

The Stage-2 regression network ``TargetMLP`` lives in
:mod:`prism.models.regressors` because it is the main predictor rather than a
lightweight probe.
"""

import torch
import torch.nn as nn

from prism.models.layers import init_linear

__all__ = [
    "ProjectionHead",
    "ACPHead",
    "LinearHead",
    "ThinHead",
]


class ProjectionHead(nn.Module):
    """
    2-layer projection head for contrastive learning.

    Changes vs original:
    - SimCLR 논문 표준에 따라 첫 번째 Linear 후 BatchNorm 추가
    - BN이 없으면 NT-Xent loss의 gradient가 불안정해져 학습 발산 가능
    """

    def __init__(self, input_dim: int = 128, proj_dim: int = 128):
        super().__init__()
        hidden_dim = max(256, input_dim * 2)

        # Projection head: 첫 번째 BN은 SimCLR 표준 유지.
        # Output BN 제거: loss에서 F.normalize로 L2 정규화하므로 중복.
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),   # SimCLR 표준 (Chen et al. 2020)
            nn.SiLU(),
            nn.Linear(hidden_dim, proj_dim),
        )

        for m in self.proj:
            if isinstance(m, nn.Linear):
                init_linear(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class ACPHead(nn.Module):
    """
    Auxiliary regression head: latent z → ACP 예측.

    목적:
    - Contrastive loss만으로는 ACP 정보가 latent에 잘 녹지 않음 (ablation 확인)
    - 직접 ACP를 예측하는 regression loss를 추가해 강한 supervision 제공
    - 학습 완료 후 이 head는 버리고 encoder만 downstream에 사용
    """

    def __init__(self, latent_dim: int, acp_dim: int):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim // 2),
            nn.SiLU(),
            nn.Linear(latent_dim // 2, acp_dim),
        )
        for m in self.head:
            if isinstance(m, nn.Linear):
                init_linear(m, nonlinearity="linear")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)


class LinearHead(nn.Module):
    """Minimal linear regression head for probing latent quality."""

    def __init__(self, input_dim: int, output_dim: int = 4):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        init_linear(self.linear, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ThinHead(nn.Module):
    """Small nonlinear head that is weaker than TargetMLP but more expressive than linear."""

    def __init__(self, input_dim: int, output_dim: int = 4, dropout: float = 0.05):
        super().__init__()
        hidden_dim = max(32, min(128, input_dim))
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                init_linear(m, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
