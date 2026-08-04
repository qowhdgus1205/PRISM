"""Regressors: the main Stage-2 prediction networks.

    - ``TargetMLP``         : default MLP regressor on ``[X ‖ z]``.
    - ``SharedMLPRegressor``: Encoder + TargetMLP composite used for fair,
                              architecture-matched baseline comparisons.
    - ``ResBlock`` /         : deeper residual alternative when a stronger
      ``TargetResMLP``         regressor is wanted.
"""

from typing import Optional

import torch
import torch.nn as nn

from prism.models.encoders import Encoder
from prism.models.layers import init_linear

__all__ = [
    "TargetMLP",
    "SharedMLPRegressor",
    "ResBlock",
    "TargetResMLP",
]


class TargetMLP(nn.Module):
    """
    MLP regressor for Stage-2 regression on [X ‖ z] input.

    Changes vs original:
    - 272→68→34 급압축 제거: input→256→128→64→out 으로 점진적 압축
    - 각 레이어 후 LayerNorm + Dropout 추가 (과적합 방지)
    - 마지막 레이어 직전 Dropout 제거 (prediction layer는 정규화 불필요)
    """

    def __init__(self, input_dim: int = 136, output_dim: int = 4, dropout: float = 0.1):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SiLU(),

            nn.Linear(64, output_dim),
        )

        for m in self.model:
            if isinstance(m, nn.Linear):
                init_linear(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class SharedMLPRegressor(nn.Module):
    """Encoder-head MLP used for fair comparisons across MLP variants.

    The same Encoder class is used for X -> Y, X -> ACP, [X, ACP_hat] -> Y,
    and [X, true ACP] -> Y oracle models. Only the input/output dimensions and
    the training protocol differ.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        latent_dim: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = Encoder(input_dim=input_dim, latent_dim=latent_dim, dropout=dropout)
        self.head = TargetMLP(input_dim=latent_dim, output_dim=output_dim, dropout=dropout)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


class ResBlock(nn.Module):
    """
    Residual block with pre-LayerNorm and dropout.

    Changes vs original:
    - res_scale 0.5 유지 (작은 residual로 초기 학습 안정화)
    - fc2 gain 0.1으로 축소 (residual branch를 identity에 가깝게 시작)
    - Dropout 위치 조정: fc1 후에도 dropout 적용
    """

    def __init__(self, dim_in: int, dim_hidden: int, dropout: float = 0.1, res_scale: float = 0.1):
        super().__init__()
        self.pre = nn.LayerNorm(dim_in)
        self.fc1 = nn.Linear(dim_in, dim_hidden)
        self.act = nn.SiLU()
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim_hidden, dim_in)
        self.drop2 = nn.Dropout(dropout)
        self.res_scale = float(res_scale)

        init_linear(self.fc1, nonlinearity="linear")
        init_linear(self.fc2, gain=0.1)  # residual branch를 거의 0으로 시작

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre(x)
        h = self.act(self.fc1(h))
        h = self.drop1(h)
        h = self.drop2(self.fc2(h))
        return x + self.res_scale * h


class TargetResMLP(nn.Module):
    """
    Deeper residual MLP for regression.

    Changes vs original:
    - Stem에 BatchNorm 추가 (입력 스케일 정규화)
    - ResBlock의 res_scale을 0.5 → 0.1 (초기 학습 안정화, 깊어질수록 중요)
    - Head의 중간 LayerNorm 추가
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 4,
        width: Optional[int] = None,
        n_blocks: int = 6,
        dropout: float = 0.1,
        res_scale: float = 0.1,
    ):
        super().__init__()
        width = int(width or max(256, input_dim * 2))

        self.stem = nn.Sequential(
            nn.BatchNorm1d(input_dim),   # 입력 feature 스케일 정규화
            nn.Linear(input_dim, width),
            nn.SiLU(),
        )
        for m in self.stem:
            if isinstance(m, nn.Linear):
                init_linear(m)

        self.blocks = nn.Sequential(
            *[
                ResBlock(width, dim_hidden=width * 2, dropout=dropout, res_scale=res_scale)
                for _ in range(int(n_blocks))
            ]
        )

        head_hidden = max(width // 2, 64)
        self.head = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, head_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(head_hidden),
            nn.Linear(head_hidden, output_dim),
        )
        for m in self.head:
            if isinstance(m, nn.Linear):
                init_linear(m, nonlinearity="linear")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x
