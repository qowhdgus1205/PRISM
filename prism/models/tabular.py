"""Backward-compatibility facade for the model components.

The implementations were split into focused modules:

    - ``prism.models.encoders``   : Encoder, CNNEncoder, FTTransformerEncoder,
                                    EGRConditionedEncoder
    - ``prism.models.heads``      : ProjectionHead, ACPHead, LinearHead, ThinHead
    - ``prism.models.regressors`` : TargetMLP, SharedMLPRegressor, ResBlock,
                                    TargetResMLP
    - ``prism.models.factory``    : build_encoder, build_target_head,
                                    build_role_model
    - ``prism.models.layers``     : init_linear

Existing code that did ``from prism.models.tabular import X`` keeps working.
Prefer importing from the focused modules (or the ``prism.models`` package) in
new code.
"""

from prism.models.encoders import (
    CNNEncoder,
    EGRConditionedEncoder,
    Encoder,
    FTTransformerEncoder,
)
from prism.models.factory import build_encoder, build_role_model, build_target_head
from prism.models.heads import ACPHead, LinearHead, ProjectionHead, ThinHead
from prism.models.layers import init_linear as _init_linear  # legacy private alias
from prism.models.regressors import (
    ResBlock,
    SharedMLPRegressor,
    TargetMLP,
    TargetResMLP,
)

__all__ = [
    "Encoder",
    "ProjectionHead",
    "ACPHead",
    "TargetMLP",
    "SharedMLPRegressor",
    "ResBlock",
    "TargetResMLP",
    "CNNEncoder",
    "FTTransformerEncoder",
    "EGRConditionedEncoder",
    "LinearHead",
    "ThinHead",
    "build_role_model",
    "build_encoder",
    "build_target_head",
]
