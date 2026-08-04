"""Model components used by PRISM experiments.

Organized by role:
    - ``encoders``   : X -> latent z backbones.
    - ``heads``      : latent -> output heads (projection, ACP, probes).
    - ``regressors`` : Stage-2 prediction networks.
    - ``factory``    : build components by name.
    - ``registry``   : pluggable model-family registry.
"""

from prism.models.encoders import (
    CNNEncoder,
    EGRConditionedEncoder,
    Encoder,
    FTTransformerEncoder,
)
from prism.models.factory import build_encoder, build_role_model, build_target_head
from prism.models.heads import ACPHead, LinearHead, ProjectionHead, ThinHead
from prism.models.regressors import (
    ResBlock,
    SharedMLPRegressor,
    TargetMLP,
    TargetResMLP,
)
from prism.models.registry import (
    ModelRequest,
    available_model_families,
    build_registered_model,
    import_model_modules,
    register_model_family,
)

__all__ = [
    # encoders
    "Encoder",
    "CNNEncoder",
    "FTTransformerEncoder",
    "EGRConditionedEncoder",
    # heads
    "ProjectionHead",
    "ACPHead",
    "LinearHead",
    "ThinHead",
    # regressors
    "TargetMLP",
    "SharedMLPRegressor",
    "ResBlock",
    "TargetResMLP",
    # factory
    "build_role_model",
    "build_encoder",
    "build_target_head",
    # registry
    "ModelRequest",
    "available_model_families",
    "build_registered_model",
    "import_model_modules",
    "register_model_family",
]
