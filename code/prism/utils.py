"""Compatibility facade for shared PRISM utilities.

New code should prefer focused modules such as `prism.metrics`, `prism.losses`,
`prism.io`, and `prism.reproducibility`; this module keeps existing imports
working.
"""

from prism.io import load_torch_state, save_json
from prism.losses import (
    curvature_loss,
    egr_axis_augmentation,
    large_range_egr_mono_loss,
    monotonicity_loss,
    perturbation_monotonicity_loss,
    supervised_physics_contrastive_loss,
)
from prism.metrics import compute_metrics
from prism.reproducibility import set_seed
from prism.training_utils import evaluate_on_test, train_model

__all__ = [
    "compute_metrics",
    "curvature_loss",
    "egr_axis_augmentation",
    "evaluate_on_test",
    "large_range_egr_mono_loss",
    "load_torch_state",
    "monotonicity_loss",
    "perturbation_monotonicity_loss",
    "save_json",
    "set_seed",
    "supervised_physics_contrastive_loss",
    "train_model",
]
