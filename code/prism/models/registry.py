from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

import torch.nn as nn


@dataclass(frozen=True)
class ModelRequest:
    role: str
    input_dim: int
    output_dim: int
    latent_dim: int
    dropout: Optional[float] = None


ModelBuilder = Callable[[ModelRequest], nn.Module]
_MODEL_FAMILIES: Dict[str, ModelBuilder] = {}


VALID_ROLES = {"simple_mlp", "acp_predictor", "two_stage_mlp", "oracle_mlp"}


def register_model_family(name: str) -> Callable[[ModelBuilder], ModelBuilder]:
    key = name.strip().lower()
    if not key:
        raise ValueError("Model family name cannot be empty")

    def decorator(builder: ModelBuilder) -> ModelBuilder:
        if key in _MODEL_FAMILIES:
            raise ValueError(f"Model family '{key}' is already registered")
        _MODEL_FAMILIES[key] = builder
        return builder

    return decorator


def available_model_families() -> list[str]:
    return sorted(_MODEL_FAMILIES)


def import_model_modules(module_names: Iterable[str]) -> None:
    for module_name in module_names:
        importlib.import_module(module_name)


def build_registered_model(
    family: str,
    role: str,
    input_dim: int,
    output_dim: int,
    latent_dim: int,
    dropout: Optional[float] = None,
) -> nn.Module:
    family_key = family.strip().lower()
    role_key = role.strip().lower()
    if role_key not in VALID_ROLES:
        raise ValueError(f"Unknown model role '{role}'. Choices: {sorted(VALID_ROLES)}")
    if family_key not in _MODEL_FAMILIES:
        raise ValueError(
            f"Unknown model family '{family}'. Registered families: {available_model_families()}"
        )
    request = ModelRequest(
        role=role_key,
        input_dim=input_dim,
        output_dim=output_dim,
        latent_dim=latent_dim,
        dropout=dropout,
    )
    return _MODEL_FAMILIES[family_key](request)


@register_model_family("shared_encoder")
def _build_shared_encoder(request: ModelRequest) -> nn.Module:
    from prism.models.regressors import SharedMLPRegressor

    role_dropout = {
        "simple_mlp": 0.10,
        "acp_predictor": 0.08,
        "two_stage_mlp": 0.10,
        "oracle_mlp": 0.10,
    }[request.role]
    return SharedMLPRegressor(
        input_dim=request.input_dim,
        output_dim=request.output_dim,
        latent_dim=request.latent_dim,
        dropout=role_dropout if request.dropout is None else request.dropout,
    )


@register_model_family("legacy")
def _build_legacy(request: ModelRequest) -> nn.Module:
    from prism.models.regressors import TargetMLP

    return TargetMLP(
        input_dim=request.input_dim,
        output_dim=request.output_dim,
        dropout=0.10 if request.dropout is None else request.dropout,
    )
