"""Shared low-level building blocks reused across model components.

Keeping the weight-init helper in one place guarantees every encoder, head,
and regressor follows the same initialization convention.
"""

from typing import Optional

import torch.nn as nn

__all__ = ["init_linear"]


def init_linear(
    module: nn.Linear,
    nonlinearity: str = "leaky_relu",
    gain: Optional[float] = None,
) -> None:
    """Kaiming/Xavier init with optional gain; biases set to zero.

    Parameters
    ----------
    module
        The ``nn.Linear`` layer to initialize in place.
    nonlinearity
        Passed to Kaiming init when ``gain`` is ``None`` (e.g. ``"linear"``
        for prediction layers, ``"leaky_relu"`` for hidden layers).
    gain
        If provided, Xavier-uniform init with this gain is used instead of
        Kaiming. Small gains (e.g. ``0.1``) start a residual branch near
        identity.
    """
    if gain is not None:
        nn.init.xavier_uniform_(module.weight, gain=gain)
    else:
        nn.init.kaiming_uniform_(module.weight, a=0.0, nonlinearity=nonlinearity)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
