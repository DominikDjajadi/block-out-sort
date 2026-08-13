"""Supervised policy-value training on the exact A* oracle dataset.

This subpackage adds a stable neural encoding, a fixed action space, a PyTorch
residual policy-value network, dataset/splitting utilities, losses, metrics, and
training/evaluation/prediction CLIs. It does not implement search, self-play, or
any web/serving code (see the milestone scope).
"""

from __future__ import annotations

from .config import (
    COLOR_ORDER,
    DIRECTION_ORDER,
    EncodingConfig,
    EncodingError,
    ModelConfig,
    ValueNormConfig,
)

__all__ = [
    "COLOR_ORDER",
    "DIRECTION_ORDER",
    "EncodingConfig",
    "EncodingError",
    "ModelConfig",
    "ValueNormConfig",
]
