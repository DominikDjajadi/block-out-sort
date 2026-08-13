"""Supervised policy-value dataset tooling (exact labels from the A* oracle)."""

from __future__ import annotations

from .schema import (
    DATASET_VERSION,
    LABEL_EXACT_PATH_POLICY,
    LABEL_FULL_EXACT,
    POLICY_SINGLE_VERIFIED_OPTIMAL,
    build_exact_path_record,
    build_record,
    deserialize_state,
    serialize_state,
)
from .targets import (
    soft_regret_policy,
    uniform_optimal_policy,
    value_target,
)

__all__ = [
    "DATASET_VERSION",
    "LABEL_EXACT_PATH_POLICY",
    "LABEL_FULL_EXACT",
    "POLICY_SINGLE_VERIFIED_OPTIMAL",
    "build_exact_path_record",
    "build_record",
    "deserialize_state",
    "serialize_state",
    "soft_regret_policy",
    "uniform_optimal_policy",
    "value_target",
]
