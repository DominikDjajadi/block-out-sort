"""Stable seed derivation for explicit stochastic search trials."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from ..signature import static_level_signature

_MAX_SEED = (1 << 63) - 1


def derive_trial_seed(
    base_seed: int,
    *,
    trial_index: int,
    level_identity: str | bytes | int | None = None,
    evaluation_context: str | None = None,
) -> int:
    """Derive a stable, schedule-independent 63-bit seed for one trial."""
    if isinstance(base_seed, bool) or not isinstance(base_seed, int):
        raise TypeError("base_seed must be an integer")
    if (isinstance(trial_index, bool) or not isinstance(trial_index, int)
            or trial_index < 0):
        raise ValueError("trial_index must be a non-negative integer")
    if evaluation_context is not None and not isinstance(evaluation_context, str):
        raise TypeError("evaluation_context must be a string or None")
    if not isinstance(level_identity, (str, bytes, int, type(None))):
        raise TypeError("level_identity must be str, bytes, int, or None")

    if isinstance(level_identity, bytes):
        identity: Any = {"type": "bytes", "hex": level_identity.hex()}
    elif level_identity is None:
        identity = None
    else:
        identity = {"type": type(level_identity).__name__,
                    "value": str(level_identity)}
    payload = json.dumps(
        {
            "version": 1,
            "base_seed": base_seed,
            "trial_index": trial_index,
            "level_identity": identity,
            "evaluation_context": evaluation_context,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & _MAX_SEED


def level_search_identity(env, level) -> str:
    """Compact identity including both static rules and initial positions."""
    static = static_level_signature(level)
    dynamic = env.canonical_key(env.initial_state(level))
    return hashlib.sha256(f"{static}|{dynamic}".encode("utf-8")).hexdigest()
