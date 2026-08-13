"""Small, explicit validators for user-controlled numeric configuration."""

from __future__ import annotations

import math


def validate_positive_finite(name: str, value: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a finite number greater than 0; got {value!r}"
        ) from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and greater than 0; got {value!r}")
    return number


def validate_positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(
            f"{name} must be an integer greater than or equal to 1; got {value!r}")
    return value


def validate_nonnegative_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer; got {value!r}")
    return value
