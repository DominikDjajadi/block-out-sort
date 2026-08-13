"""Configuration objects and fixed orderings for neural encoding/training.

The orderings here are part of the on-disk contract: they are saved in every
checkpoint so a model can always be paired with the exact encoding that produced
its training tensors. See ``docs/neural_encoding.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math

# Fixed, never-reordered palette (matches the JS engine palette order).
COLOR_ORDER: tuple[str, ...] = (
    "red", "blue", "green", "yellow", "purple", "orange", "teal", "pink",
)

# Fixed direction order; index is used in the action encoding.
DIRECTION_ORDER: tuple[str, ...] = ("up", "down", "left", "right")


class EncodingError(ValueError):
    """Raised when a level/state/action cannot be represented by the config."""


@dataclass(frozen=True)
class EncodingConfig:
    """Board/action encoding limits and normalization constants."""

    max_rows: int = 8
    max_cols: int = 8
    max_slide_distance: int = 8
    max_blocks: int = 16
    colors: tuple[str, ...] = COLOR_ORDER

    # ----- derived sizes -----

    @property
    def num_colors(self) -> int:
        return len(self.colors)

    @property
    def move_code_count(self) -> int:
        """Slide distances 1..D plus a dedicated EXIT slot."""
        return self.max_slide_distance + 1

    @property
    def exit_move_code(self) -> int:
        """Move-code index reserved for EXIT (the final slot)."""
        return self.max_slide_distance

    # Per-direction gate block: num_colors color planes + active/locked/remaining.
    @property
    def gate_dir_stride(self) -> int:
        return self.num_colors + 3

    @property
    def num_board_channels(self) -> int:
        # See docs/neural_encoding.md for the channel map.
        # 4 structure + num_colors occupancy + 5 identity + 2 frozen
        # + 4 directions * (num_colors + 3) gate planes + 2 scalar planes
        return 4 + self.num_colors + 5 + 2 + 4 * self.gate_dir_stride + 2

    @property
    def num_global_features(self) -> int:
        return 6

    @property
    def policy_channels(self) -> int:
        return 4 * self.move_code_count

    @property
    def action_space_size(self) -> int:
        return self.max_rows * self.max_cols * 4 * self.move_code_count

    def to_dict(self) -> dict:
        d = asdict(self)
        d["colors"] = list(self.colors)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "EncodingConfig":
        return cls(
            max_rows=int(data["max_rows"]),
            max_cols=int(data["max_cols"]),
            max_slide_distance=int(data["max_slide_distance"]),
            max_blocks=int(data["max_blocks"]),
            colors=tuple(data.get("colors", COLOR_ORDER)),
        )


@dataclass(frozen=True)
class ValueNormConfig:
    """Value normalization: ``normalized = -raw / constant`` for the default."""

    scheme: str = "neg_over_constant"
    constant: float = 20.0

    def __post_init__(self) -> None:
        from .validation import validate_positive_finite
        if self.scheme != "neg_over_constant":
            raise ValueError(f"unknown value scheme: {self.scheme}")
        validate_positive_finite("value normalization constant", self.constant)

    def normalize(self, raw_moves: float) -> float:
        raw = float(raw_moves)
        if not math.isfinite(raw):
            raise ValueError(f"raw_moves must be finite; got {raw_moves!r}")
        result = -raw / self.constant
        if not math.isfinite(result):
            raise ValueError("normalized value must be finite")
        return result

    def denormalize(self, normalized: float) -> float:
        value = float(normalized)
        if not math.isfinite(value):
            raise ValueError(f"normalized value must be finite; got {normalized!r}")
        result = -value * self.constant
        if not math.isfinite(result):
            raise ValueError("denormalized value must be finite")
        return result

    def to_dict(self) -> dict:
        return {"scheme": self.scheme, "constant": self.constant}

    @classmethod
    def from_dict(cls, data: dict) -> "ValueNormConfig":
        return cls(scheme=data.get("scheme", "neg_over_constant"),
                   constant=float(data.get("constant", 20.0)))


@dataclass(frozen=True)
class ModelConfig:
    """Residual policy-value network architecture."""

    channels: int = 128
    residual_blocks: int = 6
    value_hidden_size: int = 256
    normalization: str = "batch_norm"
    policy_adapter_blocks: int = 0

    def __post_init__(self) -> None:
        if (isinstance(self.channels, bool) or not isinstance(self.channels, int)
                or self.channels <= 0):
            raise ValueError("model channels must be a positive integer")
        if (isinstance(self.residual_blocks, bool)
                or not isinstance(self.residual_blocks, int)
                or self.residual_blocks < 0):
            raise ValueError(
                "model residual_blocks must be a non-negative integer")
        if (isinstance(self.value_hidden_size, bool)
                or not isinstance(self.value_hidden_size, int)
                or self.value_hidden_size <= 0):
            raise ValueError(
                "model value_hidden_size must be a positive integer")
        if (isinstance(self.policy_adapter_blocks, bool)
                or not isinstance(self.policy_adapter_blocks, int)
                or self.policy_adapter_blocks < 0):
            raise ValueError(
                "model policy_adapter_blocks must be a non-negative integer")
        if self.normalization not in ("batch_norm", "group_norm"):
            raise ValueError(
                "model normalization must be 'batch_norm' or 'group_norm'")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(
            channels=int(data["channels"]),
            residual_blocks=int(data["residual_blocks"]),
            value_hidden_size=int(data["value_hidden_size"]),
            normalization=str(data.get("normalization", "batch_norm")),
            policy_adapter_blocks=int(data.get("policy_adapter_blocks", 0)),
        )


MODEL_PROFILES: dict[str, ModelConfig] = {
    # About 71k parameters with the default 8x8 encoding. GroupNorm avoids
    # running-statistic drift on small, non-IID replay updates.
    "small_groupnorm": ModelConfig(
        channels=32,
        residual_blocks=2,
        value_hidden_size=64,
        normalization="group_norm",
    ),
    # Reproduces the original approximately 2M-parameter architecture.
    "legacy_batchnorm": ModelConfig(),
}


def model_profile(name: str) -> ModelConfig:
    try:
        return MODEL_PROFILES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown model profile {name!r}; expected one of "
            f"{', '.join(sorted(MODEL_PROFILES))}") from exc
