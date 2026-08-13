"""Configuration for the expert-iteration loop."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import math
from numbers import Real

from ..training.splits import SplitRatios
from ..training.validation import validate_positive_integer
from .promotion import (
    effective_eval_budgets, promotion_metric_requires_budget_sweep)
from .labeling import LABEL_MODES, LABEL_MODE_HYBRID_PATH


@dataclass(frozen=True)
class ExpertIterationConfig:
    # I/O
    initial_checkpoint: str = ""
    base_dataset: str = ""
    output_dir: str = "runs/expert_iteration"
    extra_levels: str | None = None       # optional JSON pool of generated levels

    # Loop
    iterations: int = 3
    levels_per_iteration: int = 100
    states_per_level: int = 8
    seed: int = 42

    # Teacher labeling
    astar_max_nodes: int = 300_000
    search_simulations: int = 800
    search_c_puct: float = 1.5
    label_policy_temperature: float = 1.0   # soft visit policy for search labels
    label_mode: str = LABEL_MODE_HYBRID_PATH

    # Frozen split (created once, then reused)
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Replay buffer
    max_replay_examples: int = 50_000

    # Training
    epochs: int = 8
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    train_sample_size: int = 4_000        # weighted samples drawn per iteration
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    # Search Q-values are bounded estimates, not exact V*.  Zero keeps their
    # visit-policy supervision while disabling pointwise scalar-value loss.
    search_value_loss_weight: float = 0.0
    policy_anchor_weight: float = 0.0
    device: str = "auto"

    # Source sampling and policy-loss weights (exact weighted more by default).
    weight_exact_historical: float = 1.0
    weight_exact_new: float = 1.5
    weight_search: float = 0.5
    # An exact path proves one action is optimal but does not disprove unknown
    # alternatives, so treat its one-hot policy as lower-confidence imitation.
    exact_path_policy_confidence: float = 0.5

    # Evaluation / promotion
    eval_budgets: tuple[int, ...] = (1, 100, 400)
    eval_limit: int | None = None         # cap eval states (keeps A* affordable)
    promotion_metric: str = "search_confirmed_optimal_rate"
    promotion_budget: int = 400
    promotion_margin: float = 0.0

    def __post_init__(self) -> None:
        SplitRatios(
            train=1.0 - self.val_ratio - self.test_ratio,
            validation=self.val_ratio,
            test=self.test_ratio,
        ).validate()
        validate_positive_integer("expert iteration epochs", self.epochs)
        source_weights = (
            ("weight_exact_historical", self.weight_exact_historical),
            ("weight_exact_new", self.weight_exact_new),
            ("weight_search", self.weight_search),
        )
        for name, value in source_weights:
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(float(value)) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if not any(float(value) > 0 for _name, value in source_weights):
            raise ValueError("at least one source weight must be positive")
        if (isinstance(self.exact_path_policy_confidence, bool)
                or not isinstance(self.exact_path_policy_confidence, Real)
                or not math.isfinite(float(self.exact_path_policy_confidence))
                or not 0 <= self.exact_path_policy_confidence <= 1):
            raise ValueError(
                "exact_path_policy_confidence must be finite and in [0, 1]")
        if self.label_mode not in LABEL_MODES:
            raise ValueError(
                f"label_mode must be one of: {', '.join(LABEL_MODES)}")
        if (isinstance(self.search_value_loss_weight, bool)
                or not isinstance(self.search_value_loss_weight, Real)
                or not math.isfinite(float(self.search_value_loss_weight))
                or self.search_value_loss_weight < 0):
            raise ValueError(
                "search_value_loss_weight must be finite and non-negative")
        if (isinstance(self.policy_anchor_weight, bool)
                or not isinstance(self.policy_anchor_weight, Real)
                or not math.isfinite(float(self.policy_anchor_weight))
                or self.policy_anchor_weight < 0):
            raise ValueError(
                "policy_anchor_weight must be finite and non-negative")
        if self.eval_limit is not None:
            validate_positive_integer("expert iteration eval_limit",
                                      self.eval_limit)
        if (isinstance(self.promotion_margin, bool)
                or not isinstance(self.promotion_margin, Real)
                or not math.isfinite(float(self.promotion_margin))
                or self.promotion_margin < 0):
            raise ValueError("promotion_margin must be finite and non-negative")
        if promotion_metric_requires_budget_sweep(self.promotion_metric):
            raise ValueError(
                "budget-sweep promotion is supported by co-training only")
        object.__setattr__(
            self,
            "eval_budgets",
            effective_eval_budgets(
                self.eval_budgets,
                promotion_metric=self.promotion_metric,
                promotion_budget=self.promotion_budget,
            ),
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["eval_budgets"] = list(self.eval_budgets)
        return d
