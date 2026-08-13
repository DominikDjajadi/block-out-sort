"""Configuration + curriculum state for co-training."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
from numbers import Real

from ..designer.config import RewardConfig
from ..expert_iteration.promotion import (
    effective_eval_budgets, normalize_budget_sweep_config,
    promotion_metric_requires_budget_sweep)
from ..expert_iteration.labeling import LABEL_MODES, LABEL_MODE_HYBRID_PATH
from ..expert_iteration.train import TRAINABLE_PARTS
from ..training.splits import SplitRatios
from ..training.validation import validate_positive_integer
from .eval_split import DEFAULT_EVAL_SPLIT_SEED
from .policy_targets import (
    DEFAULT_POLICY_TARGET_PROFILE,
    POLICY_TARGET_PROFILES,
)


@dataclass(frozen=True)
class CurriculumState:
    """The currently active generation difficulty knobs."""

    # Begin inside the bundled protagonist's 5x5, 3-7-block pretraining
    # envelope. Curriculum adaptation can expand difficulty after demonstrated
    # frontier performance instead of changing every structural axis at once.
    rows: int = 5
    cols: int = 5
    color_count: int = 3
    density: float = 0.35
    mutation_budget: int = 4           # also the reverse-slide depth
    protagonist_simulations: int = 100
    structural_threshold: int = 1      # min forced extra moves to "prefer"

    def to_dict(self) -> dict:
        return asdict(self)

    def replace(self, **kw) -> "CurriculumState":
        return replace(self, **kw)


@dataclass(frozen=True)
class CurriculumConfig:
    """Frontier band + bounds + step sizes for adaptive difficulty."""

    frontier_min_solve_rate: float = 0.2
    frontier_max_solve_rate: float = 0.7
    min_frontier_acceptance_rate: float = 0.1
    frontier_imbalance_margin: float = 0.05

    min_density: float = 0.3
    max_density: float = 0.65
    density_step: float = 0.05

    min_mutation_budget: int = 2
    max_mutation_budget: int = 24
    mutation_step: int = 2

    min_structural_threshold: int = 0
    max_structural_threshold: int = 6
    structural_step: int = 1

    # Protagonist search budget is the first lever: raise it when levels are too
    # hard (directly increases solve rate without changing the level
    # distribution) and lower it when they are too easy.
    min_protagonist_simulations: int = 20
    max_protagonist_simulations: int = 400
    protagonist_sim_step: int = 60

    # Colors / blocks.
    min_color_count: int = 1
    max_color_count: int = 4
    color_step: int = 1

    # Board size (shrunk when persistently too hard, widened when too easy).
    min_board: int = 3
    max_board: int = 8
    board_step: int = 1
    grow_board_when_density_maxed: bool = True
    shrink_board_when_floored: bool = True

    def __post_init__(self) -> None:
        for name, value in (
                ("frontier_min_solve_rate", self.frontier_min_solve_rate),
                ("frontier_max_solve_rate", self.frontier_max_solve_rate),
                ("min_frontier_acceptance_rate",
                 self.min_frontier_acceptance_rate),
                ("frontier_imbalance_margin",
                 self.frontier_imbalance_margin)):
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 1.0):
                raise ValueError(f"{name} must be a finite rate in [0, 1]")
        if self.frontier_min_solve_rate > self.frontier_max_solve_rate:
            raise ValueError(
                "frontier_min_solve_rate cannot exceed "
                "frontier_max_solve_rate")


@dataclass(frozen=True)
class CoTrainingConfig:
    # I/O
    protagonist_checkpoint: str = ""
    designer_checkpoint: str = ""
    base_dataset: str = ""
    # Optional learner ancestry for a fresh shadow-learner run. The checkpoint
    # is imported into the new run; it never replaces the official champion.
    initial_learner_checkpoint: str | None = None
    initial_base_split: str | None = None
    initial_protagonist_replay: str | None = None
    initial_designer_replay: str | None = None
    output_dir: str = "runs/cotraining"
    initialize_only: bool = False
    # Operational disk policy. After a round is durably committed, discard
    # only superseded replay snapshots and non-milestone sample copies.
    prune_superseded_round_artifacts: bool = False

    # Loop
    rounds: int = 3
    # End the current run immediately after the first durably committed
    # promotion. This supports sequential milestone selection: confirm the
    # promoted checkpoint on a separately sealed pool before any later update.
    stop_after_promotion: bool = False
    levels_per_round: int = 100
    seed: int = 42
    device: str = "auto"

    # Frontier estimation
    solve_rate_trials: int = 5
    # Root exploration noise for the *frontier* solve-rate estimate only. Without
    # noise the bounded protagonist is deterministic, so a per-level solve rate
    # is always 0 or 1 and the [min,max] frontier band is unreachable. Noise lets
    # repeated solves differ so moderately hard levels land inside the band.
    frontier_dirichlet_alpha: float = 0.5
    frontier_dirichlet_weight: float = 0.4
    # Search each level across a geometric range around the curriculum's
    # nominal protagonist budget. This turns deterministic budget sensitivity
    # into a useful frontier signal instead of relying on noise alone.
    frontier_budget_min_ratio: float = 0.25
    frontier_budget_max_ratio: float = 4.0
    # Strict frontier membership remains the designer-quality metric. When
    # strict yield is sparse, retain the nearest additional oracle-solvable
    # levels until this training quota is reached.
    frontier_backfill_target: int = 10

    # Teacher labeling / evaluation budgets
    astar_max_nodes: int = 200_000
    # Exact search on generated/training states is exploratory: construction
    # already proves generated levels solvable, and labeling falls back to
    # neural search when this wall-clock limit is reached.
    exploratory_astar_time_limit_seconds: float | None = 5.0
    training_astar_time_limit_seconds: float | None = 5.0
    eval_astar_max_nodes: int = 20_000
    eval_astar_time_limit_seconds: float | None = 3.0
    oracle_simulations: int = 1000
    eval_budgets: tuple[int, ...] = (1, 100, 400)
    # Limit distinct held-out levels, not correlated states within a level.
    # None evaluates every level in the frozen split.
    eval_limit: int | None = None
    promotion_metric: str = "search_confirmed_optimal_rate"
    promotion_budget: int = 400
    promotion_budgets: tuple[int, ...] = ()
    promotion_budget_weights: tuple[float, ...] = ()
    promotion_margin: float = 0.0
    # Optional stronger gate for full-level budget-sweep promotion. It uses
    # paired level outcomes, applies an inclusive margin, rejects excessive
    # regression at any budget, and requires a positive bootstrap lower bound.
    promotion_paired_gate_enabled: bool = False
    promotion_max_per_budget_regression: float = 0.02
    promotion_bootstrap_confidence: float = 0.95
    promotion_bootstrap_replicates: int = 10_000
    promotion_bootstrap_seed: int = 8216

    # Optional cumulative shadow learner. The champion remains the curriculum
    # generator and promotion baseline; the learner continues across rejected
    # rounds and is challenged only at pre-registered milestones.
    shadow_learner_enabled: bool = False
    learner_milestone_interval: int = 5
    learner_max_policy_kl: float = 0.25
    learner_min_entropy_ratio: float = 0.70
    # Optional frozen development pool used to detect difficulty-band
    # retention regressions at shadow-learner milestones.  Monitoring and
    # enforcement are intentionally distinct so a new guard can be observed
    # before it is allowed to roll a learner back.
    learner_retention_dataset: str | None = None
    learner_retention_budgets: tuple[int, ...] = (64, 95, 128)
    learner_retention_per_band: int = 20
    # Preserve intentionally unequal strata (for example 100/100/200/100)
    # instead of truncating every band to ``learner_retention_per_band``.
    learner_retention_use_full_pool: bool = False
    learner_retention_max_regression: float = 0.05
    learner_retention_enforce: bool = False

    # Protagonist fine-tuning (expert-iteration training)
    # A positive value prevents replay-only updates when a round produced too
    # little genuinely new material. Zero explicitly disables the guard.
    min_fresh_levels_to_train: int = 10
    states_per_level: int = 4
    train_sample_size: int = 2_000
    replay_sample_with_replacement: bool = False
    # Reserve explicit sample shares for the active round, a short recent
    # window, and older replay. Empty buckets redistribute their share
    # proportionally across the remaining non-empty buckets. Source-aware loss
    # weights below then determine the realized policy-gradient mass.
    replay_current_fraction: float = 0.35
    replay_recent_fraction: float = 0.25
    replay_historical_fraction: float = 0.40
    replay_recent_window: int = 2
    # Small, non-IID co-training updates default to the policy head only. This
    # leaves trunk/value parameters and their BatchNorm state untouched.
    epochs: int = 1
    batch_size: int = 128
    learning_rate: float = 3e-5
    trainable_part: str = "policy_head"
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    # Source preferences affect both replay inclusion and the selected
    # records' policy-loss mass. Reports expose the realized mass explicitly.
    weight_exact_historical: float = 1.0
    weight_exact_new: float = 1.5
    weight_search: float = 0.5
    exact_path_policy_confidence: float = 0.5
    # Search records retain visit-policy supervision but do not supervise a
    # point scalar value unless this confidence multiplier is explicitly set.
    search_value_loss_weight: float = 0.0
    # Preserve the incumbent's legal-action distribution on historical replay
    # while still allowing new round data to change the policy.
    policy_anchor_weight: float = 0.0
    # Exact full-oracle labels retain their optimal support while using the
    # incumbent to break ties without repeatedly sharpening an already
    # concentrated policy across promoted rounds.
    policy_target_profile: str = DEFAULT_POLICY_TARGET_PROFILE
    max_protagonist_replay: int = 50_000

    # Operational-only generation durability and visibility. These do not
    # affect generated examples because each level has its own derived RNG.
    generation_checkpoint_interval: int = 5
    generation_progress_interval: int = 5

    # Frozen split
    val_ratio: float = 0.1
    test_ratio: float = 0.1

    # Designer training (per round, against the frozen promoted protagonist)
    designer_episodes: int = 48
    designer_episodes_per_iter: int = 12
    designer_validation_episodes: int = 8
    # Bonus added to the existing adversarial reward when repeated noisy
    # protagonist trials land inside (or near) the curriculum frontier.
    designer_frontier_alignment_weight: float = 1.0
    designer_ppo_epochs: int = 4
    designer_entropy_coef: float = 0.02
    max_designer_replay: int = 5_000

    # Benchmark
    benchmark_count: int = 8
    benchmark_total_limit: int = 40
    ood_rows: int = 6
    ood_cols: int = 6
    pretrained_designer_checkpoint: str | None = None
    skip_forgetting_benchmark: bool = False
    forgetting_only_on_promotion: bool = True

    # Optional held-out harder evaluation levels (JSONL of {level, state, ...}).
    # The manifest's promotion-validation role drives promotion; its final-test
    # role stays sealed for explicit post-training evaluation. All held-out
    # signatures are excluded from training.
    eval_levels_dataset: str | None = None
    eval_split_manifest: str | None = None
    eval_split_seed: int = DEFAULT_EVAL_SPLIT_SEED
    # ``None`` trusts the exact count persisted in the required manifest.
    eval_validation_count: int | None = None

    # Evaluation-only level signatures (e.g. the frozen final benchmark) that
    # must never enter protagonist training or replay.
    excluded_signatures: tuple[str, ...] = ()

    # Ablation toggles.
    reward: RewardConfig = field(default_factory=RewardConfig)
    curriculum_enabled: bool = True
    use_designer_replay: bool = True
    seed_historical_replay: bool = True
    # full exact -> exact root path -> neural search (default). ``hybrid``
    # preserves the legacy full-exact -> neural-search behavior.
    label_mode: str = LABEL_MODE_HYBRID_PATH
    train_designer_each_round: bool = True

    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    initial_curriculum: CurriculumState = field(default_factory=CurriculumState)

    def __post_init__(self) -> None:
        SplitRatios(
            train=1.0 - self.val_ratio - self.test_ratio,
            validation=self.val_ratio,
            test=self.test_ratio,
        ).validate()
        validate_positive_integer("co-training epochs", self.epochs)
        if self.trainable_part not in TRAINABLE_PARTS:
            choices = ", ".join(sorted(TRAINABLE_PARTS))
            raise ValueError(f"trainable part must be one of: {choices}")
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
        if self.policy_target_profile not in POLICY_TARGET_PROFILES:
            choices = ", ".join(sorted(POLICY_TARGET_PROFILES))
            raise ValueError(
                f"policy_target_profile must be one of: {choices}")
        if (self.policy_target_profile != "recorded"
                and self.trainable_part not in
                ("all", "policy_adapter", "policy_head")):
            raise ValueError(
                "incumbent-guided policy targets require an all-parameter "
                "or policy-only update")
        validate_positive_integer(
            "co-training solve_rate_trials", self.solve_rate_trials)
        validate_positive_integer(
            "co-training designer_validation_episodes",
            self.designer_validation_episodes)
        replay_fractions = (
            ("replay_current_fraction", self.replay_current_fraction),
            ("replay_recent_fraction", self.replay_recent_fraction),
            ("replay_historical_fraction", self.replay_historical_fraction),
        )
        for name, value in replay_fractions:
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(float(value)) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isclose(
                sum(float(value) for _name, value in replay_fractions),
                1.0,
                rel_tol=0.0,
                abs_tol=1e-6):
            raise ValueError("replay age fractions must sum to 1.0")
        if self.replay_current_fraction <= 0:
            raise ValueError("replay_current_fraction must be positive")
        validate_positive_integer(
            "replay_recent_window", self.replay_recent_window)
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
        for name, value in (
                ("frontier_backfill_target", self.frontier_backfill_target),
                ("min_fresh_levels_to_train",
                 self.min_fresh_levels_to_train)):
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value < 0):
                raise ValueError(f"{name} must be a non-negative integer")
        for name, value in (
                ("generation_checkpoint_interval",
                 self.generation_checkpoint_interval),
                ("generation_progress_interval",
                 self.generation_progress_interval)):
            if (isinstance(value, bool) or not isinstance(value, int)
                    or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        for name, value in (
                ("exploratory_astar_time_limit_seconds",
                 self.exploratory_astar_time_limit_seconds),
                ("training_astar_time_limit_seconds",
                 self.training_astar_time_limit_seconds)):
            if value is not None and (
                    isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(float(value)) or value <= 0):
                raise ValueError(
                    f"{name} must be None or a finite positive number")
        if (isinstance(self.designer_frontier_alignment_weight, bool)
                or not isinstance(
                    self.designer_frontier_alignment_weight, Real)
                or not math.isfinite(
                    float(self.designer_frontier_alignment_weight))
                or self.designer_frontier_alignment_weight < 0):
            raise ValueError(
                "designer_frontier_alignment_weight must be finite and "
                "non-negative")
        if (self.designer_frontier_alignment_weight > 0
                and self.solve_rate_trials < 2):
            raise ValueError(
                "designer frontier alignment requires at least two "
                "solve_rate_trials")
        if (isinstance(self.frontier_dirichlet_alpha, bool)
                or not isinstance(self.frontier_dirichlet_alpha, Real)
                or not math.isfinite(float(self.frontier_dirichlet_alpha))
                or self.frontier_dirichlet_alpha <= 0):
            raise ValueError(
                "frontier_dirichlet_alpha must be finite and positive")
        if (isinstance(self.frontier_dirichlet_weight, bool)
                or not isinstance(self.frontier_dirichlet_weight, Real)
                or not math.isfinite(float(self.frontier_dirichlet_weight))
                or not 0 < self.frontier_dirichlet_weight <= 1):
            raise ValueError(
                "frontier_dirichlet_weight must be a finite rate in (0, 1]")
        for name, value in (
                ("frontier_budget_min_ratio",
                 self.frontier_budget_min_ratio),
                ("frontier_budget_max_ratio",
                 self.frontier_budget_max_ratio)):
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(float(value)) or value <= 0):
                raise ValueError(
                    f"{name} must be finite and positive")
        if self.frontier_budget_min_ratio > self.frontier_budget_max_ratio:
            raise ValueError(
                "frontier_budget_min_ratio cannot exceed "
                "frontier_budget_max_ratio")
        if self.eval_limit is not None:
            validate_positive_integer("co-training eval_limit", self.eval_limit)
        if (isinstance(self.promotion_margin, bool)
                or not isinstance(self.promotion_margin, Real)
                or not math.isfinite(float(self.promotion_margin))
                or self.promotion_margin < 0):
            raise ValueError("promotion_margin must be finite and non-negative")
        if not isinstance(self.promotion_paired_gate_enabled, bool):
            raise ValueError("promotion_paired_gate_enabled must be boolean")
        if not isinstance(self.stop_after_promotion, bool):
            raise ValueError("stop_after_promotion must be boolean")
        if (isinstance(self.promotion_max_per_budget_regression, bool)
                or not isinstance(
                    self.promotion_max_per_budget_regression, Real)
                or not math.isfinite(float(
                    self.promotion_max_per_budget_regression))
                or not 0 <= self.promotion_max_per_budget_regression <= 1):
            raise ValueError(
                "promotion_max_per_budget_regression must be finite and in "
                "[0, 1]")
        if (isinstance(self.promotion_bootstrap_confidence, bool)
                or not isinstance(self.promotion_bootstrap_confidence, Real)
                or not math.isfinite(float(
                    self.promotion_bootstrap_confidence))
                or not 0 < self.promotion_bootstrap_confidence < 1):
            raise ValueError(
                "promotion_bootstrap_confidence must be finite and in (0, 1)")
        validate_positive_integer(
            "promotion_bootstrap_replicates",
            self.promotion_bootstrap_replicates)
        if (isinstance(self.promotion_bootstrap_seed, bool)
                or not isinstance(self.promotion_bootstrap_seed, int)):
            raise ValueError("promotion_bootstrap_seed must be an integer")
        if (isinstance(self.learner_milestone_interval, bool)
                or not isinstance(self.learner_milestone_interval, int)
                or self.learner_milestone_interval <= 0):
            raise ValueError(
                "learner_milestone_interval must be a positive integer")
        if (self.initial_learner_checkpoint
                and not self.shadow_learner_enabled):
            raise ValueError(
                "initial_learner_checkpoint requires shadow_learner_enabled")
        if (isinstance(self.learner_max_policy_kl, bool)
                or not isinstance(self.learner_max_policy_kl, Real)
                or not math.isfinite(float(self.learner_max_policy_kl))
                or self.learner_max_policy_kl <= 0):
            raise ValueError(
                "learner_max_policy_kl must be finite and positive")
        if (isinstance(self.learner_min_entropy_ratio, bool)
                or not isinstance(self.learner_min_entropy_ratio, Real)
                or not math.isfinite(float(self.learner_min_entropy_ratio))
                or not 0 <= self.learner_min_entropy_ratio <= 1):
            raise ValueError(
                "learner_min_entropy_ratio must be finite and in [0, 1]")
        if self.learner_retention_dataset and not self.shadow_learner_enabled:
            raise ValueError(
                "learner_retention_dataset requires shadow_learner_enabled")
        if self.learner_retention_enforce and not self.learner_retention_dataset:
            raise ValueError(
                "learner_retention_enforce requires a retention dataset")
        if not isinstance(self.learner_retention_enforce, bool):
            raise ValueError("learner_retention_enforce must be boolean")
        if not isinstance(self.learner_retention_use_full_pool, bool):
            raise ValueError(
                "learner_retention_use_full_pool must be boolean")
        if (self.learner_retention_use_full_pool
                and not self.learner_retention_dataset):
            raise ValueError(
                "learner_retention_use_full_pool requires a retention "
                "dataset")
        if (isinstance(self.learner_retention_per_band, bool)
                or not isinstance(self.learner_retention_per_band, int)
                or self.learner_retention_per_band <= 0):
            raise ValueError(
                "learner_retention_per_band must be a positive integer")
        if not self.learner_retention_budgets or any(
                isinstance(value, bool) or not isinstance(value, int)
                or value <= 0 for value in self.learner_retention_budgets):
            raise ValueError(
                "learner_retention_budgets must contain positive integers")
        if len(set(self.learner_retention_budgets)) != len(
                self.learner_retention_budgets):
            raise ValueError("learner_retention_budgets must be unique")
        if (isinstance(self.learner_retention_max_regression, bool)
                or not isinstance(self.learner_retention_max_regression, Real)
                or not math.isfinite(float(
                    self.learner_retention_max_regression))
                or not 0 <= self.learner_retention_max_regression <= 1):
            raise ValueError(
                "learner_retention_max_regression must be finite and in "
                "[0, 1]")
        if isinstance(self.eval_split_seed, bool) \
                or not isinstance(self.eval_split_seed, int):
            raise ValueError("eval_split_seed must be an integer")
        if self.eval_validation_count is not None:
            validate_positive_integer(
                "eval_validation_count", self.eval_validation_count)
        if bool(self.eval_levels_dataset) != bool(self.eval_split_manifest):
            raise ValueError(
                "eval_levels_dataset and eval_split_manifest must be provided "
                "together; held-out membership is never recomputed implicitly")
        if promotion_metric_requires_budget_sweep(self.promotion_metric):
            if (self.promotion_metric ==
                    "budget_sweep_confirmed_optimal_rate" and
                    self.promotion_budget_weights):
                raise ValueError(
                    "promotion_budget_weights require the weighted budget-sweep "
                    "promotion metric")
            budgets, weights = normalize_budget_sweep_config(
                self.promotion_budgets or None,
                self.promotion_budget_weights or None,
            )
            if self.promotion_metric == "budget_sweep_confirmed_optimal_rate":
                weights = tuple(1.0 / len(budgets) for _ in budgets)
            object.__setattr__(self, "promotion_budgets", budgets)
            object.__setattr__(self, "promotion_budget_weights", weights)
        elif self.promotion_budgets or self.promotion_budget_weights:
            raise ValueError(
                "promotion_budgets and promotion_budget_weights are only valid "
                "with a budget-sweep promotion metric")
        if (self.promotion_paired_gate_enabled
                and self.promotion_metric !=
                "weighted_budget_sweep_solve_rate"):
            raise ValueError(
                "promotion_paired_gate_enabled requires "
                "weighted_budget_sweep_solve_rate")
        object.__setattr__(
            self,
            "eval_budgets",
            effective_eval_budgets(
                self.eval_budgets,
                promotion_metric=self.promotion_metric,
                promotion_budget=self.promotion_budget,
                promotion_budgets=self.promotion_budgets,
            ),
        )

    def to_dict(self) -> dict:
        skip = ("curriculum", "initial_curriculum", "reward")
        d = {k: v for k, v in asdict(self).items() if k not in skip}
        d["eval_budgets"] = list(self.eval_budgets)
        d["promotion_budgets"] = list(self.promotion_budgets)
        d["promotion_budget_weights"] = list(self.promotion_budget_weights)
        d["reward"] = asdict(self.reward)
        d["curriculum"] = asdict(self.curriculum)
        d["initial_curriculum"] = self.initial_curriculum.to_dict()
        return d
