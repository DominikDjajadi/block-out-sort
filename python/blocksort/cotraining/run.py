"""Alternating protagonist-designer co-training CLI.

    python -m blocksort.cotraining.run \\
        --protagonist-checkpoint runs/pv/best.pt \\
        --designer-checkpoint runs/designer/best.pt \\
        --base-dataset data/training/pv_examples.jsonl \\
        --output-dir runs/cotraining \\
        --rounds 3 --levels-per-round 100 \\
        --frontier-min-solve-rate 0.2 --frontier-max-solve-rate 0.7 --seed 42

Re-running the same command resumes from the last completed round (round state,
replay buffers, frozen split, and frozen benchmark are all persisted on disk).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import CoTrainingConfig, CurriculumConfig, CurriculumState
from .loop import run_cotraining
from .policy_targets import (
    DEFAULT_POLICY_TARGET_PROFILE,
    POLICY_TARGET_PROFILES,
)
from ..expert_iteration.labeling import LABEL_MODES, LABEL_MODE_HYBRID_PATH


_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_HARD_EVAL_DATASET = str(
    _REPOSITORY_ROOT / "data" / "eval" / "hard_pool_v1_20260723.jsonl")
_DEFAULT_HARD_EVAL_SPLIT = str(
    _REPOSITORY_ROOT / "data" / "eval" /
    "hard_pool_v1_20260723_split.json")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Protagonist-designer co-training")
    p.add_argument("--protagonist-checkpoint", required=True)
    p.add_argument("--designer-checkpoint", required=True)
    p.add_argument("--base-dataset", required=True,
                   help="handcrafted dataset for the frozen validation/test split "
                        "and the handcrafted benchmark group")
    p.add_argument(
        "--initial-learner-checkpoint",
        default=None,
        help="optional cumulative learner imported into a fresh shadow-learner "
             "run without replacing the official champion",
    )
    p.add_argument(
        "--initial-base-split",
        default=None,
        help="frozen base-dataset split manifest imported into a new run",
    )
    p.add_argument(
        "--initial-protagonist-replay",
        default=None,
        help="committed protagonist replay snapshot imported into a new run",
    )
    p.add_argument(
        "--initial-designer-replay",
        default=None,
        help="committed designer level-replay snapshot imported into a new run",
    )
    p.add_argument("--output-dir", default="runs/cotraining")
    p.add_argument(
        "--initialize-only",
        action="store_true",
        help="commit and verify initial checkpoints/replays, then stop before "
             "benchmark construction or round execution",
    )
    p.add_argument(
        "--prune-superseded-round-artifacts",
        action="store_true",
        help="after each durable round commit, remove superseded replay "
             "snapshots and non-milestone training-sample copies",
    )
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument(
        "--stop-after-promotion",
        action="store_true",
        help="durably commit the first promoted milestone, then stop before "
             "performing any later training round",
    )
    p.add_argument("--levels-per-round", type=int, default=100)
    p.add_argument("--frontier-min-solve-rate", type=float, default=0.2)
    p.add_argument("--frontier-max-solve-rate", type=float, default=0.7)
    p.add_argument("--min-frontier-acceptance-rate", type=float, default=0.1)
    p.add_argument("--frontier-imbalance-margin", type=float, default=0.05)
    p.add_argument(
        "--frontier-backfill-target", type=int, default=10,
        help="minimum usable levels retained per round by ranking otherwise "
             "eligible levels nearest to the strict frontier")
    p.add_argument("--solve-rate-trials", type=int, default=5)
    p.add_argument("--frontier-budget-min-ratio", type=float, default=0.25)
    p.add_argument("--frontier-budget-max-ratio", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")

    # Initial curriculum
    p.add_argument("--rows", type=int, default=5)
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--color-count", type=int, default=3)
    p.add_argument("--density", type=float, default=0.35)
    p.add_argument("--mutation-budget", type=int, default=4)
    p.add_argument("--protagonist-simulations", type=int, default=100)
    p.add_argument("--structural-threshold", type=int, default=1)

    # Teacher / evaluation
    p.add_argument("--astar-max-nodes", type=int, default=200_000,
                   help="A* budget for training labels (exactness matters)")
    p.add_argument(
        "--exploratory-astar-time-limit-seconds", type=float, default=5.0,
        help="wall-clock cap for exact scoring of selected generated levels "
             "(0 = none)")
    p.add_argument(
        "--training-astar-time-limit-seconds", type=float, default=5.0,
        help="per-state exact-label search cap before neural fallback "
             "(0 = none)")
    p.add_argument("--eval-astar-max-nodes", type=int, default=20_000,
                   help="A* budget for promotion / benchmark evaluation")
    p.add_argument("--eval-astar-time-limit-seconds", type=float, default=3.0,
                   help="Per-state A* time cap during evaluation (0 = none)")
    p.add_argument("--oracle-simulations", type=int, default=1000)
    p.add_argument(
        "--label-mode", choices=LABEL_MODES, default=LABEL_MODE_HYBRID_PATH,
        help=(
            "hybrid_path keeps a proven root path when successor proofs "
            "exhaust (default); hybrid uses legacy neural fallback; "
            "search_only disables A* labeling"),
    )
    p.add_argument(
        "--eval-budgets", type=int, nargs="+", default=[1, 2, 4, 8, 16],
        help="search budgets evaluated on held-out levels")
    p.add_argument(
        "--eval-limit", type=int, default=None,
        help="Max distinct held-out levels and benchmark levels sampled per "
             "group (default: all held-out levels; benchmark-total-limit "
             "still applies)")
    p.add_argument("--benchmark-total-limit", type=int, default=40,
                   help="Global cap on benchmark states (stratified across groups)")
    p.add_argument("--pretrained-designer-checkpoint", default=None,
                   help="Optional BC designer for the pretrained_designer group")
    p.add_argument("--skip-forgetting-benchmark", action="store_true")
    p.add_argument("--forgetting-on-rejection", action="store_true",
                   help="Run forgetting even when the candidate is rejected")
    p.add_argument(
        "--promotion-metric", default="weighted_budget_sweep_solve_rate",
        help="larger-is-better validation metric (default: weighted solve rate "
             "across the 4/8/16 search-budget sweep)")
    p.add_argument("--promotion-budget", type=int, default=400)
    p.add_argument("--promotion-budgets", type=int, nargs="+", default=None,
                   help="budget list for budget-sweep promotion metrics")
    p.add_argument("--promotion-budget-weights", type=float, nargs="+",
                   default=None,
                   help="optional weights aligned with --promotion-budgets")
    p.add_argument("--promotion-margin", type=float, default=0.01)
    p.add_argument(
        "--promotion-paired-gate", action="store_true",
        help="require the inclusive margin, a per-budget regression guard, "
             "and a positive paired-bootstrap lower bound")
    p.add_argument(
        "--promotion-max-per-budget-regression", type=float, default=0.02)
    p.add_argument(
        "--promotion-bootstrap-confidence", type=float, default=0.95)
    p.add_argument(
        "--promotion-bootstrap-replicates", type=int, default=10_000)
    p.add_argument("--promotion-bootstrap-seed", type=int, default=8216)
    p.add_argument(
        "--shadow-learner", action="store_true",
        help="retain safe learner updates across rounds while keeping the "
             "official champion immutable until promotion")
    p.add_argument(
        "--learner-milestone-interval", type=int, default=5,
        help="round cadence for continuation-safety and promotion evaluation")
    p.add_argument(
        "--learner-max-policy-kl", type=float, default=0.25,
        help="maximum champion-to-candidate mean policy KL at a milestone")
    p.add_argument(
        "--learner-min-entropy-ratio", type=float, default=0.70,
        help="minimum candidate/anchor policy entropy ratio at a milestone")
    p.add_argument(
        "--learner-retention-dataset", default=None,
        help="frozen development JSONL with baseline difficulty strata used "
             "for milestone retention checks")
    p.add_argument(
        "--learner-retention-budgets", type=int, nargs="+",
        default=[64, 95, 128],
        help="search budgets used for difficulty-band retention checks")
    p.add_argument(
        "--learner-retention-per-band", type=int, default=20,
        help="deterministically selected levels per baseline difficulty band")
    p.add_argument(
        "--learner-retention-use-full-pool", action="store_true",
        help="use every level in an intentionally unequal stratified pool "
             "instead of truncating each band to one common count")
    p.add_argument(
        "--learner-retention-max-regression", type=float, default=0.05,
        help="largest allowed candidate solve-rate loss in any band/budget")
    p.add_argument(
        "--learner-retention-enforce", action="store_true",
        help="roll back a milestone learner when a retention band exceeds "
             "the preregistered regression tolerance")

    # Protagonist fine-tune
    p.add_argument(
        "--min-fresh-levels-to-train", type=int, default=10,
        help="skip protagonist fine-tuning below this many newly accepted "
             "levels (0 disables the guard)")
    p.add_argument("--states-per-level", type=int, default=4)
    p.add_argument("--train-sample-size", type=int, default=2000)
    p.add_argument(
        "--replay-sample-with-replacement", action="store_true",
        help="allow duplicate replay records before all eligible records have "
             "been sampled (legacy behavior)")
    p.add_argument(
        "--replay-current-fraction", type=float, default=0.35,
        help="training sample share reserved for the active round")
    p.add_argument(
        "--replay-recent-fraction", type=float, default=0.25,
        help="training sample share reserved for recent earlier rounds")
    p.add_argument(
        "--replay-historical-fraction", type=float, default=0.40,
        help="training sample share reserved for older/base replay")
    p.add_argument(
        "--replay-recent-window", type=int, default=2,
        help="number of earlier rounds included in the recent replay bucket")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=3e-5)
    p.add_argument(
        "--trainable-part",
        choices=("all", "policy_adapter", "policy_head", "value_head"),
        default="policy_head",
        help="network portion updated during protagonist fine-tuning",
    )
    p.add_argument(
        "--search-value-loss-weight", type=float, default=0.0,
        help="scalar value-loss confidence for approximate search labels; "
             "zero keeps policy supervision only")
    p.add_argument(
        "--exact-path-policy-confidence", type=float, default=0.5,
        help="policy-loss multiplier for incomplete exact-path demonstrations")
    p.add_argument(
        "--policy-anchor-weight", type=float, default=0.0,
        help="KL penalty preserving the incumbent policy on historical replay")
    p.add_argument(
        "--policy-target-profile",
        choices=sorted(POLICY_TARGET_PROFILES),
        default=DEFAULT_POLICY_TARGET_PROFILE,
        help=(
            "exact full-oracle policy target formulation; the default "
            "temperature-1 profile preserves optimal support and uses "
            "incumbent probabilities without repeated sharpening"),
    )
    p.add_argument("--max-protagonist-replay", type=int, default=50_000)
    p.add_argument("--generation-checkpoint-interval", type=int, default=5)
    p.add_argument("--generation-progress-interval", type=int, default=5)

    # Designer training per round
    p.add_argument("--designer-episodes", type=int, default=48)
    p.add_argument("--designer-episodes-per-iter", type=int, default=12)
    p.add_argument(
        "--designer-validation-episodes", type=int, default=8,
        help="fixed generated levels used to select each post-update designer "
             "checkpoint")
    p.add_argument(
        "--designer-frontier-alignment-weight", type=float, default=1.0,
        help="bonus weight for generated levels whose repeated protagonist "
             "solve rate lands in the configured frontier band")
    p.add_argument("--designer-ppo-epochs", type=int, default=4)
    p.add_argument("--designer-entropy-coef", type=float, default=0.02)
    p.add_argument("--max-designer-replay", type=int, default=5_000)

    # Benchmark
    p.add_argument("--benchmark-count", type=int, default=8)
    p.add_argument("--ood-rows", type=int, default=6)
    p.add_argument("--ood-cols", type=int, default=6)
    p.add_argument(
        "--eval-levels-dataset",
        default=_DEFAULT_HARD_EVAL_DATASET,
        help="JSONL of held-out levels for promotion (non-saturated eval); "
             "signatures are excluded from training (default: bundled frozen "
             "hard pool)",
    )
    p.add_argument(
        "--eval-split-manifest",
        default=_DEFAULT_HARD_EVAL_SPLIT,
        help="immutable split manifest for --eval-levels-dataset (default: "
             "bundled 60/60 promotion/final split)",
    )
    p.add_argument("--eval-split-seed", type=int, default=1729)
    p.add_argument("--eval-validation-count", type=int, default=60)
    p.add_argument(
        "--no-bundled-hard-eval",
        action="store_true",
        help="disable the bundled frozen hard pool and use only the base "
             "dataset's frozen validation split",
    )
    return p


def config_from_args(args: argparse.Namespace) -> CoTrainingConfig:
    curriculum = CurriculumConfig(
        frontier_min_solve_rate=args.frontier_min_solve_rate,
        frontier_max_solve_rate=args.frontier_max_solve_rate,
        min_frontier_acceptance_rate=args.min_frontier_acceptance_rate,
        frontier_imbalance_margin=args.frontier_imbalance_margin)
    initial = CurriculumState(
        rows=args.rows, cols=args.cols, color_count=args.color_count,
        density=args.density, mutation_budget=args.mutation_budget,
        protagonist_simulations=args.protagonist_simulations,
        structural_threshold=args.structural_threshold)
    promotion_budgets = tuple(args.promotion_budgets or ())
    promotion_budget_weights = tuple(args.promotion_budget_weights or ())
    if (args.promotion_metric == "weighted_budget_sweep_solve_rate"
            and not promotion_budgets and not promotion_budget_weights):
        promotion_budgets = (4, 8, 16)
        promotion_budget_weights = (0.2, 0.3, 0.5)
    return CoTrainingConfig(
        protagonist_checkpoint=args.protagonist_checkpoint,
        designer_checkpoint=args.designer_checkpoint,
        base_dataset=args.base_dataset,
        initial_learner_checkpoint=args.initial_learner_checkpoint,
        initial_base_split=args.initial_base_split,
        initial_protagonist_replay=args.initial_protagonist_replay,
        initial_designer_replay=args.initial_designer_replay,
        output_dir=args.output_dir, initialize_only=args.initialize_only,
        prune_superseded_round_artifacts=(
            args.prune_superseded_round_artifacts),
        rounds=args.rounds, stop_after_promotion=args.stop_after_promotion,
        levels_per_round=args.levels_per_round,
        seed=args.seed, device=args.device,
        solve_rate_trials=args.solve_rate_trials,
        frontier_budget_min_ratio=args.frontier_budget_min_ratio,
        frontier_budget_max_ratio=args.frontier_budget_max_ratio,
        frontier_backfill_target=args.frontier_backfill_target,
        astar_max_nodes=args.astar_max_nodes,
        exploratory_astar_time_limit_seconds=(
            None if args.exploratory_astar_time_limit_seconds <= 0
            else args.exploratory_astar_time_limit_seconds),
        training_astar_time_limit_seconds=(
            None if args.training_astar_time_limit_seconds <= 0
            else args.training_astar_time_limit_seconds),
        eval_astar_max_nodes=args.eval_astar_max_nodes,
        eval_astar_time_limit_seconds=(
            None if args.eval_astar_time_limit_seconds <= 0
            else args.eval_astar_time_limit_seconds),
        oracle_simulations=args.oracle_simulations,
        label_mode=args.label_mode,
        eval_budgets=tuple(args.eval_budgets), eval_limit=args.eval_limit,
        benchmark_total_limit=args.benchmark_total_limit,
        pretrained_designer_checkpoint=args.pretrained_designer_checkpoint,
        skip_forgetting_benchmark=args.skip_forgetting_benchmark,
        forgetting_only_on_promotion=not args.forgetting_on_rejection,
        promotion_metric=args.promotion_metric,
        promotion_budget=args.promotion_budget,
        promotion_budgets=promotion_budgets,
        promotion_budget_weights=promotion_budget_weights,
        promotion_margin=args.promotion_margin,
        promotion_paired_gate_enabled=args.promotion_paired_gate,
        promotion_max_per_budget_regression=(
            args.promotion_max_per_budget_regression),
        promotion_bootstrap_confidence=args.promotion_bootstrap_confidence,
        promotion_bootstrap_replicates=args.promotion_bootstrap_replicates,
        promotion_bootstrap_seed=args.promotion_bootstrap_seed,
        shadow_learner_enabled=args.shadow_learner,
        learner_milestone_interval=args.learner_milestone_interval,
        learner_max_policy_kl=args.learner_max_policy_kl,
        learner_min_entropy_ratio=args.learner_min_entropy_ratio,
        learner_retention_dataset=args.learner_retention_dataset,
        learner_retention_budgets=tuple(args.learner_retention_budgets),
        learner_retention_per_band=args.learner_retention_per_band,
        learner_retention_use_full_pool=(
            args.learner_retention_use_full_pool),
        learner_retention_max_regression=(
            args.learner_retention_max_regression),
        learner_retention_enforce=args.learner_retention_enforce,
        min_fresh_levels_to_train=args.min_fresh_levels_to_train,
        states_per_level=args.states_per_level,
        train_sample_size=args.train_sample_size, epochs=args.epochs,
        replay_sample_with_replacement=args.replay_sample_with_replacement,
        replay_current_fraction=args.replay_current_fraction,
        replay_recent_fraction=args.replay_recent_fraction,
        replay_historical_fraction=args.replay_historical_fraction,
        replay_recent_window=args.replay_recent_window,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        trainable_part=args.trainable_part,
        search_value_loss_weight=args.search_value_loss_weight,
        exact_path_policy_confidence=args.exact_path_policy_confidence,
        policy_anchor_weight=args.policy_anchor_weight,
        policy_target_profile=args.policy_target_profile,
        max_protagonist_replay=args.max_protagonist_replay,
        generation_checkpoint_interval=args.generation_checkpoint_interval,
        generation_progress_interval=args.generation_progress_interval,
        designer_episodes=args.designer_episodes,
        designer_episodes_per_iter=args.designer_episodes_per_iter,
        designer_validation_episodes=args.designer_validation_episodes,
        designer_frontier_alignment_weight=(
            args.designer_frontier_alignment_weight),
        designer_ppo_epochs=args.designer_ppo_epochs,
        designer_entropy_coef=args.designer_entropy_coef,
        max_designer_replay=args.max_designer_replay,
        benchmark_count=args.benchmark_count, ood_rows=args.ood_rows,
        ood_cols=args.ood_cols,
        eval_levels_dataset=(
            None if args.no_bundled_hard_eval else args.eval_levels_dataset),
        eval_split_manifest=(
            None if args.no_bundled_hard_eval else args.eval_split_manifest),
        eval_split_seed=args.eval_split_seed,
        eval_validation_count=(
            None if args.no_bundled_hard_eval
            else args.eval_validation_count),
        curriculum=curriculum, initial_curriculum=initial)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = run_cotraining(config_from_args(args))
    print(f"\nfinished {result['rounds']} co-training round(s); "
          f"output in {args.output_dir}")


if __name__ == "__main__":
    main()
