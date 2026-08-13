"""CLI: run hybrid expert iteration.

Example::

    python -m blocksort.expert_iteration.run \\
        --initial-checkpoint runs/pv_smoke/best.pt \\
        --base-dataset data/training/pv_smoke.jsonl \\
        --output-dir runs/expert_iteration \\
        --iterations 3 --levels-per-iteration 100 \\
        --astar-max-nodes 300000 --search-simulations 800 --seed 42

Re-running with the same ``--output-dir`` resumes: completed iterations are
skipped and the loop continues from the current best checkpoint.
"""

from __future__ import annotations

import argparse

from .config import ExpertIterationConfig
from .iterate import run_expert_iteration
from .labeling import LABEL_MODES, LABEL_MODE_HYBRID_PATH


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hybrid expert iteration for Block Out Sort")
    p.add_argument("--initial-checkpoint", required=True)
    p.add_argument("--base-dataset", required=True)
    p.add_argument("--output-dir", default="runs/expert_iteration")
    p.add_argument("--extra-levels", default=None,
                   help="optional JSON pool of procedurally generated levels")
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--levels-per-iteration", type=int, default=100)
    p.add_argument("--states-per-level", type=int, default=8)
    p.add_argument("--astar-max-nodes", type=int, default=300_000)
    p.add_argument("--search-simulations", type=int, default=800)
    p.add_argument("--search-c-puct", type=float, default=1.5)
    p.add_argument("--label-policy-temperature", type=float, default=1.0)
    p.add_argument(
        "--label-mode", choices=LABEL_MODES, default=LABEL_MODE_HYBRID_PATH,
        help=(
            "hybrid_path keeps a proven root path when successor proofs "
            "exhaust (default); hybrid uses legacy neural fallback; "
            "search_only disables A* labeling"),
    )
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--max-replay-examples", type=int, default=50_000)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--train-sample-size", type=int, default=4_000)
    p.add_argument(
        "--weight-exact-historical", type=float, default=1.0,
        help="replay-sampling preference for historical exact records")
    p.add_argument(
        "--weight-exact-new", type=float, default=1.5,
        help="replay-sampling preference for current-iteration exact records")
    p.add_argument(
        "--weight-search", type=float, default=0.5,
        help="replay-sampling preference for search-derived records")
    p.add_argument(
        "--exact-path-policy-confidence", type=float, default=0.5,
        help="policy-loss multiplier for incomplete exact-path demonstrations")
    p.add_argument(
        "--search-value-loss-weight", type=float, default=0.0,
        help="scalar value-loss confidence for approximate search labels; "
             "zero keeps policy supervision only")
    p.add_argument(
        "--policy-anchor-weight", type=float, default=0.0,
        help="KL penalty preserving the incumbent policy on historical replay")
    p.add_argument("--eval-budgets", type=int, nargs="+", default=[1, 100, 400])
    p.add_argument("--eval-limit", type=int, default=None)
    p.add_argument(
        "--promotion-metric", default="search_confirmed_optimal_rate",
        help="larger-is-better validation metric; conditional accuracies require "
             "full coverage")
    p.add_argument("--promotion-budget", type=int, default=400)
    p.add_argument("--promotion-margin", type=float, default=0.0)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p


def config_from_args(args: argparse.Namespace) -> ExpertIterationConfig:
    return ExpertIterationConfig(
        initial_checkpoint=args.initial_checkpoint,
        base_dataset=args.base_dataset,
        output_dir=args.output_dir,
        extra_levels=args.extra_levels,
        iterations=args.iterations,
        levels_per_iteration=args.levels_per_iteration,
        states_per_level=args.states_per_level,
        astar_max_nodes=args.astar_max_nodes,
        search_simulations=args.search_simulations,
        search_c_puct=args.search_c_puct,
        label_policy_temperature=args.label_policy_temperature,
        label_mode=args.label_mode,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        max_replay_examples=args.max_replay_examples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        train_sample_size=args.train_sample_size,
        weight_exact_historical=args.weight_exact_historical,
        weight_exact_new=args.weight_exact_new,
        weight_search=args.weight_search,
        exact_path_policy_confidence=args.exact_path_policy_confidence,
        search_value_loss_weight=args.search_value_loss_weight,
        policy_anchor_weight=args.policy_anchor_weight,
        eval_budgets=tuple(args.eval_budgets),
        eval_limit=args.eval_limit,
        promotion_metric=args.promotion_metric,
        promotion_budget=args.promotion_budget,
        promotion_margin=args.promotion_margin,
        device=args.device,
        seed=args.seed,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    result = run_expert_iteration(cfg)
    promoted = sum(1 for h in result["run_state"]["history"] if h["promoted"])
    print(f"\nfinished {len(result['run_state']['completed_iterations'])} "
          f"iteration(s); {promoted} promoted; "
          f"best={result['run_state']['best_checkpoint']}")


if __name__ == "__main__":
    main()
