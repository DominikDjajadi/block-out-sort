"""Multi-seed designer-only calibration for frontier generation.

This freezes one protagonist, trains independent designer attempts, and then
measures each selected designer on a separate generated holdout batch. Completed
seed attempts are reusable, while incomplete attempts are preserved and a new
attempt directory is created on restart.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..designer.actions import DesignerActionSpace
from ..designer.checkpoint import designer_from_checkpoint, load_designer
from ..designer.config import GeneratorConfig
from ..designer.env import DesignerEnv
from ..designer.ppo import PPOConfig, rollout_episode
from ..designer.replay import level_fingerprint
from ..designer.roles import Protagonist
from ..designer.train import TrainConfig, train_designer
from ..training.checkpoint import (
    configs_from_checkpoint, load_checkpoint, model_from_checkpoint)
from ..training.transaction import (
    atomic_write_json, atomic_write_text, sha256_file)
from .frontier import (
    estimate_solve_rate, geometric_budget_sweep, in_frontier)
from .config import CurriculumConfig


_CALIBRATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CalibrationConfig:
    protagonist_checkpoint: str
    designer_checkpoint: str
    output_dir: str
    seeds: tuple[int, ...] = (2036, 2037, 2038)
    episodes: int = 48
    episodes_per_iter: int = 12
    validation_episodes: int = 32
    holdout_levels: int = 100
    mutation_budget: int = 6
    protagonist_simulations: int = 220
    oracle_simulations: int = 1000
    astar_max_nodes: int = 200_000
    astar_time_limit_seconds: float | None = 5.0
    solve_rate_trials: int = 5
    frontier_min_solve_rate: float = 0.2
    frontier_max_solve_rate: float = 0.7
    frontier_alignment_weight: float = 1.0
    frontier_dirichlet_alpha: float = 0.5
    frontier_dirichlet_weight: float = 0.4
    frontier_budget_min_ratio: float = 0.25
    frontier_budget_max_ratio: float = 4.0
    frontier_min_simulations: int = 20
    frontier_max_simulations: int = 400
    min_validation_frontier_rate: float = 0.10
    min_holdout_frontier_rate: float = 0.10
    rows: int = 6
    cols: int = 6
    color_count: int = 3
    density: float = 0.4
    designer_ppo_epochs: int = 4
    designer_entropy_coef: float = 0.02
    max_designer_replay: int = 5_000
    device: str = "auto"

    def validate(self) -> None:
        if not self.seeds or any(
                isinstance(seed, bool) or not isinstance(seed, int)
                for seed in self.seeds):
            raise ValueError("calibration seeds must be non-empty integers")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("calibration seeds must be unique")
        for name, value in (
                ("episodes", self.episodes),
                ("episodes_per_iter", self.episodes_per_iter),
                ("validation_episodes", self.validation_episodes),
                ("holdout_levels", self.holdout_levels),
                ("mutation_budget", self.mutation_budget),
                ("protagonist_simulations", self.protagonist_simulations),
                ("solve_rate_trials", self.solve_rate_trials)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"calibration {name} must be a positive integer")
        if self.solve_rate_trials < 2:
            raise ValueError(
                "calibration solve_rate_trials must be at least two")
        for name, value in (
                ("frontier_min_solve_rate", self.frontier_min_solve_rate),
                ("frontier_max_solve_rate", self.frontier_max_solve_rate),
                ("min_validation_frontier_rate",
                 self.min_validation_frontier_rate),
                ("min_holdout_frontier_rate",
                 self.min_holdout_frontier_rate)):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"calibration {name} must be in [0, 1]")
        if self.frontier_min_solve_rate > self.frontier_max_solve_rate:
            raise ValueError(
                "calibration frontier minimum cannot exceed maximum")
        if self.frontier_alignment_weight <= 0:
            raise ValueError(
                "calibration frontier_alignment_weight must be positive")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self)))


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _next_attempt(seed_root: Path) -> Path:
    existing = sorted(seed_root.glob("attempt_*"))
    number = (
        max(int(path.name.split("_")[-1]) for path in existing) + 1
        if existing else 1
    )
    return seed_root / f"attempt_{number:03d}"


def _completed_attempt(seed_root: Path) -> dict[str, Any] | None:
    completed = sorted(
        seed_root.glob("attempt_*/calibration_result.json"), reverse=True)
    if not completed:
        return None
    path = completed[0]
    result = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = Path(result["designer_checkpoint"])
    if not checkpoint.is_file():
        raise RuntimeError(
            f"completed calibration checkpoint is missing: {checkpoint}")
    observed = sha256_file(checkpoint)
    if observed != result["designer_checkpoint_sha256"]:
        raise RuntimeError(
            f"completed calibration checkpoint hash mismatch: {checkpoint}")
    return result


def _evaluate_holdout(
    cfg: CalibrationConfig,
    *,
    seed: int,
    designer_checkpoint: str,
) -> dict[str, Any]:
    device = _resolve_device(cfg.device)
    protagonist_checkpoint = load_checkpoint(
        cfg.protagonist_checkpoint, map_location="cpu")
    encoding, _model_cfg, value_norm = configs_from_checkpoint(
        protagonist_checkpoint)
    protagonist_model = model_from_checkpoint(
        protagonist_checkpoint, map_location=device)
    protagonist = Protagonist(
        DesignerEnv(
            GeneratorConfig(
                rows=cfg.rows, cols=cfg.cols, color_count=cfg.color_count,
                density=cfg.density),
            mutation_budget=cfg.mutation_budget,
            encoding=encoding,
        ).env,
        protagonist_model,
        encoding,
        value_norm,
        device,
        simulations=cfg.protagonist_simulations,
        dirichlet_alpha=cfg.frontier_dirichlet_alpha,
        dirichlet_weight=cfg.frontier_dirichlet_weight,
    )
    designer_model, designer_encoding, _designer_model_cfg = (
        designer_from_checkpoint(
            load_designer(designer_checkpoint, map_location=device),
            map_location=device))
    environment = DesignerEnv(
        GeneratorConfig(
            rows=cfg.rows, cols=cfg.cols, color_count=cfg.color_count,
            density=cfg.density),
        mutation_budget=cfg.mutation_budget,
        encoding=encoding,
    )
    action_space = DesignerActionSpace(encoding)
    budgets = geometric_budget_sweep(
        center=cfg.protagonist_simulations,
        trials=cfg.solve_rate_trials,
        minimum_ratio=cfg.frontier_budget_min_ratio,
        maximum_ratio=cfg.frontier_budget_max_ratio,
        minimum_simulations=cfg.frontier_min_simulations,
        maximum_simulations=cfg.frontier_max_simulations,
    )
    frontier_cfg = CurriculumConfig(
        frontier_min_solve_rate=cfg.frontier_min_solve_rate,
        frontier_max_solve_rate=cfg.frontier_max_solve_rate)
    rng = random.Random(seed ^ 0xCA11B4A7)
    rates: list[float] = []
    fingerprints: set[str] = set()
    invalid = strict = below = above = 0
    designer_model.eval()
    with torch.no_grad():
        for index in range(cfg.holdout_levels):
            episode_seed = seed * 1_000_003 + 9_000_001 + index
            episode = rollout_episode(
                environment, designer_model, action_space,
                designer_encoding, seed=episode_seed, device=device,
                rng=rng, verify_finalize=False)
            if not episode.finalize.valid:
                invalid += 1
                continue
            fingerprints.add(
                level_fingerprint(environment.env, episode.finalize.level))
            estimate = estimate_solve_rate(
                protagonist,
                episode.finalize.level,
                trials=cfg.solve_rate_trials,
                base_seed=episode_seed,
                evaluation_context="designer.calibration.holdout",
                simulation_budgets=budgets,
            )
            rate = estimate.solve_rate
            rates.append(rate)
            if in_frontier(rate, frontier_cfg):
                strict += 1
            elif rate < cfg.frontier_min_solve_rate:
                below += 1
            else:
                above += 1
    valid = len(rates)
    distribution = {
        f"{solved}/{cfg.solve_rate_trials}": sum(
            1 for rate in rates
            if abs(rate - solved / cfg.solve_rate_trials) <= 1e-12)
        for solved in range(cfg.solve_rate_trials + 1)
    }
    return {
        "levels": cfg.holdout_levels,
        "valid_count": valid,
        "invalid_count": invalid,
        "construction_proven_solvable_count": valid,
        "unique_count": len(fingerprints),
        "strict_frontier_count": strict,
        "strict_frontier_rate": strict / valid if valid else 0.0,
        "below_frontier_count": below,
        "above_frontier_count": above,
        "mean_solve_rate": sum(rates) / valid if valid else 0.0,
        "solve_rate_distribution": distribution,
        "simulation_budgets": list(budgets),
    }


def _training_config(
    cfg: CalibrationConfig,
    *,
    seed: int,
    output_dir: Path,
) -> TrainConfig:
    return TrainConfig(
        protagonist_checkpoint=cfg.protagonist_checkpoint,
        output_dir=str(output_dir),
        init_designer=cfg.designer_checkpoint,
        episodes=cfg.episodes,
        episodes_per_iter=cfg.episodes_per_iter,
        validation_episodes=cfg.validation_episodes,
        mutation_budget=cfg.mutation_budget,
        protagonist_simulations=cfg.protagonist_simulations,
        oracle_simulations=cfg.oracle_simulations,
        astar_max_nodes=cfg.astar_max_nodes,
        astar_time_limit_seconds=cfg.astar_time_limit_seconds,
        frontier_solve_rate_trials=cfg.solve_rate_trials,
        frontier_min_solve_rate=cfg.frontier_min_solve_rate,
        frontier_max_solve_rate=cfg.frontier_max_solve_rate,
        frontier_alignment_weight=cfg.frontier_alignment_weight,
        frontier_dirichlet_alpha=cfg.frontier_dirichlet_alpha,
        frontier_dirichlet_weight=cfg.frontier_dirichlet_weight,
        frontier_budget_min_ratio=cfg.frontier_budget_min_ratio,
        frontier_budget_max_ratio=cfg.frontier_budget_max_ratio,
        frontier_min_simulations=cfg.frontier_min_simulations,
        frontier_max_simulations=cfg.frontier_max_simulations,
        seed=seed,
        device=cfg.device,
        max_replay=cfg.max_designer_replay,
        generator=GeneratorConfig(
            rows=cfg.rows, cols=cfg.cols, color_count=cfg.color_count,
            density=cfg.density),
        ppo=PPOConfig(
            epochs=cfg.designer_ppo_epochs,
            entropy_coef=cfg.designer_entropy_coef),
    )


def _run_seed(cfg: CalibrationConfig, seed: int) -> dict[str, Any]:
    seed_root = Path(cfg.output_dir) / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    completed = _completed_attempt(seed_root)
    if completed is not None:
        print(f"seed {seed}: reusing completed calibration", flush=True)
        return completed

    attempt = _next_attempt(seed_root)
    training = train_designer(
        _training_config(cfg, seed=seed, output_dir=attempt / "designer"))
    checkpoint = training["best_checkpoint"]
    holdout = _evaluate_holdout(
        cfg, seed=seed, designer_checkpoint=checkpoint)
    validation = training["best_validation_metrics"]
    validation_rate = float(validation["frontier_in_band_rate"])
    holdout_rate = float(holdout["strict_frontier_rate"])
    passed = (
        validation_rate >= cfg.min_validation_frontier_rate
        and holdout_rate >= cfg.min_holdout_frontier_rate
        and int(holdout["invalid_count"]) == 0
    )
    result = {
        "schema_version": _CALIBRATION_SCHEMA_VERSION,
        "seed": seed,
        "attempt": attempt.name,
        "designer_checkpoint": checkpoint,
        "designer_checkpoint_sha256": sha256_file(checkpoint),
        "validation": validation,
        "selection_metric": training["best_selection_metric"],
        "training_frontier_simulation_budgets":
            training["frontier_simulation_budgets"],
        "holdout": holdout,
        "thresholds": {
            "min_validation_frontier_rate":
                cfg.min_validation_frontier_rate,
            "min_holdout_frontier_rate": cfg.min_holdout_frontier_rate,
            "require_zero_invalid": True,
        },
        "passed": passed,
    }
    atomic_write_json(attempt / "calibration_result.json", result)
    return result


def run_calibration(cfg: CalibrationConfig) -> dict[str, Any]:
    cfg.validate()
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    requested = cfg.to_dict()
    if config_path.exists():
        persisted = json.loads(config_path.read_text(encoding="utf-8"))
        if persisted != requested:
            raise RuntimeError(
                "calibration output directory belongs to different settings")
    else:
        atomic_write_json(config_path, requested)

    results = []
    for seed in cfg.seeds:
        print(f"=== designer calibration seed {seed} ===", flush=True)
        results.append(_run_seed(cfg, seed))
    passed = all(result["passed"] for result in results)
    summary = {
        "schema_version": _CALIBRATION_SCHEMA_VERSION,
        "seeds": list(cfg.seeds),
        "passed_seed_count": sum(1 for result in results if result["passed"]),
        "required_seed_count": len(results),
        "passed": passed,
        "decision": "go" if passed else "no_go",
        "results": results,
    }
    atomic_write_json(root / "summary.json", summary)
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=[
        "seed", "passed", "validation_frontier_rate",
        "validation_alignment", "holdout_frontier_rate",
        "holdout_frontier_count", "holdout_valid_count",
        "holdout_invalid_count", "mean_holdout_solve_rate"])
    writer.writeheader()
    for result in results:
        validation = result["validation"]
        holdout = result["holdout"]
        writer.writerow({
            "seed": result["seed"],
            "passed": result["passed"],
            "validation_frontier_rate":
                validation["frontier_in_band_rate"],
            "validation_alignment":
                validation["mean_frontier_alignment"],
            "holdout_frontier_rate": holdout["strict_frontier_rate"],
            "holdout_frontier_count": holdout["strict_frontier_count"],
            "holdout_valid_count": holdout["valid_count"],
            "holdout_invalid_count": holdout["invalid_count"],
            "mean_holdout_solve_rate": holdout["mean_solve_rate"],
        })
    atomic_write_text(root / "summary.csv", handle.getvalue())
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-seed designer-only frontier calibration")
    parser.add_argument("--protagonist-checkpoint", required=True)
    parser.add_argument("--designer-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[2036, 2037, 2038])
    parser.add_argument("--episodes", type=int, default=48)
    parser.add_argument("--episodes-per-iter", type=int, default=12)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--holdout-levels", type=int, default=100)
    parser.add_argument("--mutation-budget", type=int, default=6)
    parser.add_argument("--protagonist-simulations", type=int, default=220)
    parser.add_argument("--oracle-simulations", type=int, default=1000)
    parser.add_argument("--astar-max-nodes", type=int, default=200_000)
    parser.add_argument("--astar-time-limit-seconds", type=float, default=5.0)
    parser.add_argument("--solve-rate-trials", type=int, default=5)
    parser.add_argument("--frontier-min-solve-rate", type=float, default=0.2)
    parser.add_argument("--frontier-max-solve-rate", type=float, default=0.7)
    parser.add_argument("--frontier-alignment-weight", type=float, default=1.0)
    parser.add_argument("--frontier-dirichlet-alpha", type=float, default=0.5)
    parser.add_argument("--frontier-dirichlet-weight", type=float, default=0.4)
    parser.add_argument("--frontier-budget-min-ratio", type=float, default=0.25)
    parser.add_argument("--frontier-budget-max-ratio", type=float, default=4.0)
    parser.add_argument("--frontier-min-simulations", type=int, default=20)
    parser.add_argument("--frontier-max-simulations", type=int, default=400)
    parser.add_argument("--min-validation-frontier-rate",
                        type=float, default=0.10)
    parser.add_argument("--min-holdout-frontier-rate",
                        type=float, default=0.10)
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=6)
    parser.add_argument("--color-count", type=int, default=3)
    parser.add_argument("--density", type=float, default=0.4)
    parser.add_argument("--designer-ppo-epochs", type=int, default=4)
    parser.add_argument("--designer-entropy-coef", type=float, default=0.02)
    parser.add_argument("--max-designer-replay", type=int, default=5_000)
    parser.add_argument("--device", default="auto")
    return parser


def config_from_args(args: argparse.Namespace) -> CalibrationConfig:
    return CalibrationConfig(
        protagonist_checkpoint=args.protagonist_checkpoint,
        designer_checkpoint=args.designer_checkpoint,
        output_dir=args.output_dir,
        seeds=tuple(args.seeds),
        episodes=args.episodes,
        episodes_per_iter=args.episodes_per_iter,
        validation_episodes=args.validation_episodes,
        holdout_levels=args.holdout_levels,
        mutation_budget=args.mutation_budget,
        protagonist_simulations=args.protagonist_simulations,
        oracle_simulations=args.oracle_simulations,
        astar_max_nodes=args.astar_max_nodes,
        astar_time_limit_seconds=(
            None if args.astar_time_limit_seconds <= 0
            else args.astar_time_limit_seconds),
        solve_rate_trials=args.solve_rate_trials,
        frontier_min_solve_rate=args.frontier_min_solve_rate,
        frontier_max_solve_rate=args.frontier_max_solve_rate,
        frontier_alignment_weight=args.frontier_alignment_weight,
        frontier_dirichlet_alpha=args.frontier_dirichlet_alpha,
        frontier_dirichlet_weight=args.frontier_dirichlet_weight,
        frontier_budget_min_ratio=args.frontier_budget_min_ratio,
        frontier_budget_max_ratio=args.frontier_budget_max_ratio,
        frontier_min_simulations=args.frontier_min_simulations,
        frontier_max_simulations=args.frontier_max_simulations,
        min_validation_frontier_rate=args.min_validation_frontier_rate,
        min_holdout_frontier_rate=args.min_holdout_frontier_rate,
        rows=args.rows,
        cols=args.cols,
        color_count=args.color_count,
        density=args.density,
        designer_ppo_epochs=args.designer_ppo_epochs,
        designer_entropy_coef=args.designer_entropy_coef,
        max_designer_replay=args.max_designer_replay,
        device=args.device,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_calibration(config_from_args(args))
    print(
        f"\ndesigner calibration decision: {summary['decision']} "
        f"({summary['passed_seed_count']}/"
        f"{summary['required_seed_count']} seeds passed)",
        flush=True)


if __name__ == "__main__":
    main()
