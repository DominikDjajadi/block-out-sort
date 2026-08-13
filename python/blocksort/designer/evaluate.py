"""Evaluate and compare level generators against the oracle + protagonist.

Compares the existing random reverse-construction generator, the (optional)
pre-adversarial designer, and the trained adversarial designer on a freshly
generated benchmark (a disjoint seed range, written to disk so it is frozen and
reproducible). Reports validity, oracle-confirmed solvability, protagonist/oracle
solve rates, adversarial regret, structural metrics, duplicate rate, generation
time, and reward-component distributions.

    python -m blocksort.designer.evaluate \\
        --designer-checkpoint runs/designer/best.pt \\
        --protagonist-checkpoint runs/pv/best.pt --count 100
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any, Optional

import torch

from ..serialization import level_to_dict
from ..training.checkpoint import (configs_from_checkpoint, load_checkpoint,
                                    model_from_checkpoint)
from .actions import DesignerAction, DesignerActionSpace
from .checkpoint import designer_from_checkpoint, load_designer
from .config import GeneratorConfig, RewardConfig
from .env import DesignerEnv
from .ppo import rollout_episode
from .replay import level_fingerprint
from .roles import Oracle, Protagonist
from .score import score_level


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _random_generate(env: DesignerEnv, seed: int):
    state = env.reset(seed)
    rng = random.Random(seed ^ 0x9E3779B9)
    steps = rng.randint(0, env.mutation_budget)
    for _ in range(steps):
        moves = env.legal_moves(state)
        if not moves:
            break
        m = rng.choice(moves)
        state = env.step(state, DesignerAction(kind="reverse", anchor=m.anchor,
                                               direction=m.direction,
                                               distance=m.distance))
    return env.finalize(state, verify=False)


def _designer_generate(env: DesignerEnv, model, action_space, enc, seed, device,
                       rng):
    ep = rollout_episode(env, model, action_space, enc, seed=seed, device=device,
                         rng=rng, verify_finalize=False)
    return ep.finalize, ep.trajectory


def _mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def _aggregate(scored_list, gen_times, fingerprints) -> dict[str, Any]:
    n = len(scored_list)
    unique = len(set(fingerprints))
    comp_keys = ("adversarial_regret", "structural_difficulty", "novelty",
                 "invalidity_penalty", "unsolved_by_oracle_penalty",
                 "triviality_penalty")
    reward_components = {
        k: _mean([s.reward.components.get(k) for s in scored_list])
        for k in comp_keys}
    return {
        "count": n,
        "valid_rate": _mean([1.0 if s.valid else 0.0 for s in scored_list]),
        "oracle_confirmed_solvable_rate":
            _mean([1.0 if s.oracle.solved else 0.0 for s in scored_list]),
        "oracle_solve_rate":
            _mean([1.0 if s.oracle.solved else 0.0 for s in scored_list]),
        "protagonist_solve_rate":
            _mean([1.0 if s.protagonist.solved else 0.0 for s in scored_list]),
        "mean_adversarial_regret":
            _mean([s.reward.components["adversarial_regret"] for s in scored_list
                   if s.oracle.solved]),
        "mean_optimal_moves":
            _mean([s.structural.optimal_moves for s in scored_list]),
        "mean_extra_moves": _mean([s.structural.extra_moves for s in scored_list]),
        "mean_first_exit_depth":
            _mean([s.structural.first_exit_depth for s in scored_list]),
        "mean_rehandled_blocks":
            _mean([s.structural.rehandled_blocks for s in scored_list]),
        "duplicate_rate": (1.0 - unique / n) if n else 0.0,
        "mean_generation_seconds": _mean(gen_times),
        "reward_components_mean": reward_components,
    }


def evaluate_generators(*, protagonist_checkpoint: str, designer_checkpoint: str,
                        baseline_designer_checkpoint: Optional[str] = None,
                        count: int = 50, seed: int = 1_000,
                        generator: Optional[GeneratorConfig] = None,
                        reward: Optional[RewardConfig] = None,
                        mutation_budget: int = 12,
                        protagonist_simulations: int = 100,
                        oracle_simulations: int = 1000,
                        astar_max_nodes: int = 200_000,
                        device: str = "auto",
                        benchmark_out: Optional[str] = None) -> dict[str, Any]:
    dev = _resolve_device(device)
    pck = load_checkpoint(protagonist_checkpoint, map_location="cpu")
    enc, _mc, value_norm = configs_from_checkpoint(pck)
    prot_model = model_from_checkpoint(pck, map_location=dev)

    gen = generator or GeneratorConfig()
    reward = reward or RewardConfig()
    env = DesignerEnv(gen, mutation_budget=mutation_budget, encoding=enc)
    action_space = DesignerActionSpace(enc)
    protagonist = Protagonist(env.env, prot_model, enc, value_norm, dev,
                              simulations=protagonist_simulations)
    oracle = Oracle(env.env, prot_model, enc, value_norm, dev,
                    astar_max_nodes=astar_max_nodes,
                    search_simulations=oracle_simulations)

    designer, _e, _m = designer_from_checkpoint(
        load_designer(designer_checkpoint, map_location=dev), map_location=dev)
    baseline = None
    if baseline_designer_checkpoint:
        baseline, _be, _bm = designer_from_checkpoint(
            load_designer(baseline_designer_checkpoint, map_location=dev),
            map_location=dev)

    kind_offsets = {"random": 11, "designer_baseline": 23, "designer": 37}

    def run_generator(kind, model):
        rng = random.Random((seed * 1000 + kind_offsets[kind]) & 0xFFFFFFFF)
        scored_list, gen_times, fps, levels = [], [], [], []
        for i in range(count):
            s = seed + i
            t0 = time.perf_counter()
            if kind == "random":
                finalize = _random_generate(env, s)
            else:
                finalize, _traj = _designer_generate(env, model, action_space,
                                                      enc, s, dev, rng)
            gen_times.append(time.perf_counter() - t0)
            fp = (level_fingerprint(env.env, finalize.level)
                  if finalize.valid else f"invalid-{i}")
            fps.append(fp)
            scored = score_level(env.env, finalize, protagonist=protagonist,
                                 oracle=oracle, reward_cfg=reward, novelty=0.0,
                                 seed=seed, astar_max_nodes=astar_max_nodes)
            scored_list.append(scored)
            levels.append(level_to_dict(finalize.level))
        return _aggregate(scored_list, gen_times, fps), levels

    report: dict[str, Any] = {"count": count, "seed": seed, "generators": {}}
    report["generators"]["random"], rand_levels = run_generator("random", None)
    if baseline is not None:
        report["generators"]["designer_baseline"], _ = run_generator(
            "designer_baseline", baseline)
    report["generators"]["designer"], designer_levels = run_generator(
        "designer", designer)

    if benchmark_out:
        Path(benchmark_out).parent.mkdir(parents=True, exist_ok=True)
        Path(benchmark_out).write_text(json.dumps(
            {"random": rand_levels, "designer": designer_levels}, indent=2),
            encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate/compare level generators")
    p.add_argument("--protagonist-checkpoint", required=True)
    p.add_argument("--designer-checkpoint", required=True)
    p.add_argument("--baseline-designer-checkpoint", default=None)
    p.add_argument("--count", type=int, default=50)
    p.add_argument("--seed", type=int, default=1_000)
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--color-count", type=int, default=3)
    p.add_argument("--mutation-budget", type=int, default=12)
    p.add_argument("--protagonist-simulations", type=int, default=100)
    p.add_argument("--oracle-simulations", type=int, default=1000)
    p.add_argument("--astar-max-nodes", type=int, default=200_000)
    p.add_argument("--device", default="auto")
    p.add_argument("--benchmark-out", default=None)
    p.add_argument("--report-out", default=None)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    gen = GeneratorConfig(rows=args.rows, cols=args.cols,
                          color_count=args.color_count)
    report = evaluate_generators(
        protagonist_checkpoint=args.protagonist_checkpoint,
        designer_checkpoint=args.designer_checkpoint,
        baseline_designer_checkpoint=args.baseline_designer_checkpoint,
        count=args.count, seed=args.seed, generator=gen,
        mutation_budget=args.mutation_budget,
        protagonist_simulations=args.protagonist_simulations,
        oracle_simulations=args.oracle_simulations,
        astar_max_nodes=args.astar_max_nodes, device=args.device,
        benchmark_out=args.benchmark_out)
    text = json.dumps(report, indent=2)
    if args.report_out:
        Path(args.report_out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
