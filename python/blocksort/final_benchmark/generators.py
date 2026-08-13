"""Generator comparison over multiple seeds.

Compares the random reverse-construction generator, the behaviour-cloned
designer, the adversarial designer, and the co-trained designer using equal
level counts and comparable generation budgets. Reports validity, oracle-
confirmed solvability, protagonist/oracle solve rates, adversarial regret,
structural difficulty signals (search-based, *not* human difficulty), duplicate
rate, and generation time, with mean / std / per-seed values across seeds.
"""

from __future__ import annotations

import random
import statistics
import time
from typing import Any, Optional

import torch

from ..environment import Environment
from ..designer.actions import DesignerAction, DesignerActionSpace
from ..designer.config import GeneratorConfig, RewardConfig
from ..designer.env import DesignerEnv
from ..designer.ppo import rollout_episode
from ..designer.replay import level_fingerprint
from ..designer.roles import Oracle, Protagonist
from ..designer.score import score_level
from .common import Protagonist as ProtagonistBundle
from .common import (
    designer_generation_identity, load_designer_bundle, resolve_device,
    validate_designer_generation_config)


def _stat(values: list[Optional[float]]):
    vs = [v for v in values if v is not None]
    if not vs:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": statistics.fmean(vs),
            "std": statistics.pstdev(vs) if len(vs) > 1 else 0.0,
            "n": len(vs)}


def _random_finalize(env: DesignerEnv, seed: int):
    state = env.reset(seed)
    rng = random.Random(seed ^ 0x9E3779B9)
    for _ in range(rng.randint(0, env.mutation_budget)):
        moves = env.legal_moves(state)
        if not moves:
            break
        m = rng.choice(moves)
        state = env.step(state, DesignerAction(kind="reverse", anchor=m.anchor,
                                               direction=m.direction,
                                               distance=m.distance))
    return env.finalize(state, verify=False)


def _aggregate_seed(scored, gen_times, fps) -> dict[str, Any]:
    n = len(scored)
    unique = len(set(fps))

    def m(f):
        return statistics.fmean([f(s) for s in scored]) if n else None

    def mopt(attr):
        vals = [getattr(s.structural, attr) for s in scored
                if getattr(s.structural, attr) is not None]
        return statistics.fmean(vals) if vals else None

    regrets = [s.reward.components["adversarial_regret"] for s in scored
               if s.oracle.solved]
    structs = [s.reward.components.get("structural_difficulty") for s in scored]
    structs = [x for x in structs if x is not None]
    return {
        "count": n,
        "valid_rate": m(lambda s: 1.0 if s.valid else 0.0),
        "oracle_confirmed_solvable": m(lambda s: 1.0 if s.oracle.solved else 0.0),
        "oracle_solve_rate": m(lambda s: 1.0 if s.oracle.solved else 0.0),
        "protagonist_solve_rate": m(lambda s: 1.0 if s.protagonist.solved else 0.0),
        "mean_adversarial_regret": (statistics.fmean(regrets) if regrets else None),
        "mean_optimal_moves": mopt("optimal_moves"),
        "mean_extra_moves": mopt("extra_moves"),
        "mean_first_exit_depth": mopt("first_exit_depth"),
        "mean_distinct_setup_blocks": mopt("distinct_setup_blocks"),
        "mean_rehandled_blocks": mopt("rehandled_blocks"),
        "mean_immediately_exitable": mopt("immediately_exitable"),
        "mean_opening_requires_setup": mopt("opening_requires_setup"),
        "mean_structural_score": (statistics.fmean(structs) if structs else None),
        "duplicate_rate": (1.0 - unique / n) if n else 0.0,
        "mean_generation_seconds": (statistics.fmean(gen_times) if gen_times else None),
    }


def compare_generators(
    *,
    protagonist_checkpoint: str,
    designer_checkpoints: dict[str, Optional[str]],   # name -> ckpt (None=random)
    count: int = 30,
    seeds: list[int] = (1000, 2000, 3000),
    generator: Optional[GeneratorConfig] = None,
    reward: Optional[RewardConfig] = None,
    mutation_budget: int = 10,
    protagonist_simulations: int = 100,
    oracle_simulations: int = 1000,
    astar_max_nodes: int = 200_000,
    device: str = "auto",
) -> dict[str, Any]:
    dev = resolve_device(device)
    prot = ProtagonistBundle(protagonist_checkpoint, dev)
    gen_cfg = generator or GeneratorConfig()
    reward = reward or RewardConfig()

    scoring_env = DesignerEnv(
        gen_cfg, mutation_budget=mutation_budget, encoding=prot.enc)
    protagonist = Protagonist(
        scoring_env.env, prot.model, prot.enc, prot.value_norm, dev,
                              simulations=protagonist_simulations)
    oracle = Oracle(scoring_env.env, prot.model, prot.enc, prot.value_norm, dev,
                    astar_max_nodes=astar_max_nodes,
                    search_simulations=oracle_simulations)

    models = {}
    for name, ckpt in designer_checkpoints.items():
        models[name] = (None if ckpt is None
                        else load_designer_bundle(ckpt, dev))

    report: dict[str, Any] = {"count": count, "seeds": list(seeds),
                              "generators": {}}
    name_offsets = {n: (idx + 1) * 101 for idx, n in enumerate(models)}
    for name, bundle in models.items():
        encoding = prot.enc if bundle is None else bundle.encoding
        if bundle is not None:
            validate_designer_generation_config(bundle, gen_cfg)
        env = DesignerEnv(
            gen_cfg, mutation_budget=mutation_budget, encoding=encoding)
        action_space = DesignerActionSpace(encoding)
        per_seed = []
        for sd in seeds:
            rng = random.Random(sd * 1009 + name_offsets[name])
            scored, times, fps = [], [], []
            for i in range(count):
                s = sd + i
                t0 = time.perf_counter()
                if bundle is None:
                    fin = _random_finalize(env, s)
                else:
                    ep = rollout_episode(
                        env, bundle.model, action_space, bundle.encoding,
                                         seed=s, device=dev, rng=rng,
                                         verify_finalize=False)
                    fin = ep.finalize
                times.append(time.perf_counter() - t0)
                fps.append(level_fingerprint(env.env, fin.level)
                           if fin.valid else f"invalid-{sd}-{i}")
                scored.append(score_level(env.env, fin, protagonist=protagonist,
                                          oracle=oracle, reward_cfg=reward,
                                          novelty=0.0, seed=sd,
                                          astar_max_nodes=astar_max_nodes))
            per_seed.append(_aggregate_seed(scored, times, fps))
        # mean/std across seeds for each metric.
        keys = [k for k in per_seed[0] if k != "count"]
        summary = {k: _stat([ps[k] for ps in per_seed]) for k in keys}
        summary["count"] = count
        provenance = (
            {"source_kind": "procedurally-generated", "checkpoint": None}
            if bundle is None else {
                **bundle.provenance(),
                "source_kind": "designer-generated",
                "generation_identities": {
                    str(seed): designer_generation_identity(
                        bundle, gen_cfg, mutation_budget=mutation_budget,
                        count=count, seed=seed)
                    for seed in seeds
                },
            })
        report["generators"][name] = {
            "per_seed": per_seed, "summary": summary,
            "provenance": provenance}
    return report
