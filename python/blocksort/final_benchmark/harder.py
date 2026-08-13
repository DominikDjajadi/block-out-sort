"""Build a harder, non-saturated frozen benchmark + held-out promotion levels.

Groups (all evaluation-only, never used for training/replay):

* ``handcrafted``          -- levels from the existing handcrafted dataset
* ``random``               -- random reverse-construction generator
* ``pretrained_designer``  -- behaviour-cloned designer rollouts
* ``adversarial_designer`` -- adversarial designer rollouts
* ``prior_cotraining``     -- accepted levels from prior co-training rounds
* ``ood_large_dense``      -- larger / denser out-of-distribution boards
* ``frontier_selected``    -- levels whose bounded-protagonist solve rate sits
                              strictly between 0 and 1 (near a solver frontier)

A held-out promotion set (``eval_levels.jsonl``, validation+test) is drawn from
the harder pool so co-training promotion decisions use non-saturated levels
rather than the saturated base-dataset split.
"""

from __future__ import annotations

import json
import hashlib
import random
from pathlib import Path
from typing import Any, Optional

import torch

from ..environment import Environment
from ..schema import Level
from ..solver import solve_astar
from ..validation import validate_level
from ..signature import static_level_signature
from ..search.seeding import derive_trial_seed, level_search_identity
from ..serialization import level_from_dict, level_to_dict
from ..dataset.schema import serialize_state
from ..cotraining.eval_split import (
    DEFAULT_EVAL_SPLIT_SEED, create_eval_split_manifest,
    evaluation_split_identity, load_eval_split_manifest)
from ..training.dataset import load_records
from ..designer.config import GeneratorConfig
from ..designer.replay import LevelReplayBuffer
from ..designer.roles import Protagonist as BoundedProtagonist
from .common import (
    Protagonist, designer_generation_identity, designer_levels,
    load_designer_bundle, resolve_device)
from ..cotraining.generation import random_level


def _solvable(env: Environment, level: Level, *, max_nodes: int) -> bool:
    return solve_astar(env, env.initial_state(level), max_nodes=max_nodes).solvable is True


def _fmt_rate(r) -> str:
    return "n/a" if r is None else f"{r:.3f}"


def _dedup(levels: list[Level]) -> list[Level]:
    seen: set[str] = set()
    out = []
    for lv in levels:
        sig = static_level_signature(lv)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(lv)
    return out


def _handcrafted(dataset: str, count: int) -> list[Level]:
    records = load_records(dataset)
    seen, out = set(), []
    for r in records:
        sig = r.get("static_level_signature") or r["level_id"]
        if sig in seen:
            continue
        seen.add(sig)
        out.append(level_from_dict(r["level"]))
        if len(out) >= count:
            break
    return out


def _dataset_signatures(path: Optional[str]) -> set[str]:
    if not path or not Path(path).is_file():
        return set()
    return {
        record.get("static_level_signature") or record["level_id"]
        for record in load_records(path)}


def _file_sha256(path: Optional[str]) -> Optional[str]:
    if not path or not Path(path).is_file():
        return None
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _random_group(env: Environment, cfg: GeneratorConfig, *, count: int,
                  reverse_depth: int, seed: int, max_nodes: int) -> list[Level]:
    rng = random.Random(seed)
    out: list[Level] = []
    for _ in range(count * 4):
        lv = random_level(env, cfg, rng, reverse_depth=reverse_depth)
        if lv is not None and _solvable(env, lv, max_nodes=max_nodes):
            out.append(lv)
        if len(out) >= count:
            break
    return out


def _prior_cotraining(root: Path, count: int) -> list[Level]:
    out: list[Level] = []
    replay_dirs = list(root.glob("round_*/designer/replay"))
    replay_dirs += list(
        root.glob("round_*/designer_attempts/attempt_*/replay"))
    for shard_dir in sorted(replay_dirs):
        try:
            buf = LevelReplayBuffer(shard_dir).load()
            out.extend(buf.levels())
        except Exception:
            continue
    # also the top-level co-training level replay if populated
    top = root / "level_replay"
    if top.exists():
        try:
            out.extend(LevelReplayBuffer(top).load().levels())
        except Exception:
            pass
    return _dedup(out)[:count]


def build_harder_benchmark(
    output_dir: str,
    *,
    protagonist_checkpoint: str,
    adversarial_designer_checkpoint: str,
    pretrained_designer_checkpoint: Optional[str],
    handcrafted_dataset: str,
    training_dataset: Optional[str] = None,
    prior_cotraining_dir: Optional[str],
    count: int = 40,
    device: str = "auto",
    seed: int = 2026,
    astar_max_nodes: int = 200_000,
    gen_astar_max_nodes: int = 4_000,
    frontier_budgets: tuple[int, ...] = (50, 150),
    frontier_trials: int = 2,
) -> dict[str, Any]:
    dev = resolve_device(device)
    root = Path(output_dir)
    prot = Protagonist(protagonist_checkpoint, dev)
    adv_bundle = load_designer_bundle(adversarial_designer_checkpoint, dev)
    bc_bundle = None
    if pretrained_designer_checkpoint:
        bc_bundle = load_designer_bundle(pretrained_designer_checkpoint, dev)
    root.mkdir(parents=True, exist_ok=True)
    env = Environment()

    groups: dict[str, list[Level]] = {}
    groups["handcrafted"] = _handcrafted(handcrafted_dataset, count)
    print(f"  handcrafted: {len(groups['handcrafted'])}", flush=True)
    groups["random"] = _random_group(
        env, GeneratorConfig(rows=5, cols=5, color_count=3, density=0.55),
        count=count, reverse_depth=10, seed=seed * 3 + 1,
        max_nodes=gen_astar_max_nodes)
    print(f"  random: {len(groups['random'])}", flush=True)
    groups["adversarial_designer"] = _dedup(designer_levels(
        env, adv_bundle,
        GeneratorConfig(rows=5, cols=5, color_count=3, density=0.5),
        mutation_budget=10, count=count * 2, device=dev, seed=seed * 5 + 2))[:count]
    print(f"  adversarial_designer: {len(groups['adversarial_designer'])}", flush=True)
    if bc_bundle is not None:
        groups["pretrained_designer"] = _dedup(designer_levels(
            env, bc_bundle,
            GeneratorConfig(rows=5, cols=5, color_count=3, density=0.5),
            mutation_budget=10, count=count * 2, device=dev, seed=seed * 7 + 3))[:count]
    else:
        groups["pretrained_designer"] = []
    print(f"  pretrained_designer: {len(groups['pretrained_designer'])}", flush=True)
    # OOD: denser 6x6 boards (kept small; dense 7x7 A* is too expensive here).
    groups["ood_large_dense"] = (
        _random_group(env, GeneratorConfig(rows=6, cols=6, color_count=3,
                                           density=0.55),
                      count=count // 2, reverse_depth=10, seed=seed * 11 + 4,
                      max_nodes=gen_astar_max_nodes)
        + _random_group(env, GeneratorConfig(rows=6, cols=6, color_count=4,
                                             density=0.6),
                        count=count // 2, reverse_depth=12, seed=seed * 13 + 5,
                        max_nodes=gen_astar_max_nodes))
    print(f"  ood_large_dense: {len(groups['ood_large_dense'])}", flush=True)
    groups["prior_cotraining"] = (
        _prior_cotraining(Path(prior_cotraining_dir), count)
        if prior_cotraining_dir else [])
    print(f"  prior_cotraining: {len(groups['prior_cotraining'])}", flush=True)

    # Frontier-selected: pull from a mixed candidate pool, keep levels the
    # bounded protagonist solves sometimes-but-not-always. Root exploration
    # noise is required: a noiseless search is deterministic, so a per-level
    # solve rate would only ever be 0 or 1.
    bounded = BoundedProtagonist(env, prot.model, prot.enc, prot.value_norm, dev,
                                 simulations=max(frontier_budgets),
                                 dirichlet_alpha=0.5, dirichlet_weight=0.4)
    frontier_trials = max(frontier_trials, 4)
    # Dedicated frontier candidate pool at *calibrated* intermediate difficulty
    # (density 0.3, shallow reverse depth) where the bounded protagonist solve
    # rate is genuinely fractional. (Dense random/OOD levels are uniformly too
    # hard -> rate ~ 0, so they cannot populate the frontier.)
    pool = (
        _random_group(env, GeneratorConfig(rows=5, cols=5, color_count=3,
                                           density=0.3),
                      count=count, reverse_depth=4, seed=seed * 17 + 6,
                      max_nodes=gen_astar_max_nodes)
        + _random_group(env, GeneratorConfig(rows=5, cols=5, color_count=3,
                                             density=0.3),
                        count=count, reverse_depth=6, seed=seed * 19 + 7,
                        max_nodes=gen_astar_max_nodes))
    frontier: list[Level] = []
    fr_records: list[dict[str, Any]] = []
    for lv in pool:
        identity = level_search_identity(env, lv)
        trial_seeds = [
            derive_trial_seed(
                seed,
                trial_index=t,
                level_identity=identity,
                evaluation_context="final_benchmark.frontier_selection",
            )
            for t in range(frontier_trials)
        ]
        solved = 0
        for trial_seed in trial_seeds:
            if bounded.solve(lv, seed=trial_seed).solved:
                solved += 1
        rate = solved / frontier_trials
        if 0.0 < rate < 1.0:
            frontier.append(lv)
            fr_records.append({"signature": static_level_signature(lv),
                               "solve_rate": rate})
        if len(frontier) >= count:
            break
    groups["frontier_selected"] = frontier
    print(f"  frontier_selected: {len(frontier)} (from pool {len(pool)})", flush=True)

    # Persist the frozen benchmark manifest.
    serial = {g: [level_to_dict(l) for l in lst] for g, lst in groups.items()}
    sigs = {g: sorted({static_level_signature(l) for l in lst})
            for g, lst in groups.items()}
    all_sigs = sorted({s for lst in sigs.values() for s in lst})
    training_sigs = (
        _dataset_signatures(training_dataset) | set(sigs["handcrafted"]))
    replay_sigs = set(sigs["prior_cotraining"])
    promotion_sigs = (
        set(sigs["random"]) | set(sigs["ood_large_dense"])
        | set(sigs["frontier_selected"]))
    final_candidates = ("adversarial_designer", "pretrained_designer")
    group_provenance: dict[str, dict[str, Any]] = {}
    definitions = {
        "handcrafted": ("retention", "training-derived", False),
        "prior_cotraining": ("retention", "replay-derived", False),
        "random": ("promotion_validation", "procedurally-generated", False),
        "ood_large_dense": (
            "promotion_validation_ood", "procedurally-generated-ood", False),
        "frontier_selected": (
            "promotion_challenge", "frontier-selected-generated", False),
        "adversarial_designer": (
            "held_out_final", "adversarially-generated", True),
        "pretrained_designer": (
            "held_out_final", "pretrained-generator-transfer", True),
    }
    for group, (role, source_kind, final_candidate) in definitions.items():
        group_sigs = set(sigs[group])
        overlap_reference = training_sigs | replay_sigs | promotion_sigs
        if group in final_candidates:
            for other in final_candidates:
                if other != group:
                    overlap_reference |= set(sigs[other])
        overlap = sorted(group_sigs & overlap_reference)
        held_out = bool(final_candidate and not overlap and group_sigs)
        provenance: dict[str, Any] = {
            "group_name": group,
            "evaluation_role": role,
            "source_kind": source_kind,
            "training_overlap_policy": (
                "excluded-from-held-out" if not final_candidate
                else "must-be-disjoint"),
            "signature_exclusion_applied": True,
            "generation_seed": seed,
            "held_out_eligible": held_out,
            "disjointness_verified": bool(final_candidate),
            "overlap_count": len(overlap),
            "overlap_signatures": overlap,
        }
        if group == "handcrafted":
            provenance.update({
                "source_dataset_or_checkpoint": handcrafted_dataset,
                "source_hash": _file_sha256(handcrafted_dataset),
            })
        elif group == "prior_cotraining":
            provenance["source_dataset_or_checkpoint"] = prior_cotraining_dir
        elif group == "adversarial_designer":
            provenance.update(adv_bundle.provenance())
            provenance.update({
                "source_dataset_or_checkpoint":
                    str(adv_bundle.checkpoint_path),
                "source_hash": adv_bundle.checkpoint_sha256,
                "generation_config": {
                    "rows": 5, "cols": 5, "color_count": 3, "density": 0.5,
                    "mutation_budget": 10, "requested_count": count * 2,
                },
            })
            provenance["generation_identity"] = designer_generation_identity(
                adv_bundle,
                GeneratorConfig(rows=5, cols=5, color_count=3, density=0.5),
                mutation_budget=10, count=count * 2, seed=seed * 5 + 2)
        elif group == "pretrained_designer" and bc_bundle is not None:
            provenance.update(bc_bundle.provenance())
            provenance.update({
                "source_dataset_or_checkpoint":
                    str(bc_bundle.checkpoint_path),
                "source_hash": bc_bundle.checkpoint_sha256,
                "generation_config": {
                    "rows": 5, "cols": 5, "color_count": 3, "density": 0.5,
                    "mutation_budget": 10, "requested_count": count * 2,
                },
            })
            provenance["generation_identity"] = designer_generation_identity(
                bc_bundle,
                GeneratorConfig(rows=5, cols=5, color_count=3, density=0.5),
                mutation_budget=10, count=count * 2, seed=seed * 7 + 3)
        group_provenance[group] = provenance
    manifest = {
        "schema_version": 2,
        "semantics": (
            "group provenance separates retention, promotion-held-out, and "
            "verified final-held-out groups"),
        "seed": seed,
        "count_per_group": count,
        "group_sizes": {g: len(lst) for g, lst in groups.items()},
        "signatures": sigs,
        "all_signatures": all_sigs,
        "frontier_solve_rates": fr_records,
        "group_provenance": group_provenance,
        "training_reference": {
            "dataset": training_dataset,
            "dataset_sha256": _file_sha256(training_dataset),
            "signature_count": len(training_sigs),
        },
    }
    (root / "benchmark.json").write_text(json.dumps(serial, indent=2),
                                         encoding="utf-8")
    (root / "benchmark_manifest.json").write_text(json.dumps(manifest, indent=2),
                                                  encoding="utf-8")

    # Held-out promotion levels (validation + test) from harder, non-handcrafted
    # groups. These are eval-only; their signatures are excluded from training.
    promo_pool = _dedup(groups["random"] + groups["ood_large_dense"]
                        + groups["frontier_selected"])
    eval_records = []
    for lv in promo_pool:
        eval_records.append({
            "level": level_to_dict(lv),
            "state": serialize_state(env.initial_state(lv)),
            "static_level_signature": static_level_signature(lv),
            "level_id": static_level_signature(lv),
        })
    with open(root / "eval_levels.jsonl", "w", encoding="utf-8") as fh:
        for r in eval_records:
            fh.write(json.dumps(r) + "\n")

    split_manifest = None
    if len(eval_records) >= 2:
        split_path = root / "eval_split.json"
        if split_path.exists():
            split_manifest = load_eval_split_manifest(
                split_path,
                root / "eval_levels.jsonl",
                expected_split_seed=DEFAULT_EVAL_SPLIT_SEED,
                expected_validation_count=len(eval_records) // 2,
            )
        else:
            split_manifest = create_eval_split_manifest(
                root / "eval_levels.jsonl",
                split_path,
                validation_count=len(eval_records) // 2,
                split_seed=DEFAULT_EVAL_SPLIT_SEED,
            )
        manifest["evaluation_split"] = evaluation_split_identity(
            split_manifest, eval_limit=None)
        manifest["evaluation_split_manifest"] = "eval_split.json"
        (root / "benchmark_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")

    return {
        "manifest": manifest,
        "eval_levels": len(eval_records),
        "evaluation_split": (
            split_manifest["evaluation_split_fingerprint"]
            if split_manifest is not None else None),
    }


def load_benchmark_groups(output_dir: str) -> dict[str, list[Level]]:
    path = Path(output_dir) / "benchmark.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {g: [level_from_dict(d) for d in lst] for g, lst in data.items()}


def saturation_check(
    output_dir: str,
    *,
    protagonist_checkpoint: str,
    device: str = "auto",
    budget: int = 200,
    trials: int = 3,
    per_group: int = 20,
    seed: int = 7,
    astar_max_nodes: int = 200_000,
) -> dict[str, Any]:
    """Report bounded-search solve rate per group + a saturation verdict.

    A group/benchmark is "useful" when the solve rate is neither near 0 nor
    near 1 (room for measurable improvement).
    """
    dev = resolve_device(device)
    prot = Protagonist(protagonist_checkpoint, dev)
    env = Environment()
    bounded = BoundedProtagonist(env, prot.model, prot.enc, prot.value_norm, dev,
                                 simulations=budget)
    groups = load_benchmark_groups(output_dir)

    report: dict[str, Any] = {"budget": budget, "trials": trials, "groups": {}}
    all_solved = all_n = 0
    for name, levels in groups.items():
        levels = levels[:per_group]
        # Benchmark levels are solvable by construction/generation filter, so we
        # skip a redundant (expensive) A* re-check here.
        solved = n = 0
        for lv in levels:
            n += 1
            identity = level_search_identity(env, lv)
            trial_seeds = (
                derive_trial_seed(
                    seed,
                    trial_index=t,
                    level_identity=identity,
                    evaluation_context="final_benchmark.saturation",
                )
                for t in range(trials)
            )
            s = sum(1 for trial_seed in trial_seeds
                    if bounded.solve(lv, seed=trial_seed).solved)
            solved += s / trials
        rate = (solved / n) if n else None
        report["groups"][name] = {"levels": n, "bounded_solve_rate": rate}
        print(f"  saturation {name}: {_fmt_rate(rate)} (n={n})", flush=True)
        if rate is not None:
            all_solved += solved
            all_n += n
    overall = (all_solved / all_n) if all_n else None
    report["overall_bounded_solve_rate"] = overall
    report["saturated"] = (overall is not None
                           and (overall < 0.05 or overall > 0.95))
    report["useful_benchmark"] = (overall is not None
                                  and 0.05 <= overall <= 0.95)
    return report
