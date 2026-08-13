"""Frozen benchmark groups + forgetting measurement.

Benchmark groups are built once (seeded) and persisted, so every round evaluates
both the previous and candidate protagonist on *identical* states. Groups:

* ``handcrafted``        -- levels from the base (handcrafted) dataset
* ``random``             -- random reverse-construction generator
* ``pretrained_designer``-- optional behavior-cloned designer (separate checkpoint)
* ``adversarial_designer``-- the input / adversarial designer policy
* ``ood``                -- harder out-of-distribution board sizes
* ``prior_rounds``       -- accepted levels from earlier co-training rounds
                            (grown by the loop, snapshotted into the frozen file)

Oracle labels for benchmark initial states are precomputed once
(``benchmark_labels.json``) so routine evaluation does not rerun A*.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Optional

import torch

from ..environment import Environment
from ..model_identity import model_state_sha256
from ..schema import Level
from ..serialization import level_from_dict, level_to_dict
from ..signature import static_level_signature
from ..solution import serialize_action
from ..oracle import Oracle as ExactOracle, ValueResult
from ..expert_iteration.evaluate import evaluate_checkpoint
from ..training.transaction import atomic_write_json
from ..designer.actions import DesignerActionSpace
from ..designer.config import GeneratorConfig
from ..designer.env import DesignerEnv
from ..designer.model import DesignerNet
from ..designer.ppo import rollout_episode
from .generation import random_level

BENCHMARK_FILE = "benchmark.json"
BENCHMARK_LABELS_FILE = "benchmark_labels.json"
BENCHMARK_EVAL_DIR = "benchmark_eval"
EVALUATION_CACHE_FORMAT_VERSION = 1
# Bump whenever the meaning or denominator of an evaluation metric changes.
EVALUATION_METRIC_SEMANTICS_VERSION = 6


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def checkpoint_content_hash(path: str | Path) -> str:
    """Return the SHA-256 of the exact persisted checkpoint bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_dict(value: Any) -> Any:
    to_dict = getattr(value, "to_dict", None)
    return to_dict() if callable(to_dict) else value


def _designer_levels(
    env: Environment,
    model: DesignerNet,
    gen_cfg: GeneratorConfig,
    enc,
    *,
    mutation_budget: int,
    count: int,
    device: torch.device,
    seed: int,
) -> list[Level]:
    denv = DesignerEnv(gen_cfg, mutation_budget=mutation_budget, encoding=enc)
    action_space = DesignerActionSpace(enc)
    rng = random.Random(seed)
    out: list[Level] = []
    for i in range(count):
        ep = rollout_episode(denv, model, action_space, enc,
                             seed=seed * 100003 + i, device=device, rng=rng,
                             verify_finalize=False)
        if ep.finalize.valid:
            out.append(ep.finalize.level)
    return out


def sample_benchmark_groups(
    groups: dict[str, list[Level]],
    *,
    per_group_limit: int | None,
    total_limit: int | None,
    seed: int,
) -> dict[str, list[Level]]:
    """Deterministically sample in configured group order within global limits."""
    for name, value in (
        ("per_group_limit", per_group_limit),
        ("total_limit", total_limit),
    ):
        if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0):
            raise ValueError(f"{name} must be a non-negative integer or None")
    rng = random.Random(seed)
    out: dict[str, list[Level]] = {g: [] for g in groups}
    remaining = total_limit
    for group_name, source_levels in groups.items():
        if remaining == 0:
            break
        levels = list(source_levels)
        limit = len(levels) if per_group_limit is None else per_group_limit
        if remaining is not None:
            limit = min(limit, remaining)
        limit = min(limit, len(levels))
        selected = rng.sample(levels, limit) if limit < len(levels) else levels
        out[group_name] = selected
        if remaining is not None:
            remaining -= len(selected)
    if total_limit is not None:
        actual = sum(len(levels) for levels in out.values())
        if actual > total_limit:
            raise RuntimeError(
                f"benchmark sampling exceeded total_limit: {actual} > {total_limit}")
    return out


def build_benchmark(
    root: Path,
    env: Environment,
    base_records: list[dict[str, Any]],
    *,
    enc,
    gen_cfg: GeneratorConfig,
    ood_gen_cfg: GeneratorConfig,
    mutation_budget: int,
    count: int,
    device: torch.device,
    seed: int,
    adversarial_designer_model: Optional[DesignerNet] = None,
    pretrained_designer_model: Optional[DesignerNet] = None,
    designer_model: Optional[DesignerNet] = None,
) -> dict[str, list[Level]]:
    """Build (once) and persist the frozen benchmark groups."""
    if designer_model is not None and adversarial_designer_model is None:
        adversarial_designer_model = designer_model

    path = root / BENCHMARK_FILE
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        return {g: [level_from_dict(d) for d in lst] for g, lst in data.items()}

    groups: dict[str, list[Level]] = {}

    seen_sigs: set[str] = set()
    hand: list[Level] = []
    for r in base_records:
        sig = r.get("static_level_signature") or r.get("level_id")
        if sig in seen_sigs:
            continue
        seen_sigs.add(sig)
        hand.append(level_from_dict(r["level"]))
        if len(hand) >= count:
            break
    groups["handcrafted"] = hand

    rng = random.Random(seed * 31 + 1)
    rand: list[Level] = []
    for _ in range(count * 3):
        lvl = random_level(env, gen_cfg, rng, reverse_depth=mutation_budget)
        if lvl is not None:
            rand.append(lvl)
        if len(rand) >= count:
            break
    groups["random"] = rand

    if pretrained_designer_model is not None:
        groups["pretrained_designer"] = _designer_levels(
            env, pretrained_designer_model, gen_cfg, enc,
            mutation_budget=mutation_budget, count=count,
            device=device, seed=seed * 53 + 3)

    if adversarial_designer_model is not None:
        groups["adversarial_designer"] = _designer_levels(
            env, adversarial_designer_model, gen_cfg, enc,
            mutation_budget=mutation_budget, count=count,
            device=device, seed=seed * 53 + 7)

    rng_ood = random.Random(seed * 71 + 3)
    ood: list[Level] = []
    for _ in range(count * 3):
        lvl = random_level(env, ood_gen_cfg, rng_ood,
                           reverse_depth=mutation_budget + 2)
        if lvl is not None:
            ood.append(lvl)
        if len(ood) >= count:
            break
    groups["ood"] = ood

    groups["prior_rounds"] = []

    serial = {g: [level_to_dict(l) for l in lst] for g, lst in groups.items()}
    path.write_text(json.dumps(serial, indent=2), encoding="utf-8")
    return groups


def append_prior_round(root: Path, levels: list[Level]) -> None:
    """Grow the (frozen) ``prior_rounds`` group with accepted round levels."""
    path = root / BENCHMARK_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    data.setdefault("prior_rounds", [])
    existing = {
        static_level_signature(level_from_dict(item))
        for item in data["prior_rounds"]}
    changed = False
    for level in levels:
        signature = static_level_signature(level)
        if signature not in existing:
            data["prior_rounds"].append(level_to_dict(level))
            existing.add(signature)
            changed = True
    if not changed:
        return
    atomic_write_json(path, data)
    labels_path = root / BENCHMARK_LABELS_FILE
    if labels_path.exists():
        labels_path.unlink()


def _value_result_dict(vr: ValueResult) -> dict[str, Any]:
    return {"value": vr.value, "exact": vr.exact, "solvable": vr.solvable}


def _termination_reason(analysis) -> str:
    if analysis.terminal:
        return "terminal"
    if analysis.exact and not analysis.solvable:
        return "unsolvable"
    if analysis.exact:
        return "exact"
    return "exhausted"


def ensure_benchmark_labels(
    root: Path,
    env: Environment,
    groups: dict[str, list[Level]],
    label_oracle: ExactOracle,
) -> dict[str, list[dict[str, Any]]]:
    """Precompute oracle labels for every benchmark initial state (once).

    ``classification_complete`` records whether every legal successor value was
    exact.  Only then does absence from ``optimal_actions`` prove non-optimality.
    """
    path = root / BENCHMARK_LABELS_FILE
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))

    labels: dict[str, list[dict[str, Any]]] = {}
    for group_name, levels in groups.items():
        if not levels:
            labels[group_name] = []
            continue
        print(f"  precomputing benchmark labels: {group_name} "
              f"({len(levels)} levels)", flush=True)
        entries: list[dict[str, Any]] = []
        for i, lv in enumerate(levels):
            state = env.initial_state(lv)
            analysis = label_oracle.analyze(state)
            entries.append({
                "canonical_key": env.canonical_key(state),
                "static_signature": analysis.static_signature,
                "value_result": _value_result_dict(ValueResult(
                    value=analysis.value,
                    exact=analysis.exact,
                    solvable=analysis.solvable if not analysis.terminal else True,
                )),
                "optimal_actions": [a.serialized for a in analysis.actions
                                    if a.optimal],
                "classification_complete":
                    bool(analysis.all_successors_exact),
                "termination": _termination_reason(analysis),
            })
            if (i + 1) % 5 == 0 or (i + 1) == len(levels):
                print(f"    {group_name}: {i + 1}/{len(levels)}", flush=True)
        labels[group_name] = entries
    path.write_text(json.dumps(labels, indent=2), encoding="utf-8")
    return labels


def _precomputed_lookup(
    labels: dict[str, list[dict[str, Any]]],
    group_name: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for entry in labels.get(group_name, []):
        if "static_signature" not in entry:
            raise ValueError(
                f"benchmark label in group {group_name!r} is missing "
                "static_signature; unsafe dynamic-only labels are unsupported")
        if "canonical_key" not in entry:
            raise ValueError(
                f"benchmark label in group {group_name!r} is missing canonical_key")
        classification_complete = entry.get("classification_complete")
        if (classification_complete is not None
                and not isinstance(classification_complete, bool)):
            raise ValueError(
                f"benchmark label in group {group_name!r} has non-boolean "
                "classification_complete")
        optimal_actions = entry.get("optimal_actions")
        if not isinstance(optimal_actions, list):
            raise ValueError(
                f"benchmark label in group {group_name!r} has invalid "
                "optimal_actions; expected a list")
        action_keys = []
        for action in optimal_actions:
            if not isinstance(action, dict):
                raise ValueError(
                    f"benchmark label in group {group_name!r} has a "
                    "non-object optimal action")
            action_keys.append(_canonical_json(action))
        if len(action_keys) != len(set(action_keys)):
            raise ValueError(
                f"benchmark label in group {group_name!r} contains a "
                "duplicate optimal action")
        key = (entry["static_signature"], entry["canonical_key"])
        previous = lookup.get(key)
        if previous is None:
            lookup[key] = entry
            continue
        label_fields = ("value_result", "optimal_actions",
                        "classification_complete", "termination")
        if any(previous.get(field) != entry.get(field) for field in label_fields):
            raise ValueError(
                f"conflicting duplicate benchmark labels for group "
                f"{group_name!r}, key {key!r}")
    return lookup


def evaluate_groups(
    env: Environment,
    model,
    enc,
    value_norm,
    groups: dict[str, list[Level]],
    *,
    exact_oracle: ExactOracle,
    budgets: list[int],
    device: torch.device,
    c_puct: float,
    seed: int,
    precomputed_labels: dict[str, list[dict[str, Any]]] | None = None,
    progress_dir: Path | None = None,
    tag: str | None = None,
    checkpoint_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate one model on every group's initial states.

    ``tag`` is only a human-readable namespace.  Cache correctness comes from
    content identities computed below.  Normal persisted-checkpoint callers
    should also pass ``checkpoint_sha256``; the model-state hash independently
    binds the cache to the parameters actually evaluated.
    """
    out: dict[str, Any] = {}
    group_names = [n for n, lvls in groups.items() if lvls]
    total = len(group_names)

    # Validate labels and benchmark lookup identity before considering a cache.
    states_by_group = {
        name: [env.initial_state(level) for level in groups[name]]
        for name in group_names
    }
    precomputed_by_group = {}
    if precomputed_labels is not None:
        for name in group_names:
            lookup = _precomputed_lookup(precomputed_labels, name)
            missing = [
                (static_level_signature(state.level), env.canonical_key(state))
                for state in states_by_group[name]
                if (static_level_signature(state.level),
                    env.canonical_key(state)) not in lookup
            ]
            if missing:
                raise ValueError(
                    f"benchmark labels for group {name!r} are missing "
                    f"{len(missing)} benchmark state(s); first missing key: "
                    f"{missing[0]!r}")
            for state in states_by_group[name]:
                key = (static_level_signature(state.level),
                       env.canonical_key(state))
                legal_actions = {
                    _canonical_json(serialize_action(state, action))
                    for action in env.legal_actions(state)
                }
                for optimal_action in lookup[key]["optimal_actions"]:
                    if _canonical_json(optimal_action) not in legal_actions:
                        raise ValueError(
                            f"benchmark label in group {name!r} contains an "
                            f"optimal action that is not legal for key {key!r}")
            precomputed_by_group[name] = lookup

    benchmark_serial = {
        name: [level_to_dict(level) for level in levels]
        for name, levels in groups.items()
    }
    evaluation_config = {
        "encoding": _config_dict(enc),
        "value_norm": _config_dict(value_norm),
        "budgets": list(budgets),
        "c_puct": c_puct,
        "seed": seed,
        "device": str(device),
        "oracle": {
            "class": (f"{type(exact_oracle).__module__}."
                      f"{type(exact_oracle).__qualname__}"),
            "max_nodes": getattr(exact_oracle, "max_nodes", None),
            "time_limit_seconds":
                getattr(exact_oracle, "time_limit_seconds", None),
        },
    }
    cache_enabled = progress_dir is not None and tag is not None
    if cache_enabled and not checkpoint_sha256:
        raise ValueError(
            "checkpoint_sha256 is required for persisted benchmark caching")
    common_metadata = ({
        "cache_format_version": EVALUATION_CACHE_FORMAT_VERSION,
        "metric_semantics_version": EVALUATION_METRIC_SEMANTICS_VERSION,
        "model_state_sha256": model_state_sha256(model),
        "checkpoint_sha256": checkpoint_sha256,
        "benchmark_sha256": _content_hash(benchmark_serial),
        "labels_sha256": (_content_hash(precomputed_labels)
                          if precomputed_labels is not None else None),
        "evaluation_config": evaluation_config,
    } if cache_enabled else None)

    for gi, name in enumerate(group_names):
        metadata = ({
            **common_metadata,
            "group_name": name,
            "group_sha256": _content_hash(benchmark_serial[name]),
        } if common_metadata is not None else None)
        cached = None
        if cache_enabled:
            assert progress_dir is not None and tag is not None
            assert metadata is not None
            cache_key = _content_hash(metadata)
            cached = progress_dir / tag / f"{name}.{cache_key}.json"
            if cached.exists():
                try:
                    payload = json.loads(cached.read_text(encoding="utf-8"))
                    if (isinstance(payload, dict)
                            and payload.get("metadata") == metadata
                            and isinstance(payload.get("result"), dict)):
                        out[name] = payload["result"]
                        print(f"  benchmark {tag}/{name}: loaded cache "
                              f"({gi + 1}/{total})", flush=True)
                        continue
                except (OSError, UnicodeError, json.JSONDecodeError):
                    pass
                print(f"  benchmark {tag}/{name}: stale/corrupt cache; "
                      "recomputing", flush=True)

        states = states_by_group[name]
        pre = precomputed_by_group.get(name)
        print(f"  benchmark {tag}/{name}: {len(states)} states "
              f"({gi + 1}/{total})", flush=True)
        result = evaluate_checkpoint(
            env, model, enc, value_norm, states, budgets=budgets,
            oracle=exact_oracle, device=device, c_puct=c_puct, seed=seed,
            precomputed=pre)
        out[name] = result
        if cache_enabled:
            dest = cached
            assert dest is not None and metadata is not None
            dest.parent.mkdir(parents=True, exist_ok=True)
            payload = {"metadata": metadata, "result": result}
            dest.write_text(json.dumps(payload, indent=2, sort_keys=True),
                            encoding="utf-8")

    for name in groups:
        if name not in out:
            out[name] = {"states": 0}
    return out


def forgetting_report(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    metric_budget: int,
) -> dict[str, Any]:
    """Per-group drop in search-optimal accuracy vs the baseline (round 0).

    Negative ``delta`` means the candidate is worse than the baseline on that
    group (i.e. forgetting); positive means improvement.
    """
    report: dict[str, Any] = {}
    for name in baseline:
        b = baseline.get(name, {})
        c = candidate.get(name, {})
        if not b.get("states") or not c.get("states"):
            report[name] = {
                "delta": None, "baseline": None, "candidate": None,
                "baseline_coverage": None, "candidate_coverage": None,
                "baseline_classification_coverage": None,
                "candidate_classification_coverage": None,
                "baseline_exact_regret_coverage": None,
                "candidate_exact_regret_coverage": None,
            }
            continue
        b_budget = b["budgets"].get(str(metric_budget), {})
        c_budget = c["budgets"].get(str(metric_budget), {})
        bb = b_budget.get("search_optimal_acc")
        cc = c_budget.get("search_optimal_acc")
        delta = (cc - bb) if (bb is not None and cc is not None) else None
        report[name] = {
            "delta": delta, "baseline": bb, "candidate": cc,
            "baseline_coverage":
                b_budget.get("search_optimal_classification_coverage"),
            "candidate_coverage":
                c_budget.get("search_optimal_classification_coverage"),
            "baseline_classification_coverage":
                b_budget.get("search_optimal_classification_coverage"),
            "candidate_classification_coverage":
                c_budget.get("search_optimal_classification_coverage"),
            "baseline_exact_regret_coverage":
                b_budget.get("search_exact_regret_coverage"),
            "candidate_exact_regret_coverage":
                c_budget.get("search_exact_regret_coverage"),
        }
    return report
