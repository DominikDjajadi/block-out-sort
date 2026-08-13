"""Generate a versioned supervised policy-value dataset (JSON Lines).

Every exported record carries exact value and policy evidence. Full-exact mode
proves every successor; the fallback mode retains a solved root's verified
optimal path when successor enumeration exhausts.

Usage:

    python -m blocksort.dataset.generate \\
        --levels fixtures/levels.json \\
        --output data/training/pv_examples.jsonl \\
        --sampling optimal-path random-reachable near-optimal \\
        --samples-per-level 20 --seed 42 --policy-target uniform-optimal
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional

from ..environment import Environment
from ..oracle import Oracle, StateAnalysis
from ..schema import Level
from ..serialization import level_from_dict, level_to_dict
from ..signature import static_level_signature
from ..solver import solve_astar
from ..state import State
from ..training.transaction import (
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)
from ..conformance import _normalized_hashable
from .schema import (
    DATASET_VERSION,
    LABEL_EXACT_PATH_POLICY,
    LABEL_FULL_EXACT,
    build_exact_path_record,
    build_record,
)
from .targets import DEFAULT_TEMPERATURE, DEFAULT_VALUE_NORM_CONSTANT

SAMPLING_MODES = ("initial", "optimal-path", "random-reachable", "near-optimal")
LABEL_FULL_EXACT_WITH_PATH_FALLBACK = "full-exact-with-path-fallback"
LABEL_MODES = (
    LABEL_FULL_EXACT,
    LABEL_FULL_EXACT_WITH_PATH_FALLBACK,
    LABEL_EXACT_PATH_POLICY,
)
RESUMABLE_GENERATION_SCHEMA_VERSION = 1


# --------------------------------------------------------------------------
# Level loading
# --------------------------------------------------------------------------

def load_levels(path: str | Path) -> list[tuple[str, Level]]:
    """Load levels from a JSON file (a list, ``{"levels": [...]}``, or a single
    level object). Returns ``(level_id, Level)`` pairs."""
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "levels" in data:
        data = data["levels"]
    if isinstance(data, dict):  # a single level object
        data = [data]
    stem = path.stem
    return [(f"{stem}#{i}", level_from_dict(raw)) for i, raw in enumerate(data)]


# --------------------------------------------------------------------------
# Sampling walks
# --------------------------------------------------------------------------

def _optimal_path_states(
    env: Environment, initial: State, max_nodes: int,
    time_limit_seconds: Optional[float] = None,
) -> list[State]:
    """States visited along one optimal solution (excluding the terminal)."""
    result = solve_astar(
        env, initial, max_nodes=max_nodes,
        time_limit_seconds=time_limit_seconds)
    if result.solvable is not True or result.actions is None:
        return []
    states = []
    cur = initial
    for action in result.actions:
        if env.is_terminal(cur):
            break
        states.append(cur)
        cur = env.apply_action(cur, action)
    return states


def _random_walk_state(
    env: Environment, rng: random.Random, initial: State, walk_length: int
) -> State:
    """Take up to ``walk_length`` random legal moves, preferring unvisited
    successors to avoid reversible loops; return the last non-terminal state."""
    cur = initial
    last_nonterminal = initial
    seen = {env.canonical_key(cur)}
    steps = rng.randint(1, max(1, walk_length))
    for _ in range(steps):
        if env.is_terminal(cur):
            break
        legal = env.legal_actions(cur)
        if not legal:
            break
        rng.shuffle(legal)
        chosen = None
        for action in legal:
            nxt = env.apply_action(cur, action)
            if env.canonical_key(nxt) not in seen:
                chosen = nxt
                break
        if chosen is None:
            chosen = env.apply_action(cur, rng.choice(legal))
        cur = chosen
        seen.add(env.canonical_key(cur))
        if not env.is_terminal(cur):
            last_nonterminal = cur
    return last_nonterminal


def _near_optimal_states(
    env: Environment,
    oracle: Oracle,
    rng: random.Random,
    initial: State,
    deviation_prob: float,
    step_cap: int,
) -> list[State]:
    """Follow an optimal path but occasionally take a positive-regret action,
    labelling every non-terminal state along the way (recovery coverage)."""
    states: list[State] = []
    cur = initial
    for _ in range(step_cap):
        if env.is_terminal(cur):
            break
        analysis = oracle.analyze(cur)
        if not (analysis.exact and analysis.solvable and analysis.all_successors_exact):
            break
        states.append(cur)
        optimal = [a for a in analysis.actions if a.optimal]
        positive = [a for a in analysis.actions if a.regret is not None and a.regret > 0]
        if positive and rng.random() < deviation_prob:
            choice = rng.choice(positive)
        elif optimal:
            choice = rng.choice(optimal)
        else:
            choice = rng.choice(list(analysis.actions))
        cur = env.apply_action(cur, choice.action)
    return states


# --------------------------------------------------------------------------
# Candidate enumeration
# --------------------------------------------------------------------------

def _candidates(
    env: Environment,
    oracle: Oracle,
    level_id: str,
    level: Level,
    modes: Iterable[str],
    *,
    samples_per_level: int,
    seed: int,
    walk_length: int,
    deviation_prob: float,
    near_optimal_step_cap: int,
    max_nodes: int,
    time_limit_seconds: Optional[float],
) -> Iterator[tuple[State, dict[str, Any]]]:
    initial = env.initial_state(level)
    rng = random.Random(seed)
    for mode in modes:
        if mode == "initial":
            yield initial, {"sampling": "initial", "level_id": level_id, "seed": seed, "step": 0}
        elif mode == "optimal-path":
            for i, st in enumerate(_optimal_path_states(
                env, initial, max_nodes, time_limit_seconds)):
                yield st, {"sampling": "optimal-path", "level_id": level_id, "seed": seed, "step": i}
        elif mode == "random-reachable":
            for w in range(samples_per_level):
                st = _random_walk_state(env, rng, initial, walk_length)
                yield st, {"sampling": "random-reachable", "level_id": level_id, "seed": seed, "walk": w}
        elif mode == "near-optimal":
            for w in range(samples_per_level):
                for i, st in enumerate(_near_optimal_states(
                    env, oracle, rng, initial, deviation_prob, near_optimal_step_cap
                )):
                    yield st, {"sampling": "near-optimal", "level_id": level_id, "seed": seed, "walk": w, "step": i}
        else:
            raise ValueError(f"unknown sampling mode: {mode}")


# --------------------------------------------------------------------------
# Deduplication
# --------------------------------------------------------------------------

def _action_set(actions: list[dict]) -> set:
    return {_normalized_hashable(a) for a in actions}


def _merge_records(existing: dict[str, Any], new: dict[str, Any]) -> None:
    """Merge ``new`` into ``existing`` after asserting exact-label agreement."""
    if existing["optimal_remaining_moves"] != new["optimal_remaining_moves"]:
        raise ValueError(
            f"conflicting optimal value for state {existing['state_key']}: "
            f"{existing['optimal_remaining_moves']} vs {new['optimal_remaining_moves']}"
        )
    if _action_set(existing["legal_actions"]) != _action_set(new["legal_actions"]):
        raise ValueError(f"conflicting legal actions for state {existing['state_key']}")
    strength = {
        LABEL_EXACT_PATH_POLICY: 1,
        LABEL_FULL_EXACT: 2,
    }
    existing_kind = existing.get("label_kind", LABEL_FULL_EXACT)
    new_kind = new.get("label_kind", LABEL_FULL_EXACT)
    if existing_kind not in strength or new_kind not in strength:
        raise ValueError(
            f"unsupported exact label kind while merging state "
            f"{existing['state_key']}")
    merged_provenance = list(existing.get("provenance") or []) + list(
        new.get("provenance") or [])
    if strength[new_kind] > strength[existing_kind]:
        existing.clear()
        existing.update(copy.deepcopy(new))
        existing["provenance"] = merged_provenance
        return
    if strength[existing_kind] > strength[new_kind]:
        existing["provenance"] = merged_provenance
        return
    if _action_set(existing["optimal_actions"]) != _action_set(new["optimal_actions"]):
        raise ValueError(f"conflicting optimal actions for state {existing['state_key']}")
    existing["provenance"] = merged_provenance


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def generate_records(
    levels: list[tuple[str, Level]],
    *,
    modes: list[str],
    samples_per_level: int = 10,
    seed: int = 0,
    policy_target: str = "uniform-optimal",
    temperature: float = DEFAULT_TEMPERATURE,
    max_nodes: int = 300_000,
    time_limit_seconds: Optional[float] = None,
    walk_length: int = 6,
    deviation_prob: float = 0.3,
    near_optimal_step_cap: int = 40,
    value_norm_constant: float = DEFAULT_VALUE_NORM_CONSTANT,
    label_mode: str = LABEL_FULL_EXACT,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Produce deduplicated dataset records plus generation statistics."""
    _validate_limits(max_nodes, time_limit_seconds)
    _validate_label_mode(label_mode, modes)
    env = Environment()
    oracle = Oracle(
        env, max_nodes=max_nodes,
        time_limit_seconds=time_limit_seconds)
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    stats = _generation_stats()

    for idx, (level_id, level) in enumerate(levels):
        level_seed = (seed * 1_000_003 + idx) & 0xFFFFFFFF
        for state, provenance in _candidates(
            env, oracle, level_id, level, modes,
            samples_per_level=samples_per_level, seed=level_seed,
            walk_length=walk_length, deviation_prob=deviation_prob,
            near_optimal_step_cap=near_optimal_step_cap, max_nodes=max_nodes,
            time_limit_seconds=time_limit_seconds,
        ):
            stats["candidates"] += 1
            record, skip_reason = _label_candidate(
                env, oracle, state, level_id=level_id,
                label_mode=label_mode, policy_target=policy_target,
                temperature=temperature, max_nodes=max_nodes,
                time_limit_seconds=time_limit_seconds,
                value_norm_constant=value_norm_constant,
                provenance=provenance,
            )
            if record is None:
                stats[skip_reason] += 1
                continue
            dk = (record["static_level_signature"], record["state_key"])
            if dk in by_key:
                _merge_records(by_key[dk], record)
                stats["duplicates_merged"] += 1
            else:
                by_key[dk] = record
                stats["records"] += 1

    _set_record_kind_counts(stats, by_key.values())

    return list(by_key.values()), stats


def _validate_limits(
    max_nodes: int,
    time_limit_seconds: Optional[float],
) -> None:
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes <= 0:
        raise ValueError("max_nodes must be a positive integer")
    if (time_limit_seconds is not None
            and (isinstance(time_limit_seconds, bool)
                 or not isinstance(time_limit_seconds, (int, float))
                 or not math.isfinite(float(time_limit_seconds))
                 or time_limit_seconds <= 0)):
        raise ValueError("time_limit_seconds must be finite and positive")


def _validate_label_mode(label_mode: str, modes: list[str]) -> None:
    if label_mode not in LABEL_MODES:
        raise ValueError(f"unknown label mode: {label_mode}")
    if label_mode == LABEL_EXACT_PATH_POLICY and modes != ["initial"]:
        raise ValueError(
            "exact-path-policy currently requires --sampling initial; other "
            "samplers may perform hidden oracle searches")


def _record_astar_query(
    queries: list[dict[str, Any]],
    env: Environment,
    state: State,
    result,
    *,
    max_nodes: int,
    time_limit_seconds: Optional[float],
) -> None:
    queries.append({
        "query_index": len(queries) + 1,
        "query_role": "root",
        "static_level_signature": static_level_signature(state.level),
        "state_key": env.canonical_key(state),
        "remaining_blocks": state.remaining,
        "max_nodes": max_nodes,
        "time_limit_seconds": time_limit_seconds,
        "termination_reason": result.termination_reason,
        "solvable": result.solvable,
        "exact": result.optimal or result.solvable is False,
        "states_explored": result.states_explored,
        "states_generated": result.states_generated,
        "duplicate_states": result.duplicate_states,
        "max_frontier_size": result.max_frontier_size,
        "elapsed_seconds": result.elapsed_seconds,
    })


def _label_candidate(
    env: Environment,
    oracle: Oracle,
    state: State,
    *,
    level_id: str,
    label_mode: str,
    policy_target: str,
    temperature: float,
    max_nodes: int,
    time_limit_seconds: Optional[float],
    value_norm_constant: float,
    provenance: dict[str, Any],
    astar_queries: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Label one candidate and return ``(record, skip-stat-key)``."""
    if label_mode == LABEL_EXACT_PATH_POLICY:
        if env.is_terminal(state):
            return None, "skipped_terminal"
        result = solve_astar(
            env, state, max_nodes=max_nodes,
            time_limit_seconds=time_limit_seconds)
        if astar_queries is not None:
            _record_astar_query(
                astar_queries, env, state, result, max_nodes=max_nodes,
                time_limit_seconds=time_limit_seconds)
        record = build_exact_path_record(
            result, state, env, level_id=level_id,
            value_norm_constant=value_norm_constant, provenance=provenance)
        if record is not None:
            return record, ""
        if result.solvable is None:
            return None, "skipped_exhausted"
        if result.solvable is False:
            return None, "skipped_unsolvable"
        return None, "skipped_terminal"

    analysis = oracle.analyze(state)
    record = build_record(
        analysis, state, level_id=level_id,
        policy_type=policy_target, temperature=temperature,
        value_norm_constant=value_norm_constant, provenance=provenance,
    )
    if record is not None:
        return record, ""
    if (label_mode == LABEL_FULL_EXACT_WITH_PATH_FALLBACK
            and analysis.exact and analysis.solvable
            and not analysis.all_successors_exact):
        result = oracle.cached_solve_result(state)
        if result is not None:
            fallback_provenance = {
                **provenance,
                "labeling": {
                    "strategy": "full_exact_then_exact_path",
                    "fallback_reason": "successor_proof_incomplete",
                },
            }
            record = build_exact_path_record(
                result, state, env, level_id=level_id,
                value_norm_constant=value_norm_constant,
                provenance=fallback_provenance)
            if record is not None:
                return record, ""
    if analysis.terminal:
        return None, "skipped_terminal"
    if not analysis.exact:
        return None, "skipped_exhausted"
    if not analysis.solvable:
        return None, "skipped_unsolvable"
    return None, "skipped_successor_exhausted"


def write_jsonl(records: list[dict[str, Any]], output: str | Path) -> None:
    output = Path(output)
    text = "".join(
        json.dumps(record, sort_keys=True) + "\n" for record in records)
    atomic_write_text(output, text)


def _generation_stats() -> dict[str, int]:
    return {
        "candidates": 0,
        "records": 0,
        "records_full_exact": 0,
        "records_exact_path": 0,
        "duplicates_merged": 0,
        "skipped_terminal": 0,
        "skipped_exhausted": 0,
        "skipped_unsolvable": 0,
        "skipped_successor_exhausted": 0,
    }


def _set_record_kind_counts(
    stats: dict[str, int],
    records: Iterable[dict[str, Any]],
) -> None:
    stats["records_full_exact"] = 0
    stats["records_exact_path"] = 0
    for record in records:
        if record.get("label_kind") == LABEL_FULL_EXACT:
            stats["records_full_exact"] += 1
        elif record.get("label_kind") == LABEL_EXACT_PATH_POLICY:
            stats["records_exact_path"] += 1


def _run_identity(source_sha256: str, config: dict[str, Any]) -> str:
    payload = json.dumps(
        {"levels_sha256": source_sha256, "config": config},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _checkpoint_paths(output: str | Path) -> tuple[Path, Path]:
    destination = Path(output)
    return (
        Path(f"{destination}.progress.json"),
        Path(f"{destination}.parts"),
    )


def _load_part(
    path: Path,
    *,
    identity: str,
    level_index: int,
    level_id: str,
    level_signature: str,
) -> dict[str, Any]:
    try:
        part = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid exact-generation checkpoint: {path}") from exc
    expected = {
        "schema_version": RESUMABLE_GENERATION_SCHEMA_VERSION,
        "run_identity": identity,
        "level_index": level_index,
        "level_id": level_id,
        "static_level_signature": level_signature,
    }
    for key, value in expected.items():
        if part.get(key) != value:
            raise ValueError(
                f"incompatible exact-generation checkpoint {path}: "
                f"{key}={part.get(key)!r}, expected {value!r}")
    if not isinstance(part.get("records"), list) or not isinstance(
            part.get("stats"), dict):
        raise ValueError(f"malformed exact-generation checkpoint: {path}")
    return part


def _merge_parts(parts: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    stats = _generation_stats()
    for part in parts:
        part_stats = part["stats"]
        for key in stats:
            if not key.startswith("records"):
                stats[key] += int(part_stats.get(key, 0))
        for record in part["records"]:
            dk = (record["static_level_signature"], record["state_key"])
            if dk in by_key:
                _merge_records(by_key[dk], record)
                stats["duplicates_merged"] += 1
            else:
                by_key[dk] = copy.deepcopy(record)
    stats["records"] = len(by_key)
    _set_record_kind_counts(stats, by_key.values())
    return list(by_key.values()), stats


def _astar_summary(parts: list[dict[str, Any]]) -> dict[str, Any]:
    queries = [
        query
        for part in parts
        for query in part.get("astar_queries", [])
    ]
    by_role: dict[str, int] = {}
    by_termination_reason: dict[str, int] = {}
    for query in queries:
        role = str(query.get("query_role", "unknown"))
        reason = str(query.get("termination_reason", "unknown"))
        by_role[role] = by_role.get(role, 0) + 1
        by_termination_reason[reason] = by_termination_reason.get(reason, 0) + 1
    return {
        "queries": len(queries),
        "by_role": by_role,
        "by_termination_reason": by_termination_reason,
        "states_explored": sum(
            int(query.get("states_explored", 0)) for query in queries),
        "states_generated": sum(
            int(query.get("states_generated", 0)) for query in queries),
        "duplicate_states": sum(
            int(query.get("duplicate_states", 0)) for query in queries),
        "peak_frontier_size": max(
            (int(query.get("max_frontier_size", 0)) for query in queries),
            default=0,
        ),
        "elapsed_seconds": sum(
            float(query.get("elapsed_seconds", 0.0)) for query in queries),
    }


def _progress_payload(
    *,
    identity: str,
    levels_path: Path,
    levels_sha256: str,
    output: Path,
    config: dict[str, Any],
    parts: list[dict[str, Any]],
    total_levels: int,
    complete: bool = False,
) -> dict[str, Any]:
    _records, stats = _merge_parts(parts)
    payload: dict[str, Any] = {
        "schema_version": RESUMABLE_GENERATION_SCHEMA_VERSION,
        "run_identity": identity,
        "levels_source": {
            "path": str(levels_path),
            "sha256": levels_sha256,
            "count": total_levels,
        },
        "output": str(output),
        "config": config,
        "completed_levels": len(parts),
        "total_levels": total_levels,
        "elapsed_completed_level_seconds": sum(
            float(part["elapsed_seconds"]) for part in parts),
        "stats": stats,
        "astar": _astar_summary(parts),
        "complete": complete,
    }
    if complete:
        payload["output_sha256"] = sha256_file(output)
    return payload


def _load_all_parts(
    levels: list[tuple[str, Level]],
    parts_dir: Path,
    identity: str,
) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for index, (level_id, level) in enumerate(levels):
        path = parts_dir / f"level_{index:05d}.json"
        if not path.is_file():
            raise ValueError(
                f"completed exact generation is missing checkpoint shard: {path}")
        parts.append(_load_part(
            path,
            identity=identity,
            level_index=index,
            level_id=level_id,
            level_signature=static_level_signature(level),
        ))
    return parts


def _failed_report_path(output: Path) -> Path:
    return output.with_name(f"{output.stem}_report.json")


def _export_failed_levels(
    *,
    levels: list[tuple[str, Level]],
    parts: list[dict[str, Any]],
    output: str | Path,
    identity: str,
    levels_path: Path,
    levels_sha256: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    retry_output = Path(output)
    report_path = _failed_report_path(retry_output)
    if retry_output.resolve() == levels_path.resolve():
        raise ValueError("failed-level output cannot overwrite the source levels")
    if retry_output.exists() or report_path.exists():
        if not report_path.is_file():
            raise FileExistsError(
                f"failed-level output already exists without its report: "
                f"{retry_output}")
        try:
            existing_report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"invalid failed-level report: {report_path}") from exc
        if existing_report.get("run_identity") != identity:
            raise FileExistsError(
                "failed-level output belongs to a different generation run")

    failed_indices = [
        index
        for index, part in enumerate(parts)
        if int(part["stats"].get("candidates", 0)) > 0
        and int(part["stats"].get("records", 0)) == 0
    ]
    failed_levels = [level_to_dict(levels[index][1]) for index in failed_indices]
    atomic_write_json(retry_output, failed_levels)
    failures = []
    for index in failed_indices:
        part = parts[index]
        failures.append({
            "level_index": index,
            "level_id": part["level_id"],
            "static_level_signature": part["static_level_signature"],
            "stratum": part.get("stratum"),
            "elapsed_seconds": part.get("elapsed_seconds"),
            "stats": part["stats"],
            "astar": _astar_summary([part]),
            "astar_queries": part.get("astar_queries", []),
        })
    report = {
        "schema_version": RESUMABLE_GENERATION_SCHEMA_VERSION,
        "run_identity": identity,
        "source_levels": {
            "path": str(levels_path),
            "sha256": levels_sha256,
            "count": len(levels),
        },
        "source_config": config,
        "failed_level_definition": "candidate_count_positive_and_record_count_zero",
        "failed_level_count": len(failed_indices),
        "failed_level_indices": failed_indices,
        "output": {
            "path": str(retry_output),
            "sha256": sha256_file(retry_output),
        },
        "failures": failures,
    }
    atomic_write_json(report_path, report)
    return {
        "failed_levels_output": str(retry_output),
        "failed_levels_output_sha256": report["output"]["sha256"],
        "failed_levels_report": str(report_path),
        "failed_level_count": len(failed_indices),
    }


def generate_records_resumable(
    levels_path: str | Path,
    output: str | Path,
    *,
    modes: list[str],
    samples_per_level: int = 10,
    seed: int = 0,
    policy_target: str = "uniform-optimal",
    temperature: float = DEFAULT_TEMPERATURE,
    max_nodes: int = 300_000,
    time_limit_seconds: Optional[float] = None,
    walk_length: int = 6,
    deviation_prob: float = 0.3,
    near_optimal_step_cap: int = 40,
    value_norm_constant: float = DEFAULT_VALUE_NORM_CONSTANT,
    resume: bool = True,
    failed_levels_output: str | Path | None = None,
    label_mode: str = LABEL_FULL_EXACT,
) -> dict[str, Any]:
    """Generate atomically checkpointed per-level shards and merge on success."""
    _validate_limits(max_nodes, time_limit_seconds)
    _validate_label_mode(label_mode, modes)
    levels_path = Path(levels_path)
    destination = Path(output)
    levels = load_levels(levels_path)
    levels_sha256 = sha256_file(levels_path)
    config = {
        "modes": list(modes),
        "samples_per_level": samples_per_level,
        "seed": seed,
        "policy_target": policy_target,
        "temperature": temperature,
        "max_nodes": max_nodes,
        "time_limit_seconds": time_limit_seconds,
        "walk_length": walk_length,
        "deviation_prob": deviation_prob,
        "near_optimal_step_cap": near_optimal_step_cap,
        "value_norm_constant": value_norm_constant,
        "label_mode": label_mode,
    }
    identity = _run_identity(levels_sha256, config)
    progress_path, parts_dir = _checkpoint_paths(destination)
    if not resume and (destination.exists() or progress_path.exists()
                       or parts_dir.exists()):
        raise FileExistsError(
            "exact-generation artifacts already exist and --no-resume was set")
    if destination.exists():
        if not progress_path.is_file():
            raise FileExistsError(
                f"output already exists without a verified progress file: {destination}")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (progress.get("run_identity") != identity
                or not progress.get("complete")
                or progress.get("output_sha256") != sha256_file(destination)):
            raise ValueError(
                "existing exact dataset does not match this resumable run")
        if failed_levels_output is not None:
            parts = _load_all_parts(levels, parts_dir, identity)
            progress.update(_export_failed_levels(
                levels=levels,
                parts=parts,
                output=failed_levels_output,
                identity=identity,
                levels_path=levels_path,
                levels_sha256=levels_sha256,
                config=config,
            ))
            atomic_write_json(progress_path, progress)
        return progress

    stored_progress: dict[str, Any] | None = None
    if progress_path.exists():
        stored_progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if stored_progress.get("run_identity") != identity:
            raise ValueError(
                "cannot resume exact generation: input or settings changed")
    parts_dir.mkdir(parents=True, exist_ok=True)
    existing_indices: list[int] = []
    for path in parts_dir.glob("level_*.json"):
        try:
            existing_indices.append(int(path.stem.removeprefix("level_")))
        except ValueError as exc:
            raise ValueError(f"unexpected checkpoint shard name: {path}") from exc
    existing_indices.sort()
    if existing_indices != list(range(len(existing_indices))):
        raise ValueError(
            f"exact-generation checkpoint shards are not contiguous: "
            f"{existing_indices}")
    if (stored_progress is not None
            and int(stored_progress.get("completed_levels", 0))
            > len(existing_indices)):
        raise ValueError(
            "exact-generation progress references missing checkpoint shards")

    parts: list[dict[str, Any]] = []
    for index, (level_id, level) in enumerate(levels):
        signature = static_level_signature(level)
        part_path = parts_dir / f"level_{index:05d}.json"
        if part_path.is_file():
            part = _load_part(
                part_path,
                identity=identity,
                level_index=index,
                level_id=level_id,
                level_signature=signature,
            )
            parts.append(part)
            continue

        colors = len({block.color for block in level.blocks})
        stratum = f"{level.rows}x{level.cols}_c{colors}"
        print(
            f"[{index + 1}/{len(levels)}] {stratum} {level_id}: starting",
            flush=True,
        )
        started = time.perf_counter()
        # A fresh oracle per level bounds cache lifetime while retaining reuse
        # between the root and its successors.
        env = Environment()
        astar_queries: list[dict[str, Any]] = []
        oracle = Oracle(
            env, max_nodes=max_nodes,
            time_limit_seconds=time_limit_seconds,
            search_observer=astar_queries.append)
        level_seed = (seed * 1_000_003 + index) & 0xFFFFFFFF
        level_records: list[dict[str, Any]] = []
        level_stats = _generation_stats()
        by_key: dict[tuple[str, str], dict[str, Any]] = {}
        for state, provenance in _candidates(
            env, oracle, level_id, level, modes,
            samples_per_level=samples_per_level,
            seed=level_seed,
            walk_length=walk_length,
            deviation_prob=deviation_prob,
            near_optimal_step_cap=near_optimal_step_cap,
            max_nodes=max_nodes,
            time_limit_seconds=time_limit_seconds,
        ):
            level_stats["candidates"] += 1
            record, skip_reason = _label_candidate(
                env, oracle, state, level_id=level_id,
                label_mode=label_mode, policy_target=policy_target,
                temperature=temperature, max_nodes=max_nodes,
                time_limit_seconds=time_limit_seconds,
                value_norm_constant=value_norm_constant,
                provenance=provenance, astar_queries=astar_queries,
            )
            if record is None:
                level_stats[skip_reason] += 1
                continue
            dk = (record["static_level_signature"], record["state_key"])
            if dk in by_key:
                _merge_records(by_key[dk], record)
                level_stats["duplicates_merged"] += 1
            else:
                by_key[dk] = record
        level_records.extend(by_key.values())
        level_stats["records"] = len(level_records)
        _set_record_kind_counts(level_stats, level_records)
        elapsed = time.perf_counter() - started
        part = {
            "schema_version": RESUMABLE_GENERATION_SCHEMA_VERSION,
            "run_identity": identity,
            "level_index": index,
            "level_id": level_id,
            "static_level_signature": signature,
            "stratum": stratum,
            "elapsed_seconds": elapsed,
            "stats": level_stats,
            "astar_queries": astar_queries,
            "records": level_records,
        }
        atomic_write_json(part_path, part)
        parts.append(part)
        atomic_write_json(progress_path, _progress_payload(
            identity=identity,
            levels_path=levels_path,
            levels_sha256=levels_sha256,
            output=destination,
            config=config,
            parts=parts,
            total_levels=len(levels),
        ))
        skipped = sum(
            value for key, value in level_stats.items()
            if key.startswith("skipped_"))
        print(
            f"[{index + 1}/{len(levels)}] {stratum} {level_id}: "
            f"done in {elapsed:.2f}s, records={level_stats['records']}, "
            f"skipped={skipped}, astar_queries={len(astar_queries)}, "
            f"states_explored={sum(q['states_explored'] for q in astar_queries)}",
            flush=True,
        )

    records, stats = _merge_parts(parts)
    write_jsonl(records, destination)
    summary = _progress_payload(
        identity=identity,
        levels_path=levels_path,
        levels_sha256=levels_sha256,
        output=destination,
        config=config,
        parts=parts,
        total_levels=len(levels),
        complete=True,
    )
    summary.update({
        "version": DATASET_VERSION,
        "levels": len(levels),
        **stats,
    })
    if failed_levels_output is not None:
        summary.update(_export_failed_levels(
            levels=levels,
            parts=parts,
            output=failed_levels_output,
            identity=identity,
            levels_path=levels_path,
            levels_sha256=levels_sha256,
            config=config,
        ))
    atomic_write_json(progress_path, summary)
    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate exact policy-value training data.")
    p.add_argument("--levels", required=True, help="path to a levels JSON file")
    p.add_argument("--output", required=True, help="output JSONL path")
    p.add_argument("--sampling", nargs="+", default=["optimal-path"],
                   choices=SAMPLING_MODES, help="sampling modes to use")
    p.add_argument("--samples-per-level", type=int, default=10,
                   help="walks per level for random-reachable / near-optimal")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--policy-target", default="uniform-optimal",
                   choices=["uniform-optimal", "soft-regret"])
    p.add_argument(
        "--label-mode", default=LABEL_FULL_EXACT, choices=LABEL_MODES,
        help=(
            "full-exact proves every successor; "
            "full-exact-with-path-fallback retains a proven optimal root path "
            "when successor proofs exhaust; exact-path-policy performs only "
            "the root proof"),
    )
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--max-nodes", type=int, default=300_000)
    p.add_argument(
        "--time-limit-seconds",
        type=float,
        help="wall-clock cap for each individual A* query",
    )
    p.add_argument("--walk-length", type=int, default=6)
    p.add_argument("--deviation-prob", type=float, default=0.3)
    p.add_argument("--near-optimal-step-cap", type=int, default=40)
    p.add_argument("--value-normalization-constant", type=float,
                   default=DEFAULT_VALUE_NORM_CONSTANT)
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="refuse existing output/checkpoint artifacts instead of resuming",
    )
    p.add_argument(
        "--failed-levels-output",
        help=(
            "write levels that produced zero exact records to this JSON file, "
            "plus a sibling diagnostic report"),
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    summary = generate_records_resumable(
        args.levels,
        args.output,
        modes=args.sampling,
        samples_per_level=args.samples_per_level,
        seed=args.seed,
        policy_target=args.policy_target,
        temperature=args.temperature,
        max_nodes=args.max_nodes,
        time_limit_seconds=args.time_limit_seconds,
        walk_length=args.walk_length,
        deviation_prob=args.deviation_prob,
        near_optimal_step_cap=args.near_optimal_step_cap,
        value_norm_constant=args.value_normalization_constant,
        resume=not args.no_resume,
        failed_levels_output=args.failed_levels_output,
        label_mode=args.label_mode,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
