"""State generation for an expert-iteration round.

Reuses the supervised generator's sampling walks (initial states, optimal-path
states, random legal deviations, positive-regret deviations) and adds a level
provider that draws from the base train levels plus optional procedurally
generated levels. Returns deduplicated ``(State, provenance)`` candidates.
"""

from __future__ import annotations

import random
from typing import Any, Iterator

from ..dataset.generate import (
    _near_optimal_states,
    _optimal_path_states,
    _random_walk_state,
)
from ..environment import Environment
from ..oracle import Oracle
from ..schema import Level
from ..serialization import level_from_dict
from ..signature import static_level_signature
from ..state import State, canonical_key


def base_train_levels(records: list[dict[str, Any]], train_signatures: set[str]
                      ) -> list[tuple[str, Level]]:
    """Unique levels (by signature) from base records restricted to train split."""
    seen: dict[str, tuple[str, Level]] = {}
    for r in records:
        sig = r.get("static_level_signature")
        if sig in train_signatures and sig not in seen:
            seen[sig] = (r["level_id"], level_from_dict(r["level"]))
    return list(seen.values())


def load_level_pool(path: str) -> list[tuple[str, Level]]:
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "levels" in data:
        data = data["levels"]
    if isinstance(data, dict):
        data = [data]
    stem = Path(path).stem
    return [(f"{stem}#{i}", level_from_dict(raw)) for i, raw in enumerate(data)]


def sample_levels(pool: list[tuple[str, Level]], n: int, seed: int
                  ) -> list[tuple[str, Level]]:
    if not pool:
        return []
    rng = random.Random(seed)
    if n >= len(pool):
        return list(pool)
    return rng.sample(pool, n)


def generate_states(
    env: Environment,
    oracle: Oracle,
    levels: list[tuple[str, Level]],
    *,
    states_per_level: int,
    seed: int,
    astar_max_nodes: int,
    astar_time_limit_seconds: float | None = None,
    walk_length: int = 6,
    deviation_prob: float = 0.3,
    near_optimal_step_cap: int = 30,
    optimal_path_state_limit: int | None = None,
    near_optimal_walks_per_level: int | None = None,
) -> tuple[list[tuple[State, dict[str, Any]]], set[str]]:
    """Generate deduplicated candidate states. Returns ``(candidates, sigs)``.

    Candidate modes: initial, optimal-path, random legal deviations, and
    positive-regret (near-optimal) deviations.
    """
    for name, value in (
            ("optimal_path_state_limit", optimal_path_state_limit),
            ("near_optimal_walks_per_level", near_optimal_walks_per_level)):
        if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
                or value < 0):
            raise ValueError(f"{name} must be None or a non-negative integer")
    out: list[tuple[State, dict[str, Any]]] = []
    seen: set[tuple[str, str]] = set()
    signatures: set[str] = set()

    for idx, (level_id, level) in enumerate(levels):
        sig = static_level_signature(level)
        signatures.add(sig)
        lvl_seed = (seed * 1_000_003 + idx) & 0xFFFFFFFF
        rng = random.Random(lvl_seed)
        initial = env.initial_state(level)

        candidates: list[tuple[State, dict[str, Any]]] = []
        candidates.append((initial, {"sampling": "initial", "level_id": level_id}))
        optimal_states = _optimal_path_states(
            env, initial, astar_max_nodes,
            time_limit_seconds=astar_time_limit_seconds)
        # The initial state is already included above. When requested, retain a
        # deterministic spread across the remaining optimal trajectory rather
        # than allowing long solutions to explode the training-state count.
        optimal_states = optimal_states[1:]
        if (optimal_path_state_limit is not None
                and len(optimal_states) > optimal_path_state_limit):
            if optimal_path_state_limit <= 0:
                optimal_states = []
            elif optimal_path_state_limit == 1:
                optimal_states = [optimal_states[len(optimal_states) // 2]]
            else:
                last = len(optimal_states) - 1
                indices = [
                    round(i * last / (optimal_path_state_limit - 1))
                    for i in range(optimal_path_state_limit)
                ]
                optimal_states = [optimal_states[i] for i in indices]
        for i, st in enumerate(optimal_states):
            candidates.append((st, {"sampling": "optimal-path", "level_id": level_id,
                                    "step": i}))
        for w in range(states_per_level):
            st = _random_walk_state(env, rng, initial, walk_length)
            candidates.append((st, {"sampling": "random-deviation",
                                    "level_id": level_id, "walk": w}))
        near_walks = (
            states_per_level if near_optimal_walks_per_level is None
            else near_optimal_walks_per_level)
        for w in range(near_walks):
            for i, st in enumerate(_near_optimal_states(
                    env, oracle, rng, initial, deviation_prob, near_optimal_step_cap)):
                candidates.append((st, {"sampling": "positive-regret-deviation",
                                        "level_id": level_id, "walk": w, "step": i}))

        for state, prov in candidates:
            if env.is_terminal(state):
                continue
            dk = (sig, canonical_key(state))
            if dk in seen:
                continue
            seen.add(dk)
            out.append((state, prov))

    return out, signatures
