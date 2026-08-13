"""JavaScript vs Python solver parity (requires Node).

Compares solvability, minimum move count, and exit-only status (not explored
counts). Covers handcrafted levels, generated levels, conformance-fixture
setups, and random reachable states (the latter two synthesized as levels for
unlock-free bases, where the cleared count does not affect the rules).
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import pytest

from blocksort import Environment, Level, level_from_dict, level_to_dict
from blocksort.cli.compare_js import compare_level_dicts
from blocksort.dataset.schema import serialize_state

REPO_ROOT = Path(__file__).resolve().parents[2]
ENV = Environment()

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None, reason="node not available"
)


def _is_unlock_free(level: Level) -> bool:
    return (
        not level.locked_regions
        and all(b.unlock_at == 0 for b in level.blocks)
        and all(e.unlock_at == 0 for e in level.exits)
    )


def _synth_level(level: Level, block_specs: list[dict]) -> dict:
    d = level_to_dict(level)
    d["blocks"] = block_specs
    d["name"] = (d.get("name") or "synth") + " [synthesized state]"
    return d


def _random_states(level: Level, n: int, steps: int, seed: int) -> list[dict]:
    """A few unlock-free reachable states, serialized as level block specs."""
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        cur = ENV.initial_state(level)
        for _ in range(rng.randint(1, steps)):
            if ENV.is_terminal(cur):
                break
            legal = ENV.legal_actions(cur)
            if not legal:
                break
            cur = ENV.apply_action(cur, rng.choice(legal))
        if cur.remaining:
            out.append(serialize_state(cur)["blocks"])
    return out


def _collect_parity_levels() -> list[dict]:
    levels: list[dict] = []

    # Handcrafted levels (initial states, including the unlock level).
    handcrafted = json.loads((REPO_ROOT / "fixtures" / "levels.json").read_text())
    levels.extend(handcrafted)

    # Generated levels (initial states).
    gen = json.loads((REPO_ROOT / "fixtures" / "generated_levels.json").read_text())
    levels.extend(gen)

    # Conformance-fixture setups (unlock-free fixtures only).
    conf_dir = REPO_ROOT / "fixtures" / "conformance"
    for path in sorted(conf_dir.glob("*.json")):
        fixture = json.loads(path.read_text())
        level = level_from_dict(fixture["level"])
        if not _is_unlock_free(level):
            continue
        for test in fixture["tests"]:
            setup = test.get("setup")
            if setup and setup.get("blocks"):
                levels.append(_synth_level(level, setup["blocks"]))

    # Random reachable states for unlock-free handcrafted levels.
    for i, raw in enumerate(handcrafted):
        level = level_from_dict(raw)
        if not _is_unlock_free(level):
            continue
        for specs in _random_states(level, n=3, steps=4, seed=100 + i):
            levels.append(_synth_level(level, specs))

    return levels


def test_js_python_solver_parity():
    levels = _collect_parity_levels()
    report = compare_level_dicts(levels, max_nodes=200_000)
    assert report["mismatches"] == 0, json.dumps(
        [r for r in report["results"] if not r["match"]], indent=2
    )
    # Sanity: we actually compared a non-trivial number of levels.
    assert report["levels"] >= 10
