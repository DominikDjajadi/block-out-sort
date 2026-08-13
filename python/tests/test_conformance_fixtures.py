"""Run the shared conformance fixtures through the Python engine.

The same fixtures are evaluated by ``tools/run_conformance.js`` against the
JavaScript engine; agreement across both establishes parity.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blocksort import Environment, IllegalActionError, canonical_key
from blocksort.conformance import (
    actions_equal,
    build_state,
    normalized_actions,
    normalized_to_action,
)
from blocksort.serialization import level_from_dict

ENV = Environment()
CONFORMANCE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "conformance"


def _load_cases():
    cases = []
    for path in sorted(CONFORMANCE_DIR.glob("*.json")):
        fixture = json.loads(path.read_text(encoding="utf-8"))
        for test in fixture["tests"]:
            case_id = f"{path.stem}:{test['name']}"
            cases.append(pytest.param(fixture, test, id=case_id))
    return cases


CASES = _load_cases()


def test_fixtures_were_generated():
    assert CASES, "no conformance fixtures found; run python/tools/build_fixtures.py"


@pytest.mark.parametrize("fixture,test", CASES)
def test_conformance_case(fixture, test):
    level = level_from_dict(fixture["level"])
    state = build_state(ENV, level, test.get("setup"))

    expect = test.get("expect", {})
    if "canonicalKey" in expect:
        assert canonical_key(state) == expect["canonicalKey"]
    if "cleared" in expect:
        assert state.cleared == expect["cleared"]
    if "terminal" in expect:
        assert ENV.is_terminal(state) == expect["terminal"]
    if "legalActions" in expect:
        assert actions_equal(normalized_actions(ENV, state), expect["legalActions"])

    if "apply" not in test:
        return

    threw = False
    cur = state
    try:
        for norm in test["apply"]:
            action = normalized_to_action(cur, norm)
            cur = ENV.apply_action(cur, action)
    except IllegalActionError:
        threw = True

    if test.get("expectError"):
        assert threw, "expected an illegal action but all applied"
        return

    assert not threw, "unexpected illegal action"
    after = test.get("after", {})
    if "canonicalKey" in after:
        assert canonical_key(cur) == after["canonicalKey"]
    if "cleared" in after:
        assert cur.cleared == after["cleared"]
    if "terminal" in after:
        assert ENV.is_terminal(cur) == after["terminal"]
    if "legalActions" in after:
        assert actions_equal(normalized_actions(ENV, cur), after["legalActions"])
