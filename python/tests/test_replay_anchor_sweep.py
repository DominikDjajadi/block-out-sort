from __future__ import annotations

from blocksort.cotraining.replay_anchor_sweep import (
    compose_arm_sample,
    select_level_balanced_anchors,
)


def _record(level: str, iteration: int, state: str = "s") -> dict:
    return {
        "static_level_signature": level,
        "state_key": state,
        "generation_iteration": iteration,
    }


def test_level_balanced_anchor_selection_covers_every_level() -> None:
    groups = {
        f"band-{band}": {
            f"b{band}-l{level:02d}": [
                _record(f"b{band}-l{level:02d}", 0, f"s{state}")
                for state in range(3)
            ]
            for level in range(50)
        }
        for band in range(4)
    }

    light = select_level_balanced_anchors(
        groups, count=200, round_number=1)
    moderate = select_level_balanced_anchors(
        groups, count=400, round_number=2)

    assert len(light) == 200
    assert len({row["static_level_signature"] for row in light}) == 200
    assert len(moderate) == 400
    counts = {}
    for row in moderate:
        key = row["static_level_signature"]
        counts[key] = counts.get(key, 0) + 1
    assert set(counts.values()) == {2}


def test_compose_replaces_only_historical_tail() -> None:
    round_number = 10
    current = [_record(f"c{i}", round_number) for i in range(700)]
    recent = [_record(f"r{i}", round_number - 1) for i in range(500)]
    historical = [_record(f"h{i}", 0) for i in range(800)]
    anchors = [_record(f"a{i}", 0) for i in range(200)]

    rows, composition = compose_arm_sample(
        [*current, *recent, *historical],
        anchors,
        round_number=round_number,
        recent_window=2,
    )

    assert rows[:1200] == [*current, *recent]
    assert rows[1200:1800] == historical[:600]
    assert rows[1800:] == anchors
    assert composition == {
        "current": 700,
        "recent": 500,
        "ordinary_historical": 600,
        "difficulty_anchor": 200,
    }


def test_compose_preserves_redistributed_first_round_quotas() -> None:
    current = [_record(f"c{i}", 1) for i in range(933)]
    historical = [_record(f"h{i}", 0) for i in range(1067)]
    anchors = [_record(f"a{i}", 0) for i in range(400)]

    rows, composition = compose_arm_sample(
        [*current, *historical],
        anchors,
        round_number=1,
        recent_window=2,
    )

    assert len(rows) == 2000
    assert rows[:933] == current
    assert rows[933:1600] == historical[:667]
    assert rows[1600:] == anchors
    assert composition == {
        "current": 933,
        "recent": 0,
        "ordinary_historical": 667,
        "difficulty_anchor": 400,
    }
