from __future__ import annotations

import json
from pathlib import Path

import pytest

from blocksort.cotraining import transfer_audit as audit
from blocksort.serialization import level_from_dict, level_to_dict


def _level(name: str, row: int = 1):
    return level_from_dict({
        "name": name,
        "cols": 4,
        "rows": 4,
        "blocks": [{"color": "red", "cells": [[row, 1]]}],
        "exits": [{
            "edge": "left", "start": row, "length": 1, "color": "red",
        }],
    })


def _row(identity: str, *, solved: dict[int, bool]):
    return {
        "search_identity": identity,
        "static_level_signature": f"static-{identity}",
        "name": identity,
        "budgets": {
            str(budget): {
                "solved": outcome,
                "solution_length": 2 if outcome else None,
            }
            for budget, outcome in solved.items()
        },
    }


def test_paired_group_summary_reports_discordant_outcomes():
    incumbent = [
        _row("a", solved={10: True}),
        _row("b", solved={10: True}),
        _row("c", solved={10: False}),
        _row("d", solved={10: False}),
    ]
    candidate = [
        _row("a", solved={10: True}),
        _row("b", solved={10: False}),
        _row("c", solved={10: True}),
        _row("d", solved={10: False}),
    ]

    result = audit._paired_group_summary(
        "round_001", (10,), incumbent, candidate)
    item = result["per_budget"]["10"]

    assert item["both_solved"] == 1
    assert item["incumbent_only"] == 1
    assert item["candidate_only"] == 1
    assert item["neither_solved"] == 1
    assert item["solve_rate_delta"] == 0.0
    assert len(result["discordant_outcomes"]) == 2


def test_paired_group_summary_rejects_mismatched_levels():
    with pytest.raises(ValueError, match="different levels"):
        audit._paired_group_summary(
            "round_001", (10,),
            [_row("a", solved={10: True})],
            [_row("b", solved={10: True})],
        )


@pytest.mark.parametrize("budgets", [(), (1, 1), (0,), (True,)])
def test_transfer_config_rejects_invalid_budgets(tmp_path, budgets):
    files = []
    for name in ("inc.pt", "cand.pt", "pool.jsonl", "split.json", "levels.json"):
        path = tmp_path / name
        path.write_text("x", encoding="utf-8")
        files.append(path)
    cfg = audit.TransferAuditConfig(
        incumbent_checkpoint=str(files[0]),
        candidate_checkpoint=str(files[1]),
        generated_groups=(
            audit.GeneratedGroup("round_001", str(files[4])),),
        eval_levels_dataset=str(files[2]),
        eval_split_manifest=str(files[3]),
        output_dir=str(tmp_path / "out"),
        generated_budgets=budgets,
    )

    with pytest.raises(ValueError, match="generated_budgets"):
        cfg.validate()


def test_generated_group_reserves_validation_name(tmp_path):
    path = tmp_path / "levels.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="reserved"):
        audit.GeneratedGroup(
            "promotion_validation", str(path)).validate()


def test_load_level_list_accepts_hard_pool_jsonl(tmp_path):
    first = _level("one")
    second = _level("two", row=2)
    pool = tmp_path / "holdout.jsonl"
    records = [
        {
            "hard_eval_pool_schema_version": 2,
            "level": level_to_dict(level),
        }
        for level in (first, second)
    ]
    pool.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )

    loaded = audit._load_level_list(pool)

    assert [level.name for level in loaded] == ["one", "two"]


def test_run_transfer_audit_reuses_completed_group_evaluations(
    tmp_path, monkeypatch,
):
    incumbent = tmp_path / "inc.pt"
    candidate = tmp_path / "cand.pt"
    pool = tmp_path / "pool.jsonl"
    manifest = tmp_path / "split.json"
    generated = tmp_path / "levels.json"
    for path in (incumbent, candidate, pool, manifest):
        path.write_text(path.name, encoding="utf-8")
    level = _level("one")
    generated.write_text(
        json.dumps([level_to_dict(level)]), encoding="utf-8")
    monkeypatch.setattr(
        audit, "_load_validation_levels",
        lambda *_args, **_kwargs: [_level("validation", row=2)])
    calls = []

    def fake_evaluate(checkpoint, levels, *, group_name, budgets, **_kwargs):
        calls.append((Path(checkpoint).name, group_name))
        solved = Path(checkpoint).name == "cand.pt"
        return [
            _row(
                f"{group_name}-{index}",
                solved={budget: solved for budget in budgets},
            )
            for index, _level_item in enumerate(levels)
        ]

    monkeypatch.setattr(audit, "_evaluate_checkpoint_group", fake_evaluate)
    cfg = audit.TransferAuditConfig(
        incumbent_checkpoint=str(incumbent),
        candidate_checkpoint=str(candidate),
        generated_groups=(
            audit.GeneratedGroup("round_001", str(generated)),),
        eval_levels_dataset=str(pool),
        eval_split_manifest=str(manifest),
        output_dir=str(tmp_path / "out"),
        generated_budgets=(10,),
        validation_budgets=(5,),
    )

    first = audit.run_transfer_audit(cfg)
    second = audit.run_transfer_audit(cfg)

    assert len(calls) == 4
    assert first == second
    assert first["final_test_status"] == "sealed_not_evaluated"
    assert first["groups"][0]["per_budget"]["10"]["candidate_only"] == 1
    assert (tmp_path / "out" / "summary.csv").is_file()


def test_run_transfer_audit_rejects_changed_settings(tmp_path, monkeypatch):
    incumbent = tmp_path / "inc.pt"
    candidate = tmp_path / "cand.pt"
    pool = tmp_path / "pool.jsonl"
    manifest = tmp_path / "split.json"
    generated = tmp_path / "levels.json"
    for path in (incumbent, candidate, pool, manifest):
        path.write_text(path.name, encoding="utf-8")
    generated.write_text(
        json.dumps([level_to_dict(_level("one"))]), encoding="utf-8")
    monkeypatch.setattr(
        audit, "_load_validation_levels",
        lambda *_args, **_kwargs: [_level("validation", row=2)])
    monkeypatch.setattr(
        audit, "_evaluate_checkpoint_group",
        lambda _checkpoint, levels, *, group_name, budgets, **_kwargs: [
            _row(
                f"{group_name}-{index}",
                solved={budget: False for budget in budgets},
            )
            for index, _level_item in enumerate(levels)
        ])
    base = dict(
        incumbent_checkpoint=str(incumbent),
        candidate_checkpoint=str(candidate),
        generated_groups=(
            audit.GeneratedGroup("round_001", str(generated)),),
        eval_levels_dataset=str(pool),
        eval_split_manifest=str(manifest),
        output_dir=str(tmp_path / "out"),
        generated_budgets=(10,),
        validation_budgets=(5,),
    )
    audit.run_transfer_audit(audit.TransferAuditConfig(**base))

    with pytest.raises(RuntimeError, match="different inputs or settings"):
        audit.run_transfer_audit(audit.TransferAuditConfig(
            **{**base, "generated_budgets": (20,)}))


def test_run_transfer_audit_resumes_after_completed_level(
        tmp_path, monkeypatch):
    from blocksort.environment import Environment
    from blocksort.search.seeding import level_search_identity
    from blocksort.signature import static_level_signature

    incumbent = tmp_path / "inc.pt"
    candidate = tmp_path / "cand.pt"
    pool = tmp_path / "pool.jsonl"
    manifest = tmp_path / "split.json"
    generated = tmp_path / "levels.json"
    for path in (incumbent, candidate, pool, manifest):
        path.write_text(path.name, encoding="utf-8")
    levels = [_level("one"), _level("two", row=2)]
    generated.write_text(
        json.dumps([level_to_dict(level) for level in levels]),
        encoding="utf-8",
    )
    validation = _level("validation", row=3)
    monkeypatch.setattr(
        audit, "_load_validation_levels",
        lambda *_args, **_kwargs: [validation])

    env = Environment()
    interrupted = False
    resumed_prefixes = []

    def result_row(level, index, budgets):
        identity = level_search_identity(env, level)
        return {
            **_row(
                identity,
                solved={budget: False for budget in budgets},
            ),
            "index": index,
            "name": level.name,
            "static_level_signature": static_level_signature(level),
        }

    def fake_evaluate(
        checkpoint, evaluated_levels, *, budgets,
        initial_rows=None, checkpoint_rows=None, **_kwargs,
    ):
        nonlocal interrupted
        rows = list(initial_rows or [])
        resumed_prefixes.append(
            (Path(checkpoint).name, len(evaluated_levels), len(rows)))
        for index in range(len(rows), len(evaluated_levels)):
            rows.append(result_row(evaluated_levels[index], index, budgets))
            checkpoint_rows(rows)
            if not interrupted:
                interrupted = True
                raise RuntimeError("injected per-level interruption")
        return rows

    monkeypatch.setattr(audit, "_evaluate_checkpoint_group", fake_evaluate)
    cfg = audit.TransferAuditConfig(
        incumbent_checkpoint=str(incumbent),
        candidate_checkpoint=str(candidate),
        generated_groups=(
            audit.GeneratedGroup("round_001", str(generated)),),
        eval_levels_dataset=str(pool),
        eval_split_manifest=str(manifest),
        output_dir=str(tmp_path / "out"),
        generated_budgets=(10,),
        validation_budgets=(5,),
    )

    with pytest.raises(RuntimeError, match="per-level interruption"):
        audit.run_transfer_audit(cfg)
    partial = (
        tmp_path / "out" / "evaluations"
        / "round_001.incumbent.json.partial.json"
    )
    assert len(json.loads(partial.read_text(encoding="utf-8"))["levels"]) == 1

    result = audit.run_transfer_audit(cfg)

    assert ("inc.pt", 2, 1) in resumed_prefixes
    assert not partial.exists()
    assert result["groups"][0]["levels"] == 2
