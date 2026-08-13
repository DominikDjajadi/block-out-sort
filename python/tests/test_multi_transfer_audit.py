from __future__ import annotations

from pathlib import Path

import pytest

from blocksort.cotraining import multi_transfer_audit as multi
from blocksort.cotraining.transfer_audit import GeneratedGroup
from blocksort.environment import Environment
from blocksort.search.seeding import level_search_identity
from blocksort.serialization import level_from_dict
from blocksort.signature import static_level_signature


def _level(name: str, row: int):
    return level_from_dict({
        "name": name,
        "cols": 4,
        "rows": 4,
        "blocks": [{"color": "red", "cells": [[row, 1]]}],
        "exits": [{
            "edge": "left",
            "start": row,
            "length": 1,
            "color": "red",
        }],
    })


def test_multi_candidate_audit_evaluates_incumbent_once_and_resumes(
        tmp_path, monkeypatch):
    incumbent = tmp_path / "inc.pt"
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    generated_path = tmp_path / "generated.json"
    dataset = tmp_path / "pool.jsonl"
    manifest = tmp_path / "split.json"
    for path in (
            incumbent, first, second, generated_path, dataset, manifest):
        path.write_text(path.name, encoding="utf-8")
    generated = [_level("generated", 1)]
    validation = [_level("validation", 2)]
    monkeypatch.setattr(
        multi, "_load_level_list", lambda _path: generated)
    monkeypatch.setattr(
        multi, "_load_validation_levels",
        lambda *_args, **_kwargs: validation)

    env = Environment()
    calls = []

    def fake_evaluate(
        checkpoint, levels, *, budgets,
        initial_rows=None, checkpoint_rows=None, **_kwargs,
    ):
        calls.append((Path(checkpoint).name, levels[0].name))
        rows = list(initial_rows or [])
        for index in range(len(rows), len(levels)):
            level = levels[index]
            identity = level_search_identity(env, level)
            solved = Path(checkpoint).name != "inc.pt"
            rows.append({
                "index": index,
                "name": level.name,
                "static_level_signature": static_level_signature(level),
                "search_identity": identity,
                "rows": level.rows,
                "cols": level.cols,
                "budgets": {
                    str(budget): {
                        "solved": solved,
                        "solution_length": 1 if solved else None,
                    }
                    for budget in budgets
                },
            })
            checkpoint_rows(rows)
        return rows

    monkeypatch.setattr(multi, "_evaluate_checkpoint_group", fake_evaluate)
    cfg = multi.MultiTransferAuditConfig(
        incumbent_checkpoint=str(incumbent),
        candidates=(
            multi.Candidate("first", str(first)),
            multi.Candidate("second", str(second)),
        ),
        generated_groups=(
            GeneratedGroup("holdout", str(generated_path)),),
        eval_levels_dataset=str(dataset),
        eval_split_manifest=str(manifest),
        output_dir=str(tmp_path / "out"),
        generated_budgets=(10,),
        validation_budgets=(5,),
    )

    result = multi.run_multi_transfer_audit(cfg)
    repeated = multi.run_multi_transfer_audit(cfg)

    assert len(calls) == 6
    assert result == repeated
    assert result["final_test_status"] == "sealed_not_evaluated"
    assert all(
        candidate["groups"][0]["per_budget"]["10"]["candidate_only"] == 1
        for candidate in result["candidates"]
    )
    assert (tmp_path / "out" / "summary.csv").is_file()


def test_validation_gate_skips_holdout_for_regressing_candidate(
        tmp_path, monkeypatch):
    incumbent = tmp_path / "inc.pt"
    passing = tmp_path / "passing.pt"
    failing = tmp_path / "failing.pt"
    generated_path = tmp_path / "generated.json"
    dataset = tmp_path / "pool.jsonl"
    manifest = tmp_path / "split.json"
    for path in (
            incumbent, passing, failing, generated_path, dataset, manifest):
        path.write_text(path.name, encoding="utf-8")
    generated = [_level("generated", 1)]
    validation = [_level("validation", 2)]
    monkeypatch.setattr(
        multi, "_load_level_list", lambda _path: generated)
    monkeypatch.setattr(
        multi, "_load_validation_levels",
        lambda *_args, **_kwargs: validation)

    env = Environment()
    calls = []

    def fake_evaluate(
        checkpoint, levels, *, budgets,
        initial_rows=None, checkpoint_rows=None, **_kwargs,
    ):
        checkpoint_name = Path(checkpoint).name
        calls.append((checkpoint_name, levels[0].name))
        rows = list(initial_rows or [])
        for index in range(len(rows), len(levels)):
            level = levels[index]
            solved = checkpoint_name == "passing.pt"
            rows.append({
                "index": index,
                "name": level.name,
                "static_level_signature": static_level_signature(level),
                "search_identity": level_search_identity(env, level),
                "rows": level.rows,
                "cols": level.cols,
                "budgets": {
                    str(budget): {
                        "solved": solved,
                        "solution_length": 1 if solved else None,
                    }
                    for budget in budgets
                },
            })
            checkpoint_rows(rows)
        return rows

    monkeypatch.setattr(multi, "_evaluate_checkpoint_group", fake_evaluate)
    cfg = multi.MultiTransferAuditConfig(
        incumbent_checkpoint=str(incumbent),
        candidates=(
            multi.Candidate("passing", str(passing)),
            multi.Candidate("failing", str(failing)),
        ),
        generated_groups=(
            GeneratedGroup("holdout", str(generated_path)),),
        eval_levels_dataset=str(dataset),
        eval_split_manifest=str(manifest),
        output_dir=str(tmp_path / "out"),
        generated_budgets=(10,),
        validation_budgets=(5,),
        validation_first=True,
        gate_validation_budgets=(5,),
        gate_validation_weights=(1.0,),
        gate_margin=0.1,
    )

    result = multi.run_multi_transfer_audit(cfg)
    by_name = {
        candidate["name"]: candidate for candidate in result["candidates"]
    }

    assert calls == [
        ("inc.pt", "validation"),
        ("passing.pt", "validation"),
        ("failing.pt", "validation"),
        ("inc.pt", "generated"),
        ("passing.pt", "generated"),
    ]
    assert by_name["passing"]["validation_gate"]["passed"]
    assert not by_name["failing"]["validation_gate"]["passed"]
    assert [
        group["group"] for group in by_name["passing"]["groups"]
    ] == ["promotion_validation", "holdout"]
    assert [
        group["group"] for group in by_name["failing"]["groups"]
    ] == ["promotion_validation"]


def test_validation_gate_requires_strictly_more_than_margin():
    summary = {
        "per_budget": {
            "4": {
                "incumbent_solve_rate": 0.2,
                "candidate_solve_rate": 0.3,
            },
        },
    }
    cfg = multi.MultiTransferAuditConfig(
        incumbent_checkpoint="unused",
        candidates=(),
        generated_groups=(),
        eval_levels_dataset="unused",
        eval_split_manifest="unused",
        output_dir="unused",
        validation_first=True,
        gate_validation_budgets=(4,),
        gate_validation_weights=(1.0,),
        gate_margin=0.1,
    )

    gate = multi._validation_gate(summary, cfg)

    assert gate["candidate_score"] == pytest.approx(gate["required_score"])
    assert not gate["passed"]


def test_generated_only_never_loads_promotion_validation(tmp_path, monkeypatch):
    incumbent = tmp_path / "inc.pt"
    candidate = tmp_path / "candidate.pt"
    generated_path = tmp_path / "generated.json"
    for path in (incumbent, candidate, generated_path):
        path.write_text(path.name, encoding="utf-8")
    generated = [_level("generated", 1)]
    monkeypatch.setattr(multi, "_load_level_list", lambda _path: generated)
    monkeypatch.setattr(
        multi, "_load_validation_levels",
        lambda *_args, **_kwargs: pytest.fail("validation must stay unread"))

    env = Environment()
    calls = []

    def fake_evaluate(
        checkpoint, levels, *, budgets,
        initial_rows=None, checkpoint_rows=None, **_kwargs,
    ):
        calls.append(Path(checkpoint).name)
        rows = list(initial_rows or [])
        if not rows:
            level = levels[0]
            rows.append({
                "index": 0,
                "name": level.name,
                "static_level_signature": static_level_signature(level),
                "search_identity": level_search_identity(env, level),
                "rows": level.rows,
                "cols": level.cols,
                "budgets": {
                    str(budget): {"solved": True, "solution_length": 1}
                    for budget in budgets
                },
            })
            checkpoint_rows(rows)
        return rows

    monkeypatch.setattr(multi, "_evaluate_checkpoint_group", fake_evaluate)
    cfg = multi.MultiTransferAuditConfig(
        incumbent_checkpoint=str(incumbent),
        candidates=(multi.Candidate("candidate", str(candidate)),),
        generated_groups=(GeneratedGroup("holdout", str(generated_path)),),
        eval_levels_dataset=None,
        eval_split_manifest=None,
        output_dir=str(tmp_path / "out"),
        generated_budgets=(10,),
        generated_only=True,
    )

    result = multi.run_multi_transfer_audit(cfg)

    assert calls == ["inc.pt", "candidate.pt"]
    assert [group["group"] for group in result["candidates"][0]["groups"]] == [
        "holdout"
    ]
    assert result["final_test_status"] == "sealed_not_evaluated"
