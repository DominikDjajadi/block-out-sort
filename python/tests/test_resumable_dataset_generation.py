from __future__ import annotations

import json

import pytest

from blocksort.dataset import generate as generate_mod
from blocksort.dataset.schema import LABEL_EXACT_PATH_POLICY
from blocksort.serialization import level_to_dict
from blocksort import level_from_dict


SINGLE = level_from_dict({
    "name": "single", "cols": 4, "rows": 4,
    "blocks": [{"color": "red", "cells": [[2, 1]]}],
    "exits": [
        {"edge": "top", "start": 1, "length": 1, "color": "red"}
    ],
})

TWO = level_from_dict({
    "name": "two", "cols": 5, "rows": 5,
    "blocks": [
        {"color": "red", "cells": [[0, 0]]},
        {"color": "blue", "cells": [[4, 4]]},
    ],
    "exits": [
        {"edge": "top", "start": 0, "length": 1, "color": "red"},
        {"edge": "bottom", "start": 4, "length": 1, "color": "blue"},
    ],
})


def _write_levels(path) -> None:
    path.write_text(json.dumps([
        level_to_dict(SINGLE),
        level_to_dict(TWO),
    ]), encoding="utf-8")


def test_resumable_generation_keeps_completed_levels_after_interrupt(
    tmp_path, monkeypatch, capsys,
):
    levels = tmp_path / "levels.json"
    output = tmp_path / "records.jsonl"
    _write_levels(levels)
    real_oracle = generate_mod.Oracle
    constructions = 0

    def interrupt_on_second_level(*args, **kwargs):
        nonlocal constructions
        constructions += 1
        if constructions == 2:
            raise KeyboardInterrupt
        return real_oracle(*args, **kwargs)

    monkeypatch.setattr(generate_mod, "Oracle", interrupt_on_second_level)
    with pytest.raises(KeyboardInterrupt):
        generate_mod.generate_records_resumable(
            levels, output, modes=["initial"], max_nodes=10_000,
            time_limit_seconds=1.0)

    progress_path, parts_dir = generate_mod._checkpoint_paths(output)
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    assert progress["completed_levels"] == 1
    assert progress["astar"]["queries"] >= 1
    assert (parts_dir / "level_00000.json").is_file()
    assert not (parts_dir / "level_00001.json").exists()
    assert not output.exists()

    monkeypatch.setattr(generate_mod, "Oracle", real_oracle)
    summary = generate_mod.generate_records_resumable(
        levels, output, modes=["initial"], max_nodes=10_000,
        time_limit_seconds=1.0)

    assert summary["complete"] is True
    assert summary["completed_levels"] == 2
    assert summary["records"] == 2
    assert summary["astar"]["queries"] >= 2
    assert summary["astar"]["states_explored"] > 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2
    first_part = json.loads(
        (parts_dir / "level_00000.json").read_text(encoding="utf-8"))
    roles = {query["query_role"] for query in first_part["astar_queries"]}
    assert "root" in roles
    assert "successor" in roles
    assert all(query["termination_reason"] for query in first_part["astar_queries"])
    console = capsys.readouterr().out
    # The completed first level was reused, not run a second time.
    assert console.count("levels#0: starting") == 1
    assert console.count("levels#1: starting") == 2
    assert "states_explored=" in console


def test_resume_rejects_changed_input_or_settings(tmp_path):
    levels = tmp_path / "levels.json"
    output = tmp_path / "records.jsonl"
    _write_levels(levels)
    generate_mod.generate_records_resumable(
        levels, output, modes=["initial"], max_nodes=10_000,
        time_limit_seconds=1.0)

    with pytest.raises(ValueError, match="does not match|settings changed"):
        generate_mod.generate_records_resumable(
            levels, output, modes=["initial"], max_nodes=9_999,
            time_limit_seconds=1.0)


def test_resumable_exact_path_mode_records_one_root_query_per_level(tmp_path):
    levels = tmp_path / "levels.json"
    output = tmp_path / "path_records.jsonl"
    _write_levels(levels)
    summary = generate_mod.generate_records_resumable(
        levels, output, modes=["initial"], max_nodes=10_000,
        time_limit_seconds=1.0, label_mode=LABEL_EXACT_PATH_POLICY)

    assert summary["records"] == 2
    assert summary["astar"]["queries"] == 2
    assert summary["astar"]["by_role"] == {"root": 2}
    assert all(
        json.loads(line)["label_kind"] == LABEL_EXACT_PATH_POLICY
        for line in output.read_text(encoding="utf-8").splitlines())


def test_no_resume_refuses_existing_checkpoint_artifacts(tmp_path):
    levels = tmp_path / "levels.json"
    output = tmp_path / "records.jsonl"
    _write_levels(levels)
    generate_mod.generate_records_resumable(
        levels, output, modes=["initial"], max_nodes=10_000,
        time_limit_seconds=1.0)

    with pytest.raises(FileExistsError, match="--no-resume"):
        generate_mod.generate_records_resumable(
            levels, output, modes=["initial"], max_nodes=10_000,
            time_limit_seconds=1.0, resume=False)


def test_completed_run_can_export_failed_retry_pool_without_recomputation(
    tmp_path, capsys,
):
    levels = tmp_path / "levels.json"
    output = tmp_path / "records.jsonl"
    failed = tmp_path / "retry_levels.json"
    levels.write_text(json.dumps([
        level_to_dict(TWO),
        level_to_dict(TWO),
    ]), encoding="utf-8")
    first = generate_mod.generate_records_resumable(
        levels, output, modes=["initial"], max_nodes=1,
        time_limit_seconds=1.0)
    assert first["records"] == 0
    initial_console = capsys.readouterr().out
    assert initial_console.count(": starting") == 2

    resumed = generate_mod.generate_records_resumable(
        levels, output, modes=["initial"], max_nodes=1,
        time_limit_seconds=1.0, failed_levels_output=failed)

    assert capsys.readouterr().out == ""
    assert resumed["failed_level_count"] == 2
    assert len(json.loads(failed.read_text(encoding="utf-8"))) == 2
    report_path = generate_mod._failed_report_path(failed)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["failed_level_count"] == 2
    assert all(failure["astar"]["queries"] == 1
               for failure in report["failures"])
    assert all(
        failure["astar_queries"][0]["query_role"] == "root"
        for failure in report["failures"]
    )
    assert all(
        failure["astar_queries"][0]["termination_reason"] == "node_limit"
        for failure in report["failures"]
    )


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_per_query_time_limit_must_be_finite_positive(value):
    with pytest.raises(ValueError, match="time_limit_seconds"):
        generate_mod.generate_records(
            [("single", SINGLE)], modes=["initial"],
            time_limit_seconds=value)
