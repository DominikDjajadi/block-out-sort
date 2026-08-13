from __future__ import annotations

import copy
import json

import pytest

from blocksort import Environment, Oracle, level_from_dict
from blocksort.dataset.combine import combine_records, write_combined_dataset
from blocksort.dataset.generate import write_jsonl
from blocksort.dataset.schema import build_exact_path_record, build_record
from blocksort.solver import solve_astar


LEVEL = level_from_dict({
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


def _records():
    env = Environment()
    state = env.initial_state(LEVEL)
    full = build_record(
        Oracle(env).analyze(state), state, level_id="two",
        provenance={"sampling": "full"})
    path = build_exact_path_record(
        solve_astar(env, state), state, env, level_id="two",
        provenance={"sampling": "path"})
    assert full is not None and path is not None
    return full, path


def test_full_exact_record_supersedes_path_record_and_merges_provenance(tmp_path):
    full, path = _records()
    path_file = tmp_path / "path.jsonl"
    full_file = tmp_path / "full.jsonl"
    write_jsonl([path], path_file)
    write_jsonl([full], full_file)

    combined, summary = combine_records([path_file, full_file])

    assert len(combined) == 1
    assert combined[0]["label_kind"] == "full-exact"
    assert {entry["sampling"] for entry in combined[0]["provenance"]} == {
        "full", "path"}
    assert summary["duplicates"] == 1
    assert summary["stronger_replacements"] == 1


def test_conflicting_exact_root_values_are_rejected(tmp_path):
    full, _path = _records()
    conflict = copy.deepcopy(full)
    conflict["optimal_remaining_moves"] += 1
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write_jsonl([full], first)
    write_jsonl([conflict], second)

    with pytest.raises(ValueError, match="conflicting exact root values"):
        combine_records([first, second])


def test_write_combined_dataset_is_atomic_and_refuses_overwrite(tmp_path):
    full, path = _records()
    source_a = tmp_path / "a.jsonl"
    source_b = tmp_path / "b.jsonl"
    output = tmp_path / "combined.jsonl"
    write_jsonl([full], source_a)
    write_jsonl([path], source_b)

    summary = write_combined_dataset([source_a, source_b], output)
    report = output.with_name("combined_report.json")
    assert summary["output"]["records"] == 1
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1
    assert json.loads(report.read_text(encoding="utf-8"))["duplicates"] == 1
    with pytest.raises(FileExistsError, match="already exists"):
        write_combined_dataset([source_a, source_b], output)
