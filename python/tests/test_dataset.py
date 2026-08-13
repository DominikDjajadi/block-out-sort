"""Tests for policy/value targets, dataset generation, and validation."""

from __future__ import annotations

import copy
import json

import pytest

from blocksort import Environment, Oracle, level_from_dict
from blocksort.dataset.generate import (
    LABEL_FULL_EXACT_WITH_PATH_FALLBACK,
    generate_records,
    write_jsonl,
)
from blocksort.dataset.schema import (
    DATASET_VERSION,
    LABEL_EXACT_PATH_POLICY,
    POLICY_SINGLE_VERIFIED_OPTIMAL,
    build_exact_path_record,
    build_record,
    deserialize_state,
    serialize_state,
)
from blocksort.solver import solve_astar
from blocksort.dataset.targets import (
    soft_regret_policy,
    uniform_optimal_policy,
    value_target,
)
from blocksort.dataset import validate as validate_mod
from blocksort.dataset import generate as generate_mod
from blocksort.dataset.validate import validate_dataset, validate_record

ENV = Environment()

SINGLE = level_from_dict({
    "name": "single", "cols": 4, "rows": 4,
    "blocks": [{"color": "red", "cells": [[2, 1]]}],
    "exits": [{"edge": "top", "start": 1, "length": 1, "color": "red"}],
})

TWO = level_from_dict({
    "name": "two", "cols": 5, "rows": 5,
    "blocks": [{"color": "red", "cells": [[0, 0]]},
               {"color": "blue", "cells": [[4, 4]]}],
    "exits": [{"edge": "top", "start": 0, "length": 1, "color": "red"},
              {"edge": "bottom", "start": 4, "length": 1, "color": "blue"}],
})


# ---- policy / value targets ----

def _ordered_actions(level):
    o = Oracle(ENV)
    state = ENV.initial_state(level)
    analysis = o.analyze(state)
    return analysis


def test_uniform_optimal_single():
    analysis = _ordered_actions(SINGLE)
    policy = uniform_optimal_policy(analysis.actions)
    assert sum(1 for p in policy if p > 0) == 1
    assert pytest.approx(sum(policy)) == 1.0


def test_uniform_optimal_multiple():
    analysis = _ordered_actions(TWO)
    policy = uniform_optimal_policy(analysis.actions)
    nonzero = [p for p in policy if p > 0]
    assert len(nonzero) == 2
    assert all(pytest.approx(p) == 0.5 for p in nonzero)
    assert pytest.approx(sum(policy)) == 1.0


def test_soft_regret_normalization_sums_one():
    analysis = _ordered_actions(SINGLE)
    policy = soft_regret_policy(analysis.actions, temperature=1.0)
    assert pytest.approx(sum(policy)) == 1.0
    assert all(p >= 0 for p in policy)


def test_soft_regret_temperature_concentrates_on_optimal():
    analysis = _ordered_actions(SINGLE)
    opt_idx = next(i for i, a in enumerate(analysis.actions) if a.optimal)
    cold = soft_regret_policy(analysis.actions, temperature=0.1)
    hot = soft_regret_policy(analysis.actions, temperature=10.0)
    assert cold[opt_idx] > hot[opt_idx]


def test_policy_length_matches_legal_actions():
    analysis = _ordered_actions(TWO)
    policy = uniform_optimal_policy(analysis.actions)
    assert len(policy) == len(analysis.legal_actions)


def test_value_target_preserves_raw_and_normalizes():
    vt = value_target(8, constant=20.0)
    assert vt["raw_optimal_moves"] == 8
    assert pytest.approx(vt["normalized_value"]) == -8 / 20.0
    assert vt["normalization"] == {"scheme": "neg_over_constant", "constant": 20.0}


# ---- generation ----

def test_generate_deterministic_with_seed():
    a, _ = generate_records([("t", TWO)], modes=["optimal-path", "near-optimal"],
                            samples_per_level=3, seed=7)
    b, _ = generate_records([("t", TWO)], modes=["optimal-path", "near-optimal"],
                            samples_per_level=3, seed=7)
    assert [json.dumps(r, sort_keys=True) for r in a] == \
           [json.dumps(r, sort_keys=True) for r in b]


def test_jsonl_round_trip(tmp_path):
    records, _ = generate_records([("t", TWO)], modes=["optimal-path"], seed=0)
    path = tmp_path / "ds.jsonl"
    write_jsonl(records, path)
    loaded = [json.loads(line) for line in path.read_text().splitlines()]
    assert loaded == records


def test_optimal_path_sampling_labels_each_nonterminal_state():
    records, _ = generate_records([("t", TWO)], modes=["optimal-path"], seed=0)
    # TWO needs 2 moves -> two non-terminal states on the path.
    assert len(records) == 2
    assert {r["optimal_remaining_moves"] for r in records} == {2, 1}


def test_random_reachable_sampling_all_exact():
    records, stats = generate_records([("t", TWO)], modes=["random-reachable"],
                                      samples_per_level=5, seed=1)
    assert all(r["optimal_remaining_moves"] is not None for r in records)
    assert stats["skipped_exhausted"] == 0


def test_near_optimal_sampling_produces_records():
    records, _ = generate_records([("t", TWO)], modes=["near-optimal"],
                                  samples_per_level=3, seed=2, deviation_prob=0.5)
    assert records


def test_dedup_no_conflicting_duplicate_keys():
    records, stats = generate_records([("t", TWO)],
                                      modes=["initial", "optimal-path"], seed=0)
    keys = [(r["static_level_signature"], r["state_key"]) for r in records]
    assert len(keys) == len(set(keys))  # output is deduplicated
    assert stats["duplicates_merged"] >= 1  # initial overlaps optimal-path start


def test_exhausted_states_are_skipped():
    records, stats = generate_records([("t", TWO)], modes=["initial"],
                                      seed=0, max_nodes=1)
    assert records == []
    assert stats["skipped_exhausted"] >= 1


def test_every_generated_record_validates(tmp_path):
    records, _ = generate_records(
        [("t", TWO)], modes=["optimal-path", "random-reachable", "near-optimal"],
        samples_per_level=4, seed=3,
    )
    path = tmp_path / "ds.jsonl"
    write_jsonl(records, path)
    total, invalid, errors = validate_dataset(path)
    assert invalid == 0, errors
    assert total == len(records)


def test_exact_path_record_is_explicit_partial_policy_proof(tmp_path):
    state = ENV.initial_state(TWO)
    result = solve_astar(ENV, state)
    record = build_exact_path_record(
        result, state, ENV, level_id="two", provenance={"sampling": "initial"})

    assert record is not None
    assert record["label_kind"] == LABEL_EXACT_PATH_POLICY
    assert record["policy"]["type"] == POLICY_SINGLE_VERIFIED_OPTIMAL
    assert record["value_exact"] is True
    assert record["optimal_actions_complete"] is False
    assert record["action_values_complete"] is False
    assert sum(value is not None for value in record["action_costs"]) == 1
    assert sum(value == 0 for value in record["action_regrets"] if value is not None) == 1
    assert sum(probability == 1.0 for probability in record["policy_target"]) == 1
    path = tmp_path / "path.jsonl"
    write_jsonl([record], path)
    total, invalid, errors = validate_dataset(path)
    assert (total, invalid) == (1, 0), errors


def test_exact_path_generation_uses_root_only_and_rejects_hidden_sampling():
    records, stats = generate_records(
        [("two", TWO)], modes=["initial"], label_mode=LABEL_EXACT_PATH_POLICY)
    assert len(records) == 1
    assert stats["skipped_successor_exhausted"] == 0
    with pytest.raises(ValueError, match="requires --sampling initial"):
        generate_records(
            [("two", TWO)], modes=["optimal-path"],
            label_mode=LABEL_EXACT_PATH_POLICY)


def test_full_exact_falls_back_to_cached_exact_path_on_successor_exhaustion():
    strict_records, strict_stats = generate_records(
        [("two", TWO)], modes=["initial"], max_nodes=2,
        label_mode="full-exact")
    fallback_records, fallback_stats = generate_records(
        [("two", TWO)], modes=["initial"], max_nodes=2,
        label_mode=LABEL_FULL_EXACT_WITH_PATH_FALLBACK)

    assert strict_records == []
    assert strict_stats["skipped_successor_exhausted"] == 1
    assert len(fallback_records) == 1
    assert fallback_records[0]["label_kind"] == LABEL_EXACT_PATH_POLICY
    assert fallback_records[0]["provenance"][0]["labeling"] == {
        "strategy": "full_exact_then_exact_path",
        "fallback_reason": "successor_proof_incomplete",
    }
    assert fallback_stats["records_exact_path"] == 1
    assert fallback_stats["skipped_successor_exhausted"] == 0


def test_generation_dedup_keeps_full_exact_over_exact_path():
    [full], _ = generate_records(
        [("two", TWO)], modes=["initial"], max_nodes=10_000,
        label_mode="full-exact")
    [path], _ = generate_records(
        [("two", TWO)], modes=["initial"], max_nodes=10_000,
        label_mode=LABEL_EXACT_PATH_POLICY)

    generate_mod._merge_records(path, full)

    assert path["label_kind"] == "full-exact"
    assert path["action_values_complete"] is True
    assert len(path["provenance"]) == 2


def test_exact_path_validator_rejects_claims_about_unproved_actions():
    records, _ = generate_records(
        [("two", TWO)], modes=["initial"], label_mode=LABEL_EXACT_PATH_POLICY)
    record = records[0]
    unknown_index = next(
        index for index, value in enumerate(record["action_costs"])
        if value is None)
    record["action_costs"][unknown_index] = record["optimal_remaining_moves"] + 1
    errors = validate_record(record, Oracle(ENV), ENV, 1)
    assert any(error.code == "unproved_action_labelled" for error in errors)


# ---- validation detection ----

def _one_record():
    records, _ = generate_records([("t", TWO)], modes=["optimal-path"], seed=0)
    return records[0]


def test_detect_bad_version():
    rec = _one_record()
    rec["version"] = 999
    errors = validate_record(rec, Oracle(ENV), ENV, 1)
    assert any("version" in e for e in errors)


def test_detect_incorrect_state_key():
    rec = _one_record()
    rec["state_key"] = "bogus"
    errors = validate_record(rec, Oracle(ENV), ENV, 1)
    assert any("state_key" in e for e in errors)


def test_detect_illegal_stored_action():
    rec = _one_record()
    # Corrupt a legal action so it no longer matches the environment's set.
    rec["legal_actions"][0]["cells"] = [[3, 3]]
    errors = validate_record(rec, Oracle(ENV), ENV, 1)
    assert any("legal_actions" in e for e in errors)


def test_detect_incorrect_value():
    rec = _one_record()
    rec["optimal_remaining_moves"] = rec["optimal_remaining_moves"] + 1
    errors = validate_record(rec, Oracle(ENV), ENV, 1)
    assert any("optimal_remaining_moves" in e for e in errors)


def test_detect_incorrect_regret():
    rec = _one_record()
    rec["action_regrets"][0] = (rec["action_regrets"][0] or 0) + 5
    errors = validate_record(rec, Oracle(ENV), ENV, 1)
    assert any("regret" in e for e in errors)


def test_detect_policy_not_summing_to_one():
    rec = _one_record()
    rec["policy_target"] = [p / 2 for p in rec["policy_target"]]
    errors = validate_record(rec, Oracle(ENV), ENV, 1)
    assert any("sum" in e for e in errors)


def test_detect_conflicting_duplicate(tmp_path):
    rec = _one_record()
    other = copy.deepcopy(rec)
    other["optimal_remaining_moves"] = rec["optimal_remaining_moves"] + 3
    path = tmp_path / "dup.jsonl"
    path.write_text(json.dumps(rec, sort_keys=True) + "\n" +
                    json.dumps(other, sort_keys=True) + "\n")
    total, invalid, errors = validate_dataset(path)
    assert invalid >= 1
    assert any("duplicate" in e for e in errors)


def test_malformed_dataset_rows_produce_structured_line_diagnostics_and_continue(
    tmp_path,
):
    valid = _one_record()
    path = tmp_path / "mixed.jsonl"
    path.write_text(
        "[]\n"
        + json.dumps({"version": DATASET_VERSION, "state": {}}) + "\n"
        + json.dumps(valid) + "\n",
        encoding="utf-8",
    )
    total, invalid, diagnostics = validate_dataset(path)
    assert (total, invalid) == (3, 2)
    assert [d.line for d in diagnostics[:2]] == [1, 2]
    assert diagnostics[0].dataset_path == str(path)
    assert diagnostics[0].code == "non_object_row"
    assert any(d.code == "missing_level" for d in diagnostics)
    assert all(d.field and d.message for d in diagnostics)


@pytest.mark.parametrize(("mutate", "code"), [
    (lambda r: r.update(legal_actions="bad"), "invalid_legal_actions"),
    (lambda r: r["legal_actions"].__setitem__(0, "bad"),
     "malformed_action_list"),
    (lambda r: r["legal_actions"][0].update(color=[]),
     "invalid_action_identity"),
    (lambda r: r["legal_actions"][0].update(cells=[[{"bad": True}, 1]]),
     "invalid_action_identity"),
    (lambda r: r.update(state=[]), "invalid_state_shape"),
    (lambda r: r.update(optimal_actions=[]), "empty_optimal_actions"),
    (lambda r: r["value_target"].update(normalized_value=float("nan")),
     "nonfinite_value_target"),
    (lambda r: r["value_target"]["normalization"].update(constant=0),
     "invalid_normalization_constant"),
    (lambda r: r["action_regrets"].__setitem__(0, -1),
     "invalid_action_regret"),
    (lambda r: r["policy"].update(type="not-a-policy"),
     "invalid_policy_type"),
])
def test_record_validation_normalizes_malformed_fields(mutate, code):
    record = _one_record()
    mutate(record)
    diagnostics = validate_record(record, Oracle(ENV), ENV, 7)
    assert any(d.code == code and d.line == 7 for d in diagnostics)


def test_training_loader_rejects_non_object_with_line_number(tmp_path):
    from blocksort.training.dataset import load_records
    path = tmp_path / "bad.jsonl"
    path.write_text("{}\n[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"line 2.*JSON row must be an object"):
        load_records(path)


@pytest.mark.parametrize("value", [[], {"bad": "object"}])
def test_unhashable_state_key_produces_identity_diagnostic_and_continues(
        tmp_path, value):
    malformed = _one_record()
    malformed["state_key"] = value
    valid = _one_record()
    valid["level_id"] = "valid-after-malformed"
    path = tmp_path / "unhashable.jsonl"
    path.write_text(
        json.dumps(malformed) + "\n" + json.dumps(valid) + "\n",
        encoding="utf-8")

    total, invalid, diagnostics = validate_dataset(path)

    assert (total, invalid) == (2, 1)
    diagnostic = next(
        item for item in diagnostics
        if item.field == "state_key" and item.code == "invalid_identity_type")
    assert diagnostic.line == 1
    assert diagnostic.dataset_path == str(path)
    assert "list" in diagnostic.message or "dict" in diagnostic.message


def test_nested_malformed_signature_and_missing_identity_are_structured(
        tmp_path):
    nested = _one_record()
    nested["static_level_signature"] = {"nested": []}
    missing = _one_record()
    del missing["state_key"]
    valid = _one_record()
    path = tmp_path / "identity-errors.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in (nested, missing, valid)) + "\n",
        encoding="utf-8")

    total, invalid, diagnostics = validate_dataset(path)

    assert (total, invalid) == (3, 2)
    assert {(item.line, item.field, item.code) for item in diagnostics} >= {
        (1, "static_level_signature", "invalid_identity_type"),
        (2, "state_key", "missing_state_key"),
    }


def test_valid_duplicate_detection_survives_malformed_identity_rows(tmp_path):
    malformed = _one_record()
    malformed["state_key"] = []
    first = _one_record()
    duplicate = copy.deepcopy(first)
    duplicate["optimal_remaining_moves"] += 1
    different = _one_record()
    different["state_key"] = "different-valid-key"
    path = tmp_path / "duplicate-after-malformed.jsonl"
    path.write_text(
        "\n".join(json.dumps(row) for row in (
            malformed, first, duplicate, different)) + "\n",
        encoding="utf-8")

    total, invalid, diagnostics = validate_dataset(path)

    assert total == 4
    assert invalid == 3
    assert any(
        item.line == 3 and item.code == "conflicting_duplicate"
        for item in diagnostics)
    assert not any(
        item.line == 4 and item.code == "conflicting_duplicate"
        for item in diagnostics)


def test_dataset_validation_does_not_swallow_internal_programming_errors(
        tmp_path, monkeypatch):
    path = tmp_path / "valid.jsonl"
    path.write_text(json.dumps(_one_record()) + "\n", encoding="utf-8")

    def unexpected(_level):
        raise RuntimeError("unexpected validator defect")

    monkeypatch.setattr(validate_mod, "static_level_signature", unexpected)
    with pytest.raises(RuntimeError, match="unexpected validator defect"):
        validate_dataset(path)


def test_validation_cli_emits_json_line_identity_diagnostics(
        tmp_path, capsys):
    malformed = _one_record()
    malformed["state_key"] = []
    valid = _one_record()
    valid["level_id"] = "later-valid-record"
    path = tmp_path / "cli.jsonl"
    path.write_text(
        json.dumps(malformed) + "\n" + json.dumps(valid) + "\n",
        encoding="utf-8")

    assert validate_mod.main([str(path)]) == 1
    output = capsys.readouterr().out.splitlines()
    diagnostic = json.loads(output[0])
    assert diagnostic["dataset_path"] == str(path)
    assert diagnostic["line"] == 1
    assert diagnostic["record_id"].startswith(malformed["level_id"])
    assert diagnostic["field"] == "state_key"
    assert diagnostic["code"] == "invalid_identity_type"
    assert diagnostic["message"]
