"""Validate a supervised policy-value dataset (JSON Lines).

Every record is re-derived from the environment + exact oracle and checked for
internal consistency. Run:

    python -m blocksort.dataset.validate data/training/pv_examples.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

from ..conformance import _normalized_hashable, normalized_to_action
from ..environment import Environment, IllegalActionError
from ..oracle import Oracle
from ..serialization import level_from_dict
from ..signature import static_level_signature
from ..solution import deserialize_solution, verify_solution
from ..solver import solve_astar
from .schema import (
    DATASET_VERSION,
    LABEL_EXACT_PATH_POLICY,
    LABEL_FULL_EXACT,
    POLICY_SINGLE_VERIFIED_OPTIMAL,
    POLICY_SOFT_REGRET,
    POLICY_UNIFORM_OPTIMAL,
    deserialize_state,
)

PROB_TOLERANCE = 1e-6
_RECORD_DATA_ERRORS = (
    AttributeError, KeyError, TypeError, ValueError, IndexError,
    ZeroDivisionError, OverflowError,
)


@dataclass(frozen=True)
class ValidationDiagnostic:
    dataset_path: str
    line: int
    record_id: str
    field: str
    code: str
    message: str

    def __str__(self) -> str:
        return (
            f"{self.dataset_path}: line {self.line} [{self.record_id}] "
            f"{self.code} ({self.field}): {self.message}"
        )

    def __contains__(self, text: str) -> bool:
        """Retain convenient substring checks used by existing callers/tests."""
        return text in str(self)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _valid_identifier_part(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


def _record_id(record: dict[str, Any]) -> str:
    level_id = _valid_identifier_part(record.get("level_id")) or "?"
    state_key = _valid_identifier_part(record.get("state_key")) or "?"
    return f"{level_id} key={state_key}"


def _validate_record_identity(
    record: dict[str, Any],
    line_no: int,
    *,
    dataset_path: str,
) -> tuple[tuple[str, str] | None, list[ValidationDiagnostic]]:
    """Validate and canonicalize fields used as the dataset identity key."""
    diagnostics: list[ValidationDiagnostic] = []

    def invalid(field: str, message: str, code: str) -> None:
        diagnostics.append(ValidationDiagnostic(
            dataset_path=dataset_path,
            line=line_no,
            record_id=_record_id(record),
            field=field,
            code=code,
            message=message,
        ))

    values: dict[str, str] = {}
    for field in ("static_level_signature", "state_key"):
        if field not in record:
            invalid(field, "required identity field is missing", f"missing_{field}")
            continue
        value = record[field]
        if not isinstance(value, str):
            invalid(
                field,
                f"{field} must be a string; received {type(value).__name__}",
                "invalid_identity_type",
            )
            continue
        values[field] = value

    if "level_id" in record and _valid_identifier_part(
            record["level_id"]) is None:
        invalid(
            "level_id",
            "level_id must be a string or integer; "
            f"received {type(record['level_id']).__name__}",
            "invalid_identity_type",
        )

    if diagnostics:
        return None, diagnostics
    return (
        values["static_level_signature"],
        values["state_key"],
    ), diagnostics


def _canonical_action_identity(action: dict[str, Any]) -> tuple[Any, ...]:
    """Validate one serialized action before it is used as a set/dict key."""
    required = ("color", "cells", "dir", "distance", "exit")
    missing = [field for field in required if field not in action]
    if missing:
        raise ValueError(f"missing action identity fields: {missing}")
    if not isinstance(action["color"], str):
        raise ValueError("action color must be a string")
    if not isinstance(action["dir"], str):
        raise ValueError("action dir must be a string")
    if (isinstance(action["distance"], bool)
            or not isinstance(action["distance"], int)):
        raise ValueError("action distance must be an integer")
    if not isinstance(action["exit"], bool):
        raise ValueError("action exit must be a boolean")
    cells = action["cells"]
    if not isinstance(cells, list):
        raise ValueError("action cells must be a list")
    canonical_cells: list[tuple[int, int]] = []
    for cell in cells:
        if (not isinstance(cell, list) or len(cell) != 2
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in cell)):
            raise ValueError(
                "every action cell must be a two-integer list")
        canonical_cells.append((cell[0], cell[1]))
    return (
        action["color"],
        tuple(canonical_cells),
        action["dir"],
        action["distance"],
        action["exit"],
    )


def validate_record(
    record: dict[str, Any], oracle: Oracle, env: Environment, line_no: int,
    *, dataset_path: str = "<memory>",
) -> list[ValidationDiagnostic]:
    """Return structured problems with one record (empty means valid)."""
    errors: list[ValidationDiagnostic] = []

    def err(
        msg: str,
        *,
        field: str = "record",
        code: str = "invalid_record",
    ) -> None:
        rid = _record_id(record) if isinstance(record, dict) else "<non-object>"
        errors.append(ValidationDiagnostic(
            dataset_path=dataset_path, line=line_no, record_id=rid,
            field=field, code=code, message=msg))

    if not isinstance(record, dict):
        err("JSON row must be an object", code="non_object_row")
        return errors

    _identity, identity_errors = _validate_record_identity(
        record, line_no, dataset_path=dataset_path)
    errors.extend(identity_errors)

    # Version.
    if record.get("version") != DATASET_VERSION:
        err(f"unsupported version {record.get('version')!r}",
            field="version", code="unknown_schema_version")
        return errors

    required = (
        "level", "state", "static_level_signature", "state_key",
        "legal_actions", "optimal_actions", "action_costs", "action_regrets",
        "policy_target", "value_target", "optimal_remaining_moves",
    )
    missing = [field for field in required if field not in record]
    for field in missing:
        if field in {"static_level_signature", "state_key"}:
            continue
        err("required field is missing", field=field, code=f"missing_{field}")
    if missing:
        return errors
    if not isinstance(record["level"], dict):
        err("level must be an object", field="level", code="invalid_level")
    if not isinstance(record["state"], dict):
        err("state must be an object", field="state", code="invalid_state_shape")
    if errors:
        return errors

    for field in (
        "legal_actions", "optimal_actions", "action_costs", "action_regrets",
        "policy_target",
    ):
        if not isinstance(record[field], list):
            err("field must be a list", field=field, code=f"invalid_{field}")
    if errors:
        return errors
    if not record["legal_actions"]:
        err("labelled nonterminal record must have legal actions",
            field="legal_actions", code="empty_legal_actions")
    if not record["optimal_actions"]:
        err("exact labelled record must have at least one optimal action",
            field="optimal_actions", code="empty_optimal_actions")
    if any(not isinstance(action, dict) for action in record["legal_actions"]):
        err("every legal action must be an object",
            field="legal_actions", code="malformed_action_list")
    if any(not isinstance(action, dict) for action in record["optimal_actions"]):
        err("every optimal action must be an object",
            field="optimal_actions", code="malformed_action_list")
    if errors:
        return errors

    canonical_legal: list[tuple[Any, ...]] = []
    canonical_optimal: list[tuple[Any, ...]] = []
    for field, destination in (
        ("legal_actions", canonical_legal),
        ("optimal_actions", canonical_optimal),
    ):
        for index, action in enumerate(record[field]):
            try:
                destination.append(_canonical_action_identity(action))
            except ValueError as exc:
                err(
                    f"{field}[{index}] has invalid identity: {exc}",
                    field=field,
                    code="invalid_action_identity",
                )
    if errors:
        return errors

    policy = record["policy_target"]
    for i, value in enumerate(policy):
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            err(f"policy_target[{i}] must be a finite number; got {value!r}",
                field="policy_target", code="nonfinite_policy_target")
    for field in ("action_costs", "action_regrets"):
        for i, value in enumerate(record[field]):
            if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)):
                err(f"{field}[{i}] must be finite or null; got {value!r}",
                    field=field, code=f"invalid_{field[:-1]}")
            elif field == "action_regrets" and value is not None and value < 0:
                err(f"action_regrets[{i}] must be non-negative; got {value!r}",
                    field=field, code="invalid_action_regret")
    vt = record["value_target"]
    if not isinstance(vt, dict):
        err("value_target must be an object",
            field="value_target", code="invalid_value_target")
        return errors
    normalized = vt.get("normalized_value")
    if (isinstance(normalized, bool) or not isinstance(normalized, (int, float))
            or not math.isfinite(normalized)):
        err(f"normalized_value must be finite; got {normalized!r}",
            field="value_target.normalized_value",
            code="nonfinite_value_target")
    norm = vt.get("normalization")
    if not isinstance(norm, dict):
        err("normalization must be an object",
            field="value_target.normalization", code="invalid_normalization")
    else:
        constant = norm.get("constant")
        if (isinstance(constant, bool)
                or not isinstance(constant, (int, float))
                or not math.isfinite(constant) or constant <= 0):
            err(f"normalization constant must be finite and > 0; got {constant!r}",
                field="value_target.normalization.constant",
                code="invalid_normalization_constant")
    if errors:
        return errors

    # Level + state reconstruction.
    try:
        level = level_from_dict(record["level"])
        state = deserialize_state(level, record["state"])
    except _RECORD_DATA_ERRORS as exc:
        err(f"could not reconstruct level/state: {exc}",
            field="level/state", code="invalid_state_shape")
        return errors

    # Static signature matches the level.
    if record["static_level_signature"] != static_level_signature(level):
        err("static_level_signature does not match the level",
            field="static_level_signature", code="inconsistent_level_signature")

    # Dynamic state key matches.
    if record["state_key"] != env.canonical_key(state):
        err("state_key does not match the reconstructed state",
            field="state_key", code="inconsistent_state_key")

    # Cleared/remaining bookkeeping.
    if record.get("remaining_blocks") != state.remaining:
        err("remaining_blocks mismatch")
    if record.get("cleared_blocks") != state.cleared:
        err("cleared_blocks mismatch")

    # Legal actions match the environment.
    env_legal = {_normalized_hashable(a) for a in
                 _normalized_legal(env, state)}
    rec_legal_list = record["legal_actions"]
    rec_legal = set(canonical_legal)
    if env_legal != rec_legal:
        err("stored legal_actions do not match environment legal actions",
            field="legal_actions", code="inconsistent_legal_actions")
        return errors  # alignment checks below would be meaningless

    # Every stored action must be replayable in this state.
    for a in rec_legal_list:
        try:
            normalized_to_action(state, a)
        except _RECORD_DATA_ERRORS as exc:
            err(f"stored action not replayable: {a} ({exc})",
                field="legal_actions", code="invalid_action")

    # Every optimal action is legal.
    for a, action_key in zip(record["optimal_actions"], canonical_optimal):
        if action_key not in rec_legal:
            err(f"optimal action not in legal actions: {a}")

    costs = record["action_costs"]
    regrets = record["action_regrets"]
    policy = record["policy_target"]
    n = len(rec_legal_list)
    if not (len(costs) == len(regrets) == len(policy) == n):
        err("aligned arrays (legal_actions/costs/regrets/policy) length mismatch")
        return errors
    for i, probability in enumerate(policy):
        if probability < -PROB_TOLERANCE:
            err(f"negative policy probability at [{i}]")
    total = sum(policy)
    if not math.isclose(total, 1.0, abs_tol=1e-6):
        err(f"policy probabilities sum to {total}, expected ~1.0")

    label_kind = record.get("label_kind", LABEL_FULL_EXACT)
    if label_kind == LABEL_EXACT_PATH_POLICY:
        errors.extend(_validate_exact_path_record(
            record, state, oracle, env, line_no,
            dataset_path=dataset_path,
            canonical_legal=canonical_legal,
            canonical_optimal=canonical_optimal,
        ))
        return errors
    if label_kind != LABEL_FULL_EXACT:
        err(f"unsupported label_kind {label_kind!r}",
            field="label_kind", code="invalid_label_kind")
        return errors
    for field in (
        "value_exact", "policy_exact", "optimal_actions_complete",
        "action_values_complete",
    ):
        if field in record and record[field] is not True:
            err(f"full-exact record requires {field}=true", field=field,
                code="inconsistent_exactness_flag")

    # Recompute exact analysis and compare.
    analysis = oracle.analyze(state)
    if not (analysis.exact and analysis.solvable):
        err("oracle could not prove an exact value for this state")
        return errors

    if record["optimal_remaining_moves"] != analysis.value:
        err(f"optimal_remaining_moves {record['optimal_remaining_moves']} "
            f"!= oracle V* {analysis.value}")

    # Terminal states should not appear (no policy); guard anyway.
    if analysis.terminal or state.remaining == 0:
        err("terminal state should not be a labelled record")
        return errors

    # Build aligned lookup by action key.
    by_key = {_normalized_hashable(a.serialized): a for a in analysis.actions}
    has_zero_regret = False
    for i, a in enumerate(rec_legal_list):
        key = canonical_legal[i]
        analysis_a = by_key.get(key)
        if analysis_a is None:
            err(f"action {a} missing from oracle analysis")
            continue
        # Cost = 1 + successor optimal value.
        if analysis_a.cost is None:
            if costs[i] is not None:
                err(f"action cost should be null (infinite) for {a}")
        else:
            if costs[i] != analysis_a.cost:
                err(f"action_cost[{i}] {costs[i]} != 1 + successor value "
                    f"{analysis_a.cost}")
            # Regret = cost - V*.
            expected_regret = analysis_a.cost - analysis.value
            if regrets[i] != expected_regret:
                err(f"action_regret[{i}] {regrets[i]} != {expected_regret}")
            if expected_regret < 0:
                err(f"negative regret at [{i}]")
            if expected_regret == 0:
                has_zero_regret = True
    if not has_zero_regret:
        err("no legal action has zero regret for a solvable non-terminal state")

    # Policy-type specific checks.
    policy_meta = record.get("policy")
    if not isinstance(policy_meta, dict):
        err("policy metadata must be an object",
            field="policy", code="invalid_policy_type")
        return errors
    ptype = policy_meta.get("type")
    if ptype not in (POLICY_UNIFORM_OPTIMAL, POLICY_SOFT_REGRET):
        err(f"unsupported policy type {ptype!r}",
            field="policy.type", code="invalid_policy_type")
        return errors
    if ptype == POLICY_UNIFORM_OPTIMAL:
        opt_count = sum(1 for a in analysis.actions if a.optimal)
        expected_p = 1.0 / opt_count if opt_count else 0.0
        for i, a in enumerate(rec_legal_list):
            analysis_a = by_key.get(canonical_legal[i])
            target = expected_p if (analysis_a and analysis_a.optimal) else 0.0
            if not math.isclose(policy[i], target, abs_tol=1e-6):
                err(f"uniform-optimal policy[{i}] {policy[i]} != {target}")

    # Value target consistency.
    vt = record["value_target"]
    if vt.get("raw_optimal_moves") != analysis.value:
        err("value_target.raw_optimal_moves mismatch")
    norm = vt.get("normalization", {})
    if norm.get("scheme") == "neg_over_constant":
        const = norm.get("constant")
        expected = -analysis.value / const
        if not math.isclose(vt.get("normalized_value"), expected, rel_tol=1e-9, abs_tol=1e-12):
            err("value_target.normalized_value inconsistent with scheme/constant")

    return errors


def _validate_exact_path_record(
    record: dict[str, Any],
    state,
    oracle: Oracle,
    env: Environment,
    line_no: int,
    *,
    dataset_path: str,
    canonical_legal: list[tuple[Any, ...]],
    canonical_optimal: list[tuple[Any, ...]],
) -> list[ValidationDiagnostic]:
    """Validate a root-only exact path proof without analyzing successors."""
    errors: list[ValidationDiagnostic] = []

    def err(msg: str, *, field: str = "record", code: str = "invalid_record") -> None:
        errors.append(ValidationDiagnostic(
            dataset_path=dataset_path, line=line_no,
            record_id=_record_id(record), field=field, code=code, message=msg))

    expected_flags = {
        "value_exact": True,
        "policy_exact": True,
        "optimal_actions_complete": False,
        "action_values_complete": False,
    }
    for field, expected in expected_flags.items():
        if record.get(field) is not expected:
            err(f"exact-path-policy requires {field}={str(expected).lower()}",
                field=field, code="inconsistent_exactness_flag")

    policy_meta = record.get("policy")
    if not isinstance(policy_meta, dict) or policy_meta.get(
            "type") != POLICY_SINGLE_VERIFIED_OPTIMAL:
        err("exact-path-policy requires single-verified-optimal policy metadata",
            field="policy.type", code="invalid_policy_type")

    if len(canonical_optimal) != 1:
        err("exact-path-policy must store exactly one proven optimal action",
            field="optimal_actions", code="invalid_path_optimal_actions")
        return errors
    selected_key = canonical_optimal[0]
    try:
        selected_index = canonical_legal.index(selected_key)
    except ValueError:
        return errors  # common validation already reported non-legal optimum

    root_value = record.get("optimal_remaining_moves")
    if (isinstance(root_value, bool) or not isinstance(root_value, int)
            or root_value <= 0):
        err("exact-path-policy requires a positive integer root value",
            field="optimal_remaining_moves", code="invalid_exact_path_value")
        return errors

    for index, (cost, regret, probability) in enumerate(zip(
            record["action_costs"], record["action_regrets"],
            record["policy_target"])):
        if index == selected_index:
            if cost != root_value:
                err(f"selected action cost must equal root value {root_value}",
                    field=f"action_costs[{index}]",
                    code="inconsistent_path_action_cost")
            if regret != 0:
                err("selected action regret must be zero",
                    field=f"action_regrets[{index}]",
                    code="inconsistent_path_action_regret")
            if not math.isclose(probability, 1.0, abs_tol=PROB_TOLERANCE):
                err("selected action policy probability must be one",
                    field=f"policy_target[{index}]",
                    code="inconsistent_path_policy")
        else:
            if cost is not None or regret is not None:
                err("unproved actions must have null cost and regret",
                    field=f"action_costs/action_regrets[{index}]",
                    code="unproved_action_labelled")
            if not math.isclose(probability, 0.0, abs_tol=PROB_TOLERANCE):
                err("unselected action policy probability must be zero",
                    field=f"policy_target[{index}]",
                    code="inconsistent_path_policy")

    proof = record.get("policy_proof")
    if not isinstance(proof, dict):
        err("policy_proof must be an object", field="policy_proof",
            code="invalid_policy_proof")
        return errors
    path = proof.get("actions")
    if (proof.get("type") != "verified-optimal-path"
            or not isinstance(path, list) or not path
            or proof.get("length") != root_value
            or len(path) != root_value):
        err("policy_proof must contain a non-empty path of the exact root length",
            field="policy_proof", code="invalid_policy_proof")
        return errors
    try:
        first_key = _canonical_action_identity(path[0])
    except (TypeError, ValueError) as exc:
        err(f"invalid first proof action: {exc}", field="policy_proof.actions",
            code="invalid_policy_proof")
        return errors
    if first_key != selected_key:
        err("proof path first action does not match the policy action",
            field="policy_proof.actions", code="inconsistent_policy_proof")
    try:
        actions = deserialize_solution(env, state, path)
    except _RECORD_DATA_ERRORS as exc:
        err(f"proof path is not replayable: {exc}", field="policy_proof.actions",
            code="invalid_policy_proof")
        return errors
    if not verify_solution(env, state, actions, expected_move_count=root_value):
        err("proof path does not reach a terminal state at the stored length",
            field="policy_proof.actions", code="invalid_policy_proof")

    # A separate root-only A* proof establishes that the replayed path length is
    # globally optimal.  No successor analysis is performed here.
    result = solve_astar(
        env, state, max_nodes=oracle.max_nodes,
        time_limit_seconds=oracle.time_limit_seconds)
    if not (result.solvable is True and result.optimal):
        err("validator could not independently prove the root optimum",
            field="optimal_remaining_moves", code="root_proof_exhausted")
    elif result.move_count != root_value:
        err(f"stored root value {root_value} != A* optimum {result.move_count}",
            field="optimal_remaining_moves", code="inconsistent_exact_path_value")

    vt = record["value_target"]
    if vt.get("raw_optimal_moves") != root_value:
        err("value_target.raw_optimal_moves mismatch", field="value_target",
            code="inconsistent_value_target")
    norm = vt.get("normalization", {})
    if norm.get("scheme") == "neg_over_constant":
        const = norm.get("constant")
        expected = -root_value / const
        if not math.isclose(
                vt.get("normalized_value"), expected,
                rel_tol=1e-9, abs_tol=1e-12):
            err("value_target.normalized_value inconsistent with scheme/constant",
                field="value_target", code="inconsistent_value_target")
    return errors


def _normalized_legal(env: Environment, state):
    from ..conformance import action_to_normalized
    return [action_to_normalized(state, a) for a in env.legal_actions(state)]


def validate_dataset(
    path: str | Path, *, max_nodes: int = 300_000
) -> tuple[int, int, list[ValidationDiagnostic]]:
    """Validate every record. Returns ``(records, invalid, error_messages)``.

    Also checks that duplicate ``(signature, state_key)`` records do not conflict
    on their exact labels.
    """
    env = Environment()
    oracle = Oracle(env, max_nodes=max_nodes)
    dataset_path = str(Path(path))
    errors: list[ValidationDiagnostic] = []
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    total = 0
    invalid = 0

    with open(path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(ValidationDiagnostic(
                    dataset_path=dataset_path, line=line_no,
                    record_id="<unparsed>", field="json",
                    code="invalid_json", message=str(exc)))
                invalid += 1
                continue

            if not isinstance(record, dict):
                rec_errors = validate_record(
                    record, oracle, env, line_no, dataset_path=dataset_path)
            else:
                identity, identity_errors = _validate_record_identity(
                    record, line_no, dataset_path=dataset_path)
                rec_errors = identity_errors

                # Duplicate-conflict detection happens only after the complete
                # identity has been type-checked and canonicalized.
                if identity is not None:
                    if identity in seen:
                        prev = seen[identity]
                        if (prev.get("optimal_remaining_moves")
                                != record.get("optimal_remaining_moves")):
                            rec_errors.append(ValidationDiagnostic(
                                dataset_path=dataset_path, line=line_no,
                                record_id=_record_id(record),
                                field="optimal_remaining_moves",
                                code="conflicting_duplicate",
                                message=(
                                    "duplicate conflicts with an earlier line")))
                    else:
                        seen[identity] = record
                    rec_errors.extend(validate_record(
                        record, oracle, env, line_no,
                        dataset_path=dataset_path))
                else:
                    # Full validation can add independent structural
                    # diagnostics, but this row never enters duplicate state.
                    rec_errors = validate_record(
                        record, oracle, env, line_no,
                        dataset_path=dataset_path)

            if rec_errors:
                invalid += 1
                errors.extend(rec_errors)

    return total, invalid, errors


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate an exact policy-value dataset.")
    p.add_argument("dataset", help="path to a JSONL dataset")
    p.add_argument("--max-nodes", type=int, default=300_000)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    total, invalid, errors = validate_dataset(args.dataset, max_nodes=args.max_nodes)
    for diagnostic in errors:
        print(json.dumps(diagnostic.to_dict(), sort_keys=True))
    print(json.dumps({"records": total, "invalid": invalid,
                      "valid": total - invalid}, indent=2))
    return 0 if invalid == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
