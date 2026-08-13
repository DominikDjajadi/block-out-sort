"""Deterministic policy-target profiles shared by live and diagnostic updates.

Incumbent-guided profiles never change which actions the exact oracle marks as
optimal. They only redistribute probability within that complete optimal set,
using the incumbent's legal-action probabilities as a stable tie-breaker.
Search-derived and incomplete exact-path targets pass through unchanged.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from typing import Any

import torch
from torch.utils.data import DataLoader

from ..dataset.schema import LABEL_FULL_EXACT
from ..expert_iteration.train import ExpertDataset, collate
from ..training.action_encoding import normalized_action_index
from ..training.losses import masked_policy_probs


POLICY_TARGET_PROFILES = frozenset((
    "recorded",
    "incumbent_optimal",
    "incumbent_optimal_blend50",
    "incumbent_optimal_sharp",
))
POLICY_TARGET_EXPONENTS = {
    "incumbent_optimal": 1.0,
    "incumbent_optimal_blend50": 1.0,
    "incumbent_optimal_sharp": 2.0,
}
POLICY_TARGET_UNIFORM_MIX = {
    "incumbent_optimal": 0.0,
    "incumbent_optimal_blend50": 0.5,
    "incumbent_optimal_sharp": 0.0,
}
DEFAULT_POLICY_TARGET_PROFILE = "incumbent_optimal"


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def policy_target_sha256(records: list[dict[str, Any]]) -> str:
    return _canonical_sha256([
        record["policy_target"] for record in records
    ])


def recorded_policy_target_summary(
    records: list[dict[str, Any]],
    *,
    sample_sha256: str | None = None,
    sample_path: str | None = None,
) -> dict[str, Any]:
    summary = {
        "profile": "recorded",
        "records": len(records),
        "policy_target_sha256": policy_target_sha256(records),
    }
    if sample_path is not None:
        summary["sample"] = sample_path
    if sample_sha256 is not None:
        summary["sample_sha256"] = sample_sha256
    return summary


def _policy_entropy(target: list[float]) -> float:
    return -sum(
        probability * math.log(probability)
        for probability in target
        if probability > 0
    )


def _is_complete_exact_policy_record(record: dict[str, Any]) -> bool:
    return (
        record.get("label_kind") == LABEL_FULL_EXACT
        and record.get("policy_exact") is True
        and record.get("optimal_actions_complete") is True
    )


def incumbent_legal_probabilities(
    records: list[dict[str, Any]],
    *,
    model,
    encoding_config,
    value_norm,
    device: torch.device,
    batch_size: int,
) -> list[list[float]]:
    """Return incumbent probabilities aligned with each record's legal actions."""
    dataset = ExpertDataset(
        records,
        [1.0] * len(records),
        encoding_config=encoding_config,
        value_norm=value_norm,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
    )
    dense_probabilities = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            board = batch["board"].to(device)
            glob = batch["global_features"].to(device)
            legal_mask = batch["legal_action_mask"].to(device)
            logits, _value = model(board, glob)
            dense_probabilities.extend(
                masked_policy_probs(logits, legal_mask).detach().cpu())
    if len(dense_probabilities) != len(records):
        raise RuntimeError(
            "incumbent policy evaluation did not return one row per record")
    aligned = []
    for record, dense in zip(records, dense_probabilities):
        row = [
            float(dense[normalized_action_index(action, encoding_config)])
            for action in record["legal_actions"]
        ]
        if not math.isclose(sum(row), 1.0, abs_tol=1e-5):
            raise RuntimeError(
                "incumbent legal-action probabilities do not sum to one for "
                f"state {record.get('state_key')}")
        aligned.append(row)
    return aligned


def condition_policy_targets(
    records: list[dict[str, Any]],
    incumbent_probabilities: list[list[float]],
    *,
    profile: str,
    incumbent_checkpoint_sha256: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Condition complete exact targets on incumbent optimal-action priors."""
    if profile not in POLICY_TARGET_EXPONENTS:
        raise ValueError(f"unsupported conditioned policy profile: {profile}")
    if len(records) != len(incumbent_probabilities):
        raise ValueError(
            "one incumbent probability row is required per training record")
    exponent = POLICY_TARGET_EXPONENTS[profile]
    uniform_mix = POLICY_TARGET_UNIFORM_MIX[profile]
    transformed = copy.deepcopy(records)
    eligible = changed = 0
    entropy_before = entropy_after = 0.0
    max_suboptimal_mass = 0.0
    for source, result, incumbent_row in zip(
            records, transformed, incumbent_probabilities):
        target = source.get("policy_target")
        legal_actions = source.get("legal_actions")
        if not isinstance(target, list) or not isinstance(legal_actions, list):
            raise ValueError("policy records require list targets and actions")
        if len(target) != len(legal_actions):
            raise ValueError(
                "policy target length must match legal actions for state "
                f"{source.get('state_key')}")
        if len(incumbent_row) != len(target):
            raise ValueError(
                "incumbent probability length must match policy target for "
                f"state {source.get('state_key')}")
        if not _is_complete_exact_policy_record(source):
            continue
        optimal_indices = [
            index for index, probability in enumerate(target)
            if float(probability) > 0
        ]
        if not optimal_indices:
            raise ValueError(
                "complete exact record has empty optimal support for state "
                f"{source.get('state_key')}")
        guided = [0.0] * len(target)
        denominator = sum(
            float(incumbent_row[index]) ** exponent
            for index in optimal_indices
        )
        if not math.isfinite(denominator) or denominator <= 0:
            raise ValueError(
                "incumbent assigns no finite probability to exact optimal "
                f"support for state {source.get('state_key')}")
        for index in optimal_indices:
            guided[index] = (
                float(incumbent_row[index]) ** exponent / denominator)
        if uniform_mix > 0:
            uniform_probability = 1.0 / len(optimal_indices)
            for index in optimal_indices:
                guided[index] = (
                    (1.0 - uniform_mix) * guided[index]
                    + uniform_mix * uniform_probability)
        if not math.isclose(sum(guided), 1.0, abs_tol=1e-12):
            raise RuntimeError("conditioned policy target does not sum to one")
        optimal_set = set(optimal_indices)
        suboptimal_mass = sum(
            probability for index, probability in enumerate(guided)
            if index not in optimal_set
        )
        max_suboptimal_mass = max(max_suboptimal_mass, suboptimal_mass)
        if suboptimal_mass != 0:
            raise RuntimeError(
                "conditioned policy target assigned mass to a suboptimal action")
        eligible += 1
        entropy_before += _policy_entropy([float(value) for value in target])
        entropy_after += _policy_entropy(guided)
        if any(
                not math.isclose(float(left), right, abs_tol=1e-12)
                for left, right in zip(target, guided)):
            changed += 1
        result["policy_target"] = guided
        result["policy"] = {
            "type": "incumbent-conditioned-optimal",
            "temperature": 1.0 / exponent,
            "conditioning_exponent": exponent,
            "uniform_optimal_mix": uniform_mix,
            "incumbent_checkpoint_sha256": incumbent_checkpoint_sha256,
            "source_policy": copy.deepcopy(source.get("policy")),
        }
        result["policy_target_profile"] = profile
    summary = {
        "profile": profile,
        "conditioning_exponent": exponent,
        "temperature": 1.0 / exponent,
        "uniform_optimal_mix": uniform_mix,
        "records": len(records),
        "eligible_complete_exact_records": eligible,
        "changed_records": changed,
        "unchanged_records": len(records) - changed,
        "mean_eligible_entropy_before": (
            entropy_before / eligible if eligible else 0.0),
        "mean_eligible_entropy_after": (
            entropy_after / eligible if eligible else 0.0),
        "max_suboptimal_probability_mass": max_suboptimal_mass,
        "incumbent_checkpoint_sha256": incumbent_checkpoint_sha256,
        "policy_target_sha256": policy_target_sha256(transformed),
    }
    return transformed, summary
