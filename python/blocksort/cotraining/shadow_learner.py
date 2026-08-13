"""Safety and persistence helpers for cumulative shadow learners."""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch

from ..training.action_encoding import build_legal_action_mask
from ..training.encoding import encode_state
from ..training.losses import masked_policy_probs


def learner_milestone(round_number: int, interval: int) -> bool:
    """Return whether ``round_number`` is a pre-registered learner milestone."""
    return round_number % interval == 0


def model_integrity_report(
    model,
    *,
    parent_model_state_sha256: str,
    candidate_model_state_sha256: str,
    training_performed: bool,
) -> dict[str, Any]:
    """Check inexpensive invariants before a candidate may continue learning."""
    finite = all(
        bool(torch.isfinite(parameter.detach()).all())
        for parameter in model.parameters()
    )
    changed_when_trained = (
        not training_performed
        or candidate_model_state_sha256 != parent_model_state_sha256
    )
    passed = finite and changed_when_trained
    reasons = []
    if not finite:
        reasons.append("non_finite_model_parameter")
    if not changed_when_trained:
        reasons.append("training_produced_no_model_change")
    return {
        "passed": passed,
        "finite_parameters": finite,
        "changed_when_trained": changed_when_trained,
        "reasons": reasons,
    }


@torch.no_grad()
def policy_drift_report(
    env,
    states: Iterable,
    *,
    reference_model,
    candidate_model,
    encoding_config,
    device: torch.device,
) -> dict[str, Any]:
    """Measure legal-policy drift on a frozen learner-safety state set."""
    reference_model.eval()
    candidate_model.eval()
    count = 0
    reference_entropy = candidate_entropy = 0.0
    kl_reference_to_candidate = total_variation = 0.0
    epsilon = torch.finfo(torch.float32).tiny
    for state in states:
        legal_mask = build_legal_action_mask(env, state, encoding_config)
        if not bool((legal_mask > 0).any()):
            continue
        encoded = encode_state(env, state, encoding_config)
        board = encoded.board.unsqueeze(0).to(device)
        features = encoded.global_features.unsqueeze(0).to(device)
        mask = legal_mask.unsqueeze(0).to(device)
        reference_logits, _ = reference_model(board, features)
        candidate_logits, _ = candidate_model(board, features)
        reference = masked_policy_probs(reference_logits, mask)[0]
        candidate = masked_policy_probs(candidate_logits, mask)[0]
        legal = mask[0] > 0
        reference = reference[legal]
        candidate = candidate[legal]
        reference_entropy += float(
            -(reference * reference.clamp_min(epsilon).log()).sum().item())
        candidate_entropy += float(
            -(candidate * candidate.clamp_min(epsilon).log()).sum().item())
        kl_reference_to_candidate += float((
            reference * (
                reference.clamp_min(epsilon).log()
                - candidate.clamp_min(epsilon).log()
            )
        ).sum().item())
        total_variation += float(
            0.5 * torch.abs(reference - candidate).sum().item())
        count += 1
    if count == 0:
        raise ValueError("learner-safety state set contains no legal decisions")
    reference_entropy /= count
    candidate_entropy /= count
    entropy_ratio = (
        candidate_entropy / reference_entropy
        if reference_entropy > 0 else (1.0 if candidate_entropy == 0 else math.inf)
    )
    return {
        "decision_states": count,
        "reference_mean_entropy": reference_entropy,
        "candidate_mean_entropy": candidate_entropy,
        "candidate_to_reference_entropy_ratio": entropy_ratio,
        "mean_kl_reference_to_candidate": kl_reference_to_candidate / count,
        "mean_total_variation": total_variation / count,
    }


def continuation_decision(
    *,
    training_performed: bool,
    milestone: bool,
    integrity: dict[str, Any],
    drift: dict[str, Any] | None,
    max_policy_kl: float,
    min_entropy_ratio: float,
) -> dict[str, Any]:
    """Apply the distinct, deliberately loose learner-continuation gate."""
    reasons = list(integrity.get("reasons", []))
    if not training_performed:
        return {
            "decision": "retain_previous_learner",
            "accepted": False,
            "milestone": milestone,
            "reasons": ["training_not_performed"],
        }
    if not integrity.get("passed", False):
        return {
            "decision": "rollback_to_anchor",
            "accepted": False,
            "milestone": milestone,
            "reasons": reasons,
        }
    if not milestone:
        return {
            "decision": "continue_pending_milestone",
            "accepted": True,
            "milestone": False,
            "reasons": [],
        }
    if drift is None:
        raise ValueError("milestone continuation requires policy-drift evidence")
    if drift["mean_kl_reference_to_candidate"] > max_policy_kl:
        reasons.append("policy_kl_exceeds_limit")
    if drift["candidate_to_reference_entropy_ratio"] < min_entropy_ratio:
        reasons.append("policy_entropy_below_floor")
    return {
        "decision": "accept_as_anchor" if not reasons else "rollback_to_anchor",
        "accepted": not reasons,
        "milestone": True,
        "reasons": reasons,
        "limits": {
            "max_policy_kl": max_policy_kl,
            "min_entropy_ratio": min_entropy_ratio,
        },
    }


def apply_learner_transition(
    state: dict[str, Any],
    *,
    round_number: int,
    candidate_checkpoint: str,
    candidate_checkpoint_sha256: str,
    continuation: dict[str, Any],
) -> None:
    """Apply one continuation result to the pending transactional run state."""
    if continuation["accepted"]:
        state["active_learner_checkpoint"] = candidate_checkpoint
        state["active_learner_sha256"] = candidate_checkpoint_sha256
        state["active_learner_source_round"] = round_number
        if continuation["milestone"]:
            state["active_learner_anchor_checkpoint"] = candidate_checkpoint
            state["active_learner_anchor_sha256"] = candidate_checkpoint_sha256
            state["active_learner_anchor_source_round"] = round_number
        return
    if continuation["decision"] != "rollback_to_anchor":
        return
    state["active_learner_checkpoint"] = state[
        "active_learner_anchor_checkpoint"]
    state["active_learner_sha256"] = state["active_learner_anchor_sha256"]
    state["active_learner_source_round"] = state[
        "active_learner_anchor_source_round"]
    state.setdefault("learner_rollbacks", []).append({
        "round": round_number,
        "candidate_checkpoint_sha256": candidate_checkpoint_sha256,
        "restored_anchor_sha256": state["active_learner_anchor_sha256"],
        "reasons": list(continuation["reasons"]),
    })
