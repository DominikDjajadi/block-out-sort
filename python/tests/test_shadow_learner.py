from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from blocksort import Environment, level_from_dict
from blocksort.cotraining.config import CoTrainingConfig
from blocksort.cotraining.run import build_parser, config_from_args
from blocksort.cotraining.shadow_learner import (
    apply_learner_transition,
    continuation_decision,
    learner_milestone,
    model_integrity_report,
    policy_drift_report,
)
from blocksort.dataset.schema import deserialize_state
from blocksort.training.config import EncodingConfig, ModelConfig
from blocksort.training.dataset import load_records
from blocksort.training.model import PolicyValueNet


def test_shadow_learner_config_defaults_are_deliberately_loose():
    cfg = CoTrainingConfig(shadow_learner_enabled=True)
    assert cfg.learner_milestone_interval == 5
    assert cfg.learner_max_policy_kl == pytest.approx(0.25)
    assert cfg.learner_min_entropy_ratio == pytest.approx(0.70)


def test_shadow_learner_cli_is_explicit_and_configurable():
    args = build_parser().parse_args([
        "--protagonist-checkpoint", "protagonist.pt",
        "--designer-checkpoint", "designer.pt",
        "--base-dataset", "base.jsonl",
        "--shadow-learner",
        "--stop-after-promotion",
        "--learner-milestone-interval", "3",
        "--learner-max-policy-kl", "0.4",
        "--learner-min-entropy-ratio", "0.6",
    ])
    cfg = config_from_args(args)
    assert cfg.shadow_learner_enabled is True
    assert cfg.stop_after_promotion is True
    assert cfg.learner_milestone_interval == 3
    assert cfg.learner_max_policy_kl == pytest.approx(0.4)
    assert cfg.learner_min_entropy_ratio == pytest.approx(0.6)


@pytest.mark.parametrize("round_number, expected", [
    (1, False), (4, False), (5, True), (10, True), (11, False),
])
def test_learner_milestone_is_preregistered(round_number, expected):
    assert learner_milestone(round_number, 5) is expected


def test_nonmilestone_candidate_continues_after_mechanical_checks():
    decision = continuation_decision(
        training_performed=True,
        milestone=False,
        integrity={"passed": True, "reasons": []},
        drift=None,
        max_policy_kl=0.25,
        min_entropy_ratio=0.70,
    )
    assert decision == {
        "decision": "continue_pending_milestone",
        "accepted": True,
        "milestone": False,
        "reasons": [],
    }


def test_safe_milestone_becomes_new_anchor():
    decision = continuation_decision(
        training_performed=True,
        milestone=True,
        integrity={"passed": True, "reasons": []},
        drift={
            "mean_kl_reference_to_candidate": 0.14,
            "candidate_to_reference_entropy_ratio": 0.91,
        },
        max_policy_kl=0.25,
        min_entropy_ratio=0.70,
    )
    assert decision["decision"] == "accept_as_anchor"
    assert decision["accepted"] is True


@pytest.mark.parametrize(
    "drift, reason",
    [
        ({"mean_kl_reference_to_candidate": 0.26,
          "candidate_to_reference_entropy_ratio": 0.90},
         "policy_kl_exceeds_limit"),
        ({"mean_kl_reference_to_candidate": 0.10,
          "candidate_to_reference_entropy_ratio": 0.69},
         "policy_entropy_below_floor"),
    ],
)
def test_unsafe_milestone_rolls_back(drift, reason):
    decision = continuation_decision(
        training_performed=True,
        milestone=True,
        integrity={"passed": True, "reasons": []},
        drift=drift,
        max_policy_kl=0.25,
        min_entropy_ratio=0.70,
    )
    assert decision["decision"] == "rollback_to_anchor"
    assert decision["accepted"] is False
    assert reason in decision["reasons"]


def test_integrity_rejects_nonfinite_or_unchanged_trained_model():
    model = torch.nn.Linear(2, 1)
    unchanged = model_integrity_report(
        model,
        parent_model_state_sha256="same",
        candidate_model_state_sha256="same",
        training_performed=True,
    )
    assert unchanged["passed"] is False
    assert "training_produced_no_model_change" in unchanged["reasons"]

    with torch.no_grad():
        model.weight[0, 0] = math.inf
    nonfinite = model_integrity_report(
        model,
        parent_model_state_sha256="before",
        candidate_model_state_sha256="after",
        training_performed=True,
    )
    assert nonfinite["passed"] is False
    assert "non_finite_model_parameter" in nonfinite["reasons"]


def test_learner_state_accumulates_anchors_and_rolls_back():
    state = {
        "active_learner_checkpoint": "champion.pt",
        "active_learner_sha256": "champion",
        "active_learner_source_round": 0,
        "active_learner_anchor_checkpoint": "champion.pt",
        "active_learner_anchor_sha256": "champion",
        "active_learner_anchor_source_round": 0,
        "learner_rollbacks": [],
    }
    apply_learner_transition(
        state, round_number=1, candidate_checkpoint="round_001.pt",
        candidate_checkpoint_sha256="one",
        continuation={
            "accepted": True, "milestone": False,
            "decision": "continue_pending_milestone", "reasons": [],
        })
    assert state["active_learner_sha256"] == "one"
    assert state["active_learner_anchor_sha256"] == "champion"

    apply_learner_transition(
        state, round_number=5, candidate_checkpoint="round_005.pt",
        candidate_checkpoint_sha256="five",
        continuation={
            "accepted": True, "milestone": True,
            "decision": "accept_as_anchor", "reasons": [],
        })
    assert state["active_learner_sha256"] == "five"
    assert state["active_learner_anchor_sha256"] == "five"

    apply_learner_transition(
        state, round_number=10, candidate_checkpoint="round_010.pt",
        candidate_checkpoint_sha256="unsafe",
        continuation={
            "accepted": False, "milestone": True,
            "decision": "rollback_to_anchor",
            "reasons": ["policy_kl_exceeds_limit"],
        })
    assert state["active_learner_sha256"] == "five"
    assert state["active_learner_source_round"] == 5
    assert state["learner_rollbacks"] == [{
        "round": 10,
        "candidate_checkpoint_sha256": "unsafe",
        "restored_anchor_sha256": "five",
        "reasons": ["policy_kl_exceeds_limit"],
    }]


def test_policy_drift_is_zero_for_identical_models():
    records = load_records(
        Path(__file__).resolve().parents[2] / "data/training/pv_smoke.jsonl")
    record = records[0]
    level = level_from_dict(record["level"])
    state = deserialize_state(level, record["state"])
    encoding = EncodingConfig()
    model = PolicyValueNet(
        encoding, ModelConfig(channels=4, residual_blocks=1,
                              value_hidden_size=8))
    report = policy_drift_report(
        Environment(), [state], reference_model=model, candidate_model=model,
        encoding_config=encoding, device=torch.device("cpu"))
    assert report["decision_states"] == 1
    assert report["mean_kl_reference_to_candidate"] == pytest.approx(0.0)
    assert report["mean_total_variation"] == pytest.approx(0.0)
    assert report["candidate_to_reference_entropy_ratio"] == pytest.approx(1.0)


@pytest.mark.parametrize("field, value", [
    ("learner_milestone_interval", 0),
    ("learner_max_policy_kl", 0.0),
    ("learner_min_entropy_ratio", -0.1),
    ("learner_min_entropy_ratio", 1.1),
])
def test_invalid_shadow_learner_config_is_rejected(field, value):
    with pytest.raises(ValueError):
        CoTrainingConfig(**{field: value})
