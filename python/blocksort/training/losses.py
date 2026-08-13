"""Supervised policy-value losses with legal-action masking."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def masked_log_softmax(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Log-softmax over legal actions only (illegal set to a large negative).

    A fully-masked row yields a finite (uniform) distribution rather than NaNs.
    """
    neg = torch.finfo(logits.dtype).min
    masked = torch.where(legal_mask > 0, logits, torch.full_like(logits, neg))
    return F.log_softmax(masked, dim=-1)


def masked_policy_probs(logits: torch.Tensor, legal_mask: torch.Tensor) -> torch.Tensor:
    """Softmax probabilities with zero mass on illegal actions."""
    probs = torch.exp(masked_log_softmax(logits, legal_mask))
    return probs * (legal_mask > 0).to(probs.dtype)


def policy_loss_fn(
    logits: torch.Tensor, policy_target: torch.Tensor, legal_mask: torch.Tensor
) -> torch.Tensor:
    """Distributional cross-entropy ``-sum(target * log_softmax(masked logits))``.

    Samples whose target has no mass (e.g. terminal states with no legal action)
    are excluded so they never contribute NaNs.
    """
    log_probs = masked_log_softmax(logits, legal_mask)
    per_sample = -(policy_target * log_probs).sum(dim=-1)
    valid = policy_target.sum(dim=-1) > 0
    denom = valid.sum().clamp(min=1)
    return (per_sample * valid).sum() / denom


def value_loss_fn(
    value_pred: torch.Tensor, value_target: torch.Tensor, *, loss_type: str = "huber"
) -> torch.Tensor:
    pred = value_pred.reshape(-1)
    target = value_target.reshape(-1)
    if loss_type == "huber":
        return F.smooth_l1_loss(pred, target)
    if loss_type == "mse":
        return F.mse_loss(pred, target)
    raise ValueError(f"unknown value loss type: {loss_type}")


def compute_losses(
    logits: torch.Tensor,
    value_pred: torch.Tensor,
    batch: dict[str, Any],
    *,
    policy_weight: float = 1.0,
    value_weight: float = 1.0,
    value_loss_type: str = "huber",
) -> dict[str, torch.Tensor]:
    p = policy_loss_fn(logits, batch["policy_target"], batch["legal_action_mask"])
    v = value_loss_fn(value_pred, batch["value_target"], loss_type=value_loss_type)
    total = policy_weight * p + value_weight * v
    return {"total": total, "policy": p, "value": v}
