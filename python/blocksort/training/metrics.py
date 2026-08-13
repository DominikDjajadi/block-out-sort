"""Policy and value metrics, plus simple non-neural baselines.

A policy prediction is "correct" if the model's selected legal action is **any**
exact optimal (zero-regret) action, not one arbitrarily-chosen optimal action.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .config import ValueNormConfig
from .losses import masked_policy_probs
from .dataset import _INF_REGRET


class MetricAccumulator:
    """Streams batch predictions and computes aggregate policy/value metrics."""

    def __init__(self, value_norm: ValueNormConfig) -> None:
        self.value_norm = value_norm
        self._reset()

    def _reset(self) -> None:
        self.n = 0
        # policy
        self.top1 = 0.0
        self.top3 = 0.0
        self.opt_mass = 0.0
        self.complete_policy_n = 0
        self.verified_top1 = 0.0
        self.policy_n = 0
        self.entropy = 0.0
        self.legal_before_mask = 0.0
        self.sel_regret_sum = 0.0
        self.sel_regret_n = 0
        self.ce_sum = 0.0
        self.ce_n = 0
        # value (normalized + raw)
        self.norm_abs = 0.0
        self.norm_sq = 0.0
        self.raw_abs = 0.0
        self.raw_sq = 0.0
        # grouping records
        self.group_rows: list[dict[str, Any]] = []

    @torch.no_grad()
    def update(
        self, logits: torch.Tensor, value_pred: torch.Tensor, batch: dict[str, Any]
    ) -> None:
        legal = batch["legal_action_mask"]
        target = batch["policy_target"]
        regret = batch["action_regret"]
        regret_known = batch.get("action_regret_known_mask")
        if regret_known is None:
            regret_known = ((regret < _INF_REGRET / 2) & (legal > 0)).float()
        complete = batch.get("optimal_actions_complete")
        if complete is None:
            complete = torch.ones(logits.shape[0], dtype=torch.bool,
                                  device=logits.device)
        probs = masked_policy_probs(logits, legal)
        B = logits.shape[0]

        # Selected legal action (after masking).
        masked_logits = torch.where(legal > 0, logits,
                                    torch.full_like(logits, torch.finfo(logits.dtype).min))
        sel = masked_logits.argmax(dim=-1)
        raw_sel = logits.argmax(dim=-1)

        # zero-regret legal actions.
        is_opt = (regret == 0) & (legal > 0) & (regret_known > 0)

        # top-3 over legal.
        k = min(3, logits.shape[-1])
        top3_idx = masked_logits.topk(k, dim=-1).indices

        value_pred = value_pred.reshape(-1)
        value_target = batch["value_target"].reshape(-1)
        raw_true = batch["raw_optimal_moves"].reshape(-1).float()

        for b in range(B):
            has_legal = bool(legal[b].sum() > 0)
            sel_b = int(sel[b].item())
            if has_legal:
                self.policy_n += 1
                verified_hit = 1.0 if bool(is_opt[b, sel_b]) else 0.0
                self.verified_top1 += verified_hit
                if bool(complete[b]):
                    self.complete_policy_n += 1
                    self.top1 += verified_hit
                    self.top3 += 1.0 if bool(is_opt[b, top3_idx[b]].any()) else 0.0
                    self.opt_mass += float(
                        (probs[b] * is_opt[b].float()).sum().item())
                self.legal_before_mask += 1.0 if bool(legal[b, int(raw_sel[b])] > 0) else 0.0
                p = probs[b]
                nz = p > 0
                self.entropy += float(-(p[nz] * torch.log(p[nz])).sum().item())
                r = float(regret[b, sel_b].item())
                if bool(regret_known[b, sel_b]) and r < _INF_REGRET / 2:
                    self.sel_regret_sum += r
                    self.sel_regret_n += 1
                tgt = target[b]
                tnz = tgt > 0
                if bool(tnz.any()):
                    ce = float(-(tgt[tnz] * torch.log(probs[b][tnz].clamp_min(1e-12))).sum().item())
                    self.ce_sum += ce
                    self.ce_n += 1

            # value
            np_ = float(value_pred[b].item())
            nt = float(value_target[b].item())
            self.norm_abs += abs(np_ - nt)
            self.norm_sq += (np_ - nt) ** 2
            raw_pred = self.value_norm.denormalize(np_)
            rt = float(raw_true[b].item())
            self.raw_abs += abs(raw_pred - rt)
            self.raw_sq += (raw_pred - rt) ** 2

            self.group_rows.append({
                "rows": int(batch["rows"][b].item()),
                "cols": int(batch["cols"][b].item()),
                "remaining": int(batch["remaining_blocks"][b].item()),
                "legal_count": int(batch["legal_action_count"][b].item()),
                "optimal_moves": int(rt),
                "provenance": batch["provenance_type"][b],
                "policy_strategy": (
                    "immediately_exitable"
                    if bool(batch.get(
                        "optimal_exit_available",
                        torch.tensor(False, device=logits.device))[b])
                    else "setup_required"
                ) if "optimal_exit_available" in batch else "unknown",
                "raw_abs": abs(raw_pred - rt),
                "top1": (
                    1.0 if bool(is_opt[b, sel_b]) else 0.0
                ) if (has_legal and bool(complete[b])) else None,
                "verified_top1": (
                    1.0 if (has_legal and bool(is_opt[b, sel_b])) else 0.0),
            })
            self.n += 1

    def compute(self) -> dict[str, float]:
        n = max(self.n, 1)
        pn = max(self.ce_n, 1)  # samples with legal actions (all, here)
        legal_n = max(self.policy_n, 1)
        complete_n = self.complete_policy_n
        return {
            "count": self.n,
            "policy_cross_entropy": self.ce_sum / pn,
            "policy_top1_optimal_acc": (
                self.top1 / complete_n if complete_n else float("nan")),
            "policy_top3_optimal_acc": (
                self.top3 / complete_n if complete_n else float("nan")),
            "policy_optimal_mass": (
                self.opt_mass / complete_n if complete_n else float("nan")),
            "policy_complete_label_count": complete_n,
            "policy_verified_action_top1_acc": self.verified_top1 / legal_n,
            "policy_entropy": self.entropy / legal_n,
            "policy_legal_before_mask": self.legal_before_mask / legal_n,
            "policy_selected_regret_mean": (self.sel_regret_sum / self.sel_regret_n
                                            if self.sel_regret_n else float("nan")),
            "value_norm_mae": self.norm_abs / n,
            "value_norm_rmse": math.sqrt(self.norm_sq / n),
            "value_raw_mae": self.raw_abs / n,
            "value_raw_rmse": math.sqrt(self.raw_sq / n),
        }

    def grouped(self, key: str) -> dict[Any, dict[str, float]]:
        """Mean raw-value MAE and top-1 accuracy grouped by a field."""
        groups: dict[Any, list[dict[str, Any]]] = {}
        for g in self.group_rows:
            gk = (g["rows"], g["cols"]) if key == "board" else g[key]
            groups.setdefault(gk, []).append(g)
        out: dict[Any, dict[str, float]] = {}
        for gk, rows in sorted(groups.items(), key=lambda kv: str(kv[0])):
            complete_rows = [r for r in rows if r["top1"] is not None]
            out[gk] = {
                "count": len(rows),
                "value_raw_mae": sum(r["raw_abs"] for r in rows) / len(rows),
                "policy_top1_optimal_acc": (
                    sum(r["top1"] for r in complete_rows) / len(complete_rows)
                    if complete_rows else float("nan")),
                "policy_complete_label_count": len(complete_rows),
                "policy_verified_action_top1_acc": (
                    sum(r["verified_top1"] for r in rows) / len(rows)),
            }
        return out


# ---------------------------------------------------------------------------
# Baselines (no neural network)
# ---------------------------------------------------------------------------

def policy_baselines(items: list[dict[str, Any]], config) -> dict[str, float]:
    """Expected zero-regret hit rate for trivial policies.

    - uniform random legal action: ``P(optimal) = #optimal / #legal``.
    - exit-preferring: choose uniformly among legal EXIT actions if any exist,
      else uniformly among legal actions (EXIT detected via the move-code slot).
    """
    m = config.move_code_count
    exit_code = config.exit_move_code
    uniform = 0.0
    exit_pref = 0.0
    n = 0
    for it in items:
        if not bool(it.get("optimal_actions_complete", True)):
            continue
        legal = it["legal_action_mask"]
        regret = it["action_regret"]
        legal_idx = (legal > 0).nonzero(as_tuple=True)[0]
        if legal_idx.numel() == 0:
            continue
        n += 1
        opt = (regret == 0) & (legal > 0)
        uniform += float(opt.sum().item()) / float(legal_idx.numel())

        exit_idx = [int(i) for i in legal_idx.tolist() if i % m == exit_code]
        if exit_idx:
            opt_exit = sum(1 for i in exit_idx if bool(opt[i]))
            exit_pref += opt_exit / len(exit_idx)
        else:
            exit_pref += float(opt.sum().item()) / float(legal_idx.numel())
    n = max(n, 1)
    return {
        "uniform_random_top1_optimal_acc": uniform / n,
        "exit_preferring_top1_optimal_acc": exit_pref / n,
    }


def policy_baselines_by_strategy(
    items: list[dict[str, Any]], config,
) -> dict[str, dict[str, float]]:
    """Policy baselines split by whether an optimal exit is available now."""
    groups = {
        "immediately_exitable": [
            item for item in items
            if bool(item.get("optimal_exit_available", False))],
        "setup_required": [
            item for item in items
            if not bool(item.get("optimal_exit_available", False))],
    }
    return {
        name: {
            "count": len(rows),
            **policy_baselines(rows, config),
        }
        for name, rows in groups.items()
    }


def value_baselines(
    items: list[dict[str, Any]], value_norm: ValueNormConfig
) -> dict[str, float]:
    """Raw-move error for: predict remaining-block count, predict train mean."""
    raws = [it["raw_optimal_moves"] for it in items]
    mean_moves = sum(raws) / max(len(raws), 1)
    rb_abs = rb_sq = mean_abs = mean_sq = 0.0
    for it in items:
        rt = it["raw_optimal_moves"]
        rb = it["remaining_blocks"]
        rb_abs += abs(rb - rt); rb_sq += (rb - rt) ** 2
        mean_abs += abs(mean_moves - rt); mean_sq += (mean_moves - rt) ** 2
    n = max(len(items), 1)
    return {
        "predict_remaining_blocks_raw_mae": rb_abs / n,
        "predict_remaining_blocks_raw_rmse": math.sqrt(rb_sq / n),
        "predict_mean_raw_mae": mean_abs / n,
        "predict_mean_raw_rmse": math.sqrt(mean_sq / n),
        "train_mean_optimal_moves": mean_moves,
    }
