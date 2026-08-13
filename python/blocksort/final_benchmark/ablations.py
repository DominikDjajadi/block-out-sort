"""Compact ablations for the most important design choices.

These are intentionally small (directional evidence, not hyperparameter search):

Designer-reward ablations (short designer training against a fixed protagonist):
  1. no adversarial regret  (w_regret = 0)
  2. no structural terms     (w_structural = 0 and structural metric weights = 0)
  3. no novelty              (w_novelty = 0)

Designer level-replay note (4): in the current implementation the designer
level replay is an *archive* (accepted levels are stored for analysis) and is
not fed back into the on-policy PPO update, so disabling it does not change
designer learning. This is reported rather than re-run.

Co-training ablations (short runs; designer training skipped to isolate the
protagonist/labelling/curriculum effects):
  5. no protagonist historical replay (seed_historical_replay = False)
  6. search-only labels vs exact-first/path-retaining labels (label_mode)
  7. curriculum disabled vs adaptive curriculum (curriculum_enabled)

  8. graph transposition sharing vs tree separation: no separate tree search
     exists; transposition-hit statistics from the solver comparison quantify
     sharing. Reported there, not re-run here.
"""

from __future__ import annotations

import dataclasses
import math
from numbers import Real
from pathlib import Path
from typing import Any, Optional

from ..designer.config import GeneratorConfig, RewardConfig
from ..designer.model import DesignerModelConfig
from ..designer.ppo import PPOConfig
from ..designer.train import TrainConfig, train_designer
from ..cotraining.config import (CoTrainingConfig, CurriculumConfig,
                                 CurriculumState)
from ..cotraining.loop import run_cotraining


_FORGETTING_STATUSES = {"measured", "skipped", "unavailable", "error"}


def _normalize_forgetting_group(name: str, entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise ValueError(
            f"malformed forgetting group {name!r}: expected a mapping")
    status = entry.get("status")
    if status is not None and status not in _FORGETTING_STATUSES:
        raise ValueError(
            f"malformed forgetting group {name!r}: unknown status {status!r}")
    if entry.get("skipped") is True:
        status = "skipped"
    delta = entry.get("delta")
    if status is None:
        if "delta" not in entry:
            raise ValueError(
                f"malformed forgetting group {name!r}: missing delta/status")
        status = "unavailable" if delta is None else "measured"
    if status == "measured":
        if (isinstance(delta, bool) or not isinstance(delta, Real)
                or not math.isfinite(float(delta))):
            raise ValueError(
                f"malformed forgetting group {name!r}: measured delta must "
                f"be finite numeric, got {delta!r}")
        delta = float(delta)
    elif delta is not None:
        raise ValueError(
            f"malformed forgetting group {name!r}: status {status!r} "
            "must not carry a numeric delta")
    return {
        "status": status,
        "delta": delta,
        "baseline": entry.get("baseline"),
        "candidate": entry.get("candidate"),
        "reason": entry.get("reason"),
    }


def normalize_forgetting(forgetting: Any) -> dict[str, Any]:
    if not isinstance(forgetting, dict):
        raise ValueError("malformed forgetting result: expected a mapping")
    if forgetting.get("skipped") is True:
        return {
            "status": "skipped",
            "reason": forgetting.get("reason"),
            "groups": {},
        }
    if not forgetting:
        return {
            "status": "unavailable",
            "reason": "no forgetting groups",
            "groups": {},
        }
    groups = {
        name: _normalize_forgetting_group(name, entry)
        for name, entry in forgetting.items()
    }
    measured = any(
        entry["status"] == "measured" for entry in groups.values())
    return {
        "status": "measured" if measured else "unavailable",
        "reason": None if measured else "no measured forgetting groups",
        "groups": groups,
    }


def _designer_train_variant(name, *, root, protagonist_checkpoint, init_designer,
                            reward, gen_cfg, model_cfg, episodes, sims, oracle_sims,
                            astar_nodes, seed, device) -> dict[str, Any]:
    tc = TrainConfig(
        protagonist_checkpoint=protagonist_checkpoint,
        output_dir=str(Path(root) / name), init_designer=init_designer,
        episodes=episodes, episodes_per_iter=episodes, mutation_budget=8,
        protagonist_simulations=sims, oracle_simulations=oracle_sims,
        astar_max_nodes=astar_nodes, seed=seed, device=device, max_replay=200,
        generator=gen_cfg, reward=reward, model=model_cfg,
        ppo=PPOConfig(epochs=2, entropy_coef=0.02))
    summary = train_designer(tc)
    last = summary["history"][-1] if summary["history"] else {}
    return {
        "mean_reward": summary.get("best_mean_reward"),
        "valid_rate": last.get("valid_rate"),
        "oracle_solve_rate": last.get("oracle_solve_rate"),
        "protagonist_solve_rate": last.get("protagonist_solve_rate"),
        "mean_adversarial_regret": last.get("mean_adversarial_regret"),
        "mean_extra_moves": last.get("mean_extra_moves"),
    }


def reward_term_ablations(*, root, protagonist_checkpoint, init_designer,
                          model_cfg: DesignerModelConfig,
                          gen_cfg: GeneratorConfig, episodes: int, sims: int,
                          oracle_sims: int, astar_nodes: int, seed: int,
                          device: str) -> dict[str, Any]:
    base = RewardConfig()
    variants = {
        "full": base,
        "no_regret": dataclasses.replace(base, w_regret=0.0, cost_gap_scale=0.0),
        "no_structural": dataclasses.replace(
            base, w_structural=0.0, s_extra_moves=0.0, s_first_exit_depth=0.0,
            s_rehandled=0.0, s_few_exitable=0.0),
        "no_novelty": dataclasses.replace(base, w_novelty=0.0),
    }
    out = {}
    for name, reward in variants.items():
        out[name] = _designer_train_variant(
            name, root=Path(root) / "reward_terms",
            protagonist_checkpoint=protagonist_checkpoint,
            init_designer=init_designer, reward=reward, gen_cfg=gen_cfg,
            model_cfg=model_cfg, episodes=episodes, sims=sims,
            oracle_sims=oracle_sims, astar_nodes=astar_nodes, seed=seed,
            device=device)
    return out


def _cotrain_variant(name, *, root, base_cfg: CoTrainingConfig, **overrides
                     ) -> dict[str, Any]:
    cfg = dataclasses.replace(base_cfg, output_dir=str(Path(root) / name),
                              **overrides)
    result = run_cotraining(cfg)
    hist = result["run_state"]["history"]
    return {
        "rounds": len(hist),
        "per_round": [{
            "accepted": h["accepted"],
            "frontier_acceptance_rate": h["frontier_acceptance_rate"],
            "mean_solve_rate": h["mean_solve_rate"],
            "label_exact": h["label_exact"],
            "label_exact_path": h.get("label_exact_path", 0),
            "label_search": h["label_search"],
            "promoted": h["promoted"],
            "promotion_score_candidate": h["promotion_score_candidate"],
            "curriculum_direction": h["curriculum_adjustment"]["direction"],
            "forgetting": normalize_forgetting(h["forgetting"]),
        } for h in hist],
    }


def cotraining_ablations(*, root, base_cfg: CoTrainingConfig) -> dict[str, Any]:
    # Skip designer training in ablation runs to isolate protagonist effects.
    base = dataclasses.replace(base_cfg, train_designer_each_round=False)
    out = {}
    out["baseline"] = _cotrain_variant("baseline", root=root, base_cfg=base)
    out["no_historical_replay"] = _cotrain_variant(
        "no_historical_replay", root=root, base_cfg=base,
        seed_historical_replay=False)
    out["search_only_labels"] = _cotrain_variant(
        "search_only_labels", root=root, base_cfg=base, label_mode="search_only")
    out["curriculum_disabled"] = _cotrain_variant(
        "curriculum_disabled", root=root, base_cfg=base, curriculum_enabled=False)
    return out
