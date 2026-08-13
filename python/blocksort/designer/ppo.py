"""PPO training for the designer against a frozen protagonist.

Episodic, sparse terminal reward: the designer mutates a base level until it
submits (or exhausts its budget), then the finalized level is scored once by the
oracle + protagonist. Returns equal the terminal reward (gamma = 1); advantages
use the learned value baseline. The update is a standard clipped PPO objective
with a value loss and an entropy bonus. The protagonist is never updated here
(co-training, if any, happens outside this loop).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F

from ..environment import Environment
from ..training.checkpoint import (configs_from_checkpoint, load_checkpoint,
                                    model_from_checkpoint)
from .actions import STOP, DesignerActionSpace
from .checkpoint import save_designer
from .config import DesignerConfig
from .encoding import encode_designer_state
from .env import DesignerEnv
from .model import DesignerModelConfig, DesignerNet
from .replay import LevelReplayBuffer, build_level_record, level_fingerprint
from .roles import Oracle, Protagonist
from .score import score_level


# ----------------------------------------------------------------------
# Masked policy helpers
# ----------------------------------------------------------------------

def masked_log_probs(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    neg = torch.finfo(logits.dtype).min
    masked = torch.where(mask, logits, torch.full_like(logits, neg))
    return torch.log_softmax(masked, dim=-1)


def masked_entropy(log_probs: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    probs = log_probs.exp() * mask
    return -(probs * torch.where(mask, log_probs, torch.zeros_like(log_probs))
             ).sum(dim=-1)


def _sample_seeded_probabilities(
    probabilities: list[float],
    legal_mask: list[bool],
    rng: random.Random,
) -> int:
    """Sample one legal action with the historical seeded-CDF semantics.

    ``probabilities`` is deliberately a CPU/Python list.  Rollout copies the
    complete CUDA probability vector once before calling this helper, avoiding
    one device synchronization per action-space entry.
    """
    if len(probabilities) != len(legal_mask):
        raise ValueError("probabilities and legal mask must have equal length")
    legal_indices = [index for index, legal in enumerate(legal_mask) if legal]
    if not legal_indices:
        raise ValueError("designer rollout has no legal action")
    threshold = rng.random()
    cumulative = 0.0
    fallback = legal_indices[-1]
    for index, probability in enumerate(probabilities):
        if not legal_mask[index] or probability <= 0:
            continue
        cumulative += probability
        if threshold <= cumulative:
            return index
    return fallback


# ----------------------------------------------------------------------
# Rollout
# ----------------------------------------------------------------------

@dataclass
class Step:
    board: torch.Tensor
    globals: torch.Tensor
    mask: torch.Tensor          # bool [A]
    action_index: int
    log_prob: float
    value: float


@dataclass
class Episode:
    steps: list[Step]
    final_state: Any
    finalize: Any
    trajectory: list[int]
    reward: float = 0.0
    scored: Any = None


@torch.no_grad()
def rollout_episode(env: DesignerEnv, model: DesignerNet,
                    action_space: DesignerActionSpace, encoding, *, seed: int,
                    device: torch.device, rng: random.Random,
                    verify_finalize: bool = True, finalize_max_nodes: int = 200_000
                    ) -> Episode:
    state = env.reset(seed)
    steps: list[Step] = []
    trajectory: list[int] = []
    model.eval()

    while True:
        mask_list = env.legal_mask(state)
        mask_cpu = torch.tensor(mask_list, dtype=torch.bool)
        mask = mask_cpu.to(device)
        enc = encode_designer_state(env.env, state, encoding)
        board = enc.board.unsqueeze(0).to(device)
        glob = enc.global_features.unsqueeze(0).to(device)
        logits, value = model(board, glob)
        log_probs = masked_log_probs(logits[0], mask)
        probs = log_probs.exp()
        # One vector transfer replaces up to 2,049 CUDA scalar conversions.
        probabilities = probs.detach().cpu().tolist()
        action_index = _sample_seeded_probabilities(
            probabilities, mask_list, rng)
        selected_log_prob, predicted_value = torch.stack((
            log_probs[action_index], value[0])).detach().cpu().tolist()

        steps.append(Step(
            board=enc.board,
            globals=enc.global_features,
            mask=mask_cpu,
            action_index=action_index,
            log_prob=float(selected_log_prob),
            value=float(predicted_value),
        ))
        trajectory.append(action_index)

        action = action_space.from_index(action_index)
        state = env.step(state, action)
        if state.stopped:
            break

    finalize = env.finalize(state, verify=verify_finalize,
                            max_nodes=finalize_max_nodes)
    return Episode(steps=steps, final_state=state, finalize=finalize,
                   trajectory=trajectory)


# ----------------------------------------------------------------------
# PPO update
# ----------------------------------------------------------------------

@dataclass(frozen=True)
class PPOConfig:
    epochs: int = 4
    minibatch_size: int = 64
    clip_eps: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    learning_rate: float = 3e-4
    max_grad_norm: float = 1.0
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if (isinstance(self.gamma, bool)
                or not isinstance(self.gamma, (int, float))
                or not math.isfinite(self.gamma)
                or not 0.0 <= self.gamma <= 1.0):
            raise ValueError(f"PPO gamma must be finite and in [0, 1]; got {self.gamma!r}")
        if self.gamma != 1.0:
            raise ValueError(
                "PPO gamma is not currently configurable; expected 1.0 "
                "for undiscounted episodic terminal returns")


def ppo_update(model: DesignerNet, optimizer, episodes: list[Episode],
               cfg: PPOConfig, device: torch.device, *, seed: int = 0
               ) -> dict[str, float]:
    boards, globs, masks, actions = [], [], [], []
    old_logp, old_values, returns = [], [], []
    for ep in episodes:
        for st in ep.steps:
            boards.append(st.board)
            globs.append(st.globals)
            masks.append(st.mask)
            actions.append(st.action_index)
            old_logp.append(st.log_prob)
            old_values.append(st.value)
            returns.append(ep.reward)
    if not boards:
        return {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "steps": 0}

    boards = torch.stack(boards).to(device)
    globs = torch.stack(globs).to(device)
    masks = torch.stack(masks).to(device)
    actions = torch.tensor(actions, dtype=torch.long, device=device)
    old_logp = torch.tensor(old_logp, dtype=torch.float32, device=device)
    old_values = torch.tensor(old_values, dtype=torch.float32, device=device)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    advantages = returns - old_values
    if advantages.numel() > 1:
        advantages = ((advantages - advantages.mean())
                      / (advantages.std() + 1e-8))

    n = boards.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "steps": n}
    nb = 0
    model.train()
    for _ in range(cfg.epochs):
        perm = torch.randperm(n, generator=gen).to(device)
        for start in range(0, n, cfg.minibatch_size):
            idx = perm[start:start + cfg.minibatch_size]
            logits, value = model(boards[idx], globs[idx])
            lp = masked_log_probs(logits, masks[idx])
            chosen = lp.gather(1, actions[idx].unsqueeze(1)).squeeze(1)
            adv = advantages[idx]
            ratio = torch.exp(chosen - old_logp[idx])
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value, returns[idx])
            entropy = masked_entropy(lp, masks[idx]).mean()
            loss = (policy_loss + cfg.value_coef * value_loss
                    - cfg.entropy_coef * entropy)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
            stats["policy_loss"] += float(policy_loss.detach())
            stats["value_loss"] += float(value_loss.detach())
            stats["entropy"] += float(entropy.detach())
            nb += 1
    for k in ("policy_loss", "value_loss", "entropy"):
        stats[k] /= max(1, nb)
    return stats
