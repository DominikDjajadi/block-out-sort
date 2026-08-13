"""Weighted training over mixed exact + search-labeled examples.

Reuses the supervised state/action encoders. Each example carries separate
policy and value weights plus a ``value_exact`` flag. Policy loss is
distributional cross-entropy against the target distribution (uniform-optimal
for exact, normalized visit policy for search). Exact records supervise scalar
value normally; uncertain search estimates have zero scalar value weight by
default while retaining their visit-policy supervision.
"""

from __future__ import annotations

import math
import random
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ..environment import Environment
from ..serialization import level_from_dict
from ..dataset.schema import (
    LABEL_EXACT_PATH_POLICY,
    LABEL_FULL_EXACT,
    deserialize_state,
)
from ..training.action_encoding import (
    encode_sparse_policy_target,
    legal_mask_from_record,
    normalized_action_index,
)
from ..training.config import EncodingConfig, ValueNormConfig
from ..training.encoding import encode_state
from ..training.losses import masked_log_softmax
from ..training.validation import validate_positive_integer
from ..training.model import PolicyValueNet
from .records import ensure_record_label_metadata


TRAINABLE_PARTS = frozenset(
    ("all", "policy_adapter", "policy_head", "policy_trunk", "value_head"))
POLICY_SEARCH_UTILITY_FIELD = "policy_search_utility"
TRACE_PREFERENCE_FIELD = "trace_preference"


def configure_trainable_part(
    model: PolicyValueNet,
    trainable_part: str,
) -> dict[str, int | str]:
    """Restrict an expert-iteration update to the requested network head."""
    if trainable_part not in TRAINABLE_PARTS:
        choices = ", ".join(sorted(TRAINABLE_PARTS))
        raise ValueError(f"trainable part must be one of: {choices}")

    for parameter in model.parameters():
        parameter.requires_grad_(trainable_part == "all")
    if trainable_part == "policy_head":
        modules = (model.policy_conv,)
    elif trainable_part == "policy_adapter":
        if len(model.policy_adapter) == 0:
            raise ValueError(
                "policy_adapter requires at least one configured adapter block")
        modules = (model.policy_adapter, model.policy_conv)
    elif trainable_part == "policy_trunk":
        modules = (model.stem, model.trunk, model.policy_conv)
    elif trainable_part == "value_head":
        modules = (model.value_conv, model.value_mlp)
    else:
        modules = ()
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters()
        if parameter.requires_grad
    )
    if trainable == 0:
        raise RuntimeError(
            f"{trainable_part} selected no trainable model parameters")
    return {
        "trainable_part": trainable_part,
        "trainable_parameters": trainable,
        "total_parameters": total,
    }


class ExpertDataset(Dataset):
    """Encodes mixed exact/search records with per-sample weights."""

    def __init__(self, records: list[dict[str, Any]], weights: list[float], *,
                 encoding_config: EncodingConfig, value_norm: ValueNormConfig,
                 value_weights: list[float] | None = None,
                 env: Optional[Environment] = None) -> None:
        if len(records) != len(weights):
            raise ValueError("one policy weight is required per training record")
        if value_weights is None:
            value_weights = weights
        if len(records) != len(value_weights):
            raise ValueError("one value weight is required per training record")
        for label, values in (
                ("policy", weights), ("value", value_weights)):
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError(
                    f"{label} weights must be finite and non-negative")
        if weights and sum(weights) <= 0:
            raise ValueError("policy weights must have positive total mass")
        self.encoding_config = encoding_config
        self.value_norm = value_norm
        self.env = env or Environment()
        self.items: list[dict[str, Any]] = []
        for rec, policy_weight, value_weight in zip(
                records, weights, value_weights):
            self.items.append(self._encode(
                rec, policy_weight, value_weight))

    def _encode(
        self,
        record: dict[str, Any],
        policy_weight: float,
        value_weight: float,
    ) -> dict[str, Any]:
        record = ensure_record_label_metadata(record)
        level = level_from_dict(record["level"])
        state = deserialize_state(level, record["state"])
        enc = encode_state(self.env, state, self.encoding_config)
        legal_mask = legal_mask_from_record(record, self.encoding_config)
        policy_target = encode_sparse_policy_target(record, self.encoding_config)
        policy_rank_utility = torch.zeros(
            self.encoding_config.action_space_size, dtype=torch.float32)
        policy_rank_mask = torch.zeros(
            self.encoding_config.action_space_size, dtype=torch.bool)
        trace_preferred_index = torch.tensor(0, dtype=torch.int64)
        trace_competing_index = torch.tensor(0, dtype=torch.int64)
        trace_pair_valid = torch.tensor(False, dtype=torch.bool)
        sparse_utility = record.get(POLICY_SEARCH_UTILITY_FIELD)
        if sparse_utility is not None:
            if record.get("label_kind") != LABEL_FULL_EXACT:
                raise ValueError(
                    "policy search utility requires a full-exact record")
            if not bool(record.get("optimal_actions_complete", False)):
                raise ValueError(
                    "policy search utility requires complete optimal actions")
            if (not isinstance(sparse_utility, list)
                    or len(sparse_utility) != len(record["legal_actions"])):
                raise ValueError(
                    "policy search utility must align with legal actions")
            for action, target_probability, utility in zip(
                    record["legal_actions"], record["policy_target"],
                    sparse_utility):
                if utility is None:
                    continue
                value = float(utility)
                if not math.isfinite(value):
                    raise ValueError(
                        "policy search utility values must be finite")
                if float(target_probability) <= 0:
                    raise ValueError(
                        "policy search utility may label only oracle-optimal "
                        "actions")
                index = normalized_action_index(action, self.encoding_config)
                policy_rank_utility[index] = value
                policy_rank_mask[index] = True

        trace_preference = record.get(TRACE_PREFERENCE_FIELD)
        if trace_preference is not None:
            if record.get("label_kind") != LABEL_FULL_EXACT:
                raise ValueError(
                    "trace preference requires a full-exact record")
            if not bool(record.get("optimal_actions_complete", False)):
                raise ValueError(
                    "trace preference requires complete optimal actions")
            if not isinstance(trace_preference, dict):
                raise ValueError("trace preference must be an object")
            if trace_preference.get(
                    "both_actions_full_exact_oracle_optimal") is not True:
                raise ValueError(
                    "trace preference requires two proven-optimal actions")
            preferred_action = trace_preference.get("preferred_action")
            competing_action = trace_preference.get("competing_action")
            if not isinstance(preferred_action, dict) \
                    or not isinstance(competing_action, dict):
                raise ValueError(
                    "trace preference actions must be serialized objects")
            preferred_legal = int(trace_preference.get(
                "preferred_action_index", -1))
            competing_legal = int(trace_preference.get(
                "competing_action_index", -1))
            legal_actions = record["legal_actions"]
            if not (0 <= preferred_legal < len(legal_actions)) \
                    or legal_actions[preferred_legal] != preferred_action:
                raise ValueError(
                    "trace preferred action index does not match legal actions")
            if not (0 <= competing_legal < len(legal_actions)) \
                    or legal_actions[competing_legal] != competing_action:
                raise ValueError(
                    "trace competing action index does not match legal actions")
            if float(record["policy_target"][preferred_legal]) <= 0 \
                    or float(record["policy_target"][competing_legal]) <= 0:
                raise ValueError(
                    "trace preference may compare only oracle-optimal actions")
            preferred_global = normalized_action_index(
                preferred_action, self.encoding_config)
            competing_global = normalized_action_index(
                competing_action, self.encoding_config)
            if preferred_global == competing_global:
                raise ValueError(
                    "trace preference actions must encode to distinct indices")
            trace_preferred_index = torch.tensor(
                preferred_global, dtype=torch.int64)
            trace_competing_index = torch.tensor(
                competing_global, dtype=torch.int64)
            trace_pair_valid = torch.tensor(True, dtype=torch.bool)

        psum = float(policy_target.sum())
        if psum <= 0:
            raise ValueError(f"empty policy target for state {record.get('state_key')}")
        if abs(psum - 1.0) > 1e-4:        # renormalize defensively (search visits)
            policy_target = policy_target / psum

        raw = float(record["value_target"]["raw_optimal_moves"])
        value = self.value_norm.normalize(raw)
        return {
            "board": enc.board,
            "global_features": enc.global_features,
            "legal_action_mask": legal_mask,
            "policy_target": policy_target,
            "policy_rank_utility": policy_rank_utility,
            "policy_rank_mask": policy_rank_mask,
            "trace_preferred_index": trace_preferred_index,
            "trace_competing_index": trace_competing_index,
            "trace_pair_valid": trace_pair_valid,
            "value_target": torch.tensor(value, dtype=torch.float32),
            "weight": torch.tensor(float(policy_weight), dtype=torch.float32),
            "value_weight":
                torch.tensor(float(value_weight), dtype=torch.float32),
            "value_exact": bool(record.get("value_exact", True)),
            "generation_iteration": torch.tensor(
                int(record.get("generation_iteration", 0)),
                dtype=torch.int64),
        }

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict[str, Any]:
        return self.items[i]


_TENSOR_KEYS = (
    "board", "global_features", "legal_action_mask", "policy_target",
    "policy_rank_utility", "policy_rank_mask", "value_target", "weight",
    "value_weight", "generation_iteration", "trace_preferred_index",
    "trace_competing_index", "trace_pair_valid",
)


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out = {k: torch.stack([b[k] for b in batch]) for k in _TENSOR_KEYS}
    out["value_exact"] = torch.tensor([b["value_exact"] for b in batch],
                                      dtype=torch.bool)
    return out


def pairwise_policy_ranking_loss(
    logits: torch.Tensor,
    utilities: torch.Tensor,
    utility_mask: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rank higher-utility labeled actions above lower-utility actions.

    Utilities are sparse and intentionally limited to fully proven optimal
    actions. Equal-utility pairs abstain. Each example is normalized by its
    total utility gap so states with many optimal actions do not dominate.
    """
    if logits.shape != utilities.shape or logits.shape != utility_mask.shape:
        raise ValueError(
            "ranking logits, utilities, and masks must have identical shapes")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("policy ranking margin must be finite and non-negative")
    per_example: list[torch.Tensor] = []
    valid_rows: list[bool] = []
    pair_counts: list[int] = []
    for row_logits, row_utilities, row_mask in zip(
            logits, utilities, utility_mask):
        indices = torch.nonzero(row_mask, as_tuple=False).flatten()
        if indices.numel() < 2:
            per_example.append(row_logits.sum() * 0.0)
            valid_rows.append(False)
            pair_counts.append(0)
            continue
        selected_utilities = row_utilities[indices]
        utility_gap = (
            selected_utilities[:, None] - selected_utilities[None, :])
        preferred = utility_gap > 1e-8
        pair_count = int(preferred.sum().item())
        if pair_count == 0:
            per_example.append(row_logits.sum() * 0.0)
            valid_rows.append(False)
            pair_counts.append(0)
            continue
        selected_logits = row_logits[indices]
        logit_gap = selected_logits[:, None] - selected_logits[None, :]
        pair_weights = utility_gap[preferred]
        pair_losses = F.softplus(margin - logit_gap[preferred])
        per_example.append(
            (pair_losses * pair_weights).sum() / pair_weights.sum())
        valid_rows.append(True)
        pair_counts.append(pair_count)
    return (
        torch.stack(per_example),
        torch.tensor(valid_rows, dtype=torch.bool, device=logits.device),
        torch.tensor(pair_counts, dtype=torch.int64, device=logits.device),
    )


def trace_pairwise_hinge_loss(
    logits: torch.Tensor,
    preferred_indices: torch.Tensor,
    competing_indices: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    """Weakly rank one search-success action over one competing action.

    The caller is responsible for supplying only locally evidenced pairs where
    both actions are fully proven oracle-optimal.  The mean is intentionally
    left to the caller so auxiliary batching can report exact example mass.
    """
    if logits.ndim != 2:
        raise ValueError("trace ranking logits must be a matrix")
    if preferred_indices.shape != (len(logits),) \
            or competing_indices.shape != (len(logits),):
        raise ValueError(
            "one preferred and competing index is required per logit row")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError(
            "trace ranking margin must be finite and non-negative")
    for label, indices in (
            ("preferred", preferred_indices),
            ("competing", competing_indices)):
        if indices.dtype != torch.int64:
            raise ValueError(f"trace {label} indices must be int64")
        if torch.any(indices < 0) or torch.any(indices >= logits.shape[1]):
            raise ValueError(f"trace {label} index is outside logit space")
    preferred = logits.gather(1, preferred_indices[:, None]).squeeze(1)
    competing = logits.gather(1, competing_indices[:, None]).squeeze(1)
    return F.relu(float(margin) - (preferred - competing))


def _spread_auxiliary_batches(
    item_count: int,
    batch_count: int,
    *,
    seed: int,
) -> list[list[int]]:
    """Shuffle one auxiliary pass and spread it across optimizer steps."""
    if item_count < 0 or batch_count <= 0:
        raise ValueError("invalid auxiliary item or batch count")
    batches: list[list[int]] = [[] for _ in range(batch_count)]
    if item_count == 0:
        return batches
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(
        item_count, generator=generator).tolist()
    for position, index in enumerate(permutation):
        batch_index = min(
            batch_count - 1, position * batch_count // item_count)
        batches[batch_index].append(index)
    if sorted(index for batch in batches for index in batch) \
            != list(range(item_count)):
        raise RuntimeError("auxiliary batching lost or duplicated records")
    return batches


def source_weights_for(records: list[dict[str, Any]], current_iteration: int, *,
                       weight_exact_historical: float, weight_exact_new: float,
                       weight_search: float,
                       exact_path_policy_confidence: float = 0.5,
                       ) -> list[float]:
    """Map provenance and label completeness to policy-gradient weight."""
    from .records import SOURCE_SEARCH

    if (not math.isfinite(exact_path_policy_confidence)
            or not 0 <= exact_path_policy_confidence <= 1):
        raise ValueError(
            "exact_path_policy_confidence must be finite and in [0, 1]")

    weights = []
    for r in records:
        src = r.get("target_source")
        it = r.get("generation_iteration", 0)
        if src == SOURCE_SEARCH:
            weight = weight_search
        elif it >= current_iteration and current_iteration > 0:
            weight = weight_exact_new
        else:
            weight = weight_exact_historical
        if r.get("label_kind") == LABEL_EXACT_PATH_POLICY:
            weight *= exact_path_policy_confidence
        weights.append(weight)
    return weights


def uniform_loss_weights_for(
    records: list[dict[str, Any]],
) -> list[float]:
    """Give every record equal policy mass for explicit uniform ablations."""
    return [1.0] * len(records)


def value_supervision_weights_for(
    records: list[dict[str, Any]],
    base_weights: list[float],
    *,
    search_value_loss_weight: float = 0.0,
) -> list[float]:
    """Return scalar-value weights without treating search estimates as V*.

    ``search_value_loss_weight`` is an explicit confidence multiplier.  Its
    safe default is zero: search records still train the visit policy, but only
    exact-oracle records supervise scalar value.  A future bounded/interval
    objective can opt in deliberately without changing record provenance.
    """
    if len(records) != len(base_weights):
        raise ValueError("one base value weight is required per training record")
    if (not math.isfinite(search_value_loss_weight)
            or search_value_loss_weight < 0):
        raise ValueError(
            "search_value_loss_weight must be finite and non-negative")
    return [
        float(weight) * (
            1.0 if bool(record.get("value_exact", True))
            else search_value_loss_weight)
        for record, weight in zip(records, base_weights)
    ]


def train_expert(
    model: PolicyValueNet,
    records: list[dict[str, Any]],
    weights: list[float],
    *,
    encoding_config: EncodingConfig,
    value_norm: ValueNormConfig,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    device: torch.device,
    seed: int,
    policy_loss_weight: float = 1.0,
    value_loss_weight: float = 1.0,
    value_weights: list[float] | None = None,
    search_value_loss_weight: float = 0.0,
    value_anchor_model: PolicyValueNet | None = None,
    value_anchor_weight: float = 0.0,
    policy_anchor_model: PolicyValueNet | None = None,
    policy_anchor_weight: float = 0.0,
    policy_anchor_before_iteration: int | None = None,
    policy_anchor_records: list[dict[str, Any]] | None = None,
    policy_ranking_weight: float = 0.0,
    policy_ranking_margin: float = 0.25,
    trace_ranking_records: list[dict[str, Any]] | None = None,
    trace_ranking_weight: float = 0.0,
    trace_ranking_margin: float = 0.05,
) -> dict[str, Any]:
    """Fine-tune ``model`` in place on the weighted mixed dataset."""
    # Canonicalize before deriving any supervision weights. In particular,
    # legacy graph-search records may omit ``value_exact``; treating that
    # missing field as exact even briefly would incorrectly enable scalar-value
    # loss before ExpertDataset gets a chance to migrate the record.
    records = [
        ensure_record_label_metadata(dict(record))
        for record in records
    ]
    validate_positive_integer("expert training epochs", epochs)
    if not torch.isfinite(torch.tensor(value_anchor_weight)) \
            or value_anchor_weight < 0:
        raise ValueError("value anchor weight must be finite and non-negative")
    if value_anchor_weight > 0 and value_anchor_model is None:
        raise ValueError(
            "a value anchor model is required when anchor weight is positive")
    if not torch.isfinite(torch.tensor(policy_anchor_weight)) \
            or policy_anchor_weight < 0:
        raise ValueError("policy anchor weight must be finite and non-negative")
    if policy_anchor_weight > 0 and policy_anchor_model is None:
        raise ValueError(
            "a policy anchor model is required when anchor weight is positive")
    if policy_anchor_records is not None and not policy_anchor_records:
        raise ValueError("policy anchor records must be non-empty or None")
    if policy_anchor_records is not None and policy_anchor_model is None:
        raise ValueError(
            "separate policy anchor records require a policy anchor model")
    if (policy_anchor_records is not None
            and policy_anchor_before_iteration is not None):
        raise ValueError(
            "separate policy anchor records cannot use an iteration cutoff")
    if policy_anchor_before_iteration is not None and (
            isinstance(policy_anchor_before_iteration, bool)
            or not isinstance(policy_anchor_before_iteration, int)
            or policy_anchor_before_iteration < 0):
        raise ValueError(
            "policy_anchor_before_iteration must be a non-negative integer "
            "or None")
    if (not torch.isfinite(torch.tensor(policy_ranking_weight))
            or policy_ranking_weight < 0):
        raise ValueError(
            "policy ranking weight must be finite and non-negative")
    if (not torch.isfinite(torch.tensor(policy_ranking_margin))
            or policy_ranking_margin < 0):
        raise ValueError(
            "policy ranking margin must be finite and non-negative")
    if trace_ranking_records is not None and not trace_ranking_records:
        raise ValueError("trace ranking records must be non-empty or None")
    if (not torch.isfinite(torch.tensor(trace_ranking_weight))
            or trace_ranking_weight < 0):
        raise ValueError(
            "trace ranking weight must be finite and non-negative")
    if (not torch.isfinite(torch.tensor(trace_ranking_margin))
            or trace_ranking_margin < 0):
        raise ValueError(
            "trace ranking margin must be finite and non-negative")
    if trace_ranking_weight > 0 and trace_ranking_records is None:
        raise ValueError(
            "positive trace ranking weight requires separate trace records")
    base_value_weights = weights if value_weights is None else value_weights
    effective_value_weights = value_supervision_weights_for(
        records,
        base_value_weights,
        search_value_loss_weight=search_value_loss_weight,
    )
    random.seed(seed)
    torch.manual_seed(seed)
    dataset = ExpertDataset(records, weights, encoding_config=encoding_config,
                            value_norm=value_norm,
                            value_weights=effective_value_weights)
    ranking_eligible_examples = 0
    ranking_preference_pairs = 0
    for item in dataset.items:
        utilities = item["policy_rank_utility"].unsqueeze(0)
        utility_mask = item["policy_rank_mask"].unsqueeze(0)
        dummy_logits = torch.zeros_like(utilities)
        _, valid_rows, pair_counts = pairwise_policy_ranking_loss(
            dummy_logits, utilities, utility_mask,
            margin=policy_ranking_margin)
        ranking_eligible_examples += int(valid_rows.sum().item())
        ranking_preference_pairs += int(pair_counts.sum().item())
    if policy_ranking_weight > 0 and ranking_eligible_examples == 0:
        raise ValueError(
            "positive policy ranking weight requires at least one strict "
            "search-utility preference")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate, generator=generator)
    anchor_dataset = None
    if policy_anchor_records is not None:
        anchor_dataset = ExpertDataset(
            policy_anchor_records,
            [1.0] * len(policy_anchor_records),
            value_weights=[0.0] * len(policy_anchor_records),
            encoding_config=encoding_config,
            value_norm=value_norm,
        )
    trace_dataset = None
    trace_eligible_examples = 0
    if trace_ranking_records is not None:
        trace_dataset = ExpertDataset(
            trace_ranking_records,
            [1.0] * len(trace_ranking_records),
            value_weights=[0.0] * len(trace_ranking_records),
            encoding_config=encoding_config,
            value_norm=value_norm,
        )
        trace_eligible_examples = sum(
            bool(item["trace_pair_valid"]) for item in trace_dataset.items)
        if trace_eligible_examples != len(trace_dataset):
            raise ValueError(
                "every separate trace ranking record must contain one valid "
                "preference pair")
    trainable_parameters = [
        parameter for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise ValueError("expert training requires at least one trainable parameter")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate,
                                  weight_decay=weight_decay)
    history = []
    model.train()
    _set_frozen_modules_eval(model)
    if value_anchor_model is not None:
        value_anchor_model.to(device)
        value_anchor_model.eval()
        for parameter in value_anchor_model.parameters():
            parameter.requires_grad_(False)
    if policy_anchor_model is not None:
        policy_anchor_model.to(device)
        policy_anchor_model.eval()
        for parameter in policy_anchor_model.parameters():
            parameter.requires_grad_(False)
    for epoch in range(1, epochs + 1):
        policy_numerator = policy_mass_total = 0.0
        value_numerator = value_mass_total = 0.0
        anchor_numerator = anchor_mass_total = 0.0
        policy_anchor_numerator = 0.0
        policy_anchor_mass_total = 0.0
        ranking_numerator = ranking_mass_total = 0.0
        ranking_pairs_total = 0
        trace_numerator = trace_mass_total = 0.0
        anchor_batches: list[list[int]] = []
        if anchor_dataset is not None:
            main_batch_count = len(loader)
            anchor_count = len(anchor_dataset)
            base, remainder = divmod(anchor_count, main_batch_count)
            anchor_generator = torch.Generator().manual_seed(
                seed * 1_000_003 + epoch)
            permutation = torch.randperm(
                anchor_count, generator=anchor_generator).tolist()
            offset = 0
            for batch_index in range(main_batch_count):
                size = base + (1 if batch_index < remainder else 0)
                anchor_batches.append(permutation[offset:offset + size])
                offset += size
            if offset != anchor_count:
                raise RuntimeError("policy anchor batching lost records")
        average_anchor_batch = (
            len(anchor_dataset) / len(loader)
            if anchor_dataset is not None else 0.0)
        trace_batches: list[list[int]] = [[] for _ in range(len(loader))]
        if trace_dataset is not None and trace_ranking_weight > 0:
            trace_batches = _spread_auxiliary_batches(
                len(trace_dataset), len(loader),
                seed=seed * 1_000_033 + epoch)
        average_trace_batch = (
            len(trace_dataset) / len(loader)
            if trace_dataset is not None else 0.0)
        for batch_index, batch in enumerate(loader):
            board = batch["board"].to(device)
            glob = batch["global_features"].to(device)
            mask = batch["legal_action_mask"].to(device)
            ptgt = batch["policy_target"].to(device)
            vtgt = batch["value_target"].to(device)
            w = batch["weight"].to(device)
            vw = batch["value_weight"].to(device)
            generation_iteration = batch["generation_iteration"].to(device)
            rank_utility = batch["policy_rank_utility"].to(device)
            rank_mask = batch["policy_rank_mask"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits, value = model(board, glob)
            logp = masked_log_softmax(logits, mask)
            per_policy = -(ptgt * logp).sum(dim=-1)        # [B]
            valid = ptgt.sum(dim=-1) > 0
            per_value = F.smooth_l1_loss(value.reshape(-1), vtgt.reshape(-1),
                                         reduction="none")  # [B]

            wp = w * valid
            policy_loss = (per_policy * wp).sum() / wp.sum().clamp(min=1e-8)
            value_loss = (
                (per_value * vw).sum() / vw.sum().clamp(min=1e-8))
            per_ranking, ranking_valid, ranking_pairs = (
                pairwise_policy_ranking_loss(
                    logits, rank_utility, rank_mask,
                    margin=policy_ranking_margin))
            ranking_weights = w * ranking_valid
            ranking_mass = ranking_weights.sum()
            ranking_loss = (
                (per_ranking * ranking_weights).sum()
                / ranking_mass.clamp(min=1e-8))
            anchor_loss = torch.zeros((), device=device)
            if value_anchor_model is not None and value_anchor_weight > 0:
                with torch.no_grad():
                    _, anchor_value = value_anchor_model(board, glob)
                per_anchor = F.smooth_l1_loss(
                    value.reshape(-1),
                    anchor_value.reshape(-1),
                    reduction="none",
                )
                anchor_loss = (
                    (per_anchor * w).sum() / w.sum().clamp(min=1e-8))
            policy_anchor_loss = torch.zeros((), device=device)
            policy_anchor_mass = torch.zeros((), device=device)
            if policy_anchor_model is not None and policy_anchor_weight > 0:
                anchor_scale = 1.0
                if anchor_dataset is not None:
                    indices = anchor_batches[batch_index]
                    if indices:
                        anchor_batch = collate([
                            anchor_dataset[index] for index in indices])
                        anchor_board = anchor_batch["board"].to(device)
                        anchor_glob = anchor_batch["global_features"].to(device)
                        anchor_mask = anchor_batch[
                            "legal_action_mask"].to(device)
                        anchor_logits_candidate, _ = model(
                            anchor_board, anchor_glob)
                        anchor_scale = len(indices) / average_anchor_batch
                    else:
                        anchor_board = board[:0]
                        anchor_glob = glob[:0]
                        anchor_mask = mask[:0]
                        anchor_logits_candidate = logits[:0]
                        anchor_scale = 0.0
                else:
                    anchor_board = board
                    anchor_glob = glob
                    anchor_mask = mask
                    anchor_logits_candidate = logits
                with torch.no_grad():
                    anchor_logits, _ = policy_anchor_model(
                        anchor_board, anchor_glob)
                    anchor_logp = masked_log_softmax(
                        anchor_logits, anchor_mask)
                    anchor_probability = anchor_logp.exp()
                candidate_anchor_logp = masked_log_softmax(
                    anchor_logits_candidate, anchor_mask)
                # KL(incumbent || candidate) over legal actions. Computing the
                # zero-probability terms explicitly as zero avoids 0 * inf.
                # Illegal actions have -inf log-probability in both models.
                # Mask the log-ratio *before* multiplying so no 0 * NaN term
                # can enter autograd.
                legal_log_ratio = torch.where(
                    anchor_mask.to(torch.bool),
                    anchor_logp - candidate_anchor_logp,
                    torch.zeros_like(anchor_logp),
                )
                per_policy_anchor = (
                    anchor_probability * legal_log_ratio).sum(dim=-1)
                if anchor_dataset is not None:
                    policy_anchor_weights = torch.ones(
                        len(anchor_board), dtype=w.dtype, device=device)
                elif policy_anchor_before_iteration is None:
                    policy_anchor_weights = w
                else:
                    historical = (
                        generation_iteration
                        < policy_anchor_before_iteration).to(w.dtype)
                    policy_anchor_weights = w * historical
                policy_anchor_mass = policy_anchor_weights.sum()
                policy_anchor_loss = (
                    (per_policy_anchor * policy_anchor_weights).sum()
                    / policy_anchor_mass.clamp(min=1e-8)) * anchor_scale
            trace_ranking_loss = torch.zeros((), device=device)
            trace_ranking_raw_loss = torch.zeros((), device=device)
            trace_ranking_mass = 0
            if trace_dataset is not None and trace_ranking_weight > 0:
                trace_indices = trace_batches[batch_index]
                if trace_indices:
                    trace_batch = collate([
                        trace_dataset[index] for index in trace_indices])
                    trace_board = trace_batch["board"].to(device)
                    trace_glob = trace_batch["global_features"].to(device)
                    trace_preferred = trace_batch[
                        "trace_preferred_index"].to(device)
                    trace_competing = trace_batch[
                        "trace_competing_index"].to(device)
                    trace_valid = trace_batch["trace_pair_valid"].to(device)
                    if not bool(torch.all(trace_valid)):
                        raise RuntimeError(
                            "trace ranking batch contains an invalid pair")
                    trace_logits, _ = model(trace_board, trace_glob)
                    per_trace_ranking = trace_pairwise_hinge_loss(
                        trace_logits, trace_preferred, trace_competing,
                        margin=trace_ranking_margin)
                    trace_ranking_raw_loss = per_trace_ranking.mean()
                    trace_scale = len(trace_indices) / average_trace_batch
                    trace_ranking_loss = (
                        trace_ranking_raw_loss * trace_scale)
                    trace_ranking_mass = len(trace_indices)
            loss = (
                policy_loss_weight * policy_loss
                + value_loss_weight * value_loss
                + value_anchor_weight * anchor_loss
                + policy_anchor_weight * policy_anchor_loss
                + policy_ranking_weight * ranking_loss
                + trace_ranking_weight * trace_ranking_loss
            )
            loss.backward()
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, grad_clip)
            optimizer.step()
            policy_mass = float(wp.sum().item())
            value_mass = float(vw.sum().item())
            anchor_mass = (
                float(w.sum().item())
                if value_anchor_model is not None and value_anchor_weight > 0
                else 0.0
            )
            policy_anchor_mass_value = float(policy_anchor_mass.item())
            ranking_mass_value = float(ranking_mass.item())
            policy_numerator += float(policy_loss.item()) * policy_mass
            policy_mass_total += policy_mass
            value_numerator += float(value_loss.item()) * value_mass
            value_mass_total += value_mass
            anchor_numerator += float(anchor_loss.item()) * anchor_mass
            anchor_mass_total += anchor_mass
            policy_anchor_numerator += (
                float(policy_anchor_loss.item()) * policy_anchor_mass_value)
            policy_anchor_mass_total += policy_anchor_mass_value
            ranking_numerator += (
                float(ranking_loss.item()) * ranking_mass_value)
            ranking_mass_total += ranking_mass_value
            ranking_pairs_total += int(ranking_pairs.sum().item())
            trace_numerator += (
                float(trace_ranking_raw_loss.item()) * trace_ranking_mass)
            trace_mass_total += trace_ranking_mass
        policy_mean = (
            policy_numerator / policy_mass_total
            if policy_mass_total else 0.0)
        value_mean = (
            value_numerator / value_mass_total
            if value_mass_total else 0.0)
        anchor_mean = (
            anchor_numerator / anchor_mass_total
            if anchor_mass_total else 0.0)
        policy_anchor_mean = (
            policy_anchor_numerator / policy_anchor_mass_total
            if policy_anchor_mass_total else 0.0)
        ranking_mean = (
            ranking_numerator / ranking_mass_total
            if ranking_mass_total else 0.0)
        trace_ranking_mean = (
            trace_numerator / trace_mass_total
            if trace_mass_total else 0.0)
        total_mean = (
            policy_loss_weight * policy_mean
            + value_loss_weight * value_mean
            + value_anchor_weight * anchor_mean
            + policy_anchor_weight * policy_anchor_mean
            + policy_ranking_weight * ranking_mean
            + trace_ranking_weight * trace_ranking_mean
        )
        history.append({
            "epoch": epoch,
            "loss": total_mean,
            "policy_loss": policy_mean,
            "value_loss": value_mean,
            "value_anchor_loss": anchor_mean,
            "policy_anchor_loss": policy_anchor_mean,
            "policy_anchor_weight_mass": policy_anchor_mass_total,
            "policy_ranking_loss": ranking_mean,
            "policy_ranking_weight_mass": ranking_mass_total,
            "policy_ranking_preference_pairs": ranking_pairs_total,
            "trace_ranking_loss": trace_ranking_mean,
            "trace_ranking_weight_mass": trace_mass_total,
            "trace_ranking_preference_pairs": int(trace_mass_total),
        })
    exact_value_examples = sum(
        bool(record.get("value_exact", True)) for record in records)
    search_value_examples = len(records) - exact_value_examples
    policy_mass_by_source: dict[str, float] = {}
    policy_mass_by_label_kind: dict[str, float] = {}
    value_mass_by_source: dict[str, float] = {}
    for record, policy_weight, value_weight in zip(
            records, weights, effective_value_weights):
        source = str(record.get("target_source", "exact_oracle"))
        policy_mass_by_source[source] = (
            policy_mass_by_source.get(source, 0.0) + float(policy_weight))
        label_kind = str(record.get("label_kind", "full_exact"))
        policy_mass_by_label_kind[label_kind] = (
            policy_mass_by_label_kind.get(label_kind, 0.0)
            + float(policy_weight))
        value_mass_by_source[source] = (
            value_mass_by_source.get(source, 0.0) + float(value_weight))
    return {
        "history": history,
        "examples": len(dataset),
        "value_supervision": {
            "search_value_loss_weight": search_value_loss_weight,
            "exact_examples": exact_value_examples,
            "search_estimate_examples": search_value_examples,
            "exact_weight_mass": sum(
                weight for record, weight in zip(
                    records, effective_value_weights)
                if bool(record.get("value_exact", True))),
            "search_estimate_weight_mass": sum(
                weight for record, weight in zip(
                    records, effective_value_weights)
                if not bool(record.get("value_exact", True))),
            "total_weight_mass": sum(effective_value_weights),
        },
        "gradient_weight_mass": {
            "policy_by_source": dict(sorted(policy_mass_by_source.items())),
            "policy_by_label_kind": dict(sorted(
                policy_mass_by_label_kind.items())),
            "value_by_source": dict(sorted(value_mass_by_source.items())),
            "policy_total": sum(float(weight) for weight in weights),
            "value_total": sum(effective_value_weights),
        },
        "loss_weighting_policy": (
            "uniform_after_weighted_replay_sampling_v1"
            if all(math.isclose(float(weight), 1.0) for weight in weights)
            else "caller_provided_nonuniform_loss_weights_v1"
        ),
        "policy_anchor": {
            "weight": float(policy_anchor_weight),
            "before_iteration": policy_anchor_before_iteration,
            "eligible_examples": sum(
                1 for record in (
                    policy_anchor_records
                    if policy_anchor_records is not None else records)
                if (policy_anchor_records is not None
                    or policy_anchor_before_iteration is None
                    or int(record.get("generation_iteration", 0))
                    < policy_anchor_before_iteration)),
            "scope": (
                "separate_anchor_records"
                if policy_anchor_records is not None
                else (
                    "all_records"
                    if policy_anchor_before_iteration is None
                    else "historical_records_only")),
            "direction": "kl_incumbent_to_candidate_on_legal_actions",
        },
        "policy_ranking": {
            "weight": float(policy_ranking_weight),
            "margin": float(policy_ranking_margin),
            "eligible_examples": ranking_eligible_examples,
            "preference_pairs": ranking_preference_pairs,
            "utility_field": POLICY_SEARCH_UTILITY_FIELD,
            "scope": "strict_preferences_among_full_exact_optimal_actions",
            "loss": "utility_gap_weighted_pairwise_softplus_margin",
        },
        "trace_ranking": {
            "weight": float(trace_ranking_weight),
            "margin": float(trace_ranking_margin),
            "eligible_examples": trace_eligible_examples,
            "preference_pairs": trace_eligible_examples,
            "scope": "separate_records_first_divergent_action_pair_only",
            "target_loss_weight": 0.0,
            "value_loss_weight": 0.0,
            "batching": "one_shuffled_evenly_spread_pass_per_epoch_v1",
            "loss": "pairwise_hinge_margin",
        },
    }


def _set_frozen_modules_eval(model: torch.nn.Module) -> None:
    """Stop frozen submodules from mutating state such as BatchNorm statistics."""
    for module in model.modules():
        if module is model:
            continue
        parameters = tuple(module.parameters(recurse=True))
        if parameters and not any(
                parameter.requires_grad for parameter in parameters):
            module.eval()
