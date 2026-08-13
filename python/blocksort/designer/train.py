"""Designer adversarial training driver + CLI.

Builds a frozen protagonist + oracle from a protagonist checkpoint, rolls out
designer episodes, scores finalized levels, runs a PPO update, and retains
accepted difficult levels in a level replay buffer.

    python -m blocksort.designer.train \\
        --protagonist-checkpoint runs/pv/best.pt \\
        --output-dir runs/designer --episodes 10000 --mutation-budget 12 \\
        --protagonist-simulations 100 --oracle-simulations 1000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Optional

import torch

from ..environment import Environment
from ..cotraining.frontier import geometric_budget_sweep
from ..model_identity import model_state_sha256
from ..search.seeding import derive_trial_seed, level_search_identity
from ..training.checkpoint import (configs_from_checkpoint, load_checkpoint,
                                    model_from_checkpoint)
from ..training.experiment_identity import (
    EVALUATION_SEMANTICS_VERSION, TRANSACTION_SCHEMA_VERSION,
    build_experiment_spec, ensure_fresh_output_directory, file_identity,
    hash_canonical_value, hash_file_streaming, runtime_device_provenance,
    semantic_dataclass_config,
    validate_or_initialize_experiment)
from ..training.transaction import atomic_write_json
from ..training.validation import (
    validate_nonnegative_integer, validate_positive_integer)
from .actions import DesignerActionSpace
from .checkpoint import (
    designer_state_dict_from_checkpoint,
    load_designer,
    save_designer,
)
from .config import GeneratorConfig, RewardConfig
from .encoding import encode_designer_state
from .env import DesignerEnv
from .model import DesignerModelConfig, DesignerNet
from .ppo import PPOConfig, ppo_update, rollout_episode
from .replay import LevelReplayBuffer, build_level_record, level_fingerprint
from .reward import RewardBreakdown
from .roles import Oracle, Protagonist, SolveOutcome
from .score import score_level


@dataclass
class TrainConfig:
    protagonist_checkpoint: str
    output_dir: str
    init_designer: Optional[str] = None
    episodes: int = 200
    episodes_per_iter: int = 16
    mutation_budget: int = 12
    protagonist_simulations: int = 100
    oracle_simulations: int = 1000
    astar_max_nodes: int = 200_000
    astar_time_limit_seconds: Optional[float] = None
    seed: int = 42
    device: str = "auto"
    max_replay: int = 5_000
    validation_episodes: int = 1
    frontier_solve_rate_trials: int = 1
    frontier_min_solve_rate: float = 0.2
    frontier_max_solve_rate: float = 0.7
    frontier_alignment_weight: float = 0.0
    frontier_dirichlet_alpha: float = 0.5
    frontier_dirichlet_weight: float = 0.4
    frontier_budget_min_ratio: float = 0.25
    frontier_budget_max_ratio: float = 4.0
    frontier_min_simulations: int = 20
    frontier_max_simulations: int = 400

    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    model: DesignerModelConfig = field(default_factory=DesignerModelConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        validate_positive_integer("designer episodes", self.episodes)
        validate_positive_integer(
            "designer episodes_per_iter", self.episodes_per_iter)
        validate_nonnegative_integer(
            "designer validation_episodes", self.validation_episodes)
        validate_positive_integer(
            "designer frontier_solve_rate_trials",
            self.frontier_solve_rate_trials)
        for name, value in (
                ("frontier_min_solve_rate", self.frontier_min_solve_rate),
                ("frontier_max_solve_rate", self.frontier_max_solve_rate),
                ("frontier_dirichlet_weight", self.frontier_dirichlet_weight)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or not 0.0 <= float(value) <= 1.0):
                raise ValueError(
                    f"designer {name} must be a finite rate in [0, 1]")
        if self.frontier_min_solve_rate > self.frontier_max_solve_rate:
            raise ValueError(
                "designer frontier_min_solve_rate cannot exceed "
                "frontier_max_solve_rate")
        for name, value in (
                ("frontier_alignment_weight", self.frontier_alignment_weight),
                ("frontier_dirichlet_alpha", self.frontier_dirichlet_alpha),
                ("frontier_budget_min_ratio",
                 self.frontier_budget_min_ratio),
                ("frontier_budget_max_ratio",
                 self.frontier_budget_max_ratio)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(float(value)) or value < 0):
                raise ValueError(
                    f"designer {name} must be finite and non-negative")
        if self.frontier_budget_min_ratio <= 0:
            raise ValueError(
                "designer frontier_budget_min_ratio must be positive")
        if self.frontier_budget_max_ratio <= 0:
            raise ValueError(
                "designer frontier_budget_max_ratio must be positive")
        if self.frontier_budget_min_ratio > self.frontier_budget_max_ratio:
            raise ValueError(
                "designer frontier_budget_min_ratio cannot exceed "
                "frontier_budget_max_ratio")
        for name, value in (
                ("frontier_min_simulations", self.frontier_min_simulations),
                ("frontier_max_simulations", self.frontier_max_simulations)):
            validate_positive_integer(f"designer {name}", value)
        if self.frontier_min_simulations > self.frontier_max_simulations:
            raise ValueError(
                "designer frontier_min_simulations cannot exceed "
                "frontier_max_simulations")
        if self.frontier_alignment_weight > 0:
            if self.frontier_solve_rate_trials < 2:
                raise ValueError(
                    "designer frontier alignment requires at least two "
                    "solve-rate trials")
            if self.frontier_dirichlet_alpha <= 0:
                raise ValueError(
                    "designer frontier alignment requires positive "
                    "frontier_dirichlet_alpha")
            if self.frontier_dirichlet_weight <= 0:
                raise ValueError(
                    "designer frontier alignment requires positive "
                    "frontier_dirichlet_weight")
        if self.astar_time_limit_seconds is not None and (
                isinstance(self.astar_time_limit_seconds, bool)
                or not isinstance(self.astar_time_limit_seconds, (int, float))
                or not math.isfinite(float(self.astar_time_limit_seconds))
                or self.astar_time_limit_seconds <= 0):
            raise ValueError(
                "designer astar_time_limit_seconds must be None or a finite "
                "positive number")


_DESIGNER_INPUT_FIELDS = ("protagonist_checkpoint", "init_designer")
_DESIGNER_OPERATIONAL_FIELDS = ("output_dir",)
_DESIGNER_DERIVED_FIELDS = ("device",)
_DESIGNER_SEMANTIC_FIELDS = (
    "episodes", "episodes_per_iter", "mutation_budget",
    "protagonist_simulations", "oracle_simulations", "astar_max_nodes",
    "astar_time_limit_seconds", "seed", "max_replay", "validation_episodes",
    "frontier_solve_rate_trials", "frontier_min_solve_rate",
    "frontier_max_solve_rate", "frontier_alignment_weight",
    "frontier_dirichlet_alpha", "frontier_dirichlet_weight",
    "frontier_budget_min_ratio", "frontier_budget_max_ratio",
    "frontier_min_simulations", "frontier_max_simulations",
    "generator", "reward", "model", "ppo")


def _designer_training_spec(
    cfg: TrainConfig, *, resolved_device: str | torch.device | None = None
) -> dict[str, Any]:
    protagonist = load_checkpoint(cfg.protagonist_checkpoint, map_location="cpu")
    encoding_max_blocks = int(
        protagonist["encoding_config"]["max_blocks"])
    if cfg.generator.max_blocks > encoding_max_blocks:
        raise ValueError(
            "generator max_blocks exceeds protagonist checkpoint encoding "
            f"limit: {cfg.generator.max_blocks} > {encoding_max_blocks}")
    inputs = {
        "protagonist_checkpoint": file_identity(
            cfg.protagonist_checkpoint, kind="protagonist_checkpoint",
            format_version=int(protagonist["checkpoint_version"]),
            extra={
                "encoding_config": protagonist["encoding_config"],
                "model_config": protagonist["model_config"],
                "value_norm": protagonist["value_norm"],
            }),
        "initial_designer": None,
    }
    if cfg.init_designer:
        initial = load_designer(cfg.init_designer, map_location="cpu")
        inputs["initial_designer"] = file_identity(
            cfg.init_designer, kind="initial_designer_checkpoint",
            format_version=int(initial["designer_checkpoint_version"]),
            extra={
                "encoding_config": initial["encoding_config"],
                "model_config": initial["model_config"],
            })
    semantic = semantic_dataclass_config(
        cfg, semantic_fields=_DESIGNER_SEMANTIC_FIELDS,
        operational_fields=_DESIGNER_OPERATIONAL_FIELDS,
        input_fields=_DESIGNER_INPUT_FIELDS,
        derived_fields=_DESIGNER_DERIVED_FIELDS)
    return build_experiment_spec(
        pipeline="designer_training", semantic_config=semantic, inputs=inputs,
        software_semantics={
            "evaluation_semantics_version": EVALUATION_SEMANTICS_VERSION,
            "transaction_schema_version": TRANSACTION_SCHEMA_VERSION,
            "experiment_identity_version": 1,
            "generated_level_solvability_policy":
                "reverse_construction_exact_first_bounded_no_fallback_v1",
            "designer_frontier_reward_policy":
                "budget_sweep_centered_plateau_alignment_v3",
            "frontier_estimation_policy":
                "geometric_search_budget_sweep_v1",
            "designer_checkpoint_selection_policy":
                "frontier_in_band_alignment_reward_lexicographic_v1",
            "encoding_block_limit_policy":
                "generator_and_state_must_not_exceed_checkpoint_v1",
            "runtime": runtime_device_provenance(
                requested_device=cfg.device,
                resolved_device=resolved_device or _resolve_device(cfg.device)),
        })


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def training_episode_batches(episodes: int, episodes_per_iter: int) -> tuple[int, ...]:
    """Exact deterministic batch plan, including a partial final iteration."""
    validate_positive_integer("designer episodes", episodes)
    validate_positive_integer(
        "designer episodes_per_iter", episodes_per_iter)
    full, remainder = divmod(episodes, episodes_per_iter)
    return ((episodes_per_iter,) * full
            + ((remainder,) if remainder else ()))


def frontier_alignment_score(
    solve_rate: float,
    *,
    minimum: float,
    maximum: float,
) -> float:
    """Plateau reward for the learning frontier, tapering to zero at 0 and 1."""
    rate = min(1.0, max(0.0, float(solve_rate)))
    if minimum <= rate <= maximum:
        return 1.0
    if rate < minimum:
        return rate / minimum if minimum > 0 else 1.0
    return ((1.0 - rate) / (1.0 - maximum)
            if maximum < 1.0 else 1.0)


class _FixedOutcomeProtagonist:
    """Adapter that lets score_level reuse an already-computed trial outcome."""

    def __init__(self, outcome: SolveOutcome) -> None:
        self.outcome = outcome

    def solve(self, _level, *, seed: int = 0) -> SolveOutcome:
        return self.outcome


def _score_generated_level(
    env: Environment,
    finalize,
    *,
    protagonist: Protagonist,
    oracle: Oracle,
    reward_cfg: RewardConfig,
    novelty: float,
    seed: int,
    astar_max_nodes: int,
    frontier_solve_rate_trials: int,
    frontier_min_solve_rate: float,
    frontier_max_solve_rate: float,
    frontier_alignment_weight: float,
    evaluation_context: str,
    frontier_simulation_budgets: tuple[int, ...] | None = None,
):
    """Score once with the oracle and optionally add repeated-trial alignment."""
    if frontier_alignment_weight <= 0 or not finalize.valid:
        return score_level(
            env, finalize, protagonist=protagonist, oracle=oracle,
            reward_cfg=reward_cfg, novelty=novelty, seed=seed,
            astar_max_nodes=astar_max_nodes, construction_solvable=True)

    identity = level_search_identity(env, finalize.level)
    trial_seeds = tuple(
        derive_trial_seed(
            seed,
            trial_index=index,
            level_identity=identity,
            evaluation_context=evaluation_context,
        )
        for index in range(frontier_solve_rate_trials)
    )
    budgets = tuple(frontier_simulation_budgets or ())
    if budgets and len(budgets) != len(trial_seeds):
        raise ValueError(
            "designer frontier simulation budget count must equal trials")
    outcomes = tuple(
        protagonist.solve(
            finalize.level,
            seed=trial_seed,
            **({"simulations": budgets[index]} if budgets else {}),
        )
        for index, trial_seed in enumerate(trial_seeds)
    )
    scored = score_level(
        env, finalize,
        protagonist=_FixedOutcomeProtagonist(outcomes[0]),
        oracle=oracle, reward_cfg=reward_cfg, novelty=novelty, seed=seed,
        astar_max_nodes=astar_max_nodes, construction_solvable=True)
    solved = sum(1 for outcome in outcomes if outcome.solved)
    solve_rate = solved / len(outcomes)
    alignment = frontier_alignment_score(
        solve_rate,
        minimum=frontier_min_solve_rate,
        maximum=frontier_max_solve_rate,
    )
    eligible = bool(scored.valid and scored.oracle.solved)
    # A one-sided bonus leaves the old adversarial-regret reward free to make
    # always-failed levels competitive with frontier levels. Centering the
    # adjustment makes the desired band positive, the 0/1 extremes negative,
    # and preserves a dense taper between those endpoints.
    bonus = frontier_alignment_weight * alignment if eligible else 0.0
    extremity_penalty = (
        frontier_alignment_weight * (1.0 - alignment) if eligible else 0.0
    )
    adjustment = bonus - extremity_penalty
    components = dict(scored.reward.components)
    components.update({
        "frontier_solve_rate": solve_rate,
        "frontier_solve_rate_trials": len(outcomes),
        "frontier_solved_trials": solved,
        "frontier_simulation_budgets": list(budgets),
        "frontier_min_solve_rate": frontier_min_solve_rate,
        "frontier_max_solve_rate": frontier_max_solve_rate,
        "frontier_in_band": eligible and (
            frontier_min_solve_rate <= solve_rate
            <= frontier_max_solve_rate),
        "frontier_alignment": alignment,
        "frontier_alignment_weight": frontier_alignment_weight,
        "frontier_alignment_bonus": bonus,
        "frontier_extremity_penalty": extremity_penalty,
        "frontier_alignment_adjustment": adjustment,
        "frontier_reward_eligible": eligible,
    })
    return replace(
        scored,
        reward=RewardBreakdown(
            total=scored.reward.total + adjustment,
            components=components,
        ),
    )


def _aggregate(episodes) -> dict[str, Any]:
    n = len(episodes)
    valid = sum(1 for e in episodes if e.scored.valid)
    oracle_solved = sum(1 for e in episodes if e.scored.oracle.solved)
    prot_solved = sum(1 for e in episodes if e.scored.protagonist.solved)
    regrets = [e.scored.reward["adversarial_regret"] for e in episodes
               if e.scored.oracle.solved]
    extras = [e.scored.structural.extra_moves for e in episodes
              if e.scored.structural.extra_moves is not None]
    rewards = [e.reward for e in episodes]
    frontier_components = [
        e.scored.reward.components
        for e in episodes
        if isinstance(getattr(e.scored.reward, "components", None), dict)
        and "frontier_solve_rate" in e.scored.reward.components
    ]
    frontier_rates = [
        float(components["frontier_solve_rate"])
        for components in frontier_components
    ]
    frontier_alignments = [
        float(components["frontier_alignment"])
        for components in frontier_components
    ]
    frontier_in_band = sum(
        1 for components in frontier_components
        if components["frontier_in_band"])
    return {
        "episodes": n,
        "valid_rate": valid / n if n else 0.0,
        "oracle_solve_rate": oracle_solved / n if n else 0.0,
        "protagonist_solve_rate": prot_solved / n if n else 0.0,
        "mean_reward": sum(rewards) / n if n else 0.0,
        "mean_adversarial_regret": sum(regrets) / len(regrets) if regrets else 0.0,
        "mean_extra_moves": sum(extras) / len(extras) if extras else None,
        "mean_frontier_solve_rate": (
            sum(frontier_rates) / len(frontier_rates)
            if frontier_rates else None),
        "frontier_in_band_rate": (
            frontier_in_band / len(frontier_components)
            if frontier_components else None),
        "mean_frontier_alignment": (
            sum(frontier_alignments) / len(frontier_alignments)
            if frontier_alignments else None),
        "frontier_evaluated_count": len(frontier_components),
    }


def _evaluate_designer(
    env: DesignerEnv,
    model: DesignerNet,
    action_space: DesignerActionSpace,
    encoding,
    *,
    protagonist: Protagonist,
    oracle: Oracle,
    reward_cfg: RewardConfig,
    validation_episodes: int,
    seed: int,
    device: torch.device,
    astar_max_nodes: int,
    frontier_solve_rate_trials: int = 1,
    frontier_min_solve_rate: float = 0.2,
    frontier_max_solve_rate: float = 0.7,
    frontier_alignment_weight: float = 0.0,
    frontier_simulation_budgets: tuple[int, ...] | None = None,
):
    """Evaluate ``model`` deterministically without touching training state."""
    episodes = []
    rng = random.Random(seed ^ 0x5EED5EED)
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            for i in range(validation_episodes):
                episode_seed = seed * 1_000_003 + 700_001 + i
                ep = rollout_episode(
                    env, model, action_space, encoding, seed=episode_seed,
                    device=device, rng=rng, verify_finalize=False)
                scored = _score_generated_level(
                    env.env, ep.finalize, protagonist=protagonist, oracle=oracle,
                    # Validation has no mutable novelty history: use the same
                    # fixed novelty contribution on every evaluation.
                    reward_cfg=reward_cfg, novelty=0.0, seed=episode_seed,
                    astar_max_nodes=astar_max_nodes,
                    frontier_solve_rate_trials=frontier_solve_rate_trials,
                    frontier_min_solve_rate=frontier_min_solve_rate,
                    frontier_max_solve_rate=frontier_max_solve_rate,
                    frontier_alignment_weight=frontier_alignment_weight,
                    evaluation_context="designer.validation.frontier",
                    frontier_simulation_budgets=(
                        frontier_simulation_budgets))
                ep.reward = scored.reward.total
                ep.scored = scored
                episodes.append(ep)
    finally:
        model.train(was_training)
    return episodes


_DESIGNER_SELECTION_POLICY = (
    "frontier_in_band_alignment_reward_lexicographic_v1"
)


def designer_selection_key(
    validation_metrics: dict[str, Any],
) -> tuple[float, float, float]:
    """Rank validation results by frontier quality, then ordinary reward."""
    evaluated = int(validation_metrics.get("frontier_evaluated_count") or 0)
    if evaluated > 0:
        in_band_rate = float(
            validation_metrics.get("frontier_in_band_rate") or 0.0)
        alignment = float(
            validation_metrics.get("mean_frontier_alignment") or 0.0)
    else:
        # Standalone runs can explicitly disable frontier scoring. Keep their
        # legacy reward ordering without pretending the frontier was measured.
        in_band_rate = -1.0
        alignment = -1.0
    reward = float(validation_metrics["mean_reward"])
    return in_band_rate, alignment, reward


def designer_selection_metric(
    validation_metrics: dict[str, Any],
) -> dict[str, Any]:
    key = designer_selection_key(validation_metrics)
    return {
        "name": _DESIGNER_SELECTION_POLICY,
        "priority": [
            "frontier_in_band_rate",
            "mean_frontier_alignment",
            "validation_mean_reward",
        ],
        "value": {
            "frontier_in_band_rate": key[0],
            "mean_frontier_alignment": key[1],
            "validation_mean_reward": key[2],
        },
    }


def train_designer(cfg: TrainConfig) -> dict[str, Any]:
    cfg.validate()
    root = Path(cfg.output_dir)
    ensure_fresh_output_directory(
        root, pipeline_label="Designer training")
    device = _resolve_device(cfg.device)
    experiment_fingerprint, _ = validate_or_initialize_experiment(
        root, _designer_training_spec(cfg, resolved_device=device),
        run_state=None)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / "config.json", asdict(cfg))

    # Frozen protagonist + oracle from the protagonist checkpoint.
    pck = load_checkpoint(cfg.protagonist_checkpoint, map_location="cpu")
    enc, _mc, value_norm = configs_from_checkpoint(pck)
    prot_model = model_from_checkpoint(pck, map_location=device)

    env = DesignerEnv(cfg.generator, mutation_budget=cfg.mutation_budget,
                      encoding=enc)
    action_space = DesignerActionSpace(enc)
    protagonist = Protagonist(env.env, prot_model, enc, value_norm, device,
                              simulations=cfg.protagonist_simulations,
                              c_puct=1.5,
                              dirichlet_alpha=(
                                  cfg.frontier_dirichlet_alpha
                                  if cfg.frontier_alignment_weight > 0 else 0.0),
                              dirichlet_weight=(
                                  cfg.frontier_dirichlet_weight
                                  if cfg.frontier_alignment_weight > 0 else 0.0))
    frontier_simulation_budgets = geometric_budget_sweep(
        center=cfg.protagonist_simulations,
        trials=cfg.frontier_solve_rate_trials,
        minimum_ratio=cfg.frontier_budget_min_ratio,
        maximum_ratio=cfg.frontier_budget_max_ratio,
        minimum_simulations=cfg.frontier_min_simulations,
        maximum_simulations=cfg.frontier_max_simulations,
    )
    oracle = Oracle(env.env, prot_model, enc, value_norm, device,
                    astar_max_nodes=cfg.astar_max_nodes,
                    astar_time_limit_seconds=cfg.astar_time_limit_seconds,
                    search_simulations=cfg.oracle_simulations, c_puct=1.5,
                    fallback_on_astar_exhaustion=False)

    # Seed before constructing a fresh designer so its initial parameters are
    # determined by the run configuration rather than ambient process RNG state.
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    # Designer model (optionally warm-started from a pretrained checkpoint).
    model = DesignerNet(enc, cfg.model).to(device)
    if cfg.init_designer:
        init = load_designer(cfg.init_designer, map_location=device)
        model.load_state_dict(designer_state_dict_from_checkpoint(init))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.ppo.learning_rate)

    buffer = LevelReplayBuffer(root / "replay", max_levels=cfg.max_replay,
                               seed=cfg.seed).load()
    seen = buffer.fingerprints()

    rng = random.Random(cfg.seed)

    episode_batches = training_episode_batches(
        cfg.episodes, cfg.episodes_per_iter)
    iterations = len(episode_batches)
    history: list[dict[str, Any]] = []
    best_selection: tuple[float, float, float] | None = None
    best_selection_metric: dict[str, Any] = {}
    best_reward = -float("inf")
    best_validation_metrics: dict[str, Any] = {}
    episode_counter = 0
    validation_episode_counter = 0

    for it, episodes_this_iteration in enumerate(episode_batches):
        generator_model_state_sha256 = model_state_sha256(model)
        episodes = []
        records = []
        for _ in range(episodes_this_iteration):
            episode_seed = cfg.seed * 100003 + episode_counter
            ep = rollout_episode(env, model, action_space, enc,
                                 seed=episode_seed,
                                 device=device, rng=rng, verify_finalize=False)
            episode_counter += 1

            fp = (level_fingerprint(env.env, ep.finalize.level)
                  if ep.finalize.valid else None)
            novelty = 1.0 if (fp is not None and fp not in seen) else 0.0
            scored = _score_generated_level(
                env.env, ep.finalize, protagonist=protagonist,
                oracle=oracle, reward_cfg=cfg.reward,
                novelty=novelty, seed=episode_seed,
                astar_max_nodes=cfg.astar_max_nodes,
                frontier_solve_rate_trials=cfg.frontier_solve_rate_trials,
                frontier_min_solve_rate=cfg.frontier_min_solve_rate,
                frontier_max_solve_rate=cfg.frontier_max_solve_rate,
                frontier_alignment_weight=cfg.frontier_alignment_weight,
                evaluation_context="designer.training.frontier",
                frontier_simulation_budgets=frontier_simulation_budgets)
            ep.reward = scored.reward.total
            ep.scored = scored
            if fp is not None:
                seen.add(fp)

            if scored.valid and scored.oracle.solved:
                records.append(build_level_record(
                    env.env, ep.finalize.level, trajectory=ep.trajectory,
                    designer_checkpoint=str(root / "last.pt"),
                    generator_model_state_sha256=generator_model_state_sha256,
                    protagonist_checkpoint=cfg.protagonist_checkpoint,
                    oracle_result=scored.oracle_result(),
                    reward_components=scored.reward.components,
                    structural_metrics=scored.structural.to_dict(),
                    solver_metrics=scored.solver_metrics(),
                    generation_iteration=it, reward_total=ep.reward))
            episodes.append(ep)

        add_stats = buffer.add(records)
        agg = _aggregate(episodes)
        agg["training_mean_reward"] = agg["mean_reward"]
        agg.update({"iteration": it, "accepted": add_stats["added"],
                    "duplicates": add_stats["duplicates"],
                    "replay_size": len(buffer)})

        update_stats = ppo_update(model, optimizer, episodes, cfg.ppo, device,
                                  seed=cfg.seed + it)
        agg["ppo"] = update_stats

        validation_episodes = _evaluate_designer(
            env, model, action_space, enc, protagonist=protagonist,
            oracle=oracle, reward_cfg=cfg.reward,
            validation_episodes=cfg.validation_episodes, seed=cfg.seed,
            device=device, astar_max_nodes=cfg.astar_max_nodes,
            frontier_solve_rate_trials=cfg.frontier_solve_rate_trials,
            frontier_min_solve_rate=cfg.frontier_min_solve_rate,
            frontier_max_solve_rate=cfg.frontier_max_solve_rate,
            frontier_alignment_weight=cfg.frontier_alignment_weight,
            frontier_simulation_budgets=frontier_simulation_budgets)
        validation_episode_counter += len(validation_episodes)
        validation_metrics = _aggregate(validation_episodes)
        agg["validation_mean_reward"] = validation_metrics["mean_reward"]
        agg["validation"] = validation_metrics
        selection_key = designer_selection_key(validation_metrics)
        selection_metric = designer_selection_metric(validation_metrics)
        agg["designer_selection_metric"] = selection_metric

        if best_selection is None or selection_key >= best_selection:
            best_selection = selection_key
            best_selection_metric = selection_metric
            best_reward = agg["validation_mean_reward"]
            best_validation_metrics = dict(validation_metrics)
            save_designer(
                root / "best.pt", model=model, encoding_config=enc,
                model_config=cfg.model, seed=cfg.seed,
                metadata={
                    "experiment_fingerprint": experiment_fingerprint,
                    "iteration": it,
                    "metrics": dict(agg),
                    "checkpoint_model_state": "post_update",
                    "selection_metric": selection_metric,
                })

        history.append(agg)
        print(json.dumps({k: agg[k] for k in (
            "iteration", "training_mean_reward", "validation_mean_reward",
            "valid_rate", "oracle_solve_rate", "protagonist_solve_rate",
            "mean_adversarial_regret", "mean_frontier_solve_rate",
            "frontier_in_band_rate", "mean_frontier_alignment", "accepted")}),
            flush=True)

        save_designer(root / "last.pt", model=model, encoding_config=enc,
                      model_config=cfg.model, seed=cfg.seed,
                      metadata={
                          "experiment_fingerprint": experiment_fingerprint,
                          "iteration": it,
                          "metrics": agg,
                          "checkpoint_model_state": "post_update",
                          "training_metrics_model_state": "pre_update",
                          "validation_metrics_model_state": "post_update",
                      })
        buffer.persist()

    summary = {"iterations": iterations, "episodes": episode_counter,
               "requested_training_episodes": cfg.episodes,
               "completed_training_episodes": episode_counter,
               "validation_episodes": validation_episode_counter,
               "experiment_fingerprint": experiment_fingerprint,
               "best_mean_reward": best_reward,
               "best_validation_mean_reward": best_reward,
               "best_validation_metrics": best_validation_metrics,
               "best_selection_metric": best_selection_metric,
               "frontier_simulation_budgets":
                   list(frontier_simulation_budgets),
               "history": history,
               "replay_size": len(buffer),
               "best_checkpoint": str(root / "best.pt"),
               "last_checkpoint": str(root / "last.pt"),
               "best_checkpoint_sha256": hash_file_streaming(root / "best.pt"),
               "last_checkpoint_sha256": hash_file_streaming(root / "last.pt"),
               "encoding_fingerprint": hash_canonical_value({
                   "encoding_config": enc.to_dict(),
                   "model_config": cfg.model.to_dict(),
               })}
    (root / "summary.json").write_text(json.dumps(summary, indent=2),
                                       encoding="utf-8")
    return summary


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Adversarial designer training")
    p.add_argument("--protagonist-checkpoint", required=True)
    p.add_argument("--output-dir", default="runs/designer")
    p.add_argument("--init-designer", default=None)
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--episodes-per-iter", type=int, default=16)
    p.add_argument("--mutation-budget", type=int, default=12)
    p.add_argument("--protagonist-simulations", type=int, default=100)
    p.add_argument("--oracle-simulations", type=int, default=1000)
    p.add_argument("--astar-max-nodes", type=int, default=200_000)
    p.add_argument(
        "--astar-time-limit-seconds", type=float, default=0.0,
        help="per-level exact-search cap (0 = none); generated levels retain "
             "their reverse-construction solvability proof")
    p.add_argument("--rows", type=int, default=6)
    p.add_argument("--cols", type=int, default=6)
    p.add_argument("--color-count", type=int, default=3)
    p.add_argument("--density", type=float, default=0.5)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--residual-blocks", type=int, default=2)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--ppo-epochs", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--max-replay", type=int, default=5_000)
    p.add_argument("--validation-episodes", type=int, default=8)
    p.add_argument("--frontier-solve-rate-trials", type=int, default=1)
    p.add_argument("--frontier-min-solve-rate", type=float, default=0.2)
    p.add_argument("--frontier-max-solve-rate", type=float, default=0.7)
    p.add_argument("--frontier-alignment-weight", type=float, default=0.0)
    p.add_argument("--frontier-dirichlet-alpha", type=float, default=0.5)
    p.add_argument("--frontier-dirichlet-weight", type=float, default=0.4)
    p.add_argument("--frontier-budget-min-ratio", type=float, default=0.25)
    p.add_argument("--frontier-budget-max-ratio", type=float, default=4.0)
    p.add_argument("--frontier-min-simulations", type=int, default=20)
    p.add_argument("--frontier-max-simulations", type=int, default=400)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p


def config_from_args(args: argparse.Namespace) -> TrainConfig:
    gen = GeneratorConfig(rows=args.rows, cols=args.cols,
                          color_count=args.color_count, density=args.density)
    model = DesignerModelConfig(channels=args.channels,
                                residual_blocks=args.residual_blocks,
                                hidden_size=args.hidden_size)
    ppo = PPOConfig(epochs=args.ppo_epochs, learning_rate=args.learning_rate,
                    entropy_coef=args.entropy_coef)
    return TrainConfig(
        protagonist_checkpoint=args.protagonist_checkpoint,
        output_dir=args.output_dir, init_designer=args.init_designer,
        episodes=args.episodes, episodes_per_iter=args.episodes_per_iter,
        mutation_budget=args.mutation_budget,
        protagonist_simulations=args.protagonist_simulations,
        oracle_simulations=args.oracle_simulations,
        astar_max_nodes=args.astar_max_nodes,
        astar_time_limit_seconds=(
            None if args.astar_time_limit_seconds <= 0
            else args.astar_time_limit_seconds),
        seed=args.seed, device=args.device,
        max_replay=args.max_replay, validation_episodes=args.validation_episodes,
        frontier_solve_rate_trials=args.frontier_solve_rate_trials,
        frontier_min_solve_rate=args.frontier_min_solve_rate,
        frontier_max_solve_rate=args.frontier_max_solve_rate,
        frontier_alignment_weight=args.frontier_alignment_weight,
        frontier_dirichlet_alpha=args.frontier_dirichlet_alpha,
        frontier_dirichlet_weight=args.frontier_dirichlet_weight,
        frontier_budget_min_ratio=args.frontier_budget_min_ratio,
        frontier_budget_max_ratio=args.frontier_budget_max_ratio,
        frontier_min_simulations=args.frontier_min_simulations,
        frontier_max_simulations=args.frontier_max_simulations,
        generator=gen, model=model, ppo=ppo)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = train_designer(config_from_args(args))
    print(f"\nfinished {summary['iterations']} iteration(s); "
          f"{summary['episodes']} episodes; best_mean_reward="
          f"{summary['best_mean_reward']:.4f}; replay={summary['replay_size']}")


if __name__ == "__main__":
    main()
