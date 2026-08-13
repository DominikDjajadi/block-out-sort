"""Trace paired PUCT searches on levels whose audit outcomes disagreed.

The diagnostic is bound to a completed multi-candidate transfer audit.  It
loads the exact checkpoints, promotion-validation split, budgets, seed scheme,
and discordant level identities recorded by that audit, then records every
simulation path and root-edge update for one selected candidate.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..environment import Environment
from ..search.config import SearchConfig
from ..search.graph_search import BlocksortAdapter, GraphSearch
from ..search.seeding import derive_trial_seed, level_search_identity
from ..signature import static_level_signature
from ..training.checkpoint import (
    configs_from_checkpoint,
    load_checkpoint,
    model_from_checkpoint,
)
from ..training.transaction import atomic_write_json, sha256_file
from .transfer_audit import _load_level_list, _load_validation_levels


SCHEMA_VERSION = 1
SEMANTICS = "paired_discordant_puct_trace_v1"
AUDIT_SEMANTICS = frozenset((
    "multi_candidate_paired_transfer_audit_v2",
    "multi_candidate_paired_transfer_audit_v3",
))


@dataclass(frozen=True)
class SearchTraceConfig:
    audit_summary: str
    candidate_name: str
    output_dir: str
    group_name: str = "promotion_validation"
    device: str = "cuda"
    budgets: tuple[int, ...] = ()

    def validate(self) -> None:
        if not Path(self.audit_summary).is_file():
            raise ValueError(
                f"audit summary does not exist: {self.audit_summary}")
        if not self.candidate_name:
            raise ValueError("candidate_name must not be empty")
        if not self.group_name:
            raise ValueError("group_name must not be empty")
        if self.budgets:
            if len(self.budgets) != len(set(self.budgets)):
                raise ValueError("budgets must not contain duplicates")
            if any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value <= 0
                    for value in self.budgets):
                raise ValueError("budgets must contain positive integers")


def _load_audit_context(
    cfg: SearchTraceConfig,
) -> dict[str, Any]:
    summary_path = Path(cfg.audit_summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("semantics") not in AUDIT_SEMANTICS:
        raise ValueError(
            "search tracing requires a supported multi-candidate transfer "
            "audit")
    if summary.get("final_test_status") != "sealed_not_evaluated":
        raise ValueError("audit does not report a sealed final test")

    candidate = next(
        (
            item for item in summary.get("candidates", [])
            if item.get("name") == cfg.candidate_name
        ),
        None,
    )
    if candidate is None:
        raise ValueError(
            f"candidate is not present in audit: {cfg.candidate_name}")
    group = next(
        (
            item for item in candidate.get("groups", [])
            if item.get("group") == cfg.group_name
        ),
        None,
    )
    if group is None:
        raise ValueError(
            f"candidate audit has no {cfg.group_name!r} group")
    discordant = group.get("discordant_outcomes")
    if not isinstance(discordant, list) or not discordant:
        raise ValueError(
            "candidate has no discordant outcomes to trace in group "
            f"{cfg.group_name!r}")

    experiment_path = summary_path.with_name("experiment.json")
    if not experiment_path.is_file():
        raise ValueError(
            f"audit experiment metadata is missing: {experiment_path}")
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    if experiment.get("semantics") != summary.get("semantics"):
        raise ValueError("audit experiment semantics do not match summary")
    audit_config = experiment.get("config")
    if not isinstance(audit_config, dict):
        raise ValueError("audit experiment has no configuration")

    if cfg.group_name == "promotion_validation":
        audit_budgets = tuple(
            int(value) for value in audit_config["validation_budgets"])
        dataset = audit_config.get("eval_levels_dataset")
        manifest = audit_config.get("eval_split_manifest")
        for label, path in (
                ("evaluation dataset", dataset),
                ("evaluation split manifest", manifest)):
            if not path or not Path(path).is_file():
                raise ValueError(f"{label} does not exist: {path}")
        inputs = experiment.get("inputs", {})
        if sha256_file(dataset) != inputs.get("eval_levels_dataset_sha256"):
            raise RuntimeError(
                "evaluation dataset content differs from the source audit")
        if sha256_file(manifest) != inputs.get(
                "eval_split_manifest_sha256"):
            raise RuntimeError(
                "evaluation split manifest differs from the source audit")
        level_source = {
            "kind": "promotion_validation",
            "dataset": dataset,
            "dataset_sha256": sha256_file(dataset),
            "split_manifest": manifest,
            "split_manifest_sha256": sha256_file(manifest),
        }
    else:
        generated = next(
            (
                item for item in audit_config.get("generated_groups", [])
                if item.get("name") == cfg.group_name
            ),
            None,
        )
        if generated is None:
            raise ValueError(
                f"audit configuration has no generated group "
                f"{cfg.group_name!r}")
        dataset = generated.get("path")
        manifest = None
        if not dataset or not Path(dataset).is_file():
            raise ValueError(
                f"generated group file does not exist: {dataset}")
        expected_hash = experiment.get("inputs", {}).get(
            "generated_groups", {}).get(cfg.group_name)
        if sha256_file(dataset) != expected_hash:
            raise RuntimeError(
                "generated group content differs from the source audit")
        audit_budgets = tuple(
            int(value) for value in audit_config["generated_budgets"])
        level_source = {
            "kind": "generated_holdout",
            "path": dataset,
            "sha256": expected_hash,
        }
    selected_budgets = (
        cfg.budgets
        if cfg.budgets else tuple(sorted({
            int(item["budget"]) for item in discordant
        }))
    )
    missing = set(selected_budgets).difference(audit_budgets)
    if missing:
        raise ValueError(
            "trace budgets were not evaluated by the source audit: "
            f"{sorted(missing)}")
    selected_discordant = [
        item for item in discordant
        if int(item["budget"]) in selected_budgets
    ]
    if not selected_discordant:
        raise ValueError(
            "no discordant outcomes remain at the selected budgets")

    incumbent_path = str(summary["incumbent_checkpoint"])
    candidate_path = str(candidate["checkpoint"])
    for label, path, expected_hash in (
            ("incumbent", incumbent_path,
             summary["incumbent_checkpoint_sha256"]),
            ("candidate", candidate_path, candidate["checkpoint_sha256"])):
        if not Path(path).is_file():
            raise ValueError(f"{label} checkpoint does not exist: {path}")
        if sha256_file(path) != expected_hash:
            raise RuntimeError(
                f"{label} checkpoint content differs from the source audit")

    return {
        "summary": summary,
        "candidate": candidate,
        "group": group,
        "experiment": experiment,
        "audit_config": audit_config,
        "incumbent_checkpoint": incumbent_path,
        "candidate_checkpoint": candidate_path,
        "dataset": dataset,
        "manifest": manifest,
        "audit_budgets": audit_budgets,
        "selected_budgets": selected_budgets,
        "discordant": selected_discordant,
        "level_source": level_source,
    }


def _root_selection_scores(root, config: SearchConfig) -> list[float]:
    # Retained for compatibility with existing diagnostics/tests.
    class _Search:
        pass
    search = _Search()
    search.config = config
    return GraphSearch._root_selection_scores(search, root)


def _run_traced_search(
    adapter,
    state,
    *,
    config: SearchConfig,
) -> dict[str, Any]:
    """Run one search while capturing every simulation path."""
    search = GraphSearch(adapter, config)
    search._validate_config()
    search._reset_per_run_state(None)
    root_key = adapter.key(state)
    root, created = search.table.get_or_create(root_key, state)
    if not created:
        raise RuntimeError("fresh search unexpectedly reused its root")
    search._expand(root, state)

    root_edges = [
        {
            "index": index,
            "locator": copy.deepcopy(root.locators[index]),
            "prior": root.priors[index],
        }
        for index in range(root.num_actions)
    ]
    timeline = []
    batch_evaluator = getattr(
        adapter, "evaluate_batch_with_legal_actions", None)
    if config.inference_batch_size > 1 and callable(batch_evaluator):
        remaining = config.simulations
        completed_total = 0
        while remaining > 0:
            batch_rows: list[dict[str, Any]] = []
            completed = search._simulate_batch(
                root,
                state,
                min(config.inference_batch_size, remaining),
                batch_evaluator,
                trace_rows=batch_rows,
            )
            if completed <= 0:
                raise RuntimeError(
                    "batched trace made no simulation progress")
            batch_end = completed_total + completed
            for offset, detail in enumerate(batch_rows, start=1):
                timeline.append({
                    "simulation": completed_total + offset,
                    "batch_end_simulation": batch_end,
                    **detail,
                })
            completed_total += completed
            remaining -= completed
    else:
        for simulation in range(1, config.simulations + 1):
            visits_before = list(root.N)
            search.stats.simulations += 1
            detail = search._simulate(root, state, trace=True)
            assert detail is not None
            changed = [
                index for index, (before, after)
                in enumerate(zip(visits_before, root.N))
                if after != before
            ]
            if len(changed) > 1:
                raise RuntimeError(
                    "one simulation changed more than one root edge")
            selected_root_edge = changed[0] if changed else None
            if selected_root_edge != detail["root_edge_index"]:
                raise RuntimeError(
                    "simulation trace root edge disagrees with visit counts")
            timeline.append({
                "simulation": simulation,
                "batch_end_simulation": simulation,
                **detail,
                "root_visit_counts": list(root.N),
                "root_action_q_costs": list(root.Q),
                "next_root_selection_scores":
                    _root_selection_scores(root, config),
                "principal_variation": copy.deepcopy(
                    search._principal_variation(root, state)),
            })

    search.stats.unique_states = len(search.table)
    search.stats.transposition_hits = search.table.hits
    result = search._build_result(root, state)
    return {
        "seed": config.seed,
        "inference_batch_size": config.inference_batch_size,
        "root_value_cost_model": result.root_value_cost_model,
        "root_edges": root_edges,
        "timeline": timeline,
        "final": {
            "solved": bool(result.solved),
            "solution_length": result.solution_length,
            "solution_verified": bool(result.solution_verified),
            "termination_reason": result.termination_reason,
            "search_value_cost": result.search_value_cost,
            "visit_counts": result.visit_counts,
            "action_q_costs": result.action_q_cost,
            "chosen_action_locator": result.chosen_action_locator,
            "principal_variation": result.principal_variation,
            "stats": {
                "simulations": result.stats.simulations,
                "nodes_expanded": result.stats.nodes_expanded,
                "unique_states": result.stats.unique_states,
                "transposition_hits": result.stats.transposition_hits,
                "cycle_rejections": result.stats.cycle_rejections,
                "deadlocks": result.stats.deadlocks,
            },
        },
    }


def _common_prefix(left: list[Any], right: list[Any]) -> int:
    count = 0
    for left_item, right_item in zip(left, right):
        if left_item != right_item:
            break
        count += 1
    return count


def _first_solution_simulation(trace: dict[str, Any]) -> int | None:
    for item in trace["timeline"]:
        if item["best_solution_length"] is not None:
            return int(item["simulation"])
    return None


def _paired_report(
    incumbent: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    incumbent_edges = incumbent["root_edges"]
    candidate_edges = candidate["root_edges"]
    incumbent_locators = [item["locator"] for item in incumbent_edges]
    candidate_locators = [item["locator"] for item in candidate_edges]
    if incumbent_locators != candidate_locators:
        raise RuntimeError(
            "root legal actions differ between paired searches")
    prior_delta = max(
        (
            abs(left["prior"] - right["prior"])
            for left, right in zip(incumbent_edges, candidate_edges)
        ),
        default=0.0,
    )

    simulation_rows = []
    for left, right in zip(
            incumbent["timeline"], candidate["timeline"]):
        if left["simulation"] != right["simulation"]:
            raise RuntimeError("paired simulation indices differ")
        simulation_rows.append({
            "simulation": left["simulation"],
            "same_root_edge":
                left["root_edge_index"] == right["root_edge_index"],
            "same_full_path":
                left["path_locators"] == right["path_locators"],
            "path_common_prefix_length": _common_prefix(
                left["path_locators"], right["path_locators"]),
            "incumbent_root_edge_index": left["root_edge_index"],
            "candidate_root_edge_index": right["root_edge_index"],
            "incumbent_leaf_reason": left["leaf_reason"],
            "candidate_leaf_reason": right["leaf_reason"],
            "incumbent_leaf_cost": left["leaf_cost"],
            "candidate_leaf_cost": right["leaf_cost"],
        })
    root_divergences = [
        row["simulation"] for row in simulation_rows
        if not row["same_root_edge"]
    ]
    path_divergences = [
        row["simulation"] for row in simulation_rows
        if not row["same_full_path"]
    ]
    final_visit_l1 = sum(
        abs(left - right)
        for left, right in zip(
            incumbent["final"]["visit_counts"],
            candidate["final"]["visit_counts"],
        )
    )
    return {
        "root_value_cost_delta": (
            candidate["root_value_cost_model"]
            - incumbent["root_value_cost_model"]
        ),
        "max_root_prior_absolute_delta": prior_delta,
        "first_root_edge_divergence_simulation":
            root_divergences[0] if root_divergences else None,
        "root_edge_divergence_count": len(root_divergences),
        "first_full_path_divergence_simulation":
            path_divergences[0] if path_divergences else None,
        "full_path_divergence_count": len(path_divergences),
        "incumbent_first_solution_simulation":
            _first_solution_simulation(incumbent),
        "candidate_first_solution_simulation":
            _first_solution_simulation(candidate),
        "final_root_visit_l1_distance": final_visit_l1,
        "simulation_comparison": simulation_rows,
    }


def _load_models(
    context: dict[str, Any],
    *,
    device: torch.device,
) -> tuple[dict[str, Any], Any, Any]:
    checkpoints = {
        "incumbent": context["incumbent_checkpoint"],
        "candidate": context["candidate_checkpoint"],
    }
    loaded = {
        role: load_checkpoint(path, map_location="cpu")
        for role, path in checkpoints.items()
    }
    configs = {
        role: configs_from_checkpoint(checkpoint)
        for role, checkpoint in loaded.items()
    }
    reference = tuple(
        config.to_dict() for config in configs["incumbent"])
    if tuple(
            config.to_dict() for config in configs["candidate"]) != reference:
        raise ValueError(
            "candidate checkpoint configuration differs from incumbent")
    encoding, _model_config, value_norm = configs["incumbent"]
    models = {
        role: model_from_checkpoint(checkpoint, map_location=device)
        for role, checkpoint in loaded.items()
    }
    return models, encoding, value_norm


def _expected_discordance(
    context: dict[str, Any],
) -> dict[tuple[str, int], dict[str, Any]]:
    return {
        (str(item["static_level_signature"]), int(item["budget"])): item
        for item in context["discordant"]
    }


def run_search_trace(cfg: SearchTraceConfig) -> dict[str, Any]:
    cfg.validate()
    context = _load_audit_context(cfg)
    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but this Python environment has no "
            "CUDA-enabled PyTorch device")
    device = torch.device(cfg.device)
    levels = (
        _load_validation_levels(context["dataset"], context["manifest"])
        if context["manifest"] is not None
        else _load_level_list(context["dataset"])
    )
    by_signature = {
        static_level_signature(level): level for level in levels}
    wanted_signatures = sorted({
        str(item["static_level_signature"])
        for item in context["discordant"]
    })
    missing = set(wanted_signatures).difference(by_signature)
    if missing:
        raise RuntimeError(
            "discordant audit levels are absent from the selected group: "
            f"{sorted(missing)}")

    models, encoding, value_norm = _load_models(
        context, device=device)
    env = Environment()
    adapters = {
        role: BlocksortAdapter(
            env, model, encoding, value_norm, device)
        for role, model in models.items()
    }
    expected = _expected_discordance(context)
    audit_budgets = context["audit_budgets"]
    audit_seed = int(context["audit_config"]["seed"])
    c_puct = float(context["audit_config"]["c_puct"])

    reports = []
    for level_index, signature in enumerate(wanted_signatures, start=1):
        level = by_signature[signature]
        state = env.initial_state(level)
        identity = level_search_identity(env, level)
        budget_reports = {}
        for budget in context["selected_budgets"]:
            trial_index = audit_budgets.index(budget)
            trial_seed = derive_trial_seed(
                audit_seed,
                trial_index=trial_index,
                level_identity=identity,
                evaluation_context=
                    f"transfer_audit.{cfg.group_name}",
            )
            traces = {}
            for role, adapter in adapters.items():
                trace_config = SearchConfig(
                    simulations=budget,
                    c_puct=c_puct,
                    temperature=0.0,
                    value_normalization_constant=getattr(
                        value_norm, "constant", 20.0),
                    seed=trial_seed,
                )
                traces[role] = _run_traced_search(
                    adapter, state, config=trace_config)
            pair = _paired_report(
                traces["incumbent"], traces["candidate"])
            expected_outcome = expected.get((signature, budget))
            if expected_outcome is not None:
                if (
                    traces["incumbent"]["final"]["solved"]
                    != bool(expected_outcome["incumbent_solved"])
                    or traces["candidate"]["final"]["solved"]
                    != bool(expected_outcome["candidate_solved"])
                ):
                    raise RuntimeError(
                        "trace did not reproduce source audit outcome for "
                        f"{signature} at budget {budget}")
            budget_reports[str(budget)] = {
                "source_audit_discordance": expected_outcome,
                "incumbent": traces["incumbent"],
                "candidate": traces["candidate"],
                "paired": pair,
            }
        reports.append({
            "name": level.name,
            "static_level_signature": signature,
            "search_identity": identity,
            "rows": level.rows,
            "cols": level.cols,
            "budgets": budget_reports,
        })
        print(
            f"traced {level_index}/{len(wanted_signatures)} "
            f"discordant levels",
            flush=True,
        )

    result = {
        "schema_version": SCHEMA_VERSION,
        "semantics": SEMANTICS,
        "config": json.loads(json.dumps(asdict(cfg))),
        "source_audit": {
            "path": cfg.audit_summary,
            "sha256": sha256_file(cfg.audit_summary),
            "candidate_name": cfg.candidate_name,
            "semantics": context["summary"]["semantics"],
        },
        "inputs": {
            "incumbent_checkpoint":
                context["incumbent_checkpoint"],
            "incumbent_checkpoint_sha256":
                context["summary"]["incumbent_checkpoint_sha256"],
            "candidate_checkpoint":
                context["candidate_checkpoint"],
            "candidate_checkpoint_sha256":
                context["candidate"]["checkpoint_sha256"],
            "group_name": cfg.group_name,
            "level_source": context["level_source"],
        },
        "device": str(device),
        "budgets": list(context["selected_budgets"]),
        "levels": reports,
        "source_audit_reproduction": "passed",
        "final_test_status": "sealed_not_evaluated",
    }
    output = Path(cfg.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output / "summary.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Trace incumbent/candidate PUCT simulations on levels in an "
            "audited group whose saved outcomes disagree."))
    parser.add_argument("--audit-summary", required=True)
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--group-name", default="promotion_validation")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--budgets", type=int, nargs="*", default=[])
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_search_trace(SearchTraceConfig(
        audit_summary=args.audit_summary,
        candidate_name=args.candidate_name,
        output_dir=args.output_dir,
        group_name=args.group_name,
        device=args.device,
        budgets=tuple(args.budgets),
    ))
    print(
        f"traced {len(result['levels'])} levels at budgets "
        f"{result['budgets']}")
    print("source audit reproduction: passed")
    print("final test: sealed and not evaluated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
