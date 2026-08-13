"""Paired per-level audit of protagonist learning transfer.

This diagnostic deliberately uses only generated training levels and the
promotion-validation split.  It never accepts a final-test role.  Each
checkpoint/group evaluation is committed separately so a long CPU run can
resume without repeating completed work.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..environment import Environment
from ..expert_iteration.budget_sweep import select_evaluation_records
from ..schema import Level
from ..serialization import level_from_dict
from ..signature import static_level_signature
from ..training.transaction import (
    atomic_write_json, atomic_write_text, sha256_file)


_SCHEMA_VERSION = 2
_SEMANTICS = "paired_deterministic_graph_search_transfer_v2"
_GROUP_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


@dataclass(frozen=True)
class GeneratedGroup:
    name: str
    path: str

    def validate(self) -> None:
        if not _GROUP_NAME.fullmatch(self.name):
            raise ValueError(
                "generated group names must contain only letters, numbers, "
                "underscores, and hyphens")
        if self.name == "promotion_validation":
            raise ValueError(
                "promotion_validation is reserved for the immutable split")
        if not Path(self.path).is_file():
            raise ValueError(f"generated level file does not exist: {self.path}")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "path": self.path}


@dataclass(frozen=True)
class TransferAuditConfig:
    incumbent_checkpoint: str
    candidate_checkpoint: str
    generated_groups: tuple[GeneratedGroup, ...]
    eval_levels_dataset: str
    eval_split_manifest: str
    output_dir: str
    generated_budgets: tuple[int, ...] = (55, 90, 148, 244, 400)
    validation_budgets: tuple[int, ...] = (4, 8, 16, 32, 64)
    c_puct: float = 1.5
    seed: int = 2045
    device: str = "cpu"
    progress_interval: int = 5

    def validate(self) -> None:
        for label, value in (
                ("incumbent checkpoint", self.incumbent_checkpoint),
                ("candidate checkpoint", self.candidate_checkpoint),
                ("evaluation dataset", self.eval_levels_dataset),
                ("evaluation split manifest", self.eval_split_manifest)):
            if not Path(value).is_file():
                raise ValueError(f"{label} does not exist: {value}")
        if not self.generated_groups:
            raise ValueError("at least one generated group is required")
        names: set[str] = set()
        for group in self.generated_groups:
            group.validate()
            if group.name in names:
                raise ValueError(f"duplicate generated group name: {group.name}")
            names.add(group.name)
        _validate_budgets("generated_budgets", self.generated_budgets)
        _validate_budgets("validation_budgets", self.validation_budgets)
        if (not math.isfinite(self.c_puct) or self.c_puct < 0):
            raise ValueError("c_puct must be finite and non-negative")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise ValueError("seed must be an integer")
        if (isinstance(self.progress_interval, bool)
                or not isinstance(self.progress_interval, int)
                or self.progress_interval <= 0):
            raise ValueError("progress_interval must be a positive integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "incumbent_checkpoint": self.incumbent_checkpoint,
            "candidate_checkpoint": self.candidate_checkpoint,
            "generated_groups": [
                group.to_dict() for group in self.generated_groups],
            "eval_levels_dataset": self.eval_levels_dataset,
            "eval_split_manifest": self.eval_split_manifest,
            "output_dir": self.output_dir,
            "generated_budgets": list(self.generated_budgets),
            "validation_budgets": list(self.validation_budgets),
            "c_puct": self.c_puct,
            "seed": self.seed,
            "device": self.device,
            "progress_interval": self.progress_interval,
        }


def _validate_budgets(label: str, budgets: tuple[int, ...]) -> None:
    if not budgets:
        raise ValueError(f"{label} must not be empty")
    if len(set(budgets)) != len(budgets):
        raise ValueError(f"{label} must not contain duplicates")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in budgets):
        raise ValueError(f"{label} must contain positive integers")


def _fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_generated_group(value: str) -> GeneratedGroup:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError(
            "generated groups must use NAME=PATH syntax")
    return GeneratedGroup(name=name, path=path)


def _load_level_list(path: str | Path) -> list[Level]:
    from ..search.seeding import level_search_identity

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        try:
            decoded = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"generated level file is not valid JSON or JSONL: {path}"
            ) from exc
    raw = decoded if isinstance(decoded, list) else [decoded]
    if not raw:
        raise ValueError(
            f"generated level file must contain at least one level: {path}")
    levels = []
    identities: set[str] = set()
    env = Environment()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(
                f"generated level file {path} item {index} is not an object")
        level_data = item
        if "hard_eval_pool_schema_version" in item:
            level_data = item.get("level")
            if not isinstance(level_data, dict):
                raise ValueError(
                    f"generated level file {path} item {index} has no level")
        level = level_from_dict(level_data)
        identity = level_search_identity(env, level)
        if identity in identities:
            raise ValueError(
                f"generated level group contains a duplicate level: {identity}")
        identities.add(identity)
        levels.append(level)
    return levels


def _load_validation_levels(
    dataset: str | Path,
    manifest: str | Path,
) -> list[Level]:
    records, split = select_evaluation_records(
        dataset, split_manifest_path=manifest,
        split_role="promotion_validation")
    if not split or split.get("role") != "promotion_validation":
        raise RuntimeError("transfer audit must use promotion_validation")
    return [level_from_dict(record["level"]) for record in records]


def _identity(cfg: TransferAuditConfig) -> dict[str, Any]:
    result = {
        "schema_version": _SCHEMA_VERSION,
        "semantics": _SEMANTICS,
        "config": cfg.to_dict(),
        "inputs": {
            "incumbent_checkpoint_sha256":
                sha256_file(cfg.incumbent_checkpoint),
            "candidate_checkpoint_sha256":
                sha256_file(cfg.candidate_checkpoint),
            "eval_levels_dataset_sha256":
                sha256_file(cfg.eval_levels_dataset),
            "eval_split_manifest_sha256":
                sha256_file(cfg.eval_split_manifest),
            "generated_groups": {
                group.name: sha256_file(group.path)
                for group in cfg.generated_groups
            },
        },
    }
    result["fingerprint"] = _fingerprint(result)
    return result


def _evaluate_checkpoint_group(
    checkpoint_path: str | Path,
    levels: list[Level],
    *,
    group_name: str,
    budgets: tuple[int, ...],
    c_puct: float,
    seed: int,
    device: str,
    progress_interval: int,
    initial_rows: list[dict[str, Any]] | None = None,
    checkpoint_rows=None,
) -> list[dict[str, Any]]:
    import torch

    from ..search.config import SearchConfig
    from ..search.graph_search import BlocksortAdapter, GraphSearch
    from ..search.seeding import derive_trial_seed, level_search_identity
    from ..training.checkpoint import (
        configs_from_checkpoint, load_checkpoint, model_from_checkpoint)

    env = Environment()
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    encoding, _model_cfg, value_norm = configs_from_checkpoint(checkpoint)
    torch_device = torch.device(device)
    model = model_from_checkpoint(checkpoint, map_location=torch_device)
    model.eval()
    adapter = BlocksortAdapter(
        env, model, encoding, value_norm, torch_device)

    rows: list[dict[str, Any]] = list(initial_rows or [])
    total = len(levels)
    for index in range(len(rows), total):
        level = levels[index]
        state = env.initial_state(level)
        identity = level_search_identity(env, level)
        budget_results: dict[str, Any] = {}
        for trial_index, budget in enumerate(budgets):
            trial_seed = derive_trial_seed(
                seed, trial_index=trial_index, level_identity=identity,
                evaluation_context=f"transfer_audit.{group_name}")
            search_cfg = SearchConfig(
                simulations=budget,
                c_puct=c_puct,
                temperature=0.0,
                value_normalization_constant=getattr(
                    value_norm, "constant", 20.0),
                seed=trial_seed,
            )
            result = GraphSearch(adapter, search_cfg).run(state)
            budget_results[str(budget)] = {
                "solved": bool(result.solved),
                "solution_length": result.solution_length,
                "solution_verified": bool(result.solution_verified),
                "termination_reason": result.termination_reason,
                "root_value_cost_model": result.root_value_cost_model,
                "search_value_cost": result.search_value_cost,
                "stats": {
                    "simulations": result.stats.simulations,
                    "nodes_expanded": result.stats.nodes_expanded,
                    "unique_states": result.stats.unique_states,
                    "transposition_hits": result.stats.transposition_hits,
                    "cycle_rejections": result.stats.cycle_rejections,
                    "deadlocks": result.stats.deadlocks,
                },
            }
        rows.append({
            "index": index,
            "name": level.name,
            "static_level_signature": static_level_signature(level),
            "search_identity": identity,
            "rows": level.rows,
            "cols": level.cols,
            "budgets": budget_results,
        })
        if checkpoint_rows is not None:
            checkpoint_rows(rows)
        if (index + 1) % progress_interval == 0 or index + 1 == total:
            print(
                f"{group_name}: {Path(checkpoint_path).name}: "
                f"{index + 1}/{total} levels",
                flush=True,
            )
    return rows


def _validate_partial_rows(
    path: Path,
    result: dict[str, Any],
    *,
    expected_header: dict[str, Any],
    levels: list[Level],
    budgets: tuple[int, ...],
) -> list[dict[str, Any]]:
    from ..search.seeding import level_search_identity

    header = {key: result.get(key) for key in expected_header}
    if header != expected_header:
        raise RuntimeError(
            f"incompatible partial transfer evaluation: {path}")
    rows = result.get("levels")
    if not isinstance(rows, list) or len(rows) > len(levels):
        raise RuntimeError(
            f"invalid partial transfer evaluation rows: {path}")
    env = Environment()
    expected_budgets = {str(budget) for budget in budgets}
    for index, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("index") != index:
            raise RuntimeError(
                f"partial transfer evaluation is not a contiguous prefix: "
                f"{path}")
        expected_identity = level_search_identity(env, levels[index])
        if row.get("search_identity") != expected_identity:
            raise RuntimeError(
                f"partial transfer evaluation level mismatch at index "
                f"{index}: {path}")
        row_budgets = row.get("budgets")
        if not isinstance(row_budgets, dict) \
                or set(row_budgets) != expected_budgets:
            raise RuntimeError(
                f"partial transfer evaluation budget mismatch at index "
                f"{index}: {path}")
    return rows


def _paired_group_summary(
    group_name: str,
    budgets: tuple[int, ...],
    incumbent: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
) -> dict[str, Any]:
    incumbent_by_id = {
        row["search_identity"]: row for row in incumbent}
    candidate_by_id = {
        row["search_identity"]: row for row in candidate}
    if set(incumbent_by_id) != set(candidate_by_id):
        raise ValueError(
            f"checkpoint evaluations contain different levels for {group_name}")

    per_budget: dict[str, Any] = {}
    discordant: list[dict[str, Any]] = []
    for budget in budgets:
        counts = {
            "both_solved": 0,
            "incumbent_only": 0,
            "candidate_only": 0,
            "neither_solved": 0,
        }
        length_delta_sum = 0
        length_delta_count = 0
        for identity in sorted(incumbent_by_id):
            inc_row = incumbent_by_id[identity]
            cand_row = candidate_by_id[identity]
            inc = inc_row["budgets"][str(budget)]
            cand = cand_row["budgets"][str(budget)]
            inc_solved = bool(inc["solved"])
            cand_solved = bool(cand["solved"])
            if inc_solved and cand_solved:
                counts["both_solved"] += 1
                if (inc["solution_length"] is not None
                        and cand["solution_length"] is not None):
                    length_delta_sum += (
                        cand["solution_length"] - inc["solution_length"])
                    length_delta_count += 1
            elif inc_solved:
                counts["incumbent_only"] += 1
            elif cand_solved:
                counts["candidate_only"] += 1
            else:
                counts["neither_solved"] += 1
            if inc_solved != cand_solved:
                discordant.append({
                    "budget": budget,
                    "search_identity": identity,
                    "static_level_signature":
                        inc_row["static_level_signature"],
                    "name": inc_row["name"],
                    "incumbent_solved": inc_solved,
                    "candidate_solved": cand_solved,
                })
        total = len(incumbent_by_id)
        incumbent_solved = (
            counts["both_solved"] + counts["incumbent_only"])
        candidate_solved = (
            counts["both_solved"] + counts["candidate_only"])
        per_budget[str(budget)] = {
            "levels": total,
            **counts,
            "incumbent_solved": incumbent_solved,
            "candidate_solved": candidate_solved,
            "incumbent_solve_rate":
                incumbent_solved / total if total else None,
            "candidate_solve_rate":
                candidate_solved / total if total else None,
            "solve_rate_delta":
                (candidate_solved - incumbent_solved) / total
                if total else None,
            "mean_solution_length_delta_on_both_solved":
                length_delta_sum / length_delta_count
                if length_delta_count else None,
        }
    return {
        "group": group_name,
        "levels": len(incumbent_by_id),
        "budgets": list(budgets),
        "per_budget": per_budget,
        "discordant_outcomes": discordant,
    }


def _write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    handle = io.StringIO()
    writer = csv.DictWriter(handle, fieldnames=[
        "group", "budget", "levels", "incumbent_solved",
        "candidate_solved", "candidate_only", "incumbent_only",
        "both_solved", "neither_solved", "solve_rate_delta",
        "mean_solution_length_delta_on_both_solved",
    ])
    writer.writeheader()
    for summary in summaries:
        for budget in summary["budgets"]:
            item = summary["per_budget"][str(budget)]
            writer.writerow({
                "group": summary["group"],
                "budget": budget,
                **{key: item[key] for key in writer.fieldnames
                   if key not in ("group", "budget")},
            })
    atomic_write_text(path, handle.getvalue())


def run_transfer_audit(cfg: TransferAuditConfig) -> dict[str, Any]:
    cfg.validate()
    root = Path(cfg.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    identity = _identity(cfg)
    identity_path = root / "experiment.json"
    if identity_path.exists():
        persisted = json.loads(identity_path.read_text(encoding="utf-8"))
        if persisted != identity:
            raise RuntimeError(
                "transfer-audit output directory belongs to different inputs "
                "or settings")
    else:
        atomic_write_json(identity_path, identity)

    groups: list[tuple[str, list[Level], tuple[int, ...], dict[str, Any]]] = []
    for group in cfg.generated_groups:
        groups.append((
            group.name,
            _load_level_list(group.path),
            cfg.generated_budgets,
            {"kind": "generated_training", "path": group.path,
             "sha256": sha256_file(group.path)},
        ))
    groups.append((
        "promotion_validation",
        _load_validation_levels(
            cfg.eval_levels_dataset, cfg.eval_split_manifest),
        cfg.validation_budgets,
        {
            "kind": "promotion_validation",
            "dataset": cfg.eval_levels_dataset,
            "dataset_sha256": sha256_file(cfg.eval_levels_dataset),
            "split_manifest": cfg.eval_split_manifest,
            "split_manifest_sha256":
                sha256_file(cfg.eval_split_manifest),
        },
    ))

    evaluation_dir = root / "evaluations"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    for group_name, levels, budgets, source in groups:
        role_results: dict[str, list[dict[str, Any]]] = {}
        for role, checkpoint in (
                ("incumbent", cfg.incumbent_checkpoint),
                ("candidate", cfg.candidate_checkpoint)):
            path = evaluation_dir / f"{group_name}.{role}.json"
            partial_path = path.with_name(path.name + ".partial.json")
            expected_header = {
                "schema_version": _SCHEMA_VERSION,
                "semantics": _SEMANTICS,
                "experiment_fingerprint": identity["fingerprint"],
                "group": group_name,
                "role": role,
                "checkpoint": checkpoint,
                "checkpoint_sha256": sha256_file(checkpoint),
                "source": source,
                "budgets": list(budgets),
                "seed": cfg.seed,
                "c_puct": cfg.c_puct,
            }
            if path.exists():
                result = json.loads(path.read_text(encoding="utf-8"))
                header = {key: result.get(key) for key in expected_header}
                if header != expected_header:
                    raise RuntimeError(
                        f"incompatible cached transfer evaluation: {path}")
                if partial_path.exists():
                    partial_path.unlink()
                print(f"reusing {group_name} {role} evaluation", flush=True)
            else:
                initial_rows: list[dict[str, Any]] = []
                if partial_path.exists():
                    partial = json.loads(
                        partial_path.read_text(encoding="utf-8"))
                    initial_rows = _validate_partial_rows(
                        partial_path, partial,
                        expected_header=expected_header,
                        levels=levels,
                        budgets=budgets,
                    )
                    print(
                        f"resuming {group_name} {role} evaluation at "
                        f"{len(initial_rows)}/{len(levels)} levels",
                        flush=True,
                    )
                else:
                    print(
                        f"evaluating {group_name} with {role}", flush=True)

                def checkpoint_rows(rows):
                    atomic_write_json(
                        partial_path, {**expected_header, "levels": rows})

                rows = _evaluate_checkpoint_group(
                    checkpoint, levels, group_name=group_name,
                    budgets=budgets, c_puct=cfg.c_puct, seed=cfg.seed,
                    device=cfg.device,
                    progress_interval=cfg.progress_interval,
                    initial_rows=initial_rows,
                    checkpoint_rows=checkpoint_rows)
                result = {**expected_header, "levels": rows}
                atomic_write_json(path, result)
                if partial_path.exists():
                    partial_path.unlink()
            role_results[role] = result["levels"]
        summaries.append(_paired_group_summary(
            group_name, budgets, role_results["incumbent"],
            role_results["candidate"]))

    result = {
        "schema_version": _SCHEMA_VERSION,
        "semantics": _SEMANTICS,
        "experiment_fingerprint": identity["fingerprint"],
        "incumbent_checkpoint": cfg.incumbent_checkpoint,
        "incumbent_checkpoint_sha256":
            identity["inputs"]["incumbent_checkpoint_sha256"],
        "candidate_checkpoint": cfg.candidate_checkpoint,
        "candidate_checkpoint_sha256":
            identity["inputs"]["candidate_checkpoint_sha256"],
        "groups": summaries,
        "final_test_status": "sealed_not_evaluated",
    }
    atomic_write_json(root / "summary.json", result)
    _write_summary_csv(root / "summary.csv", summaries)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare incumbent and candidate transfer on generated levels and "
            "promotion-validation without evaluating final-test levels."))
    parser.add_argument("--incumbent-checkpoint", required=True)
    parser.add_argument("--candidate-checkpoint", required=True)
    parser.add_argument(
        "--generated-group", action="append", type=_parse_generated_group,
        required=True, metavar="NAME=PATH")
    parser.add_argument("--eval-levels-dataset", required=True)
    parser.add_argument("--eval-split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--generated-budgets", type=int, nargs="+",
        default=[55, 90, 148, 244, 400])
    parser.add_argument(
        "--validation-budgets", type=int, nargs="+",
        default=[4, 8, 16, 32, 64])
    parser.add_argument("--c-puct", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=2045)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--progress-interval", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = TransferAuditConfig(
        incumbent_checkpoint=args.incumbent_checkpoint,
        candidate_checkpoint=args.candidate_checkpoint,
        generated_groups=tuple(args.generated_group),
        eval_levels_dataset=args.eval_levels_dataset,
        eval_split_manifest=args.eval_split_manifest,
        output_dir=args.output_dir,
        generated_budgets=tuple(args.generated_budgets),
        validation_budgets=tuple(args.validation_budgets),
        c_puct=args.c_puct,
        seed=args.seed,
        device=args.device,
        progress_interval=args.progress_interval,
    )
    result = run_transfer_audit(cfg)
    for group in result["groups"]:
        print(f"=== {group['group']} ===")
        for budget in group["budgets"]:
            item = group["per_budget"][str(budget)]
            print(
                f"budget {budget}: incumbent={item['incumbent_solved']}/"
                f"{item['levels']}, candidate={item['candidate_solved']}/"
                f"{item['levels']}, delta={item['solve_rate_delta']:+.3f}, "
                f"candidate_only={item['candidate_only']}, "
                f"incumbent_only={item['incumbent_only']}")
    print("final test: sealed and not evaluated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
