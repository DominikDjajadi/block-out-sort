"""Orchestrate the final-benchmark experiment under ``runs/final_benchmark/``.

Phases (run selectively with ``--phases``):

    benchmark   build the harder frozen benchmark + held-out promotion levels;
                run the saturation check on the initial protagonist
    cotrain     longer co-training using the harder held-out promotion levels +
                an extended curriculum that can lower difficulty
    solver      solver comparison on identical frozen (harder) states
    generators  generator comparison over multiple seeds
    ablations   compact reward-term + co-training ablations
    report      assemble summary.json + report.md from the artifacts

Each phase writes machine-readable JSON so phases can be run independently and
resumed. Defaults point at the repository's existing checkpoints.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from ..environment import Environment
from ..serialization import level_from_dict
from ..designer.config import GeneratorConfig
from ..designer.model import DesignerModelConfig
from ..cotraining.config import (CoTrainingConfig, CurriculumConfig,
                                 CurriculumState)
from ..cotraining.loop import run_cotraining
from ..training.experiment_identity import (
    EVALUATION_SEMANTICS_VERSION, PROMOTION_CONTRACT_VERSION,
    TRANSACTION_SCHEMA_VERSION, ExperimentSpecIntegrityError,
    build_experiment_spec, file_identity, hash_canonical_value,
    hash_file_streaming, runtime_device_provenance,
    validate_field_classification,
    validate_or_initialize_experiment)
from ..training.transaction import atomic_write_json
from .common import (
    Protagonist, ValidatedCheckpointSource, checkpoint_identity,
    resolve_device, validate_checkpoint_source)
from . import harder, solver as solver_mod, generators as gen_mod, ablations, report


DEFAULTS = {
    "supervised": "runs/pv_smoke/best.pt",
    "expert_iteration": "runs/expert_iteration/best.pt",
    "adversarial_designer": "runs/designer/best.pt",
    "pretrained_designer": "runs/designer_pretrain/best.pt",
    "base_dataset": "../data/training/pv_smoke.jsonl",
    "handcrafted_dataset": "../data/training/pv_examples.jsonl",
}


def _write(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


_INTERNAL_SOURCES_FILE = "internal_checkpoint_sources.json"


def _internal_source_record(
    source: ValidatedCheckpointSource, *, origin: str,
) -> dict[str, Any]:
    return {**source.to_dict(), "benchmark_source_origin": origin}


def _read_internal_source_records(root: Path) -> dict[str, dict[str, Any]]:
    path = root / _INTERNAL_SOURCES_FILE
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExperimentSpecIntegrityError(
            f"cannot validate benchmark internal source identity: {path}") from exc
    if (not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or not isinstance(payload.get("sources"), dict)):
        raise ExperimentSpecIntegrityError(
            f"malformed benchmark internal source identity: {path}")
    expected_identity = hash_canonical_value({
        "schema_version": 1,
        "sources": payload["sources"],
    })
    if payload.get("source_identity_sha256") != expected_identity:
        raise ExperimentSpecIntegrityError(
            f"benchmark internal source identity hash mismatch: {path}")
    records = payload["sources"]
    if any(
            not isinstance(name, str) or not isinstance(record, dict)
            for name, record in records.items()):
        raise ExperimentSpecIntegrityError(
            f"malformed benchmark internal source identity: {path}")
    return records


def _validate_pinned_internal_sources(
    root: Path,
) -> dict[str, dict[str, Any]]:
    records = _read_internal_source_records(root)
    for name, expected in records.items():
        checkpoint = expected.get("checkpoint_path") or expected.get("path")
        if not isinstance(checkpoint, str):
            raise ExperimentSpecIntegrityError(
                f"internal benchmark source {name!r} has no checkpoint path")
        try:
            observed_source = validate_checkpoint_source(checkpoint)
        except (FileNotFoundError, ExperimentSpecIntegrityError) as exc:
            raise ExperimentSpecIntegrityError(
                "Cannot validate internally produced benchmark source "
                f"{name!r}: {exc}") from exc
        observed = _internal_source_record(
            observed_source,
            origin=str(expected.get("benchmark_source_origin")
                       or "internal_phase"),
        )
        if observed != expected:
            differences = sorted(
                key for key in set(expected) | set(observed)
                if expected.get(key) != observed.get(key))
            raise ExperimentSpecIntegrityError(
                "Cannot resume final benchmark: internally produced source "
                f"{name!r} changed; different fields={differences}")
    return records


def _pin_internal_sources(
    root: Path,
    sources: dict[str, tuple[ValidatedCheckpointSource, str]],
) -> dict[str, dict[str, Any]]:
    existing = _validate_pinned_internal_sources(root)
    requested = {
        name: _internal_source_record(source, origin=origin)
        for name, (source, origin) in sources.items()
    }
    if existing:
        if existing != requested:
            raise ExperimentSpecIntegrityError(
                "Cannot resume final benchmark: internal checkpoint source "
                "generation differs from the persisted source identity")
        return existing
    payload = {"schema_version": 1, "sources": requested}
    payload["source_identity_sha256"] = hash_canonical_value(payload)
    atomic_write_json(root / _INTERNAL_SOURCES_FILE, payload)
    return requested


def _cotrained_sources(
    root: Path,
) -> tuple[ValidatedCheckpointSource | None, ValidatedCheckpointSource | None]:
    """Resolve co-training outputs through the shared authority boundary."""
    run_root = root / "cotraining"
    prot = run_root / "best.pt"
    if not prot.is_file():
        if ((run_root / "experiment_spec.json").exists()
                or (run_root / "run_state.json").exists()):
            raise ExperimentSpecIntegrityError(
                "Cannot validate internally produced benchmark source: "
                f"committed best.pt mirror is missing: {prot}")
        return None, None
    try:
        protagonist_source = validate_checkpoint_source(prot)
    except (FileNotFoundError, ExperimentSpecIntegrityError) as exc:
        raise ExperimentSpecIntegrityError(
            "Cannot validate internally produced benchmark source: "
            f"pipeline=cotraining checkpoint={prot}: {exc}") from exc

    state_path = run_root / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    configured = state.get("designer_checkpoint")
    if not configured:
        return protagonist_source, None
    try:
        designer_source = validate_checkpoint_source(configured)
    except (FileNotFoundError, ExperimentSpecIntegrityError) as exc:
        raise ExperimentSpecIntegrityError(
            "Cannot validate internally produced benchmark designer source: "
            f"checkpoint={configured}: {exc}") from exc
    return protagonist_source, designer_source


_FINAL_INPUT_FIELDS = {
    "supervised", "expert_iteration", "adversarial_designer",
    "pretrained_designer", "base_dataset", "handcrafted_dataset",
    "prior_cotraining_dir"}
_FINAL_OPERATIONAL_FIELDS = {"output_dir", "phases"}
_FINAL_DERIVED_FIELDS = {"device"}
_FINAL_SEMANTIC_FIELDS = {
    "seed", "astar_max_nodes", "benchmark_count", "saturation_budget",
    "rounds", "levels_per_round", "designer_episodes", "solver_states",
    "budgets", "comparison_budget", "gen_count", "gen_seeds",
    "ablation_designer_episodes", "ablation_rounds"}


def _optional_file_input(path: str | None, *, kind: str) -> dict[str, Any] | None:
    if path is None:
        return None
    source = Path(path)
    if not source.is_file():
        return {"kind": kind, "path_hint": source.as_posix(), "exists": False}
    return file_identity(
        source, kind=kind, count_lines=kind.endswith("dataset"))


def _checkpoint_input(path: str | None, *, kind: str) -> dict[str, Any] | None:
    if path is None:
        return None
    identity = checkpoint_identity(path)
    return {
        "kind": kind,
        "path_hint": Path(path).as_posix(),
        "exists": identity["exists"],
        **({
            "bytes": identity["bytes"],
            "sha256": identity["sha256"],
            "source_experiment_fingerprint":
                identity.get("source_experiment_fingerprint"),
            "source_kind": identity.get("source_kind"),
            "source_pipeline": identity.get("pipeline"),
            "committed_role": identity.get("committed_role"),
            "committed_progress": identity.get("committed_progress"),
            "encoding_fingerprint": identity.get("encoding_fingerprint"),
        } if identity["exists"] else {}),
    }


def _prior_cotraining_identity(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    root = Path(path)
    if not root.is_dir():
        return {
            "kind": "prior_cotraining_replay",
            "path_hint": root.as_posix(),
            "exists": False,
        }
    files = sorted({
        candidate
        for pattern in ("round_*/designer/replay/*.json*",
                        "round_*/designer_attempts/attempt_*/replay/*.json*",
                        "level_replay/*.json*")
        for candidate in root.glob(pattern)
        if candidate.is_file()
    })
    entries = [{
        "relative_path": candidate.relative_to(root).as_posix(),
        "bytes": candidate.stat().st_size,
        "sha256": hash_file_streaming(candidate),
    } for candidate in files]
    source_fingerprint = None
    spec_path = root / "experiment_spec.json"
    if spec_path.exists():
        from ..training.experiment_identity import load_persisted_experiment_spec
        _spec, source_fingerprint = load_persisted_experiment_spec(root)
        state_path = root / "run_state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("experiment_fingerprint") != source_fingerprint:
                raise ExperimentSpecIntegrityError(
                    "prior co-training run-state fingerprint does not match "
                    "its experiment specification")
    return {
        "kind": "prior_cotraining_replay",
        "path_hint": root.as_posix(),
        "exists": True,
        "file_count": len(entries),
        "manifest_sha256": hash_canonical_value(entries),
        "source_experiment_fingerprint": source_fingerprint,
    }


def _final_benchmark_spec(
    args: argparse.Namespace, *, resolved_device=None
) -> dict[str, Any]:
    values = vars(args)
    validate_field_classification(set(values), {
        "semantic": _FINAL_SEMANTIC_FIELDS,
        "operational": _FINAL_OPERATIONAL_FIELDS,
        "input": _FINAL_INPUT_FIELDS,
        "derived": _FINAL_DERIVED_FIELDS,
    })
    semantic = {
        name: values[name] for name in sorted(_FINAL_SEMANTIC_FIELDS)}
    semantic["budgets"] = sorted(set(args.budgets))
    semantic["gen_seeds"] = sorted(set(args.gen_seeds))
    semantic["comparison_budget"] = (
        args.comparison_budget
        if args.comparison_budget is not None else max(args.budgets))
    inputs = {
        "supervised": _checkpoint_input(
            args.supervised, kind="supervised_checkpoint"),
        "expert_iteration": _checkpoint_input(
            args.expert_iteration, kind="expert_iteration_checkpoint"),
        "adversarial_designer": _checkpoint_input(
            args.adversarial_designer, kind="adversarial_designer_checkpoint"),
        "pretrained_designer": _checkpoint_input(
            args.pretrained_designer, kind="pretrained_designer_checkpoint"),
        "base_dataset": _optional_file_input(
            args.base_dataset, kind="base_dataset"),
        "handcrafted_dataset": _optional_file_input(
            args.handcrafted_dataset, kind="handcrafted_dataset"),
        "prior_cotraining": _prior_cotraining_identity(
            args.prior_cotraining_dir),
    }
    return build_experiment_spec(
        pipeline="final_benchmark", semantic_config=semantic, inputs=inputs,
        software_semantics={
            "benchmark_provenance_schema_version": 2,
            "evaluation_semantics_version": EVALUATION_SEMANTICS_VERSION,
            "promotion_contract_version": PROMOTION_CONTRACT_VERSION,
            "transaction_schema_version": TRANSACTION_SCHEMA_VERSION,
            "experiment_identity_version": 1,
            "runtime": runtime_device_provenance(
                requested_device=args.device,
                resolved_device=resolved_device or resolve_device(args.device)),
        })


# ----------------------------------------------------------------------
# Phases
# ----------------------------------------------------------------------

def phase_benchmark(root: Path, args) -> None:
    bdir = root / "benchmark"
    res = harder.build_harder_benchmark(
        str(bdir), protagonist_checkpoint=args.expert_iteration,
        adversarial_designer_checkpoint=args.adversarial_designer,
        pretrained_designer_checkpoint=args.pretrained_designer,
        handcrafted_dataset=args.handcrafted_dataset,
        training_dataset=args.base_dataset,
        prior_cotraining_dir=args.prior_cotraining_dir,
        count=args.benchmark_count, device=args.device, seed=args.seed,
        astar_max_nodes=args.astar_max_nodes)
    sat = harder.saturation_check(
        str(bdir), protagonist_checkpoint=args.expert_iteration,
        device=args.device, budget=args.saturation_budget,
        astar_max_nodes=args.astar_max_nodes)
    _write(root / "saturation.json", sat)
    _write(root / "checkpoints.json", {
        name: checkpoint_identity(getattr(args, name, DEFAULTS.get(name, "")))
        for name in ("supervised", "expert_iteration", "adversarial_designer",
                     "pretrained_designer")})
    print(json.dumps({"benchmark": res["manifest"]["group_sizes"],
                      "eval_levels": res["eval_levels"],
                      "overall_bounded_solve_rate": sat["overall_bounded_solve_rate"],
                      "useful_benchmark": sat["useful_benchmark"]}, indent=2))


def _cotrain_config(root: Path, args) -> CoTrainingConfig:
    # Difficulty knobs calibrated so the bounded protagonist's solve rate is
    # intermediate (see report); the designer is adversarial so we start near
    # the lower density / shallow reverse-depth regime and let the curriculum
    # adapt from there.
    excluded: tuple[str, ...] = ()
    manifest_path = root / "benchmark" / "benchmark_manifest.json"
    if manifest_path.exists():
        excluded = tuple(json.loads(manifest_path.read_text(encoding="utf-8"))
                         .get("all_signatures", []))

    curriculum = CurriculumConfig(
        frontier_min_solve_rate=0.2, frontier_max_solve_rate=0.7,
        min_density=0.2, max_density=0.45, density_step=0.05,
        min_mutation_budget=1, mutation_step=1, min_board=3,
        protagonist_sim_step=80, max_protagonist_simulations=400,
        min_protagonist_simulations=80)
    initial = CurriculumState(rows=5, cols=5, color_count=3, density=0.3,
                              mutation_budget=4, protagonist_simulations=150)
    protagonist_source = validate_checkpoint_source(args.expert_iteration)
    designer_source = validate_checkpoint_source(args.adversarial_designer)
    return CoTrainingConfig(
        protagonist_checkpoint=str(protagonist_source.checkpoint_path),
        designer_checkpoint=str(designer_source.checkpoint_path),
        base_dataset=args.base_dataset,
        output_dir=str(root / "cotraining"),
        eval_levels_dataset=str(root / "benchmark" / "eval_levels.jsonl"),
        eval_split_manifest=str(root / "benchmark" / "eval_split.json"),
        eval_split_seed=1729,
        excluded_signatures=excluded,
        rounds=args.rounds, levels_per_round=args.levels_per_round,
        seed=args.seed, device=args.device, solve_rate_trials=5,
        astar_max_nodes=args.astar_max_nodes, oracle_simulations=150,
        eval_budgets=(1, 50, 200), eval_limit=24, promotion_budget=50,
        states_per_level=2, train_sample_size=512, epochs=2, batch_size=64,
        designer_episodes=args.designer_episodes,
        designer_episodes_per_iter=args.designer_episodes,
        benchmark_count=8, ood_rows=6, ood_cols=6,
        curriculum=curriculum, initial_curriculum=initial)


def phase_cotrain(root: Path, args) -> None:
    cfg = _cotrain_config(root, args)
    result = run_cotraining(cfg)
    protagonist_source, designer_source = _cotrained_sources(root)
    if protagonist_source is None:
        raise ExperimentSpecIntegrityError(
            "co-training completed without an authoritative protagonist source")
    internal_sources = {
        "cotrained_protagonist": (
            protagonist_source, "internal_cotraining_phase"),
    }
    if designer_source is not None:
        internal_sources["cotrained_designer"] = (
            designer_source, "internal_cotraining_state")
    source_records = _pin_internal_sources(root, internal_sources)
    _write(root / "cotraining_summary.json",
           {"rounds": result["rounds"],
            "history": result["run_state"]["history"],
            "evaluation_split":
                result["run_state"].get("evaluation_split"),
            "best_protagonist": result["run_state"]["best_protagonist"],
            "designer_checkpoint": result["run_state"]["designer_checkpoint"],
            "validated_sources": source_records})
    checkpoints_path = root / "checkpoints.json"
    checkpoints = (
        json.loads(checkpoints_path.read_text(encoding="utf-8"))
        if checkpoints_path.is_file() else {})
    checkpoints.update(source_records)
    _write(checkpoints_path, checkpoints)
    print(f"co-training done: {result['rounds']} rounds")


def _harder_states(root: Path, groups=("frontier_selected", "ood_large_dense",
                                       "random"), limit=60):
    data = json.loads((root / "benchmark" / "benchmark.json")
                      .read_text(encoding="utf-8"))
    env = Environment()
    states = []
    for g in groups:
        for d in data.get(g, []):
            states.append(env.initial_state(level_from_dict(d)))
    return states[:limit]


def phase_solver(root: Path, args) -> None:
    dev = resolve_device(args.device)
    cot_prot, cot_designer = _cotrained_sources(root)
    models = {"supervised": Protagonist(args.supervised, dev),
              "expert_iteration": Protagonist(args.expert_iteration, dev)}
    if cot_prot is not None:
        models["cotrained"] = Protagonist(cot_prot, dev)
    states = _harder_states(root, limit=args.solver_states)
    rep, per_state = solver_mod.compare_solvers(
        states, models=models, budgets=list(args.budgets),
        astar_max_nodes=args.astar_max_nodes, device=dev, seed=args.seed,
        comparison_budget=args.comparison_budget)
    if cot_prot is not None:
        sources = {
            "cotrained_protagonist": (
                cot_prot, "internal_cotraining_phase"),
        }
        if cot_designer is not None:
            sources["cotrained_designer"] = (
                cot_designer, "internal_cotraining_state")
        rep["validated_internal_sources"] = _pin_internal_sources(root, sources)
    _write(root / "solver.json", rep)
    with open(root / "per_state_solver.jsonl", "w", encoding="utf-8") as fh:
        for row in per_state:
            fh.write(json.dumps(row) + "\n")
    print(json.dumps({"states": rep["states"],
                      "common_solved": rep["common_solved_count"]}, indent=2))


def phase_generators(root: Path, args) -> None:
    cot_prot, cot_designer = _cotrained_sources(root)
    designers = {"random": None,
                 "bc_designer": args.pretrained_designer,
                 "adversarial_designer": args.adversarial_designer}
    if cot_designer is not None:
        designers["cotrained_designer"] = str(cot_designer.checkpoint_path)
    rep = gen_mod.compare_generators(
        protagonist_checkpoint=args.expert_iteration,
        designer_checkpoints=designers, count=args.gen_count,
        seeds=list(args.gen_seeds),
        generator=GeneratorConfig(rows=5, cols=5, color_count=3, density=0.5),
        mutation_budget=10, protagonist_simulations=100, oracle_simulations=150,
        astar_max_nodes=args.astar_max_nodes, device=args.device)
    if cot_prot is not None:
        sources = {
            "cotrained_protagonist": (
                cot_prot, "internal_cotraining_phase"),
        }
        if cot_designer is not None:
            sources["cotrained_designer"] = (
                cot_designer, "internal_cotraining_state")
        rep["validated_internal_sources"] = _pin_internal_sources(root, sources)
    _write(root / "generators.json", rep)
    print("generator comparison written")


def phase_ablations(root: Path, args) -> None:
    adir = root / "ablations"
    protagonist_source = validate_checkpoint_source(args.expert_iteration)
    designer_source = validate_checkpoint_source(args.pretrained_designer)
    reward = ablations.reward_term_ablations(
        root=str(adir),
        protagonist_checkpoint=str(protagonist_source.checkpoint_path),
        init_designer=str(designer_source.checkpoint_path),
        model_cfg=DesignerModelConfig(), gen_cfg=GeneratorConfig(
            rows=5, cols=5, color_count=3, density=0.5),
        episodes=args.ablation_designer_episodes, sims=30, oracle_sims=120,
        astar_nodes=args.astar_max_nodes, seed=args.seed, device=args.device)
    base_cfg = dataclasses.replace(_cotrain_config(root, args),
                                   rounds=args.ablation_rounds)
    cotrain = ablations.cotraining_ablations(root=str(adir), base_cfg=base_cfg)
    _write(root / "ablations.json", {"reward_terms": reward,
                                     "cotraining": cotrain})
    print("ablations written")


def phase_report(root: Path, args) -> None:
    _validate_pinned_internal_sources(root)
    report.build_report(root, defaults=DEFAULTS, args_dict=vars(args))
    print(f"report written to {root / 'report.md'}")


PHASES = {
    "benchmark": phase_benchmark,
    "cotrain": phase_cotrain,
    "solver": phase_solver,
    "generators": phase_generators,
    "ablations": phase_ablations,
    "report": phase_report,
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Final benchmark experiment driver")
    p.add_argument("--output-dir", default="runs/final_benchmark")
    p.add_argument("--phases", nargs="+",
                   default=["benchmark", "cotrain", "solver", "generators",
                            "ablations", "report"])
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=2026)
    for name, default in DEFAULTS.items():
        p.add_argument(f"--{name.replace('_', '-')}", default=default)
    p.add_argument("--prior-cotraining-dir", default="runs/cotraining")
    p.add_argument("--astar-max-nodes", type=int, default=20_000)
    p.add_argument("--benchmark-count", type=int, default=40)
    p.add_argument("--saturation-budget", type=int, default=200)
    # cotraining
    p.add_argument("--rounds", type=int, default=6)
    p.add_argument("--levels-per-round", type=int, default=16)
    p.add_argument("--designer-episodes", type=int, default=12)
    # solver
    p.add_argument("--solver-states", type=int, default=60)
    p.add_argument("--budgets", type=int, nargs="+", default=[1, 50, 200])
    p.add_argument(
        "--comparison-budget", type=int, default=None,
        help="one common search budget for every controlled model comparison; "
             "defaults to the largest configured --budgets value")
    # generators
    p.add_argument("--gen-count", type=int, default=30)
    p.add_argument("--gen-seeds", type=int, nargs="+", default=[1000, 2000, 3000])
    # ablations
    p.add_argument("--ablation-designer-episodes", type=int, default=12)
    p.add_argument("--ablation-rounds", type=int, default=2)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = Path(args.output_dir)
    resolved_device = resolve_device(args.device)
    requested_spec = _final_benchmark_spec(
        args, resolved_device=resolved_device)
    _validate_pinned_internal_sources(root)
    validate_or_initialize_experiment(
        root, requested_spec,
        run_state=None,
        extra_legacy_markers=(
            "benchmark", "cotraining_summary.json", "saturation.json",
            "solver.json", "generators.json", "ablations.json", "report.md",
            "checkpoints.json"))
    root.mkdir(parents=True, exist_ok=True)
    for ph in args.phases:
        if ph not in PHASES:
            raise SystemExit(f"unknown phase: {ph}")
        print(f"\n===== phase: {ph} =====", flush=True)
        PHASES[ph](root, args)


if __name__ == "__main__":
    main()
