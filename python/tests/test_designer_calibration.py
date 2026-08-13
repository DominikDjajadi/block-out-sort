from dataclasses import replace
from pathlib import Path

import pytest

from blocksort.cotraining import designer_calibration as calibration


def test_calibration_aggregates_thresholds_and_reuses_completed_seeds(
        tmp_path, monkeypatch):
    training_calls = []

    def fake_train(cfg):
        training_calls.append(cfg.seed)
        checkpoint = Path(cfg.output_dir) / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(f"designer-{cfg.seed}".encode("utf-8"))
        return {
            "best_checkpoint": str(checkpoint),
            "best_validation_metrics": {
                "frontier_in_band_rate": 0.125,
                "mean_frontier_alignment": 0.25,
                "mean_reward": 0.5,
            },
            "best_selection_metric": {"name": "frontier-first"},
            "frontier_simulation_budgets": [25, 50, 100, 200, 400],
        }

    def fake_holdout(cfg, *, seed, designer_checkpoint):
        strict = 20 if seed == 11 else 5
        return {
            "levels": 100,
            "valid_count": 100,
            "invalid_count": 0,
            "construction_proven_solvable_count": 100,
            "unique_count": 100,
            "strict_frontier_count": strict,
            "strict_frontier_rate": strict / 100,
            "below_frontier_count": 100 - strict,
            "above_frontier_count": 0,
            "mean_solve_rate": 0.2,
            "solve_rate_distribution": {},
            "simulation_budgets": [25, 50, 100, 200, 400],
        }

    monkeypatch.setattr(calibration, "train_designer", fake_train)
    monkeypatch.setattr(calibration, "_evaluate_holdout", fake_holdout)
    cfg = calibration.CalibrationConfig(
        protagonist_checkpoint="protagonist.pt",
        designer_checkpoint="designer.pt",
        output_dir=str(tmp_path / "calibration"),
        seeds=(11, 12),
    )

    first = calibration.run_calibration(cfg)
    second = calibration.run_calibration(cfg)

    assert first["decision"] == "no_go"
    assert first["passed_seed_count"] == 1
    assert second == first
    assert training_calls == [11, 12]
    assert (Path(cfg.output_dir) / "summary.csv").is_file()

    with pytest.raises(RuntimeError, match="different settings"):
        calibration.run_calibration(replace(
            cfg, min_holdout_frontier_rate=0.2))


@pytest.mark.parametrize(
    "kwargs,message",
    [
        ({"seeds": ()}, "seeds"),
        ({"seeds": (1, 1)}, "unique"),
        ({"solve_rate_trials": 1}, "at least two"),
        ({"validation_episodes": 0}, "positive"),
        ({"frontier_min_solve_rate": 0.8,
          "frontier_max_solve_rate": 0.2}, "cannot exceed"),
    ],
)
def test_calibration_config_rejects_invalid_values(kwargs, message):
    cfg = calibration.CalibrationConfig(
        protagonist_checkpoint="p",
        designer_checkpoint="d",
        output_dir="out",
        **kwargs,
    )
    with pytest.raises(ValueError, match=message):
        cfg.validate()
