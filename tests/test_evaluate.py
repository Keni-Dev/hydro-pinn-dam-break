"""Small independent arithmetic checks; these are not final study results."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hydro_pinn.config import load_config
from hydro_pinn.evaluate import (
    EvaluationError,
    aggregate_metric_rows,
    common_grid_fields,
    error_metrics,
    evaluate_fields,
    evaluation_axes,
    unavailable_metric_rows,
)
from hydro_pinn.references import wave_diagnostics

ROOT = Path(__file__).resolve().parents[1]


def evaluation(cells: int = 4) -> dict:
    return {
        "grid": {"kind": "cell_centres", "domain_m": [-5.0, 5.0], "cells": cells},
        "times_s": [0.1, 0.5],
        "velocity_wet_threshold_m": 0.001,
        "wave_band_half_width_m": 0.1,
    }


def identity(variant: str = "baseline", seed: int = 11) -> dict:
    return {
        "run_id": f"fixture-{variant}-{seed}",
        "case_id": "E2",
        "method": "pinn",
        "variant": variant,
        "seed": seed,
        "cells": None,
    }


def fields(config: dict, h: list, q: list) -> dict:
    x, times = evaluation_axes(config)
    return {"x_m": x, "t_s": times, "h_m": np.array(h), "q_m2_s": np.array(q)}


def row(rows: list[dict], field: str, region: str = "whole", time_s: float = 0.1):
    return next(
        item
        for item in rows
        if item["field"] == field
        and item["region"] == region
        and item["time_s"] == time_s
    )


def test_known_error_arithmetic_and_undefined_denominator() -> None:
    # Errors (3,4) give RMSE sqrt(25/2), reference norm sqrt(1+4).
    result = error_metrics([4.0, 6.0], [1.0, 2.0])
    assert_allclose(result["rmse"]["value"], np.sqrt(12.5), rtol=0, atol=1e-15)
    assert_allclose(result["relative_l2"]["value"], np.sqrt(5), rtol=0, atol=1e-15)
    zero = error_metrics([1.0, -1.0], [0.0, 0.0])
    assert zero["rmse"] == {"value": 1.0, "reason": None}
    assert zero["relative_l2"] == {
        "value": None,
        "reason": "zero_or_negligible_reference_norm",
    }
    assert error_metrics([], [])["rmse"]["reason"] == "no_selected_points"


def test_reference_wet_mask_and_incomplete_candidate_velocity() -> None:
    case = load_config(ROOT / "configs/cases/e2_dry.json")
    config = evaluation()
    # Exactly 0.001 m is excluded; the two dry/threshold cells have large q/h
    # errors so a prediction-derived or non-strict mask would fail this check.
    exact = fields(config, [[1, 0.001, 0, 2]] * 2, [[2, 0, 0, 8]] * 2)
    predicted = fields(config, [[1, 1, 1, 2]] * 2, [[3, 99, 100, 10]] * 2)
    rows = evaluate_fields(case, exact, predicted, config, identity())
    velocity = row(rows, "u_m_s")
    assert velocity["selected_points"] == velocity["eligible_points"] == 2
    assert velocity["reference_wet_points"] == 2
    assert velocity["coverage_fraction"] == 0.5
    assert velocity["rmse"]["value"] == 1
    assert_allclose(velocity["relative_l2"]["value"], np.sqrt(2 / 20))
    # A model cannot hide its missed wet point by predicting dry depth.
    predicted["h_m"][0, 0], predicted["q_m2_s"][0, 0] = 0, 0
    incomplete = row(
        evaluate_fields(case, exact, predicted, config, identity()), "u_m_s"
    )
    assert incomplete["selected_points"] == 1 and incomplete["eligible_points"] == 2
    assert incomplete["rmse"]["value"] is None
    assert incomplete["rmse"]["reason"] == "incomplete_candidate_velocity_coverage"
    assert row(rows, "h_m")["selected_points"] == 4


@pytest.mark.parametrize(
    "case_name,wave", [("e1_wet", "shock"), ("e2_dry", "dry_front")]
)
def test_exact_wave_band_and_complement_are_reported(case_name: str, wave: str) -> None:
    case = load_config(ROOT / f"configs/cases/{case_name}.json")
    config = evaluation(400)
    x, times = evaluation_axes(config)
    exact = fields(config, np.ones((2, 400)), np.zeros((2, 400)))
    rows = evaluate_fields(
        case, exact, exact, config, {**identity(), "case_id": case["case_id"]}
    )
    speed = wave_diagnostics(case)["wave_speeds_m_s"][wave]
    for elapsed in times:
        inside = row(rows, "h_m", "wave_band", elapsed)
        outside = row(rows, "h_m", "outside_wave_band", elapsed)
        expected = np.count_nonzero(np.abs(x - elapsed * speed) <= 0.1)
        assert inside["selected_points"] == expected
        assert outside["selected_points"] == 400 - expected
        assert inside["exact_wave_position_m"] == elapsed * speed
        assert inside["wave_kind"] == wave


def test_native_reconstruction_right_face_and_exact_times() -> None:
    config = evaluation(5)  # A common centre is the exact native internal face 0.
    native = {
        "x_m": [-2.5, 2.5],
        "t_s": [0.1, 0.5],
        "cell_edges_m": [-5.0, 0.0, 5.0],
        "h_m": [[10, 20], [30, 40]],
        "q_m2_s": [[1, 2], [3, 4]],
    }
    reconstructed = common_grid_fields(native, config)
    assert_allclose(reconstructed["h_m"], [[10, 10, 20, 20, 20], [30, 30, 40, 40, 40]])
    assert_allclose(reconstructed["q_m2_s"], [[1, 1, 2, 2, 2], [3, 3, 4, 4, 4]])
    assert np.asarray(native["h_m"]).shape == (2, 2)
    native["t_s"] = [0.1, np.nextafter(0.5, 1)]
    with pytest.raises(EvaluationError, match="exactly"):
        common_grid_fields(native, config)


def test_seed_aggregation_keeps_failures_missing_seeds_and_pairs() -> None:
    case = load_config(ROOT / "configs/cases/e2_dry.json")
    config = evaluation()
    exact = fields(config, [[1, 1, 1, 1]] * 2, [[0, 0, 0, 0]] * 2)
    baseline = fields(config, [[2, 2, 2, 2]] * 2, [[0, 0, 0, 0]] * 2)
    adaptive = fields(config, [[3, 3, 3, 3]] * 2, [[0, 0, 0, 0]] * 2)
    rows = evaluate_fields(case, exact, baseline, config, identity())
    rows += evaluate_fields(case, exact, adaptive, config, identity("adaptive"))
    rows += unavailable_metric_rows(
        case, config, identity("baseline", 22), "training_failed"
    )
    rows += evaluate_fields(
        case,
        exact,
        exact,
        config,
        {
            **identity(),
            "run_id": "fixture-reference",
            "method": "analytical_reference",
            "seed": None,
            "variant": None,
        },
    )
    report = aggregate_metric_rows(rows, [11, 22, 33])
    unseeded = next(
        item for item in report["groups"] if item["method"] == "analytical_reference"
    )
    assert unseeded["coverage_kind"] == "single_unseeded_run"
    assert unseeded["rmse"]["single_run_metric"]["value"] == 0
    assert "defined_seed_count" not in unseeded["rmse"]
    group = next(
        item
        for item in report["groups"]
        if item["variant"] == "baseline"
        and item["field"] == "h_m"
        and item["region"] == "whole"
        and item["time_s"] == 0.1
    )
    assert group["rmse"]["mean"]["value"] == 1
    assert group["rmse"]["defined_seed_count"] == 1
    assert group["rmse"]["expected_seed_count"] == 3
    assert group["rmse"]["missing_seeds"] == [33]
    assert group["rmse"]["undefined_seeds"] == {"22": "training_failed"}
    assert group["rmse"]["sample_standard_deviation"]["value"] is None
    pair = next(
        item
        for item in report["paired_ablation"]
        if item["field"] == "h_m"
        and item["region"] == "whole"
        and item["time_s"] == 0.1
    )
    assert pair["seeds"][0]["rmse"] == {"value": 1.0, "reason": None}
    assert pair["seeds"][1]["rmse"]["reason"] == "missing_paired_seed_run"
    assert pair["seeds"][2]["rmse"]["reason"] == "missing_paired_seed_run"
