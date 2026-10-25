"""Independent mass arithmetic and separately recorded checkpoint evaluations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose

from hydro_pinn.config import load_config
from hydro_pinn.conservation import (
    auxiliary_record,
    file_sha256,
    native_solver_mass,
    prescribed_initial_volume,
    quadrature_mass,
    quadrature_sensitivity,
)
from hydro_pinn.references import evaluate_reference
from hydro_pinn.train import predict_run, train_run

ROOT = Path(__file__).resolve().parents[1]


def samples(x, t, h, q) -> dict:
    shape = (len(t), len(x))
    return {
        "x_m": np.array(x),
        "t_s": np.array(t),
        "h_m": np.broadcast_to(h, shape).copy(),
        "q_m2_s": np.broadcast_to(q, shape).copy(),
    }


def test_boundary_flux_sign_initial_error_and_native_cell_sums() -> None:
    case = load_config(ROOT / "configs/cases/e1_wet.json")
    times = np.array([0.0, 0.25, 0.5])
    x = -5 + (np.arange(4) + 0.5) * 2.5
    volume = samples(x, times, ((6 - 0.3 * times) / 10)[:, None], 0)
    boundary = samples([-5, 5], np.linspace(0, 0.5, 5), 1, [0.2, 0.5])
    result = quadrature_mass(case, volume, boundary)
    assert prescribed_initial_volume(case) == 5.5
    assert_allclose(result["learned_initial_volume_error_m2"], 0.5)
    assert_allclose(result["boundary_net_outflow_m2"], 0.3 * times)
    assert_allclose(result["signed_balance_m2"], 0.5)
    assert_allclose(result["maximum_absolute_normalized_imbalance"], 0.5 / 5.5)
    assert_allclose(result["integrated_absolute_boundary_mismatch_m2"], 0.7 * times)
    assert result["maximum_absolute_boundary_mismatch_m2_s"] == 0.5
    diagnostics = {
        "cell_sum_mass": [{"time_s": t, "mass_m2": 5.5 - 0.4 * t} for t in times],
        "steps": [
            {
                "time_start_s": times[i],
                "time_end_s": times[i + 1],
                "dt_s": 0.25,
                "left_boundary_flux_m2_s": 0.1,
                "right_boundary_flux_m2_s": 0.5,
            }
            for i in range(2)
        ],
    }
    native = samples([-2.5, 2.5], times[1:], ((5.5 - 0.4 * times[1:]) / 10)[:, None], 0)
    native["cell_edges_m"] = np.array([-5, 0, 5])
    result = native_solver_mass(case, diagnostics, native)
    assert_allclose(result["signed_balance_m2"], 0, atol=1e-15)
    assert_allclose(result["boundary_net_outflow_m2"], [0, 0.1, 0.2])
    assert result["boundary_t_s"] == [0, 0.25, 0.5]
    assert result["left_boundary_q_m2_s"] == [0.1, 0.1, 0.1]
    shifted_diagnostics = {
        **diagnostics,
        "cell_sum_mass": [
            {**row, "mass_m2": row["mass_m2"] + 0.5}
            for row in diagnostics["cell_sum_mass"]
        ],
    }
    biased = native_solver_mass(
        case, shifted_diagnostics, {**native, "h_m": native["h_m"] + 0.05}
    )
    assert_allclose(biased["normalized_initial_volume_error"], 0.5 / 5.5)
    assert_allclose(biased["maximum_absolute_normalized_imbalance"], 0.5 / 5.5)


def test_ritter_closed_form_mass_and_doubled_quadrature() -> None:
    case = load_config(ROOT / "configs/cases/e2_dry.json")
    times = [0, 0.1, 0.25, 0.5]
    results = []
    for count, nodes in [(2000, 1001), (4000, 2001)]:
        x = -5 + (np.arange(count) + 0.5) * 10 / count
        volume = evaluate_reference(case, x, times)
        boundary = evaluate_reference(case, [-5, 5], np.linspace(0, 0.5, nodes))
        result = quadrature_mass(case, vars(volume), vars(boundary))
        # Exact Ritter volume is 5: integrate the left reservoir and quadratic fan.
        assert_allclose(result["volume_m2"], 5, atol=1e-5, rtol=0)
        assert_allclose(result["boundary_net_outflow_m2"], 0, atol=0)
        assert result["learned_initial_volume_error_m2"] == 0
        results.append(result)
    sensitivity = quadrature_sensitivity(*results)
    assert sensitivity["base"] == [2000, 1001]
    assert sensitivity["doubled"] == [4000, 2001]
    assert max(abs(np.array(results[1]["signed_balance_m2"]))) < max(
        abs(np.array(results[0]["signed_balance_m2"]))
    )


def test_checkpoint_auxiliary_path_binds_source_and_preserves_training(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    output_root = root / "runs"
    train_path = train_run(
        ROOT / "configs/cases/e2_dry.json",
        ROOT / "configs/training/baseline.json",
        11,
        "cpu",
        "float64",
        "training",
        smoke_path=ROOT / "configs/smoke.json",
        output_root=output_root,
        overrides={
            "training": {
                "network": {"hidden_layers": [4]},
                "sampling": {
                    "interior_points": 4,
                    "initial_points": 4,
                    "boundary_points_per_side": 2,
                },
                "optimizer": {"steps": 1},
            }
        },
    )
    original_hashes = {
        p.name: file_sha256(p) for p in train_path.iterdir() if p.is_file()
    }
    prediction = predict_run(
        "training", ROOT / "configs/smoke.json", "prediction", output_root
    )
    source = json.loads((prediction / "resolved_config.json").read_text())
    axes = {
        "rule": "midpoint",
        "cells": 4,
        "x_m": [-3.75, -1.25, 1.25, 3.75],
        "t_s": [0, 0.1, 0.5],
    }
    protocol = {
        "protocol_sha256": "fixture-protocol",
        "resolved": {"evaluation": source["evaluation"]},
        "conservation": {"volume_axes": {"4": axes}},
    }
    plan = {
        "kind": "volume",
        "axis_key": "4",
        "source_kind": "predict",
        "run_id": "planned-volume",
        "source_stage_id": "prediction-stage",
        "evaluation_sha256": "fixture-evaluation",
    }
    execution = {"output_root": "runs", "run_ids": {"planned-volume": "replay-volume"}}
    arrays, artifact = auxiliary_record(
        protocol, plan, "prediction", project_root=root, execution=execution
    )
    assert arrays["h_m"].shape == (3, 4)
    assert artifact["run_id"] == "replay-volume"
    assert artifact["checkpoint_sha256"] == original_hashes["checkpoint.pt"]
    assert artifact["source_resolved_config_sha256"] == file_sha256(
        prediction / "resolved_config.json"
    )
    loaded, repeated = auxiliary_record(
        protocol, plan, "prediction", project_root=root, execution=execution
    )
    assert_allclose(loaded["h_m"], arrays["h_m"], atol=0, rtol=0)
    assert repeated == artifact
    assert original_hashes == {
        p.name: file_sha256(p) for p in train_path.iterdir() if p.is_file()
    }
