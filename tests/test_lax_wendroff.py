"""Hand update, conservation, intended timing, and explicit failure capture."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hydro_pinn.cli import main
from hydro_pinn.config import load_config
from hydro_pinn.lax_wendroff import (
    NumericalRunFailed,
    SolverFailure,
    advance_one_step,
    generate_solver_run,
    initial_cell_averages,
    solve,
)

ROOT = Path(__file__).resolve().parents[1]


def test_constant_state_and_exact_dry_flux() -> None:
    for value in ([1.0, 0.3], [0.0, 0.0]):
        initial = np.tile(value, (4, 1))
        updated, flux, predictor = advance_one_step(initial, value, value, 1, 0.1)
        assert_allclose(updated, initial, atol=0, rtol=0)
        assert_allclose(predictor, np.tile(value, (5, 1)), atol=0, rtol=0)
        assert_allclose(flux, np.tile(flux[0], (5, 1)), atol=0, rtol=0)
        if value[0] == 0:
            assert np.count_nonzero(flux) == 0


def test_hand_checked_two_cell_step() -> None:
    # g=2, dx=1, dt=.1: midpoint U=(1.5,.15), F=(.15,2.265).
    initial = np.array([[2.0, 0.0], [1.0, 0.0]])
    updated, flux, predictor = advance_one_step(
        initial, initial[0], initial[-1], 1.0, 0.1, 2.0
    )
    assert_allclose(predictor, [[2, 0], [1.5, 0.15], [1, 0]], atol=1e-15)
    assert_allclose(flux, [[0, 4], [0.15, 2.265], [0, 1]], atol=1e-15)
    assert_allclose(updated, [[1.985, 0.1735], [1.015, 0.1265]], atol=1e-15)


def test_nonzero_numerical_boundary_flux_telescopes() -> None:
    initial = np.array([[2.0, 0.3], [1.5, 0.2], [1.0, 0.0]])
    dx, dt = 0.2, 0.01
    updated, flux, _ = advance_one_step(initial, [2, 0.4], [1, -0.1], dx, dt)
    difference = dx * (updated.sum(axis=0) - initial.sum(axis=0))
    assert_allclose(difference + dt * (flux[-1] - flux[0]), 0, atol=4e-16)
    # Fixed reservoir q does not force the predictor mass flux to that q.
    assert flux[0, 0] != 0.4 and flux[-1, 0] != -0.1


@pytest.mark.parametrize("case_name,volume", [("e1_wet", 5.5), ("e2_dry", 5.0)])
def test_native_initial_averages_timing_cfl_and_mass(
    case_name: str, volume: float
) -> None:
    case = load_config(ROOT / f"configs/cases/{case_name}.json")
    evaluation = load_config(ROOT / "configs/verification.json")
    x, edges, initial = initial_cell_averages(case, 100)
    assert_allclose(np.diff(edges), 0.1, atol=1e-15)
    assert_allclose(x, -5 + (np.arange(100) + 0.5) * 0.1, atol=0, rtol=0)
    assert_allclose(initial[:50, 0], 1, atol=0, rtol=0)
    assert_allclose(initial[50:, 0], case["right_depth_m"], atol=0, rtol=0)
    assert_allclose(0.1 * initial[:, 0].sum(), volume, atol=1e-15)
    result = solve(case, 100, evaluation["times_s"])
    assert result.failure is None
    assert np.array_equal(result.t_s, evaluation["times_s"])
    assert result.h_m.shape == result.q_m2_s.shape == (6, 100)
    diagnostics = result.diagnostics
    assert diagnostics["maximum_cfl"] <= 0.45 + 1e-15
    assert diagnostics["maximum_absolute_balance_m2"] < 3e-15
    assert abs(diagnostics["initial_volume_error_m2"]) < 1e-15
    steps = diagnostics["steps"]
    assert steps[0]["dt_s"] == pytest.approx(0.45 * 0.1 / np.sqrt(9.81))
    ends = [row["time_end_s"] for row in steps]
    assert all(t in ends for t in evaluation["times_s"])
    assert all(
        not (row["time_start_s"] < t < row["time_end_s"])
        for row in steps
        for t in evaluation["times_s"]
    )
    integrated_flux = sum(
        row["dt_s"] * (row["right_boundary_flux_m2_s"] - row["left_boundary_flux_m2_s"])
        for row in steps
    )
    final = diagnostics["cell_sum_mass"][-1]
    assert_allclose(
        final["mass_m2"] - volume + integrated_flux, final["signed_balance_m2"], atol=0
    )


@pytest.mark.parametrize(
    "state,dt,stage",
    [([[1, -2], [1, 2]], 1.0, "predictor"), ([[2, 0], [1, 0]], 2.0, "corrector")],
)
def test_rejected_step_reports_raw_stage(state: list, dt: float, stage: str) -> None:
    values = np.array(state, dtype=float)
    untouched = values.copy()
    with pytest.raises(SolverFailure) as caught:
        advance_one_step(values, values[0], values[-1], 1, dt, 2, time_s=0.25, step=7)
    failure = caught.value
    assert failure.stage == stage and failure.step == 7
    assert failure.time_s == 0.25 + dt * (0.5 if stage == "predictor" else 1)
    assert failure.as_dict()["negative_depth_count"] > 0
    assert np.min(failure.state[:, 0]) < 0
    assert np.array_equal(values, untouched)


def test_current_failure_zero_speed_and_nonfinite_are_explained() -> None:
    case = load_config(ROOT / "configs/cases/e2_dry.json")
    for values, reason in (
        (np.zeros((2, 2)), "wave speed"),
        (np.array([[np.nan, 0], [0, 0]]), "nonfinite"),
        (np.array([[0, 1], [0, 0]]), "zero depth"),
        (np.array([[1e-320, 1], [0, 0]]), "invalid current velocity"),
    ):
        result = solve(case, 2, [0.113], initial_state=values)
        assert result.failure["stage"] == "current"
        assert reason in result.failure["reason"]
        assert result.h_m.shape == (0, 2)
        json.dumps(result.diagnostics, allow_nan=False)


def test_run_success_and_persisted_failure_have_distinct_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_path, evaluation_path = (
        ROOT / "configs/cases/e1_wet.json",
        ROOT / "configs/verification.json",
    )
    path = generate_solver_run(case_path, evaluation_path, 100, "complete", tmp_path)
    assert json.loads((path / "status.json").read_text())["state"] == "completed"
    with np.load(path / "outputs.npz") as arrays:
        assert arrays["h_m"].shape == (6, 100)
        assert np.array_equal(arrays["t_s"], load_config(evaluation_path)["times_s"])
        assert len(arrays["cell_edges_m"]) == 101
    with pytest.raises(FileExistsError):
        generate_solver_run(case_path, evaluation_path, 100, "complete", tmp_path)

    # Diagnostic input produces an actual solver rejection. Only run I/O is
    # injected here so canonical case files remain unchanged at CFL=.45.
    case = load_config(case_path)
    initial = np.zeros((100, 2))
    rejected = solve(case, 100, [0.113], initial_state=initial)
    monkeypatch.setattr("hydro_pinn.lax_wendroff.solve", lambda *args: rejected)
    with pytest.raises(NumericalRunFailed) as caught:
        generate_solver_run(case_path, evaluation_path, 100, "failed", tmp_path)
    failed_path = caught.value.path
    status = json.loads((failed_path / "status.json").read_text())
    assert status["state"] == "failed"
    assert status["diagnostics"]["stage"] == "current"
    assert (failed_path / "last_valid_state.npz").is_file()
    assert (failed_path / "failed_state.npz").is_file()
    assert not (failed_path / "outputs.npz").exists()
    monkeypatch.chdir(tmp_path)
    assert (
        main(
            [
                "solve",
                "--config",
                str(case_path),
                "--evaluation",
                str(evaluation_path),
                "--cells",
                "100",
                "--run-id",
                "cli-failure",
            ]
        )
        == 3
    )
