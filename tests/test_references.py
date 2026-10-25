"""Analytical identities, branch conventions, and saved-reference integrity."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hydro_pinn.config import load_config
from hydro_pinn.references import (
    ReferenceError,
    evaluate_reference,
    generate_reference_run,
    solve_wet_star,
    wave_diagnostics,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def wet() -> dict:
    return load_config(ROOT / "configs/cases/e1_wet.json")


@pytest.fixture
def dry() -> dict:
    return load_config(ROOT / "configs/cases/e2_dry.json")


def test_wet_star_bracket_jump_and_independent_swashes_relation(wet: dict) -> None:
    star = solve_wet_star(wet)
    assert 0.1 < star.h_m < 1
    assert star.converged and star.stop_reason == "converged"
    assert abs(star.root_residual_m_s) < 1e-10
    assert_allclose(
        [star.h_m, star.u_m_s, star.shock_speed_m_s],
        [0.3961748167994429, 2.3213549956407444, 3.1051336506682135],
        atol=1e-11,
        rtol=0,
    )
    # SWASHESv7 section4.1.1 gives a polynomial in c_m, independently of
    # this implementation's rarefaction/shock velocity-intersection equation.
    gravity, hr = wet["gravity_m_s2"], wet["right_depth_m"]
    c0, cm = math.sqrt(gravity * wet["left_depth_m"]), star.c_m_s
    polynomial = -8 * gravity * hr * cm**2 * (c0 - cm) ** 2
    polynomial += (cm**2 - gravity * hr) ** 2 * (cm**2 + gravity * hr)
    assert abs(polynomial) < 1e-10
    swashes_shock = 2 * cm**2 * (c0 - cm) / (cm**2 - gravity * hr)
    assert_allclose(star.shock_speed_m_s, swashes_shock, atol=1e-12, rtol=0)
    checks = wave_diagnostics(wet)
    assert abs(checks["rankine_hugoniot_mass_residual_m2_s"]) < 1e-10
    assert abs(checks["rankine_hugoniot_momentum_residual_m3_s2"]) < 1e-10


@pytest.mark.parametrize("case_fixture", ["wet", "dry"])
def test_t0_uses_prescribed_initial_condition_and_left_dam_equality(
    case_fixture: str, request: pytest.FixtureRequest
) -> None:
    case = request.getfixturevalue(case_fixture)
    result = evaluate_reference(case, [-0.1, 0, 0.1], [0])
    assert_allclose(result.h_m[0], [1, 1, case["right_depth_m"]], rtol=0, atol=0)
    assert np.count_nonzero(result.q_m2_s) == 0
    assert np.count_nonzero(result.u_m_s) == 0
    assert result.velocity_valid[0, -1] == (case["right_depth_m"] > 0)


def test_wet_four_branches_and_shock_equality(wet: dict) -> None:
    star = solve_wet_star(wet)
    elapsed = 0.5
    points = (
        np.array(
            [
                star.fan_head_speed_m_s - 0.1,
                star.fan_head_speed_m_s,
                0.0,
                star.fan_tail_speed_m_s,
                (star.fan_tail_speed_m_s + star.shock_speed_m_s) / 2,
                star.shock_speed_m_s,
                star.shock_speed_m_s + 0.1,
            ]
        )
        * elapsed
    )
    result = evaluate_reference(wet, points, [elapsed])
    c0 = math.sqrt(wet["gravity_m_s2"])
    assert_allclose(
        result.h_m[0], [1, 1, 4 / 9, star.h_m, star.h_m, 0.1, 0.1], atol=1e-12, rtol=0
    )
    assert_allclose(
        result.u_m_s[0],
        [0, 0, 2 * c0 / 3, star.u_m_s, star.u_m_s, 0, 0],
        atol=1e-12,
        rtol=0,
    )
    assert_allclose(result.q_m2_s, result.h_m * result.u_m_s, atol=0, rtol=0)
    assert result.velocity_valid.all()


def test_ritter_fan_and_front_retain_exact_zero(dry: dict) -> None:
    c0 = math.sqrt(dry["gravity_m_s2"])
    elapsed = 0.5
    result = evaluate_reference(
        dry, np.array([-c0 - 0.1, -c0, 0, 2 * c0, 2 * c0 + 0.1]) * elapsed, [elapsed]
    )
    assert_allclose(result.h_m[0], [1, 1, 4 / 9, 0, 0], atol=1e-14, rtol=0)
    assert_allclose(result.u_m_s[0], [0, 0, 2 * c0 / 3, 0, 0], atol=1e-14, rtol=0)
    assert np.array_equal(result.q_m2_s[0, -2:], [0.0, 0.0])
    assert np.array_equal(result.velocity_valid[0], [True, True, True, False, False])


def test_continuous_fan_edges_and_actual_boundary_horizons(
    wet: dict, dry: dict
) -> None:
    wet_checks, dry_checks = wave_diagnostics(wet), wave_diagnostics(dry)
    for checks in (wet_checks, dry_checks):
        assert checks["identity_checks_passed"]
        assert checks["all_waves_strictly_inside"]
        assert checks["first_boundary_contact_s"] > 0.5
    assert_allclose(wet_checks["fan_tail_depth_continuity_error_m"], 0, atol=1e-12)
    assert_allclose(wet_checks["fan_tail_velocity_continuity_error_m_s"], 0, atol=1e-12)
    assert_allclose(
        wet_checks["positions_at_final_time_m"]["shock"], 1.5525668253341067, atol=1e-11
    )
    assert_allclose(
        dry_checks["positions_at_final_time_m"]["dry_front"],
        3.132091952673165,
        atol=1e-12,
    )
    assert_allclose(
        dry_checks["first_boundary_contact_s"], 0.7981885710176261, atol=1e-12
    )
    unsafe = copy.deepcopy(dry)
    unsafe["final_time_s"] = 0.8
    with pytest.raises(ValueError, match="boundary"):
        evaluate_reference(unsafe, [0], [0.8])


def test_incompatible_reference_assumptions_are_rejected(wet: dict, dry: dict) -> None:
    with pytest.raises(ReferenceError, match="wet_bed"):
        solve_wet_star(dry)
    incompatible = copy.deepcopy(wet)
    incompatible["reference"]["kind"] = "ritter"
    with pytest.raises(ValueError, match="Ritter"):
        evaluate_reference(incompatible, [0], [0.5])
    incompatible = copy.deepcopy(dry)
    incompatible["assumptions"]["friction"] = "manning"
    with pytest.raises(ValueError, match="friction"):
        evaluate_reference(incompatible, [0], [0.5])


def test_reference_axes_and_physical_array_schema(wet: dict) -> None:
    result = evaluate_reference(wet, [-5, 0, 5], [0, 0.113, 0.5])
    assert result.h_m.shape == result.q_m2_s.shape == (3, 3)
    assert result.h_m.dtype == result.q_m2_s.dtype == np.float64
    assert np.isfinite(result.h_m).all() and np.isfinite(result.q_m2_s).all()
    assert_allclose(result.h_m[:, 0], 1, atol=0, rtol=0)
    assert_allclose(result.h_m[:, -1], 0.1, atol=0, rtol=0)
    with pytest.raises(ReferenceError, match="t_s"):
        evaluate_reference(wet, [0], [-0.1])
    with pytest.raises(ReferenceError, match="x_m"):
        evaluate_reference(wet, [float("nan")], [0.5])


@pytest.mark.parametrize("case_file", ["e1_wet.json", "e2_dry.json"])
def test_reference_run_readback_provenance_and_no_overwrite(
    tmp_path: Path, case_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hydro_pinn import references

    root_solves = []
    original_solve = references._wet_star

    def counted_solve(case: dict) -> references.WetStar:
        result = original_solve(case)
        root_solves.append(result)
        return result

    monkeypatch.setattr(references, "_wet_star", counted_solve)
    run = generate_reference_run(
        ROOT / "configs/cases" / case_file,
        ROOT / "configs/verification.json",
        "reference",
        tmp_path / "runs",
    )
    assert json.loads((run / "status.json").read_text())["state"] == "completed"
    resolved = json.loads((run / "resolved_config.json").read_text())
    metadata = json.loads((run / "metadata.json").read_text())
    assert resolved["evaluation"]["partition"] == "implementation_verification"
    assert metadata["dtype"] == "float64" and metadata["device"]["requested"] == "cpu"
    assert metadata["stage_metadata"]["data_origin"] == "analytical_generated"
    assert len(root_solves) == (1 if case_file == "e1_wet.json" else 0)
    if root_solves:
        assert metadata["execution"]["iterations"] == root_solves[0].iterations
        assert (
            metadata["execution"]["function_evaluations"]
            == root_solves[0].function_calls
        )
    with np.load(run / "outputs.npz", allow_pickle=False) as arrays:
        assert arrays["h_m"].shape == arrays["q_m2_s"].shape == (6, 129)
        assert np.array_equal(arrays["t_s"], [0.037, 0.113, 0.229, 0.347, 0.463, 0.5])
        assert arrays["velocity_valid"].dtype == np.bool_
        assert arrays["h_m"].dtype == np.float64
    saved = (run / "outputs.npz").read_bytes()
    with pytest.raises(FileExistsError):
        generate_reference_run(
            ROOT / "configs/cases" / case_file,
            ROOT / "configs/verification.json",
            "reference",
            tmp_path / "runs",
        )
    assert (run / "outputs.npz").read_bytes() == saved


def test_failed_root_records_failure_and_produces_no_final_arrays(
    tmp_path: Path, wet: dict
) -> None:
    wet["reference"]["root_max_iterations"] = 1
    case_path = tmp_path / "nonconverging_case.json"
    case_path.write_text(json.dumps(wet))
    with pytest.raises(ReferenceError, match="did not converge"):
        generate_reference_run(
            case_path,
            ROOT / "configs/verification.json",
            "root-failed",
            tmp_path / "runs",
        )
    run = tmp_path / "runs/root-failed"
    status = json.loads((run / "status.json").read_text())
    assert status["state"] == "failed"
    assert "did not converge" in status["stopping_reason"]
    assert not (run / "outputs.npz").exists()
