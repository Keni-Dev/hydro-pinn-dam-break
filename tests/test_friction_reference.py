"""Material checks for the separately named wet HLLE/Manning reference."""

import copy

import numpy as np
import pytest

from hydro_pinn.config import load_config
from hydro_pinn.extensions.friction_reference import (
    FrictionReferenceError,
    conservative_project,
    friction_source_step,
    hlle_flux,
    solve_reference,
    zero_n_reference_check,
)


def test_exact_source_both_signs_zero_identity_and_semigroup():
    state = np.array([[0.4, 0.3], [0.4, -0.3], [0.4, 0.0]])
    actual = friction_source_step(state, 0.03, 0.2)
    coefficient = 9.81 * 0.03**2 / 0.4 ** (7 / 3)
    expected = state[:, 1] / (1 + coefficient * np.abs(state[:, 1]) * 0.2)
    np.testing.assert_allclose(actual[:, 1], expected, rtol=2e-15)
    np.testing.assert_array_equal(actual[:, 0], state[:, 0])
    assert np.all(np.abs(actual[:, 1]) <= np.abs(state[:, 1]))
    np.testing.assert_array_equal(friction_source_step(state, 0.0, 0.2), state)
    np.testing.assert_allclose(
        friction_source_step(friction_source_step(state, 0.03, 0.07), 0.03, 0.13),
        actual,
        rtol=2e-15,
    )
    # Independent solve_ivp integrates the source ODE, without using this update.
    from scipy.integrate import solve_ivp

    for row in state:
        integrated = solve_ivp(
            lambda t, y: [-coefficient * y[0] * abs(y[0])],
            [0, 0.2],
            [row[1]],
            rtol=1e-12,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            integrated.y[0, -1],
            friction_source_step(row, 0.03, 0.2)[1],
            rtol=1e-11,
            atol=1e-14,
        )


def test_flat_constant_still_water_and_native_budgets():
    case = copy.deepcopy(load_config("configs/cases/e1_wet.json"))
    case["right_depth_m"] = 1.0
    case["boundary"]["right_h_m"] = 1.0
    arr = solve_reference(case, 0.03, 80, 0.45, np.array([0.07, 0.2, 0.5]))
    np.testing.assert_array_equal(arr["h_m"], np.ones((3, 80)))
    np.testing.assert_array_equal(arr["q_m2_s"], np.zeros((3, 80)))
    np.testing.assert_array_equal(arr["step_source_impulse_m3_s"], 0.0)
    np.testing.assert_array_equal(arr["step_mass_budget_defect_m2"], 0.0)
    np.testing.assert_array_equal(arr["step_momentum_budget_defect_m3_s"], 0.0)


def test_dam_native_mass_and_momentum_budget_including_actual_drag():
    case = load_config("configs/cases/e1_wet.json")
    arr = solve_reference(case, 0.03, 200, 0.45, np.array([0.1, 0.5]))
    dt = arr["step_dt_s"]
    boundary = np.cumsum(
        dt[:, None]
        * (arr["step_right_numerical_flux"] - arr["step_left_numerical_flux"]),
        axis=0,
    )
    mass_defect = arr["step_mass_m2"] - 5.5 + boundary[:, 0]
    momentum_defect = (
        arr["step_momentum_m3_s"]
        + boundary[:, 1]
        - np.cumsum(arr["step_source_impulse_m3_s"])
    )
    assert np.max(np.abs(mass_defect)) < 2e-13
    assert np.max(np.abs(momentum_defect)) < 2e-13
    assert arr["cumulative_source_impulse_m3_s"][-1] < -0.005
    assert arr["step_running_minimum_h_m"].min() >= 0.1 - 1e-14
    assert arr["step_cfl"].max() <= 0.45 + 1e-14


def test_refined_zero_n_hlle_approaches_exact_e1():
    check = zero_n_reference_check(load_config("configs/cases/e1_wet.json"))
    assert check["h_decreases"] and check["q_decreases"], check
    assert check["errors"][-1]["h_l1_m"] < 0.003
    assert check["errors"][-1]["q_l1_m2_s"] < 0.006


def test_conservative_projection_preserves_integrals_and_rejects_bad_states():
    src = np.array([0.0, 0.2, 0.7, 1.0])
    dst = np.array([0.0, 0.1, 0.8, 1.0])
    values = np.array([[2.0, 3.0, -1.0], [1.0, 5.0, 4.0]])
    result = conservative_project(values, src, dst)
    np.testing.assert_allclose(result @ np.diff(dst), values @ np.diff(src), atol=1e-15)
    with pytest.raises(FrictionReferenceError):
        hlle_flux(np.array([[0.0, 0.0]]), np.array([[1.0, 0.0]]))
    with pytest.raises(FrictionReferenceError):
        friction_source_step(np.array([[np.nan, 0.0]]), 0.03, 0.1)
    case = load_config("configs/cases/e1_wet.json")
    case["right_depth_m"] = 0.0
    with pytest.raises(FrictionReferenceError):
        solve_reference(case, 0.03, 80, 0.45, np.array([0.1]))
