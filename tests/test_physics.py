"""Scientific checks for the shared source-free shallow-water representation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from hydro_pinn.config import load_config
from hydro_pinn.physics import (
    PhysicsError,
    Scales,
    conservative_flux,
    dimensionless_flux,
    state_from_depth_velocity,
    velocity_from_state,
)

ROOT = Path(__file__).resolve().parents[1]
SCALES = Scales.from_case(load_config(ROOT / "configs/cases/e1_wet.json"))
ATOL = 1e-12
RTOL = 1e-12


def array(values: object, backend: str):
    if backend == "torch":
        return torch.tensor(values, dtype=torch.float64, device="cpu")
    return np.asarray(values, dtype=np.float64)


def assert_close(actual, expected) -> None:
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(
            actual,
            torch.as_tensor(expected, dtype=torch.float64),
            atol=ATOL,
            rtol=RTOL,
        )
    else:
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)


def test_both_core_cases_share_the_declared_characteristic_scales() -> None:
    dry_scales = Scales.from_case(load_config(ROOT / "configs/cases/e2_dry.json"))
    assert SCALES == dry_scales
    assert SCALES.H_m == 1.0
    assert SCALES.Lc_m == 5.0
    assert_close(SCALES.Uc_m_s, np.sqrt(9.81))
    assert_close(SCALES.Tc_s, 5 / np.sqrt(9.81))
    assert_close(SCALES.Qc_m2_s, SCALES.H_m * SCALES.Uc_m_s)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_physical_scaled_round_trips(backend: str) -> None:
    examples = {
        "x": [-5.0, -1.0, 0.0, 5.0],
        "t": [0.0, 0.1, 0.25, 0.5],
        "h": [0.0, 0.1, 1.0, 2.0],
        "u": [-2.0, 0.0, 1.0, 2.0],
        "q": [-0.2, 0.0, 0.5, 2.0],
    }
    for quantity, values in examples.items():
        physical = array(values, backend)
        scaled = getattr(SCALES, f"{quantity}_to_scaled")(physical)
        recovered = getattr(SCALES, f"{quantity}_to_physical")(scaled)
        assert_close(recovered, physical)
        if backend == "torch":
            assert recovered.dtype == torch.float64
            assert recovered.device.type == "cpu"


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_q_equals_hu_before_and_after_scaling(backend: str) -> None:
    h, u = array([1.0, 0.1, 0.0], backend), array([2.0, -1.0, 0.0], backend)
    state = state_from_depth_velocity(h, u)
    assert state.shape == (3, 2)
    assert_close(state[..., 1], h * u)
    scaled = SCALES.state_to_scaled(state)
    assert_close(scaled[..., 1], SCALES.h_to_scaled(h) * SCALES.u_to_scaled(u))
    assert_close(SCALES.state_to_physical(scaled), state)
    velocity, wet = velocity_from_state(state)
    assert_close(velocity, u)
    assert wet.tolist() == [True, True, False]


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_hand_flux_and_unit_gravity_are_distinct(backend: str) -> None:
    state = array([2.0, 6.0], backend)
    physical = conservative_flux(state)
    nondimensional = dimensionless_flux(state)
    assert physical.shape == (2,)
    # q²/h=36/2=18; gh²/2=19.62 in SI and 2 with unit gravity.
    assert_close(physical, [6.0, 37.62])
    assert_close(nondimensional, [6.0, 20.0])


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_unit_gravity_flux_matches_characteristic_scaling(backend: str) -> None:
    physical = array([[1.0, 2.0], [0.1, -0.2], [0.0, 0.0]], backend)
    physical_flux = conservative_flux(physical)
    scaled_flux = dimensionless_flux(SCALES.state_to_scaled(physical))
    assert_close(scaled_flux[..., 0], physical_flux[..., 0] / SCALES.Qc_m2_s)
    assert_close(
        scaled_flux[..., 1], physical_flux[..., 1] / (SCALES.H_m * SCALES.Uc_m_s**2)
    )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_exact_dry_state_has_zero_flux_and_masked_velocity(backend: str) -> None:
    dry = array([[0.0, 0.0]], backend)
    # NumPy raises on any invalid intermediate quotient in this block.
    with np.errstate(divide="raise", invalid="raise"):
        assert_close(conservative_flux(dry), [[0.0, 0.0]])
        velocity, wet = velocity_from_state(dry)
    assert_close(velocity, [0.0])
    assert wet.tolist() == [False]


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([1.0, 0.0, 0.0], "shape"),
        ([-0.1, 0.0], "nonnegative"),
        ([1.0, float("inf")], "finite"),
        ([0.0, 0.1], "dry state"),
    ],
)
def test_invalid_states_are_rejected(
    backend: str, values: list[float], message: str
) -> None:
    state = array(values, backend)
    for operation in (conservative_flux, velocity_from_state):
        with pytest.raises(PhysicsError, match=message):
            operation(state)


def test_depth_velocity_construction_rejects_mismatched_shapes() -> None:
    with pytest.raises(PhysicsError, match="same shape"):
        state_from_depth_velocity(np.ones(3), np.zeros((3, 1)))


def test_tensor_flux_derivative_matches_the_hand_jacobian() -> None:
    state = torch.tensor([2.0, 6.0], dtype=torch.float64, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(conservative_flux, state)
    # dF2/dh=-q²/h²+gh=-9+19.62; dF2/dq=2q/h=6.
    assert_close(jacobian, [[0.0, 1.0], [10.62, 6.0]])


def test_tensor_state_construction_and_coordinate_scales_preserve_gradients() -> None:
    h = torch.tensor([1.0, 0.1], dtype=torch.float64, requires_grad=True)
    u = torch.tensor([2.0, -1.0], dtype=torch.float64, requires_grad=True)
    q = state_from_depth_velocity(h, u)[..., 1]
    dh, du = torch.autograd.grad(q.sum(), (h, u))
    assert_close(dh, u)
    assert_close(du, h)
    x = torch.tensor([-1.0, 2.0], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([0.1, 0.4], dtype=torch.float64, requires_grad=True)
    expression = SCALES.x_to_scaled(x) ** 2 + 3 * SCALES.t_to_scaled(t)
    dx, dt = torch.autograd.grad(expression.sum(), (x, t))
    assert_close(dx, 2 * x / SCALES.Lc_m**2)
    assert_close(dt, torch.full_like(t, 3 / SCALES.Tc_s))


def test_tensor_dry_branch_has_finite_backward_values() -> None:
    state = torch.tensor(
        [[0.0, 0.0], [1.0, 0.2]], dtype=torch.float64, requires_grad=True
    )
    gradient = torch.autograd.grad(conservative_flux(state).sum(), state)[0]
    assert bool(torch.isfinite(gradient).all())
    assert_close(gradient[0], [0.0, 1.0])
