"""Independent wet Manning source/scale checks and unchanged-core n=0 limit."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from torch import Tensor, nn

from hydro_pinn.config import load_config
from hydro_pinn.extensions.friction_physics import (
    dimensionless_coefficient,
    friction_loss_components,
    friction_residuals,
    manning_slope,
    momentum_source,
)
from hydro_pinn.losses import LossError, component_losses, residuals
from hydro_pinn.model import ShallowWaterPINN
from hydro_pinn.physics import PhysicsError, Scales

ROOT = Path(__file__).resolve().parents[1]


def _samples() -> dict[str, np.ndarray]:
    return {
        "interior_xt_nd": np.array(
            [
                [-0.7, 0.03],
                [-0.3, 0.1],
                [0.2, 0.15],
                [0.4, 0.21],
                [0.8, 0.3],
            ]
        ),
        "initial_xt_nd": np.array([[-0.8, 0], [-0.3, 0], [0.2, 0], [0.7, 0]]),
        "initial_hq_nd": np.array([[1.0, 0], [1.0, 0], [0.1, 0], [0.1, 0]]),
        "boundary_xt_nd": np.array([[-1.0, 0.03], [-1.0, 0.2], [1.0, 0.1], [1.0, 0.3]]),
        "boundary_hq_nd": np.array([[1.0, 0], [1.0, 0], [0.1, 0], [0.1, 0]]),
    }


class SmoothState(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.3, dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor(-0.2, dtype=torch.float64))

    def forward(self, coords: Tensor) -> Tensor:
        x, t = coords[:, 0], coords[:, 1]
        h, u = 2 + self.a * x + t.square(), self.b * t + x
        return torch.stack((h, h * u), dim=1)


def test_physical_source_sign_units_zero_and_scaled_chain_factors() -> None:
    h = np.array([0.4, 0.4, 0.4])
    q = np.array([0.7, -0.7, 0.0])
    n, g = 0.03, 9.81
    slope = manning_slope(h, q, n)
    source = momentum_source(h, q, n, g)
    assert slope[0] > 0 and slope[1] < 0 and slope[2] == 0
    assert source[0] < 0 and source[1] > 0 and source[2] == 0
    assert_allclose(source, -g * h * slope, rtol=3e-16, atol=0)
    # SI dimension exponents (m,s): n²*q²/h^(10/3)=(0,0);
    # multiplying by g*h gives momentum-source dimensions (2,-2).
    assert_allclose(
        2 * np.array([-1 / 3, 1]) + 2 * np.array([2, -1]) - (10 / 3) * np.array([1, 0]),
        [0, 0],
        atol=1e-15,
    )
    assert_allclose(np.array([1, -2]) + np.array([1, 0]), [2, -2])
    # Nonunit H and L expose an omitted H^(4/3) or L chain factor.
    H, L = 2.3, 7.1
    U = np.sqrt(g * H)
    scales = Scales(H, L, U, L / U)
    h_nd, q_nd = h / H, q / scales.Qc_m2_s
    coefficient = dimensionless_coefficient(scales, n, g)
    expected = -coefficient * q_nd * abs(q_nd) / h_nd ** (7 / 3)
    assert_allclose(source * scales.Tc_s / scales.Qc_m2_s, expected, rtol=1e-15)
    assert_allclose(momentum_source(h, q, 0), 0, atol=0, rtol=0)
    for bad_h in (np.array([0.0]), np.array([-0.1]), np.array([np.nan])):
        with pytest.raises(PhysicsError, match="strictly positive|finite"):
            momentum_source(bad_h, np.array([0.0]), n)

    with pytest.raises(PhysicsError, match="nonfinite Manning source"):
        momentum_source(np.array([1e-300]), np.array([1.0]), n)


def test_manufactured_residual_and_parameter_derivatives() -> None:
    model = SmoothState()
    coords = torch.tensor([[-0.3, 0.1], [0.2, 0.21], [0.6, 0.3]], dtype=torch.float64)
    scales = Scales(2.3, 7.1, np.sqrt(9.81 * 2.3), 7.1 / np.sqrt(9.81 * 2.3))
    roughness = torch.tensor(0.03, dtype=torch.float64, requires_grad=True)
    actual = friction_residuals(model, coords, scales, roughness)
    x, t = coords[:, 0], coords[:, 1]
    h, u = 2 + model.a * x + t.square(), model.b * t + x
    q = h * u
    # Direct derivatives: h_t=2t, q_x=a*u+h, q_t=2t*u+h*b,
    # d_x(h*u²+h²/2)=a*u²+2hu+h*a.
    C = 9.81 * roughness.square() * scales.Lc_m / scales.H_m ** (4 / 3)
    expected = torch.stack(
        (
            2 * t + model.a * u + h,
            2 * t * u
            + h * model.b
            + model.a * u.square()
            + 2 * h * u
            + h * model.a
            + C * q * abs(q) / h ** (7 / 3),
        ),
        dim=1,
    )
    assert_allclose(actual.detach(), expected.detach(), rtol=1e-15, atol=1e-15)
    actual_loss = actual.square().sum(dim=1).mean()
    expected_loss = expected.square().sum(dim=1).mean()
    actual_grad = torch.autograd.grad(actual_loss, (model.a, model.b, roughness))
    expected_grad = torch.autograd.grad(expected_loss, (model.a, model.b, roughness))
    for actual_value, expected_value in zip(actual_grad, expected_grad, strict=True):
        assert_allclose(actual_value, expected_value, rtol=2e-15, atol=2e-14)
    # Independent central differences include source dependence on network h,q.
    for parameter, exact in zip(
        (model.a, model.b, roughness), actual_grad, strict=True
    ):
        centre, delta = float(parameter.detach()), 1e-6
        values = []
        for sign in (1, -1):
            with torch.no_grad():
                parameter.fill_(centre + sign * delta)
            values.append(
                float(
                    friction_residuals(model, coords, scales, roughness)
                    .square()
                    .sum(dim=1)
                    .mean()
                    .detach()
                )
            )
        with torch.no_grad():
            parameter.fill_(centre)
        assert_allclose(
            exact, (values[0] - values[1]) / (2 * delta), rtol=3e-8, atol=1e-8
        )
    with torch.no_grad():
        assert_allclose(
            friction_residuals(model, coords, scales, 0.03).detach(),
            expected.detach(),
            rtol=1e-15,
            atol=1e-15,
        )


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_zero_n_exact_core_residual_objective_all_gradients_and_one_adam(
    dtype: str,
) -> None:
    case = load_config(ROOT / "configs/cases/e1_wet.json")
    training = load_config(ROOT / "configs/training/baseline.json")
    torch.manual_seed(11)
    core = ShallowWaterPINN(case, training, dtype=dtype)
    friction = copy.deepcopy(core)
    coords = torch.tensor(_samples()["interior_xt_nd"], dtype=getattr(torch, dtype))
    assert torch.equal(
        residuals(core, coords), friction_residuals(friction, coords, core.scales, 0)
    )
    left = component_losses(core, _samples())
    right = friction_loss_components(friction, _samples(), core.scales, 0)
    for name in ("pde", "ic", "bc"):
        assert torch.equal(left[name], right[name])
    weights = {"pde": 1.0, "ic": 10.0, "bc": 10.0}
    loss_core = sum(weights[name] * left[name] for name in weights)
    loss_friction = sum(weights[name] * right[name] for name in weights)
    assert torch.equal(loss_core, loss_friction)
    optimizer_core = torch.optim.Adam(core.parameters(), lr=0.001)
    optimizer_friction = torch.optim.Adam(friction.parameters(), lr=0.001)
    loss_core.backward()
    loss_friction.backward()
    for (name_a, a), (name_b, b) in zip(
        core.named_parameters(), friction.named_parameters(), strict=True
    ):
        assert name_a == name_b and a.grad is not None and b.grad is not None
        assert torch.equal(a.grad, b.grad), name_a
    optimizer_core.step()
    optimizer_friction.step()
    for name, value in core.state_dict().items():
        assert torch.equal(value, friction.state_dict()[name]), name
    for a, b in zip(
        optimizer_core.state.values(), optimizer_friction.state.values(), strict=True
    ):
        for name in a:
            assert torch.equal(a[name], b[name]), name


def test_positive_n_cpu32_complete_gradients_chunks_and_label_rejection() -> None:
    case = load_config(ROOT / "configs/cases/e1_wet.json")
    training = load_config(ROOT / "configs/training/baseline.json")
    torch.manual_seed(11)
    model = ShallowWaterPINN(case, training, dtype="float32")
    full = friction_loss_components(model, _samples(), model.scales, 0.03)
    chunked = friction_loss_components(
        model, _samples(), model.scales, 0.03, chunk_size=2
    )
    for name in full:
        torch.testing.assert_close(full[name], chunked[name], rtol=2e-6, atol=2e-7)
    sum(
        full[name] * weight for name, weight in {"pde": 1, "ic": 10, "bc": 10}.items()
    ).backward()
    assert all(
        parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    bad = _samples()
    bad["interior_hq_nd"] = np.zeros_like(bad["interior_xt_nd"])
    with pytest.raises(LossError, match="IC/BC targets only"):
        friction_loss_components(model, bad, model.scales, 0.03)
