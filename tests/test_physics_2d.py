"""Focused equation, chain-factor and independent-state checks for 2D."""

import math

import torch

from hydro_pinn.extensions.model_2d import PlanarPINN
from hydro_pinn.extensions.physics_2d import physical_residuals, residuals
from hydro_pinn.losses import residuals as residuals_1d
from hydro_pinn.physics import Scales


def _scales():
    speed = math.sqrt(9.81)
    return Scales(H_m=1.0, Lc_m=5.0, Uc_m_s=speed, Tc_s=5.0 / speed)


def _smooth(coords):
    x, y, t = coords.unbind(dim=1)
    h = 2 + 0.1 * x + 0.2 * y + 0.3 * t
    u = 0.4 + 0.2 * x - 0.1 * y + 0.05 * t
    v = -0.2 + 0.07 * x + 0.09 * y - 0.03 * t
    return torch.stack((h, h * u, h * v), dim=1)


def _smooth_expected(coords):
    state = _smooth(coords)
    h, qx, qy = state.unbind(dim=1)
    u, v = qx / h, qy / h
    # Closed-form product-rule derivatives, independent of autograd.
    ht, hx, hy = 0.3, 0.1, 0.2
    ut, ux, uy = 0.05, 0.2, -0.1
    vt, vx, vy = -0.03, 0.07, 0.09
    mass = ht + hx * u + h * ux + hy * v + h * vy
    mx = (
        ht * u
        + h * ut
        + hx * u.square()
        + 2 * h * u * ux
        + h * hx
        + hy * u * v
        + h * uy * v
        + h * u * vy
    )
    my = (
        ht * v
        + h * vt
        + hx * u * v
        + h * ux * v
        + h * u * vx
        + hy * v.square()
        + 2 * h * v * vy
        + h * hy
    )
    return torch.stack((mass, mx, my), dim=1)


def test_constant_and_smooth_cross_flux_residuals():
    coords = torch.tensor([[-0.8, 0.7, 0.2], [0.5, -0.4, 0.3]], dtype=torch.float64)

    def constant(c):
        return c.new_tensor([2.0, 0.6, -0.3]).expand(len(c), 3)

    torch.testing.assert_close(
        residuals(constant, coords), torch.zeros_like(coords), rtol=0, atol=0
    )
    torch.testing.assert_close(
        residuals(_smooth, coords), _smooth_expected(coords), rtol=2e-14, atol=2e-14
    )


def test_reduction_to_unchanged_1d_on_smooth_rarefaction_interior():
    # Fan samples have x*/t* in [-0.8,-0.2], away from dam/shock/fan edges.
    xt = torch.tensor([[-0.08, 0.1], [-0.12, 0.2], [-0.06, 0.3]], dtype=torch.float64)

    def fan(c):
        ratio = c[:, 0] / c[:, 1]
        h = (2 - ratio).square() / 9
        u = 2 * (1 + ratio) / 3
        return torch.stack((h, h * u), dim=1)

    def embedded(c):
        state = fan(c[:, (0, 2)])
        return torch.cat((state, torch.zeros_like(state[:, :1])), dim=1)

    xyt = torch.stack((xt[:, 0], torch.tensor([0.1, -0.7, 0.8]), xt[:, 1]), dim=1)
    r1, r2 = residuals_1d(fan, xt), residuals(embedded, xyt)
    torch.testing.assert_close(r2[:, :2], r1, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(r2, torch.zeros_like(r2), rtol=0, atol=3e-14)


def test_physical_chain_factors_and_units_against_direct_si_derivatives():
    scales = _scales()

    class Manufactured:
        def __init__(self):
            self.scales = scales

        def __call__(self, c):
            return _smooth(c)

    coords = torch.tensor([[-2.0, 1.0, 0.2], [3.0, -3.5, 0.4]], dtype=torch.float64)
    c = coords.clone().requires_grad_(True)
    nd = c / c.new_tensor((scales.Lc_m, scales.Lc_m, scales.Tc_s))
    state = _smooth(nd) * c.new_tensor((scales.H_m, scales.Qc_m2_s, scales.Qc_m2_s))
    h, qx, qy = state.unbind(dim=1)
    fx = (qx, qx.square() / h + 0.5 * 9.81 * h.square(), qx * qy / h)
    fy = (qy, qx * qy / h, qy.square() / h + 0.5 * 9.81 * h.square())

    def derivative(field):
        return torch.autograd.grad(field.sum(), c, retain_graph=True)[0]

    direct = torch.stack(
        tuple(
            derivative(state[:, i])[:, 2]
            + derivative(fx[i])[:, 0]
            + derivative(fy[i])[:, 1]
            for i in range(3)
        ),
        dim=1,
    )
    torch.testing.assert_close(
        physical_residuals(Manufactured(), coords), direct, rtol=2e-14, atol=2e-14
    )


def test_model_uses_y_and_independent_transverse_output_with_positive_depth():
    torch.manual_seed(11)
    model = PlanarPINN(_scales(), hidden_layers=(8, 8), dtype=torch.float64)
    coords = torch.tensor(
        [[0.2, -0.8, 0.1], [0.2, 0.8, 0.1]], dtype=torch.float64, requires_grad=True
    )
    state = model(coords)
    assert state.shape == (2, 3)
    assert bool((state[:, 0] > 1e-6).all())
    assert bool((state[0] != state[1]).any())
    dqy = torch.autograd.grad(state[:, 2].sum(), coords)[0]
    assert bool((dqy[:, 1].abs() > 0).all())
    factors = coords.new_tensor((1.0, model.scales.Qc_m2_s, model.scales.Qc_m2_s))
    physical = coords * coords.new_tensor((5.0, 5.0, model.scales.Tc_s))
    torch.testing.assert_close(model.physical_state(physical), state * factors)
