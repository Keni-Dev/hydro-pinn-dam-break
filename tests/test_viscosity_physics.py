"""Necessary derivative/scale/zero-limit checks for constant momentum diffusion."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from hydro_pinn import losses
from hydro_pinn import train as core_train
from hydro_pinn.config import load_config
from hydro_pinn.extensions import viscosity_physics as viscosity
from hydro_pinn.model import ShallowWaterPINN
from hydro_pinn.sampling import generate_validation_samples

ROOT = Path(__file__).resolve().parents[1]
POINTS = np.array([[-0.8, 0.03], [-0.3, 0.11], [0.2, 0.19], [0.7, 0.25], [1.1, 0.31]])


def tolerance(dtype: torch.dtype) -> dict:
    return (
        {"rtol": 1e-10, "atol": 1e-11}
        if dtype == torch.float64
        else {"rtol": 5e-5, "atol": 1e-5}
    )


class SmoothState(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.2, dtype=dtype))
        self.b = nn.Parameter(torch.tensor(-0.13, dtype=dtype))
        self.absent = nn.Parameter(torch.tensor(0.9, dtype=dtype))

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x, t = coords[:, 0], coords[:, 1]
        return torch.stack(
            (2 + 0.1 * torch.sin(x), self.a * x**3 + self.b * t**2 * x), dim=1
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_smooth_second_derivative_sign_and_parameter_graph(dtype: torch.dtype) -> None:
    model = SmoothState(dtype)
    coordinates = torch.tensor(POINTS, dtype=dtype)
    observed = []

    def state(points):
        observed.append(points)
        return model(points)

    dual = viscosity.dual_residuals(state, coordinates)
    assert len(observed) == 1
    assert observed[0].is_leaf and observed[0].requires_grad
    assert coordinates.requires_grad is False
    x, t = coordinates[:, 0], coordinates[:, 1]
    h, hx = 2 + 0.1 * x.sin(), 0.1 * x.cos()
    q, qx = model.a * x**3 + model.b * t**2 * x, 3 * model.a * x**2 + model.b * t**2
    expected = torch.stack(
        (
            qx,
            2 * model.b * t * x
            + 2 * q * qx / h
            - q.square() * hx / h.square()
            + h * hx,
        ),
        dim=1,
    )
    torch.testing.assert_close(dual["inviscid"], expected, **tolerance(dtype))
    torch.testing.assert_close(dual["q_xx"], 6 * model.a * x, **tolerance(dtype))
    assert torch.equal(dual["modified"][:, 0], dual["inviscid"][:, 0])
    difference = dual["modified"][:, 1] - dual["inviscid"][:, 1]
    torch.testing.assert_close(
        difference, -6 * viscosity.NU_STAR * model.a * x, **tolerance(dtype)
    )
    grad_a, grad_b = torch.autograd.grad(
        difference.sum(), (model.a, model.b), retain_graph=True
    )
    torch.testing.assert_close(
        grad_a, -6 * viscosity.NU_STAR * x.sum(), **tolerance(dtype)
    )
    torch.testing.assert_close(grad_b, torch.zeros_like(grad_b), **tolerance(dtype))
    graph_gradient = torch.autograd.grad(dual["q_xx"].sum(), model.a)[0]
    torch.testing.assert_close(graph_gradient, 6 * x.sum(), **tolerance(dtype))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["zero", "constant", "linear"])
def test_constant_and_linear_q_have_safe_zero_second_derivative(dtype, kind) -> None:
    a = nn.Parameter(torch.tensor(0.3, dtype=dtype))
    coords = torch.tensor(POINTS, dtype=dtype)

    def state(points):
        x, t = points[:, 0], points[:, 1]
        q = {
            "zero": torch.zeros_like(x),
            "constant": a.expand_as(x),
            "linear": a * x + 0.2 * t,
        }[kind]
        return torch.stack((torch.ones_like(x) * 2, q), dim=1)

    result = viscosity.dual_residuals(state, coords)
    assert torch.equal(result["q_xx"], torch.zeros(len(coords), dtype=dtype))
    assert torch.equal(result["inviscid"], result["modified"])
    assert torch.isfinite(result["modified"]).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_nonunit_physical_scaling_and_qxx_chain_rule(dtype) -> None:
    # Nonunit H=2m,Lc=3m,Uc=3m/s,Tc=1s, corresponding to g=4.5m/s².
    H, Lc, Uc, Tc, gravity = 2.0, 3.0, 3.0, 1.0, 4.5
    Qc = H * Uc
    nu_physical = viscosity.NU_STAR * Lc * Uc
    model = SmoothState(dtype)
    nd = torch.tensor(POINTS, dtype=dtype)
    physical = (nd * torch.tensor([Lc, Tc], dtype=dtype)).requires_grad_(True)
    state = model(physical / torch.tensor([Lc, Tc], dtype=dtype))
    h, q = H * state[:, 0], Qc * state[:, 1]

    def derivative(value):
        return torch.autograd.grad(
            value,
            physical,
            torch.ones_like(value),
            create_graph=True,
            retain_graph=True,
        )[0]

    dh, dq = derivative(h), derivative(q)
    qxx_physical = derivative(dq[:, 0])[:, 0]
    physical_flux = q.square() / h + 0.5 * gravity * h.square()
    flux_dx = derivative(physical_flux)[:, 0]
    physical_residual = torch.stack(
        (dh[:, 1] + dq[:, 0], dq[:, 1] + flux_dx - nu_physical * qxx_physical),
        dim=1,
    )
    expected_qxx = (Qc / Lc**2) * 6 * model.a * nd[:, 0]
    torch.testing.assert_close(qxx_physical, expected_qxx, **tolerance(dtype))
    scaled = physical_residual / torch.tensor([H / Tc, Qc / Tc], dtype=dtype)
    observed = viscosity.viscosity_residuals(model, nd, viscosity.NU_STAR)
    torch.testing.assert_close(scaled, observed, **tolerance(dtype))


def small_samples() -> dict:
    points = POINTS.copy()
    return {
        "interior_xt_nd": points,
        "initial_xt_nd": np.column_stack((points[:, 0], np.zeros(len(points)))),
        "initial_hq_nd": np.column_stack(
            (np.full(len(points), 2.0), np.zeros(len(points)))
        ),
        "boundary_xt_nd": points[[0, -1]],
        "boundary_hq_nd": np.array([[2.0, 0.0], [2.0, 0.0]]),
    }


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("fixture", ["smooth", "actual_model"])
def test_zero_delegates_exact_residual_components_gradients_and_adam(
    dtype,
    fixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.set_num_threads(4)
    if fixture == "smooth":
        baseline, extension = SmoothState(dtype), SmoothState(dtype)
        samples = small_samples()
    else:
        case = load_config(ROOT / "configs/cases/e1_wet.json")
        training = load_config(ROOT / "configs/training/baseline.json")
        torch.manual_seed(11)
        baseline = ShallowWaterPINN(case, training, dtype="float32")
        extension = ShallowWaterPINN(case, training, dtype="float32")
        extension.load_state_dict(baseline.state_dict())
        # Cast the same float32-representable initial weights to either dtype.
        baseline, extension = baseline.to(dtype=dtype), extension.to(dtype=dtype)
        validation = generate_validation_samples(case, training, 11)
        indices = np.floor(np.arange(16) * 364 / 15).astype(int)
        samples = {
            **validation,
            "interior_xt_nd": validation["interior_xt_nd"][indices],
        }
        assert [
            len(samples[k])
            for k in ("interior_xt_nd", "initial_xt_nd", "boundary_xt_nd")
        ] == [16, 72, 10]

    def forbidden_second_derivative(*args, **kwargs):
        raise AssertionError("nu=0 must never enter second-derivative path")

    monkeypatch.setattr(viscosity, "_dual", forbidden_second_derivative)
    coords = torch.tensor(samples["interior_xt_nd"], dtype=dtype)
    assert torch.equal(
        losses.residuals(baseline, coords),
        viscosity.viscosity_residuals(extension, coords, 0),
    )
    values, gradients, norms = {}, {}, {}
    for component in losses.COMPONENTS:
        value, gradient = core_train.component_gradients(
            baseline, samples, component, None
        )
        actual_value, actual_gradient = viscosity.component_gradients(
            extension, samples, component, 0
        )
        assert value == actual_value
        assert all(
            torch.equal(a, b) for a, b in zip(gradient, actual_gradient, strict=True)
        )
        values[component], gradients[component] = value, gradient
        norms[component] = losses.gradient_rms(gradient, baseline.parameters())
    total = sum(viscosity.WEIGHTS[name] * values[name] for name in losses.COMPONENTS)
    weighted = [
        sum(viscosity.WEIGHTS[name] * gradients[name][i] for name in losses.COMPONENTS)
        for i, _ in enumerate(baseline.parameters())
    ]
    actual_total, actual_values, actual_norms, actual_weighted = (
        viscosity.objective_gradients(extension, samples, 0)
    )
    assert total == actual_total and values == actual_values and norms == actual_norms
    assert all(
        torch.equal(a, b) for a, b in zip(weighted, actual_weighted, strict=True)
    )
    optimizers = [
        torch.optim.Adam(model.parameters(), lr=0.001)
        for model in (baseline, extension)
    ]
    for optimizer, model, grads in zip(
        optimizers, (baseline, extension), (weighted, actual_weighted), strict=True
    ):
        for parameter, grad in zip(model.parameters(), grads, strict=True):
            parameter.grad = grad
        optimizer.step()
    assert all(
        torch.equal(a, b)
        for a, b in zip(baseline.parameters(), extension.parameters(), strict=True)
    )
    for a, b in zip(
        optimizers[0].state.values(), optimizers[1].state.values(), strict=True
    ):
        assert a.keys() == b.keys()
        assert all(
            torch.equal(a[k], b[k]) if isinstance(a[k], torch.Tensor) else a[k] == b[k]
            for k in a
        )
    if fixture == "smooth":
        assert torch.equal(weighted[-1], torch.zeros_like(weighted[-1]))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_positive_objective_normalization_validation_and_ic_bc_delegation(
    dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, samples = SmoothState(dtype), small_samples()
    coords = torch.tensor(samples["interior_xt_nd"], dtype=dtype)
    residual = viscosity.viscosity_residuals(model, coords, viscosity.NU_STAR)
    expected_pde = float(residual.square().sum(dim=1).mean().detach())
    total, values, _, gradients = viscosity.objective_gradients(
        model, samples, viscosity.NU_STAR
    )
    assert values["pde"] == expected_pde
    for component in ("ic", "bc"):
        core_value, core_grads = core_train.component_gradients(
            model, samples, component
        )
        value, grads = viscosity.component_gradients(
            model, samples, component, viscosity.NU_STAR
        )
        assert value == core_value
        assert all(torch.equal(a, b) for a, b in zip(grads, core_grads, strict=True))
    assert total == values["pde"] + 10 * values["ic"] + 10 * values["bc"]
    assert len(gradients) == len(list(model.parameters()))
    assert all(torch.isfinite(g).all() for g in gradients)
    assert all(parameter.grad is None for parameter in model.parameters())

    def forbidden_parameter_gradients(*args, **kwargs):
        raise AssertionError(
            "validation must not compute component parameter gradients"
        )

    monkeypatch.setattr(viscosity, "component_gradients", forbidden_parameter_gradients)
    with torch.no_grad():
        report = viscosity.validation(model, samples, viscosity.NU_STAR)
    assert report["modified_pde"] == expected_pde
    assert report["modified_total"] == total
    assert report["inviscid_total"] == core_train._validation(
        model, samples, viscosity.WEIGHTS, None
    )
    assert all(parameter.grad is None for parameter in model.parameters())
