"""Independent derivatives, complete means, gradient norms, and update rule."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose
from torch import Tensor, nn

from hydro_pinn.config import load_config
from hydro_pinn.losses import (
    LossError,
    component_loss,
    component_losses,
    gradient_rms,
    iter_component_losses,
    residuals,
    update_adaptive_weights,
)

ROOT = Path(__file__).resolve().parents[1]


class SmoothState(nn.Module):
    """Small explicit state with a nontrivial parameter derivative oracle."""

    def __init__(self) -> None:
        super().__init__()
        self.a = nn.Parameter(torch.tensor(0.3, dtype=torch.float64))
        self.b = nn.Parameter(torch.tensor(-0.2, dtype=torch.float64))

    def forward(self, coords: Tensor) -> Tensor:
        x, t = coords[:, 0], coords[:, 1]
        h = 2 + self.a * x + t.square()
        u = self.b * t + x
        return torch.stack((h, h * u), dim=1)


@pytest.fixture
def samples() -> dict[str, np.ndarray]:
    return {
        "interior_xt_nd": np.array(
            [
                [-0.8, 0.03],
                [-0.5, 0.07],
                [-0.2, 0.1],
                [0.1, 0.15],
                [0.3, 0.19],
                [0.5, 0.22],
                [0.7, 0.3],
            ],
            dtype=np.float64,
        ),
        "initial_xt_nd": np.array([[-0.8, 0], [-0.4, 0], [0.1, 0], [0.4, 0], [0.8, 0]]),
        "initial_hq_nd": np.array([[1, 0], [1, 0], [0.1, 0], [0.1, 0], [0.1, 0]]),
        "boundary_xt_nd": np.array([[-1, 0.04], [-1, 0.15], [1, 0.24]]),
        "boundary_hq_nd": np.array([[1, 0], [1, 0], [0.1, 0]]),
    }


def test_constant_and_hand_smooth_residuals() -> None:
    coords = torch.tensor([[-0.4, 0.1], [0.1, 0.2], [0.5, 0.3]], dtype=torch.float64)

    def constant(points: Tensor) -> Tensor:
        return torch.tensor([2.0, 0.8], dtype=points.dtype).expand(len(points), 2)

    assert_allclose(residuals(constant, coords).detach(), 0, atol=0, rtol=0)

    def smooth(points: Tensor) -> Tensor:
        x, t = points[:, 0], points[:, 1]
        h, u = 2 + x + t, x - 2 * t
        return torch.stack((h, h * u), dim=1)

    x, t = coords[:, 0], coords[:, 1]
    h, u = 2 + x + t, x - 2 * t
    # h_t=1; q_x=u+h; q_t=u-2h; (h*u²+h²/2)_x=u²+2hu+h.
    expected = torch.stack((1 + u + h, u.square() + u + 2 * h * u - h), dim=1)
    assert_allclose(residuals(smooth, coords).detach(), expected, atol=2e-15, rtol=0)
    with torch.no_grad():
        assert_allclose(
            residuals(smooth, coords).detach(), expected, atol=2e-15, rtol=0
        )


def test_residual_parameter_gradient_matches_finite_difference() -> None:
    model = SmoothState()
    coords = torch.tensor([[-0.3, 0.1], [0.2, 0.21], [0.6, 0.3]], dtype=torch.float64)
    loss = residuals(model, coords).square().sum(dim=1).mean()
    derivative = float(torch.autograd.grad(loss, model.a)[0])
    centre, delta = float(model.a.detach()), 1e-6
    with torch.no_grad():
        model.a.fill_(centre + delta)
    plus = float(residuals(model, coords).square().sum(dim=1).mean().detach())
    with torch.no_grad():
        model.a.fill_(centre - delta)
    minus = float(residuals(model, coords).square().sum(dim=1).mean().detach())
    with torch.no_grad():
        model.a.fill_(centre)
    assert_allclose(derivative, (plus - minus) / (2 * delta), rtol=2e-8, atol=1e-8)


def test_complete_means_and_uneven_chunk_gradients(samples: dict) -> None:
    model = SmoothState()
    params = list(model.parameters())
    full = component_losses(model, samples)
    for component in ("pde", "ic", "bc"):
        expected_value = float(full[component].detach())
        full_grads = torch.autograd.grad(full[component], params)
        chunked = component_loss(model, samples, component, chunk_size=2)
        assert_allclose(float(chunked.detach()), expected_value, atol=2e-14, rtol=1e-14)
        summed_grads = [torch.zeros_like(parameter) for parameter in params]
        fractions = []
        for fraction, chunk in iter_component_losses(model, samples, component, 2):
            fractions.append(fraction)
            gradients = torch.autograd.grad(fraction * chunk, params)
            for total, gradient in zip(summed_grads, gradients, strict=True):
                total.add_(gradient)
        assert_allclose(sum(fractions), 1, atol=2e-16, rtol=0)
        for complete, accumulated in zip(full_grads, summed_grads, strict=True):
            assert_allclose(accumulated, complete, atol=3e-14, rtol=2e-14)
        assert_allclose(
            gradient_rms(summed_grads, params),
            gradient_rms(full_grads, params),
            atol=2e-14,
        )
    # IC/BC average the per-point sum, not half that sum or equally weighted side means.
    for name, prefix in (("ic", "initial"), ("bc", "boundary")):
        points = torch.tensor(samples[f"{prefix}_xt_nd"], dtype=torch.float64)
        target = torch.tensor(samples[f"{prefix}_hq_nd"], dtype=torch.float64)
        direct = (model(points) - target).square().sum(dim=1).mean()
        assert_allclose(
            float(full[name].detach()), float(direct.detach()), atol=0, rtol=0
        )


def test_physics_only_schema_refuses_interior_labels(samples: dict) -> None:
    samples["interior_hq_nd"] = np.full_like(samples["interior_xt_nd"], 999)
    with pytest.raises(LossError, match="IC/BC targets only"):
        component_losses(SmoothState(), samples)


def test_rms_uses_same_parameter_count_and_detaches_statistics() -> None:
    parameters = [
        nn.Parameter(torch.zeros(2, dtype=torch.float64)),
        nn.Parameter(torch.zeros(3, dtype=torch.float64)),
    ]
    gradient = torch.tensor([3.0, 4.0], dtype=torch.float64, requires_grad=True)
    norm = gradient_rms([gradient, None], parameters)
    assert type(norm) is float
    assert_allclose(norm, np.sqrt(25 / 5), atol=0, rtol=0)
    assert gradient.grad is None
    assert gradient_rms([None, None], parameters) == 0.0
    with pytest.raises(LossError, match="finite"):
        gradient_rms([torch.tensor([float("nan"), 0]), None], parameters)


def test_adaptive_hand_schedule_clipping_and_nonfinite() -> None:
    weighting = load_config(ROOT / "configs/training/adaptive.json")["weighting"]
    weights = {"pde": 1.0, "ic": 10.0, "bc": 10.0}
    norms = {"pde": 4.0, "ic": 2.0, "bc": 0.0}
    assert update_adaptive_weights(weights, {}, 99, weighting) == weights
    updated = update_adaptive_weights(weights, norms, 100, weighting)
    assert_allclose(updated["ic"], 0.9 * 10 + 0.1 * (4 / (2 + 1e-12)), atol=2e-15)
    assert_allclose(updated["bc"], 19.0, atol=2e-15)
    assert updated["pde"] == 1.0 and weights["ic"] == 10.0
    assert update_adaptive_weights(updated, {}, 101, weighting) == updated
    low_norms = {"pde": 0.0, "ic": 2.0, "bc": 2.0}
    second = update_adaptive_weights(updated, low_norms, 200, weighting)
    assert_allclose(second["ic"], 0.9 * updated["ic"] + 0.1 * 0.01, atol=2e-15)
    assert_allclose(second["bc"], 0.9 * 19 + 0.1 * 0.01, atol=2e-15)
    detached = update_adaptive_weights(
        weights,
        {"pde": torch.tensor(4.0, requires_grad=True), "ic": 2.0, "bc": 2.0},
        100,
        weighting,
    )
    assert all(type(value) is float for value in detached.values())
    fixed = load_config(ROOT / "configs/training/baseline.json")["weighting"]
    assert update_adaptive_weights(weights, {}, 100, fixed) == weights
    with pytest.raises(LossError, match="finite"):
        update_adaptive_weights(
            weights, {"pde": float("inf"), "ic": 2, "bc": 2}, 100, weighting
        )
