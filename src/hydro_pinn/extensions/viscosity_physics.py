"""Scaled constant momentum diffusion, separate from the frozen inviscid core.

The continuity operator is unchanged. Momentum uses r_q - nu_star*q_star_xx
with derivatives in the already scaled x/Lc coordinate. In SI units,
nu = nu_star*Lc*Uc [m²/s]; no extra affine map or chain factor is applied here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import torch
from torch import Tensor, nn

from hydro_pinn import losses
from hydro_pinn import train as core_train

NU_STAR = 0.001
WEIGHTS = {"pde": 1.0, "ic": 10.0, "bc": 10.0}


def _coefficient(nu_star: float) -> float:
    if (
        isinstance(nu_star, bool)
        or not isinstance(nu_star, (int, float))
        or not math.isfinite(nu_star)
        or nu_star < 0
    ):
        raise losses.LossError("nu_star must be a finite nonnegative scalar")
    return float(nu_star)


def _coordinates(coords_nd: Tensor) -> Tensor:
    if (
        not isinstance(coords_nd, Tensor)
        or not coords_nd.is_floating_point()
        or coords_nd.ndim != 2
        or coords_nd.shape[1] != 2
        or len(coords_nd) == 0
        or not bool(torch.isfinite(coords_nd).all())
    ):
        raise losses.LossError(
            "coords_nd must be a finite nonempty floating (N,2) tensor"
        )
    # Own one coordinate leaf for both the core operator and q_xx. Reuse an
    # existing requires-grad leaf so callers can differentiate returned fields.
    return (
        coords_nd
        if coords_nd.is_leaf and coords_nd.requires_grad
        else coords_nd.detach().clone().requires_grad_(True)
    )


@torch.enable_grad()
def _dual(
    state_fn: Callable[[Tensor], Tensor], coords_nd: Tensor, nu_star: float
) -> dict[str, Tensor]:
    coords = _coordinates(coords_nd)
    captured = []

    def capture(points: Tensor) -> Tensor:
        state = state_fn(points)
        captured.append(state)
        return state

    inviscid = losses.residuals(capture, coords)
    # The accepted helper retains create_graph and returns differentiable zeros
    # for truly constant/linear coordinate derivatives. Nonzero parameter
    # dependencies in q_xx remain in the graph.
    q = captured[0][:, 1]
    q_x = losses._coordinate_derivatives(q, coords)[:, 0]
    q_xx = losses._coordinate_derivatives(q_x, coords)[:, 0]
    modified = torch.stack((inviscid[:, 0], inviscid[:, 1] - nu_star * q_xx), dim=1)
    if not bool(torch.isfinite(q_xx).all() & torch.isfinite(modified).all()):
        raise losses.LossError("nonfinite dimensionless viscosity derivative/residual")
    return {"inviscid": inviscid, "q_xx": q_xx, "modified": modified}


def viscosity_residuals(
    state_fn: Callable[[Tensor], Tensor], coords_nd: Tensor, nu_star: float
) -> Tensor:
    """Return scaled continuity/momentum residuals; zero skips q_xx entirely."""
    coefficient = _coefficient(nu_star)
    if coefficient == 0:
        return losses.residuals(state_fn, coords_nd)
    return _dual(state_fn, coords_nd, coefficient)["modified"]


def dual_residuals(
    state_fn: Callable[[Tensor], Tensor], coords_nd: Tensor
) -> dict[str, Tensor]:
    """Return inviscid, q_xx and modified operators for the fixed nu_star=.001."""
    return _dual(state_fn, coords_nd, NU_STAR)


def component_gradients(
    model: nn.Module, samples: Mapping[str, Any], component: str, nu_star: float
) -> tuple[float, list[Tensor]]:
    """Compute one complete unchunked component in accepted parameter order."""
    coefficient = _coefficient(nu_star)
    if coefficient == 0 or component in ("ic", "bc"):
        return core_train.component_gradients(model, samples, component, None)
    if set(samples) != losses.SAMPLE_KEYS:
        raise losses.LossError(
            "sample schema must contain coordinates and IC/BC targets only"
        )
    if component != "pde":
        raise losses.LossError(f"unknown component {component!r}")
    points = losses._tensor_samples(model, samples["interior_xt_nd"], "interior_xt_nd")
    difference = viscosity_residuals(model, points.detach().clone(), coefficient)
    loss = difference.square().sum(dim=1).mean()
    if not bool(torch.isfinite(loss)):
        raise core_train.TrainingError("nonfinite pde loss")
    parameters = list(model.parameters())
    accumulated = [torch.zeros_like(parameter) for parameter in parameters]
    value = 0.0
    # Keep the accepted one-chunk value/gradient arithmetic, including fraction1.
    fraction = 1.0
    value += fraction * float(loss.detach())
    if loss.requires_grad:
        gradients = torch.autograd.grad(fraction * loss, parameters, allow_unused=True)
        for buffer, gradient in zip(accumulated, gradients, strict=True):
            if gradient is not None:
                buffer.add_(gradient.detach())
    losses.gradient_rms(accumulated, parameters)
    return value, accumulated


def objective_gradients(
    model: nn.Module, samples: Mapping[str, Any], nu_star: float
) -> tuple[float, dict[str, float], dict[str, float], list[Tensor]]:
    """Return fixed1/10/10 objective and complete gradients in PDE/IC/BC order."""
    values, gradients, norms = {}, {}, {}
    for name in losses.COMPONENTS:
        values[name], gradients[name] = component_gradients(
            model, samples, name, nu_star
        )
        norms[name] = losses.gradient_rms(gradients[name], model.parameters())
    weighted = [
        sum(WEIGHTS[name] * gradients[name][index] for name in losses.COMPONENTS)
        for index, _ in enumerate(model.parameters())
    ]
    total = sum(WEIGHTS[name] * values[name] for name in losses.COMPONENTS)
    if not math.isfinite(total) or any(
        not bool(torch.isfinite(gradient).all()) for gradient in weighted
    ):
        raise core_train.TrainingError("nonfinite viscosity objective or gradient")
    return total, values, norms, weighted


@torch.enable_grad()
def validation(
    model: nn.Module, samples: Mapping[str, Any], nu_star: float
) -> dict[str, float]:
    """Evaluate both physics operators without computing parameter gradients."""
    coefficient = _coefficient(nu_star)
    if set(samples) != losses.SAMPLE_KEYS:
        raise losses.LossError(
            "sample schema must contain coordinates and IC/BC targets only"
        )
    coords = losses._tensor_samples(model, samples["interior_xt_nd"], "interior_xt_nd")
    if coefficient == 0:
        inviscid = losses.residuals(model, coords)
        modified = inviscid
    else:
        dual = _dual(model, coords, coefficient)
        inviscid, modified = dual["inviscid"], dual["modified"]
    values = {
        "modified_pde": float(modified.square().sum(dim=1).mean().detach()),
        "inviscid_pde": float(inviscid.square().sum(dim=1).mean().detach()),
        "rh_mse": float(inviscid[:, 0].square().mean().detach()),
        "rq_mse": float(inviscid[:, 1].square().mean().detach()),
        "rqnu_mse": float(modified[:, 1].square().mean().detach()),
    }
    for name in ("ic", "bc"):
        value = 0.0
        for fraction, loss in losses.iter_component_losses(model, samples, name, None):
            value += fraction * float(loss.detach())
        values[name] = value
    for branch in ("modified", "inviscid"):
        components = {
            "pde": values[f"{branch}_pde"],
            "ic": values["ic"],
            "bc": values["bc"],
        }
        values[f"{branch}_total"] = sum(
            WEIGHTS[name] * components[name] for name in losses.COMPONENTS
        )
    if any(not math.isfinite(value) for value in values.values()):
        raise core_train.TrainingError("nonfinite viscosity validation")
    return values
