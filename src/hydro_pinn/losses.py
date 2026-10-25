"""Dimensionless physics-only losses and the project's gradient-ratio rule.

Coordinates are (x/Lc,t/Tc), model states are (h/H,q/(H*Uc)), and gravity is
one in the scaled flux. PDE, initial, and boundary losses each average a sum
over the two equations/state fields. Boundary normalization counts every point
equally, including when the two sides or the final chunk have unequal counts.

The adaptive rule uses each complete unweighted component gradient's RMS over
the same parameter count. It is the project-specific ratio/clipping/EMA rule,
not an exact reproduction of a published weighting algorithm.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

COMPONENTS = ("pde", "ic", "bc")
SAMPLE_KEYS = frozenset(
    {
        "interior_xt_nd",
        "initial_xt_nd",
        "initial_hq_nd",
        "boundary_xt_nd",
        "boundary_hq_nd",
    }
)


class LossError(ValueError):
    """A loss, derivative, or adaptive statistic cannot be evaluated safely."""


def _coordinate_derivatives(field: Tensor, coords: Tensor) -> Tensor:
    """Differentiate point-independent fields while retaining parameter graphs."""
    if not field.requires_grad:
        return coords * 0
    derivative = torch.autograd.grad(
        field,
        coords,
        grad_outputs=torch.ones_like(field),
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )[0]
    return coords * 0 if derivative is None else derivative


@torch.enable_grad()
def residuals(state_fn: Callable[[Tensor], Tensor], coords_nd: Tensor) -> Tensor:
    """Return (rh,rq) with derivative graphs for trainable parameters.

    ``state_fn`` must evaluate each row independently. Callers supply scaled
    coordinates directly; no further affine mapping or omitted chain factor is
    hidden here. Constant states have exactly zero coordinate derivatives.
    Coordinate autograd is enabled even for a validation caller using no_grad.
    """
    if (
        not isinstance(coords_nd, Tensor)
        or not coords_nd.is_floating_point()
        or coords_nd.ndim != 2
        or coords_nd.shape[1] != 2
        or len(coords_nd) == 0
        or not bool(torch.isfinite(coords_nd).all())
    ):
        raise LossError("coords_nd must be a finite nonempty floating (N,2) tensor")
    coords = (
        coords_nd
        if coords_nd.requires_grad
        else coords_nd.detach().clone().requires_grad_(True)
    )
    state = state_fn(coords)
    if not isinstance(state, Tensor) or state.shape != coords.shape:
        raise LossError("state_fn must return (N,2) scaled (h,q) states")
    if not bool(torch.isfinite(state).all()) or bool((state[:, 0] <= 0).any()):
        raise LossError("residual states must be finite with strictly positive depth")
    h, q = state[:, 0], state[:, 1]
    dh = _coordinate_derivatives(h, coords)
    dq = _coordinate_derivatives(q, coords)
    momentum_flux = q.square() / h + 0.5 * h.square()
    dflux = _coordinate_derivatives(momentum_flux, coords)
    result = torch.stack((dh[:, 1] + dq[:, 0], dq[:, 1] + dflux[:, 0]), dim=1)
    if not bool(torch.isfinite(result).all()):
        raise LossError("nonfinite dimensionless residual")
    return result


def _tensor_samples(model: nn.Module, value: Any, name: str) -> Tensor:
    parameters = list(model.parameters())
    if not parameters:
        raise LossError("component losses require a model with parameters")
    parameter = parameters[0]
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2 or not len(array):
        raise LossError(f"{name} must have nonempty shape (N,2)")
    if not np.isfinite(array).all():
        raise LossError(f"{name} must contain finite values")
    return torch.as_tensor(array, dtype=parameter.dtype, device=parameter.device)


def iter_component_losses(
    model: nn.Module,
    samples: Mapping[str, Any],
    component: str,
    chunk_size: int | None = None,
) -> Iterator[tuple[float, Tensor]]:
    """Yield (point fraction, chunk mean) without retaining previous graphs.

    Training can backward/accumulate each fraction-weighted chunk immediately,
    then compute a norm from the summed full-component gradients. Do not average
    chunk norms, or update parameters until the complete objective is accumulated.
    The strict schema permits no analytical interior targets.
    """
    if set(samples) != SAMPLE_KEYS:
        raise LossError("sample schema must contain coordinates and IC/BC targets only")
    if component not in COMPONENTS:
        raise LossError(f"unknown component {component!r}")
    if chunk_size is not None and (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise LossError("chunk_size must be a positive integer or None")
    prefix = {"pde": "interior", "ic": "initial", "bc": "boundary"}[component]
    points = _tensor_samples(model, samples[f"{prefix}_xt_nd"], f"{prefix}_xt_nd")
    targets = None
    if component != "pde":
        targets = _tensor_samples(model, samples[f"{prefix}_hq_nd"], f"{prefix}_hq_nd")
        if targets.shape != points.shape:
            raise LossError("target and coordinate shapes must match")
    count = len(points)
    size = count if chunk_size is None else chunk_size
    for start in range(0, count, size):
        stop = min(start + size, count)
        coordinates = points[start:stop].detach().clone()
        if component == "pde":
            difference = residuals(model, coordinates)
        else:
            difference = model(coordinates) - targets[start:stop]
        chunk_loss = difference.square().sum(dim=1).mean()
        if not bool(torch.isfinite(chunk_loss)):
            raise LossError(f"nonfinite {component} loss")
        yield (stop - start) / count, chunk_loss


def component_loss(
    model: nn.Module,
    samples: Mapping[str, Any],
    component: str,
    *,
    chunk_size: int | None = None,
) -> Tensor:
    """Return one complete unweighted mean, optionally assembled from chunks.

    This convenience function retains all chunk graphs. Memory-limited training
    should use ``iter_component_losses`` and backward each contribution instead.
    """
    return sum(
        fraction * loss
        for fraction, loss in iter_component_losses(
            model, samples, component, chunk_size
        )
    )


def component_losses(
    model: nn.Module,
    samples: Mapping[str, Any],
    *,
    chunk_size: int | None = None,
) -> dict[str, Tensor]:
    """Return PDE/IC/BC means; no analytical interior solution labels are used."""
    return {
        name: component_loss(model, samples, name, chunk_size=chunk_size)
        for name in COMPONENTS
    }


def gradient_rms(grads: Iterable[Tensor | None], parameters: Iterable[Tensor]) -> float:
    """Detached ||grad||₂/sqrt(P), including absent gradients as zero in P."""
    gradients, params = list(grads), list(parameters)
    if len(gradients) != len(params):
        raise LossError("gradients and parameters must have matching lengths")
    count = sum(parameter.numel() for parameter in params)
    if count == 0:
        raise LossError("gradient RMS requires at least one parameter")
    squared_norm = 0.0
    for gradient, parameter in zip(gradients, params, strict=True):
        if gradient is None:
            continue
        if gradient.shape != parameter.shape or not bool(
            torch.isfinite(gradient).all()
        ):
            raise LossError("component gradients must match parameters and be finite")
        # Statistics are detached CPU float64, including on float32 backends
        # whose device cannot allocate float64 (MPS). Model gradients stay put.
        squared_norm += float(
            gradient.detach().to(device="cpu", dtype=torch.float64).square().sum()
        )
    norm = math.sqrt(squared_norm / count)
    if not math.isfinite(norm):
        raise LossError("nonfinite component gradient RMS")
    return norm


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise LossError(f"{name} must be scalar")
        value = float(value.detach())
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LossError(f"{name} must be a finite nonnegative number")
    if not math.isfinite(value) or value < 0:
        raise LossError(f"{name} must be finite and nonnegative")
    return float(value)


def update_adaptive_weights(
    weights: Mapping[str, float],
    norms: Mapping[str, float],
    step: int,
    weighting: Mapping[str, Any],
) -> dict[str, float]:
    """Apply the specified update before one-based Adam steps 100,200,... .

    Input ``weighting`` comes from the validated training configuration. The
    returned floats have no autograd graph; ordinary objective graphs are not
    consumed or changed. Fixed mode and intervening steps preserve the weights.
    """
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise LossError("Adam step numbering starts at one")
    if set(weights) != set(COMPONENTS):
        raise LossError("weights must contain pde,ic,bc")
    result = {
        name: _finite_nonnegative(weights[name], f"weight {name}")
        for name in COMPONENTS
    }
    if result["pde"] != 1.0:
        raise LossError("the project adaptive rule fixes lambda_pde=1")
    if weighting["mode"] == "fixed":
        return result
    if weighting["mode"] != "gradient_norm_ema":
        raise LossError("unknown weighting mode")
    interval = weighting["update_interval"]
    if step % interval:
        return result
    if set(norms) != set(COMPONENTS):
        raise LossError("adaptive updates need pde,ic,bc full-component norms")
    checked = {
        name: _finite_nonnegative(norms[name], f"gradient RMS {name}")
        for name in COMPONENTS
    }
    low, high = weighting["target_bounds"]
    epsilon, decay = weighting["denominator_epsilon"], weighting["ema_decay"]
    for name in ("ic", "bc"):
        ratio = checked["pde"] / (checked[name] + epsilon)
        if not math.isfinite(ratio):
            raise LossError("nonfinite adaptive target ratio")
        target = min(high, max(low, ratio))
        result[name] = decay * result[name] + (1 - decay) * target
    return result
