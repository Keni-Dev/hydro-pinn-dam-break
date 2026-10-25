"""Wet flat-bed Manning drag, separate from the immutable source-free core.

Physical slope is n²q|q|/h^(10/3), dimensionless, and physical momentum
source is -gn²q|q|/h^(7/3), in m²/s². n has units s/m^(1/3).
The scaled source coefficient is gn²Lc/H^(4/3). No mass source, depth
threshold, source limiter, topography, or dry-state prescription is added.

The PINN consumes the existing scaled (x/Lc,t/Tc) and (h/H,q/Qc) contracts.
The core residual and IC/BC losses remain unchanged; momentum receives the
negative of the dimensionless source. Tensor inputs retain derivative graphs.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor, nn

from hydro_pinn.losses import (
    COMPONENTS,
    SAMPLE_KEYS,
    LossError,
    component_loss,
    residuals,
)
from hydro_pinn.physics import PhysicsError, Scales

type Array = NDArray[np.float64] | Tensor


def _gravity(gravity: float) -> float:
    if (
        isinstance(gravity, bool)
        or not isinstance(gravity, (int, float))
        or not math.isfinite(gravity)
        or gravity <= 0
    ):
        raise PhysicsError("gravity must be a finite positive scalar in m/s²")
    return float(gravity)


def _roughness(n: float | Tensor, template: Tensor | None = None) -> float | Tensor:
    if isinstance(n, Tensor):
        if (
            not n.is_floating_point()
            or n.ndim != 0
            or not bool(torch.isfinite(n))
            or bool(n < 0)
        ):
            raise PhysicsError("Manning n must be a finite nonnegative scalar")
        if template is not None and (
            n.dtype != template.dtype or n.device != template.device
        ):
            raise PhysicsError("Manning n tensor must match state dtype and device")
        return n
    if (
        isinstance(n, bool)
        or not isinstance(n, (int, float))
        or not math.isfinite(n)
        or n < 0
    ):
        raise PhysicsError("Manning n must be a finite nonnegative scalar")
    return float(n)


def _wet_pair(h: ArrayLike | Tensor, q: ArrayLike | Tensor) -> tuple[Array, Array]:
    template = h if isinstance(h, Tensor) else q
    if isinstance(template, Tensor):
        if not template.is_floating_point():
            raise PhysicsError("h and q must be real floating-point tensors")
        converted = []
        for value in (h, q):
            if isinstance(value, Tensor):
                if value.dtype != template.dtype or value.device != template.device:
                    raise PhysicsError("h and q tensors must share dtype and device")
                converted.append(value)
            else:
                converted.append(
                    torch.as_tensor(
                        value,
                        **{
                            "dtype": template.dtype,
                            "device": template.device,
                        },
                    )
                )
        depth, discharge = converted
        finite = bool(torch.isfinite(depth).all() & torch.isfinite(discharge).all())
    else:
        try:
            raw_h, raw_q = np.asarray(h), np.asarray(q)
            if raw_h.dtype.kind not in "iuf" or raw_q.dtype.kind not in "iuf":
                raise PhysicsError("h and q must contain real numbers")
            depth = raw_h.astype(np.float64, copy=False)
            discharge = raw_q.astype(np.float64, copy=False)
        except (TypeError, ValueError, OverflowError) as exc:
            raise PhysicsError(f"invalid wet state: {exc}") from exc
        finite = bool(np.isfinite(depth).all() and np.isfinite(discharge).all())
    if depth.shape != discharge.shape:
        raise PhysicsError("h and q must have matching shapes")
    if not finite or bool((depth <= 0).any()):
        raise PhysicsError(
            "friction requires finite states and strictly positive depth"
        )
    return depth, discharge


def _finite_source(result: Array) -> Array:
    finite = (
        torch.isfinite(result).all()
        if isinstance(result, Tensor)
        else np.isfinite(result).all()
    )
    if not bool(finite):
        raise PhysicsError("nonfinite Manning source; no clipping applied")
    return result


def manning_slope(
    h: ArrayLike | Tensor, q: ArrayLike | Tensor, n: float | Tensor
) -> Array:
    """Return dimensionless friction slope for wet SI h[m] and q[m²/s]."""
    h, q = _wet_pair(h, q)
    n = _roughness(n, q if isinstance(q, Tensor) else None)
    if isinstance(n, Tensor) and not isinstance(q, Tensor):
        raise PhysicsError("tensor Manning n requires tensor states")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result = n**2 * q * abs(q) / h ** (10 / 3)
    return _finite_source(result)


def momentum_source(
    h: ArrayLike | Tensor,
    q: ArrayLike | Tensor,
    n: float | Tensor,
    gravity: float = 9.81,
) -> Array:
    """Return physical Sq[m²/s²]; its sign opposes q and there is no mass source."""
    h, q = _wet_pair(h, q)
    n = _roughness(n, q if isinstance(q, Tensor) else None)
    if isinstance(n, Tensor) and not isinstance(q, Tensor):
        raise PhysicsError("tensor Manning n requires tensor states")
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        result = -_gravity(gravity) * n**2 * q * abs(q) / h ** (7 / 3)
    return _finite_source(result)


def dimensionless_coefficient(
    scales: Scales, n: float | Tensor, gravity: float = 9.81
) -> float | Tensor:
    """Return C=gn²Lc/H^(4/3) for source -C q*|q*|/h*^(7/3)."""
    if not isinstance(scales, Scales):
        raise PhysicsError("scales must be the existing 1D Scales contract")
    n = _roughness(n)
    return _gravity(gravity) * n**2 * scales.Lc_m / scales.H_m ** (4 / 3)


@torch.enable_grad()
def friction_residuals(
    state_fn: Callable[[Tensor], Tensor],
    coords_nd: Tensor,
    scales: Scales,
    n: float | Tensor,
    *,
    gravity: float = 9.81,
) -> Tensor:
    """Return (N,2) scaled residuals, with positive drag on the momentum LHS.

    Derivatives use x*=x/Lc,t*=t/Tc directly. No second affine coordinate
    transform is introduced. With scalar n=0 this returns the exact core path.
    """
    coefficient = dimensionless_coefficient(scales, n, gravity)
    base = residuals(state_fn, coords_nd)
    if not isinstance(n, Tensor) and n == 0:
        return base
    state = state_fn(coords_nd)
    h, q = _wet_pair(state[:, 0], state[:, 1])
    _roughness(n, q)
    drag = coefficient * q * q.abs() / h ** (7 / 3)
    result = torch.stack((base[:, 0], base[:, 1] + drag), dim=1)
    if not bool(torch.isfinite(result).all()):
        raise LossError("nonfinite friction residual")
    return result


def friction_loss_components(
    model: nn.Module,
    samples: Mapping[str, Any],
    scales: Scales,
    n: float | Tensor,
    *,
    gravity: float = 9.81,
    chunk_size: int | None = None,
) -> dict[str, Tensor]:
    """Return complete PDE/IC/BC means under the unchanged physics-only schema.

    IC/BC delegate to the core. PDE chunks use point fractions, matching its
    complete mean of the per-point sum of both squared residuals. The returned
    tensors retain all chunk graphs, as does core component_losses.
    """
    if set(samples) != SAMPLE_KEYS:
        raise LossError("sample schema must contain coordinates and IC/BC targets only")
    dimensionless_coefficient(scales, n, gravity)
    if not isinstance(n, Tensor) and n == 0:
        return {
            name: component_loss(model, samples, name, chunk_size=chunk_size)
            for name in COMPONENTS
        }
    if chunk_size is not None and (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise LossError("chunk_size must be a positive integer or None")
    parameter = next(model.parameters(), None)
    if parameter is None:
        raise LossError("component losses require a model with parameters")
    array = np.asarray(samples["interior_xt_nd"], dtype=np.float64)
    if (
        array.ndim != 2
        or array.shape[1] != 2
        or not len(array)
        or not np.isfinite(array).all()
    ):
        raise LossError("interior_xt_nd must be a finite nonempty (N,2) array")
    points = torch.as_tensor(array, dtype=parameter.dtype, device=parameter.device)
    count = len(points)
    size = count if chunk_size is None else chunk_size
    pde = sum(
        len(points[start : start + size])
        / count
        * friction_residuals(
            model,
            points[start : start + size].detach().clone(),
            scales,
            n,
            gravity=gravity,
        )
        .square()
        .sum(dim=1)
        .mean()
        for start in range(0, count, size)
    )
    result = {"pde": pde}
    for component in ("ic", "bc"):
        result[component] = component_loss(
            model, samples, component, chunk_size=chunk_size
        )
    return result
