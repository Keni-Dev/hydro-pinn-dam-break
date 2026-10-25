"""Flat, source-free 2D shallow-water residuals in (x,y,t) order.

Scaled states are (h/H,qx/(H*Uc),qy/(H*Uc)). Both spatial axes use Lc,
Tc=Lc/Uc and Uc=sqrt(gH), so scaled gravity is one. Physical mass residual
has units m/s; both momentum residuals have units m²/s². No 1D contracts
are changed here, and no solution invariance is imposed.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor

from hydro_pinn.physics import Scales


def _coordinates(coords: Tensor) -> None:
    if (
        not isinstance(coords, Tensor)
        or not coords.is_floating_point()
        or coords.ndim != 2
        or coords.shape[1] != 3
        or not len(coords)
        or not bool(torch.isfinite(coords).all())
    ):
        raise ValueError("coordinates require finite nonempty floating (N,3) x,y,t")


def _derivative(field: Tensor, coords: Tensor) -> Tensor:
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


def conservative_fluxes(state: Tensor, gravity: float = 1.0) -> tuple[Tensor, Tensor]:
    """Return x/y fluxes, each ordered mass,x-momentum,y-momentum.

    With physical states use physical g; with scaled states use g*=1. The
    caller must supply positive depth, as the network does by construction.
    """
    if state.ndim != 2 or state.shape[1] != 3:
        raise ValueError("state requires shape (N,3), ordered h,qx,qy")
    if not bool(torch.isfinite(state).all()) or bool((state[:, 0] <= 0).any()):
        raise ValueError("states require finite values and strictly positive depth")
    h, qx, qy = state.unbind(dim=1)
    cross = qx * qy / h
    pressure = 0.5 * gravity * h.square()
    return (
        torch.stack((qx, qx.square() / h + pressure, cross), dim=1),
        torch.stack((qy, cross, qy.square() / h + pressure), dim=1),
    )


@torch.enable_grad()
def residuals(state_fn: Callable[[Tensor], Tensor], coords_nd: Tensor) -> Tensor:
    """Return scaled mass/x/y momentum residuals with parameter graphs.

    Each state row must depend only on its corresponding coordinate row.
    Constant states have zero derivatives. Validation also needs coordinate
    autograd, even when called within an outer no_grad context.
    """
    _coordinates(coords_nd)
    coords = (
        coords_nd
        if coords_nd.requires_grad
        else coords_nd.detach().clone().requires_grad_(True)
    )
    state = state_fn(coords)
    if not isinstance(state, Tensor) or state.shape != coords.shape:
        raise ValueError("state_fn must return scaled (N,3) h,qx,qy")
    flux_x, flux_y = conservative_fluxes(state)
    dh = _derivative(state[:, 0], coords)
    dqx = _derivative(state[:, 1], coords)
    dqy = _derivative(state[:, 2], coords)
    dxx = _derivative(flux_x[:, 1], coords)
    dxy = _derivative(flux_x[:, 2], coords)
    dyy = _derivative(flux_y[:, 2], coords)
    result = torch.stack(
        (
            dh[:, 2] + dqx[:, 0] + dqy[:, 1],
            dqx[:, 2] + dxx[:, 0] + dxy[:, 1],
            dqy[:, 2] + dxy[:, 0] + dyy[:, 1],
        ),
        dim=1,
    )
    if not bool(torch.isfinite(result).all()):
        raise ValueError("nonfinite 2D residual")
    return result


def physical_residuals(model: Callable[[Tensor], Tensor], coords_m_s: Tensor) -> Tensor:
    """Return SI residuals from a model exposing the shared Scales object.

    d/dx=(1/Lc)d/dx*, d/dy=(1/Lc)d/dy*, d/dt=(1/Tc)d/dt*.
    Therefore rh=(H/Tc)rh*, rx=(H*Uc/Tc)rx*, ry=(H*Uc/Tc)ry*.
    The identities Tc=Lc/Uc and Uc²=gH make every flux chain factor match.
    """
    _coordinates(coords_m_s)
    scales: Scales = model.scales
    coords_nd = torch.stack(
        (
            coords_m_s[:, 0] / scales.Lc_m,
            coords_m_s[:, 1] / scales.Lc_m,
            coords_m_s[:, 2] / scales.Tc_s,
        ),
        dim=1,
    )
    factors = coords_m_s.new_tensor(
        (
            scales.H_m / scales.Tc_s,
            scales.Qc_m2_s / scales.Tc_s,
            scales.Qc_m2_s / scales.Tc_s,
        )
    )
    return residuals(model, coords_nd) * factors
