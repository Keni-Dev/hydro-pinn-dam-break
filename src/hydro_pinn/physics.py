"""Source-free, flat-bed 1D shallow-water states and characteristic scaling.

Physical states are U=(h,q), with h in m and q=hu in m²/s per unit width.
Physical fluxes are F=(q,q²/h+gh²/2), with gravity in m/s². Dimensionless
states use H, Uc=sqrt(gH), Qc=H*Uc, Lc, and Tc=Lc/Uc, giving unit gravity.

At h=q=0, flux is exactly zero. Dry velocity is undefined; this module stores
zero together with a False wet mask. No depth floor is applied. NumPy inputs
use float64. Tensor inputs retain their dtype, device, and differentiation
graph; only CPU float64 behavior is verified in this phase.

Friction and topography are outside this module. For a future wide-channel
Manning extension, friction slope is Sf=n²q|q|/h^(10/3), while the momentum
source is -ghSf=-gn²q|q|/h^(7/3). These are different physical quantities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Self

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray
from torch import Tensor

from hydro_pinn.config import validate_config

type Array = NDArray[np.float64] | Tensor
type NumericInput = ArrayLike | Tensor


class PhysicsError(ValueError):
    """An array or state violates the physical representation contract."""


def _numeric(value: NumericInput, name: str) -> Array:
    if isinstance(value, Tensor):
        if not value.is_floating_point():
            raise PhysicsError(f"{name} must be a real floating-point tensor")
        if not bool(torch.isfinite(value).all()):
            raise PhysicsError(f"{name} must contain only finite values")
        return value
    try:
        array = np.asarray(value)
        if array.dtype.kind not in "iuf":
            raise PhysicsError(f"{name} must contain real numbers")
        array = array.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PhysicsError(f"{name} must contain real numbers: {exc}") from exc
    if not np.isfinite(array).all():
        raise PhysicsError(f"{name} must contain only finite values")
    return array


def _any(condition: NDArray[np.bool_] | Tensor) -> bool:
    return bool(condition.any())


def _stack(first: Array, second: Array) -> Array:
    if isinstance(first, Tensor):
        return torch.stack((first, second), dim=-1)
    return np.stack((first, second), axis=-1)


def _pair(first: NumericInput, second: NumericInput) -> tuple[Array, Array]:
    """Keep tensor operands on one backend without detaching either graph."""
    template = first if isinstance(first, Tensor) else second
    if isinstance(template, Tensor):
        for value in (first, second):
            if isinstance(value, Tensor) and (
                value.device != template.device or value.dtype != template.dtype
            ):
                raise PhysicsError("h and u tensors must share device and dtype")
        first = (
            first
            if isinstance(first, Tensor)
            else torch.as_tensor(first, dtype=template.dtype, device=template.device)
        )
        second = (
            second
            if isinstance(second, Tensor)
            else torch.as_tensor(second, dtype=template.dtype, device=template.device)
        )
    h, u = _numeric(first, "h"), _numeric(second, "u")
    if h.shape != u.shape:
        raise PhysicsError("h and u must have the same shape; broadcasting is explicit")
    return h, u


def _state(value: NumericInput) -> tuple[Array, Array, Array]:
    state = _numeric(value, "state")
    if state.ndim < 1 or state.shape[-1] != 2:
        raise PhysicsError("state must have shape (..., 2), ordered (h,q)")
    h, q = state[..., 0], state[..., 1]
    if _any(h < 0):
        raise PhysicsError("depth h must be nonnegative")
    if _any((h == 0) & (q != 0)):
        raise PhysicsError("a dry state h=0 requires q=0")
    return state, h, q


def state_from_depth_velocity(h: NumericInput, u: NumericInput) -> Array:
    """Return (...,2) states from matching h[m] and u[m/s] arrays, with q=hu.

    Scalar h/u pairs return shape (2,). A supplied dry velocity carries no
    physical meaning: h=0 gives q=0 and velocity_from_state supplies its mask.
    """
    h, u = _pair(h, u)
    if _any(h < 0):
        raise PhysicsError("depth h must be nonnegative")
    result = _stack(h, h * u)
    _state(result)
    return result


def velocity_from_state(
    state: NumericInput,
) -> tuple[Array, NDArray[np.bool_] | Tensor]:
    """Return u[m/s] and h>0 validity mask; stored dry velocity is zero.

    Replace only dry denominators before division. Masking q/h after a zero
    division would leave invalid intermediate values and tensor derivatives.
    """
    _, h, q = _state(state)
    wet = h > 0
    if isinstance(h, Tensor):
        denominator = torch.where(wet, h, torch.ones_like(h))
        velocity = torch.where(wet, q / denominator, torch.zeros_like(q))
    else:
        denominator = np.where(wet, h, 1.0)
        velocity = np.where(wet, q / denominator, 0.0)
    return _numeric(velocity, "velocity"), wet


def conservative_flux(state: NumericInput, gravity: float = 9.81) -> Array:
    """Return F=(q,q²/h+gh²/2) for physical U=(h,q), preserving its shape.

    The mass flux has units m²/s; momentum flux has units m³/s² per unit width.
    The genuinely dry state has zero flux and no invalid quotient. Gravity is
    a finite positive scalar; it is separate from the state differentiation.
    """
    if isinstance(gravity, bool) or not isinstance(gravity, (int, float)):
        raise PhysicsError("gravity must be a finite positive scalar in m/s²")
    if not math.isfinite(gravity) or gravity <= 0:
        raise PhysicsError("gravity must be finite and positive")
    _, h, q = _state(state)
    velocity, _ = velocity_from_state(state)
    result = _stack(q, q * velocity + 0.5 * gravity * h * h)
    return _numeric(result, "flux")


def dimensionless_flux(state: NumericInput) -> Array:
    """Return unit-gravity flux for state (h/H,q/(H*Uc))."""
    return conservative_flux(state, gravity=1.0)


@dataclass(frozen=True)
class Scales:
    """Characteristic SI scales; construct from a validated case when possible."""

    H_m: float
    Lc_m: float
    Uc_m_s: float
    Tc_s: float

    def __post_init__(self) -> None:
        for name in ("H_m", "Lc_m", "Uc_m_s", "Tc_s"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise PhysicsError(f"{name} must be finite and positive")
        if not math.isfinite(self.Qc_m2_s) or self.Qc_m2_s <= 0:
            raise PhysicsError("H_m*Uc_m_s must be a finite positive discharge scale")

    @property
    def Qc_m2_s(self) -> float:
        """Discharge per-unit-width scale H*Uc in m²/s."""
        return self.H_m * self.Uc_m_s

    @classmethod
    def from_case(cls, case: dict[str, Any]) -> Self:
        """Reuse the established case validator, including Uc/Tc identities."""
        checked = validate_config(case)
        if checked["config_type"] != "case":
            raise PhysicsError("Scales.from_case requires a case configuration")
        return cls(**checked["scales"])

    def _convert(self, value: NumericInput, scale: float, *, inverse: bool) -> Array:
        array = _numeric(value, "quantity")
        result = array * scale if inverse else array / scale
        return _numeric(result, "scaled quantity")

    def x_to_scaled(self, x_m: NumericInput) -> Array:
        """x*=x/Lc, with no shift or additional network mapping."""
        return self._convert(x_m, self.Lc_m, inverse=False)

    def x_to_physical(self, x_scaled: NumericInput) -> Array:
        """Return x[m] from x*."""
        return self._convert(x_scaled, self.Lc_m, inverse=True)

    def t_to_scaled(self, t_s: NumericInput) -> Array:
        """t*=t/Tc."""
        return self._convert(t_s, self.Tc_s, inverse=False)

    def t_to_physical(self, t_scaled: NumericInput) -> Array:
        """Return t[s] from t*."""
        return self._convert(t_scaled, self.Tc_s, inverse=True)

    def h_to_scaled(self, h_m: NumericInput) -> Array:
        """h*=h/H; no positivity floor is introduced."""
        return self._convert(h_m, self.H_m, inverse=False)

    def h_to_physical(self, h_scaled: NumericInput) -> Array:
        """Return h[m] from h*."""
        return self._convert(h_scaled, self.H_m, inverse=True)

    def u_to_scaled(self, u_m_s: NumericInput) -> Array:
        """u*=u/Uc."""
        return self._convert(u_m_s, self.Uc_m_s, inverse=False)

    def u_to_physical(self, u_scaled: NumericInput) -> Array:
        """Return u[m/s] from u*."""
        return self._convert(u_scaled, self.Uc_m_s, inverse=True)

    def q_to_scaled(self, q_m2_s: NumericInput) -> Array:
        """q*=q/(H*Uc)."""
        return self._convert(q_m2_s, self.Qc_m2_s, inverse=False)

    def q_to_physical(self, q_scaled: NumericInput) -> Array:
        """Return q[m²/s per unit width] from q*."""
        return self._convert(q_scaled, self.Qc_m2_s, inverse=True)

    def state_to_scaled(self, state: NumericInput) -> Array:
        """Scale a valid physical (...,2) state to (h*,q*)."""
        _, h, q = _state(state)
        return _stack(self.h_to_scaled(h), self.q_to_scaled(q))

    def state_to_physical(self, state: NumericInput) -> Array:
        """Unscale a valid dimensionless (...,2) state to SI units."""
        _, h, q = _state(state)
        return _stack(self.h_to_physical(h), self.q_to_physical(q))
