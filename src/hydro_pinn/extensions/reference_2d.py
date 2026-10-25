"""Strictly wet 2D Cartesian finite-volume numerical kernels.

Normal-interface HLLE speeds and full-vector middle state are adapted from
Clawpack riemann src/rpn2_shallow_hlle.f90, commit
c29dcab4af30913fc16e4406fe56fe5f0a322b1f (v5.12.0 observed).
Pinned local source and same-commit license:
docs/evidence/phase_10/genuine_reference_sources_preflight_v01/rpn2_shallow_hlle.f90
docs/evidence/phase_10/genuine_reference_sources_preflight_v01/LICENSE
The unsplit update, ghosts, geometry, projection and diagnostics are local
choices. This module does not execute PyClaw. States are float64 (h,qx,qy),
with positive depth only; either momentum may be signed or zero.

BSD 3-Clause License

Copyright (c) 1994-2018, Clawpack Developers
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import time
from pathlib import Path
from typing import NamedTuple

import numpy as np

from hydro_pinn.records import RunRecord, write_json_atomic

DEFAULT_SPEED = math.sqrt(9.81)


class Reference2DError(ValueError):
    """Retain offending raw arrays and location for the caller's failure record."""

    def __init__(self, message, *, stage, raw_arrays=None, location=None, context=None):
        super().__init__(message)
        self.stage = stage
        self.raw_arrays = {
            name: np.array(value, copy=True)
            for name, value in (raw_arrays or {}).items()
        }
        self.location = None if location is None else tuple(int(i) for i in location)
        self.context = dict(context or {})


class FluxResult(NamedTuple):
    flux: np.ndarray
    s_left: np.ndarray
    s_right: np.ndarray
    intermediate: np.ndarray


class FaceFluxes(NamedTuple):
    fx: np.ndarray
    fy: np.ndarray
    a_x: float
    a_y: float
    diagnostics: dict


def _first(mask):
    indices = np.argwhere(mask)
    return tuple(indices[0]) if len(indices) else None


def validate_state(state, *, stage="input", context=None):
    """Require (...,3) finite states and h>0; do not constrain momentum signs."""
    values = np.asarray(state, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 3 or values.size == 0:
        raise Reference2DError(
            "state requires nonempty (...,3), ordered h,qx,qy",
            stage=stage,
            raw_arrays={"state": values},
            context=context,
        )
    invalid = ~np.isfinite(values).all(axis=-1) | (values[..., 0] <= 0)
    if np.any(invalid):
        raise Reference2DError(
            "nonfinite conserved component or nonpositive depth",
            stage=stage,
            raw_arrays={"state": values},
            location=_first(invalid),
            context=context,
        )
    return values


def _positive(value, name):
    if (
        isinstance(value, bool)
        or not np.isscalar(value)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise Reference2DError(
            f"{name} must be finite and positive",
            stage="parameters",
            context={name: value},
        )
    return float(value)


def _axis(axis):
    if axis in ("x", 0) and not isinstance(axis, bool):
        return 1
    if axis in ("y", 1) and not isinstance(axis, bool):
        return 2
    raise Reference2DError(
        "axis must be x/0 or y/1", stage="parameters", context={"axis": axis}
    )


def physical_flux(state, axis, gravity=9.81):
    """SI flux vector in global h,qx,qy order for one positive axis normal."""
    values = validate_state(state, stage="physical_flux_input")
    normal = _axis(axis)
    tangent = 3 - normal
    g = _positive(gravity, "gravity")
    h, qn, qt = values[..., 0], values[..., normal], values[..., tangent]
    flux = np.empty_like(values)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        flux[..., 0] = qn
        flux[..., normal] = qn * (qn / h) + 0.5 * g * h * h
        flux[..., tangent] = qn * (qt / h)
    if not np.isfinite(flux).all():
        raise Reference2DError(
            "nonfinite physical flux",
            stage="physical_flux",
            raw_arrays={"state": values, "flux": flux},
            location=_first(~np.isfinite(flux).all(axis=-1)),
        )
    return flux


def hlle_flux(left, right, axis, gravity=9.81):
    """Pinned wet HLLE with Einfeldt speed estimates and full tangential flux.

    sL=min(uL-cL,u_hat-c_hat), sR=max(uR+cR,u_hat+c_hat), with
    h_hat=(hL+hR)/2 and sqrt(h)-weighted u_hat. The middle state is
    (sR*UR-sL*UL-(FR-FL))/(sR-sL). Upwind FL or FR applies outside the fan.
    """
    normal = _axis(axis)
    g = _positive(gravity, "gravity")
    left = validate_state(left, stage="hlle_left", context={"axis": axis})
    right = validate_state(right, stage="hlle_right", context={"axis": axis})
    if left.shape != right.shape:
        raise Reference2DError(
            "left/right state shapes must match",
            stage="hlle_shape",
            raw_arrays={"left_state": left, "right_state": right},
        )
    hl, hr = left[..., 0], right[..., 0]
    fl, fr = physical_flux(left, axis, g), physical_flux(right, axis, g)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ul, ur = left[..., normal] / hl, right[..., normal] / hr
        root_l, root_r = np.sqrt(hl), np.sqrt(hr)
        uhat = (ul * root_l + ur * root_r) / (root_l + root_r)
        chat = np.sqrt(g * 0.5 * (hl + hr))
        sl = np.minimum(ul - np.sqrt(g * hl), uhat - chat)
        sr = np.maximum(ur + np.sqrt(g * hr), uhat + chat)
        denominator = sr - sl
    invalid = (
        ~np.isfinite(sl)
        | ~np.isfinite(sr)
        | ~np.isfinite(denominator)
        | (denominator <= 0)
    )
    raw = {"left_state": left, "right_state": right, "s_left": sl, "s_right": sr}
    if np.any(invalid):
        raise Reference2DError(
            "invalid HLLE wave fan denominator",
            stage="hlle_speeds",
            raw_arrays=raw,
            location=_first(invalid),
            context={"axis": axis},
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        middle = (
            sr[..., None] * right - sl[..., None] * left - (fr - fl)
        ) / denominator[..., None]
    raw["intermediate"] = middle
    invalid = ~np.isfinite(middle).all(axis=-1) | (middle[..., 0] <= 0)
    if np.any(invalid):
        raise Reference2DError(
            "invalid raw HLLE intermediate state",
            stage="hlle_intermediate",
            raw_arrays=raw,
            location=_first(invalid),
            context={"axis": axis},
        )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        fan_flux = (
            sr[..., None] * fl
            - sl[..., None] * fr
            + (sl * sr)[..., None] * (right - left)
        ) / denominator[..., None]
        flux = np.where(
            (sl >= 0)[..., None], fl, np.where((sr <= 0)[..., None], fr, fan_flux)
        )
    if not np.isfinite(flux).all():
        raw["flux"] = flux
        raise Reference2DError(
            "nonfinite HLLE flux",
            stage="hlle_flux",
            raw_arrays=raw,
            location=_first(~np.isfinite(flux).all(axis=-1)),
            context={"axis": axis},
        )
    return FluxResult(flux, sl, sr, middle)


def ghost_cells(state, boundary="reflective"):
    """One ghost cell; x/y reflections commute at all four corners."""
    values = validate_state(state, stage="ghost_input")
    if values.ndim != 3:
        raise Reference2DError(
            "grid state requires (ny,nx,3)",
            stage="ghost_input",
            raw_arrays={"state": values},
        )
    if boundary == "periodic":
        padded = np.pad(values, ((1, 1), (1, 1), (0, 0)), mode="wrap")
    elif boundary == "reflective":
        padded = np.pad(values, ((1, 1), (1, 1), (0, 0)), mode="edge")
        padded[:, (0, -1), 1] *= -1
        padded[(0, -1), :, 2] *= -1
    else:
        raise Reference2DError(
            "boundary must be reflective or periodic",
            stage="boundary",
            context={"boundary": boundary},
        )
    return validate_state(padded, stage="ghost_state")


def fluxfaces(state, boundary="reflective", gravity=9.81):
    """Build both directional arrays from the same old-time state and ghosts."""
    old = validate_state(state, stage="fluxfaces_input")
    ghosts = ghost_cells(old, boundary)
    try:
        x = hlle_flux(ghosts[1:-1, :-1], ghosts[1:-1, 1:], "x", gravity)
        y = hlle_flux(ghosts[:-1, 1:-1], ghosts[1:, 1:-1], "y", gravity)
    except Reference2DError as exc:
        exc.raw_arrays["old_state"] = old.copy()
        exc.raw_arrays["ghost_state"] = ghosts.copy()
        exc.context["boundary"] = boundary
        raise
    ax = float(np.maximum(np.abs(x.s_left), np.abs(x.s_right)).max())
    ay = float(np.maximum(np.abs(y.s_left), np.abs(y.s_right)).max())
    diagnostics = {
        "minimum_intermediate_depth_m": float(
            min(x.intermediate[..., 0].min(), y.intermediate[..., 0].min())
        ),
        "x_minimum_intermediate_depth_m": float(x.intermediate[..., 0].min()),
        "y_minimum_intermediate_depth_m": float(y.intermediate[..., 0].min()),
        "x_s_left_min_m_s": float(x.s_left.min()),
        "x_s_right_max_m_s": float(x.s_right.max()),
        "y_s_left_min_m_s": float(y.s_left.min()),
        "y_s_right_max_m_s": float(y.s_right.max()),
    }
    return FaceFluxes(x.flux, y.flux, ax, ay, diagnostics)


def _face_shapes(state, fx, fy):
    if state.ndim != 3:
        raise Reference2DError("grid state requires (ny,nx,3)", stage="face_shape")
    ny, nx, _ = state.shape
    if fx.shape != (ny, nx + 1, 3) or fy.shape != (ny + 1, nx, 3):
        raise Reference2DError(
            "face arrays disagree with physical grid",
            stage="face_shape",
            raw_arrays={"old_state": state, "flux_x": fx, "flux_y": fy},
        )


def unsplit_step(state, fx, fy, dt, dx, dy, *, context=None):
    """One unsplit Euler update using both old-time flux arrays, without repair."""
    old = validate_state(state, stage="step_input", context=context)
    fx, fy = np.asarray(fx, dtype=np.float64), np.asarray(fy, dtype=np.float64)
    _face_shapes(old, fx, fy)
    dt, dx, dy = (
        _positive(value, name) for value, name in ((dt, "dt"), (dx, "dx"), (dy, "dy"))
    )
    raw = {"old_state": old, "flux_x": fx, "flux_y": fy}
    if not np.isfinite(fx).all() or not np.isfinite(fy).all():
        raise Reference2DError(
            "nonfinite old-time face flux",
            stage="step_flux",
            raw_arrays=raw,
            context=context,
        )
    with np.errstate(over="ignore", invalid="ignore"):
        updated = (
            old - (dt / dx) * (fx[:, 1:] - fx[:, :-1]) - (dt / dy) * (fy[1:] - fy[:-1])
        )
    raw["updated_state"] = updated
    invalid = ~np.isfinite(updated).all(axis=-1) | (updated[..., 0] <= 0)
    if np.any(invalid):
        raise Reference2DError(
            "invalid raw unsplit updated state",
            stage="updated_state",
            raw_arrays=raw,
            location=_first(invalid),
            context={**(context or {}), "dt": dt, "dx": dx, "dy": dy},
        )
    return updated


def outward_rates(fx, fy, dx, dy):
    """Four outward full-vector rates: units (m³/s,m⁴/s²,m⁴/s²)."""
    fx, fy = np.asarray(fx, dtype=np.float64), np.asarray(fy, dtype=np.float64)
    dx, dy = _positive(dx, "dx"), _positive(dy, "dy")
    if (
        fx.ndim != 3
        or fy.ndim != 3
        or fx.shape[-1] != 3
        or fy.shape[-1] != 3
        or fx.shape[0] + 1 != fy.shape[0]
        or fx.shape[1] != fy.shape[1] + 1
        or not np.isfinite(fx).all()
        or not np.isfinite(fy).all()
    ):
        raise Reference2DError(
            "invalid vector face arrays",
            stage="boundary_rates",
            raw_arrays={"flux_x": fx, "flux_y": fy},
        )
    return {
        "left": -dy * fx[:, 0].sum(axis=0),
        "right": dy * fx[:, -1].sum(axis=0),
        "bottom": -dx * fy[0].sum(axis=0),
        "top": dx * fy[-1].sum(axis=0),
    }


def _edges(edges, name):
    a = np.asarray(edges, dtype=np.float64)
    if (
        a.ndim != 1
        or len(a) < 2
        or not np.isfinite(a).all()
        or not np.all(np.diff(a) > 0)
    ):
        raise Reference2DError(
            f"{name} requires finite strictly increasing edges",
            stage="geometry",
            raw_arrays={name: a},
        )
    return a


def rectangle_initial(
    x_edges,
    y_edges,
    *,
    reservoir_x=(-1.0, 1.0),
    reservoir_y=(-0.5, 0.5),
    h_inside=1.0,
    h_outside=0.1,
):
    """Exact area-overlap cell means of a constant-depth rectangular reservoir."""
    xe, ye = _edges(x_edges, "x_edges"), _edges(y_edges, "y_edges")
    rx, ry = _edges(reservoir_x, "reservoir_x"), _edges(reservoir_y, "reservoir_y")
    if len(rx) != 2 or len(ry) != 2:
        raise Reference2DError("reservoir needs two edges per axis", stage="geometry")
    hi, ho = _positive(h_inside, "h_inside"), _positive(h_outside, "h_outside")
    ox = np.maximum(0.0, np.minimum(xe[1:], rx[1]) - np.maximum(xe[:-1], rx[0]))
    oy = np.maximum(0.0, np.minimum(ye[1:], ry[1]) - np.maximum(ye[:-1], ry[0]))
    fraction = np.outer(oy / np.diff(ye), ox / np.diff(xe))
    state = np.zeros((len(ye) - 1, len(xe) - 1, 3), dtype=np.float64)
    state[..., 0] = ho + (hi - ho) * fraction
    return validate_state(state, stage="initial_state")


def overlap_weights(source_edges, target_edges):
    """Normalized target-by-source overlap for conservative cell means.

    The source must cover every target cell. Coverage defects above 64eps are
    rejected; no row renormalization conceals a missing source strip.
    """
    source = _edges(source_edges, "source_edges")
    target = _edges(target_edges, "target_edges")
    lengths = np.maximum(
        0.0,
        np.minimum(target[1:, None], source[None, 1:])
        - np.maximum(target[:-1, None], source[None, :-1]),
    )
    weights = lengths / np.diff(target)[:, None]
    defect = np.abs(weights.sum(axis=1) - 1.0)
    if np.any(defect > 64 * np.finfo(np.float64).eps):
        raise Reference2DError(
            "target cell is not fully covered by source",
            stage="projection_coverage",
            raw_arrays={
                "weights": weights,
                "source_edges": source,
                "target_edges": target,
            },
            location=_first(defect > 64 * np.finfo(float).eps),
        )
    return weights


def conservative_project(state, wx, wy):
    """Project (...,y,x,3) cell means using separable normalized overlap weights."""
    values = validate_state(state, stage="projection_input")
    wx, wy = np.asarray(wx, dtype=np.float64), np.asarray(wy, dtype=np.float64)
    if (
        values.ndim < 3
        or wx.ndim != 2
        or wy.ndim != 2
        or wx.shape[1] != values.shape[-2]
        or wy.shape[1] != values.shape[-3]
        or not np.isfinite(wx).all()
        or not np.isfinite(wy).all()
        or np.any(wx < 0)
        or np.any(wy < 0)
        or np.any(np.abs(wx.sum(axis=1) - 1) > 64 * np.finfo(float).eps)
        or np.any(np.abs(wy.sum(axis=1) - 1) > 64 * np.finfo(float).eps)
    ):
        raise Reference2DError(
            "invalid separable projection operators",
            stage="projection_operator",
            raw_arrays={"weights_x": wx, "weights_y": wy},
        )
    projected = np.einsum("Jj,...jic,Ii->...JIc", wy, values, wx, optimize=True)
    return validate_state(projected, stage="projected_state")


def parity_diagnostics(state, H=1.0, Uc=DEFAULT_SPEED):
    """Scaled reflection defects for h even, qx odd-x, qy odd-y."""
    values = validate_state(state, stage="parity_input")
    if values.ndim != 3:
        raise Reference2DError("parity requires (ny,nx,3)", stage="parity_input")
    scale = np.array((_positive(H, "H"), H * _positive(Uc, "Uc"), H * Uc))
    xdefect = (
        np.max(np.abs(values - values[:, ::-1] * np.array((1, -1, 1))), axis=(0, 1))
        / scale
    )
    ydefect = (
        np.max(np.abs(values - values[::-1] * np.array((1, 1, -1))), axis=(0, 1))
        / scale
    )
    return {
        "x_scaled": xdefect.tolist(),
        "y_scaled": ydefect.tolist(),
        "maximum_scaled": float(max(xdefect.max(), ydefect.max())),
    }


def outer_strip_diagnostics(state, ambient_depth=0.1, H=1.0, Uc=DEFAULT_SPEED):
    """Scaled outermost-cell departure per side; a discrete diagnostic only."""
    values = validate_state(state, stage="outer_strip_input")
    if values.ndim != 3:
        raise Reference2DError(
            "outer strips require (ny,nx,3)", stage="outer_strip_input"
        )
    scale = np.array((_positive(H, "H"), H * _positive(Uc, "Uc"), H * Uc))
    ambient = np.array((_positive(ambient_depth, "ambient_depth"), 0.0, 0.0))
    strips = {
        "left": values[:, 0],
        "right": values[:, -1],
        "bottom": values[0],
        "top": values[-1],
    }
    return {
        side: (np.max(np.abs(strip - ambient), axis=0) / scale).tolist()
        for side, strip in strips.items()
    }


# The following record/advance/protocol choices are local, not Clawpack code.
EDGES = ("left", "right", "bottom", "top")
PROVENANCE_BASE = "docs/evidence/phase_10/genuine_reference_sources_preflight_v01"
INPUT_PATHS = (
    "docs/coordination/extension_genuine_2d_reference_selection.json",
    "docs/coordination/phase_10_2_reference_gate_plan_v01.json",
    "docs/coordination/phase_10_2_reference_plan_review_v01.json",
    "docs/coordination/phase_10_2_source_preflight_v01.json",
    f"{PROVENANCE_BASE}/manifest.json",
    f"{PROVENANCE_BASE}/rpn2_shallow_hlle.f90",
    f"{PROVENANCE_BASE}/LICENSE",
    f"{PROVENANCE_BASE}/source_interpretation.json",
)
SOURCE_PATHS = (
    "src/hydro_pinn/extensions/reference_2d.py",
    "src/hydro_pinn/cli.py",
    "src/hydro_pinn/records.py",
    "src/hydro_pinn/config.py",
    "pyproject.toml",
    "uv.lock",
)
PINNED_COMMIT = "c29dcab4af30913fc16e4406fe56fe5f0a322b1f"
PINNED_SOURCE_SHA = "9d2b334c8d5056ca8ec99e444237ba8d5ec4a554a86228d91c4e8ce598e2280f"
PINNED_LICENSE_SHA = "668317944192d8217d63c0fd0730f889bd07f759b69ed6b657e1fb9902145ebe"
EPS = np.finfo(np.float64).eps


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path):
    return json.loads(Path(path).read_text())


def _save_npz(path, **arrays):
    """Preserve interrupted write attempts; publish a complete payload once."""
    path = Path(path)
    if path.exists():
        return
    attempt = 1
    while (path.parent / f"{path.stem}.write_attempt_{attempt:02d}.npz").exists():
        attempt += 1
    staging = path.parent / f"{path.stem}.write_attempt_{attempt:02d}.npz"
    with staging.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
    # This is the newly owned staging file only. Failed partial attempts remain.
    staging.rename(path)


def _digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _write(path, value, *, overwrite=False):
    write_json_atomic(Path(path), value, overwrite=overwrite)


def gamma(count):
    """Standard float64 roundoff accumulation factor; count is fixed by work."""
    product = int(count) * EPS
    if not 0 <= product < 1:
        raise ValueError("invalid roundoff operation count")
    return product / (1 - product)


def arithmetic_allowance(count, absolute_magnitude, unit_scale):
    return 32 * gamma(count) * np.asarray(absolute_magnitude) + 32 * EPS * np.asarray(
        unit_scale
    )


def tolerance_definitions():
    return {
        "epsilon": float(EPS),
        "gamma": "gamma(n)=n*eps/(1-n*eps), n*eps<1",
        "arithmetic_R": "R(n,M,S)=32*gamma(n)*M+32*eps*S; raw defects retained",
        "edge_geometry_m": "32*eps*Lc (Lc=10m)",
        "deep_area_m2": "R(Ncells+8,sum(abs(overlap_areas)),Lc^2)",
        "initial_volume_m3": "R(Ncells+8,sum(abs(h*area)),Lc^2*Hc)",
        "projection_coverage": "64*eps per normalized overlap row; no renormalization",
        "projection_integral": (
            "R(Nnative+Ncommon+nx+ny+16,max(sum(abs(U)*area),sum(abs(PU)*"
            "common_area)),Lc^2*S_j)"
        ),
        "constant_planar_parity": (
            "32*gamma(32*k+64)*max(S_j,max|U_j|,max|expected_j|)+32*eps*S"
            "_j; parity reported divided by S_j"
        ),
        "budget": (
            "32*[gamma(Ncells+4)*(L_k+L_0)+gamma(16)*W_k+(gamma(Nedge+4)+"
            "gamma(k+4))*E_k]+32*eps*(Lc^2*S_j)"
        ),
        "budget_magnitudes": (
            "L=sum(area*abs(U)); E=sum_steps "
            "dt*sum_boundary_faces(length*abs(flux)); W=sum_steps "
            "sum_cells area*[abs(Uold)+(dt/dx)*(abs(Fleft)+abs(Fright))+("
            "dt/dy)*(abs(Gbottom)+abs(Gtop))]"
        ),
        "state_scales": "S=(Hc,Hc*Uc,Hc*Uc), Hc=1m,Uc=sqrt(9.81)m/s",
        "budget_units": ["m^3", "m^4/s", "m^4/s"],
        "cfl_allowance": "64*eps added to requested sum-CFL",
        "boundary_safety": (
            "Dmax<4m and all components of all4 outer strips <1e-10 at "
            "every accepted step, including initial strips"
        ),
        "limits": (
            "Arithmetic allowances only; no physical accuracy/error "
            "bound, statistical gate or claimed convergence rate."
        ),
    }


def load_case_2d(config_path, root=Path(".")):
    """Validate this dedicated new case locally, without changing core schemas."""
    cfg = _json(Path(root) / config_path)
    expected = {
        "schema_version": 1,
        "case_id": "wet_rectangular_reservoir_2d_v01",
        "dimension": 2,
        "domain_x_m": [-5.0, 5.0],
        "domain_y_m": [-5.0, 5.0],
        "reservoir_x_m": [-1.0, 1.0],
        "reservoir_y_m": [-0.5, 0.5],
        "inside_depth_m": 1.0,
        "outside_depth_m": 0.1,
        "initial_qx_m2_s": 0.0,
        "initial_qy_m2_s": 0.0,
        "gravity_m_s2": 9.81,
        "bed_elevation_m": 0.0,
        "manning_n": 0.0,
        "rainfall_m_s": 0.0,
        "boundary": "four_reflective_slip_walls",
        "final_time_s": 0.25,
        "positive_output_dt_s": 0.005,
        "positive_output_count": 50,
        "initial_snapshot": True,
        "mathematical_initial_volume_m3": 11.8,
        "scales": {"Hc_m": 1.0, "Lc_m": 10.0},
        "backend": "numpy_cpu",
        "dtype": "float64",
        "method": "local_first_order_unsplit_forward_Euler_full_vector_HLLE",
        "common_grid": [80, 80],
        "matrix": [
            {"id": f"ref_rect{n}_{c}_v01", "nx": n, "ny": n, "cfl_sum": v}
            for n, c, v in [
                (80, "c045", 0.45),
                (160, "c045", 0.45),
                (320, "c045", 0.45),
                (320, "c0225", 0.225),
            ]
        ],
        "preselected_reference_id": "ref_rect320_c0225_v01",
        "first_cell_numerical_wall_cap_s": 60.0,
        "boundary_travel_limit_m": 4.0,
        "outer_strip_scaled_departure_limit": 1e-10,
    }
    if cfg != expected:
        raise ValueError("case differs from adopted wet rectangular v01 protocol")
    return cfg


def requested_times():
    return np.r_[0.0, 0.005 * np.arange(1, 51, dtype=np.float64)]


def _grid(nx, ny):
    xe, ye = np.linspace(-5.0, 5.0, nx + 1), np.linspace(-5.0, 5.0, ny + 1)
    return {
        "x_edges_m": xe,
        "y_edges_m": ye,
        "x_m": (xe[:-1] + xe[1:]) / 2,
        "y_m": (ye[:-1] + ye[1:]) / 2,
        "cell_area_m2": np.outer(np.diff(ye), np.diff(xe)),
        "t_s": requested_times(),
    }


def freeze_reference_protocol(config_path, destination, *, root=Path(".")):
    """Freeze source/config/axes/area operators; advance no time steps."""
    root, destination = Path(root).resolve(), Path(destination)
    output = root / destination
    if output.exists():
        raise FileExistsError(f"protocol destination already exists: {output}")
    cfg = load_case_2d(config_path, root)
    if (
        sha256(root / INPUT_PATHS[5]) != PINNED_SOURCE_SHA
        or sha256(root / INPUT_PATHS[6]) != PINNED_LICENSE_SHA
    ):
        raise ValueError("pinned primary source/license bytes changed")
    output.mkdir(parents=True, exist_ok=False)
    common = _grid(80, 80)
    _save_npz(output / "common_axes.npz", **common)
    geometry, artifacts = {}, {}
    for n in (80, 160, 320):
        axes = _grid(n, n)
        wx = overlap_weights(axes["x_edges_m"], common["x_edges_m"])
        wy = overlap_weights(axes["y_edges_m"], common["y_edges_m"])
        initial = rectangle_initial(axes["x_edges_m"], axes["y_edges_m"])
        area = axes["cell_area_m2"]
        deep = (initial[..., 0] - 0.1) / 0.9 * area
        volume = float(np.sum(initial[..., 0] * area))
        area_value = float(deep.sum())
        area_tol = float(arithmetic_allowance(n * n + 8, np.abs(deep).sum(), 100.0))
        volume_tol = float(
            arithmetic_allowance(
                n * n + 8, np.sum(np.abs(initial[..., 0] * area)), 100.0
            )
        )
        geometry[str(n)] = {
            "deep_area_m2": area_value,
            "deep_area_defect_m2": area_value - 2.0,
            "deep_area_tolerance_m2": area_tol,
            "initial_volume_m3": volume,
            "initial_volume_defect_m3": volume - 11.8,
            "initial_volume_tolerance_m3": volume_tol,
            "passes": bool(
                abs(area_value - 2.0) <= area_tol and abs(volume - 11.8) <= volume_tol
            ),
            "time_updates_executed": 0,
        }
        if not geometry[str(n)]["passes"]:
            raise ValueError("initial geometry arithmetic violates frozen tolerance")
        _save_npz(
            output / f"axes_operator_n{n}.npz", **axes, weights_x=wx, weights_y=wy
        )
    _write(output / "geometry_readback.json", geometry)
    artifacts = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    protocol = {
        "schema_version": 1,
        "state": "candidate_frozen_no_canonical_cell_released",
        "case": cfg,
        "config_path": str(config_path),
        "config_sha256": sha256(root / config_path),
        "source_sha256": {p: sha256(root / p) for p in SOURCE_PATHS},
        "input_sha256": {p: sha256(root / p) for p in INPUT_PATHS},
        "artifact_sha256": artifacts,
        "tolerances": tolerance_definitions(),
        "provenance": {
            "commit": PINNED_COMMIT,
            "source_sha256": PINNED_SOURCE_SHA,
            "license_sha256": PINNED_LICENSE_SHA,
            "license": "BSD-3-Clause",
            "adapted": (
                "normal wet full-vector interface speeds/middle state/flux translation"
            ),
            "local": (
                "unsplit Euler/sum-"
                "CFL/ghosts/geometry/projection/diagnostics/persistence"
            ),
            "PyClaw_executed": False,
        },
        "array_contract": {
            "location": "cell_average",
            "orientation": "time,y,x",
            "fields": ["h_m", "qx_m2_s", "qy_m2_s"],
            "integral_units": ["m^3", "m^4/s", "m^4/s"],
            "boundary_vector_rate_units": ["m^3/s", "m^4/s^2", "m^4/s^2"],
        },
        "norms": {
            "area_mean_MAE": "sum(A*abs(d))/sum(A)",
            "area_mean_RMSE": "sqrt(sum(A*d^2)/sum(A))",
            "integral_L1": "sum(A*abs(d))",
            "integral_L2": "sqrt(sum(A*d^2))",
            "relative_L2": (
                "100*L2(d)/L2(reference), null if reference norm=0; no "
                "epsilon denominator"
            ),
            "time_weighting": (
                "equal weights over the50positive snapshots; initial is "
                "separate evidence"
            ),
        },
        "boundary_diagnostics": {
            "D_x": "sum(dt*a_x)",
            "D_y": "sum(dt*a_y)",
            "D_max": "sum(dt*max(a_x,a_y))",
            "outer_strips": (
                "per-side/component max(abs(U-[.1,0,0])/S), every accepted "
                "step and initial"
            ),
            "interpretation": (
                "discrete diagnostic, not analytical wave-speed/error bound; "
                "walls model pressure forces"
            ),
        },
        "runtime": {
            "origin": (
                "run_reference entry before config/release/source validation "
                "and RunRecord setup"
            ),
            "included": (
                "validation/setup/initialization/flux/update/diagnostics/proj"
                "ection/output and final persistence"
            ),
            "cap_check": (
                "before each next time step; final actual wall may exceed by "
                "one step/final writes"
            ),
            "first_cell_cap_s": 60.0,
            "later_caps": "positive exact cap in separate later root release",
            "partial": (
                "last valid U/t/step, completed snapshots, all accepted "
                "dt/speeds/budgets/four edge fluxes retained"
            ),
            "termination_kinds": [
                "completed",
                "budget_limit",
                "interrupt",
                "numerical_failure",
            ],
        },
        "release_schema": {
            "state": "released",
            "run_id": "one exact matrix cell ID",
            "protocol_sha256": "this protocol content digest",
            "numerical_wall_cap_s": (
                "60forfirst cell; later explicit measured-budget release"
            ),
        },
        "operator": (
            "P(U)=einsum(wy,U,wx), separable area overlaps, no centre "
            "sampling/renormalization"
        ),
        "future_refinement": (
            "signed160-80,320-160 at.45 "
            "coupledmesh/time;320(.225)-320(.45) timeonly; Dspace+ Dtime "
            "sensitivity envelope not errorbound"
        ),
        "canonical_time_updates_executed_at_freeze": 0,
    }
    protocol["protocol_sha256"] = _digest(protocol)
    _write(output / "protocol.json", protocol)
    return output / "protocol.json"


def verify_reference_protocol(path, root=Path(".")):
    root, path = Path(root).resolve(), Path(path)
    path = root / path
    p = _json(path)
    if (
        _digest({k: v for k, v in p.items() if k != "protocol_sha256"})
        != p["protocol_sha256"]
    ):
        raise ValueError("reference protocol content digest mismatch")
    for name, expected in {
        **p["source_sha256"],
        **p["input_sha256"],
        p["config_path"]: p["config_sha256"],
    }.items():
        if sha256(root / name) != expected:
            raise ValueError(f"reference source/input changed: {name}")
    for name, expected in p["artifact_sha256"].items():
        if sha256(path.parent / name) != expected:
            raise ValueError(f"frozen reference axes/operator changed: {name}")
    return p


def _memory():
    return {
        "peak_process_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if platform.system() == "Darwin" else 1024)
        ),
        "host_load_1_5_15": list(os.getloadavg()),
        "limit": (
            "process-lifetime maximum, not system pressure/instantaneous allocator peak"
        ),
    }


def _persist_run(
    run,
    state,
    t,
    step,
    snapshots,
    snapshot_times,
    rows,
    edge_faces,
    xe,
    ye,
    failure=None,
):
    """Write available observations only; never fill missing requested times."""
    areas = np.outer(np.diff(ye), np.diff(xe))
    fields = np.asarray(snapshots, dtype=np.float64).reshape(
        (-1, len(ye) - 1, len(xe) - 1, 3)
    )
    _save_npz(
        run / "last_valid_state.npz",
        state_h_qx_qy=state,
        t_s=np.asarray(t),
        step=np.asarray(step),
        x_edges_m=xe,
        y_edges_m=ye,
    )
    _save_npz(
        run / "outputs.npz",
        x_edges_m=xe,
        y_edges_m=ye,
        x_m=(xe[:-1] + xe[1:]) / 2,
        y_m=(ye[:-1] + ye[1:]) / 2,
        cell_area_m2=areas,
        t_s=np.asarray(snapshot_times),
        h_m=fields[..., 0],
        qx_m2_s=fields[..., 1],
        qy_m2_s=fields[..., 2],
    )
    keys = (
        "t_s",
        "dt_s",
        "a_x_m_s",
        "a_y_m_s",
        "x_s_left_min_m_s",
        "x_s_right_max_m_s",
        "y_s_left_min_m_s",
        "y_s_right_max_m_s",
        "sum_cfl",
        "minimum_depth_m",
        "minimum_intermediate_depth_m",
        "totals",
        "budget_defect",
        "budget_tolerance",
        "outward_rates",
        "cumulative_outward",
        "travel_m",
        "outer_strips_scaled",
        "parity_scaled",
        "parity_tolerance_scaled",
        "budget_absolute_boundary_sum",
        "budget_absolute_update_sum",
    )
    shapes = {
        "totals": (3,),
        "budget_defect": (3,),
        "budget_tolerance": (3,),
        "outward_rates": (4, 3),
        "cumulative_outward": (4, 3),
        "travel_m": (3,),
        "outer_strips_scaled": (4, 3),
        "parity_scaled": (2, 3),
        "parity_tolerance_scaled": (3,),
        "budget_absolute_boundary_sum": (3,),
        "budget_absolute_update_sum": (3,),
    }
    arrays = {
        k: np.asarray([r[k] for r in rows], dtype=np.float64).reshape(
            (len(rows), *shapes.get(k, ()))
        )
        for k in keys
    }
    for edge in EDGES:
        size = len(ye) - 1 if edge in EDGES[:2] else len(xe) - 1
        arrays[f"step_{edge}_numerical_flux"] = np.asarray(
            edge_faces[edge], dtype=np.float64
        ).reshape((len(rows), size, 3))
    _save_npz(run / "diagnostics.npz", **arrays)
    if failure is not None:
        _save_npz(run / "raw_failure.npz", **failure.raw_arrays)
        _write(
            run
            / (
                "failure_recovery.json"
                if (run / "failure.json").exists()
                else "failure.json"
            ),
            {
                "type": type(failure).__name__,
                "message": str(failure),
                "stage": failure.stage,
                "location": failure.location,
                "context": failure.context,
            },
        )


def _advance_record(
    record,
    initial_state,
    xe,
    ye,
    times,
    *,
    cfl=0.45,
    wall_cap_s=60.0,
    start=None,
    boundary="reflective",
    imposed_dt=None,
    canonical=False,
    protocol=None,
    operator=None,
):
    start = time.perf_counter() if start is None else start
    run = record.path
    state = validate_state(initial_state, stage="initial_state").copy()
    dx, dy = float(np.diff(xe)[0]), float(np.diff(ye)[0])
    area, ncell = dx * dy, state.shape[0] * state.shape[1]
    nedge = 2 * (state.shape[0] + state.shape[1])
    scales = np.array((1.0, DEFAULT_SPEED, DEFAULT_SPEED))
    initial_totals = area * state.sum(axis=(0, 1))
    initial_absolute = area * np.abs(state).sum(axis=(0, 1))
    boundary_integrals = np.zeros((4, 3))
    absolute_boundary, absolute_updates, travel = np.zeros(3), np.zeros(3), np.zeros(3)
    snapshots, snap_times, rows = [state.copy()], [0.0], []
    edge_faces = {e: [] for e in EDGES}
    t, step, termination_kind, failure = 0.0, 0, "completed", None
    accepted = (
        state,
        t,
        step,
        boundary_integrals,
        absolute_boundary,
        absolute_updates,
        travel,
    )
    stage_cost = {
        "flux": 0.0,
        "update": 0.0,
        "diagnostics": 0.0,
        "projection": 0.0,
        "persistence": 0.0,
    }
    outer_initial = outer_strip_diagnostics(state)
    try:
        _save_npz(
            run / "initial_state.npz",
            state_h_qx_qy=state,
            x_edges_m=xe,
            y_edges_m=ye,
            requested_t_s=times,
        )
        for target in times[1:]:
            while t < target:
                if time.perf_counter() - start >= wall_cap_s:
                    termination_kind = "budget_limit"
                    raise TimeoutError("pre-step numerical wall budget reached")
                now = time.perf_counter()
                faces = fluxfaces(state, boundary=boundary)
                stage_cost["flux"] += time.perf_counter() - now
                remaining = float(target - t)
                dt_cfl = cfl / (faces.a_x / dx + faces.a_y / dy)
                dt = min(dt_cfl if imposed_dt is None else imposed_dt, remaining)
                if not np.isfinite(dt) or dt <= 0 or t + dt == t:
                    raise Reference2DError(
                        "nonadvancing dt",
                        stage="time_step",
                        raw_arrays={"old_state": state},
                        context={"t_s": t, "step": step, "dt_s": dt},
                    )
                now = time.perf_counter()
                updated = unsplit_step(
                    state,
                    faces.fx,
                    faces.fy,
                    dt,
                    dx,
                    dy,
                    context={
                        "step": step + 1,
                        "old_t_s": t,
                        "target_s": float(target),
                        "sum_cfl": dt * (faces.a_x / dx + faces.a_y / dy),
                    },
                )
                stage_cost["update"] += time.perf_counter() - now
                now = time.perf_counter()
                rate_map = outward_rates(faces.fx, faces.fy, dx, dy)
                rates = np.array([rate_map[e] for e in EDGES])
                pending_boundary = boundary_integrals + dt * rates
                pending_absolute_boundary = absolute_boundary + dt * (
                    dy
                    * (np.abs(faces.fx[:, 0]).sum(0) + np.abs(faces.fx[:, -1]).sum(0))
                    + dx * (np.abs(faces.fy[0]).sum(0) + np.abs(faces.fy[-1]).sum(0))
                )
                pending_absolute_updates = absolute_updates + area * (
                    np.abs(state)
                    + (dt / dx) * (np.abs(faces.fx[:, 1:]) + np.abs(faces.fx[:, :-1]))
                    + (dt / dy) * (np.abs(faces.fy[1:]) + np.abs(faces.fy[:-1]))
                ).sum(axis=(0, 1))
                pending_travel = travel + dt * np.array(
                    [faces.a_x, faces.a_y, max(faces.a_x, faces.a_y)]
                )
                pending_step = step + 1
                pending_t = float(target) if dt == remaining else t + dt
                totals = area * updated.sum(axis=(0, 1))
                defect = totals - initial_totals + pending_boundary.sum(0)
                absolute = area * np.abs(updated).sum(axis=(0, 1))
                allowance = (
                    32
                    * (
                        gamma(ncell + 4) * (absolute + initial_absolute)
                        + gamma(16) * pending_absolute_updates
                        + (gamma(nedge + 4) + gamma(pending_step + 4))
                        * pending_absolute_boundary
                    )
                    + 32 * EPS * 100 * scales
                )
                parity = parity_diagnostics(updated)
                parity_tol = (
                    32
                    * gamma(32 * pending_step + 64)
                    * np.maximum(scales, np.max(np.abs(updated), axis=(0, 1)))
                    + 32 * EPS * scales
                ) / scales
                strips = outer_strip_diagnostics(updated)
                rows.append(
                    {
                        "t_s": pending_t,
                        "dt_s": dt,
                        "a_x_m_s": faces.a_x,
                        "a_y_m_s": faces.a_y,
                        **{
                            k: faces.diagnostics[k]
                            for k in (
                                "x_s_left_min_m_s",
                                "x_s_right_max_m_s",
                                "y_s_left_min_m_s",
                                "y_s_right_max_m_s",
                            )
                        },
                        "sum_cfl": dt * (faces.a_x / dx + faces.a_y / dy),
                        "minimum_depth_m": float(updated[..., 0].min()),
                        "minimum_intermediate_depth_m": faces.diagnostics[
                            "minimum_intermediate_depth_m"
                        ],
                        "totals": totals,
                        "budget_defect": defect,
                        "budget_tolerance": allowance,
                        "outward_rates": rates,
                        "cumulative_outward": pending_boundary.copy(),
                        "travel_m": pending_travel.copy(),
                        "outer_strips_scaled": np.array([strips[e] for e in EDGES]),
                        "parity_scaled": np.array(
                            [parity["x_scaled"], parity["y_scaled"]]
                        ),
                        "parity_tolerance_scaled": parity_tol,
                        "budget_absolute_boundary_sum": (
                            pending_absolute_boundary.copy()
                        ),
                        "budget_absolute_update_sum": pending_absolute_updates.copy(),
                    }
                )
                edge_faces["left"].append(faces.fx[:, 0].copy())
                edge_faces["right"].append(faces.fx[:, -1].copy())
                edge_faces["bottom"].append(faces.fy[0].copy())
                edge_faces["top"].append(faces.fy[-1].copy())
                stage_cost["diagnostics"] += time.perf_counter() - now
                # Single accepted-root assignment follows all diagnostic/history work.
                # An interrupt during append is rolled back to this root below.
                accepted = (
                    updated,
                    pending_t,
                    pending_step,
                    pending_boundary,
                    pending_absolute_boundary,
                    pending_absolute_updates,
                    pending_travel,
                )
                (
                    state,
                    t,
                    step,
                    boundary_integrals,
                    absolute_boundary,
                    absolute_updates,
                    travel,
                ) = accepted
            snapshots.append(state.copy())
            snap_times.append(float(target))
    except TimeoutError:
        pass
    except KeyboardInterrupt:
        termination_kind = "interrupt"
    except Reference2DError as exc:
        termination_kind, failure = "numerical_failure", exc
        failure.context.update(last_valid_step=step, last_valid_t_s=t)
    except Exception as exc:
        termination_kind = "numerical_failure"
        failure = Reference2DError(
            str(exc),
            stage="unexpected_execution_exception",
            raw_arrays={"old_state": state},
            context={"type": type(exc).__name__, "step": step, "t_s": t},
        )
    # Reconcile partially appended histories to the last atomic accepted root.
    (
        state,
        t,
        step,
        boundary_integrals,
        absolute_boundary,
        absolute_updates,
        travel,
    ) = accepted
    del rows[step:]
    for edge in EDGES:
        del edge_faces[edge][step:]
    available = min(len(snapshots), len(snap_times))
    snapshots, snap_times = snapshots[:available], snap_times[:available]
    criteria = {"finalization_complete": False}
    try:
        now = time.perf_counter()
        _persist_run(
            run,
            state,
            t,
            step,
            snapshots,
            snap_times,
            rows,
            edge_faces,
            xe,
            ye,
            failure,
        )
        stage_cost["persistence"] += time.perf_counter() - now
        criteria = {
            "finite_positive_depth": bool(
                all(
                    r["minimum_depth_m"] > 0 and r["minimum_intermediate_depth_m"] > 0
                    for r in rows
                )
            ),
            "sum_cfl": bool(all(r["sum_cfl"] <= cfl + 64 * EPS for r in rows)),
            "budgets": bool(
                all(
                    np.all(np.abs(r["budget_defect"]) <= r["budget_tolerance"])
                    for r in rows
                )
            ),
            "parity": bool(
                all(
                    np.all(r["parity_scaled"] <= r["parity_tolerance_scaled"])
                    for r in rows
                )
            ),
            "travel_Dmax": bool(travel[2] < 4.0),
            "outer_strips": bool(
                np.max(list(outer_initial.values())) < 1e-10
                and all(np.max(r["outer_strips_scaled"]) < 1e-10 for r in rows)
            ),
            "all_requested_snapshots": bool(
                np.array_equal(np.asarray(snap_times), times)
            ),
        }
        if canonical and termination_kind == "completed" and not all(criteria.values()):
            termination_kind = "numerical_failure"
            _write(
                run / "validity_failure.json",
                {
                    "stage": "reference_diagnostic_gate",
                    "criteria": criteria,
                    "meaning": (
                        "Finite cell run retained but not accepted as a valid early-"
                        "time reference; no cropping or retuning."
                    ),
                },
            )
        if canonical and termination_kind == "completed" and operator is not None:
            now = time.perf_counter()
            fields = np.asarray(snapshots)
            projected = conservative_project(
                fields, operator["weights_x"], operator["weights_y"]
            )
            axes = _grid(80, 80)
            _save_npz(
                run / "common_reference.npz",
                **axes,
                h_m=projected[..., 0],
                qx_m2_s=projected[..., 1],
                qy_m2_s=projected[..., 2],
            )
            native_integrals = area * fields.sum(axis=(1, 2))
            common_integrals = (axes["cell_area_m2"][None, :, :, None] * projected).sum(
                axis=(1, 2)
            )
            magnitude = np.maximum(
                area * np.abs(fields).sum(axis=(1, 2)),
                (axes["cell_area_m2"][None, :, :, None] * np.abs(projected)).sum(
                    axis=(1, 2)
                ),
            )
            project_tol = arithmetic_allowance(
                ncell + 6400 + len(xe) + len(ye) + 14, magnitude, 100 * scales
            )
            _write(
                run / "projection_readback.json",
                {
                    "maximum_abs_integral_defect": np.max(
                        np.abs(native_integrals - common_integrals), axis=0
                    ).tolist(),
                    "tolerance_per_time_component": project_tol.tolist(),
                    "passed": bool(
                        np.all(
                            np.abs(native_integrals - common_integrals) <= project_tol
                        )
                    ),
                },
            )
            if not np.all(np.abs(native_integrals - common_integrals) <= project_tol):
                termination_kind = "numerical_failure"
            stage_cost["projection"] += time.perf_counter() - now
    except (KeyboardInterrupt, Exception) as exc:
        termination_kind = (
            "interrupt" if isinstance(exc, KeyboardInterrupt) else "numerical_failure"
        )
        finalization_failure = Reference2DError(
            str(exc) or "interrupted output finalization",
            stage="output_finalization",
            raw_arrays={"old_state": state},
            context={
                "type": type(exc).__name__,
                "last_valid_step": step,
                "last_valid_t_s": t,
            },
        )
        _write(
            run / "finalization_attempt_01_failure.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "stage": "output_finalization",
                "termination_kind": termination_kind,
                "last_valid_step": step,
                "last_valid_t_s": t,
                "policy": (
                    "One bounded recovery write; completed payloads are kept and "
                    "failed staging files remain."
                ),
            },
        )
        _persist_run(
            run,
            state,
            t,
            step,
            snapshots,
            snap_times,
            rows,
            edge_faces,
            xe,
            ye,
            finalization_failure,
        )
        if failure is None and termination_kind == "numerical_failure":
            failure = finalization_failure
    summary = {
        "case_id": "wet_rectangular_reservoir_2d_v01"
        if canonical
        else "genuine_2d_small_fixture",
        "method": "local_unsplit_full_vector_wet_HLLE",
        "dimension": 2,
        "field_location": "cell_average",
        "termination_kind": termination_kind,
        "accepted_steps": step,
        "last_valid_t_s": t,
        "completed_snapshot_count": len(snap_times),
        "requested_snapshot_count": len(times),
        "initial_integrals": initial_totals.tolist(),
        "initial_volume_discrepancy_m3": float(initial_totals[0] - 11.8)
        if canonical
        else None,
        "final_integrals": (area * state.sum(axis=(0, 1))).tolist(),
        "travel_Dx_Dy_Dmax_m": travel.tolist(),
        "criteria": criteria,
        "criteria_enforced": canonical,
        "cost_s": stage_cost,
        "protocol_sha256": None if protocol is None else protocol["protocol_sha256"],
        "imposed_dt_fixture_only": imposed_dt,
        "wall_cap_s": wall_cap_s,
        "timer_includes": (
            "entry validation/setup/init/loop/diagnostics/projection/persistence"
        ),
        "memory": _memory(),
        "provisional": (
            "One coarse cell cannot establish refinement-accepted reference accuracy."
        ),
    }

    def terminal_bundle(kind, *, overwrite=False):
        summary["termination_kind"] = kind
        summary["final_observed_wall_s"] = time.perf_counter() - start
        summary["wall_cap_overshoot_s"] = max(
            0.0, summary["final_observed_wall_s"] - wall_cap_s
        )
        _write(run / "summary.json", summary, overwrite=overwrite)
        _write(
            run / "termination.json",
            {
                "termination_kind": kind,
                "last_valid_step": step,
                "last_valid_t_s": t,
                "completed_snapshot_count": len(snap_times),
                "actual_wall_observed_before_commit_s": summary[
                    "final_observed_wall_s"
                ],
                "wall_cap_s": wall_cap_s,
                "stage": None if failure is None else failure.stage,
                "location": None if failure is None else failure.location,
                "context": {} if failure is None else failure.context,
                "timing_limit": (
                    "RunRecord terminal metadata records entry-to-commit elapsed "
                    "after mandatory payload hashing; external launcher wall "
                    "remains separate."
                ),
            },
            overwrite=overwrite,
        )
        _write(
            run / "execution_wall.json",
            {
                "actual_entry_through_payload_persistence_wall_s": time.perf_counter()
                - start,
                "wall_cap_s": wall_cap_s,
                "termination_kind": kind,
                "included": (
                    "setup/init/loop/diagnostics/native+common output and "
                    "terminal payload writes"
                ),
                "terminal_metadata": (
                    "metadata.execution.elapsed_timings_s.entry_to_terminal_commi"
                    "t records later elapsed observation; lifecycle commit is the"
                    " final write."
                ),
            },
            overwrite=overwrite,
        )
        excluded = {"status.json", "metadata.json", "hashes.json"}
        _write(
            run / "hashes.json",
            {
                "source_sha256": {
                    "src/hydro_pinn/extensions/reference_2d.py": sha256(Path(__file__))
                },
                "files": {
                    p.name: sha256(p)
                    for p in run.iterdir()
                    if p.is_file() and p.name not in excluded
                },
                "mutable_lifecycle_files_excluded": ["metadata.json", "status.json"],
                "reason": (
                    "The terminal lifecycle status is committed only after all "
                    "mandatory payload/terminal/hash files. Final lifecycle "
                    "hashes belong in an external launch/readback receipt."
                ),
            },
            overwrite=overwrite,
        )

    def commit(kind):
        elapsed = {
            **stage_cost,
            "entry_to_terminal_commit": time.perf_counter() - start,
        }
        if kind == "completed":
            record.complete(
                stopping_reason=(
                    "requested snapshots and mandatory terminal evidence completed"
                ),
                iterations=step,
                elapsed_timings_s=elapsed,
            )
        elif kind in ("budget_limit", "interrupt"):
            record._transition(
                "interrupted",
                kind,
                {
                    "termination_kind": kind,
                    "last_valid_state": {
                        "file": "last_valid_state.npz",
                        "step": step,
                        "t_s": t,
                    },
                },
                timings=elapsed,
                iterations=step,
            )
        else:
            record._transition(
                "failed",
                str(failure)
                if failure
                else "reference diagnostic/projection gate failed",
                {
                    "termination_kind": "numerical_failure",
                    "stage": failure.stage if failure else "reference_validity_gate",
                    "step": step,
                    "time_s": t,
                    "last_valid_state": {"file": "last_valid_state.npz"},
                },
                timings=elapsed,
                iterations=step,
            )

    try:
        terminal_bundle(termination_kind)
        # Success status is the final commit marker, after every required write.
        commit(termination_kind)
    except (KeyboardInterrupt, Exception) as exc:
        prior_kind = termination_kind
        termination_kind = (
            "interrupt" if isinstance(exc, KeyboardInterrupt) else "numerical_failure"
        )
        final_failure = Reference2DError(
            str(exc) or "terminal finalization interrupted",
            stage="terminal_finalization",
            raw_arrays={"old_state": state},
            context={
                "type": type(exc).__name__,
                "last_valid_step": step,
                "last_valid_t_s": t,
                "prior_termination_kind": prior_kind,
            },
        )
        _write(
            run / "terminal_finalization_attempt_01_failure.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "stage": "terminal_finalization",
                "prior_termination_kind": prior_kind,
                "termination_kind": termination_kind,
                "last_valid_step": step,
                "last_valid_t_s": t,
                "policy": (
                    "One bounded failed-finalization persistence attempt; prior "
                    "raw/staging files remain."
                ),
            },
        )
        if failure is None:
            failure = final_failure
        _persist_run(
            run,
            state,
            t,
            step,
            snapshots,
            snap_times,
            rows,
            edge_faces,
            xe,
            ye,
            final_failure,
        )
        terminal_bundle(termination_kind, overwrite=True)
        commit(termination_kind)
    if failure is not None:
        raise failure
    return run


def _record(output_root, run_id, resolved, *, root=Path(".")):
    record = RunRecord.create(output_root, run_id, resolved, project_root=root)
    metadata = _json(record.path / "metadata.json")
    metadata.update(
        case=resolved["case"]["case_id"],
        dimension=2,
        scientific_authority=(
            "local independent NumPy numerical approximation; "
            "not E1/Stoker/PyClaw execution"
        ),
        numerical_floor=None,
        reference_provenance={
            "commit": PINNED_COMMIT,
            "license": "BSD-3-Clause",
            "source_sha256": PINNED_SOURCE_SHA,
        },
        array_contract={
            "orientation": "time,y,x",
            "location": "cell_average",
            "fields": ["h_m", "qx_m2_s", "qy_m2_s"],
            "integral_units": ["m^3", "m^4/s", "m^4/s"],
        },
    )
    _write(record.path / "metadata.json", metadata, overwrite=True)
    return record


def run_fixture(
    output_root,
    run_id,
    initial_state,
    x_edges,
    y_edges,
    times,
    *,
    imposed_dt=None,
    boundary="reflective",
    wall_cap_s=5.0,
):
    """Small verification records only; an imposed unstable dt is explicit."""
    if not run_id.startswith("fixture_genuine_"):
        raise ValueError("small fixtures require distinct fixture_genuine_ prefix")
    start = time.perf_counter()
    xe, ye = _edges(x_edges, "x_edges"), _edges(y_edges, "y_edges")
    times = np.asarray(times, dtype=np.float64)
    if (
        times.ndim != 1
        or len(times) < 2
        or times[0] != 0
        or np.any(np.diff(times) <= 0)
    ):
        raise ValueError("fixture requires initial and increasing positive times")
    if len(xe) > 33 or len(ye) > 33:
        raise ValueError("fixture grids are bounded at32cells per axis")
    resolved = {
        "case": {"case_id": "genuine_2d_small_fixture", "dimension": 2},
        "request": {"run_id": run_id, "device": "cpu", "dtype": "float64"},
        "fixture": {
            "imposed_dt": imposed_dt,
            "boundary": boundary,
            "wall_cap_s": wall_cap_s,
        },
        "source_sha256": sha256(Path(__file__)),
    }
    with _record(output_root, run_id, resolved) as record:
        return _advance_record(
            record,
            initial_state,
            xe,
            ye,
            times,
            imposed_dt=imposed_dt,
            boundary=boundary,
            wall_cap_s=wall_cap_s,
            start=start,
        )


def run_reference(
    config_path, run_id, *, protocol_path, release_path=None, root=Path(".")
):
    """Run ONE explicitly released matrix cell; never dispatch the whole matrix."""
    start = time.perf_counter()
    root = Path(root).resolve()
    cfg = load_case_2d(config_path, root)
    if release_path is None:
        raise ValueError(
            "canonical reference solve is not released: provide exact root cell release"
        )
    protocol = verify_reference_protocol(protocol_path, root)
    if protocol["case"] != cfg:
        raise ValueError("configuration differs from frozen reference protocol")
    release = _json(root / release_path)
    if (
        release.get("state") != "released"
        or release.get("protocol_sha256") != protocol["protocol_sha256"]
        or release.get("run_id") != run_id
    ):
        raise ValueError("root cell release/protocol/run ID mismatch")
    cells = [c for c in cfg["matrix"] if c["id"] == run_id]
    if len(cells) != 1:
        raise ValueError("unknown canonical reference cell ID")
    cell = cells[0]
    cap = release.get("numerical_wall_cap_s")
    if (
        isinstance(cap, bool)
        or not isinstance(cap, (int, float))
        or not math.isfinite(cap)
        or cap <= 0
    ):
        raise ValueError("release requires exact positive numerical wall cap")
    if run_id == "ref_rect80_c045_v01" and cap != 60.0:
        raise ValueError("first canonical cell cap is frozen at60seconds")
    output_root = root / "results/extensions/two_dimensional/reference_v01"
    resolved = {
        "case": cfg,
        "cell": cell,
        "protocol_sha256": protocol["protocol_sha256"],
        "request": {"run_id": run_id, "device": "cpu", "dtype": "float64"},
        "release_path": str(release_path),
        "release_sha256": sha256(root / release_path),
    }
    with _record(output_root, run_id, resolved, root=root) as record:
        axes = _grid(cell["nx"], cell["ny"])
        initial = rectangle_initial(axes["x_edges_m"], axes["y_edges_m"])
        with np.load(
            (root / protocol_path).parent / f"axes_operator_n{cell['nx']}.npz"
        ) as a:
            operator = {k: a[k].copy() for k in ("weights_x", "weights_y")}
        _write(record.path / "protocol.json", protocol)
        _write(record.path / "root_release.json", release)
        return _advance_record(
            record,
            initial,
            axes["x_edges_m"],
            axes["y_edges_m"],
            axes["t_s"],
            cfl=cell["cfl_sum"],
            wall_cap_s=float(cap),
            start=start,
            canonical=True,
            protocol=protocol,
            operator=operator,
        )


def area_difference_metrics(difference, reference, cell_areas):
    """Area norms with equal positive-snapshot weights, retaining zero norms."""
    d, r, a = (
        np.asarray(v, dtype=np.float64) for v in (difference, reference, cell_areas)
    )
    if (
        d.ndim != 4
        or d.shape != r.shape
        or d.shape[-1] != 3
        or a.shape != d.shape[1:3]
        or np.any(a <= 0)
        or not all(np.isfinite(v).all() for v in (d, r, a))
    ):
        raise ValueError("norms require matching finite(time,y,x,3), positive areas")
    weight = a[None, :, :, None]
    l1 = (weight * np.abs(d)).sum(axis=(1, 2))
    l2_squared = (weight * d * d).sum(axis=(1, 2))
    ref_squared = (weight * r * r).sum(axis=(1, 2))

    def row(first, second, denominator):
        answer = []
        for j in range(3):
            answer.append(
                {
                    "area_mean_MAE": float(first[j] / a.sum()),
                    "area_mean_RMSE": float(np.sqrt(second[j] / a.sum())),
                    "integral_L1": float(first[j]),
                    "integral_L2": float(np.sqrt(second[j])),
                    "relative_L2_pct": float(100 * np.sqrt(second[j] / denominator[j]))
                    if denominator[j] > 0
                    else None,
                    "relative_reason": None
                    if denominator[j] > 0
                    else "zero reference norm",
                }
            )
        return answer

    return {
        "component_order": ["h", "qx", "qy"],
        "per_time": [
            row(x, y, z) for x, y, z in zip(l1, l2_squared, ref_squared, strict=True)
        ],
        "equal_positive_time_aggregate": row(
            l1.mean(0), l2_squared.mean(0), ref_squared.mean(0)
        ),
        "integral_units_L1": ["m^3", "m^4/s", "m^4/s"],
        "integral_units_L2": ["m^2", "m^3/s", "m^3/s"],
    }


def compare_completed_cells(protocol_path, output_dir, *, root=Path(".")):
    """Summarize existing valid four-cell records; never execute a solver."""
    root, output = Path(root).resolve(), Path(output_dir)
    protocol = verify_reference_protocol(protocol_path, root)
    if (root / output).exists():
        raise FileExistsError("reference comparison output must be new")
    fields, artifacts = {}, {}
    for cell in protocol["case"]["matrix"]:
        run = root / "results/extensions/two_dimensional/reference_v01" / cell["id"]
        status, summary = _json(run / "status.json"), _json(run / "summary.json")
        if (
            status["state"] != "completed"
            or summary["termination_kind"] != "completed"
            or summary["protocol_sha256"] != protocol["protocol_sha256"]
            or not all(summary["criteria"].values())
        ):
            raise ValueError(
                f"incomplete/invalid required reference cell: {cell['id']}"
            )
        hashes = _json(run / "hashes.json")
        for name, expected in hashes["files"].items():
            if sha256(run / name) != expected:
                raise ValueError(f"saved reference cell artifact changed: {run / name}")
        with np.load(run / "common_reference.npz") as f:
            if not np.array_equal(f["t_s"], requested_times()):
                raise ValueError("common comparison times differ")
            fields[cell["id"]] = np.stack(
                [f[k][1:] for k in ("h_m", "qx_m2_s", "qy_m2_s")], axis=-1
            )
            areas = f["cell_area_m2"].copy()
        artifacts[cell["id"]] = {
            "common_reference_sha256": sha256(run / "common_reference.npz"),
            "summary_sha256": sha256(run / "summary.json"),
        }
    a, b, c, d = (
        fields[k]
        for k in (
            "ref_rect80_c045_v01",
            "ref_rect160_c045_v01",
            "ref_rect320_c045_v01",
            "ref_rect320_c0225_v01",
        )
    )
    (root / output).mkdir(parents=True, exist_ok=False)
    space_first, space_last, time_difference = b - a, c - b, d - c
    _save_npz(
        root / output / "signed_differences_and_sensitivity.npz",
        **{k: v for k, v in _grid(80, 80).items() if k != "t_s"},
        t_s=requested_times()[1:],
        coupled_160_minus_80=space_first,
        coupled_320_minus_160=space_last,
        fixed_grid_half_cfl_minus_cfl=time_difference,
        D_space=np.abs(space_last),
        D_time=np.abs(time_difference),
        D_sum=np.abs(space_last) + np.abs(time_difference),
    )
    _write(
        root / output / "norms.json",
        {
            "coupled_160_minus_80": area_difference_metrics(space_first, b, areas),
            "coupled_320_minus_160": area_difference_metrics(space_last, c, areas),
            "fixed_grid_half_cfl_minus_cfl": area_difference_metrics(
                time_difference, d, areas
            ),
            "protocol_sha256": protocol["protocol_sha256"],
            "inputs": artifacts,
            "preselected_reference": "ref_rect320_c0225_v01",
            "qualification": (
                "Signed coupled mesh/time and separate fixed-grid time "
                "sensitivities. Dsum is not an error bound, confidence "
                "interval, Richardson extrapolation or proof of asymptotic "
                "order."
            ),
        },
    )
    return root / output
