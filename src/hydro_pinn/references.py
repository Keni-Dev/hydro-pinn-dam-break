"""Pointwise, CPU float64 exact references for the two rest-state dam breaks.

The equations are independently implemented from SWASHES v7 sections 4.1.1-2
and the Clawpack Riemann-book shallow-water chapter. No external solver code is
copied. These are idealized analytical values, not measured field observations.
"""

from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from hydro_pinn.config import resolve_run_config, validate_config
from hydro_pinn.physics import conservative_flux, state_from_depth_velocity
from hydro_pinn.records import RunRecord, write_json_atomic

FloatArray = NDArray[np.float64]

REFERENCE_SOURCES = [
    {
        "title": "SWASHES: a compilation of Shallow Water Analytic Solutions",
        "authors": "Delestre et al.",
        "version": "arXiv:1110.0288v7, revised 2016-01-21",
        "url": "https://arxiv.org/abs/1110.0288v7",
        "sections": [
            "4.1.1 wet-domain Stoker solution",
            "4.1.2 dry-domain Ritter solution",
        ],
        "license_url": "https://arxiv.org/licenses/nonexclusive-distrib/1.0/license.html",
        "software_license": "CeCILL-V2; SWASHES software is not reused here",
        "reuse": "mathematical definitions only; no manuscript text or software copied",
    },
    {
        "title": "Clawpack Riemann book: The shallow water equations",
        "url": "https://faculty.washington.edu/rjl/riemann_book/Shallow_water.html",
        "source_url": "https://github.com/clawpack/riemann_book/blob/master/exact_solvers/shallow_water.py",
        "version": "5b171f11fd7a3e810ee7a74a4d9f2dff01d65a6e (2025-11-30)",
        "pinned_source_url": "https://raw.githubusercontent.com/clawpack/riemann_book/5b171f11fd7a3e810ee7a74a4d9f2dff01d65a6e/exact_solvers/shallow_water.py",
        "code_license": "BSD-3-Clause, LICENSE-CODE",
        "text_license": "CC-BY-NC-ND-4.0, LICENSE-TEXT.md",
        "reuse": "independent comparison of wave/fan definitions; no code copied",
    },
]


class ReferenceError(ValueError):
    """An analytical assumption, root solve, or evaluation request is invalid."""


@dataclass(frozen=True)
class WetStar:
    """Wet-bed intermediate state, physical wave speeds, and root diagnostics."""

    h_m: float
    u_m_s: float
    q_m2_s: float
    c_m_s: float
    shock_speed_m_s: float
    fan_head_speed_m_s: float
    fan_tail_speed_m_s: float
    root_residual_m_s: float
    iterations: int
    function_calls: int
    converged: bool
    stop_reason: str


@dataclass(frozen=True)
class ReferenceSolution:
    """SI axes and fields; every field has shape (len(t_s), len(x_m)).

    ``velocity_valid`` is h>0. A stored zero velocity on genuinely dry points is
    an explicit convention; later metric wet masks may use a larger threshold.
    """

    x_m: FloatArray
    t_s: FloatArray
    h_m: FloatArray
    q_m2_s: FloatArray
    u_m_s: FloatArray
    velocity_valid: NDArray[np.bool_]


def _case(case: dict[str, Any]) -> dict[str, Any]:
    validated = validate_config(case)
    if validated["config_type"] != "case":
        raise ReferenceError("reference input must be a case configuration")
    return validated


def _wet_star(case: dict[str, Any]) -> WetStar:
    if case["reference"]["kind"] != "wet_bed":
        raise ReferenceError("wet intermediate solve requires the wet_bed reference")
    gravity = case["gravity_m_s2"]
    h_left, h_right = case["left_depth_m"], case["right_depth_m"]
    c0 = math.sqrt(gravity * h_left)

    def mismatch(depth: float) -> float:
        rarefaction_velocity = 2 * (c0 - math.sqrt(gravity * depth))
        shock_velocity = (depth - h_right) * math.sqrt(
            gravity * (1 / depth + 1 / h_right) / 2
        )
        return rarefaction_velocity - shock_velocity

    root, result = brentq(
        mismatch,
        h_right,
        h_left,
        xtol=case["reference"]["root_tolerance_m"],
        rtol=4 * np.finfo(np.float64).eps,
        maxiter=case["reference"]["root_max_iterations"],
        full_output=True,
        disp=False,
    )
    if not result.converged or not h_right < root < h_left:
        raise ReferenceError(
            f"wet-bed bracketed root did not converge in "
            f"{result.iterations} iterations: {result.flag}"
        )
    sound = math.sqrt(gravity * root)
    velocity = 2 * (c0 - sound)
    discharge = root * velocity
    return WetStar(
        h_m=float(root),
        u_m_s=velocity,
        q_m2_s=discharge,
        c_m_s=sound,
        shock_speed_m_s=discharge / (root - h_right),
        fan_head_speed_m_s=-c0,
        fan_tail_speed_m_s=velocity - sound,
        root_residual_m_s=mismatch(root),
        iterations=int(result.iterations),
        function_calls=int(result.function_calls),
        converged=True,
        stop_reason=str(result.flag),
    )


def solve_wet_star(case: dict[str, Any]) -> WetStar:
    """Find the left-rarefaction/right-shock state using bracketed Brent solve."""
    return _wet_star(_case(case))


def _wave_diagnostics(case: dict[str, Any]) -> dict[str, Any]:
    gravity = case["gravity_m_s2"]
    c0 = math.sqrt(gravity * case["left_depth_m"])
    dam, horizon = case["dam_position_m"], case["final_time_s"]
    left, right = case["domain_m"]
    wet = case["reference"]["kind"] == "wet_bed"
    star = _wet_star(case) if wet else None
    speeds = {"fan_head": -c0}
    speeds.update(
        {"fan_tail": star.fan_tail_speed_m_s, "shock": star.shock_speed_m_s}
        if star is not None
        else {"dry_front": 2 * c0}
    )
    contacts = [
        (left - dam) / speed if speed < 0 else (right - dam) / speed
        for speed in speeds.values()
        if speed != 0
    ]
    safe_horizon = min(contacts)
    positions = {name: dam + speed * horizon for name, speed in speeds.items()}
    if not horizon < safe_horizon or not all(
        left < value < right for value in positions.values()
    ):
        raise ReferenceError(
            f"case.final_time_s={horizon} must precede exact boundary contact "
            f"at {safe_horizon:.16g} s; revised boundary treatment is required"
        )
    checks: dict[str, Any] = {
        "reference_kind": case["reference"]["kind"],
        "wave_speeds_m_s": speeds,
        "positions_at_final_time_m": positions,
        "first_boundary_contact_s": safe_horizon,
        "requested_horizon_s": horizon,
        "all_waves_strictly_inside": True,
        "fan_head_depth_continuity_error_m": (2 * c0 + c0) ** 2 / (9 * gravity)
        - case["left_depth_m"],
        "fan_head_velocity_continuity_error_m_s": 2 * (c0 + speeds["fan_head"]) / 3,
        "branch_conventions": {
            "t0": "left state for x<=dam; right state for x>dam",
            "fan_head": "left state at equality",
            "fan_tail": "wet intermediate state at equality",
            "shock": "right state at equality",
            "dry_front": "exact dry state at equality",
        },
    }
    if star is not None:
        left_state = state_from_depth_velocity(np.array(star.h_m), np.array(star.u_m_s))
        right_state = state_from_depth_velocity(
            np.array(case["right_depth_m"]), np.array(0.0)
        )
        jump = conservative_flux(right_state, gravity) - conservative_flux(
            left_state, gravity
        )
        jump -= star.shock_speed_m_s * (right_state - left_state)
        tail = star.fan_tail_speed_m_s
        checks.update(
            wet_star=asdict(star),
            rankine_hugoniot_mass_residual_m2_s=float(jump[0]),
            rankine_hugoniot_momentum_residual_m3_s2=float(jump[1]),
            fan_tail_depth_continuity_error_m=(2 * c0 - tail) ** 2 / (9 * gravity)
            - star.h_m,
            fan_tail_velocity_continuity_error_m_s=2 * (c0 + tail) / 3 - star.u_m_s,
        )
    else:
        checks.update(dry_front_depth_m=0.0, dry_front_discharge_m2_s=0.0)
    tolerances = {
        "fan_head_depth_continuity_error_m": 1e-12,
        "fan_head_velocity_continuity_error_m_s": 1e-12,
        "fan_tail_depth_continuity_error_m": 1e-12,
        "fan_tail_velocity_continuity_error_m_s": 1e-12,
        "rankine_hugoniot_mass_residual_m2_s": 1e-10,
        "rankine_hugoniot_momentum_residual_m3_s2": 1e-10,
    }
    errors = {key: value for key, value in checks.items() if key in tolerances}
    checks["absolute_check_tolerances"] = {key: tolerances[key] for key in errors}
    if star is not None:
        errors["root_residual_m_s"] = star.root_residual_m_s
        checks["absolute_check_tolerances"]["root_residual_m_s"] = 1e-10
    checks["identity_checks_passed"] = all(
        math.isfinite(value) and abs(value) <= checks["absolute_check_tolerances"][key]
        for key, value in errors.items()
    )
    if not checks["identity_checks_passed"]:
        raise ReferenceError(f"analytical identity checks failed: {errors}")
    return checks


def wave_diagnostics(case: dict[str, Any]) -> dict[str, Any]:
    """Return root, jump, continuity, and actual boundary-contact diagnostics."""
    return _wave_diagnostics(_case(case))


def _axis(values: ArrayLike, name: str) -> FloatArray:
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0 or raw.dtype.kind not in "iuf":
        raise ReferenceError(f"{name} must be a nonempty one-dimensional real axis")
    result = np.array(raw, dtype=np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ReferenceError(f"{name} must contain only finite SI values")
    return result


def evaluate_reference(
    case: dict[str, Any], x_m: ArrayLike, t_s: ArrayLike
) -> ReferenceSolution:
    """Evaluate physical point values on arbitrary one-dimensional SI axes.

    Initial dam equality belongs to the left state. At positive time the wet
    shock equality belongs to the right state; continuous fan equalities follow
    the documented reservoir/intermediate conventions. No depth floor is used.
    """
    case = _case(case)
    diagnostics = _wave_diagnostics(case)
    return _evaluate_reference_with_diagnostics(case, x_m, t_s, diagnostics)


def _evaluate_reference_with_diagnostics(
    case: dict[str, Any],
    x_m: ArrayLike,
    t_s: ArrayLike,
    diagnostics: dict[str, Any],
) -> ReferenceSolution:
    """Reuse a validated case's checked intermediate state without another solve."""
    x, times = _axis(x_m, "x_m"), _axis(t_s, "t_s")
    left, right = case["domain_m"]
    if np.any((x < left) | (x > right)):
        raise ReferenceError("x_m must lie within case.domain_m")
    if np.any((times < 0) | (times > case["final_time_s"])):
        raise ReferenceError("t_s must lie in [0,case.final_time_s]")
    depth = np.empty((len(times), len(x)), dtype=np.float64)
    velocity = np.zeros_like(depth)
    dam, gravity = case["dam_position_m"], case["gravity_m_s2"]
    c0 = math.sqrt(gravity * case["left_depth_m"])
    star = diagnostics.get("wet_star")
    for index, elapsed in enumerate(times):
        if elapsed == 0:
            depth[index] = np.where(
                x <= dam, case["left_depth_m"], case["right_depth_m"]
            )
            continue
        xi = (x - dam) / elapsed
        fan_head = xi <= -c0
        if star is None:
            fan = (xi > -c0) & (xi < 2 * c0)
            depth[index].fill(0.0)
        else:
            fan = (xi > -c0) & (xi < star["fan_tail_speed_m_s"])
            middle = (xi >= star["fan_tail_speed_m_s"]) & (xi < star["shock_speed_m_s"])
            depth[index].fill(case["right_depth_m"])
            depth[index, middle] = star["h_m"]
            velocity[index, middle] = star["u_m_s"]
        depth[index, fan_head] = case["left_depth_m"]
        depth[index, fan] = (2 * c0 - xi[fan]) ** 2 / (9 * gravity)
        velocity[index, fan] = 2 * (c0 + xi[fan]) / 3
    discharge = depth * velocity
    return ReferenceSolution(x, times, depth, discharge, velocity, depth > 0)


def _save_outputs(path: Path, solution: ReferenceSolution) -> None:
    """Atomically claim a new NPZ; never replace previously saved arrays."""
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npz.tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            np.savez(handle, **asdict(solution))
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def generate_reference_run(
    case_path: str | Path,
    evaluation_path: str | Path,
    run_id: str,
    output_root: str | Path = "runs",
) -> Path:
    """Resolve immutable inputs and create one analytical point-array record.

    Consumers must require ``status.json`` state ``completed``. Verification
    coordinates remain distinct from any later frozen final-evaluation runs.
    """
    resolved = resolve_run_config(
        case_path,
        evaluation_path=evaluation_path,
        device="cpu",
        dtype="float64",
        run_id=run_id,
    )
    metadata = {
        "stage": "analytical_reference",
        "data_origin": "analytical_generated",
        "state_location": "point_value",
        "velocity_dry_convention": "zero with velocity_valid=False",
        "array_units": {
            "x_m": "m",
            "t_s": "s",
            "h_m": "m",
            "q_m2_s": "m^2/s",
            "u_m_s": "m/s",
        },
        "sources": REFERENCE_SOURCES,
    }
    with RunRecord.create(output_root, run_id, resolved, metadata=metadata) as run:
        started = time.perf_counter()
        case, evaluation = resolved["case"], resolved["evaluation"]
        left, right = evaluation["grid"]["domain_m"]
        cells = evaluation["grid"]["cells"]
        x = left + (np.arange(cells, dtype=np.float64) + 0.5) * (right - left) / cells
        checks = wave_diagnostics(case)
        solution = _evaluate_reference_with_diagnostics(
            case, x, evaluation["times_s"], checks
        )
        write_json_atomic(run.path / "checks.json", checks)
        write_json_atomic(run.path / "reference_provenance.json", metadata)
        _save_outputs(run.path / "outputs.npz", solution)
        star = checks.get("wet_star", {})
        run.complete(
            stopping_reason="analytical values and reference diagnostics saved",
            elapsed_timings_s={"reference_generation": time.perf_counter() - started},
            iterations=star.get("iterations", 0),
            function_evaluations=star.get("function_calls", 0),
        )
    return run.path
