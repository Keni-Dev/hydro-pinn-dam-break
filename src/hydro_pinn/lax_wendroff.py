"""Unlimited two-step (Richtmyer) Lax-Wendroff for flat-bed 1D SWE.

States are physical cell averages U=(h,q). One fixed reservoir ghost cell on
each side supplies the boundary predictor. The corrector uses those numerical
interface fluxes, including at the two boundaries. There is no clipping,
limiter, viscosity, depth floor, or retry with a smaller step after failure.
"""

from __future__ import annotations

import hashlib
import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hydro_pinn.config import resolve_run_config, validate_config
from hydro_pinn.physics import PhysicsError, conservative_flux, velocity_from_state
from hydro_pinn.records import RunRecord, write_json_atomic

type FloatArray = NDArray[np.float64]
METHOD = "unlimited_two_step_lax_wendroff"


class SolverFailure(ValueError):
    """A rejected state, with enough context to preserve the failed attempt."""

    def __init__(
        self,
        reason: str,
        stage: str,
        state: FloatArray,
        time_s: float,
        step: int,
        offending_index: int | None = None,
    ) -> None:
        super().__init__(reason)
        self.reason, self.stage = reason, stage
        self.state = np.array(state, dtype=np.float64, copy=True)
        self.time_s, self.step = time_s, step
        self.offending_index = offending_index

    def as_dict(self) -> dict[str, Any]:
        bad = (
            ~np.isfinite(self.state).all(axis=-1)
            | (self.state[..., 0] < 0)
            | ((self.state[..., 0] == 0) & (self.state[..., 1] != 0))
        )
        indices = np.flatnonzero(bad)
        index = self.offending_index
        if index is None and len(indices):
            index = int(indices[0])
        raw = self.state[index] if index is not None else None
        return {
            "reason": self.reason,
            "stage": self.stage,
            "time_s": self.time_s,
            "step": self.step,
            "offending_index": index,
            "offending_state_h_q": None
            if raw is None
            else [float(v) if np.isfinite(v) else None for v in raw],
            "nonfinite_encoding": (
                "null in JSON; raw values retained in failed_state.npz"
            ),
            "negative_depth_count": int(np.count_nonzero(self.state[..., 0] < 0)),
            "nonfinite_value_count": int(np.count_nonzero(~np.isfinite(self.state))),
            "inconsistent_dry_count": int(
                np.count_nonzero((self.state[..., 0] == 0) & (self.state[..., 1] != 0))
            ),
            "index_semantics": (
                "interface index 0..N (including boundary interfaces)"
                if self.stage == "predictor"
                else "cell index; current step checks include one ghost per side"
            ),
        }


class NumericalRunFailed(ValueError):
    """The CLI run was saved as failed; its path is available for readback."""

    def __init__(self, path: Path, failure: dict[str, Any]) -> None:
        self.path, self.failure = path, failure
        super().__init__(
            f"{failure['stage']} at t={failure['time_s']:.16g} s, "
            f"step {failure['step']}: {failure['reason']}; evidence: {path}"
        )


def _check_state(state: FloatArray, stage: str, time_s: float, step: int) -> None:
    reason = None
    if not np.isfinite(state).all():
        reason = "nonfinite state"
    elif np.any(state[..., 0] < 0):
        reason = "negative depth"
    elif np.any((state[..., 0] == 0) & (state[..., 1] != 0)):
        reason = "zero depth with nonzero discharge"
    if reason:
        raise SolverFailure(reason, stage, state, time_s, step)


def _flux(
    state: FloatArray, gravity: float, stage: str, time_s: float, step: int
) -> FloatArray:
    _check_state(state, stage, time_s, step)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            return conservative_flux(state, gravity)
    except (PhysicsError, FloatingPointError) as exc:
        raise SolverFailure(f"invalid flux: {exc}", stage, state, time_s, step) from exc


def advance_one_step(
    state: ArrayLike,
    left_state: ArrayLike,
    right_state: ArrayLike,
    dx_m: float,
    dt_s: float,
    gravity_m_s2: float = 9.81,
    *,
    time_s: float = 0.0,
    step: int = 1,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return accepted cells, actual N+1 interface fluxes, and predictor.

    This explicit step interface permits hand calculations and failure checks.
    ``solve`` chooses CFL-controlled dt; this function uses the supplied dt.
    The current/predictor/corrector checks happen before unsafe flux evaluation.
    """
    state = np.asarray(state, dtype=np.float64)
    left, right = np.asarray(left_state), np.asarray(right_state)
    if state.ndim != 2 or state.shape[1] != 2 or len(state) < 1:
        raise ValueError("state must have shape (cells,2), ordered (h,q)")
    if left.shape != (2,) or right.shape != (2,):
        raise ValueError("reservoir states must have shape (2,)")
    for name, value in (("dx_m", dx_m), ("dt_s", dt_s), ("gravity", gravity_m_s2)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    extended = np.vstack((left, state, right)).astype(np.float64)
    flux = _flux(extended, gravity_m_s2, "current", time_s, step)
    with np.errstate(over="ignore", invalid="ignore"):
        predictor = 0.5 * (extended[:-1] + extended[1:])
        predictor -= dt_s / (2 * dx_m) * (flux[1:] - flux[:-1])
    interface_flux = _flux(
        predictor, gravity_m_s2, "predictor", time_s + dt_s / 2, step
    )
    with np.errstate(over="ignore", invalid="ignore"):
        updated = state - dt_s / dx_m * (interface_flux[1:] - interface_flux[:-1])
    _check_state(updated, "corrector", time_s + dt_s, step)
    return updated, interface_flux, predictor


def initial_cell_averages(case: dict[str, Any], cells: int) -> tuple[FloatArray, ...]:
    """Return native centres, edges, and exact dam-aligned initial averages."""
    if isinstance(cells, bool) or not isinstance(cells, int) or cells < 2 or cells % 2:
        raise ValueError("cells must be an even integer >=2")
    a, b = case["domain_m"]
    dx = (b - a) / cells
    face = (case["dam_position_m"] - a) / dx
    if not math.isclose(face, round(face), abs_tol=1e-10):
        raise ValueError("dam must align with a cell interface for the selected cells")
    edges = np.linspace(a, b, cells + 1, dtype=np.float64)
    x = a + (np.arange(cells, dtype=np.float64) + 0.5) * dx
    h = np.where(
        np.arange(cells) < round(face), case["left_depth_m"], case["right_depth_m"]
    )
    return x, edges, np.column_stack((h, np.zeros(cells)))


def piecewise_constant_reconstruct(
    cell_edges_m: ArrayLike, native_values: ArrayLike, x_m: ArrayLike
) -> FloatArray:
    """Sample (...,N) cell averages; choose the right cell at internal faces."""
    edges = np.asarray(cell_edges_m, dtype=np.float64)
    values, x = (
        np.asarray(native_values, dtype=np.float64),
        np.asarray(x_m, dtype=np.float64),
    )
    if edges.ndim != 1 or len(edges) < 2 or not np.isfinite(edges).all():
        raise ValueError("cell edges must be a finite one-dimensional axis")
    if (
        np.any(np.diff(edges) <= 0)
        or values.ndim < 1
        or values.shape[-1] != len(edges) - 1
    ):
        raise ValueError("increasing cell edges must match the last native value axis")
    if (
        x.ndim != 1
        or not np.isfinite(x).all()
        or np.any((x < edges[0]) | (x > edges[-1]))
    ):
        raise ValueError(
            "sample coordinates must be finite and inside the native domain"
        )
    indices = np.searchsorted(edges, x, side="right") - 1
    indices[x == edges[-1]] = len(edges) - 2
    return values[..., indices]


@dataclass
class SolverResult:
    """Requested native outputs, raw histories, and explicit failure state."""

    x_m: FloatArray
    cell_edges_m: FloatArray
    t_s: FloatArray
    h_m: FloatArray
    q_m2_s: FloatArray
    diagnostics: dict[str, Any]
    failure: dict[str, Any] | None
    last_valid_state: FloatArray | None
    last_valid_time_s: float | None
    rejected_state: FloatArray | None = None


def solve(
    case: dict[str, Any],
    cells: int,
    output_times_s: ArrayLike,
    *,
    initial_state: ArrayLike | None = None,
) -> SolverResult:
    """Integrate CPU float64 cell averages, stopping on the first bad state.

    The optional initial state is an explicit diagnostic hook; canonical CLI
    runs always use the prescribed cell averages. Zero maximum wave speed
    stops with a current-stage diagnostic because the selected CFL rule has
    no finite step. No manufactured final snapshot is returned after failure.
    """
    case = validate_config(case)
    if case["config_type"] != "case":
        raise ValueError("solve requires a physical case configuration")
    times = np.asarray(output_times_s, dtype=np.float64)
    if (
        times.ndim != 1
        or len(times) == 0
        or not np.isfinite(times).all()
        or np.any(times < 0)
        or np.any(np.diff(times) <= 0)
        or times[-1] > case["final_time_s"]
    ):
        raise ValueError("output times must increase within [0,case.final_time_s]")
    x, edges, prescribed = initial_cell_averages(case, cells)
    state = (
        prescribed.copy()
        if initial_state is None
        else np.array(initial_state, dtype=np.float64, copy=True)
    )
    if state.shape != (cells, 2):
        raise ValueError("initial_state must have shape (cells,2)")
    a, b = case["domain_m"]
    dx, gravity, cfl = (b - a) / cells, case["gravity_m_s2"], case["grid"]["cfl"]
    boundary = case["boundary"]
    left = np.array([boundary["left_h_m"], boundary["left_q_m2_s"]])
    right = np.array([boundary["right_h_m"], boundary["right_q_m2_s"]])
    prescribed_mass = float(
        (case["dam_position_m"] - a) * case["left_depth_m"]
        + (b - case["dam_position_m"]) * case["right_depth_m"]
    )
    initial_mass = float(dx * state[:, 0].sum()) if np.isfinite(state).all() else None
    history: list[dict[str, Any]] = []
    masses: list[dict[str, Any]] = []
    saved_times, snapshots = [], []
    elapsed, step, integrated_flux = 0.0, 0, 0.0
    last_valid, last_time, rejected, failure = None, None, None, None
    attempted_dt, attempted_cfl = None, None

    def mass_record() -> dict[str, Any]:
        mass = float(dx * state[:, 0].sum())
        return {
            "time_s": elapsed,
            "mass_m2": mass,
            "integrated_boundary_outflow_m2": integrated_flux,
            "signed_balance_m2": mass - prescribed_mass + integrated_flux,
            "minimum_depth_m": float(state[:, 0].min()),
            "maximum_depth_m": float(state[:, 0].max()),
            "negative_depth_count": int(np.count_nonzero(state[:, 0] < 0)),
            "nonfinite_value_count": int(np.count_nonzero(~np.isfinite(state))),
        }

    try:
        _check_state(state, "current", elapsed, step)
        last_valid, last_time = state.copy(), elapsed
        masses.append(mass_record())
        targets = sorted(set(times.tolist() + [case["final_time_s"]]))
        for target in targets:
            while elapsed < target:
                step += 1
                _check_state(state, "current", elapsed, step)
                try:
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        velocity, _ = velocity_from_state(state)
                except (PhysicsError, FloatingPointError) as exc:
                    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                        raw_u = state[:, 1] / np.where(state[:, 0] > 0, state[:, 0], 1)
                    bad_index = int(np.flatnonzero(~np.isfinite(raw_u))[0])
                    raise SolverFailure(
                        f"invalid current velocity: {exc}",
                        "current",
                        state,
                        elapsed,
                        step,
                        bad_index,
                    ) from exc
                with np.errstate(over="ignore", invalid="ignore"):
                    speeds = np.abs(velocity) + np.sqrt(gravity * state[:, 0])
                    speed = float(np.max(speeds))
                if not math.isfinite(speed) or speed <= 0:
                    raise SolverFailure(
                        "maximum wave speed is nonfinite or zero; CFL step undefined",
                        "current",
                        state,
                        elapsed,
                        step,
                        int(np.argmax(speeds)),
                    )
                remaining = target - elapsed
                attempted_dt = min(cfl * dx / speed, remaining)
                attempted_cfl = attempted_dt * speed / dx
                if attempted_dt <= 0 or elapsed + attempted_dt == elapsed:
                    raise SolverFailure(
                        "time step cannot advance floating-point time",
                        "current",
                        state,
                        elapsed,
                        step,
                    )
                updated, flux, predictor = advance_one_step(
                    state,
                    left,
                    right,
                    dx,
                    attempted_dt,
                    gravity,
                    time_s=elapsed,
                    step=step,
                )
                next_time = (
                    target if attempted_dt == remaining else elapsed + attempted_dt
                )
                left_q, right_q = float(flux[0, 0]), float(flux[-1, 0])
                integrated_flux += attempted_dt * (right_q - left_q)
                history.append(
                    {
                        "step": step,
                        "time_start_s": elapsed,
                        "time_end_s": next_time,
                        "dt_s": attempted_dt,
                        "cfl": attempted_cfl,
                        "maximum_wave_speed_m_s": speed,
                        "left_boundary_flux_m2_s": left_q,
                        "right_boundary_flux_m2_s": right_q,
                        "left_boundary_flux_mismatch_m2_s": left_q - left[1],
                        "right_boundary_flux_mismatch_m2_s": right_q - right[1],
                        "predictor_minimum_depth_m": float(predictor[:, 0].min()),
                        "predictor_maximum_depth_m": float(predictor[:, 0].max()),
                    }
                )
                state, elapsed = updated, next_time
                last_valid, last_time = state.copy(), elapsed
                masses.append(mass_record())
            if target in times:
                saved_times.append(target)
                snapshots.append(state.copy())
    except SolverFailure as exc:
        failure, rejected = exc.as_dict(), exc.state
        failure.update(attempted_dt_s=attempted_dt, attempted_cfl=attempted_cfl)
    states = np.array(snapshots, dtype=np.float64).reshape((-1, cells, 2))
    diagnostics = {
        "method": METHOD,
        "state_location": "cell_average",
        "boundary_treatment": (
            "one fixed reservoir ghost per side; predictor flux at each boundary"
        ),
        "mass_rule": "dx*sum(h); signed balance=M-M_IC+sum(dt*(Fh_right-Fh_left))",
        "boundary_flux_quadrature": (
            "actual corrector flux, rectangular in each accepted step"
        ),
        "time_output_policy": "hit_requested_times; no time interpolation",
        "cfl_target": cfl,
        "cells": cells,
        "dx_m": dx,
        "prescribed_initial_mass_m2": prescribed_mass,
        "initial_mass_m2": initial_mass,
        "initial_volume_error_m2": None
        if initial_mass is None
        else initial_mass - prescribed_mass,
        "initial_state_source": "prescribed cell averages"
        if initial_state is None
        else "explicit diagnostic override",
        "requested_output_times_s": times.tolist(),
        "completed_output_times_s": saved_times,
        "final_time_reached_s": last_time,
        "accepted_steps": len(history),
        "attempted_steps": step,
        "maximum_cfl": max((row["cfl"] for row in history), default=None),
        "maximum_absolute_balance_m2": max(
            (abs(row["signed_balance_m2"]) for row in masses), default=None
        ),
        "minimum_accepted_depth_m": min(
            (row["minimum_depth_m"] for row in masses), default=None
        ),
        "steps": history,
        "cell_sum_mass": masses,
        "failure": failure,
    }
    return SolverResult(
        x,
        edges,
        np.array(saved_times),
        states[..., 0],
        states[..., 1],
        diagnostics,
        failure,
        last_valid,
        last_time,
        rejected,
    )


def _save_npz(path: Path, **arrays: Any) -> None:
    """Claim one new array file atomically, preserving any existing evidence."""
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".npz.tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            np.savez(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def generate_solver_run(
    case_path: str | Path,
    evaluation_path: str | Path,
    cells: int,
    run_id: str,
    output_root: str | Path = "runs",
) -> Path:
    """Create a canonical native solver record; numerical failure raises after save."""
    resolved = resolve_run_config(
        case_path,
        evaluation_path=evaluation_path,
        device="cpu",
        dtype="float64",
        run_id=run_id,
    )
    # Validate requested grid before claiming a run directory.
    initial_cell_averages(resolved["case"], cells)
    resolved["solver"] = {
        "method": METHOD,
        "cells": cells,
        "state_location": "cell_average",
        "numerical_floor": None,
    }
    module_paths = [
        Path(__file__),
        Path(__file__).with_name("physics.py"),
        Path(__file__).with_name("config.py"),
        Path(__file__).with_name("cli.py"),
        Path(__file__).with_name("records.py"),
    ]
    metadata = {
        "stage": "numerical_solver",
        "method": METHOD,
        "data_origin": "numerical_approximation",
        "state_location": "cell_average",
        "numerical_floor": None,
        "modifications": [],
        "source_sha256": {
            str(p.relative_to(Path(__file__).parents[2])): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in module_paths
        },
        "array_units": {
            "x_m": "m",
            "cell_edges_m": "m",
            "t_s": "s",
            "h_m": "m",
            "q_m2_s": "m^2/s",
        },
    }
    failure = None
    with RunRecord.create(output_root, run_id, resolved, metadata=metadata) as run:
        started = time.perf_counter()
        result = solve(resolved["case"], cells, resolved["evaluation"]["times_s"])
        write_json_atomic(run.path / "diagnostics.json", result.diagnostics)
        if result.failure is not None:
            failure = result.failure
            if result.last_valid_state is not None:
                _save_npz(
                    run.path / "last_valid_state.npz",
                    x_m=result.x_m,
                    cell_edges_m=result.cell_edges_m,
                    t_s=np.array([result.last_valid_time_s]),
                    h_m=result.last_valid_state[None, :, 0],
                    q_m2_s=result.last_valid_state[None, :, 1],
                )
            if result.rejected_state is not None:
                _save_npz(
                    run.path / "failed_state.npz", state_h_q=result.rejected_state
                )
            if len(result.t_s):
                _save_npz(
                    run.path / "partial_outputs.npz",
                    x_m=result.x_m,
                    cell_edges_m=result.cell_edges_m,
                    t_s=result.t_s,
                    h_m=result.h_m,
                    q_m2_s=result.q_m2_s,
                )
            run.fail(
                failure["reason"],
                stage=failure["stage"],
                time_s=failure["time_s"],
                step=failure["step"],
                last_valid_state="last_valid_state.npz"
                if result.last_valid_state is not None
                else None,
                category="numerical_failure",
            )
        else:
            _save_npz(
                run.path / "outputs.npz",
                x_m=result.x_m,
                cell_edges_m=result.cell_edges_m,
                t_s=result.t_s,
                h_m=result.h_m,
                q_m2_s=result.q_m2_s,
            )
            run.complete(
                stopping_reason="all requested native snapshots and final time reached",
                elapsed_timings_s={
                    "numerical_integration_and_save": time.perf_counter() - started
                },
                iterations=result.diagnostics["accepted_steps"],
            )
    if failure is not None:
        raise NumericalRunFailed(run.path, failure)
    return run.path
