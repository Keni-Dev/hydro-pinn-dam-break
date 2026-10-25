"""Wet flat-bed HLLE/Manning reference, independent of the core comparator.

The Einfeldt wave estimates follow the pinned Clawpack Riemann-book equations.
This is an independent NumPy implementation; external solver code is not copied.
Only strictly positive, finite depths are supported. No depth floor is applied.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from hydro_pinn.config import load_config
from hydro_pinn.records import collect_metadata, write_json_atomic
from hydro_pinn.references import evaluate_reference


class FrictionReferenceError(ValueError):
    """Unsupported wet-reference input or invalid numerical stage."""


def _state(state: np.ndarray, stage: str) -> np.ndarray:
    state = np.asarray(state, dtype=np.float64)
    if state.ndim < 1 or state.shape[-1] != 2:
        raise FrictionReferenceError(f"{stage}: expected (...,2) depth/discharge")
    if not np.isfinite(state).all() or np.any(state[..., 0] <= 0):
        raise FrictionReferenceError(f"{stage}: nonfinite or nonpositive wet state")
    return state


def _physical(n: float, gravity: float) -> None:
    if not np.isfinite([n, gravity]).all() or n < 0 or gravity <= 0:
        raise FrictionReferenceError("n must be finite nonnegative; g finite positive")


def friction_source_step(
    state: np.ndarray, n: float, dt: float, gravity: float = 9.81
) -> np.ndarray:
    """Integrate q'=-g n² q|q|/h^(7/3) exactly while holding h fixed."""
    state = _state(state, "source input")
    _physical(n, gravity)
    if not np.isfinite(dt) or dt < 0:
        raise FrictionReferenceError("source dt must be finite nonnegative")
    result = state.copy()
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        result[..., 1] /= 1 + dt * gravity * n**2 * np.abs(state[..., 1]) / state[
            ..., 0
        ] ** (7 / 3)
    return _state(result, "source output")


def hlle_flux(
    left: np.ndarray, right: np.ndarray, gravity: float = 9.81
) -> tuple[np.ndarray, np.ndarray]:
    """Return conservative HLLE flux and interface maximum wave-speed modulus.

    uhat=(sqrt(hL)uL+sqrt(hR)uR)/(sqrt(hL)+sqrt(hR));
    chat=sqrt(g*(hL+hR)/2). sL=min(uL-cL,uhat-chat),
    sR=max(uR+cR,uhat+chat). Only the flux selection includes zero.
    """
    left, right = _state(left, "HLLE left"), _state(right, "HLLE right")
    _physical(0, gravity)
    h_l, q_l = left[..., 0], left[..., 1]
    h_r, q_r = right[..., 0], right[..., 1]
    u_l, u_r = q_l / h_l, q_r / h_r
    root_l, root_r = np.sqrt(h_l), np.sqrt(h_r)
    u_hat = (root_l * u_l + root_r * u_r) / (root_l + root_r)
    c_hat = np.sqrt(gravity * (h_l + h_r) / 2)
    s_l = np.minimum(u_l - np.sqrt(gravity * h_l), u_hat - c_hat)
    s_r = np.maximum(u_r + np.sqrt(gravity * h_r), u_hat + c_hat)
    f_l = np.stack((q_l, q_l * u_l + gravity * h_l**2 / 2), axis=-1)
    f_r = np.stack((q_r, q_r * u_r + gravity * h_r**2 / 2), axis=-1)
    gap = s_r - s_l
    middle = s_r[..., None] * right - s_l[..., None] * left - f_r + f_l
    middle /= gap[..., None]
    _state(middle, "HLLE intermediate")
    flux_middle = (
        s_r[..., None] * f_l
        - s_l[..., None] * f_r
        + (s_l * s_r)[..., None] * (right - left)
    ) / gap[..., None]
    flux = np.where(
        (s_l >= 0)[..., None],
        f_l,
        np.where((s_r <= 0)[..., None], f_r, flux_middle),
    )
    if not np.isfinite(flux).all():
        raise FrictionReferenceError("nonfinite HLLE flux")
    return flux, np.maximum(np.abs(s_l), np.abs(s_r))


def conservative_project(
    averages: np.ndarray, source_edges: np.ndarray, target_edges: np.ndarray
) -> np.ndarray:
    """Integrate the source piecewise-constant cell field over target cells.

    Last dimension is the source cell axis. The cumulative integral is linear
    inside each source cell, so evaluating it at target edges is conservative.
    """
    values = np.asarray(averages, dtype=np.float64)
    source, target = np.asarray(source_edges), np.asarray(target_edges)
    if (
        source.ndim != 1
        or target.ndim != 1
        or len(target) < 2
        or values.shape[-1] != len(source) - 1
        or np.any(np.diff(source) <= 0)
        or np.any(np.diff(target) <= 0)
        or target[0] < source[0]
        or target[-1] > source[-1]
        or not all(np.isfinite(a).all() for a in (values, source, target))
    ):
        raise FrictionReferenceError("invalid conservative projection axes/values")
    cumulative = np.concatenate(
        (
            np.zeros(values.shape[:-1] + (1,)),
            np.cumsum(values * np.diff(source), axis=-1),
        ),
        axis=-1,
    )
    flat = cumulative.reshape(-1, len(source))
    integrals = np.stack([np.interp(target, source, row) for row in flat])
    return (np.diff(integrals) / np.diff(target)).reshape(
        values.shape[:-1] + (len(target) - 1,)
    )


def solve_reference(
    case: dict[str, Any], n: float, cells: int, cfl: float, times: np.ndarray
) -> dict[str, np.ndarray]:
    """Solve strictly wet flat-bed E1 and retain every actual step budget.

    One ghost cell at each edge is reset to its prescribed reservoir state.
    Homogeneous forward Euler is followed by exact fixed-h source integration.
    The timestep uses all interface Einfeldt speeds, with CFL<=0.5, and is
    shortened to requested output times. There is no clipping or retry path.
    """
    gravity = float(case["gravity_m_s2"])
    _physical(n, gravity)
    times = np.asarray(times, dtype=np.float64)
    if (
        isinstance(cells, bool)
        or not isinstance(cells, (int, np.integer))
        or cells < 2
        or cells % 2
        or not np.isfinite(cfl)
        or not 0 < cfl <= 0.5
        or times.ndim != 1
        or times.size == 0
        or not np.isfinite(times).all()
        or times[0] < 0
        or np.any(np.diff(times) <= 0)
        or times[-1] > case["final_time_s"]
    ):
        raise FrictionReferenceError("invalid even cells, CFL<=.5 or output times")
    a, b = case["domain_m"]
    edges = np.linspace(a, b, cells + 1)
    x, dx = (edges[:-1] + edges[1:]) / 2, (b - a) / cells
    dam = case["dam_position_m"]
    if not np.any(np.isclose(edges, dam, rtol=0, atol=1e-13)):
        raise FrictionReferenceError("initial dam must align with a cell face")
    h = np.where(x < dam, case["left_depth_m"], case["right_depth_m"])
    u = np.where(x < dam, case["left_velocity_m_s"], case["right_velocity_m_s"])
    state = _state(np.stack((h, h * u), axis=-1), "initial")
    boundary = case["boundary"]
    left = np.array([[boundary["left_h_m"], boundary["left_q_m2_s"]]])
    right = np.array([[boundary["right_h_m"], boundary["right_q_m2_s"]]])
    _state(left, "left boundary")
    _state(right, "right boundary")
    initial_totals = dx * state.sum(axis=0)
    step_rows, left_fluxes, right_fluxes, snapshots = [], [], [], []
    t, boundary_integral, impulse = 0.0, np.zeros(2), 0.0
    min_h = float(state[:, 0].min())
    start = time.perf_counter()
    for target in times:
        while t < target:
            ghost = np.concatenate((left, state, right))
            flux, speeds = hlle_flux(ghost[:-1], ghost[1:], gravity)
            dt = min(cfl * dx / float(speeds.max()), float(target - t))
            if dt <= 0 or t + dt == t:
                raise FrictionReferenceError(f"nonadvancing dt at t={t}")
            homogeneous = _state(state - dt / dx * np.diff(flux, axis=0), "homogeneous")
            new_state = friction_source_step(homogeneous, n, dt, gravity)
            source_impulse = float(dx * (new_state[:, 1] - homogeneous[:, 1]).sum())
            boundary_integral += dt * (flux[-1] - flux[0])
            impulse += source_impulse
            state, t = new_state, min(t + dt, float(target))
            totals = dx * state.sum(axis=0)
            defects = totals - initial_totals + boundary_integral - [0.0, impulse]
            min_h = min(min_h, float(state[:, 0].min()), float(homogeneous[:, 0].min()))
            step_rows.append(
                [
                    t,
                    dt,
                    totals[0],
                    totals[1],
                    source_impulse,
                    impulse,
                    defects[0],
                    defects[1],
                    min_h,
                    float(speeds.max()) * dt / dx,
                ]
            )
            left_fluxes.append(flux[0].copy())
            right_fluxes.append(flux[-1].copy())
        snapshots.append(state.copy())
    fields, budgets = np.asarray(snapshots), np.asarray(step_rows).reshape(-1, 10)
    return {
        "x_m": x,
        "cell_edges_m": edges,
        "t_s": times,
        "h_m": fields[..., 0],
        "q_m2_s": fields[..., 1],
        "step_t_s": budgets[:, 0],
        "step_dt_s": budgets[:, 1],
        "step_mass_m2": budgets[:, 2],
        "step_momentum_m3_s": budgets[:, 3],
        "step_source_impulse_m3_s": budgets[:, 4],
        "cumulative_source_impulse_m3_s": budgets[:, 5],
        "step_mass_budget_defect_m2": budgets[:, 6],
        "step_momentum_budget_defect_m3_s": budgets[:, 7],
        "step_running_minimum_h_m": budgets[:, 8],
        "step_cfl": budgets[:, 9],
        "step_left_numerical_flux": np.asarray(left_fluxes).reshape(-1, 2),
        "step_right_numerical_flux": np.asarray(right_fluxes).reshape(-1, 2),
        "initial_mass_m2": np.asarray(initial_totals[0]),
        "initial_momentum_m3_s": np.asarray(initial_totals[1]),
        "solve_wall_s": np.asarray(time.perf_counter() - start),
    }


def numerical_front(solution: dict[str, np.ndarray], dam: float = 0.0) -> np.ndarray:
    """First maximum negative depth slope on downstream native interfaces."""
    edges = solution["cell_edges_m"][1:-1]
    eligible = edges >= dam
    slope = -np.diff(solution["h_m"], axis=-1) / np.diff(solution["x_m"])
    return edges[eligible][np.argmax(slope[:, eligible], axis=-1)]


def load_reference(path: str | Path) -> dict[str, np.ndarray]:
    """Load trusted local numerical arrays; a packet root loads common averages."""
    source = Path(path)
    if source.is_dir():
        source = source / (
            "common_reference.npz"
            if (source / "common_reference.npz").exists()
            else "outputs.npz"
        )
    with np.load(source, allow_pickle=False) as arrays:
        return {k: arrays[k].copy() for k in arrays.files}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stats(values: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean(values**2))),
        "max_abs": float(np.max(np.abs(values))),
    }


def run_reference(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Create one immutable procedure and full four-level refinement packet."""
    start = time.perf_counter()
    config_path, output = Path(config_path), Path(output_dir)
    config = json.loads(config_path.read_text())
    if output.exists():
        raise FileExistsError(f"reference output already exists: {output}")
    if config["bed_elevation_m"] != 0 or config["rainfall_m_s"] != 0:
        raise FrictionReferenceError("selected reference supports flat bed/no rain")
    case_path = Path(config["case"])
    case = load_config(case_path)
    n = float(config["manning_n_s_m_minus_one_third"])
    levels, cfl = config["spatial_cells"], config["cfl"]
    if (
        levels != [800, 1600, 3200]
        or cfl != 0.45
        or config["time_check_cfl"] != 0.225
        or n != 0.03
    ):
        raise FrictionReferenceError("reference matrix differs from selected v01")
    time_spec = config["output_times_s"]
    times = np.linspace(time_spec["first"], time_spec["last"], time_spec["count"])
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        "config": config,
        "case": case,
        "config_sha256": _sha(config_path),
        "case_sha256": _sha(case_path),
        "source_sha256": _sha(Path(__file__)),
        "state_location": "cell_average",
        "freeze": "Before reference matrix and PINN comparison",
    }
    write_json_atomic(output / "protocol.json", protocol)
    write_json_atomic(output / "resolved_config.json", protocol)
    write_json_atomic(output / "status.json", {"state": "running"})
    metadata = collect_metadata(
        {"case": case, "request": {"device": "cpu", "dtype": "float64"}}
    )
    write_json_atomic(output / "metadata.json", metadata)
    all_runs, records = {}, []
    try:
        for cells, dt_cfl in [(c, cfl) for c in levels] + [
            (levels[-1], config["time_check_cfl"])
        ]:
            name = f"n{cells}_cfl{str(dt_cfl).replace('.', 'p')}"
            run_dir = output / name
            run_dir.mkdir()
            arrays = solve_reference(case, n, cells, dt_cfl, times)
            np.savez_compressed(run_dir / "outputs.npz", **arrays)
            record = {
                "id": name,
                "cells": cells,
                "dx_m": 10 / cells,
                "cfl": dt_cfl,
                "steps": len(arrays["step_t_s"]),
                "solve_wall_s": float(arrays["solve_wall_s"]),
                "minimum_h_m": float(arrays["h_m"].min()),
                "dt_min_s": float(arrays["step_dt_s"].min()),
                "dt_max_s": float(arrays["step_dt_s"].max()),
                "max_mass_budget_defect_m2": float(
                    np.max(np.abs(arrays["step_mass_budget_defect_m2"]))
                ),
                "max_momentum_budget_defect_m3_s": float(
                    np.max(np.abs(arrays["step_momentum_budget_defect_m3_s"]))
                ),
                "final_cumulative_source_impulse_m3_s": float(
                    arrays["cumulative_source_impulse_m3_s"][-1]
                ),
                "boundary_h_max_mismatch_m": float(
                    max(
                        np.max(np.abs(arrays["h_m"][:, 0] - case["left_depth_m"])),
                        np.max(np.abs(arrays["h_m"][:, -1] - case["right_depth_m"])),
                    )
                ),
                "boundary_q_max_mismatch_m2_s": float(
                    np.max(np.abs(arrays["q_m2_s"][:, [0, -1]]))
                ),
                "outputs_sha256": _sha(run_dir / "outputs.npz"),
            }
            write_json_atomic(run_dir / "diagnostics.json", record)
            all_runs[name], records = arrays, records + [record]
        selected = all_runs[config["selected_reference"]]
        edges = np.linspace(*case["domain_m"], config["common_cells"] + 1)

        def project(s, k):
            return conservative_project(s[k], s["cell_edges_m"], edges)

        h_selected, q_selected = project(selected, "h_m"), project(selected, "q_m2_s")
        coarse, medium, fine = [all_runs[f"n{c}_cfl0p45"] for c in levels]
        fronts = {
            name: numerical_front(arr, case["dam_position_m"])
            for name, arr in all_runs.items()
        }
        selected_front = fronts[config["selected_reference"]]
        front_difference = np.max(
            np.abs(np.stack(list(fronts.values())) - selected_front), axis=0
        )
        comparisons = []
        for name, earlier, later in [
            ("spatial800_1600", coarse, medium),
            ("spatial1600_3200", medium, fine),
            ("time3200", fine, selected),
        ]:
            comparisons.append(
                {
                    "pair": name,
                    "h_m": _stats(project(later, "h_m") - project(earlier, "h_m")),
                    "q_m2_s": _stats(
                        project(later, "q_m2_s") - project(earlier, "q_m2_s")
                    ),
                    "front_max_difference_m": float(
                        np.max(
                            np.abs(numerical_front(later) - numerical_front(earlier))
                        )
                    ),
                }
            )
        common = {
            "x_m": (edges[1:] + edges[:-1]) / 2,
            "cell_edges_m": edges,
            "t_s": times,
            "h_m": h_selected,
            "q_m2_s": q_selected,
            "h_measured_uncertainty_m": np.abs(
                project(fine, "h_m") - project(medium, "h_m")
            )
            + np.abs(h_selected - project(fine, "h_m")),
            "q_measured_uncertainty_m2_s": np.abs(
                project(fine, "q_m2_s") - project(medium, "q_m2_s")
            )
            + np.abs(q_selected - project(fine, "q_m2_s")),
            "front_x_m": selected_front,
            "front_uncertainty_m": front_difference + 10 / levels[-1],
            "front_half_width_m": np.maximum(0.1, front_difference + 10 / levels[-1]),
        }
        np.savez_compressed(output / "common_reference.npz", **common)
        summary = {
            "state": "complete_reference_with_measured_uncertainty",
            "runs": records,
            "comparison_operator": (
                "Conservative overlap integration of native piecewise-constant "
                "averages onto 400 equal cells; PINN integrates over same cells"
            ),
            "selected_reference": config["selected_reference"],
            "refinement": comparisons,
            "front_max_uncertainty_m": float(common["front_uncertainty_m"].max()),
            "front_max_half_width_m": float(common["front_half_width_m"].max()),
            "uncertainty_h_m": _stats(common["h_measured_uncertainty_m"]),
            "uncertainty_q_m2_s": _stats(common["q_measured_uncertainty_m2_s"]),
            "reference_is_ground_truth": False,
            "common_sha256": _sha(output / "common_reference.npz"),
            "wall_s_including_output": time.perf_counter() - start,
            "python": sys.version,
            "platform": platform.platform(),
        }
        write_json_atomic(output / "summary.json", summary)
        write_json_atomic(
            output / "status.json", {"state": "completed"}, overwrite=True
        )
        return summary
    except Exception as exc:
        write_json_atomic(
            output / "status.json",
            {"state": "failed", "reason": repr(exc), "completed_runs": records},
            overwrite=True,
        )
        raise


def zero_n_reference_check(
    case: dict[str, Any], levels: tuple[int, ...] = (200, 400, 800)
) -> dict[str, Any]:
    """Refined zero-n HLLE approaches exact E1 cell averages at t=.5.

    Sixteen-node Gauss integration per cell approximates exact cell averages;
    this check is separate from the frictional comparison and neural evidence.
    """
    nodes, weights = np.polynomial.legendre.leggauss(16)
    errors = []
    for cells in levels:
        arr = solve_reference(case, 0.0, cells, 0.45, np.array([0.5]))
        x, dx = arr["x_m"], 10 / cells
        quadrature = (x[:, None] + nodes[None, :] * dx / 2).ravel()
        exact = evaluate_reference(case, quadrature, [0.5])
        h = (exact.h_m.reshape(cells, 16) * weights / 2).sum(axis=1)
        q = (exact.q_m2_s.reshape(cells, 16) * weights / 2).sum(axis=1)
        errors.append(
            {
                "cells": cells,
                "h_l1_m": float(np.mean(np.abs(arr["h_m"][0] - h))),
                "q_l1_m2_s": float(np.mean(np.abs(arr["q_m2_s"][0] - q))),
                "solve_wall_s": float(arr["solve_wall_s"]),
            }
        )
    return {
        "quadrature": "16-node Gauss per native cell, exact E1 pointwise integrand",
        "errors": errors,
        "h_decreases": all(
            a["h_l1_m"] > b["h_l1_m"] for a, b in zip(errors, errors[1:], strict=False)
        ),
        "q_decreases": all(
            a["q_l1_m2_s"] > b["q_l1_m2_s"]
            for a, b in zip(errors, errors[1:], strict=False)
        ),
    }
