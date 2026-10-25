"""Independent dense-quadrature and native-flux conservation evidence.

Auxiliary evaluations reload the already selected checkpoint. They neither train
nor write into the source run, and preserve the rectangular physical-array schema.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.integrate import cumulative_trapezoid

from hydro_pinn.model import ShallowWaterPINN
from hydro_pinn.records import RunRecord, validate_run_id, write_json_atomic
from hydro_pinn.references import evaluate_reference
from hydro_pinn.train import load_training_checkpoint


class ConservationError(ValueError):
    """Missing axes or incompatible provenance prevent conservation evaluation."""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prescribed_initial_volume(case: dict) -> float:
    a, b = case["domain_m"]
    dam = case["dam_position_m"]
    return (dam - a) * case["left_depth_m"] + (b - dam) * case["right_depth_m"]


def _axis(values: Any, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if (
        result.ndim != 1
        or result.size == 0
        or not np.isfinite(result).all()
        or np.any(np.diff(result) <= 0)
    ):
        raise ConservationError(f"{name} must be a finite increasing axis")
    return result


def _fields(arrays: dict) -> tuple[np.ndarray, ...]:
    x, t = _axis(arrays["x_m"], "x_m"), _axis(arrays["t_s"], "t_s")
    h, q = np.asarray(arrays["h_m"]), np.asarray(arrays["q_m2_s"])
    if h.shape != (len(t), len(x)) or q.shape != h.shape:
        raise ConservationError("physical fields must have shape (times,positions)")
    if not np.isfinite(h).all() or not np.isfinite(q).all() or np.any(h < 0):
        raise ConservationError("auxiliary fields contain nonfinite/negative depths")
    return x, t, h, q


def quadrature_mass(case: dict, volume: dict, boundary: dict) -> dict:
    """Compute signed balance with independent midpoint/trapezoid samples.

    Both axes include zero; boundary samples must include every volume time
    exactly. No interpolation or estimate from the comparison grid is used.
    """
    x, times, depth, _ = _fields(volume)
    bx, bt, _, discharge = _fields(boundary)
    a, b = case["domain_m"]
    expected_x = a + (np.arange(len(x)) + 0.5) * (b - a) / len(x)
    if not np.array_equal(x, expected_x) or not np.array_equal(bx, [a, b]):
        raise ConservationError("mass requires midpoint cells and exact endpoints")
    if times[0] != 0 or bt[0] != 0 or bt[-1] != case["final_time_s"]:
        raise ConservationError("mass quadrature must include t0 and boundary T")
    indices = np.searchsorted(bt, times)
    if np.any(indices >= len(bt)) or not np.array_equal(bt[indices], times):
        raise ConservationError("every volume time must be a boundary time exactly")
    prescribed = prescribed_initial_volume(case)
    if prescribed <= 0:
        raise ConservationError("prescribed initial volume must be positive")
    mass = np.sum(depth, axis=1, dtype=np.float64) * (b - a) / len(x)
    left, right = discharge[:, 0], discharge[:, 1]
    left_mismatch = left - case["boundary"]["left_q_m2_s"]
    right_mismatch = right - case["boundary"]["right_q_m2_s"]
    net_integral = cumulative_trapezoid(right - left, bt, initial=0.0)
    absolute_mismatch = cumulative_trapezoid(
        np.abs(left_mismatch) + np.abs(right_mismatch), bt, initial=0.0
    )
    balance = mass - prescribed + net_integral[indices]
    return {
        "rule": "midpoint volume; trapezoid actual physical endpoint discharge",
        "volume_cells": len(x),
        "boundary_nodes": len(bt),
        "t_s": times.tolist(),
        "volume_m2": mass.tolist(),
        "prescribed_initial_volume_m2": prescribed,
        "learned_initial_volume_error_m2": float(mass[0] - prescribed),
        "normalized_initial_volume_error": float((mass[0] - prescribed) / prescribed),
        "boundary_net_outflow_m2": net_integral[indices].tolist(),
        "signed_balance_m2": balance.tolist(),
        "normalized_signed_balance": (balance / prescribed).tolist(),
        "maximum_absolute_normalized_imbalance": float(
            np.max(np.abs(balance / prescribed))
        ),
        "boundary_t_s": bt.tolist(),
        "left_boundary_q_m2_s": left.tolist(),
        "right_boundary_q_m2_s": right.tolist(),
        "left_boundary_mismatch_m2_s": left_mismatch.tolist(),
        "right_boundary_mismatch_m2_s": right_mismatch.tolist(),
        "integrated_absolute_boundary_mismatch_m2": absolute_mismatch[indices].tolist(),
        "maximum_absolute_boundary_mismatch_m2_s": float(
            max(np.max(np.abs(left_mismatch)), np.max(np.abs(right_mismatch)))
        ),
    }


def quadrature_sensitivity(base: dict, doubled: dict) -> dict:
    if base["t_s"] != doubled["t_s"]:
        raise ConservationError("quadrature sensitivity requires identical times")
    differences = {}
    for key in ("volume_m2", "boundary_net_outflow_m2", "signed_balance_m2"):
        delta = np.asarray(doubled[key]) - np.asarray(base[key])
        differences[key] = {
            "doubled_minus_base": delta.tolist(),
            "maximum_absolute_difference": float(np.max(np.abs(delta))),
        }
    return {
        "base": [base["volume_cells"], base["boundary_nodes"]],
        "doubled": [doubled["volume_cells"], doubled["boundary_nodes"]],
        "t_s": base["t_s"],
        "differences": differences,
        "interpretation": "quadrature sensitivity, not an error bound",
    }


def native_solver_mass(
    case: dict, diagnostics: dict, native: dict | None = None
) -> dict:
    """Recompute balance from saved native cell sums and actual interface fluxes."""
    samples, steps = diagnostics["cell_sum_mass"], diagnostics["steps"]
    if len(samples) != len(steps) + 1 or not samples or samples[0]["time_s"] != 0:
        raise ConservationError("native mass/step history is incomplete")
    times = _axis([row["time_s"] for row in samples], "native mass times")
    volume = np.array([row["mass_m2"] for row in samples], dtype=np.float64)
    integrated, left, right, boundary_times = [0.0], [], [], []
    for index, step in enumerate(steps):
        if (
            step["time_start_s"] != times[index]
            or step["time_end_s"] != times[index + 1]
        ):
            raise ConservationError("native step times do not match mass history")
        dt = step["dt_s"]
        if not np.isclose(dt, times[index + 1] - times[index], rtol=1e-12, atol=1e-15):
            raise ConservationError("native step dt is inconsistent")
        lq, rq = step["left_boundary_flux_m2_s"], step["right_boundary_flux_m2_s"]
        integrated.append(integrated[-1] + dt * (rq - lq))
        left.append(lq)
        right.append(rq)
        boundary_times.append(step["time_end_s"])
    prescribed = prescribed_initial_volume(case)
    balance = volume - prescribed + np.asarray(integrated)
    if not np.isfinite(volume).all() or not np.isfinite(balance).all():
        raise ConservationError("nonfinite native mass/flux history")
    if native is not None:
        edges = _axis(native["cell_edges_m"], "cell edges")
        weights = np.diff(edges)
        native_volume = np.asarray(native["h_m"], dtype=np.float64) @ weights
        indices = np.searchsorted(times, native["t_s"])
        if np.any(indices >= len(times)) or not np.array_equal(
            times[indices], native["t_s"]
        ):
            raise ConservationError("native output times are absent from mass history")
        if not np.allclose(native_volume, volume[indices], rtol=1e-13, atol=1e-13):
            raise ConservationError("native cell sums disagree with saved mass history")
    return {
        "rule": "native cell sums; actual corrector interface flux times accepted dt",
        "t_s": times.tolist(),
        "volume_m2": volume.tolist(),
        "prescribed_initial_volume_m2": prescribed,
        "learned_initial_volume_error_m2": float(volume[0] - prescribed),
        "normalized_initial_volume_error": float((volume[0] - prescribed) / prescribed),
        "boundary_net_outflow_m2": integrated,
        "signed_balance_m2": balance.tolist(),
        "normalized_signed_balance": (balance / prescribed).tolist(),
        "maximum_absolute_normalized_imbalance": float(
            np.max(np.abs(balance / prescribed))
        ),
        "boundary_t_s": [0.0, *boundary_times] if steps else [],
        "left_boundary_q_m2_s": [left[0], *left] if steps else [],
        "right_boundary_q_m2_s": [right[0], *right] if steps else [],
        "boundary_values_are": (
            "numerical mass interface flux, constant over each accepted step"
        ),
        "boundary_plot_convention": (
            "steps-pre on t0 plus step ends; first known flux repeated at t0"
        ),
        "failure": diagnostics.get("failure"),
    }


def _read_record(path: Path, expected: dict, axes: dict) -> tuple[dict, dict]:
    for name in (
        "resolved_config.json",
        "metadata.json",
        "status.json",
        "outputs.npz",
        "outputs_manifest.json",
    ):
        if not (path / name).is_file() or (path / name).is_symlink():
            raise ConservationError(
                f"missing or unsafe auxiliary artifact: {path / name}"
            )
    resolved = json.loads((path / "resolved_config.json").read_text())
    metadata = json.loads((path / "metadata.json").read_text())
    status = json.loads((path / "status.json").read_text())
    if (
        status["state"] != "completed"
        or status["run_id"] != path.name
        or metadata["run_id"] != path.name
        or resolved["request"]["run_id"] != path.name
        or resolved["conservation_source"] != expected
    ):
        raise ConservationError("auxiliary source identity/state mismatch")
    if metadata["resolved_config_sha256"] != file_sha256(path / "resolved_config.json"):
        raise ConservationError("auxiliary resolved configuration hash mismatch")
    if resolved["auxiliary_axes"] != axes:
        raise ConservationError("auxiliary axes/configuration mismatch")
    output_manifest = json.loads((path / "outputs_manifest.json").read_text())
    if (
        output_manifest["outputs_sha256"] != file_sha256(path / "outputs.npz")
        or output_manifest["conservation_source"] != expected
        or output_manifest["resolved_config_sha256"]
        != metadata["resolved_config_sha256"]
    ):
        raise ConservationError("auxiliary output hash mismatch")
    with np.load(path / "outputs.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    _fields(arrays)
    if not np.array_equal(arrays["x_m"], axes["x_m"]) or not np.array_equal(
        arrays["t_s"], axes["t_s"]
    ):
        raise ConservationError("saved auxiliary arrays differ from frozen axes")
    artifact = {
        "run_id": path.name,
        "run_path": str(path),
        "resolved_config_sha256": metadata["resolved_config_sha256"],
        "outputs_sha256": file_sha256(path / "outputs.npz"),
        "files_sha256": {
            name: file_sha256(path / name)
            for name in (
                "resolved_config.json",
                "metadata.json",
                "status.json",
                "outputs.npz",
                "outputs_manifest.json",
            )
        },
        **expected,
    }
    return arrays, artifact


def auxiliary_record(
    protocol: dict,
    plan: dict,
    source_run_id: str,
    *,
    project_root: str | Path,
    execution: dict,
) -> tuple[dict, dict]:
    """Create or verify one frozen, separately recorded forward evaluation."""
    root = Path(project_root).absolute()
    output_root = root / execution["output_root"]
    validate_run_id(source_run_id)
    source_path = output_root / source_run_id
    if source_path.is_symlink():
        raise ConservationError("source run must not be a symlink")
    source = json.loads((source_path / "resolved_config.json").read_text())
    source_status = json.loads((source_path / "status.json").read_text())
    if source_status["state"] != "completed":
        raise ConservationError("auxiliary evaluation requires completed source")
    axes_key = "volume_axes" if plan["kind"] == "volume" else "boundary_axes"
    axes = protocol["conservation"][axes_key][plan["axis_key"]]
    expected = {
        "protocol_sha256": protocol["protocol_sha256"],
        "evaluation_sha256": plan["evaluation_sha256"],
        "source_stage_id": plan["source_stage_id"],
        "source_run_id": source_run_id,
        "source_resolved_config_sha256": file_sha256(
            source_path / "resolved_config.json"
        ),
        "kind": plan["kind"],
        "axis_key": plan["axis_key"],
        "checkpoint_sha256": None,
    }
    checkpoint = training = None
    if plan["source_kind"] == "predict":
        prediction_source = source["prediction_source"]
        checkpoint, training, checkpoint_path = load_training_checkpoint(
            prediction_source["training_run_id"], output_root, require_completed=True
        )
        expected["checkpoint_sha256"] = file_sha256(checkpoint_path / "checkpoint.pt")
        if expected["checkpoint_sha256"] != prediction_source["checkpoint_sha256"]:
            raise ConservationError(
                "auxiliary checkpoint differs from saved prediction"
            )
    run_id = execution["run_ids"][plan["run_id"]]
    path = output_root / validate_run_id(run_id)
    if path.exists():
        return _read_record(path, expected, axes)
    resolved = {
        "case": copy.deepcopy(source["case"]),
        "training": copy.deepcopy(source.get("training")),
        "evaluation": copy.deepcopy(protocol["resolved"]["evaluation"]),
        "request": {**source["request"], "run_id": run_id},
        "conservation_source": expected,
        "auxiliary_axes": copy.deepcopy(axes),
        "provenance": {"purpose": "frozen_conservation_forward_only"},
    }
    metadata = {
        "stage": "conservation_auxiliary",
        "state_location": "point_value",
        "no_retraining_or_checkpoint_selection": True,
        "forward_chunk_points": 20000 if checkpoint is not None else None,
        **expected,
        "source_sha256": {
            "src/hydro_pinn/conservation.py": file_sha256(Path(__file__))
        },
    }
    with RunRecord.create(
        output_root, run_id, resolved, metadata=metadata, project_root=root
    ) as run:
        started = time.perf_counter()
        x, times = np.array(axes["x_m"]), np.array(axes["t_s"])
        if checkpoint is None:
            solution = evaluate_reference(source["case"], x, times)
            depth, discharge = solution.h_m, solution.q_m2_s
        else:
            model = ShallowWaterPINN(
                training["case"],
                training["training"],
                device=training["request"]["device"],
                dtype=training["request"]["dtype"],
            )
            model.load_state_dict(checkpoint["selected_model_state"], strict=True)
            model.eval()
            xx, tt = np.meshgrid(x, times)
            coordinates = np.column_stack((xx.ravel(), tt.ravel()))
            parameter = next(model.parameters())
            fields = []
            with torch.no_grad():
                for start in range(0, len(coordinates), 20000):
                    query = torch.tensor(
                        coordinates[start : start + 20000],
                        dtype=parameter.dtype,
                        device=parameter.device,
                    )
                    fields.append(model.physical_state(query).cpu().numpy())
            state = np.concatenate(fields).reshape(len(times), len(x), 2)
            depth, discharge = state[..., 0], state[..., 1]
        arrays = {"x_m": x, "t_s": times, "h_m": depth, "q_m2_s": discharge}
        _fields(arrays)
        with (run.path / "outputs.npz").open("xb") as handle:
            np.savez(handle, **arrays)
        write_json_atomic(
            run.path / "outputs_manifest.json",
            {
                "outputs_sha256": file_sha256(run.path / "outputs.npz"),
                "resolved_config_sha256": file_sha256(
                    run.path / "resolved_config.json"
                ),
                "conservation_source": expected,
            },
        )
        run.complete(
            stopping_reason=(
                "frozen auxiliary forward values saved; checkpoint unchanged"
            ),
            elapsed_timings_s={"forward_and_save": time.perf_counter() - started},
        )
    return _read_record(path, expected, axes)
