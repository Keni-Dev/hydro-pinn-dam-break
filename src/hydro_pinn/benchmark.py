"""Repeated, workload-matched timing of frozen core outputs.

Production timing requires a terminal verified matrix and an explicit scheduling
signal. These measurements do not establish accuracy, generalization, or field
validity. Training is read from its original record and is never repeated here.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hydro_pinn.experiments import _verify_record, execution_plan
from hydro_pinn.lax_wendroff import piecewise_constant_reconstruct, solve
from hydro_pinn.model import ShallowWaterPINN
from hydro_pinn.protocol import ROOT, file_sha256, load_protocol
from hydro_pinn.records import collect_metadata, undefined_metric, write_json_atomic
from hydro_pinn.references import evaluate_reference
from hydro_pinn.train import load_training_checkpoint


class BenchmarkError(ValueError):
    """Timing inputs or the measured output do not match the declared workload."""


def synchronize(device: str) -> None:
    """Finish device work before reading a wall clock; CPU calls are synchronous."""
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    elif device != "cpu":
        raise BenchmarkError(f"unsupported synchronization device {device!r}")


def _host_observation() -> dict:
    try:
        loads = list(os.getloadavg())
    except (AttributeError, OSError):
        loads = None
    return {
        "at_utc": datetime.now(UTC).isoformat(),
        "load_average_1_5_15_minutes": loads,
        "logical_cpu_count": os.cpu_count(),
        "process_cpu_time_s": time.process_time(),
    }


def timing_summary(samples_s: list[float]) -> dict:
    """Retain raw samples and report median/IQR with the NumPy linear quantile."""
    values = np.asarray(samples_s, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0:
        raise BenchmarkError("timing summary requires at least one sample")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise BenchmarkError("timing samples must be finite and nonnegative")
    q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
    return {
        "raw_samples_s": values.tolist(),
        "measured_repetitions": len(values),
        "median_s": float(median),
        "q1_s": float(q1),
        "q3_s": float(q3),
        "iqr_s": float(q3 - q1),
        "quantile_rule": "numpy.quantile method=linear",
    }


def measure_workload(
    workload: Callable[[], Any],
    validate: Callable[[Any], None],
    *,
    device: str = "cpu",
    warmups: int = 10,
    repetitions: int = 30,
    clock: Callable[[], float] = time.perf_counter,
    sync: Callable[[str], None] = synchronize,
) -> dict:
    """Measure complete work, synchronizing on both sides of every interval.

    Output validation runs after the timed interval. A failed repetition retains
    its prior samples but has no summary; it is never summarized as a full run.
    Smaller counts are available for bounded software fixtures only. Production
    entry points enforce the frozen ten warmups and thirty measurements.
    """
    if (
        isinstance(warmups, bool)
        or not isinstance(warmups, int)
        or warmups < 0
        or isinstance(repetitions, bool)
        or not isinstance(repetitions, int)
        or repetitions < 1
    ):
        raise BenchmarkError(
            "warmups/repetitions must be nonnegative/positive integers"
        )
    record: dict[str, Any] = {
        "state": "running",
        "device": device,
        "warmup_repetitions_requested": warmups,
        "warmup_repetitions_completed": 0,
        "measured_repetitions_requested": repetitions,
        "raw_samples_s": [],
        "synchronization": "device synchronization before and after each interval",
        "validation_scope": "every output after the timed interval",
        "host_before": _host_observation(),
    }
    try:
        for _ in range(warmups):
            sync(device)
            result = workload()
            sync(device)
            validate(result)
            del result
            record["warmup_repetitions_completed"] += 1
        for _ in range(repetitions):
            sync(device)
            started = clock()
            result = workload()
            sync(device)
            elapsed = clock() - started
            if not math.isfinite(elapsed) or elapsed < 0:
                raise BenchmarkError("invalid elapsed wall time")
            record["raw_samples_s"].append(elapsed)
            validate(result)
            del result
        record.update(timing_summary(record["raw_samples_s"]), state="completed")
    except (Exception, KeyboardInterrupt) as exc:
        record.update(
            state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            reason=f"{type(exc).__name__}: {exc}",
            measured_repetitions=len(record["raw_samples_s"]),
            median_s=None,
            iqr_s=None,
        )
    record["host_after"] = _host_observation()
    return record


def speed_ratio(numerator: dict, denominator: dict) -> dict:
    """Return a qualified time ratio, preserving invalid/incomplete cases."""
    if numerator.get("state") != "completed" or denominator.get("state") != "completed":
        return undefined_metric(
            "ratio requires two complete validated timing workloads"
        )
    first, second = numerator.get("median_s"), denominator.get("median_s")
    if (
        first is None
        or second is None
        or not math.isfinite(first)
        or not math.isfinite(second)
        or first <= 0
        or second <= 0
    ):
        return undefined_metric("ratio requires finite, strictly positive medians")
    return {"value": first / second, "reason": None}


def common_coordinates(evaluation: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct the declared physical cell-centre grid and time-major pairs."""
    left, right = evaluation["grid"]["domain_m"]
    cells = evaluation["grid"]["cells"]
    x = left + (np.arange(cells, dtype=np.float64) + 0.5) * (right - left) / cells
    times = np.asarray(evaluation["times_s"], dtype=np.float64)
    xx, tt = np.meshgrid(x, times)
    return x, times, np.column_stack((xx.ravel(), tt.ravel()))


def solver_workload(case: dict, cells: int, x: np.ndarray, times: np.ndarray):
    """Return a closure that integrates afresh and reconstructs both fields."""

    def run():
        result = solve(case, cells, times)
        if result.failure is not None:
            raise BenchmarkError(
                f"fresh numerical integration failed: {result.failure}"
            )
        if not np.array_equal(result.t_s, times):
            raise BenchmarkError("solver did not hit every exact requested output time")
        h = piecewise_constant_reconstruct(result.cell_edges_m, result.h_m, x)
        q = piecewise_constant_reconstruct(result.cell_edges_m, result.q_m2_s, x)
        return np.stack((h, q), axis=-1)

    return run


def network_workload(
    model: ShallowWaterPINN, coordinates: torch.Tensor, chunk_size=None
):
    """Return complete physical h/q forward work with resident model and inputs."""
    if chunk_size is not None and (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size < 1
    ):
        raise BenchmarkError("chunk_size must be a positive integer or null")

    def run():
        with torch.no_grad():
            if chunk_size is None:
                return model.physical_state(coordinates)
            return torch.cat(
                [model.physical_state(batch) for batch in coordinates.split(chunk_size)]
            )

    return run


def _validate_output(result: Any, shape: tuple[int, ...]) -> None:
    if tuple(result.shape) != shape:
        raise BenchmarkError(
            f"workload produced {tuple(result.shape)}, expected {shape}"
        )
    if isinstance(result, torch.Tensor):
        valid = bool(torch.isfinite(result).all()) and not bool(
            (result[..., 0] < 0).any()
        )
    else:
        valid = np.isfinite(result).all() and not np.any(result[..., 0] < 0)
    if not valid:
        raise BenchmarkError("workload produced nonfinite fields or negative depth")


def _path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else root / path
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise BenchmarkError(
            "benchmark paths must not use symlinks or parent traversal"
        )
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise BenchmarkError(
            "benchmark paths must remain within the project root"
        ) from exc
    return path


def _read(path: Path) -> dict:
    def reject(value):
        raise BenchmarkError(f"nonfinite JSON constant {value}")

    result = json.loads(path.read_text(), parse_constant=reject)
    if not isinstance(result, dict):
        raise BenchmarkError(f"expected JSON object: {path}")
    return result


def _verified_manifest(protocol: dict, path: Path, root: Path) -> tuple[dict, dict]:
    manifest = _read(path)
    if (
        manifest.get("protocol_sha256") != protocol["protocol_sha256"]
        or manifest.get("protocol_id") != protocol["protocol_id"]
        or manifest.get("purpose") != protocol["purpose"]
    ):
        raise BenchmarkError("manifest does not belong to the frozen protocol")
    if manifest.get("workflow_status") not in {
        "completed",
        "completed_with_diagnostics",
    }:
        raise BenchmarkError("production benchmark requires a complete terminal matrix")
    entries = manifest.get("entries", [])
    if len(entries) != len(protocol["matrix"]):
        raise BenchmarkError("manifest does not cover the complete frozen matrix")
    by_id = {entry["stage_id"]: entry for entry in entries}
    if len(by_id) != len(entries):
        raise BenchmarkError("duplicate manifest stage identity")
    execution = manifest.get("execution") or execution_plan(protocol)
    verified = {}
    for stage in protocol["matrix"]:
        entry = by_id.get(stage["stage_id"])
        if entry is None or entry.get("frozen_stage") != stage:
            raise BenchmarkError("manifest stage differs from the frozen matrix")
        state = entry["state"]
        if state != "completed" and not (
            state == "failed"
            and stage["stage"] == "solve"
            and entry.get("failure_category") == "numerical_failure"
        ):
            raise BenchmarkError(
                "manifest contains a nonterminal or invalid required stage"
            )
        training_source = None
        if stage["stage"] == "predict":
            training_source = by_id[stage["training_stage_id"]]["actual_run_id"]
        info = _verify_record(
            protocol,
            stage,
            entry["actual_run_id"],
            root,
            training_source_id=training_source,
            require_completed=state == "completed",
            execution=execution,
        )
        if info["record_state"] != state:
            raise BenchmarkError(
                "manifest state differs from independently read run state"
            )
        recorded = entry.get("result", {})
        if recorded.get("files_sha256") != info["files_sha256"]:
            raise BenchmarkError("run files changed after matrix verification")
        verified[stage["stage_id"]] = info
    return manifest, verified


def _one_time(work: Callable, device: str) -> tuple[Any, float]:
    synchronize(device)
    started = time.perf_counter()
    value = work()
    synchronize(device)
    return value, time.perf_counter() - started


def _native_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path / "outputs.npz", allow_pickle=False) as arrays:
        return tuple(arrays[name].copy() for name in ("cell_edges_m", "h_m", "q_m2_s"))


def benchmark_protocol(
    protocol_path: str | Path,
    output_path: str | Path,
    *,
    project_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
    operationally_ready: bool = False,
) -> dict:
    """Time frozen runs after the orchestrator releases an uncontended window.

    ``operationally_ready`` is a scheduling assertion, separate from scientific
    protocol validity. It must stay false while long training competes for CPU.
    A distinct manifest may bind a namespaced, otherwise identical replay.
    """
    if operationally_ready is not True:
        raise BenchmarkError(
            "production timing awaits orchestrator readiness; training must be idle"
        )
    root = Path(project_root).absolute() if project_root else ROOT
    destination = _path(root, output_path)
    if destination.exists():
        raise FileExistsError(f"timing output already exists: {destination}")
    protocol_source = _path(root, protocol_path)
    protocol = load_protocol(protocol_source, project_root=root)
    evaluation, settings = protocol["resolved"]["evaluation"], protocol["settings"]
    warmups = protocol["benchmark"]["warmup_repetitions"]
    repetitions = protocol["benchmark"]["measured_repetitions"]
    if warmups != 10 or repetitions != 30:
        raise BenchmarkError(
            "production benchmark requires ten warmups and thirty repetitions"
        )
    x, times, coordinate_array = common_coordinates(evaluation)
    if (len(times), len(x)) != (50, 400):
        raise BenchmarkError(
            "production benchmark requires the frozen 400-by-50 workload"
        )
    manifest_source = _path(root, manifest_path or protocol["paths"]["manifest"])
    manifest, verified = _verified_manifest(protocol, manifest_source, root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The matrix's execution output root also supports an isolated paper replay.
    output_root = _path(
        root, manifest.get("execution", protocol["paths"])["output_root"]
    )
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(settings["cpu_threads"])
    report: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "frozen_core_equivalent_output_timing",
        "source_protocol_purpose": protocol["purpose"],
        "state": "running",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "provenance": {
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": protocol["protocol_sha256"],
            "protocol_file_sha256": file_sha256(protocol_source),
            "manifest_path": str(manifest_source.relative_to(root)),
            "manifest_sha256": file_sha256(manifest_source),
            "source_sha256": {"src/hydro_pinn/benchmark.py": file_sha256(__file__)},
            "frozen_scientific_source_sha256": protocol["scientific_source_sha256"],
        },
        "workload": {
            "shape_h_q": [50, 400, 2],
            "fields": ["h_m", "q_m2_s"],
            "x_m": x.tolist(),
            "t_s": times.tolist(),
            "warmups": warmups,
            "repetitions": repetitions,
            "reference_and_error_calculation_in_main_timing": False,
            "plotting_in_main_timing": False,
            "numerical": (
                "fresh prescribed-state integration and right-face "
                "piecewise-constant reconstruction"
            ),
            "network": (
                "resident checkpoint/coordinates; complete physical h/q forward; "
                "outputs resident"
            ),
            "compile": "eager PyTorch; no torch.compile; warmups excluded from samples",
        },
        "environment": collect_metadata(
            {"request": {"device": settings["device"], "dtype": settings["dtype"]}},
            project_root=root,
        ),
        "cpu_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "host_contention": {
            "orchestrator_reported_study_training_idle": True,
            "before": _host_observation(),
            "limitation": (
                "load averages are observations; other processes are not "
                "experimentally controlled"
            ),
        },
        "costs": [],
        "ratios": [],
    }
    by_id = {entry["stage_id"]: entry for entry in manifest["entries"]}
    solver_records, network_records = {}, []
    interrupted = False
    try:
        for stage in protocol["matrix"]:
            entry = by_id[stage["stage_id"]]
            info = verified[stage["stage_id"]]
            path = _path(root, info["run_path"])
            identity = {
                "stage_id": stage["stage_id"],
                "run_id": entry["actual_run_id"],
                "case_id": stage["case_id"],
                "variant": stage.get("variant"),
                "seed": stage.get("seed"),
                "cells": stage.get("cells"),
                "run_provenance": info,
            }
            case = protocol["resolved"]["cases"][stage["case_id"]]
            if stage["stage"] == "train":
                metadata = _read(path / "metadata.json")
                report["costs"].append(
                    {
                        **identity,
                        "category": "saved_actual_training_and_persistence",
                        "repeated_for_timing": False,
                        "execution": metadata["execution"],
                        "history_sha256": file_sha256(path / "history.csv"),
                        "weighting_overhead": (
                            "included in recorded training wall time; not isolated"
                        ),
                    }
                )
                continue
            if stage["stage"] == "solve":
                if entry["state"] != "completed":
                    row = {
                        **identity,
                        "category": "numerical_integration_and_reconstruction",
                        "state": "invalid_saved_run",
                        "reason": info["stop_reason"],
                        "median_s": None,
                    }
                    report["costs"].append(row)
                    solver_records[(stage["case_id"], stage["cells"])] = row
                    continue
                measured = measure_workload(
                    solver_workload(case, stage["cells"], x, times),
                    lambda value: _validate_output(value, (50, 400, 2)),
                    warmups=warmups,
                    repetitions=repetitions,
                )
                row = {
                    **identity,
                    "category": "numerical_integration_and_reconstruction",
                    "dtype": "float64",
                    "device": "cpu",
                    "output_location": "host_numpy",
                    **measured,
                }
                report["costs"].append(row)
                solver_records[(stage["case_id"], stage["cells"])] = row
                (edges, h, q), cache_load_s = _one_time(
                    lambda path=path: _native_cache(path), "cpu"
                )

                def cached_reconstruction(edges=edges, h=h, q=q):
                    return np.stack(
                        (
                            piecewise_constant_reconstruct(edges, h, x),
                            piecewise_constant_reconstruct(edges, q, x),
                        ),
                        axis=-1,
                    )

                cache = measure_workload(
                    cached_reconstruction,
                    lambda value: _validate_output(value, (50, 400, 2)),
                    warmups=warmups,
                    repetitions=repetitions,
                )
                report["costs"].append(
                    {
                        **identity,
                        "category": "resident_classical_cache_reconstruction",
                        "cache_scope": (
                            "saved native h/q arrays resident; file loading excluded"
                        ),
                        "saved_native_file_loading_s": cache_load_s,
                        "file_loading_observations": 1,
                        **cache,
                    }
                )
            elif stage["stage"] == "reference":

                def reference_work(case=case):
                    result = evaluate_reference(case, x, times)
                    return np.stack((result.h_m, result.q_m2_s), axis=-1)

                measured = measure_workload(
                    reference_work,
                    lambda value: _validate_output(value, (50, 400, 2)),
                    warmups=warmups,
                    repetitions=repetitions,
                )
                report["costs"].append(
                    {
                        **identity,
                        "category": "analytical_reference_generation",
                        "device": "cpu",
                        "dtype": "float64",
                        **measured,
                    }
                )
            elif stage["stage"] == "predict":
                training_id = by_id[stage["training_stage_id"]]["actual_run_id"]

                def load_model(training_id=training_id):
                    checkpoint, source, checkpoint_path = load_training_checkpoint(
                        training_id, output_root, require_completed=True
                    )
                    model = ShallowWaterPINN(
                        source["case"],
                        source["training"],
                        device="cpu",
                        dtype=settings["dtype"],
                    )
                    model.load_state_dict(
                        checkpoint["selected_model_state"], strict=True
                    )
                    model.eval()
                    return model, checkpoint_path

                (model, checkpoint_path), loading_s = _one_time(load_model, "cpu")
                device = settings["device"]
                model, model_transfer_s = _one_time(
                    lambda model=model, device=device: model.to(device=device), device
                )
                coordinates, input_transfer_s = _one_time(
                    lambda model=model, device=device: torch.tensor(
                        coordinate_array,
                        dtype=next(model.parameters()).dtype,
                        device=device,
                    ),
                    device,
                )
                workload = network_workload(model, coordinates, settings["chunk_size"])
                measured = measure_workload(
                    workload,
                    lambda value: _validate_output(value, (20000, 2)),
                    device=device,
                    warmups=warmups,
                    repetitions=repetitions,
                )
                resident = workload()
                transfer = measure_workload(
                    lambda resident=resident: resident.cpu().numpy().copy(),
                    lambda value: _validate_output(value, (20000, 2)),
                    device=device,
                    warmups=warmups,
                    repetitions=repetitions,
                )
                row = {
                    **identity,
                    "category": "resident_network_physical_forward",
                    "training_run_id": training_id,
                    "checkpoint_sha256": file_sha256(checkpoint_path / "checkpoint.pt"),
                    "dtype": str(next(model.parameters()).dtype),
                    "device": device,
                    "batch_points": 20000,
                    "chunk_size": settings["chunk_size"],
                    "output_location": f"resident {device} tensor",
                    "loading_and_transfer": {
                        "trusted_checkpoint_validation_and_cpu_model_load_s": loading_s,
                        "model_cpu_to_device_s": model_transfer_s,
                        "input_tensor_creation_and_transfer_s": input_transfer_s,
                        "observations_per_setup_cost": 1,
                        "output_transfer_and_host_copy": transfer,
                    },
                    **measured,
                }
                report["costs"].append(row)
                network_records.append(row)

                def forward_host_export(workload=workload):
                    return workload().cpu().numpy().copy()

                host_export = measure_workload(
                    forward_host_export,
                    lambda value: _validate_output(value, (20000, 2)),
                    device=device,
                    warmups=warmups,
                    repetitions=repetitions,
                )
                report["costs"].append(
                    {
                        **identity,
                        "category": "network_forward_and_host_export",
                        "device": device,
                        "dtype": str(next(model.parameters()).dtype),
                        "output_location": "host numpy copy",
                        "setup_costs_excluded": (
                            "checkpoint validation/loading, model transfer, and "
                            "input creation/transfer are separately recorded"
                        ),
                        **host_export,
                    }
                )
                del resident, model, coordinates
            if any(row.get("state") == "interrupted" for row in report["costs"]):
                interrupted = True
                break
        for network in network_records:
            primary = solver_records.get(
                (network["case_id"], 400), {"state": "missing"}
            )
            report["ratios"].append(
                {
                    "case_id": network["case_id"],
                    "network_run_id": network["run_id"],
                    "numerical_run_id": primary.get("run_id"),
                    "numerical_cells": 400,
                    "numerical_median_over_network_median": speed_ratio(
                        primary, network
                    ),
                    "hardware_precision_context": (
                        "CPU float64 numerical solver versus "
                        f"{network['device']} {network['dtype']} network; "
                        "ratio includes these differences"
                    ),
                    "cache_context": (
                        "fresh classical integration ratio; resident cached "
                        "classical reconstruction is a separate cost"
                    ),
                }
            )
        primary_invalid = [
            case
            for case in ("E1", "E2")
            if solver_records.get((case, 400), {}).get("state") != "completed"
        ]
        report["primary_comparator"] = {
            "cells": 400,
            "invalid_cases": primary_invalid,
            "state": "protocol_resolution_required" if primary_invalid else "available",
            "alternative_grid_selected": False,
        }
        report["state"] = (
            "interrupted"
            if interrupted
            else (
                "completed"
                if all(
                    row.get("state", "completed") == "completed"
                    for row in report["costs"]
                )
                else "completed_with_invalid_workloads"
            )
        )
        report["host_contention"]["after"] = _host_observation()
        report["interpretation_limits"] = [
            "Read common-grid error and mass evidence alongside these timing samples.",
            "A new scenario requiring retraining includes the recorded training cost.",
            "Classical reuse requires the separate cached reconstruction comparison.",
            "Unavailable hardware has no timing evidence in this report.",
        ]
        write_json_atomic(destination, report)
    except (Exception, KeyboardInterrupt) as exc:
        report.update(
            state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            reason=f"{type(exc).__name__}: {exc}",
        )
        report["host_contention"]["after"] = _host_observation()
        # Preserve all samples already collected, including a setup/readback failure.
        if not destination.exists():
            write_json_atomic(destination, report)
        raise
    finally:
        torch.set_num_threads(previous_threads)
    return report
