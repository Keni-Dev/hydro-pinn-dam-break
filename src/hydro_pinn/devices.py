"""Observed device inventory and bounded, nonfinal training pilots.

The pilot calls the same training path used by experiments. Precision checks
copy one CPU float64 initialization and use the same physics-only samples.
Unavailable hardware and unsupported dtypes retain diagnostic run records.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import resource
import shutil
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hydro_pinn.config import load_config, validate_config
from hydro_pinn.losses import COMPONENTS, residuals
from hydro_pinn.model import ShallowWaterPINN
from hydro_pinn.records import RunRecord, collect_metadata, write_json_atomic
from hydro_pinn.sampling import generate_training_samples, load_samples, save_samples
from hydro_pinn.train import component_gradients, load_training_checkpoint, train_run


def _query(arguments: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["rtk", "proxy", *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if result.returncode:
            return {"value": None, "reason": result.stderr.strip()}
        return {"value": result.stdout.strip(), "source": " ".join(arguments)}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"value": None, "reason": f"{type(exc).__name__}: {exc}"}


def memory_snapshot(device: str = "cpu") -> dict[str, Any]:
    """Report measured process high-water RSS and accessible allocator values.

    RSS is a process-lifetime high water, not a per-run delta or GPU peak.
    MPS allocator snapshots are endpoints/observations, not a true peak counter.
    """
    result: dict[str, Any] = {
        "process_lifetime_peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024)
        ),
        "rss_scope": "process-lifetime high water; includes earlier pilot work",
        "device_allocator": {},
    }
    if device == "mps" and torch.backends.mps.is_available():
        for name in (
            "current_allocated_memory",
            "driver_allocated_memory",
            "recommended_max_memory",
        ):
            method = getattr(torch.mps, name, None)
            if method is not None:
                try:
                    result["device_allocator"][f"{name}_bytes"] = int(method())
                except RuntimeError as exc:
                    result["device_allocator"][name] = {"reason": str(exc)}
        result["allocator_scope"] = "MPS observed snapshot; no true peak API used"
    elif device == "cuda" and torch.cuda.is_available():
        result["device_allocator"] = {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        result["allocator_scope"] = "CUDA process/device allocator counters"
    else:
        result["allocator_scope"] = "CPU RSS only or unavailable accelerator"
    return result


def inspect_devices() -> dict[str, Any]:
    """Inspect this process's environment without claiming execution support."""
    metadata = collect_metadata({"request": {"device": "cpu", "dtype": "float64"}})
    usage = shutil.disk_usage(Path.cwd())
    cuda = []
    if torch.cuda.is_available():
        for index in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(index)
            cuda.append(
                {"index": index, "model": prop.name, "vram_bytes": prop.total_memory}
            )
    observed = {
        "os": metadata["hardware"],
        "software": metadata["software"],
        "cpu": metadata["device"],
        "torch_build": torch.__config__.show(),
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "backends": {
            "cpu": {"available": True},
            "mps": {
                "built": torch.backends.mps.is_built(),
                "available": torch.backends.mps.is_available(),
                "float64": "unsupported; no CPU fallback changes this limit",
            },
            "cuda": {
                "available": torch.cuda.is_available(),
                "build": torch.version.cuda,
                "devices": cuda,
            },
        },
        "storage": {
            "path": str(Path.cwd()),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
            "source": "shutil.disk_usage for project filesystem",
        },
        "memory": memory_snapshot(),
    }
    if sys.platform == "darwin":
        observed["mac_model_identifier"] = _query(["sysctl", "-n", "hw.model"])
        observed["mac_cpu_identifier"] = _query(
            ["sysctl", "-n", "machdep.cpu.brand_string"]
        )
        observed["mac_gpu_inventory"] = _query(
            ["system_profiler", "SPDisplaysDataType", "-json"]
        )
        observed["memory_pressure_snapshot"] = _query(["vm_stat"])
    return {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "observed": observed,
        "reported_hardware": {
            "mac": "User reports M5 MacBook Air, 24 GB memory, 1 TB storage",
            "victus": "User reports HP Victus 15 with RTX 2050; RAM/VRAM unverified",
        },
        "environment": {
            name: os.environ.get(name)
            for name in (
                "PYTORCH_ENABLE_MPS_FALLBACK",
                "PYTORCH_MPS_HIGH_WATERMARK_RATIO",
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
            )
        },
        "limits": [
            "Backend availability alone does not verify PINN gradients or throughput",
            "Only devices exposed to this process were inventoried",
            "Unavailable Victus is untested; no comparative speed or VRAM inferred",
            "Filesystem capacity is observed volume capacity, not physical SSD model",
        ],
    }


def write_doctor(output_path: str | Path) -> dict[str, Any]:
    report = inspect_devices()
    write_json_atomic(output_path, report)
    return report


def _backend_reason(device: str, dtype: str) -> str | None:
    if device not in {"cpu", "cuda", "mps"}:
        return f"unsupported requested backend {device!r}"
    if dtype not in {"float32", "float64"}:
        return f"unsupported requested dtype {dtype!r}"
    if device == "mps" and dtype == "float64":
        return "MPS does not support float64; no CPU fallback applied"
    if device == "mps" and os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
        return "MPS CPU fallback enabled; native MPS pilot requires fallback disabled"
    if device == "mps" and not torch.backends.mps.is_available():
        return "MPS unavailable to this process; hardware untested; no fallback"
    if device == "cuda" and not torch.cuda.is_available():
        return "CUDA unavailable to this process; hardware untested; no fallback"
    return None


def _synchronize(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def compare_arrays(
    actual: np.ndarray, reference: np.ndarray, tolerance: dict[str, float]
) -> dict[str, Any]:
    """Use the predeclared elementwise atol+rtol*abs(reference) criterion."""
    actual = np.asarray(actual, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    difference = actual - reference
    reference_norm = float(np.linalg.norm(reference.ravel()))
    return {
        "passed": bool(np.allclose(actual, reference, **tolerance)),
        "max_abs_difference": float(np.max(np.abs(difference))),
        "difference_l2": float(np.linalg.norm(difference.ravel())),
        "reference_l2": reference_norm,
        "relative_l2_difference": (
            float(np.linalg.norm(difference.ravel())) / reference_norm
            if reference_norm
            else None
        ),
        "relative_l2_reason": None if reference_norm else "zero reference norm",
        "tolerance": tolerance,
        "criterion": "all abs(actual-reference) <= atol+rtol*abs(reference)",
    }


def _model_measurements(model, samples, chunk_size=None) -> dict[str, np.ndarray]:
    parameter = next(model.parameters())
    coords = torch.as_tensor(
        samples["interior_xt_nd"], device=parameter.device, dtype=parameter.dtype
    )
    result = {
        "residual": residuals(model, coords).detach().cpu().double().numpy(),
    }
    for name in COMPONENTS:
        value, gradients = component_gradients(model, samples, name, chunk_size)
        result[f"loss_{name}"] = np.asarray([value], dtype=np.float64)
        result[f"gradient_{name}"] = np.concatenate(
            [g.detach().cpu().double().numpy().ravel() for g in gradients]
        )
    return result


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    with path.open("xb") as handle:
        np.savez(handle, **arrays)


def _precision_attempt(
    candidate: dict,
    config: dict,
    case: dict,
    training: dict,
    samples: dict,
    state: dict,
    reference: dict,
    input_hash: str,
) -> dict:
    device, dtype, run_id = (candidate[k] for k in ("device", "dtype", "run_id"))
    resolved = {
        "case": case,
        "training": training,
        "request": {
            "seed": config["seed"],
            "device": device,
            "dtype": dtype,
            "run_id": run_id,
        },
        "pilot": {
            "purpose": "precision_and_chunk_diagnostic",
            "candidate": candidate,
            "precision": config["precision"],
            "config_sha256": input_hash,
        },
    }
    result = {"run_id": run_id, "device": device, "dtype": dtype}
    with RunRecord.create(
        config["output_root"],
        run_id,
        resolved,
        metadata={"stage": "device_precision_pilot"},
    ) as run:
        result["run_path"] = str(run.path)
        reason = _backend_reason(device, dtype)
        if reason:
            result.update(
                outcome="unsupported_or_unavailable",
                reason=reason,
                execution_tested=False,
                passed=False,
            )
            write_json_atomic(run.path / "precision.json", result)
            run.fail(reason, stage="device_preflight")
            return result
        try:
            save_samples(run.path / "sampling.npz", samples)
            _save_npz(
                run.path / "initial_cpu64_state.npz",
                {k: v.numpy() for k, v in state.items()},
            )
            model = ShallowWaterPINN(case, training, device=device, dtype=dtype)
            model.load_state_dict(state, strict=True)
            started = time.perf_counter()
            _synchronize(device)
            actual = _model_measurements(model, samples)
            chunked = _model_measurements(
                model, samples, config["precision"]["chunk_size"]
            )
            _synchronize(device)
            elapsed = time.perf_counter() - started
            tolerance = config["precision"]["tolerances"][dtype]
            comparisons = {
                k: compare_arrays(actual[k], reference[k], tolerance) for k in reference
            }
            chunks = {
                k: compare_arrays(chunked[k], actual[k], tolerance) for k in actual
            }
            passed = all(v["passed"] for v in (*comparisons.values(), *chunks.values()))
            _save_npz(
                run.path / "precision_arrays.npz",
                {
                    **{f"reference_{k}": v for k, v in reference.items()},
                    **{f"actual_{k}": v for k, v in actual.items()},
                    **{f"chunked_{k}": v for k, v in chunked.items()},
                },
            )
            result.update(
                outcome="passed" if passed else "precision_disagreement",
                passed=passed,
                execution_tested=True,
                parameter_count=sum(p.numel() for p in model.parameters()),
                reference="identical CPU float64 initialization and samples",
                comparisons=comparisons,
                chunk_comparisons=chunks,
                elapsed_s=elapsed,
                memory=memory_snapshot(device),
                actual_device=str(next(model.parameters()).device),
                actual_dtype=str(next(model.parameters()).dtype),
            )
            write_json_atomic(run.path / "precision.json", result)
            if passed:
                run.complete(
                    stopping_reason=(
                        "precision/chunk comparisons passed declared tolerances"
                    ),
                    elapsed_timings_s={"precision_comparisons": elapsed},
                )
            else:
                run.fail(
                    "precision or chunk comparison exceeded declared tolerance",
                    stage="precision_comparison",
                )
        except (RuntimeError, ValueError, OSError) as exc:
            result.update(
                outcome="execution_failed",
                passed=False,
                reason=f"{type(exc).__name__}: {exc}",
                execution_tested=True,
            )
            if not (run.path / "precision.json").exists():
                write_json_atomic(run.path / "precision.json", result)
            run.fail(result["reason"], stage="precision_execution")
    return result


def summarize_timings(events: list[dict], warmup_steps: int) -> dict[str, Any]:
    included = [e for e in events if e["step"] > warmup_steps]
    if not included:
        raise ValueError("pilot must measure at least one step after warmup")
    result: dict[str, Any] = {
        "actual_steps": len(events),
        "warmup_steps_excluded": warmup_steps,
        "measured_steps": len(included),
        "warmup_total_s": sum(
            e["total_s"] for e in events if e["step"] <= warmup_steps
        ),
        "weights_updated_steps": [e["step"] for e in events if e["weights_updated"]],
        "scope": (
            "synchronized objective+Adam+validation/selection; checkpoint I/O separate"
        ),
    }
    for field in ("compute_s", "checkpoint_io_s", "total_s"):
        values = [e[field] for e in included]
        result[field] = {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "minimum": min(values),
            "maximum": max(values),
            "sum": sum(values),
        }
    result["updated_step_timings"] = [
        {k: e[k] for k in ("step", "compute_s", "checkpoint_io_s", "total_s")}
        for e in included
        if e["weights_updated"]
    ]
    return result


def _unsupported_workload(workload: dict, config: dict, reason: str) -> dict:
    run_id = workload["run_id"]
    resolved = {
        "request": {
            "seed": config["seed"],
            "device": workload["device"],
            "dtype": workload["dtype"],
            "run_id": run_id,
        },
        "pilot_workload": workload,
    }
    result = {
        **workload,
        "outcome": "unsupported_or_blocked",
        "reason": reason,
        "actual_steps": 0,
        "execution_tested": False,
    }
    with RunRecord.create(
        config["output_root"],
        run_id,
        resolved,
        metadata={"stage": "blocked_training_pilot"},
    ) as run:
        result["run_path"] = str(run.path)
        write_json_atomic(run.path / "pilot_timing.json", result)
        run.fail(reason, stage="pilot_preflight")
    return result


def _training_attempt(workload: dict, config: dict, precision_passed: bool) -> dict:
    device, dtype, run_id = (workload[k] for k in ("device", "dtype", "run_id"))
    reason = _backend_reason(device, dtype)
    if reason or not precision_passed:
        return _unsupported_workload(
            workload, config, reason or "matching precision comparison did not pass"
        )
    events: list[dict] = []
    observed_memory: list[dict] = []
    before = memory_snapshot(device)

    def observe(event):
        events.append(event.copy())
        observed_memory.append(memory_snapshot(device))

    overrides = {
        "training": {
            "sampling": workload["sampling"],
            "optimizer": {"steps": workload["steps"]},
            "lbfgs": {"enabled": False},
        }
    }
    case_path = workload.get("case", config["case"])
    started = time.perf_counter()
    path = Path(config["output_root"]).absolute() / run_id
    result = {
        **workload,
        "case": case_path,
        "run_path": str(path),
        "execution_tested": True,
        "memory_before": before,
    }
    try:
        _synchronize(device)
        train_run(
            case_path,
            config["training"][workload["variant"]],
            config["seed"],
            device,
            dtype,
            run_id,
            output_root=config["output_root"],
            chunk_size=workload["chunk_size"],
            overrides=overrides,
            step_observer=observe,
            synchronize_steps=True,
        )
        _synchronize(device)
        elapsed = time.perf_counter() - started
        checkpoint, resolved, _ = load_training_checkpoint(
            run_id, config["output_root"], require_completed=True
        )
        model = ShallowWaterPINN(
            resolved["case"], resolved["training"], device=device, dtype=dtype
        )
        model.load_state_dict(checkpoint["selected_model_state"], strict=True)
        samples = load_samples(path / "validation.npz")
        parameter = next(model.parameters())
        coordinates = torch.as_tensor(
            samples["interior_xt_nd"][:17],
            device=parameter.device,
            dtype=parameter.dtype,
        )
        with torch.no_grad():
            first = model(coordinates).cpu().double().numpy()
        second_checkpoint, _, _ = load_training_checkpoint(
            run_id, config["output_root"], require_completed=True
        )
        model.load_state_dict(second_checkpoint["selected_model_state"], strict=True)
        with torch.no_grad():
            second = model(coordinates).cpu().double().numpy()
        if not np.isfinite(first).all() or not np.array_equal(first, second):
            raise ValueError(
                "selected checkpoint readback predictions disagree or nonfinite"
            )
        _save_npz(
            path / "pilot_reload_predictions.npz",
            {
                "coordinates_xt_nd": samples["interior_xt_nd"][:17],
                "first_hq_nd": first,
                "second_hq_nd": second,
            },
        )
        result.update(
            outcome="completed_finite_budget",
            timings=summarize_timings(events, workload["warmup_steps"]),
            elapsed_training_and_save_s=elapsed,
            unattributed_setup_observer_finalization_s=elapsed
            - sum(e["total_s"] for e in events),
            unattributed_timing_scope=(
                "initial setup/validation, observer memory queries, synchronization "
                "gaps and finalization outside recorded step intervals"
            ),
            actual_steps=checkpoint["step"],
            selected_step=checkpoint["selected_step"],
            selected_validation_score=checkpoint["selected_validation_score"],
            checkpoint_reload={
                "exact_agreement": True,
                "coordinate_partition": "validation_only",
                "finite": True,
                "points": len(first),
            },
            actual_device=str(parameter.device),
            actual_dtype=str(parameter.dtype),
            memory_after=memory_snapshot(device),
            observed_peak_process_rss_bytes=max(
                m["process_lifetime_peak_rss_bytes"] for m in [before, *observed_memory]
            ),
            allocator_observations=observed_memory,
            timing_events=events,
            interpretation=(
                "finite bounded training; convergence and final accuracy untested"
            ),
            resolved_config_sha256=hashlib.sha256(
                (path / "resolved_config.json").read_bytes()
            ).hexdigest(),
        )
    except (RuntimeError, ValueError, OSError) as exc:
        result.update(
            outcome="execution_failed",
            reason=f"{type(exc).__name__}: {exc}",
            actual_steps=len(events),
            timing_events=events,
            memory_after=memory_snapshot(device),
        )
        if not path.exists():
            return _unsupported_workload(workload, config, result["reason"])
    write_json_atomic(path / "pilot_timing.json", result)
    return result


def _read_pilot_config(path: str | Path) -> dict:
    def reject_constant(value):
        raise ValueError(f"nonfinite JSON constant {value}")

    config = json.loads(Path(path).read_text(), parse_constant=reject_constant)
    if (
        config.get("schema_version") != 1
        or config.get("purpose") != "bounded_hardware_pilot"
    ):
        raise ValueError(
            "pilot config requires schema_version=1 and bounded_hardware_pilot purpose"
        )
    if not isinstance(config["cpu_threads"], int) or config["cpu_threads"] < 1:
        raise ValueError("cpu_threads must be a positive integer")
    ids = [c["run_id"] for c in config["precision"]["candidates"]]
    ids.extend(w["run_id"] for w in config["workloads"])
    if len(ids) != len(set(ids)):
        raise ValueError("pilot run IDs must be unique")
    for workload in config["workloads"]:
        steps, warmup = workload["steps"], workload["warmup_steps"]
        if not isinstance(steps, int) or not 0 <= warmup < steps <= 200:
            raise ValueError("pilot requires 0<=warmup_steps<steps<=200")
        if (
            workload["kind"] == "representative"
            and workload["variant"] == "adaptive"
            and steps < 100
        ):
            raise ValueError("representative adaptive pilot must exercise step100")
    for tolerance in config["precision"]["tolerances"].values():
        if set(tolerance) != {"atol", "rtol"} or any(
            not math.isfinite(v) or v < 0 for v in tolerance.values()
        ):
            raise ValueError(
                "precision tolerances must be finite nonnegative atol/rtol"
            )
    return config


def run_pilot(config_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Run the declared bounded attempts; preserve failures and refuse rerun IDs."""
    config = _read_pilot_config(config_path)
    if Path(output_path).exists():
        raise FileExistsError(f"pilot findings already exist: {output_path}")
    identifiers = [c["run_id"] for c in config["precision"]["candidates"]]
    identifiers.extend(w["run_id"] for w in config["workloads"])
    for run_id in identifiers:
        if (Path(config["output_root"]) / run_id).exists():
            raise FileExistsError(f"pilot run exists; declare fresh run IDs: {run_id}")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(config["cpu_threads"])
    try:
        case = load_config(config["case"])
        training = load_config(config["training"]["baseline"])
        training = copy.deepcopy(training)
        training["sampling"].update(config["precision"]["sampling"])
        validate_config(training)
        torch.manual_seed(config["seed"])
        reference_model = ShallowWaterPINN(
            case, training, device="cpu", dtype="float64"
        )
        state = {k: v.detach().clone() for k, v in reference_model.state_dict().items()}
        samples = generate_training_samples(case, training, config["seed"])
        reference = _model_measurements(reference_model, samples)
        digest = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
        precision_results = [
            _precision_attempt(
                c, config, case, training, samples, state, reference, digest
            )
            for c in config["precision"]["candidates"]
        ]
        accepted = {(r["device"], r["dtype"]): r["passed"] for r in precision_results}
        results = []
        for workload in config["workloads"]:
            result = _training_attempt(
                workload,
                config,
                accepted.get((workload["device"], workload["dtype"]), False),
            )
            results.append(result)
            print(
                json.dumps(
                    {
                        "pilot_run": result["run_id"],
                        "outcome": result["outcome"],
                        "actual_steps": result["actual_steps"],
                    }
                ),
                flush=True,
            )
        report = {
            "schema_version": 1,
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "purpose": "hardware_and_training_pilot_not_final_accuracy",
            "config_path": str(config_path),
            "config_sha256": digest,
            "cpu_threads": torch.get_num_threads(),
            "seed": config["seed"],
            "inventory": inspect_devices(),
            "precision": precision_results,
            "workloads": results,
            "final_evaluation_consumed": False,
            "limitations": [
                "Bounded pilot timing does not establish sustained thermal performance",
                "RSS is process-lifetime high-water; MPS memory is observed snapshots",
                "No final analytical errors or checkpoint tuning used",
                "Victus/CUDA untested unless available in this process",
            ],
        }
        write_json_atomic(output_path, report)
        return report
    finally:
        torch.set_num_threads(previous_threads)
