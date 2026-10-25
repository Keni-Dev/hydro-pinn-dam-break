"""Physics-only optimization, trusted local checkpoints, and SI prediction.

Validation uses separate physics/IC/BC coordinates without interior solution
labels. Its fixed configured weights select the minimum validation objective.
The selected model and the latest finite optimizer state are stored separately
so checkpoint selection does not silently change continuation semantics.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import pickle
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hydro_pinn.config import load_config, resolve_run_config, validate_config
from hydro_pinn.losses import (
    gradient_rms,
    iter_component_losses,
    update_adaptive_weights,
)
from hydro_pinn.model import ShallowWaterPINN
from hydro_pinn.records import RunRecord, validate_run_id, write_json_atomic
from hydro_pinn.sampling import (
    generate_training_samples,
    generate_validation_samples,
    load_samples,
    save_samples,
)

COMPONENTS = ("pde", "ic", "bc")
SELECTION = "minimum_separate_validation_physics_ic_bc_loss"
HISTORY_FIELDS = [
    "phase",
    "step",
    "loss_pde",
    "loss_ic",
    "loss_bc",
    "loss_total",
    "lambda_pde",
    "lambda_ic",
    "lambda_bc",
    "grad_pde",
    "grad_ic",
    "grad_bc",
    "weights_updated",
    "validation_score",
    "selected_step",
    "elapsed_s",
    "stop_reason",
]


class TrainingError(ValueError):
    """Incompatible state or a nonfinite objective/gradient stopped execution."""


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_hash(config: dict) -> str:
    return hashlib.sha256(
        (json.dumps(config, allow_nan=False, sort_keys=True, indent=2) + "\n").encode()
    ).hexdigest()


def _safe_run(output_root: str | Path, run_id: str) -> Path:
    validate_run_id(run_id)
    path = Path(output_root).absolute() / run_id
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise TrainingError("training run path must not traverse parents or symlinks")
    return path


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(child) for child in value)
    if isinstance(value, list):
        return [_cpu_tree(child) for child in value]
    return copy.deepcopy(value)


def _rng_state(device: str) -> dict:
    legacy = np.random.get_state()
    state = {
        "torch_cpu": torch.get_rng_state(),
        "python": random.getstate(),
        "numpy": [legacy[0], legacy[1].tolist(), legacy[2], legacy[3], legacy[4]],
    }
    if device == "cuda":
        state["torch_device"] = torch.cuda.get_rng_state_all()
    elif device == "mps":
        state["torch_device"] = torch.mps.get_rng_state()
    return state


def _restore_rng(state: dict, device: str) -> None:
    torch.set_rng_state(state["torch_cpu"])
    random.setstate(state["python"])
    legacy = state["numpy"]
    np.random.set_state((legacy[0], np.array(legacy[1], dtype=np.uint32), *legacy[2:]))
    if device == "cuda":
        torch.cuda.set_rng_state_all(state["torch_device"])
    elif device == "mps":
        torch.mps.set_rng_state(state["torch_device"])


def _finite_model(model: torch.nn.Module) -> None:
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise TrainingError("nonfinite model parameters")


def component_gradients(
    model: torch.nn.Module, samples: dict, component: str, chunk_size: int | None = None
) -> tuple[float, list[torch.Tensor]]:
    """Accumulate the complete unweighted component gradient before its norm.

    Each chunk graph is released after autograd; chunk fractions sum to one.
    No optimizer update occurs here. Absent gradients occupy zero buffers.
    """
    parameters = list(model.parameters())
    accumulated = [torch.zeros_like(p) for p in parameters]
    value = 0.0
    for fraction, loss in iter_component_losses(model, samples, component, chunk_size):
        if not bool(torch.isfinite(loss)):
            raise TrainingError(f"nonfinite {component} loss")
        value += fraction * float(loss.detach())
        if loss.requires_grad:
            gradients = torch.autograd.grad(
                fraction * loss, parameters, allow_unused=True
            )
            for buffer, gradient in zip(accumulated, gradients, strict=True):
                if gradient is not None:
                    buffer.add_(gradient.detach())
    gradient_rms(accumulated, parameters)  # also checks finiteness
    return value, accumulated


def _validation(
    model: torch.nn.Module, samples: dict, weights: dict, chunk_size: int | None
) -> float:
    values = {}
    with torch.enable_grad():
        for name in COMPONENTS:
            value = 0.0
            for fraction, loss in iter_component_losses(
                model, samples, name, chunk_size
            ):
                value += fraction * float(loss.detach())
            values[name] = value
    score = sum(weights[name] * values[name] for name in COMPONENTS)
    if not math.isfinite(score):
        raise TrainingError("nonfinite separate validation objective")
    return score


def _save_checkpoint(path: Path, checkpoint: dict) -> None:
    """Atomically update only this running run's trusted local checkpoint."""
    if path.is_symlink():
        raise TrainingError("checkpoint path must not be a symlink")
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".pt.tmp", delete=False
    ) as f:
        temporary = Path(f.name)
        try:
            torch.save(_cpu_tree(checkpoint), f)
            f.flush()
            os.fsync(f.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    manifest = {
        "schema_version": 1,
        "filename": path.name,
        "sha256": _hash(path),
        "run_id": checkpoint["run_id"],
        "step": checkpoint["step"],
        "selected_step": checkpoint["selected_step"],
        "resolved_config_sha256": checkpoint["resolved_config_sha256"],
        "partition_hashes": checkpoint["partition_hashes"],
    }
    manifest_path = path.with_name("checkpoint_manifest.json")
    write_json_atomic(manifest_path, manifest, overwrite=manifest_path.exists())


def load_training_checkpoint(
    run_id: str, output_root: str | Path = "runs", *, require_completed: bool = False
) -> tuple[dict, dict, Path]:
    """Read a hash-bound local checkpoint with weights_only=True.

    This is a local provenance/integrity contract, not a remote artifact trust
    service. All source artifacts must remain regular files within the run.
    """
    path = _safe_run(output_root, run_id)
    required = [
        "resolved_config.json",
        "metadata.json",
        "status.json",
        "checkpoint.pt",
        "checkpoint_manifest.json",
        "sampling.npz",
        "validation.npz",
    ]
    for name in required:
        if not (path / name).is_file() or (path / name).is_symlink():
            raise TrainingError(
                f"missing or unsafe trusted checkpoint artifact: {name}"
            )
    try:
        resolved = json.loads((path / "resolved_config.json").read_text())
        metadata = json.loads((path / "metadata.json").read_text())
        status = json.loads((path / "status.json").read_text())
        manifest = json.loads((path / "checkpoint_manifest.json").read_text())
        if any(record["run_id"] != run_id for record in (metadata, status, manifest)):
            raise TrainingError("source artifact run identity mismatch")
        if status["state"] not in {"completed", "failed", "interrupted"}:
            raise TrainingError("source training run is not terminal")
        if require_completed and status["state"] != "completed":
            raise TrainingError("prediction requires a completed training run")
        if metadata["stage_metadata"]["stage"] != "pinn_training":
            raise TrainingError("source run is not a PINN training stage")
        digest = _hash(path / "resolved_config.json")
        if (
            digest != metadata["resolved_config_sha256"]
            or digest != manifest["resolved_config_sha256"]
        ):
            raise TrainingError("resolved configuration integrity mismatch")
        if _hash(path / "checkpoint.pt") != manifest["sha256"]:
            raise TrainingError("checkpoint SHA256 mismatch")
        checkpoint = torch.load(
            path / "checkpoint.pt", map_location="cpu", weights_only=True
        )
        if checkpoint["schema_version"] != 1 or checkpoint["run_id"] != run_id:
            raise TrainingError("checkpoint schema/run identity mismatch")
        if (
            checkpoint["resolved_config_sha256"] != digest
            or checkpoint["resolved_config"] != resolved
        ):
            raise TrainingError("checkpoint configuration mismatch")
        if checkpoint["scales"] != resolved["case"]["scales"]:
            raise TrainingError("checkpoint scales mismatch")
        if (
            checkpoint["step"] != manifest["step"]
            or checkpoint["selected_step"] != manifest["selected_step"]
        ):
            raise TrainingError("checkpoint step manifest mismatch")
        for name in ("sampling.npz", "validation.npz"):
            actual = _hash(path / name)
            if (
                actual != manifest["partition_hashes"][name]
                or actual != checkpoint["partition_hashes"][name]
            ):
                raise TrainingError(f"partition integrity mismatch: {name}")
            load_samples(path / name)
        validate_config(resolved["case"])
        validate_config(resolved["training"])
        for model_state in (
            checkpoint["latest_model_state"],
            checkpoint["selected_model_state"],
        ):
            if any(not bool(torch.isfinite(v).all()) for v in model_state.values()):
                raise TrainingError("nonfinite checkpoint model state")
        return checkpoint, resolved, path
    except (
        KeyError,
        TypeError,
        RuntimeError,
        OSError,
        ValueError,
        pickle.UnpicklingError,
        EOFError,
    ) as exc:
        if isinstance(exc, TrainingError):
            raise
        raise TrainingError(f"invalid trusted local checkpoint: {exc}") from exc


def _resume_compatible(previous: dict, resolved: dict, checkpoint: dict) -> dict:
    old, new = copy.deepcopy(previous), copy.deepcopy(resolved)
    old_budget, new_budget = (
        old["training"]["optimizer"].pop("steps"),
        new["training"]["optimizer"].pop("steps"),
    )
    for name in ("case", "training", "evaluation", "execution_settings"):
        if old.get(name) != new.get(name):
            raise TrainingError(f"incompatible resume {name}")
    for name in ("seed", "device", "dtype"):
        if old["request"][name] != new["request"][name]:
            raise TrainingError(f"incompatible resume request.{name}")
    if checkpoint.get("lbfgs_completed"):
        raise TrainingError("resume after completed LBFGS is a new experiment")
    if new_budget <= checkpoint["step"] or new_budget < old_budget:
        raise TrainingError(
            "resume needs an explicit compatible total budget beyond the saved step"
        )
    return {
        "source_total_adam_steps": old_budget,
        "requested_total_adam_steps": new_budget,
        "budget_extension_steps": max(0, new_budget - old_budget),
        "budget_extension_source": "explicit resolved training optimizer.steps",
    }


def train_run(
    case_path: str | Path,
    training_path: str | Path,
    seed: int,
    device: str,
    dtype: str,
    run_id: str,
    *,
    smoke_path: str | Path | None = None,
    resume_from: str | None = None,
    output_root: str | Path = "runs",
    chunk_size: int | None = None,
    overrides: dict | None = None,
) -> Path:
    """Execute a bounded or explicit Adam budget with validation-only selection."""
    if chunk_size is not None and (
        isinstance(chunk_size, bool)
        or not isinstance(chunk_size, int)
        or chunk_size <= 0
    ):
        raise TrainingError("chunk_size must be a positive integer or None")
    resolved = resolve_run_config(
        case_path,
        training_path,
        seed=seed,
        device=device,
        dtype=dtype,
        run_id=run_id,
        smoke_path=smoke_path,
        overrides=overrides,
    )
    resolved["execution_settings"] = {"chunk_size": chunk_size}
    case, training = resolved["case"], resolved["training"]
    resume_checkpoint, ancestry = None, None
    if resume_from is not None:
        resume_checkpoint, previous, source_path = load_training_checkpoint(
            resume_from, output_root
        )
        ancestry = {
            "source_run_id": resume_from,
            "checkpoint_sha256": _hash(source_path / "checkpoint.pt"),
            "resume_step": resume_checkpoint["step"],
            **_resume_compatible(previous, resolved, resume_checkpoint),
        }
        resolved["resume_ancestry"] = ancestry
    # Validate backend/dtype before claiming output ownership. No fallback cast.
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    model = ShallowWaterPINN(case, training, device=device, dtype=dtype)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=training["optimizer"]["learning_rate"]
    )
    if resume_checkpoint is None:
        samples = generate_training_samples(case, training, seed)
        validation = generate_validation_samples(case, training, seed)
    else:
        samples, validation = (
            load_samples(source_path / "sampling.npz"),
            load_samples(source_path / "validation.npz"),
        )
        model.load_state_dict(resume_checkpoint["latest_model_state"], strict=True)
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        _restore_rng(resume_checkpoint["rng_state"], device)
    weights = {
        name: float(training["weighting"][f"lambda_{name}"]) for name in COMPONENTS
    }
    selection_weights = weights.copy()
    if resume_checkpoint is not None:
        weights = resume_checkpoint["weights"].copy()
    stage_metadata = {
        "stage": "pinn_training",
        "data_origin": "physics_only_generated_coordinates",
        "interior_labels": False,
        "lambda_data": 0.0,
        "selection_rule": SELECTION,
        "selection_weights": selection_weights,
        "selection_note": (
            "minimum separate validation PDE+IC+BC with fixed initial configured "
            "weights; no analytical interior labels"
        ),
        "chunk_size": chunk_size,
        "resume_ancestry": ancestry,
        "source_sha256": {
            f"src/hydro_pinn/{name}": _hash(Path(__file__).with_name(name))
            for name in ("model.py", "sampling.py", "losses.py", "train.py", "cli.py")
        },
    }
    with RunRecord.create(
        output_root, run_id, resolved, metadata=stage_metadata
    ) as run:
        started = time.perf_counter()
        save_samples(run.path / "sampling.npz", samples)
        save_samples(run.path / "validation.npz", validation)
        partition_hashes = {
            name: _hash(run.path / name) for name in ("sampling.npz", "validation.npz")
        }
        step = 0 if resume_checkpoint is None else resume_checkpoint["step"]
        best_score, best_step, best_state = (
            math.inf,
            step,
            _cpu_tree(model.state_dict()),
        )
        if resume_checkpoint is not None:
            best_score, best_step = (
                resume_checkpoint["selected_validation_score"],
                resume_checkpoint["selected_step"],
            )
            best_state = resume_checkpoint["selected_model_state"]
        history: list[dict] = []
        lbfgs_info: dict = {"enabled": training["lbfgs"]["enabled"], "closure_calls": 0}

        def checkpoint() -> dict:
            return {
                "schema_version": 1,
                "run_id": run_id,
                "resolved_config": resolved,
                "resolved_config_sha256": _config_hash(resolved),
                "scales": case["scales"],
                "step": step,
                "latest_model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": None,
                "weights": weights.copy(),
                "weight_update_state": {
                    "last_completed_adam_step": step,
                    "next_update_before_step": (
                        (step // training["weighting"]["update_interval"]) + 1
                    )
                    * training["weighting"]["update_interval"],
                },
                "sampling_references": {name: name for name in partition_hashes},
                "partition_hashes": partition_hashes,
                "rng_state": _rng_state(device),
                "selection_rule": SELECTION,
                "selection_weights": selection_weights,
                "selected_model_state": best_state,
                "selected_step": best_step,
                "selected_validation_score": best_score,
                "lbfgs_info": lbfgs_info,
                "lbfgs_completed": bool(lbfgs_info.get("completed", False)),
            }

        def save_history() -> None:
            with (run.path / "history.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
                writer.writeheader()
                writer.writerows(history)

        try:
            initial_score = _validation(
                model, validation, selection_weights, chunk_size
            )
            if initial_score < best_score:
                best_score, best_step, best_state = (
                    initial_score,
                    step,
                    _cpu_tree(model.state_dict()),
                )
            _save_checkpoint(run.path / "checkpoint.pt", checkpoint())
            for attempted_step in range(step + 1, training["optimizer"]["steps"] + 1):
                values, gradients, norms = {}, {}, {}
                for name in COMPONENTS:
                    values[name], gradients[name] = component_gradients(
                        model, samples, name, chunk_size
                    )
                    norms[name] = gradient_rms(gradients[name], model.parameters())
                updated_weights = update_adaptive_weights(
                    weights, norms, attempted_step, training["weighting"]
                )
                weights_changed = updated_weights != weights
                weights = updated_weights
                optimizer.zero_grad(set_to_none=True)
                for index, parameter in enumerate(model.parameters()):
                    parameter.grad = sum(
                        weights[name] * gradients[name][index] for name in COMPONENTS
                    )
                    if not bool(torch.isfinite(parameter.grad).all()):
                        raise TrainingError("nonfinite weighted optimizer gradient")
                optimizer.step()
                _finite_model(model)
                score = _validation(model, validation, selection_weights, chunk_size)
                step = attempted_step
                if score < best_score:
                    best_score, best_step, best_state = (
                        score,
                        step,
                        _cpu_tree(model.state_dict()),
                    )
                history.append(
                    {
                        "phase": "adam",
                        "step": step,
                        **{f"loss_{name}": values[name] for name in COMPONENTS},
                        "loss_total": sum(
                            weights[name] * values[name] for name in COMPONENTS
                        ),
                        **{f"lambda_{name}": weights[name] for name in COMPONENTS},
                        **{f"grad_{name}": norms[name] for name in COMPONENTS},
                        "weights_updated": weights_changed,
                        "validation_score": score,
                        "selected_step": best_step,
                        "elapsed_s": time.perf_counter() - started,
                        "stop_reason": "",
                    }
                )
                _save_checkpoint(run.path / "checkpoint.pt", checkpoint())
                save_history()
            if training["lbfgs"]["enabled"]:
                lbfgs = torch.optim.LBFGS(
                    model.parameters(),
                    max_iter=training["lbfgs"]["max_iterations"],
                    history_size=training["lbfgs"]["history_size"],
                    line_search_fn=None,
                )
                lbfgs_info.update(
                    max_iterations=training["lbfgs"]["max_iterations"],
                    max_evaluations=lbfgs.defaults["max_eval"],
                    history_size=training["lbfgs"]["history_size"],
                    line_search=None,
                    tolerance_grad=lbfgs.defaults["tolerance_grad"],
                    tolerance_change=lbfgs.defaults["tolerance_change"],
                    weights_frozen=weights.copy(),
                )
                lbfgs_started = time.perf_counter()

                def closure() -> torch.Tensor:
                    lbfgs_info["closure_calls"] += 1
                    lbfgs.zero_grad(set_to_none=True)
                    total = 0.0
                    for name in COMPONENTS:
                        for fraction, loss in iter_component_losses(
                            model, samples, name, chunk_size
                        ):
                            weighted = weights[name] * fraction * loss
                            if not bool(torch.isfinite(weighted)):
                                raise TrainingError("nonfinite LBFGS closure loss")
                            weighted.backward()
                            total += float(weighted.detach())
                    gradient_rms(
                        [p.grad for p in model.parameters()], model.parameters()
                    )
                    return next(model.parameters()).new_tensor(total)

                lbfgs.step(closure)
                _finite_model(model)
                state = lbfgs.state[next(iter(model.parameters()))]
                lbfgs_info.update(
                    completed=True,
                    iterations=state.get("n_iter", 0),
                    function_evaluations=state.get("func_evals", 0),
                    elapsed_s=time.perf_counter() - lbfgs_started,
                    stopping_reason="PyTorch LBFGS returned; convergence not inferred",
                    optimizer_state=_cpu_tree(lbfgs.state_dict()),
                )
                score = _validation(model, validation, selection_weights, chunk_size)
                if score < best_score:
                    best_score, best_step, best_state = (
                        score,
                        step,
                        _cpu_tree(model.state_dict()),
                    )
                _save_checkpoint(run.path / "checkpoint.pt", checkpoint())
            if smoke_path is not None:
                # Probe the held selected state before any torch.load. Reusing
                # this model and restoring its latest state consumes no RNG.
                latest_state = _cpu_tree(model.state_dict())
                model.load_state_dict(best_state, strict=True)
                evaluation = resolved["evaluation"]
                a, b = evaluation["grid"]["domain_m"]
                count = evaluation["grid"]["cells"]
                x = a + (np.arange(count) + 0.5) * (b - a) / count
                times = np.array(evaluation["times_s"], dtype=np.float64)
                xx, tt = np.meshgrid(x, times)
                parameter = next(model.parameters())
                coordinates = torch.tensor(
                    np.column_stack((xx.ravel(), tt.ravel())),
                    dtype=parameter.dtype,
                    device=parameter.device,
                )
                with torch.no_grad():
                    probe = model.physical_state(coordinates).cpu().numpy()
                probe = probe.reshape(len(times), count, 2)
                with (run.path / "reload_probe.npz").open("xb") as handle:
                    np.savez(
                        handle,
                        x_m=x,
                        t_s=times,
                        h_m=probe[..., 0],
                        q_m2_s=probe[..., 1],
                    )
                model.load_state_dict(latest_state, strict=True)
            reason = "finite Adam budget exhausted; convergence not established"
            if training["lbfgs"]["enabled"]:
                reason += "; configured LBFGS returned"
            if history:
                history[-1]["stop_reason"] = reason
            save_history()
            write_json_atomic(
                run.path / "training_diagnostics.json",
                {
                    "completed_adam_steps": step,
                    "selected_step": best_step,
                    "selected_validation_score": best_score,
                    "selection_rule": SELECTION,
                    "selection_weights": selection_weights,
                    "lbfgs": {
                        key: value
                        for key, value in lbfgs_info.items()
                        if key != "optimizer_state"
                    },
                    "component_normalization": (
                        "mean over points of summed h/q or residual squares"
                    ),
                    "history_loss_timing": (
                        "Adam components before update; validation after update"
                    ),
                    "interior_labels": False,
                    "reload_probe": (
                        "reload_probe.npz: held selected model before disk reload; "
                        "physical point values on smoke evaluation only"
                    )
                    if smoke_path is not None
                    else None,
                    "stopping_reason": reason,
                },
            )
            run.complete(
                stopping_reason=reason,
                iterations=step,
                function_evaluations=lbfgs_info["closure_calls"],
                elapsed_timings_s={"training_and_save": time.perf_counter() - started},
            )
        except BaseException as exc:
            reason = f"{type(exc).__name__}: {exc}"
            history.append(
                {
                    "phase": "failure",
                    "step": step,
                    "stop_reason": reason,
                    "elapsed_s": time.perf_counter() - started,
                }
            )
            save_history()
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                run.interrupt(
                    reason,
                    last_valid_state="checkpoint.pt"
                    if (run.path / "checkpoint.pt").exists()
                    else None,
                )
            else:
                run.fail(
                    reason,
                    stage="pinn_training",
                    step=step,
                    last_valid_state="checkpoint.pt"
                    if (run.path / "checkpoint.pt").exists()
                    else None,
                )
            raise
    return run.path


def predict_run(
    training_run: str,
    evaluation_path: str | Path,
    run_id: str,
    output_root: str | Path = "runs",
) -> Path:
    """Reload the selected trusted model and save pointwise physical h/q."""
    checkpoint, source, path = load_training_checkpoint(
        training_run, output_root, require_completed=True
    )
    evaluation_input = load_config(evaluation_path)
    evaluation = (
        evaluation_input["evaluation"]
        if evaluation_input["config_type"] == "smoke"
        else evaluation_input
    )
    if evaluation["config_type"] != "evaluation":
        raise TrainingError(
            "prediction requires evaluation or typed smoke configuration"
        )
    if (
        evaluation["grid"]["domain_m"] != source["case"]["domain_m"]
        or evaluation["times_s"][-1] > source["case"]["final_time_s"]
    ):
        raise TrainingError("prediction evaluation is incompatible with physical case")
    request = {**source["request"], "run_id": run_id}
    resolved = {
        "case": source["case"],
        "training": source["training"],
        "evaluation": evaluation,
        "request": request,
        "provenance": {
            "input_files": [
                {
                    "role": "prediction_evaluation",
                    "path": str(evaluation_path),
                    "sha256": _hash(Path(evaluation_path)),
                }
            ],
            "purpose": evaluation["partition"],
        },
        "prediction_source": {
            "training_run_id": training_run,
            "checkpoint_sha256": _hash(path / "checkpoint.pt"),
            "training_resolved_config_sha256": checkpoint["resolved_config_sha256"],
            "selected_step": checkpoint["selected_step"],
        },
    }
    model = ShallowWaterPINN(
        source["case"],
        source["training"],
        device=request["device"],
        dtype=request["dtype"],
    )
    model.load_state_dict(checkpoint["selected_model_state"], strict=True)
    model.eval()
    _finite_model(model)
    metadata = {
        "stage": "pinn_prediction",
        "state_location": "point_value",
        "data_origin": "network_approximation",
        **resolved["prediction_source"],
        "array_units": {"x_m": "m", "t_s": "s", "h_m": "m", "q_m2_s": "m^2/s"},
        "numerical_floor_nd": source["training"]["network"]["depth_floor_nd"],
    }
    with RunRecord.create(output_root, run_id, resolved, metadata=metadata) as run:
        started = time.perf_counter()
        a, b = evaluation["grid"]["domain_m"]
        count = evaluation["grid"]["cells"]
        x = a + (np.arange(count) + 0.5) * (b - a) / count
        times = np.array(evaluation["times_s"], dtype=np.float64)
        xx, tt = np.meshgrid(x, times)
        parameter = next(model.parameters())
        coordinates = torch.tensor(
            np.column_stack((xx.ravel(), tt.ravel())),
            dtype=parameter.dtype,
            device=parameter.device,
        )
        with torch.no_grad():
            state = model.physical_state(coordinates)
        if not bool(torch.isfinite(state).all()) or bool((state[:, 0] <= 0).any()):
            raise TrainingError("nonfinite or nonpositive reloaded physical prediction")
        state_np = state.cpu().numpy().reshape(len(times), count, 2)
        # This run directory is exclusively owned and this new array is not replaced.
        with (run.path / "outputs.npz").open("xb") as handle:
            np.savez(
                handle, x_m=x, t_s=times, h_m=state_np[..., 0], q_m2_s=state_np[..., 1]
            )
        run.complete(
            stopping_reason=(
                "selected checkpoint reloaded; physical point predictions saved"
            ),
            elapsed_timings_s={"prediction_and_save": time.perf_counter() - started},
            function_evaluations=1,
        )
    return run.path
