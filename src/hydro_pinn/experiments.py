"""Execute a frozen matrix with inspectable, immutable run attempts.

The mutable manifest records workflow progress. Individual runs retain their
own inputs, status, arrays and checkpoint identity. A terminal failed/interrupted
training attempt can continue only through an explicit new-ID resume request.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import socket
import time
import zipfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from hydro_pinn.lax_wendroff import NumericalRunFailed, generate_solver_run
from hydro_pinn.protocol import load_protocol
from hydro_pinn.records import validate_run_id, write_json_atomic
from hydro_pinn.references import generate_reference_run
from hydro_pinn.sampling import load_samples
from hydro_pinn.train import load_training_checkpoint, predict_run, train_run

ROOT = Path(__file__).resolve().parents[2]
BINDING = "protocol_binding.json"


class MatrixError(ValueError):
    """A frozen execution, resume request, or existing artifact is incompatible."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise MatrixError(f"missing or unsafe matrix artifact: {path}")
    document = json.loads(path.read_text())
    if not isinstance(document, dict):
        raise MatrixError(f"expected a JSON object: {path}")
    return document


def _observed_state(path: Path) -> dict:
    """Retain a recorder fact even when complete artifact verification fails."""
    try:
        return {"record_state_observed": _read(path / "status.json").get("state")}
    except (ValueError, OSError) as exc:
        return {
            "record_state_observed": None,
            "record_status_read_error": f"{type(exc).__name__}: {exc}",
        }


def _path(root: Path, value: str | Path) -> Path:
    result = Path(value)
    if not result.is_absolute():
        result = root / result
    if ".." in result.parts or any(p.is_symlink() for p in (result, *result.parents)):
        raise MatrixError("matrix paths must not traverse parents or symlinks")
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise MatrixError("matrix paths must remain inside project root") from exc
    return result


def execution_plan(
    protocol: dict,
    *,
    namespace: str | None = None,
    output_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict:
    """Derive destinations/IDs for the same frozen science without editing it.

    The map reserves both core and planned auxiliary IDs. Actual checkpoint
    bindings remain runtime facts of this particular execution.
    """
    if namespace is None and (output_root is not None or manifest_path is not None):
        raise MatrixError("execution destination overrides require a fresh namespace")
    if namespace is not None:
        validate_run_id(namespace)
    frozen_ids = [row["run_id"] for row in protocol["matrix"]]
    frozen_ids.extend(
        row["run_id"]
        for row in protocol.get("conservation", {}).get("auxiliary_records", [])
    )
    if len(frozen_ids) != len(set(frozen_ids)):
        raise MatrixError("duplicate frozen core/auxiliary run IDs")
    mapped = {
        run_id: f"{namespace}-{run_id}" if namespace else run_id
        for run_id in frozen_ids
    }
    for run_id in mapped.values():
        validate_run_id(run_id)
    default_manifest = Path(protocol["paths"]["manifest"])
    if namespace is not None:
        default_manifest = default_manifest.with_name(
            f"{default_manifest.stem}-{namespace}{default_manifest.suffix}"
        )
    return {
        "namespace": namespace,
        "output_root": str(output_root)
        if output_root is not None
        else protocol["paths"]["output_root"],
        "manifest": str(manifest_path)
        if manifest_path is not None
        else str(default_manifest),
        "parent_protocol_sha256": protocol["protocol_sha256"],
        "run_ids": mapped,
    }


@contextmanager
def _lock(path: Path):
    """An OS advisory lock survives no process; its owner log is retained."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                if path.stat().st_size == 0:
                    handle.write(" ")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as exc:
            raise MatrixError(f"matrix already has a live writer: {path}") from exc
        owner = {
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "acquired_at_utc": _now(),
            "state": "held",
        }
        handle.seek(0)
        handle.truncate()
        json.dump(owner, handle, allow_nan=False)
        handle.flush()
        try:
            yield owner
        finally:
            owner.update(state="released", released_at_utc=_now())
            handle.seek(0)
            handle.truncate()
            json.dump(owner, handle, allow_nan=False)
            handle.flush()
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _binding(
    protocol: dict, stage: dict, run_id: str, execution: dict | None = None
) -> dict:
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol["protocol_sha256"],
        "purpose": protocol["purpose"],
        "stage_id": stage["stage_id"],
        "frozen_run_id": stage["run_id"],
        "actual_run_id": run_id,
        "execution": execution or execution_plan(protocol),
        "scientific_source_sha256": protocol["scientific_source_sha256"],
        "final_evaluation_sha256": protocol["inputs"]["evaluation"]["sha256"],
    }


def _bind(
    path: Path, protocol: dict, stage: dict, run_id: str, execution: dict | None = None
) -> None:
    expected = _binding(protocol, stage, run_id, execution)
    if (path / BINDING).exists():
        if _read(path / BINDING) != expected:
            raise MatrixError("run protocol binding mismatch")
    else:
        write_json_atomic(path / BINDING, expected)


def _software(metadata: dict, protocol: dict) -> None:
    expected = protocol["provenance"]["software"]
    actual = metadata["software"]
    for key in ("python", "implementation"):
        if actual[key] != expected[key]:
            raise MatrixError(f"run software {key} differs from frozen environment")
    for name in ("torch", "numpy", "scipy"):
        if actual["packages"][name] != expected["packages"][name]:
            raise MatrixError(f"run {name} version differs from frozen environment")
    environment = protocol.get("environment", {})
    if environment.get("torch_git_version") is not None:
        if metadata["device"].get("git_version") != environment["torch_git_version"]:
            raise MatrixError("run PyTorch build differs from frozen environment")


def _partition_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for key, value in sorted(load_samples(path).items()):
        digest.update(key.encode())
        digest.update(str(value.shape).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _verify_record_contents(
    protocol: dict,
    stage: dict,
    run_id: str,
    root: Path,
    *,
    training_source_id: str | None = None,
    require_completed: bool = False,
    require_binding: bool = True,
    execution: dict | None = None,
) -> dict:
    """Check an actual record against frozen inputs before reuse or resume."""
    validate_run_id(run_id)
    execution = execution or execution_plan(protocol)
    path = _path(root, Path(execution["output_root"]) / run_id)
    status = _read(path / "status.json")
    metadata = _read(path / "metadata.json")
    resolved = _read(path / "resolved_config.json")
    if status.get("run_id") != run_id or metadata.get("run_id") != run_id:
        raise MatrixError("run artifact identity mismatch")
    if metadata["resolved_config_sha256"] != _sha(path / "resolved_config.json"):
        raise MatrixError("run resolved configuration SHA256 mismatch")
    if resolved["request"]["run_id"] != run_id:
        raise MatrixError("resolved request run identity mismatch")
    if resolved["case"] != protocol["resolved"]["cases"][stage["case_id"]]:
        raise MatrixError("run physical case differs from frozen case")
    _software(metadata, protocol)
    if require_binding and _read(path / BINDING) != _binding(
        protocol, stage, run_id, execution
    ):
        raise MatrixError("run protocol binding mismatch")
    frozen_sources = protocol["scientific_source_sha256"]
    for name, digest in metadata["stage_metadata"].get("source_sha256", {}).items():
        if name in frozen_sources and digest != frozen_sources[name]:
            raise MatrixError(f"run scientific source differs: {name}")
    if require_completed and status["state"] != "completed":
        raise MatrixError("completed stage points to a noncompleted record")
    if stage["stage"] in {"train", "predict"}:
        settings = protocol["settings"]
        for key, expected in (
            ("seed", stage["seed"]),
            ("device", settings["device"]),
            ("dtype", settings["dtype"]),
        ):
            if resolved["request"][key] != expected:
                raise MatrixError(f"run request {key} differs from frozen settings")
        if resolved["training"] != protocol["resolved"]["training"][stage["variant"]]:
            raise MatrixError("run training settings differ from frozen variant")
    if stage["stage"] == "train":
        if resolved.get("evaluation") is not None:
            raise MatrixError(
                "training unexpectedly contains final evaluation coordinates"
            )
        if (
            resolved["execution_settings"]["chunk_size"]
            != protocol["settings"]["chunk_size"]
        ):
            raise MatrixError("run chunks differ from frozen settings")
        required_sources = {
            f"src/hydro_pinn/{name}.py"
            for name in ("model", "sampling", "losses", "train")
        }
        if not required_sources <= metadata["stage_metadata"]["source_sha256"].keys():
            raise MatrixError("training source hashes missing")
        expected_inputs = (
            protocol["inputs"]["cases"][stage["case_id"]],
            protocol["inputs"]["training"][stage["variant"]],
        )
    elif stage["stage"] == "predict":
        expected_inputs = (protocol["inputs"]["evaluation"],)
        if resolved["prediction_source"]["training_run_id"] != training_source_id:
            raise MatrixError("prediction uses a different actual training attempt")
        training_path = _path(root, Path(execution["output_root"]) / training_source_id)
        if resolved["prediction_source"]["checkpoint_sha256"] != _sha(
            training_path / "checkpoint.pt"
        ):
            raise MatrixError("prediction checkpoint identity differs from training")
    else:
        expected_inputs = (
            protocol["inputs"]["cases"][stage["case_id"]],
            protocol["inputs"]["evaluation"],
        )
    input_hashes = {
        entry["sha256"]
        for entry in resolved.get("provenance", {}).get("input_files", [])
    }
    if not all(entry["sha256"] in input_hashes for entry in expected_inputs):
        raise MatrixError("run input hashes do not match frozen stage inputs")
    if stage["stage"] != "train":
        if resolved["evaluation"] != protocol["resolved"]["evaluation"]:
            raise MatrixError("run uses a different final evaluation configuration")
    if stage["stage"] == "solve" and resolved["solver"]["cells"] != stage["cells"]:
        raise MatrixError("solver grid differs from frozen grid")
    info = {
        "run_id": run_id,
        "run_path": str(path.relative_to(root)),
        "record_state": status["state"],
        "stop_reason": status["stopping_reason"],
        "diagnostics": status["diagnostics"],
        "metadata_path": str((path / "metadata.json").relative_to(root)),
        "software": metadata["software"],
        "device": metadata["device"],
        "source_sha256": metadata["stage_metadata"].get("source_sha256", {}),
        "resolved_config_sha256": metadata["resolved_config_sha256"],
    }
    if status["state"] == "completed" and stage["stage"] != "train":
        if (path / "outputs.npz").is_symlink():
            raise MatrixError("saved output must be a regular file inside its run")
        with np.load(path / "outputs.npz", allow_pickle=False) as arrays:
            evaluation = protocol["resolved"]["evaluation"]
            cells = stage.get("cells", evaluation["grid"]["cells"])
            times = np.array(evaluation["times_s"], dtype=np.float64)
            if not np.array_equal(arrays["t_s"], times):
                raise MatrixError("saved output times differ from frozen evaluation")
            for field in ("h_m", "q_m2_s"):
                if (
                    arrays[field].shape != (len(times), cells)
                    or not np.isfinite(arrays[field]).all()
                ):
                    raise MatrixError(f"invalid completed array {field}")
            if arrays["x_m"].shape != (cells,) or not np.isfinite(arrays["x_m"]).all():
                raise MatrixError("invalid completed spatial axis")
            left, right = protocol["resolved"]["cases"][stage["case_id"]]["domain_m"]
            expected_x = left + (np.arange(cells) + 0.5) * (right - left) / cells
            if not np.allclose(arrays["x_m"], expected_x, rtol=0, atol=1e-13):
                raise MatrixError("saved spatial coordinates differ from frozen grid")
            if (arrays["h_m"] < 0).any():
                raise MatrixError("negative completed depth")
            if stage["stage"] == "solve" and arrays["cell_edges_m"].shape != (
                cells + 1,
            ):
                raise MatrixError("invalid native cell edges")
        info["output_path"] = str((path / "outputs.npz").relative_to(root))
    if stage["stage"] == "train" and status["state"] in {
        "completed",
        "failed",
        "interrupted",
    }:
        if (path / "checkpoint.pt").exists():
            checkpoint, _, _ = load_training_checkpoint(
                run_id, path.parent, require_completed=require_completed
            )
            budget = protocol["resolved"]["training"][stage["variant"]]["optimizer"][
                "steps"
            ]
            if status["state"] == "completed" and checkpoint["step"] != budget:
                raise MatrixError("completed training did not exhaust frozen budget")
            info.update(
                checkpoint_sha256=_sha(path / "checkpoint.pt"),
                completed_adam_steps=checkpoint["step"],
                selected_step=checkpoint["selected_step"],
                selected_validation_score=checkpoint["selected_validation_score"],
                selection_rule=checkpoint["selection_rule"],
                partition_content_sha256={
                    name: _partition_hash(path / name)
                    for name in ("sampling.npz", "validation.npz")
                },
            )
        elif status["state"] == "completed":
            raise MatrixError("completed training lacks a checkpoint")
    info["files_sha256"] = {
        p.name: _sha(p) for p in sorted(path.iterdir()) if p.is_file()
    }
    return info


def _verify_record(*args, **kwargs) -> dict:
    try:
        return _verify_record_contents(*args, **kwargs)
    except (KeyError, TypeError, EOFError, zipfile.BadZipFile) as exc:
        raise MatrixError(f"invalid run artifact structure: {exc}") from exc


def _new_manifest(
    protocol: dict, protocol_path: Path, owner: dict, execution: dict
) -> dict:
    return {
        "schema_version": 1,
        "purpose": protocol["purpose"],
        "execution": copy.deepcopy(execution),
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol["protocol_sha256"],
        "protocol_path": str(protocol_path),
        "created_at_utc": _now(),
        "updated_at_utc": _now(),
        "workflow_status": "running",
        "owner": owner.copy(),
        "entries": [
            {
                "frozen_stage": copy.deepcopy(stage),
                "stage_id": stage["stage_id"],
                "state": "not_started",
                "actual_run_id": None,
                "attempts": [],
                "progress": None,
                "failure_category": None,
            }
            for stage in protocol["matrix"]
        ],
        "scientific_status": {
            "core_comparison": "pending",
            "E1_classical_comparator": "pending",
            "final_accuracy": "not_evaluated",
            "failures": [],
        },
    }


def _save(path: Path, manifest: dict) -> None:
    manifest["updated_at_utc"] = _now()
    write_json_atomic(path, manifest, overwrite=path.exists())


def _stage_info(
    protocol,
    stage,
    entry,
    entries,
    root,
    *,
    require_completed=False,
    require_binding=True,
    execution=None,
):
    source = None
    if stage["stage"] == "predict":
        source = entries[stage["training_stage_id"]]["actual_run_id"]
    return _verify_record(
        protocol,
        stage,
        entry["actual_run_id"],
        root,
        training_source_id=source,
        require_completed=require_completed,
        require_binding=require_binding,
        execution=execution,
    )


def _reconcile(protocol: dict, manifest: dict, root: Path, execution: dict) -> None:
    if manifest.get("protocol_sha256") != protocol["protocol_sha256"]:
        raise MatrixError("manifest belongs to a different frozen protocol")
    if manifest.get("execution") != execution:
        raise MatrixError("manifest execution destinations/namespace differ")
    entries = {entry["stage_id"]: entry for entry in manifest["entries"]}
    if len(entries) != len(protocol["matrix"]):
        raise MatrixError("manifest matrix entry count mismatch")
    for stage in protocol["matrix"]:
        entry = entries.get(stage["stage_id"])
        if entry is None or entry["frozen_stage"] != stage:
            raise MatrixError("manifest immutable stage identity mismatch")
        if entry["actual_run_id"] is None:
            continue
        path = _path(root, Path(execution["output_root"]) / entry["actual_run_id"])
        if entry["state"] == "completed":
            info = _stage_info(
                protocol,
                stage,
                entry,
                entries,
                root,
                require_completed=True,
                execution=execution,
            )
            for name, digest in entry["result"]["files_sha256"].items():
                if info["files_sha256"].get(name) != digest:
                    raise MatrixError(f"completed run artifact changed: {path / name}")
        elif entry["state"] == "running":
            if not path.exists():
                entry.update(
                    state="blocked",
                    failure_category="missing_attempt_record",
                    reason=(
                        "writer ended before run record creation; "
                        "explicit technical recovery required"
                    ),
                )
                continue
            info = _stage_info(
                protocol,
                stage,
                entry,
                entries,
                root,
                require_binding=False,
                execution=execution,
            )
            if info["record_state"] == "running":
                entry.update(
                    state="blocked",
                    failure_category="orphan_running",
                    reason=(
                        "prior writer ended; running source preserved; "
                        "terminal status required for compatible resume"
                    ),
                )
                entry["attempts"][-1]["state"] = "orphan_running"
            else:
                _bind(path, protocol, stage, entry["actual_run_id"], execution)
                info = _stage_info(
                    protocol, stage, entry, entries, root, execution=execution
                )
                entry.update(
                    state=info["record_state"],
                    result=info,
                    failure_category=info["diagnostics"].get("category"),
                )
                entry["attempts"][-1].update(
                    state=info["record_state"],
                    result=info,
                    reconciled_after_writer_exit=True,
                )


def _summary(manifest: dict) -> None:
    entries = manifest["entries"]
    completed = [entry for entry in entries if entry["state"] == "completed"]
    failures = [entry for entry in entries if entry["state"] != "completed"]
    numerical = [
        entry for entry in failures if entry["failure_category"] == "numerical_failure"
    ]
    if not failures:
        manifest["workflow_status"] = "completed"
    elif len(numerical) == len(failures):
        manifest["workflow_status"] = "completed_with_diagnostics"
    else:
        manifest["workflow_status"] = "incomplete"
    e1_solver = any(
        entry["frozen_stage"]["case_id"] == "E1"
        and entry["frozen_stage"]["stage"] == "solve"
        for entry in completed
    )
    references_ok = (
        sum(entry["frozen_stage"]["stage"] == "reference" for entry in completed) == 2
    )
    predictions_ok = (
        sum(entry["frozen_stage"]["stage"] == "predict" for entry in completed) == 7
    )
    manifest["scientific_status"] = {
        "core_comparison": "available_for_evaluation"
        if e1_solver and references_ok and predictions_ok
        else "incomplete",
        "E1_classical_comparator": "available" if e1_solver else "missing",
        "final_accuracy": "not_evaluated",
        "failures": [
            {
                "stage_id": entry["stage_id"],
                "state": entry["state"],
                "category": entry["failure_category"],
                "reason": entry.get("reason"),
            }
            for entry in failures
        ],
    }


def run_matrix(
    protocol_path: str | Path,
    *,
    project_root: str | Path | None = None,
    resume_stage: str | None = None,
    resume_from: str | None = None,
    resume_run_id: str | None = None,
    execution_namespace: str | None = None,
    manifest_path: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict:
    """Execute/reconcile frozen stages; explicit terminal resumes use fresh IDs."""
    root = Path(project_root).absolute() if project_root else ROOT
    protocol_path = _path(root, protocol_path)
    protocol = load_protocol(protocol_path, project_root=root)
    resume = (resume_stage, resume_from, resume_run_id)
    if any(value is not None for value in resume) and not all(
        value is not None for value in resume
    ):
        raise MatrixError(
            "resume requires --resume-stage, --resume-from and --resume-run-id together"
        )
    execution = execution_plan(
        protocol,
        namespace=execution_namespace,
        manifest_path=manifest_path,
        output_root=output_root,
    )
    manifest_path = _path(root, execution["manifest"])
    output_root = _path(root, execution["output_root"])
    execution["manifest"] = str(manifest_path.relative_to(root))
    execution["output_root"] = str(output_root.relative_to(root))
    previous_threads = torch.get_num_threads()
    with _lock(manifest_path.with_suffix(".lock")) as owner:
        manifest = (
            _read(manifest_path)
            if manifest_path.exists()
            else _new_manifest(protocol, protocol_path, owner, execution)
        )
        if (
            execution_namespace is not None
            and manifest_path.exists()
            and manifest["workflow_status"]
            in {"completed", "completed_with_diagnostics"}
        ):
            raise MatrixError(
                "completed execution namespace already exists; use a fresh namespace"
            )
        if execution_namespace is not None and not manifest_path.exists():
            collisions = [
                run_id
                for run_id in execution["run_ids"].values()
                if (output_root / run_id).exists()
            ]
            if collisions:
                raise MatrixError(f"fresh namespace run ID collision: {collisions}")
        _reconcile(protocol, manifest, root, execution)
        entries = {entry["stage_id"]: entry for entry in manifest["entries"]}
        if resume_stage is not None:
            entry = entries.get(resume_stage)
            if entry is None or entry["frozen_stage"]["stage"] != "train":
                raise MatrixError("resume-stage must identify a frozen training stage")
            if entry["state"] == "completed":
                raise MatrixError("a completed frozen training stage cannot be resumed")
            if resume_from not in {attempt["run_id"] for attempt in entry["attempts"]}:
                raise MatrixError(
                    "resume source is not a recorded attempt of this stage"
                )
            validate_run_id(resume_run_id)
            used_ids = set(execution["run_ids"].values())
            used_ids.update(
                attempt["run_id"]
                for row in entries.values()
                for attempt in row["attempts"]
            )
            if resume_run_id in used_ids or (output_root / resume_run_id).exists():
                raise MatrixError("resume requires a fresh unique run ID")
            stage = entry["frozen_stage"]
            info = _verify_record(
                protocol, stage, resume_from, root, execution=execution
            )
            if info["record_state"] not in {"failed", "interrupted"}:
                raise MatrixError(
                    "resume source must be terminal failed/interrupted; "
                    "running orphan is preserved"
                )
            budget = protocol["resolved"]["training"][stage["variant"]]["optimizer"][
                "steps"
            ]
            if info.get("completed_adam_steps", budget) >= budget:
                raise MatrixError("resume cannot extend the frozen optimizer budget")
        manifest.update(owner=owner.copy(), workflow_status="running")
        _save(manifest_path, manifest)
        torch.set_num_threads(protocol["settings"]["cpu_threads"])
        try:
            for stage in protocol["matrix"]:
                # Core hashes/inputs are checked between stages; additive CLI
                # changes do not invalidate the frozen scientific implementation.
                load_protocol(protocol_path, project_root=root)
                entry = entries[stage["stage_id"]]
                is_resume = stage["stage_id"] == resume_stage
                if entry["state"] == "completed":
                    continue
                unavailable = [
                    key
                    for key in stage["depends_on"]
                    if entries[key]["state"] != "completed"
                ]
                if unavailable:
                    entry.update(
                        state="blocked",
                        failure_category="dependency_failure",
                        reason=f"required completed stages unavailable: {unavailable}",
                    )
                    _save(manifest_path, manifest)
                    continue
                if entry["state"] in {"failed", "interrupted"} and not is_resume:
                    continue
                if (
                    entry["state"] == "blocked"
                    and entry["failure_category"] != "dependency_failure"
                    and not is_resume
                ):
                    continue
                actual_id = (
                    resume_run_id
                    if is_resume
                    else execution["run_ids"][stage["run_id"]]
                )
                path = output_root / actual_id
                if path.exists():
                    raise MatrixError(
                        "unowned existing run cannot be replaced or adopted: "
                        f"{actual_id}"
                    )
                attempt = {
                    "run_id": actual_id,
                    "state": "running",
                    "started_at_utc": _now(),
                    "resume_from": resume_from if is_resume else None,
                    "protocol_sha256": protocol["protocol_sha256"],
                }
                entry["attempts"].append(attempt)
                entry.update(
                    state="running",
                    actual_run_id=actual_id,
                    progress=None,
                    failure_category=None,
                    reason=None,
                )
                started = time.perf_counter()
                _save(manifest_path, manifest)

                def observe(
                    event,
                    current=entry,
                    current_stage=stage,
                    current_path=path,
                    step_started=started,
                ):
                    budget = protocol["resolved"]["training"][current_stage["variant"]][
                        "optimizer"
                    ]["steps"]
                    if (
                        event["step"] != 1
                        and event["step"] % 100
                        and event["step"] != budget
                    ):
                        return
                    _bind(
                        current_path,
                        protocol,
                        current_stage,
                        current["actual_run_id"],
                        execution,
                    )
                    current["progress"] = {
                        **event,
                        "total_adam_steps": budget,
                        "stage_elapsed_s": time.perf_counter() - step_started,
                        "checkpoint_manifest": _read(
                            current_path / "checkpoint_manifest.json"
                        ),
                    }
                    _save(manifest_path, manifest)
                    print(
                        json.dumps(
                            {
                                "stage_id": current_stage["stage_id"],
                                "run_id": current["actual_run_id"],
                                "step": event["step"],
                                "total_steps": budget,
                            },
                            allow_nan=False,
                        ),
                        flush=True,
                    )

                try:
                    case_path = _path(
                        root, protocol["inputs"]["cases"][stage["case_id"]]["path"]
                    )
                    evaluation = _path(root, protocol["inputs"]["evaluation"]["path"])
                    if stage["stage"] == "reference":
                        generate_reference_run(
                            case_path, evaluation, actual_id, output_root
                        )
                    elif stage["stage"] == "solve":
                        generate_solver_run(
                            case_path,
                            evaluation,
                            stage["cells"],
                            actual_id,
                            output_root,
                        )
                    elif stage["stage"] == "train":
                        settings = protocol["settings"]
                        train_run(
                            case_path,
                            _path(
                                root,
                                protocol["inputs"]["training"][stage["variant"]][
                                    "path"
                                ],
                            ),
                            stage["seed"],
                            settings["device"],
                            settings["dtype"],
                            actual_id,
                            output_root=output_root,
                            chunk_size=settings["chunk_size"],
                            resume_from=resume_from if is_resume else None,
                            step_observer=observe,
                            synchronize_steps=True,
                        )
                    elif stage["stage"] == "predict":
                        source = entries[stage["training_stage_id"]]["actual_run_id"]
                        predict_run(source, evaluation, actual_id, output_root)
                    else:
                        raise MatrixError(f"unsupported frozen stage: {stage['stage']}")
                    _bind(path, protocol, stage, actual_id, execution)
                    info = _stage_info(
                        protocol,
                        stage,
                        entry,
                        entries,
                        root,
                        require_completed=True,
                        execution=execution,
                    )
                    entry.update(state="completed", result=info)
                    attempt.update(state="completed", result=info)
                except (KeyboardInterrupt, SystemExit) as exc:
                    entry.update(
                        state="interrupted",
                        failure_category="caller_interruption",
                        reason=str(exc),
                    )
                    attempt.update(state="interrupted", reason=str(exc))
                    if path.exists():
                        attempt.update(_observed_state(path))
                        try:
                            _bind(path, protocol, stage, actual_id, execution)
                            info = _stage_info(
                                protocol,
                                stage,
                                entry,
                                entries,
                                root,
                                execution=execution,
                            )
                            entry["result"] = info
                            attempt["result"] = info
                        except (ValueError, OSError, RuntimeError) as integrity:
                            attempt["readback_error"] = (
                                f"{type(integrity).__name__}: {integrity}"
                            )
                    manifest["workflow_status"] = "interrupted"
                    _save(manifest_path, manifest)
                    raise
                except (ValueError, OSError, RuntimeError) as exc:
                    category = "execution_defect_or_environment"
                    if path.exists():
                        attempt.update(_observed_state(path))
                        try:
                            _bind(path, protocol, stage, actual_id, execution)
                            info = _stage_info(
                                protocol,
                                stage,
                                entry,
                                entries,
                                root,
                                execution=execution,
                            )
                            if (
                                isinstance(exc, NumericalRunFailed)
                                and info["record_state"] == "failed"
                                and info["diagnostics"].get("category")
                                == "numerical_failure"
                            ):
                                category = "numerical_failure"
                            entry["result"] = info
                            attempt["result"] = info
                        except (ValueError, OSError, RuntimeError) as integrity:
                            category = "artifact_verification_failure"
                            attempt["readback_error"] = (
                                f"{type(integrity).__name__}: {integrity}"
                            )
                    entry.update(
                        state="failed",
                        failure_category=category,
                        reason=f"{type(exc).__name__}: {exc}",
                    )
                    attempt.update(
                        state="failed", reason=entry["reason"], category=category
                    )
                attempt.update(
                    finished_at_utc=_now(), elapsed_s=time.perf_counter() - started
                )
                _save(manifest_path, manifest)
                print(
                    json.dumps(
                        {
                            "stage_id": stage["stage_id"],
                            "state": entry["state"],
                            "actual_run_id": actual_id,
                        },
                        allow_nan=False,
                    ),
                    flush=True,
                )
            # Pair verification uses the saved complete coordinate arrays, not
            # ZIP timestamps or a promise that equal seeds produced equal data.
            for pair in protocol["pairs"]:
                paired = [entries[key] for key in pair["train_stage_ids"]]
                if all(entry["state"] == "completed" for entry in paired):
                    if (
                        paired[0]["result"]["partition_content_sha256"]
                        != paired[1]["result"]["partition_content_sha256"]
                    ):
                        raise MatrixError(
                            "completed pair used different saved partitions"
                        )
            _summary(manifest)
            manifest["finished_at_utc"] = _now()
            _save(manifest_path, manifest)
            return manifest
        except (ValueError, OSError, RuntimeError) as exc:
            manifest.update(
                workflow_status="incomplete",
                execution_error=f"{type(exc).__name__}: {exc}",
            )
            _save(manifest_path, manifest)
            raise
        finally:
            torch.set_num_threads(previous_threads)
