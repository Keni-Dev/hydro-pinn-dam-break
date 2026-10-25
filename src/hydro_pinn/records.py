"""Local run provenance and atomic lifecycle records.

Creating a record never runs a numerical method. Callers own stage-specific
arrays/checkpoints and must finish or fail the record explicitly.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}\Z")
TERMINAL_STATES = frozenset({"completed", "failed", "interrupted"})


class RecordError(ValueError):
    """A run path, value, or lifecycle transition violates the record contract."""


def validate_run_id(run_id: str) -> str:
    """Accept identifiers, never paths, including on Windows checkouts."""
    if not isinstance(run_id, str) or not RUN_ID_PATTERN.fullmatch(run_id):
        raise RecordError(
            "run_id must be 1-96 ASCII letters/digits/underscores/hyphens, "
            "starting with a letter or digit"
        )
    if run_id.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        raise RecordError("run_id must not use a reserved Windows device name")
    return run_id


def undefined_metric(reason: str) -> dict[str, Any]:
    """Represent an undefined result with an explicit reason, not NaN or zero."""
    if not isinstance(reason, str) or not reason.strip():
        raise RecordError("an undefined value requires a nonempty reason")
    return {"value": None, "reason": reason}


def _json_bytes(data: Any) -> bytes:
    try:
        return (
            json.dumps(data, allow_nan=False, sort_keys=True, indent=2) + "\n"
        ).encode()
    except (ValueError, TypeError) as exc:
        raise RecordError(f"record must contain strict JSON values: {exc}") from exc


def _reject_symlink_path(path: Path) -> None:
    if ".." in path.parts:
        raise RecordError("output path must not contain parent traversal ('..')")
    for entry in (path, *path.parents):
        if entry.is_symlink():
            raise RecordError(f"output path must not follow a symlink: {entry}")


def write_json_atomic(path: str | Path, data: Any, *, overwrite: bool = False) -> None:
    """Write strict JSON atomically; new evidence uses exclusive creation.

    Only lifecycle metadata/status updates should request ``overwrite=True``.
    A temporary sibling makes replacement atomic on the destination filesystem.
    """
    destination = Path(path).absolute()
    _reject_symlink_path(destination)
    payload = _json_bytes(data)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        _reject_symlink_path(destination)
        if overwrite:
            os.replace(temporary, destination)
        else:
            # link fails if another writer has already claimed the destination.
            os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _command(args: list[str], cwd: Path | None = None) -> tuple[str | None, str | None]:
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if result.returncode:
        return None, result.stderr.strip() or f"exit {result.returncode}"
    return result.stdout.strip(), None


def _code_state(project_root: Path) -> dict[str, Any]:
    inside, error = _command(
        ["git", "rev-parse", "--is-inside-work-tree"], project_root
    )
    if inside != "true":
        return {
            "revision": None,
            "revision_reason": error or "not a Git worktree",
            "dirty": None,
            "dirty_reason": "Git worktree is unavailable",
        }
    revision, revision_error = _command(
        ["git", "rev-parse", "--verify", "HEAD"], project_root
    )
    changes, changes_error = _command(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        project_root,
    )
    return {
        "revision": revision,
        "revision_reason": revision_error,
        "dirty": bool(changes) if changes_error is None else None,
        "dirty_reason": changes_error,
        "working_tree_status": changes,
    }


def _memory_bytes() -> dict[str, Any]:
    try:
        size = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        if size > 0:
            return {"value": size, "source": "os.sysconf physical pages and page size"}
    except (ValueError, OSError, AttributeError):
        pass
    if sys.platform == "darwin":
        size, error = _command(["sysctl", "-n", "hw.memsize"])
        if size is not None and size.isdigit():
            return {"value": int(size), "source": "sysctl hw.memsize"}
        return undefined_metric(error or "physical memory could not be measured")
    return undefined_metric("physical memory query is unavailable on this platform")


def _torch_facts(requested_device: str) -> dict[str, Any]:
    try:
        import torch
    except (ImportError, OSError) as exc:
        return {
            "available": None,
            "availability_reason": f"PyTorch unavailable: {exc}",
            "execution_tested": False,
        }
    facts: dict[str, Any] = {
        "build": str(torch.__version__),
        "git_version": getattr(torch.version, "git_version", None),
        "cuda_build": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "execution_tested": False,
    }
    if requested_device == "cpu":
        model = platform.processor()
        facts.update(available=True, model=model or None)
        facts["model_source"] = "platform.processor"
        if model.lower() in {"", "arm", "arm64", "aarch64", "x86_64", "amd64"}:
            facts["model"] = None
            facts["model_reason"] = "platform.processor does not identify the CPU model"
        if sys.platform == "darwin":
            model, _ = _command(["sysctl", "-n", "machdep.cpu.brand_string"])
            if model:
                facts["model"] = model
                facts["model_source"] = "sysctl machdep.cpu.brand_string"
                facts.pop("model_reason", None)
    elif requested_device == "mps":
        facts["available"] = bool(torch.backends.mps.is_available())
        facts["model"] = None
        facts["model_reason"] = (
            "backend availability does not identify the accelerator model"
        )
    elif requested_device == "cuda" or requested_device.startswith("cuda:"):
        available = bool(torch.cuda.is_available())
        try:
            index = (
                int(requested_device.split(":", 1)[1]) if ":" in requested_device else 0
            )
            available = available and 0 <= index < torch.cuda.device_count()
            if available:
                properties = torch.cuda.get_device_properties(index)
                facts.update(
                    model=properties.name, device_memory_bytes=properties.total_memory
                )
        except (ValueError, RuntimeError, AssertionError) as exc:
            facts["availability_reason"] = str(exc)
            available = False
        facts["available"] = available
    else:
        facts.update(
            available=False, availability_reason="unsupported device identifier"
        )
    if facts["available"] is False and "availability_reason" not in facts:
        facts["availability_reason"] = (
            "requested backend/device is unavailable in this environment"
        )
    return facts


def collect_metadata(
    resolved_config: dict[str, Any],
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Capture observations; availability is distinct from executed device work."""
    root = Path(project_root) if project_root is not None else Path.cwd()
    request = resolved_config.get("request", {})
    training = resolved_config.get("training") or {}
    case = resolved_config.get("case") or {}
    versions: dict[str, Any] = {}
    for name in (
        "hydro-pinn",
        "numpy",
        "scipy",
        "matplotlib",
        "torch",
        "pytest",
        "ruff",
    ):
        try:
            versions[name] = {"version": importlib.metadata.version(name)}
        except importlib.metadata.PackageNotFoundError:
            versions[name] = undefined_metric(
                "distribution not installed in this interpreter"
            )
    device = request.get("device", resolved_config.get("device", "cpu"))
    dtype = request.get("dtype", resolved_config.get("dtype"))
    facts = _torch_facts(device)
    if device == "mps" and dtype == "float64":
        facts.update(
            available=False, availability_reason="MPS does not support float64"
        )
    weighting_mode = training.get("weighting", {}).get("mode")
    variant = training.get("variant", training.get("name"))
    if variant is None and weighting_mode is not None:
        variant = {"fixed": "baseline", "gradient_norm_ema": "adaptive"}.get(
            weighting_mode, weighting_mode
        )
    return {
        "schema_version": 1,
        "case": case.get("case_id", case.get("id")),
        "variant": variant,
        "variant_reason": None
        if training
        else "this stage has no training configuration",
        "variant_source": "training variant/name or weighting.mode",
        "seed": request.get("seed", resolved_config.get("seed")),
        "dtype": dtype,
        "input_hashes": copy.deepcopy(
            resolved_config.get(
                "input_hashes",
                {
                    entry["path"]: entry["sha256"]
                    for entry in resolved_config.get("provenance", {}).get(
                        "input_files", []
                    )
                },
            )
        ),
        "provenance": copy.deepcopy(resolved_config.get("provenance", {})),
        "code": _code_state(root),
        "software": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": sys.executable,
            "packages": versions,
        },
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "logical_cpu_count": os.cpu_count(),
            "physical_memory_bytes": _memory_bytes(),
        },
        "device": {"requested": device, **facts},
        "numerical_floor": copy.deepcopy(
            training.get("network", {}).get("depth_floor_nd")
        ),
        "loss_weights": copy.deepcopy(
            training.get("weighting", training.get("loss_weights"))
        ),
        "data_partitions": copy.deepcopy(
            resolved_config.get(
                "data_partitions",
                {
                    "evaluation": resolved_config.get("evaluation"),
                    "validation": training.get("validation"),
                },
            )
        ),
        "reference_provenance": copy.deepcopy(case.get("reference")),
        "execution": {
            "elapsed_timings_s": undefined_metric("stage has not finished"),
            "iterations": undefined_metric("stage has not reported iterations"),
            "function_evaluations": undefined_metric(
                "stage has not reported function evaluations"
            ),
            "stopping_reason": None,
        },
    }


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class RunRecord:
    """An exclusively created local run and its persisted lifecycle state."""

    path: Path
    _started_monotonic: float

    @classmethod
    def create(
        cls,
        output_root: str | Path,
        run_id: str,
        resolved_config: dict[str, Any],
        *,
        metadata: dict[str, Any] | None = None,
        project_root: str | Path | None = None,
    ) -> Self:
        started_monotonic = time.perf_counter()
        validate_run_id(run_id)
        request = resolved_config.get("request", {})
        declared_id = request.get("run_id", resolved_config.get("run_id"))
        if declared_id is not None and declared_id != run_id:
            raise RecordError(
                "run_id differs from the resolved configuration request.run_id"
            )
        config_payload = _json_bytes(resolved_config)
        extra = copy.deepcopy(metadata or {})
        _json_bytes(extra)
        # Probe before claiming output ownership, so unavailable metadata queries
        # cannot leave a directory that looks like an attempted numerical run.
        observations = collect_metadata(resolved_config, project_root=project_root)
        _json_bytes(observations)
        root = Path(output_root).absolute()
        _reject_symlink_path(root)
        root.mkdir(parents=True, exist_ok=True)
        directory = root / run_id
        _reject_symlink_path(directory)
        directory.mkdir(exist_ok=False)
        created = _utc_now()
        observations.update(
            run_id=run_id,
            created_at_utc=created,
            resolved_config_sha256=hashlib.sha256(config_payload).hexdigest(),
            stage_metadata=extra,
        )
        status = {
            "schema_version": 1,
            "run_id": run_id,
            "state": "running",
            "started_at_utc": created,
            "updated_at_utc": created,
            "stopping_reason": None,
            "diagnostics": {},
        }
        # A status marker precedes the other writes. Preserve a partial directory
        # on error, and mark failed initialization when the filesystem permits it.
        try:
            write_json_atomic(directory / "status.json", status)
            write_json_atomic(directory / "resolved_config.json", resolved_config)
            write_json_atomic(directory / "metadata.json", observations)
        except BaseException as exc:
            status.update(
                state="interrupted"
                if isinstance(exc, (KeyboardInterrupt, SystemExit))
                else "failed",
                updated_at_utc=_utc_now(),
                stopping_reason=f"{type(exc).__name__}: {exc}",
                diagnostics={
                    "category": "record_initialization",
                    "stage": "record_creation",
                },
            )
            try:
                write_json_atomic(directory / "status.json", status, overwrite=True)
            except (OSError, RecordError):
                # A full filesystem failure can prevent even failure evidence.
                # Keep the claimed directory and propagate the original error.
                pass
            raise
        return cls(directory, started_monotonic)

    @property
    def status(self) -> dict[str, Any]:
        _reject_symlink_path(self.path / "status.json")
        return json.loads((self.path / "status.json").read_text())

    def _transition(
        self,
        state: str,
        reason: str,
        diagnostics: dict[str, Any] | None = None,
        *,
        timings: dict[str, float] | None = None,
        iterations: int | None = None,
        function_evaluations: int | None = None,
    ) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise RecordError(
                "a terminal run status requires a nonempty stopping reason"
            )
        current = self.status
        if current["state"] != "running":
            raise RecordError(f"cannot change terminal run state {current['state']!r}")
        for label, count in (
            ("iterations", iterations),
            ("function_evaluations", function_evaluations),
        ):
            if count is not None and (
                isinstance(count, bool) or not isinstance(count, int) or count < 0
            ):
                raise RecordError(f"{label} must be a nonnegative integer or None")
        if timings is not None:
            for key, value in timings.items():
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (float, int))
                    or not math.isfinite(value)
                    or value < 0
                ):
                    raise RecordError(
                        f"elapsed timing {key!r} must be finite and nonnegative seconds"
                    )
        details = copy.deepcopy(diagnostics or {})
        _json_bytes(details)
        _reject_symlink_path(self.path / "metadata.json")
        metadata = json.loads((self.path / "metadata.json").read_text())
        config_path = self.path / "resolved_config.json"
        expected_hash = metadata["resolved_config_sha256"]
        snapshot_hash = None
        integrity_reason = None
        try:
            _reject_symlink_path(config_path)
            snapshot_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
            if snapshot_hash != expected_hash:
                integrity_reason = (
                    "resolved configuration snapshot changed after run creation"
                )
        except (OSError, RecordError) as exc:
            integrity_reason = f"resolved configuration snapshot unavailable: {exc}"
        if integrity_reason is not None:
            if state == "completed":
                raise RecordError(integrity_reason)
            # Corrupted inputs must remain reportable as failures/interruption.
            # Preserve the snapshot and its original provenance hash unchanged.
            details["input_integrity"] = {
                "path": "resolved_config.json",
                "expected_sha256": expected_hash,
                "observed_sha256": snapshot_hash,
                "reason": integrity_reason,
            }
        if state == "completed" and metadata["device"].get("available") is False:
            raise RecordError(
                "unavailable requested device cannot be recorded "
                "as successful execution"
            )
        elapsed = time.perf_counter() - self._started_monotonic
        metadata["execution"] = {
            "elapsed_timings_s": {**(timings or {}), "record_lifetime": elapsed},
            "iterations": iterations
            if iterations is not None
            else undefined_metric("not reported by this stage"),
            "function_evaluations": function_evaluations
            if function_evaluations is not None
            else undefined_metric("not reported by this stage"),
            "stopping_reason": reason,
        }
        updated = {
            **current,
            "state": state,
            "updated_at_utc": _utc_now(),
            "stopping_reason": reason,
            "diagnostics": details,
        }
        # Status is the final commit marker after provenance is safely updated.
        write_json_atomic(self.path / "metadata.json", metadata, overwrite=True)
        write_json_atomic(self.path / "status.json", updated, overwrite=True)

    def complete(
        self,
        *,
        stopping_reason: str = "completed",
        elapsed_timings_s: dict[str, float] | None = None,
        iterations: int | None = None,
        function_evaluations: int | None = None,
    ) -> None:
        self._transition(
            "completed",
            stopping_reason,
            timings=elapsed_timings_s,
            iterations=iterations,
            function_evaluations=function_evaluations,
        )

    def fail(
        self,
        reason: str,
        *,
        stage: str | None = None,
        time_s: float | None = None,
        step: int | None = None,
        last_valid_state: Any = None,
        category: str = "execution_failure",
    ) -> None:
        if time_s is not None and (
            isinstance(time_s, bool)
            or not isinstance(time_s, (int, float))
            or not math.isfinite(time_s)
            or time_s < 0
        ):
            raise RecordError("failure time_s must be finite nonnegative seconds")
        if step is not None and (
            isinstance(step, bool) or not isinstance(step, int) or step < 0
        ):
            raise RecordError("failure step must be a nonnegative integer")
        self._transition(
            "failed",
            reason,
            {
                "category": category,
                "stage": stage,
                "time_s": time_s,
                "step": step,
                "last_valid_state": last_valid_state,
            },
        )

    def interrupt(
        self, reason: str = "interrupted by caller", *, last_valid_state: Any = None
    ) -> None:
        self._transition("interrupted", reason, {"last_valid_state": last_valid_state})

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exception_type: Any, exception: BaseException | None, traceback: Any
    ) -> bool:
        if self.status["state"] != "running":
            return False
        if isinstance(exception, (KeyboardInterrupt, SystemExit)):
            self.interrupt(f"{type(exception).__name__}: {exception}")
        elif exception is not None:
            self.fail(f"{type(exception).__name__}: {exception}")
        else:
            self.fail(
                "context exited without explicit completion",
                category="unfinished_stage",
            )
        return False
