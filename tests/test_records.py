"""Consequential output-safety and provenance checks using temporary runs only."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from hydro_pinn import records
from hydro_pinn.records import RecordError, RunRecord, undefined_metric, validate_run_id


@pytest.fixture
def resolved() -> dict[str, Any]:
    return {
        "case": {"case_id": "E2", "h_right_m": 0, "reference": {"kind": "ritter"}},
        "training": {
            "network": {"depth_floor_nd": 1e-6},
            "weighting": {"mode": "fixed", "lambda_data": 0},
        },
        "evaluation": {"partition": "implementation_verification"},
        "request": {"seed": 11, "device": "cpu", "dtype": "float64", "run_id": None},
        "provenance": {
            "input_files": [{"path": "configs/cases/e2_dry.json", "sha256": "a" * 64}]
        },
    }


@pytest.fixture(autouse=True)
def lightweight_device_facts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        records,
        "_torch_facts",
        lambda device: {
            "available": device == "cpu",
            "model": "test fixture CPU",
            "execution_tested": False,
        },
    )


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


@pytest.mark.parametrize(
    "run_id",
    [
        "../escape",
        "/tmp/run",
        "a/b",
        "a\\b",
        ".",
        "..",
        "",
        "x.y",
        "x" * 97,
        "é",
        "NUL",
        "COM1",
    ],
)
def test_unsafe_identifiers_rejected(run_id: str) -> None:
    with pytest.raises(RecordError, match="run_id"):
        validate_run_id(run_id)


def test_create_preserves_resolved_snapshot_and_hash(
    tmp_path: Path, resolved: dict[str, Any]
) -> None:
    run = RunRecord.create(
        tmp_path / "runs", "e2-baseline-11", resolved, project_root=tmp_path
    )
    payload = (run.path / "resolved_config.json").read_bytes()
    metadata = read(run.path / "metadata.json")
    resolved["case"]["h_right_m"] = 99
    assert read(run.path / "resolved_config.json")["case"]["h_right_m"] == 0
    assert metadata["resolved_config_sha256"] == hashlib.sha256(payload).hexdigest()
    assert metadata["input_hashes"]["configs/cases/e2_dry.json"] == "a" * 64
    assert metadata["seed"] == 11
    assert metadata["variant"] == "baseline"
    assert metadata["dtype"] == "float64"
    assert metadata["numerical_floor"] == 1e-6
    assert metadata["loss_weights"]["lambda_data"] == 0
    assert metadata["code"]["revision"] is None
    assert metadata["code"]["revision_reason"]
    assert run.status["state"] == "running"
    assert not (run.path / "outputs.npz").exists()


def test_existing_directory_never_overwritten(
    tmp_path: Path, resolved: dict[str, Any]
) -> None:
    run = RunRecord.create(tmp_path / "runs", "retain", resolved)
    run.complete()
    before = {path.name: path.read_bytes() for path in run.path.iterdir()}
    with pytest.raises(FileExistsError):
        RunRecord.create(tmp_path / "runs", "retain", resolved)
    assert before == {path.name: path.read_bytes() for path in run.path.iterdir()}


def test_running_directory_also_cannot_be_reused(
    tmp_path: Path, resolved: dict[str, Any]
) -> None:
    RunRecord.create(tmp_path / "runs", "running", resolved)
    with pytest.raises(FileExistsError):
        RunRecord.create(tmp_path / "runs", "running", resolved)


def test_symlink_root_or_run_rejected(tmp_path: Path, resolved: dict[str, Any]) -> None:
    target = tmp_path / "outside"
    target.mkdir()
    link = tmp_path / "runs"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(RecordError, match="symlink"):
        RunRecord.create(link, "escape", resolved)
    assert not (target / "escape").exists()


def test_output_root_parent_traversal_rejected(
    tmp_path: Path, resolved: dict[str, Any]
) -> None:
    with pytest.raises(RecordError, match="traversal"):
        RunRecord.create(tmp_path / "sub" / ".." / "runs", "escape", resolved)


def test_requested_id_cannot_differ(tmp_path: Path, resolved: dict[str, Any]) -> None:
    resolved["request"]["run_id"] = "original"
    with pytest.raises(RecordError, match="differs"):
        RunRecord.create(tmp_path / "runs", "different", resolved)
    assert not (tmp_path / "runs").exists()


def test_failure_and_last_state_are_persisted(
    tmp_path: Path, resolved: dict[str, Any]
) -> None:
    run = RunRecord.create(tmp_path / "runs", "failed", resolved)
    run.fail(
        "negative predictor depth",
        stage="predictor",
        time_s=0.01,
        step=2,
        last_valid_state={"h_m": [1.0, 0.1], "q_m2_s": [0, 0]},
    )
    status = read(run.path / "status.json")
    assert status["state"] == "failed"
    assert status["stopping_reason"] == "negative predictor depth"
    assert status["diagnostics"]["stage"] == "predictor"
    assert status["diagnostics"]["time_s"] == 0.01
    assert status["diagnostics"]["step"] == 2
    assert status["diagnostics"]["last_valid_state"]["h_m"] == [1.0, 0.1]
    assert (
        read(run.path / "metadata.json")["execution"]["stopping_reason"]
        == "negative predictor depth"
    )
    with pytest.raises(RecordError, match="terminal"):
        run.complete()


def test_interruption_persists_and_exception_propagates(
    tmp_path: Path, resolved: dict[str, Any]
) -> None:
    run = RunRecord.create(tmp_path / "runs", "interrupted", resolved)
    with pytest.raises(KeyboardInterrupt):
        with run:
            raise KeyboardInterrupt("manual stop")
    assert run.status["state"] == "interrupted"
    assert "manual stop" in run.status["stopping_reason"]


def test_context_does_not_turn_an_unfinished_stage_into_success(
    tmp_path: Path, resolved: dict[str, Any]
) -> None:
    with RunRecord.create(tmp_path / "runs", "unfinished", resolved) as run:
        pass
    assert run.status["state"] == "failed"
    assert run.status["diagnostics"]["category"] == "unfinished_stage"


def test_context_exception_persists_failure(
    tmp_path: Path, resolved: dict[str, Any]
) -> None:
    run = RunRecord.create(tmp_path / "runs", "exception", resolved)
    with pytest.raises(RuntimeError, match="bad state"):
        with run:
            raise RuntimeError("bad state")
    assert run.status["state"] == "failed"
    assert "RuntimeError: bad state" == run.status["stopping_reason"]


def test_undefined_metric_is_null_with_reason(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    records.write_json_atomic(
        path, {"relative_l2_q": undefined_metric("zero reference norm")}
    )
    assert read(path)["relative_l2_q"] == {
        "value": None,
        "reason": "zero reference norm",
    }
    assert "NaN" not in path.read_text()
    with pytest.raises(RecordError, match="reason"):
        undefined_metric("")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_json_rejected_without_claiming_run(
    tmp_path: Path, resolved: dict[str, Any], value: float
) -> None:
    resolved["bad_metric"] = value
    with pytest.raises(RecordError, match="strict JSON"):
        RunRecord.create(tmp_path / "runs", "invalid", resolved)
    assert not (tmp_path / "runs").exists()


def test_atomic_status_failure_preserves_previous_status(
    tmp_path: Path, resolved: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    run = RunRecord.create(tmp_path / "runs", "atomic", resolved)
    before = (run.path / "status.json").read_bytes()
    replace = records.os.replace

    def failing_replace(source: Path, destination: Path) -> None:
        if destination.name == "status.json":
            raise OSError("simulated filesystem interruption")
        replace(source, destination)

    monkeypatch.setattr(records.os, "replace", failing_replace)
    with pytest.raises(OSError, match="filesystem interruption"):
        run.complete()
    assert (run.path / "status.json").read_bytes() == before
    assert not list(run.path.glob("*.tmp"))


def test_unavailable_device_is_separate_from_success(
    tmp_path: Path, resolved: dict[str, Any]
) -> None:
    resolved["request"]["device"] = "cuda"
    run = RunRecord.create(tmp_path / "runs", "unsupported", resolved)
    assert read(run.path / "metadata.json")["device"]["available"] is False
    with pytest.raises(RecordError, match="unavailable"):
        run.complete()
    run.fail("CUDA unavailable", stage="device_setup", category="unsupported_device")
    assert run.status["diagnostics"]["category"] == "unsupported_device"


def test_execution_counts_and_timing_are_real_inputs(
    tmp_path: Path, resolved: dict[str, Any]
) -> None:
    run = RunRecord.create(tmp_path / "runs", "finished", resolved)
    run.complete(
        stopping_reason="requested output times reached",
        elapsed_timings_s={"solve": 0.01},
        iterations=2,
        function_evaluations=4,
    )
    execution = read(run.path / "metadata.json")["execution"]
    assert execution["iterations"] == 2
    assert execution["function_evaluations"] == 4
    assert execution["elapsed_timings_s"]["solve"] == 0.01
    assert execution["elapsed_timings_s"]["record_lifetime"] >= 0
    assert run.status["state"] == "completed"


def test_stage_metadata_cannot_replace_observed_provenance(
    tmp_path: Path, resolved: dict[str, Any]
) -> None:
    run = RunRecord.create(
        tmp_path / "runs", "safe-extra", resolved, metadata={"code": "invented"}
    )
    metadata = read(run.path / "metadata.json")
    assert isinstance(metadata["code"], dict)
    assert metadata["stage_metadata"]["code"] == "invented"


def test_modified_snapshot_cannot_be_declared_complete(
    tmp_path: Path, resolved: dict[str, Any]
) -> None:
    run = RunRecord.create(tmp_path / "runs", "modified", resolved)
    snapshot = run.path / "resolved_config.json"
    snapshot.write_text("{}\n")
    with pytest.raises(RecordError, match="snapshot changed"):
        run.complete()
    assert run.status["state"] == "running"


@pytest.mark.parametrize("terminal_state", ["failed", "interrupted"])
def test_snapshot_corruption_can_be_persisted_as_a_terminal_outcome(
    tmp_path: Path, resolved: dict[str, Any], terminal_state: str
) -> None:
    run = RunRecord.create(tmp_path / "runs", f"corrupted-{terminal_state}", resolved)
    original_hash = read(run.path / "metadata.json")["resolved_config_sha256"]
    snapshot = run.path / "resolved_config.json"
    corrupted_payload = b'{"corrupted":true}\n'
    snapshot.write_bytes(corrupted_payload)
    with pytest.raises(RecordError, match="snapshot changed"):
        run.complete()
    if terminal_state == "failed":
        run.fail("resolved snapshot corrupted", stage="input_integrity")
    else:
        run.interrupt("stopped after detecting snapshot corruption")
    status = read(run.path / "status.json")
    assert status["state"] == terminal_state
    integrity = status["diagnostics"]["input_integrity"]
    assert integrity["expected_sha256"] == original_hash
    assert integrity["observed_sha256"] == hashlib.sha256(corrupted_payload).hexdigest()
    assert "snapshot changed" in integrity["reason"]
    assert read(run.path / "metadata.json")["resolved_config_sha256"] == original_hash
    assert snapshot.read_bytes() == corrupted_payload


def test_symlink_status_cannot_redirect_updates(
    tmp_path: Path, resolved: dict[str, Any]
) -> None:
    run = RunRecord.create(tmp_path / "runs", "redirected", resolved)
    outside = tmp_path / "outside.json"
    outside.write_text('{"state":"running"}')
    status = run.path / "status.json"
    # Deliberately move only this test's temporary status to simulate tampering.
    status.rename(run.path / "original_status.json")
    status.symlink_to(outside)
    with pytest.raises(RecordError, match="symlink"):
        run.complete()
    assert outside.read_text() == '{"state":"running"}'


def test_identifier_matches_config_validator() -> None:
    from hydro_pinn.config import validate_run_id as validate_config_run_id

    for identifier in ("CON_foo", "lpt1_seed22", "con-run", "valid_11", "X" * 96):
        assert validate_run_id(identifier) == validate_config_run_id(identifier)
    for identifier in ("CON", "LPT1"):
        with pytest.raises(ValueError):
            validate_run_id(identifier)
        with pytest.raises(ValueError):
            validate_config_run_id(identifier)


def test_partial_creation_failure_is_retained_and_marked(
    tmp_path: Path, resolved: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = records.write_json_atomic

    def injected_failure(path: Path, data: Any, *, overwrite: bool = False) -> None:
        if path.name == "metadata.json":
            raise OSError("simulated metadata write failure")
        writer(path, data, overwrite=overwrite)

    monkeypatch.setattr(records, "write_json_atomic", injected_failure)
    with pytest.raises(OSError, match="metadata write failure"):
        RunRecord.create(tmp_path / "runs", "partial", resolved)
    directory = tmp_path / "runs" / "partial"
    assert (directory / "resolved_config.json").exists()
    status = read(directory / "status.json")
    assert status["state"] == "failed"
    assert status["diagnostics"]["category"] == "record_initialization"
    with pytest.raises(FileExistsError):
        RunRecord.create(tmp_path / "runs", "partial", resolved)
