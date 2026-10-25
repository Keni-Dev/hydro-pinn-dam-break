"""Bounded real matrix workflow, preserved failures and explicit resume.

These tests replace only the strict protocol loader with a labeled software
fixture. They never weaken the production freezer or run the final matrix.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch

from hydro_pinn import experiments, train
from hydro_pinn.config import load_config, resolve_run_config
from hydro_pinn.lax_wendroff import NumericalRunFailed
from hydro_pinn.protocol import SCIENTIFIC_MODULES, _matrix
from hydro_pinn.records import RunRecord, collect_metadata, write_json_atomic
from hydro_pinn.references import ReferenceError

ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _input(root: Path, path: str) -> dict:
    return {"path": path, "sha256": _sha(root / path)}


@pytest.fixture
def fixture_matrix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path.resolve()
    (root / "configs/cases").mkdir(parents=True)
    (root / "configs/training").mkdir()
    (root / "docs").mkdir()
    cases = {}
    training = {}
    for case_id, name in (("E1", "e1_wet"), ("E2", "e2_dry")):
        path = f"configs/cases/{name}.json"
        cases[case_id] = load_config(ROOT / path)
        write_json_atomic(root / path, cases[case_id])
    for variant in ("baseline", "adaptive"):
        path = f"configs/training/{variant}.json"
        value = load_config(ROOT / path)
        value["network"]["hidden_layers"] = [4, 4]
        value["sampling"].update(
            interior_points=7, initial_points=6, boundary_points_per_side=3
        )
        value["optimizer"]["steps"] = 2
        value["validation"]["grid"]["cells"] = 7
        training[variant] = value
        write_json_atomic(root / path, value)
    evaluation = load_config(ROOT / "configs/evaluation.json")
    evaluation["grid"]["cells"] = 20
    evaluation["times_s"] = [0.11, 0.31]
    evaluation["profile_times_s"] = [0.31]
    write_json_atomic(root / "configs/evaluation.json", evaluation)
    settings = {
        "device": "cpu",
        "dtype": "float32",
        "cpu_threads": 1,
        "chunk_size": None,
        "seeds": [11, 22, 33],
        "e2_diagnostic_seed": 11,
    }
    matrix, pairs = _matrix("fixture-core", settings)
    metadata = collect_metadata({"request": {"device": "cpu", "dtype": "float32"}})
    protocol = {
        "schema_version": 1,
        "purpose": "software_fixture_not_final_results",
        "protocol_id": "fixture-core",
        "protocol_sha256": "fixture-only-identity",
        "paths": {"output_root": "runs", "manifest": "docs/final_run_manifest.json"},
        "inputs": {
            "cases": {
                case_id: _input(root, f"configs/cases/{name}.json")
                for case_id, name in (("E1", "e1_wet"), ("E2", "e2_dry"))
            },
            "training": {
                variant: _input(root, f"configs/training/{variant}.json")
                for variant in training
            },
            "evaluation": _input(root, "configs/evaluation.json"),
        },
        "resolved": {"cases": cases, "training": training, "evaluation": evaluation},
        "settings": settings,
        "matrix": matrix,
        "pairs": pairs,
        "scientific_source_sha256": {
            f"src/hydro_pinn/{name}.py": _sha(ROOT / f"src/hydro_pinn/{name}.py")
            for name in SCIENTIFIC_MODULES
        },
        "provenance": {"software": metadata["software"]},
    }
    protocol["conservation"] = {
        "auxiliary_records": [
            {"run_id": f"{stage['run_id']}-{kind}-{resolution}"}
            for stage in matrix
            if stage["stage"] in {"reference", "predict"}
            for kind, resolutions in (
                ("volume", (2000, 4000)),
                ("boundary_flux", (1001, 2001)),
            )
            for resolution in resolutions
        ]
    }
    path = root / "configs/software_fixture_protocol.json"
    write_json_atomic(path, protocol)

    def fixture_loader(*args, **kwargs):
        return copy.deepcopy(protocol)

    monkeypatch.setattr(experiments, "load_protocol", fixture_loader)
    return root, path, protocol


def _entries(manifest):
    return {entry["stage_id"]: entry for entry in manifest["entries"]}


def test_real_bounded_matrix_reuse_reconcile_and_corruption(
    fixture_matrix, monkeypatch
):
    root, path, protocol = fixture_matrix
    manifest = experiments.run_matrix(path, project_root=root)
    assert manifest["purpose"] == "software_fixture_not_final_results"
    assert manifest["workflow_status"] == "completed"
    assert len(manifest["entries"]) == 24
    assert all(entry["state"] == "completed" for entry in manifest["entries"])
    assert manifest["scientific_status"]["final_accuracy"] == "not_evaluated"
    preserved = {
        str(p.relative_to(root)): _sha(p)
        for p in (root / "runs").rglob("*")
        if p.is_file()
    }
    # Simulate a completed run whose manifest completion was not committed.
    entry = manifest["entries"][0]
    entry["state"] = "running"
    entry.pop("result")
    entry["attempts"][-1]["state"] = "running"
    write_json_atomic(root / protocol["paths"]["manifest"], manifest, overwrite=True)

    def no_rerun(*args, **kwargs):
        pytest.fail("a verified completed stage was rerun")

    for name in (
        "generate_reference_run",
        "generate_solver_run",
        "train_run",
        "predict_run",
    ):
        monkeypatch.setattr(experiments, name, no_rerun)
    second = experiments.run_matrix(path, project_root=root)
    assert second["workflow_status"] == "completed"
    assert second["entries"][0]["attempts"][-1]["reconciled_after_writer_exit"]
    assert all(_sha(root / name) == digest for name, digest in preserved.items())
    training_entry = next(
        e for e in second["entries"] if e["frozen_stage"]["stage"] == "train"
    )
    checkpoint = root / "runs" / training_entry["actual_run_id"] / "checkpoint.pt"
    with checkpoint.open("ab") as handle:
        handle.write(b"fixture corruption")
    with pytest.raises(ValueError, match="SHA256"):
        experiments.run_matrix(path, project_root=root)


def _failed_record(
    case_path, evaluation, run_id, output_root, *, cells=None, numerical=False
):
    resolved = resolve_run_config(
        case_path,
        evaluation_path=evaluation,
        device="cpu",
        dtype="float64",
        run_id=run_id,
    )
    if cells is not None:
        resolved["solver"] = {"cells": cells}
    with RunRecord.create(
        output_root, run_id, resolved, metadata={"stage": "constructed_test_failure"}
    ) as record:
        record.fail(
            "constructed fixture",
            stage="predictor" if numerical else "reference",
            category="numerical_failure" if numerical else "execution_failure",
        )
    return record.path


@pytest.mark.parametrize("failure", ["numerical", "reference"])
def test_saved_numerical_failure_and_reference_dependency_gate(
    fixture_matrix, monkeypatch, failure
):
    root, path, _ = fixture_matrix
    if failure == "numerical":
        original = experiments.generate_solver_run

        def solver(case, evaluation, cells, run_id, output_root):
            if "e2" in str(case) and cells == 100:
                target = _failed_record(
                    case, evaluation, run_id, output_root, cells=cells, numerical=True
                )
                raise NumericalRunFailed(
                    target,
                    {
                        "stage": "predictor",
                        "time_s": 0.0,
                        "step": 0,
                        "reason": "constructed fixture",
                    },
                )
            return original(case, evaluation, cells, run_id, output_root)

        monkeypatch.setattr(experiments, "generate_solver_run", solver)
    else:
        original = experiments.generate_reference_run

        def reference(case, evaluation, run_id, output_root):
            if "e1" in str(case):
                _failed_record(case, evaluation, run_id, output_root)
                raise ReferenceError("constructed invalid reference fixture")
            return original(case, evaluation, run_id, output_root)

        monkeypatch.setattr(experiments, "generate_reference_run", reference)
    manifest = experiments.run_matrix(path, project_root=root)
    if failure == "numerical":
        assert manifest["workflow_status"] == "completed_with_diagnostics"
        failed = [e for e in manifest["entries"] if e["state"] == "failed"]
        assert len(failed) == 1 and failed[0]["failure_category"] == "numerical_failure"
        assert (
            manifest["scientific_status"]["core_comparison"]
            == "available_for_evaluation"
        )
    else:
        assert manifest["workflow_status"] == "incomplete"
        for entry in manifest["entries"]:
            stage = entry["frozen_stage"]
            if stage["case_id"] == "E1" and stage["stage"] != "reference":
                assert entry["state"] == "blocked"
        assert any(
            e["state"] == "completed" and e["frozen_stage"]["case_id"] == "E2"
            for e in manifest["entries"]
        )


@pytest.mark.parametrize("orphan", [False, True])
def test_interrupted_training_resume_or_preserved_running_orphan(
    fixture_matrix, monkeypatch, orphan
):
    root, path, protocol = fixture_matrix
    original = train.component_gradients
    calls = 0

    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise KeyboardInterrupt("constructed interruption before Adam step2")
        return original(*args, **kwargs)

    monkeypatch.setattr(train, "component_gradients", interrupt)
    with pytest.raises(KeyboardInterrupt):
        experiments.run_matrix(path, project_root=root)
    monkeypatch.setattr(train, "component_gradients", original)
    manifest_path = root / protocol["paths"]["manifest"]
    manifest = json.loads(manifest_path.read_text())
    assert manifest["workflow_status"] == "interrupted"
    entry = next(e for e in manifest["entries"] if e["state"] == "interrupted")
    source = root / "runs" / entry["actual_run_id"]
    if orphan:
        status = json.loads((source / "status.json").read_text())
        status["state"] = "running"
        write_json_atomic(source / "status.json", status, overwrite=True)
        entry["state"] = "running"
        write_json_atomic(manifest_path, manifest, overwrite=True)
    preserved = {p.name: _sha(p) for p in source.iterdir() if p.is_file()}
    arguments = dict(
        project_root=root,
        resume_stage=entry["stage_id"],
        resume_from=entry["actual_run_id"],
        resume_run_id="fixture-resumed-train",
    )
    if orphan:
        blocked = experiments.run_matrix(path, project_root=root)
        assert (
            _entries(blocked)[entry["stage_id"]]["failure_category"] == "orphan_running"
        )
        with pytest.raises(
            experiments.MatrixError, match="terminal failed/interrupted"
        ):
            experiments.run_matrix(path, **arguments)
        assert json.loads((source / "status.json").read_text())["state"] == "running"
    else:
        reserved_auxiliary = protocol["conservation"]["auxiliary_records"][0]["run_id"]
        with pytest.raises(experiments.MatrixError, match="fresh unique"):
            experiments.run_matrix(
                path, **dict(arguments, resume_run_id=reserved_auxiliary)
            )
        result = experiments.run_matrix(path, **arguments)
        assert result["workflow_status"] == "completed"
        resumed = _entries(result)[entry["stage_id"]]
        assert len(resumed["attempts"]) == 2
        assert resumed["actual_run_id"] == "fixture-resumed-train"
        prediction = next(
            e
            for e in result["entries"]
            if e["frozen_stage"].get("training_stage_id") == entry["stage_id"]
        )
        resolved = json.loads(
            (
                root / prediction["result"]["run_path"] / "resolved_config.json"
            ).read_text()
        )
        assert (
            resolved["prediction_source"]["training_run_id"] == "fixture-resumed-train"
        )
    assert all(_sha(source / name) == digest for name, digest in preserved.items())


def test_bad_completed_output_records_failed_manifest_fact(fixture_matrix, monkeypatch):
    root, path, _ = fixture_matrix
    original = experiments.generate_reference_run

    def malformed(case, evaluation, run_id, output_root):
        record = original(case, evaluation, run_id, output_root)
        if "e1" in str(case):
            (record / "outputs.npz").write_bytes(
                b"constructed malformed output fixture"
            )
        return record

    monkeypatch.setattr(experiments, "generate_reference_run", malformed)
    result = experiments.run_matrix(path, project_root=root)
    assert result["workflow_status"] == "incomplete"
    entry = next(
        e
        for e in result["entries"]
        if e["frozen_stage"]["case_id"] == "E1"
        and e["frozen_stage"]["stage"] == "reference"
    )
    assert entry["state"] == "failed"
    assert entry["failure_category"] == "artifact_verification_failure"
    assert entry["attempts"][-1]["record_state_observed"] == "completed"
    assert entry["attempts"][-1]["readback_error"]
    assert (
        json.loads(
            (root / "runs" / entry["actual_run_id"] / "status.json").read_text()
        )["state"]
        == "completed"
    )


def test_matrix_guard_runs_before_manifest_creation_and_lock_is_exclusive(
    fixture_matrix,
    monkeypatch,
):
    root, path, protocol = fixture_matrix

    def mismatch(*args, **kwargs):
        raise experiments.MatrixError("constructed scientific source hash mismatch")

    monkeypatch.setattr(experiments, "load_protocol", mismatch)
    with pytest.raises(experiments.MatrixError, match="source hash"):
        experiments.run_matrix(path, project_root=root)
    assert not (root / protocol["paths"]["manifest"]).exists()
    lock = root / "docs/test.lock"
    with experiments._lock(lock):
        with pytest.raises(experiments.MatrixError, match="live writer"):
            with experiments._lock(lock):
                pytest.fail("second writer acquired lock")


def test_fresh_namespace_reproduces_same_science_without_touching_main(fixture_matrix):
    root, path, protocol = fixture_matrix
    main = experiments.run_matrix(path, project_root=root)
    original_manifest = _sha(root / protocol["paths"]["manifest"])
    original_protocol = _sha(path)
    original_runs = {
        str(p.relative_to(root)): _sha(p)
        for p in (root / "runs").rglob("*")
        if p.is_file()
    }
    frozen_science = copy.deepcopy(protocol["resolved"])
    with pytest.raises(experiments.MatrixError, match="require a fresh namespace"):
        experiments.execution_plan(protocol, output_root="runs/paper")
    plan = experiments.execution_plan(
        protocol,
        namespace="replay",
        output_root="runs/paper",
        manifest_path="docs/paper_manifest.json",
    )
    assert len(plan["run_ids"]) == 60
    replay = experiments.run_matrix(
        path,
        project_root=root,
        execution_namespace="replay",
        output_root="runs/paper",
        manifest_path="docs/paper_manifest.json",
    )
    assert replay["workflow_status"] == "completed"
    assert replay["execution"] == plan
    assert (
        replay["protocol_sha256"]
        == main["protocol_sha256"]
        == protocol["protocol_sha256"]
    )
    assert protocol["resolved"] == frozen_science
    assert all(
        entry["actual_run_id"] == plan["run_ids"][entry["frozen_stage"]["run_id"]]
        for entry in replay["entries"]
    )
    for old, new in zip(main["entries"], replay["entries"], strict=True):
        assert old["frozen_stage"] == new["frozen_stage"]
        if old["frozen_stage"]["stage"] == "train":
            first, _, _ = train.load_training_checkpoint(
                old["actual_run_id"], root / "runs"
            )
            second, _, _ = train.load_training_checkpoint(
                new["actual_run_id"], root / "runs/paper"
            )
            assert all(
                torch.equal(
                    first["latest_model_state"][key], second["latest_model_state"][key]
                )
                for key in first["latest_model_state"]
            )
    assert _sha(root / protocol["paths"]["manifest"]) == original_manifest
    assert _sha(path) == original_protocol
    assert all(_sha(root / name) == digest for name, digest in original_runs.items())
    with pytest.raises(experiments.MatrixError, match="namespace already exists"):
        experiments.run_matrix(
            path,
            project_root=root,
            execution_namespace="replay",
            output_root="runs/paper",
            manifest_path="docs/paper_manifest.json",
        )
