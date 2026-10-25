"""Bounded reproduction checks; these do not run the full paper matrix."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hydro_pinn import experiments, reproduce
from hydro_pinn.protocol import load_protocol
from hydro_pinn.records import write_json_atomic

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path: Path) -> tuple[Path, dict]:
    """Copy the real frozen inputs; only the explicit smoke budget is reduced."""
    root = tmp_path.resolve()
    protocol = json.loads((ROOT / "configs/final_protocol.json").read_text())
    paths = {"configs/final_protocol.json", "configs/smoke.json"}

    def inputs(value):
        if isinstance(value, dict):
            if {"path", "sha256"} <= value.keys():
                paths.add(value["path"])
            else:
                for item in value.values():
                    inputs(item)
        elif isinstance(value, list):
            for item in value:
                inputs(item)

    inputs(protocol["inputs"])
    paths.update(protocol["scientific_source_sha256"])
    paths.update(
        f"src/hydro_pinn/{name}.py"
        for name in ("evaluation_workflow", "evaluate", "conservation", "benchmark")
    )
    for name in paths:
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    # This typed smoke config is not a paper input. The network remains the
    # frozen 4x64 architecture; two steps keep pipeline verification bounded.
    smoke_path = root / "configs/smoke.json"
    smoke = json.loads(smoke_path.read_text())
    smoke["training_overrides"]["optimizer"]["steps"] = 2
    smoke["training_overrides"]["sampling"].update(
        interior_points=7, initial_points=6, boundary_points_per_side=3
    )
    smoke["validation"]["grid"]["cells"] = 7
    write_json_atomic(smoke_path, smoke, overwrite=True)
    manifest = experiments._new_manifest(
        protocol,
        root / "configs/final_protocol.json",
        {"purpose": "unexecuted parent manifest fixture"},
        experiments.execution_plan(protocol),
    )
    write_json_atomic(root / protocol["paths"]["manifest"], manifest)
    assert load_protocol("configs/final_protocol.json", project_root=root) == protocol
    return root, protocol


def test_actual_smoke_runs_fresh_stages_and_preserves_parent(project):
    root, protocol = project
    original = {
        str(path.relative_to(root)): reproduce._sha(path)
        for path in root.rglob("*")
        if path.is_file()
    }
    report = reproduce.reproduce(
        "configs/final_protocol.json",
        "smoke",
        "results/reproduction/smoke",
        project_root=root,
    )
    assert report["state"] in {"completed", "completed_with_diagnostics"}
    assert report["mode"] == "smoke"
    assert not report["paper_reproduction_established"]
    assert report["saved_output_replay"]["state"] == "not_executed"
    assert report["parent"]["manifest"]["workflow_status_at_observation"] == "running"
    assert report["parent"]["frozen_training_budget"] == {
        "baseline": 10000,
        "adaptive": 10000,
    }
    stages = report["fresh_execution"]["stages"]
    assert len(stages) == 10
    assert {row["stage"] for row in stages} == {
        "reference",
        "solve",
        "train",
        "predict",
    }
    trained = [row for row in stages if row["stage"] == "train"]
    assert len(trained) == 3
    assert all(row["completed_adam_steps"] == 2 for row in trained)
    assert all(
        row["run_id"].startswith(report["execution_namespace"]) for row in stages
    )
    for row in stages:
        path = root / "runs" / row["run_id"]
        assert (path / "reproduction_binding.json").is_file()
        if row["stage"] == "predict" and row["state"] == "completed":
            assert row["held_model_vs_checkpoint_reload"]["within_tolerance"]
        resolved = json.loads((path / "resolved_config.json").read_text())
        assert resolved["case"] == protocol["resolved"]["cases"][row["case_id"]]
    metrics = json.loads(
        (root / "results/reproduction/smoke/smoke_metrics.json").read_text()
    )
    assert metrics["rows"]
    assert metrics["conservation"] == "not_executed"
    assert all(
        reproduce._sha(root / name) == digest for name, digest in original.items()
    )
    assert not any(
        (root / "runs" / row["run_id"]).exists() for row in protocol["matrix"]
    )


@pytest.mark.parametrize("problem", ["changed_input", "missing_input"])
def test_invalid_frozen_input_preserves_failure_report_before_execution(
    project, problem
):
    root, _ = project
    target = root / "configs/evaluation.json"
    if problem == "changed_input":
        target.write_text(target.read_text() + "\n")
    else:
        # Relocation keeps the fixture bytes while testing a missing required path.
        target.rename(target.with_name("evaluation_retained.json"))
    with pytest.raises(reproduce.ReproductionError) as captured:
        reproduce.reproduce(
            "configs/final_protocol.json",
            "smoke",
            f"results/{problem}",
            project_root=root,
        )
    report = json.loads(captured.value.report_path.read_text())
    assert report["state"] == "failed"
    assert report["fresh_execution"]["stages"] == []
    assert not (root / "runs").exists()


def test_existing_destination_is_preserved(project):
    root, _ = project
    directory = root / "results/existing"
    directory.mkdir(parents=True)
    retained = directory / "report.json"
    retained.write_text("preserved existing result")
    with pytest.raises(FileExistsError):
        reproduce.reproduce(
            "configs/final_protocol.json",
            "smoke",
            directory,
            project_root=root,
        )
    assert retained.read_text() == "preserved existing result"
    assert not (root / "runs").exists()


def test_failed_fresh_stage_never_becomes_success_or_deletes_evidence(
    project, monkeypatch
):
    root, _ = project
    original = reproduce.generate_reference_run

    def fail_after_record(*args, **kwargs):
        path = original(*args, **kwargs)
        raise RuntimeError(f"constructed failure after producing {path.name}")

    monkeypatch.setattr(reproduce, "generate_reference_run", fail_after_record)
    with pytest.raises(reproduce.ReproductionError) as captured:
        reproduce.reproduce(
            "configs/final_protocol.json",
            "smoke",
            "results/failure",
            project_root=root,
        )
    report = json.loads(captured.value.report_path.read_text())
    assert report["state"] == "failed"
    assert not report["paper_reproduction_established"]
    row = report["fresh_execution"]["stages"][0]
    assert row["state"] == "failed"
    assert row["record_state_observed"] == "completed"
    assert (root / row["run_path"] / "outputs.npz").is_file()
    assert (root / row["run_path"] / "reproduction_binding.json").is_file()
    assert row["reason"].startswith("RuntimeError: constructed failure")


def test_paper_pending_results_stops_before_fresh_training(project, monkeypatch):
    root, _ = project

    def no_matrix(*args, **kwargs):
        pytest.fail("paper rerun started before canonical results were ready")

    monkeypatch.setattr(reproduce, "run_matrix", no_matrix)
    with pytest.raises(reproduce.ReproductionError, match="complete terminal"):
        reproduce.reproduce(
            "configs/final_protocol.json",
            "paper",
            "results/paper_pending",
            project_root=root,
        )
    report = json.loads((root / "results/paper_pending/report.json").read_text())
    assert report["mode"] == "paper"
    assert report["state"] == "failed"
    assert report["fresh_execution"]["stages"] == []
    assert not (root / "runs").exists()


@pytest.mark.parametrize("outcome", ["incomplete", "interrupted"])
def test_paper_forwards_same_frozen_budget_and_new_destinations(
    project, monkeypatch, outcome
):
    """API contract only; mocks cannot establish an executed paper reproduction."""
    from hydro_pinn import evaluation_workflow

    root, protocol = project
    frozen = copy.deepcopy(protocol)
    parent_manifest = json.loads((root / protocol["paths"]["manifest"]).read_text())
    (root / "results").mkdir()
    write_json_atomic(root / "results/metrics.json", {})
    write_json_atomic(root / "results/timings.json", {})
    monkeypatch.setattr(
        reproduce,
        "_result_dependencies",
        lambda *args: (
            parent_manifest,
            {"rows": [], "runs": [], "provenance": {"source_files": []}},
            {"state": "completed"},
        ),
    )
    monkeypatch.setattr(
        reproduce,
        "_rendering_bindings",
        lambda *args: {
            "source_sha256": {},
            "preserved_files_sha256": {},
            "parent_canonical_ledger": {},
            "parent_core_ledger": {},
            "parent_export_resolutions": [],
        },
    )
    monkeypatch.setattr(
        evaluation_workflow,
        "evaluate_manifest",
        lambda *args, **kwargs: {
            "rows": [],
            "runs": [],
            "provenance": {"execution": experiments.execution_plan(protocol)},
        },
    )
    monkeypatch.setattr(
        evaluation_workflow, "plot_manifest", lambda *args, **kwargs: {"figures": []}
    )
    calls = []

    def matrix(path, **kwargs):
        calls.append(kwargs)
        checked = load_protocol(path, project_root=kwargs["project_root"])
        assert checked == frozen
        assert (
            sum(
                checked["resolved"]["training"][row["variant"]]["optimizer"]["steps"]
                for row in checked["matrix"]
                if row["stage"] == "train"
            )
            == 70000
        )
        assert kwargs["execution_namespace"].startswith("p07-")
        assert kwargs["output_root"] == "runs"
        assert kwargs["manifest_path"] != root / protocol["paths"]["manifest"]
        retained_report = json.loads(
            (kwargs["manifest_path"].parent / "report.json").read_text()
        )
        planned = retained_report["fresh_execution"]
        assert planned["state"] == "planned"
        assert planned["manifest_path"] == str(
            kwargs["manifest_path"].relative_to(root)
        )
        assert len(planned["execution"]["run_ids"]) == 60
        if outcome == "interrupted":
            training = next(row for row in checked["matrix"] if row["stage"] == "train")
            write_json_atomic(
                kwargs["manifest_path"],
                {
                    "purpose": "mocked_interruption_fixture_not_numerical_run",
                    "protocol_sha256": checked["protocol_sha256"],
                    "execution": planned["execution"],
                    "workflow_status": "interrupted",
                    "entries": [
                        {
                            "frozen_stage": training,
                            "state": "interrupted",
                            "actual_run_id": planned["execution"]["run_ids"][
                                training["run_id"]
                            ],
                        }
                    ],
                },
            )
            raise KeyboardInterrupt("constructed matrix interruption")
        return {"execution": {}, "entries": [], "workflow_status": "incomplete"}

    monkeypatch.setattr(reproduce, "run_matrix", matrix)
    expected_error = (
        KeyboardInterrupt if outcome == "interrupted" else reproduce.ReproductionError
    )
    expected_message = (
        "constructed matrix interruption"
        if outcome == "interrupted"
        else "fresh paper matrix incomplete"
    )
    with pytest.raises(expected_error, match=expected_message):
        reproduce.reproduce(
            "configs/final_protocol.json",
            "paper",
            "results/paper_fixture",
            project_root=root,
        )
    assert len(calls) == 1
    report = json.loads((root / "results/paper_fixture/report.json").read_text())
    assert (
        report["fresh_execution"]["manifest_path"]
        == "results/paper_fixture/fresh_manifest.json"
    )
    if outcome == "interrupted":
        assert report["state"] == "interrupted"
        assert report["fresh_execution"]["observed_workflow_status"] == "interrupted"
        assert report["fresh_execution"]["stages"][0]["state"] == "interrupted"
        assert not report["fresh_execution"]["observed_artifacts_verified"]
        assert (root / report["fresh_execution"]["manifest_path"]).is_file()
    assert load_protocol("configs/final_protocol.json", project_root=root) == frozen
    assert not (root / "runs").exists()


def test_numerical_differences_are_retained_with_tolerances(tmp_path):
    first, second = tmp_path / "first.npz", tmp_path / "second.npz"
    np.savez(first, x_m=np.array([0.0]), t_s=np.array([0.1]), h_m=np.array([[1.0]]))
    np.savez(second, x_m=np.array([0.0]), t_s=np.array([0.1]), h_m=np.array([[1.01]]))
    result = reproduce._compare_arrays(first, second, "network_float32")
    assert not result["within_tolerance"]
    assert result["fields"]["h_m"]["max_absolute_difference"] == pytest.approx(0.01)
    assert (
        result["fields"]["h_m"]["tolerance"] == reproduce.TOLERANCES["network_float32"]
    )
    # Matching field tolerances cannot conceal a different training stop or
    # validation-selected checkpoint step.
    stage = {"stage": "train", "stage_id": "training", "run_id": "training"}
    parent = {
        "entries": [
            {
                "stage_id": "training",
                "frozen_stage": stage,
                "actual_run_id": "parent-train",
                "state": "completed",
                "result": {
                    "run_path": "runs/parent-train",
                    "completed_adam_steps": 10000,
                    "selected_step": 9010,
                    "partition_content_sha256": "same-partitions",
                },
            }
        ],
    }
    fresh = copy.deepcopy(parent)
    fresh["entries"][0]["actual_run_id"] = "fresh-train"
    fresh["entries"][0]["result"]["run_path"] = "runs/fresh-train"
    assert reproduce._paper_comparison(parent, fresh, {}, tmp_path)[
        "within_declared_tolerances"
    ]
    fresh["entries"][0]["result"]["selected_step"] += 1
    selected = reproduce._paper_comparison(parent, fresh, {}, tmp_path)
    assert not selected["within_declared_tolerances"]
    assert not selected["stages"][0]["checkpoint_selection_step_matches"]
    fresh["entries"][0]["result"]["selected_step"] -= 1
    fresh["entries"][0]["result"]["completed_adam_steps"] -= 1
    steps = reproduce._paper_comparison(parent, fresh, {}, tmp_path)
    assert not steps["within_declared_tolerances"]
    assert not steps["stages"][0]["completed_step_count_matches"]


def test_conservation_and_all_auxiliary_outputs_are_compared(project):
    root, protocol = project
    parent_execution = experiments.execution_plan(protocol)
    fresh_execution = experiments.execution_plan(protocol, namespace="tiny")
    parent = {
        "execution": parent_execution,
        "entries": [
            {"actual_run_id": row["run_id"], "frozen_stage": row}
            for row in protocol["matrix"]
        ],
    }
    fresh = {"execution": fresh_execution}
    for plan in protocol["conservation"]["auxiliary_records"]:
        for execution in (parent_execution, fresh_execution):
            directory = root / "runs" / execution["run_ids"][plan["run_id"]]
            directory.mkdir(parents=True)
            np.savez(
                directory / "outputs.npz",
                x_m=np.array([0.0]),
                t_s=np.array([0.1]),
                h_m=np.array([[1.0]]),
                q_m2_s=np.array([[0.0]]),
            )
    arrays = reproduce._compare_auxiliary(parent, fresh, protocol, root)
    assert arrays["within_tolerance"]
    assert arrays["expected_records"] == arrays["compared_records"] == 36
    target = root / arrays["records"][-1]["fresh_path"]
    np.savez(
        target,
        x_m=np.array([0.0]),
        t_s=np.array([0.1]),
        h_m=np.array([[1.1]]),
        q_m2_s=np.array([[0.0]]),
    )
    changed = reproduce._compare_auxiliary(parent, fresh, protocol, root)
    assert not changed["within_tolerance"]
    assert changed["records"][-1]["comparison"]["fields"]["h_m"][
        "max_absolute_difference"
    ] == pytest.approx(0.1)

    stage = next(row for row in protocol["matrix"] if row["stage"] == "predict")
    metric = {
        "rows": [
            {
                "run_id": stage["run_id"],
                "time_s": 0.1,
                "field": "h_m",
                "region": "whole",
                "rmse": {"value": 0.1, "reason": None},
            }
        ],
        "runs": [
            {
                "run_id": stage["run_id"],
                "mass": {
                    "learned_initial_volume_error_m2": 0.1,
                    "left_boundary_mismatch_m2_s": [0.0, 0.1],
                    "base_resolution": {"volume_m2": [5.5, 5.6]},
                    "sensitivity": {
                        "differences": {
                            "signed_balance_m2": {"doubled_minus_base": [0.0, 0.01]}
                        }
                    },
                    "auxiliary_records": [{"run_path": "parent-path-is-provenance"}],
                },
            }
        ],
    }
    newer = copy.deepcopy(metric)
    newer["provenance"] = {"execution": fresh_execution}
    newer["rows"][0]["run_id"] = fresh_execution["run_ids"][stage["run_id"]]
    newer["runs"][0]["run_id"] = newer["rows"][0]["run_id"]
    newer["runs"][0]["mass"]["auxiliary_records"] = [
        {"run_path": "fresh-path-is-provenance"}
    ]
    assert reproduce._compare_metrics(metric, newer, parent, protocol)[
        "within_tolerance"
    ]
    metric["rows"][0]["rmse"] = {
        "value": None,
        "reason": f"failure; evidence: runs/{stage['run_id']}",
    }
    newer["rows"][0]["rmse"] = {
        "value": None,
        "reason": f"failure; evidence: runs/{newer['rows'][0]['run_id']}",
    }
    assert reproduce._compare_metrics(metric, newer, parent, protocol)[
        "within_tolerance"
    ]
    mass = newer["runs"][0]["mass"]
    mass["base_resolution"]["volume_m2"][1] += 0.1
    mass["learned_initial_volume_error_m2"] += 0.1
    mass["left_boundary_mismatch_m2_s"][1] += 0.1
    mass["sensitivity"]["differences"]["signed_balance_m2"]["doubled_minus_base"][
        1
    ] += 0.1
    compared = reproduce._compare_metrics(metric, newer, parent, protocol)
    assert not compared["within_tolerance"]
    mass_comparison = compared["conservation"][0]["comparison"]["fields"]
    assert not mass_comparison["base_resolution"]["within_tolerance"]
    assert not mass_comparison["learned_initial_volume_error_m2"]["within_tolerance"]
    assert not mass_comparison["left_boundary_mismatch_m2_s"]["within_tolerance"]
    assert not mass_comparison["sensitivity"]["within_tolerance"]


@pytest.mark.parametrize("changed_module", ["conservation", "benchmark"])
def test_additive_evaluation_source_drift_stops_before_retraining(
    project, monkeypatch, changed_module
):
    from hydro_pinn import evaluation_workflow

    root, protocol = project
    manifest_path = root / protocol["paths"]["manifest"]
    manifest = json.loads(manifest_path.read_text())
    monkeypatch.setattr(
        evaluation_workflow, "validated_manifest", lambda *args: (manifest, protocol)
    )
    provenance = {
        "protocol_sha256": protocol["protocol_sha256"],
        "manifest_sha256": reproduce._sha(manifest_path),
    }
    modules = ["evaluation_workflow", "evaluate", "conservation"]
    metrics = {
        "state": "completed",
        "provenance": {
            **provenance,
            "source_files": [],
            "evaluation_source_sha256": {
                f"src/hydro_pinn/{name}.py": reproduce._sha(
                    root / f"src/hydro_pinn/{name}.py"
                )
                for name in modules
            },
        },
    }
    timing = {
        "state": "completed",
        "provenance": {
            **provenance,
            "source_sha256": {
                "src/hydro_pinn/benchmark.py": reproduce._sha(
                    root / "src/hydro_pinn/benchmark.py"
                ),
            },
        },
    }
    (root / "results").mkdir()
    write_json_atomic(root / "results/metrics.json", metrics)
    write_json_atomic(root / "results/timings.json", timing)
    target = root / f"src/hydro_pinn/{changed_module}.py"
    target.write_text(target.read_text() + "\n# changed fixture\n")
    with pytest.raises(ValueError, match="additive implementation changed"):
        reproduce._result_dependencies(protocol, root)
    assert not (root / "runs").exists()


def test_timing_figure_integration_preserves_core_and_writes_complete_ledger(
    tmp_path, monkeypatch
):
    """Only the shared utility invocation is mocked; no rendering or timing runs."""
    root = tmp_path.resolve()
    output = root / "results/paper-fixture"
    figures = output / "fresh_figures"
    figures.mkdir(parents=True)
    sources = {}
    for name in ("src/hydro_pinn/plot.py", reproduce.TIMING_FIGURE_HELPER):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# Explicit integration fixture; not the accepted renderer.\n")
        sources[name] = reproduce._sha(path)
    provenance = {
        "protocol_sha256": "a" * 64,
        "protocol_file_sha256": "b" * 64,
        "manifest_sha256": "c" * 64,
    }
    run_ids = [f"fixture-run-{index}" for index in range(11)]
    metrics = {
        "state": "completed",
        "provenance": provenance,
        "runs": [{"run_id": run_id} for run_id in run_ids],
    }
    write_json_atomic(output / "fresh_metrics.json", metrics)
    write_json_atomic(output / "fresh_timings.json", {"state": "completed"})
    core = {"plot_source_sha256": sources["src/hydro_pinn/plot.py"], "figures": []}
    for index in range(34):
        artifacts = []
        for suffix in ("png", "pdf"):
            path = figures / f"core-{index}.{suffix}"
            path.write_bytes(f"fixture-{index}-{suffix}".encode())
            artifacts.append(
                {"path": str(path.relative_to(root)), "sha256": reproduce._sha(path)}
            )
        core["figures"].append({"figure_id": f"core-{index}", "artifacts": artifacts})
    write_json_atomic(output / "fresh_figure_manifest.json", core)
    write_json_atomic(figures / "figure_manifest.json", core)
    before = reproduce._sha(output / "fresh_figure_manifest.json")
    calls = []

    def utility(command, **kwargs):
        calls.append(command)
        assert command[0] == reproduce.sys.executable
        assert command[1] == str(root / reproduce.TIMING_FIGURE_HELPER)
        assert kwargs == {
            "cwd": root,
            "capture_output": True,
            "text": True,
            "check": False,
        }
        assert (
            json.loads((output / "fresh_timings.json").read_text())["state"]
            == "completed"
        )
        artifacts = []
        for suffix in ("png", "pdf"):
            path = figures / f"timing_accuracy.{suffix}"
            path.write_bytes(f"mock-utility-{suffix}".encode())
            artifacts.append(
                {"path": str(path.relative_to(root)), "sha256": reproduce._sha(path)}
            )
        evidence = {
            "state": "completed",
            "figure_id": "timing_accuracy",
            "kind": "timing_and_accuracy",
            "points": [{"run_id": run_ids[index % 11]} for index in range(13)],
            "run_ids": run_ids,
            "artifacts": artifacts,
            "provenance": provenance,
            "plot_source_sha256": {
                reproduce.TIMING_FIGURE_HELPER: sources[reproduce.TIMING_FIGURE_HELPER]
            },
            "performance_measurements_executed_by_renderer": False,
            "source_artifacts": [
                {
                    "path": str((output / name).relative_to(root)),
                    "sha256": reproduce._sha(output / name),
                }
                for name in ("fresh_metrics.json", "fresh_timings.json")
            ],
        }
        write_json_atomic(output / "fresh_timing_figure_evidence.json", evidence)
        return SimpleNamespace(
            returncode=0, stdout="mocked utility invocation", stderr=""
        )

    monkeypatch.setattr(reproduce.subprocess, "run", utility)
    report = reproduce._timing_figure_complete_manifest(root, output, sources)
    complete = json.loads((output / "fresh_figure_manifest_complete.json").read_text())
    assert len(calls) == 1
    assert len(complete["figures"]) == 35
    assert sum(len(row["artifacts"]) for row in complete["figures"]) == 70
    assert reproduce._sha(output / "fresh_figure_manifest.json") == before
    assert reproduce._sha(figures / "figure_manifest.json") == before
    assert report["count"] == 35 and report["exports"] == 70
    assert report["visual_inspection"] == "pending_root_review"
    assert complete["supplemental_plot_source_sha256"] == {
        reproduce.TIMING_FIGURE_HELPER: sources[reproduce.TIMING_FIGURE_HELPER]
    }
    with pytest.raises(FileExistsError):
        reproduce._timing_figure_complete_manifest(root, output, sources)
    assert len(calls) == 1
