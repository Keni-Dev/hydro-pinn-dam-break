"""Bounded timing arithmetic, synchronization, and fresh-work checks."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from hydro_pinn import benchmark
from hydro_pinn.config import load_config
from hydro_pinn.model import ShallowWaterPINN
from hydro_pinn.protocol import ROOT


def test_samples_quantiles_synchronization_and_failed_coverage():
    events = []
    clock_values = iter([0, 1, 2, 4, 5, 8, 9, 13])

    def work():
        events.append("work")
        return np.ones((2, 2))

    def validate(value):
        events.append("validate")
        benchmark._validate_output(value, (2, 2))

    timing = benchmark.measure_workload(
        work,
        validate,
        warmups=1,
        repetitions=4,
        clock=lambda: next(clock_values),
        sync=lambda device: events.append(f"sync:{device}"),
    )
    assert events == ["sync:cpu", "work", "sync:cpu", "validate"] * 5
    assert timing["state"] == "completed"
    assert timing["raw_samples_s"] == [1, 2, 3, 4]
    assert timing["median_s"] == 2.5
    assert timing["iqr_s"] == 1.5
    assert timing["warmup_repetitions_completed"] == 1
    attempts = 0

    def failing_work():
        nonlocal attempts
        attempts += 1
        if attempts == 2:
            raise ValueError("constructed workload failure")
        return np.ones((2, 2))

    failed = benchmark.measure_workload(
        failing_work, validate, warmups=0, repetitions=3
    )
    assert failed["state"] == "failed"
    assert failed["measured_repetitions"] == 1
    assert len(failed["raw_samples_s"]) == 1
    assert failed["median_s"] is None
    assert benchmark.speed_ratio(timing, failed)["value"] is None
    assert benchmark.speed_ratio(timing, {"state": "completed", "median_s": 0})[
        "reason"
    ]


def test_solver_reintegrates_and_reconstructs_both_physical_fields(monkeypatch):
    calls = []
    native = SimpleNamespace(
        failure=None,
        t_s=np.array([0.1, 0.2]),
        cell_edges_m=np.array([-1.0, 0.0, 1.0]),
        h_m=np.array([[1.0, 2.0], [3.0, 4.0]]),
        q_m2_s=np.array([[5.0, 6.0], [7.0, 8.0]]),
    )

    def integrate(case, cells, times):
        calls.append((case, cells, times.copy()))
        return native

    monkeypatch.setattr(benchmark, "solve", integrate)
    work = benchmark.solver_workload(
        {"fixture": True}, 2, np.array([-0.5, 0.0, 0.5]), native.t_s
    )
    first, second = work(), work()
    assert len(calls) == 2
    assert first.shape == second.shape == (2, 3, 2)
    np.testing.assert_array_equal(first[..., 0], [[1, 2, 2], [3, 4, 4]])
    np.testing.assert_array_equal(first[..., 1], [[5, 6, 6], [7, 8, 8]])
    native.failure = {"reason": "constructed invalid intermediate depth"}
    with pytest.raises(benchmark.BenchmarkError, match="integration failed"):
        work()


def test_network_workload_has_complete_h_q_and_chunk_equivalence():
    case = load_config(ROOT / "configs/cases/e1_wet.json")
    training = load_config(ROOT / "configs/training/baseline.json")
    training["network"]["hidden_layers"] = [4, 4]
    model = ShallowWaterPINN(case, training, dtype="float64")
    coordinates = torch.tensor(
        [[-1.0, 0.1], [0.0, 0.1], [1.0, 0.2]], dtype=torch.float64
    )
    whole = benchmark.network_workload(model, coordinates)()
    chunked = benchmark.network_workload(model, coordinates, 2)()
    benchmark._validate_output(whole, (3, 2))
    assert whole.requires_grad is False
    torch.testing.assert_close(whole, chunked, rtol=1e-13, atol=1e-13)


def test_production_gate_and_terminal_manifest_requirement(tmp_path):
    with pytest.raises(benchmark.BenchmarkError, match="orchestrator readiness"):
        benchmark.benchmark_protocol("absent.json", "timings.json")
    path = tmp_path / "manifest.json"
    path.write_text(
        '{"protocol_id":"fixture","protocol_sha256":"fixture",'
        '"purpose":"fixture","workflow_status":"running"}'
    )
    protocol = {
        "protocol_id": "fixture",
        "protocol_sha256": "fixture",
        "purpose": "fixture",
    }
    with pytest.raises(benchmark.BenchmarkError, match="complete terminal matrix"):
        benchmark._verified_manifest(protocol, path, tmp_path)
    assert not (tmp_path / "timings.json").exists()


def test_labeled_orchestration_fixture_uses_namespaced_checkpoint_and_cost_scopes(
    tmp_path, monkeypatch
):
    """Exercise orchestration with synthetic outputs, never a performance result."""
    case = load_config(ROOT / "configs/cases/e1_wet.json")
    training = load_config(ROOT / "configs/training/baseline.json")
    training["network"]["hidden_layers"] = [4]
    source = {"case": case, "training": training}
    model = ShallowWaterPINN(case, training, dtype="float32")
    evaluation = load_config(ROOT / "configs/evaluation.json")
    x, times, _ = benchmark.common_coordinates(evaluation)
    stages = [
        {"stage_id": kind, "stage": kind, "case_id": "E1", "run_id": kind}
        for kind in ("reference", "solve", "train", "predict")
    ]
    stages[1]["cells"] = 2  # Synthetic two-cell cache, not a frozen production grid.
    stages[2].update(variant="baseline", seed=11)
    stages[3].update(variant="baseline", seed=11, training_stage_id="train")
    protocol = {
        "purpose": "software_fixture_not_performance_results",
        "protocol_id": "fixture",
        "protocol_sha256": "fixture-only",
        "scientific_source_sha256": {},
        "paths": {"output_root": "runs", "manifest": "manifest.json"},
        "settings": {
            "device": "cpu",
            "dtype": "float32",
            "cpu_threads": 1,
            "chunk_size": 5000,
        },
        "resolved": {"cases": {"E1": case}, "evaluation": evaluation},
        "benchmark": {"warmup_repetitions": 10, "measured_repetitions": 30},
        "matrix": stages,
    }
    protocol_path, manifest_path = (
        tmp_path / "protocol.json",
        tmp_path / "manifest.json",
    )
    protocol_path.write_text(json.dumps(protocol))
    manifest_path.write_text('{"fixture":true}')
    verified = {}
    entries = []
    for stage in stages:
        name = f"replay-{stage['stage']}"
        path = tmp_path / "runs/paper" / name
        path.mkdir(parents=True)
        verified[stage["stage_id"]] = {"run_path": str(path.relative_to(tmp_path))}
        entries.append(
            {"stage_id": stage["stage_id"], "state": "completed", "actual_run_id": name}
        )
    training_path = tmp_path / "runs/paper/replay-train"
    (training_path / "metadata.json").write_text('{"execution":{"fixture":true}}')
    (training_path / "history.csv").write_text("fixture\n")
    (training_path / "checkpoint.pt").write_text("synthetic checkpoint identity only\n")
    solver_path = tmp_path / "runs/paper/replay-solve"
    edges = np.array([-5.0, 0.0, 5.0])
    h, q = np.ones((50, 2)), np.zeros((50, 2))
    np.savez(solver_path / "outputs.npz", cell_edges_m=edges, h_m=h, q_m2_s=q)
    manifest = {
        "entries": entries,
        "execution": {"output_root": "runs/paper"},
    }
    monkeypatch.setattr(benchmark, "load_protocol", lambda *a, **kw: protocol)
    monkeypatch.setattr(
        benchmark, "_verified_manifest", lambda *a: (manifest, verified)
    )
    monkeypatch.setattr(
        benchmark, "collect_metadata", lambda *a, **kw: {"fixture": True}
    )
    monkeypatch.setattr(
        benchmark,
        "solve",
        lambda *a: SimpleNamespace(
            failure=None, t_s=times, cell_edges_m=edges, h_m=h, q_m2_s=q
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "evaluate_reference",
        lambda *a: SimpleNamespace(
            h_m=np.ones((50, len(x))), q_m2_s=np.zeros((50, len(x)))
        ),
    )
    checkpoint_calls = []

    def load_checkpoint(run_id, output_root, **kwargs):
        checkpoint_calls.append((run_id, output_root, kwargs))
        return {"selected_model_state": model.state_dict()}, source, training_path

    monkeypatch.setattr(benchmark, "load_training_checkpoint", load_checkpoint)
    measured_calls = []

    def fixture_measure(work, validate, **kwargs):
        measured_calls.append(kwargs)
        validate(work())
        return {
            "state": "completed",
            "median_s": 1.0,
            "purpose": "synthetic fixture; no timing samples measured",
        }

    monkeypatch.setattr(benchmark, "measure_workload", fixture_measure)
    output = tmp_path / "fixture_timings.json"
    report = benchmark.benchmark_protocol(
        protocol_path, output, project_root=tmp_path, operationally_ready=True
    )
    assert checkpoint_calls == [
        ("replay-train", tmp_path / "runs/paper", {"require_completed": True})
    ]
    assert (
        len(measured_calls) == 6
    )  # Reference, fresh/cache, NN, transfer, host export.
    assert all(
        call["warmups"] == 10 and call["repetitions"] == 30 for call in measured_calls
    )
    assert (
        report["source_protocol_purpose"] == "software_fixture_not_performance_results"
    )
    categories = {row["category"] for row in report["costs"]}
    assert "network_forward_and_host_export" in categories
    assert "saved_actual_training_and_persistence" in categories
    assert report["primary_comparator"]["alternative_grid_selected"] is False
    assert report["ratios"][0]["numerical_median_over_network_median"]["value"] is None
    assert (
        json.loads(output.read_text())["source_protocol_purpose"] == protocol["purpose"]
    )
