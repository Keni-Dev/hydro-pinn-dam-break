"""Targeted device diagnostics, declared tolerances and real pilot workflow."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from hydro_pinn.devices import (
    _backend_reason,
    _read_pilot_config,
    compare_arrays,
    run_pilot,
    summarize_timings,
    write_doctor,
)

ROOT = Path(__file__).resolve().parents[1]


def test_doctor_reports_observed_and_reported_facts(tmp_path: Path) -> None:
    path = tmp_path.resolve() / "doctor.json"
    report = write_doctor(path)
    assert json.loads(path.read_text()) == report
    assert report["observed"]["backends"]["cpu"]["available"] is True
    assert report["observed"]["storage"]["free_bytes"] > 0
    assert report["observed"]["memory"]["process_lifetime_peak_rss_bytes"] > 0
    assert report["reported_hardware"]["victus"]
    assert report["observed"]["cpu_threads"] > 0


def test_mps_float64_and_fallback_are_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    assert "does not support float64" in _backend_reason("mps", "float64")
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    assert "fallback enabled" in _backend_reason("mps", "float32")


def test_declared_elementwise_tolerance_and_zero_reference() -> None:
    tolerance = {"atol": 1e-6, "rtol": 1e-3}
    accepted = compare_arrays(np.array([1.0005, 1e-7]), np.array([1.0, 0.0]), tolerance)
    rejected = compare_arrays(np.array([1.0005, 1e-4]), np.array([1.0, 0.0]), tolerance)
    assert accepted["passed"] and not rejected["passed"]
    zero = compare_arrays(np.zeros(2), np.zeros(2), tolerance)
    assert zero["relative_l2_difference"] is None and zero["relative_l2_reason"]


def test_warmup_and_adaptive_update_timing_are_distinct() -> None:
    events = [
        {
            "step": 99,
            "compute_s": 8.0,
            "checkpoint_io_s": 1.0,
            "total_s": 9.0,
            "weights_updated": False,
        },
        {
            "step": 100,
            "compute_s": 2.0,
            "checkpoint_io_s": 0.2,
            "total_s": 2.2,
            "weights_updated": True,
        },
    ]
    result = summarize_timings(events, 99)
    assert result["warmup_total_s"] == 9.0
    assert result["compute_s"]["mean"] == 2.0
    assert result["weights_updated_steps"] == [100]
    assert result["updated_step_timings"][0]["step"] == 100


def test_representative_adaptive_requires_real_step100(tmp_path: Path) -> None:
    config = json.loads((ROOT / "configs/pilot.json").read_text())
    config["workloads"] = [
        next(
            copy.deepcopy(w) for w in config["workloads"] if w["variant"] == "adaptive"
        )
    ]
    config["workloads"][0]["steps"] = 20
    path = tmp_path.resolve() / "bad_pilot.json"
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="step100"):
        _read_pilot_config(path)


def test_cpu_pilot_precision_chunks_training_reload_and_preservation(
    tmp_path: Path,
) -> None:
    directory = tmp_path.resolve()
    config = json.loads((ROOT / "configs/pilot.json").read_text())
    config["cpu_threads"] = 1
    config["case"] = str(ROOT / config["case"])
    config["training"] = {k: str(ROOT / v) for k, v in config["training"].items()}
    config["output_root"] = str(directory / "runs")
    config["precision"]["candidates"] = [
        {"device": "cpu", "dtype": "float64", "run_id": "precision64"},
        {"device": "cpu", "dtype": "float32", "run_id": "precision32"},
        {"device": "mps", "dtype": "float64", "run_id": "unsupported64"},
    ]
    config["workloads"] = [
        {
            "run_id": "tiny",
            "kind": "smoke",
            "variant": "baseline",
            "device": "cpu",
            "dtype": "float64",
            "steps": 2,
            "warmup_steps": 1,
            "chunk_size": 5,
            "sampling": {
                "interior_points": 7,
                "initial_points": 6,
                "boundary_points_per_side": 3,
            },
        }
    ]
    path = directory / "pilot.json"
    path.write_text(json.dumps(config))
    output = directory / "findings.json"
    report = run_pilot(path, output)
    assert [r["passed"] for r in report["precision"]] == [True, True, False]
    unsupported = directory / "runs/unsupported64/status.json"
    assert json.loads(unsupported.read_text())["state"] == "failed"
    workload = report["workloads"][0]
    assert workload["outcome"] == "completed_finite_budget"
    assert workload["actual_steps"] == 2
    assert workload["timings"]["measured_steps"] == 1
    assert workload["checkpoint_reload"]["exact_agreement"]
    assert workload["actual_device"] == "cpu"
    assert workload["timings"]["total_s"]["mean"] > 0
    assert report["final_evaluation_consumed"] is False
    prior = (directory / "runs/tiny/checkpoint.pt").read_bytes()
    with pytest.raises(FileExistsError):
        run_pilot(path, directory / "second_findings.json")
    assert (directory / "runs/tiny/checkpoint.pt").read_bytes() == prior
