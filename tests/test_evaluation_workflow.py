"""Bounded report plumbing; final-run evaluation remains separately gated."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from hydro_pinn import evaluation_workflow as workflow
from hydro_pinn.config import load_config
from hydro_pinn.evaluate import EvaluationError, evaluation_axes
from hydro_pinn.references import evaluate_reference

ROOT = Path(__file__).resolve().parents[1]


def test_production_evaluation_requires_readiness_and_terminal_manifest(
    tmp_path: Path,
) -> None:
    root = tmp_path.resolve()
    with pytest.raises(EvaluationError, match="readiness"):
        workflow.evaluate_manifest("missing.json", "metrics.json", project_root=root)
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"workflow_status": "running"}))
    with pytest.raises(EvaluationError, match="complete terminal"):
        workflow.validated_manifest(manifest, root)
    assert not (root / "metrics.json").exists()


def test_fixture_report_keeps_failed_rows_and_actual_seed_coverage(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path.resolve()
    case = load_config(ROOT / "configs/cases/e1_wet.json")
    evaluation = load_config(ROOT / "configs/evaluation.json")
    evaluation["grid"]["cells"] = 4
    evaluation["times_s"] = [0.1, 0.5]
    evaluation["profile_times_s"] = [0.1, 0.5]
    x, times = evaluation_axes(evaluation)
    reference = evaluate_reference(case, x, times)
    reference_fields = {
        key: getattr(reference, key) for key in ("x_m", "t_s", "h_m", "q_m2_s")
    }
    common = {
        "case_id": "E1",
        "cells": None,
        "variant": None,
        "seed": None,
        "state": "completed",
        "reason": None,
    }
    datasets = [
        {
            **reference_fields,
            **common,
            "run_id": "fixture-reference",
            "stage_id": "reference",
            "method": "analytical_reference",
        }
    ]
    for variant, offset in [("baseline", 0.1), ("adaptive", 0.2)]:
        datasets.append(
            {
                **reference_fields,
                **common,
                "run_id": f"fixture-{variant}",
                "stage_id": variant,
                "variant": variant,
                "seed": 11,
                "method": "pinn",
                "h_m": reference.h_m + offset,
            }
        )
    datasets.append(
        {
            **common,
            "run_id": "fixture-failed",
            "stage_id": "failed",
            "method": "lax_wendroff",
            "cells": 400,
            "state": "failed",
            "reason": "fixture numerical failure",
        }
    )
    # Fixture report plumbing receives checked fields; actual-record loading and
    # conservation checkpoint forward passes are tested in their own modules.
    (root / "protocol.json").write_text("{}")
    (root / "manifest.json").write_text('{"purpose":"bounded fixture"}')
    manifest = {"protocol_path": "protocol.json", "execution": {"namespace": "fixture"}}
    protocol = {
        "protocol_id": "fixture",
        "protocol_sha256": "fixture-sha",
        "resolved": {"cases": {"E1": case}, "evaluation": evaluation},
        "settings": {"cpu_threads": 1, "seeds": [11, 22, 33], "e2_diagnostic_seed": 11},
    }
    monkeypatch.setattr(
        workflow, "validated_manifest", lambda *args: (manifest, protocol)
    )
    monkeypatch.setattr(
        workflow, "saved_datasets", lambda *args: copy.deepcopy(datasets)
    )
    monkeypatch.setattr(
        workflow, "_conservation_for_source", lambda *args: {"fixture": True}
    )
    monkeypatch.setattr(workflow, "_source_files", lambda *args: [])
    report = workflow.evaluate_manifest(
        "manifest.json",
        "results/metrics.json",
        project_root=root,
        operationally_ready=True,
    )
    failed = [row for row in report["rows"] if row["run_id"] == "fixture-failed"]
    assert report["state"] == "incomplete_required_comparison"
    assert report["scientific_status"]["E1_classical_comparator"] == "missing"
    assert report["aggregation"]["E2"]["paired_ablation"] == []
    assert report["aggregation"]["E2"]["paired_ablation_status"].startswith(
        "not_planned"
    )
    assert len(failed) == 18
    assert all(
        row["rmse"] == {"value": None, "reason": "fixture numerical failure"}
        for row in failed
    )
    fixed = next(
        row
        for row in report["aggregation"]["E1"]["groups"]
        if row["variant"] == "baseline"
        and row["field"] == "h_m"
        and row["region"] == "whole"
    )
    assert fixed["rmse"]["defined_seed_count"] == 1
    assert fixed["rmse"]["expected_seed_count"] == 3
    assert fixed["rmse"]["missing_seeds"] == [22, 33]
    assert fixed["rmse"]["mean"]["value"] == pytest.approx(0.1)
    assert (root / "results/metrics.csv").is_file()
    raw = (root / "results/metrics.json").read_text()
    assert "NaN" not in raw and "Infinity" not in raw
    with pytest.raises(FileExistsError):
        workflow.evaluate_manifest(
            "manifest.json",
            "results/metrics.json",
            project_root=root,
            operationally_ready=True,
        )
