"""Bounded saved-evidence plot workflow and preservation check."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hydro_pinn.plot import plot_report


def test_saved_fixture_figures_keep_failures_provenance_and_exact_times(
    tmp_path: Path,
) -> None:
    x, times = np.linspace(-1, 1, 7), np.array([0.1, 0.5])
    reference_h = np.tile(np.linspace(1, 0.1, 7), (2, 1))
    datasets, rows = [], []
    descriptors = [
        ("fixture_reference", "analytical_reference", None, None, None, 0),
        ("fixture_fixed", "pinn", "baseline", 11, None, 0.1),
        ("fixture_adaptive", "pinn", "adaptive", 11, None, 0.2),
        ("fixture_solver", "lax_wendroff", None, None, 100, 0.05),
    ]
    for run_id, method, variant, seed, cells, offset in descriptors:
        source = tmp_path / f"{run_id}.npz"
        np.savez(
            source,
            x_m=x,
            t_s=times,
            h_m=reference_h + offset,
            q_m2_s=np.full((2, 7), offset),
        )
        with np.load(source, allow_pickle=False) as saved:
            data = {key: saved[key] for key in saved.files}
        data.update(
            run_id=run_id,
            case_id="E1",
            method=method,
            variant=variant,
            seed=seed,
            cells=cells,
            state="completed",
            reason=None,
            resolved_config_sha256="fixture-config-sha",
            provenance={
                "source_files": [
                    {
                        "path": str(source),
                        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    }
                ]
            },
            mass={
                "t_s": [0, 0.1, 0.5],
                "normalized_signed_balance": [0, offset, offset],
                "boundary_net_outflow_m2": [0, 0, 0],
                "boundary_t_s": [0.1, 0.5],
                "left_boundary_q_m2_s": [0, offset],
                "right_boundary_q_m2_s": [0, 0],
                "boundary_values_are": "numerical mass interface flux"
                if method == "lax_wendroff"
                else "physical endpoint discharge",
            },
        )
        if method == "pinn":
            data["history"] = [
                {
                    "step": step,
                    "loss_pde": 1 / step,
                    "loss_ic": 2 / step,
                    "loss_bc": 3 / step,
                    "lambda_ic": 10,
                    "lambda_bc": 10,
                }
                for step in (1, 2)
            ]
        datasets.append(data)
        if method == "analytical_reference":
            continue
        for time in times:
            for field in ("h_m", "q_m2_s", "u_m_s"):
                for region in ("whole", "wave_band", "outside_wave_band"):
                    rows.append(
                        {
                            "run_id": run_id,
                            "case_id": "E1",
                            "field": field,
                            "time_s": float(time),
                            "region": region,
                            "rmse": {"value": offset, "reason": None},
                            "relative_l2": {
                                "value": None,
                                "reason": "zero reference norm",
                            }
                            if field == "q_m2_s"
                            else {"value": offset, "reason": None},
                            "coverage_fraction": 1 if region == "whole" else 0.5,
                        }
                    )
    datasets.append(
        {
            "run_id": "fixture_failed",
            "case_id": "E1",
            "method": "lax_wendroff",
            "cells": 200,
            "state": "failed",
            "reason": "Negative predictor depth; fixture",
        }
    )
    report = {
        "purpose": "bounded_fixture_not_final_results",
        "provenance": {
            "protocol_sha256": "fixture-protocol",
            "manifest_sha256": "fixture-manifest",
        },
        "evaluation": {"profile_times_s": [0.1, 0.25, 0.5]},
        "rows": rows,
    }
    output = tmp_path / "figures"
    manifest = plot_report(report, datasets, output)
    persisted = json.loads((output / "figure_manifest.json").read_text())
    assert persisted == manifest
    assert manifest["purpose"] == "bounded_fixture_not_final_results"
    assert manifest["provenance"] == report["provenance"]
    assert manifest["missing_profile_outputs"] == [
        {"run_id": item[0], "time_s": 0.25} for item in descriptors
    ]
    assert "fixture_failed" in manifest["figures"][0]["run_ids"]
    for figure in manifest["figures"][1:]:
        assert "fixture_failed" not in figure["run_ids"]
    assert {figure["kind"] for figure in manifest["figures"]} >= {
        "profiles",
        "depth_and_error_map",
        "per_time_errors",
        "mass_boundary",
        "training_components",
        "grid_comparison",
        "paired_ablation",
        "velocity_mask_coverage",
        "comparison_status",
    }
    for figure in manifest["figures"]:
        for artifact in figure["artifacts"]:
            path = Path(artifact["path"])
            assert path.stat().st_size > 1000
            assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact["sha256"]
    before = hashlib.sha256((output / "figure_manifest.json").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        plot_report(report, datasets, output)
    assert (
        hashlib.sha256((output / "figure_manifest.json").read_bytes()).hexdigest()
        == before
    )
