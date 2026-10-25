"""Focused viscosity workflow contracts; no pilot, fit or final query is run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hydro_pinn.cli import build_parser
from hydro_pinn.config import load_config
from hydro_pinn.extensions import viscosity_experiment as experiment
from hydro_pinn.protocol import _conservation
from hydro_pinn.sampling import load_samples

ROOT = Path(__file__).resolve().parents[1]
INITIAL_IDENTITY = "827083c31cbc04bae49f321d52e0071d9aedcf2f6f5b2384ffa82cb420c26d33"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def coordinate_pairs(axes: dict, scales: dict, dtype=np.float64) -> set[tuple]:
    x, t = np.meshgrid(axes["x_m"], axes["t_s"])
    points = np.column_stack((x.ravel(), t.ravel()))
    points = (points / [scales["Lc_m"], scales["Tc_s"]]).astype(dtype)
    return {tuple(row) for row in points}


def test_axes_match_frozen_core_counts_and_partition_membership() -> None:
    evaluation_path = ROOT / "configs/evaluation.json"
    evaluation = load_config(evaluation_path)
    axes = experiment.build_axes(evaluation, sha(evaluation_path))
    core = _conservation([], evaluation, sha(evaluation_path))
    assert axes["volume_axes"] == core["volume_axes"]
    assert axes["boundary_axes"] == core["boundary_axes"]
    primary = axes["primary"]
    assert np.array_equal(
        primary["x_m"], [-5 + (i + 0.5) * 10 / 400 for i in range(400)]
    )
    assert primary["t_s"] == evaluation["times_s"]
    primary_count = len(primary["x_m"]) * len(primary["t_s"])
    volume_count = sum(
        len(v["x_m"]) * len(v["t_s"]) for v in axes["volume_axes"].values()
    )
    boundary_count = sum(
        len(v["x_m"]) * len(v["t_s"]) for v in axes["boundary_axes"].values()
    )
    assert (primary_count, volume_count, boundary_count) == (20000, 306000, 6004)
    assert primary_count + volume_count + boundary_count == 332004
    assert [len(v["t_s"]) for v in axes["volume_axes"].values()] == [51, 51]
    assert [len(v["t_s"]) for v in axes["boundary_axes"].values()] == [1001, 2001]
    case = load_config(ROOT / "configs/cases/e1_wet.json")
    historical = ROOT / "runs/p11_hybrid_full_v01_control"
    partitions = [
        load_samples(historical / name) for name in ("sampling.npz", "validation.npz")
    ]
    for dtype in (np.float64, np.float32):
        fitting_or_validation = {
            tuple(row)
            for samples in partitions
            for name in ("interior_xt_nd", "initial_xt_nd", "boundary_xt_nd")
            for row in samples[name].astype(dtype)
        }
        final_points = coordinate_pairs(primary, case["scales"], dtype)
        assert len(final_points) == 20000
        assert final_points.isdisjoint(fitting_or_validation)
    validation_boundary = {tuple(row) for row in partitions[1]["boundary_xt_nd"]}
    # Auxiliary endpoint diagnostics intentionally revisit ten known validation
    # BC coordinates. They are declared physics probes, not held-out labels.
    for boundary in axes["boundary_axes"].values():
        overlaps = coordinate_pairs(boundary, case["scales"]) & validation_boundary
        assert len(overlaps) == 10
        assert set([0.0, *evaluation["times_s"]]).issubset(set(boundary["t_s"]))


def test_historical_binding_and_fresh_initial_parameter_identity() -> None:
    checkpoint, resolved, directory = experiment.load_historical(ROOT)
    selection = json.loads(
        (ROOT / "docs/coordination/extension_viscosity_selection.json").read_text()
    )
    for filename, digest in selection["historical_control"]["binding_sha256"].items():
        assert sha(ROOT / filename) == digest
    assert directory == ROOT / "runs/p11_hybrid_full_v01_control"
    assert checkpoint["step"] == 10000 and checkpoint["selected_step"] == 6512
    assert resolved["extension"]["variant"] == "control"
    case = load_config(ROOT / "configs/cases/e1_wet.json")
    training = load_config(ROOT / "configs/training/baseline.json")
    model = experiment.initial_model(case, training)
    assert experiment.initial_hash(model) == INITIAL_IDENTITY
    assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert torch.get_num_threads() == 4
    assert any(
        not torch.equal(value, checkpoint["selected_model_state"][name])
        for name, value in model.state_dict().items()
    )


def test_viscosity_two_named_cli_commands() -> None:
    parser = build_parser()
    run = parser.parse_args(
        [
            "extension-viscosity-run",
            "--config",
            "viscosity.json",
            "--manifest",
            "pilot.json",
        ]
    )
    assert run.command == "extension-viscosity-run"
    evaluation = parser.parse_args(
        [
            "extension-viscosity-evaluate",
            "--manifest",
            "pilot.json",
            "--output-dir",
            "eval",
        ]
    )
    assert evaluation.command == "extension-viscosity-evaluate"


def test_pilot_quality_gate_refuses_before_model_or_reference_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "pilot.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "completed",
                "mode": "pilot",
                "purpose": "exploratory_single_seed_viscosity",
                "steps": 25,
            }
        )
    )
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError("pilot quality evaluation must stop before any query")

    for name in (
        "evaluate_reference",
        "initial_model",
        "ShallowWaterPINN",
        "query_checkpoint",
        "_ve_query",
        "_ve_npz",
    ):
        monkeypatch.setattr(experiment, name, forbidden, raising=False)
    output = tmp_path / "forbidden_evaluation"
    with pytest.raises(ValueError, match="pilot|full|release"):
        experiment.evaluate_experiment(manifest, output, root=tmp_path)
    assert calls == []
    assert not output.exists()
