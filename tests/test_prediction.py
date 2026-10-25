"""Trusted selected-checkpoint reload, SI state conversion, and safe records."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose

from hydro_pinn.config import load_config
from hydro_pinn.model import ShallowWaterPINN
from hydro_pinn.train import (
    TrainingError,
    load_training_checkpoint,
    predict_run,
    train_run,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "configs/training/baseline.json"
SMOKE = ROOT / "configs/smoke.json"


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    """Use the physical temporary path under the record no-symlink contract."""
    return tmp_path.resolve()


def train_fixture(run_root: Path, case_name: str = "e1_wet") -> Path:
    return train_run(
        ROOT / f"configs/cases/{case_name}.json",
        BASELINE,
        11,
        "cpu",
        "float64",
        "training",
        smoke_path=SMOKE,
        output_root=run_root,
        overrides={
            "training": {
                "network": {"hidden_layers": [4, 4]},
                "sampling": {
                    "interior_points": 7,
                    "initial_points": 6,
                    "boundary_points_per_side": 3,
                },
                "optimizer": {"steps": 2},
                "validation": {
                    "grid": {"cells": 7},
                    "times_s": [0.023, 0.193],
                    "profile_times_s": [0.193],
                },
            }
        },
    )


@pytest.mark.parametrize("case_name", ["e1_wet", "e2_dry"])
def test_selected_checkpoint_prediction_matches_physical_reload(
    run_root: Path, case_name: str
) -> None:
    training_path = train_fixture(run_root, case_name)
    checkpoint, resolved, _ = load_training_checkpoint(
        "training", run_root, require_completed=True
    )
    source_hash = hashlib.sha256(
        (training_path / "checkpoint.pt").read_bytes()
    ).hexdigest()
    path = predict_run("training", SMOKE, "prediction", run_root)
    status = json.loads((path / "status.json").read_text())
    metadata = json.loads((path / "metadata.json").read_text())
    prediction_config = json.loads((path / "resolved_config.json").read_text())
    assert status["state"] == "completed"
    assert prediction_config["evaluation"]["partition"] == "smoke_evaluation"
    provenance = prediction_config["prediction_source"]
    assert provenance["training_run_id"] == "training"
    assert provenance["checkpoint_sha256"] == source_hash
    assert provenance["selected_step"] == checkpoint["selected_step"]
    assert metadata["stage_metadata"]["state_location"] == "point_value"
    assert metadata["stage_metadata"]["array_units"] == {
        "x_m": "m",
        "t_s": "s",
        "h_m": "m",
        "q_m2_s": "m^2/s",
    }
    with np.load(path / "outputs.npz", allow_pickle=False) as arrays:
        x, times = arrays["x_m"], arrays["t_s"]
        h, q = arrays["h_m"], arrays["q_m2_s"]
        assert h.shape == q.shape == (4, 64)
        assert h.dtype == q.dtype == np.float64
        assert np.array_equal(times, load_config(SMOKE)["evaluation"]["times_s"])
        assert np.isfinite(h).all() and np.isfinite(q).all()
        assert np.all(
            h
            > resolved["case"]["scales"]["H_m"]
            * resolved["training"]["network"]["depth_floor_nd"]
        )
        xx, tt = np.meshgrid(x, times)
        points = torch.tensor(
            np.column_stack((xx.ravel(), tt.ravel())), dtype=torch.float64
        )
        model = ShallowWaterPINN(resolved["case"], resolved["training"])
        model.load_state_dict(checkpoint["selected_model_state"])
        with torch.no_grad():
            physical = model.physical_state(points).numpy().reshape(4, 64, 2)
            scaled_points = torch.stack(
                (points[:, 0] / model.scales.Lc_m, points[:, 1] / model.scales.Tc_s),
                dim=1,
            )
            raw = model.network(scaled_points)
            physical_velocity = (raw[:, 1] * model.scales.Uc_m_s).numpy().reshape(4, 64)
        assert_allclose(h, physical[..., 0], atol=0, rtol=0)
        assert_allclose(q, physical[..., 1], atol=0, rtol=0)
        assert_allclose(q, h * physical_velocity, atol=3e-16, rtol=2e-15)
    if case_name == "e2_dry":
        assert resolved["case"]["right_depth_m"] == 0.0
        with np.load(training_path / "sampling.npz", allow_pickle=False) as samples:
            assert np.any(samples["initial_hq_nd"][:, 0] == 0)
    assert (
        hashlib.sha256((training_path / "checkpoint.pt").read_bytes()).hexdigest()
        == source_hash
    )


def test_prediction_preserves_outputs_and_rejects_untrusted_source(
    run_root: Path,
) -> None:
    training_path = train_fixture(run_root)
    path = predict_run("training", SMOKE, "prediction", run_root)
    output_hash = hashlib.sha256((path / "outputs.npz").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        predict_run("training", SMOKE, "prediction", run_root)
    assert (
        hashlib.sha256((path / "outputs.npz").read_bytes()).hexdigest() == output_hash
    )
    with pytest.raises(ValueError, match="run_id"):
        predict_run("../outside", SMOKE, "unsafe-source", run_root)
    assert not (run_root / "unsafe-source").exists()
    with pytest.raises(ValueError, match="run_id"):
        predict_run("training", SMOKE, "../unsafe-output", run_root)
    assert not (run_root.parent / "unsafe-output").exists()
    with (training_path / "checkpoint.pt").open("ab") as handle:
        handle.write(b"fixture-corruption")
    with pytest.raises(TrainingError, match="SHA256 mismatch"):
        predict_run("training", SMOKE, "corrupt-source", run_root)
    assert not (run_root / "corrupt-source").exists()
