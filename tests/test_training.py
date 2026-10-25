"""Tiny optimizer, complete gradients, resume, and truthful terminal records."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose

from hydro_pinn.config import load_config
from hydro_pinn.model import ShallowWaterPINN
from hydro_pinn.sampling import generate_training_samples, load_samples
from hydro_pinn.train import (
    TrainingError,
    component_gradients,
    load_training_checkpoint,
    train_run,
)

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "configs/cases/e1_wet.json"
BASELINE = ROOT / "configs/training/baseline.json"
ADAPTIVE = ROOT / "configs/training/adaptive.json"
SMOKE = ROOT / "configs/smoke.json"


@pytest.fixture
def run_root(tmp_path: Path) -> Path:
    """Use the physical temporary path under the record no-symlink contract."""
    return tmp_path.resolve()


def tiny_overrides(steps: int) -> dict:
    return {
        "training": {
            "network": {"hidden_layers": [4, 4]},
            "sampling": {
                "interior_points": 7,
                "initial_points": 6,
                "boundary_points_per_side": 3,
            },
            "optimizer": {"steps": steps},
            "validation": {
                "grid": {"cells": 7},
                "times_s": [0.023, 0.193],
                "profile_times_s": [0.193],
            },
        }
    }


def train_tiny(run_root: Path, run_id: str, steps: int, **kwargs: Any) -> Path:
    return train_run(
        CASE,
        BASELINE,
        11,
        "cpu",
        "float64",
        run_id,
        smoke_path=SMOKE,
        output_root=run_root,
        overrides=tiny_overrides(steps),
        **kwargs,
    )


def assert_same_tree(first: Any, second: Any) -> None:
    if isinstance(first, torch.Tensor):
        assert_allclose(first, second, rtol=0, atol=1e-12)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for name in first:
            assert_same_tree(first[name], second[name])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for one, two in zip(first, second, strict=True):
            assert_same_tree(one, two)
    else:
        assert first == second


def test_optimizer_latest_state_and_compatible_resume(run_root: Path) -> None:
    full_path = train_tiny(run_root, "whole", 4)
    short_path = train_tiny(run_root, "short", 2)
    source_hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in short_path.iterdir()
        if p.is_file()
    }
    resumed_path = train_tiny(run_root, "resumed", 4, resume_from="short")
    full, resolved, _ = load_training_checkpoint(
        "whole", run_root, require_completed=True
    )
    short, _, _ = load_training_checkpoint("short", run_root, require_completed=True)
    resumed, resumed_config, _ = load_training_checkpoint(
        "resumed", run_root, require_completed=True
    )
    assert full["step"] == resumed["step"] == 4 and short["step"] == 2
    assert_same_tree(full["latest_model_state"], resumed["latest_model_state"])
    assert_same_tree(full["optimizer_state"], resumed["optimizer_state"])
    assert_same_tree(full["weights"], resumed["weights"])
    assert_same_tree(full["rng_state"], resumed["rng_state"])
    torch.manual_seed(11)
    initial = ShallowWaterPINN(resolved["case"], resolved["training"])
    assert any(
        not torch.equal(value, initial.state_dict()[name])
        for name, value in full["latest_model_state"].items()
    )
    assert full["selected_step"] <= full["step"]
    assert set(full["selected_model_state"]) == set(full["latest_model_state"])
    assert full["selection_rule"] == "minimum_separate_validation_physics_ic_bc_loss"
    assert full["weight_update_state"]["next_update_before_step"] == 100
    ancestry = resumed_config["resume_ancestry"]
    assert ancestry["source_run_id"] == "short" and ancestry["resume_step"] == 2
    assert ancestry["budget_extension_steps"] == 2
    assert ancestry["requested_total_adam_steps"] == 4
    for name, digest in source_hashes.items():
        assert hashlib.sha256((short_path / name).read_bytes()).hexdigest() == digest
    for name in ("sampling.npz", "validation.npz"):
        original = load_samples(short_path / name)
        continued = load_samples(resumed_path / name)
        for key in original:
            assert np.array_equal(original[key], continued[key])
    with (full_path / "history.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["step"]) for row in rows] == [1, 2, 3, 4]
    assert all(row["weights_updated"] == "False" for row in rows)
    assert all(np.isfinite(float(row["loss_total"])) for row in rows)
    assert "convergence not established" in rows[-1]["stop_reason"]
    assert json.loads((full_path / "status.json").read_text())["state"] == "completed"


def test_full_and_uneven_chunk_component_gradients() -> None:
    case, training = load_config(CASE), load_config(BASELINE)
    patch = tiny_overrides(2)["training"]
    training["network"].update(patch["network"])
    training["sampling"].update(patch["sampling"])
    torch.manual_seed(11)
    model = ShallowWaterPINN(case, training)
    samples = generate_training_samples(case, training, 11)
    for name in ("pde", "ic", "bc"):
        full_value, full_gradients = component_gradients(model, samples, name)
        chunk_value, chunk_gradients = component_gradients(
            model, samples, name, chunk_size=2
        )
        assert_allclose(chunk_value, full_value, atol=3e-14, rtol=2e-14)
        for full, chunk in zip(full_gradients, chunk_gradients, strict=True):
            assert_allclose(chunk, full, atol=3e-14, rtol=2e-14)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_incompatible_corrupt_and_overwrite_refused_before_new_run(
    run_root: Path,
) -> None:
    path = train_tiny(run_root, "source", 2)
    checkpoint_hash = hashlib.sha256((path / "checkpoint.pt").read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        train_tiny(run_root, "source", 2)
    assert (
        hashlib.sha256((path / "checkpoint.pt").read_bytes()).hexdigest()
        == checkpoint_hash
    )
    with pytest.raises(TrainingError, match="incompatible resume training"):
        train_run(
            CASE,
            ADAPTIVE,
            11,
            "cpu",
            "float64",
            "wrong-weighting",
            smoke_path=SMOKE,
            resume_from="source",
            output_root=run_root,
            overrides=tiny_overrides(4),
        )
    assert not (run_root / "wrong-weighting").exists()
    with (path / "checkpoint.pt").open("ab") as handle:
        handle.write(b"fixture-corruption")
    with pytest.raises(TrainingError, match="SHA256 mismatch"):
        train_tiny(run_root, "corrupt-resume", 4, resume_from="source")
    assert not (run_root / "corrupt-resume").exists()


@pytest.mark.parametrize("failure", ["nonfinite", "interruption"])
def test_failure_preserves_initial_finite_checkpoint(
    run_root: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    original = component_gradients

    def fail_during_first_step(model, samples, component, chunk_size=None):
        if failure == "interruption":
            raise KeyboardInterrupt("injected training interruption")
        value, gradients = original(model, samples, component, chunk_size)
        gradients[0].fill_(float("nan"))
        return value, gradients

    monkeypatch.setattr("hydro_pinn.train.component_gradients", fail_during_first_step)
    error = KeyboardInterrupt if failure == "interruption" else ValueError
    with pytest.raises(error):
        train_tiny(run_root, failure, 2)
    path = run_root / failure
    status = json.loads((path / "status.json").read_text())
    assert status["state"] == ("interrupted" if failure == "interruption" else "failed")
    assert (path / "checkpoint.pt").is_file()
    checkpoint, _, _ = load_training_checkpoint(failure, run_root)
    assert checkpoint["step"] == 0
    assert all(
        bool(torch.isfinite(value).all())
        for value in checkpoint["latest_model_state"].values()
    )
    with (path / "history.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows[-1]["phase"] == "failure" and rows[-1]["stop_reason"]
    assert not (path / "outputs.npz").exists()


def test_optional_lbfgs_records_repeated_fixed_closures(run_root: Path) -> None:
    overrides = tiny_overrides(1)
    overrides["training"]["lbfgs"] = {
        "enabled": True,
        "max_iterations": 2,
        "history_size": 5,
    }
    path = train_run(
        CASE,
        BASELINE,
        11,
        "cpu",
        "float64",
        "lbfgs",
        output_root=run_root,
        overrides=overrides,
        chunk_size=2,
    )
    checkpoint, _, _ = load_training_checkpoint(
        "lbfgs", run_root, require_completed=True
    )
    diagnostics = json.loads((path / "training_diagnostics.json").read_text())
    info = diagnostics["lbfgs"]
    assert checkpoint["lbfgs_completed"] is True
    assert info["closure_calls"] >= 2
    assert info["function_evaluations"] == info["closure_calls"]
    assert info["iterations"] <= info["max_iterations"] == 2
    assert info["history_size"] == 5 and info["line_search"] is None
    assert info["weights_frozen"] == checkpoint["weights"]
    assert "convergence not inferred" in info["stopping_reason"]
    assert (
        checkpoint["partition_hashes"]["sampling.npz"]
        == hashlib.sha256((path / "sampling.npz").read_bytes()).hexdigest()
    )
