"""Focused checks for generated-label provenance, partitions and fit isolation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from hydro_pinn.evaluate import evaluation_axes
from hydro_pinn.extensions.hybrid_data import (
    array_sha256,
    file_sha256,
    generate_labels,
    load_label_manifest,
    load_training_labels,
    verify_label_integrity,
)
from hydro_pinn.references import evaluate_reference

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config() -> dict:
    return {"case": str(ROOT / "configs/cases/e1_wet.json"), "labels": {}}


def test_generated_source_units_membership_and_readback(
    tmp_path: Path, config: dict
) -> None:
    run = generate_labels(config, "labels", root=tmp_path)
    manifest = load_label_manifest(run)
    assert manifest["data_origin"] == "analytical_generated"
    assert manifest["field_observations"] is False
    assert manifest["units"] == {
        "x_m": "m",
        "t_s": "s",
        "h_m": "m",
        "q_m2_s": "m^2/s",
    }
    assert manifest["source"]["source_sha256"] == file_sha256(
        ROOT / "src/hydro_pinn/references.py"
    )
    assert manifest["source"]["versioned_references"][0]["version"]
    assert manifest["label_rng"]["seed"] == 110211
    assert manifest["scales"]["Q_m2_s"] == pytest.approx(np.sqrt(9.81))
    assert all(verify_label_integrity(run).values())
    train = load_training_labels(run)
    assert train["xt_m_s"].shape == train["h_q"].shape == (256, 2)
    xt = train["xt_m_s"]
    assert np.all((0 < xt[:, 1]) & (xt[:, 1] < 0.35))
    with np.load(run / "validation_labels.npz", allow_pickle=False) as validation:
        times = validation["xt_m_s"][:, 1]
        assert times.shape == (128,)
        assert np.all((0.35 <= times) & (times < 0.40))
    with np.load(run / "final_axes.npz", allow_pickle=False) as final:
        assert set(final.files) == {"x_m", "t_s"}
        assert final["x_m"].size == 400
        expected_x, _ = evaluation_axes(
            {
                "grid": {"kind": "cell_centres", "domain_m": [-5, 5], "cells": 400},
                "times_s": np.linspace(0.40, 0.50, 11).tolist(),
            }
        )
        assert_array_equal(final["x_m"], expected_x)
        assert_array_equal(final["t_s"], np.linspace(0.40, 0.50, 11))
    reference = evaluate_reference(manifest["case"], xt[:, 0], xt[:, 1])
    assert_array_equal(train["h_q"][:, 0], np.diag(reference.h_m))
    assert_array_equal(train["h_q"][:, 1], np.diag(reference.q_m2_s))
    # Generated labels include both physical undisturbed reservoir states.
    assert np.any(train["h_q"][:, 0] == 1.0)
    assert np.any(train["h_q"][:, 0] == 0.1)
    assert np.all(train["h_q"][xt[:, 0] < -2.0] == [1.0, 0.0])
    assert manifest["generation_workload"]["final_reference_points"] == 0
    assert manifest["generation_workload"]["saved_label_count"] == 384
    assert manifest["cost_s"]["analytical_label_evaluation"] >= 0
    assert json.loads((run / "status.json").read_text())["state"] == "completed"


def test_rng_is_independent_and_exact_membership_repeats(
    tmp_path: Path, config: dict
) -> None:
    np.random.seed(8765)
    state_before = np.random.get_state()
    first = generate_labels(config, "first", root=tmp_path)
    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    assert_array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]
    second = generate_labels(config, "second", root=tmp_path)
    a, b = load_label_manifest(first), load_label_manifest(second)
    for partition in ("training", "validation", "final"):
        assert (
            a["partitions"][partition]["coordinate_sha256"]
            == (b["partitions"][partition]["coordinate_sha256"])
        )
    assert array_sha256(load_training_labels(first)["h_q"]) == array_sha256(
        load_training_labels(second)["h_q"]
    )


def test_training_loader_never_opens_validation_or_final(
    tmp_path: Path,
    config: dict,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = generate_labels(config, "isolated", root=tmp_path)
    real_load = np.load
    opened = []

    def guarded_load(path: Path, *args: object, **kwargs: object) -> object:
        opened.append(Path(path).name)
        assert Path(path).name == "train_labels.npz"
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded_load)
    assert load_training_labels(run)["h_q"].shape == (256, 2)
    assert opened == ["train_labels.npz"]


def test_existing_runs_and_corrupted_labels_are_not_accepted(
    tmp_path: Path, config: dict
) -> None:
    run = generate_labels(config, "preserved", root=tmp_path)
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    with pytest.raises(FileExistsError):
        generate_labels(config, "preserved", root=tmp_path)
    assert before == {p.name: p.read_bytes() for p in run.iterdir()}
    # Corrupt only this temporary test run, preserving all production artifacts.
    with (run / "train_labels.npz").open("ab") as handle:
        handle.write(b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_training_labels(run)
