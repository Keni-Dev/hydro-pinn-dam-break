"""Focused matched-control checks without final analytical accuracy inspection."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.testing import assert_allclose, assert_array_equal

from hydro_pinn.cli import build_parser
from hydro_pinn.config import load_config
from hydro_pinn.extensions import hybrid_experiment as hybrid
from hydro_pinn.extensions.hybrid_data import (
    file_sha256,
    generate_labels,
    load_training_labels,
)

ROOT = Path(__file__).resolve().parents[1]


def test_full_workload_zero_data_equivalence_to_frozen_baseline() -> None:
    case = load_config(ROOT / "configs/cases/e1_wet.json")
    training = load_config(ROOT / "configs/training/baseline.json")
    report = hybrid.equivalence_report(case, training)
    assert report["passed"]
    assert report["workload"]["interior_points"] == 10000
    assert report["workload"]["initial_points"] == 1000
    assert report["workload"]["boundary_points_per_side"] == 500
    assert report["initial_parameters_bitwise_equal"]
    assert report["objective_exact_equal"]
    assert report["all_gradients_bitwise_equal"]
    assert report["maximum_absolute_gradient_difference"] == 0
    assert report["gradient_parameter_count"] == 10
    assert report["all_parameters_after_one_adam_bitwise_equal"]
    assert report["adam_state_bitwise_equal"]
    assert report["control_payload"] is None
    assert report["baseline_source_sha256"] == file_sha256(
        ROOT / "src/hydro_pinn/train.py"
    )


def test_scaled_h_q_loss_sums_fields_then_averages_label_points() -> None:
    case = load_config(ROOT / "configs/cases/e1_wet.json")
    scales = case["scales"]
    q_scale = scales["H_m"] * scales["Uc_m_s"]
    labels = {
        "xt_m_s": np.array([[2.5, 0.1], [-2.5, 0.2]], dtype=np.float64),
        "h_q": np.array([[1.0, 2 * q_scale], [4.0, 6 * q_scale]]),
    }
    payload = hybrid.scaled_label_payload(labels, case)
    assert_allclose(
        payload["xt_nd"],
        labels["xt_m_s"] / np.array([scales["Lc_m"], scales["Tc_s"]]),
        rtol=0,
        atol=0,
    )
    assert_allclose(payload["hq_nd"], [[1.0, 2.0], [4.0, 6.0]], rtol=2e-16)
    model = torch.nn.Linear(2, 2, dtype=torch.float64)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    loss = hybrid.data_loss(model, payload)
    assert loss.item() == pytest.approx(28.5)  # Sum fields, average points.
    loss.backward()
    torch.testing.assert_close(
        model.bias.grad, torch.tensor([-5.0, -8.0]).double(), rtol=2e-15, atol=0
    )
    with pytest.raises(ValueError, match="training-window"):
        hybrid.scaled_label_payload(
            {**labels, "xt_m_s": np.array([[0.0, 0.35], [0.0, 0.40]])}, case
        )


def test_physics_only_objective_rejects_any_interior_label_payload() -> None:
    model = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="no interior-label payload"):
        hybrid.objective_gradients(
            model,
            samples=None,
            weights={"pde": 1.0, "ic": 10.0, "bc": 10.0},
            lambda_data=0.0,
            payload={},
        )


def test_three_named_cli_registrations() -> None:
    parser = build_parser()
    requests = [
        ["extension-hybrid-labels", "--config", "hybrid.json", "--run-id", "labels"],
        [
            "extension-hybrid-run",
            "--config",
            "hybrid.json",
            "--labels-run",
            "labels",
            "--manifest",
            "pair.json",
            "--pilot-steps",
            "1",
        ],
        [
            "extension-hybrid-evaluate",
            "--manifest",
            "pair.json",
            "--output-dir",
            "evaluation",
        ],
    ]
    for argv in requests:
        args = parser.parse_args(argv)
        assert args.command == argv[0]


def test_one_step_pair_checkpoint_readback_and_label_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = json.loads((ROOT / "configs/extensions/hybrid.json").read_text())
    config["case"] = str(ROOT / "configs/cases/e1_wet.json")
    config["training"] = str(ROOT / "configs/training/baseline.json")
    config_path = tmp_path / "hybrid.json"
    config_path.write_text(json.dumps(config))
    # Freeze hashes of real source bytes inside this isolated temporary fixture.
    shutil.copytree(
        ROOT / "src/hydro_pinn",
        tmp_path / "src/hydro_pinn",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    label_dir = generate_labels(config_path, "labels", root=tmp_path)
    expected_training_labels = load_training_labels(label_dir)
    opened, training_calls = [], []
    real_load, real_training_loader = np.load, hybrid.load_training_labels

    def guarded_np_load(path, *args, **kwargs):
        filename = Path(path).name
        opened.append(filename)
        assert filename not in {"validation_labels.npz", "final_axes.npz"}
        return real_load(path, *args, **kwargs)

    def counted_training_loader(directory):
        training_calls.append(Path(directory))
        return real_training_loader(directory)

    with monkeypatch.context() as patch:
        patch.setattr(np, "load", guarded_np_load)
        patch.setattr(hybrid, "load_training_labels", counted_training_loader)
        manifest_path = hybrid.run_pair(
            config_path,
            "labels",
            "one_step_pair.json",
            pilot_steps=1,
            root=tmp_path,
        )
    assert training_calls == [label_dir]
    assert "train_labels.npz" in opened
    manifest = json.loads(manifest_path.read_text())
    assert manifest["state"] == "completed" and manifest["mode"] == "pilot"
    assert manifest["steps_per_variant"] == 1
    assert manifest["final_analytical_accuracy_inspected"] is False
    assert all(manifest["matched_readback"].values())
    protocol_path = tmp_path / manifest["protocol"]["path"]
    assert file_sha256(protocol_path) == manifest["protocol"]["sha256"]
    protocol = json.loads(protocol_path.read_text())
    assert protocol["source_sha256"]
    assert protocol["selection"]["data_validation_used"] is False
    assert protocol["selection"]["final_reference_used"] is False
    checkpoints = []
    for entry in manifest["entries"]:
        checkpoint, resolved, directory = hybrid.load_hybrid_checkpoint(
            entry["run_id"], root=tmp_path
        )
        checkpoints.append(checkpoint)
        assert checkpoint["step"] == 1
        assert 0 <= checkpoint["selected_step"] <= 1
        assert checkpoint["selection_rule"] == protocol["selection"]["rule"]
        assert checkpoint["selection_weights"] == protocol["selection"]["weights"]
        assert file_sha256(directory / "checkpoint.pt") == entry["checkpoint_sha256"]
        for name, digest in checkpoint["partition_hashes"].items():
            assert file_sha256(directory / name) == digest
        metadata = json.loads((directory / "metadata.json").read_text())
        stage = metadata["stage_metadata"]
        assert stage["validation_solution_labels_used"] is False
        assert stage["final_solution_values_generated"] is False
        with (directory / "history.csv").open(newline="") as handle:
            history = list(csv.DictReader(handle))
        assert len(history) == 1 and history[0]["step"] == "1"
        extension = resolved["extension"]
        if entry["variant"] == "control":
            assert extension["lambda_data"] == 0
            assert extension["labels_run"] is None
            assert extension["fitting_payload"] is None
            assert not (directory / "training_labels.npz").exists()
            assert history[0]["loss_data"] == "0.0"
            assert stage["interior_labels"] is False
        else:
            assert extension["lambda_data"] == 1
            assert stage["interior_labels"] is True
            with np.load(
                directory / "training_labels.npz", allow_pickle=False
            ) as labels:
                assert set(labels.files) == {"xt_m_s", "h_q"}
                for key, expected in expected_training_labels.items():
                    assert_array_equal(labels[key], expected)
                assert np.all(labels["xt_m_s"][:, 1] < 0.35)
    first, second = checkpoints
    assert first["initial_model_sha256"] == second["initial_model_sha256"]
    for name in ("sampling.npz", "validation.npz"):
        assert first["partition_hashes"][name] == second["partition_hashes"][name]

    # A pilot must be refused before any final analytical reference is requested.
    def forbidden_reference(*args, **kwargs):
        raise AssertionError("pilot must not generate final analytical values")

    import hydro_pinn.references as references

    monkeypatch.setattr(references, "evaluate_reference", forbidden_reference)
    monkeypatch.setattr(
        hybrid, "evaluate_reference", forbidden_reference, raising=False
    )
    with pytest.raises(ValueError, match="pilot|full"):
        hybrid.evaluate_pair(manifest_path, tmp_path / "forbidden_eval", root=tmp_path)
    assert not (tmp_path / "forbidden_eval").exists()
