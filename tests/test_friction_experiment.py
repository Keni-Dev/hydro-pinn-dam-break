"""Focused friction protocol, comparison-operator and persistence checks.

These fixtures perform no PINN training and do not inspect pilot accuracy.
"""

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from hydro_pinn.extensions.friction_experiment import (
    digest,
    evaluate_friction,
    final_memberships,
    load_friction_config,
    mass_workload,
    model_identity,
    partition_receipt,
    query_cell_averages,
    run_friction,
    sha,
    validate_reference_packet,
    verify_protocol,
    verify_saved_training,
)
from hydro_pinn.model import ShallowWaterPINN
from hydro_pinn.sampling import (
    generate_training_samples,
    generate_validation_samples,
)

CONFIG = Path("configs/extensions/friction_experiment.json")


def _inputs():
    return load_friction_config(CONFIG)


class PolynomialState(nn.Module):
    """Known SI polynomial averages; no neural parameters or fitting."""

    def physical_state(self, coordinates):
        # Return64 after the production float32 coordinate conversion to isolate
        # quadrature arithmetic from polynomial evaluation rounding.
        x, t = coordinates.double().unbind(1)
        return torch.stack((2 + x.square() + t, 3 + 2 * x + t), dim=1)


class ConservedState(nn.Module):
    """h_t+q_x=0 with two nonzero boundary fluxes and initial-volume bias."""

    def physical_state(self, coordinates):
        x, t = coordinates.double().unbind(1)
        return torch.stack((1 - 0.2 * t, 2 + 0.2 * x + 0.3 * t), dim=1)


def test_fixed_design_reuses_core_samples_and_keeps_all_query_memberships_disjoint(
    tmp_path,
):
    cfg, case, training = _inputs()
    changed = copy.deepcopy(cfg)
    changed["manning_n_s_m_minus_one_third"] = 0.04
    altered = tmp_path / "altered.json"
    altered.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="fixed friction v01"):
        load_friction_config(altered)
    train = generate_training_samples(case, training, cfg["sampling_seed"])
    val = generate_validation_samples(case, training, cfg["sampling_seed"])
    axes = final_memberships(cfg, case, train, val)
    receipt = partition_receipt(train, val, axes, case)
    assert train["interior_xt_nd"].shape == (10000, 2)
    assert train["initial_xt_nd"].shape == (1000, 2)
    assert train["boundary_xt_nd"].shape == (1000, 2)
    assert not receipt["interior_labels"]
    for dtype in ("float64", "float32"):
        assert set(receipt["membership"][dtype]["pairwise_overlap"].values()) == {0}
    for level in (0, 1):
        bt = axes[f"boundary_t_{level}_s"]
        assert np.all(np.diff(bt) > 0)
        assert bt[0] == 0 and bt[-1] == 0.5
        assert np.isin(axes["report_t_s"], bt).all()
    # A real coordinate collision must fail instead of being labelled held out.
    damaged = {k: v.copy() for k, v in val.items()}
    damaged["interior_xt_nd"][0] = train["interior_xt_nd"][0]
    with pytest.raises(ValueError, match="partition overlap"):
        partition_receipt(train, damaged, axes, case)


def test_gauss_cell_averages_use_physical_cells_and_preserve_orientation():
    # h=2+x²+t has mean 2+(a²+ab+b²)/3+t on each cell. Both orders
    # integrate it exactly up to disclosed float32 query-coordinate rounding.
    edges = np.array([-2.0, -0.5, 1.0, 3.0])
    centres = (edges[:-1] + edges[1:]) / 2
    times = np.array([0.125, 0.375])
    axes = {"t_s": times}
    for order in (3, 5):
        z, w = np.polynomial.legendre.leggauss(order)
        axes[f"gauss_x_{order}_m"] = centres[:, None] + np.diff(edges)[:, None] * z / 2
        axes[f"gauss_w_{order}"] = w / 2
        answer = query_cell_averages(PolynomialState(), axes, order, chunk=4)
        expected_h = (
            2
            + (edges[:-1] ** 2 + edges[:-1] * edges[1:] + edges[1:] ** 2) / 3
            + times[:, None]
        )
        expected_q = 3 + 2 * centres[None, :] + times[:, None]
        assert answer.shape == (2, 3, 2)
        np.testing.assert_allclose(answer[..., 0], expected_h, atol=5e-7, rtol=0)
        np.testing.assert_allclose(answer[..., 1], expected_q, atol=5e-7, rtol=0)
        # Off-centre quadratic cells cannot silently be interpreted as point h.
        assert np.max(np.abs(answer[..., 0] - (2 + centres**2 + times[:, None]))) > 0.1


def test_both_boundary_fluxes_correct_mass_and_keep_initial_bias_separate(tmp_path):
    times = np.array([0.0, 0.125, 0.25, 0.5])
    axes = {"report_t_s": times}
    cfg = {"evaluation": {"query_chunk_size": 7}}
    for level, cells in enumerate((8, 16)):
        axes[f"mass_x_{level}_m"] = -5 + (np.arange(cells) + 0.5) * 10 / cells
        axes[f"boundary_t_{level}_s"] = np.unique(
            np.r_[np.linspace(0, 0.5, 9 * (level + 1)), times]
        )
        report = mass_workload(ConservedState(), axes, cfg, tmp_path, level)
        np.testing.assert_allclose(report["volume_m2"], 10 - 2 * times, atol=1e-14)
        np.testing.assert_allclose(report["outward_integral_m2"], 2 * times, atol=1e-8)
        assert report["predicted_initial_bias_m2"] == pytest.approx(4.5)
        np.testing.assert_allclose(report["corrected_balance_m2"], 4.5, atol=1e-8)
        np.testing.assert_allclose(
            report["balance_without_initial_bias_m2"], 0, atol=1e-8
        )
        assert report["boundary"]["left"]["signed_outward_integral_m2"] < 0
        assert report["boundary"]["right"]["signed_outward_integral_m2"] > 0
        with np.load(tmp_path / f"boundaries_{level}.npz") as raw:
            assert raw["q_m2_s"].shape == (len(raw["t_s"]), 2)
            np.testing.assert_array_equal(raw["x_m"], [-5, 5])


def _fixture_protocol(tmp_path):
    run = tmp_path / "fixture_run"
    run.mkdir()
    sections = {
        "source_sha256": {"source.py": "source"},
        "input_sha256": {"config.json": "input"},
        "reference_sha256": {"reference.npz": "reference"},
        "membership_sha256": {"members.npz": "membership"},
    }
    protocol = {}
    for section, files in sections.items():
        protocol[section] = {}
        for name, content in files.items():
            path = (run if section == "membership_sha256" else tmp_path) / name
            path.write_text(content)
            protocol[section][name] = sha(path)
    protocol["protocol_sha256"] = digest(protocol)
    path = run / "protocol.json"
    path.write_text(json.dumps(protocol))
    return path, protocol


def test_frozen_protocol_rejects_source_input_reference_membership_and_content_changes(
    tmp_path,
):
    path, protocol = _fixture_protocol(tmp_path)
    assert verify_protocol(path, tmp_path) == protocol
    for target in (
        tmp_path / "source.py",
        tmp_path / "config.json",
        tmp_path / "reference.npz",
        path.parent / "members.npz",
    ):
        original = target.read_bytes()
        target.write_bytes(original + b"changed")
        with pytest.raises(ValueError, match="changed"):
            verify_protocol(path, tmp_path)
        target.write_bytes(original)
    protocol["added"] = "unhashed mutation"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="content hash mismatch"):
        verify_protocol(path, tmp_path)


def test_pilot_accuracy_and_unmatched_full_release_are_refused_before_fitting(tmp_path):
    run = tmp_path / "runs/p08_friction_pilot_fixture"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(
        json.dumps({"mode": "pilot", "actual_updates": 25})
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"state": "completed", "run": str(run.relative_to(tmp_path))})
    )
    with pytest.raises(ValueError, match="pilot refused"):
        evaluate_friction(manifest, "uncreated_accuracy", root=tmp_path)
    assert not (tmp_path / "uncreated_accuracy").exists()
    cfg, _, _ = _inputs()
    for source in (CONFIG, Path(cfg["case"]), Path(cfg["training_template"])):
        destination = tmp_path / source
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    protocol = {
        "config": cfg,
        "source_sha256": {},
        "input_sha256": {},
        "reference_sha256": {},
        "membership_sha256": {},
    }
    protocol["protocol_sha256"] = digest(protocol)
    (tmp_path / "protocol.json").write_text(json.dumps(protocol))
    (tmp_path / "release.json").write_text(
        json.dumps({"state": "released", "protocol_sha256": "different"})
    )
    with pytest.raises(ValueError, match="technical release/config mismatch"):
        run_friction(
            CONFIG,
            "uncreated_full_manifest.json",
            run_id="p08_friction_unreleased_fixture",
            root=tmp_path,
            protocol_path=Path("protocol.json"),
            release_path=Path("release.json"),
        )
    assert not (tmp_path / "uncreated_full_manifest.json").exists()
    assert not (tmp_path / "runs/p08_friction_unreleased_fixture").exists()


def test_saved_history_keeps_initial_state_and_earliest_strict_minimum(tmp_path):
    # Synthetic state/optimizer/history receipts only. No optimizer is run.
    _, case, training = _inputs()
    torch.manual_seed(11)
    model = ShallowWaterPINN(case, training, dtype="float32")
    identity = model_identity(model)
    protocol = {
        "case_template": case,
        "training_template": training,
        "protocol_sha256": "synthetic",
        "membership_sha256": {"synthetic": "not a fit"},
        "initial_identity": identity,
    }
    scores = [1.0, 1.0, 0.5, 0.5]
    history = tmp_path / "history.csv"

    def write_history(rows):
        with history.open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=("step", "val_objective", "selected")
            )
            writer.writeheader()
            writer.writerows(rows)

    rows = [
        {"step": i, "val_objective": score, "selected": i in (0, 2)}
        for i, score in enumerate(scores)
    ]
    write_history(rows)
    for name, step in (("initial.pt", 0), ("checkpoint.pt", 2), ("latest.pt", 3)):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001).state_dict()
        if step:
            optimizer["state"] = {
                i: {
                    "step": torch.tensor(float(step)),
                    "exp_avg": torch.zeros_like(p),
                    "exp_avg_sq": torch.zeros_like(p),
                }
                for i, p in enumerate(model.parameters())
            }
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer,
                "step": step,
                "validation_score": scores[step],
                "protocol_sha256": protocol["protocol_sha256"],
                "membership_sha256": protocol["membership_sha256"],
                "initial_identity": identity,
            },
            tmp_path / name,
        )
    summary = {
        "mode": "pilot",
        "actual_updates": 3,
        "selected_step": 2,
        "selected_validation_score": 0.5,
        "checkpoint_write_counts": {"initial": 1, "selected": 2, "latest": 4},
        "checkpoint_sha256": sha(tmp_path / "checkpoint.pt"),
        "history_sha256": sha(history),
    }
    assert (
        verify_saved_training(tmp_path, summary, protocol)[
            "earliest_strict_minimum_step"
        ]
        == 2
    )
    rows[3]["selected"] = True
    write_history(rows)
    summary["history_sha256"] = sha(history)
    with pytest.raises(ValueError, match="earliest strict minimum"):
        verify_saved_training(tmp_path, summary, protocol)
    rows[3]["selected"] = False
    write_history(rows[1:])
    summary["history_sha256"] = sha(history)
    with pytest.raises(ValueError, match="initial and every update"):
        verify_saved_training(tmp_path, summary, protocol)


def test_reference_gate_rejects_incomplete_and_nonfinite_summary(tmp_path):
    """A hashed file and completed status cannot make a NaN budget acceptable."""
    cfg, _, _ = _inputs()
    packet = tmp_path / cfg["reference_packet"]
    packet.mkdir(parents=True)
    dependencies = {
        "configs/extensions/friction_reference.json": "reference configuration",
        cfg["case"]: "case template",
        "src/hydro_pinn/extensions/friction_reference.py": "reference source",
    }
    for name, text in dependencies.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    procedure = {
        "config_sha256": sha(tmp_path / "configs/extensions/friction_reference.json"),
        "case_sha256": sha(tmp_path / cfg["case"]),
        "source_sha256": sha(
            tmp_path / "src/hydro_pinn/extensions/friction_reference.py"
        ),
    }
    (packet / "protocol.json").write_text(json.dumps(procedure))
    np.savez(
        packet / "common_reference.npz", h_m=np.ones((2, 2)), q_m2_s=np.zeros((2, 2))
    )
    runs = []
    for i, (cells, cfl) in enumerate(
        ((800, 0.45), (1600, 0.45), (3200, 0.45), (3200, 0.225))
    ):
        run = packet / f"fixture_{i}"
        run.mkdir()
        np.savez(run / "outputs.npz", h_m=np.ones((2, 2)), q_m2_s=np.zeros((2, 2)))
        runs.append(
            {
                "id": run.name,
                "cells": cells,
                "cfl": cfl,
                "outputs_sha256": sha(run / "outputs.npz"),
                "minimum_h_m": 0.1,
                "max_mass_budget_defect_m2": 0.0,
                "max_momentum_budget_defect_m3_s": 0.0,
                "solve_wall_s": 0.01,
            }
        )
    summary = {
        "state": "complete_reference_with_measured_uncertainty",
        "reference_is_ground_truth": False,
        "common_sha256": sha(packet / "common_reference.npz"),
        "runs": runs,
        "refinement": [
            {"h_m": {"rmse": value}, "q_m2_s": {"rmse": value}}
            for value in (0.03, 0.02, 0.01)
        ],
        "uncertainty_h_m": {"rmse": 0.02, "max_abs": 0.03},
        "uncertainty_q_m2_s": {"rmse": 0.02, "max_abs": 0.03},
    }
    summary_path = packet / "summary.json"
    summary_path.write_text(json.dumps(summary))
    status_path = packet / "status.json"
    status_path.write_text(json.dumps({"state": "failed"}))
    with pytest.raises(ValueError, match="incomplete or unsupported"):
        validate_reference_packet(cfg, tmp_path)
    status_path.write_text(json.dumps({"state": "completed"}))
    assert (
        validate_reference_packet(cfg, tmp_path)["state"]
        == "checked_reference_with_measured_uncertainty"
    )
    summary["runs"][0]["max_momentum_budget_defect_m3_s"] = float("nan")
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        validate_reference_packet(cfg, tmp_path)
