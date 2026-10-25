"""Focused planar orientation, boundary, objective and persistence checks."""

import copy
import csv
import json

import numpy as np
import pytest
import torch
from torch import nn

from hydro_pinn.cli import build_parser
from hydro_pinn.extensions.model_2d import PlanarPINN
from hydro_pinn.extensions.planar_2d import (
    EDGES,
    _checkpoint,
    _identity,
    evaluate_planar,
    load_checkpoint,
    load_planar_config,
    losses,
    mass_queries,
    partition_receipt,
    precision_check,
    query_grid,
    reference_fields,
    samples,
    sha,
    tensor_samples,
    verify_saved_training,
)
from hydro_pinn.physics import Scales


class ManufacturedState(nn.Module):
    """Smooth SI fields with exact nonzero flux through all four edges."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def physical_state(self, coordinates):
        x, y, t = coordinates.unbind(dim=1)
        return torch.stack(
            (
                1 - 0.6 * t,
                1 + 0.1 * x + 0.2 * y + 0.3 * t,
                2 + 0.4 * x + 0.5 * y + 0.6 * t,
            ),
            dim=1,
        )


def _config_case():
    return load_planar_config("configs/extensions/planar_2d.json")


def test_grid_orientation_and_extruded_reference():
    _, case = _config_case()
    x = np.array([-3.0, 0.5, 4.0])
    y = np.array([-2.0, 3.0])
    t = np.array([0.1, 0.25, 0.5])
    fields = query_grid(ManufacturedState(), x, y, t, chunk_size=4)
    assert fields.shape == (3, 2, 3, 3)
    tt, yy, xx = np.meshgrid(t, y, x, indexing="ij")
    np.testing.assert_allclose(fields[..., 0], 1 - 0.6 * tt, atol=1e-15)
    np.testing.assert_allclose(
        fields[..., 1], 1 + 0.1 * xx + 0.2 * yy + 0.3 * tt, atol=1e-15
    )
    np.testing.assert_allclose(
        fields[..., 2], 2 + 0.4 * xx + 0.5 * yy + 0.6 * tt, atol=1e-15
    )
    reference = reference_fields(case, {"x_m": x, "y_m": y, "t_s": t})
    for name in ("h_m", "qx_m2_s", "qy_m2_s", "reference_wet_mask"):
        assert reference[name].shape == (3, 2, 3)
        np.testing.assert_array_equal(reference[name][:, 0], reference[name][:, 1])
    assert not reference["qy_m2_s"].any()
    assert reference["reference_wet_mask"].all()


def test_four_nonzero_edge_signs_and_initial_volume_bias(tmp_path):
    cfg, _ = _config_case()
    cfg = copy.deepcopy(cfg)
    # Tiny quadratures exercise the production path without a model fit.
    cfg["evaluation"].update(
        nx=6,
        ny=3,
        nt=5,
        first_time_s=0.1,
        volume_nx=[6, 12],
        volume_ny=[3, 6],
        edge_nx=[6, 12],
        edge_ny=[3, 6],
        edge_time_nodes=[11, 21],
        query_chunk_size=7,
    )
    reports = [
        mass_queries(ManufacturedState(), cfg, tmp_path, level) for level in (0, 1)
    ]
    for level, report in enumerate(reports):
        t = np.asarray(report["report_t_s"])
        np.testing.assert_allclose(report["volume_m3"], 100 - 60 * t, atol=3e-14)
        np.testing.assert_allclose(report["outward_integral_m3"], 60 * t, atol=3e-14)
        # The fixture starts at100m3; the experiment prescription stays55m3.
        assert report["predicted_initial_bias_m3"] == pytest.approx(45)
        np.testing.assert_allclose(report["corrected_balance_m3"], 45, atol=3e-14)
        np.testing.assert_allclose(
            report["balance_without_initial_bias_m3"], 0, atol=3e-14
        )
        raw = np.load(tmp_path / f"edges_{level}.npz")
        et = raw["t_s"]
        expected = {
            "left": -5 - 3 * et,
            "right": 15 + 3 * et,
            "bottom": 5 - 6 * et,
            "top": 45 + 6 * et,
        }
        for edge in EDGES:
            np.testing.assert_allclose(
                raw[f"{edge}_outward_rate_m3_s"], expected[edge], atol=2e-14
            )
            edge_report = report["edge"][edge]
            assert edge_report["absolute_normal_flux_mismatch_integral_m3"] >= (
                abs(edge_report["final_signed_outward_integral_m3"]) - 1e-13
            )
        volume_raw = np.load(tmp_path / f"volume_{level}.npz")
        assert volume_raw["h_m"].shape == (
            len(t),
            cfg["evaluation"]["volume_ny"][level],
            cfg["evaluation"]["volume_nx"][level],
        )
    np.testing.assert_allclose(
        reports[0]["corrected_balance_m3"],
        reports[1]["corrected_balance_m3"],
        atol=4e-14,
    )


def test_partitions_qy_only_lateral_supervision_and_objective_gradients():
    cfg, case = _config_case()
    train, val = samples(cfg, case, "training"), samples(cfg, case, "validation")
    receipt = partition_receipt(train, val)
    assert receipt["overlap_exact_float64_rows"] == 0
    assert train["pde_xyt_nd"].shape == (10000, 3)
    assert train["ic_xyt_nd"].shape == (1000, 3)
    assert set(train) == {
        "pde_xyt_nd",
        "ic_xyt_nd",
        "ic_state_nd",
        *{f"{e}_{suffix}" for e in EDGES for suffix in ("xyt_nd", "state_nd", "mask")},
    }
    for edge in ("bottom", "top"):
        np.testing.assert_array_equal(train[f"{edge}_mask"], [0, 0, 1])
    small = {
        key: value[:8] if value.ndim == 2 else value for key, value in train.items()
    }
    torch.manual_seed(11)
    model = PlanarPINN(Scales.from_case(case), hidden_layers=(8, 8))
    baseline = losses(model, tensor_samples(model, small))
    changed = copy.deepcopy(small)
    for edge in ("bottom", "top"):
        changed[f"{edge}_state_nd"][:, :2] = 1000
    altered = losses(model, tensor_samples(model, changed))
    torch.testing.assert_close(
        baseline["objective"], altered["objective"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        baseline["objective"],
        baseline["pde"] + 10 * baseline["ic"] + 10 * baseline["bc"],
    )
    baseline["objective"].backward()
    assert all(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for p in model.parameters()
    )


def test_small_cpu_float32_objective_gradient_agrees_with_float64():
    cfg, case = _config_case()
    receipt = precision_check(cfg, case)
    assert receipt["passed"]
    assert receipt["objective_relative_difference"] < 2e-5
    assert receipt["gradient_relative_l2_difference"] < 2e-4


def test_checkpoint_reload_and_cli_pilot_guard(tmp_path):
    _, case = _config_case()
    torch.manual_seed(11)
    model = PlanarPINN(Scales.from_case(case))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    protocol = {
        "protocol_sha256": "fixture-protocol",
        "case": case,
        "membership_sha256": {"training_samples.npz": "fixture-samples"},
    }
    identity = _identity(model)
    path = tmp_path / "checkpoint.pt"
    torch.save(_checkpoint(model, optimizer, 0, 1.5, protocol, identity), path)
    restored = load_checkpoint(path, protocol)
    assert _identity(restored) == identity
    coordinates = torch.tensor([[0.2, -0.7, 0.1], [-0.3, 0.4, 0.2]])
    torch.testing.assert_close(
        model(coordinates), restored(coordinates), rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="checkpoint/protocol mismatch"):
        load_checkpoint(path, {**protocol, "protocol_sha256": "different"})
    with pytest.raises(ValueError, match="checkpoint/sample mismatch"):
        load_checkpoint(path, {**protocol, "membership_sha256": {}})
    args = build_parser().parse_args(
        [
            "extension-2d-planar",
            "--config",
            "configs/extensions/planar_2d.json",
            "--run-id",
            "p10_planar_parser_fixture",
        ]
    )
    assert args.command == "extension-2d-planar"
    assert args.run_id == "p10_planar_parser_fixture"
    run = tmp_path / "p10_planar_pilot_fixture"
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps({"mode": "pilot", "actual_updates": 25})
    )
    with pytest.raises(ValueError, match="pilots refused"):
        evaluate_planar(run.name, "uncreated_evaluation", root=tmp_path)
    assert not (tmp_path / "uncreated_evaluation").exists()


def test_saved_history_requires_every_step_and_earliest_strict_minimum(tmp_path):
    _, case = _config_case()
    torch.manual_seed(11)
    model = PlanarPINN(Scales.from_case(case))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    protocol = {
        "protocol_sha256": "synthetic-history-fixture",
        "membership_sha256": {"training_samples.npz": "synthetic-membership"},
        "case": case,
        "config": {"full_steps": 2},
    }
    identity = _identity(model)
    # Synthetic history and unchanged parameters exercise readback only.
    # The tie at step2 must keep step1 as the earliest strict minimum.
    for filename, step, score in (
        ("initial.pt", 0, 3.0),
        ("checkpoint.pt", 1, 2.0),
        ("latest.pt", 2, 2.0),
    ):
        torch.save(
            _checkpoint(model, optimizer, step, score, protocol, identity),
            tmp_path / filename,
        )
    summary = {
        "actual_updates": 2,
        "checkpoint_write_counts": {"initial": 1, "selected": 2, "latest": 3},
        "selected_step": 1,
        "selected_validation_score": 2.0,
        "initial_identity": identity,
        "checkpoint_sha256": sha(tmp_path / "checkpoint.pt"),
        "mode": "pilot",
    }
    rows = [
        {"step": 0, "val_objective": 3.0, "selected": "True"},
        {"step": 1, "val_objective": 2.0, "selected": "True"},
        {"step": 2, "val_objective": 2.0, "selected": "False"},
    ]

    def write_history():
        with (tmp_path / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        summary["history_sha256"] = sha(tmp_path / "history.csv")

    write_history()
    readback = verify_saved_training(tmp_path, summary, protocol)
    assert readback["earliest_strict_minimum_step"] == 1
    assert readback["history_steps"] == 3
    assert readback["initial_identity_verified"]
    rows[2]["selected"] = "True"
    write_history()
    with pytest.raises(ValueError, match="earliest strict minimum"):
        verify_saved_training(tmp_path, summary, protocol)
    rows[2]["selected"] = "False"
    rows[2]["step"] = 3
    write_history()
    with pytest.raises(ValueError, match="initial state and every update"):
        verify_saved_training(tmp_path, summary, protocol)
    rows[2]["step"] = 2
    write_history()
    summary["mode"] = "full"
    release_path = tmp_path / "root_release.json"
    release_path.write_text(
        json.dumps(
            {
                "state": "released",
                "protocol_sha256": protocol["protocol_sha256"],
            }
        )
    )
    assert verify_saved_training(tmp_path, summary, protocol)["release_verified"]
    release_path.write_text(
        json.dumps(
            {
                "state": "released",
                "protocol_sha256": "different-protocol",
            }
        )
    )
    with pytest.raises(ValueError, match="release/budget mismatch"):
        verify_saved_training(tmp_path, summary, protocol)
