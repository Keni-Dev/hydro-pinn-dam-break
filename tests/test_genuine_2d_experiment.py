"""Necessary small checks for the separately selected genuine-2D PINN path.

The main implementation owner executes this file. It contains no precision
comparison, pilot, full fit, reference solve or final model-quality check.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from hydro_pinn.extensions import evaluate_2d as evaluation
from hydro_pinn.extensions import experiment_2d as experiment
from hydro_pinn.extensions.physics_2d import physical_residuals, residuals

ROOT = Path(__file__).resolve().parents[1]
EDGES = ("left", "right", "bottom", "top")


@pytest.fixture(scope="module")
def memberships():
    config = json.loads(
        (ROOT / "configs/extensions/genuine_2d_neural.json").read_text()
    )
    return (
        config,
        experiment.generate_samples(config, "training"),
        experiment.generate_samples(config, "validation"),
    )


class SmoothState:
    def __init__(self):
        self.scales = experiment.scales()

    def __call__(self, coordinates):
        x, y, t = coordinates.unbind(1)
        return torch.stack(
            (
                2 + 0.2 * x + 0.1 * y + 0.05 * t,
                0.3 + 0.4 * x - 0.2 * y + 0.15 * t,
                -0.2 + 0.25 * x + 0.35 * y - 0.1 * t,
            ),
            dim=1,
        )


class ConstantState(nn.Module):
    def __init__(self, h=0.2, qx=0.3, qy=-0.4):
        super().__init__()
        self.scales = experiment.scales()
        self.state = nn.Parameter(torch.tensor([h, qx, qy], dtype=torch.float64))

    def forward(self, coordinates):
        return self.state.expand(len(coordinates), -1)


def test_smooth_three_residuals_and_physical_H1_L10_chain_factors():
    model = SmoothState()
    scale = model.scales
    assert scale.H_m == 1 and scale.Lc_m == 10
    assert scale.Uc_m_s == math.sqrt(9.81)
    assert scale.Tc_s == 10 / math.sqrt(9.81)
    coords = torch.tensor(
        [[-0.31, 0.23, 0.017], [0.19, -0.37, 0.061]], dtype=torch.float64
    )
    h, qx, qy = model(coords).unbind(1)
    expected = torch.stack(
        (
            torch.full_like(h, 0.05 + 0.4 + 0.35),
            0.15
            + 2 * qx * 0.4 / h
            - qx.square() * 0.2 / h.square()
            + h * 0.2
            + (-0.2 * qy + qx * 0.35) / h
            - qx * qy * 0.1 / h.square(),
            -0.1
            + (0.4 * qy + qx * 0.25) / h
            - qx * qy * 0.2 / h.square()
            + 2 * qy * 0.35 / h
            - qy.square() * 0.1 / h.square()
            + h * 0.1,
        ),
        dim=1,
    )
    torch.testing.assert_close(
        residuals(model, coords), expected, rtol=1e-12, atol=1e-12
    )
    physical = coords * coords.new_tensor([10, 10, scale.Tc_s])
    factors = coords.new_tensor(
        [scale.H_m / scale.Tc_s, scale.Qc_m2_s / scale.Tc_s, scale.Qc_m2_s / scale.Tc_s]
    )
    torch.testing.assert_close(
        physical_residuals(model, physical), expected * factors, rtol=1e-12, atol=1e-12
    )


def _actual_input(si):
    si32 = torch.as_tensor(si, dtype=torch.float32)
    return (si32 / si32.new_tensor([10, 10, 10 / math.sqrt(9.81)])).numpy()


def test_actual_stratified_IC_execution_support_and_normal_only_wall_loss(memberships):
    _, training, validation = memberships
    for samples, expected in (
        (training, [20, 400, 400, 90, 90]),
        (validation, [8, 160, 160, 36, 36]),
    ):
        labels, counts = np.unique(samples["ic_stratum"], return_counts=True)
        assert len(labels) == 5
        np.testing.assert_array_equal(counts, expected)
        np.testing.assert_array_equal(
            counts / counts.sum(), [0.02, 0.4, 0.4, 0.09, 0.09]
        )
        np.testing.assert_array_equal(
            np.sort(samples["ic_unshuffled_index"]), np.arange(sum(expected))
        )
        for name in ("pde", "ic", *EDGES):
            si, actual = samples[f"{name}_si64"], samples[f"{name}_input32"]
            assert si.dtype == np.float64 and actual.dtype == np.float32
            np.testing.assert_array_equal(
                samples[f"{name}_si32"], si.astype(np.float32)
            )
            np.testing.assert_array_equal(actual, _actual_input(si))
            np.testing.assert_array_equal(
                experiment.execution_coordinates(si, dtype=torch.float32).numpy(),
                actual,
            )
            assert np.isfinite(actual).all()
        pde = samples["pde_input32"]
        assert np.all((pde[:, :2] > -0.5) & (pde[:, :2] < 0.5))
        upper_time = np.float32(0.25) / np.float32(10 / math.sqrt(9.81))
        assert np.all((pde[:, 2] > 0) & (pde[:, 2] < upper_time))
        ic = samples["ic_input32"]
        assert np.all((ic[:, :2] > -0.5) & (ic[:, :2] < 0.5))
        assert np.all(ic[:, 2] == 0)
        deep = (
            (ic[:, 0] >= np.float32(-0.1))
            & (ic[:, 0] < np.float32(0.1))
            & (ic[:, 1] >= np.float32(-0.05))
            & (ic[:, 1] < np.float32(0.05))
        )
        np.testing.assert_array_equal(
            samples["ic_target64"][:, 0], np.where(deep, 1.0, 0.1)
        )
        np.testing.assert_array_equal(samples["ic_target64"][:, 1:], 0)
        np.testing.assert_array_equal(
            experiment.initial_depth(samples["ic_si64"]), samples["ic_target64"][:, 0]
        )
        region_masks = (
            deep,
            ic[:, 0] < np.float32(-0.1),
            ic[:, 0] >= np.float32(0.1),
            (ic[:, 0] >= np.float32(-0.1))
            & (ic[:, 0] < np.float32(0.1))
            & (ic[:, 1] < np.float32(-0.05)),
            (ic[:, 0] >= np.float32(-0.1))
            & (ic[:, 0] < np.float32(0.1))
            & (ic[:, 1] >= np.float32(0.05)),
        )
        for label, mask in enumerate(region_masks):
            np.testing.assert_array_equal(mask, samples["ic_stratum"] == label)
        for edge, axis, value in (
            ("left", 0, -0.5),
            ("right", 0, 0.5),
            ("bottom", 1, -0.5),
            ("top", 1, 0.5),
        ):
            wall = samples[f"{edge}_input32"]
            assert np.all(wall[:, axis] == value)
            assert np.all((wall[:, 1 - axis] > -0.5) & (wall[:, 1 - axis] < 0.5))
            assert np.all((wall[:, 2] > 0) & (wall[:, 2] < upper_time))
    assert experiment.WALL_SELECTORS == {"left": 1, "right": 1, "bottom": 2, "top": 2}
    # Keep the exact IC proportions, with only two constant-state PDE/wall rows.
    loss_fixture = dict(training)
    for name in ("pde", *EDGES):
        loss_fixture[f"{name}_si64"] = training[f"{name}_si64"][:2]
    values = experiment.component_losses(ConstantState(), loss_fixture)
    expected_h = 0.02 * (0.2 - 1) ** 2 + 0.98 * (0.2 - 0.1) ** 2
    assert float(values["ic_h"].detach()) == pytest.approx(expected_h, abs=1e-14)
    assert float(values["ic_qx"].detach()) == pytest.approx(0.09, abs=1e-14)
    assert float(values["ic_qy"].detach()) == pytest.approx(0.16, abs=1e-14)
    for edge, expected in (
        ("left", 0.09),
        ("right", 0.09),
        ("bottom", 0.16),
        ("top", 0.16),
    ):
        assert float(values[f"bc_{edge}"].detach()) == pytest.approx(
            expected, abs=1e-14
        )
    assert float(values["bc"].detach()) == pytest.approx(0.125, abs=1e-14)
    assert float(values["pde"].detach()) == 0
    assert float(values["objective"].detach()) == pytest.approx(
        10 * (expected_h + 0.09 + 0.16) + 10 * 0.125, abs=1e-13
    )


def test_manufactured_Gauss_means_match_streamed_cell_accumulation():
    xc, yc, dx, dy = 0.3, -0.4, 0.2, 0.4
    x2, y2 = xc * xc + dx * dx / 12, yc * yc + dy * dy / 12
    expected = np.array(
        [
            2 + 0.1 * xc + 0.2 * yc + x2 + 2 * y2,
            -0.3 + xc * yc + 3 * x2,
            0.4 - 0.2 * xc + 4 * y2,
        ]
    )
    for order in (3, 5):
        nodes, weights = np.polynomial.legendre.leggauss(order)
        yy, xx = np.meshgrid(yc + dy * nodes / 2, xc + dx * nodes / 2, indexing="ij")
        values = np.stack(
            [
                2 + 0.1 * xx + 0.2 * yy + xx**2 + 2 * yy**2,
                -0.3 + xx * yy + 3 * xx**2,
                0.4 - 0.2 * xx + 4 * yy**2,
            ],
            axis=-1,
        )
        np.testing.assert_allclose(
            evaluation.gauss_cell_means(values, weights),
            expected,
            rtol=3e-14,
            atol=3e-14,
        )
        accumulated = np.zeros((1, 3), dtype=np.float64)
        point_weights = np.outer(weights, weights).ravel() / 4
        flattened = values.reshape(-1, 3)
        for start in range(0, len(flattened), 4):
            stop = min(start + 4, len(flattened))
            evaluation.accumulate_gauss_batch(
                accumulated,
                np.zeros(stop - start, dtype=np.int64),
                flattened[start:stop],
                point_weights[start:stop],
            )
        np.testing.assert_allclose(accumulated[0], expected, rtol=3e-14, atol=3e-14)

        # Exercise the same lazy iterator and weights used for production means.
        operator = {
            "x_nodes_m": (xc + dx * nodes / 2)[None],
            "y_nodes_m": (yc + dy * nodes / 2)[None],
            "t_s": np.array([0.005, 0.01]),
            "legendre_nodes": nodes,
            "legendre_weights": weights,
        }
        streamed = np.zeros((2, 3), dtype=np.float64)
        seen = []
        for batch in evaluation.iter_query_points(operator, "gauss", batch_size=4):
            x, y, _ = batch["points_si64"].T
            states = np.column_stack(
                (
                    2 + 0.1 * x + 0.2 * y + x**2 + 2 * y**2,
                    -0.3 + x * y + 3 * x**2,
                    0.4 - 0.2 * x + 4 * y**2,
                )
            )
            assert len(states) <= 4
            evaluation.accumulate_gauss_batch(
                streamed, batch["cell_indices"], states, batch["point_weights"]
            )
            seen.extend(batch["flat_indices"].tolist())
        assert seen == list(range(2 * order**2))
        np.testing.assert_allclose(
            streamed, np.broadcast_to(expected, (2, 3)), rtol=3e-14, atol=3e-14
        )


def test_actual_memberships_and_factorized_final_axes_are_disjoint(
    memberships, tmp_path
):
    _, training, validation = memberships
    centres = -5 + (np.arange(80, dtype=np.float64) + 0.5) / 8
    times = 0.005 * np.arange(1, 51, dtype=np.float64)
    for order in (3, 5):
        nodes, _ = np.polynomial.legendre.leggauss(order)
        axis = centres[:, None] + nodes[None] / 16
        np.savez(
            tmp_path / f"gauss{order}_operator.npz",
            x_nodes_m=axis,
            y_nodes_m=axis.copy(),
            t_s=times,
        )
    axes = list(evaluation.final_gauss_axes(tmp_path))
    assert [item[0] for item in axes] == [3, 5]
    for order, x, y, t in axes:
        assert x.ndim == y.ndim == t.ndim == 1
        assert x.dtype == y.dtype == t.dtype == np.float64
        assert x.size == y.size == 80 * order and t.size == 50
        np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(t, times)
    shared = np.intersect1d(axes[0][1], axes[1][1])
    np.testing.assert_array_equal(shared, centres)
    assert sum(a.nbytes for _, *parts in axes for a in parts) == 8 * (
        2 * 80 * (3 + 5) + 2 * 50
    )

    # Independently read the actual CPU32 row sets, including all four walls.
    row_sets = []
    for samples in (training, validation):
        actual_rows = np.concatenate(
            [samples[f"{name}_input32"] for name in ("pde", "ic", *EDGES)]
        ).copy()
        actual_rows[actual_rows == 0] = 0
        unique_rows = {row.tobytes() for row in actual_rows}
        assert len(unique_rows) == len(actual_rows)
        row_sets.append(unique_rows)
    assert row_sets[0].isdisjoint(row_sets[1])

    # Only the factorized one-dimensional axes are supplied; no final row union.
    receipt = experiment.verify_memberships(training, validation, axes)
    assert receipt["passed"] is True
    assert receipt["final_rules_intentionally_share_centres"] == 320_000
    assert set(receipt["representations"]) == {"si64", "scaled64", "input32"}
    for facts in receipt["representations"].values():
        assert facts["training_duplicates"] == 0
        assert facts["validation_duplicates"] == 0
        assert facts["train_validation_intersection"] == 0
        assert facts["final_intersections"] == {
            "3": {"training": 0, "validation": 0},
            "5": {"training": 0, "validation": 0},
        }
    np.testing.assert_array_equal(
        experiment.row_bytes(np.array([[0.0, -0.0, 0.0]], dtype=np.float32)),
        experiment.row_bytes(np.array([[-0.0, 0.0, -0.0]], dtype=np.float32)),
    )

    # Deliberately put one held-out row on a final Gauss3 query coordinate.
    collision = {key: value.copy() for key, value in validation.items()}
    point = np.array([[axes[0][1][0], axes[0][2][0], times[0]]])
    converted = evaluation.execution_representations(point, experiment.scales())
    for saved_name, conversion_name in (
        ("si64", "si64"),
        ("scaled64", "scaled64"),
        ("si32", "si32_physical"),
        ("input32", "model_nd32"),
    ):
        collision[f"pde_{saved_name}"][0] = converted[conversion_name][0]
    with pytest.raises(experiment.Experiment2DError) as rejected:
        experiment.verify_memberships(training, collision, axes)
    for facts in rejected.value.receipt["representations"].values():
        assert facts["final_intersections"]["3"]["validation"] == 1


def test_mass_schedules_midpoint_weights_and_four_outward_signs():
    exact_outputs = 0.005 * np.arange(51, dtype=np.float64)
    for nodes in (501, 1001):
        times, output_indices, reported = evaluation.mass_time_schedule(nodes)
        expected_indices = np.arange(51, dtype=np.int64) * ((nodes - 1) // 50)
        expected_times = 0.25 * np.arange(nodes, dtype=np.float64) / (nodes - 1)
        expected_times[expected_indices] = exact_outputs
        assert times.dtype == reported.dtype == np.float64
        assert output_indices.dtype == np.int64
        assert len(times) == nodes and times[0] == 0 and times[-1] == 0.25
        np.testing.assert_array_equal(times, expected_times)
        np.testing.assert_array_equal(output_indices, expected_indices)
        np.testing.assert_array_equal(reported, exact_outputs)
        np.testing.assert_array_equal(times[output_indices], exact_outputs)
        assert np.all(np.diff(times) > 0)
        actual = evaluation.execution_representations(
            np.column_stack((np.zeros(nodes), np.zeros(nodes), times)),
            experiment.scales(),
        )["model_nd32"][:, 2]
        assert np.all(np.diff(actual) > 0)
        np.testing.assert_array_equal(
            actual[output_indices],
            _actual_input(np.column_stack((np.zeros(51), np.zeros(51), exact_outputs)))[
                :, 2
            ],
        )
        values = np.column_stack((np.ones(nodes), 2 + 3 * times))
        np.testing.assert_allclose(
            evaluation.trapezoid_prefix(values, times),
            np.column_stack((times, 2 * times + 1.5 * times**2)),
            rtol=3e-14,
            atol=3e-14,
        )
    for n in (80, 160):
        centres, edges, areas = evaluation.midpoint_axes(n)
        np.testing.assert_array_equal(centres, (edges[:-1] + edges[1:]) / 2)
        np.testing.assert_array_equal(areas, np.full((n, n), 100 / n**2))
        assert areas.sum() == pytest.approx(100)
        assert len(centres) * (10 / n) == 10
    assert evaluation.NORMAL_COMPONENT == experiment.WALL_SELECTORS
    assert evaluation.OUTWARD_SIGN == {"left": -1, "right": 1, "bottom": -1, "top": 1}
    normal = {
        "left": np.array([[1, 3], [-1, 1]], dtype=np.float64),
        "right": np.array([[-2, 4], [2, -6]], dtype=np.float64),
        "bottom": np.array([[5, -1], [-3, -7]], dtype=np.float64),
        "top": np.array([[2, -4], [-1, 5]], dtype=np.float64),
    }
    rates = evaluation.outward_mass_rates(normal, np.array([5.0, 5.0]))
    for wall, signed, absolute in (
        ("left", [-20, 0], [20, 10]),
        ("right", [10, -20], [30, 40]),
        ("bottom", [-20, 50], [30, 50]),
        ("top", [-10, 20], [30, 30]),
    ):
        np.testing.assert_array_equal(rates["signed"][wall], signed)
        np.testing.assert_array_equal(rates["absolute"][wall], absolute)
        np.testing.assert_array_equal(
            rates["max_abs_normal"][wall], np.max(np.abs(normal[wall]), axis=1)
        )
    np.testing.assert_array_equal(sum(rates["signed"].values()), [-40, 50])
    np.testing.assert_array_equal(sum(rates["absolute"].values()), [110, 130])


def test_changing_region_metrics_use_integral_ratio_and_preserve_nulls():
    reference = np.zeros((2, 1, 2, 2), dtype=np.float64)
    reference[..., 0] = 2
    error = np.array([[[[1, -2], [100, 100]]], [[[1, 2], [3, -2]]]], dtype=np.float64)
    areas = np.array([[1.0, 3.0]])
    mask = np.array([[[True, False]], [[True, True]]])
    times = np.array([0.005, 0.01])
    result = evaluation.region_metrics(
        reference + error, reference, areas, mask, times, ("h_m", "qx_m2_s")
    )
    assert result["empty_region_times"] == []
    np.testing.assert_array_equal(
        [row["region_area_m2"] for row in result["per_time"]], [1, 4]
    )
    for row, mae, rmse in zip(
        result["per_time"], (1, 2.5), (1, math.sqrt(7)), strict=True
    ):
        h = row["fields"]["h_m"]
        assert h["area_mean_MAE"] == pytest.approx(mae)
        assert h["area_mean_RMSE"] == pytest.approx(rmse)
        q = row["fields"]["qx_m2_s"]
        assert q["area_mean_MAE"] == 2 and q["area_mean_RMSE"] == 2
        assert q["relative_L2_percent"] is None
        assert q["relative_reason"] == "zero_reference_norm"
    pooled_h = result["pooled_equal_time"]["h_m"]
    assert pooled_h["area_mean_MAE"] == 1.75
    assert pooled_h["area_mean_RMSE"] == 2
    assert pooled_h["pooled_integral_relative_L2_percent"] == pytest.approx(
        100 * math.sqrt(29 / 20)
    )
    assert pooled_h["pooled_integral_relative_L2_percent"] != pytest.approx(100)
    pooled_q = result["pooled_equal_time"]["qx_m2_s"]
    assert pooled_q["area_mean_MAE"] == 2 and pooled_q["area_mean_RMSE"] == 2
    assert pooled_q["pooled_integral_relative_L2_percent"] is None
    assert pooled_q["relative_reason"] == "zero_reference_norm"

    empty_mask = mask.copy()
    empty_mask[0] = False
    empty = evaluation.region_metrics(
        reference + error, reference, areas, empty_mask, times, ("h_m", "qx_m2_s")
    )
    assert empty["empty_region_times"] == [0]
    for name in ("h_m", "qx_m2_s"):
        assert empty["per_time"][0]["fields"][name]["reason"] == "empty_region"
        assert empty["per_time"][1]["fields"][name]["area_mean_MAE"] is not None
        pooled = empty["pooled_equal_time"][name]
        assert pooled["reason"] == "empty_region_at_one_or_more_times_no_omission"
        for key in (
            "area_mean_MAE",
            "area_mean_RMSE",
            "integral_L1",
            "integral_L2",
            "pooled_integral_relative_L2_percent",
        ):
            assert pooled[key] is None


def test_fresh_discarded_IC_Adam_fixture_checkpoint_and_provenance(tmp_path):
    """One tiny fixture update is a persistence check, never pilot/model screening."""
    model, optimizer = experiment.fresh_initial_state()
    initial = {key: value.clone() for key, value in model.state_dict().items()}
    identity = experiment.model_identity(initial)
    same, same_optimizer = experiment.fresh_initial_state()
    assert sum(parameter.numel() for parameter in model.parameters()) == 12_931
    assert experiment.model_identity(same) == identity
    assert not optimizer.state and not same_optimizer.state
    assert optimizer.param_groups[0]["lr"] == 0.001
    for name, value in same.state_dict().items():
        assert value.dtype == torch.float32 and value.device.type == "cpu"
        assert torch.equal(value, initial[name])

    fixture_si = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 0.0]], dtype=np.float64)
    coordinates = experiment.execution_coordinates(fixture_si)
    targets = torch.tensor([[1.0, 0.0, 0.0], [0.1, 0.0, 0.0]], dtype=torch.float32)
    optimizer.zero_grad(set_to_none=True)
    fixture_loss = (model(coordinates) - targets).square().mean()
    fixture_loss.backward()
    optimizer.step()  # Exactly one local fixture update, discarded after this test.
    assert experiment.model_identity(model) != identity
    with torch.no_grad():
        fixture_score = float((model(coordinates) - targets).square().mean())
    binding = {
        "tiny_ic_fixture_si64": hashlib.sha256(fixture_si.tobytes()).hexdigest(),
        "tiny_ic_fixture_targets32": hashlib.sha256(
            targets.numpy().tobytes()
        ).hexdigest(),
    }
    protocol = {
        "protocol_sha256": hashlib.sha256(
            json.dumps(binding, sort_keys=True).encode()
        ).hexdigest(),
        "membership_sha256": binding,
        "initial_identity": identity,
    }
    payload = experiment.checkpoint_payload(
        model, optimizer, 1, fixture_score, protocol, identity
    )
    checkpoint = tmp_path / "discarded_fixture.pt"
    torch.save(payload, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    loaded = experiment.load_checkpoint(checkpoint, protocol)
    reloaded_optimizer = torch.optim.Adam(loaded.parameters(), lr=0.001)
    reloaded_optimizer.load_state_dict(saved["optimizer"])
    assert saved["step"] == 1
    assert saved["validation_score"] == fixture_score
    assert saved["initial_identity"] == identity
    assert saved["sample_sha256"] == binding
    assert torch.equal(saved["torch_rng_state"], payload["torch_rng_state"])
    with torch.no_grad():
        assert torch.equal(model(coordinates), loaded(coordinates))
        assert float((loaded(coordinates) - targets).square().mean()) == fixture_score
    original_adam = optimizer.state_dict()
    restored_adam = reloaded_optimizer.state_dict()
    assert restored_adam["param_groups"] == original_adam["param_groups"]
    assert len(restored_adam["state"]) == 10
    for index, state in original_adam["state"].items():
        assert set(state) == set(restored_adam["state"][index])
        assert float(state["step"]) == 1
        for key, value in state.items():
            assert torch.equal(value, restored_adam["state"][index][key])
    fresh, fresh_optimizer = experiment.fresh_initial_state()
    assert experiment.model_identity(fresh) == identity
    assert not fresh_optimizer.state
    assert experiment.select_checkpoint(2.0, math.inf)
    assert not experiment.select_checkpoint(2.0, 2.0)
    assert experiment.select_checkpoint(1.0, 2.0)
    for key in ("protocol_sha256", "membership_sha256", "initial_identity"):
        changed = copy.deepcopy(protocol)
        changed[key] = "incorrect_fixture_binding"
        with pytest.raises(experiment.Experiment2DError, match="provenance"):
            experiment.load_checkpoint(checkpoint, changed)
