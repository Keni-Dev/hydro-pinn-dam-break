"""R1–R5 small deterministic checks for the independent wet 2D reference."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hydro_pinn.cli import build_parser
from hydro_pinn.extensions.friction_reference import hlle_flux as hlle_flux_1d
from hydro_pinn.extensions.reference_2d import (
    Reference2DError,
    area_difference_metrics,
    conservative_project,
    fluxfaces,
    ghost_cells,
    hlle_flux,
    outward_rates,
    overlap_weights,
    rectangle_initial,
    unsplit_step,
)

EPS = np.finfo(np.float64).eps
ROUND_TOL = 256 * EPS


def _receipt(path, value):
    (path / "raw_defects.json").write_text(json.dumps(value, indent=2, allow_nan=False))


def test_r1_exact_rectangle_geometry_layout_and_signed_axis_rotation(tmp_path):
    # Deliberately nonaligned small cells exercise area fractions, not masks.
    xe = np.array([-5.0, -1.5, -0.7, 0.0, 0.6, 1.3, 5.0])
    ye = np.array([-5.0, -0.8, -0.2, 0.2, 0.7, 5.0])
    initial = rectangle_initial(xe, ye)
    area = np.diff(ye)[:, None] * np.diff(xe)[None, :]
    assert initial.shape == (5, 6, 3)
    np.testing.assert_array_equal(initial[..., 1:], 0)
    np.testing.assert_allclose(
        np.sum((initial[..., 0] - 0.1) / 0.9 * area),
        2.0,
        rtol=0,
        atol=ROUND_TOL,
    )
    np.testing.assert_allclose(
        np.sum(initial[..., 0] * area),
        11.8,
        rtol=0,
        atol=ROUND_TOL,
    )
    # One full-inside and one partial-overlap cell identify x/y orientation.
    assert initial[2, 2, 0] == 1.0
    expected_partial = 0.1 + 0.9 * (0.3 * 0.3) / (0.8 * 0.6)
    np.testing.assert_allclose(initial[1, 1, 0], expected_partial, atol=ROUND_TOL)

    left = np.array([[1.0, 0.3, -0.6], [0.8, -0.12, 0.4]])
    right = np.array([[1.0, 0.3, 0.5], [1.2, 0.24, -0.3]])
    result_x = hlle_flux(left, right, axis="x")
    result_y = hlle_flux(left[:, [0, 2, 1]], right[:, [0, 2, 1]], axis="y")
    np.testing.assert_allclose(
        result_x.flux,
        result_y.flux[:, [0, 2, 1]],
        rtol=ROUND_TOL,
        atol=ROUND_TOL,
    )
    # Independent full-HLL transverse flux, including its signed jump term.
    sl, sr = result_x.s_left, result_x.s_right
    tangential_left = left[:, 1] * left[:, 2] / left[:, 0]
    tangential_right = right[:, 1] * right[:, 2] / right[:, 0]
    expected_tangential = (
        sr * tangential_left
        - sl * tangential_right
        + sl * sr * (right[:, 2] - left[:, 2])
    ) / (sr - sl)
    np.testing.assert_allclose(
        result_x.flux[:, 2],
        expected_tangential,
        rtol=ROUND_TOL,
        atol=ROUND_TOL,
    )
    assert np.any(result_x.flux[:, 2] < 0)
    assert np.any(left[:, 1:] < 0)
    signed = np.array(
        [
            [[1.0, -0.2, 0.3], [0.8, 0.4, -0.5]],
            [[0.7, 0.1, -0.2], [1.1, -0.3, 0.2]],
        ]
    )
    ghosts = ghost_cells(signed, boundary="reflective")
    np.testing.assert_array_equal(ghosts[1:-1, 0], signed[:, 0] * [1, -1, 1])
    np.testing.assert_array_equal(ghosts[1:-1, -1], signed[:, -1] * [1, -1, 1])
    np.testing.assert_array_equal(ghosts[0, 1:-1], signed[0] * [1, 1, -1])
    np.testing.assert_array_equal(ghosts[-1, 1:-1], signed[-1] * [1, 1, -1])
    np.testing.assert_array_equal(ghosts[0, 0], signed[0, 0] * [1, -1, -1])
    _receipt(
        tmp_path,
        {
            "deep_area_defect_m2": float(
                np.sum((initial[..., 0] - 0.1) / 0.9 * area) - 2
            ),
            "initial_volume_defect_m3": float(np.sum(initial[..., 0] * area) - 11.8),
            "maximum_axis_rotated_flux_defect": float(
                np.max(np.abs(result_x.flux - result_y.flux[:, [0, 2, 1]]))
            ),
            "maximum_transverse_flux_defect": float(
                np.max(np.abs(result_x.flux[:, 2] - expected_tangential))
            ),
            "geometry_absolute_tolerance": ROUND_TOL,
        },
    )


def test_r2_uniform_still_lake_four_reflective_walls_is_unchanged(tmp_path):
    state = np.zeros((3, 5, 3), dtype=np.float64)
    state[..., 0] = 0.7
    expanded = ghost_cells(state, boundary="reflective")
    np.testing.assert_array_equal(expanded[..., 0], 0.7)
    np.testing.assert_array_equal(expanded[..., 1:], 0)
    expected = state.copy()
    for dt in (0.001, 0.002, 0.0015):
        faces = fluxfaces(state, boundary="reflective")
        state = unsplit_step(state, faces.fx, faces.fy, dt, 0.5, 0.75)
    np.testing.assert_array_equal(state, expected)
    _receipt(
        tmp_path,
        {
            "maximum_component_defects": np.max(
                np.abs(state - expected), axis=(0, 1)
            ).tolist(),
            "updates": 3,
            "comparison": "bit-exact",
        },
    )


def test_r3_both_planar_axes_match_accepted_1d_with_identical_dt_and_ghosts(tmp_path):
    initial_1d = np.column_stack(
        (
            [1.0, 1.0, 0.6, 0.3, 0.1, 0.1],
            [0.08, 0.02, -0.03, -0.01, 0.04, 0.02],
        )
    )
    defects = {"x": [], "y": []}
    for axis in ("x", "y"):
        one = initial_1d.copy()
        if axis == "x":
            state = np.zeros((3, len(one), 3))
            state[..., 0] = one[:, 0][None, :]
            state[..., 1] = one[:, 1][None, :]
            normal, transverse = 1, 2
        else:
            state = np.zeros((len(one), 3, 3))
            state[..., 0] = one[:, 0][:, None]
            state[..., 2] = one[:, 1][:, None]
            normal, transverse = 2, 1
        for dt in (0.001, 0.002, 0.0015):
            # The accepted wrapper uses reservoir ghosts; this comparator
            # deliberately supplies matching reflective ghosts at each update.
            ghosts = np.concatenate(
                (
                    one[:1] * [1, -1],
                    one,
                    one[-1:] * [1, -1],
                )
            )
            flux_1d, _ = hlle_flux_1d(ghosts[:-1], ghosts[1:])
            one = one - dt / 0.5 * (flux_1d[1:] - flux_1d[:-1])
            faces = fluxfaces(state, boundary="reflective")
            state = unsplit_step(state, faces.fx, faces.fy, dt, 0.5, 0.5)
            comparison = (
                state[1, :, :][:, [0, normal]]
                if axis == "x"
                else (state[:, 1, :][:, [0, normal]])
            )
            np.testing.assert_allclose(
                comparison,
                one,
                rtol=ROUND_TOL,
                atol=ROUND_TOL,
            )
            np.testing.assert_array_equal(state[..., transverse], 0)
            defects[axis].append(float(np.max(np.abs(comparison - one))))
    _receipt(
        tmp_path,
        {
            "maximum_component_defect_by_update": defects,
            "imposed_dt_s": [0.001, 0.002, 0.0015],
            "relative_and_absolute_tolerance": ROUND_TOL,
            "orthogonal_discharge_bit_exact_zero": True,
        },
    )


def test_r4_all_component_four_edge_budget_and_conservative_area_projection(tmp_path):
    ny, nx, dx, dy, dt = 2, 3, 0.7, 1.3, 0.002
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    old = np.stack((2 + 0.1 * jj + 0.2 * ii, 0.3 - 0.2 * ii, -0.1 + 0.1 * jj), axis=-1)
    yf, xf = np.meshgrid(np.arange(ny), np.arange(nx + 1), indexing="ij")
    fx = np.stack(
        (
            0.1 + 0.2 * xf + 0.3 * yf,
            -0.7 + 0.4 * xf - 0.2 * yf,
            0.9 - 0.1 * xf + 0.5 * yf,
        ),
        axis=-1,
    )
    yf, xf = np.meshgrid(np.arange(ny + 1), np.arange(nx), indexing="ij")
    fy = np.stack(
        (
            -0.2 + 0.1 * xf - 0.3 * yf,
            0.6 - 0.5 * xf + 0.2 * yf,
            -0.8 + 0.4 * xf + 0.3 * yf,
        ),
        axis=-1,
    )
    rates = outward_rates(fx, fy, dx, dy)
    independently_signed = {
        "left": -dy * np.sum(fx[:, 0], axis=0),
        "right": dy * np.sum(fx[:, -1], axis=0),
        "bottom": -dx * np.sum(fy[0], axis=0),
        "top": dx * np.sum(fy[-1], axis=0),
    }
    for edge, expected in independently_signed.items():
        np.testing.assert_allclose(rates[edge], expected, rtol=0, atol=ROUND_TOL)
        assert np.all(expected != 0)
    updated = unsplit_step(old, fx, fy, dt, dx, dy)
    integrated_change = np.sum(updated - old, axis=(0, 1)) * dx * dy
    defect = integrated_change + dt * sum(independently_signed.values())
    scale = np.sum(np.abs(old), axis=(0, 1)) * dx * dy
    assert np.all(np.abs(defect) <= ROUND_TOL * scale)

    xe = np.array([-5.0, -2.0, 0.0, 1.0, 5.0])
    ye = np.array([-5.0, -1.0, 2.0, 5.0])
    target_x = np.array([-5.0, -3.0, 2.0, 5.0])
    target_y = np.array([-5.0, 0.0, 5.0])
    wx, wy = overlap_weights(xe, target_x), overlap_weights(ye, target_y)
    for weights in (wx, wy):
        np.testing.assert_allclose(weights.sum(axis=1), 1, rtol=0, atol=ROUND_TOL)
    j, i = np.meshgrid(np.arange(3), np.arange(4), indexing="ij")
    nonconstant = np.stack(
        (1 + 0.1 * j + 0.03 * i, 0.4 * i - 0.2 * j, 0.1 * j - 0.3 * i), axis=-1
    )
    snapshots = np.stack((nonconstant, nonconstant * [1.1, 0.9, 1.2]))
    projected = conservative_project(snapshots, wx, wy)
    assert projected.shape == (2, 2, 3, 3)
    # Target[-5,-3]×[-5,0] covers only source xcell0 and 4:1 y overlap.
    np.testing.assert_allclose(
        projected[:, 0, 0],
        (4 * snapshots[:, 0, 0] + snapshots[:, 1, 0]) / 5,
        rtol=ROUND_TOL,
        atol=ROUND_TOL,
    )
    source_areas = np.diff(ye)[:, None] * np.diff(xe)[None, :]
    target_areas = np.diff(target_y)[:, None] * np.diff(target_x)[None, :]
    native_integrals = np.sum(snapshots * source_areas[None, ..., None], axis=(1, 2))
    target_integrals = np.sum(projected * target_areas[None, ..., None], axis=(1, 2))
    np.testing.assert_allclose(
        target_integrals,
        native_integrals,
        rtol=ROUND_TOL,
        atol=ROUND_TOL,
    )
    areas = np.array([[1.0, 3.0], [2.0, 4.0]])
    delta = np.broadcast_to([0.2, -0.3, 0.4], (2, 2, 2, 3)).copy()
    reference = np.broadcast_to([2.0, -3.0, 0.0], delta.shape).copy()
    reference[:, :, 1, 1] = 3.0
    norms = area_difference_metrics(delta, reference, areas)
    assert len(norms["per_time"]) == 2
    for row in [*norms["per_time"], norms["equal_positive_time_aggregate"]]:
        for component, value in enumerate((0.2, 0.3, 0.4)):
            np.testing.assert_allclose(row[component]["area_mean_MAE"], value)
            np.testing.assert_allclose(row[component]["area_mean_RMSE"], value)
            np.testing.assert_allclose(row[component]["integral_L1"], 10 * value)
            np.testing.assert_allclose(
                row[component]["integral_L2"], np.sqrt(10) * value
            )
        np.testing.assert_allclose(row[0]["relative_L2_pct"], 10.0)
        np.testing.assert_allclose(row[1]["relative_L2_pct"], 10.0)
        assert row[2]["relative_L2_pct"] is None
        assert row[2]["relative_reason"] == "zero reference norm"
    assert norms["integral_units_L1"] == ["m^3", "m^4/s", "m^4/s"]
    assert norms["integral_units_L2"] == ["m^2", "m^3/s", "m^3/s"]
    _receipt(
        tmp_path,
        {
            "signed_budget_defect_all3_components": defect.tolist(),
            "budget_tolerance_all3_components": (ROUND_TOL * scale).tolist(),
            "signed_projection_integral_defects": (
                target_integrals - native_integrals
            ).tolist(),
            "maximum_projection_coverage_defects_xy": [
                float(np.max(np.abs(w.sum(axis=1) - 1))) for w in (wx, wy)
            ],
            "norm_fixture": norms,
        },
    )


def test_r5_invalid_update_raw_persistence_hashes_and_exclusive_destination(
    tmp_path, monkeypatch
):
    from hydro_pinn.extensions import reference_2d
    from hydro_pinn.extensions.reference_2d import run_fixture, run_reference

    args = build_parser().parse_args(
        [
            "extension-2d-reference",
            "--config",
            "configs/extensions/case_2d.json",
            "--run-id",
            "ref_rect80_c045_v01",
        ]
    )
    assert args.command == "extension-2d-reference"
    assert args.release is None
    with pytest.raises(ValueError, match="not released"):
        run_reference(
            args.config.resolve(),
            args.run_id,
            protocol_path=args.protocol,
            root=tmp_path,
        )
    assert not (tmp_path / "results").exists()

    xe, ye = np.linspace(-1.0, 1.0, 9), np.linspace(-0.5, 0.5, 5)
    initial = np.zeros((4, 8, 3), dtype=np.float64)
    initial[..., 0] = 0.1
    initial[:, :4, 0] = 1.0
    run_id = "fixture_genuine_invalid_update"
    with pytest.raises(Reference2DError, match="updated state"):
        run_fixture(
            tmp_path,
            run_id,
            initial,
            xe,
            ye,
            np.array([0.0, 0.25]),
            imposed_dt=0.25,
        )
    run = tmp_path / run_id
    status = json.loads((run / "status.json").read_text())
    termination = json.loads((run / "termination.json").read_text())
    assert status["state"] == "failed"
    assert termination["termination_kind"] == "numerical_failure"
    assert termination["stage"] == "updated_state"
    assert termination["location"] is not None
    assert termination["context"]["step"] == 1
    assert termination["context"]["old_t_s"] == 0.0
    assert termination["context"]["dt"] == 0.25
    assert termination["context"]["sum_cfl"] > 1
    raw = np.load(run / "raw_failure.npz")
    np.testing.assert_array_equal(raw["old_state"], initial)
    assert np.any(raw["updated_state"][..., 0] <= 0)
    last = np.load(run / "last_valid_state.npz")
    np.testing.assert_array_equal(last["state_h_qx_qy"], initial)
    assert float(last["t_s"]) == 0.0
    assert int(last["step"]) == 0
    saved = np.load(run / "outputs.npz")
    np.testing.assert_array_equal(saved["t_s"], [0.0])
    np.testing.assert_array_equal(saved["h_m"][0], initial[..., 0])
    np.testing.assert_array_equal(saved["qx_m2_s"][0], initial[..., 1])
    np.testing.assert_array_equal(saved["qy_m2_s"][0], initial[..., 2])
    assert (run / "diagnostics.npz").is_file()
    hashes = json.loads((run / "hashes.json").read_text())
    assert set(hashes["mutable_lifecycle_files_excluded"]) == {
        "status.json",
        "metadata.json",
    }
    assert not ({"status.json", "metadata.json"} & set(hashes["files"]))
    for name, expected in hashes["files"].items():
        assert hashlib.sha256((run / name).read_bytes()).hexdigest() == expected
    for name, expected in hashes["source_sha256"].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected
    preserved = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in run.iterdir()
        if p.is_file()
    }
    with pytest.raises(FileExistsError):
        run_fixture(
            tmp_path,
            run_id,
            initial,
            xe,
            ye,
            np.array([0.0, 0.25]),
            imposed_dt=0.25,
        )
    assert preserved == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in run.iterdir()
        if p.is_file()
    }

    original_write = reference_2d._write
    observed_status = []
    failures_injected = 0

    def fail_first_termination_write(path, value, **kwargs):
        nonlocal failures_injected
        path = Path(path)
        if path.name == "termination.json":
            observed_status.append(
                json.loads((path.parent / "status.json").read_text())["state"]
            )
            if failures_injected == 0:
                failures_injected += 1
                raise OSError("injected termination finalization failure")
        return original_write(path, value, **kwargs)

    monkeypatch.setattr(reference_2d, "_write", fail_first_termination_write)
    still = np.zeros((2, 3, 3), dtype=np.float64)
    still[..., 0] = 0.7
    finalize_id = "fixture_genuine_finalization_failure"
    with pytest.raises(Reference2DError, match="injected termination"):
        run_fixture(
            tmp_path,
            finalize_id,
            still,
            np.linspace(-1.0, 1.0, 4),
            np.linspace(-1.0, 1.0, 3),
            np.array([0.0, 0.001]),
        )
    finalization_run = tmp_path / finalize_id
    finalization_status = json.loads((finalization_run / "status.json").read_text())
    finalization_termination = json.loads(
        (finalization_run / "termination.json").read_text()
    )
    assert failures_injected == 1
    assert "completed" not in observed_status
    assert finalization_status["state"] == "failed"
    assert finalization_termination["termination_kind"] == "numerical_failure"
    assert finalization_termination["stage"] == "terminal_finalization"
    np.testing.assert_array_equal(
        np.load(finalization_run / "last_valid_state.npz")["state_h_qx_qy"], still
    )
    np.testing.assert_array_equal(
        np.load(finalization_run / "raw_failure.npz")["old_state"], still
    )
    _receipt(
        tmp_path,
        {
            "failure_injection": "one OSError on first termination.json write",
            "observed_status_at_termination_writes": observed_status,
            "final_status": finalization_status["state"],
            "failure_stage": finalization_termination["stage"],
        },
    )
