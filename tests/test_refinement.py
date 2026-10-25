"""Verification-grid reconstruction and bounded analytical comparisons.

These checks verify the comparison operator and required data partitions. They
make no monotonic-error or convergence-order assertion for a shock solution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from hydro_pinn.config import load_config
from hydro_pinn.lax_wendroff import piecewise_constant_reconstruct, solve
from hydro_pinn.references import evaluate_reference, solve_wet_star

ROOT = Path(__file__).resolve().parents[1]


def test_reconstruction_edges_faces_and_immediate_neighbors() -> None:
    edges = np.array([-5.0, -2.5, 0.0, 2.5, 5.0])
    values = np.array([10.0, 20.0, 30.0, 40.0])
    x = np.array(
        [
            -5.0,
            np.nextafter(-2.5, -np.inf),
            -2.5,
            np.nextafter(-2.5, np.inf),
            np.nextafter(0.0, -np.inf),
            0.0,
            np.nextafter(0.0, np.inf),
            2.5,
            5.0,
        ]
    )
    assert_allclose(
        piecewise_constant_reconstruct(edges, values, x),
        [10, 10, 20, 20, 20, 30, 30, 40, 40],
        rtol=0,
        atol=0,
    )


def test_reconstruction_preserves_native_averages_and_leading_axes() -> None:
    edges = np.array([-1.0, 0.0, 1.0])
    x = np.array([-0.75, -0.25, 0.25, 0.75])
    # Exact cell averages of x^2 equal 1/3 on both cells. Reconstruction
    # retains those averages rather than pretending they are x^2 point values.
    values = np.array([[1 / 3, 1 / 3], [2 / 3, 2 / 3]])
    reconstructed = piecewise_constant_reconstruct(edges, values, x)
    assert reconstructed.shape == (2, 4)
    assert_allclose(reconstructed, np.repeat(values, 2, axis=1), rtol=0, atol=0)
    assert not np.allclose(reconstructed[0], x**2)
    assert_allclose(np.sum(values[0] * np.diff(edges)), 2 / 3, atol=1e-15)


def test_verification_grid_covers_wave_features_and_far_fields() -> None:
    wet = load_config(ROOT / "configs/cases/e1_wet.json")
    dry = load_config(ROOT / "configs/cases/e2_dry.json")
    verification = load_config(ROOT / "configs/verification.json")
    left, right = verification["grid"]["domain_m"]
    count = verification["grid"]["cells"]
    x = left + (np.arange(count) + 0.5) * (right - left) / count
    elapsed = verification["times_s"][-1]
    band = verification["wave_band_half_width_m"]
    star = solve_wet_star(wet)
    c0 = np.sqrt(wet["gravity_m_s2"] * wet["left_depth_m"])
    fan = (x > -c0 * elapsed) & (x < star.fan_tail_speed_m_s * elapsed)
    middle = (x > star.fan_tail_speed_m_s * elapsed) & (
        x < star.shock_speed_m_s * elapsed
    )
    assert fan.any() and middle.any()
    for wave in (star.shock_speed_m_s * elapsed, 2 * c0 * elapsed):
        assert np.any((x < wave) & (np.abs(x - wave) <= band))
        assert np.any((x > wave) & (np.abs(x - wave) <= band))
    wet_values = evaluate_reference(wet, x, [elapsed])
    dry_values = evaluate_reference(dry, x, [elapsed])
    assert_allclose(wet_values.h_m[0, middle], star.h_m, rtol=0, atol=1e-12)
    assert_allclose(wet_values.h_m[0, 0], 1, rtol=0, atol=0)
    assert_allclose(wet_values.h_m[0, -1], 0.1, rtol=0, atol=0)
    assert_allclose(dry_values.h_m[0, -1], 0, rtol=0, atol=0)
    assert_allclose(dry_values.q_m2_s[0, -1], 0, rtol=0, atol=0)


@pytest.mark.parametrize("cells", [100, 200])
def test_wet_comparator_uses_exact_times_and_common_point_reference(cells: int) -> None:
    case = load_config(ROOT / "configs/cases/e1_wet.json")
    verification = load_config(ROOT / "configs/verification.json")
    times = np.asarray(verification["times_s"], dtype=np.float64)
    result = solve(case, cells, times)
    assert result.failure is None
    assert np.array_equal(result.t_s, times)
    assert result.h_m.shape == result.q_m2_s.shape == (len(times), cells)
    assert_allclose(np.diff(result.cell_edges_m), 10 / cells, rtol=0, atol=2e-15)
    count = verification["grid"]["cells"]
    x = -5 + (np.arange(count) + 0.5) * 10 / count
    reference = evaluate_reference(case, x, times)
    for native, exact in (
        (result.h_m, reference.h_m),
        (result.q_m2_s, reference.q_m2_s),
    ):
        compared = piecewise_constant_reconstruct(result.cell_edges_m, native, x)
        assert compared.shape == exact.shape == (len(times), 129)
        assert np.isfinite(compared).all()
        rmse = np.sqrt(np.mean((compared - exact) ** 2, axis=1))
        relative_l2 = np.linalg.norm(compared - exact, axis=1) / np.linalg.norm(
            exact, axis=1
        )
        assert np.isfinite(rmse).all() and np.isfinite(relative_l2).all()
        # Nonzero error is expected from the numerical method and its
        # reconstruction. No accuracy target is invented for this check.
        assert np.any(rmse > 0)
