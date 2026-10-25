"""Fixed physics-only samples, exact IC/BC branches, and partition readback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from hydro_pinn.config import load_config, resolve_run_config
from hydro_pinn.sampling import (
    SAMPLE_KEYS,
    generate_training_samples,
    generate_validation_samples,
    load_samples,
    save_samples,
)

ROOT = Path(__file__).resolve().parents[1]


def resolved(case_name: str = "e1_wet") -> dict:
    return resolve_run_config(
        ROOT / f"configs/cases/{case_name}.json",
        training_path=ROOT / "configs/training/baseline.json",
        smoke_path=ROOT / "configs/smoke.json",
        seed=11,
    )


@pytest.mark.parametrize("case_name", ["e1_wet", "e2_dry"])
def test_seeded_samples_preserve_exact_physical_targets_and_no_labels(
    case_name: str,
) -> None:
    config = resolved(case_name)
    case, training = config["case"], config["training"]
    first = generate_training_samples(case, training, 11)
    repeated = generate_training_samples(case, training, 11)
    different = generate_training_samples(case, training, 22)
    assert set(first) == SAMPLE_KEYS
    for key in first:
        assert_array_equal(first[key], repeated[key])
        assert first[key].dtype == np.float64
    assert not np.array_equal(first["interior_xt_nd"], different["interior_xt_nd"])
    assert first["interior_xt_nd"].shape == (128, 2)
    assert first["initial_xt_nd"].shape == first["initial_hq_nd"].shape == (64, 2)
    assert first["boundary_xt_nd"].shape == first["boundary_hq_nd"].shape == (64, 2)
    domain = np.array(case["domain_m"]) / case["scales"]["Lc_m"]
    interior = first["interior_xt_nd"]
    assert np.all((interior[:, 0] > domain[0]) & (interior[:, 0] < domain[1]))
    assert np.all(
        (interior[:, 1] > 0)
        & (interior[:, 1] < case["final_time_s"] / case["scales"]["Tc_s"])
    )
    dam = case["dam_position_m"] / case["scales"]["Lc_m"]
    initial_x = first["initial_xt_nd"][:, 0]
    assert np.count_nonzero(initial_x < dam) == np.count_nonzero(initial_x > dam) == 32
    assert np.count_nonzero(initial_x == dam) == 0
    assert np.count_nonzero(first["initial_xt_nd"][:, 1]) == 0
    for component in ("initial", "boundary"):
        points, targets = first[f"{component}_xt_nd"], first[f"{component}_hq_nd"]
        expected_h = (
            np.where(points[:, 0] < dam, case["left_depth_m"], case["right_depth_m"])
            / case["scales"]["H_m"]
        )
        assert_array_equal(targets[:, 0], expected_h)
        assert np.count_nonzero(targets[:, 1]) == 0
    assert_array_equal(first["boundary_xt_nd"][:32, 0], np.full(32, domain[0]))
    assert_array_equal(first["boundary_xt_nd"][32:, 0], np.full(32, domain[1]))
    if case_name == "e2_dry":
        assert np.count_nonzero(first["initial_hq_nd"][initial_x > dam]) == 0
        assert np.count_nonzero(first["boundary_hq_nd"][32:]) == 0


def test_validation_declared_coordinates_and_training_evaluation_separation() -> None:
    config = resolved()
    case, training = config["case"], config["training"]
    validation = generate_validation_samples(case, training, 11)
    train = generate_training_samples(case, training, 11)
    scales = case["scales"]
    points = validation["interior_xt_nd"]
    assert points.shape == (31 * 5, 2)
    assert_array_equal(
        np.unique(points[:, 1]),
        np.array(training["validation"]["times_s"]) / scales["Tc_s"],
    )
    assert len(validation["initial_xt_nd"]) == 30  # Centre x=dam omitted.
    assert len(validation["boundary_xt_nd"]) == 10
    assert 0 not in validation["initial_xt_nd"][:, 0]
    train_coordinates = {
        tuple(row)
        for key in ("interior_xt_nd", "initial_xt_nd", "boundary_xt_nd")
        for row in train[key]
    }
    validation_coordinates = {
        tuple(row)
        for key in ("interior_xt_nd", "initial_xt_nd", "boundary_xt_nd")
        for row in validation[key]
    }
    assert train_coordinates.isdisjoint(validation_coordinates)
    smoke_times = config["evaluation"]["times_s"]
    final_times = load_config(ROOT / "configs/evaluation.json")["times_s"]
    assert set(training["validation"]["times_s"]).isdisjoint(smoke_times)
    assert set(training["validation"]["times_s"]).isdisjoint(final_times)
    other_seed = generate_validation_samples(case, training, 22)
    for key in validation:
        assert_array_equal(validation[key], other_seed[key])


def test_saved_samples_roundtrip_exclusive_creation_and_schema(tmp_path: Path) -> None:
    config = resolved("e2_dry")
    samples = generate_training_samples(config["case"], config["training"], 11)
    path = tmp_path / "sampling.npz"
    save_samples(path, samples)
    loaded = load_samples(path)
    for key in samples:
        assert_array_equal(loaded[key], samples[key])
    with pytest.raises(FileExistsError):
        save_samples(path, samples)
    leaked = tmp_path / "interior_label_leak.npz"
    np.savez(leaked, **samples, interior_hq_nd=np.zeros((128, 2)))
    with pytest.raises(ValueError, match="no interior labels"):
        load_samples(leaked)
