"""Focused fixtures for gauge means, distance adjustment, and frozen inputs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from hydro_pinn.extensions import inverse_sensitivity as sensitivity


def test_conservative_gauge_extraction_and_time_major_order():
    # The last target cell contains half of source cell 2 and all of cell 3.
    native = {
        "cell_edges_m": np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
        "t_s": np.array([0.1, 0.2]),
        "h_m": np.array([[2.0, 4.0, 6.0, 8.0], [12.0, 14.0, 16.0, 18.0]]),
    }
    common_edges = np.array([0.0, 0.5, 2.5, 4.0])
    actual = sensitivity.extract_gauge_means(native, common_edges, [0, 1], [0, 2])
    np.testing.assert_allclose(actual, [2.0, 22 / 3, 12.0, 52 / 3], rtol=1e-15)
    assert actual.shape == (4,)


def test_signed_pair_distance_adjustment_and_independent_accumulation():
    delta = np.array([0.006, -0.008, 0.004])
    indicator = np.array([0.001, 0.003, 0.006])
    stats = sensitivity.pair_statistics(delta, indicator, 0.01)
    np.testing.assert_array_equal(stats["signed_delta_m"], delta)
    np.testing.assert_array_equal(stats["combined_indicator_U_m"], indicator)
    np.testing.assert_allclose(stats["adjusted_magnitude_s_m"], [0.005, 0.005, 0])
    assert stats["raw_RMS_m"] == pytest.approx(np.sqrt(116e-6 / 3))
    assert stats["raw_max_abs_m"] == pytest.approx(0.008)
    assert stats["raw_L2_m"] == pytest.approx(np.sqrt(116e-6))
    assert stats["U_L2_m"] == pytest.approx(np.sqrt(46e-6))
    assert stats["adjusted_RMS_m"] == pytest.approx(np.sqrt(50e-6 / 3))
    assert stats["adjusted_max_m"] == pytest.approx(0.005)
    assert stats["adjusted_L2_m"] == pytest.approx(np.sqrt(50e-6))
    assert stats["d_raw"] == pytest.approx(np.sqrt(1.16))
    assert stats["d_adj"] == pytest.approx(np.sqrt(0.5))
    assert stats["max_adjusted_over_sigma"] == pytest.approx(0.5)
    assert stats["visible_sample_count"] == 2
    assert not stats["indicator_dominated"]
    assert stats["raw_pass"]
    assert not stats["adjusted_pass"]
    assert stats["adjustment_changes_screen"]

    # Independent samples accumulate without a per-component one-sigma gate.
    accumulated = sensitivity.pair_statistics(np.full(15, 0.006), np.zeros(15), 0.01)
    assert accumulated["max_adjusted_over_sigma"] < 1
    assert accumulated["d_adj"] == pytest.approx(np.sqrt(15) * 0.6)
    assert accumulated["adjusted_pass"]


def test_frozen_protocol_and_source_hash_reject_changed_bytes(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("original source\n")
    expected = {str(source): hashlib.sha256(source.read_bytes()).hexdigest()}
    sensitivity.assert_hashes(expected)
    config = json.loads(Path("configs/extensions/inverse_sensitivity.json").read_text())
    protocol = tmp_path / "protocol.json"
    original = json.dumps({"config": config, "input_sha256": expected})
    protocol.write_text(original)
    protocol.with_suffix(".sha256").write_text(
        hashlib.sha256(protocol.read_bytes()).hexdigest() + "\n"
    )
    assert sensitivity.verify_frozen(protocol)["input_sha256"] == expected
    protocol.write_text(original + "\n")
    with pytest.raises(ValueError):
        sensitivity.verify_frozen(protocol)
    protocol.write_text(original)
    source.write_text("changed source\n")
    with pytest.raises(ValueError):
        sensitivity.assert_hashes(expected)
    with pytest.raises(ValueError):
        sensitivity.verify_frozen(protocol)


def test_prepare_refuses_existing_result_or_evidence_root(tmp_path, monkeypatch):
    config = json.loads(Path("configs/extensions/inverse_sensitivity.json").read_text())
    monkeypatch.setattr(sensitivity, "assert_hashes", lambda _: None)
    for existing_root_key in ("output_root", "evidence_root"):
        scope = tmp_path / existing_root_key
        output = scope / "results"
        evidence = scope / "evidence"
        config["output_root"] = str(output)
        config["evidence_root"] = str(evidence)
        existing = output if existing_root_key == "output_root" else evidence
        existing.mkdir(parents=True)
        sentinel = existing / "retained.txt"
        sentinel.write_text("retain these bytes\n")
        config_path = tmp_path / f"{existing_root_key}.json"
        config_path.write_text(json.dumps(config))
        with pytest.raises(FileExistsError):
            sensitivity.prepare(config_path)
        assert sentinel.read_text() == "retain these bytes\n"


def test_execute_refuses_second_launch_without_a_solve(tmp_path, monkeypatch):
    output, evidence = tmp_path / "results", tmp_path / "evidence"
    evidence.mkdir()
    launch = evidence / "launch.json"
    launch.write_text("existing launch must remain\n")
    frozen = {"config": {"output_root": str(output), "evidence_root": str(evidence)}}
    monkeypatch.setattr(sensitivity, "verify_frozen", lambda _: frozen)
    monkeypatch.setattr(sensitivity, "_host", lambda: {})
    monkeypatch.setattr(
        sensitivity,
        "solve_reference",
        lambda *_args, **_kwargs: pytest.fail("second launch called the solver"),
    )
    protocol = tmp_path / "protocol.json"
    protocol.write_text("test protocol\n")
    with pytest.raises(FileExistsError):
        sensitivity.execute(protocol)
    assert launch.read_text() == "existing launch must remain\n"
