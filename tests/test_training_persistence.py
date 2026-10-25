"""CSV append durability and unchanged compatible checkpoint continuation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from hydro_pinn import train as training_module
from hydro_pinn.train import TrainingError, load_training_checkpoint, train_run

ROOT = Path(__file__).resolve().parents[1]


def _train(directory: Path, run_id: str, **kwargs: Any) -> Path:
    return train_run(
        ROOT / "configs/cases/e1_wet.json",
        ROOT / "configs/training/baseline.json",
        11,
        "cpu",
        "float64",
        run_id,
        output_root=directory,
        overrides={
            "training": {
                "network": {"hidden_layers": [4, 4]},
                "sampling": {
                    "interior_points": 7,
                    "initial_points": 6,
                    "boundary_points_per_side": 3,
                },
                "optimizer": {"steps": 3},
                "validation": {"grid": {"cells": 7}},
            }
        },
        **kwargs,
    )


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _assert_exact(first: Any, second: Any) -> None:
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_exact(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second, strict=True):
            _assert_exact(left, right)
    else:
        assert first == second


def test_step_rows_are_readable_and_full_rewrite_occurs_only_at_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = tmp_path.resolve()
    original = csv.DictWriter.writerows
    full_writes: list[int] = []
    observed_steps: list[int] = []

    def count_full_rewrite(writer, rows):
        full_writes.append(len(rows))
        return original(writer, rows)

    def observe(event):
        rows = _rows(directory / "append/history.csv")
        step = event["step"]
        assert [int(row["step"]) for row in rows] == list(range(1, step + 1))
        assert all(row["phase"] == "adam" for row in rows)
        manifest = json.loads(
            (directory / "append/checkpoint_manifest.json").read_text()
        )
        assert manifest["step"] == step
        assert full_writes == []
        observed_steps.append(step)

    monkeypatch.setattr(csv.DictWriter, "writerows", count_full_rewrite)
    path = _train(directory, "append", step_observer=observe)
    assert observed_steps == [1, 2, 3]
    assert full_writes == [3]
    rows = _rows(path / "history.csv")
    assert len(rows) == 3
    assert "convergence not established" in rows[-1]["stop_reason"]
    assert (
        sum("phase" in line for line in (path / "history.csv").read_text().splitlines())
        == 1
    )


@pytest.mark.parametrize("failure", ["failure", "interruption"])
def test_second_step_stop_preserves_first_row_and_resumes_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    directory = tmp_path.resolve()
    _train(directory, "whole")
    original = training_module.component_gradients
    calls = 0

    def stop_on_second_step(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 4:
            if failure == "interruption":
                raise KeyboardInterrupt("stop during attempted Adam step 2")
            raise TrainingError("stop during attempted Adam step 2")
        return original(*args, **kwargs)

    monkeypatch.setattr(training_module, "component_gradients", stop_on_second_step)
    error = KeyboardInterrupt if failure == "interruption" else TrainingError
    with pytest.raises(error, match="attempted Adam step 2"):
        _train(directory, "stopped")
    monkeypatch.setattr(training_module, "component_gradients", original)
    path = directory / "stopped"
    rows = _rows(path / "history.csv")
    assert [row["phase"] for row in rows] == ["adam", "failure"]
    assert rows[0]["step"] == "1" and rows[1]["stop_reason"]
    status = json.loads((path / "status.json").read_text())
    assert status["state"] == ("interrupted" if failure == "interruption" else "failed")
    stopped, _, _ = load_training_checkpoint("stopped", directory)
    assert stopped["step"] == 1
    preserved = {
        file.name: hashlib.sha256(file.read_bytes()).hexdigest()
        for file in path.iterdir()
        if file.is_file()
    }
    resumed_path = _train(directory, "resumed", resume_from="stopped")
    whole, _, _ = load_training_checkpoint("whole", directory, require_completed=True)
    resumed, _, _ = load_training_checkpoint(
        "resumed", directory, require_completed=True
    )
    for key in (
        "latest_model_state",
        "selected_model_state",
        "optimizer_state",
        "weights",
        "rng_state",
        "weight_update_state",
        "selected_step",
        "selected_validation_score",
    ):
        _assert_exact(whole[key], resumed[key])
    assert [row["step"] for row in _rows(resumed_path / "history.csv")] == ["2", "3"]
    for name, digest in preserved.items():
        assert hashlib.sha256((path / name).read_bytes()).hexdigest() == digest
