"""Verify installed package behavior and honest CLI capabilities."""

import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path

import pytest

import hydro_pinn

ROOT = Path(__file__).resolve().parents[1]


def invoke(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "hydro_pinn", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def test_installed_import_and_version() -> None:
    assert hydro_pinn.__version__ == "0.1.0"
    result = invoke("--version")
    assert result.returncode == 0
    assert result.stdout.strip() == hydro_pinn.__version__


def test_help_only_advertises_implemented_command() -> None:
    result = invoke("--help")
    assert result.returncode == 0
    assert "validate-config" in result.stdout
    for future_command in ("reference --", "solve --", "train --", "reproduce --"):
        assert future_command not in result.stdout
    child = invoke("validate-config", "--help")
    assert child.returncode == 0
    assert "metres" in child.stdout
    assert "does not create runs/" in " ".join(child.stdout.split())


def test_cli_validates_case_without_creating_output() -> None:
    result = invoke(
        "validate-config", "--config", str(ROOT / "configs/cases/e2_dry.json")
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["valid"] is True


def test_cli_reports_missing_and_invalid_inputs(tmp_path: Path) -> None:
    missing = invoke("validate-config", "--config", str(tmp_path / "absent.json"))
    assert missing.returncode == 2
    assert "Invalid configuration" in missing.stderr
    assert "Traceback" not in missing.stderr
    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"config_type": "case"}', encoding="utf-8")
    result = invoke("validate-config", "--config", str(invalid))
    assert result.returncode == 2
    assert "Invalid configuration" in result.stderr
    assert "Traceback" not in result.stderr


def test_cli_rejects_unimplemented_command() -> None:
    result = invoke("train")
    assert result.returncode == 2
    assert "invalid choice" in result.stderr


def test_scientific_stack_cpu_float64(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Check usable dependencies on CPU without running a research experiment."""
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    import matplotlib
    import numpy as np
    import torch
    from matplotlib.figure import Figure
    from scipy.optimize import brentq

    matplotlib.use("Agg")
    x = torch.tensor([1.0, 2.0], dtype=torch.float64, device="cpu", requires_grad=True)
    gradient = torch.autograd.grad((x**3).sum(), x)[0]
    np.testing.assert_allclose(gradient.detach().numpy(), [3.0, 12.0], rtol=0, atol=0)
    assert brentq(lambda value: value**2 - 2, 1.0, 2.0) == pytest.approx(2**0.5)
    figure = Figure()
    figure.subplots().plot([0, 1], [0, 1])
    output = BytesIO()
    figure.savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")


def test_resolved_inputs_and_actual_metadata_survive_failed_record(
    tmp_path: Path,
) -> None:
    """Exercise the public foundation APIs together with actual CPU facts."""
    from hydro_pinn.config import resolve_run_config
    from hydro_pinn.records import RunRecord

    resolved = resolve_run_config(
        ROOT / "configs/cases/e2_dry.json",
        ROOT / "configs/training/baseline.json",
        ROOT / "configs/evaluation.json",
        seed=11,
        run_id="foundation-contract-check",
    )
    with RunRecord.create(
        tmp_path / "runs",
        "foundation-contract-check",
        resolved,
        project_root=ROOT,
    ) as record:
        metadata = json.loads((record.path / "metadata.json").read_text())
        assert metadata["case"] == "E2"
        assert metadata["variant"] == "baseline"
        assert metadata["seed"] == 11
        assert metadata["numerical_floor"] == 1e-6
        assert metadata["device"]["requested"] == "cpu"
        assert metadata["device"]["available"] is True
        assert metadata["software"]["packages"]["torch"]["version"]
        assert len(metadata["input_hashes"]) == 3
        snapshot = json.loads((record.path / "resolved_config.json").read_text())
        assert snapshot["case"]["right_depth_m"] == 0
        record.fail("intentional software fixture", stage="foundation_contract_check")
    assert record.status["state"] == "failed"
    assert not (record.path / "outputs.npz").exists()
