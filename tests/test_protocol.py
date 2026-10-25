"""Required freeze, paired fairness, heldout and source-binding checks."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from hydro_pinn import protocol as protocol_module
from hydro_pinn.protocol import (
    SCIENTIFIC_MODULES,
    ProtocolError,
    file_sha256,
    freeze_protocol,
    load_protocol,
    protocol_digest,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path.resolve()
    candidate = json.loads((ROOT / "configs/protocol_candidate.json").read_text())
    inputs = [
        "configs/protocol_candidate.json",
        "docs/coordination/phase_04_review.json",
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        *candidate["input_sha256"],
    ]
    inputs += [
        f"src/hydro_pinn/{name}.py" for name in (*SCIENTIFIC_MODULES, "cli", "__main__")
    ]
    for name in inputs:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if (ROOT / name).is_file():
            shutil.copyfile(ROOT / name, destination)
        else:
            # The protocol test only hashes this disjoint worker's not-yet-written
            # source. Production freeze requires its actual source to exist.
            assert name == "src/hydro_pinn/experiments.py"
            destination.write_text("# Runner source fixture for protocol-only tests.\n")
    return root


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _freeze(root: Path) -> dict:
    return freeze_protocol(
        "configs/protocol_candidate.json", "configs/frozen_test.json", project_root=root
    )


def _update_fixture_inputs_and_review(root: Path, candidate: dict) -> None:
    candidate["input_sha256"] = {
        name: file_sha256(root / name) for name in candidate["input_sha256"]
    }
    _write(root / "configs/protocol_candidate.json", candidate)
    review_path = root / "docs/coordination/phase_04_review.json"
    review = json.loads(review_path.read_text())
    review["candidate_sha256"] = file_sha256(root / "configs/protocol_candidate.json")
    _write(review_path, review)


def test_freeze_canonical_matrix_exact_auxiliary_axes_and_no_execution(
    project: Path,
) -> None:
    frozen = _freeze(project)
    assert load_protocol("configs/frozen_test.json", project_root=project) == frozen
    assert len(frozen["matrix"]) == 24
    assert len([s for s in frozen["matrix"] if s["stage"] == "train"]) == 7
    assert len(frozen["pairs"]) == 3
    assert frozen["settings"]["seeds"] == [11, 22, 33]
    assert frozen["settings"]["adam_steps"] == 10000
    assert frozen["settings"]["optimizer_execution"]["defaults"]["betas"] == [
        0.9,
        0.999,
    ]
    assert frozen["settings"]["checkpoint_selection"]["selection_weights"] == {
        "pde": 1.0,
        "ic": 10.0,
        "bc": 10.0,
    }
    conservation = frozen["conservation"]
    assert len(conservation["auxiliary_records"]) == 36
    assert all(r["state"] == "planned" for r in conservation["auxiliary_records"])
    assert all(
        r["binding"]["status"] == "runtime_required"
        for r in conservation["auxiliary_records"]
    )
    reported = frozen["resolved"]["evaluation"]["times_s"]
    for count in (2000, 4000):
        axes = conservation["volume_axes"][str(count)]
        assert len(axes["x_m"]) == count
        assert axes["t_s"] == [0.0, *reported]
        assert all(-5.0 < x < 5.0 for x in axes["x_m"])
    for count in (1001, 2001):
        axes = conservation["boundary_axes"][str(count)]
        assert len(axes["t_s"]) == count
        assert axes["t_s"][0] == 0.0 and axes["t_s"][-1] == 0.5
        assert axes["x_m"] == [-5.0, 5.0]
        assert all(t in axes["t_s"] for t in reported)
    assert not (project / "runs").exists()
    assert not (project / "docs/final_run_manifest.json").exists()
    original = (project / "configs/frozen_test.json").read_bytes()
    with pytest.raises(FileExistsError, match="already exists"):
        _freeze(project)
    assert (project / "configs/frozen_test.json").read_bytes() == original


@pytest.mark.parametrize("problem", ["pair", "leakage", "reference"])
def test_reject_pair_mismatch_leakage_and_incompatible_case(
    project: Path, problem: str
) -> None:
    candidate = json.loads((project / "configs/protocol_candidate.json").read_text())
    if problem == "pair":
        path = project / candidate["training"]["adaptive"]
        document = json.loads(path.read_text())
        document["network"]["hidden_layers"][0] = 65
        _write(path, document)
    elif problem == "leakage":
        for name in ("baseline", "adaptive"):
            path = project / candidate["training"][name]
            document = json.loads(path.read_text())
            document["validation"]["times_s"][0] = 0.01
            _write(path, document)
        candidate["checkpoint_selection"]["validation"]["times_s"][0] = 0.01
    else:
        path = project / candidate["cases"]["E2"]
        document = json.loads(path.read_text())
        document["right_depth_m"] = 0.001
        document["boundary"]["right_h_m"] = 0.001
        _write(path, document)
    _update_fixture_inputs_and_review(project, candidate)
    with pytest.raises(
        ValueError,
        match={
            "pair": "pair mismatch",
            "leakage": "overlap",
            "reference": "right_depth|reference",
        }[problem],
    ):
        _freeze(project)
    assert not (project / "configs/frozen_test.json").exists()


def test_duplicate_matrix_ids_and_missing_quadrature_endpoint_rejected(
    project: Path,
) -> None:
    frozen = _freeze(project)
    duplicate = copy.deepcopy(frozen)
    duplicate["matrix"][1]["run_id"] = duplicate["matrix"][0]["run_id"]
    duplicate["protocol_sha256"] = protocol_digest(duplicate)
    _write(project / "configs/duplicate.json", duplicate)
    with pytest.raises(ProtocolError, match="duplicate"):
        load_protocol("configs/duplicate.json", project_root=project)
    bad_axes = copy.deepcopy(frozen)
    bad_axes["conservation"]["boundary_axes"]["1001"]["t_s"].pop(0)
    bad_axes["protocol_sha256"] = protocol_digest(bad_axes)
    _write(project / "configs/axes.json", bad_axes)
    with pytest.raises(ProtocolError, match="conservation axes"):
        load_protocol("configs/axes.json", project_root=project)


@pytest.mark.parametrize(
    "target", ["configs/evaluation.json", "src/hydro_pinn/train.py", "uv.lock"]
)
def test_changed_scientific_input_or_module_is_rejected(
    project: Path, target: str
) -> None:
    _freeze(project)
    with (project / target).open("a") as handle:
        handle.write("\n")
    with pytest.raises(ProtocolError, match="frozen.*hash"):
        load_protocol("configs/frozen_test.json", project_root=project)


def test_cli_additions_are_historical_and_runtime_change_is_not(
    project: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _freeze(project)
    for name in ("cli", "__main__"):
        with (project / f"src/hydro_pinn/{name}.py").open("a") as handle:
            handle.write("\n# Additive dispatch provenance fixture.\n")
    assert load_protocol("configs/frozen_test.json", project_root=project) == frozen
    altered = copy.deepcopy(frozen["environment"])
    altered["packages"]["torch"] = "changed-build-fixture"
    monkeypatch.setattr(protocol_module, "environment_snapshot", lambda: altered)
    with pytest.raises(ProtocolError, match="runtime environment"):
        load_protocol("configs/frozen_test.json", project_root=project)


def test_content_digest_is_verified_before_structure_or_execution(
    project: Path,
) -> None:
    frozen = _freeze(project)
    frozen["settings"]["adam_steps"] = 1
    _write(project / "configs/tampered.json", frozen)
    with pytest.raises(ProtocolError, match="protocol digest"):
        load_protocol("configs/tampered.json", project_root=project)
