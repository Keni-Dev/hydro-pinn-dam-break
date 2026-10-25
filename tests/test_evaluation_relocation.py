"""Relocated strict protocol and a real minimal complete record fixture.

The one-stage fixture substitutes only the fixed final-core matrix shape. The
ordinary loader still checks canonical digest, environment, every input and
scientific source; record verification and analytical execution are real.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest

from hydro_pinn import evaluation_workflow as workflow
from hydro_pinn import experiments
from hydro_pinn import protocol as protocol_module
from hydro_pinn.protocol import file_sha256, protocol_digest
from hydro_pinn.records import write_json_atomic

ROOT = Path(__file__).resolve().parents[1]


def copy_frozen_project(root: Path) -> dict:
    protocol = json.loads((ROOT / "configs/final_protocol.json").read_text())
    paths = {"configs/final_protocol.json", *protocol["scientific_source_sha256"]}

    def collect(value):
        if isinstance(value, dict):
            if {"path", "sha256"} <= value.keys():
                paths.add(value["path"])
            else:
                for item in value.values():
                    collect(item)
        elif isinstance(value, list):
            for item in value:
                collect(item)

    collect(protocol["inputs"])
    for name in paths:
        destination = root / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    return protocol


def test_strict_protocol_relocation_preserves_hash_and_refuses_other_valid_digest(
    tmp_path: Path,
) -> None:
    origin = tmp_path.resolve() / "origin"
    protocol = copy_frozen_project(origin)
    relocated = tmp_path.resolve() / "extracted"
    shutil.copytree(origin, relocated)
    original_path = origin / "configs/final_protocol.json"
    original_hash = file_sha256(original_path)
    manifest = {
        "protocol_path": str(original_path),
        "protocol_sha256": protocol["protocol_sha256"],
    }
    preserved = copy.deepcopy(manifest)
    local, loaded = workflow.resolve_manifest_protocol(manifest, relocated)
    assert local == relocated / "configs/final_protocol.json"
    assert loaded == protocol
    assert file_sha256(local) == original_hash
    assert manifest == preserved
    # A different but structurally valid, self-consistent local protocol must
    # not be rescued by the still-existing valid original checkout.
    changed = copy.deepcopy(protocol)
    changed["created_at_utc"] = "fixture-other-protocol-identity"
    changed["protocol_sha256"] = protocol_digest(changed)
    write_json_atomic(local, changed, overwrite=True)
    assert protocol_module.load_protocol(local, project_root=relocated) == changed
    with pytest.raises(ValueError, match="another protocol"):
        workflow.resolve_manifest_protocol(manifest, relocated)
    assert file_sha256(original_path) == original_hash
    assert manifest == preserved


def test_actual_minimal_complete_fixture_validates_after_extraction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    origin = tmp_path.resolve() / "fixture-origin"
    canonical = copy_frozen_project(origin)
    fixture = copy.deepcopy(canonical)
    stage = {
        "stage_id": "relocation-fixture-reference",
        "stage": "reference",
        "case_id": "E1",
        "run_id": "relocation-fixture-reference",
        "depends_on": [],
    }
    fixture.update(
        purpose="bounded_relocation_fixture_not_final_results",
        protocol_id="relocation-fixture",
        matrix=[stage],
        pairs=[],
        conservation={"auxiliary_records": []},
    )
    fixture["protocol_sha256"] = protocol_digest(fixture)
    protocol_path = origin / "configs/final_protocol.json"
    protocol_path.rename(protocol_path.with_name("preserved_strict_protocol.json"))
    write_json_atomic(protocol_path, fixture)
    strict_structure = protocol_module._validate_structure

    def minimal_fixture_structure(document):
        if document.get("purpose") != fixture["purpose"]:
            return strict_structure(document)
        assert document["matrix"] == [stage]
        assert document["pairs"] == []
        assert document["conservation"] == {"auxiliary_records": []}
        for name in ("settings", "resolved", "paths", "scientific_source_sha256"):
            assert document[name] == fixture[name]

    monkeypatch.setattr(
        protocol_module, "_validate_structure", minimal_fixture_structure
    )
    # No training, benchmark, or final-error evaluation: one actual analytical
    # record is enough to exercise the complete manifest's relocation path.
    manifest = experiments.run_matrix(protocol_path, project_root=origin)
    assert manifest["workflow_status"] == "completed"
    assert len(manifest["entries"]) == 1
    assert manifest["entries"][0]["state"] == "completed"
    original_hashes = {
        str(path.relative_to(origin)): file_sha256(path)
        for path in origin.rglob("*")
        if path.is_file()
    }
    relocated = tmp_path.resolve() / "fixture-extracted"
    shutil.copytree(origin, relocated)
    local_manifest = relocated / fixture["paths"]["manifest"]
    manifest_hash = file_sha256(local_manifest)
    observed, loaded = workflow.validated_manifest(local_manifest, relocated)
    assert observed == manifest
    assert loaded == fixture
    assert file_sha256(relocated / "configs/final_protocol.json") == file_sha256(
        protocol_path
    )
    assert file_sha256(local_manifest) == manifest_hash
    assert all(
        file_sha256(origin / name) == digest for name, digest in original_hashes.items()
    )
    assert all(
        file_sha256(relocated / name) == digest
        for name, digest in original_hashes.items()
    )
