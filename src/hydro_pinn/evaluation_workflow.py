"""Hash-bound evaluation and plotting of a terminal frozen run matrix."""

from __future__ import annotations

import copy
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from hydro_pinn.conservation import (
    auxiliary_record,
    file_sha256,
    native_solver_mass,
    quadrature_mass,
    quadrature_sensitivity,
)
from hydro_pinn.evaluate import (
    EvaluationError,
    aggregate_metric_rows,
    common_grid_fields,
    evaluate_fields,
    metric_definitions,
    unavailable_metric_rows,
)
from hydro_pinn.experiments import _stage_info, execution_plan
from hydro_pinn.protocol import load_protocol
from hydro_pinn.records import write_json_atomic

ROOT = Path(__file__).resolve().parents[2]


def _path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if ".." in path.parts:
        raise EvaluationError("evaluation paths must not contain parent traversal")
    path = path if path.is_absolute() else root / path
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise EvaluationError("evaluation paths must not follow symlinks")
    return path


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def validated_manifest(
    manifest_path: str | Path, project_root: str | Path
) -> tuple[dict, dict]:
    """Verify terminal matrix identities and the actual immutable run artifacts."""
    root = Path(project_root).absolute()
    manifest = _json(_path(root, manifest_path))
    if manifest.get("workflow_status") not in {
        "completed",
        "completed_with_diagnostics",
    }:
        raise EvaluationError("final evaluation requires the complete terminal matrix")
    protocol = load_protocol(manifest["protocol_path"], project_root=root)
    if manifest["protocol_sha256"] != protocol["protocol_sha256"]:
        raise EvaluationError("manifest belongs to another protocol")
    execution = manifest["execution"]
    if execution["namespace"] is None:
        expected_execution = execution_plan(protocol)
    else:
        expected_execution = execution_plan(
            protocol,
            namespace=execution["namespace"],
            output_root=execution["output_root"],
            manifest_path=execution["manifest"],
        )
    if execution != expected_execution:
        raise EvaluationError("manifest execution destinations or IDs are inconsistent")
    entries = {entry["stage_id"]: entry for entry in manifest["entries"]}
    if len(entries) != len(protocol["matrix"]) or len(entries) != len(
        manifest["entries"]
    ):
        raise EvaluationError("manifest matrix coverage differs from frozen protocol")
    for stage in protocol["matrix"]:
        entry = entries.get(stage["stage_id"])
        if entry is None or entry["frozen_stage"] != stage:
            raise EvaluationError("manifest stage differs from immutable frozen matrix")
        if entry["state"] != "completed" and not (
            stage["stage"] == "solve"
            and entry["state"] == "failed"
            and entry["failure_category"] == "numerical_failure"
        ):
            raise EvaluationError(
                "matrix has unresolved non-numerical/incomplete stages"
            )
        info = _stage_info(
            protocol,
            stage,
            entry,
            entries,
            root,
            require_completed=entry["state"] == "completed",
            execution=execution,
        )
        if info["record_state"] != entry["state"]:
            raise EvaluationError("manifest/record terminal states differ")
        if entry["result"]["files_sha256"] != info["files_sha256"]:
            raise EvaluationError("saved run artifacts changed after matrix execution")
    return manifest, protocol


def _identity(entry: dict) -> dict:
    stage = entry["frozen_stage"]
    return {
        "run_id": entry["actual_run_id"],
        "case_id": stage["case_id"],
        "method": {
            "reference": "analytical_reference",
            "solve": "lax_wendroff",
            "predict": "pinn",
            "train": "pinn_training",
        }[stage["stage"]],
        "variant": stage.get("variant"),
        "seed": stage.get("seed"),
        "cells": stage.get("cells"),
    }


def saved_datasets(manifest: dict, protocol: dict, root: Path) -> list[dict]:
    """Read source arrays/diagnostics without mutating completed source records."""
    datasets = []
    entries = {entry["stage_id"]: entry for entry in manifest["entries"]}
    for entry in manifest["entries"]:
        stage = entry["frozen_stage"]
        if stage["stage"] == "train":
            continue
        path = _path(
            root, Path(manifest["execution"]["output_root"]) / entry["actual_run_id"]
        )
        metadata = _json(path / "metadata.json")
        data = {
            **_identity(entry),
            "stage_id": stage["stage_id"],
            "state": entry["state"],
            "reason": entry.get("reason") or entry["result"]["stop_reason"],
            "provenance": {
                "resolved_config_sha256": metadata["resolved_config_sha256"],
                "files_sha256": entry["result"]["files_sha256"],
                "run_path": str(path.relative_to(root)),
            },
            "resolved_config_sha256": metadata["resolved_config_sha256"],
        }
        if entry["state"] == "completed":
            with np.load(path / "outputs.npz", allow_pickle=False) as saved:
                native = {name: saved[name].copy() for name in saved.files}
            data.update(common_grid_fields(native, protocol["resolved"]["evaluation"]))
            data["native_fields"] = native
        if (path / "diagnostics.json").exists():
            data["diagnostics"] = _json(path / "diagnostics.json")
        if stage["stage"] == "solve":
            data["mass"] = native_solver_mass(
                protocol["resolved"]["cases"][stage["case_id"]],
                data["diagnostics"],
                data.get("native_fields"),
            )
        if stage["stage"] == "predict":
            train_entry = entries[stage["training_stage_id"]]
            train_path = _path(
                root,
                Path(manifest["execution"]["output_root"])
                / train_entry["actual_run_id"],
            )
            with (train_path / "history.csv").open(newline="") as handle:
                data["history"] = list(csv.DictReader(handle))
            data["training_run_id"] = train_entry["actual_run_id"]
        datasets.append(data)
    return datasets


def _conservation_for_source(
    protocol: dict, manifest: dict, dataset: dict, root: Path
) -> dict:
    arrays, artifacts = {}, []
    plans = [
        plan
        for plan in protocol["conservation"]["auxiliary_records"]
        if plan["source_stage_id"] == dataset["stage_id"]
    ]
    for plan in plans:
        values, artifact = auxiliary_record(
            protocol,
            plan,
            dataset["run_id"],
            project_root=root,
            execution=manifest["execution"],
        )
        arrays[(plan["kind"], plan["axis_key"])] = values
        artifacts.append(artifact)
    cells = protocol["resolved"]["evaluation"]["mass_quadrature"]["volume"]["cells"]
    nodes = protocol["resolved"]["evaluation"]["mass_quadrature"]["boundary_flux"][
        "nodes"
    ]
    case = protocol["resolved"]["cases"][dataset["case_id"]]
    levels = [
        quadrature_mass(
            case, arrays[("volume", str(n))], arrays[("boundary_flux", str(k))]
        )
        for n, k in zip(cells, nodes, strict=True)
    ]
    return {
        **levels[1],
        "base_resolution": levels[0],
        "sensitivity": quadrature_sensitivity(*levels),
        "auxiliary_records": artifacts,
        "primary_reported_resolution": "doubled",
    }


def _source_files(manifest: dict, root: Path) -> list[dict]:
    result = []
    for entry in manifest["entries"]:
        directory = Path(manifest["execution"]["output_root"]) / entry["actual_run_id"]
        result.extend(
            {"path": str(directory / name), "sha256": digest}
            for name, digest in entry["result"]["files_sha256"].items()
        )
    return result


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "run_id",
        "case_id",
        "method",
        "variant",
        "seed",
        "cells",
        "time_s",
        "field",
        "region",
        "rmse",
        "rmse_reason",
        "relative_l2",
        "relative_l2_reason",
        "selected_points",
        "eligible_points",
        "total_points",
        "coverage_fraction",
    ]
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in fields}
            for metric in ("rmse", "relative_l2"):
                flat[metric] = row[metric]["value"]
                flat[f"{metric}_reason"] = row[metric]["reason"]
            writer.writerow(flat)


def evaluate_manifest(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    project_root: str | Path | None = None,
    operationally_ready: bool = False,
) -> dict:
    """Produce final metrics only after the coordinating root releases evaluation."""
    if not operationally_ready:
        raise EvaluationError("production evaluation awaits root readiness signal")
    root = Path(project_root).absolute() if project_root else ROOT
    output = _path(root, output_path)
    csv_path = output.with_suffix(".csv")
    if output.exists() or csv_path.exists():
        raise FileExistsError("preserve prior metrics; choose a new output path")
    manifest, protocol = validated_manifest(manifest_path, root)
    torch.set_num_threads(protocol["settings"]["cpu_threads"])
    datasets = saved_datasets(manifest, protocol, root)
    references = {
        data["case_id"]: data
        for data in datasets
        if data["method"] == "analytical_reference"
    }
    rows = []
    for data in datasets:
        case = protocol["resolved"]["cases"][data["case_id"]]
        identity = {
            key: data.get(key)
            for key in ("run_id", "case_id", "method", "variant", "seed", "cells")
        }
        if data["state"] == "completed":
            rows.extend(
                evaluate_fields(
                    case,
                    references[data["case_id"]],
                    data,
                    protocol["resolved"]["evaluation"],
                    identity,
                )
            )
            if data["method"] != "lax_wendroff":
                data["mass"] = _conservation_for_source(protocol, manifest, data, root)
        else:
            rows.extend(
                unavailable_metric_rows(
                    case, protocol["resolved"]["evaluation"], identity, data["reason"]
                )
            )
    scientific_status = copy.deepcopy(manifest.get("scientific_status", {}))
    e1_comparator = any(
        data["case_id"] == "E1"
        and data["method"] == "lax_wendroff"
        and data["state"] == "completed"
        for data in datasets
    )
    scientific_status.update(
        final_accuracy="computed_for_available_runs",
        E1_classical_comparator="available" if e1_comparator else "missing",
        core_comparison="available_for_evaluation" if e1_comparator else "incomplete",
    )
    source_files = _source_files(manifest, root)
    for data in datasets:
        for artifact in data.get("mass", {}).get("auxiliary_records", []):
            directory = Path(artifact["run_path"])
            source_files.extend(
                {"path": str((directory / name).relative_to(root)), "sha256": digest}
                for name, digest in artifact["files_sha256"].items()
            )
    aggregation = {
        case_id: aggregate_metric_rows(
            [row for row in rows if row["case_id"] == case_id], seeds
        )
        for case_id, seeds in [
            ("E1", protocol["settings"]["seeds"]),
            ("E2", [protocol["settings"]["e2_diagnostic_seed"]]),
        ]
    }
    aggregation["E1"]["paired_ablation_status"] = "planned_three_seed_comparison"
    aggregation["E2"]["paired_ablation"] = []
    aggregation["E2"]["paired_ablation_status"] = "not_planned_baseline_only_diagnostic"
    report = {
        "schema_version": 1,
        "purpose": "final_frozen_core_evaluation",
        "state": "completed" if e1_comparator else "incomplete_required_comparison",
        "scientific_status": scientific_status,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "metric_definitions": metric_definitions(),
        "evaluation": protocol["resolved"]["evaluation"],
        "provenance": {
            "protocol_sha256": protocol["protocol_sha256"],
            "protocol_file_sha256": file_sha256(_path(root, manifest["protocol_path"])),
            "manifest_path": str(_path(root, manifest_path)),
            "manifest_sha256": file_sha256(_path(root, manifest_path)),
            "execution": manifest["execution"],
            "source_files": source_files,
            "evaluation_source_sha256": {
                str(path.relative_to(ROOT)): file_sha256(path)
                for path in (
                    Path(__file__),
                    ROOT / "src/hydro_pinn/evaluate.py",
                    ROOT / "src/hydro_pinn/conservation.py",
                )
            },
        },
        "rows": rows,
        "runs": [
            {
                key: value
                for key, value in data.items()
                if key
                not in {"x_m", "t_s", "h_m", "q_m2_s", "native_fields", "history"}
            }
            for data in datasets
        ],
        "aggregation": aggregation,
        "limits": [
            "Idealized generated cases; no field validation or generalization claim",
            "Finite approximation errors and solver failures remain included",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    json.dumps(report, allow_nan=False)
    _write_csv(csv_path, rows)
    write_json_atomic(output, report)
    return report


def plot_manifest(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    metrics_path: str | Path = "results/metrics.json",
    project_root: str | Path | None = None,
    figure_manifest_path: str | Path | None = "docs/figure_manifest.json",
) -> dict:
    """Regenerate figures from hash-checked saved arrays and matching metrics."""
    from hydro_pinn.plot import plot_report

    root = Path(project_root).absolute() if project_root else ROOT
    manifest, protocol = validated_manifest(manifest_path, root)
    metrics_file = _path(root, metrics_path)
    report = _json(metrics_file)
    if report["provenance"]["manifest_sha256"] != file_sha256(
        _path(root, manifest_path)
    ):
        raise EvaluationError("metrics were produced from another manifest revision")
    if report["provenance"]["protocol_sha256"] != protocol["protocol_sha256"]:
        raise EvaluationError("metrics were produced from another frozen protocol")
    for source in report["provenance"]["source_files"]:
        if file_sha256(_path(root, source["path"])) != source["sha256"]:
            raise EvaluationError("figure source changed after metric generation")
    canonical_manifest = (
        _path(root, figure_manifest_path) if figure_manifest_path is not None else None
    )
    if canonical_manifest is not None and canonical_manifest.exists():
        raise FileExistsError("preserve prior figure manifest; choose a new path")
    report["provenance"]["metrics_artifact"] = {
        "path": str(metrics_file.relative_to(root)),
        "sha256": file_sha256(metrics_file),
    }
    datasets = saved_datasets(manifest, protocol, root)
    by_id = {data["run_id"]: data for data in report["runs"]}
    for data in datasets:
        if "mass" in by_id[data["run_id"]]:
            data["mass"] = by_id[data["run_id"]]["mass"]
    figures = plot_report(report, datasets, _path(root, output_dir), project_root=root)
    if canonical_manifest is not None:
        canonical_manifest.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(canonical_manifest, figures)
    return figures
