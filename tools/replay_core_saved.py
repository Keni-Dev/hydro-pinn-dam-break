#!/usr/bin/env python3
"""Reanalyze the published core arrays without training or new flow solutions.

This route verifies the frozen scientific bytes and stored run records, then uses
the unchanged metric, conservation and plotting functions. It deliberately records
the actual runtime instead of claiming the historical execution environment.
The strict experiment/reproduction CLI remains unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

EXPECTED_PROTOCOL = "e73e9a1bd6973cebaeb62435cb3e96da57f3bf6308afb0694a2c36bc916216db"
EXPECTED_METRICS = "173ca0567eda591bc160ba8904cda450a965f441f1ac8a9207b343566b567e0a"
EXPECTED_FIGURE_LEDGER = (
    "db9b482478c797cbdc2ff34a3861bcf0126e6fa66b5ca86336e4d9a2dd0b01a0"
)
EXPECTED_ADDITIVE_CLI = (
    "2a136a615c76bda38fdf73c1797125678f3fdfba424465e319d75bc4211d860b"
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def local(root: Path, name: str | Path) -> Path:
    p = Path(name)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"Expected a checkout-relative input: {name}")
    target = root / p
    if any(x.is_symlink() for x in [target, *target.parents]):
        raise ValueError(f"Input must not use symlinks: {name}")
    if not target.is_file():
        raise FileNotFoundError(
            f"Required input missing: {name}; restore core-data first"
        )
    return target


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def check_hash(root: Path, name: str, expected: str, checked: dict) -> None:
    target = local(root, name)
    actual = digest(target)
    if actual != expected:
        raise ValueError(f"Input SHA256 mismatch: {name}")
    prior = checked.get(name)
    if prior is not None and prior != expected:
        raise ValueError(f"Conflicting published bindings: {name}")
    checked[name] = actual


def anchored_inputs(root: Path) -> tuple[dict, dict, dict, dict, dict]:
    checked = {}
    check_hash(root, "results/metrics.json", EXPECTED_METRICS, checked)
    accepted = read(root / "results/metrics.json")
    check_hash(
        root,
        "docs/final_run_manifest.json",
        accepted["provenance"]["manifest_sha256"],
        checked,
    )
    check_hash(
        root,
        "configs/final_protocol.json",
        accepted["provenance"]["protocol_file_sha256"],
        checked,
    )
    manifest = read(root / "docs/final_run_manifest.json")
    protocol = read(root / "configs/final_protocol.json")
    payload = {k: v for k, v in protocol.items() if k != "protocol_sha256"}
    identity = hashlib.sha256(
        (json.dumps(payload, allow_nan=False, sort_keys=True, indent=2) + "\n").encode()
    ).hexdigest()
    if (
        not identity
        == protocol["protocol_sha256"]
        == manifest["protocol_sha256"]
        == EXPECTED_PROTOCOL
    ):
        raise ValueError("Frozen protocol identity changed")
    inputs = protocol["inputs"]
    bound = [
        inputs["candidate"],
        *inputs["cases"].values(),
        *inputs["training"].values(),
        inputs["evaluation"],
        *inputs["environment"],
        inputs["approval"],
    ]
    for item in bound:
        check_hash(root, item["path"], item["sha256"], checked)
    candidate = read(local(root, inputs["candidate"]["path"]))
    for name, expected in candidate["input_sha256"].items():
        check_hash(root, name, expected, checked)
    for name, expected in protocol["scientific_source_sha256"].items():
        check_hash(root, name, expected, checked)
    for name, expected in accepted["provenance"]["evaluation_source_sha256"].items():
        check_hash(root, name, expected, checked)
    check_hash(root, "docs/figure_manifest.json", EXPECTED_FIGURE_LEDGER, checked)
    ledger = read(root / "docs/figure_manifest.json")
    check_hash(root, "src/hydro_pinn/plot.py", ledger["plot_source_sha256"], checked)
    cli = protocol["cli_provenance"]
    if cli["enforcement"] != "historical_only; additive commands permitted":
        raise ValueError("Unknown historical CLI provenance policy")
    if set(cli["source_sha256"]) != {
        "src/hydro_pinn/cli.py",
        "src/hydro_pinn/__main__.py",
    }:
        raise ValueError("Historical CLI provenance is incomplete")
    check_hash(
        root,
        "src/hydro_pinn/__main__.py",
        cli["source_sha256"]["src/hydro_pinn/__main__.py"],
        checked,
    )
    check_hash(root, "src/hydro_pinn/cli.py", EXPECTED_ADDITIVE_CLI, checked)
    for item in accepted["provenance"]["source_files"]:
        check_hash(root, item["path"], item["sha256"], checked)
    for entry in manifest["entries"]:
        run = Path(manifest["execution"]["output_root"]) / entry["actual_run_id"]
        for name, expected in entry["result"]["files_sha256"].items():
            check_hash(root, str(run / name), expected, checked)
    return protocol, manifest, accepted, ledger, checked


def validate_records(root, raw_protocol, manifest):
    from hydro_pinn.config import load_config
    from hydro_pinn.experiments import _stage_info, execution_plan
    from hydro_pinn.protocol import SCIENTIFIC_MODULES, load_protocol

    protocol = load_protocol(
        root / "configs/final_protocol.json", project_root=root, check_current=False
    )
    if protocol != raw_protocol:
        raise ValueError("Parsed protocol differs from verified input")
    expected_sources = {f"src/hydro_pinn/{name}.py" for name in SCIENTIFIC_MODULES}
    if set(protocol["scientific_source_sha256"]) != expected_sources:
        raise ValueError("Frozen scientific source enforcement set changed")
    for group in ("cases", "training"):
        for key, item in protocol["inputs"][group].items():
            if (
                load_config(local(root, item["path"]))
                != protocol["resolved"][group][key]
            ):
                raise ValueError(f"Resolved configuration changed: {group}.{key}")
    if (
        load_config(local(root, protocol["inputs"]["evaluation"]["path"]))
        != protocol["resolved"]["evaluation"]
    ):
        raise ValueError("Resolved final evaluation changed")
    if manifest["workflow_status"] not in {"completed", "completed_with_diagnostics"}:
        raise ValueError("Saved matrix is not terminal")
    execution = manifest["execution"]
    expected_execution = (
        execution_plan(protocol)
        if execution["namespace"] is None
        else execution_plan(
            protocol,
            namespace=execution["namespace"],
            output_root=execution["output_root"],
            manifest_path=execution["manifest"],
        )
    )
    if execution != expected_execution:
        raise ValueError("Saved execution IDs/destinations differ from the frozen plan")
    entries = {e["stage_id"]: e for e in manifest["entries"]}
    if len(entries) != len(manifest["entries"]) or len(entries) != len(
        protocol["matrix"]
    ):
        raise ValueError("Saved matrix coverage changed")
    for stage in protocol["matrix"]:
        entry = entries.get(stage["stage_id"])
        if entry is None or entry["frozen_stage"] != stage:
            raise ValueError("Saved stage differs from frozen matrix")
        if entry["state"] != "completed" and not (
            stage["stage"] == "solve"
            and entry["state"] == "failed"
            and entry["failure_category"] == "numerical_failure"
        ):
            raise ValueError("Unresolved saved stage")
        info = _stage_info(
            protocol,
            stage,
            entry,
            entries,
            root,
            require_completed=entry["state"] == "completed",
            execution=execution,
        )
        if (
            info["record_state"] != entry["state"]
            or info["files_sha256"] != entry["result"]["files_sha256"]
        ):
            raise ValueError("Saved run identity/state/files changed")
    return protocol


def saved_mass(protocol, manifest, dataset, root):
    from hydro_pinn.conservation import (
        _read_record,
        quadrature_mass,
        quadrature_sensitivity,
    )

    arrays, artifacts = {}, []
    source_dir = Path(manifest["execution"]["output_root"]) / dataset["run_id"]
    source = read(local(root, source_dir / "resolved_config.json"))
    source_hash = digest(local(root, source_dir / "resolved_config.json"))
    plans = [
        p
        for p in protocol["conservation"]["auxiliary_records"]
        if p["source_stage_id"] == dataset["stage_id"]
    ]
    for plan in plans:
        axes = protocol["conservation"][
            "volume_axes" if plan["kind"] == "volume" else "boundary_axes"
        ][plan["axis_key"]]
        checkpoint_hash = None
        if plan["source_kind"] == "predict":
            prediction = source["prediction_source"]
            checkpoint_hash = digest(
                local(
                    root,
                    Path(manifest["execution"]["output_root"])
                    / prediction["training_run_id"]
                    / "checkpoint.pt",
                )
            )
            if checkpoint_hash != prediction["checkpoint_sha256"]:
                raise ValueError("Auxiliary checkpoint binding changed")
        expected = {
            "protocol_sha256": protocol["protocol_sha256"],
            "evaluation_sha256": plan["evaluation_sha256"],
            "source_stage_id": plan["source_stage_id"],
            "source_run_id": dataset["run_id"],
            "source_resolved_config_sha256": source_hash,
            "kind": plan["kind"],
            "axis_key": plan["axis_key"],
            "checkpoint_sha256": checkpoint_hash,
        }
        run_id = manifest["execution"]["run_ids"][plan["run_id"]]
        directory = root / manifest["execution"]["output_root"] / run_id
        for name in [
            "resolved_config.json",
            "metadata.json",
            "status.json",
            "outputs.npz",
            "outputs_manifest.json",
        ]:
            local(root, directory.relative_to(root) / name)
        values, artifact = _read_record(directory, expected, axes)
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


def write(path: Path, data: dict) -> None:
    with path.open("x") as stream:
        json.dump(data, stream, allow_nan=False, indent=2, sort_keys=True)
        stream.write("\n")


def run(root: Path, output: Path, figures=True) -> dict:
    root = root.resolve(strict=True)
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Choose a fresh saved-analysis output directory")
    if output.is_relative_to(root) and output.relative_to(root).parts[0] in {
        "src",
        "tests",
        "configs",
        "runs",
        "docs",
    }:
        raise ValueError(
            "Analysis output must not be inside protected source/input folders"
        )
    raw_protocol, manifest, accepted, ledger, checked = anchored_inputs(root)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    from hydro_pinn.evaluate import (
        aggregate_metric_rows,
        evaluate_fields,
        metric_definitions,
        unavailable_metric_rows,
    )
    from hydro_pinn.evaluation_workflow import _write_csv, saved_datasets
    from hydro_pinn.protocol import environment_snapshot

    protocol = validate_records(root, raw_protocol, manifest)
    actual_environment = environment_snapshot()
    datasets = saved_datasets(manifest, protocol, root)
    references = {
        d["case_id"]: d for d in datasets if d["method"] == "analytical_reference"
    }
    rows = []
    for data in datasets:
        case = protocol["resolved"]["cases"][data["case_id"]]
        identity = {
            k: data.get(k)
            for k in ("run_id", "case_id", "method", "variant", "seed", "cells")
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
                data["mass"] = saved_mass(protocol, manifest, data, root)
        else:
            rows.extend(
                unavailable_metric_rows(
                    case, protocol["resolved"]["evaluation"], identity, data["reason"]
                )
            )
    aggregation = {
        case: aggregate_metric_rows([r for r in rows if r["case_id"] == case], seeds)
        for case, seeds in [
            ("E1", protocol["settings"]["seeds"]),
            ("E2", [protocol["settings"]["e2_diagnostic_seed"]]),
        ]
    }
    aggregation["E1"]["paired_ablation_status"] = "planned_three_seed_comparison"
    aggregation["E2"].update(
        paired_ablation=[],
        paired_ablation_status="not_planned_baseline_only_diagnostic",
    )
    accepted_runs = {d["run_id"]: d for d in accepted["runs"]}
    mass_checks = {}
    for data in datasets:
        if "mass" in data:
            actual = {k: v for k, v in data["mass"].items() if k != "auxiliary_records"}
            expected = {
                k: v
                for k, v in accepted_runs[data["run_id"]]["mass"].items()
                if k != "auxiliary_records"
            }
            mass_checks[data["run_id"]] = actual == expected
    comparisons = {
        "metric_row_count": len(rows) == 7650,
        "all_metric_rows_exact": rows == accepted["rows"],
        "all_aggregates_exact": aggregation == accepted["aggregation"],
        "all_mass_diagnostics_exact": all(mass_checks.values()),
        "metric_definitions_exact": metric_definitions()
        == accepted["metric_definitions"],
    }
    if not all(comparisons.values()):
        raise ValueError(f"Saved numerical comparison failed: {comparisons}")
    report = {
        "schema_version": 1,
        "purpose": "saved_core_array_reanalysis",
        "state": "completed",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_id": protocol["protocol_id"],
        "evaluation": protocol["resolved"]["evaluation"],
        "metric_definitions": metric_definitions(),
        "scientific_status": copy.deepcopy(accepted["scientific_status"]),
        "rows": rows,
        "aggregation": aggregation,
        "runs": [
            {
                k: v
                for k, v in d.items()
                if k not in {"x_m", "t_s", "h_m", "q_m2_s", "native_fields", "history"}
            }
            for d in datasets
        ],
        "provenance": {
            "protocol_sha256": EXPECTED_PROTOCOL,
            "protocol_file_sha256": checked["configs/final_protocol.json"],
            "manifest_sha256": checked["docs/final_run_manifest.json"],
            "manifest_path": str(root / "docs/final_run_manifest.json"),
            "execution": manifest["execution"],
            "protocol_resolution": {
                "recorded_path": manifest["protocol_path"],
                "resolved_local_path": "configs/final_protocol.json",
                "original_manifest_unchanged": True,
            },
            "source_files": [
                {"path": k, "sha256": v} for k, v in sorted(checked.items())
            ],
            "actual_runtime": actual_environment,
            "historical_runtime": protocol["environment"],
            "runtime_matches_historical_execution": actual_environment
            == protocol["environment"],
            "historical_runtime_equality_enforced": False,
            "other_input_source_data_bindings_enforced": True,
            "historical_cli_policy": protocol["cli_provenance"],
            "current_additive_cli_sha256": EXPECTED_ADDITIVE_CLI,
            "no_training_solver_inference_or_benchmark_execution": True,
            "reanalysis_helper_sha256": digest(Path(__file__)),
        },
        "comparison_with_published_results": comparisons,
        "mass_diagnostic_comparisons": mass_checks,
        "limits": [
            "Reanalysis of stored arrays and diagnostic samples; "
            "not a new experiment or historical-runtime reproduction.",
            "All original reported results and limitations remain unchanged.",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    write(output / "metrics.json", report)
    _write_csv(output / "metrics.csv", rows)
    figure_count = 0
    if figures:
        from hydro_pinn.plot import plot_report

        figures_report = plot_report(
            report, datasets, output / "figures", project_root=root
        )
        figure_count = len(figures_report["figures"])
        if figure_count != 34:
            raise ValueError(f"Expected 34 core figures, got {figure_count}")
    for name, expected in checked.items():
        if digest(local(root, name)) != expected:
            raise ValueError(f"Input changed during reanalysis: {name}")
    summary = {
        "pass": True,
        "scope": "Saved-array core reanalysis with explicit actual runtime; "
        "strict original CLI unchanged.",
        "metric_rows": len(rows),
        "datasets": len(datasets),
        "mass_diagnostics": len(mass_checks),
        "core_figures": figure_count,
        "checked_input_files": len(checked),
        "comparisons": comparisons,
        "runtime_matches_historical_execution": actual_environment
        == protocol["environment"],
        "actual_runtime": actual_environment,
        "historical_runtime": protocol["environment"],
        "source_inputs_unchanged": True,
        "new_experiments_executed": False,
        "supplemental_timing_figure": "Regenerate separately from existing saved "
        "timing records using rebuild_timing_accuracy.py; do not run benchmarks.",
    }
    write(output / "verification.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(run(args.root, args.output, figures=not args.no_figures), indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
