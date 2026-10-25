"""Fresh, namespaced execution and explicitly separate saved-output replay.

Smoke exercises the pipeline with declared reductions. Paper requires completed
parent results and executes the unchanged frozen matrix; neither path adopts or
overwrites original records. Reports survive failed preflight and execution.
"""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch

from hydro_pinn.config import load_config
from hydro_pinn.evaluate import (
    common_grid_fields,
    evaluate_fields,
    unavailable_metric_rows,
)
from hydro_pinn.experiments import execution_plan, run_matrix
from hydro_pinn.lax_wendroff import NumericalRunFailed, generate_solver_run
from hydro_pinn.protocol import environment_snapshot, load_protocol
from hydro_pinn.records import write_json_atomic
from hydro_pinn.references import generate_reference_run
from hydro_pinn.train import load_training_checkpoint, predict_run, train_run

ROOT = Path(__file__).resolve().parents[2]
TIMING_FIGURE_HELPER = "docs/evidence/phase_06/rebuild_timing_accuracy.py"
ACCEPTED_RENDERER_SHA256 = {
    "src/hydro_pinn/plot.py": (
        "74df4c54c87cd2d403d91a6c5fec7f5978df6de072f826c6b9ec8e3487ea616c"
    ),
    TIMING_FIGURE_HELPER: (
        "4103a9fb9576eec3a89985eac48a4b7f56967132940d04beb5400c4ae2a9e47e"
    ),
}
TOLERANCES = {
    "coordinates": {"rtol": 0.0, "atol": 1e-13},
    "reference_and_solver": {"rtol": 1e-10, "atol": 1e-12},
    "network_float32": {"rtol": 1e-4, "atol": 1e-5},
    "network_float64": {"rtol": 1e-9, "atol": 1e-11},
}


class ReproductionError(ValueError):
    """A failed reproduction whose report and any new runs remain available."""

    def __init__(self, message: str, report_path: Path):
        super().__init__(f"{message}; report retained at {report_path}")
        self.report_path = report_path


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else root / path
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("reproduction paths must not use parent traversal or symlinks")
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError("reproduction paths must remain inside project root") from exc
    return path


def _read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"missing or unsafe reproduction input: {path}")
    result = json.loads(path.read_text())
    if not isinstance(result, dict):
        raise ValueError(f"reproduction input must be a JSON object: {path}")
    return result


def _save(path: Path, report: dict) -> None:
    report["updated_at_utc"] = _now()
    write_json_atomic(path, report, overwrite=path.exists())


def _files(directory: Path, root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): _sha(path)
        for path in sorted(directory.iterdir())
        if path.is_file() and not path.is_symlink()
    }


def _parent_snapshot(protocol: dict, root: Path, output: Path) -> dict:
    path = _path(root, protocol["paths"]["manifest"])
    raw = path.read_bytes()
    manifest = json.loads(raw)
    if manifest["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError("parent manifest belongs to another frozen protocol")
    if manifest["execution"] != execution_plan(protocol):
        raise ValueError("parent manifest execution differs from frozen destinations")
    stages = {row["stage_id"]: row for row in manifest["entries"]}
    if len(stages) != len(manifest["entries"]) or set(stages) != {
        row["stage_id"] for row in protocol["matrix"]
    }:
        raise ValueError("parent manifest coverage differs from frozen matrix")
    for stage in protocol["matrix"]:
        if stages[stage["stage_id"]]["frozen_stage"] != stage:
            raise ValueError("parent manifest stage differs from frozen matrix")
    # A snapshot keeps a changing main run's observed state reproducible without
    # claiming that mutable in-progress checkpoint files were verified.
    with (output / "parent_manifest_snapshot.json").open("xb") as handle:
        handle.write(raw)
    return {
        "path": str(path.relative_to(root)),
        "sha256_at_observation": hashlib.sha256(raw).hexdigest(),
        "workflow_status_at_observation": manifest["workflow_status"],
        "snapshot": str((output / "parent_manifest_snapshot.json").relative_to(root)),
        "completed_artifacts_verified": False,
    }


def _compare_values(first: np.ndarray, second: np.ndarray, tolerance: dict) -> dict:
    if first.shape != second.shape:
        return {"within_tolerance": False, "reason": "array shape differs"}
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        return {"within_tolerance": False, "reason": "nonfinite comparison array"}
    delta = second.astype(np.float64) - first.astype(np.float64)
    norm = float(np.linalg.norm(first.astype(np.float64)))
    return {
        "shape": list(first.shape),
        "max_absolute_difference": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "rms_difference": float(np.sqrt(np.mean(delta**2))) if delta.size else 0.0,
        "relative_l2_difference": float(np.linalg.norm(delta) / norm) if norm else None,
        "relative_l2_reason": None if norm else "zero parent norm",
        "within_tolerance": bool(np.allclose(first, second, **tolerance)),
        "tolerance": tolerance,
    }


def _compare_arrays(first: Path, second: Path, kind: str) -> dict:
    with (
        np.load(first, allow_pickle=False) as parent,
        np.load(second, allow_pickle=False) as fresh,
    ):
        if set(parent.files) != set(fresh.files):
            return {"within_tolerance": False, "reason": "saved field names differ"}
        fields = {
            key: _compare_values(
                parent[key],
                fresh[key],
                TOLERANCES["coordinates"]
                if key in {"x_m", "t_s", "cell_edges_m"}
                else TOLERANCES[kind],
            )
            for key in parent.files
        }
    return {
        "fields": fields,
        "within_tolerance": all(value["within_tolerance"] for value in fields.values()),
    }


def _compare_tree(parent, fresh, tolerance: dict) -> dict:
    """Compare physical metric values, retaining differences at named fields."""
    if isinstance(parent, dict) and isinstance(fresh, dict):
        if set(parent) != set(fresh):
            return {
                "within_tolerance": False,
                "reason": "metric fields differ",
                "parent_fields": sorted(parent),
                "fresh_fields": sorted(fresh),
            }
        fields = {
            key: _compare_tree(parent[key], fresh[key], tolerance) for key in parent
        }
        return {
            "within_tolerance": all(row["within_tolerance"] for row in fields.values()),
            "fields": fields,
        }
    if isinstance(parent, (list, int, float)) and isinstance(fresh, (list, int, float)):
        first, second = np.asarray(parent), np.asarray(fresh)
        if first.dtype.kind in "biuf" and second.dtype.kind in "biuf":
            # Axis lengths, mask counts, and boolean facts must agree exactly.
            chosen = (
                {"rtol": 0.0, "atol": 0.0} if first.dtype.kind in "biu" else tolerance
            )
            result = _compare_values(first, second, chosen)
            if first.ndim == second.ndim == 0:
                result.update(parent_value=parent, fresh_value=fresh)
            return result
    return {
        "within_tolerance": parent == fresh,
        "parent_value": parent,
        "fresh_value": fresh,
    }


def _compare_metrics(parent: dict, fresh: dict, manifest: dict, protocol: dict) -> dict:
    """Compare errors and every numeric native/dense mass diagnostic."""
    source = {
        entry["actual_run_id"]: entry["frozen_stage"] for entry in manifest["entries"]
    }
    fresh_stage = {entry["run_id"]: entry for entry in protocol["matrix"]}
    # Canonical stages have IDs equal to their frozen run IDs. Namespace binding
    # is supplied by the fresh metric report rather than inferred from strings.
    fresh_ids = fresh["provenance"]["execution"]["run_ids"]
    stages = {fresh_ids[key]: value for key, value in fresh_stage.items()}

    def rows_by_identity(report, mapping):
        result = {}
        for row in report["rows"]:
            stage = mapping[row["run_id"]]
            key = (stage["stage_id"], row["time_s"], row["field"], row["region"])
            if key in result:
                raise ValueError(
                    "duplicate metric identity during reproduction comparison"
                )
            result[key] = {
                name: value for name, value in row.items() if name != "run_id"
            }
            for metric in ("rmse", "relative_l2"):
                if metric in result[key] and result[key][metric].get("reason"):
                    result[key][metric] = {
                        **result[key][metric],
                        "reason": result[key][metric]["reason"].replace(
                            row["run_id"], stage["run_id"]
                        ),
                    }
        return result

    first_rows, second_rows = (
        rows_by_identity(parent, source),
        rows_by_identity(fresh, stages),
    )
    if set(first_rows) != set(second_rows):
        return {"within_tolerance": False, "reason": "metric row coverage differs"}
    matrix = {row["stage_id"]: row for row in protocol["matrix"]}
    comparisons = []
    for key, row in first_rows.items():
        stage = matrix[key[0]]
        kind = (
            f"network_{protocol['settings']['dtype']}"
            if stage["stage"] == "predict"
            else "reference_and_solver"
        )
        comparisons.append(
            {
                "stage_id": key[0],
                "time_s": key[1],
                "field": key[2],
                "region": key[3],
                "comparison": _compare_tree(row, second_rows[key], TOLERANCES[kind]),
            }
        )
    first_mass = {
        source[row["run_id"]]["stage_id"]: row.get("mass") for row in parent["runs"]
    }
    second_mass = {
        stages[row["run_id"]]["stage_id"]: row.get("mass") for row in fresh["runs"]
    }
    if set(first_mass) != set(second_mass):
        return {
            "within_tolerance": False,
            "reason": "conservation run coverage differs",
        }
    mass = []
    for stage_id, value in first_mass.items():
        stage = matrix[stage_id]
        kind = (
            f"network_{protocol['settings']['dtype']}"
            if stage["stage"] == "predict"
            else "reference_and_solver"
        )
        # Ancestry paths/hashes necessarily change. All physical values,
        # quadrature levels, sensitivities, and failure facts remain compared.
        first = (
            {key: data for key, data in value.items() if key != "auxiliary_records"}
            if isinstance(value, dict)
            else value
        )
        second = (
            {
                key: data
                for key, data in second_mass[stage_id].items()
                if key != "auxiliary_records"
            }
            if isinstance(second_mass[stage_id], dict)
            else second_mass[stage_id]
        )
        mass.append(
            {
                "stage_id": stage_id,
                "comparison": _compare_tree(first, second, TOLERANCES[kind]),
            }
        )
    return {
        "within_tolerance": all(
            row["comparison"]["within_tolerance"] for row in [*comparisons, *mass]
        ),
        "error_rows": comparisons,
        "conservation": mass,
        "failure_reason_run_ids_normalized": True,
    }


def _compare_auxiliary(parent: dict, fresh: dict, protocol: dict, root: Path) -> dict:
    stages = {row["stage_id"]: row for row in protocol["matrix"]}
    rows = []
    for plan in protocol["conservation"]["auxiliary_records"]:
        stage = stages[plan["source_stage_id"]]
        kind = (
            f"network_{protocol['settings']['dtype']}"
            if stage["stage"] == "predict"
            else "reference_and_solver"
        )
        paths = []
        for manifest in (parent, fresh):
            execution = manifest["execution"]
            path = _path(
                root,
                Path(execution["output_root"])
                / execution["run_ids"][plan["run_id"]]
                / "outputs.npz",
            )
            paths.append(path)
        rows.append(
            {
                "frozen_run_id": plan["run_id"],
                "source_stage_id": plan["source_stage_id"],
                "parent_path": str(paths[0].relative_to(root)),
                "fresh_path": str(paths[1].relative_to(root)),
                "parent_sha256": _sha(paths[0]),
                "fresh_sha256": _sha(paths[1]),
                "comparison": _compare_arrays(paths[0], paths[1], kind),
            }
        )
    return {
        "within_tolerance": all(row["comparison"]["within_tolerance"] for row in rows),
        "expected_records": len(protocol["conservation"]["auxiliary_records"]),
        "compared_records": len(rows),
        "records": rows,
    }


def _record_stage(
    report: dict, report_path: Path, root: Path, stage: dict, operation
) -> dict:
    row = {**stage, "state": "running", "started_at_utc": _now()}
    report["fresh_execution"]["stages"].append(row)
    _save(report_path, report)
    path = root / report["fresh_execution"]["output_root"] / stage["run_id"]
    if path.exists():
        row.update(state="failed", reason="fresh run ID already exists")
        _save(report_path, report)
        raise FileExistsError(f"preserve existing run; choose a new run ID: {path}")
    try:
        operation()
        status = _read(path / "status.json")
        if status["state"] != "completed":
            raise ValueError("fresh stage did not complete its run record")
        row.update(state="completed", stopping_reason=status["stopping_reason"])
    except (KeyboardInterrupt, SystemExit) as exc:
        row.update(state="interrupted", reason=f"{type(exc).__name__}: {exc}")
        raise
    except (ValueError, OSError, RuntimeError) as exc:
        numerical = isinstance(exc, NumericalRunFailed)
        row.update(
            state="failed",
            failure_category="numerical_failure" if numerical else "execution_failure",
            reason=f"{type(exc).__name__}: {exc}",
        )
        if path.exists():
            row["record_state_observed"] = _read(path / "status.json")["state"]
    finally:
        if path.exists():
            binding = {
                "purpose": "bounded_smoke_not_paper_reproduction",
                "parent_protocol_sha256": report["parent"]["protocol_sha256"],
                "parent_manifest": report["parent"]["manifest"],
                "reproduction_report": str(report_path.relative_to(root)),
                "stage": stage,
            }
            write_json_atomic(path / "reproduction_binding.json", binding)
            row["run_path"] = str(path.relative_to(root))
            row["files_sha256"] = _files(path, root)
        row["finished_at_utc"] = _now()
        _save(report_path, report)
    return row


def _smoke(protocol: dict, root: Path, output: Path, report: dict) -> None:
    report_path = output / "report.json"
    smoke_path = _path(root, "configs/smoke.json")
    smoke = load_config(smoke_path)
    if smoke["config_type"] != "smoke":
        raise ValueError("smoke reproduction requires typed configs/smoke.json")
    write_json_atomic(output / "smoke_config_snapshot.json", smoke)
    evaluation_path = output / "smoke_evaluation.json"
    write_json_atomic(evaluation_path, smoke["evaluation"])
    settings = protocol["settings"]
    namespace = report["execution_namespace"]
    runs_root = _path(root, protocol["paths"]["output_root"])
    report["fresh_execution"].update(
        purpose="bounded_pipeline_check_not_paper_reproduction",
        smoke_input={"path": "configs/smoke.json", "sha256": _sha(smoke_path)},
        reductions={
            "training_overrides": smoke["training_overrides"],
            "validation": smoke["validation"],
            "evaluation": smoke["evaluation"],
            "solver_cells": smoke["evaluation"]["grid"]["cells"],
            "seed_coverage": {
                "E1": [settings["seeds"][0]],
                "E2": [settings["e2_diagnostic_seed"]],
            },
            "variants": {"E1": ["baseline", "adaptive"], "E2": ["baseline"]},
            "network_architecture": "unchanged from parent",
            "conservation_and_benchmark": "not_executed_in_smoke",
        },
    )
    report["saved_output_replay"].update(
        state="not_executed",
        reason="smoke executes reduced fresh stages and makes no paper comparison",
    )
    reference_paths = {}
    datasets = []
    for case_id in ("E1", "E2"):
        case_path = _path(root, protocol["inputs"]["cases"][case_id]["path"])
        seed = (
            settings["seeds"][0] if case_id == "E1" else settings["e2_diagnostic_seed"]
        )
        variants = ["baseline", "adaptive"] if case_id == "E1" else ["baseline"]

        def stage(kind: str, suffix: str = "", variant=None, case=case_id, value=seed):
            return {
                "stage": kind,
                "case_id": case,
                "variant": variant,
                "seed": value if variant else None,
                "run_id": f"{namespace}-{case.lower()}-{kind}{suffix}",
            }

        ref = stage("reference")
        load_protocol(report["parent"]["protocol_path"], project_root=root)
        ref_row = _record_stage(
            report,
            report_path,
            root,
            ref,
            partial(
                generate_reference_run,
                case_path,
                evaluation_path,
                ref["run_id"],
                runs_root,
            ),
        )
        if ref_row["state"] != "completed":
            raise ValueError(
                f"smoke {case_id} reference failed; dependent stages not executed"
            )
        reference_paths[case_id] = runs_root / ref["run_id"] / "outputs.npz"
        solver = stage("solve")
        solver_row = _record_stage(
            report,
            report_path,
            root,
            solver,
            partial(
                generate_solver_run,
                case_path,
                evaluation_path,
                smoke["evaluation"]["grid"]["cells"],
                solver["run_id"],
                runs_root,
            ),
        )
        datasets.append((solver_row, protocol["resolved"]["cases"][case_id]))
        for variant in variants:
            training = stage("train", f"-{variant}-s{seed}", variant)
            load_protocol(report["parent"]["protocol_path"], project_root=root)
            training_row = _record_stage(
                report,
                report_path,
                root,
                training,
                partial(
                    train_run,
                    case_path,
                    _path(root, protocol["inputs"]["training"][variant]["path"]),
                    seed,
                    settings["device"],
                    settings["dtype"],
                    training["run_id"],
                    smoke_path=smoke_path,
                    output_root=runs_root,
                    chunk_size=settings["chunk_size"],
                    synchronize_steps=True,
                ),
            )
            if training_row["state"] != "completed":
                continue
            checkpoint, _, train_path = load_training_checkpoint(
                training["run_id"], runs_root, require_completed=True
            )
            training_row.update(
                completed_adam_steps=checkpoint["step"],
                selected_step=checkpoint["selected_step"],
                checkpoint_sha256=_sha(train_path / "checkpoint.pt"),
            )
            prediction = stage("predict", f"-{variant}-s{seed}", variant)
            prediction_row = _record_stage(
                report,
                report_path,
                root,
                prediction,
                partial(
                    predict_run,
                    training["run_id"],
                    smoke_path,
                    prediction["run_id"],
                    runs_root,
                ),
            )
            if prediction_row["state"] == "completed":
                prediction_row["held_model_vs_checkpoint_reload"] = _compare_arrays(
                    train_path / "reload_probe.npz",
                    runs_root / prediction["run_id"] / "outputs.npz",
                    f"network_{settings['dtype']}",
                )
                if not prediction_row["held_model_vs_checkpoint_reload"][
                    "within_tolerance"
                ]:
                    raise ValueError(
                        "smoke checkpoint reload differs from held selected model"
                    )
            datasets.append((prediction_row, protocol["resolved"]["cases"][case_id]))
    if _sha(smoke_path) != report["fresh_execution"]["smoke_input"]["sha256"]:
        raise ValueError("smoke input changed during execution")
    metrics = []
    for row, case in datasets:
        identity = {
            **{key: row[key] for key in ("run_id", "case_id", "variant", "seed")},
            "method": "lax_wendroff" if row["stage"] == "solve" else "pinn",
            "cells": smoke["evaluation"]["grid"]["cells"]
            if row["stage"] == "solve"
            else None,
        }
        if row["state"] != "completed":
            metrics.extend(
                unavailable_metric_rows(
                    case,
                    smoke["evaluation"],
                    identity,
                    row["reason"],
                )
            )
            continue
        with np.load(reference_paths[row["case_id"]], allow_pickle=False) as data:
            reference = {name: data[name].copy() for name in data.files}
        with np.load(
            runs_root / row["run_id"] / "outputs.npz", allow_pickle=False
        ) as data:
            fields = {name: data[name].copy() for name in data.files}
        metrics.extend(
            evaluate_fields(
                case,
                reference,
                common_grid_fields(fields, smoke["evaluation"]),
                smoke["evaluation"],
                identity,
            )
        )
    write_json_atomic(
        output / "smoke_metrics.json",
        {
            "purpose": "fresh_reduced_smoke_metrics_not_paper_results",
            "rows": metrics,
            "evaluation": smoke["evaluation"],
            "conservation": "not_executed",
        },
    )
    report["fresh_execution"]["evaluation"] = {
        "state": "completed",
        "path": str((output / "smoke_metrics.json").relative_to(root)),
        "sha256": _sha(output / "smoke_metrics.json"),
        "rows": len(metrics),
    }
    failed = [
        row for row in report["fresh_execution"]["stages"] if row["state"] == "failed"
    ]
    if any(row["failure_category"] != "numerical_failure" for row in failed):
        raise ValueError("one or more fresh smoke stages failed to execute")
    report["state"] = "completed_with_diagnostics" if failed else "completed"


def _result_dependencies(protocol: dict, root: Path) -> tuple[dict, dict, dict]:
    from hydro_pinn.evaluation_workflow import validated_manifest

    manifest, _ = validated_manifest(protocol["paths"]["manifest"], root)
    metrics = _read(root / "results/metrics.json")
    timing = _read(root / "results/timings.json")
    manifest_sha = _sha(_path(root, protocol["paths"]["manifest"]))
    for name, data in (("metrics", metrics), ("timings", timing)):
        allowed_states = {"completed"}
        if name == "timings":
            allowed_states.add("completed_with_invalid_workloads")
        if data["state"] not in allowed_states:
            raise ValueError(f"paper reproduction requires completed parent {name}")
        provenance = data["provenance"]
        if (
            provenance["protocol_sha256"] != protocol["protocol_sha256"]
            or provenance["manifest_sha256"] != manifest_sha
        ):
            raise ValueError(
                f"parent {name} belongs to another protocol/manifest revision"
            )
        source_key = (
            "evaluation_source_sha256" if name == "metrics" else "source_sha256"
        )
        sources = provenance.get(source_key)
        expected = (
            {
                "src/hydro_pinn/evaluation_workflow.py",
                "src/hydro_pinn/evaluate.py",
                "src/hydro_pinn/conservation.py",
            }
            if name == "metrics"
            else {"src/hydro_pinn/benchmark.py"}
        )
        if not isinstance(sources, dict) or set(sources) != expected:
            raise ValueError(f"parent {name} additive source hash set is incomplete")
        for source, digest in sources.items():
            if _sha(_path(root, source)) != digest:
                raise ValueError(
                    f"parent {name} additive implementation changed: {source}"
                )
    for source in metrics["provenance"]["source_files"]:
        if _sha(_path(root, source["path"])) != source["sha256"]:
            raise ValueError("parent metric source artifact changed")
    declared = {source["path"] for source in metrics["provenance"]["source_files"]}
    # Saved-output replay must reuse already recorded quadrature arrays. Missing
    # auxiliary outputs must not trigger fresh checkpoint forward passes while
    # the report calls the operation saved-output replay.
    for plan in protocol["conservation"]["auxiliary_records"]:
        directory = _path(root, Path(protocol["paths"]["output_root"]) / plan["run_id"])
        for name in (
            "outputs.npz",
            "resolved_config.json",
            "metadata.json",
            "status.json",
        ):
            path = directory / name
            if not path.is_file() or str(path.relative_to(root)) not in declared:
                raise ValueError(
                    "saved-output replay requires recorded parent auxiliary arrays"
                )
    return manifest, metrics, timing


def _paper_comparison(parent: dict, fresh: dict, protocol: dict, root: Path) -> dict:
    original = {entry["stage_id"]: entry for entry in parent["entries"]}
    comparisons = []
    for entry in fresh["entries"]:
        first = original[entry["stage_id"]]
        stage = entry["frozen_stage"]
        row = {
            "stage_id": entry["stage_id"],
            "stage": stage["stage"],
            "parent_run_id": first["actual_run_id"],
            "fresh_run_id": entry["actual_run_id"],
            "parent_state": first["state"],
            "fresh_state": entry["state"],
            "terminal_state_matches": first["state"] == entry["state"],
        }
        if first["state"] == entry["state"] == "completed":
            first_path = _path(root, first["result"]["run_path"])
            second_path = _path(root, entry["result"]["run_path"])
            if stage["stage"] == "train":
                row.update(
                    parent_completed_adam_steps=first["result"]["completed_adam_steps"],
                    fresh_completed_adam_steps=entry["result"]["completed_adam_steps"],
                    parent_selected_step=first["result"]["selected_step"],
                    fresh_selected_step=entry["result"]["selected_step"],
                    completed_step_count_matches=first["result"]["completed_adam_steps"]
                    == entry["result"]["completed_adam_steps"],
                    checkpoint_selection_step_matches=first["result"]["selected_step"]
                    == entry["result"]["selected_step"],
                    partitions_match=first["result"]["partition_content_sha256"]
                    == entry["result"]["partition_content_sha256"],
                )
            else:
                row["array_comparison"] = _compare_arrays(
                    first_path / "outputs.npz",
                    second_path / "outputs.npz",
                    f"network_{protocol['settings']['dtype']}"
                    if stage["stage"] == "predict"
                    else "reference_and_solver",
                )
        elif first["state"] == entry["state"] == "failed":
            row.update(
                parent_failure_category=first["failure_category"],
                fresh_failure_category=entry["failure_category"],
                parent_reason=first.get("reason"),
                fresh_reason=entry.get("reason"),
                parent_diagnostics=first.get("result", {}).get("diagnostics"),
                fresh_diagnostics=entry.get("result", {}).get("diagnostics"),
            )
            row["failure_signature_matches"] = first["failure_category"] == entry[
                "failure_category"
            ] and all(
                (row["parent_diagnostics"] or {}).get(key)
                == (row["fresh_diagnostics"] or {}).get(key)
                for key in ("stage", "time_s", "step", "reason")
            )
        comparisons.append(row)
    matches = all(
        row["terminal_state_matches"]
        and row.get("array_comparison", {}).get("within_tolerance", True)
        and row.get("partitions_match", True)
        and row.get("completed_step_count_matches", True)
        and row.get("checkpoint_selection_step_matches", True)
        and row.get("failure_signature_matches", True)
        for row in comparisons
    )
    return {
        "within_declared_tolerances": matches,
        "checkpoint_agreement_rule": (
            "Exact completed Adam step count, selected step, and partition agreement "
            "are required in addition to declared field-array tolerances."
        ),
        "stages": comparisons,
    }


def _rendering_bindings(root: Path) -> dict:
    """Bind the accepted parent ledgers, all exports, and current renderer bytes."""
    canonical_path = root / "docs/figure_manifest.json"
    canonical = _read(canonical_path)
    entries = canonical["figures"]
    if len(entries) != 35 or len({row["figure_id"] for row in entries}) != 35:
        raise ValueError(
            "paper reproduction requires the complete 35-figure parent ledger"
        )
    exports = [artifact for row in entries for artifact in row["artifacts"]]
    if len(exports) != 70:
        raise ValueError("paper reproduction requires all 70 parent exports")
    sources = {
        name: _sha(_path(root, name))
        for name in ("src/hydro_pinn/plot.py", TIMING_FIGURE_HELPER)
    }
    if sources != ACCEPTED_RENDERER_SHA256:
        raise ValueError("figure renderer differs from the root-accepted source bytes")
    if sources["src/hydro_pinn/plot.py"] != canonical["plot_source_sha256"]:
        raise ValueError(
            "core plot implementation differs from accepted parent figures"
        )
    preserved = {"docs/figure_manifest.json": _sha(canonical_path), **sources}
    core_path = root / "results/figures/figure_manifest.json"
    core = _read(core_path)
    if len(core["figures"]) != 34:
        raise ValueError("parent core exporter ledger must retain its 34 figures")
    preserved[str(core_path.relative_to(root))] = _sha(core_path)
    archived_core = canonical["core_export_manifest"]
    if _sha(_path(root, archived_core["path"])) != archived_core["sha256"]:
        raise ValueError("parent archived core figure ledger changed")
    preserved[archived_core["path"]] = archived_core["sha256"]
    resolutions = []
    for artifact in exports:
        recorded = Path(artifact["path"])
        # Historical absolute artifact paths may belong to the producing
        # checkout. Only the fixed local export directory is considered after
        # relocation, and its bytes must still match the accepted ledger.
        relocated = recorded.is_absolute() and not recorded.is_relative_to(root)
        local = (
            root / "results/figures" / recorded.name
            if relocated
            else _path(root, recorded)
        )
        local = _path(root, local)
        if _sha(local) != artifact["sha256"]:
            raise ValueError(f"parent figure export changed: {artifact['path']}")
        name = str(local.relative_to(root))
        if name in preserved:
            raise ValueError("parent figure ledger repeats an export path")
        preserved[name] = artifact["sha256"]
        resolutions.append(
            {
                "recorded_path": str(recorded),
                "local_path": name,
                "sha256": artifact["sha256"],
                "relocated": relocated,
            }
        )
    return {
        "source_sha256": sources,
        "preserved_files_sha256": preserved,
        "parent_canonical_ledger": {
            "path": "docs/figure_manifest.json",
            "sha256": _sha(canonical_path),
        },
        "parent_core_ledger": {
            "path": str(core_path.relative_to(root)),
            "sha256": _sha(core_path),
        },
        "parent_export_resolutions": resolutions,
    }


def _timing_figure_complete_manifest(
    root: Path, output: Path, sources: dict[str, str]
) -> dict:
    """Invoke the shared saved-data utility after benchmarking, then add a ledger."""
    metrics_path, timing_path = (
        output / "fresh_metrics.json",
        output / "fresh_timings.json",
    )
    core_path = output / "fresh_figure_manifest.json"
    core_internal_path = output / "fresh_figures/figure_manifest.json"
    evidence_path = output / "fresh_timing_figure_evidence.json"
    complete_path = output / "fresh_figure_manifest_complete.json"
    command_path = output / "fresh_timing_figure_command.json"
    for path in (evidence_path, complete_path, command_path):
        if path.exists():
            raise FileExistsError(
                f"preserve existing supplemental figure artifact: {path}"
            )
    metrics, timing, core = _read(metrics_path), _read(timing_path), _read(core_path)
    if metrics.get("state") != "completed" or timing.get("state") != "completed":
        raise ValueError("timing figure requires completed fresh metrics and benchmark")
    if (
        len(core["figures"]) != 34
        or core["plot_source_sha256"] != sources["src/hydro_pinn/plot.py"]
    ):
        raise ValueError("fresh core ledger or renderer binding differs")
    core_sha = _sha(core_path)
    if _sha(core_internal_path) != core_sha:
        raise ValueError("fresh core exporter ledgers differ")
    input_hashes = {
        str(path.relative_to(root)): _sha(path)
        for path in (metrics_path, timing_path, core_path, core_internal_path)
    }
    for name, digest in sources.items():
        if _sha(_path(root, name)) != digest:
            raise ValueError(
                f"figure renderer changed after reproduction preflight: {name}"
            )
    command = [
        sys.executable,
        str(_path(root, TIMING_FIGURE_HELPER)),
        "--metrics",
        str(metrics_path),
        "--timings",
        str(timing_path),
        "--output-dir",
        str(output / "fresh_figures"),
        "--evidence-output",
        str(evidence_path),
    ]
    invocation = {
        "state": "running",
        "command": command,
        "cwd": str(root),
        "source_sha256": sources,
        "input_sha256": input_hashes,
        "performance_measurements_executed": False,
    }
    write_json_atomic(command_path, invocation)
    try:
        completed = subprocess.run(
            command, cwd=root, capture_output=True, text=True, check=False
        )
        invocation.update(
            state="completed" if completed.returncode == 0 else "failed",
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
        write_json_atomic(command_path, invocation, overwrite=True)
        if completed.returncode:
            raise ValueError(f"timing figure utility failed; see {command_path}")
    except BaseException as exc:
        invocation.update(state="failed", reason=f"{type(exc).__name__}: {exc}")
        write_json_atomic(command_path, invocation, overwrite=True)
        raise
    evidence = _read(evidence_path)
    if (
        evidence.get("state") != "completed"
        or evidence.get("figure_id") != "timing_accuracy"
        or len(evidence.get("points", [])) != 13
        or len(evidence.get("run_ids", [])) != 11
        or {point.get("run_id") for point in evidence.get("points", [])}
        != set(evidence.get("run_ids", []))
        or not set(evidence.get("run_ids", [])).issubset(
            {row["run_id"] for row in metrics["runs"]}
        )
        or evidence.get("performance_measurements_executed_by_renderer") is not False
        or evidence.get("plot_source_sha256")
        != {TIMING_FIGURE_HELPER: sources[TIMING_FIGURE_HELPER]}
    ):
        raise ValueError(
            "supplemental figure evidence has incomplete coverage or source binding"
        )
    for key in ("protocol_sha256", "protocol_file_sha256", "manifest_sha256"):
        if evidence["provenance"][key] != metrics["provenance"][key]:
            raise ValueError("supplemental figure belongs to another fresh attempt")
    source_artifacts = {
        row["path"]: row["sha256"] for row in evidence["source_artifacts"]
    }
    for path in (metrics_path, timing_path):
        key = str(path.relative_to(root))
        if source_artifacts.get(key) != input_hashes[key]:
            raise ValueError(
                "supplemental figure does not bind the fresh metric/timing inputs"
            )
    for name, digest in {**input_hashes, **sources, **source_artifacts}.items():
        if _sha(_path(root, name)) != digest:
            raise ValueError(f"supplemental figure input/source changed: {name}")
    exports = [artifact for row in core["figures"] for artifact in row["artifacts"]]
    exports.extend(evidence["artifacts"])
    expected_supplement = {
        str((output / "fresh_figures" / f"timing_accuracy.{suffix}").relative_to(root))
        for suffix in ("png", "pdf")
    }
    if {row["path"] for row in evidence["artifacts"]} != expected_supplement or len(
        exports
    ) != 70:
        raise ValueError(
            "fresh figure exports must include 68 core and two timing artifacts"
        )
    destinations = set()
    for artifact in exports:
        path = _path(root, artifact["path"])
        if path in destinations or _sha(path) != artifact["sha256"]:
            raise ValueError("fresh figure export is duplicated or its hash changed")
        destinations.add(path)
    complete = copy.deepcopy(core)
    complete["figures"].append(evidence)
    complete.update(
        core_export_manifest={
            "path": str(core_path.relative_to(root)),
            "sha256": core_sha,
        },
        supplemental_plot_source_sha256={
            TIMING_FIGURE_HELPER: sources[TIMING_FIGURE_HELPER]
        },
        supplemental_evidence_artifact={
            "path": str(evidence_path.relative_to(root)),
            "sha256": _sha(evidence_path),
        },
        timing_source_artifact={
            "path": str(timing_path.relative_to(root)),
            "sha256": input_hashes[str(timing_path.relative_to(root))],
        },
        artifact_counts={
            "core_figures": 34,
            "supplemental_figures": 1,
            "figures": 35,
            "exports": 70,
        },
        visual_inspection={
            "state": "pending_root_review",
            "scope": (
                "generated artifacts and hashes checked; visual acceptance is separate"
            ),
        },
    )
    write_json_atomic(complete_path, complete)
    if _sha(core_path) != core_sha or _sha(core_internal_path) != core_sha:
        raise ValueError("supplemental integration changed a preserved core ledger")
    return {
        "state": "generated_hash_verified_visual_review_pending",
        "count": 35,
        "exports": 70,
        "core_count": 34,
        "supplemental_points": 13,
        "supplemental_distinct_run_ids": 11,
        "core_manifest": {"path": str(core_path.relative_to(root)), "sha256": core_sha},
        "canonical_manifest": {
            "path": str(complete_path.relative_to(root)),
            "sha256": _sha(complete_path),
        },
        "supplemental_evidence": {
            "path": str(evidence_path.relative_to(root)),
            "sha256": _sha(evidence_path),
        },
        "command_record": str(command_path.relative_to(root)),
        "source_sha256": sources,
        "visual_inspection": "pending_root_review",
    }


def _paper(protocol: dict, root: Path, output: Path, report: dict) -> None:
    from hydro_pinn.benchmark import benchmark_protocol
    from hydro_pinn.evaluation_workflow import evaluate_manifest, plot_manifest

    parent, metrics, timing = _result_dependencies(protocol, root)
    if (
        _sha(_path(root, protocol["paths"]["manifest"]))
        != report["parent"]["manifest"]["sha256_at_observation"]
    ):
        raise ValueError(
            "parent manifest changed during preflight; retry a fresh destination"
        )
    report["parent"]["manifest"]["completed_artifacts_verified"] = True
    preserved = {
        source["path"]: source["sha256"]
        for source in metrics["provenance"]["source_files"]
    }
    preserved.update(metrics["provenance"].get("evaluation_source_sha256", {}))
    preserved.update(timing.get("provenance", {}).get("source_sha256", {}))
    rendering = _rendering_bindings(root)
    preserved.update(rendering["preserved_files_sha256"])
    report["rendering_sources"] = rendering["source_sha256"]
    report["parent"]["figures"] = {
        "canonical_manifest": rendering["parent_canonical_ledger"],
        "core_manifest": rendering["parent_core_ledger"],
        "figure_count": 35,
        "export_count": len(rendering["parent_export_resolutions"]),
        "exports": rendering["parent_export_resolutions"],
    }
    for name in (
        protocol["paths"]["manifest"],
        "results/metrics.json",
        "results/timings.json",
    ):
        preserved[name] = _sha(_path(root, name))
    _save(output / "report.json", report)
    saved = output / "saved_output_replay"
    saved.mkdir()
    replay_metrics = evaluate_manifest(
        protocol["paths"]["manifest"],
        saved / "metrics.json",
        project_root=root,
        operationally_ready=True,
    )
    replay_figures = plot_manifest(
        protocol["paths"]["manifest"],
        saved / "figures",
        metrics_path=saved / "metrics.json",
        project_root=root,
        figure_manifest_path=saved / "figure_manifest.json",
    )
    saved_comparison = _compare_metrics(metrics, replay_metrics, parent, protocol)
    if not saved_comparison["within_tolerance"]:
        raise ValueError("saved-output metric replay differs from parent metrics")
    report["saved_output_replay"].update(
        state="completed",
        metrics_recomputed_from_saved_arrays=True,
        figures_regenerated_from_saved_arrays=True,
        fresh_reference_solver_training_prediction=False,
        metrics_path=str((saved / "metrics.json").relative_to(root)),
        figures=len(replay_figures["figures"]),
        physical_metric_comparison=saved_comparison,
    )
    _save(output / "report.json", report)
    manifest_path = output / "fresh_manifest.json"
    planned = execution_plan(
        protocol,
        namespace=report["execution_namespace"],
        manifest_path=str(manifest_path.relative_to(root)),
        output_root=protocol["paths"]["output_root"],
    )
    report["fresh_execution"].update(
        purpose="unchanged_frozen_paper_matrix",
        manifest_path=str(manifest_path.relative_to(root)),
        execution=planned,
        state="planned",
    )
    _save(output / "report.json", report)
    try:
        fresh = run_matrix(
            report["parent"]["protocol_path"],
            project_root=root,
            execution_namespace=report["execution_namespace"],
            manifest_path=manifest_path,
            output_root=protocol["paths"]["output_root"],
        )
    except BaseException as original:
        report["fresh_execution"]["state"] = (
            "interrupted"
            if isinstance(original, (KeyboardInterrupt, SystemExit))
            else "failed"
        )
        # Read only the fresh manifest. This observation does not verify all
        # stage artifacts or mutate the matrix writer's retained failure facts.
        try:
            if manifest_path.exists():
                observed = _read(manifest_path)
                if (
                    observed["protocol_sha256"] != protocol["protocol_sha256"]
                    or observed["execution"] != planned
                    or not isinstance(observed["entries"], list)
                ):
                    raise ValueError(
                        "fresh manifest identity differs from planned execution"
                    )
                report["fresh_execution"].update(
                    stages=observed["entries"],
                    observed_workflow_status=observed.get("workflow_status"),
                    manifest_sha256_at_observation=_sha(manifest_path),
                    observed_artifacts_verified=False,
                )
        except BaseException as readback:
            report["fresh_execution"]["manifest_readback_error"] = (
                f"{type(readback).__name__}: {readback}"
            )
        try:
            _save(output / "report.json", report)
        except BaseException:
            # Preserve the matrix exception if best-effort reporting also fails.
            pass
        raise
    report["fresh_execution"].update(
        execution=fresh["execution"],
        stages=fresh["entries"],
        state=fresh["workflow_status"],
    )
    _save(output / "report.json", report)
    if fresh["workflow_status"] not in {"completed", "completed_with_diagnostics"}:
        raise ValueError(
            "fresh paper matrix incomplete; failed stages and runs retained"
        )
    fresh_metrics = evaluate_manifest(
        manifest_path,
        output / "fresh_metrics.json",
        project_root=root,
        operationally_ready=True,
    )
    figures = plot_manifest(
        manifest_path,
        output / "fresh_figures",
        metrics_path=output / "fresh_metrics.json",
        project_root=root,
        figure_manifest_path=output / "fresh_figure_manifest.json",
    )
    fresh_timing = benchmark_protocol(
        report["parent"]["protocol_path"],
        output / "fresh_timings.json",
        project_root=root,
        manifest_path=manifest_path,
        operationally_ready=True,
    )
    report["fresh_execution"].update(
        evaluation={
            "state": fresh_metrics["state"],
            "path": str((output / "fresh_metrics.json").relative_to(root)),
        },
        figures={"state": "completed", "count": len(figures["figures"])},
        timing={
            "state": fresh_timing["state"],
            "path": str((output / "fresh_timings.json").relative_to(root)),
            "comparison": (
                "raw timing reports retained; wall times are not "
                "deterministic numerical outputs"
            ),
            "parent_state": timing["state"],
        },
    )
    _save(output / "report.json", report)
    report["fresh_execution"]["figures"] = _timing_figure_complete_manifest(
        root, output, rendering["source_sha256"]
    )
    _save(output / "report.json", report)
    report["comparison"] = _paper_comparison(parent, fresh, protocol, root)
    report["comparison"].update(
        physical_metrics=_compare_metrics(metrics, fresh_metrics, parent, protocol),
        auxiliary_arrays=_compare_auxiliary(parent, fresh, protocol, root),
    )
    report["comparison"]["within_declared_tolerances"] = (
        report["comparison"]["within_declared_tolerances"]
        and report["comparison"]["physical_metrics"]["within_tolerance"]
        and report["comparison"]["auxiliary_arrays"]["within_tolerance"]
    )
    changed = [
        name for name, digest in preserved.items() if _sha(_path(root, name)) != digest
    ]
    report["parent_artifacts_preserved"] = not changed
    if changed:
        raise ValueError(f"parent artifacts changed during reproduction: {changed}")
    if fresh_metrics["state"] != "completed" or fresh_timing["state"] not in {
        "completed",
        "completed_with_invalid_workloads",
    }:
        raise ValueError("fresh paper evaluation or timing is incomplete")
    if not report["comparison"]["within_declared_tolerances"]:
        report["state"] = "completed_with_differences"
        return
    report["paper_reproduction_established"] = True
    diagnostics = (
        fresh["workflow_status"] == "completed_with_diagnostics"
        or fresh_timing["state"] == "completed_with_invalid_workloads"
    )
    report["state"] = "completed_with_diagnostics" if diagnostics else "completed"


def reproduce(
    protocol_path: str | Path,
    mode: str,
    output_dir: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict:
    """Run smoke or full-paper reproduction into fresh destinations.

    Paper execution can be expensive. The caller schedules it after the parent
    matrix, evaluation and benchmarks finish. Dependencies are checked here;
    there is no hidden reduction of the frozen paper budget.
    """
    if mode not in {"smoke", "paper"}:
        raise ValueError("reproduction mode must be smoke or paper")
    root = Path(project_root).absolute() if project_root else ROOT
    output = _path(root, output_dir)
    output.mkdir(parents=True, exist_ok=False)
    report_path = output / "report.json"
    report = {
        "schema_version": 1,
        "purpose": "course_reproduction_evidence",
        "mode": mode,
        "state": "running",
        "created_at_utc": _now(),
        "execution_namespace": f"p07-{uuid4().hex[:12]}",
        "paper_reproduction_established": False,
        "independent_human_collaborator_reproduction": (
            "not_established_by_agent_execution"
        ),
        "fresh_environment": "not_established_by_execution_in_current_environment",
        "parent": {},
        "fresh_execution": {"stages": []},
        "saved_output_replay": {"state": "not_started"},
        "comparison": {"state": "not_executed"},
        "numerical_tolerances": copy.deepcopy(TOLERANCES),
        "tolerance_scope": (
            "repeated-output agreement only; "
            "not scientific accuracy acceptance thresholds"
        ),
    }
    _save(report_path, report)
    previous_threads = torch.get_num_threads()
    try:
        source = _path(root, protocol_path)
        protocol = load_protocol(source, project_root=root)
        report["parent"].update(
            protocol_path=str(source.relative_to(root)),
            protocol_sha256=protocol["protocol_sha256"],
            protocol_file_sha256=_sha(source),
            manifest=_parent_snapshot(protocol, root, output),
            frozen_training_budget={
                name: value["optimizer"]["steps"]
                for name, value in protocol["resolved"]["training"].items()
            },
            frozen_settings=protocol["settings"],
        )
        report["environment"] = environment_snapshot()
        report["fresh_execution"]["output_root"] = protocol["paths"]["output_root"]
        torch.set_num_threads(protocol["settings"]["cpu_threads"])
        _save(report_path, report)
        if mode == "smoke":
            _smoke(protocol, root, output, report)
        else:
            _paper(protocol, root, output, report)
        load_protocol(source, project_root=root)
        report["finished_at_utc"] = _now()
        _save(report_path, report)
        return report
    except (KeyboardInterrupt, SystemExit) as exc:
        report.update(
            state="interrupted",
            reason=f"{type(exc).__name__}: {exc}",
            finished_at_utc=_now(),
        )
        _save(report_path, report)
        raise
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        report.update(
            state="failed",
            reason=f"{type(exc).__name__}: {exc}",
            finished_at_utc=_now(),
        )
        _save(report_path, report)
        raise ReproductionError(str(exc), report_path) from exc
    finally:
        torch.set_num_threads(previous_threads)
