"""Freeze the accepted core study before consuming final evaluation outputs.

Scientific inputs and execution modules are enforced byte-for-byte. CLI and
entry-point hashes are historical provenance, so later additive evaluation and
reproduction commands do not change the frozen scientific implementation.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hydro_pinn.config import load_config, resolve_run_config, validate_config
from hydro_pinn.records import collect_metadata, validate_run_id, write_json_atomic

ROOT = Path(__file__).resolve().parents[2]
SCIENTIFIC_MODULES = (
    "config",
    "records",
    "physics",
    "references",
    "lax_wendroff",
    "model",
    "sampling",
    "losses",
    "train",
    "protocol",
    "experiments",
)
SELECTION = "minimum_separate_validation_physics_ic_bc_loss"


class ProtocolError(ValueError):
    """A protocol, paired comparison or frozen source has changed."""


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_digest(document: dict) -> str:
    payload = {k: v for k, v in document.items() if k != "protocol_sha256"}
    return hashlib.sha256(
        (json.dumps(payload, allow_nan=False, sort_keys=True, indent=2) + "\n").encode()
    ).hexdigest()


def environment_snapshot() -> dict:
    """Record runtime facts that can change numerical execution."""
    import torch

    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_build": list(platform.python_build()),
        "python_compiler": platform.python_compiler(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("torch", "numpy", "scipy")
        },
        "torch_git_version": torch.version.git_version,
    }


def _adam_defaults() -> dict:
    import torch

    parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    defaults = torch.optim.Adam([parameter], lr=0.001).defaults
    return {"class": "torch.optim.Adam", "defaults": json.loads(json.dumps(defaults))}


def _read_json(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ProtocolError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    def constant(value):
        raise ProtocolError(f"nonfinite JSON constant {value}")

    result = json.loads(
        path.read_text(), object_pairs_hook=pairs, parse_constant=constant
    )
    if not isinstance(result, dict):
        raise ProtocolError("protocol/candidate must be a JSON object")
    return result


def _path(root: Path, path: str | Path) -> Path:
    value = Path(path)
    result = value if value.is_absolute() else root / value
    if ".." in result.parts or any(p.is_symlink() for p in (result, *result.parents)):
        raise ProtocolError(
            "protocol paths must not follow symlinks or parent traversal"
        )
    try:
        result.relative_to(root)
    except ValueError as exc:
        raise ProtocolError("protocol paths must stay within the project root") from exc
    return result


def _input(root: Path, path: str | Path) -> dict:
    resolved = _path(root, path)
    if not resolved.is_file():
        raise ProtocolError(f"missing protocol input {path}")
    return {"path": str(resolved.relative_to(root)), "sha256": file_sha256(resolved)}


def _require(value: Any, expected: Any, name: str) -> None:
    if value != expected:
        detail = (
            "the frozen expected value"
            if isinstance(expected, (dict, list, set))
            else repr(expected)
        )
        raise ProtocolError(f"{name} must equal {detail}")


def _pair(training: dict) -> None:
    first, second = (
        copy.deepcopy(training["baseline"]),
        copy.deepcopy(training["adaptive"]),
    )
    _require(first["weighting"].pop("mode"), "fixed", "baseline weighting mode")
    _require(
        second["weighting"].pop("mode"), "gradient_norm_ema", "adaptive weighting mode"
    )
    if first != second:
        raise ProtocolError("fixed/adaptive pair mismatch beyond weighting mode")


def _settings(candidate: dict) -> dict:
    fields = (
        "device",
        "dtype",
        "cpu_threads",
        "chunk_size",
        "seeds",
        "e2_diagnostic_seed",
        "adam_steps",
        "learning_rate",
        "checkpoint_selection",
        "diagnostic_front_depth_threshold_m",
    )
    result = {name: copy.deepcopy(candidate[name]) for name in fields}
    result["optimizer_execution"] = _adam_defaults()
    return result


def _validate_resolved(resolved: dict, settings: dict) -> None:
    _require(set(resolved["cases"]), {"E1", "E2"}, "case IDs")
    for case_id, case in resolved["cases"].items():
        validate_config(case)
        _require(case["case_id"], case_id, "case identity")
        for name, expected in {
            "domain_m": [-5.0, 5.0],
            "dam_position_m": 0.0,
            "gravity_m_s2": 9.81,
            "final_time_s": 0.5,
            "left_depth_m": 1.0,
            "right_depth_m": 0.1 if case_id == "E1" else 0.0,
            "left_velocity_m_s": 0.0,
            "right_velocity_m_s": 0.0,
        }.items():
            _require(case[name], expected, f"{case_id}.{name}")
        _require(
            case["reference"]["kind"],
            "wet_bed" if case_id == "E1" else "ritter",
            f"{case_id} reference",
        )
        _require(case["grid"]["cell_counts"], [100, 200, 400, 800], "numerical grids")
        _require(case["grid"]["cfl"], 0.45, "numerical CFL")
    evaluation = validate_config(resolved["evaluation"])
    _require(evaluation["partition"], "final_evaluation", "evaluation partition")
    _require(
        evaluation["grid"],
        {"kind": "cell_centres", "domain_m": [-5.0, 5.0], "cells": 400},
        "final evaluation grid",
    )
    _require(
        evaluation["times_s"], [i / 100 for i in range(1, 51)], "final evaluation times"
    )
    _require(evaluation["profile_times_s"], [0.1, 0.25, 0.5], "profile times")
    _require(evaluation["velocity_wet_threshold_m"], 0.001, "reference-wet mask")
    _require(evaluation["wave_band_half_width_m"], 0.1, "wave bands")
    _require(
        evaluation["mass_quadrature"],
        {
            "volume": {
                "rule": "midpoint",
                "cells": [2000, 4000],
                "include_t0": True,
                "include_reported_times": True,
            },
            "boundary_flux": {
                "rule": "trapezoid",
                "nodes": [1001, 2001],
                "include_t0": True,
                "include_final_time": True,
                "include_reported_times": True,
                "spatial_locations": "domain_endpoints",
            },
        },
        "mass quadrature",
    )
    _require(set(resolved["training"]), {"baseline", "adaptive"}, "training variants")
    _pair(resolved["training"])
    baseline = resolved["training"]["baseline"]
    for name, training in resolved["training"].items():
        validate_config(training)
        _require(training["network"]["hidden_layers"], [64] * 4, "network widths")
        _require(training["network"]["depth_floor_nd"], 1e-6, "network depth floor")
        _require(training["sampling"]["interior_points"], 10000, "interior count")
        _require(training["sampling"]["initial_points"], 1000, "initial count")
        _require(
            training["sampling"]["boundary_points_per_side"], 500, "boundary count"
        )
        _require(
            training["optimizer"],
            {"kind": "adam", "steps": 10000, "learning_rate": 0.001},
            "Adam budget",
        )
        _require(training["seeds"], [11, 22, 33], "training seeds")
        _require(
            training["weighting"],
            {
                "mode": "fixed" if name == "baseline" else "gradient_norm_ema",
                "lambda_pde": 1.0,
                "lambda_ic": 10.0,
                "lambda_bc": 10.0,
                "lambda_data": 0.0,
                "update_interval": 100,
                "ema_decay": 0.9,
                "target_bounds": [0.01, 100.0],
                "denominator_epsilon": 1e-12,
                "gradient_norm": "l2_over_sqrt_parameter_count_absent_gradients_zero",
                "update_before_step": True,
                "detach_statistics": True,
            },
            "adaptive rule",
        )
        _require(training["checkpoint_selection"], SELECTION, "checkpoint selection")
        _require(training["lbfgs"]["enabled"], False, "LBFGS selection")
        if training["validation"]["partition"] != "validation":
            raise ProtocolError("checkpoint partition must remain separate validation")
        if any(
            math.isclose(t, u, rel_tol=0, abs_tol=1e-14)
            for t in training["validation"]["times_s"]
            for u in evaluation["times_s"]
        ):
            raise ProtocolError("validation/final evaluation times overlap")
        _require(
            training["validation"]["grid"]["cells"], 73, f"{name} validation cells"
        )
    for name, expected in {
        "device": "cpu",
        "dtype": "float32",
        "cpu_threads": 4,
        "chunk_size": None,
        "seeds": [11, 22, 33],
        "e2_diagnostic_seed": 11,
        "adam_steps": 10000,
        "learning_rate": 0.001,
        "diagnostic_front_depth_threshold_m": 0.001,
    }.items():
        _require(settings[name], expected, f"selected {name}")
    selection = settings["checkpoint_selection"]
    _require(selection["rule"], SELECTION, "selection rule")
    _require(selection["validation"], baseline["validation"], "selection validation")
    _require(
        selection["selection_weights"],
        {"pde": 1.0, "ic": 10.0, "bc": 10.0},
        "selection weights",
    )
    _require(selection["final_analytical_errors_used"], False, "selection leakage flag")
    _require(
        selection["tie_handling"],
        "earliest step; strict validation improvement only",
        "checkpoint tie handling",
    )
    _require(settings["optimizer_execution"], _adam_defaults(), "Adam defaults")


def _matrix(protocol_id: str, settings: dict) -> tuple[list[dict], list[dict]]:
    matrix, pairs = [], []
    for case_id in ("E1", "E2"):
        prefix = f"{protocol_id}-{case_id.lower()}"
        ref = f"{prefix}-reference"
        matrix.append(
            {
                "stage_id": ref,
                "stage": "reference",
                "case_id": case_id,
                "run_id": ref,
                "depends_on": [],
            }
        )
        for cells in (100, 200, 400, 800):
            run_id = f"{prefix}-solve-n{cells}"
            matrix.append(
                {
                    "stage_id": run_id,
                    "stage": "solve",
                    "case_id": case_id,
                    "cells": cells,
                    "run_id": run_id,
                    "depends_on": [ref],
                }
            )
    for case_id, seeds, variants in (
        ("E1", settings["seeds"], ("baseline", "adaptive")),
        ("E2", [settings["e2_diagnostic_seed"]], ("baseline",)),
    ):
        for seed in seeds:
            pair_id = (
                f"{protocol_id}-{case_id.lower()}-s{seed}" if case_id == "E1" else None
            )
            training_ids = []
            for variant in variants:
                prefix = f"{protocol_id}-{case_id.lower()}-{variant}-s{seed}"
                train_id, predict_id = f"{prefix}-train", f"{prefix}-predict"
                shared = {
                    "case_id": case_id,
                    "variant": variant,
                    "seed": seed,
                    "pair_id": pair_id,
                }
                matrix.append(
                    {
                        "stage_id": train_id,
                        "stage": "train",
                        "run_id": train_id,
                        "depends_on": [f"{protocol_id}-{case_id.lower()}-reference"],
                        **shared,
                    }
                )
                matrix.append(
                    {
                        "stage_id": predict_id,
                        "stage": "predict",
                        "run_id": predict_id,
                        "depends_on": [train_id],
                        "training_stage_id": train_id,
                        **shared,
                    }
                )
                training_ids.append(train_id)
            if pair_id:
                pairs.append(
                    {
                        "pair_id": pair_id,
                        "seed": seed,
                        "train_stage_ids": training_ids,
                        "only_difference": "weighting.mode",
                    }
                )
    return matrix, pairs


def _conservation(matrix: list[dict], evaluation: dict, evaluation_hash: str) -> dict:
    left, right = evaluation["grid"]["domain_m"]
    times = [0.0, *evaluation["times_s"]]
    volume, boundary, records = {}, {}, []
    for cells in (2000, 4000):
        volume[str(cells)] = {
            "rule": "midpoint",
            "cells": cells,
            "x_m": [left + (i + 0.5) * (right - left) / cells for i in range(cells)],
            "t_s": times,
        }
    for nodes in (1001, 2001):
        axis = [i * 0.5 / (nodes - 1) for i in range(nodes)]
        for time_s in times:
            index = round(time_s / 0.5 * (nodes - 1))
            if not math.isclose(axis[index], time_s, rel_tol=0, abs_tol=1e-14):
                raise ProtocolError(
                    "reported time is not on boundary quadrature base grid"
                )
            axis[index] = time_s
        boundary[str(nodes)] = {
            "rule": "trapezoid",
            "nodes": nodes,
            "x_m": [left, right],
            "t_s": axis,
        }
    for source in matrix:
        if source["stage"] not in {"reference", "predict"}:
            continue
        for kind, axes in (("volume", volume), ("boundary_flux", boundary)):
            for resolution in axes:
                run_id = f"{source['run_id']}-{kind}-{resolution}"
                records.append(
                    {
                        "run_id": run_id,
                        "run_path": f"runs/{run_id}",
                        "source_stage_id": source["stage_id"],
                        "source_kind": source["stage"],
                        "case_id": source["case_id"],
                        "kind": kind,
                        "axis_key": resolution,
                        "state": "planned",
                        "evaluation_sha256": evaluation_hash,
                        "binding": {
                            "status": "runtime_required",
                            "checkpoint_sha256_required": source["stage"] == "predict",
                            "checkpoint_source_stage_id": source.get(
                                "training_stage_id"
                            ),
                            "resolved_config_sha256_required": True,
                        },
                    }
                )
    return {
        "volume_axes": volume,
        "boundary_axes": boundary,
        "auxiliary_records": records,
        "native_solver_mass": (
            "saved cell sums and actual numerical interface flux over native steps"
        ),
    }


def _validate_structure(protocol: dict) -> None:
    _require(protocol.get("schema_version"), 1, "protocol schema version")
    _require(protocol.get("protocol_version"), 1, "protocol version")
    _require(protocol.get("purpose"), "final_core_study", "protocol purpose")
    validate_run_id(protocol["protocol_id"])
    _validate_resolved(protocol["resolved"], protocol["settings"])
    expected_matrix, expected_pairs = _matrix(
        protocol["protocol_id"], protocol["settings"]
    )
    ids = [stage["run_id"] for stage in protocol["matrix"]]
    ids.extend(row["run_id"] for row in protocol["conservation"]["auxiliary_records"])
    if len(ids) != len(set(ids)):
        raise ProtocolError("duplicate matrix or auxiliary run IDs")
    for run_id in ids:
        validate_run_id(run_id)
    _require(protocol["matrix"], expected_matrix, "immutable matrix and pair settings")
    _require(protocol["pairs"], expected_pairs, "paired comparison identities")
    _require(
        protocol["paths"],
        {"output_root": "runs", "manifest": "docs/final_run_manifest.json"},
        "protocol output paths",
    )
    expected = _conservation(
        expected_matrix,
        protocol["resolved"]["evaluation"],
        protocol["inputs"]["evaluation"]["sha256"],
    )
    _require(protocol["conservation"], expected, "conservation axes/paths/bindings")
    _require(protocol["benchmark"]["warmup_repetitions"], 10, "benchmark warmups")
    _require(protocol["benchmark"]["measured_repetitions"], 30, "benchmark repetitions")


def freeze_protocol(
    config_path: str | Path,
    output_path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict:
    """Create a new final-core protocol; refuse replacement or unreviewed changes."""
    root = Path(project_root).absolute() if project_root else ROOT
    destination = _path(root, output_path)
    if destination.exists():
        raise FileExistsError(f"frozen protocol already exists: {destination}")
    candidate_path = _path(root, config_path)
    candidate = _read_json(candidate_path)
    approval_path = _path(root, "docs/coordination/phase_04_review.json")
    approval = _read_json(approval_path)
    _require(approval["state"], "accepted", "Phase04 review")
    _require(
        approval["candidate_sha256"],
        file_sha256(candidate_path),
        "accepted candidate hash",
    )
    settings = _settings(candidate)
    resolved = {
        "cases": {
            key: load_config(_path(root, path))
            for key, path in candidate["cases"].items()
        },
        "training": {
            key: load_config(_path(root, path))
            for key, path in candidate["training"].items()
        },
        "evaluation": load_config(_path(root, candidate["evaluation"])),
    }
    _validate_resolved(resolved, settings)
    for name in ("network", "sampling", "lbfgs"):
        _require(
            candidate[name], resolved["training"]["baseline"][name], f"candidate {name}"
        )
    _require(
        candidate["mass_quadrature"],
        resolved["evaluation"]["mass_quadrature"],
        "candidate mass quadrature",
    )
    _require(candidate["coverage_reductions"], [], "coverage reductions")
    _require(
        candidate["numerical"]["cells"],
        [100, 200, 400, 800],
        "candidate numerical grids",
    )
    _require(candidate["numerical"]["cfl"], 0.45, "candidate numerical CFL")
    for path, digest in candidate["input_sha256"].items():
        _require(file_sha256(_path(root, path)), digest, f"candidate input hash {path}")
    for case_id in ("E1", "E2"):
        for variant in ("baseline", "adaptive"):
            resolve_run_config(
                _path(root, candidate["cases"][case_id]),
                _path(root, candidate["training"][variant]),
                _path(root, candidate["evaluation"]),
                seed=11,
                device=settings["device"],
                dtype=settings["dtype"],
                run_id="freeze-check",
            )
    inputs = {
        "candidate": _input(root, candidate_path),
        "cases": {key: _input(root, path) for key, path in candidate["cases"].items()},
        "training": {
            key: _input(root, path) for key, path in candidate["training"].items()
        },
        "evaluation": _input(root, candidate["evaluation"]),
        "environment": [
            _input(root, name)
            for name in ("pyproject.toml", "uv.lock", ".python-version")
        ],
        "approval": _input(root, approval_path),
    }
    scientific = {
        f"src/hydro_pinn/{name}.py": file_sha256(
            _path(root, f"src/hydro_pinn/{name}.py")
        )
        for name in SCIENTIFIC_MODULES
    }
    matrix, pairs = _matrix(candidate["candidate_id"], settings)
    metadata = collect_metadata(
        {"request": {"device": settings["device"], "dtype": settings["dtype"]}},
        project_root=root,
    )
    protocol = {
        "schema_version": 1,
        "protocol_version": 1,
        "protocol_id": candidate["candidate_id"],
        "purpose": "final_core_study",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "paths": {"output_root": "runs", "manifest": "docs/final_run_manifest.json"},
        "inputs": inputs,
        "environment": environment_snapshot(),
        "resolved": resolved,
        "settings": settings,
        "scientific_source_sha256": scientific,
        "cli_provenance": {
            "enforcement": "historical_only; additive commands permitted",
            "source_sha256": {
                f"src/hydro_pinn/{name}.py": file_sha256(
                    _path(root, f"src/hydro_pinn/{name}.py")
                )
                for name in ("cli", "__main__")
            },
        },
        "matrix": matrix,
        "pairs": pairs,
        "conservation": _conservation(
            matrix, resolved["evaluation"], inputs["evaluation"]["sha256"]
        ),
        "benchmark": {
            **candidate["benchmark_candidate"],
            "workload": "same final400centres x50times; solver includes reconstruction",
            "cost_categories": [
                "training and persistence",
                "inference only",
                "reference generation",
                "numerical integration and reconstruction",
                "end-to-end generation",
            ],
            "training_repetition": (
                "actual training once per job; not 30 retraining repetitions"
            ),
        },
        "stopping_rules": copy.deepcopy(candidate["stopping_rules"]),
        "output_schema": {
            "fields": ["x_m", "t_s", "h_m", "q_m2_s"],
            "shape": "(number_of_times,number_of_positions)",
            "units": {"x_m": "m", "t_s": "s", "h_m": "m", "q_m2_s": "m^2/s"},
            "solver_state": "native cell averages with cell_edges_m",
            "reference_and_prediction_state": "point values",
            "training_partitions": (
                "fixed training and separate validation; no interior labels"
            ),
        },
        "provenance": {
            "code": metadata["code"],
            "software": metadata["software"],
            "phase04_review": inputs["approval"],
            "freeze_is_not_execution": True,
        },
    }
    _validate_structure(protocol)
    protocol["protocol_sha256"] = protocol_digest(protocol)
    write_json_atomic(destination, protocol)
    return protocol


def load_protocol(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
    check_current: bool = True,
) -> dict:
    """Verify digest, immutable structure and current scientific inputs/sources."""
    root = Path(project_root).absolute() if project_root else ROOT
    protocol = _read_json(_path(root, path))
    _require(
        protocol.get("protocol_sha256"), protocol_digest(protocol), "protocol digest"
    )
    _validate_structure(protocol)
    if check_current:
        _require(
            protocol["environment"],
            environment_snapshot(),
            "frozen runtime environment",
        )
        inputs = protocol["inputs"]
        entries = [
            inputs["candidate"],
            *inputs["cases"].values(),
            *inputs["training"].values(),
            inputs["evaluation"],
            *inputs["environment"],
            inputs["approval"],
        ]
        for entry in entries:
            _require(
                file_sha256(_path(root, entry["path"])),
                entry["sha256"],
                f"frozen input hash {entry['path']}",
            )
        for name in ("cases", "training"):
            for key, entry in inputs[name].items():
                _require(
                    load_config(_path(root, entry["path"])),
                    protocol["resolved"][name][key],
                    f"resolved {name}.{key}",
                )
        _require(
            load_config(_path(root, inputs["evaluation"]["path"])),
            protocol["resolved"]["evaluation"],
            "resolved final evaluation",
        )
        expected_modules = {f"src/hydro_pinn/{name}.py" for name in SCIENTIFIC_MODULES}
        _require(
            set(protocol["scientific_source_sha256"]),
            expected_modules,
            "scientific source enforcement set",
        )
        for name, digest in protocol["scientific_source_sha256"].items():
            _require(
                file_sha256(_path(root, name)),
                digest,
                f"frozen scientific source hash {name}",
            )
    return protocol
