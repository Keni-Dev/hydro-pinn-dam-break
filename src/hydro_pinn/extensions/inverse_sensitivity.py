"""Finite Phase09 numerical depth-response screen; no inverse data or trainer.

Run ``python -m hydro_pinn.extensions.inverse_sensitivity prepare`` before
``execute``. Preparation freezes inputs and axes without calling a solver.
Execution claims one immutable launch and cannot resume, replace or retry it.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import itertools
import json
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from hydro_pinn.extensions.friction_reference import (
    conservative_project,
    load_reference,
    solve_reference,
)
from hydro_pinn.records import write_json_atomic

PLAN_SHA = "298ebfd506094ceb42e9997d1c78602b87cadb52384e5469f4ee82eb4646a05c"
DEFAULT_CONFIG = "configs/extensions/inverse_sensitivity.json"
DEFAULT_PROTOCOL = "results/extensions/inverse_sensitivity/gate_v01/protocol.json"
LEVELS = [
    {"id": "n1600_cfl0p45", "cells": 1600, "cfl": 0.45},
    {"id": "n3200_cfl0p45", "cells": 3200, "cfl": 0.45},
    {"id": "n3200_cfl0p225", "cells": 3200, "cfl": 0.225},
]


def utc() -> str:
    return datetime.now(UTC).isoformat()


def sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_hashes(expected: dict[str, str]) -> None:
    for name, digest in expected.items():
        if not Path(name).is_file() or sha(name) != digest:
            raise ValueError(f"frozen input changed or missing: {name}")


def save_arrays(path: Path, **arrays: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def extract_gauge_means(
    native: dict[str, np.ndarray],
    common_edges: np.ndarray,
    rows: list[int],
    cells: list[int],
) -> np.ndarray:
    """Conservative interval means, flattened by time then declared gauge."""
    common = conservative_project(native["h_m"], native["cell_edges_m"], common_edges)
    return common[np.ix_(rows, cells)].ravel(order="C")


def pair_statistics(delta: np.ndarray, U: np.ndarray, sigma: float) -> dict:
    """Heuristic numerical adjustment and one-vector noise-unit distances."""
    delta, U = np.asarray(delta, dtype=np.float64), np.asarray(U, dtype=np.float64)
    if delta.shape != U.shape or np.any(U < 0) or sigma <= 0:
        raise ValueError("pair vectors must match with nonnegative U and sigma>0")
    if not np.isfinite(delta).all() or not np.isfinite(U).all():
        raise ValueError("nonfinite pair vector")
    s = np.maximum(np.abs(delta) - U, 0)
    raw, adj, uncertainty = map(
        float, (np.linalg.norm(delta), np.linalg.norm(s), np.linalg.norm(U))
    )
    return {
        "signed_delta_m": delta.tolist(),
        "combined_indicator_U_m": U.tolist(),
        "adjusted_magnitude_s_m": s.tolist(),
        "raw_RMS_m": float(np.sqrt(np.mean(delta**2))),
        "raw_max_abs_m": float(np.max(np.abs(delta))),
        "raw_L2_m": raw,
        "U_L2_m": uncertainty,
        "adjusted_RMS_m": float(np.sqrt(np.mean(s**2))),
        "adjusted_max_m": float(s.max()),
        "adjusted_L2_m": adj,
        "d_raw": raw / sigma,
        "d_adj": adj / sigma,
        "max_adjusted_over_sigma": float(s.max() / sigma),
        "visible_sample_count": int(np.count_nonzero(np.abs(delta) > U)),
        "indicator_dominated": raw <= uncertainty,
        "raw_pass": raw / sigma >= 1.0,
        "adjusted_pass": adj / sigma >= 1.0,
        "adjustment_changes_screen": (raw / sigma >= 1.0) != (adj / sigma >= 1.0),
    }


def _host() -> dict:
    observed = {}
    for label, command in {
        "cpu_brand": ["sysctl", "-n", "machdep.cpu.brand_string"],
        "physical_memory_bytes": ["sysctl", "-n", "hw.memsize"],
        "vm_stat": ["vm_stat"],
    }.items():
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=5)
            observed[label] = {
                "command": command,
                "exit": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        except OSError as exc:
            observed[label] = {"unavailable": repr(exc)}
    return {
        "recorded_at_utc": utc(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "pid": os.getpid(),
        "numpy": np.__version__,
        "observations": observed,
        "ru_maxrss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "ru_maxrss_units": "bytes on macOS; process high-water resident memory",
        "thread_environment": {
            k: os.environ.get(k)
            for k in [
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            ]
        },
    }


def _validate_config(c: dict) -> None:
    fixed = {
        "plan_sha256": PLAN_SHA,
        "n_values": [0.020, 0.025, 0.030, 0.035, 0.040],
        "candidate_ids": ["n0p020", "n0p025", "n0p030", "n0p035", "n0p040"],
        "truth_n": 0.03,
        "levels": LEVELS,
        "selected_level": "n3200_cfl0p225",
        "time_row_indices": [9, 19, 29, 39, 49],
        "gauge_common_cell_indices": [220, 240, 260],
        "common_cells": 400,
        "sigma_m": 0.01,
        "threshold": 1.0,
        "numerical_matrix_budget_s": 60.0,
        "bed_elevation_m": 0.0,
        "rainfall_m_s": 0.0,
        "dtype": "float64",
        "device": "cpu",
        "case": "configs/cases/e1_wet.json",
        "reference_root": "results/extensions/friction/reference_v01",
        "plan": "docs/coordination/phase_09_sensitivity_gate_plan_v01.json",
        "selection": "docs/coordination/extension_inverse_sensitivity_selection.json",
    }
    for key, value in fixed.items():
        if c.get(key) != value:
            raise ValueError(f"config changes adopted gate: {key}")


def prepare(config_path: str | Path = DEFAULT_CONFIG) -> dict:
    """Hash-check accepted evidence and freeze exact axes before any new solve."""
    c = json.loads(Path(config_path).read_text())
    out, evidence = Path(c["output_root"]), Path(c["evidence_root"])
    if out.exists() or evidence.exists():
        raise FileExistsError("gate output/evidence already exists; never overwrite")
    _validate_config(c)
    assert_hashes({c["plan"]: PLAN_SHA})
    plan = json.loads(Path(c["plan"]).read_text())
    selection = json.loads(Path(c["selection"]).read_text())
    if selection["state"] != "released_selected_bounded_sensitivity_gate_only":
        raise ValueError("bounded gate not released")
    protected = {
        **selection["protected_nonCLI_sha256"],
        "src/hydro_pinn/cli.py": selection["protected_CLI_sha256"],
        **selection["protected_archives"],
        **plan["observed_input_sha256"],
    }
    ref = Path(c["reference_root"])
    protected.update({str(p): sha(p) for p in ref.rglob("*") if p.is_file()})
    assert_hashes(
        {
            **plan["observed_input_sha256"],
            **selection["protected_nonCLI_sha256"],
            "src/hydro_pinn/cli.py": selection["protected_CLI_sha256"],
            **selection["protected_archives"],
        }
    )
    common = load_reference(ref)
    times, edges = common["t_s"], common["cell_edges_m"]
    if times.shape != (50,) or edges.shape != (401,) or times.dtype != np.float64:
        raise ValueError("accepted common axes differ")
    rows, cells = c["time_row_indices"], c["gauge_common_cell_indices"]
    np.testing.assert_allclose(
        times[rows], [0.1, 0.2, 0.3, 0.4, 0.5], atol=1e-15, rtol=0
    )
    for level in LEVELS:
        native = load_reference(ref / level["id"])
        np.testing.assert_array_equal(native["t_s"], times)
        np.testing.assert_array_equal(
            native["cell_edges_m"], np.linspace(-5, 5, level["cells"] + 1)
        )
    out.mkdir(parents=True, exist_ok=False)
    evidence.mkdir(parents=True, exist_ok=False)
    axes = {
        "t_s": times,
        "common_cell_edges_m": edges,
        "time_row_indices": np.array(rows),
        "gauge_cell_indices": np.array(cells),
    }
    for count in [1600, 3200]:
        axes[f"native_edges_{count}_m"] = np.linspace(-5, 5, count + 1)
    save_arrays(out / "frozen_axes.npz", **axes)
    inputs = {
        **protected,
        str(config_path): sha(config_path),
        c["selection"]: sha(c["selection"]),
        "docs/coordination/phase_08_completion_review.json": sha(
            "docs/coordination/phase_08_completion_review.json"
        ),
        str(Path(__file__).relative_to(Path.cwd())): sha(__file__),
        "tests/test_inverse_sensitivity.py": sha("tests/test_inverse_sensitivity.py"),
        "docs/extensions/inverse_sensitivity_protocol_v01.md": sha(
            "docs/extensions/inverse_sensitivity_protocol_v01.md"
        ),
        str(out / "frozen_axes.npz"): sha(out / "frozen_axes.npz"),
    }
    matrix = []
    for candidate, n in zip(c["candidate_ids"], c["n_values"], strict=True):
        for level in LEVELS:
            reused = n == 0.03
            path = ref / level["id"] if reused else out / candidate / level["id"]
            matrix.append(
                {
                    "candidate_id": candidate,
                    "n_s_m_minus_one_third": n,
                    **level,
                    "reused": reused,
                    "path": str(path / "outputs.npz"),
                }
            )
    protocol = {
        "schema_version": 1,
        "frozen_at_utc": utc(),
        "config": c,
        "input_sha256": inputs,
        "preserved_sha256": protected,
        "case_template": json.loads(Path(c["case"]).read_text()),
        "source_override": "Sq=-g*n^2*q*abs(q)/h^(7/3); positive n, flat bed/no rain",
        "method": plan["numerical_matrix"]["method"],
        "operator_source_sha256": hashlib.sha256(
            (
                inspect.getsource(conservative_project)
                + inspect.getsource(extract_gauge_means)
            ).encode()
        ).hexdigest(),
        "full_t_s": times.tolist(),
        "selected_t_s": times[rows].tolist(),
        "gauge_edges_m": [[float(edges[i]), float(edges[i + 1])] for i in cells],
        "sample_order": [
            {
                "time_row": row,
                "actual_t_s": float(times[row]),
                "gauge_id": f"G{j + 1}",
                "common_cell": cell,
            }
            for row in rows
            for j, cell in enumerate(cells)
        ],
        "matrix": matrix,
        "new_solve_count": 12,
        "reused_count": 3,
        "budget": "60s matrix wall including I/O; checked between solves",
        "noise": "Independent Gaussian sigma=.01m illustrative only; no draws",
        "adjustment": "e=|mesh|+|time|; U=e_a+e_b; s=max(|delta|-U,0); heuristic",
        "screen": "L2(s)/sigma>=1 for BOTH truth vs .025 and .035; weak screen only",
        "truth_context": (
            "Numerical truth; accepted08 forward flow recovery remains incomplete"
        ),
        "no_followon": "No inverse data/fit, pilot, refinements or trainer released",
    }
    write_json_atomic(out / "protocol.json", protocol)
    with (out / "protocol.sha256").open("x") as stream:
        stream.write(sha(out / "protocol.json") + "\n")
    # Store the executable driver/config bytes in addition to hashes.
    for source, destination in [
        (Path(__file__), evidence / "frozen_driver.py"),
        (Path(config_path), evidence / "frozen_config.json"),
    ]:
        with destination.open("xb") as stream:
            stream.write(source.read_bytes())
    write_json_atomic(
        evidence / "prepare.json",
        {
            "state": "frozen_before_any_new_solve",
            "at_utc": utc(),
            "command": [sys.executable, *sys.argv],
            "cwd": str(Path.cwd()),
            "protocol_sha256": sha(out / "protocol.json"),
            "host": _host(),
            "all_protected_hashes_match": True,
        },
    )
    return protocol


def verify_frozen(protocol_path: str | Path = DEFAULT_PROTOCOL) -> dict:
    path = Path(protocol_path)
    if sha(path) != path.with_suffix(".sha256").read_text().strip():
        raise ValueError("frozen protocol hash mismatch")
    p = json.loads(path.read_text())
    _validate_config(p["config"])
    assert_hashes(p["input_sha256"])
    return p


def _diagnostics(a: dict, entry: dict) -> dict:
    dt = a["step_dt_s"]
    flux = np.cumsum(
        dt[:, None] * (a["step_right_numerical_flux"] - a["step_left_numerical_flux"]),
        axis=0,
    )
    mass = a["step_mass_m2"] - a["initial_mass_m2"] + flux[:, 0]
    momentum = (
        a["step_momentum_m3_s"]
        - a["initial_momentum_m3_s"]
        + flux[:, 1]
        - np.cumsum(a["step_source_impulse_m3_s"])
    )
    np.testing.assert_allclose(
        mass, a["step_mass_budget_defect_m2"], atol=1e-13, rtol=0
    )
    np.testing.assert_allclose(
        momentum, a["step_momentum_budget_defect_m3_s"], atol=1e-13, rtol=0
    )
    if (
        not all(np.isfinite(v).all() for v in a.values())
        or a["step_running_minimum_h_m"].min() <= 0
        or a["step_cfl"].max() > entry["cfl"] + 1e-14
        or np.max(np.abs(mass)) > 2e-12
        or np.max(np.abs(momentum)) > 2e-12
    ):
        raise ValueError("native finite/positivity/CFL/budget evidence failed")
    return {
        "steps": len(dt),
        "solve_wall_s": float(a["solve_wall_s"]),
        "dt_min_s": float(dt.min()),
        "dt_max_s": float(dt.max()),
        "maximum_actual_cfl": float(a["step_cfl"].max()),
        "minimum_stage_h_m": float(a["step_running_minimum_h_m"].min()),
        "minimum_saved_h_m": float(a["h_m"].min()),
        "native_initial_mass_m2": float(a["initial_mass_m2"]),
        "native_final_mass_m2": float(a["step_mass_m2"][-1]),
        "native_final_momentum_m3_s": float(a["step_momentum_m3_s"][-1]),
        "final_boundary_flux_integral": flux[-1].tolist(),
        "final_source_impulse_m3_s": float(a["cumulative_source_impulse_m3_s"][-1]),
        "max_abs_mass_defect_m2": float(np.max(np.abs(mass))),
        "max_abs_momentum_defect_m3_s": float(np.max(np.abs(momentum))),
        "budget_recomputed_from_saved_step_flux_and_source": True,
        "strictly_wet_stage_validation": (
            "Accepted solver validates wet stages; no clipping/floor/retry"
        ),
    }


def analyze(responses: dict, c: dict) -> dict:
    selected = c["selected_level"]
    indicators = {}
    for candidate in c["candidate_ids"]:
        r = responses[candidate]
        mesh = r["n3200_cfl0p45"] - r["n1600_cfl0p45"]
        temporal = r[selected] - r["n3200_cfl0p45"]
        indicators[candidate] = {
            "signed_mesh_difference_m": mesh.tolist(),
            "signed_time_difference_m": temporal.tolist(),
            "abs_mesh_m": np.abs(mesh).tolist(),
            "abs_time_m": np.abs(temporal).tolist(),
            "e_m": (np.abs(mesh) + np.abs(temporal)).tolist(),
        }
    pairs = []
    for a, b in itertools.combinations(c["candidate_ids"], 2):
        contrasts = {
            level["id"]: (
                responses[b][level["id"]] - responses[a][level["id"]]
            ).tolist()
            for level in LEVELS
        }
        U = np.array(indicators[a]["e_m"]) + indicators[b]["e_m"]
        pairs.append(
            {
                "a": a,
                "b": b,
                "sign": "b minus a",
                "signed_contrasts_all_levels_m": contrasts,
                "candidate_a_indicators": indicators[a],
                "candidate_b_indicators": indicators[b],
                **pair_statistics(np.array(contrasts[selected]), U, c["sigma_m"]),
            }
        )
    truth_pairs = [p for p in pairs if "n0p030" in [p["a"], p["b"]]]
    local = [
        p
        for p in truth_pairs
        if any(n in [p["a"], p["b"]] for n in ["n0p025", "n0p035"])
    ]
    outer = [p for p in truth_pairs if p not in local]
    local_pass = all(p["adjusted_pass"] for p in local)
    unresolved = any(
        p["indicator_dominated"] or p["adjustment_changes_screen"] for p in local
    )
    outcome = (
        "unresolved_at_this_numerical_operator"
        if unresolved
        else "passes_weak_local_screen"
        if local_pass
        else "weak_at_fixed_1cm_noise_scale"
    )
    J = (responses["n0p035"][selected] - responses["n0p025"][selected]) / 0.010
    second = (
        responses["n0p035"][selected]
        + responses["n0p025"][selected]
        - 2 * responses["n0p030"][selected]
    )
    return {
        "state": "complete_bounded_sensitivity_screen",
        "outcome": outcome,
        "responses_all_levels_m": {
            n: {level: v.tolist() for level, v in r.items()}
            for n, r in responses.items()
        },
        "candidate_indicators": indicators,
        "pairs": pairs,
        "primary_local_both_pass": local_pass,
        "range_truth_all_pass": all(p["adjusted_pass"] for p in truth_pairs),
        "truth_local_pairs": [[p["a"], p["b"]] for p in local],
        "truth_outer_pairs": [[p["a"], p["b"]] for p in outer],
        "ambiguous_pairs_on_declared_grid": [
            [p["a"], p["b"]] for p in pairs if not p["adjusted_pass"]
        ],
        "finite_difference": {
            "J_m_per_n": J.tolist(),
            "J_times_truth_n_over_H": (J * 0.03 / 1).tolist(),
            "central_second_difference_m": second.tolist(),
        },
        "noise_sigma_m": 0.01,
        "noise_draws": 0,
        "threshold_interpretation": (
            "One-noise-unit weak screen; no confidence/recovery guarantee"
        ),
        "numerical_indicator_limit": (
            "Heuristic adjustment; no certified bound or identifiability claim"
        ),
        "finite_difference_limit": (
            "Descriptive finite difference; no exact gradient or Fisher interval"
        ),
        "inverse_observation_dataset_created": False,
        "inverse_trainer_created": False,
        "phase09_complete": False,
        "followon_released": False,
    }


def _figure(report: dict, out: Path, c: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7), layout="constrained")
    responses = report["responses_all_levels_m"]
    truth = np.array(responses["n0p030"][c["selected_level"]])
    raw, adjusted = [], []
    for candidate, n in zip(c["candidate_ids"], c["n_values"], strict=True):
        v = np.array(responses[candidate][c["selected_level"]]) - truth
        axes[0].plot(np.arange(1, 16), v * 1000, ".-", lw=1, label=f"n={n:.3f}")
        match = next(
            (
                p
                for p in report["pairs"]
                if set([p["a"], p["b"]]) == set([candidate, "n0p030"])
            ),
            None,
        )
        raw.append(match["d_raw"] if match else 0.0)
        adjusted.append(match["d_adj"] if match else 0.0)
    axes[0].axhspan(-10, 10, color="0.8", alpha=0.4, label="Illustrative ±σ (10 mm)")
    axes[0].set(
        xlabel="Fixed sample (time-major, G1/G2/G3)",
        ylabel="Selected numerical mean minus n=.030 (mm)",
    )
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].plot(c["n_values"], raw, "o-", label="Raw L2 / σ")
    axes[1].plot(c["n_values"], adjusted, "s-", label="Adjusted L2 / σ")
    axes[1].axhline(1, color="0.4", ls="--", label="Weak one-unit screen")
    axes[1].set(xlabel="Manning n (s m⁻¹ᐟ³)", ylabel="Aggregate distance to n=.030")
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Fixed numerical depth response: three interval means × five times", fontsize=11
    )
    fig.supxlabel(
        (
            "Heuristic mesh/time adjustment. No noisy "
            "data or fit; truth zero is tautological."
        ),
        fontsize=8,
    )
    destination = out / "sensitivity_profile.png"
    with destination.open("xb") as stream:
        fig.savefig(stream, format="png", dpi=160)
    plt.close(fig)


def execute(protocol_path: str | Path = DEFAULT_PROTOCOL) -> dict:
    start = time.perf_counter()
    p = verify_frozen(protocol_path)
    c = p["config"]
    out, evidence = Path(c["output_root"]), Path(c["evidence_root"])
    launch = {
        "at_utc": utc(),
        "pid": os.getpid(),
        "cwd": str(Path.cwd()),
        "command": [sys.executable, *sys.argv],
        "protocol_sha256": sha(protocol_path),
        "host": _host(),
        "exactly_once_no_resume": True,
    }
    write_json_atomic(evidence / "launch.json", launch)
    with np.load(out / "frozen_axes.npz", allow_pickle=False) as axes:
        times, edges = axes["t_s"].copy(), axes["common_cell_edges_m"].copy()
    records, responses = [], {n: {} for n in c["candidate_ids"]}
    matrix_start = time.perf_counter()
    for entry in p["matrix"]:
        attempt = {**entry, "at_utc": utc(), "pid": os.getpid()}
        if not entry["reused"] and time.perf_counter() - matrix_start >= 60:
            records.append({**attempt, "state": "not_started_budget_limit"})
            continue
        attempt_start = time.perf_counter()
        run_dir = Path(entry["path"]).parent
        try:
            if entry["reused"]:
                a = load_reference(entry["path"])
            else:
                run_dir.mkdir(parents=True, exist_ok=False)
                write_json_atomic(run_dir / "attempt.json", attempt)
                a = solve_reference(
                    p["case_template"],
                    entry["n_s_m_minus_one_third"],
                    entry["cells"],
                    entry["cfl"],
                    times,
                )
                storage_start = time.perf_counter()
                save_arrays(Path(entry["path"]), **a)
                attempt["native_storage_wall_s"] = time.perf_counter() - storage_start
            np.testing.assert_array_equal(a["t_s"], times)
            np.testing.assert_array_equal(
                a["cell_edges_m"], np.linspace(-5, 5, entry["cells"] + 1)
            )
            extraction_start = time.perf_counter()
            mu = extract_gauge_means(
                a, edges, c["time_row_indices"], c["gauge_common_cell_indices"]
            )
            attempt["projection_extraction_wall_s"] = (
                time.perf_counter() - extraction_start
            )
            responses[entry["candidate_id"]][entry["id"]] = mu
            record = {
                **attempt,
                "state": "reused" if entry["reused"] else "completed",
                "outputs_sha256": sha(entry["path"]),
                "gauge_means_m": mu.tolist(),
                "diagnostics": _diagnostics(a, entry),
                "total_record_wall_s": time.perf_counter() - attempt_start,
            }
            if not entry["reused"]:
                write_json_atomic(run_dir / "receipt.json", record)
            print(
                json.dumps(
                    {
                        "candidate": entry["candidate_id"],
                        "level": entry["id"],
                        "state": record["state"],
                        "steps": record["diagnostics"]["steps"],
                        "elapsed_matrix_s": time.perf_counter() - matrix_start,
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            responses[entry["candidate_id"]].pop(entry["id"], None)
            record = {
                **attempt,
                "state": "failed",
                "reason": repr(exc),
                "traceback": traceback.format_exc(),
                "total_record_wall_s": time.perf_counter() - attempt_start,
                "partial_native_path": entry["path"]
                if Path(entry["path"]).exists()
                else None,
                "partial_stage_arrays_available": False,
                "failure_limit": (
                    "No intermediate state returned by accepted solver; no retry"
                ),
            }
            if not entry["reused"]:
                write_json_atomic(run_dir / "failure.json", record)
        records.append(record)
    matrix_wall = time.perf_counter() - matrix_start
    write_json_atomic(
        out / "matrix.json",
        {
            "records": records,
            "wall_s": matrix_wall,
            "budget_s": 60,
            "budget_exceeded": matrix_wall >= 60,
        },
    )
    complete = matrix_wall < 60 and all(
        r["state"] in ["completed", "reused"] for r in records
    )
    report_start = time.perf_counter()
    if complete:
        report = analyze(responses, c)
        write_json_atomic(out / "analysis.json", report)
        save_arrays(
            out / "gauge_means.npz",
            n_values=np.array(c["n_values"]),
            means_m=np.array(
                [
                    [responses[n][level["id"]] for level in LEVELS]
                    for n in c["candidate_ids"]
                ]
            ),
            selected_t_s=times[c["time_row_indices"]],
            common_cell_edges_m=edges,
        )
        _figure(report, out, c)
    else:
        report = {
            "state": "incomplete_bounded_gate",
            "outcome": "missing_or_failed_numerical_evidence",
            "budget_limited": matrix_wall >= 60
            or any(r["state"] == "not_started_budget_limit" for r in records),
            "inverse_observation_dataset_created": False,
            "inverse_trainer_created": False,
            "followon_released": False,
            "phase09_complete": False,
        }
        write_json_atomic(out / "incomplete_decision.json", report)
    reporting_wall = time.perf_counter() - report_start
    assert_hashes(p["preserved_sha256"])
    assert_hashes(p["input_sha256"])
    preservation = {
        "at_utc": utc(),
        "all_match": True,
        "before_sha256": p["preserved_sha256"],
        "after_sha256": {name: sha(name) for name in p["preserved_sha256"]},
    }
    write_json_atomic(evidence / "preservation_after.json", preservation)
    terminal = {
        "at_utc": utc(),
        "pid": os.getpid(),
        "state": report["state"],
        "outcome": report["outcome"],
        "protocol_sha256": sha(protocol_path),
        "new_completed": sum(r["state"] == "completed" for r in records),
        "new_failed": sum(r["state"] == "failed" for r in records),
        "reused": sum(r["state"] == "reused" for r in records),
        "matrix_wall_s": matrix_wall,
        "reporting_wall_s": reporting_wall,
        "total_wall_s": time.perf_counter() - start,
        "bytes_under_output": sum(
            f.stat().st_size for f in out.rglob("*") if f.is_file()
        ),
        "host_after": _host(),
        "all_protected_hashes_match": True,
    }
    write_json_atomic(evidence / "terminal.json", terminal)
    return {
        "terminal": terminal,
        "decision": {
            k: report[k]
            for k in [
                "state",
                "outcome",
                "inverse_observation_dataset_created",
                "inverse_trainer_created",
            ]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "execute"])
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--protocol", default=DEFAULT_PROTOCOL)
    args = parser.parse_args()
    if args.action == "prepare":
        p = prepare(args.config)
        print(json.dumps({"state": "frozen", "new_solves": p["new_solve_count"]}))
    else:
        print(json.dumps(execute(args.protocol), indent=2))


if __name__ == "__main__":
    main()
