"""Separate wet flat-bed Manning forward study and bounded CPU cost pilot.

The core case is an immutable geometry/IC/BC/scales template. The extension
configuration supplies n=0.03 and overrides its source-free physical assumption
only within this module. Final comparisons use common-cell averages: conservative
FV projection versus Gauss-Legendre integration of PINN h and q. Pilot predictions
are persisted to measure workload, but never compared with reference accuracy.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import cumulative_trapezoid

from hydro_pinn.config import load_config
from hydro_pinn.extensions.friction_physics import friction_loss_components
from hydro_pinn.model import ShallowWaterPINN
from hydro_pinn.physics import Scales
from hydro_pinn.records import RunRecord, validate_run_id, write_json_atomic
from hydro_pinn.sampling import (
    generate_training_samples,
    generate_validation_samples,
    load_samples,
    save_samples,
)

SOURCES = (
    "src/hydro_pinn/extensions/friction_physics.py",
    "src/hydro_pinn/extensions/friction_reference.py",
    "src/hydro_pinn/extensions/friction_experiment.py",
    "src/hydro_pinn/cli.py",
    "src/hydro_pinn/config.py",
    "src/hydro_pinn/model.py",
    "src/hydro_pinn/physics.py",
    "src/hydro_pinn/losses.py",
    "src/hydro_pinn/sampling.py",
    "src/hydro_pinn/records.py",
    "pyproject.toml",
    "uv.lock",
)
COORD_KEYS = ("interior_xt_nd", "initial_xt_nd", "boundary_xt_nd")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, value: dict, *, overwrite: bool = False) -> None:
    write_json_atomic(path, value, overwrite=overwrite)


def load_friction_config(path: str | Path, root: Path = Path(".")) -> tuple:
    cfg = _read(root / path)
    expected = {
        "schema_version": 1,
        "extension": "wet_flat_manning_forward_v01",
        "case": "configs/cases/e1_wet.json",
        "training_template": "configs/training/baseline.json",
        "reference_packet": "results/extensions/friction/reference_v01",
        "manning_n_s_m_minus_one_third": 0.03,
        "bed_elevation_m": 0.0,
        "rainfall": 0.0,
        "seed": 11,
        "sampling_seed": 11,
        "device": "cpu",
        "dtype": "float32",
        "cpu_threads": 4,
        "hidden_layers": [64] * 4,
        "depth_floor_nd": 1e-6,
        "adam_lr": 0.001,
        "full_steps": 10000,
        "pilot_steps": 25,
        "weights": {"pde": 1.0, "ic": 10.0, "bc": 10.0},
        "sampling": {"pde": 10000, "ic": 1000, "per_boundary": 500},
        "evaluation": {
            "common_cells": 400,
            "times_s": [i / 100 for i in range(1, 51)],
            "pinn_cell_gauss_orders": [3, 5],
            "reference_wet_threshold_m": 0.001,
            "mass_midpoint_cells": [800, 1600],
            "boundary_trapezoid_nodes_before_disjoint_filter": [1001, 2001],
            "query_chunk_size": 8192,
        },
    }
    if cfg != expected:
        raise ValueError("configuration differs from fixed friction v01 proposal")
    case, training = (
        load_config(root / cfg["case"]),
        load_config(root / cfg["training_template"]),
    )
    if (
        case["case_id"] != "E1"
        or case["domain_m"] != [-5.0, 5.0]
        or case["left_depth_m"] != 1.0
        or case["right_depth_m"] != 0.1
        or case["final_time_s"] != 0.5
        or case["gravity_m_s2"] != 9.81
        or case["boundary"]["kind"] != "fixed_reservoir"
    ):
        raise ValueError("friction v01 requires the unchanged selected wet E1 template")
    if (
        training["network"]["hidden_layers"] != cfg["hidden_layers"]
        or training["sampling"]["interior_points"] != 10000
        or training["sampling"]["initial_points"] != 1000
        or training["sampling"]["boundary_points_per_side"] != 500
        or training["weighting"]["mode"] != "fixed"
        or training["optimizer"]["learning_rate"] != 0.001
        or training["lbfgs"]["enabled"]
    ):
        raise ValueError("core training template no longer matches selected design")
    return cfg, case, training


def _rows(arrays: list[np.ndarray], dtype: str) -> set[tuple]:
    return {tuple(row) for a in arrays for row in a.astype(dtype)}


def final_memberships(cfg: dict, case: dict, train: dict, val: dict) -> dict:
    """Freeze all query axes; remove auxiliary boundary collisions before fitting.

    Trapezoid nodes remain ordered and include 0,T and all report times. The
    removed nodes are nonreport nodes equal to any training/validation boundary
    time after float32 conversion. This avoids calling reused BC points held out.
    """
    e = cfg["evaluation"]
    a, b = case["domain_m"]
    edges = np.linspace(a, b, e["common_cells"] + 1)
    result = {
        "cell_edges_m": edges,
        "x_m": (edges[1:] + edges[:-1]) / 2,
        "t_s": np.array(e["times_s"]),
        "report_t_s": np.r_[0.0, e["times_s"]],
    }
    for order in e["pinn_cell_gauss_orders"]:
        z, w = np.polynomial.legendre.leggauss(order)
        result[f"gauss_x_{order}_m"] = (
            result["x_m"][:, None] + np.diff(edges)[:, None] * z / 2
        )
        result[f"gauss_w_{order}"] = w / 2
    sc = Scales.from_case(case)
    forbidden = np.concatenate([p["boundary_xt_nd"][:, 1] for p in (train, val)])
    for level, (cells, nodes) in enumerate(
        zip(
            e["mass_midpoint_cells"],
            e["boundary_trapezoid_nodes_before_disjoint_filter"],
            strict=True,
        )
    ):
        result[f"mass_x_{level}_m"] = a + (np.arange(cells) + 0.5) * (b - a) / cells
        times = np.unique(
            np.r_[np.linspace(0, case["final_time_s"], nodes), result["report_t_s"]]
        )
        hit = (
            np.isin((times / sc.Tc_s).astype("float32"), forbidden.astype("float32"))
            | np.isin(
                times.astype("float32") / np.float32(sc.Tc_s),
                forbidden.astype("float32"),
            )
            | np.isin(times / sc.Tc_s, forbidden)
        )
        if np.any(np.isin(times[hit], result["report_t_s"])):
            raise ValueError("required report node collides with IC/BC membership")
        result[f"boundary_t_{level}_s"] = times[~hit]
    return result


def final_coordinate_arrays(
    axes: dict, case: dict, dtype: str = "float64"
) -> list[np.ndarray]:
    sc = Scales.from_case(case)
    out = []

    def scaled(x: np.ndarray, t: np.ndarray) -> np.ndarray:
        # physical_state casts SI coordinates before scaling. Match that order.
        numeric_type = np.dtype(dtype).type
        return np.column_stack(
            (
                x.ravel().astype(dtype) / numeric_type(sc.Lc_m),
                t.ravel().astype(dtype) / numeric_type(sc.Tc_s),
            )
        )

    for order in (3, 5):
        x, t = np.meshgrid(axes[f"gauss_x_{order}_m"].ravel(), axes["t_s"])
        out.append(scaled(x, t))
    for level in (0, 1):
        x, t = np.meshgrid(axes[f"mass_x_{level}_m"], axes["report_t_s"])
        out.append(scaled(x, t))
        t = axes[f"boundary_t_{level}_s"]
        for wall in case["domain_m"]:
            out.append(scaled(np.full(len(t), wall), t))
    return out


def partition_receipt(train: dict, val: dict, axes: dict, case: dict) -> dict:
    groups = {
        "training": [train[k] for k in COORD_KEYS],
        "validation": [val[k] for k in COORD_KEYS],
    }
    result = {}
    for dtype in ("float64", "float32"):
        groups["final_and_auxiliary"] = final_coordinate_arrays(axes, case, dtype)
        sets = {k: _rows(v, dtype) for k, v in groups.items()}
        overlap = {
            f"{a}_{b}": len(sets[a] & sets[b])
            for a, b in (
                ("training", "validation"),
                ("training", "final_and_auxiliary"),
                ("validation", "final_and_auxiliary"),
            )
        }
        if any(overlap.values()):
            raise ValueError(f"coordinate partition overlap at {dtype}: {overlap}")
        result[dtype] = {
            "unique_coordinate_counts": {k: len(v) for k, v in sets.items()},
            "pairwise_overlap": overlap,
        }
    return {
        "coordinates": "x/Lc,t/Tc",
        "membership": result,
        "training_shapes": {k: list(v.shape) for k, v in train.items()},
        "validation_shapes": {k: list(v.shape) for k, v in val.items()},
        "interior_labels": False,
        "auxiliary_boundary_nodes": [len(axes[f"boundary_t_{i}_s"]) for i in (0, 1)],
    }


def definitions() -> dict:
    return {
        "physics": "wet flat bed; fixed illustrative n=.03 s/m^(1/3); no mass source",
        "case_template": (
            "core E1 supplies geometry/IC/BC/scales; extension n "
            "overrides only source assumption"
        ),
        "loss": (
            "unchanged dimensionless core residual plus C q|q|/h^(7/3) "
            "momentum; C=g n² Lc/H^(4/3); fixed1/10/10"
        ),
        "selection": (
            "earliest strict minimum separate physics/IC/BC validation, "
            "initial plus every update; no final labels"
        ),
        "operator": (
            "reference FV averages conservatively projected to400commoncells; "
            "PINN h/q cell averages via3point Gauss-Legendre,5point sensitivity"
        ),
        "velocity": (
            "ratio of common-cell q/h, reference h>.001 fixed mask; "
            "not point velocity or average of point velocity"
        ),
        "regions": (
            "whole, numerical front band, outside; front and per-time "
            "uncertainty frozen from reference refinement, no analytical shock claim"
        ),
        "mass": (
            "midpoint h integrals800/1600, report0+.01:.01:.5; two-endpoint q "
            "trapezoid1001/2001 candidate nodes union reports minus membership "
            "collisions; M-5.5+integral(qR-qL)"
        ),
        "boundary": (
            "save both h/q states and individual left/right flux mismatches "
            "against reservoir h1/.1,q0; prevent cancellation"
        ),
        "pilot": (
            "one25update workload; persist neural query arrays but no neural "
            "error/mass quality screening; diagnostics use reference-only fixtures"
        ),
        "limits": (
            "one seed/case; refinement differences are measured uncertainty "
            "indicators, not rigorous bounds; no nonflat/dry/inverse/field validity"
        ),
    }


def model_identity(model: ShallowWaterPINN) -> str:
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def memory_observation() -> dict:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "process_lifetime_peak_rss_bytes": int(
            peak * (1 if platform.system() == "Darwin" else 1024)
        ),
        "host_load_1_5_15": list(os.getloadavg()),
        "limits": (
            "ru_maxrss lifetime maximum; no instantaneous RSS/system "
            "pressure/thermal measurement; load is host-wide"
        ),
    }


def validate_reference_packet(cfg: dict, root: Path) -> dict:
    """Check performed matrix identity and conservation before any neural pilot."""
    packet = root / cfg["reference_packet"]
    summary = _read(packet / "summary.json")
    procedure = _read(packet / "protocol.json")
    if (
        _read(packet / "status.json")["state"] != "completed"
        or summary["state"] != "complete_reference_with_measured_uncertainty"
        or summary["reference_is_ground_truth"] is not False
    ):
        raise ValueError("reference packet incomplete or unsupported")
    if (
        procedure["config_sha256"]
        != sha(root / "configs/extensions/friction_reference.json")
        or procedure["case_sha256"] != sha(root / cfg["case"])
        or procedure["source_sha256"] != sha(root / SOURCES[1])
        or summary["common_sha256"] != sha(packet / "common_reference.npz")
    ):
        raise ValueError("reference source/input/output integrity mismatch")
    expected = [(800, 0.45), (1600, 0.45), (3200, 0.45), (3200, 0.225)]
    if [(r["cells"], r["cfl"]) for r in summary["runs"]] != expected:
        raise ValueError("reference matrix differs from selected levels")
    for run in summary["runs"]:
        if sha(packet / run["id"] / "outputs.npz") != run["outputs_sha256"]:
            raise ValueError("reference native output changed")
        if not np.isfinite(
            [
                run["minimum_h_m"],
                run["max_mass_budget_defect_m2"],
                run["max_momentum_budget_defect_m3_s"],
                run["solve_wall_s"],
            ]
        ).all():
            raise ValueError("nonfinite reference diagnostic")
        if (
            run["minimum_h_m"] <= 0
            or run["max_mass_budget_defect_m2"] > 1e-10
            or run["max_momentum_budget_defect_m3_s"] > 1e-10
        ):
            raise ValueError("reference wet state or discrete budget unsupported")
    spatial_a, spatial_b, _ = summary["refinement"]
    for field in ("h_m", "q_m2_s"):
        if not spatial_b[field]["rmse"] < spatial_a[field]["rmse"]:
            raise ValueError("reference refinement differences fail to decrease")
    with np.load(packet / "common_reference.npz", allow_pickle=False) as arrays:
        if any(not np.isfinite(arrays[k]).all() for k in arrays.files) or np.any(
            arrays["h_m"] <= 0
        ):
            raise ValueError("nonfinite/nonpositive common reference")
    return {
        "state": "checked_reference_with_measured_uncertainty",
        "budget_roundoff_tolerance": 1e-10,
        "rationale": (
            "1e-10 absolute budget allowance is over500 times "
            "observed roundoff on this fixed O(1) SI case; no accuracy threshold"
        ),
        "uncertainty_h_m": summary["uncertainty_h_m"],
        "uncertainty_q_m2_s": summary["uncertainty_q_m2_s"],
    }


def freeze_protocol(
    cfg: dict,
    case: dict,
    training: dict,
    config_path: Path,
    run: Path,
    model: ShallowWaterPINN,
    root: Path,
) -> dict:
    reference_receipt = validate_reference_packet(cfg, root)
    train = generate_training_samples(case, training, cfg["sampling_seed"])
    val = generate_validation_samples(case, training, cfg["sampling_seed"])
    axes = final_memberships(cfg, case, train, val)
    partitions = partition_receipt(train, val, axes, case)
    save_samples(run / "training_samples.npz", train)
    save_samples(run / "validation_samples.npz", val)
    np.savez_compressed(run / "final_memberships.npz", **axes)
    torch.save(model.state_dict(), run / "initial_state.pt")
    reference_dir = root / cfg["reference_packet"]
    reference_hashes = {
        str(p.relative_to(root)): sha(p)
        for p in sorted(reference_dir.rglob("*"))
        if p.is_file()
    }
    source_evidence = root / "docs/evidence/phase_08/friction_v01/reference_sources"
    reference_hashes.update(
        {
            str(p.relative_to(root)): sha(p)
            for p in sorted(source_evidence.rglob("*"))
            if p.is_file()
        }
    )
    protocol = {
        "schema_version": 1,
        "state": "immutable_proposed_full_fit_not_released",
        "config": cfg,
        "case_template": case,
        "training_template": training,
        "physical_case": {
            "n_s_m_minus_one_third": 0.03,
            "bed_elevation_m": 0,
            "rainfall": 0,
            "initial_volume_m2_per_unit_width": 5.5,
        },
        "source_sha256": {p: sha(root / p) for p in SOURCES},
        "input_sha256": {
            str(config_path): sha(root / config_path),
            cfg["case"]: sha(root / cfg["case"]),
            cfg["training_template"]: sha(root / cfg["training_template"]),
            "configs/extensions/friction_reference.json": sha(
                root / "configs/extensions/friction_reference.json"
            ),
        },
        "reference_sha256": reference_hashes,
        "reference_gate": reference_receipt,
        "membership_sha256": {
            p: sha(run / p)
            for p in (
                "training_samples.npz",
                "validation_samples.npz",
                "final_memberships.npz",
                "initial_state.pt",
            )
        },
        "partitions": partitions,
        "initial_identity": model_identity(model),
        "diagnostics": definitions(),
        "full_fit_release": (
            "root receipt state=released, exact protocol_sha256 "
            "and10000updates; no inverse release"
        ),
    }
    protocol["protocol_sha256"] = digest(protocol)
    _write(run / "protocol.json", protocol)
    return protocol


def verify_protocol(path: Path, root: Path) -> dict:
    p = _read(path)
    if (
        digest({k: v for k, v in p.items() if k != "protocol_sha256"})
        != p["protocol_sha256"]
    ):
        raise ValueError("protocol content hash mismatch")
    for key in ("source_sha256", "input_sha256", "reference_sha256"):
        for name, expected in p[key].items():
            if sha(root / name) != expected:
                raise ValueError(f"protocol source/input/reference changed: {name}")
    for name, expected in p["membership_sha256"].items():
        if sha(path.parent / name) != expected:
            raise ValueError(f"protocol membership/initialization changed: {name}")
    return p


def losses(model: ShallowWaterPINN, samples: dict, cfg: dict) -> dict:
    result = friction_loss_components(
        model, samples, model.scales, cfg["manning_n_s_m_minus_one_third"]
    )
    result["objective"] = sum(
        cfg["weights"][k] * result[k] for k in ("pde", "ic", "bc")
    )
    if not all(bool(torch.isfinite(v)) for v in result.values()):
        raise RuntimeError("nonfinite friction objective")
    return result


def numbers(values: dict) -> dict:
    return {k: float(v.detach()) for k, v in values.items()}


def load_checkpoint(path: Path, protocol: dict) -> ShallowWaterPINN:
    cp = torch.load(path, map_location="cpu", weights_only=True)
    if (
        cp["protocol_sha256"] != protocol["protocol_sha256"]
        or cp["membership_sha256"] != protocol["membership_sha256"]
        or cp["initial_identity"] != protocol["initial_identity"]
    ):
        raise ValueError("checkpoint provenance mismatch")
    model = ShallowWaterPINN(
        protocol["case_template"],
        protocol["training_template"],
        device="cpu",
        dtype="float32",
    )
    model.load_state_dict(cp["model"])
    return model


@torch.no_grad()
def query(model: ShallowWaterPINN, points: np.ndarray, chunk: int) -> np.ndarray:
    result = np.empty((len(points), 2))
    for start in range(0, len(points), chunk):
        coords = torch.as_tensor(points[start : start + chunk], dtype=torch.float32)
        result[start : start + len(coords)] = model.physical_state(coords).cpu().numpy()
    if not np.isfinite(result).all() or np.any(result[:, 0] <= 0):
        raise RuntimeError("invalid PINN query states")
    return result


def query_grid(
    model: ShallowWaterPINN, x: np.ndarray, t: np.ndarray, chunk: int
) -> np.ndarray:
    result = np.empty((len(t), len(x), 2))
    for i, ti in enumerate(t):
        result[i] = query(model, np.column_stack((x, np.full(len(x), ti))), chunk)
    return result


def query_cell_averages(
    model: ShallowWaterPINN, axes: dict, order: int, chunk: int
) -> np.ndarray:
    x = axes[f"gauss_x_{order}_m"]
    values = query_grid(model, x.ravel(), axes["t_s"], chunk)
    return np.einsum(
        "tnkf,k->tnf",
        values.reshape(len(axes["t_s"]), len(x), order, 2),
        axes[f"gauss_w_{order}"],
    )


def mass_workload(
    model: ShallowWaterPINN, axes: dict, cfg: dict, directory: Path, level: int
) -> dict:
    start = time.perf_counter()
    x, t = axes[f"mass_x_{level}_m"], axes["report_t_s"]
    values = query_grid(model, x, t, cfg["evaluation"]["query_chunk_size"])
    volume = values[..., 0].mean(1) * 10
    volume_s = time.perf_counter() - start
    start = time.perf_counter()
    np.savez_compressed(
        directory / f"volume_{level}.npz",
        x_m=x,
        t_s=t,
        h_m=values[..., 0],
        q_m2_s=values[..., 1],
    )
    volume_io = time.perf_counter() - start
    start = time.perf_counter()
    bt = axes[f"boundary_t_{level}_s"]
    walls = query_grid(
        model, np.array([-5.0, 5.0]), bt, cfg["evaluation"]["query_chunk_size"]
    )
    integral = cumulative_trapezoid(walls[:, 1, 1] - walls[:, 0, 1], bt, initial=0)
    idx = np.searchsorted(bt, t)
    if not np.array_equal(bt[idx], t):
        raise ValueError("report nodes missing from boundary quadrature")
    corrected = volume - 5.5 + integral[idx]
    edge_s = time.perf_counter() - start
    start = time.perf_counter()
    np.savez_compressed(
        directory / f"boundaries_{level}.npz",
        x_m=np.array([-5.0, 5.0]),
        t_s=bt,
        h_m=walls[..., 0],
        q_m2_s=walls[..., 1],
        outward_integral_m2=integral,
    )
    edge_io = time.perf_counter() - start
    return {
        "volume_m2": volume.tolist(),
        "report_t_s": t.tolist(),
        "outward_integral_m2": integral[idx].tolist(),
        "prescribed_initial_volume_m2": 5.5,
        "predicted_initial_bias_m2": float(volume[0] - 5.5),
        "corrected_balance_m2": corrected.tolist(),
        "balance_without_initial_bias_m2": (corrected - (volume[0] - 5.5)).tolist(),
        "sampled_max_abs_corrected_m2": float(np.max(np.abs(corrected))),
        "boundary": {
            name: {
                "max_abs_h_mismatch_m": float(np.max(np.abs(walls[:, i, 0] - target))),
                "max_abs_q_mismatch_m2_s": float(np.max(np.abs(walls[:, i, 1]))),
                "absolute_flux_mismatch_integral_m2": float(
                    np.trapezoid(np.abs(walls[:, i, 1]), bt)
                ),
                "signed_outward_integral_m2": float(
                    np.trapezoid(walls[:, i, 1] * (-1 if i == 0 else 1), bt)
                ),
            }
            for i, (name, target) in enumerate((("left", 1.0), ("right", 0.1)))
        },
        "cost_s": {
            "volume_query": volume_s,
            "volume_io": volume_io,
            "boundary_query": edge_s,
            "boundary_io": edge_io,
        },
        "query_points": len(x) * len(t) + len(bt) * 2,
    }


def metrics(pred: np.ndarray, ref: np.ndarray) -> dict:
    if not pred.size:
        return {
            "count": 0,
            "rmse": None,
            "relative_l2_pct": None,
            "reason": "empty region",
        }
    norm = np.linalg.norm(ref.ravel())
    return {
        "count": int(pred.size),
        "rmse": float(np.sqrt(np.mean((pred - ref) ** 2))),
        "relative_l2_pct": float(100 * np.linalg.norm((pred - ref).ravel()) / norm)
        if norm
        else None,
        "relative_reason": None if norm else "zero reference norm",
    }


def field_diagnostics(fields: np.ndarray, ref: dict, axes: dict) -> dict:
    h, q = ref["h_m"], ref["q_m2_s"]
    wet = h > 0.001
    band = (
        np.abs(axes["x_m"][None, :] - ref["front_x_m"][:, None])
        <= ref["front_half_width_m"][:, None]
    )
    variables = {
        "h_m": (fields[..., 0], h),
        "q_m2_s": (fields[..., 1], q),
        "cell_ratio_velocity_m_s": (fields[..., 1] / fields[..., 0], q / h),
    }
    regions = {
        "whole": np.ones_like(wet),
        "numerical_front_band": band,
        "outside_band": ~band,
    }

    def region_report(mask: np.ndarray) -> dict:
        return {
            name: metrics(
                a[mask & wet] if name == "cell_ratio_velocity_m_s" else a[mask],
                b[mask & wet] if name == "cell_ratio_velocity_m_s" else b[mask],
            )
            for name, (a, b) in variables.items()
        }

    # Keep per-time diagnostics through the same implementation; masks isolate one row.
    rows = []
    for i, t in enumerate(axes["t_s"]):
        row_regions = {}
        for name, mask in regions.items():
            select = np.zeros_like(mask)
            select[i] = mask[i]
            row_regions[name] = region_report(select)
        rows.append({"t_s": float(t), "regions": row_regions})
    return {
        "aggregate": {k: region_report(v) for k, v in regions.items()},
        "per_time": rows,
        "reference_wet_count": int(wet.sum()),
        "reference_wet_coverage": float(wet.mean()),
        "definition": definitions(),
    }


def workload_queries(
    model: ShallowWaterPINN,
    protocol: dict,
    run: Path,
    directory: Path,
    root: Path,
    *,
    neural_errors: bool = False,
) -> dict:
    start = time.perf_counter()
    directory.mkdir(exist_ok=False)
    cfg = protocol["config"]
    with np.load(run / "final_memberships.npz", allow_pickle=False) as z:
        axes = dict(z)
    now = time.perf_counter()
    with np.load(
        root / cfg["reference_packet"] / "common_reference.npz", allow_pickle=False
    ) as z:
        ref = dict(z)
    if not np.allclose(
        ref["t_s"], axes["t_s"], atol=1e-15, rtol=0
    ) or not np.array_equal(ref["cell_edges_m"], axes["cell_edges_m"]):
        raise ValueError("reference/common-cell axes mismatch")
    costs = {"reference_packet_load": time.perf_counter() - now}
    fields = []
    for order in (3, 5):
        now = time.perf_counter()
        f = query_cell_averages(
            model, axes, order, cfg["evaluation"]["query_chunk_size"]
        )
        costs[f"pinn_gauss{order}_query"] = time.perf_counter() - now
        now = time.perf_counter()
        np.savez_compressed(
            directory / f"common_cell_gauss{order}.npz",
            x_m=axes["x_m"],
            cell_edges_m=axes["cell_edges_m"],
            t_s=axes["t_s"],
            h_m=f[..., 0],
            q_m2_s=f[..., 1],
        )
        costs[f"pinn_gauss{order}_io"] = time.perf_counter() - now
        fields.append(f)
    now = time.perf_counter()
    if neural_errors:
        _write(directory / "errors.json", field_diagnostics(fields[0], ref, axes))
        _write(
            directory / "pinn_cell_quadrature_sensitivity.json",
            {
                "h_max_abs_m": float(
                    np.max(np.abs(fields[0][..., 0] - fields[1][..., 0]))
                ),
                "q_max_abs_m2_s": float(
                    np.max(np.abs(fields[0][..., 1] - fields[1][..., 1]))
                ),
                "gauss5_errors": field_diagnostics(fields[1], ref, axes),
                "limits": "3-to5point sensitivity is not a proven error bound",
            },
        )
    else:
        fixture = np.stack((ref["h_m"], ref["q_m2_s"]), axis=-1)
        _write(
            directory / "reference_only_diagnostic_fixture.json",
            field_diagnostics(fixture, ref, axes),
        )
        _write(
            directory / "reference_only_quadrature_output_fixture.json",
            {
                "h_max_abs_m": 0.0,
                "q_max_abs_m2_s": 0.0,
                "gauss5_errors": field_diagnostics(fixture, ref, axes),
                "limits": "reference-only JSON shape fixture; no pilot errors",
            },
        )
    _write(
        directory / "numerical_reference_uncertainty.json",
        {
            "definition": (
                "cellwise abs(spatial3200-spatial1600) plus "
                "abs(CFL.225-CFL.45); measured differences, no rigorous bound"
            ),
            "h_rmse_indicator_m": float(
                np.sqrt(np.mean(ref["h_measured_uncertainty_m"] ** 2))
            ),
            "q_rmse_indicator_m2_s": float(
                np.sqrt(np.mean(ref["q_measured_uncertainty_m2_s"] ** 2))
            ),
            "h_max_indicator_m": float(ref["h_measured_uncertainty_m"].max()),
            "q_max_indicator_m2_s": float(ref["q_measured_uncertainty_m2_s"].max()),
            "front_x_m": ref["front_x_m"].tolist(),
            "front_uncertainty_m": ref["front_uncertainty_m"].tolist(),
            "front_half_width_m": ref["front_half_width_m"].tolist(),
        },
    )
    costs[
        "diagnostics_and_json"
        if neural_errors
        else "reference_only_diagnostic_fixture_and_json"
    ] = time.perf_counter() - now
    reports = [mass_workload(model, axes, cfg, directory, i) for i in (0, 1)]
    costs["mass_queries_io_and_calculation"] = sum(
        sum(r["cost_s"].values()) for r in reports
    )
    now = time.perf_counter()
    if neural_errors:
        _write(
            directory / "mass.json",
            {
                "base": reports[0],
                "refined": reports[1],
                "combined_refinement_sensitivity": {
                    k: float(
                        np.max(
                            np.abs(np.array(reports[0][k]) - np.array(reports[1][k]))
                        )
                    )
                    for k in (
                        "volume_m2",
                        "outward_integral_m2",
                        "corrected_balance_m2",
                    )
                },
                "limits": (
                    "midpoint space/trapezoid time difference, "
                    "not rigorous bound; sampled extrema"
                ),
            },
        )
    else:
        # Match summary serialization cost while withholding pilot quality values.
        def shape_fixture(value):
            if isinstance(value, dict):
                return {k: shape_fixture(v) for k, v in value.items()}
            if isinstance(value, list):
                return [shape_fixture(v) for v in value]
            return 0.0 if isinstance(value, (float, int)) else value

        _write(
            directory / "mass_summary_shape_fixture.json",
            {
                "base": shape_fixture(reports[0]),
                "refined": shape_fixture(reports[1]),
                "label": "shape-only fixture; no pilot mass quality reported",
            },
        )
    costs["mass_summary_json"] = time.perf_counter() - now
    receipt = {
        "neural_errors_computed": neural_errors,
        "mass_quality_reported": neural_errors,
        "cell_query_points": 400 * 50 * (3 + 5),
        "mass_and_boundary_query_points": sum(r["query_points"] for r in reports),
        "cost_s": costs,
        "memory": memory_observation(),
        "output_bytes_before_receipt": sum(
            p.stat().st_size for p in directory.iterdir()
        ),
        "wall_s": time.perf_counter() - start,
    }
    _write(directory / "query_receipt.json", receipt)
    return receipt


def verify_saved_training(run: Path, summary: dict, protocol: dict) -> dict:
    for name, key in (
        ("history.csv", "history_sha256"),
        ("checkpoint.pt", "checkpoint_sha256"),
    ):
        if sha(run / name) != summary[key]:
            raise ValueError(f"saved training changed: {name}")
    with (run / "history.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    steps = summary["actual_updates"]
    if [int(r["step"]) for r in rows] != list(range(steps + 1)):
        raise ValueError("history must include initial and every update")
    best = float("inf")
    best_step = None
    selected_count = 0
    for row in rows:
        value = float(row["val_objective"])
        if not np.isfinite(value):
            raise ValueError("nonfinite saved validation")
        selected = value < best
        if row["selected"] != str(selected):
            raise ValueError("earliest strict minimum violated")
        if selected:
            best = value
            best_step = int(row["step"])
            selected_count += 1
    if (
        summary["selected_step"] != best_step
        or summary["selected_validation_score"] != best
    ):
        raise ValueError("selected summary/history mismatch")
    if summary["checkpoint_write_counts"] != {
        "initial": 1,
        "selected": selected_count,
        "latest": steps + 1,
    }:
        raise ValueError("checkpoint count mismatch")
    for name, step in (
        ("initial.pt", 0),
        ("checkpoint.pt", best_step),
        ("latest.pt", steps),
    ):
        cp = torch.load(run / name, map_location="cpu", weights_only=True)
        checkpoint_model = load_checkpoint(run / name, protocol)
        if cp["step"] != step or cp["validation_score"] != float(
            rows[step]["val_objective"]
        ):
            raise ValueError(f"checkpoint step/score mismatch: {name}")
        expected_opt_steps = {step} if step else set()
        observed = {int(v["step"].item()) for v in cp["optimizer"]["state"].values()}
        if observed != expected_opt_steps:
            raise ValueError("saved optimizer update count mismatch")
        groups = cp["optimizer"]["param_groups"]
        params = list(checkpoint_model.parameters())
        ids = [i for group in groups for i in group["params"]]
        states = cp["optimizer"]["state"]
        if (
            len(groups) != 1
            or len(ids) != len(params)
            or len(set(ids)) != len(params)
            or set(states) != (set(ids) if step else set())
            or groups[0]["lr"] != 0.001
        ):
            raise ValueError("saved optimizer parameter membership mismatch")
        for index, param in zip(ids, params, strict=True):
            if not step:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                moment = states[index][key]
                if moment.shape != param.shape or not bool(
                    torch.isfinite(moment).all()
                ):
                    raise ValueError("saved optimizer moment shape/finite mismatch")
    if (
        model_identity(load_checkpoint(run / "initial.pt", protocol))
        != protocol["initial_identity"]
    ):
        raise ValueError("initial identity mismatch")
    if summary["mode"] == "full":
        release = _read(run / "root_release.json")
        if (
            release.get("state") != "released"
            or release.get("protocol_sha256") != protocol["protocol_sha256"]
            or steps != protocol["config"]["full_steps"]
        ):
            raise ValueError("saved full release mismatch")
    return {
        "history_rows": len(rows),
        "earliest_strict_minimum_step": best_step,
        "initial_and_all_optimizer_counts_verified": True,
    }


def run_friction(
    config_path: str | Path,
    manifest_path: str | Path,
    *,
    run_id: str,
    root: Path = Path("."),
    protocol_path: Path | None = None,
    release_path: Path | None = None,
) -> Path:
    """Execute exactly one25step pilot; full10000updates require hash-bound release."""
    overall = time.perf_counter()
    root = root.resolve()
    config_path = Path(config_path)
    manifest_path = root / manifest_path
    validate_run_id(run_id)
    if not run_id.startswith("p08_friction_"):
        raise ValueError("fresh p08_friction_ prefix required")
    if manifest_path.exists():
        raise ValueError("manifest exists; no overwrite")
    cfg, case, training = load_friction_config(config_path, root)
    full = release_path is not None
    if full:
        if protocol_path is None:
            raise ValueError("full fit requires prior frozen protocol")
        protocol = verify_protocol(root / protocol_path, root)
        release = _read(root / release_path)
        if (
            release.get("state") != "released"
            or release.get("protocol_sha256") != protocol["protocol_sha256"]
            or protocol["config"] != cfg
        ):
            raise ValueError("root technical release/config mismatch")
    elif protocol_path is not None:
        raise ValueError("pilot freezes its own new protocol")
    torch.set_num_threads(4)
    torch.manual_seed(11)
    model = ShallowWaterPINN(case, training, device="cpu", dtype="float32")
    resolved = {
        "case": case,
        "training": training,
        "extension": cfg,
        "physical_source_override": {"n": 0.03, "bed_m": 0, "rainfall": 0},
        "request": {"run_id": run_id, "device": "cpu", "dtype": "float32", "seed": 11},
    }
    manifest = {
        "state": "running",
        "mode": "full" if full else "pilot",
        "run": f"runs/{run_id}",
    }
    _write(manifest_path, manifest)
    try:
        with RunRecord.create(
            root / "runs", run_id, resolved, project_root=root
        ) as record:
            run = record.path
            if full:
                for name in (*protocol["membership_sha256"], "protocol.json"):
                    shutil.copyfile((root / protocol_path).parent / name, run / name)
                _write(run / "root_release.json", release)
                model.load_state_dict(
                    torch.load(run / "initial_state.pt", weights_only=True)
                )
            else:
                protocol = freeze_protocol(
                    cfg, case, training, config_path, run, model, root
                )
            if model_identity(model) != protocol["initial_identity"]:
                raise ValueError("initialization differs")
            train, val = (
                load_samples(run / "training_samples.npz"),
                load_samples(run / "validation_samples.npz"),
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["adam_lr"])
            best = float("inf")
            selected_step = 0
            saves = {"initial": 0, "selected": 0, "latest": 0}
            timings = []
            observations = [{"stage": "before_training", **memory_observation()}]
            steps = cfg["full_steps"] if full else cfg["pilot_steps"]
            started = time.perf_counter()
            with (run / "history.csv").open("x", newline="") as f:
                timing_keys = (
                    "objective_s",
                    "backward_s",
                    "optimizer_s",
                    "validation_s",
                    "checkpoint_io_s",
                    "full_step_s",
                )
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "step",
                        *[f"train_{k}" for k in ("pde", "ic", "bc", "objective")],
                        *[f"val_{k}" for k in ("pde", "ic", "bc", "objective")],
                        "selected",
                        *timing_keys,
                    ],
                )
                writer.writeheader()
                for step in range(steps + 1):
                    st = time.perf_counter()
                    times = {k: 0.0 for k in timing_keys}
                    training_values = {}
                    if step:
                        optimizer.zero_grad(set_to_none=True)
                        now = time.perf_counter()
                        c = losses(model, train, cfg)
                        training_values = numbers(c)
                        times["objective_s"] = time.perf_counter() - now
                        now = time.perf_counter()
                        c["objective"].backward()
                        if not all(
                            p.grad is not None and bool(torch.isfinite(p.grad).all())
                            for p in model.parameters()
                        ):
                            raise RuntimeError("nonfinite/absent parameter gradient")
                        times["backward_s"] = time.perf_counter() - now
                        now = time.perf_counter()
                        optimizer.step()
                        times["optimizer_s"] = time.perf_counter() - now
                    now = time.perf_counter()
                    v = numbers(losses(model, val, cfg))
                    times["validation_s"] = time.perf_counter() - now
                    score = v["objective"]
                    selected = score < best
                    now = time.perf_counter()
                    cp = {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "step": step,
                        "validation_score": score,
                        "protocol_sha256": protocol["protocol_sha256"],
                        "membership_sha256": protocol["membership_sha256"],
                        "initial_identity": protocol["initial_identity"],
                        "torch_rng_state": torch.get_rng_state(),
                    }
                    if step == 0:
                        torch.save(cp, run / "initial.pt")
                        saves["initial"] += 1
                    if selected:
                        best = score
                        selected_step = step
                        torch.save(cp, run / "checkpoint.pt")
                        saves["selected"] += 1
                    torch.save(cp, run / "latest.pt")
                    saves["latest"] += 1
                    times["checkpoint_io_s"] = time.perf_counter() - now
                    times["full_step_s"] = time.perf_counter() - st
                    writer.writerow(
                        {
                            "step": step,
                            **{f"train_{k}": a for k, a in training_values.items()},
                            **{f"val_{k}": a for k, a in v.items()},
                            "selected": selected,
                            **times,
                        }
                    )
                    f.flush()
                    if step:
                        timings.append(times)
                    if step in (0, steps):
                        observations.append(
                            {"stage": f"step_{step}", **memory_observation()}
                        )
            train_wall = time.perf_counter() - started
            now = time.perf_counter()
            selected_model = load_checkpoint(run / "checkpoint.pt", protocol)
            latest_model = load_checkpoint(run / "latest.pt", protocol)
            probe = torch.as_tensor(val["initial_xt_nd"][:16], dtype=torch.float32)
            with torch.no_grad():
                if not torch.equal(model(probe), latest_model(probe)):
                    raise RuntimeError("latest reload mismatch")
            if numbers(losses(selected_model, val, cfg))["objective"] != best:
                raise RuntimeError("selected score reload mismatch")
            reload_s = time.perf_counter() - now
            summary = {
                "state": "completed",
                "mode": "full" if full else "pilot",
                "actual_updates": steps,
                "validation_evaluations": steps + 1,
                "checkpoint_write_counts": saves,
                "selected_step": selected_step,
                "selected_validation_score": best,
                "initial_identity": protocol["initial_identity"],
                "protocol_sha256": protocol["protocol_sha256"],
                "checkpoint_sha256": sha(run / "checkpoint.pt"),
                "history_sha256": sha(run / "history.csv"),
                "training_wall_s": train_wall,
                "reload_s": reload_s,
                "reload_score_exact": True,
                "latest_reload_prediction_exact": True,
                "raw_step_costs_s": timings,
                "memory_observations": observations,
                "neural_reference_errors_computed": False,
            }
            summary["saved_training_readback"] = verify_saved_training(
                run, summary, protocol
            )
            if not full:
                setup_s = time.perf_counter() - overall - train_wall
                query_report = workload_queries(
                    selected_model, protocol, run, run / "pilot_query_workload", root
                )
                summary["query_workload"] = query_report
                raw = np.array([t["full_step_s"] for t in timings])
                reference_summary = _read(
                    root / cfg["reference_packet"] / "summary.json"
                )
                reference_cost = reference_summary["wall_s_including_output"]
                mean = float(raw.mean())
                slow = float(raw.max())
                summary["projection"] = {
                    "full_updates": 10000,
                    "step_mean_s": mean,
                    "step_median_s": float(np.median(raw)),
                    "step_min_max_s": [float(raw.min()), slow],
                    "training_mean_projection_s": mean * 10000,
                    "training_scaled_wall_projection_s": train_wall / steps * 10000,
                    "setup_reload_and_readback_s": setup_s,
                    "full_query_workload_s": query_report["wall_s"],
                    "complete_reference_and_refinement_packet_s": reference_cost,
                    "complete_matrix_mean_projection_s": mean * 10000
                    + setup_s
                    + query_report["wall_s"]
                    + reference_cost,
                    "complete_matrix_slowest_observed_step_projection_s": slow * 10000
                    + setup_s
                    + query_report["wall_s"]
                    + reference_cost,
                    "limits": (
                        "one25step pilot; thermal/load/checkpoint frequency may "
                        "differ over10000updates. Reference generation measured "
                        "once; diagnostic JSON cost uses reference-only fixture. "
                        "Later figures/archive unmeasured."
                    ),
                }
                summary["full_fit_state"] = "held_pending_root_technical_release"
            summary["wall_before_summary_s"] = time.perf_counter() - overall
            _write(run / "summary.json", summary)
            record.complete(
                stopping_reason="declared10000updates"
                if full
                else "single25update feasibility pilot",
                iterations=steps,
                function_evaluations=2 * steps + 2,
                elapsed_timings_s={
                    "training": train_wall,
                    "checkpoint_reload": reload_s,
                },
            )
        manifest.update(
            state="completed",
            protocol=f"runs/{run_id}/protocol.json",
            protocol_sha256=protocol["protocol_sha256"],
            summary=f"runs/{run_id}/summary.json",
        )
        _write(manifest_path, manifest, overwrite=True)
    except BaseException as exc:
        manifest.update(
            state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            reason=f"{type(exc).__name__}: {exc}",
        )
        _write(manifest_path, manifest, overwrite=True)
        raise
    return root / "runs" / run_id


def evaluate_friction(
    manifest_path: str | Path, output_dir: str | Path, *, root: Path = Path(".")
) -> Path:
    root = root.resolve()
    manifest = _read(root / manifest_path)
    run = root / manifest["run"]
    summary = _read(run / "summary.json")
    if (
        manifest["state"] != "completed"
        or summary["mode"] != "full"
        or summary["actual_updates"] != 10000
        or _read(run / "status.json")["state"] != "completed"
    ):
        raise ValueError(
            "final accuracy requires released completed full10000update run; "
            "pilot refused"
        )
    protocol = verify_protocol(run / "protocol.json", root)
    verify_saved_training(run, summary, protocol)
    if manifest["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError("manifest/protocol mismatch")
    torch.set_num_threads(4)
    output = root / output_dir
    receipt = workload_queries(
        load_checkpoint(run / "checkpoint.pt", protocol),
        protocol,
        run,
        output,
        root,
        neural_errors=True,
    )
    _write(
        output / "provenance.json",
        {
            "run": manifest["run"],
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "protocol_sha256": protocol["protocol_sha256"],
            "reference_sha256": protocol["reference_sha256"],
            "comparison_operator": definitions()["operator"],
            "costs": receipt,
            "scope": (
                "one fixed-positive-n wet flat E1; reference has measured "
                "numerical uncertainty"
            ),
        },
    )
    return output
