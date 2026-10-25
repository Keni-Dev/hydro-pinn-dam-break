"""Additive planar E1 experiment; a pilot cannot establish solution accuracy.

The sole CLI route runs the bounded pilot. Full training requires a hash-bound
root release passed to ``run_planar``; evaluation requires a terminal full run.
All scalar grids are (time,y,x) point values. Edges define midpoint quadrature
cells, not cell averages. Existing 1D records and scientific modules are unchanged.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import cumulative_trapezoid

from hydro_pinn.config import load_config
from hydro_pinn.extensions.model_2d import PlanarPINN
from hydro_pinn.extensions.physics_2d import residuals
from hydro_pinn.physics import Scales
from hydro_pinn.records import RunRecord, validate_run_id, write_json_atomic
from hydro_pinn.references import evaluate_reference, wave_diagnostics

EDGES = ("left", "right", "bottom", "top")
LOSS_KEYS = (
    "pde_mass",
    "pde_x_momentum",
    "pde_y_momentum",
    "ic_h",
    "ic_qx",
    "ic_qy",
    "bc_left",
    "bc_right",
    "bc_bottom",
    "bc_top",
)
SOURCES = (
    "src/hydro_pinn/extensions/physics_2d.py",
    "src/hydro_pinn/extensions/model_2d.py",
    "src/hydro_pinn/extensions/planar_2d.py",
    "src/hydro_pinn/cli.py",
    "src/hydro_pinn/physics.py",
    "src/hydro_pinn/references.py",
    "src/hydro_pinn/config.py",
    "src/hydro_pinn/records.py",
    "pyproject.toml",
    "uv.lock",
)


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


def _write(path: Path, data: dict, *, overwrite: bool = False) -> None:
    write_json_atomic(path, data, overwrite=overwrite)


def load_planar_config(path: str | Path, root: Path = Path(".")) -> tuple[dict, dict]:
    cfg = _read(root / path)
    # This version is one declared experiment, not a hyperparameter search API.
    expected = {
        "schema_version": 1,
        "extension": "planar_e1_consistency_v01",
        "case": "configs/cases/e1_wet.json",
        "domain_y_m": [-5.0, 5.0],
        "seed": 11,
        "device": "cpu",
        "dtype": "float32",
        "cpu_threads": 4,
        "hidden_layers": [64] * 4,
        "depth_floor_nd": 1e-6,
        "adam_lr": 0.001,
        "full_steps": 10000,
        "pilot_steps": 25,
        "weights": {"pde": 1.0, "ic": 10.0, "bc": 10.0},
        "sampling": {
            "algorithm": "numpy_PCG64_fixed_uniform",
            "training": {"seed": 110101, "pde": 10000, "ic": 1000, "per_edge": 500},
            "validation": {"seed": 110102, "pde": 2000, "ic": 400, "per_edge": 200},
        },
        "evaluation": {
            "nx": 400,
            "ny": 21,
            "nt": 50,
            "first_time_s": 0.01,
            "last_time_s": 0.5,
            "wet_threshold_m": 0.001,
            "shock_band_half_width_m": 0.1,
            "volume_nx": [800, 1600],
            "volume_ny": [21, 42],
            "edge_nx": [400, 800],
            "edge_ny": [21, 42],
            "edge_time_nodes": [1001, 2001],
            "query_chunk_size": 8192,
        },
    }
    if cfg != expected:
        raise ValueError("configuration differs from the declared planar v01 protocol")
    case = load_config(root / cfg["case"])
    if case["case_id"] != "E1":
        raise ValueError("planar consistency requires verified wet E1")
    wave_diagnostics(case)
    return cfg, case


def samples(cfg: dict, case: dict, partition: str) -> dict[str, np.ndarray]:
    """Independent PCG64 streams; no interior solution labels or y-wall h/qx."""
    spec = cfg["sampling"][partition]
    rng = np.random.Generator(np.random.PCG64(spec["seed"]))
    sc = Scales.from_case(case)
    upper_t = 0.5 / sc.Tc_s

    def points(count: int) -> np.ndarray:
        a = rng.uniform([-1.0, -1.0, 0.0], [1.0, 1.0, upper_t], (count, 3))
        a[:, 2] = np.maximum(a[:, 2], np.nextafter(0.0, 1.0))
        return a

    pde, ic = points(spec["pde"]), points(spec["ic"])
    ic[:, 2] = 0.0
    if np.any(ic[:, 0] == 0):
        raise ValueError("IC sampler hit dam discontinuity")
    targets = np.zeros((len(ic), 3))
    targets[:, 0] = np.where(ic[:, 0] < 0, 1.0, 0.1)
    out = {"pde_xyt_nd": pde, "ic_xyt_nd": ic, "ic_state_nd": targets}
    for edge in EDGES:
        a = points(spec["per_edge"])
        a[:, 0 if edge in EDGES[:2] else 1] = (
            -1.0 if edge in ("left", "bottom") else 1.0
        )
        target = np.zeros((len(a), 3))
        mask = np.array([1.0, 1.0, 1.0] if edge in EDGES[:2] else [0.0, 0.0, 1.0])
        if edge in EDGES[:2]:
            target[:, 0] = 1.0 if edge == "left" else 0.1
        out[f"{edge}_xyt_nd"] = a
        out[f"{edge}_state_nd"] = target
        out[f"{edge}_mask"] = mask
    return out


def partition_receipt(train: dict, val: dict) -> dict:
    coordinates = [k for k in train if k.endswith("xyt_nd")]
    train_rows = {tuple(row) for key in coordinates for row in train[key]}
    val_rows = {tuple(row) for key in coordinates for row in val[key]}
    overlap = len(train_rows & val_rows)
    if overlap:
        raise ValueError("training/validation coordinate overlap")
    return {
        "coordinate_order": ["x/Lc", "y/Lc", "t/Tc"],
        "overlap_exact_float64_rows": overlap,
        "counts": {
            p: {k: list(v.shape) for k, v in a.items()}
            for p, a in (("training", train), ("validation", val))
        },
        "interior_labels": False,
        "lateral_boundary_supervision": "qy only; h/qx mask zero",
    }


def tensor_samples(model: PlanarPINN, data: dict) -> dict:
    dtype = next(model.parameters()).dtype
    return {k: torch.as_tensor(v, dtype=dtype) for k, v in data.items()}


def losses(model: PlanarPINN, data: dict) -> dict[str, torch.Tensor]:
    """Means of sums over equations; BC is an equal-point four-edge mean."""
    r = residuals(model, data["pde_xyt_nd"])
    initial = model(data["ic_xyt_nd"]) - data["ic_state_nd"]
    result = {key: r[:, i].square().mean() for i, key in enumerate(LOSS_KEYS[:3])}
    result.update(
        {key: initial[:, i].square().mean() for i, key in enumerate(LOSS_KEYS[3:6])}
    )
    for edge in EDGES:
        diff = model(data[f"{edge}_xyt_nd"]) - data[f"{edge}_state_nd"]
        result[f"bc_{edge}"] = (diff * data[f"{edge}_mask"]).square().sum(1).mean()
    result["pde"] = sum(result[k] for k in LOSS_KEYS[:3])
    result["ic"] = sum(result[k] for k in LOSS_KEYS[3:6])
    counts = [len(data[f"{e}_xyt_nd"]) for e in EDGES]
    result["bc"] = sum(
        result[f"bc_{e}"] * n for e, n in zip(EDGES, counts, strict=True)
    ) / sum(counts)
    result["objective"] = result["pde"] + 10 * result["ic"] + 10 * result["bc"]
    if not all(bool(torch.isfinite(v)) for v in result.values()):
        raise RuntimeError("nonfinite objective component")
    return result


def _numbers(values: dict) -> dict:
    return {key: float(value.detach()) for key, value in values.items()}


def _identity(model: PlanarPINN) -> str:
    h = hashlib.sha256()
    for key, value in model.state_dict().items():
        h.update(key.encode())
        h.update(value.detach().numpy().tobytes())
    return h.hexdigest()


def memory_observation() -> dict:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "process_lifetime_peak_rss_bytes": int(
            peak * (1 if platform.system() == "Darwin" else 1024)
        ),
        "host_load_1_5_15": list(os.getloadavg()),
        "limitations": (
            "ru_maxrss is process lifetime maximum; no instantaneous or "
            "system-pressure measurement. CPU host load is not per-"
            "process utilization."
        ),
    }


def centres(low: float, high: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(low, high, count + 1)
    return (edges[:-1] + edges[1:]) / 2, edges


def final_axes(cfg: dict) -> dict:
    e = cfg["evaluation"]
    x, xe = centres(-5.0, 5.0, e["nx"])
    y, ye = centres(-5.0, 5.0, e["ny"])
    return {
        "x_m": x,
        "y_m": y,
        "t_s": np.linspace(e["first_time_s"], e["last_time_s"], e["nt"]),
        "x_edges_m": xe,
        "y_edges_m": ye,
    }


@torch.no_grad()
def query_points(model: PlanarPINN, points: np.ndarray, chunk_size: int) -> np.ndarray:
    dtype = next(model.parameters()).dtype
    answer = np.empty((len(points), 3), dtype=np.float64)
    for start in range(0, len(points), chunk_size):
        c = torch.as_tensor(points[start : start + chunk_size], dtype=dtype)
        answer[start : start + len(c)] = model.physical_state(c).numpy()
    if not np.isfinite(answer).all() or np.any(answer[:, 0] <= 0):
        raise RuntimeError("invalid queried state")
    return answer


def query_grid(
    model: PlanarPINN, x: np.ndarray, y: np.ndarray, t: np.ndarray, chunk_size: int
) -> np.ndarray:
    """Stream one time slab, then bounded network chunks; output(time,y,x,3)."""
    yy, xx = np.meshgrid(y, x, indexing="ij")
    result = np.empty((len(t), len(y), len(x), 3), dtype=np.float64)
    for i, ti in enumerate(t):
        points = np.column_stack((xx.ravel(), yy.ravel(), np.full(xx.size, ti)))
        result[i] = query_points(model, points, chunk_size).reshape(len(y), len(x), 3)
    return result


def reference_fields(case: dict, axes: dict) -> dict:
    ref = evaluate_reference(case, axes["x_m"], axes["t_s"])
    shape = (len(axes["t_s"]), len(axes["y_m"]), len(axes["x_m"]))
    return {
        "h_m": np.broadcast_to(ref.h_m[:, None, :], shape).copy(),
        "qx_m2_s": np.broadcast_to(ref.q_m2_s[:, None, :], shape).copy(),
        "qy_m2_s": np.zeros(shape),
        "reference_wet_mask": np.broadcast_to(
            ref.h_m[:, None, :] > 0.001, shape
        ).copy(),
    }


def four_edge_rates(states: dict, x: np.ndarray, y: np.ndarray) -> dict:
    """SI outward rates: left/right use qx, bottom/top use qy."""
    return {
        "left": -np.mean(states["left"][..., 1], axis=1) * 10.0,
        "right": np.mean(states["right"][..., 1], axis=1) * 10.0,
        "bottom": -np.mean(states["bottom"][..., 2], axis=1) * 10.0,
        "top": np.mean(states["top"][..., 2], axis=1) * 10.0,
    }


def mass_queries(model: PlanarPINN, cfg: dict, directory: Path, level: int) -> dict:
    """Save raw area and all four edge fields with the exact quadrature axes."""
    e = cfg["evaluation"]
    x, xe = centres(-5.0, 5.0, e["volume_nx"][level])
    y, ye = centres(-5.0, 5.0, e["volume_ny"][level])
    report_t = np.r_[0.0, final_axes(cfg)["t_s"]]
    started = time.perf_counter()
    field = query_grid(model, x, y, report_t, e["query_chunk_size"])
    volume = field[..., 0].mean(axis=(1, 2)) * 100.0
    query_volume_s = time.perf_counter() - started
    started = time.perf_counter()
    np.savez_compressed(
        directory / f"volume_{level}.npz",
        x_m=x,
        y_m=y,
        t_s=report_t,
        x_edges_m=xe,
        y_edges_m=ye,
        h_m=field[..., 0],
        qx_m2_s=field[..., 1],
        qy_m2_s=field[..., 2],
    )
    volume_io_s = time.perf_counter() - started
    del field
    ex, exe = centres(-5.0, 5.0, e["edge_nx"][level])
    ey, eye = centres(-5.0, 5.0, e["edge_ny"][level])
    et = np.unique(np.r_[np.linspace(0.0, 0.5, e["edge_time_nodes"][level]), report_t])
    started = time.perf_counter()
    states = {}
    for edge in EDGES:
        tangent = ey if edge in EDGES[:2] else ex
        tt, zz = np.meshgrid(et, tangent, indexing="ij")
        wall = -5.0 if edge in ("left", "bottom") else 5.0
        points = np.column_stack(
            (
                np.full(tt.size, wall) if edge in EDGES[:2] else zz.ravel(),
                zz.ravel() if edge in EDGES[:2] else np.full(tt.size, wall),
                tt.ravel(),
            )
        )
        states[edge] = query_points(model, points, e["query_chunk_size"]).reshape(
            len(et), -1, 3
        )
    edge_query_s = time.perf_counter() - started
    rates = four_edge_rates(states, ex, ey)
    # Reference reservoir flux and slip-wall flux are identically zero at all edges.
    normal = {edge: states[edge][..., 1 if edge in EDGES[:2] else 2] for edge in EDGES}
    outward = {
        edge: value * (-1.0 if edge in ("left", "bottom") else 1.0)
        for edge, value in normal.items()
    }
    absolute_rates = {
        edge: np.abs(value).mean(axis=1) * 10.0 for edge, value in outward.items()
    }
    integrals = {
        edge: cumulative_trapezoid(rate, et, initial=0.0)
        for edge, rate in rates.items()
    }
    index = np.searchsorted(et, report_t)
    if not np.array_equal(et[index], report_t):
        raise RuntimeError("report times absent from edge quadrature")
    flux = sum(integrals.values())[index]
    started = time.perf_counter()
    np.savez_compressed(
        directory / f"edges_{level}.npz",
        x_m=ex,
        y_m=ey,
        t_s=et,
        x_edges_m=exe,
        y_edges_m=eye,
        **{f"{k}_state_h_qx_qy": v for k, v in states.items()},
        **{f"{k}_outward_rate_m3_s": v for k, v in rates.items()},
        **{f"{k}_outward_integral_m3": v for k, v in integrals.items()},
    )
    edge_io_s = time.perf_counter() - started
    balance = volume - 55.0 + flux
    return {
        "volume_m3": volume.tolist(),
        "report_t_s": report_t.tolist(),
        "predicted_initial_bias_m3": float(volume[0] - 55.0),
        "outward_integral_m3": flux.tolist(),
        "corrected_balance_m3": balance.tolist(),
        "balance_without_initial_bias_m3": (balance - (volume[0] - 55.0)).tolist(),
        "corrected_balance_pct_prescribed": (100 * balance / 55.0).tolist(),
        "sampled_max_abs_corrected_balance_m3": float(np.max(np.abs(balance))),
        "edge": {
            edge: {
                "final_signed_outward_integral_m3": float(integrals[edge][-1]),
                "absolute_normal_flux_mismatch_integral_m3": float(
                    np.trapezoid(absolute_rates[edge], et)
                ),
                "sampled_max_abs_normal_flux_m2_s": float(np.max(np.abs(normal[edge]))),
            }
            for edge in EDGES
        },
        "axes": {
            "volume_nx": len(x),
            "volume_ny": len(y),
            "volume_nt": len(report_t),
            "edge_nx": len(ex),
            "edge_ny": len(ey),
            "edge_nt_actual": len(et),
        },
        "cost_s": {
            "volume_query": query_volume_s,
            "volume_io": volume_io_s,
            "edge_query": edge_query_s,
            "edge_io": edge_io_s,
        },
        "query_points": len(x) * len(y) * len(report_t)
        + len(et) * (2 * len(ex) + 2 * len(ey)),
    }


def diagnostic_definitions() -> dict:
    return {
        "scope": (
            "single-seed planar consistency; independently trained "
            "unconstrained y input/state"
        ),
        "array_contract": (
            "point values (time,y,x); fields h_m,qx_m2_s,qy_m2_s; x/y "
            "edges define quadrature cells"
        ),
        "loss": (
            "PDE mean sum of 3 squared nd residuals; IC mean sum of 3 "
            "squared nd errors; BC equal-point mean, x all3 fields and y "
            "qy only; fixed 1/10/10"
        ),
        "selection": (
            "initial state and every Adam update, earliest strict minimum"
            " separate validation objective; no analytical validation "
            "labels"
        ),
        "errors": (
            "equal-cell RMSE and 100*L2(error)/L2(reference), "
            "whole/shock/outside; per-time per-y slice and aggregate; no "
            "zero-reference relative transverse error"
        ),
        "shock": "abs(x-shock_speed*t)<=0.1m; all y; never differentiated",
        "velocity": (
            "reference h>0.001m fixed mask and coverage; predicted u=qx/h, v=qy/h"
        ),
        "y_variation": (
            "max absolute and RMS deviation from arithmetic y mean, "
            "h/qx/qy/u/v, global and per time; sampled maxima"
        ),
        "mass": (
            "midpoint area sum h dxdy; M(t)-55+integral all4 outward "
            "flux; initial bias separate; also bias-subtracted balance; "
            "all4 absolute normal mismatch integrals prevent cancellation"
        ),
        "quadrature": (
            "base/refined volumes800x21/1600x42 at t0+50 times; edge "
            "tangent x400/800,y21/42; time1001/2001 union exact reporting"
            " times; midpoint space/trapezoid time"
        ),
        "sensitivity": (
            "combined x/y/time refinement difference of "
            "volumes/flux/balance, not error bound or per-axis "
            "attribution"
        ),
        "limits": (
            "no genuine2D, multi-seed variability, continuous-time "
            "extrema, convergence, field validation or comparison of "
            "controlled training cost with 1D"
        ),
    }


def freeze_protocol(
    cfg: dict,
    case: dict,
    config_path: Path,
    run: Path,
    train: dict,
    val: dict,
    root: Path,
) -> dict:
    np.savez_compressed(run / "training_samples.npz", **train)
    np.savez_compressed(run / "validation_samples.npz", **val)
    np.savez_compressed(run / "final_axes.npz", **final_axes(cfg))
    protocol = {
        "schema_version": 1,
        "state": "immutable_proposed_full_fit_not_released",
        "config": cfg,
        "case": case,
        "scales": case["scales"],
        "source_sha256": {p: sha(root / p) for p in SOURCES},
        "input_sha256": {
            str(config_path): sha(root / config_path),
            cfg["case"]: sha(root / cfg["case"]),
        },
        "membership_sha256": {
            p: sha(run / p)
            for p in (
                "training_samples.npz",
                "validation_samples.npz",
                "final_axes.npz",
            )
        },
        "partitions": partition_receipt(train, val),
        "diagnostics": diagnostic_definitions(),
        "wave_diagnostics": wave_diagnostics(case),
        "prescribed_initial_volume_m3": 55.0,
        "full_fit_release": (
            "root receipt state=released with matching protocol_sha256 "
            "required; pilot is not a release"
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
    for name, expected in {**p["source_sha256"], **p["input_sha256"]}.items():
        if sha(root / name) != expected:
            raise ValueError(f"protocol source/input changed: {name}")
    for name, expected in p["membership_sha256"].items():
        if sha(path.parent / name) != expected:
            raise ValueError(f"protocol membership changed: {name}")
    return p


def precision_check(cfg: dict, case: dict) -> dict:
    """One small objective/gradient comparison; no device or precision sweep."""
    subset = samples(cfg, case, "training")
    subset = {k: v[: min(len(v), 32)] if v.ndim == 2 else v for k, v in subset.items()}
    torch.manual_seed(cfg["seed"])
    model32 = PlanarPINN(Scales.from_case(case))
    model64 = PlanarPINN(Scales.from_case(case), dtype=torch.float64)
    model64.load_state_dict({k: v.double() for k, v in model32.state_dict().items()})
    observed = []
    for model in (model32, model64):
        components = losses(model, tensor_samples(model, subset))
        components["objective"].backward()
        gradient = np.concatenate(
            [p.grad.detach().double().numpy().ravel() for p in model.parameters()]
        )
        observed.append((_numbers(components), gradient))
    a, b = observed
    objective_relative = abs(a[0]["objective"] - b[0]["objective"]) / abs(
        b[0]["objective"]
    )
    gradient_relative_l2 = float(np.linalg.norm(a[1] - b[1]) / np.linalg.norm(b[1]))
    gradient_max_abs = float(np.max(np.abs(a[1] - b[1])))
    # Aggregate tolerances accommodate float32 reductions and nested autograd;
    # absolute allowance also covers components/parameters whose true value is zero.
    passed = bool(
        objective_relative < 2e-5
        and gradient_relative_l2 < 2e-4
        and np.allclose(a[1], b[1], rtol=2e-3, atol=2e-5)
    )
    receipt = {
        "passed": passed,
        "points_per_membership_max": 32,
        "initial_state_cast_from_float32": True,
        "objective_relative_difference": objective_relative,
        "gradient_relative_l2_difference": gradient_relative_l2,
        "gradient_max_absolute_difference": gradient_max_abs,
        "float32_components": a[0],
        "float64_components": b[0],
        "tolerances": {
            "objective_relative": 2e-5,
            "gradient_relative_l2": 2e-4,
            "gradient_element_rtol": 2e-3,
            "gradient_element_atol": 2e-5,
        },
        "rationale": (
            "Identical float32 initialization cast to64; aggregate "
            "tolerance covers float32 reduction and autograd rounding; "
            "element absolute tolerance protects near-zero gradients. "
            "This checks one small objective, not long-fit trajectory "
            "equality."
        ),
    }
    if not passed:
        raise RuntimeError(f"CPU32/64 sanity failed: {receipt}")
    return receipt


def _checkpoint(
    model: PlanarPINN,
    optimizer: torch.optim.Optimizer,
    step: int,
    score: float,
    protocol: dict,
    initial_identity: str,
) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "validation_score": score,
        "protocol_sha256": protocol["protocol_sha256"],
        "initial_identity": initial_identity,
        "torch_rng_state": torch.get_rng_state(),
        "sample_sha256": protocol["membership_sha256"],
    }


def load_checkpoint(path: Path, protocol: dict) -> PlanarPINN:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["protocol_sha256"] != protocol["protocol_sha256"]:
        raise ValueError("checkpoint/protocol mismatch")
    if checkpoint["sample_sha256"] != protocol["membership_sha256"]:
        raise ValueError("checkpoint/sample mismatch")
    model = PlanarPINN(Scales.from_case(protocol["case"]))
    model.load_state_dict(checkpoint["model"])
    return model


def verify_saved_training(run: Path, summary: dict, protocol: dict) -> dict:
    """Read actual ordered history and checkpoints, including the initial state."""
    if sha(run / "history.csv") != summary["history_sha256"]:
        raise ValueError("training history hash changed")
    if sha(run / "checkpoint.pt") != summary["checkpoint_sha256"]:
        raise ValueError("selected checkpoint hash changed")
    with (run / "history.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    steps = summary["actual_updates"]
    if [int(row["step"]) for row in rows] != list(range(steps + 1)):
        raise ValueError("history does not record initial state and every update")
    best, best_step, selected_count = float("inf"), None, 0
    for row in rows:
        score = float(row["val_objective"])
        if not np.isfinite(score):
            raise ValueError("nonfinite saved validation score")
        selected = score < best
        if row["selected"] != str(selected):
            raise ValueError("saved selection flag violates earliest strict minimum")
        if selected:
            best, best_step = score, int(row["step"])
            selected_count += 1
    counts = {"initial": 1, "selected": selected_count, "latest": steps + 1}
    if summary["checkpoint_write_counts"] != counts:
        raise ValueError("checkpoint write counts disagree with history")
    if (
        summary["selected_step"] != best_step
        or summary["selected_validation_score"] != best
    ):
        raise ValueError("summary selection disagrees with history")
    for filename, step in (
        ("initial.pt", 0),
        ("checkpoint.pt", best_step),
        ("latest.pt", steps),
    ):
        payload = torch.load(run / filename, map_location="cpu", weights_only=True)
        if (
            payload["step"] != step
            or payload["protocol_sha256"] != protocol["protocol_sha256"]
            or payload["sample_sha256"] != protocol["membership_sha256"]
            or payload["initial_identity"] != summary["initial_identity"]
        ):
            raise ValueError(f"checkpoint provenance disagrees: {filename}")
        if payload["validation_score"] != float(rows[step]["val_objective"]):
            raise ValueError(f"checkpoint score disagrees: {filename}")
    if (
        _identity(load_checkpoint(run / "initial.pt", protocol))
        != summary["initial_identity"]
    ):
        raise ValueError("initial parameter identity changed")
    if summary["mode"] == "full":
        release = _read(run / "root_release.json")
        if (
            release.get("state") != "released"
            or release.get("protocol_sha256") != protocol["protocol_sha256"]
            or steps != protocol["config"]["full_steps"]
        ):
            raise ValueError("saved full fit release/budget mismatch")
    return {
        "history_steps": len(rows),
        "earliest_strict_minimum_step": best_step,
        "checkpoint_counts": counts,
        "initial_identity_verified": True,
        "release_verified": summary["mode"] == "full",
    }


def workload_queries(
    model: PlanarPINN,
    cfg: dict,
    case: dict,
    directory: Path,
    *,
    analytical_errors: bool = False,
) -> dict:
    """Full output/query/I/O cost; pilot leaves solution-error computation disabled."""
    workload_start = time.perf_counter()
    directory.mkdir(exist_ok=False)
    axes = final_axes(cfg)
    start = time.perf_counter()
    fields = query_grid(
        model,
        axes["x_m"],
        axes["y_m"],
        axes["t_s"],
        cfg["evaluation"]["query_chunk_size"],
    )
    query_s = time.perf_counter() - start
    start = time.perf_counter()
    reference = reference_fields(case, axes)
    reference_s = time.perf_counter() - start
    start = time.perf_counter()
    np.savez_compressed(
        directory / "network_fields.npz",
        **axes,
        h_m=fields[..., 0],
        qx_m2_s=fields[..., 1],
        qy_m2_s=fields[..., 2],
        reference_wet_mask=reference["reference_wet_mask"],
    )
    np.savez_compressed(directory / "reference_fields.npz", **axes, **reference)
    final_io_s = time.perf_counter() - start
    report = {
        "analytical_model_errors_computed": analytical_errors,
        "final_query_points": int(np.prod(fields.shape[:-1])),
        "cost_s": {
            "final_model_query": query_s,
            "final_reference": reference_s,
            "final_fields_io": final_io_s,
        },
    }
    if analytical_errors:
        start = time.perf_counter()
        _write(
            directory / "errors.json", field_diagnostics(fields, reference, axes, case)
        )
        report["cost_s"]["error_diagnostics"] = time.perf_counter() - start
    else:
        # Measure the same diagnostic/JSON workload using reference against
        # itself. Never compare pilot network fields with analytical values.
        start = time.perf_counter()
        reference_stack = np.stack(
            (reference["h_m"], reference["qx_m2_s"], reference["qy_m2_s"]), axis=-1
        )
        _write(
            directory / "reference_only_diagnostic_fixture.json",
            field_diagnostics(reference_stack, reference, axes, case),
        )
        report["cost_s"]["reference_only_diagnostic_and_json_fixture"] = (
            time.perf_counter() - start
        )
        del reference_stack
    del fields, reference
    reports = [mass_queries(model, cfg, directory, i) for i in (0, 1)]
    report["mass_query_points"] = sum(r["query_points"] for r in reports)
    report["cost_s"]["mass_all_query_and_io"] = sum(
        sum(r["cost_s"].values()) for r in reports
    )
    report["mass_axes"] = [r["axes"] for r in reports]
    # Mass diagnostics are computed only at final evaluation; pilot merely queries
    # and persists matching raw workloads and records costs, never quality values.
    if analytical_errors:
        a, b = reports
        _write(
            directory / "mass.json",
            {
                "base": a,
                "refined": b,
                "combined_refinement_sensitivity": {
                    key: float(np.max(np.abs(np.array(a[key]) - np.array(b[key]))))
                    for key in (
                        "volume_m3",
                        "outward_integral_m3",
                        "corrected_balance_m3",
                    )
                },
                "limitations": (
                    "Combined spatial/time sensitivity is not an error bound. "
                    "Maxima occur at saved samples, not all continuous times."
                ),
            },
        )
    report["memory"] = memory_observation()
    report["output_bytes"] = sum(p.stat().st_size for p in directory.iterdir())
    report["wall_s"] = time.perf_counter() - workload_start
    _write(directory / "query_receipt.json", report)
    return report


def _metrics(a: np.ndarray, b: np.ndarray) -> dict:
    if not a.size:
        return {
            "count": 0,
            "rmse": None,
            "relative_l2_pct": None,
            "reason": "empty region",
        }
    norm = float(np.linalg.norm(b.ravel()))
    return {
        "count": int(a.size),
        "rmse": float(np.sqrt(np.mean((a - b) ** 2))),
        "relative_l2_pct": float(100 * np.linalg.norm((a - b).ravel()) / norm)
        if norm > 0
        else None,
        "relative_reason": None if norm > 0 else "zero reference norm",
    }


def field_diagnostics(fields: np.ndarray, ref: dict, axes: dict, case: dict) -> dict:
    speed = wave_diagnostics(case)["wave_speeds_m_s"]["shock"]
    band = np.abs(axes["x_m"][None, :] - speed * axes["t_s"][:, None]) <= 0.1
    regions = {"whole": np.ones_like(band), "shock_band": band, "outside_band": ~band}
    wet = ref["reference_wet_mask"]
    u, v = fields[..., 1] / fields[..., 0], fields[..., 2] / fields[..., 0]
    ur = ref["qx_m2_s"] / ref["h_m"]
    aggregate, rows = {}, []
    variables = {
        "h_m": (fields[..., 0], ref["h_m"]),
        "qx_m2_s": (fields[..., 1], ref["qx_m2_s"]),
        "qy_m2_s": (fields[..., 2], ref["qy_m2_s"]),
        "u_m_s": (u, ur),
        "v_m_s": (v, np.zeros_like(v)),
    }
    for region, mask_tx in regions.items():
        mask = np.broadcast_to(mask_tx[:, None, :], wet.shape)
        aggregate[region] = {
            name: _metrics(
                a[mask & wet] if name in ("u_m_s", "v_m_s") else a[mask],
                b[mask & wet] if name in ("u_m_s", "v_m_s") else b[mask],
            )
            for name, (a, b) in variables.items()
        }
    for ti, t in enumerate(axes["t_s"]):
        for yi, y in enumerate(axes["y_m"]):
            row = {"t_s": float(t), "y_m": float(y), "regions": {}}
            for region, mask_tx in regions.items():
                mask = mask_tx[ti]
                row["regions"][region] = {
                    name: _metrics(
                        a[ti, yi, mask & wet[ti, yi]]
                        if name in ("u_m_s", "v_m_s")
                        else a[ti, yi, mask],
                        b[ti, yi, mask & wet[ti, yi]]
                        if name in ("u_m_s", "v_m_s")
                        else b[ti, yi, mask],
                    )
                    for name, (a, b) in variables.items()
                }
            rows.append(row)
    variation = {}
    for name, (a, _) in variables.items():
        d = a - a.mean(axis=1, keepdims=True)
        variation[name] = {
            "max_absolute": float(np.max(np.abs(d))),
            "rms": float(np.sqrt(np.mean(d**2))),
            "per_time_max_absolute": np.max(np.abs(d), axis=(1, 2)).tolist(),
            "per_time_rms": np.sqrt(np.mean(d**2, axis=(1, 2))).tolist(),
        }
    return {
        "aggregate": aggregate,
        "per_time_y_slice": rows,
        "y_variation": variation,
        "reference_wet_count": int(wet.sum()),
        "reference_wet_coverage": float(wet.mean()),
        "definitions": diagnostic_definitions(),
    }


def run_planar(
    config_path: str | Path,
    run_id: str,
    *,
    root: Path = Path("."),
    protocol_path: Path | None = None,
    release_path: Path | None = None,
) -> Path:
    """One 25-step pilot by default; full run requires prior protocol + root release."""
    overall_start = time.perf_counter()
    root = root.resolve()
    config_path = Path(config_path)
    validate_run_id(run_id)
    if not run_id.startswith("p10_planar_"):
        raise ValueError("planar runs require a fresh p10_planar_ prefix")
    cfg, case = load_planar_config(config_path, root)
    full = release_path is not None
    if full:
        if protocol_path is None:
            raise ValueError("full fit requires the frozen pilot protocol")
        protocol = verify_protocol(root / protocol_path, root)
        release = _read(root / release_path)
        if (
            release.get("state") != "released"
            or release.get("protocol_sha256") != protocol["protocol_sha256"]
        ):
            raise ValueError("root release does not match the proposed protocol")
        if protocol["config"] != cfg:
            raise ValueError("full config differs from frozen proposal")
    elif protocol_path is not None:
        raise ValueError("a pilot must freeze its own new protocol")
    torch.set_num_threads(cfg["cpu_threads"])
    resolved = {
        "case": case,
        "extension": cfg,
        "request": {"run_id": run_id, "device": "cpu", "dtype": "float32", "seed": 11},
    }
    with RunRecord.create(root / "runs", run_id, resolved, project_root=root) as record:
        run = record.path
        if full:
            for filename in (
                "training_samples.npz",
                "validation_samples.npz",
                "final_axes.npz",
            ):
                (run / filename).write_bytes(
                    (root / protocol_path).parent.joinpath(filename).read_bytes()
                )
            _write(run / "protocol.json", protocol)
            train = dict(np.load(run / "training_samples.npz"))
            val = dict(np.load(run / "validation_samples.npz"))
            _write(run / "root_release.json", release)
        else:
            train, val = (
                samples(cfg, case, "training"),
                samples(cfg, case, "validation"),
            )
            protocol = freeze_protocol(cfg, case, config_path, run, train, val, root)
            precision = precision_check(cfg, case)
            _write(run / "precision_check.json", precision)
        torch.manual_seed(11)
        model = PlanarPINN(Scales.from_case(case))
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg["adam_lr"])
        initial_identity = _identity(model)
        train_t, val_t = tensor_samples(model, train), tensor_samples(model, val)
        best = float("inf")
        selected_step, saves = 0, {"initial": 0, "selected": 0, "latest": 0}
        timings = []
        observations = [{"stage": "before_training", **memory_observation()}]
        steps = cfg["full_steps"] if full else cfg["pilot_steps"]
        start = time.perf_counter()
        with (run / "history.csv").open("x", newline="") as handle:
            columns = [
                "step",
                *[f"train_{k}" for k in (*LOSS_KEYS, "pde", "ic", "bc", "objective")],
                *[f"val_{k}" for k in (*LOSS_KEYS, "pde", "ic", "bc", "objective")],
                "selected",
                "objective_s",
                "backward_s",
                "optimizer_s",
                "validation_s",
                "checkpoint_io_s",
                "full_step_s",
            ]
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for step in range(steps + 1):
                step_start = time.perf_counter()
                times = {k: 0.0 for k in ("objective_s", "backward_s", "optimizer_s")}
                training = {}
                if step:
                    optimizer.zero_grad(set_to_none=True)
                    now = time.perf_counter()
                    components = losses(model, train_t)
                    training = _numbers(components)
                    times["objective_s"] = time.perf_counter() - now
                    now = time.perf_counter()
                    components["objective"].backward()
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
                validation = _numbers(losses(model, val_t))
                times["validation_s"] = time.perf_counter() - now
                score = validation["objective"]
                selected = score < best
                now = time.perf_counter()
                payload = _checkpoint(
                    model, optimizer, step, score, protocol, initial_identity
                )
                if step == 0:
                    torch.save(payload, run / "initial.pt")
                    saves["initial"] += 1
                if selected:
                    best, selected_step = score, step
                    torch.save(payload, run / "checkpoint.pt")
                    saves["selected"] += 1
                torch.save(payload, run / "latest.pt")
                saves["latest"] += 1
                times["checkpoint_io_s"] = time.perf_counter() - now
                times["full_step_s"] = time.perf_counter() - step_start
                writer.writerow(
                    {
                        "step": step,
                        **{f"train_{k}": v for k, v in training.items()},
                        **{f"val_{k}": v for k, v in validation.items()},
                        "selected": selected,
                        **times,
                    }
                )
                handle.flush()
                if step:
                    timings.append(times)
                if step in (0, steps):
                    observations.append(
                        {"stage": f"step_{step}", **memory_observation()}
                    )
        train_wall = time.perf_counter() - start
        selected_model = load_checkpoint(run / "checkpoint.pt", protocol)
        latest_model = load_checkpoint(run / "latest.pt", protocol)
        probe = torch.as_tensor(val["ic_xyt_nd"][:16], dtype=torch.float32)
        with torch.no_grad():
            if not torch.equal(model(probe), latest_model(probe)):
                raise RuntimeError("latest checkpoint reload prediction mismatch")
        selection = torch.load(run / "checkpoint.pt", weights_only=True)
        reloaded_score = _numbers(losses(selected_model, val_t))["objective"]
        if reloaded_score != best or selection["step"] != selected_step:
            raise RuntimeError("selected checkpoint reload selection mismatch")
        summary = {
            "state": "completed",
            "mode": "full" if full else "pilot",
            "actual_updates": steps,
            "validation_evaluations": steps + 1,
            "checkpoint_write_counts": saves,
            "selected_step": selected_step,
            "selected_validation_score": best,
            "reload_score_exact": reloaded_score == best,
            "latest_reload_prediction_exact": True,
            "initial_identity": initial_identity,
            "protocol_sha256": protocol["protocol_sha256"],
            "checkpoint_sha256": sha(run / "checkpoint.pt"),
            "history_sha256": sha(run / "history.csv"),
            "training_wall_s": train_wall,
            "raw_step_costs_s": timings,
            "memory_observations": observations,
            "analytical_model_errors_computed": False,
        }
        summary["saved_training_readback"] = verify_saved_training(
            run, summary, protocol
        )
        if not full:
            setup_reload_s = time.perf_counter() - overall_start - train_wall
            queries = workload_queries(
                selected_model, cfg, case, run / "pilot_query_workload"
            )
            summary["query_workload"] = queries
            component_means = {
                k: float(np.mean([t[k] for t in timings])) for k in timings[0]
            }
            mean_step = component_means["full_step_s"]
            query_cost = queries["wall_s"]
            summary["projection"] = {
                "full_updates": cfg["full_steps"],
                "mean_measured_step_cost_s": component_means,
                "training_wall_scaled_s": train_wall / steps * cfg["full_steps"],
                "setup_precision_reload_observed_s": setup_reload_s,
                "complete_measured_workload_projection_s": train_wall
                / steps
                * cfg["full_steps"]
                + query_cost
                + setup_reload_s,
                "step_median_s": float(np.median([t["full_step_s"] for t in timings])),
                "step_min_max_s": [
                    min(t["full_step_s"] for t in timings),
                    max(t["full_step_s"] for t in timings),
                ],
                "step_mean_projection_s": mean_step * cfg["full_steps"],
                "complete_with_slowest_observed_step_s": max(
                    t["full_step_s"] for t in timings
                )
                * cfg["full_steps"]
                + query_cost
                + setup_reload_s,
                "limits": (
                    "One25step sample including validation/checkpoint I/O. "
                    "Scaling includes initial-state overhead; sustained "
                    "thermal/load/storage cost may differ. Error diagnostic/JSON "
                    "cost uses reference-only fixture; no pilot network accuracy "
                    "screening. Future figures/archive are unmeasured packaging "
                    "overhead. No timing repeats."
                ),
            }
            summary["full_fit_state"] = "held_pending_root_technical_release"
        summary["observed_run_wall_before_summary_s"] = (
            time.perf_counter() - overall_start
        )
        _write(run / "summary.json", summary)
        record.complete(
            stopping_reason="declared full updates"
            if full
            else "single bounded feasibility pilot",
            iterations=steps,
            function_evaluations=2 * steps + 1,
            elapsed_timings_s={"training": train_wall},
        )
        return run


def evaluate_planar(
    run_path: str | Path, output_dir: str | Path, *, root: Path = Path(".")
) -> Path:
    root = root.resolve()
    run, output = root / run_path, root / output_dir
    summary = _read(run / "summary.json")
    if (
        summary["mode"] != "full"
        or summary["actual_updates"] != 10000
        or _read(run / "status.json")["state"] != "completed"
    ):
        raise ValueError(
            "final evaluation requires completed full10000step run; pilots refused"
        )
    protocol = verify_protocol(run / "protocol.json", root)
    verify_saved_training(run, summary, protocol)
    torch.set_num_threads(4)
    model = load_checkpoint(run / "checkpoint.pt", protocol)
    report = workload_queries(
        model, protocol["config"], protocol["case"], output, analytical_errors=True
    )
    _write(
        output / "provenance.json",
        {
            "run": str(run_path),
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "protocol_sha256": protocol["protocol_sha256"],
            "actual_costs": report,
            "field_location": "point_value",
            "scope": "planar consistency only",
        },
    )
    return output
