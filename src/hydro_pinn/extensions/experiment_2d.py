"""Frozen, physics-only wet rectangular 2D neural experiment.

Preparation fixes coordinates, initial CPU32 tensors/fresh Adam, operators and
scientific input hashes. The one25-update pilot has no neural quality analysis.
A separate exact release is required for one fresh10000-update run. Historical
archives belong to preservation audits, not this study's runtime dependencies.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import platform
import resource
import shutil
import time
from pathlib import Path

import numpy as np
import torch

from hydro_pinn.extensions.model_2d import PlanarPINN
from hydro_pinn.extensions.physics_2d import residuals
from hydro_pinn.physics import Scales
from hydro_pinn.records import RunRecord, validate_run_id, write_json_atomic

EXPECTED_CONFIG = {
    "adam_lr": 0.001,
    "budget": {
        "new_full_decision_s": 1200,
        "type": "prelaunch_decision_not_runtime_cap",
        "unmeasured_one_time_allowance_s": 120,
    },
    "case_path": "configs/extensions/case_2d.json",
    "cpu_threads": 4,
    "depth_floor_nd": 1e-06,
    "device": "cpu",
    "dtype": "float32",
    "experiment": "genuine_2d_wet_rectangle_physics_only_v01",
    "full_steps": 10000,
    "hidden_layers": [64, 64, 64, 64],
    "interop_threads": 4,
    "pilot_steps": 25,
    "precision": {
        "element_atol": 2e-05,
        "element_rtol": 0.002,
        "gradient_relative_L2_lt": 0.0002,
        "objective_relative_lt": 2e-05,
        "pde": 32,
        "per_wall": 32,
        "strata_counts": [2, 40, 40, 9, 9],
        "zero_objective_absolute_le": 2e-05,
    },
    "query": {
        "batch_size": 20000,
        "common_n": 80,
        "gauss_orders": [3, 5],
        "mass": [
            {"id": "base", "n": 80, "time_nodes": 501},
            {"id": "spatial", "n": 160, "time_nodes": 501},
            {"id": "temporal", "n": 80, "time_nodes": 1001},
        ],
        "positive_dt_s": 0.005,
        "positive_times": 50,
    },
    "sampling": {
        "training": {
            "ic": 1000,
            "pde": 10000,
            "per_wall": 500,
            "seed": 110201,
            "strata_counts": [20, 400, 400, 90, 90],
        },
        "validation": {
            "ic": 400,
            "pde": 2000,
            "per_wall": 200,
            "seed": 110202,
            "strata_counts": [8, 160, 160, 36, 36],
        },
    },
    "schema_version": 1,
    "seed": 11,
    "weights": {"bc": 10.0, "ic": 10.0, "pde": 1.0},
}
WALL_SELECTORS = {"left": 1, "right": 1, "bottom": 2, "top": 2}
STRATA = (
    {
        "name": "deep",
        "x": (-1.0, 1.0),
        "y": (-0.5, 0.5),
        "training": 20,
        "validation": 8,
        "precision": 2,
    },
    {
        "name": "left",
        "x": (-5.0, -1.0),
        "y": (-5.0, 5.0),
        "training": 400,
        "validation": 160,
        "precision": 40,
    },
    {
        "name": "right",
        "x": (1.0, 5.0),
        "y": (-5.0, 5.0),
        "training": 400,
        "validation": 160,
        "precision": 40,
    },
    {
        "name": "middle_below",
        "x": (-1.0, 1.0),
        "y": (-5.0, -0.5),
        "training": 90,
        "validation": 36,
        "precision": 9,
    },
    {
        "name": "middle_above",
        "x": (-1.0, 1.0),
        "y": (0.5, 5.0),
        "training": 90,
        "validation": 36,
        "precision": 9,
    },
)
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
    "pde",
    "ic",
    "bc",
    "objective",
)
SOURCE_PATHS = (
    "src/hydro_pinn/__init__.py",
    "src/hydro_pinn/__main__.py",
    "src/hydro_pinn/extensions/__init__.py",
    "src/hydro_pinn/extensions/experiment_2d.py",
    "src/hydro_pinn/extensions/evaluate_2d.py",
    "src/hydro_pinn/extensions/model_2d.py",
    "src/hydro_pinn/extensions/physics_2d.py",
    "src/hydro_pinn/physics.py",
    "src/hydro_pinn/records.py",
    "src/hydro_pinn/config.py",
    "src/hydro_pinn/cli.py",
    "pyproject.toml",
    "uv.lock",
)
CONTRACT_PATHS = (
    "docs/coordination/extension_genuine_2d_neural_selection.json",
    "docs/coordination/phase_10_2_neural_study_plan_v01.json",
    "docs/coordination/phase_10_2_neural_pre_freeze_clarifications_v01.json",
    "docs/coordination/phase_10_2_neural_plan_science_review_v01.json",
    "docs/coordination/phase_10_2_neural_plan_cost_review_v01.json",
    "docs/coordination/phase_10_2_reference_completion_review_v01.json",
)
EVALUATION_PATHS = {
    "selected_reference": (
        "results/extensions/two_dimensional/reference_v01/ref_rect320"
        "_c0225_v01/common_reference.npz"
    ),
    "gradient_region": (
        "results/extensions/two_dimensional/reference_v01/r6_analysis"
        "_v01/selected_reference_gradient_region.npz"
    ),
    "sensitivity": (
        "results/extensions/two_dimensional/reference_v01/r6_analysis"
        "_v01/signed_differences_and_sensitivity.npz"
    ),
    "reference_norms": (
        "results/extensions/two_dimensional/reference_v01/r6_analysis_v01/norms.json"
    ),
}


class Experiment2DError(ValueError):
    def __init__(self, message, receipt=None):
        super().__init__(message)
        self.receipt = receipt


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1048576):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    write_json_atomic(Path(path), value)


def scales():
    speed = math.sqrt(9.81)
    return Scales(H_m=1.0, Lc_m=10.0, Uc_m_s=speed, Tc_s=10.0 / speed)


def load_config(path, root=Path(".")):
    value = read(Path(root) / path)
    if value != EXPECTED_CONFIG:
        raise Experiment2DError("configuration differs from fixed genuine2D v01 study")
    return value


def default_config():
    return copy.deepcopy(EXPECTED_CONFIG)


def configure_threads():
    before = {
        "intra": torch.get_num_threads(),
        "interop": torch.get_num_interop_threads(),
    }
    torch.set_num_threads(4)
    if torch.get_num_interop_threads() != 4:
        torch.set_num_interop_threads(4)
    after = {
        "intra": torch.get_num_threads(),
        "interop": torch.get_num_interop_threads(),
    }
    if after != {"intra": 4, "interop": 4}:
        raise RuntimeError("fresh process must provide four intra/inter-op threads")
    return {
        "before": before,
        "actual": after,
        "limit": (
            "Interop can only be configured before parallel work in a fre"
            "sh process; no fallback."
        ),
    }


def new_model(dtype=torch.float32):
    return PlanarPINN(scales(), hidden_layers=(64, 64, 64, 64), dtype=dtype)


def fresh_initial_state():
    torch.manual_seed(11)
    model = new_model()
    return model, torch.optim.Adam(model.parameters(), lr=0.001)


def model_identity(model_or_state):
    state = (
        model_or_state.state_dict()
        if hasattr(model_or_state, "state_dict")
        else model_or_state
    )
    h = hashlib.sha256()
    for name, value in state.items():
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def execution_coordinates(si, dtype=torch.float32):
    """Use the same SIcast then divide sequence as physical_state."""
    s = scales()
    a = torch.as_tensor(np.asarray(si), dtype=dtype, device="cpu")
    return torch.stack((a[:, 0] / s.Lc_m, a[:, 1] / s.Lc_m, a[:, 2] / s.Tc_s), dim=1)


def initial_depth(si):
    a = np.asarray(si)
    deep = (a[:, 0] >= -1.0) & (a[:, 0] < 1.0) & (a[:, 1] >= -0.5) & (a[:, 1] < 0.5)
    return np.where(deep, 1.0, 0.1)


def _representations(points):
    s = scales()
    return {
        "si64": np.asarray(points, dtype=np.float64),
        "scaled64": np.asarray(points, dtype=np.float64) / [s.Lc_m, s.Lc_m, s.Tc_s],
        "si32": np.asarray(points, dtype=np.float32),
        "input32": execution_coordinates(points).numpy().copy(),
    }


def generate_samples(cfg, partition="training"):
    """Explicit PCG64 call order; no resampling, repair, or interior targets."""
    spec = cfg["sampling"][partition]
    rng = np.random.Generator(np.random.PCG64(spec["seed"]))
    pde = np.column_stack(
        (
            rng.uniform(-5, 5, spec["pde"]),
            rng.uniform(-5, 5, spec["pde"]),
            rng.uniform(0, 0.25, spec["pde"]),
        )
    )
    arrays, ic_parts, strata = {}, [], []
    for index, stratum in enumerate(STRATA):
        count = spec["strata_counts"][index]
        ic_parts.append(
            np.column_stack(
                (
                    rng.uniform(*stratum["x"], count),
                    rng.uniform(*stratum["y"], count),
                    np.zeros(count),
                )
            )
        )
        strata.extend([index] * count)
    unshuffled = np.concatenate(ic_parts)
    permutation = rng.permutation(spec["ic"])
    ic = unshuffled[permutation]
    arrays["ic_stratum"] = np.asarray(strata, dtype=np.int64)[permutation]
    arrays["ic_unshuffled_index"] = permutation
    targets = np.zeros((len(ic), 3), dtype=np.float64)
    targets[:, 0] = initial_depth(ic)
    arrays["ic_target64"] = targets
    for prefix, points in (("pde", pde), ("ic", ic)):
        arrays.update(
            {
                f"{prefix}_{name}": value
                for name, value in _representations(points).items()
            }
        )
    for wall in WALL_SELECTORS:
        count = spec["per_wall"]
        tangent, t = rng.uniform(-5, 5, count), rng.uniform(0, 0.25, count)
        fixed = np.full(count, -5.0 if wall in ("left", "bottom") else 5.0)
        points = (
            np.column_stack((fixed, tangent, t))
            if wall in ("left", "right")
            else np.column_stack((tangent, fixed, t))
        )
        arrays.update(
            {
                f"{wall}_{name}": value
                for name, value in _representations(points).items()
            }
        )
    return arrays


def support_receipt(data):
    s = scales()
    checks = {}
    pde = data["pde_input32"].astype(np.float64)
    upper = float(np.float32(0.25) / np.float32(s.Tc_s))
    checks["PDE_open_execution_support"] = bool(
        np.all(np.abs(pde[:, :2]) < 0.5)
        and np.all((pde[:, 2] > 0) & (pde[:, 2] < upper))
    )
    ic = data["ic_input32"].astype(np.float64)
    checks["IC_execution_labels_stable"] = bool(
        np.array_equal(
            initial_depth(ic * [10.0, 10.0, s.Tc_s]), data["ic_target64"][:, 0]
        )
    )
    checks["IC_execution_support"] = bool(
        np.all(np.abs(ic[:, :2]) < 0.5) and np.all(ic[:, 2] == 0)
    )
    physical_ic = ic[:, :2] * 10.0
    for index, stratum in enumerate(STRATA):
        points = physical_ic[data["ic_stratum"] == index]
        checks[stratum["name"] + "_open_execution_stratum"] = bool(
            len(points) > 0
            and np.all(
                (points[:, 0] > stratum["x"][0]) & (points[:, 0] < stratum["x"][1])
            )
            and np.all(
                (points[:, 1] > stratum["y"][0]) & (points[:, 1] < stratum["y"][1])
            )
        )
    for wall in WALL_SELECTORS:
        a = data[f"{wall}_input32"]
        normal, tangent = (0, 1) if wall in ("left", "right") else (1, 0)
        location = -0.5 if wall in ("left", "bottom") else 0.5
        checks[wall + "_execution_support"] = bool(
            np.all(a[:, normal] == location)
            and np.all(np.abs(a[:, tangent]) < 0.5)
            and np.all((a[:, 2] > 0) & (a[:, 2] < upper))
        )
    checks["all_saved_arrays_finite"] = all(
        bool(np.isfinite(a).all()) for a in data.values()
    )
    result = {
        "passed": all(checks.values()),
        "checks": checks,
        "IC_stratum_counts": [int(np.sum(data["ic_stratum"] == i)) for i in range(5)],
    }
    if not result["passed"]:
        raise Experiment2DError(
            "fixed membership support or IC classification failed", result
        )
    return result


def row_bytes(rows):
    a = np.array(rows, copy=True, order="C")
    if a.ndim != 2 or a.shape[1] != 3 or not np.isfinite(a).all():
        raise Experiment2DError("membership rows require finite (N,3)")
    a[a == 0] = 0  # Both signs of zero have one canonical byte representation.
    return a.view(np.dtype((np.void, a.dtype.itemsize * 3))).reshape(-1)


def verify_memberships(train, val, gauss_axes):
    """Small row-byte sets and factorized axes; never form a final Cartesian set."""
    final = list(gauss_axes)
    result = {
        "algorithm": (
            "canonical C-row bytes; signed zero -> positive zero; factori"
            "zed exact final axes"
        ),
        "representations": {},
        "final_rules_intentionally_share_centres": 320000,
    }
    for representation in ("si64", "scaled64", "input32"):
        a = np.concatenate(
            [
                train[f"{name}_{representation}"]
                for name in ("pde", "ic", *WALL_SELECTORS)
            ]
        )
        b = np.concatenate(
            [val[f"{name}_{representation}"] for name in ("pde", "ic", *WALL_SELECTORS)]
        )
        ra, rb = row_bytes(a), row_bytes(b)
        facts = {
            "training_duplicates": len(ra) - len(np.unique(ra)),
            "validation_duplicates": len(rb) - len(np.unique(rb)),
            "train_validation_intersection": len(np.intersect1d(ra, rb)),
            "final_intersections": {},
        }
        for order, x, y, t in final:
            axes = (np.asarray(x), np.asarray(y), np.asarray(t))
            if representation == "scaled64":
                axes = tuple(
                    v / scale
                    for v, scale in zip(axes, (10.0, 10.0, scales().Tc_s), strict=True)
                )
            elif representation == "input32":
                axes = tuple(
                    (torch.as_tensor(v, dtype=torch.float32) / scale).numpy()
                    for v, scale in zip(axes, (10.0, 10.0, scales().Tc_s), strict=True)
                )
            facts["final_intersections"][str(order)] = {
                name: int(
                    np.sum(
                        np.isin(rows[:, 0], axes[0])
                        & np.isin(rows[:, 1], axes[1])
                        & np.isin(rows[:, 2], axes[2])
                    )
                )
                for name, rows in (("training", a), ("validation", b))
            }
        result["representations"][representation] = facts
    result["passed"] = all(
        not facts["training_duplicates"]
        and not facts["validation_duplicates"]
        and not facts["train_validation_intersection"]
        and all(
            not value
            for intersections in facts["final_intersections"].values()
            for value in intersections.values()
        )
        for facts in result["representations"].values()
    )
    if not result["passed"]:
        raise Experiment2DError(
            "fixed train/validation/final memberships overlap", result
        )
    return result


def component_losses(model, data):
    dtype = next(model.parameters()).dtype
    r = residuals(model.forward, execution_coordinates(data["pde_si64"], dtype))
    ic = model(execution_coordinates(data["ic_si64"], dtype)) - torch.as_tensor(
        data["ic_target64"], dtype=dtype
    )
    losses = {key: r[:, i].square().mean() for i, key in enumerate(LOSS_KEYS[:3])}
    losses.update(
        {key: ic[:, i].square().mean() for i, key in enumerate(LOSS_KEYS[3:6])}
    )
    for wall, selector in WALL_SELECTORS.items():
        state = model(execution_coordinates(data[f"{wall}_si64"], dtype))
        losses["bc_" + wall] = state[:, selector].square().mean()
    losses["pde"] = sum(losses[key] for key in LOSS_KEYS[:3])
    losses["ic"] = sum(losses[key] for key in LOSS_KEYS[3:6])
    losses["bc"] = sum(losses["bc_" + wall] for wall in WALL_SELECTORS) / 4
    losses["objective"] = losses["pde"] + 10 * losses["ic"] + 10 * losses["bc"]
    if not all(bool(torch.isfinite(value)) for value in losses.values()):
        raise RuntimeError("nonfinite genuine2D objective")
    return losses


def _numbers(values):
    return {name: float(value.detach()) for name, value in values.items()}


def select_checkpoint(score, best):
    return score < best


def memory_observation():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "process_lifetime_peak_RSS_bytes": int(
            rss * (1 if platform.system() == "Darwin" else 1024)
        ),
        "host_load_1_5_15": list(os.getloadavg()),
        "limits": (
            "Process-lifetime high water, not instantaneous allocator/sys"
            "tem-pressure/thermal telemetry."
        ),
    }


def precision_indices(data):
    return {
        "pde": np.arange(32),
        "ic": np.concatenate(
            [
                np.flatnonzero(data["ic_stratum"] == i)[:count]
                for i, count in enumerate((2, 40, 40, 9, 9))
            ]
        ),
        **{wall: np.arange(32) for wall in WALL_SELECTORS},
    }


def _subset(data, indices):
    out = {}
    for key, value in data.items():
        prefix = key.split("_", 1)[0]
        out[key] = value[indices[prefix]] if prefix in indices else value
    return out


def prepare_protocol(config_path, destination, *, root=Path(".")):
    """Freeze only. The explicit precision check is separate and runs once."""
    from hydro_pinn.extensions.evaluate_2d import final_gauss_axes, prepare_operators

    root = Path(root).resolve()
    destination = root / destination
    destination.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        cfg = load_config(config_path, root)
        threads = configure_threads()
        write(destination / "thread_setup.json", threads)
        train, val = (
            generate_samples(cfg, "training"),
            generate_samples(cfg, "validation"),
        )
        np.savez_compressed(destination / "training_samples.npz", **train)
        np.savez_compressed(destination / "validation_samples.npz", **val)
        indices = precision_indices(train)
        np.savez_compressed(destination / "precision_indices.npz", **indices)
        support = {
            "training": support_receipt(train),
            "validation": support_receipt(val),
        }
        operators = prepare_operators(cfg, destination, scales())
        membership = verify_memberships(train, val, final_gauss_axes(destination))
        write(
            destination / "membership_receipt.json",
            {"support": support, "membership": membership},
        )
        model, optimizer = fresh_initial_state()
        if sum(p.numel() for p in model.parameters()) != 12931:
            raise RuntimeError("unexpected model parameter count")
        identity = model_identity(model)
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": 0,
                "initial_identity": identity,
                "torch_rng_state": torch.get_rng_state(),
            },
            destination / "initial_model_adam.pt",
        )
        write(destination / "config.json", cfg)
        sampling_schema = {
            label: {
                name: {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "C_bytes_sha256": hashlib.sha256(
                        value.tobytes(order="C")
                    ).hexdigest(),
                }
                for name, value in arrays.items()
            }
            for label, arrays in (
                ("training", train),
                ("validation", val),
                ("precision_indices", indices),
            )
        }
        write(destination / "sampling_schema.json", sampling_schema)
        artifacts = {p.name: sha(p) for p in destination.iterdir() if p.is_file()}
        runtime = {
            name: sha(root / name)
            for name in (
                *SOURCE_PATHS,
                *CONTRACT_PATHS,
                str(config_path),
                cfg["case_path"],
                *EVALUATION_PATHS.values(),
            )
        }
        protocol = {
            "schema_version": 1,
            "state": "frozen_genuine2D_pilot25_full_not_released",
            "config": cfg,
            "scales": scales().__dict__,
            "operator_directory": str(destination.relative_to(root)),
            "runtime_bindings": runtime,
            "artifact_bindings": artifacts,
            "membership_sha256": {
                name: artifacts[name]
                for name in (
                    "training_samples.npz",
                    "validation_samples.npz",
                    "precision_indices.npz",
                )
            },
            "initial_identity": identity,
            "initial_adam_identity": digest(optimizer.state_dict()),
            "initial_tensor_schema": {
                name: {
                    "dtype": str(value.dtype),
                    "shape": list(value.shape),
                    "bytes_sha256": hashlib.sha256(
                        value.detach().contiguous().numpy().tobytes()
                    ).hexdigest(),
                }
                for name, value in model.state_dict().items()
            },
            "initial_model_adam_file": "initial_model_adam.pt",
            "initial_Adam_state_entries": 0,
            "parameter_count": 12931,
            "evaluation_inputs": {
                name: {"path": path, "sha256": runtime[path]}
                for name, path in EVALUATION_PATHS.items()
            },
            "operators": operators,
            "memberships": membership,
            "precision_criteria": cfg["precision"],
            "sampling_order": (
                "train110201 then independentval110202; PDE x,y,t vectors; ea"
                "ch IC stratum x,y vectors then fixed permutation; walls left"
                ",right,bottom,top tangent,time vectors"
            ),
            "checkpoint_policy": (
                "initial+latest+selected strictminimum at step0; latest every"
                "update, selected everystrictimprovement; full optimizer each"
                "write"
            ),
            "history_semantics": (
                "step0 training+validation atinitial; stepN trainbeforeAdamN,"
                " validationafterAdamN; no reference labels"
            ),
            "precision_requirement": (
                "single separately persisted check on frozenindices/initial t"
                "ensors beforepilot; never repeated by existing-protocol pilo"
                "t"
            ),
            "query_quality_screening_in_pilot": False,
            "reference_archive_provenance": {
                "path": "results/extensions/two_dimensional/"
                "genuine_2d_reference_v01.zip",
                "sha256": "5346679a435eebb1198d69ebc3208d92"
                "b250b725433fc5a7e2529b9b205809b5",
                "runtime_dependency": False,
            },
            "budget_equation": (
                "10000*completecyclemean/median/max + measuredsetupandstep0 +"
                "2*(measuredreload+fullrawqueryIO) +120s unmeasuredone-time a"
                "llowance;1200s prelaunch decision only"
            ),
            "historical_archives": (
                "Preservation/provenance only; not runtime requirements or re"
                "cursively checked by selection JSON"
            ),
            "preparation_setup_wall_s_before_protocol_write": time.perf_counter()
            - started,
        }
        protocol["protocol_sha256"] = digest(protocol)
        write(destination / "protocol.json", protocol)
        return protocol
    except BaseException as exc:
        write(
            destination / "preparation_failure.json",
            {
                "state": "failed",
                "type": type(exc).__name__,
                "message": str(exc),
                "receipt": getattr(exc, "receipt", None),
                "elapsed_s": time.perf_counter() - started,
            },
        )
        raise


def verify_protocol(path, root=Path(".")):
    root, path = Path(root).resolve(), Path(path)
    if not path.is_absolute():
        path = root / path
    p = read(path)
    if (
        digest({k: v for k, v in p.items() if k != "protocol_sha256"})
        != p["protocol_sha256"]
    ):
        raise Experiment2DError("protocol content digest changed")
    for name, expected in p["runtime_bindings"].items():
        if sha(root / name) != expected:
            raise Experiment2DError("runtime input changed: " + name)
    for name, expected in p["artifact_bindings"].items():
        if sha(path.parent / name) != expected:
            raise Experiment2DError("protocol artifact changed: " + name)
    return p


def _initial_model_optimizer(directory, protocol):
    saved = torch.load(
        Path(directory) / protocol["initial_model_adam_file"],
        map_location="cpu",
        weights_only=True,
    )
    model = new_model()
    model.load_state_dict(saved["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.load_state_dict(saved["optimizer"])
    if (
        model_identity(model) != protocol["initial_identity"]
        or optimizer.state
        or saved["step"] != 0
        or digest(optimizer.state_dict()) != protocol["initial_adam_identity"]
    ):
        raise Experiment2DError("frozen fresh model/Adam identity changed")
    torch.set_rng_state(saved["torch_rng_state"])
    return model, optimizer


def precision_check(protocol_path, output_dir, *, root=Path(".")):
    root, protocol_path = Path(root).resolve(), Path(protocol_path)
    if not protocol_path.is_absolute():
        protocol_path = root / protocol_path
    p = verify_protocol(protocol_path, root)
    output = root / output_dir
    output.mkdir(parents=True, exist_ok=False)
    try:
        data = dict(np.load(protocol_path.parent / "training_samples.npz"))
        indices = dict(np.load(protocol_path.parent / "precision_indices.npz"))
        subset = _subset(data, indices)
        model32, _ = _initial_model_optimizer(protocol_path.parent, p)
        model64 = new_model(torch.float64)
        model64.load_state_dict(
            {name: value.double() for name, value in model32.state_dict().items()}
        )
        observed = []
        for label, model in (("32", model32), ("64", model64)):
            raw = residuals(
                model.forward,
                execution_coordinates(
                    subset["pde_si64"], next(model.parameters()).dtype
                ),
            )
            values = component_losses(model, subset)
            values["objective"].backward()
            gradients = {
                name: parameter.grad.detach().numpy().copy()
                for name, parameter in model.named_parameters()
            }
            if any(not np.isfinite(value).all() for value in gradients.values()):
                raise RuntimeError("nonfinite precision gradient")
            payload = {
                "residuals": raw.detach().numpy(),
                **{
                    "component_" + name: value.detach().numpy().copy()
                    for name, value in values.items()
                },
                **{"gradient_" + name: value for name, value in gradients.items()},
            }
            np.savez_compressed(output / ("raw" + label + ".npz"), **payload)
            observed.append((payload, gradients))
        a, b = observed
        ga = np.concatenate(
            [value.ravel().astype(np.float64) for value in a[1].values()]
        )
        gb = np.concatenate(
            [value.ravel().astype(np.float64) for value in b[1].values()]
        )
        objective_a, objective_b = (
            float(a[0]["component_objective"]),
            float(b[0]["component_objective"]),
        )
        objective_delta = abs(objective_a - objective_b)
        objective_relative = (
            objective_delta / abs(objective_b) if objective_b != 0 else None
        )
        gnorm = float(np.linalg.norm(gb))
        gradient_relative = (
            float(np.linalg.norm(ga - gb)) / gnorm if gnorm != 0 else None
        )
        residuals_ok = bool(
            np.allclose(a[0]["residuals"], b[0]["residuals"], rtol=0.002, atol=0.00002)
        )
        residual_comparison = []
        for index in range(3):
            cpu64 = b[0]["residuals"][:, index]
            delta = a[0]["residuals"][:, index].astype(np.float64) - cpu64
            norm = float(np.linalg.norm(cpu64))
            residual_comparison.append(
                {
                    "component": LOSS_KEYS[index],
                    "cpu64_norm": norm,
                    "absolute_L2_difference": float(np.linalg.norm(delta)),
                    "relative_L2_difference": float(np.linalg.norm(delta)) / norm
                    if norm
                    else None,
                    "relative_reason": None if norm else "zero_cpu64_norm",
                    "elements_allclose": bool(
                        np.allclose(
                            a[0]["residuals"][:, index],
                            cpu64,
                            rtol=0.002,
                            atol=0.00002,
                        )
                    ),
                }
            )
        checks = {
            "objective": objective_relative < 0.00002
            if objective_relative is not None
            else objective_delta <= 0.00002,
            "gradient_relative": gradient_relative < 0.0002
            if gradient_relative is not None
            else bool(np.max(np.abs(ga - gb)) <= 0.00002),
            "gradient_elements": bool(np.allclose(ga, gb, rtol=0.002, atol=0.00002)),
            "residual_elements": residuals_ok,
        }
        receipt = {
            "passed": all(checks.values()),
            "checks": checks,
            "protocol_sha256": p["protocol_sha256"],
            "initial_identity": p["initial_identity"],
            "objective_relative": objective_relative,
            "objective_relative_reason": None
            if objective_relative is not None
            else "zero_cpu64_norm",
            "objective_absolute": objective_delta,
            "gradient_relative_L2": gradient_relative,
            "gradient_relative_reason": None
            if gradient_relative is not None
            else "zero_cpu64_norm",
            "gradient_cpu64_norm": gnorm,
            "objective_cpu64_norm": abs(objective_b),
            "gradient_max_abs_delta": float(np.abs(ga - gb).max()),
            "residual_max_abs_delta_by_component": np.abs(
                a[0]["residuals"] - b[0]["residuals"]
            )
            .max(0)
            .tolist(),
            "named_parameter_order": list(a[1]),
            "residual_component_comparisons": residual_comparison,
            "criteria": p["precision_criteria"],
            "files": {name: sha(output / name) for name in ("raw32.npz", "raw64.npz")},
            "limits": (
                "Single arithmetic/autograd comparison, not trajectory or phy"
                "sical accuracy."
            ),
        }
        write(output / "receipt.json", receipt)
        if not receipt["passed"]:
            raise Experiment2DError("frozen CPU32/64 check failed", receipt)
        return receipt
    except BaseException as exc:
        write(
            output / "failure.json",
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "receipt": getattr(exc, "receipt", None),
            },
        )
        raise


def checkpoint_payload(model, optimizer, step, score, protocol, initial_identity):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "validation_score": score,
        "protocol_sha256": protocol["protocol_sha256"],
        "sample_sha256": protocol["membership_sha256"],
        "initial_identity": initial_identity,
        "torch_rng_state": torch.get_rng_state(),
    }


def load_checkpoint(path, protocol):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        payload["protocol_sha256"] != protocol["protocol_sha256"]
        or payload["sample_sha256"] != protocol["membership_sha256"]
        or payload["initial_identity"] != protocol["initial_identity"]
    ):
        raise Experiment2DError("checkpoint provenance mismatch")
    if any(
        value.dtype != torch.float32
        or value.device.type != "cpu"
        or not bool(torch.isfinite(value).all())
        for value in payload["model"].values()
    ):
        raise Experiment2DError("checkpoint must contain finite CPU32 tensors")
    model = new_model()
    model.load_state_dict(payload["model"])
    return model


def verify_training_readback(run_dir, protocol, summary):
    run = Path(run_dir)
    if (
        sha(run / "history.csv") != summary["history_sha256"]
        or sha(run / "checkpoint.pt") != summary["checkpoint_sha256"]
        or sha(run / "initial.pt") != summary["initial_checkpoint_sha256"]
        or sha(run / "latest.pt") != summary["latest_checkpoint_sha256"]
    ):
        raise Experiment2DError("saved training hash changed")
    with (run / "history.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    n = summary["actual_updates"]
    if [int(row["step"]) for row in rows] != list(range(n + 1)):
        raise Experiment2DError("history must include initial and everyupdate")
    best, selected, writes = math.inf, None, 0
    for row in rows:
        score = float(row["val_objective"])
        flag = score < best
        if not math.isfinite(score) or row["selected"] != str(flag):
            raise Experiment2DError("history selection is not earliest strictminimum")
        if flag:
            best, selected, writes = score, int(row["step"]), writes + 1
    if (
        summary["checkpoint_write_counts"]
        != {"initial": 1, "latest": n + 1, "selected": writes}
        or summary["selected_step"] != selected
        or summary["selected_validation_score"] != best
    ):
        raise Experiment2DError("selection/writecount summary mismatch")
    for name, step in (
        ("initial.pt", 0),
        ("latest.pt", n),
        ("checkpoint.pt", selected),
    ):
        saved = torch.load(run / name, map_location="cpu", weights_only=True)
        load_checkpoint(run / name, protocol)
        if saved["step"] != step or saved["validation_score"] != float(
            rows[step]["val_objective"]
        ):
            raise Experiment2DError("checkpoint step/score mismatch")
        counts = [
            float(value["step"]) for value in saved["optimizer"]["state"].values()
        ]
        if (step == 0 and counts) or (
            step and (len(counts) != 10 or any(value != step for value in counts))
        ):
            raise Experiment2DError("Adam update counters mismatch")
        for state in saved["optimizer"]["state"].values():
            if any(
                state[key].dtype != torch.float32
                or state[key].device.type != "cpu"
                or not bool(torch.isfinite(state[key]).all())
                for key in ("exp_avg", "exp_avg_sq")
            ):
                raise Experiment2DError("Adam moments must be finite CPU32 tensors")
    if (
        model_identity(torch.load(run / "initial.pt", weights_only=True)["model"])
        != protocol["initial_identity"]
    ):
        raise Experiment2DError("initial parameter tensors changed")
    if (
        model_identity(torch.load(run / "checkpoint.pt", weights_only=True)["model"])
        != summary["selected_model_identity"]
    ):
        raise Experiment2DError("selected parameter identity differs from summary")
    return {
        "passed": True,
        "history_rows": len(rows),
        "selected_step": selected,
        "checkpoint_counts": summary["checkpoint_write_counts"],
    }


def run_experiment(
    config_path,
    run_id,
    *,
    protocol_path=None,
    release_path=None,
    prepare_only=False,
    root=Path("."),
):
    """Prepare once, or consume prepared inputs for pilot/future released fullfit."""
    from hydro_pinn.extensions.evaluate_2d import run_raw_queries

    started = time.perf_counter()
    root = Path(root).resolve()
    validate_run_id(run_id)
    if not run_id.startswith("p10_genuine_2d_"):
        raise Experiment2DError("fresh genuine2D run namespace required")
    cfg = load_config(config_path, root)
    if prepare_only:
        if protocol_path is not None or release_path is not None:
            raise Experiment2DError("prepare-only cannot consume protocol/release")
        directory = root / "runs" / run_id
        p = prepare_protocol(config_path, directory, root=root)
        try:
            precision = precision_check(
                directory / "protocol.json", directory / "precision", root=root
            )
        except BaseException as exc:
            write(
                directory / "preparation_receipt.json",
                {
                    "state": "failed",
                    "passed": False,
                    "protocol_sha256": p["protocol_sha256"],
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "elapsed_s": time.perf_counter() - started,
                },
            )
            raise
        write(
            directory / "preparation_receipt.json",
            {
                "state": "completed",
                "passed": True,
                "protocol_sha256": p["protocol_sha256"],
                "precision_receipt_sha256": sha(directory / "precision/receipt.json"),
                "precision_files": precision["files"],
                "setup_and_precision_wall_s": time.perf_counter() - started,
                "memory": memory_observation(),
            },
        )
        return directory
    if protocol_path is None:
        raise Experiment2DError("prepare-only frozen protocol required before pilot")
    protocol_path = root / protocol_path
    p = verify_protocol(protocol_path, root)
    if p["config"] != cfg:
        raise Experiment2DError("config differs from frozen study")
    prepared = read(protocol_path.parent / "preparation_receipt.json")
    precision_receipt = read(protocol_path.parent / "precision/receipt.json")
    if (
        prepared.get("passed") is not True
        or prepared["protocol_sha256"] != p["protocol_sha256"]
        or prepared["precision_receipt_sha256"]
        != sha(protocol_path.parent / "precision/receipt.json")
        or precision_receipt["passed"] is not True
        or precision_receipt["protocol_sha256"] != p["protocol_sha256"]
        or precision_receipt["initial_identity"] != p["initial_identity"]
        or prepared["precision_files"] != precision_receipt["files"]
        or any(
            sha(protocol_path.parent / "precision" / name) != expected
            for name, expected in precision_receipt["files"].items()
        )
    ):
        raise Experiment2DError("passed immutable preparation/precision required")
    full = release_path is not None
    if full:
        release = read(root / release_path)
        if (
            release.get("state") != "released"
            or release.get("protocol_sha256") != p["protocol_sha256"]
            or release.get("run_id") != run_id
            or release.get("updates") != 10000
        ):
            raise Experiment2DError("exact future full10000 release required")
    else:
        release = None
        selection = read(root / CONTRACT_PATHS[0])
        if selection["state"] != (
            "released_additive_implementation_necessary_checks_and_one25_"
            "update_fullcost_pilot_only"
        ):
            raise Experiment2DError("single25pilot selection unavailable")
    write(
        protocol_path.parent
        / ("full_launch_claim.json" if full else "pilot_launch_claim.json"),
        {
            "run_id": run_id,
            "protocol_sha256": p["protocol_sha256"],
            "mode": "full" if full else "pilot",
        },
    )
    steps = 10000 if full else 25
    threads = configure_threads()
    resolved = {
        "case": read(root / cfg["case_path"]),
        "extension": cfg,
        "request": {"run_id": run_id, "seed": 11, "device": "cpu", "dtype": "float32"},
    }
    with RunRecord.create(root / "runs", run_id, resolved, project_root=root) as record:
        run = record.path
        for name in p["artifact_bindings"]:
            shutil.copyfile(protocol_path.parent / name, run / name)
        write(run / "protocol.json", p)
        write(run / "preparation_receipt.json", prepared)
        shutil.copytree(protocol_path.parent / "precision", run / "precision")
        metadata = read(run / "metadata.json")
        metadata.update(
            {
                "dimension": 2,
                "scientific_authority": "physics-only genuine2D neural approximation",
                "scales": p["scales"],
                "numerical_floor": 1e-6,
                "loss_weights": cfg["weights"],
                "thread_setup": threads,
                "reference_provenance": {
                    "kind": "numerical_approximation",
                    "selected": "ref_rect320_c0225_v01",
                    "inputs": p["evaluation_inputs"],
                    "used_for_training_or_checkpoint_selection": False,
                },
                "data_partitions": p["membership_sha256"],
            }
        )
        write_json_atomic(run / "metadata.json", metadata, overwrite=True)
        if release is not None:
            write(run / "root_release.json", release)
        model, optimizer = _initial_model_optimizer(run, p)
        train, val = (
            dict(np.load(run / "training_samples.npz")),
            dict(np.load(run / "validation_samples.npz")),
        )
        setup_s = time.perf_counter() - started
        best, selected_step = math.inf, 0
        saves = {"initial": 0, "latest": 0, "selected": 0}
        timings, memory = [], [{"stage": "beforetraining", **memory_observation()}]
        training_started = time.perf_counter()
        with (run / "history.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "step",
                    *[
                        prefix + key
                        for prefix in ("train_", "val_")
                        for key in LOSS_KEYS
                    ],
                    "selected",
                    "objective_s",
                    "backward_s",
                    "optimizer_s",
                    "validation_selection_s",
                    "checkpoint_io_s",
                ],
            )
            writer.writeheader()
            for step in range(steps + 1):
                cycle = time.perf_counter()
                now = time.perf_counter()
                losses = component_losses(model, train)
                training = _numbers(losses)
                clock = {
                    "objective_s": time.perf_counter() - now,
                    "backward_s": 0.0,
                    "optimizer_s": 0.0,
                }
                if step:
                    now = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    losses["objective"].backward()
                    if any(
                        parameter.grad is None
                        or not bool(torch.isfinite(parameter.grad).all())
                        for parameter in model.parameters()
                    ):
                        raise RuntimeError("nonfinite/absent gradient")
                    clock["backward_s"] = time.perf_counter() - now
                    now = time.perf_counter()
                    optimizer.step()
                    clock["optimizer_s"] = time.perf_counter() - now
                now = time.perf_counter()
                validation = _numbers(component_losses(model, val))
                score = validation["objective"]
                selected = select_checkpoint(score, best)
                if selected:
                    best, selected_step = score, step
                clock["validation_selection_s"] = time.perf_counter() - now
                now = time.perf_counter()
                payload = checkpoint_payload(
                    model, optimizer, step, score, p, p["initial_identity"]
                )
                if step == 0:
                    torch.save(payload, run / "initial.pt")
                    saves["initial"] += 1
                torch.save(payload, run / "latest.pt")
                saves["latest"] += 1
                if selected:
                    torch.save(payload, run / "checkpoint.pt")
                    saves["selected"] += 1
                clock["checkpoint_io_s"] = time.perf_counter() - now
                now = time.perf_counter()
                writer.writerow(
                    {
                        "step": step,
                        **{"train_" + key: value for key, value in training.items()},
                        **{"val_" + key: value for key, value in validation.items()},
                        "selected": selected,
                        **clock,
                    }
                )
                handle.flush()
                clock["history_io_s"] = time.perf_counter() - now
                clock["complete_cycle_s"] = time.perf_counter() - cycle
                timings.append({"step": step, **clock})
        train_s = time.perf_counter() - training_started
        memory.append({"stage": "aftertraining", **memory_observation()})
        reload_start = time.perf_counter()
        selected_model = load_checkpoint(run / "checkpoint.pt", p)
        latest = load_checkpoint(run / "latest.pt", p)
        with torch.no_grad():
            probe = execution_coordinates(val["ic_si64"][:16])
            latest_exact = torch.equal(model(probe), latest(probe))
        reloaded_score = _numbers(component_losses(selected_model, val))["objective"]
        if not latest_exact or reloaded_score != best:
            raise RuntimeError("trusted selected/latest reload mismatch")
        reload_s = time.perf_counter() - reload_start
        query_start = time.perf_counter()
        queries = run_raw_queries(selected_model, run, run / "raw_queries")
        query_s = time.perf_counter() - query_start
        if queries["actual_query_count"] != 13639680:
            raise RuntimeError("raw query count differs from frozen complete workload")
        memory.append({"stage": "afterquery", **memory_observation()})
        cycles = np.array([row["complete_cycle_s"] for row in timings[1:]])
        fixed = (
            setup_s + timings[0]["complete_cycle_s"] + 2 * (reload_s + query_s) + 120.0
        )
        projections = {
            name: float(10000 * value + fixed)
            for name, value in (
                ("mean", cycles.mean()),
                ("median", np.median(cycles)),
                ("observed_max", cycles.max()),
            )
        }
        summary = {
            "state": "completed",
            "mode": "full" if full else "pilot",
            "actual_updates": steps,
            "validation_evaluations": steps + 1,
            "selected_step": selected_step,
            "selected_validation_score": best,
            "selected_model_identity": model_identity(selected_model),
            "initial_identity": p["initial_identity"],
            "checkpoint_write_counts": saves,
            "protocol_sha256": p["protocol_sha256"],
            "protocol_directory": str(protocol_path.parent.relative_to(root)),
            "checkpoint_sha256": sha(run / "checkpoint.pt"),
            "initial_checkpoint_sha256": sha(run / "initial.pt"),
            "latest_checkpoint_sha256": sha(run / "latest.pt"),
            "history_sha256": sha(run / "history.csv"),
            "trusted_selected_reload_score_exact": reloaded_score == best,
            "latest_reload_prediction_exact": bool(latest_exact),
            "raw_queries": queries,
            "raw_query_wall_s": query_s,
            "reload_wall_s": reload_s,
            "training_wall_s": train_s,
            "setup_wall_s": setup_s,
            "step0_wall_s": timings[0]["complete_cycle_s"],
            "raw_cycle_timings_s": timings,
            "threads": threads,
            "memory": memory,
            "projections_complete_new_full_s": projections,
            "projection_equation": (
                "10000*completecycle statistic +setup+step0 +2*(reload+rawque"
                "ryIO) +120unmeasuredone-timeallowance"
            ),
            "decision_budget_s": 1200,
            "limits": (
                "Mean/median/max are25step proxies, not bounds; selectedwrite"
                " frequency/full sustainedload may differ.120s allowance unme"
                "asured finalanalysis/startup/figures/archive/extraction/comp"
                "arison. No pilot neural quality screening."
            ),
            "analytical_or_numeric_model_errors_computed": False,
            "observed_outer_wall_s_before_summary": time.perf_counter() - started,
            "output_bytes_before_summary": sum(
                path.stat().st_size for path in run.rglob("*") if path.is_file()
            ),
        }
        summary["training_readback"] = verify_training_readback(run, p, summary)
        write(run / "summary.json", summary)
        write(
            run / "hashes.json",
            {
                "files": {
                    str(path.relative_to(run)): sha(path)
                    for path in run.rglob("*")
                    if path.is_file()
                    and path.name not in ("status.json", "metadata.json", "hashes.json")
                }
            },
        )
        record.complete(
            stopping_reason="declared full10000 updates and rawquery"
            if full
            else "single25pilot and fullrawquery",
            iterations=steps,
            elapsed_timings_s={
                "training": train_s,
                "reload": reload_s,
                "raw_query": query_s,
                "entry_before_terminal": time.perf_counter() - started,
            },
        )
        return run
