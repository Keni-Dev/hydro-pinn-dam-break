"""Frozen one-factor viscosity gate and separately released full experiment.

The historical zero control is an explicitly hash-bound input, not a rerun or
a retagged old study. Pilot queries save raw arrays for cost only; scientific
accuracy, fronts and mass summaries require the separate full evaluator.
"""

from __future__ import annotations

import copy
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
from hydro_pinn.conservation import quadrature_mass, quadrature_sensitivity
from hydro_pinn.evaluate import (
    common_grid_fields,
    error_metrics,
    evaluate_fields,
    evaluation_axes,
    metric_definitions,
)
from hydro_pinn.extensions.hybrid_experiment import load_hybrid_checkpoint
from hydro_pinn.extensions.viscosity_physics import (
    NU_STAR,
    WEIGHTS,
    dual_residuals,
    objective_gradients,
    validation,
    viscosity_residuals,
)
from hydro_pinn.losses import residuals
from hydro_pinn.model import ShallowWaterPINN
from hydro_pinn.protocol import _conservation
from hydro_pinn.records import RunRecord, validate_run_id, write_json_atomic
from hydro_pinn.references import wave_diagnostics
from hydro_pinn.sampling import (
    generate_training_samples,
    generate_validation_samples,
    load_samples,
)
from hydro_pinn.train import _cpu_tree, _finite_model, _rng_state, component_gradients

SELECTION_PATH = "docs/coordination/extension_viscosity_selection.json"
PLAN_PATH = "docs/coordination/phase_11_1_viscosity_gate_plan_v01.json"
EXPECTED_PLAN_SHA = "0539decabb36d58b7342716ee2dedc058b8f53e551938ee842c8d5cbc18ba4ca"
INITIAL_IDENTITY = "827083c31cbc04bae49f321d52e0071d9aedcf2f6f5b2384ffa82cb420c26d33"
COMPONENTS = ("pde", "ic", "bc")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    return json.loads(Path(path).read_text(), parse_constant=reject)


def digest(value):
    return hashlib.sha256(
        (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    ).hexdigest()


def initial_hash(model):
    return hashlib.sha256(
        b"".join(p.detach().cpu().numpy().tobytes() for p in model.parameters())
    ).hexdigest()


def initial_model(case, training, dtype="float32", initial_state=None):
    torch.set_num_threads(4)
    torch.manual_seed(11)
    model = ShallowWaterPINN(case, training, device="cpu", dtype=dtype)
    if initial_state is not None:
        model.load_state_dict(initial_state, strict=True)
    return model


def load_config_extension(path, root=Path(".")):
    root = Path(root).resolve()
    cfg = read_json(root / path)
    expected = {
        "seed": 11,
        "device": "cpu",
        "dtype": "float32",
        "cpu_threads": 4,
        "nu_star": 0.001,
        "pilot_updates": 25,
        "full_updates": 10000,
        "query_batch_points": 20000,
        "pilot_quality_analysis": False,
        "no_momentum_budget": True,
    }
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError("settings differ from the adopted viscosity design")
    case, training, evaluation = [
        load_config(root / cfg[k]) for k in ("case", "training", "evaluation")
    ]
    if (
        case["case_id"] != "E1"
        or training["weighting"]["mode"] != "fixed"
        or training["lbfgs"]["enabled"]
    ):
        raise ValueError(
            "requires unchanged frictionless E1/fixed baseline without LBFGS"
        )
    if training["optimizer"] != {
        "kind": "adam",
        "steps": 10000,
        "learning_rate": 0.001,
    }:
        raise ValueError("requires unchanged Adam settings")
    if [training["weighting"][f"lambda_{k}"] for k in COMPONENTS] != [
        1.0,
        10.0,
        10.0,
    ] or training["sampling"]["interior_labels"]:
        raise ValueError("requires physics-only fixed1/10/10")
    return cfg, case, training, evaluation


def load_historical(root=Path(".")):
    root = Path(root).resolve()
    selected = read_json(root / SELECTION_PATH)
    for mapping in (
        selected["protected_existing_nonCLI_sha256"],
        selected["historical_control"]["binding_sha256"],
    ):
        for filename, expected in mapping.items():
            if sha(root / filename) != expected:
                raise ValueError(
                    f"immutable historical/shared source mismatch: {filename}"
                )
    checkpoint, resolved, directory = load_hybrid_checkpoint(
        "p11_hybrid_full_v01_control", root
    )
    if (
        checkpoint["step"] != 10000
        or checkpoint["selected_step"] != 6512
        or resolved["extension"]["lambda_data"] != 0
    ):
        raise ValueError("historical zero input identity mismatch")
    return checkpoint, resolved, directory


def build_axes(evaluation, evaluation_sha):
    x, times = evaluation_axes(evaluation)
    aux = _conservation([], evaluation, evaluation_sha)
    return {
        "primary": {"x_m": x.tolist(), "t_s": times.tolist()},
        "volume_axes": aux["volume_axes"],
        "boundary_axes": aux["boundary_axes"],
    }


def compare_values(candidate, reference, *, rtol, atol):
    a, b = (
        np.asarray(candidate, dtype=np.float64),
        np.asarray(reference, dtype=np.float64),
    )
    difference = a - b
    return {
        "passed": bool(np.allclose(a, b, rtol=rtol, atol=atol)),
        "maximum_absolute_difference": float(np.max(np.abs(difference))),
        "l2_difference": float(np.linalg.norm(difference)),
        "norm_scaled_l2_difference": float(
            np.linalg.norm(difference) / max(np.linalg.norm(b), 1e-12)
        ),
        "reference_l2_norm": float(np.linalg.norm(b)),
        "rtol": rtol,
        "atol": atol,
    }


def precision_report(case, training, state, validation_samples):
    indices = [k * 364 // 15 for k in range(16)]
    small = {
        key: value.astype(np.float32).astype(np.float64)
        for key, value in validation_samples.items()
    }
    small["interior_xt_nd"] = small["interior_xt_nd"][indices]
    models = {
        dtype: initial_model(case, training, dtype, state)
        for dtype in ("float32", "float64")
    }
    results = {}
    for dtype, model in models.items():
        coords = torch.tensor(
            small["interior_xt_nd"], dtype=next(model.parameters()).dtype
        )
        dual = dual_residuals(model, coords)
        objective, _, _, gradients = objective_gradients(model, small, NU_STAR)
        results[dtype] = {
            "dual": {k: v.detach().numpy() for k, v in dual.items()},
            "objective": np.array([objective]),
            "gradients": [g.detach().numpy() for g in gradients],
        }
    checks = {
        key: compare_values(
            results["float32"]["dual"][key],
            results["float64"]["dual"][key],
            rtol=5e-4,
            atol=5e-6,
        )
        for key in results["float32"]["dual"]
    }
    checks["objective"] = compare_values(
        results["float32"]["objective"],
        results["float64"]["objective"],
        rtol=1e-3,
        atol=1e-6,
    )
    gradient_checks = [
        compare_values(a, b, rtol=1e-3, atol=1e-6)
        for a, b in zip(
            results["float32"]["gradients"],
            results["float64"]["gradients"],
            strict=True,
        )
    ]
    return {
        "passed": all(v["passed"] for v in checks.values())
        and all(v["passed"] for v in gradient_checks),
        "indices": indices,
        "fixture_counts": {"pde": 16, "ic": 72, "bc": 10},
        "input_cast": (
            "same float32-representable initial weights/coordinates/targets "
            "cast to float64"
        ),
        "checks": checks,
        "every_parameter_gradient": gradient_checks,
    }


def zero_equivalence_report(case, training, state, validation_samples):
    small = copy.deepcopy(validation_samples)
    small["interior_xt_nd"] = small["interior_xt_nd"][
        [k * 364 // 15 for k in range(16)]
    ]
    outputs = {}
    for dtype in ("float32", "float64"):
        old, new = [initial_model(case, training, dtype, state) for _ in range(2)]
        coords = torch.tensor(
            small["interior_xt_nd"], dtype=next(old.parameters()).dtype
        )
        residual_equal = torch.equal(
            residuals(old, coords), viscosity_residuals(new, coords, 0.0)
        )
        values, component_g = {}, {}
        for name in COMPONENTS:
            values[name], component_g[name] = component_gradients(
                old, small, name, None
            )
        total = sum(WEIGHTS[k] * values[k] for k in COMPONENTS)
        old_gradients = [
            sum(WEIGHTS[k] * component_g[k][i] for k in COMPONENTS)
            for i, _ in enumerate(old.parameters())
        ]
        new_total, new_values, _, new_gradients = objective_gradients(new, small, 0.0)
        per_gradient = [
            bool(torch.equal(a, b))
            for a, b in zip(old_gradients, new_gradients, strict=True)
        ]
        optimizers = [
            torch.optim.Adam(model.parameters(), lr=0.001) for model in (old, new)
        ]
        for model, optimizer, gradients in zip(
            (old, new), optimizers, (old_gradients, new_gradients), strict=True
        ):
            optimizer.zero_grad(set_to_none=True)
            for p, gradient in zip(model.parameters(), gradients, strict=True):
                p.grad = gradient
            optimizer.step()
        parameters_equal = all(
            torch.equal(a, b)
            for a, b in zip(old.parameters(), new.parameters(), strict=True)
        )
        optimizer_equal = all(
            torch.equal(a[key], b[key])
            for a, b in zip(
                optimizers[0].state.values(), optimizers[1].state.values(), strict=True
            )
            for key in a
        )
        outputs[dtype] = {
            "residual_exact": bool(residual_equal),
            "component_values_exact": values == new_values,
            "objective_exact": total == new_total,
            "objective": total,
            "every_parameter_gradient_exact": per_gradient,
            "one_adam_parameters_exact": parameters_equal,
            "all_adam_moments_counts_exact": optimizer_equal,
        }
        outputs[dtype]["passed"] = (
            residual_equal
            and values == new_values
            and total == new_total
            and all(per_gradient)
            and parameters_equal
            and optimizer_equal
        )
    return {
        "passed": all(value["passed"] for value in outputs.values()),
        "dtypes": outputs,
        "nu_zero_delegates_core": True,
        "fixture_counts": {"pde": 16, "ic": 72, "bc": 10},
    }


def observations():
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    try:
        load = list(os.getloadavg())
    except OSError:
        load = None
    return {
        "process_lifetime_peak_rss_bytes": int(
            peak * (1 if platform.system() == "Darwin" else 1024)
        ),
        "host_load_1_5_15_min": load,
        "current_rss_bytes": None,
        "limits": (
            "Lifetime peak, not instantaneous/free RAM or pressure. Current "
            "RSS/process inventory not probed again after prior sandbox ps "
            "denials; load averages are host observations, not per-process "
            "CPU or thermal telemetry."
        ),
    }


def prepare_protocol(config_path, root=Path(".")):
    root = Path(root).resolve()
    cfg, case, training, evaluation = load_config_extension(config_path, root)
    historical, historical_resolved, old_path = load_historical(root)
    if (
        case != historical_resolved["case"]
        or training != historical_resolved["training"]
    ):
        raise ValueError("case/training differ from the reused control")
    selected = read_json(root / SELECTION_PATH)
    if sha(root / PLAN_PATH) != EXPECTED_PLAN_SHA:
        raise ValueError("adopted plan identity mismatch")
    model = initial_model(case, training)
    if initial_hash(model) != INITIAL_IDENTITY:
        raise ValueError("fresh seed11 initial tensors differ")
    samples = load_samples(old_path / "sampling.npz")
    validation_samples = load_samples(old_path / "validation.npz")
    generated = generate_training_samples(case, training, 11)
    if any(not np.array_equal(generated[k], samples[k]) for k in samples):
        raise ValueError("accepted physics membership differs from current generator")
    if any(
        not np.array_equal(
            generate_validation_samples(case, training, 11)[k], validation_samples[k]
        )
        for k in validation_samples
    ):
        raise ValueError("accepted validation membership differs")
    state = _cpu_tree(model.state_dict())
    zero = zero_equivalence_report(case, training, state, validation_samples)
    precision = precision_report(case, training, state, validation_samples)
    resolved = {
        "case": case,
        "training": training,
        "evaluation": evaluation,
        "extension": cfg,
        "request": {
            "run_id": cfg["protocol_run_id"],
            "device": "cpu",
            "dtype": "float32",
            "seed": 11,
        },
    }
    with RunRecord.create(
        root / "runs",
        cfg["protocol_run_id"],
        resolved,
        project_root=root,
        metadata={"stage": "viscosity_protocol_freeze", "no_training": True},
    ) as run:
        write_json_atomic(run.path / "zero_equivalence.json", zero)
        write_json_atomic(run.path / "precision.json", precision)
        if not zero["passed"] or not precision["passed"]:
            raise ValueError("zero-equivalence/precision gate failed; pilot withheld")
        torch.save(state, run.path / "initial_state.pt")
        for filename in ("sampling.npz", "validation.npz"):
            shutil.copyfile(old_path / filename, run.path / filename)
        axes = build_axes(evaluation, sha(root / cfg["evaluation"]))
        primary = axes["primary"]
        xx, tt = np.meshgrid(primary["x_m"], primary["t_s"])
        coords = np.column_stack((xx.ravel(), tt.ravel()))
        nd_final = coords / [case["scales"]["Lc_m"], case["scales"]["Tc_s"]]
        primary_membership = {}
        for dtype in (np.float64, np.float32):
            known = {
                tuple(row)
                for partition in (samples, validation_samples)
                for name in ("interior_xt_nd", "initial_xt_nd", "boundary_xt_nd")
                for row in partition[name].astype(dtype)
            }
            final_set = set(map(tuple, nd_final.astype(dtype)))
            if len(final_set) != 20000 or not final_set.isdisjoint(known):
                raise ValueError(
                    "primary final coordinates overlap known physics points"
                )
            primary_membership[np.dtype(dtype).name] = {
                "unique_final_coordinates": len(final_set),
                "disjoint_known_physics_points": True,
            }
        physical32 = torch.tensor(coords, dtype=torch.float32)
        actual_query_nd = torch.stack(
            (
                model.scales.x_to_scaled(physical32[:, 0]),
                model.scales.t_to_scaled(physical32[:, 1]),
            ),
            dim=1,
        ).numpy()
        actual_set = set(map(tuple, actual_query_nd))
        known32 = {
            tuple(row)
            for partition in (samples, validation_samples)
            for name in ("interior_xt_nd", "initial_xt_nd", "boundary_xt_nd")
            for row in partition[name].astype(np.float32)
        }
        if len(actual_set) != 20000 or not actual_set.isdisjoint(known32):
            raise ValueError("actual float32 queries overlap known physics points")
        primary_membership["actual_float32_physical_then_scale"] = {
            "unique_final_coordinates": len(actual_set),
            "disjoint_known_physics_points": True,
        }
        write_json_atomic(run.path / "axes.json", axes)
        evidence = read_json(
            root / "docs/evidence/phase_11/viscosity_v01/axis_interface_review.json"
        )
        reference = {
            "primary": {
                "path": cfg["reference_primary"],
                "sha256": evidence["saved_primary_reference"]["files_sha256"][
                    "outputs.npz"
                ],
            }
        }
        if sha(root / reference["primary"]["path"]) != reference["primary"]["sha256"]:
            raise ValueError("saved inviscid reference identity mismatch")
        with np.load(root / reference["primary"]["path"], allow_pickle=False) as arrays:
            if not np.array_equal(arrays["x_m"], primary["x_m"]) or not np.array_equal(
                arrays["t_s"], primary["t_s"]
            ):
                raise ValueError("saved reference axes mismatch")
        current_sources = {
            str(p.relative_to(root)): sha(p)
            for p in sorted((root / "src/hydro_pinn").rglob("*.py"))
        }
        inputs = {
            str(p): sha(root / p)
            for p in (
                str(config_path),
                cfg["case"],
                cfg["training"],
                cfg["evaluation"],
                SELECTION_PATH,
                PLAN_PATH,
                "docs/coordination/phase_11_1_historical_control_reuse_note.json",
            )
        }
        artifacts = {
            str(p.relative_to(root)): sha(p)
            for p in sorted(run.path.iterdir())
            if p.name not in ("metadata.json", "status.json")
        }
        protocol = {
            "schema_version": 1,
            "purpose": "fixed_artificial_momentum_diffusion_E1_single_seed",
            "config": cfg,
            "case": case,
            "training": training,
            "evaluation": evaluation,
            "historical": selected["historical_control"],
            "source_sha256": current_sources,
            "input_sha256": inputs,
            "preflight_artifacts_sha256": artifacts,
            "initial_state": str((run.path / "initial_state.pt").relative_to(root)),
            "initial_parameter_byte_identity": INITIAL_IDENTITY,
            "physics": {
                "nu_star": NU_STAR,
                "dimensional_nu_m2_s": NU_STAR
                * case["scales"]["Lc_m"]
                * case["scales"]["Uc_m_s"],
                "momentum_residual": "r_q - nu_star*q_star_xstarxstar",
                "continuity_unchanged": True,
            },
            "primary_partition_checks": primary_membership,
            "partitions_run": cfg["protocol_run_id"],
            "axes": axes,
            "reference": reference,
            "zero_cache_run": cfg["zero_cache_run"],
            "selection": cfg["selection"],
            "cast_policy": (
                "Saved coordinates/targets float64; explicit torch cast "
                "to model float32 for full work."
            ),
            "auxiliary_membership": (
                "IC/BC mass probes are physics diagnostics; shared imposed "
                "coordinates may repeat. No filtering/moving."
            ),
            "pilot_quality_analysis": False,
            "scientific_limits": [
                "Project-specified momentum regularization; inviscid analytical "
                "comparison is not an exact viscous solution.",
                "Soft reservoir penalties do not establish a well-posed viscous BVP.",
                "No entropy, energy-dissipation or unconditional-stability guarantee.",
                "Momentum budgets withheld: diffusive boundary flux would be required.",
            ],
        }
        write_json_atomic(run.path / "protocol.json", protocol)
        run.complete(
            stopping_reason=(
                "source/control/initialization/partitions/axes and checks "
                "frozen before pilot"
            )
        )
    return run.path / "protocol.json"


def check_protocol(protocol, root):
    for name in ("source_sha256", "input_sha256", "preflight_artifacts_sha256"):
        for filename, expected in protocol[name].items():
            if sha(root / filename) != expected:
                raise ValueError(f"frozen {name} mismatch: {filename}")
    for filename, expected in protocol["historical"]["binding_sha256"].items():
        if sha(root / filename) != expected:
            raise ValueError(f"historical control binding changed: {filename}")
    binding = protocol["reference"]["primary"]
    if sha(root / binding["path"]) != binding["sha256"]:
        raise ValueError("bound saved analytical reference changed")


def _checkpoint_save(directory, checkpoint):
    temporary = directory / "checkpoint.pt.tmp"
    torch.save(_cpu_tree(checkpoint), temporary)
    temporary.replace(directory / "checkpoint.pt")
    target = directory / "checkpoint_manifest.json"
    write_json_atomic(
        target,
        {
            "run_id": checkpoint["run_id"],
            "sha256": sha(directory / "checkpoint.pt"),
            "step": checkpoint["step"],
            "selected_step": checkpoint["selected_step"],
            "resolved_config_sha256": checkpoint["resolved_config_sha256"],
            "partition_hashes": checkpoint["partition_hashes"],
        },
        overwrite=target.exists(),
    )


def load_viscosity_checkpoint(run_id, root=Path(".")):
    validate_run_id(run_id)
    directory = Path(root).resolve() / "runs" / run_id
    status, resolved, binding = [
        read_json(directory / name)
        for name in ("status.json", "resolved_config.json", "checkpoint_manifest.json")
    ]
    if status["state"] != "completed" or binding["sha256"] != sha(
        directory / "checkpoint.pt"
    ):
        raise ValueError("requires terminal complete hash-bound viscosity checkpoint")
    checkpoint = torch.load(
        directory / "checkpoint.pt", weights_only=True, map_location="cpu"
    )
    if (
        checkpoint["resolved_config_sha256"] != sha(directory / "resolved_config.json")
        or checkpoint["run_id"] != run_id
        or binding["run_id"] != run_id
        or binding["step"] != checkpoint["step"]
        or binding["selected_step"] != checkpoint["selected_step"]
        or checkpoint["protocol_sha256"] != resolved["protocol_sha256"]
    ):
        raise ValueError("checkpoint configuration/identity mismatch")
    for name, expected in checkpoint["partition_hashes"].items():
        if (
            sha(directory / name) != expected
            or binding["partition_hashes"][name] != expected
        ):
            raise ValueError("checkpoint membership changed")
    return checkpoint, resolved, directory


def fit_positive(protocol, run_id, mode, root=Path(".")):
    root = Path(root).resolve()
    check_protocol(protocol, root)
    setup = time.perf_counter()
    case, training, cfg = protocol["case"], protocol["training"], protocol["config"]
    state = torch.load(
        root / protocol["initial_state"], weights_only=True, map_location="cpu"
    )
    model = initial_model(case, training, initial_state=state)
    if initial_hash(model) != INITIAL_IDENTITY:
        raise ValueError("positive fit must start from fresh common initial tensors")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    partition_dir = root / "runs" / protocol["partitions_run"]
    samples, valid = (
        load_samples(partition_dir / "sampling.npz"),
        load_samples(partition_dir / "validation.npz"),
    )
    steps = cfg["pilot_updates"] if mode == "pilot" else cfg["full_updates"]
    resolved = {
        "case": case,
        "training": training,
        "evaluation": protocol["evaluation"],
        "request": {"run_id": run_id, "seed": 11, "device": "cpu", "dtype": "float32"},
        "extension": {
            "kind": "viscosity",
            "mode": mode,
            "nu_star": NU_STAR,
            "steps": steps,
        },
        "protocol_sha256": digest(protocol),
        "initial_parameter_byte_identity": INITIAL_IDENTITY,
        "execution_settings": {"cpu_threads": 4, "chunk_size": None},
    }
    with RunRecord.create(
        root / "runs",
        run_id,
        resolved,
        project_root=root,
        metadata={
            "stage": "viscosity_positive_fit",
            "physics_only": True,
            "selection": protocol["selection"],
            "no_interior_labels": True,
        },
    ) as run:
        for filename in ("sampling.npz", "validation.npz"):
            shutil.copyfile(partition_dir / filename, run.path / filename)
        partitions = {
            name: sha(run.path / name) for name in ("sampling.npz", "validation.npz")
        }
        first = validation(model, valid, NU_STAR)
        best_score, best_step, best_state = (
            first["modified_total"],
            0,
            _cpu_tree(model.state_dict()),
        )
        measures, step = [], 0
        setup_s = time.perf_counter() - setup

        def checkpoint():
            return {
                "run_id": run_id,
                "schema_version": 1,
                "kind": "viscosity",
                "step": step,
                "selected_step": best_step,
                "selected_score": best_score,
                "latest_model_state": model.state_dict(),
                "selected_model_state": best_state,
                "optimizer_state": optimizer.state_dict(),
                "rng_state": _rng_state("cpu"),
                "initial_parameter_byte_identity": INITIAL_IDENTITY,
                "protocol_sha256": digest(protocol),
                "partition_hashes": partitions,
                "resolved_config_sha256": sha(run.path / "resolved_config.json"),
            }

        _checkpoint_save(run.path, checkpoint())
        history_fields = [
            "step",
            "loss_pde",
            "loss_ic",
            "loss_bc",
            "loss_total",
            "grad_pde",
            "grad_ic",
            "grad_bc",
            *first.keys(),
            "selected_step",
            "elapsed_s",
        ]
        training_started = time.perf_counter()
        with (run.path / "history.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=history_fields)
            writer.writeheader()
            writer.writerow({"step": 0, **first, "selected_step": 0, "elapsed_s": 0.0})
            handle.flush()
            try:
                for step in range(1, steps + 1):
                    started = time.perf_counter()
                    total, values, norms, gradients = objective_gradients(
                        model, samples, NU_STAR
                    )
                    gradient_done = time.perf_counter()
                    optimizer.zero_grad(set_to_none=True)
                    for parameter, gradient in zip(
                        model.parameters(), gradients, strict=True
                    ):
                        parameter.grad = gradient
                    optimizer.step()
                    _finite_model(model)
                    adam_done = time.perf_counter()
                    scores = validation(model, valid, NU_STAR)
                    if scores["modified_total"] < best_score:
                        best_score, best_step, best_state = (
                            scores["modified_total"],
                            step,
                            _cpu_tree(model.state_dict()),
                        )
                    validation_done = time.perf_counter()
                    _checkpoint_save(run.path, checkpoint())
                    writer.writerow(
                        {
                            "step": step,
                            **{f"loss_{k}": v for k, v in values.items()},
                            "loss_total": total,
                            **{f"grad_{k}": v for k, v in norms.items()},
                            **scores,
                            "selected_step": best_step,
                            "elapsed_s": time.perf_counter() - training_started,
                        }
                    )
                    handle.flush()
                    finished = time.perf_counter()
                    measures.append(
                        {
                            "step": step,
                            "objective_second_derivative_backward_s": gradient_done
                            - started,
                            "adam_s": adam_done - gradient_done,
                            "dual_validation_s": validation_done - adam_done,
                            "checkpoint_hash_history_flush_s": finished
                            - validation_done,
                            "inclusive_step_s": finished - started,
                            **observations(),
                        }
                    )
            finally:
                write_json_atomic(
                    run.path / "step_measurements.json", {"observations": measures}
                )
        wall = time.perf_counter() - training_started
        write_json_atomic(
            run.path / "training_diagnostics.json",
            {
                "updates": step,
                "validation_calls": step + 1,
                "initial_validation": first,
                "selection": protocol["selection"],
                "selected_step": best_step,
                "selected_modified_validation": best_score,
                "setup_s": setup_s,
                "training_inclusive_wall_s": wall,
                "observations": observations(),
                "quality_computed": False,
            },
        )
        run.complete(
            stopping_reason=(
                "fixed budget exhausted; convergence/quality not established"
            ),
            iterations=step,
            elapsed_timings_s={"setup": setup_s, "training_inclusive": wall},
        )
    reload_started = time.perf_counter()
    loaded, _, _ = load_viscosity_checkpoint(run_id, root)
    with (run.path / "history.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    expected_best = min(
        range(len(history)), key=lambda i: float(history[i]["modified_total"])
    )
    if loaded["step"] != steps or loaded["selected_step"] != expected_best:
        raise ValueError("checkpoint/strict minimum readback failed")
    if any(
        not torch.equal(loaded["selected_model_state"][key], value)
        for key, value in best_state.items()
    ):
        raise ValueError("reloaded selected tensors differ from held selected state")
    if any(
        not torch.equal(loaded["latest_model_state"][key], value)
        for key, value in model.state_dict().items()
    ):
        raise ValueError("reloaded latest tensors differ from terminal state")
    if any(
        float(value["step"].item()) != steps
        for value in loaded["optimizer_state"]["state"].values()
    ):
        raise ValueError("optimizer update count mismatch")
    write_json_atomic(
        run.path / "trusted_reload_readback.json",
        {
            "passed": True,
            "reload_s": time.perf_counter() - reload_started,
            "updates": steps,
            "history_rows_including_initial": len(history),
            "selected_step": expected_best,
            "checkpoint_sha256": sha(run.path / "checkpoint.pt"),
            "optimizer_parameter_states": len(loaded["optimizer_state"]["state"]),
            "selected_latest_tensors_exact": True,
        },
    )
    return run.path


def query_checkpoint(protocol, checkpoint_path, output_run_id, nu_star, root=Path(".")):
    root = Path(root).resolve()
    check_protocol(protocol, root)
    started = time.perf_counter()
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path = (
        checkpoint_path if checkpoint_path.is_absolute() else root / checkpoint_path
    )
    if nu_star == 0:
        expected = protocol["historical"]["binding_sha256"][
            protocol["historical"]["run"] + "/checkpoint.pt"
        ]
        if sha(checkpoint_path) != expected:
            raise ValueError("zero queries require the bound historical checkpoint")
    load_started = time.perf_counter()
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    model = initial_model(
        protocol["case"],
        protocol["training"],
        initial_state=checkpoint["selected_model_state"],
    )
    model.eval()
    load_s = time.perf_counter() - load_started
    reference_started = time.perf_counter()
    reference_binding = protocol["reference"]["primary"]
    with np.load(root / reference_binding["path"], allow_pickle=False) as arrays:
        reference_fixture = {
            key: np.array(arrays[key], copy=True)
            for key in ("x_m", "t_s", "h_m", "q_m2_s")
        }
    if reference_fixture["h_m"].shape != (50, 400):
        raise ValueError("reference serialization fixture shape mismatch")
    reference_load_s = time.perf_counter() - reference_started
    resolved = {
        "case": protocol["case"],
        "training": protocol["training"],
        "request": {
            "run_id": output_run_id,
            "seed": 11,
            "device": "cpu",
            "dtype": "float32",
        },
        "query_source": {
            "checkpoint_sha256": sha(checkpoint_path),
            "selected_step": checkpoint["selected_step"],
            "nu_star": nu_star,
            "protocol_sha256": digest(protocol),
        },
    }
    with RunRecord.create(
        root / "runs",
        output_run_id,
        resolved,
        project_root=root,
        metadata={
            "stage": "viscosity_raw_query_cost",
            "quality_computed": False,
            "historical_control_input": nu_star == 0,
        },
    ) as run:
        axes = protocol["axes"]
        groups = {"primary": axes["primary"]}
        groups.update({f"volume_{n}": a for n, a in axes["volume_axes"].items()})
        groups.update({f"boundary_{n}": a for n, a in axes["boundary_axes"].items()})
        stages, count = {}, 0
        for name, axis in groups.items():
            assembly = time.perf_counter()
            x, times = (
                np.asarray(axis["x_m"], dtype=np.float64),
                np.asarray(axis["t_s"], dtype=np.float64),
            )
            xx, tt = np.meshgrid(x, times)
            coordinates = np.column_stack((xx.ravel(), tt.ravel()))
            count += len(coordinates)
            assembly_s = time.perf_counter() - assembly
            tick = time.perf_counter()
            pieces = []
            with torch.no_grad():
                for start in range(
                    0, len(coordinates), protocol["config"]["query_batch_points"]
                ):
                    points = torch.tensor(
                        coordinates[
                            start : start + protocol["config"]["query_batch_points"]
                        ],
                        dtype=torch.float32,
                    )
                    pieces.append(model.physical_state(points).numpy())
            values = np.concatenate(pieces).reshape(len(times), len(x), 2)
            if not np.isfinite(values).all() or np.any(values[..., 0] <= 0):
                raise ValueError("invalid raw-query state")
            query_s = time.perf_counter() - tick
            tick = time.perf_counter()
            np.savez_compressed(
                run.path / f"{name}.npz",
                x_m=x,
                t_s=times,
                h_m=values[..., 0],
                q_m2_s=values[..., 1],
            )
            stages[name] = {
                "coordinates": len(coordinates),
                "coordinate_assembly_s": assembly_s,
                "forward_transfer_s": query_s,
                "compressed_npz_io_s": time.perf_counter() - tick,
            }
        if count != 332004:
            raise ValueError("query workload must be exactly332004state coordinates")
        valid = load_samples(
            root / "runs" / protocol["partitions_run"] / "validation.npz"
        )
        coords = torch.tensor(valid["interior_xt_nd"], dtype=torch.float32)
        tick = time.perf_counter()
        dual = dual_residuals(model, coords)
        arrays = {key: value.detach().numpy() for key, value in dual.items()}
        derivative_s = time.perf_counter() - tick
        tick = time.perf_counter()
        np.savez_compressed(
            run.path / "dual_operator.npz", coords_nd=valid["interior_xt_nd"], **arrays
        )
        derivative_io_s = time.perf_counter() - tick
        # Shape-only serialization fixture. It contains no computed model errors,
        # shock descriptors or mass-quality summaries, even for the zero branch.
        fixture = [
            {
                "branch": branch,
                "time_s": t,
                "field": field,
                "region": region,
                "rmse": None,
                "relative_l2": None,
                "reason": "pilot_quality_not_computed",
            }
            for branch in ("historical_zero", "positive")
            for t in protocol["evaluation"]["times_s"]
            for field in ("h_m", "q_m2_s", "u_m_s")
            for region in ("whole", "wave_band", "outside_wave_band")
        ]
        tick = time.perf_counter()
        write_json_atomic(
            run.path / "quality_shape_fixture.json",
            {
                "rows": fixture,
                "mass": None,
                "front": None,
                "meaning": (
                    "serialization cost fixture only; all quality results withheld"
                ),
            },
        )
        with (run.path / "quality_shape_fixture.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fixture[0]))
            writer.writeheader()
            writer.writerows(fixture)
            handle.flush()
        serialization_s = time.perf_counter() - tick
        tick = time.perf_counter()
        files = {
            p.name: sha(p)
            for p in sorted(run.path.iterdir())
            if p.is_file() and p.name not in ("metadata.json", "status.json")
        }
        hashing_s = time.perf_counter() - tick
        query_manifest = {
            "schema_version": 1,
            "protocol_sha256": digest(protocol),
            "checkpoint_sha256": sha(checkpoint_path),
            "nu_star": nu_star,
            "selected_step": checkpoint["selected_step"],
            "files_sha256": files,
            "quality_computed": False,
            "state_coordinates": count,
            "dual_operator_points": len(coords),
            "axes_stages": stages,
            "cost": {
                "checkpoint_model_reload_s": load_s,
                "saved_reference_fixture_load_s": reference_load_s,
                "dual_operator_s": derivative_s,
                "dual_operator_npz_io_s": derivative_io_s,
                "shape_json_csv_s": serialization_s,
                "hashing_s": hashing_s,
                "inclusive_query_output_s": time.perf_counter() - started,
            },
            "written_bytes_before_manifest": sum(
                p.stat().st_size for p in run.path.iterdir() if p.is_file()
            ),
            "observations": observations(),
            "query_chunk_points": protocol["config"]["query_batch_points"],
            "no_momentum_budget": True,
        }
        write_json_atomic(run.path / "query_manifest.json", query_manifest)
        run.complete(
            stopping_reason=(
                "full raw-query/operator/serialization workload saved "
                "without quality analysis"
            ),
            elapsed_timings_s=query_manifest["cost"],
            iterations=0,
        )
    write_json_atomic(
        run.path / "query_cost_receipt.json",
        {
            "inclusive_through_manifest_and_lifecycle_s": time.perf_counter() - started,
            "scope": (
                "Includes model/reference reload, record setup, all queries, "
                "compressed arrays, shape JSON/CSV, hashing, manifest and "
                "lifecycle writes; excludes this final small receipt write."
            ),
            "total_written_bytes_before_receipt": sum(
                p.stat().st_size for p in run.path.iterdir() if p.is_file()
            ),
            "query_manifest_sha256": sha(run.path / "query_manifest.json"),
            "quality_computed": False,
        },
    )
    return run.path


def run_experiment(
    config_path, manifest_path, protocol_path=None, release_path=None, root=Path(".")
):
    root = Path(root).resolve()
    dispatch_started = time.perf_counter()
    destination = root / manifest_path
    if destination.exists():
        raise ValueError(
            "existing viscosity manifest preserved; duplicate dispatch refused"
        )
    if protocol_path is None:
        if release_path is not None:
            raise ValueError("full release requires an already frozen protocol")
        protocol_path = prepare_protocol(config_path, root)
    else:
        protocol_path = root / protocol_path
    preparation_s = time.perf_counter() - dispatch_started
    protocol = read_json(protocol_path)
    check_protocol(protocol, root)
    if load_config_extension(config_path, root)[0] != protocol["config"]:
        raise ValueError("configuration differs from frozen protocol")
    mode = "pilot"
    if release_path is not None:
        release = read_json(root / release_path)
        if not (
            release.get("positive_full_fit_released") is True
            and release.get("protocol_sha256") == sha(protocol_path)
            and release.get("steps") == 10000
            and release.get("nu_star") == NU_STAR
        ):
            raise ValueError(
                "requires exact hash-bound full-positive technical release"
            )
        cache_path = root / "runs" / protocol["zero_cache_run"] / "query_manifest.json"
        if not cache_path.is_file() or sha(cache_path) != release.get(
            "zero_cache_manifest_sha256"
        ):
            raise ValueError("full release must bind the existing zero cache manifest")
        cache = read_json(cache_path)
        for filename, expected in cache["files_sha256"].items():
            if sha(cache_path.parent / filename) != expected:
                raise ValueError("cached zero array changed before full fitting")
        mode = "full"
    cfg = protocol["config"]
    run_id = cfg["pilot_run_id"] if mode == "pilot" else cfg["positive_full_run_id"]
    if (root / "runs" / run_id).exists():
        raise ValueError("designated run already exists; no favorable pilot repeat")
    manifest = {
        "schema_version": 1,
        "mode": mode,
        "state": "running",
        "protocol": {
            "path": str(protocol_path.relative_to(root)),
            "sha256": sha(protocol_path),
        },
        "positive_run": run_id,
        "zero_query_run": protocol["zero_cache_run"],
        "positive_query_run": None,
        "quality_computed": False,
        "full_fit_released": mode == "full",
        "protocol_preparation_s": preparation_s,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(destination, manifest)
    try:
        positive = fit_positive(protocol, run_id, mode, root)
        manifest["positive_checkpoint_sha256"] = sha(positive / "checkpoint.pt")
        if mode == "pilot":
            query = query_checkpoint(
                protocol, positive / "checkpoint.pt", run_id + "_query", NU_STAR, root
            )
            manifest["positive_query_run"] = query.name
            historical_path = root / protocol["historical"]["run"] / "checkpoint.pt"
            zero = query_checkpoint(
                protocol, historical_path, protocol["zero_cache_run"], 0.0, root
            )
            manifest["zero_query_manifest_sha256"] = sha(zero / "query_manifest.json")
        else:
            cache = root / "runs" / protocol["zero_cache_run"] / "query_manifest.json"
            if not cache.is_file():
                raise ValueError(
                    "bound zero raw cache required; never retrain/requery silently"
                )
            manifest["zero_query_manifest_sha256"] = sha(cache)
        manifest["state"] = "completed"
        manifest["dispatch_inclusive_before_terminal_manifest_s"] = (
            time.perf_counter() - dispatch_started
        )
    except BaseException as exc:
        manifest.update(
            state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            failure=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        write_json_atomic(destination, manifest, overwrite=True)
    return destination


def _ve_path(root, value):
    path = Path(value)
    path = (path if path.is_absolute() else root / path).absolute()
    if (
        ".." in path.parts
        or not path.is_relative_to(root)
        or any(p.is_symlink() for p in (path, *path.parents))
    ):
        raise ValueError(
            "viscosity evidence paths must remain inside root without symlinks"
        )
    return path


def _ve_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def _ve_copy(root, source, destination, bindings):
    source = _ve_path(root, source)
    if not source.is_file():
        raise ValueError(f"missing source file: {source}")
    digest = sha(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, destination.open("xb") as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)
    if sha(source) != digest or sha(destination) != digest:
        raise ValueError("source changed while copying evaluation evidence")
    bindings[str(source.relative_to(root))] = digest


def _ve_query(
    root, run_id, *, protocol_sha256, checkpoint_sha256, selected_step, nu_star
):
    validate_run_id(run_id)
    directory = _ve_path(root, Path("runs") / run_id)
    if read_json(directory / "status.json")["state"] != "completed":
        raise ValueError("query source must be completed")
    manifest = read_json(directory / "query_manifest.json")
    if (
        manifest["protocol_sha256"] != protocol_sha256
        or manifest["checkpoint_sha256"] != checkpoint_sha256
        or manifest["selected_step"] != selected_step
        or manifest["nu_star"] != nu_star
    ):
        raise ValueError("query protocol/checkpoint/selection/branch binding mismatch")
    required = {
        "primary.npz",
        "volume_2000.npz",
        "volume_4000.npz",
        "boundary_1001.npz",
        "boundary_2001.npz",
        "dual_operator.npz",
    }
    if not required.issubset(manifest["files_sha256"]):
        raise ValueError("incomplete frozen query workload")
    for name, digest in manifest["files_sha256"].items():
        path = _ve_path(root, directory / name)
        if sha(path) != digest:
            raise ValueError(f"query file hash mismatch: {name}")
    arrays = {name[:-4]: _ve_npz(directory / name) for name in required}
    return directory, manifest, arrays


def _ve_pooled(case, evaluation, reference, fields, identity):
    exact = common_grid_fields(reference, evaluation)
    predicted = common_grid_fields(fields, evaluation)
    waves = wave_diagnostics(case)
    positions = (
        case["dam_position_m"] + exact["t_s"] * waves["wave_speeds_m_s"]["shock"]
    )
    band = (
        np.abs(exact["x_m"][None, :] - positions[:, None])
        <= evaluation["wave_band_half_width_m"]
    )
    masks = {
        "whole": np.ones_like(band, dtype=bool),
        "wave_band": band,
        "outside_wave_band": ~band,
    }
    wet = exact["h_m"] > evaluation["velocity_wet_threshold_m"]
    valid_velocity = predicted["h_m"] > 0
    exact["u_m_s"] = np.divide(
        exact["q_m2_s"], exact["h_m"], out=np.zeros_like(exact["h_m"]), where=wet
    )
    predicted["u_m_s"] = np.divide(
        predicted["q_m2_s"],
        predicted["h_m"],
        out=np.zeros_like(predicted["h_m"]),
        where=valid_velocity,
    )
    rows = []
    for field in ("h_m", "q_m2_s", "u_m_s"):
        for region, mask in masks.items():
            eligible = mask & wet if field == "u_m_s" else mask
            incomplete = field == "u_m_s" and np.any(eligible & ~valid_velocity)
            values = (
                {
                    key: {
                        "value": None,
                        "reason": "incomplete_candidate_velocity_coverage",
                    }
                    for key in ("rmse", "relative_l2")
                }
                if incomplete
                else error_metrics(predicted[field][eligible], exact[field][eligible])
            )
            selected_count = (
                int((eligible & valid_velocity).sum())
                if field == "u_m_s"
                else int(eligible.sum())
            )
            rows.append(
                {
                    **identity,
                    "case_id": case["case_id"],
                    "scope": "pooled_all_50_positive_times",
                    "field": field,
                    "region": region,
                    **values,
                    "eligible_points": int(eligible.sum()),
                    "selected_points": selected_count,
                    "total_points": int(mask.size),
                    "reference_wet_points": int(wet.sum()),
                    "coverage_fraction": selected_count / mask.size,
                    "eligible_coverage_fraction": float(eligible.sum() / mask.size),
                    "weighting": "equal_space_time_point_weights",
                }
            )
    return rows


def _ve_front(case, evaluation, fields, reference):
    waves = wave_diagnostics(case)
    hstar = waves["wet_star"]["h_m"]
    hright = case["right_depth_m"]
    levels = {
        "10": hright + 0.1 * (hstar - hright),
        "90": hright + 0.9 * (hstar - hright),
    }
    width = evaluation["wave_band_half_width_m"]
    if width != 0.1:
        raise ValueError("front diagnostics require the frozen +/-0.1m shock window")
    x, times = fields["x_m"], fields["t_s"]
    indices, diagnostics = [], []
    for elapsed in (0.1, 0.25, 0.5):
        selected = np.flatnonzero(times == elapsed)
        if len(selected) != 1:
            raise ValueError(
                "profile time must occur exactly once on the frozen primary axis"
            )
        i = int(selected[0])
        indices.append(i)
        shock = case["dam_position_m"] + elapsed * waves["wave_speeds_m_s"]["shock"]
        eligible = np.flatnonzero(np.abs(x - shock) <= width)
        depth = fields["h_m"][i]
        counts, crossings, plateau = {}, {}, False
        for label, level in levels.items():
            positions = []
            for j in eligible[:-1]:
                if j + 1 not in eligible:
                    continue
                a, b = float(depth[j]), float(depth[j + 1])
                plateau |= a == level and b == level
                if a >= level > b:
                    positions.append(
                        float(x[j] + (a - level) / (a - b) * (x[j + 1] - x[j]))
                    )
            counts[label], crossings[label] = len(positions), positions
        slope = None
        if len(eligible) >= 2:
            slope = float(
                np.max(np.abs(np.diff(depth[eligible]) / np.diff(x[eligible])))
            )
        reason = None
        if len(eligible) < 2:
            reason = "fewer_than_two_original_nodes_inside_window"
        elif plateau:
            reason = "threshold_plateau_has_no_unique_crossing"
        elif any(counts[k] > 1 for k in levels):
            reason = "multiple_descending_crossings_inside_window"
        elif any(counts[k] == 0 for k in levels):
            reason = "absent_crossing_or_transition_outside_fixed_window"
        elif not crossings["90"][0] < crossings["10"][0]:
            reason = "crossings_out_of_descending_order"
        elif depth[eligible[0]] < levels["90"] or depth[eligible[-1]] > levels["10"]:
            reason = "transition_extends_beyond_fixed_window"
        diagnostics.append(
            {
                "time_s": elapsed,
                "exact_shock_position_m": float(shock),
                "window_m": [float(shock - width), float(shock + width)],
                "original_node_count_in_window": len(eligible),
                "threshold_depths_m": levels,
                "descending_crossing_counts": counts,
                "descending_crossing_positions_m": crossings,
                "maximum_absolute_adjacent_depth_slope": {
                    "value": slope,
                    "reason": "insufficient_original_nodes" if slope is None else None,
                },
                "transition_width_10_to_90_m": {
                    "value": None
                    if reason
                    else crossings["10"][0] - crossings["90"][0],
                    "reason": reason,
                },
                "crossing_rule": (
                    "linear interpolation; both bracketing original nodes must "
                    "be inside the fixed window; no additional queries"
                ),
                "interpretation": (
                    "grid-dependent amplitude/position/smoothing descriptor; "
                    "shock accuracy is not established by width alone"
                ),
            }
        )
    rows = np.array(indices, dtype=int)
    profiles = {
        "x_m": x,
        "t_s": times[rows],
        "h_m": fields["h_m"][rows],
        "q_m2_s": fields["q_m2_s"][rows],
        "reference_h_m": reference["h_m"][rows],
        "reference_q_m2_s": reference["q_m2_s"][rows],
    }
    return diagnostics, profiles


def _ve_mass(case, volume, boundary):
    result = quadrature_mass(case, volume, boundary)
    prescribed = result["prescribed_initial_volume_m2"]
    curve = np.array(result["signed_balance_m2"], dtype=float)
    bias_removed = curve - curve[0]
    result["bias_subtracted_balance_m2"] = bias_removed.tolist()
    result["normalized_bias_subtracted_balance"] = (bias_removed / prescribed).tolist()
    for label, values in (("corrected", curve), ("bias_subtracted", bias_removed)):
        result[label + "_sampled_extrema"] = {
            "minimum_signed_m2": float(values.min()),
            "maximum_signed_m2": float(values.max()),
            "maximum_absolute_m2": float(np.abs(values).max()),
            "final_signed_m2": float(values[-1]),
            "maximum_absolute_normalized": float(np.abs(values).max() / prescribed),
        }
    bt = boundary["t_s"]
    mismatch = {"t_s": bt}
    summary = {}
    for index, side in enumerate(("left", "right")):
        for field, target_key, integral_unit in (
            ("h_m", side + "_h_m", "m_s"),
            ("q_m2_s", side + "_q_m2_s", "m2"),
        ):
            delta = (
                np.asarray(boundary[field][:, index], dtype=np.float64)
                - case["boundary"][target_key]
            )
            signed = cumulative_trapezoid(delta, bt, initial=0.0)
            absolute = cumulative_trapezoid(np.abs(delta), bt, initial=0.0)
            key = side + "_" + field
            mismatch[key + "_mismatch"] = delta
            mismatch[key + "_signed_integral_" + integral_unit] = signed
            mismatch[key + "_absolute_integral_" + integral_unit] = absolute
            summary[key] = {
                "maximum_absolute_mismatch": float(np.abs(delta).max()),
                "final_signed_integral": float(signed[-1]),
                "final_absolute_integral": float(absolute[-1]),
                "integral_units": integral_unit,
                "role": "continuity flux mismatch"
                if field == "q_m2_s"
                else "reservoir depth constraint mismatch; not a mass-flux term",
            }
    result["separate_boundary_h_q_mismatch"] = summary
    return result, mismatch


def _ve_dual(arrays, source_sha256):
    if set(arrays) != {"coords_nd", "inviscid", "q_xx", "modified"}:
        raise ValueError("unexpected selected-checkpoint dual-operator schema")
    if (
        arrays["coords_nd"].shape != (365, 2)
        or arrays["inviscid"].shape != (365, 2)
        or arrays["modified"].shape != (365, 2)
        or arrays["q_xx"].shape not in {(365,), (365, 1)}
        or any(not np.isfinite(value).all() for value in arrays.values())
    ):
        raise ValueError("invalid 365-location dual-operator arrays")
    if not np.array_equal(arrays["inviscid"][:, 0], arrays["modified"][:, 0]):
        raise ValueError("continuity changed between reported operators")
    fields = {
        "r_h": arrays["inviscid"][:, 0],
        "r_q": arrays["inviscid"][:, 1],
        "q_star_xx": arrays["q_xx"].ravel(),
        "r_q_nu": arrays["modified"][:, 1],
    }
    return {
        "points": 365,
        "coordinate_system": "nondimensional x/Lc,t/Tc",
        "modified_operator_gamma": 0.001,
        "source_array_sha256": source_sha256,
        "fields": {
            name: {
                "mean": float(np.mean(v, dtype=np.float64)),
                "mean_square": float(np.mean(np.asarray(v, dtype=np.float64) ** 2)),
                "rms": float(np.sqrt(np.mean(np.asarray(v, dtype=np.float64) ** 2))),
                "maximum_absolute": float(np.max(np.abs(v))),
            }
            for name, v in fields.items()
        },
        "interpretation": (
            "two frozen operators at one selected state; no checkpoint "
            "reselection or analytical-reference derivative claim"
        ),
    }


def evaluate_experiment(manifest_path, output_dir, root=Path(".")) -> Path:
    """Full-fit-only quality branch; pilot rejection precedes query/reference reads."""
    root = Path(root).resolve()
    manifest_path = _ve_path(root, manifest_path)
    manifest = read_json(manifest_path)
    if manifest.get("mode") != "full" or manifest.get("state") != "completed":
        raise ValueError(
            "accuracy analysis requires the released completed full fit; "
            "pilots are excluded"
        )
    output = _ve_path(root, output_dir)
    if output.exists():
        raise FileExistsError("existing viscosity evaluations are preserved")
    validate_run_id(output.name)
    protocol_path = _ve_path(root, manifest["protocol"]["path"])
    if sha(protocol_path) != manifest["protocol"]["sha256"]:
        raise ValueError("experiment protocol hash mismatch")
    protocol = read_json(protocol_path)
    check_protocol(protocol, root)
    case, training, evaluation = (
        protocol["case"],
        protocol["training"],
        protocol["evaluation"],
    )
    positive_id = validate_run_id(manifest["positive_run"])
    positive, resolved, positive_path = load_viscosity_checkpoint(positive_id, root)
    if (
        positive["step"] != 10000
        or resolved["case"] != case
        or resolved["training"] != training
        or sha(positive_path / "checkpoint.pt")
        != manifest["positive_checkpoint_sha256"]
        or positive["protocol_sha256"] != digest(protocol)
        or resolved["protocol_sha256"] != digest(protocol)
        or resolved["extension"]
        != {"kind": "viscosity", "mode": "full", "nu_star": NU_STAR, "steps": 10000}
    ):
        raise ValueError(
            "quality analysis requires the unchanged completed "
            "10000-update positive fit"
        )
    historical = protocol["historical"]
    zero_checkpoint_path = _ve_path(root, Path(historical["run"]) / "checkpoint.pt")
    zero_checkpoint_sha = historical["binding_sha256"][
        str(zero_checkpoint_path.relative_to(root))
    ]
    if sha(zero_checkpoint_path) != zero_checkpoint_sha:
        raise ValueError("historical selected zero checkpoint changed")
    zero_id = validate_run_id(manifest["zero_query_run"])
    if zero_id != protocol["zero_cache_run"]:
        raise ValueError("historical zero query cache differs from the frozen protocol")
    if (
        sha(root / "runs" / zero_id / "query_manifest.json")
        != manifest["zero_query_manifest_sha256"]
    ):
        raise ValueError("cached zero query manifest differs from its full binding")
    zero_path, zero_manifest, zero_arrays = _ve_query(
        root,
        zero_id,
        protocol_sha256=sha(protocol_path),
        checkpoint_sha256=zero_checkpoint_sha,
        selected_step=historical["selected_step"],
        nu_star=0.0,
    )
    reference_binding = protocol["reference"]["primary"]
    reference_path = _ve_path(root, reference_binding["path"])
    if sha(reference_path) != reference_binding["sha256"]:
        raise ValueError("saved inviscid analytical reference changed")
    source_bindings = {}
    resolved_output = {
        "case": case,
        "training": training,
        "evaluation": evaluation,
        "request": {
            "run_id": output.name,
            "seed": 11,
            "device": "cpu",
            "dtype": "float32",
        },
        "provenance": {
            "input_files": [
                {"path": str(manifest_path), "sha256": sha(manifest_path)},
                {"path": str(protocol_path), "sha256": sha(protocol_path)},
            ]
        },
    }
    with RunRecord.create(
        output.parent,
        output.name,
        resolved_output,
        project_root=root,
        metadata={
            "stage": "viscosity_final_evaluation",
            "checkpoint_reselection": False,
            "historical_zero_cache_recomputed": False,
        },
    ) as record:
        started = time.perf_counter()
        positive_query_id = manifest.get("positive_query_run")
        positive_query_started = time.perf_counter()
        created_positive_query = positive_query_id is None
        if created_positive_query:
            positive_query_id = validate_run_id(positive_id + "_query")
            query_checkpoint(
                protocol,
                positive_path / "checkpoint.pt",
                positive_query_id,
                0.001,
                root,
            )
        positive_query_wall = time.perf_counter() - positive_query_started
        positive_query_path, positive_manifest, positive_arrays = _ve_query(
            root,
            positive_query_id,
            protocol_sha256=sha(protocol_path),
            checkpoint_sha256=sha(positive_path / "checkpoint.pt"),
            selected_step=positive["selected_step"],
            nu_star=0.001,
        )
        reference = _ve_npz(reference_path)
        common_grid_fields(reference, evaluation)
        for name in (
            "primary",
            "volume_2000",
            "volume_4000",
            "boundary_1001",
            "boundary_2001",
        ):
            for axis in ("x_m", "t_s"):
                if not np.array_equal(
                    zero_arrays[name][axis], positive_arrays[name][axis]
                ):
                    raise ValueError("positive and cached zero query axes differ")
        if not np.array_equal(
            zero_arrays["dual_operator"]["coords_nd"],
            positive_arrays["dual_operator"]["coords_nd"],
        ):
            raise ValueError("dual-operator coordinate membership differs")
        _ve_copy(
            root, manifest_path, output / "experiment_manifest.json", source_bindings
        )
        _ve_copy(root, protocol_path, output / "protocol.json", source_bindings)
        _ve_copy(
            root, reference_path, output / "reference" / "primary.npz", source_bindings
        )
        for section in ("source_sha256", "input_sha256"):
            for name, expected_digest in protocol[section].items():
                path = _ve_path(root, name)
                if sha(path) != expected_digest:
                    raise ValueError(
                        "frozen source/input changed during full evaluation"
                    )
                source_bindings[str(path.relative_to(root))] = expected_digest
                if section == "source_sha256":
                    _ve_copy(
                        root,
                        path,
                        output / "source_snapshot" / path.relative_to(root),
                        source_bindings,
                    )
        metrics, pooled, fronts, mass, duals, identities = [], [], {}, {}, {}, {}
        query_paths = {"zero": zero_path, "positive": positive_query_path}
        raw = {"zero": zero_arrays, "positive": positive_arrays}
        query_manifests = {"zero": zero_manifest, "positive": positive_manifest}
        for variant in ("zero", "positive"):
            for path in sorted(query_paths[variant].rglob("*")):
                if path.is_file():
                    _ve_copy(
                        root,
                        path,
                        output
                        / "queries"
                        / variant
                        / path.relative_to(query_paths[variant]),
                        source_bindings,
                    )
            fields = raw[variant]["primary"]
            identity = {
                "run_id": Path(historical["run"]).name
                if variant == "zero"
                else positive_id,
                "method": "pinn",
                "variant": variant,
                "seed": 11,
                "cells": None,
            }
            identities[variant] = identity
            metrics.extend(
                evaluate_fields(case, reference, fields, evaluation, identity)
            )
            pooled.extend(_ve_pooled(case, evaluation, reference, fields, identity))
            fronts[variant], profiles = _ve_front(case, evaluation, fields, reference)
            np.savez_compressed(output / (variant + "_profiles.npz"), **profiles)
            levels = []
            for cells, nodes in ((2000, 1001), (4000, 2001)):
                balance, mismatches = _ve_mass(
                    case,
                    raw[variant]["volume_" + str(cells)],
                    raw[variant]["boundary_" + str(nodes)],
                )
                levels.append(balance)
                np.savez_compressed(
                    output / f"{variant}_boundary_mismatch_{nodes}.npz", **mismatches
                )
            mass[variant] = {
                "base": levels[0],
                "doubled": levels[1],
                "quadrature_sensitivity": quadrature_sensitivity(*levels),
                "momentum_budget": None,
                "momentum_budget_reason": (
                    "not part of this minimal selected experiment"
                ),
            }
            duals[variant] = _ve_dual(
                raw[variant]["dual_operator"],
                sha(query_paths[variant] / "dual_operator.npz"),
            )
        differences = []
        for rows, keys, scope in (
            (metrics, ("time_s", "field", "region"), "per_time"),
            (pooled, ("field", "region"), "pooled"),
        ):
            zero_rows = [r for r in rows if r["variant"] == "zero"]
            positive_rows = [r for r in rows if r["variant"] == "positive"]
            for a, b in zip(zero_rows, positive_rows, strict=True):
                if any(a[k] != b[k] for k in keys):
                    raise ValueError("comparison rows do not align")
                row = {k: a[k] for k in keys}
                row.update(
                    scope=scope, difference_direction="positive_minus_historical_zero"
                )
                for key in ("rmse", "relative_l2"):
                    av, bv = a[key]["value"], b[key]["value"]
                    row[key] = {
                        "value": None if av is None or bv is None else bv - av,
                        "reason": "undefined_paired_metric"
                        if av is None or bv is None
                        else None,
                    }
                differences.append(row)
        training_evidence = {}
        for directory, label in (
            (positive_path, "positive"),
            (_ve_path(root, historical["run"]), "historical_zero"),
        ):
            for filename in (
                "resolved_config.json",
                "metadata.json",
                "status.json",
                "checkpoint_manifest.json",
                "training_diagnostics.json",
                "history.csv",
                "step_measurements.json",
            ):
                path = directory / filename
                if path.is_file():
                    _ve_copy(
                        root,
                        path,
                        output / "training_evidence" / label / filename,
                        source_bindings,
                    )
                    if path.suffix == ".json":
                        training_evidence[label + "/" + filename] = read_json(path)
        costs = {
            "positive_query_newly_executed": created_positive_query,
            "positive_query_call_or_cache_validation_wall_s": positive_query_wall,
            "zero_query_recomputed": False,
            "zero_query_is_reused_pilot_cost": True,
            "raw_query_manifests": query_manifests,
            "raw_training_json_evidence": training_evidence,
            "analysis_copy_and_payload_before_report_s": time.perf_counter() - started,
            "scope": (
                "historical zero training, performed pilot work, new positive "
                "full training and final query costs are separate stages; "
                "no simultaneous paired-training or speedup claim"
            ),
        }
        report = {
            "schema_version": 1,
            "purpose": "single_seed_E1_constant_momentum_diffusion",
            "manifest_sha256": sha(manifest_path),
            "protocol_sha256": sha(protocol_path),
            "query_sources": {
                v: {
                    "run_id": query_paths[v].name,
                    "query_manifest_sha256": sha(
                        query_paths[v] / "query_manifest.json"
                    ),
                }
                for v in query_paths
            },
            "reference_source": reference_binding,
            "metrics": metrics,
            "pooled_metrics": pooled,
            "positive_minus_zero": differences,
            "front_diagnostics": fronts,
            "mass": mass,
            "dual_operator_diagnostics": duals,
            "metric_definitions": {
                **metric_definitions(),
                "paired_difference": "positive minus historical zero",
            },
            "cost": costs,
            "limitations": [
                "The comparison target is the inviscid analytical E1 solution; "
                "there is no exact positive-viscosity solution here.",
                "One positive fit and a conditionally reused historical zero "
                "selected checkpoint do not establish variability or "
                "new-case generalization.",
                "No analytical classical second derivatives at the discontinuous "
                "shock are used as a positive-viscosity reference.",
                "Front descriptors use the original grid/window and can mix "
                "smoothing, displacement and amplitude error; "
                "invalid transitions remain null.",
                "Quadrature differences are sensitivity observations, not error "
                "bounds; sampled extrema need not be continuous-time extrema.",
                "Reservoir h mismatch is reported separately from the unchanged "
                "continuity q-flux balance.",
                "No momentum budget, entropy selection, energy dissipation, "
                "well-posedness, field validity or operational guarantee "
                "is established.",
            ],
        }
        write_json_atomic(output / "evaluation.json", report)
        write_json_atomic(output / "source_bindings.json", source_bindings)
        record.complete(
            stopping_reason=(
                "saved full-fit and cached historical-zero diagnostics "
                "evaluated; no reselection"
            ),
            iterations=0,
            elapsed_timings_s={
                "quality_analysis_copy_and_save": time.perf_counter() - started
            },
        )
    files = {
        str(path.relative_to(output)): sha(path)
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    write_json_atomic(
        output / "result_integrity.json",
        {
            "schema_version": 1,
            "files_sha256": files,
            "manifest_sha256": sha(manifest_path),
            "protocol_sha256": sha(protocol_path),
            "source_bindings": source_bindings,
            "self_hash_excluded": True,
            "lifecycle_files_finalized_before_inventory": True,
        },
    )
    return output
