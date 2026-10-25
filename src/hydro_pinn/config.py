"""Strict JSON contracts for idealized cases and reproducible run requests.

All public dictionaries use explicit SI field names. Validation returns a deep
copy, never mutates source inputs, and rejects unknown fields so spelling errors
cannot silently become unused settings. Resolving a run applies file values,
then smoke overrides, then explicit nested overrides, then request arguments.
No function here generates an analytical solution or trains a network.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, TypedDict

JsonObject = dict[str, Any]


class Scales(TypedDict):
    H_m: float
    Lc_m: float
    Uc_m_s: float
    Tc_s: float


class RunRequest(TypedDict):
    seed: int | None
    device: str
    dtype: str
    run_id: str | None


class ConfigError(ValueError):
    """A configuration field or cross-file contract is invalid."""


def _error(path: str, message: str) -> None:
    raise ConfigError(f"{path}: {message}")


def _object(value: Any, path: str, fields: set[str]) -> JsonObject:
    if not isinstance(value, dict):
        _error(path, "must be an object")
    missing = fields - value.keys()
    unknown = value.keys() - fields
    if missing:
        _error(path, f"missing required fields: {', '.join(sorted(missing))}")
    if unknown:
        _error(path, f"unknown fields: {', '.join(sorted(unknown))}")
    return value


def _number(value: Any, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error(path, "must be a finite number, not a boolean")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        _error(path, "must be finite")
    if positive and value <= 0:
        _error(path, "must be positive")
    return float(value)


def _integer(value: Any, path: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        _error(path, f"must be an integer >= {minimum}")
    return value


def _choice(value: Any, path: str, choices: set[Any]) -> None:
    if not isinstance(value, str) or value not in choices:
        _error(path, f"must be one of {sorted(choices)}")


def _exact(value: Any, path: str, expected: Any) -> None:
    if type(value) is not type(expected) or value != expected:
        _error(path, f"must equal {expected!r}")


def _numbers(values: Any, path: str, *, positive: bool = False) -> list[float]:
    if not isinstance(values, list) or not values:
        _error(path, "must be a nonempty array")
    return [_number(v, f"{path}[{i}]", positive=positive) for i, v in enumerate(values)]


def _ascending(values: list[float], path: str) -> None:
    if any(b <= a for a, b in zip(values, values[1:], strict=False)):
        _error(path, "must be strictly increasing without duplicates")


def _domain(values: Any, path: str) -> tuple[float, float]:
    numbers = _numbers(values, path)
    if len(numbers) != 2 or numbers[1] <= numbers[0]:
        _error(path, "must contain two increasing finite endpoints in metres")
    _number(numbers[1] - numbers[0], f"{path} width", positive=True)
    return numbers[0], numbers[1]


def _finite_tree(value: Any, path: str = "config") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        _error(path, "must be finite")
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                _error(path, "object keys must be strings")
            _finite_tree(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _finite_tree(child, f"{path}[{index}]")


def validate_run_id(value: Any) -> str:
    """Return a single portable identifier; paths and hidden names are invalid."""
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", value) is None
    ):
        _error(
            "request.run_id",
            "must be 1-96 ASCII letters/digits/underscores/hyphens, "
            "starting alphanumeric",
        )
    if value.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        _error("request.run_id", "reserved filesystem identifier")
    return value


def validate_request(request: JsonObject) -> RunRequest:
    """Validate a requested backend/dtype without claiming it is available."""
    data = _object(request, "request", {"seed", "device", "dtype", "run_id"})
    if data["seed"] is not None:
        _integer(data["seed"], "request.seed", minimum=0)
        if data["seed"] >= 2**32:
            _error(
                "request.seed",
                "must be less than 2**32 for shared NumPy/PyTorch seeding",
            )
    _choice(data["device"], "request.device", {"cpu", "cuda", "mps"})
    _choice(data["dtype"], "request.dtype", {"float32", "float64"})
    if data["device"] == "mps" and data["dtype"] == "float64":
        _error(
            "request.dtype",
            "MPS does not support float64; choose CPU float64 or MPS float32",
        )
    if data["run_id"] is not None:
        validate_run_id(data["run_id"])
    return copy.deepcopy(data)


def _validate_case(data: JsonObject) -> None:
    _object(
        data,
        "case",
        {
            "config_type",
            "schema_version",
            "case_id",
            "description",
            "domain_m",
            "dam_position_m",
            "gravity_m_s2",
            "final_time_s",
            "left_depth_m",
            "right_depth_m",
            "left_velocity_m_s",
            "right_velocity_m_s",
            "reference",
            "assumptions",
            "boundary",
            "scales",
            "grid",
            "array_units",
        },
    )
    _choice(data["case_id"], "case.case_id", {"E1", "E2"})
    if not isinstance(data["description"], str) or not data["description"].strip():
        _error("case.description", "must be nonempty text")
    left, right = _domain(data["domain_m"], "case.domain_m")
    dam = _number(data["dam_position_m"], "case.dam_position_m")
    if not left < dam < right:
        _error("case.dam_position_m", "must lie strictly inside domain_m")
    gravity = _number(data["gravity_m_s2"], "case.gravity_m_s2", positive=True)
    horizon = _number(data["final_time_s"], "case.final_time_s", positive=True)
    h_left = _number(data["left_depth_m"], "case.left_depth_m", positive=True)
    h_right = _number(data["right_depth_m"], "case.right_depth_m")
    if not 0 <= h_right < h_left:
        _error(
            "case.right_depth_m",
            "must satisfy 0 <= hR < hL for these rest-state references",
        )
    for key in ("left_velocity_m_s", "right_velocity_m_s"):
        if _number(data[key], f"case.{key}") != 0:
            _error(f"case.{key}", "must be zero for the core rest-state reference")
    reference = _object(
        data["reference"],
        "case.reference",
        {"kind", "source_url", "root_tolerance_m", "root_max_iterations"},
    )
    _choice(reference["kind"], "case.reference.kind", {"wet_bed", "ritter"})
    if not isinstance(reference["source_url"], str) or not reference[
        "source_url"
    ].startswith("https://"):
        _error("case.reference.source_url", "must be an HTTPS attribution URL")
    _number(
        reference["root_tolerance_m"], "case.reference.root_tolerance_m", positive=True
    )
    _integer(reference["root_max_iterations"], "case.reference.root_max_iterations")
    if reference["kind"] == "ritter" and h_right != 0:
        _error(
            "case.right_depth_m",
            "Ritter requires exact hR=0; no analytical depth floor",
        )
    if reference["kind"] == "wet_bed" and h_right <= 0:
        _error("case.right_depth_m", "wet_bed requires hR>0")
    if (data["case_id"] == "E1") != (reference["kind"] == "wet_bed"):
        _error("case.reference.kind", "E1 uses wet_bed and E2 uses ritter")
    assumptions = _object(
        data["assumptions"],
        "case.assumptions",
        {"bed", "friction", "rainfall", "channel_width_convention"},
    )
    for key, value in {
        "bed": "flat",
        "friction": "none",
        "rainfall": "none",
        "channel_width_convention": "per_unit_width",
    }.items():
        _exact(assumptions[key], f"case.assumptions.{key}", value)
    boundary = _object(
        data["boundary"],
        "case.boundary",
        {"kind", "left_h_m", "left_q_m2_s", "right_h_m", "right_q_m2_s"},
    )
    _exact(boundary["kind"], "case.boundary.kind", "fixed_reservoir")
    for key, expected in {
        "left_h_m": h_left,
        "right_h_m": h_right,
        "left_q_m2_s": 0.0,
        "right_q_m2_s": 0.0,
    }.items():
        if _number(boundary[key], f"case.boundary.{key}") != expected:
            _error(f"case.boundary.{key}", "must equal its undisturbed initial state")
    # A conservative bound shared by both zero-velocity hL>hR cases. Phase01
    # will calculate the exact wet-bed shock speed independently.
    c0 = math.sqrt(gravity * h_left)
    if not math.isfinite(c0) or c0 <= 0:
        _error(
            "case.gravity_m_s2",
            "gravity * left_depth_m must yield a finite positive wave speed",
        )
    if horizon >= min((dam - left) / c0, (right - dam) / (2 * c0)):
        _error(
            "case.final_time_s",
            "must precede boundary contact under the conservative c0/2c0 wave bound",
        )
    scales = _object(data["scales"], "case.scales", {"H_m", "Lc_m", "Uc_m_s", "Tc_s"})
    for key, value in scales.items():
        _number(value, f"case.scales.{key}", positive=True)
    gravity_depth_scale = _number(
        gravity * scales["H_m"],
        "case.scales.Uc_m_s (gravity_m_s2 * H_m)",
        positive=True,
    )
    expected_u = _number(
        math.sqrt(gravity_depth_scale), "case.scales.Uc_m_s", positive=True
    )
    expected_t = _number(
        scales["Lc_m"] / expected_u, "case.scales.Tc_s (Lc_m/Uc_m_s)", positive=True
    )
    if not math.isclose(scales["Uc_m_s"], expected_u, rel_tol=1e-12):
        _error("case.scales.Uc_m_s", "must equal sqrt(gravity_m_s2 * H_m)")
    if not math.isclose(scales["Tc_s"], expected_t, rel_tol=1e-12):
        _error("case.scales.Tc_s", "must equal Lc_m/Uc_m_s")
    grid = _object(
        data["grid"],
        "case.grid",
        {"cell_counts", "state_location", "cfl", "time_output_policy"},
    )
    _exact(grid["state_location"], "case.grid.state_location", "cell_average")
    _exact(
        grid["time_output_policy"],
        "case.grid.time_output_policy",
        "hit_requested_times",
    )
    if not 0 < _number(grid["cfl"], "case.grid.cfl") <= 1:
        _error("case.grid.cfl", "must satisfy 0 < CFL <= 1")
    if not isinstance(grid["cell_counts"], list) or not grid["cell_counts"]:
        _error("case.grid.cell_counts", "must be a nonempty list of even cell counts")
    for index, cells in enumerate(grid["cell_counts"]):
        _integer(cells, f"case.grid.cell_counts[{index}]", minimum=2)
        if cells % 2:
            _error(
                f"case.grid.cell_counts[{index}]",
                "must be even for the core dam interface",
            )
        _number(cells, f"case.grid.cell_counts[{index}]")
        _number(
            (right - left) / cells,
            f"case.grid.cell_counts[{index}] derived dx_m",
            positive=True,
        )
        face_index = (dam - left) / (right - left) * cells
        if not math.isclose(face_index, round(face_index), abs_tol=1e-10):
            _error("case.dam_position_m", f"must align with a cell face for N={cells}")
    _ascending(grid["cell_counts"], "case.grid.cell_counts")
    _exact(
        data["array_units"],
        "case.array_units",
        {"x_m": "m", "t_s": "s", "h_m": "m", "q_m2_s": "m^2/s"},
    )


def _validate_evaluation(data: JsonObject, path: str = "evaluation") -> None:
    _object(
        data,
        path,
        {
            "config_type",
            "schema_version",
            "partition",
            "grid",
            "times_s",
            "profile_times_s",
            "velocity_wet_threshold_m",
            "wave_band_half_width_m",
            "mass_quadrature",
        },
    )
    _exact(data["config_type"], f"{path}.config_type", "evaluation")
    _exact(data["schema_version"], f"{path}.schema_version", 1)
    _choice(
        data["partition"],
        f"{path}.partition",
        {
            "final_evaluation",
            "implementation_verification",
            "smoke_evaluation",
            "validation",
        },
    )
    grid = _object(data["grid"], f"{path}.grid", {"kind", "domain_m", "cells"})
    _exact(grid["kind"], f"{path}.grid.kind", "cell_centres")
    _domain(grid["domain_m"], f"{path}.grid.domain_m")
    _integer(grid["cells"], f"{path}.grid.cells")
    times = _numbers(data["times_s"], f"{path}.times_s", positive=True)
    _ascending(times, f"{path}.times_s")
    profiles = _numbers(
        data["profile_times_s"], f"{path}.profile_times_s", positive=True
    )
    _ascending(profiles, f"{path}.profile_times_s")
    if not all(
        any(math.isclose(t, item, abs_tol=1e-12) for item in times) for t in profiles
    ):
        _error(f"{path}.profile_times_s", "each profile time must occur in times_s")
    _number(
        data["velocity_wet_threshold_m"],
        f"{path}.velocity_wet_threshold_m",
        positive=True,
    )
    _number(
        data["wave_band_half_width_m"], f"{path}.wave_band_half_width_m", positive=True
    )
    mass = _object(
        data["mass_quadrature"], f"{path}.mass_quadrature", {"volume", "boundary_flux"}
    )
    volume = _object(
        mass["volume"],
        f"{path}.mass_quadrature.volume",
        {"rule", "cells", "include_t0", "include_reported_times"},
    )
    _exact(volume["rule"], f"{path}.mass_quadrature.volume.rule", "midpoint")
    _exact(volume["include_t0"], f"{path}.mass_quadrature.volume.include_t0", True)
    _exact(
        volume["include_reported_times"],
        f"{path}.mass_quadrature.volume.include_reported_times",
        True,
    )
    flux = _object(
        mass["boundary_flux"],
        f"{path}.mass_quadrature.boundary_flux",
        {
            "rule",
            "nodes",
            "include_t0",
            "include_final_time",
            "include_reported_times",
            "spatial_locations",
        },
    )
    _exact(flux["rule"], f"{path}.mass_quadrature.boundary_flux.rule", "trapezoid")
    for key in ("include_t0", "include_final_time", "include_reported_times"):
        _exact(flux[key], f"{path}.mass_quadrature.boundary_flux.{key}", True)
    _exact(
        flux["spatial_locations"],
        f"{path}.mass_quadrature.boundary_flux.spatial_locations",
        "domain_endpoints",
    )
    for container, key in ((volume, "cells"), (flux, "nodes")):
        levels = container[key]
        if not isinstance(levels, list) or len(levels) != 2:
            _error(
                f"{path}.mass_quadrature.{key}",
                "must contain base and doubled resolutions",
            )
        for index, level in enumerate(levels):
            _integer(level, f"{path}.mass_quadrature.{key}[{index}]", minimum=2)
        expected = 2 * levels[0] if key == "cells" else 2 * (levels[0] - 1) + 1
        if levels[1] != expected:
            _error(
                f"{path}.mass_quadrature.{key}",
                "second resolution must double cells/time intervals",
            )


def _validate_training(data: JsonObject) -> None:
    _object(
        data,
        "training",
        {
            "config_type",
            "schema_version",
            "status",
            "network",
            "sampling",
            "optimizer",
            "lbfgs",
            "weighting",
            "seeds",
            "checkpoint_selection",
            "validation",
        },
    )
    _exact(
        data["status"], "training.status", "proposed_until_pilot_and_protocol_freeze"
    )
    network = _object(
        data["network"],
        "training.network",
        {
            "hidden_layers",
            "activation",
            "inputs",
            "outputs",
            "depth_transform",
            "depth_floor_nd",
        },
    )
    if not isinstance(network["hidden_layers"], list) or not network["hidden_layers"]:
        _error("training.network.hidden_layers", "must contain positive layer widths")
    for index, width in enumerate(network["hidden_layers"]):
        _integer(width, f"training.network.hidden_layers[{index}]")
    for key, expected in {
        "activation": "tanh",
        "inputs": ["x_over_Lc", "t_over_Tc"],
        "outputs": ["raw_depth", "scaled_velocity"],
        "depth_transform": "softplus",
    }.items():
        _exact(network[key], f"training.network.{key}", expected)
    _number(network["depth_floor_nd"], "training.network.depth_floor_nd", positive=True)
    sampling = _object(
        data["sampling"],
        "training.sampling",
        {
            "interior_points",
            "initial_points",
            "boundary_points_per_side",
            "algorithm",
            "interior_time",
            "initial_discontinuity",
            "interior_labels",
        },
    )
    for key in ("interior_points", "initial_points", "boundary_points_per_side"):
        _integer(sampling[key], f"training.sampling.{key}")
    for key, expected in {
        "algorithm": "numpy_pcg64_uniform_fixed",
        "interior_time": "strictly_positive",
        "initial_discontinuity": "unsmoothed_exclude_dam_point",
        "interior_labels": False,
    }.items():
        _exact(sampling[key], f"training.sampling.{key}", expected)
    optimizer = _object(
        data["optimizer"], "training.optimizer", {"kind", "steps", "learning_rate"}
    )
    _exact(optimizer["kind"], "training.optimizer.kind", "adam")
    _integer(optimizer["steps"], "training.optimizer.steps")
    _number(
        optimizer["learning_rate"], "training.optimizer.learning_rate", positive=True
    )
    lbfgs = _object(
        data["lbfgs"],
        "training.lbfgs",
        {"enabled", "max_iterations", "history_size", "freeze_points_and_weights"},
    )
    if not isinstance(lbfgs["enabled"], bool):
        _error("training.lbfgs.enabled", "must be a boolean")
    _integer(lbfgs["max_iterations"], "training.lbfgs.max_iterations")
    _integer(lbfgs["history_size"], "training.lbfgs.history_size")
    _exact(
        lbfgs["freeze_points_and_weights"],
        "training.lbfgs.freeze_points_and_weights",
        True,
    )
    weighting = _object(
        data["weighting"],
        "training.weighting",
        {
            "mode",
            "lambda_pde",
            "lambda_ic",
            "lambda_bc",
            "lambda_data",
            "update_interval",
            "ema_decay",
            "target_bounds",
            "denominator_epsilon",
            "gradient_norm",
            "update_before_step",
            "detach_statistics",
        },
    )
    _choice(
        weighting["mode"], "training.weighting.mode", {"fixed", "gradient_norm_ema"}
    )
    for key in ("lambda_pde", "lambda_ic", "lambda_bc"):
        _number(weighting[key], f"training.weighting.{key}", positive=True)
    if _number(weighting["lambda_data"], "training.weighting.lambda_data") != 0:
        _error(
            "training.weighting.lambda_data",
            "must be zero for the physics-only baseline/ablation",
        )
    _integer(weighting["update_interval"], "training.weighting.update_interval")
    if not 0 <= _number(weighting["ema_decay"], "training.weighting.ema_decay") < 1:
        _error("training.weighting.ema_decay", "must satisfy 0 <= decay < 1")
    bounds = _numbers(
        weighting["target_bounds"], "training.weighting.target_bounds", positive=True
    )
    if len(bounds) != 2 or bounds[1] <= bounds[0]:
        _error(
            "training.weighting.target_bounds",
            "must contain increasing positive bounds",
        )
    _number(
        weighting["denominator_epsilon"],
        "training.weighting.denominator_epsilon",
        positive=True,
    )
    _exact(
        weighting["gradient_norm"],
        "training.weighting.gradient_norm",
        "l2_over_sqrt_parameter_count_absent_gradients_zero",
    )
    _exact(
        weighting["update_before_step"], "training.weighting.update_before_step", True
    )
    _exact(weighting["detach_statistics"], "training.weighting.detach_statistics", True)
    if not isinstance(data["seeds"], list) or not data["seeds"]:
        _error("training.seeds", "must contain nonnegative integer seeds")
    for seed in data["seeds"]:
        validate_request(
            {"seed": seed, "device": "cpu", "dtype": "float64", "run_id": None}
        )
    if len(set(data["seeds"])) != len(data["seeds"]):
        _error("training.seeds", "must not contain duplicates")
    _exact(
        data["checkpoint_selection"],
        "training.checkpoint_selection",
        "minimum_separate_validation_physics_ic_bc_loss",
    )
    _validate_evaluation(data["validation"], "training.validation")
    _exact(
        data["validation"]["partition"], "training.validation.partition", "validation"
    )


def _validate_smoke(data: JsonObject) -> None:
    _object(
        data,
        "smoke",
        {
            "config_type",
            "schema_version",
            "purpose",
            "training_overrides",
            "validation",
            "evaluation",
        },
    )
    _exact(
        data["purpose"],
        "smoke.purpose",
        "bounded_pipeline_check_not_paper_reproduction",
    )
    overrides = _object(
        data["training_overrides"],
        "smoke.training_overrides",
        {"sampling", "optimizer", "lbfgs"},
    )
    sample = _object(
        overrides["sampling"],
        "smoke.training_overrides.sampling",
        {"interior_points", "initial_points", "boundary_points_per_side"},
    )
    for key, value in sample.items():
        _integer(value, f"smoke.training_overrides.sampling.{key}")
    if sample["interior_points"] > 1000:
        _error("smoke.training_overrides.sampling.interior_points", "must be <= 1000")
    optimizer = _object(
        overrides["optimizer"], "smoke.training_overrides.optimizer", {"steps"}
    )
    _integer(optimizer["steps"], "smoke.training_overrides.optimizer.steps")
    if optimizer["steps"] > 200:
        _error("smoke.training_overrides.optimizer.steps", "must be <= 200")
    _object(overrides["lbfgs"], "smoke.training_overrides.lbfgs", {"enabled"})
    _exact(
        overrides["lbfgs"]["enabled"], "smoke.training_overrides.lbfgs.enabled", False
    )
    _validate_evaluation(data["validation"], "smoke.validation")
    _validate_evaluation(data["evaluation"], "smoke.evaluation")
    _exact(data["validation"]["partition"], "smoke.validation.partition", "validation")
    _exact(
        data["evaluation"]["partition"],
        "smoke.evaluation.partition",
        "smoke_evaluation",
    )
    _check_disjoint(data["validation"], data["evaluation"], "smoke.validation")


def validate_config(data: JsonObject) -> JsonObject:
    """Validate one declared configuration type and return an independent copy."""
    if not isinstance(data, dict):
        _error("config", "must be an object")
    _finite_tree(data)
    if "config_type" not in data or "schema_version" not in data:
        _error("config", "missing required config_type/schema_version")
    _exact(data["schema_version"], "config.schema_version", 1)
    kind = data["config_type"]
    validators = {
        "case": _validate_case,
        "training": _validate_training,
        "evaluation": _validate_evaluation,
        "smoke": _validate_smoke,
    }
    if not isinstance(kind, str) or kind not in validators:
        _error("config.config_type", f"must be one of {sorted(validators)}")
    validators[kind](data)
    return copy.deepcopy(data)


def _json_pairs(pairs: list[tuple[str, Any]]) -> JsonObject:
    result: JsonObject = {}
    for key, value in pairs:
        if key in result:
            _error(key, "duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    _error("config", f"nonstandard JSON constant {value}")


def _read_config(path: str | Path) -> tuple[JsonObject, str]:
    source = Path(path)
    try:
        content = source.read_bytes()
        data = json.loads(
            content, object_pairs_hook=_json_pairs, parse_constant=_reject_constant
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"{source}: {exc}") from exc
    return validate_config(data), hashlib.sha256(content).hexdigest()


def load_config(path: str | Path) -> JsonObject:
    """Read strict UTF-8 JSON, reject duplicate keys, and validate the document."""
    return _read_config(path)[0]


def _merge(
    target: JsonObject,
    patch: JsonObject,
    prefix: str,
    history: list[JsonObject],
    source: str,
) -> None:
    for key, value in patch.items():
        path = f"{prefix}.{key}"
        if key not in target:
            _error(path, "override targets an unknown field")
        if isinstance(value, dict) and isinstance(target[key], dict):
            _merge(target[key], value, path, history, source)
        else:
            history.append(
                {
                    "path": path,
                    "before": copy.deepcopy(target[key]),
                    "after": copy.deepcopy(value),
                    "source": source,
                }
            )
            target[key] = copy.deepcopy(value)


def _check_disjoint(validation: JsonObject, evaluation: JsonObject, path: str) -> None:
    """Reject any shared space-time point, using the declared centre grids."""
    if validation["grid"]["domain_m"] != evaluation["grid"]["domain_m"]:
        _error(path, "validation and evaluation domains must match")
    shared_time = any(
        math.isclose(a, b, abs_tol=1e-12)
        for a in validation["times_s"]
        for b in evaluation["times_s"]
    )
    if not shared_time:
        return

    # Equal-domain midpoint grids share a centre iff N/gcd(N,M) and
    # M/gcd(N,M) are both odd. This avoids materializing potentially large
    # coordinate arrays merely to validate a configuration.
    n, m = validation["grid"]["cells"], evaluation["grid"]["cells"]
    common = math.gcd(n, m)
    if (n // common) % 2 == 1 and (m // common) % 2 == 1:
        _error(path, "validation coordinates overlap evaluation space-time points")


def resolve_run_config(
    case_path: str | Path,
    training_path: str | Path | None = None,
    evaluation_path: str | Path | None = None,
    *,
    seed: int | None = None,
    device: str = "cpu",
    dtype: str = "float64",
    run_id: str | None = None,
    smoke_path: str | Path | None = None,
    overrides: JsonObject | None = None,
) -> JsonObject:
    """Resolve inputs and hash exactly the bytes that were validated.

    ``overrides`` is a nested object rooted in case/training/evaluation; unknown
    fields are rejected. Request arguments always win. A training run requires
    an explicit seed; no arbitrary first seed is silently selected. A smoke file
    replaces validation/evaluation partitions and bounds training work, even when
    explicit overrides follow it. Availability of requested hardware is a later
    execution check, separate from the validity of the request.
    """
    resolved: JsonObject = {
        "case": None,
        "training": None,
        "evaluation": None,
        "request": None,
    }
    inputs: list[JsonObject] = []
    history: list[JsonObject] = []

    def read(
        role: str, path: str | Path, expected: str | tuple[str, ...]
    ) -> JsonObject:
        document, digest = _read_config(path)
        accepted = (expected,) if isinstance(expected, str) else expected
        if document["config_type"] not in accepted:
            _error(role, f"expected config_type={expected!r}")
        inputs.append({"role": role, "path": str(Path(path)), "sha256": digest})
        return document

    resolved["case"] = read("case", case_path, "case")
    if training_path is not None:
        resolved["training"] = read("training", training_path, "training")
    if evaluation_path is not None:
        evaluation = read("evaluation", evaluation_path, ("evaluation", "smoke"))
        resolved["evaluation"] = (
            evaluation["evaluation"]
            if evaluation["config_type"] == "smoke"
            else evaluation
        )
    smoke = None
    if smoke_path is not None:
        smoke = read("smoke", smoke_path, "smoke")
        if resolved["training"] is None:
            _error("smoke", "smoke training overrides require a training configuration")
        _merge(
            resolved["training"],
            smoke["training_overrides"],
            "training",
            history,
            "smoke",
        )
        history.append(
            {
                "path": "training.validation",
                "before": resolved["training"]["validation"],
                "after": smoke["validation"],
                "source": "smoke",
            }
        )
        resolved["training"]["validation"] = copy.deepcopy(smoke["validation"])
        history.append(
            {
                "path": "evaluation",
                "before": resolved["evaluation"],
                "after": smoke["evaluation"],
                "source": "smoke",
            }
        )
        resolved["evaluation"] = copy.deepcopy(smoke["evaluation"])
    if overrides is not None:
        if not isinstance(overrides, dict) or set(overrides) - {
            "case",
            "training",
            "evaluation",
        }:
            _error("overrides", "must contain only case/training/evaluation objects")
        for role, patch in overrides.items():
            if resolved[role] is None or not isinstance(patch, dict):
                _error(
                    f"overrides.{role}",
                    "must target an existing configuration with an object",
                )
            _merge(resolved[role], patch, role, history, "explicit_override")
    request = validate_request(
        {"seed": seed, "device": device, "dtype": dtype, "run_id": run_id}
    )
    for key, value in request.items():
        history.append(
            {
                "path": f"request.{key}",
                "before": None,
                "after": value,
                "source": "request_argument",
            }
        )
    resolved["request"] = request
    for role in ("case", "training", "evaluation"):
        if resolved[role] is not None:
            resolved[role] = validate_config(resolved[role])
    case = resolved["case"]
    evaluation = resolved["evaluation"]
    training = resolved["training"]
    for role, coordinates in (
        ("evaluation", evaluation),
        ("training.validation", training["validation"] if training else None),
    ):
        if coordinates is None:
            continue
        if coordinates["grid"]["domain_m"] != case["domain_m"]:
            _error(f"{role}.grid.domain_m", "must match the physical case domain_m")
        if coordinates["times_s"][-1] > case["final_time_s"]:
            _error(f"{role}.times_s", "must not exceed case.final_time_s")
    if training:
        if seed is None:
            _error("request.seed", "a training run requires an explicit seed")
        if evaluation:
            _check_disjoint(training["validation"], evaluation, "training.validation")
    if smoke:
        if (
            training["optimizer"]["steps"] > 200
            or training["sampling"]["interior_points"] > 1000
            or training["lbfgs"]["enabled"]
        ):
            _error("smoke", "resolved overrides exceed the bounded smoke budget")
    resolved["provenance"] = {
        "schema_version": 1,
        "input_files": inputs,
        "overrides": history,
        "precedence": [
            "input_files",
            "smoke",
            "explicit_overrides",
            "request_arguments",
        ],
        "purpose": "smoke" if smoke else "configured_run",
    }
    return resolved
