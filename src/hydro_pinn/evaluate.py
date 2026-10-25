"""Physical common-grid errors with fixed analytical masks and seed coverage.

This module reads arrays supplied by the evaluation workflow. It never trains a
model, selects a checkpoint, or treats missing outputs as successful results.
The independent mass quadrature workflow lives in ``conservation.py``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from hydro_pinn.lax_wendroff import piecewise_constant_reconstruct
from hydro_pinn.references import wave_diagnostics

FloatArray = NDArray[np.float64]
REFERENCE_RMS_ATOL = 1e-14
IDENTITY_FIELDS = ("run_id", "case_id", "method", "variant", "seed", "cells")
FIELDS = ("h_m", "q_m2_s", "u_m_s")
REGIONS = ("whole", "wave_band", "outside_wave_band")


class EvaluationError(ValueError):
    """Saved fields or requested coordinates violate the comparison contract."""


def metric_definitions() -> dict[str, Any]:
    """Return definitions fixed before reading final model outputs."""
    return {
        "rmse": "sqrt(sum(error^2)/selected_points), equal spatial weights",
        "relative_l2": "sqrt(sum(error^2)/sum(reference^2)); dimensionless",
        "reference_rms_absolute_tolerance": REFERENCE_RMS_ATOL,
        "reference_rms_tolerance_units": "same SI units as the evaluated field",
        "relative_l2_undefined": "reference RMS <= absolute tolerance",
        "velocity": "q/h only on reference h_m > velocity_wet_threshold_m",
        "velocity_incomplete": (
            "null when any reference-wet candidate depth is zero; do not shrink "
            "the reference mask to improve the metric"
        ),
        "wave_band": "abs(x - exact shock/front position) <= half width",
        "wave_band_boundary": "inside at equality; outside is its complement",
        "spatial_coverage": "selected_points / all common-grid positions",
        "native_reconstruction": (
            "piecewise constant cell averages; right cell at an internal face"
        ),
        "time_alignment": "exact requested output times; no interpolation",
        "seed_summary": "descriptive mean and sample SD (ddof=1) of defined seeds",
        "paired_difference": "adaptive minus baseline on the same seed",
    }


def _axis(values: ArrayLike, name: str) -> FloatArray:
    raw = np.asarray(values)
    if raw.ndim != 1 or not raw.size or raw.dtype.kind not in "iuf":
        raise EvaluationError(f"{name} must be a nonempty real axis")
    axis = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(axis).all() or np.any(np.diff(axis) <= 0):
        raise EvaluationError(f"{name} must be finite and strictly increasing")
    return axis


def evaluation_axes(evaluation: dict[str, Any]) -> tuple[FloatArray, FloatArray]:
    """Build the declared common centres and exact reported time axis."""
    grid = evaluation["grid"]
    if grid["kind"] != "cell_centres":
        raise EvaluationError("evaluation grid must use equal cell centres")
    left, right = grid["domain_m"]
    cells = grid["cells"]
    if (
        not isinstance(cells, int)
        or isinstance(cells, bool)
        or cells < 1
        or not math.isfinite(left)
        or not math.isfinite(right)
        or right <= left
    ):
        raise EvaluationError("evaluation grid needs a finite domain and cell count")
    x = left + (np.arange(cells, dtype=np.float64) + 0.5) * (right - left) / cells
    times = _axis(evaluation["times_s"], "evaluation.times_s")
    if np.any(times < 0):
        raise EvaluationError("evaluation times must be nonnegative")
    return x, times


def common_grid_fields(
    arrays: dict[str, ArrayLike], evaluation: dict[str, Any]
) -> dict[str, FloatArray]:
    """Validate saved axes and reconstruct native h/q without time interpolation.

    Presence of ``cell_edges_m`` identifies cell averages. Pointwise fields must
    already use the exact common centres. Returned fields are independent arrays;
    the caller retains native arrays for mass and stability diagnostics.
    """
    expected_x, expected_t = evaluation_axes(evaluation)
    try:
        x, times = _axis(arrays["x_m"], "x_m"), _axis(arrays["t_s"], "t_s")
        h, q = (
            np.asarray(arrays[field], dtype=np.float64) for field in ("h_m", "q_m2_s")
        )
    except KeyError as exc:
        raise EvaluationError(f"missing required array {exc.args[0]}") from exc
    if not np.array_equal(times, expected_t):
        raise EvaluationError("t_s must equal every requested time exactly")
    if h.shape != (len(times), len(x)) or q.shape != h.shape:
        raise EvaluationError("h_m and q_m2_s must have shape (times,positions)")
    if not np.isfinite(h).all() or not np.isfinite(q).all():
        raise EvaluationError("h_m and q_m2_s must contain finite physical values")
    if np.any(h < 0) or np.any((h == 0) & (q != 0)):
        raise EvaluationError("negative depth or discharge at zero depth is invalid")
    if "cell_edges_m" in arrays:
        edges = _axis(arrays["cell_edges_m"], "cell_edges_m")
        left, right = evaluation["grid"]["domain_m"]
        if len(edges) != len(x) + 1 or edges[0] != left or edges[-1] != right:
            raise EvaluationError("native cell edges must cover the declared domain")
        tolerance = 8 * np.finfo(np.float64).eps * max(1, abs(left), abs(right))
        if not np.allclose(x, (edges[:-1] + edges[1:]) / 2, rtol=0, atol=tolerance):
            raise EvaluationError("native x_m must be centres of its saved cell edges")
        h = piecewise_constant_reconstruct(edges, h, expected_x)
        q = piecewise_constant_reconstruct(edges, q, expected_x)
    elif not np.array_equal(x, expected_x):
        raise EvaluationError("point-value x_m must equal the common centres exactly")
    return {
        "x_m": expected_x.copy(),
        "t_s": expected_t.copy(),
        "h_m": h.copy(),
        "q_m2_s": q.copy(),
    }


def _metric(value: float | None, reason: str | None = None) -> dict[str, Any]:
    if value is not None and not math.isfinite(value):
        return {"value": None, "reason": "nonfinite_metric_arithmetic"}
    return {"value": value, "reason": reason}


def _rms(values: FloatArray) -> float:
    scale = float(np.max(np.abs(values)))
    return 0.0 if scale == 0 else scale * float(np.sqrt(np.mean((values / scale) ** 2)))


def error_metrics(
    prediction: ArrayLike, reference: ArrayLike
) -> dict[str, dict[str, Any]]:
    """Compute equal-weight scalar errors without dropping invalid samples."""
    candidate, exact = np.asarray(prediction), np.asarray(reference)
    if candidate.shape != exact.shape or candidate.ndim != 1:
        raise EvaluationError("metric inputs must have equal one-dimensional shapes")
    if not candidate.size:
        undefined = _metric(None, "no_selected_points")
        return {"rmse": undefined.copy(), "relative_l2": undefined.copy()}
    if not np.isfinite(candidate).all() or not np.isfinite(exact).all():
        raise EvaluationError("metric inputs must be finite; do not filter samples")
    with np.errstate(over="ignore", invalid="ignore"):
        residual = np.asarray(candidate - exact, dtype=np.float64)
        rmse = _rms(residual)
        denominator = _rms(np.asarray(exact, dtype=np.float64))
    relative = (
        _metric(None, "zero_or_negligible_reference_norm")
        if denominator <= REFERENCE_RMS_ATOL
        else _metric(rmse / denominator)
    )
    return {"rmse": _metric(rmse), "relative_l2": relative}


def _identity(identity: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    result = {name: identity.get(name) for name in IDENTITY_FIELDS}
    if not result["run_id"] or not result["method"]:
        raise EvaluationError("metric identity requires run_id and method")
    if result["case_id"] not in (None, case["case_id"]):
        raise EvaluationError("metric identity case_id does not match the case")
    result["case_id"] = case["case_id"]
    return result


def _masks(
    case: dict[str, Any], evaluation: dict[str, Any], x: FloatArray, times: FloatArray
) -> tuple[dict[str, NDArray[np.bool_]], FloatArray, str]:
    if evaluation["grid"]["domain_m"] != case["domain_m"]:
        raise EvaluationError("case and evaluation domains must match")
    if times[-1] > case["final_time_s"]:
        raise EvaluationError("evaluation exceeds the case time horizon")
    diagnostics = wave_diagnostics(case)
    wave = "shock" if case["reference"]["kind"] == "wet_bed" else "dry_front"
    positions = case["dam_position_m"] + times * diagnostics["wave_speeds_m_s"][wave]
    width = evaluation["wave_band_half_width_m"]
    if not math.isfinite(width) or width < 0:
        raise EvaluationError("wave band half width must be finite and nonnegative")
    band = np.abs(x[None, :] - positions[:, None]) <= width
    return (
        {
            "whole": np.ones_like(band, dtype=bool),
            "wave_band": band,
            "outside_wave_band": ~band,
        },
        positions,
        wave,
    )


def evaluate_fields(
    case: dict[str, Any],
    reference: dict[str, ArrayLike],
    candidate: dict[str, ArrayLike],
    evaluation: dict[str, Any],
    identity: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return per-time whole/band/outside h/q and reference-wet velocity errors."""
    if "cell_edges_m" in reference:
        raise EvaluationError("analytical reference must contain point values")
    exact = common_grid_fields(reference, evaluation)
    predicted = common_grid_fields(candidate, evaluation)
    x, times = exact["x_m"], exact["t_s"]
    masks, positions, wave = _masks(case, evaluation, x, times)
    threshold = evaluation["velocity_wet_threshold_m"]
    if not math.isfinite(threshold) or threshold < 0:
        raise EvaluationError("velocity wet threshold must be finite and nonnegative")
    wet = exact["h_m"] > threshold
    valid_velocity = predicted["h_m"] > 0
    exact["u_m_s"] = np.divide(
        exact["q_m2_s"], exact["h_m"], out=np.zeros_like(wet, dtype=float), where=wet
    )
    predicted["u_m_s"] = np.divide(
        predicted["q_m2_s"],
        predicted["h_m"],
        out=np.zeros_like(wet, dtype=float),
        where=valid_velocity,
    )
    base = _identity(identity, case)
    rows = []
    for index, elapsed in enumerate(times):
        for field in FIELDS:
            for region, region_mask in masks.items():
                eligible = (
                    region_mask[index] & wet[index]
                    if field == "u_m_s"
                    else (region_mask[index])
                )
                selected = (
                    eligible & valid_velocity[index] if field == "u_m_s" else (eligible)
                )
                count, eligible_count = int(selected.sum()), int(eligible.sum())
                if count != eligible_count:
                    values = {
                        name: _metric(None, "incomplete_candidate_velocity_coverage")
                        for name in ("rmse", "relative_l2")
                    }
                else:
                    values = error_metrics(
                        predicted[field][index, selected], exact[field][index, selected]
                    )
                rows.append(
                    {
                        **base,
                        "time_s": float(elapsed),
                        "field": field,
                        "region": region,
                        **values,
                        "selected_points": count,
                        "eligible_points": eligible_count,
                        "total_points": len(x),
                        "coverage_fraction": count / len(x),
                        "reference_wet_points": int(wet[index].sum()),
                        "velocity_wet_threshold_m": threshold,
                        "wave_kind": wave,
                        "exact_wave_position_m": float(positions[index]),
                        "wave_band_half_width_m": evaluation["wave_band_half_width_m"],
                        "spatial_weighting": "equal_cell_centres",
                    }
                )
    return rows


def unavailable_metric_rows(
    case: dict[str, Any],
    evaluation: dict[str, Any],
    identity: dict[str, Any],
    reason: str,
) -> list[dict[str, Any]]:
    """Keep all requested metric slots visible when a stage has no valid output."""
    if not reason:
        raise EvaluationError("unavailable metrics require a reason")
    x, times = evaluation_axes(evaluation)
    base = _identity(identity, case)
    return [
        {
            **base,
            "time_s": float(elapsed),
            "field": field,
            "region": region,
            "rmse": _metric(None, reason),
            "relative_l2": _metric(None, reason),
            "selected_points": 0,
            "eligible_points": None,
            "total_points": len(x),
            "coverage_fraction": 0.0,
        }
        for elapsed in times
        for field in FIELDS
        for region in REGIONS
    ]


def _seed_summary(
    rows: list[dict[str, Any]], name: str, expected: list[int]
) -> dict[str, Any]:
    indexed = {row["seed"]: row for row in rows}
    defined = {
        seed: row[name]["value"]
        for seed, row in indexed.items()
        if row[name]["value"] is not None
    }
    values = list(defined.values())
    return {
        "mean": _metric(float(np.mean(values)))
        if values
        else _metric(None, "no_defined_seed_metrics"),
        "sample_standard_deviation": _metric(float(np.std(values, ddof=1)))
        if len(values) > 1
        else _metric(None, "fewer_than_two_defined_seeds"),
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
        "defined_seed_count": len(values),
        "expected_seed_count": len(expected),
        "defined_seeds": list(defined),
        "missing_seeds": [seed for seed in expected if seed not in indexed],
        "undefined_seeds": {
            str(seed): row[name]["reason"]
            for seed, row in indexed.items()
            if row[name]["value"] is None
        },
        "seed_values": {str(seed): value for seed, value in defined.items()},
    }


def aggregate_metric_rows(
    rows: list[dict[str, Any]], expected_seeds: list[int]
) -> dict[str, Any]:
    """Summarize each metric with actual seed coverage and paired differences.

    Missing and undefined seeds stay in the report. Means describe the defined
    seeds only, and their denominators are explicit. Unseeded analytical/solver
    rows are retained as their own single-run groups rather than pseudo-seeds.
    """
    expected = list(expected_seeds)
    if len(set(expected)) != len(expected):
        raise EvaluationError("expected seeds must not contain duplicates")
    group_fields = (
        "case_id",
        "method",
        "variant",
        "cells",
        "time_s",
        "field",
        "region",
    )
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in group_fields)].append(row)
    summaries = []
    paired: dict[tuple, dict[str, dict[int, dict]]] = defaultdict(dict)
    pair_fields = tuple(key for key in group_fields if key != "variant")
    for key, members in groups.items():
        seeds = [row["seed"] for row in members]
        if len(set(seeds)) != len(seeds):
            raise EvaluationError("duplicate seed metric in an aggregation group")
        if any(seed is not None and seed not in expected for seed in seeds):
            raise EvaluationError("metric row has an undeclared seed")
        seeded = any(seed is not None for seed in seeds)
        if seeded and any(seed is None for seed in seeds):
            raise EvaluationError("do not mix seeded and unseeded metric rows")
        summary = dict(zip(group_fields, key, strict=True))
        summary.update(
            run_ids=[row["run_id"] for row in members],
            coverage_kind="seeded" if seeded else "single_unseeded_run",
            **{
                name: (
                    _seed_summary(members, name, expected)
                    if seeded
                    else {
                        "single_run_metric": members[0][name],
                        "defined_run_count": int(members[0][name]["value"] is not None),
                        "expected_run_count": 1,
                    }
                )
                for name in ("rmse", "relative_l2")
            },
        )
        summaries.append(summary)
        variant = summary["variant"]
        if seeded and variant in {"baseline", "adaptive"}:
            pair_key = tuple(summary[name] for name in pair_fields)
            paired[pair_key][variant] = {row["seed"]: row for row in members}
    comparisons = []
    for key, variants in paired.items():
        comparison = dict(zip(pair_fields, key, strict=True))
        comparison["difference_direction"] = "adaptive_minus_baseline"
        comparison["expected_seeds"] = expected.copy()
        comparison["seeds"] = []
        for seed in expected:
            baseline, adaptive = (
                variants.get(variant, {}).get(seed)
                for variant in ("baseline", "adaptive")
            )
            entry: dict[str, Any] = {
                "seed": seed,
                "baseline_run_id": baseline["run_id"] if baseline else None,
                "adaptive_run_id": adaptive["run_id"] if adaptive else None,
            }
            for name in ("rmse", "relative_l2"):
                if baseline is None or adaptive is None:
                    value = _metric(None, "missing_paired_seed_run")
                elif baseline[name]["value"] is None or adaptive[name]["value"] is None:
                    value = _metric(None, "undefined_paired_seed_metric")
                else:
                    value = _metric(adaptive[name]["value"] - baseline[name]["value"])
                entry[name] = value
            comparison["seeds"].append(entry)
        comparisons.append(comparison)
    return {
        "expected_seeds": expected,
        "groups": summaries,
        "paired_ablation": comparisons,
    }
