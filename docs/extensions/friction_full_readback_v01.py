"""Read saved friction evaluation arrays and independently verify their reports.

This helper imports no project solver/model/evaluator. It never loads a model,
queries a checkpoint, fits parameters, selects a checkpoint, or runs a reference.
It writes only its requested new JSON receipt and refuses to overwrite it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read_json(path: Path):
    return json.loads(path.read_text())


def read_arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as source:
        arrays = {key: source[key].copy() for key in source.files}
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise ValueError(f"nonfinite saved array: {path}")
    return arrays


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def error_stats(prediction: np.ndarray, target: np.ndarray) -> dict:
    count = int(prediction.size)
    if count == 0:
        return {
            "count": 0,
            "rmse": None,
            "relative_l2_pct": None,
            "reason": "empty region",
        }
    difference = prediction - target
    squared_error = float(np.sum(difference * difference))
    squared_target = float(np.sum(target * target))
    return {
        "count": count,
        "rmse": math.sqrt(squared_error / count),
        "relative_l2_pct": 100 * math.sqrt(squared_error / squared_target)
        if squared_target
        else None,
        "relative_reason": None if squared_target else "zero reference norm",
    }


def diagnostics(prediction: dict, reference: dict) -> dict:
    if np.any(prediction["h_m"] <= 0) or np.any(reference["h_m"] <= 0):
        raise ValueError("unsupported nonpositive saved wet field")
    if not np.array_equal(prediction["cell_edges_m"], reference["cell_edges_m"]):
        raise ValueError("saved common-cell edges differ")
    if not np.allclose(prediction["x_m"], reference["x_m"], rtol=0, atol=1e-15):
        raise ValueError("saved common-cell centers differ")
    if not np.allclose(prediction["t_s"], reference["t_s"], rtol=0, atol=1e-15):
        raise ValueError("saved common output times differ")
    wet = reference["h_m"] > 0.001
    front = np.abs(prediction["x_m"][None, :] - reference["front_x_m"][:, None])
    band = front <= reference["front_half_width_m"][:, None]
    regions = {
        "whole": np.ones_like(wet),
        "numerical_front_band": band,
        "outside_band": np.logical_not(band),
    }
    variables = {
        "h_m": (prediction["h_m"], reference["h_m"]),
        "q_m2_s": (prediction["q_m2_s"], reference["q_m2_s"]),
        "cell_ratio_velocity_m_s": (
            prediction["q_m2_s"] / prediction["h_m"],
            reference["q_m2_s"] / reference["h_m"],
        ),
    }

    def report(mask):
        result = {}
        for name, (predicted, exact) in variables.items():
            selected = mask & wet if name == "cell_ratio_velocity_m_s" else mask
            result[name] = error_stats(predicted[selected], exact[selected])
        return result

    per_time = []
    for row, ti in enumerate(prediction["t_s"]):
        selected_regions = {}
        for name, mask in regions.items():
            selected = np.zeros_like(mask)
            selected[row] = mask[row]
            selected_regions[name] = report(selected)
        per_time.append({"t_s": float(ti), "regions": selected_regions})
    return {
        "aggregate": {name: report(mask) for name, mask in regions.items()},
        "per_time": per_time,
        "reference_wet_count": int(wet.sum()),
        "reference_wet_coverage": float(wet.mean()),
    }


def cumulative_trapezoid(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    return np.r_[0.0, np.cumsum((values[1:] + values[:-1]) * np.diff(times) / 2)]


def mass_from_arrays(volume: dict, boundaries: dict) -> dict:
    x, times = volume["x_m"], volume["t_s"]
    edge_times = boundaries["t_s"]
    if times[0] != 0 or np.any(np.diff(times) <= 0) or np.any(np.diff(edge_times) <= 0):
        raise ValueError("invalid saved mass/boundary time axes")
    if not np.array_equal(boundaries["x_m"], [-5.0, 5.0]):
        raise ValueError("saved boundaries differ from selected endpoints")
    dx = 10 / len(x)
    expected_x = -5 + (np.arange(len(x)) + 0.5) * dx
    if not np.allclose(x, expected_x, rtol=0, atol=1e-14):
        raise ValueError("saved volume axes differ from midpoint rule")
    if np.any(volume["h_m"] <= 0) or np.any(boundaries["h_m"] <= 0):
        raise ValueError("nonpositive saved wet mass field")
    integrated_volume = np.sum(volume["h_m"], axis=1) * dx
    outward = boundaries["q_m2_s"][:, 1] - boundaries["q_m2_s"][:, 0]
    integral = cumulative_trapezoid(outward, edge_times)
    indices = np.searchsorted(edge_times, times)
    if np.any(indices >= len(edge_times)) or not np.array_equal(
        edge_times[indices], times
    ):
        raise ValueError("saved report times absent from boundary nodes")
    bias = float(integrated_volume[0] - 5.5)
    balance = integrated_volume - 5.5 + integral[indices]
    endpoint = {}
    for edge, name, target, sign in [(0, "left", 1.0, -1), (1, "right", 0.1, 1)]:
        q = boundaries["q_m2_s"][:, edge]
        endpoint[name] = {
            "max_abs_h_mismatch_m": float(
                np.max(np.abs(boundaries["h_m"][:, edge] - target))
            ),
            "max_abs_q_mismatch_m2_s": float(np.max(np.abs(q))),
            "absolute_flux_mismatch_integral_m2": float(
                cumulative_trapezoid(np.abs(q), edge_times)[-1]
            ),
            "signed_outward_integral_m2": float(
                cumulative_trapezoid(q * sign, edge_times)[-1]
            ),
        }
    return {
        "volume_m2": integrated_volume.tolist(),
        "report_t_s": times.tolist(),
        "outward_integral_m2": integral[indices].tolist(),
        "prescribed_initial_volume_m2": 5.5,
        "predicted_initial_bias_m2": bias,
        "corrected_balance_m2": balance.tolist(),
        "balance_without_initial_bias_m2": (balance - bias).tolist(),
        "sampled_max_abs_corrected_m2": float(np.max(np.abs(balance))),
        "boundary": endpoint,
        "query_points": len(x) * len(times) + 2 * len(edge_times),
        "saved_outward_integral_readback": integral.tolist(),
    }


def compare_values(expected, saved, path: str, receipt: dict):
    """Check expected report keys; retain all differences, including small errors."""
    if isinstance(expected, dict):
        if not isinstance(saved, dict):
            receipt["mismatches"].append({"path": path, "reason": "expected mapping"})
            return
        for key, value in expected.items():
            if key not in saved:
                receipt["mismatches"].append(
                    {"path": f"{path}.{key}", "reason": "missing key"}
                )
            else:
                compare_values(value, saved[key], f"{path}.{key}", receipt)
    elif isinstance(expected, list):
        if not isinstance(saved, list) or len(expected) != len(saved):
            receipt["mismatches"].append(
                {"path": path, "reason": "array length differs"}
            )
        else:
            for i, (value, other) in enumerate(zip(expected, saved, strict=True)):
                compare_values(value, other, f"{path}[{i}]", receipt)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        receipt["compared_numeric_values"] += 1
        if not isinstance(saved, (int, float)) or not np.isfinite(saved):
            receipt["mismatches"].append(
                {"path": path, "reason": "invalid saved number"}
            )
        elif not math.isclose(expected, saved, rel_tol=1e-12, abs_tol=1e-13):
            receipt["mismatches"].append(
                {"path": path, "recomputed": expected, "saved": saved}
            )
        else:
            receipt["max_abs_numeric_difference"] = max(
                receipt["max_abs_numeric_difference"], abs(expected - saved)
            )
    elif expected != saved:
        receipt["mismatches"].append(
            {"path": path, "recomputed": expected, "saved": saved}
        )


def verify(evaluation: Path, reference_path: Path) -> dict:
    packet = reference_path if reference_path.is_dir() else reference_path.parent
    common = (
        packet / "common_reference.npz" if reference_path.is_dir() else reference_path
    )
    reference = read_arrays(common)
    gauss3, gauss5 = [
        read_arrays(evaluation / f"common_cell_gauss{order}.npz") for order in (3, 5)
    ]
    receipt = {
        "compared_numeric_values": 0,
        "max_abs_numeric_difference": 0.0,
        "mismatches": [],
    }
    errors3, errors5 = diagnostics(gauss3, reference), diagnostics(gauss5, reference)
    compare_values(errors3, read_json(evaluation / "errors.json"), "errors", receipt)
    sensitivity = {
        "h_max_abs_m": float(np.max(np.abs(gauss3["h_m"] - gauss5["h_m"]))),
        "q_max_abs_m2_s": float(np.max(np.abs(gauss3["q_m2_s"] - gauss5["q_m2_s"]))),
        "gauss5_errors": errors5,
    }
    compare_values(
        sensitivity,
        read_json(evaluation / "pinn_cell_quadrature_sensitivity.json"),
        "cell_quadrature",
        receipt,
    )
    mass = []
    for level in (0, 1):
        volume = read_arrays(evaluation / f"volume_{level}.npz")
        boundary = read_arrays(evaluation / f"boundaries_{level}.npz")
        report = mass_from_arrays(volume, boundary)
        compare_values(
            report.pop("saved_outward_integral_readback"),
            boundary["outward_integral_m2"].tolist(),
            f"saved_boundary_integral_{level}",
            receipt,
        )
        mass.append(report)
    combined = {
        key: float(np.max(np.abs(np.asarray(mass[0][key]) - np.asarray(mass[1][key]))))
        for key in ("volume_m2", "outward_integral_m2", "corrected_balance_m2")
    }
    mass_report = {
        "base": mass[0],
        "refined": mass[1],
        "combined_refinement_sensitivity": combined,
    }
    compare_values(mass_report, read_json(evaluation / "mass.json"), "mass", receipt)
    uncertainty = {
        "h_rmse_indicator_m": float(
            np.sqrt(np.mean(reference["h_measured_uncertainty_m"] ** 2))
        ),
        "q_rmse_indicator_m2_s": float(
            np.sqrt(np.mean(reference["q_measured_uncertainty_m2_s"] ** 2))
        ),
        "h_max_indicator_m": float(reference["h_measured_uncertainty_m"].max()),
        "q_max_indicator_m2_s": float(reference["q_measured_uncertainty_m2_s"].max()),
        **{
            name: reference[name].tolist()
            for name in ("front_x_m", "front_uncertainty_m", "front_half_width_m")
        },
    }
    compare_values(
        uncertainty,
        read_json(evaluation / "numerical_reference_uncertainty.json"),
        "reference_uncertainty",
        receipt,
    )
    summary = read_json(packet / "summary.json")
    expected_native = {row["id"]: row["outputs_sha256"] for row in summary["runs"]}
    observed_native = {
        name: digest(packet / name / "outputs.npz") for name in expected_native
    }
    for name, expected in expected_native.items():
        compare_values(
            expected, observed_native[name], f"native_reference_hash.{name}", receipt
        )
    compare_values(
        summary["common_sha256"], digest(common), "common_reference_hash", receipt
    )
    provenance = read_json(evaluation / "provenance.json")
    project_root = Path(__file__).resolve().parents[2]
    frozen_reference_hashes = provenance["reference_sha256"]
    for name, expected in frozen_reference_hashes.items():
        compare_values(
            expected,
            digest(project_root / name),
            f"frozen_reference_hash.{name}",
            receipt,
        )
    inputs = {
        str(path): digest(path)
        for path in sorted(evaluation.iterdir())
        if path.is_file()
    }
    return {
        "state": "verified_saved_array_reports"
        if not receipt["mismatches"]
        else "failed_saved_array_report_readback",
        "evaluation_dir": str(evaluation),
        "reference": str(common),
        "verification": receipt,
        "tolerance": {"relative": 1e-12, "absolute": 1e-13},
        "aggregate_gauss3": errors3["aggregate"],
        "gauss3_per_time": errors3["per_time"],
        "gauss5_aggregate": errors5["aggregate"],
        "gauss3_gauss5_max_difference": {
            k: v for k, v in sensitivity.items() if k != "gauss5_errors"
        },
        "mass": mass_report,
        "reference_uncertainty": uncertainty,
        "native_reference_output_hashes": observed_native,
        "native_reference_budget_preservation": (
            "Full native arrays match their accepted hashes; arrays include "
            "numerical boundary fluxes and actual source impulses. No reference rerun."
        ),
        "input_file_sha256": inputs,
        "reference_sha256": digest(common),
        "helper_sha256": digest(Path(__file__)),
        "limits": [
            "Saved cell-average readback; Gauss point model outputs were not "
            "re-queried or reintegrated",
            "Reference uncertainty indicators and quadrature differences are "
            "not rigorous error bounds",
            "One seed, wet flat E1, fixed illustrative n=.03; no parameter "
            "fitting or field validation",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"readback output already exists: {args.output}")
    result = verify(args.evaluation_dir, args.reference)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as destination:
        json.dump(result, destination, indent=2, allow_nan=False)
        destination.write("\n")
    print(
        json.dumps(
            {
                "state": result["state"],
                "verification": result["verification"],
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0 if not result["verification"]["mismatches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
