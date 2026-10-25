"""Post-terminal saved-array viscosity readback and two figures; no forwards.

Prepare now; invoke only after the parent confirms that the released full fit
and its evaluation are completed. Nothing in this file imports model, solver,
training, query or analytical-evaluation code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RTOL, ATOL = 1e-12, 1e-14
REFERENCE_CHECKS_SHA = "1a6aa8362164ac1c5a64e3ada1d911b6abc87c2fe2efe18011648c4a47906288"
COLORS = {"zero": "#0072B2", "positive": "#D55E00"}
LABELS = {"zero": "Historical zero control", "positive": "Positive ν* = 0.001"}


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(value, handle, allow_nan=False, sort_keys=True, indent=2)
        handle.write("\n")


def arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def strict_path(root, value):
    path = Path(value)
    path = (path if path.is_absolute() else root / path).absolute()
    if (not path.is_relative_to(root) or ".." in path.parts
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError("readback paths must remain within project root without symlinks")
    return path


def trap(values, times):
    # Preserve input arithmetic: raw float32 flux sums precede float64 dt.
    y, t = np.asarray(values), np.asarray(times, dtype=np.float64)
    return np.r_[0.0, np.cumsum(np.diff(t) * (y[1:] + y[:-1]) / 2.0)]


def metric(prediction, reference):
    prediction = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if not len(reference):
        return {key: {"value": None, "reason": "no_selected_points"} for key in ("rmse", "relative_l2")}
    rmse = float(np.sqrt(np.mean((prediction - reference) ** 2)))
    reference_rms = float(np.sqrt(np.mean(reference ** 2)))
    return {"rmse": {"value": rmse, "reason": None},
            "relative_l2": {"value": None if reference_rms <= 1e-14 else rmse / reference_rms,
                            "reason": "zero_or_negligible_reference_norm" if reference_rms <= 1e-14 else None}}


def check_close(actual, expected, label, checks):
    """Record tolerance-bound recomputation, including exact null/reason semantics."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise AssertionError(f"mapping mismatch: {label}")
        for key, value in expected.items():
            check_close(actual[key], value, f"{label}/{key}", checks)
    elif isinstance(expected, (list, tuple, np.ndarray)):
        if len(actual) != len(expected):
            raise AssertionError(f"length mismatch: {label}")
        for index, value in enumerate(expected):
            check_close(actual[index], value, f"{label}/{index}", checks)
    elif isinstance(expected, (float, np.floating)):
        if actual is None or not math.isclose(float(actual), float(expected), rel_tol=RTOL, abs_tol=ATOL):
            raise AssertionError(f"numeric mismatch: {label}: {actual} vs {expected}")
        checks["numeric_values"] += 1
        checks["maximum_absolute_difference"] = max(checks["maximum_absolute_difference"], abs(float(actual) - float(expected)))
    elif actual != expected:
        raise AssertionError(f"value/null/reason mismatch: {label}: {actual} vs {expected}")
    else:
        checks["exact_discrete_values"] += 1


def field_metrics(case, evaluation, reference, fields, speed):
    x, times = reference["x_m"], reference["t_s"]
    shock = case["dam_position_m"] + speed * times
    band = np.abs(x[None, :] - shock[:, None]) <= evaluation["wave_band_half_width_m"]
    masks = {"whole": np.ones_like(band), "wave_band": band, "outside_wave_band": ~band}
    wet = reference["h_m"] > evaluation["velocity_wet_threshold_m"]
    exact = {key: np.asarray(reference[key], dtype=np.float64) for key in ("h_m", "q_m2_s")}
    predicted = {key: np.asarray(fields[key], dtype=np.float64) for key in exact}
    exact["u_m_s"] = np.divide(exact["q_m2_s"], exact["h_m"], out=np.zeros_like(exact["h_m"]), where=wet)
    valid = fields["h_m"] > 0
    predicted["u_m_s"] = np.divide(predicted["q_m2_s"], predicted["h_m"], out=np.zeros_like(predicted["h_m"]), where=valid)
    per_time, pooled = {}, {}
    for field in exact:
        for region, mask in masks.items():
            eligible = mask & wet if field == "u_m_s" else mask
            for index, elapsed in enumerate(times):
                selected = eligible[index]
                values = metric(predicted[field][index][selected], exact[field][index][selected])
                per_time[(float(elapsed), field, region)] = values
            pooled[(field, region)] = metric(predicted[field][eligible], exact[field][eligible])
    return per_time, pooled


def front_descriptors(case, evaluation, fields, speed, hstar):
    levels = {"10": case["right_depth_m"] + .1 * (hstar - case["right_depth_m"]),
              "90": case["right_depth_m"] + .9 * (hstar - case["right_depth_m"])}
    result = []
    x, times = fields["x_m"], fields["t_s"]
    for elapsed in (.1, .25, .5):
        index = np.flatnonzero(times == elapsed)
        if len(index) != 1:
            raise AssertionError("profile time missing from saved primary data")
        depth = fields["h_m"][int(index[0])]
        shock = case["dam_position_m"] + speed * elapsed
        inside = np.flatnonzero(np.abs(x - shock) <= evaluation["wave_band_half_width_m"])
        segments = inside[:-1][np.diff(inside) == 1]
        left, right = depth[segments].astype(np.float64), depth[segments + 1].astype(np.float64)
        crossings, counts, plateau = {}, {}, False
        for key, level in levels.items():
            plateau |= bool(np.any((left == level) & (right == level)))
            selected = (left >= level) & (right < level)
            nodes = segments[selected]
            locations = x[nodes] + (left[selected] - level) / (left[selected] - right[selected]) * (x[nodes + 1] - x[nodes])
            crossings[key], counts[key] = locations.tolist(), int(selected.sum())
        slope = None if len(inside) < 2 else float(np.max(np.abs(np.diff(depth[inside]) / np.diff(x[inside]))))
        if len(inside) < 2:
            reason = "fewer_than_two_original_nodes_inside_window"
        elif plateau:
            reason = "threshold_plateau_has_no_unique_crossing"
        elif any(value > 1 for value in counts.values()):
            reason = "multiple_descending_crossings_inside_window"
        elif any(value == 0 for value in counts.values()):
            reason = "absent_crossing_or_transition_outside_fixed_window"
        elif crossings["90"][0] >= crossings["10"][0]:
            reason = "crossings_out_of_descending_order"
        elif depth[inside[0]] < levels["90"] or depth[inside[-1]] > levels["10"]:
            reason = "transition_extends_beyond_fixed_window"
        else:
            reason = None
        result.append({"time_s": elapsed, "exact_shock_position_m": float(shock),
                       "window_m": [float(shock - .1), float(shock + .1)],
                       "original_node_count_in_window": len(inside), "threshold_depths_m": levels,
                       "descending_crossing_counts": counts, "descending_crossing_positions_m": crossings,
                       "maximum_absolute_adjacent_depth_slope": {"value": slope, "reason": "insufficient_original_nodes" if slope is None else None},
                       "transition_width_10_to_90_m": {"value": None if reason else crossings["10"][0] - crossings["90"][0], "reason": reason}})
    return result


def mass_and_boundaries(case, volume, boundary):
    a, b = case["domain_m"]
    prescribed = (case["dam_position_m"] - a) * case["left_depth_m"] + (b - case["dam_position_m"]) * case["right_depth_m"]
    mass = np.sum(volume["h_m"], axis=1, dtype=np.float64) * (b - a) / len(volume["x_m"])
    bt, times = boundary["t_s"], volume["t_s"]
    indices = np.searchsorted(bt, times)
    if not np.array_equal(bt[indices], times):
        raise AssertionError("volume times absent from boundary nodes")
    left, right = boundary["q_m2_s"][:, 0], boundary["q_m2_s"][:, 1]
    left_delta = left - case["boundary"]["left_q_m2_s"]
    right_delta = right - case["boundary"]["right_q_m2_s"]
    net = trap(right - left, bt)
    mismatch = trap(np.abs(left_delta) + np.abs(right_delta), bt)
    corrected = mass - prescribed + net[indices]
    bias_removed = corrected - corrected[0]
    result = {"t_s": times.tolist(), "volume_m2": mass.tolist(),
              "prescribed_initial_volume_m2": prescribed,
              "learned_initial_volume_error_m2": float(mass[0] - prescribed),
              "normalized_initial_volume_error": float((mass[0] - prescribed) / prescribed),
              "boundary_net_outflow_m2": net[indices].tolist(),
              "signed_balance_m2": corrected.tolist(), "normalized_signed_balance": (corrected / prescribed).tolist(),
              "maximum_absolute_normalized_imbalance": float(np.abs(corrected / prescribed).max()),
              "integrated_absolute_boundary_mismatch_m2": mismatch[indices].tolist(),
              "maximum_absolute_boundary_mismatch_m2_s": float(max(np.abs(left_delta).max(), np.abs(right_delta).max())),
              "bias_subtracted_balance_m2": bias_removed.tolist(),
              "normalized_bias_subtracted_balance": (bias_removed / prescribed).tolist()}
    for label, values in (("corrected", corrected), ("bias_subtracted", bias_removed)):
        result[label + "_sampled_extrema"] = {"minimum_signed_m2": float(values.min()), "maximum_signed_m2": float(values.max()),
                                               "maximum_absolute_m2": float(np.abs(values).max()), "final_signed_m2": float(values[-1]),
                                               "maximum_absolute_normalized": float(np.abs(values).max() / prescribed)}
    curves, summaries = {"t_s": bt}, {}
    for index, side in enumerate(("left", "right")):
        for field, target, units in (("h_m", side + "_h_m", "m_s"), ("q_m2_s", side + "_q_m2_s", "m2")):
            delta = boundary[field][:, index].astype(np.float64) - case["boundary"][target]
            signed, absolute = trap(delta, bt), trap(np.abs(delta), bt)
            key = side + "_" + field
            curves[key + "_mismatch"] = delta
            curves[key + "_signed_integral_" + units] = signed
            curves[key + "_absolute_integral_" + units] = absolute
            summaries[key] = {"maximum_absolute_mismatch": float(np.abs(delta).max()), "final_signed_integral": float(signed[-1]),
                              "final_absolute_integral": float(absolute[-1]), "integral_units": units}
    result["separate_boundary_h_q_mismatch"] = summaries
    return result, curves


def dual_statistics(dual):
    data = {"r_h": dual["inviscid"][:, 0], "r_q": dual["inviscid"][:, 1],
            "q_star_xx": dual["q_xx"].ravel(), "r_q_nu": dual["modified"][:, 1]}
    return {key: {"mean": float(np.mean(values, dtype=np.float64)),
                  "mean_square": float(np.mean(values.astype(np.float64) ** 2)),
                  "rms": float(np.sqrt(np.mean(values.astype(np.float64) ** 2))),
                  "maximum_absolute": float(np.abs(values).max())} for key, values in data.items()}


def save_figures(output, case, reference, raw, mass, boundaries, speed):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for col, elapsed in enumerate((.1, .25, .5)):
        i = int(np.flatnonzero(reference["t_s"] == elapsed)[0])
        for row, field, label in ((0, "h_m", "Depth h (m)"), (1, "q_m2_s", "Discharge q (m²/s per unit width)")):
            ax = axes[row, col]
            ax.plot(reference["x_m"], reference[field][i], color="#242424", lw=1.7, label="Inviscid analytical target", zorder=3)
            for variant in ("zero", "positive"):
                ax.plot(raw[variant]["primary"]["x_m"], raw[variant]["primary"][field][i], color=COLORS[variant], lw=1.5, label=LABELS[variant])
            shock = case["dam_position_m"] + speed * elapsed
            ax.axvspan(shock - .1, shock + .1, color="#7B7B7B", alpha=.13)
            ax.axvline(shock, color="#777777", lw=.8, ls=":")
            ax.set_xlim(case["domain_m"])
            if col == 0:
                ax.set_ylabel(label)
            if row == 0:
                ax.set_title(f"t = {elapsed:.2f} s")
            else:
                ax.set_xlabel("x (m)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, .955))
    fig.suptitle("Selected checkpoints against the inviscid E1 target", fontsize=15, y=.995)
    fig.text(.5, .015, "Seed 11 · pointwise fields · grey band = exact inviscid shock ±0.1 m · modified momentum has no exact viscous reference here", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .91))
    fig.savefig(output / "profiles_inviscid_target.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for variant in ("zero", "positive"):
        m = mass[variant]
        axes[0, 0].plot(m["t_s"], np.array(m["normalized_signed_balance"]) * 100, color=COLORS[variant], label=LABELS[variant])
        axes[0, 1].plot(m["t_s"], np.array(m["normalized_bias_subtracted_balance"]) * 100, color=COLORS[variant], label=LABELS[variant])
        for side, style in (("left", "-"), ("right", "--")):
            for ax, field in ((axes[1, 0], "q_m2_s"), (axes[1, 1], "h_m")):
                curves = boundaries[variant]
                ax.plot(curves["t_s"], curves[side + "_" + field + "_mismatch"], color=COLORS[variant], ls=style, label=f"{LABELS[variant]} · {side}")
    for ax in axes.flat:
        ax.axhline(0, color="#777777", lw=.8, zorder=0)
        ax.set_xlim(0, .5)
        ax.set_xlabel("t (s)")
    axes[0, 0].set_title("Boundary-corrected continuity balance")
    axes[0, 0].set_ylabel("Balance / prescribed initial volume (%)")
    axes[0, 0].legend(frameon=False, fontsize=9)
    axes[0, 1].set_title("Additional balance after removing initial bias")
    axes[0, 1].set_ylabel("Bias-subtracted balance / initial volume (%)")
    axes[1, 0].set_title("Endpoint discharge constraint mismatch")
    axes[1, 0].set_ylabel("q − prescribed q (m²/s)")
    axes[1, 1].set_title("Endpoint depth constraint mismatch")
    axes[1, 1].set_ylabel("h − reservoir h (m)")
    axes[1, 1].legend(frameon=False, fontsize=8)
    fig.suptitle("Continuity balance and reservoir constraint mismatches", fontsize=15)
    fig.text(.5, .015, "Midpoint volume: 4,000 cells · endpoint trapezoid: 2,001 nodes with core time replacement · sampled curves; no momentum budget", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.savefig(output / "mass_and_boundary_constraints.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/extensions/viscosity/full_v01_figures"))
    parser.add_argument("--readback", type=Path, default=Path("docs/evidence/phase_11/viscosity_v01/science_readback_full_v01.json"))
    parser.add_argument("--reference-checks", type=Path, default=Path("runs/core-v1-pilot-a03-e1-reference/checks.json"))
    args = parser.parse_args()
    started = time.perf_counter()
    root = args.root.resolve()
    evaluation_dir, output, readback_path, reference_checks = [strict_path(root, p) for p in (args.evaluation_dir, args.output_dir, args.readback, args.reference_checks)]
    if output.exists() or readback_path.exists():
        raise FileExistsError("preserve existing figures/readback; use new destinations")
    if read(evaluation_dir / "status.json")["state"] != "completed":
        raise ValueError("wait for completed full evaluation")
    manifest = read(evaluation_dir / "experiment_manifest.json")
    if manifest["mode"] != "full" or manifest["state"] != "completed" or manifest.get("full_fit_released") is not True:
        raise ValueError("readback requires the released completed full experiment")
    protocol = read(evaluation_dir / "protocol.json")
    report = read(evaluation_dir / "evaluation.json")
    integrity = read(evaluation_dir / "result_integrity.json")
    input_hashes = {str((evaluation_dir / name).relative_to(root)): sha(evaluation_dir / name) for name in integrity["files_sha256"]}
    for name, expected in integrity["files_sha256"].items():
        if input_hashes[str((evaluation_dir / name).relative_to(root))] != expected:
            raise ValueError("evaluation payload integrity mismatch")
    if sha(reference_checks) != REFERENCE_CHECKS_SHA:
        raise ValueError("frozen inviscid wave diagnostic identity changed")
    input_hashes[str(reference_checks.relative_to(root))] = REFERENCE_CHECKS_SHA
    input_hashes[str((evaluation_dir / "result_integrity.json").relative_to(root))] = sha(evaluation_dir / "result_integrity.json")
    waves = read(reference_checks)
    speed, hstar = waves["wave_speeds_m_s"]["shock"], waves["wet_star"]["h_m"]
    case, evaluation = protocol["case"], protocol["evaluation"]
    reference = arrays(evaluation_dir / "reference/primary.npz")
    raw = {variant: {name: arrays(evaluation_dir / "queries" / variant / (name + ".npz"))
                     for name in ("primary", "volume_2000", "volume_4000", "boundary_1001", "boundary_2001", "dual_operator")}
           for variant in ("zero", "positive")}
    checks = {"numeric_values": 0, "exact_discrete_values": 0, "maximum_absolute_difference": 0.0}
    recomputed, mass, boundary, fronts, duals = {}, {}, {}, {}, {}
    for variant in ("zero", "positive"):
        per_time, pooled = field_metrics(case, evaluation, reference, raw[variant]["primary"], speed)
        recomputed[variant] = {"per_time": per_time, "pooled": pooled}
        for row in report["metrics"]:
            if row["variant"] == variant:
                check_close(row, per_time[(row["time_s"], row["field"], row["region"])], variant + "/per_time", checks)
        for row in report["pooled_metrics"]:
            if row["variant"] == variant:
                check_close(row, pooled[(row["field"], row["region"])], variant + "/pooled", checks)
        fronts[variant] = front_descriptors(case, evaluation, raw[variant]["primary"], speed, hstar)
        check_close(report["front_diagnostics"][variant], fronts[variant], variant + "/front", checks)
        levels = []
        for cells, nodes, key in ((2000, 1001, "base"), (4000, 2001, "doubled")):
            level, mismatch = mass_and_boundaries(case, raw[variant]["volume_" + str(cells)], raw[variant]["boundary_" + str(nodes)])
            levels.append(level)
            check_close(report["mass"][variant][key], level, variant + "/mass/" + key, checks)
            saved_mismatch = arrays(evaluation_dir / f"{variant}_boundary_mismatch_{nodes}.npz")
            for field, values in mismatch.items():
                check_close(saved_mismatch[field], values, variant + "/boundary/" + field, checks)
            if key == "doubled":
                mass[variant], boundary[variant] = level, mismatch
        for field in ("volume_m2", "boundary_net_outflow_m2", "signed_balance_m2"):
            delta = np.array(levels[1][field]) - np.array(levels[0][field])
            check_close(report["mass"][variant]["quadrature_sensitivity"]["differences"][field],
                        {"doubled_minus_base": delta.tolist(), "maximum_absolute_difference": float(np.abs(delta).max())}, variant + "/quadrature/" + field, checks)
        duals[variant] = dual_statistics(raw[variant]["dual_operator"])
        check_close(report["dual_operator_diagnostics"][variant]["fields"], duals[variant], variant + "/dual", checks)
    for row in report["positive_minus_zero"]:
        scope = "per_time" if row["scope"] == "per_time" else "pooled"
        key = (row["time_s"], row["field"], row["region"]) if scope == "per_time" else (row["field"], row["region"])
        for field in ("rmse", "relative_l2"):
            a, b = recomputed["zero"][scope][key][field]["value"], recomputed["positive"][scope][key][field]["value"]
            expected = {"value": None if a is None or b is None else b - a,
                        "reason": "undefined_paired_metric" if a is None or b is None else None}
            check_close(row[field], expected, "positive_minus_zero", checks)
    summary = {field: {variant: recomputed[variant]["pooled"][(field, "whole")] for variant in ("zero", "positive")}
               for field in ("h_m", "q_m2_s", "u_m_s")}
    audit_s = time.perf_counter() - started
    output.mkdir(parents=True, exist_ok=False)
    render_started = time.perf_counter()
    save_figures(output, case, reference, raw, mass, boundary, speed)
    render_s = time.perf_counter() - render_started
    outputs = {p.name: {"sha256": sha(p), "bytes": p.stat().st_size,
                        "pixel_shape": list(plt.imread(p).shape)} for p in sorted(output.glob("*.png"))}
    figure_manifest = {"schema_version": 1, "source_evaluation": str(evaluation_dir.relative_to(root)),
                       "source_evaluation_sha256": sha(evaluation_dir / "evaluation.json"),
                       "helper_sha256": sha(Path(__file__)), "input_sha256": input_hashes, "rendered_outputs": outputs,
                       "cost_s": {"input_verification_and_independent_recomputation": audit_s, "figure_render_and_save": render_s},
                       "software": {"python": platform.python_version(), "numpy": np.__version__, "matplotlib": matplotlib.__version__},
                       "reference_interpretation": "inviscid idealized E1 target; not an exact positive-viscosity solution",
                       "no_model_solver_training_or_query_calls": True}
    write_new(output / "figures_manifest.json", figure_manifest)
    readback = {"schema_version": 1, "state": "passed_saved_array_science_recomputation_visual_review_pending",
                "source_evaluation": str(evaluation_dir.relative_to(root)), "input_sha256": input_hashes,
                "helper_sha256": sha(Path(__file__)), "checks": checks, "comparison_tolerances": {"rtol": RTOL, "atol": ATOL,
                "meaning": "independent saved-array arithmetic comparison, not a scientific performance target"},
                "per_time_metric_rows": len(report["metrics"]), "pooled_metric_rows": len(report["pooled_metrics"]),
                "positive_minus_zero_rows": len(report["positive_minus_zero"]), "pooled_whole_domain": summary,
                "front_diagnostics": fronts, "mass_doubled": mass, "dual_operator_fields": duals,
                "figures_manifest": str((output / "figures_manifest.json").relative_to(root)),
                "figures_manifest_sha256": sha(output / "figures_manifest.json"), "rendered_outputs": outputs,
                "cost_s": figure_manifest["cost_s"], "visual_inspection": {"state": "pending"},
                "limits": ["Saved-data recomputation only; no forward pass, differentiation, model fit, solver call or new reference calculation.",
                           "Null front widths and unfavorable values are retained without threshold/window tuning.",
                           "One seed and a historical zero control do not establish new-case generalization or independent training reproduction.",
                           "Inviscid reference comparison does not establish an exact viscous solution, entropy/energy/BVP validity or a momentum budget."]}
    write_new(readback_path, readback)
    print(json.dumps({"readback": str(readback_path.relative_to(root)), "sha256": sha(readback_path),
                      "figures": outputs, "checks": checks}, allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
