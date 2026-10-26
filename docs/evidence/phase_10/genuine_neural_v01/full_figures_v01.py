"""Three saved-data figures for the released completed genuine 2D full study.

Training, original evaluation and aggregate mass receipts must pass file-hash
guards before NumPy, Matplotlib or any predicted/reference array is loaded.
The script performs no model import, query, training or checkpoint selection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path

PROTOCOL_SHA256 = "82fd15b84fbc4333d1bb1953f137e47a172b98cf3826493db2aa4111d4ad5bd4"
EVALUATOR_SHA256 = "686eb401a359439566dcf61aff565a41e1f08dd4acdb4e815c7222e77199398e"
MASS_SCRIPT_SHA256 = "92a3e865a1bb97b5818659ab6fcf5edd2fd40a3c4b12b246f0ee657e1d79c7e4"
MASS_CONTRACT_SHA256 = (
    "9071959576225e31a3a0eed812bd615a89fd7d82a99a556ebc26f44e0b04fc70"
)
QUADRATURES = ("base", "spatial", "temporal")
FIELDS = ("h_m", "qx_m2_s", "qy_m2_s")
REGIONS = ("whole", "reference_gradient", "complement")
OUTPUT = "results/extensions/two_dimensional/neural_full_v01_figures"
QUALIFICATION = (
    "One seed and one wet rectangular case. The numerical reference retains "
    "material mesh sensitivity; reference sensitivity indicators are not error "
    "bounds. All saved cells and time nodes specified in each panel are retained."
)


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _write(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _npz(path, np):
    with np.load(path, allow_pickle=False) as saved:
        result = {name: saved[name].copy() for name in saved.files}
    _require(
        all(bool(np.isfinite(value).all()) for value in result.values()),
        "nonfinite saved figure input: " + str(path),
    )
    return result


def _guard(root, run, evaluation, mass):
    """Inspect bytes/control JSON only; imports and scientific loads follow."""
    inputs = {}

    def bind(path, expected=None):
        path = Path(path)
        observed = _sha(path)
        _require(
            expected is None or observed == expected, "input hash changed: " + str(path)
        )
        inputs[str(path.relative_to(root))] = observed

    def manifest(directory, files):
        for name, expected in files.items():
            _require(Path(name).name == name, "input manifest must use flat names")
            bind(directory / name, expected)

    release_path = (
        root / "docs/coordination/phase_10_2_neural_full_fit_release_v01.json"
    )
    release = _read(release_path)
    bind(release_path)
    _require(
        release["state"] == "released"
        and release["updates"] == 10000
        and release["protocol_sha256"] == PROTOCOL_SHA256
        and run == root / "runs" / release["run_id"]
        and evaluation == root / release["evaluation_output"]
        and mass
        == root / "results/extensions/two_dimensional/neural_full_v01_mass_totals",
        "figure inputs must be the released original full run/evaluation/mass sidecar",
    )
    summary, status, metadata = (
        _read(run / "summary.json"),
        _read(run / "status.json"),
        _read(run / "metadata.json"),
    )
    _require(
        summary.get("mode") == "full"
        and summary.get("state") == "completed"
        and summary.get("actual_updates") == 10000
        and summary.get("validation_evaluations") == 10001
        and summary.get("training_readback", {}).get("passed") is True
        and status.get("state") == "completed"
        and metadata.get("execution", {}).get("iterations") == 10000,
        "completed full 10000-update data required; pilot figures refused",
    )
    protocol = _read(run / "protocol.json")
    content = {k: v for k, v in protocol.items() if k != "protocol_sha256"}
    digest = hashlib.sha256(
        json.dumps(
            content, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    _require(
        digest == PROTOCOL_SHA256
        and protocol["protocol_sha256"] == PROTOCOL_SHA256
        and summary["protocol_sha256"] == PROTOCOL_SHA256
        and _read(run / "root_release.json") == release,
        "full release/protocol mismatch",
    )
    for name, expected in protocol["runtime_bindings"].items():
        bind(root / name, expected)
    for name, expected in protocol["artifact_bindings"].items():
        bind(run / name, expected)
    for name, key in (
        ("history.csv", "history_sha256"),
        ("checkpoint.pt", "checkpoint_sha256"),
        ("initial.pt", "initial_checkpoint_sha256"),
        ("latest.pt", "latest_checkpoint_sha256"),
    ):
        bind(run / name, summary[key])
    with (run / "history.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    _require(
        [int(row["step"]) for row in rows] == list(range(10001)),
        "full history is incomplete",
    )
    raw_report = _read(run / "raw_queries/raw_query_report.json")
    evaluated = _read(evaluation / "evaluation_receipt.json")
    analysis = evaluation / "analysis"
    analyzed = _read(analysis / "analysis_receipt.json")
    totals = _read(mass / "mass_totals_receipt.json")
    _require(
        evaluated["state"] == "completed"
        and evaluated["additional_neural_queries"] == 0
        and evaluated["actual_raw_queries"] == raw_report == summary["raw_queries"]
        and raw_report["actual_query_count"] == 13639680
        and analyzed["state"] == "completed_full10000update_quality_analysis"
        and analyzed["source_sha256"] == EVALUATOR_SHA256
        and analyzed["evaluation_input_bindings"] == protocol["evaluation_inputs"]
        and evaluated["evaluation_inputs"] == protocol["evaluation_inputs"]
        and analyzed["raw_query_report_sha256"]
        == _sha(run / "raw_queries/raw_query_report.json")
        and totals["state"] == "completed_saved_full_analysis_mass_totals"
        and totals["all_required_checks_passed"] is True
        and totals["additional_model_queries"] == 0
        and totals["script_sha256"] == MASS_SCRIPT_SHA256
        and totals["contract_file_sha256"] == MASS_CONTRACT_SHA256,
        "completed original full analysis and mass sidecar required",
    )
    for receipt in (evaluated, analyzed, totals):
        _require(
            receipt["protocol_sha256"] == PROTOCOL_SHA256
            and receipt["checkpoint_sha256"] == summary["checkpoint_sha256"],
            "analysis/sidecar protocol or checkpoint mismatch",
        )
    for name in ("summary.json", "status.json", "metadata.json", "protocol.json"):
        bind(run / name, totals["input_control_sha256"][name])
    bind(run / "root_release.json")
    bind(evaluation / "evaluation_receipt.json")
    bind(
        analysis / "analysis_receipt.json",
        totals["input_control_sha256"]["analysis_receipt.json"],
    )
    bind(
        run / "raw_queries/raw_query_report.json",
        totals["input_control_sha256"]["raw_query_report.json"],
    )
    bind(mass / "mass_totals_receipt.json")
    manifest(analysis, analyzed["output_sha256"])
    manifest(run / "raw_queries", raw_report["output_sha256"])
    manifest(mass, totals["output_sha256"])
    for name in (
        "docs/evidence/phase_10/genuine_neural_v01/full_mass_totals_postprocess_v01.py",
        "docs/evidence/phase_10/genuine_neural_v01/full_mass_totals_contract_v01.json",
    ):
        bind(root / name, release["bound_acceptance_inputs"][name])
    return inputs, summary


def _decorate(ax, ylabel):
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.tick_params(labelsize=9)


def _save(fig, path, caption, plt):
    fig.savefig(
        path,
        dpi=180,
        facecolor="white",
        metadata={
            "Software": "hydro-pinn saved-data full figures v01",
            "Description": caption,
        },
    )
    plt.close(fig)


def _fields_figure(data, destination, np, plt):
    index = len(data["t_s"]) - 1
    time_s = float(data["t_s"][index])
    _require(index == 49 and time_s == 0.25, "final field time must be saved index49")
    ref = data["reference_state_h_qx_qy"][index]
    pred = data["gauss3_state_h_qx_qy"][index]
    error = data["gauss3_minus_reference"][index]
    _require(
        ref.shape == pred.shape == error.shape == (80, 80, 3),
        "field figure shape changed",
    )
    fig, axes = plt.subplots(3, 3, figsize=(17, 12), sharex=True, sharey=True)
    labels = (("Depth h", "m"), ("Discharge qx", "m²/s"), ("Discharge qy", "m²/s"))
    ranges = []
    for row, (label, unit) in enumerate(labels):
        low, high = (
            float(min(ref[..., row].min(), pred[..., row].min())),
            float(max(ref[..., row].max(), pred[..., row].max())),
        )
        if low == high:
            low, high = low - 0.5, high + 0.5
        limit = float(np.max(np.abs(error[..., row])))
        error_limit = limit if limit else 1.0
        ranges.append(
            {
                "field": FIELDS[row],
                "state_color_limits": [low, high],
                "error_color_limits": [-error_limit, error_limit],
                "all_cells_included": True,
            }
        )
        for column, (values, title) in enumerate(
            (
                (ref, "Numerical reference"),
                (pred, "PINN Gauss3 means"),
                (error, "PINN minus reference"),
            )
        ):
            ax = axes[row, column]
            im = ax.pcolormesh(
                data["x_edges_m"],
                data["y_edges_m"],
                values[..., row],
                shading="flat",
                cmap="RdBu_r" if column == 2 else "viridis",
                vmin=-error_limit if column == 2 else low,
                vmax=error_limit if column == 2 else high,
            )
            ax.set_aspect("equal")
            ax.set_title(title + "\n" + label, fontsize=11)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            ax.set_xticks([-5, 0, 5])
            ax.set_yticks([-5, 0, 5])
            fig.colorbar(im, ax=ax, label=unit, shrink=0.8)
    caption = (
        "All 80×80 cell means at the final saved time. Error uses the saved "
        "Gauss3-minus-reference arrays. "
        "The reference is ref_rect320_c0225_v01 projected to the common grid. "
        + QUALIFICATION
    )
    fig.suptitle(f"Conserved cell means at t={time_s:.3f} s; seed11", fontsize=15)
    fig.text(0.5, 0.025, caption, ha="center", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(top=0.91, bottom=0.12, hspace=0.35, wspace=0.25)
    name = "conserved_final_time_v01.png"
    _save(fig, destination / name, caption, plt)
    return {
        "file": name,
        "caption": caption,
        "input": "comparison_fields_and_reference_sensitivity.npz",
        "time_index": index,
        "time_s": time_s,
        "x_indices": [0, 79],
        "y_indices": [0, 79],
        "conserved_component_indices": [0, 1, 2],
        "source_fields": [
            "reference_state_h_qx_qy",
            "gauss3_state_h_qx_qy",
            "gauss3_minus_reference",
        ],
        "color_ranges": ranges,
        "clipping_or_cell_exclusions": False,
    }


def _metrics_figure(metrics, sensitivity, times, destination, np, plt):
    fig, axes = plt.subplots(3, 2, figsize=(17, 12), sharex=True)
    colors = {
        "whole": "#1f77b4",
        "reference_gradient": "#d95f02",
        "complement": "#228833",
    }
    nulls, series = [], []
    for row, field in enumerate(FIELDS):
        unit = "m" if field == "h_m" else "m²/s"
        for column, (key, label) in enumerate(
            (("area_mean_MAE", "MAE"), ("area_mean_RMSE", "RMSE"))
        ):
            ax = axes[row, column]
            for rule, style in (("Gauss3", "-"), ("Gauss5", "--")):
                for region in REGIONS:
                    rows = metrics["metrics"][rule][region]["per_time"]
                    _require(
                        np.array_equal([r["t_s"] for r in rows], times),
                        "field-metric time axis changed",
                    )
                    values = [r["fields"][field][key] for r in rows]
                    missing = [i for i, value in enumerate(values) if value is None]
                    if missing:
                        nulls.append(
                            {
                                "rule": rule,
                                "region": region,
                                "field": field,
                                "metric": key,
                                "indices": missing,
                                "reasons": [
                                    rows[i]["fields"][field].get("reason")
                                    for i in missing
                                ],
                            }
                        )
                    ax.plot(
                        times,
                        [np.nan if v is None else v for v in values],
                        style,
                        color=colors[region],
                        linewidth=1.5,
                        label=rule
                        + " "
                        + region.replace("reference_gradient", "gradient"),
                    )
                    series.append(
                        {
                            "source": "field_metrics.json",
                            "rule": rule,
                            "region": region,
                            "field": field,
                            "metric": key,
                            "time_indices": [0, 49],
                            "undefined_indices": missing,
                        }
                    )
            rows = sensitivity["SI_amplitudes"]["Dsum"]["whole"]["per_time"]
            _require(
                np.array_equal([r["t_s"] for r in rows], times),
                "sensitivity time axis changed",
            )
            ax.plot(
                times,
                [r["fields"][field][key] for r in rows],
                ":",
                color="black",
                linewidth=2,
                label="Reference Dsum whole",
            )
            series.append(
                {
                    "source": "reference_sensitivity_amplitudes.json",
                    "indicator": "Dsum",
                    "region": "whole",
                    "field": field,
                    "metric": key,
                    "time_indices": [0, 49],
                }
            )
            ax.set_title(field.replace("_m2_s", "").replace("_m", "") + " " + label)
            _decorate(ax, label + " (" + unit + ")")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=4,
        fontsize=9,
    )
    caption = (
        "All 50 positive saved times; both Gauss rules and all three fixed regions. "
        "Undefined saved metrics remain gaps with reasons retained in the receipt. "
        "Dsum shows the saved whole-domain spatial+temporal reference "
        "sensitivity amplitude. " + QUALIFICATION
    )
    fig.suptitle("Conserved-field errors against the numerical reference", fontsize=15)
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(top=0.87, bottom=0.12, hspace=0.3)
    name = "field_errors_over_time_v01.png"
    _save(fig, destination / name, caption, plt)
    return {
        "file": name,
        "caption": caption,
        "series": series,
        "undefined_saved_metrics": nulls,
        "time_indices": [0, 49],
        "time_count": 50,
        "axis_scale": "linear",
        "value_clipping_or_region_exclusions": False,
    }


def _mass_figure(masses, totals, differences, destination, np, plt):
    fig, axes = plt.subplots(3, 2, figsize=(17, 13))
    colors = dict(zip(QUADRATURES, ("#1f77b4", "#d95f02", "#228833"), strict=True))
    series = []
    for name in QUADRATURES:
        mass, total, color = masses[name], totals[name], colors[name]
        _require(
            np.array_equal(mass["output_t_s"], total["output_t_s"]),
            "mass output axes differ",
        )
        _require(
            np.array_equal(
                total["edge_t_s"][total["output_indices"]], total["output_t_s"]
            ),
            "mass total output alignment changed",
        )
        for ax, field in zip(
            (axes[0, 0], axes[0, 1], axes[1, 0]),
            ("volume_m3", "corrected_balance_m3", "bias_removed_balance_m3"),
            strict=True,
        ):
            ax.plot(mass["output_t_s"], mass[field], color=color, label=name)
            series.append(
                {
                    "source": f"mass_{name}_analysis.npz",
                    "field": field,
                    "time_indices": [0, 50],
                }
            )
        for ax, fields in (
            (axes[1, 1], ("net_signed_rate_m3_s", "summed_absolute_rate_m3_s")),
            (axes[2, 0], ("net_signed_integral_m3", "summed_absolute_integral_m3")),
        ):
            for field, style, label in zip(
                fields, ("-", "--"), ("net signed", "sum absolute"), strict=True
            ):
                ax.plot(
                    total["edge_t_s"],
                    total[field],
                    style,
                    color=color,
                    label=name + " " + label,
                )
                series.append(
                    {
                        "source": f"mass_{name}_totals.npz",
                        "field": field,
                        "time_indices": [0, len(total["edge_t_s"]) - 1],
                        "all_edge_time_nodes": True,
                    }
                )
    for name, style in (("spatial", "-"), ("temporal", "--")):
        for field, color in (
            ("deltaM_m3", "#1f77b4"),
            ("deltaIout_m3", "#d95f02"),
            ("deltaB_m3", "#228833"),
        ):
            key = name + "_" + field
            axes[2, 1].plot(
                differences["output_t_s"],
                differences[key],
                style,
                color=color,
                label=name + " " + field.removesuffix("_m3"),
            )
            series.append(
                {
                    "source": "mass_spatial_and_temporal_differences.npz",
                    "field": key,
                    "time_indices": [0, 50],
                }
            )
    axes[0, 0].axhline(
        11.8, color="black", linestyle=":", label="initial target11.8 m³"
    )
    titles = (
        ("Volume including initial bias", "Volume (m³)"),
        ("Raw balance B = M −11.8 + Iout", "B (m³)"),
        ("Bias-subtracted balance B − B(0)", "Change (m³)"),
        ("All-node outward rates", "Rate (m³/s)"),
        ("All-node cumulative outward integrals", "Integral (m³)"),
        ("Quadrature changes relative to base", "Difference (m³)"),
    )
    for ax, (title, unit) in zip(axes.ravel(), titles, strict=True):
        ax.set_title(title, fontsize=11)
        _decorate(ax, unit)
        if ax is not axes[0, 0]:
            ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
        ax.legend(fontsize=8, ncol=2)
    caption = (
        "All three mass quadratures are shown. Raw B retains initial mass bias; "
        "B−B(0) is shown separately. Signed totals retain cancellation; "
        "absolute totals sum all four wall mismatches. Rates/integrals retain "
        "every 501/1001 edge time node. Spatial/temporal differences are indicators. "
        + QUALIFICATION
    )
    fig.suptitle("Mass bias, four-wall leakage and quadrature sensitivity", fontsize=15)
    fig.text(0.5, 0.02, caption, ha="center", va="bottom", fontsize=9, wrap=True)
    fig.subplots_adjust(top=0.92, bottom=0.12, hspace=0.35)
    name = "mass_budget_and_leakage_v01.png"
    _save(fig, destination / name, caption, plt)
    return {
        "file": name,
        "caption": caption,
        "series": series,
        "quadratures": list(QUADRATURES),
        "signed_and_absolute_totals_separate": True,
        "initial_bias_retained_and_separately_subtracted": True,
        "value_clipping_or_time_exclusions": False,
    }


def make_figures(run_dir, evaluation_dir, mass_totals_dir, *, root=Path(".")):
    started = time.perf_counter()
    root = Path(root).resolve()
    run, evaluation, mass = (
        root / Path(p) for p in (run_dir, evaluation_dir, mass_totals_dir)
    )
    destination = root / OUTPUT
    _require(not destination.exists(), "fixed figure output directory must be new")
    inputs, summary = _guard(root, run, evaluation, mass)
    guarded_s = time.perf_counter() - started

    import matplotlib
    import numpy as np

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    analysis = evaluation / "analysis"
    data = _npz(analysis / "comparison_fields_and_reference_sensitivity.npz", np)
    _require(
        data["t_s"].dtype == np.float64 and data["t_s"].shape == (50,),
        "positive time axis changed",
    )
    metrics = _read(analysis / "field_metrics.json")
    sensitivity = _read(analysis / "reference_sensitivity_amplitudes.json")
    masses = {
        name: _npz(analysis / f"mass_{name}_analysis.npz", np) for name in QUADRATURES
    }
    totals = {name: _npz(mass / f"mass_{name}_totals.npz", np) for name in QUADRATURES}
    differences = _npz(analysis / "mass_spatial_and_temporal_differences.npz", np)
    loaded_s = time.perf_counter() - started - guarded_s
    destination.mkdir(parents=True, exist_ok=False)
    try:
        rendered = time.perf_counter()
        figures = [
            _fields_figure(data, destination, np, plt),
            _metrics_figure(metrics, sensitivity, data["t_s"], destination, np, plt),
            _mass_figure(masses, totals, differences, destination, np, plt),
        ]
        rendering_s = time.perf_counter() - rendered
        pngs = {
            entry["file"]: {
                "sha256": _sha(destination / entry["file"]),
                "bytes": (destination / entry["file"]).stat().st_size,
            }
            for entry in figures
        }
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        _write(
            destination / "figures_receipt.json",
            {
                "state": "completed_saved_full_data_figures",
                "all_required_checks_passed": True,
                "protocol_sha256": PROTOCOL_SHA256,
                "checkpoint_sha256": summary["checkpoint_sha256"],
                "script_sha256": _sha(Path(__file__)),
                "input_sha256": inputs,
                "figures": figures,
                "outputs": pngs,
                "output_PNG_bytes": sum(item["bytes"] for item in pngs.values()),
                "wall_s": {
                    "control_and_hash_guards": guarded_s,
                    "imports_and_data_load": loaded_s,
                    "render_and_PNG_save": rendering_s,
                    "function_before_receipt": time.perf_counter() - started,
                },
                "process_lifetime_peak_RSS_bytes": int(
                    rss * (1 if platform.system() == "Darwin" else 1024)
                ),
                "environment": {
                    "numpy_version": np.__version__,
                    "matplotlib_version": matplotlib.__version__,
                    "backend": matplotlib.get_backend(),
                    "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR"),
                    "cache_directory": matplotlib.get_cachedir(),
                    "qualification": (
                        "Existing locked environment; font/cache warnings remain "
                        "in actual stderr. No installation or fresh-environment claim."
                    ),
                },
                "qualification": QUALIFICATION,
                "additional_model_queries": 0,
                "model_modules_imported": False,
                "saved_model_outputs_or_analysis_modified": False,
                "plotting_policy": (
                    "Use saved final index 49, all 80×80 cells, all 50 positive "
                    "metric times/all 3 regions/both Gauss rules, all 51 mass "
                    "output times and all 501/1001 edge nodes. Keep undefined "
                    "metrics as gaps with reasons; no value clipping, "
                    "neural-dependent mask or favorable subset selection."
                ),
            },
        )
    except BaseException as exc:
        _write(
            destination / "failure.json",
            {"type": type(exc).__name__, "message": str(exc)},
        )
        raise
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--mass-totals", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(make_figures(args.run, args.evaluation, args.mass_totals, root=args.root))


if __name__ == "__main__":
    main()
