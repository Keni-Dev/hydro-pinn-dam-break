"""Three saved-data figures for the fixed-n wet flat-bed friction study.

This helper imports no model, solver, checkpoint, or training code. It reads the
completed full evaluation packet, refuses to overwrite a figure directory, and
records input/helper/output hashes. It performs plotting reductions only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REFERENCE = "#1f2933"
PINN = "#b2452f"
FRONT = "#437a9c"
OUTSIDE = "#65843c"
SENSITIVITY = "#825ba0"
UNCERTAINTY = "#9a9a9a"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as arrays:
        result = {name: arrays[name] for name in arrays.files}
    if any(not np.isfinite(value).all() for value in result.values()):
        raise ValueError(f"nonfinite saved array in {path}")
    return result


def read_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.2,
            "grid.linewidth": 0.5,
            "legend.frameon": False,
            "savefig.dpi": 180,
            "pdf.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, output: Path, name: str) -> list[Path]:
    paths = []
    for suffix in ("png", "pdf"):
        path = output / f"{name}.{suffix}"
        fig.savefig(path, facecolor="white", bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def profiles(pred: dict, ref: dict, output: Path) -> list[Path]:
    fig, axes = plt.subplots(3, 3, figsize=(14, 10.5), sharex=True, sharey="row")
    fields = ("h_m", "q_m2_s", "cell_ratio_velocity_m_s")
    labels = (
        r"Cell-average depth $\bar h$ (m)",
        r"Cell-average discharge $\bar q$ (m²/s)",
        r"Cell-ratio velocity $\bar q/\bar h$ (m/s)",
    )
    x = ref["x_m"]
    for column, target in enumerate((0.10, 0.25, 0.50)):
        matches = np.flatnonzero(np.isclose(ref["t_s"], target, rtol=0, atol=1e-14))
        if len(matches) != 1:
            raise ValueError(f"profile time {target} absent or ambiguous")
        row_index = int(matches[0])
        centre, width = (
            ref["front_x_m"][row_index],
            ref["front_half_width_m"][row_index],
        )
        for row, name in enumerate(fields):
            ax = axes[row, column]
            if row < 2:
                truth, values = ref[name][row_index], pred[name][row_index]
                key = (
                    "h_measured_uncertainty_m"
                    if row == 0
                    else "q_measured_uncertainty_m2_s"
                )
                uncertainty = ref[key][row_index]
                ax.fill_between(
                    x,
                    truth - uncertainty,
                    truth + uncertainty,
                    color=UNCERTAINTY,
                    alpha=0.38,
                    linewidth=0,
                )
            else:
                wet = ref["h_m"][row_index] > 0.001
                truth = np.where(
                    wet, ref["q_m2_s"][row_index] / ref["h_m"][row_index], np.nan
                )
                values = np.where(
                    wet, pred["q_m2_s"][row_index] / pred["h_m"][row_index], np.nan
                )
            ax.axvspan(
                centre - width, centre + width, color=FRONT, alpha=0.12, linewidth=0
            )
            ax.axvline(centre, color=FRONT, alpha=0.55, linewidth=0.7, linestyle=":")
            ax.plot(x, truth, color=REFERENCE, linewidth=1.6)
            ax.plot(x, values, color=PINN, linewidth=1.3)
            if row == 0:
                ax.set_title(f"t = {target:.2f} s")
            if column == 0:
                ax.set_ylabel(labels[row])
            if row == 2:
                ax.set_xlabel("x (m)")
            ax.set_xlim(-5, 5)
    fig.suptitle("Wet flat-bed Manning flow, n = 0.03 s/m^(1/3)", fontsize=15, y=0.99)
    handles = [
        Line2D([], [], color=REFERENCE, label="HLLE numerical reference"),
        Line2D([], [], color=PINN, label="PINN, three-point cell quadrature"),
        Patch(
            color=UNCERTAINTY, alpha=0.38, label="Measured h/q reference uncertainty"
        ),
        Patch(color=FRONT, alpha=0.12, label="Frozen numerical-front band"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.956), ncol=2)
    fig.text(
        0.5,
        0.015,
        "Reference cell averages use conservative projection; "
        "PINN h/q use cell integration. "
        "Velocity is q̄/h̄ with the fixed reference-wet mask.\n"
        "Gray envelopes show measured grid/time differences, "
        "not confidence intervals or rigorous bounds.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.075, right=0.99, bottom=0.085, top=0.875, hspace=0.19, wspace=0.12
    )
    return save_figure(fig, output, "friction_profiles_v01")


def error_curves(
    pred: dict, refined: dict, ref: dict, report: dict, output: Path
) -> list[Path]:
    fig, axes = plt.subplots(3, 2, figsize=(13.5, 11))
    fields = ("h_m", "q_m2_s", "cell_ratio_velocity_m_s")
    units = ("m", "m²/s", "m/s")
    names = ("Cell-average depth", "Cell-average discharge", "Cell-ratio velocity")
    regions = (
        ("whole", "Whole domain", REFERENCE),
        ("numerical_front_band", "Numerical-front band", FRONT),
        ("outside_band", "Outside band", OUTSIDE),
    )
    times = np.array([entry["t_s"] for entry in report["per_time"]])
    if not np.array_equal(times, pred["t_s"]):
        raise ValueError("error-report times differ from common-cell arrays")
    for row, (field, unit, name) in enumerate(zip(fields, units, names, strict=True)):
        left, right = axes[row]
        for region, label, color in regions:
            for column, metric in enumerate(("rmse", "relative_l2_pct")):
                values = [
                    entry["regions"][region][field][metric]
                    for entry in report["per_time"]
                ]
                axes[row, column].plot(
                    times,
                    np.array(values, dtype=float),
                    color=color,
                    linewidth=1.4,
                    label=label,
                )
        if row < 2:
            key = (
                "h_measured_uncertainty_m"
                if row == 0
                else "q_measured_uncertainty_m2_s"
            )
            indicator = np.sqrt(np.mean(ref[key] ** 2, axis=1))
            difference = pred[field] - refined[field]
            left.plot(
                times,
                indicator,
                color=UNCERTAINTY,
                linestyle="--",
                linewidth=1.5,
                label="Whole-domain reference indicator",
            )
        else:
            wet = ref["h_m"] > 0.001
            difference = (
                pred["q_m2_s"] / pred["h_m"] - refined["q_m2_s"] / refined["h_m"]
            )
            difference = np.where(wet, difference, np.nan)
        sensitivity = np.sqrt(np.nanmean(difference**2, axis=1))
        left.plot(
            times,
            sensitivity,
            color=SENSITIVITY,
            linestyle=":",
            linewidth=1.8,
            label="Whole-domain Gauss 3/5 sensitivity",
        )
        left.set_yscale("symlog", linthresh=1e-8)
        left.set_ylabel(f"{name}\nRMSE ({unit}); symlog")
        right.set_ylabel(f"{name}\nRelative L2 error (%)")
        for ax in (left, right):
            ax.set_xlim(0, 0.5)
            ax.set_xlabel("t (s)")
            ax.set_ylim(bottom=0)
        left.legend(loc="best", fontsize=8)
        if row == 0:
            right.legend(loc="best", fontsize=9)
    fig.suptitle(
        "Errors over the frozen 400-cell, 50-time comparison", fontsize=15, y=0.99
    )
    fig.text(
        0.5,
        0.015,
        "Reference h/q indicators are cellwise spatial-plus-time differences "
        "reduced by RMSE. "
        "Gauss 3/5 sensitivity is a saved-query difference.\n"
        "Neither is a rigorous bound. Velocity uses q̄/h̄ and "
        "the reference h̄ > 0.001 m mask; undefined relative errors appear as gaps.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.085, right=0.99, bottom=0.085, top=0.94, hspace=0.36, wspace=0.28
    )
    return save_figure(fig, output, "friction_errors_v01")


def mass_and_boundaries(mass: dict, boundaries: list[dict], output: Path) -> list[Path]:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    base, refined = mass["base"], mass["refined"]
    t = np.array(base["report_t_s"])
    if not np.array_equal(t, refined["report_t_s"]):
        raise ValueError("mass quadrature report axes differ")
    balance, sensitivity, depth, discharge = axes.ravel()
    for level, (report, linestyle, label) in enumerate(
        ((base, "-", "Base"), (refined, "--", "Refined"))
    ):
        balance.plot(
            t,
            report["corrected_balance_m2"],
            color=PINN,
            linestyle=linestyle,
            linewidth=1.5,
            label=f"{label}: corrected balance",
        )
        balance.plot(
            t,
            report["balance_without_initial_bias_m2"],
            color=REFERENCE,
            linestyle=linestyle,
            linewidth=1.5,
            label=f"{label}: initial bias subtracted",
        )
        walls = boundaries[level]
        for side, color, target, name in (
            (0, FRONT, 1.0, "Left"),
            (1, PINN, 0.1, "Right"),
        ):
            depth.plot(
                walls["t_s"],
                walls["h_m"][:, side] - target,
                color=color,
                linestyle=linestyle,
                linewidth=1.4,
                label=f"{name}, {label.lower()}",
            )
            discharge.plot(
                walls["t_s"],
                walls["q_m2_s"][:, side],
                color=color,
                linestyle=linestyle,
                linewidth=1.4,
                label=f"{name}, {label.lower()}",
            )
    bias_base, bias_fine = (
        base["predicted_initial_bias_m2"],
        refined["predicted_initial_bias_m2"],
    )
    balance.text(
        0.03,
        0.97,
        f"Predicted t=0 volume bias\nBase {bias_base:+.6g} m²; "
        f"refined {bias_fine:+.6g} m²",
        transform=balance.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    for key, color, label in (
        ("volume_m2", FRONT, "Volume: refined − base"),
        ("outward_integral_m2", OUTSIDE, "Outward flux integral: refined − base"),
        ("corrected_balance_m2", PINN, "Corrected balance: refined − base"),
    ):
        sensitivity.plot(
            t,
            np.array(refined[key]) - np.array(base[key]),
            color=color,
            linewidth=1.5,
            label=label,
        )
    balance.set_title("Two-boundary mass balance")
    balance.set_ylabel("Balance (m² per unit width)")
    sensitivity.set_title("Combined spatial/time quadrature sensitivity")
    sensitivity.set_ylabel("Difference (m² per unit width)")
    depth.set_title("Individual boundary depth mismatches")
    depth.set_ylabel("h − reservoir target (m)")
    discharge.set_title("Individual boundary discharge mismatches")
    discharge.set_ylabel("q − 0 (m²/s per unit width)")
    for ax in axes.ravel():
        ax.axhline(0, color=UNCERTAINTY, linewidth=0.7)
        ax.set_xlim(0, 0.5)
        ax.set_xlabel("t (s)")
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Mass and reservoir-boundary diagnostics", fontsize=15, y=0.99)
    fig.text(
        0.5,
        0.015,
        "Corrected balance = M(t) − 5.5 + ∫(qR − qL)dt; initial bias = M(0) − 5.5. "
        "Both endpoints enter the flux integral.\n"
        "Base/refined quadrature: 800/1600 midpoint cells "
        "and 999/1999 frozen trapezoid nodes. "
        "Differences measure sensitivity, not a rigorous bound.",
        ha="center",
        fontsize=9,
    )
    fig.subplots_adjust(
        left=0.075, right=0.99, bottom=0.095, top=0.93, hspace=0.28, wspace=0.22
    )
    return save_figure(fig, output, "friction_mass_boundaries_v01")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation-dir",
        type=Path,
        default=Path("results/extensions/friction/full_v01_evaluation"),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path("results/extensions/friction/reference_v01/common_reference.npz"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/extensions/friction/full_v01_figures"),
    )
    args = parser.parse_args()
    start = time.perf_counter()
    source = Path(__file__).resolve()
    required = [
        args.reference,
        *[
            args.evaluation_dir / name
            for name in (
                "common_cell_gauss3.npz",
                "common_cell_gauss5.npz",
                "errors.json",
                "mass.json",
                "boundaries_0.npz",
                "boundaries_1.npz",
                "provenance.json",
                "query_receipt.json",
                "pinn_cell_quadrature_sensitivity.json",
                "numerical_reference_uncertainty.json",
            )
        ],
    ]
    if not all(path.is_file() for path in required):
        raise ValueError(
            "completed full evaluation packet required; saved inputs are missing"
        )
    provenance = read_json(args.evaluation_dir / "provenance.json")
    run = Path(provenance["run"])
    run_inputs = [
        run / name for name in ("protocol.json", "summary.json", "status.json")
    ]
    if not all(path.is_file() for path in run_inputs):
        raise ValueError("bound full-run records are missing")
    required.extend(run_inputs)
    protocol, summary, status = (read_json(path) for path in run_inputs)
    if (
        summary["mode"] != "full"
        or summary["actual_updates"] != 10000
        or status["state"] != "completed"
    ):
        raise ValueError("released completed full10000update run required")
    if (
        protocol["config"]["manning_n_s_m_minus_one_third"] != 0.03
        or provenance["protocol_sha256"] != protocol["protocol_sha256"]
    ):
        raise ValueError("fixed positive-n study/protocol mismatch")
    reference_key = protocol["config"]["reference_packet"] + "/common_reference.npz"
    if digest(args.reference) != protocol["reference_sha256"][reference_key]:
        raise ValueError("bound numerical reference hash mismatch")
    receipt = read_json(args.evaluation_dir / "query_receipt.json")
    if (
        receipt.get("neural_errors_computed") is not True
        or receipt.get("mass_quality_reported") is not True
    ):
        raise ValueError("pilot/fixture packet refused")
    inputs = {str(path.resolve()): digest(path) for path in required}
    pred = load_npz(args.evaluation_dir / "common_cell_gauss3.npz")
    refined = load_npz(args.evaluation_dir / "common_cell_gauss5.npz")
    ref = load_npz(args.reference)
    for values in (pred, refined):
        for name in ("t_s", "cell_edges_m", "x_m"):
            if not (
                np.allclose(values[name], ref[name], atol=1e-15, rtol=0)
                if name == "t_s"
                else np.array_equal(values[name], ref[name])
            ):
                raise ValueError(f"reference/prediction {name} mismatch")
        if values["h_m"].shape != (50, 400) or np.any(values["h_m"] <= 0):
            raise ValueError("positive-depth full comparison arrays required")
    boundaries = [load_npz(args.evaluation_dir / f"boundaries_{i}.npz") for i in (0, 1)]
    errors, mass = (
        read_json(args.evaluation_dir / "errors.json"),
        read_json(args.evaluation_dir / "mass.json"),
    )
    style()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    paths, timings = [], {}
    for name, function, arguments in (
        ("profiles", profiles, (pred, ref, args.output_dir)),
        ("errors", error_curves, (pred, refined, ref, errors, args.output_dir)),
        ("mass_boundaries", mass_and_boundaries, (mass, boundaries, args.output_dir)),
    ):
        now = time.perf_counter()
        paths.extend(function(*arguments))
        timings[name] = time.perf_counter() - now
    if any(digest(Path(path)) != value for path, value in inputs.items()):
        raise ValueError("saved inputs changed during rendering")
    ledger = {
        "schema_version": 1,
        "state": "rendered_from_saved_full_evaluation",
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "helper": {"path": str(source), "sha256": digest(source)},
        "inputs_sha256": inputs,
        "outputs": [
            {
                "path": str(path.resolve()),
                "sha256": digest(path),
                "bytes": path.stat().st_size,
            }
            for path in paths
        ],
        "provenance": provenance,
        "authoritative_physics": protocol["physical_case"],
        "metadata_qualification": (
            "Generic E1 metadata contains inherited source-free analytical fields. "
            "Figures use this bound extension protocol and numerical reference."
        ),
        "figure_count": 3,
        "formats": ["PNG", "PDF"],
        "cost_s": timings,
        "wall_s_including_input_load_and_output": time.perf_counter() - start,
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "platform": platform.platform(),
        },
        "operations": (
            "Saved-array/JSON reductions and plotting only; "
            "no model, reference solve, checkpoint, or training imports"
        ),
        "limits": [
            "Single synthetic wet flat-bed n=.03 study; "
            "no analytical friction solution or field validation",
            "Reference uncertainty shading is measured spatial/time difference, "
            "not a rigorous bound or confidence interval",
            "Velocity is qbar/hbar, neither point velocity nor averaged point velocity",
            "Mass quadrature and Gauss3/5 sensitivity are numerical differences, "
            "not error bounds",
            "Plots use sampled times; no continuous-time extrema "
            "or native application/human acceptance claimed",
        ],
    }
    with (args.output_dir / "figure_ledger_v01.json").open("x") as handle:
        json.dump(ledger, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "state": ledger["state"],
                "figure_count": 3,
                "output_dir": str(args.output_dir),
                "wall_s": ledger["wall_s_including_input_load_and_output"],
            }
        )
    )


if __name__ == "__main__":
    main()
