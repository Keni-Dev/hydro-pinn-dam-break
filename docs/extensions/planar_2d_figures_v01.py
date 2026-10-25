"""Render three planar-consistency figures from saved final evaluation only.

Example after terminal full evaluation:
    uv run --frozen python docs/extensions/planar_2d_figures_v01.py \
        --input-dir results/extensions/planar_2d/full_v01_evaluation \
        --output-dir results/extensions/planar_2d/full_v01_figures

No checkpoint loading, model query, training, or pilot evaluation occurs here.
Existing destinations are refused. Input/output hashes are saved in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BLUE = "#24688c"
ORANGE = "#c36827"
GREEN = "#3b7c60"
PURPLE = "#75518e"
EDGE_COLORS = (BLUE, ORANGE, GREEN, PURPLE)
EDGES = ("left", "right", "bottom", "top")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#dfe3e6",
            "grid.linewidth": 0.6,
            "grid.alpha": 0.8,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _finish(fig: plt.Figure, provenance: dict, footer: str) -> None:
    fig.text(0.02, 0.028, footer, ha="left", va="bottom", fontsize=8)
    fig.text(
        0.98,
        0.009,
        f"Planar E1 seed 11 | checkpoint {provenance['checkpoint_sha256'][:12]} "
        f"| protocol {provenance['protocol_sha256'][:12]}",
        ha="right",
        va="bottom",
        fontsize=7,
        color="#555b61",
    )


def _save(fig: plt.Figure, destination: Path, stem: str) -> dict:
    paths = []
    for suffix in ("png", "pdf"):
        path = destination / f"{stem}.{suffix}"
        fig.savefig(path, dpi=180)
        paths.append({"path": path.name, "sha256": _sha(path)})
    plt.close(fig)
    return {"figure": stem, "outputs": paths}


def _profiles(network: dict, reference: dict, provenance: dict) -> plt.Figure:
    x, t = network["x_m"], network["t_s"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharex=True)
    for col, selected_time in enumerate((0.1, 0.25, 0.5)):
        hits = np.flatnonzero(np.isclose(t, selected_time, rtol=0, atol=1e-12))
        if len(hits) != 1:
            raise ValueError(f"exact profile time unavailable: {selected_time}s")
        ti = int(hits[0])
        for row, (key, label) in enumerate(
            (("h_m", "Depth h [m]"), ("qx_m2_s", "Longitudinal discharge qx [m²/s]"))
        ):
            axis = axes[row, col]
            data = network[key][ti]
            axis.fill_between(
                x,
                data.min(axis=0),
                data.max(axis=0),
                color=BLUE,
                alpha=0.22,
                linewidth=0,
                label="Network range across y",
            )
            axis.plot(
                x, data.mean(axis=0), color=BLUE, lw=1.6, label="Network mean across y"
            )
            axis.plot(
                x,
                reference[key][ti, 0],
                color="#22272b",
                lw=1.2,
                linestyle="--",
                label="Analytical E1",
            )
            if row == 0:
                axis.set_title(f"t = {selected_time:.2f} s")
            if col == 0:
                axis.set_ylabel(label)
            if row == 1:
                axis.set_xlabel("x [m]")
            axis.set_xlim(float(x[0]), float(x[-1]))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.925),
        ncol=3,
        frameon=False,
    )
    fig.suptitle("Longitudinal profiles and variation across y", y=0.98)
    fig.subplots_adjust(
        left=0.09, right=0.98, top=0.84, bottom=0.13, hspace=0.22, wspace=0.22
    )
    _finish(
        fig,
        provenance,
        "400 x centres × 21 y centres. Shading shows sampled minimum/maximum "
        "across y; point values, not cell averages.",
    )
    return fig


def _series(errors: dict, name: str, key: str, length: int) -> np.ndarray:
    series = np.asarray(errors["y_variation"][name][key], dtype=float)
    if series.shape != (length,) or not np.isfinite(series).all():
        raise ValueError(f"invalid saved y-variation series: {name}/{key}")
    return series


def _transverse(
    network: dict, reference: dict, errors: dict, provenance: dict
) -> plt.Figure:
    t = network["t_s"]
    qy = network["qy_m2_s"]
    velocity = qy / network["h_m"]
    wet = reference["reference_wet_mask"]
    qy_rms = np.sqrt(np.mean(qy**2, axis=(1, 2)))
    qy_max = np.max(np.abs(qy), axis=(1, 2))
    v_rms = np.array(
        [np.sqrt(np.mean(v[m] ** 2)) for v, m in zip(velocity, wet, strict=True)]
    )
    v_max = np.array([np.max(np.abs(v[m])) for v, m in zip(velocity, wet, strict=True)])
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.6), sharex=True)
    for axis, rms, maximum, title, units in (
        (axes[0, 0], qy_rms, qy_max, "Absolute transverse discharge", "qy [m²/s]"),
        (axes[0, 1], v_rms, v_max, "Absolute transverse velocity", "v [m/s]"),
    ):
        axis.plot(t, rms, color=BLUE, lw=1.8, label="RMS")
        axis.plot(t, maximum, color=ORANGE, lw=1.4, label="Sampled max absolute")
        axis.set_title(title)
        axis.set_ylabel(units)
        axis.legend(frameon=False)
    axis = axes[1, 0]
    axis.plot(
        t,
        _series(errors, "h_m", "per_time_rms", len(t)),
        color=BLUE,
        lw=1.8,
        label="RMS deviation",
    )
    axis.plot(
        t,
        _series(errors, "h_m", "per_time_max_absolute", len(t)),
        color=ORANGE,
        lw=1.4,
        label="Sampled max absolute deviation",
    )
    axis.set_title("Depth variation across y")
    axis.set_ylabel("h − mean_y(h) [m]")
    axis.legend(frameon=False)
    axis = axes[1, 1]
    for name, label, color in (("qx_m2_s", "qx", BLUE), ("qy_m2_s", "qy", GREEN)):
        axis.plot(
            t,
            _series(errors, name, "per_time_rms", len(t)),
            color=color,
            lw=1.8,
            label=f"{label} RMS deviation",
        )
        axis.plot(
            t,
            _series(errors, name, "per_time_max_absolute", len(t)),
            color=color,
            lw=1.1,
            linestyle="--",
            label=f"{label} sampled max absolute deviation",
        )
    axis.set_title("Discharge variation across y")
    axis.set_ylabel("q − mean_y(q) [m²/s]")
    axis.legend(frameon=False)
    for axis in axes.flat:
        axis.ticklabel_format(
            axis="y", style="sci", scilimits=(-3, 3), useMathText=True
        )
        axis.set_xlim(float(t[0]), float(t[-1]))
    for axis in axes[1]:
        axis.set_xlabel("t [s]")
    fig.suptitle("Transverse state and sampled y variation", y=0.98)
    fig.subplots_adjust(
        left=0.09, right=0.98, top=0.9, bottom=0.14, hspace=0.33, wspace=0.28
    )
    _finish(
        fig,
        provenance,
        "Analytical qy = v = 0; relative transverse error is undefined. "
        "Velocity uses the reference-wet mask. Maxima are sampled.",
    )
    return fig


def _mass(mass: dict, edges: dict, provenance: dict) -> plt.Figure:
    base, refined = mass["base"], mass["refined"]
    t = np.asarray(base["report_t_s"])
    if not np.array_equal(t, refined["report_t_s"]):
        raise ValueError("mass reporting times differ between quadratures")
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 7.8))
    labels = (
        f"Base {base['axes']['volume_nx']} × {base['axes']['volume_ny']}",
        f"Refined {refined['axes']['volume_nx']} × {refined['axes']['volume_ny']}",
    )
    for report, color, style, label in (
        (base, BLUE, "-", labels[0]),
        (refined, ORANGE, "--", labels[1]),
    ):
        axes[0, 0].plot(
            t,
            report["corrected_balance_m3"],
            color=color,
            linestyle=style,
            lw=1.7,
            label=label,
        )
        axes[0, 0].scatter(
            [0], [report["predicted_initial_bias_m3"]], color=color, s=24, zorder=4
        )
        axes[0, 1].plot(
            t,
            report["balance_without_initial_bias_m3"],
            color=color,
            linestyle=style,
            lw=1.7,
            label=label,
        )
    axes[0, 0].set_title("Corrected balance retains initial-volume bias")
    axes[0, 0].set_ylabel("M(t) − 55 + ∫ outward flux dt [m³]")
    axes[0, 0].text(
        0.03,
        0.04,
        f"t = 0 bias: base {base['predicted_initial_bias_m3']:+.4g} m³; "
        f"refined {refined['predicted_initial_bias_m3']:+.4g} m³",
        transform=axes[0, 0].transAxes,
        fontsize=8,
    )
    axes[0, 1].set_title("Balance after subtracting initial bias")
    axes[0, 1].set_ylabel("M(t) − M(0) + ∫ outward flux dt [m³]")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False)
    for edge, color in zip(EDGES, EDGE_COLORS, strict=True):
        axes[1, 0].plot(
            edges["t_s"],
            edges[f"{edge}_outward_integral_m3"],
            color=color,
            lw=1.5,
            label=edge.capitalize(),
        )
    axes[1, 0].set_title("Four signed outward edge contributions, refined")
    axes[1, 0].set_ylabel("∫ edge outward flux dt [m³]")
    axes[1, 0].legend(frameon=False, ncol=2)
    mismatch = [
        refined["edge"][edge]["absolute_normal_flux_mismatch_integral_m3"]
        for edge in EDGES
    ]
    axes[1, 1].bar(
        [e.capitalize() for e in EDGES], mismatch, color=EDGE_COLORS, width=0.6
    )
    axes[1, 1].set_title("Absolute edge mismatch without cancellation")
    axes[1, 1].set_ylabel("∫∫ |normal discharge − target| ds dt [m³]")
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.axhline(0, color="#555b61", lw=0.7, zorder=0)
        axis.set_xlabel("t [s]")
        axis.set_xlim(0, float(t[-1]))
    fig.suptitle("Area-volume and four-edge mass diagnostics", y=0.98)
    fig.subplots_adjust(
        left=0.09, right=0.98, top=0.9, bottom=0.14, hspace=0.36, wspace=0.3
    )
    _finish(
        fig,
        provenance,
        "Prescribed volume 55 m³. Midpoint space and trapezoid time; refinement "
        "changes x, y and time together. Sensitivity is not an error bound.",
    )
    return fig


def render(input_dir: Path, output_dir: Path) -> dict:
    input_dir, output_dir = input_dir.resolve(), output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"preserve existing figure directory: {output_dir}")
    required = (
        "network_fields.npz",
        "reference_fields.npz",
        "errors.json",
        "mass.json",
        "edges_1.npz",
        "provenance.json",
        "query_receipt.json",
    )
    for name in required:
        if not (input_dir / name).is_file():
            raise ValueError(f"missing final evaluation input: {name}")
    provenance = _json(input_dir / "provenance.json")
    if (
        provenance.get("scope") != "planar consistency only"
        or not provenance.get("actual_costs", {}).get(
            "analytical_model_errors_computed"
        )
        or not _json(input_dir / "query_receipt.json").get(
            "analytical_model_errors_computed"
        )
    ):
        raise ValueError("figures require saved final evaluation; pilot inputs refused")
    for key in ("checkpoint_sha256", "protocol_sha256"):
        if len(provenance.get(key, "")) != 64:
            raise ValueError(f"missing final provenance hash: {key}")
    network = dict(np.load(input_dir / "network_fields.npz", allow_pickle=False))
    reference = dict(np.load(input_dir / "reference_fields.npz", allow_pickle=False))
    expected = tuple(len(network[key]) for key in ("t_s", "y_m", "x_m"))
    for key in ("t_s", "y_m", "x_m"):
        if not np.array_equal(network[key], reference[key]):
            raise ValueError(f"network/reference axis mismatch: {key}")
    for data in (network, reference):
        for key in ("h_m", "qx_m2_s", "qy_m2_s"):
            if data[key].shape != expected or not np.isfinite(data[key]).all():
                raise ValueError(f"invalid saved (time,y,x) field: {key}")
        if np.any(data["h_m"] <= 0):
            raise ValueError("wet E1 plot requires positive saved depths")
    if np.any(reference["qy_m2_s"]):
        raise ValueError("planar analytical transverse reference must be zero")
    output_dir.mkdir(parents=True, exist_ok=False)
    _style()
    entries = [
        _save(
            _profiles(network, reference, provenance),
            output_dir,
            "01_longitudinal_profiles",
        ),
        _save(
            _transverse(
                network, reference, _json(input_dir / "errors.json"), provenance
            ),
            output_dir,
            "02_transverse_and_y_variation",
        ),
        _save(
            _mass(
                _json(input_dir / "mass.json"),
                dict(np.load(input_dir / "edges_1.npz", allow_pickle=False)),
                provenance,
            ),
            output_dir,
            "03_mass_and_edges",
        ),
    ]
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "state": "rendered_from_saved_final_evaluation",
        "scope": "planar consistency only",
        "input_dir": str(input_dir),
        "input_sha256": {name: _sha(input_dir / name) for name in required},
        "script_sha256": _sha(Path(__file__).resolve()),
        "checkpoint_sha256": provenance["checkpoint_sha256"],
        "protocol_sha256": provenance["protocol_sha256"],
        "training_run": provenance["run"],
        "field_shape_time_y_x": list(expected),
        "field_location": "point_value",
        "figure_entries": entries,
        "limits": [
            "No fits, checkpoint reloads or model queries were executed.",
            "Minima/maxima are sampled over saved axes.",
            "Combined quadrature sensitivity is not an error bound.",
            "One planar seed does not establish genuine2D performance.",
        ],
    }
    with (output_dir / "figure_manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = render(args.input_dir, args.output_dir)
    print(
        json.dumps(
            {
                "state": manifest["state"],
                "figures": len(manifest["figure_entries"]),
                "manifest": str(args.output_dir / "figure_manifest.json"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
