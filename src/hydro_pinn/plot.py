"""Regenerate traceable Phase 06 figures from saved arrays and metric rows.

This module never trains, selects checkpoints, evaluates a reference, or fills
missing output times. A failed run is represented in the status figure and is
excluded from successful-output curves.
"""

from __future__ import annotations

import hashlib
import os
import textwrap
from pathlib import Path
from typing import Any

# Bundled DejaVu fonts keep headless exports portable and avoid OS font probes.
os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hydro_pinn.records import write_json_atomic

FIELDS = {"h_m": "Depth (m)", "q_m2_s": "Discharge (m²/s)", "u_m_s": "Velocity (m/s)"}
METHODS = {
    "analytical_reference": "Analytical reference",
    "lax_wendroff": "Lax–Wendroff",
    "pinn": "PINN",
}


class PlotError(ValueError):
    """Saved plotting evidence is inconsistent or incomplete."""


def _label(data: dict) -> str:
    label = METHODS.get(data.get("method"), str(data.get("method", "Unknown")))
    if data.get("variant"):
        label += f" {data['variant']}"
    if data.get("seed") is not None:
        label += f" seed {data['seed']}"
    if data.get("cells") is not None and data.get("method") == "lax_wendroff":
        label += f" N={data['cells']}"
    return label


def _style(data: dict) -> dict:
    if data.get("method") == "analytical_reference":
        return {"color": "#222222", "linestyle": "-"}
    if data.get("method") == "lax_wendroff":
        colors = {100: "#56B4E9", 200: "#0072B2", 400: "#5D3A9B", 800: "#777777"}
        return {"color": colors.get(data.get("cells"), "#0072B2"), "linestyle": "-"}
    return {
        "color": "#D55E00" if data.get("variant") == "baseline" else "#009E73",
        "linestyle": {11: "-", 22: "--", 33: ":"}.get(data.get("seed"), "-"),
    }


def _arrays(data: dict) -> tuple[np.ndarray, ...]:
    axes = [np.asarray(data[key], dtype=float) for key in ("x_m", "t_s")]
    if any(axis.ndim != 1 or not len(axis) for axis in axes):
        raise PlotError(
            f"{data['run_id']}: axes must be nonempty one-dimensional arrays"
        )
    if any(not np.isfinite(axis).all() for axis in axes):
        raise PlotError(f"{data['run_id']}: nonfinite coordinate")
    if any(np.any(np.diff(axis) <= 0) for axis in axes):
        raise PlotError(f"{data['run_id']}: axes must be strictly increasing")
    fields = [np.asarray(data[key], dtype=float) for key in ("h_m", "q_m2_s")]
    shape = (len(axes[1]), len(axes[0]))
    if any(field.shape != shape or not np.isfinite(field).all() for field in fields):
        raise PlotError(
            f"{data['run_id']}: fields must be finite arrays of shape {shape}"
        )
    return *axes, *fields


def _value(metric: Any) -> float:
    value = metric.get("value") if isinstance(metric, dict) else metric
    return float(value) if value is not None else float("nan")


def _note(ax: plt.Axes, message: str) -> None:
    ax.text(
        0.5, 0.5, message, transform=ax.transAxes, ha="center", va="center", wrap=True
    )


def _series(ax: plt.Axes, x: Any, y: Any, label: str, **kwargs: Any) -> bool:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or not len(x):
        return False
    # NaN keeps each missing observation as a break in the line.
    if not np.isfinite(x).all() or np.any(np.diff(x) <= 0):
        raise PlotError(f"{label}: curve coordinates must be finite and increasing")
    (line,) = ax.plot(x, y, label=label, **kwargs)
    finite = np.isfinite(y)
    isolated = finite & ~np.r_[False, finite[:-1]] & ~np.r_[finite[1:], False]
    if np.any(isolated):
        ax.plot(
            x[isolated],
            y[isolated],
            linestyle="none",
            marker="o",
            markersize=3,
            color=line.get_color(),
            label="_nolegend_",
        )
    return bool(np.isfinite(y).any())


def _legend(ax: plt.Axes) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, fontsize=8, loc="best")


class _Exporter:
    def __init__(self, path: Path, report: dict, datasets: list[dict]) -> None:
        self.path, self.report, self.datasets = path, report, datasets
        self.entries: list[dict] = []

    def save(self, fig: plt.Figure, name: str, kind: str, sources: list[dict]) -> None:
        run_ids = sorted({str(data["run_id"]) for data in sources})
        footer = "\n".join(
            line
            for run_id in run_ids
            for line in textwrap.wrap(f"Run: {run_id}", width=145)
        )
        # Keep all evidence identifiers and legends outside the plotted axes.
        # Allocate inches so a tall list cannot overrun a short figure's labels.
        width, content_height = fig.get_size_inches()
        legend_groups = []
        signatures = set()
        for ax in fig.axes:
            legend = ax.get_legend()
            if legend is None:
                continue
            handles, labels = ax.get_legend_handles_labels()
            signature = tuple(
                (
                    label,
                    str(getattr(handle, "get_color", lambda: "")()),
                    str(getattr(handle, "get_linestyle", lambda: "")()),
                )
                for handle, label in zip(handles, labels, strict=True)
            )
            legend.remove()
            if signature in signatures:
                continue
            signatures.add(signature)
            title = ax.get_title() or ax.get_ylabel()
            legend_groups.append((handles, labels, title))
        columns = 3 if width >= 10 else 2
        footer_height = 0.18 + len(footer.splitlines()) * 8.5 / 72
        legend_heights = [
            ((len(labels) + columns - 1) // columns) * 12 / 72
            + (0.22 if len(legend_groups) > 1 else 0)
            + 0.12
            for _, labels, _ in legend_groups
        ]
        extra_height = footer_height + sum(legend_heights) + 0.3
        total_height = content_height + extra_height
        fig.set_size_inches(width, total_height)
        fig.text(
            0.012,
            0.08 / total_height,
            footer,
            va="bottom",
            fontsize=6.5,
            family="monospace",
        )
        anchor = extra_height - 0.12
        for (handles, labels, title), height in zip(
            legend_groups, legend_heights, strict=True
        ):
            legend = fig.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(0.055, anchor / total_height),
                ncol=columns,
                fontsize=8,
                frameon=False,
                borderaxespad=0,
                handlelength=2.5,
                columnspacing=1.6,
                title=title if len(legend_groups) > 1 else None,
            )
            legend.get_title().set_fontsize(8)
            anchor -= height
        fig.tight_layout(
            rect=(0, extra_height / total_height, 1, 1 - 0.45 / total_height)
        )
        artifacts = []
        for extension in ("png", "pdf"):
            target = self.path / f"{name}.{extension}"
            fig.savefig(target, dpi=160, bbox_inches="tight")
            artifacts.append(
                {
                    "path": str(target),
                    "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                }
            )
        plt.close(fig)
        self.entries.append(
            {
                "figure_id": name,
                "kind": kind,
                "artifacts": artifacts,
                "run_ids": run_ids,
                "source_records": [
                    {
                        "run_id": data["run_id"],
                        "resolved_config_sha256": data.get("resolved_config_sha256"),
                        "provenance": data.get("provenance", {}),
                    }
                    for data in sources
                ],
                "report_provenance": self.report.get("provenance", {}),
            }
        )


def _status(export: _Exporter) -> None:
    rows = []
    for data in export.datasets:
        state = data.get("state", "unknown")
        reason = data.get("reason") or ""
        rows.append(
            [data["case_id"], _label(data), state, "\n".join(textwrap.wrap(reason, 52))]
        )
    fig, ax = plt.subplots(figsize=(12, max(3, 0.8 * len(rows) + 1.5)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Case", "Method", "Saved state", "Reason / diagnostic"],
        colWidths=[0.06, 0.3, 0.16, 0.48],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 3)
    fig.suptitle("Comparison coverage and saved run states")
    export.save(fig, "comparison_status", "comparison_status", export.datasets)


def _profiles(export: _Exporter, case: str, sources: list[dict], times: list) -> None:
    fig, axes = plt.subplots(2, len(times), figsize=(12, 7), squeeze=False)
    for data in sources:
        x, t, h, q = _arrays(data)
        for column, time in enumerate(times):
            index = np.flatnonzero(t == time)
            if not len(index):
                continue
            for row, field in enumerate((h, q)):
                axes[row, column].plot(
                    x,
                    field[index[0]],
                    label=_label(data),
                    linewidth=1.3,
                    **_style(data),
                )
    for column, time in enumerate(times):
        for row, field in enumerate(("h_m", "q_m2_s")):
            ax = axes[row, column]
            ax.set(title=f"t = {time:g} s", xlabel="Position (m)", ylabel=FIELDS[field])
            ax.grid(alpha=0.2)
            if not ax.lines:
                _note(ax, "No saved output at this exact time")
    _legend(axes[0, 0])
    fig.suptitle(f"{case}: saved common-grid profiles")
    export.save(fig, f"{case.lower()}_profiles", "profiles", sources)


def _maps(export: _Exporter, case: str, sources: list[dict]) -> None:
    references = [data for data in sources if data["method"] == "analytical_reference"]
    depth_min = min(float(np.min(data["h_m"])) for data in sources)
    depth_max = max(float(np.max(data["h_m"])) for data in sources)
    reference = references[0] if references else None
    error_scale = 1.0
    if reference is not None:
        rx, rt, rh, _ = _arrays(reference)
        error_scale = (
            max(float(np.max(np.abs(np.asarray(data["h_m"]) - rh))) for data in sources)
            or 1.0
        )
    for index, data in enumerate(sources):
        x, t, h, _ = _arrays(data)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
        mesh = axes[0].pcolormesh(
            x, t, h, shading="nearest", cmap="viridis", vmin=depth_min, vmax=depth_max
        )
        fig.colorbar(mesh, ax=axes[0], label="Depth (m)")
        if reference is not None:
            if not (np.array_equal(x, rx) and np.array_equal(t, rt)):
                raise PlotError("depth/error maps require exact shared axes")
            error = h - rh
            mesh = axes[1].pcolormesh(
                x,
                t,
                error,
                shading="nearest",
                cmap="RdBu_r",
                vmin=-error_scale,
                vmax=error_scale,
            )
            fig.colorbar(
                mesh, ax=axes[1], label="Depth error, candidate − reference (m)"
            )
        else:
            _note(axes[1], "Analytical reference unavailable")
        for ax in axes:
            ax.set(
                xlabel="Position (m)",
                ylabel="Time (s)",
                xlim=(x[0], x[-1]),
                ylim=(t[0], t[-1]),
            )
        axes[0].set_title("Saved depth")
        axes[1].set_title("Signed depth error")
        fig.suptitle(f"{case}: {_label(data)}")
        linked = [data] + (
            [reference] if reference is not None and reference is not data else []
        )
        export.save(
            fig, f"{case.lower()}_map_{index:02d}", "depth_and_error_map", linked
        )


def _errors(
    export: _Exporter, case: str, sources: list[dict], rows: list[dict]
) -> None:
    regions = ("whole", "wave_band", "outside_wave_band")
    width = export.report.get("evaluation", {}).get("wave_band_half_width_m", 0.1)
    region_labels = {
        "whole": "whole domain",
        "wave_band": f"within ±{width:g} m shock/front",
        "outside_wave_band": f"outside ±{width:g} m shock/front",
    }
    times = export.report.get("evaluation", {}).get("times_s", [])
    if not times:
        times = sorted({item["time_s"] for item in rows})
    for metric in ("rmse", "relative_l2"):
        fig, axes = plt.subplots(3, 3, figsize=(12, 10), squeeze=False)
        for row, field in enumerate(FIELDS):
            for column, region in enumerate(regions):
                ax = axes[row, column]
                found = False
                reasons = set()
                for data in sources:
                    selected = sorted(
                        [
                            item
                            for item in rows
                            if item["run_id"] == data["run_id"]
                            and item["field"] == field
                            and item["region"] == region
                        ],
                        key=lambda item: item["time_s"],
                    )
                    if selected:
                        reasons.update(
                            item[metric].get("reason", "undefined")
                            for item in selected
                            if isinstance(item[metric], dict)
                            and item[metric].get("value") is None
                        )
                        found |= _series(
                            ax,
                            [item["time_s"] for item in selected],
                            [_value(item[metric]) for item in selected],
                            _label(data),
                            **_style(data),
                        )
                unit = (
                    "Relative L2 (dimensionless)"
                    if metric == "relative_l2"
                    else FIELDS[field]
                )
                ax.set(
                    title=f"{field}: {region_labels[region]}",
                    xlabel="Time (s)",
                    ylabel=unit,
                )
                if len(times) > 1:
                    ax.set_xlim(times[0], times[-1])
                ax.grid(alpha=0.2)
                if not found:
                    message = "; ".join(sorted(reason for reason in reasons if reason))
                    _note(
                        ax,
                        f"Undefined: {message}"
                        if message
                        else "No defined saved metric",
                    )
                elif reasons:
                    reason_text = (
                        "Gaps: no reference-wet points"
                        if reasons == {"no_selected_points"} and field == "u_m_s"
                        else "Gaps: undefined saved metric"
                    )
                    ax.text(
                        0.02,
                        0.03,
                        reason_text,
                        transform=ax.transAxes,
                        fontsize=7,
                        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
                    )
        _legend(axes[0, 0])
        fig.suptitle(f"{case}: {metric}; velocity uses reference h > 0.001 m")
        export.save(fig, f"{case.lower()}_{metric}", "per_time_errors", sources)


def _mass(export: _Exporter, case: str, sources: list[dict]) -> None:
    fig, grid = plt.subplots(2, 2, figsize=(12, 8))
    axes = grid.ravel()
    found = [False, False, False, False]
    for data in sources:
        mass = data.get("mass") or {}
        label = _label(data)
        times = mass.get("t_s", [])
        if "normalized_signed_balance" in mass:
            found[0] |= _series(
                axes[0], times, mass["normalized_signed_balance"], label, **_style(data)
            )
        if "boundary_net_outflow_m2" in mass:
            found[1] |= _series(
                axes[1], times, mass["boundary_net_outflow_m2"], label, **_style(data)
            )
        boundary_times = mass.get("boundary_t_s", times)
        numerical_flux = "numerical" in mass.get("boundary_values_are", "")
        for side, style in (("left", "--"), ("right", "-")):
            key = f"{side}_boundary_q_m2_s"
            if key in mass:
                panel = 3 if numerical_flux else 2
                found[panel] |= _series(
                    axes[panel],
                    boundary_times,
                    mass[key],
                    f"{label}, {side}",
                    linestyle=style,
                    color=_style(data)["color"],
                    drawstyle="steps-pre" if numerical_flux else "default",
                )
    labels = [
        "Signed balance / prescribed initial volume",
        "Integrated net boundary outflow (m²)",
        "Physical endpoint discharge (m²/s)",
        "Numerical boundary mass flux (m²/s)",
    ]
    for index, ax in enumerate(axes):
        ax.set(xlabel="Time (s)", ylabel=labels[index])
        ax.grid(alpha=0.2)
        if not found[index]:
            _note(ax, "Independent mass / boundary evidence unavailable")
        _legend(ax)
    fig.suptitle(f"{case}: conservation and boundary mismatch against zero")
    export.save(fig, f"{case.lower()}_mass_boundary", "mass_boundary", sources)


def _coverage(
    export: _Exporter, case: str, sources: list[dict], rows: list[dict]
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    width = export.report.get("evaluation", {}).get("wave_band_half_width_m", 0.1)
    region_labels = {
        "whole": "whole domain",
        "wave_band": f"within ±{width:g} m shock/front",
        "outside_wave_band": f"outside ±{width:g} m shock/front",
    }
    for ax, region in zip(
        axes, ("whole", "wave_band", "outside_wave_band"), strict=True
    ):
        for data in sources:
            selected = sorted(
                [
                    item
                    for item in rows
                    if item["run_id"] == data["run_id"]
                    and item["field"] == "u_m_s"
                    and item["region"] == region
                ],
                key=lambda item: item["time_s"],
            )
            if selected:
                _series(
                    ax,
                    [item["time_s"] for item in selected],
                    [item["coverage_fraction"] for item in selected],
                    _label(data),
                    **_style(data),
                )
        ax.set(
            title=region_labels[region],
            xlabel="Time (s)",
            ylabel="Selected / total grid points",
            ylim=(-0.02, 1.02),
        )
        ax.grid(alpha=0.2)
        if not ax.lines:
            _note(ax, "No saved velocity-mask coverage")
    _legend(axes[0])
    fig.suptitle(f"{case}: velocity coverage, fixed reference depth > 0.001 m")
    export.save(
        fig, f"{case.lower()}_velocity_coverage", "velocity_mask_coverage", sources
    )


def _history(export: _Exporter, case: str, sources: list[dict]) -> None:
    histories = [data for data in sources if data.get("history")]
    if not histories:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for data in histories:
        history = data["history"]
        steps = [float(item["step"]) for item in history]
        for component in ("pde", "ic", "bc"):
            candidates = (f"loss_{component}", f"{component}_loss", component)
            key = next((name for name in candidates if name in history[0]), None)
            if key:
                _series(
                    axes[0],
                    steps,
                    [float(item[key]) for item in history],
                    f"{_label(data)}, {component}",
                )
        for component in ("ic", "bc"):
            key = f"lambda_{component}"
            if key in history[0]:
                _series(
                    axes[1],
                    steps,
                    [float(item[key]) for item in history],
                    f"{_label(data)}, {component}",
                )
    for ax, title in zip(
        axes, ("Saved unweighted components", "Saved loss weights"), strict=True
    ):
        ax.set(title=title, xlabel="Optimizer step")
        ax.grid(alpha=0.2)
        if not ax.lines:
            _note(ax, "Requested columns unavailable in saved history")
        _legend(ax)
    axes[0].set_ylabel("Dimensionless loss")
    axes[1].set_ylabel("Dimensionless weight")
    fig.suptitle(f"{case}: training-component histories")
    export.save(fig, f"{case.lower()}_training", "training_components", histories)


def _grid(export: _Exporter, case: str, sources: list[dict], rows: list[dict]) -> None:
    numerical = [data for data in sources if data["method"] == "lax_wendroff"]
    if not numerical:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, field in zip(axes, ("h_m", "q_m2_s"), strict=True):
        for data in numerical:
            selected = [
                item
                for item in rows
                if item["run_id"] == data["run_id"]
                and item["field"] == field
                and item["region"] == "whole"
            ]
            if selected:
                final = max(selected, key=lambda item: item["time_s"])
                ax.scatter(
                    data["cells"],
                    _value(final["rmse"]),
                    label=f"{_label(data)}, t={final['time_s']:g} s",
                )
        ax.set(xlabel="Native numerical cells", ylabel=f"RMSE, {FIELDS[field]}")
        ax.grid(alpha=0.2)
        _legend(ax)
        if not ax.collections:
            _note(ax, "No valid saved grid-comparison metric")
    fig.suptitle(f"{case}: grid comparison; no convergence order inferred")
    export.save(fig, f"{case.lower()}_grid_comparison", "grid_comparison", numerical)


def _ablation(
    export: _Exporter, case: str, sources: list[dict], rows: list[dict]
) -> None:
    neural = [data for data in sources if data["method"] == "pinn"]
    if not neural:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))
    for ax, field in zip(axes, ("h_m", "q_m2_s"), strict=True):
        by_seed: dict[int, dict[str, float]] = {}
        for data in neural:
            selected = [
                item
                for item in rows
                if item["run_id"] == data["run_id"]
                and item["field"] == field
                and item["region"] == "whole"
            ]
            if selected and data.get("seed") is not None:
                final = max(selected, key=lambda item: item["time_s"])
                by_seed.setdefault(data["seed"], {})[data["variant"]] = _value(
                    final["rmse"]
                )
        for seed, variants in sorted(by_seed.items()):
            values = [
                variants.get(name, float("nan")) for name in ("baseline", "adaptive")
            ]
            ax.plot([0, 1], values, "o-", label=f"Seed {seed}")
        ax.set(
            xticks=[0, 1],
            xticklabels=["Fixed", "Adaptive"],
            ylabel=f"Final-time RMSE, {FIELDS[field]}",
        )
        ax.grid(alpha=0.2)
        _legend(ax)
        if not by_seed:
            _note(ax, "No saved paired-seed metric")
        paired = sum(
            all(
                np.isfinite(values.get(name, np.nan))
                for name in ("baseline", "adaptive")
            )
            for values in by_seed.values()
        )
        ax.set_title(f"{field}: {paired} complete seed pairs")
    fig.suptitle(
        f"{case}: fixed/adaptive weighting; available seeds shown individually"
    )
    export.save(fig, f"{case.lower()}_ablation", "paired_ablation", neural)


def plot_report(
    report: dict,
    datasets: list[dict],
    output_dir: str | Path,
    *,
    project_root: str | Path | None = None,
) -> dict:
    """Export PNG/PDF figures plus a manifest without replacing prior artifacts.

    ``datasets`` must already contain the saved common-grid h/q arrays, metric
    mass curves, and optional CSV history rows. Failed or missing datasets need
    only identity, state, and reason. Root CLI integration supplies these records.
    """
    if not datasets:
        raise PlotError("at least one saved run or explicit failure record is required")
    root = Path(project_root or Path.cwd()).resolve()
    target = Path(output_dir)
    if not target.is_absolute():
        target = root / target
    # Never replace a previous attempt, including an empty claimed directory.
    target.mkdir(parents=True, exist_ok=False)
    run_ids = [data["run_id"] for data in datasets]
    if len(run_ids) != len(set(run_ids)):
        raise PlotError("duplicate run IDs in plotting inputs")
    good = [data for data in datasets if data.get("state") == "completed"]
    for data in good:
        _arrays(data)
    export = _Exporter(target, report, datasets)
    with plt.rc_context(
        {"font.size": 9, "axes.spines.top": False, "axes.spines.right": False}
    ):
        _status(export)
        for case in sorted({data["case_id"] for data in datasets}):
            sources = [data for data in good if data["case_id"] == case]
            rows = [row for row in report.get("rows", []) if row["case_id"] == case]
            if not sources:
                continue
            times = report.get("evaluation", {}).get(
                "profile_times_s", [0.1, 0.25, 0.5]
            )
            _profiles(export, case, sources, times)
            _maps(export, case, sources)
            _errors(export, case, sources, rows)
            _coverage(export, case, sources, rows)
            _mass(export, case, sources)
            _history(export, case, sources)
            _grid(export, case, sources, rows)
            _ablation(export, case, sources, rows)
    manifest = {
        "schema_version": 1,
        "purpose": report.get("purpose", "saved_evidence_figures"),
        "provenance": report.get("provenance", {}),
        "plot_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "figures": export.entries,
        "run_states": [
            {
                key: data.get(key)
                for key in ("run_id", "case_id", "method", "state", "reason")
            }
            for data in datasets
        ],
        "missing_profile_outputs": [
            {"run_id": data["run_id"], "time_s": time}
            for data in good
            for time in report.get("evaluation", {}).get(
                "profile_times_s", [0.1, 0.25, 0.5]
            )
            if not np.any(np.asarray(data["t_s"]) == time)
        ],
        "interpretation_limits": [
            "Figures use saved evidence; missing outputs are not interpolated.",
            "Failed runs appear in comparison_status without output curves.",
            "Velocity metric curves use the fixed reference-wet mask.",
            "Mass curves require separate quadrature or native solver balance.",
        ],
    }
    write_json_atomic(target / "figure_manifest.json", manifest)
    return manifest
