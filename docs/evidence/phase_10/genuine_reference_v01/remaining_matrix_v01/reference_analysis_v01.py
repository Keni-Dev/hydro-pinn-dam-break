"""Saved-data R6 analysis for the four frozen genuine-2D reference cells.

This script reads and validates completed records. It imports no solver helper,
executes no PDE update or area projection, and never writes an input record.
Its SHA256 must be supplied from the frozen analysis launch receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CONTRACT = "docs/coordination/phase_10_2_reference_analysis_contract_v01.json"
CONTRACT_SHA256 = "0e4d15aea1696290cb35431309ac460d756aeaa3178eb09b487c3112086f7d77"
PROTOCOL = (
    "results/extensions/two_dimensional/reference_v01/"
    "candidate_protocol_v01/protocol.json"
)
PROTOCOL_DIGEST = "adaa44c9f45d6a58850a693e0e88715097b0a5cf36185344863b2d6e5a87822c"
DEFAULT_OUTPUT = "results/extensions/two_dimensional/reference_v01/r6_analysis_v01"
ROOT_FROM_SCRIPT = Path(__file__).resolve().parents[5]
RUNS = (
    "ref_rect80_c045_v01",
    "ref_rect160_c045_v01",
    "ref_rect320_c045_v01",
    "ref_rect320_c0225_v01",
)
SELECTED = RUNS[-1]
PAIRS = (
    ("coupled160_minus80", RUNS[1], RUNS[0], "160 − 80, C=.45; coupled mesh/time"),
    ("coupled320_minus160", RUNS[2], RUNS[1], "320 − 160, C=.45; coupled mesh/time"),
    (
        "time320_half_minus_full",
        RUNS[3],
        RUNS[2],
        "320, C=.225 − C=.45; fixed-grid time",
    ),
)
FIELDS = ("h_m", "qx_m2_s", "qy_m2_s")
AXES = ("x_edges_m", "y_edges_m", "x_m", "y_m", "cell_area_m2", "t_s")
CRITERIA = {
    "all_requested_snapshots",
    "budgets",
    "finite_positive_depth",
    "outer_strips",
    "parity",
    "sum_cfl",
    "travel_Dmax",
}
UNITS = {
    "h_m": {"field": "m", "integral_L1": "m^3", "integral_L2": "m^2"},
    "qx_m2_s": {"field": "m^2/s", "integral_L1": "m^4/s", "integral_L2": "m^3/s"},
    "qy_m2_s": {"field": "m^2/s", "integral_L1": "m^4/s", "integral_L2": "m^3/s"},
}


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def content_digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: dict, *, replace_status: bool = False) -> None:
    if replace_status and path.name != "status.json":
        raise ValueError("only the analysis status marker may be replaced")
    with path.open("w" if replace_status else "x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def save_npz(path: Path, **arrays) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def contained(root: Path, relative: str | Path) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"path lies outside the analysis root: {relative}")
    return path


def bind_file(
    root: Path, path: Path, expected: str | None, bindings: dict[str, str]
) -> str:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"missing/unsafe regular input: {path}")
    observed = file_sha256(path)
    if expected is not None and observed != expected:
        raise ValueError(f"input SHA256 mismatch: {path}")
    name = str(path.relative_to(root))
    if name in bindings and bindings[name] != observed:
        raise ValueError(f"input changed during analysis readback: {name}")
    bindings[name] = observed
    return observed


def validate_frozen_inputs(root: Path, script_hash: str) -> tuple[dict, dict, dict]:
    bindings = {}
    script = Path(__file__).resolve()
    bind_file(root, script, script_hash, bindings)
    contract_path, protocol_path = root / CONTRACT, root / PROTOCOL
    bind_file(root, contract_path, CONTRACT_SHA256, bindings)
    contract = read_json(contract_path)
    bind_file(root, protocol_path, None, bindings)
    protocol = read_json(protocol_path)
    digest = content_digest(
        {k: v for k, v in protocol.items() if k != "protocol_sha256"}
    )
    if digest != PROTOCOL_DIGEST or protocol.get("protocol_sha256") != digest:
        raise ValueError("frozen reference protocol content digest mismatch")
    if contract.get("protocol_content_digest") != digest:
        raise ValueError("analysis contract/protocol binding mismatch")
    for name, expected in {
        **protocol["source_sha256"],
        **protocol["input_sha256"],
        protocol["config_path"]: protocol["config_sha256"],
    }.items():
        bind_file(root, contained(root, name), expected, bindings)
    for name, expected in protocol["artifact_sha256"].items():
        if Path(name).name != name:
            raise ValueError("invalid frozen axes/operator filename")
        bind_file(root, protocol_path.parent / name, expected, bindings)
    with np.load(protocol_path.parent / "common_axes.npz", allow_pickle=False) as saved:
        axes = {key: saved[key].copy() for key in AXES}
    expected_times = np.r_[0.0, 0.005 * np.arange(1, 51, dtype=np.float64)]
    if (
        not np.array_equal(axes["t_s"], expected_times)
        or axes["cell_area_m2"].shape != (80, 80)
        or any(value.dtype != np.float64 for value in axes.values())
        or any(not np.isfinite(value).all() for value in axes.values())
        or np.any(axes["cell_area_m2"] <= 0)
    ):
        raise ValueError("invalid frozen common-grid float64 axes/times/areas")
    return protocol, axes, bindings


def validate_cell(
    root: Path, run_id: str, protocol: dict, axes: dict, bindings: dict
) -> tuple[np.ndarray, dict]:
    directory = root / "results/extensions/two_dimensional/reference_v01" / run_id
    for name in ("hashes.json", "status.json", "metadata.json"):
        bind_file(root, directory / name, None, bindings)
    hashes = read_json(directory / "hashes.json")
    if set(hashes.get("mutable_lifecycle_files_excluded", [])) != {
        "status.json",
        "metadata.json",
    }:
        raise ValueError(f"unexpected lifecycle hash policy: {run_id}")
    required = {
        "common_reference.npz",
        "outputs.npz",
        "diagnostics.npz",
        "initial_state.npz",
        "last_valid_state.npz",
        "summary.json",
        "termination.json",
        "execution_wall.json",
        "projection_readback.json",
        "protocol.json",
        "resolved_config.json",
        "root_release.json",
    }
    if not required.issubset(hashes["files"]):
        raise ValueError(f"missing mandatory hashed cell records: {run_id}")
    for name, expected in hashes["files"].items():
        if Path(name).name != name:
            raise ValueError("invalid cell artifact filename")
        bind_file(root, directory / name, expected, bindings)
    expected_source = {
        "src/hydro_pinn/extensions/reference_2d.py": protocol["source_sha256"][
            "src/hydro_pinn/extensions/reference_2d.py"
        ],
    }
    if hashes["source_sha256"] != expected_source:
        raise ValueError(f"cell/frozen source binding mismatch: {run_id}")
    status, metadata = (
        read_json(directory / "status.json"),
        read_json(directory / "metadata.json"),
    )
    summary, termination = (
        read_json(directory / "summary.json"),
        read_json(directory / "termination.json"),
    )
    resolved, release = (
        read_json(directory / "resolved_config.json"),
        read_json(directory / "root_release.json"),
    )
    spec = next(cell for cell in protocol["case"]["matrix"] if cell["id"] == run_id)
    if (
        status.get("state") != "completed"
        or status.get("run_id") != run_id
        or summary.get("termination_kind") != "completed"
        or summary.get("criteria_enforced") is not True
        or set(summary.get("criteria", {})) != CRITERIA
        or not all(v is True for v in summary["criteria"].values())
        or summary.get("protocol_sha256") != PROTOCOL_DIGEST
        or summary.get("completed_snapshot_count") != 51
        or summary.get("requested_snapshot_count") != 51
        or summary.get("last_valid_t_s") != 0.25
        or summary.get("dimension") != 2
        or summary.get("field_location") != "cell_average"
        or summary.get("case_id") != protocol["case"]["case_id"]
        or summary.get("method") != "local_unsplit_full_vector_wet_HLLE"
        or termination.get("termination_kind") != "completed"
        or termination.get("completed_snapshot_count") != 51
        or termination.get("last_valid_t_s") != 0.25
        or termination.get("last_valid_step") != summary.get("accepted_steps")
        or termination.get("stage") is not None
        or metadata.get("execution", {}).get("iterations")
        != summary.get("accepted_steps")
        or metadata.get("dimension") != 2
        or metadata.get("case") != protocol["case"]["case_id"]
        or metadata.get("array_contract", {}).get("orientation") != "time,y,x"
        or metadata.get("array_contract", {}).get("location") != "cell_average"
    ):
        raise ValueError(
            f"cell is incomplete or fails saved validity criteria: {run_id}"
        )
    if (
        resolved.get("case") != protocol["case"]
        or resolved.get("cell") != spec
        or resolved.get("protocol_sha256") != PROTOCOL_DIGEST
        or resolved.get("request")
        != {"device": "cpu", "dtype": "float64", "run_id": run_id}
        or metadata.get("resolved_config_sha256")
        != file_sha256(directory / "resolved_config.json")
        or read_json(directory / "protocol.json") != protocol
        or release.get("state") != "released"
        or release.get("run_id") != run_id
        or release.get("protocol_sha256") != PROTOCOL_DIGEST
    ):
        raise ValueError(f"cell config/release/protocol mismatch: {run_id}")
    original_release = contained(root, resolved["release_path"])
    bind_file(root, original_release, resolved["release_sha256"], bindings)
    if read_json(original_release) != release:
        raise ValueError(f"saved/original root release content mismatch: {run_id}")
    if read_json(directory / "projection_readback.json").get("passed") is not True:
        raise ValueError(f"saved conservative projection readback failed: {run_id}")
    with np.load(directory / "common_reference.npz", allow_pickle=False) as saved:
        for key in AXES:
            if saved[key].dtype != np.float64 or not np.array_equal(
                saved[key], axes[key]
            ):
                raise ValueError(
                    f"exact common axis/time/area mismatch: {run_id}/{key}"
                )
        values = []
        for key in FIELDS:
            a = saved[key]
            if (
                a.dtype != np.float64
                or a.shape != (51, 80, 80)
                or not np.isfinite(a).all()
            ):
                raise ValueError(f"invalid float64 common field: {run_id}/{key}")
            values.append(a.copy())
    fields = np.stack(values, axis=-1)
    if np.any(fields[..., 0] <= 0):
        raise ValueError(f"nonpositive common-cell depth: {run_id}")
    return fields, {
        "run_id": run_id,
        "accepted_steps": summary["accepted_steps"],
        "criteria": summary["criteria"],
        "native_shape_y_x": [spec["ny"], spec["nx"]],
        "common_shape_time_y_x": [51, 80, 80],
        "dtype": "float64",
        "relative_denominator_eligible": True,
        "saved_cost_s": summary["cost_s"],
        "saved_final_observed_wall_s": summary["final_observed_wall_s"],
        "terminal_metadata_wall_s": metadata["execution"]["elapsed_timings_s"],
        "saved_memory_observation": summary["memory"],
        "initial_volume_discrepancy_m3": summary["initial_volume_discrepancy_m3"],
        "travel_Dx_Dy_Dmax_m": summary["travel_Dx_Dy_Dmax_m"],
    }


def norm_rows(
    error_l1: np.ndarray,
    error_squared: np.ndarray,
    reference_squared: np.ndarray,
    area: float,
) -> dict:
    answer = {}
    for index, key in enumerate(FIELDS):
        denominator = float(reference_squared[index])
        answer[key] = {
            "area_mean_MAE": float(error_l1[index] / area),
            "area_mean_RMSE": float(np.sqrt(error_squared[index] / area)),
            "integral_L1": float(error_l1[index]),
            "integral_L2": float(np.sqrt(error_squared[index])),
            "reference_integral_L2": float(np.sqrt(denominator)),
            "relative_L2_percent": float(
                100 * np.sqrt(error_squared[index] / denominator)
            )
            if denominator > 0
            else None,
            "relative_reason": None
            if denominator > 0
            else "zero_higher_member_reference_norm",
            "units": UNITS[key],
        }
    return answer


def pair_norms(
    difference: np.ndarray,
    higher: np.ndarray,
    areas: np.ndarray,
    times: np.ndarray,
    high_id: str,
    low_id: str,
) -> dict:
    weight = areas[None, ..., None]
    l1 = np.sum(weight * np.abs(difference), axis=(1, 2))
    squared = np.sum(weight * difference**2, axis=(1, 2))
    reference_squared = np.sum(weight * higher**2, axis=(1, 2))
    area = float(areas.sum())
    return {
        "difference_direction": f"{high_id} minus {low_id}",
        "higher_member_relative_denominator_run": high_id,
        "positive_snapshots": 50,
        "positive_time_weights": (
            "equal 1/50; pooled squared quantities before square root"
        ),
        "initial_separate": {
            "t_s": float(times[0]),
            "fields": norm_rows(l1[0], squared[0], reference_squared[0], area),
        },
        "per_positive_time": [
            {
                "t_s": float(times[i]),
                "fields": norm_rows(l1[i], squared[i], reference_squared[i], area),
            }
            for i in range(1, 51)
        ],
        "pooled_positive": norm_rows(
            l1[1:].mean(axis=0),
            squared[1:].mean(axis=0),
            reference_squared[1:].mean(axis=0),
            area,
        ),
        "pooled_definition": {
            "MAE_and_L1": "arithmetic mean of per-time L1 quantities",
            "RMSE_and_L2": (
                "square root after arithmetic mean of per-time squared integrals"
            ),
            "relative_L2_percent": (
                "100*sqrt(sum_t,A(A*d²)/sum_t,A(A*higher²)); no epsilon denominator"
            ),
        },
    }


def gradient_region(selected: np.ndarray, axes: dict) -> tuple[dict, dict]:
    dx = float(axes["x_m"][1] - axes["x_m"][0])
    dy = float(axes["y_m"][1] - axes["y_m"][0])
    gy, gx = np.gradient(selected[1:, ..., 0], dy, dx, axis=(1, 2), edge_order=1)
    magnitude = np.sqrt(gx**2 + gy**2)
    maxima = magnitude.max(axis=(1, 2))
    thresholds = 0.1 * maxima
    raw = (magnitude >= thresholds[:, None, None]) & (maxima[:, None, None] > 0)
    padded = np.pad(raw, ((0, 0), (1, 1), (1, 1)), constant_values=False)
    dilated = np.zeros_like(raw)
    for j in range(3):
        for i in range(3):
            dilated |= padded[:, j : j + raw.shape[1], i : i + raw.shape[2]]
    areas = axes["cell_area_m2"]
    raw_count, dilated_count = raw.sum(axis=(1, 2)), dilated.sum(axis=(1, 2))
    raw_coverage = (raw * areas).sum(axis=(1, 2)) / areas.sum()
    dilated_coverage = (dilated * areas).sum(axis=(1, 2)) / areas.sum()
    reasons = [
        "flat_reference_gradient" if maximum == 0 else None for maximum in maxima
    ]
    arrays = {
        **{key: axes[key] for key in AXES if key != "t_s"},
        "t_s": axes["t_s"][1:],
        "gradient_x_m_m": gx,
        "gradient_y_m_m": gy,
        "gradient_magnitude_m_m": magnitude,
        "maximum_magnitude_m_m": maxima,
        "threshold_magnitude_m_m": thresholds,
        "raw_threshold_mask": raw,
        "dilated_mask": dilated,
        "raw_count": raw_count,
        "dilated_count": dilated_count,
        "raw_area_coverage": raw_coverage,
        "dilated_area_coverage": dilated_coverage,
    }
    metadata = {
        "source_run": SELECTED,
        "source_field": "common-cell mean h_m",
        "gradient": "numpy.gradient(hbar,dy,dx,axis=(y,x),edge_order=1)",
        "gradient_units": "m/m",
        "threshold_fraction_of_per_time_maximum": 0.1,
        "dilation": "one cell, eight-neighbour/Chebyshev, false outside domain",
        "per_time": [
            {
                "t_s": float(axes["t_s"][i + 1]),
                "maximum_magnitude_m_m": float(maxima[i]),
                "threshold_magnitude_m_m": float(thresholds[i]),
                "raw_count": int(raw_count[i]),
                "dilated_count": int(dilated_count[i]),
                "raw_area_coverage": float(raw_coverage[i]),
                "dilated_area_coverage": float(dilated_coverage[i]),
                "reason": reasons[i],
            }
            for i in range(50)
        ],
        "qualification": (
            "Numerical reference-gradient region may include shocks, "
            "rarefactions and corners; not an exact2D front/shock location "
            "or error bound."
        ),
    }
    return arrays, metadata


def plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def save_figure(figure, output: Path, stem: str, description: dict) -> dict:
    artifacts = []
    for suffix in ("png", "pdf"):
        path = output / f"{stem}.{suffix}"
        with path.open("xb") as handle:
            figure.savefig(handle, format=suffix, dpi=180)
        artifacts.append({"path": path.name, "sha256": file_sha256(path)})
    plt.close(figure)
    return {"name": stem, "artifacts": artifacts, **description}


def reference_map(
    selected: np.ndarray, axes: dict, gradient: dict, output: Path
) -> dict:
    q_limit = float(np.max(np.abs(selected[1:, ..., 1:])))
    if q_limit <= 0:
        raise ValueError("selected genuine2D flow has zero discharge display range")
    gmax = float(np.max(gradient["gradient_magnitude_m_m"][-1]))
    if gmax <= 0:
        raise ValueError(
            "selected T=.25 gradient is flat; fixed contract map scale undefined"
        )
    fig, subplots = plt.subplots(2, 2, figsize=(10.7, 8.7))
    panels = (
        (selected[-1, ..., 0], "Depth h [m]", "viridis", 0.1, 1.0),
        (selected[-1, ..., 1], "Discharge qx [m²/s]", "RdBu_r", -q_limit, q_limit),
        (selected[-1, ..., 2], "Discharge qy [m²/s]", "RdBu_r", -q_limit, q_limit),
        (
            gradient["gradient_magnitude_m_m"][-1],
            "Depth-gradient magnitude [m/m]",
            "magma",
            0.0,
            gmax,
        ),
    )
    for axis, (field, title, color, low, high) in zip(
        subplots.flat, panels, strict=True
    ):
        image = axis.pcolormesh(
            axes["x_edges_m"],
            axes["y_edges_m"],
            field,
            cmap=color,
            vmin=low,
            vmax=high,
            shading="flat",
        )
        axis.set_title(title)
        axis.set_xlabel("x [m]")
        axis.set_ylabel("y [m]")
        axis.set_aspect("equal")
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Selected numerical reference at t = 0.25 s", fontsize=15, y=0.98)
    fig.subplots_adjust(
        left=0.07, right=0.94, top=0.91, bottom=0.11, wspace=0.32, hspace=0.3
    )
    fig.text(
        0.02,
        0.035,
        "320², sum-CFL .225, projected to 80² area means. Shared symmetric "
        "discharge scale uses both fields over all 50 positive snapshots.",
        fontsize=8,
    )
    fig.text(
        0.98,
        0.012,
        f"Reference-only sensitivity study | protocol {PROTOCOL_DIGEST[:12]}",
        ha="right",
        fontsize=7,
        color="#555b61",
    )
    return save_figure(
        fig,
        output,
        "01_selected_reference_maps",
        {
            "source_run": SELECTED,
            "t_s": 0.25,
            "h_limits_m": [0.1, 1.0],
            "shared_qx_qy_limits_m2_s": [-q_limit, q_limit],
            "shared_q_limit_source": (
                "max abs(qx,qy), selected reference, all50positive common snapshots"
            ),
            "gradient_limits_m_m": [0.0, gmax],
            "field_location": "80x80 common-cell area means",
        },
    )


def refinement_plot(norms: dict, times: np.ndarray, output: Path) -> dict:
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.0))
    colors = ("#24688c", "#c36827", "#3b7c60")
    styles = ("-", "--", ":")
    for field, axis, label in zip(
        FIELDS,
        axes,
        ("Depth h [m]", "Discharge qx [m²/s]", "Discharge qy [m²/s]"),
        strict=True,
    ):
        for (name, _, _, pair_label), color, style in zip(
            PAIRS, colors, styles, strict=True
        ):
            values = [
                row["fields"][field]["area_mean_RMSE"]
                for row in norms[name]["per_positive_time"]
            ]
            axis.plot(
                times[1:],
                values,
                color=color,
                linestyle=style,
                linewidth=1.7,
                label=pair_label,
            )
        axis.set_title(label)
        axis.set_xlabel("t [s]")
        axis.set_ylabel(
            "Area-weighted RMSE [m]" if field == "h_m" else "Area-weighted RMSE [m²/s]"
        )
        axis.grid(color="#dfe3e6", linewidth=0.6)
        axis.set_xlim(float(times[1]), float(times[-1]))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.89),
        ncol=1,
        frameon=False,
    )
    fig.suptitle(
        "Per-time differences between numerical reference levels", y=0.98, fontsize=15
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.67, bottom=0.2, wspace=0.3)
    fig.text(
        0.02,
        0.06,
        "50 positive snapshots, SI differences of common-cell means. Constant-CFL "
        "pairs couple mesh and time; half-CFL pair changes time at fixed grid.",
        fontsize=8,
    )
    fig.text(
        0.98,
        0.015,
        f"Sensitivity indicators, not error bounds | protocol {PROTOCOL_DIGEST[:12]}",
        ha="right",
        fontsize=7,
        color="#555b61",
    )
    return save_figure(
        fig,
        output,
        "02_refinement_rmse_by_time",
        {
            "time_set": "50 positive snapshots; initial excluded",
            "metric": "sqrt(sum_A(A*d²)/sum_A(A)) at each time",
            "difference_pairs": [name for name, *_ in PAIRS],
        },
    )


def analyze(root: Path, output: Path, expected_script_sha256: str) -> Path:
    start = time.perf_counter()
    root = root.resolve()
    output = contained(root, output)
    if output.exists():
        raise FileExistsError(f"preserve existing analysis output: {output}")
    protocol, axes, bindings = validate_frozen_inputs(root, expected_script_sha256)
    cells, cell_readback = {}, {}
    for run_id in RUNS:
        cells[run_id], cell_readback[run_id] = validate_cell(
            root, run_id, protocol, axes, bindings
        )
    validation_s = time.perf_counter() - start
    output.mkdir(parents=True, exist_ok=False)
    write_json(
        output / "status.json", {"state": "running", "scope": "saved-data R6 analysis"}
    )
    stage = "signed_differences_norms_and_sensitivity"
    try:
        now = time.perf_counter()
        differences, norms = {}, {}
        for name, high, low, _ in PAIRS:
            differences[name] = cells[high] - cells[low]
            norms[name] = pair_norms(
                differences[name],
                cells[high],
                axes["cell_area_m2"],
                axes["t_s"],
                high,
                low,
            )
        dspace = np.abs(differences["coupled320_minus160"])
        dtime = np.abs(differences["time320_half_minus_full"])
        save_npz(
            output / "signed_differences_and_sensitivity.npz",
            **axes,
            **differences,
            Dspace=dspace,
            Dtime=dtime,
            Dsum=dspace + dtime,
            component_names=np.array(FIELDS),
            positive_time_mask=axes["t_s"] > 0,
        )
        write_json(
            output / "norms.json",
            {
                "protocol_sha256": PROTOCOL_DIGEST,
                "contract_sha256": CONTRACT_SHA256,
                "field_order": list(FIELDS),
                "array_shape": "time,y,x,field",
                "pairs": norms,
                "initial_state_separate": True,
                "sensitivity_meaning": (
                    "Dspace+Dtime is an indicator, not an error bound, "
                    "confidence interval, Richardson extrapolation "
                    "or asymptotic-order proof."
                ),
            },
        )
        difference_norm_io_s = time.perf_counter() - now
        stage = "reference_gradient_region"
        now = time.perf_counter()
        gradient, gradient_metadata = gradient_region(cells[SELECTED], axes)
        save_npz(output / "selected_reference_gradient_region.npz", **gradient)
        write_json(output / "gradient_region.json", gradient_metadata)
        gradient_io_s = time.perf_counter() - now
        stage = "two_contract_figures"
        now = time.perf_counter()
        plot_style()
        figures = [
            reference_map(cells[SELECTED], axes, gradient, output),
            refinement_plot(norms, axes["t_s"], output),
        ]
        write_json(
            output / "figure_manifest.json",
            {
                "protocol_sha256": PROTOCOL_DIGEST,
                "contract_sha256": CONTRACT_SHA256,
                "script_sha256": expected_script_sha256,
                "figures": figures,
                "figure_sources": [
                    "signed_differences_and_sensitivity.npz",
                    "norms.json",
                    "selected_reference_gradient_region.npz",
                    "gradient_region.json",
                ],
                "selected_common_reference_sha256": bindings[
                    str(
                        (
                            root
                            / "results/extensions/two_dimensional/reference_v01"
                            / SELECTED
                            / "common_reference.npz"
                        ).relative_to(root)
                    )
                ],
            },
        )
        figure_io_s = time.perf_counter() - now
        stage = "final_input_integrity_readback"
        now = time.perf_counter()
        for name, expected in bindings.items():
            if file_sha256(root / name) != expected:
                raise ValueError(f"input changed during saved-data analysis: {name}")
        final_input_readback_s = time.perf_counter() - now
        write_json(
            output / "input_binding_and_cell_readback.json",
            {
                "input_sha256": bindings,
                "cells": cell_readback,
                "protocol_sha256": PROTOCOL_DIGEST,
                "contract_sha256": CONTRACT_SHA256,
                "analysis_script_sha256": expected_script_sha256,
                "native_and_common_records_hashed": True,
                "native_fields_reprojected": False,
                "all_mutable_lifecycle_hashes_captured_externally": True,
            },
        )
        write_json(
            output / "analysis_receipt.json",
            {
                "state": "completed_saved_reference_analysis",
                "created_at_utc": datetime.now(UTC).isoformat(),
                "protocol_sha256": PROTOCOL_DIGEST,
                "contract_sha256": CONTRACT_SHA256,
                "analysis_script_sha256": expected_script_sha256,
                "selected_reference": SELECTED,
                "output": str(output.relative_to(root)),
                "software": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "matplotlib": matplotlib.__version__,
                },
                "cost_s": {
                    "input_validation": validation_s,
                    "signed_differences_norms_and_io": difference_norm_io_s,
                    "reference_gradient_region_and_io": gradient_io_s,
                    "two_figures_and_io": figure_io_s,
                    "final_input_hash_readback": final_input_readback_s,
                },
                "wall_s_before_final_manifest": time.perf_counter() - start,
                "peak_process_rss_bytes": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    * (1 if platform.system() == "Darwin" else 1024)
                ),
                "memory_limit": (
                    "process-lifetime peak, not system pressure "
                    "or an instantaneous allocator peak"
                ),
                "solver_updates": 0,
                "area_operator_executions": 0,
                "neural_queries": 0,
                "limits": (
                    "Four finite numerical runs and sensitivity indicators "
                    "do not establish exact truth, physical convergence, "
                    "neural recovery, field validity or completePhase10."
                ),
            },
        )
        write_json(
            output / "output_hashes.json",
            {
                "files": {
                    p.name: file_sha256(p)
                    for p in output.iterdir()
                    if p.is_file() and p.name != "status.json"
                },
                "mutable_marker_excluded": (
                    "status.json commits after mandatory analysis outputs"
                ),
            },
        )
        write_json(
            output / "status.json",
            {
                "state": "completed",
                "scope": "saved-data R6 analysis",
                "actual_entry_to_commit_wall_s": time.perf_counter() - start,
            },
            replace_status=True,
        )
    except BaseException as exc:
        write_json(
            output / "failure.json",
            {
                "stage": stage,
                "type": type(exc).__name__,
                "message": str(exc),
                "partial_outputs_preserved": True,
                "observed_wall_s": time.perf_counter() - start,
            },
        )
        write_json(
            output / "status.json",
            {
                "state": "failed",
                "stage": stage,
                "reason": f"{type(exc).__name__}: {exc}",
            },
            replace_status=True,
        )
        raise
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT_FROM_SCRIPT)
    parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    parser.add_argument(
        "--expected-script-sha256",
        required=True,
        help="SHA256 frozen in the separate analysis launch receipt.",
    )
    args = parser.parse_args()
    output = analyze(args.root, args.output, args.expected_script_sha256)
    print(
        json.dumps(
            {"state": "completed_saved_reference_analysis", "output": str(output)}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
