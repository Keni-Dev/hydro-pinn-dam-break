"""Render a fresh core timing/accuracy comparison from explicit saved JSON inputs.

Run after the fresh Phase 07 evaluation and benchmark have completed. This utility
does not train or measure performance, and does not use production_summary.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
CATEGORIES = {
    "analytical_reference_generation": ("Exact reference", "#222222", "D"),
    "numerical_integration_and_reconstruction": ("Fresh N=400", "#5D3A9B", "s"),
    "resident_classical_cache_reconstruction": ("Cached N=400", "#0072B2", "P"),
    "resident_network_physical_forward": ("PINN", "#D55E00", "o"),
}
NETWORK_MARKERS = {
    ("baseline", 11): "o",
    ("adaptive", 11): "^",
    ("baseline", 22): "s",
    ("adaptive", 22): "v",
    ("baseline", 33): "D",
    ("adaptive", 33): "P",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_path(value: str | Path) -> Path:
    """Resolve local inputs/outputs without following a symlink destination."""
    raw = Path(value)
    absolute = Path(os.path.abspath(raw if raw.is_absolute() else ROOT / raw))
    require(absolute.is_relative_to(ROOT), f"path must be within project: {value}")
    for component in (absolute, *absolute.parents):
        if component == ROOT.parent:
            break
        require(
            not component.is_symlink(), f"symlink path is not accepted: {component}"
        )
    resolved = absolute.resolve()
    require(resolved.is_relative_to(ROOT), f"path leaves project: {value}")
    return resolved


def finite_tree(value: Any) -> None:
    if isinstance(value, float):
        require(math.isfinite(value), "nonfinite JSON number")
    elif isinstance(value, dict):
        for item in value.values():
            finite_tree(item)
    elif isinstance(value, list):
        for item in value:
            finite_tree(item)


def unique_pairs(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value: str) -> None:
    raise ValueError(f"nonstandard JSON constant: {value}")


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        data = json.load(
            handle, object_pairs_hook=unique_pairs, parse_constant=reject_constant
        )
    require(isinstance(data, dict), f"JSON object required: {path}")
    finite_tree(data)
    return data


def number(value: Any, label: str, *, positive: bool = False) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"numeric value required: {label}",
    )
    result = float(value)
    require(math.isfinite(result), f"finite value required: {label}")
    require(result > 0 if positive else result >= 0, f"invalid value: {label}")
    return result


def digest(value: Any, label: str) -> str:
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"SHA256 required: {label}",
    )
    return value


def validate_provenance(metrics: dict, timings: dict) -> tuple[dict, list[Path], dict]:
    require(metrics.get("state") == "completed", "metrics must be completed")
    require(timings.get("state") == "completed", "timings must be completed")
    mp, tp = metrics["provenance"], timings["provenance"]
    for key in ("protocol_sha256", "protocol_file_sha256", "manifest_sha256"):
        require(digest(mp.get(key), key) == digest(tp.get(key), key), f"{key} mismatch")
    require(metrics.get("protocol_id") == tp.get("protocol_id"), "protocol ID mismatch")
    protocol_path = project_path(mp["protocol_resolution"]["resolved_local_path"])
    protocol = read_json(protocol_path)
    require(sha(protocol_path) == mp["protocol_file_sha256"], "protocol file changed")
    payload = {
        key: value for key, value in protocol.items() if key != "protocol_sha256"
    }
    computed = hashlib.sha256(
        (json.dumps(payload, allow_nan=False, sort_keys=True, indent=2) + "\n").encode()
    ).hexdigest()
    require(
        computed == protocol.get("protocol_sha256") == mp["protocol_sha256"],
        "protocol digest mismatch",
    )
    require(
        protocol.get("protocol_id") == metrics["protocol_id"], "protocol ID changed"
    )
    execution_manifest = mp["execution"]["manifest"]
    require(
        isinstance(execution_manifest, str)
        and not Path(execution_manifest).is_absolute(),
        "metrics execution manifest must be project-relative",
    )
    # The recorded absolute path belongs to the producing checkout. Resolve the
    # hash-bound execution path locally; never open the historical absolute path.
    manifests = [project_path(execution_manifest), project_path(tp["manifest_path"])]
    for path in manifests:
        require(sha(path) == mp["manifest_sha256"], "manifest file changed")
        manifest = read_json(path)
        require(
            manifest.get("workflow_status")
            in {"completed", "completed_with_diagnostics"},
            "manifest matrix must be terminal and completed",
        )
        require(
            manifest.get("protocol_sha256") == mp["protocol_sha256"],
            "manifest protocol mismatch",
        )
    resolution = {
        "recorded_metrics_manifest_path": mp.get("manifest_path"),
        "metrics_execution_manifest": execution_manifest,
        "resolved_local_manifest_path": str(manifests[0].relative_to(ROOT)),
        "timing_manifest_path": tp["manifest_path"],
        "manifest_sha256": mp["manifest_sha256"],
        "historical_absolute_path_read": False,
    }
    return protocol, list(dict.fromkeys([protocol_path, *manifests])), resolution


def validate_grid(metrics: dict, timings: dict, protocol: dict) -> list[float]:
    evaluation = metrics["evaluation"]
    require(
        evaluation == protocol["resolved"]["evaluation"], "evaluation config mismatch"
    )
    grid = evaluation["grid"]
    require(
        grid.get("cells") == 400 and grid.get("kind") == "cell_centres",
        "the frozen comparison requires 400 cell centres",
    )
    times = evaluation["times_s"]
    require(
        times == [index / 100 for index in range(1, 51)], "50 frozen times required"
    )
    workload = timings["workload"]
    require(workload.get("shape_h_q") == [50, 400, 2], "h/q workload shape mismatch")
    require(workload.get("fields") == ["h_m", "q_m2_s"], "h/q workload fields mismatch")
    require(workload.get("t_s") == times, "timing and metric time axes differ")
    x = workload["x_m"]
    require(
        isinstance(x, list) and len(x) == 400, "timing x axis must contain 400 points"
    )
    a, b = grid["domain_m"]
    require(b > a, "invalid evaluation domain")
    require(
        all(
            math.isclose(
                value, a + (index + 0.5) * (b - a) / 400, rel_tol=0, abs_tol=1e-12
            )
            for index, value in enumerate(x)
        ),
        "timing x axis differs from the frozen cell centres",
    )
    require(
        workload.get("plotting_in_main_timing") is False, "plotting entered timings"
    )
    require(
        workload.get("reference_and_error_calculation_in_main_timing") is False,
        "error calculation entered main timings",
    )
    return times


def pooled_depth(metrics: dict, run: dict, times: list[float]) -> float:
    selected = [
        row
        for row in metrics["rows"]
        if row.get("run_id") == run["run_id"]
        and row.get("field") == "h_m"
        and row.get("region") == "whole"
    ]
    selected.sort(key=lambda row: row["time_s"])
    require(
        [row["time_s"] for row in selected] == times, "missing/duplicate h metric times"
    )
    errors = []
    for row in selected:
        for key in ("case_id", "method", "variant", "seed", "cells"):
            require(
                row.get(key) == run.get(key), f"metric row identity mismatch: {key}"
            )
        require(
            row.get("selected_points") == row.get("total_points") == 400
            and row.get("coverage_fraction") == 1
            and row.get("spatial_weighting") == "equal_cell_centres",
            "pooled h RMSE requires full equal-weight 400-point coverage at every time",
        )
        require(row["rmse"].get("reason") is None, "undefined h RMSE")
        errors.append(number(row["rmse"].get("value"), "whole-domain h RMSE"))
    # Each of 50 per-time RMSEs has exactly 400 equally weighted samples.
    return math.sqrt(math.fsum(error * error for error in errors) / len(errors))


def expected_costs(protocol: dict) -> set[tuple]:
    settings = protocol["settings"]
    require(settings.get("seeds") == [11, 22, 33], "unsupported core paired seeds")
    require(settings.get("e2_diagnostic_seed") == 11, "unsupported E2 diagnostic seed")
    expected = set()
    for case in ("E1", "E2"):
        for category in CATEGORIES:
            if category != "resident_network_physical_forward":
                cells = None if category == "analytical_reference_generation" else 400
                expected.add((case, category, None, None, cells))
    for variant, seed in NETWORK_MARKERS:
        expected.add(("E1", "resident_network_physical_forward", variant, seed, None))
    expected.add(("E2", "resident_network_physical_forward", "baseline", 11, None))
    return expected


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    return ordered[lower] + (index - lower) * (
        ordered[min(lower + 1, len(ordered) - 1)] - ordered[lower]
    )


def resolve_dtype(cost: dict, timings: dict) -> tuple[str, dict]:
    """Use a recorded dtype or the uniquely linked saved integration dtype."""
    source = cost["run_provenance"]
    config_hash = digest(source.get("resolved_config_sha256"), "dtype config hash")
    output_hash = digest(source["files_sha256"].get("outputs.npz"), "dtype output hash")
    origin, kind = cost, "recorded"
    if cost.get("dtype") is None:
        require(
            cost.get("category") == "resident_classical_cache_reconstruction",
            "only a saved classical cache may derive a missing dtype",
        )
        matches = [
            candidate
            for candidate in timings["costs"]
            if candidate.get("category") == "numerical_integration_and_reconstruction"
            and all(
                candidate.get(key) == cost.get(key)
                for key in ("run_id", "case_id", "cells", "stage_id")
            )
            and candidate.get("run_provenance", {}).get("resolved_config_sha256")
            == config_hash
            and candidate.get("run_provenance", {})
            .get("files_sha256", {})
            .get("outputs.npz")
            == output_hash
        ]
        require(len(matches) == 1, "cache dtype needs one matching integration record")
        origin, kind = matches[0], "derived_from_matching_integration_record"
        require(
            origin.get("state") == "completed"
            and origin["run_provenance"].get("record_state") == "completed"
            and origin["run_provenance"].get("run_id") == cost["run_id"],
            "cache dtype source must be the completed matching run",
        )
    saved_dtype = origin.get("dtype")
    dtype = saved_dtype.removeprefix("torch.") if isinstance(saved_dtype, str) else None
    require(dtype in {"float32", "float64"}, "timing dtype missing/unsupported")
    return dtype, {
        "kind": kind,
        "run_id": origin["run_id"],
        "category": origin["category"],
        "case_id": origin["case_id"],
        "cells": origin.get("cells"),
        "stage_id": origin.get("stage_id"),
        "resolved_config_sha256": config_hash,
        "outputs_npz_sha256": output_hash,
        "saved_dtype": saved_dtype,
    }


def points_from_saved(metrics: dict, timings: dict, protocol: dict) -> list[dict]:
    times = validate_grid(metrics, timings, protocol)
    runs = {run["run_id"]: run for run in metrics["runs"]}
    require(len(runs) == len(metrics["runs"]), "duplicate metric run IDs")
    expected, seen, points = expected_costs(protocol), set(), []
    repetitions = timings["workload"]["repetitions"]
    warmups = timings["workload"]["warmups"]
    require(
        repetitions == protocol["benchmark"]["measured_repetitions"]
        and warmups == protocol["benchmark"]["warmup_repetitions"],
        "timing repetition/warmup counts differ from protocol",
    )
    for cost in timings["costs"]:
        category = cost.get("category")
        if category not in CATEGORIES:
            continue
        if (
            category
            in {
                "numerical_integration_and_reconstruction",
                "resident_classical_cache_reconstruction",
            }
            and cost.get("cells") != 400
        ):
            continue
        key = tuple(
            cost.get(name)
            for name in ("case_id", "category", "variant", "seed", "cells")
        )
        require(key in expected, f"unexpected plotted cost identity: {key}")
        require(key not in seen, f"duplicate plotted cost identity: {key}")
        seen.add(key)
        require(cost.get("state") == "completed", "a plotted timing cost is incomplete")
        require(cost["run_id"] in runs, "timing run is absent from metrics")
        run = runs[cost["run_id"]]
        require(run.get("state") == "completed", "a plotted metric run is incomplete")
        for name in ("case_id", "variant", "seed", "cells", "stage_id"):
            require(
                run.get(name) == cost.get(name),
                f"timing/metric identity differs: {name}",
            )
        method = {
            "analytical_reference_generation": "analytical_reference",
            "resident_network_physical_forward": "pinn",
        }.get(category, "lax_wendroff")
        require(run.get("method") == method, "timing category/metric method mismatch")
        source = cost["run_provenance"]
        require(source.get("run_id") == run["run_id"], "timing source run ID mismatch")
        require(
            source.get("record_state") == "completed", "timing source is incomplete"
        )
        require(
            source.get("resolved_config_sha256")
            == run.get("resolved_config_sha256")
            == run["provenance"].get("resolved_config_sha256"),
            "timing/metric run configuration hash mismatch",
        )
        require(
            digest(source["files_sha256"].get("outputs.npz"), "timing output hash")
            == digest(
                run["provenance"]["files_sha256"].get("outputs.npz"),
                "metric output hash",
            ),
            "timing/metric saved solution hash mismatch",
        )
        require(
            cost.get("measured_repetitions")
            == cost.get("measured_repetitions_requested")
            == repetitions
            and cost.get("warmup_repetitions_completed")
            == cost.get("warmup_repetitions_requested")
            == warmups,
            "incomplete timing samples or warmups",
        )
        raw = [
            number(value, "raw timing sample", positive=True)
            for value in cost["raw_samples_s"]
        ]
        require(len(raw) == repetitions, "raw sample count mismatch")
        require(
            cost.get("quantile_rule") == "numpy.quantile method=linear",
            "unsupported quantile rule",
        )
        values = {}
        for name, fraction in (("median", 0.5), ("q1", 0.25), ("q3", 0.75)):
            value = number(cost[f"{name}_s"], name, positive=True)
            require(
                math.isclose(
                    value, quantile(raw, fraction), rel_tol=2e-14, abs_tol=1e-18
                ),
                f"saved {name} differs from raw samples",
            )
            values[f"{name}_ms"] = 1000 * value
        require(
            values["q1_ms"] <= values["median_ms"] <= values["q3_ms"],
            "invalid timing quartiles",
        )
        require(
            isinstance(cost.get("device"), str) and cost["device"],
            "timing device missing",
        )
        dtype, dtype_provenance = resolve_dtype(cost, timings)
        points.append(
            {
                "case_id": run["case_id"],
                "run_id": run["run_id"],
                "category": category,
                "variant": run.get("variant"),
                "seed": run.get("seed"),
                "cells": run.get("cells"),
                **values,
                "pooled_depth_rmse_m": pooled_depth(metrics, run, times),
                "device": cost["device"],
                "dtype": dtype,
                "saved_dtype": cost.get("dtype"),
                "dtype_provenance": dtype_provenance,
                "measured_repetitions": repetitions,
                "warmup_repetitions": warmups,
            }
        )
    require(
        seen == expected and len(points) == 13,
        "the full 13-point core comparison is required",
    )
    return sorted(
        points,
        key=lambda p: (
            p["case_id"],
            list(CATEGORIES).index(p["category"]),
            p["seed"] or 0,
            p["variant"] or "",
        ),
    )


def host_observations(value: Any, path: str = "timings") -> list[dict]:
    """Read every saved host sample, including nested per-workload observations."""
    observations = []
    if isinstance(value, dict):
        if "load_average_1_5_15_minutes" in value:
            loads = value["load_average_1_5_15_minutes"]
            if loads is not None:
                require(
                    isinstance(loads, list) and len(loads) == 3,
                    "invalid saved host load",
                )
                loads = [number(item, "host load average") for item in loads]
                cpus = value.get("logical_cpu_count")
                require(
                    cpus is None
                    or isinstance(cpus, int)
                    and not isinstance(cpus, bool)
                    and cpus > 0,
                    "invalid logical CPU count",
                )
                observations.append(
                    {
                        "path": path,
                        "load_average_1_5_15_minutes": loads,
                        "logical_cpu_count": cpus,
                    }
                )
        for key, item in value.items():
            observations.extend(host_observations(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            observations.extend(host_observations(item, f"{path}[{index}]"))
    return observations


def caption_from_saved(points: list[dict], timings: dict) -> tuple[str, dict]:
    observations = host_observations(timings)
    cpus = sorted(
        {
            row["logical_cpu_count"]
            for row in observations
            if row["logical_cpu_count"] is not None
        }
    )
    load = [row["load_average_1_5_15_minutes"][0] for row in observations]
    context = {
        "one_minute_load_min": min(load) if load else None,
        "one_minute_load_max": max(load) if load else None,
        "logical_cpu_counts": cpus,
        "host_observations": observations,
        "cpu_threads": timings.get("cpu_threads"),
        "torch_interop_threads": timings.get("torch_interop_threads"),
        "hardware_precision_by_category": {
            category: sorted(
                {
                    f"{point['device'].upper()} {point['dtype']}"
                    for point in points
                    if point["category"] == category
                }
            )
            for category in CATEGORIES
        },
    }
    samples = sorted({point["measured_repetitions"] for point in points})
    warmups = sorted({point["warmup_repetitions"] for point in points})
    require(len(samples) == len(warmups) == 1, "mixed sample counts are unsupported")
    hardware = "; ".join(
        f"{CATEGORIES[category][0]}: {', '.join(devices)}"
        for category, devices in context["hardware_precision_by_category"].items()
    )
    host = (
        f"Observed one-minute host load {min(load):.2f}–{max(load):.2f}"
        if load
        else "Host load observations unavailable"
    )
    host += (
        f"; logical CPU counts: {', '.join(map(str, cpus))}."
        if cpus
        else "; logical CPU count unrecorded."
    )
    caption = (
        f"Horizontal bars: Q1–Q3 of {samples[0]} samples after "
        f"{warmups[0]} warmups. {hardware}.\n"
        "Resident inputs/model/cache; training, loading and host export excluded. "
        f"{host} Other host processes were not experimentally controlled."
    )
    return caption, context


def render(points: list[dict], caption: str, destinations: list[Path]) -> list[dict]:
    # Import the renderer only after all input and destination checks have passed.
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for case_id, ax in zip(("E1", "E2"), axes, strict=True):
        for point in points:
            if point["case_id"] != case_id:
                continue
            name, color, marker = CATEGORIES[point["category"]]
            if point["category"] == "resident_network_physical_forward":
                name = f"{point['variant']} seed {point['seed']}"
                color = "#D55E00" if point["variant"] == "baseline" else "#009E73"
                marker = NETWORK_MARKERS[(point["variant"], point["seed"])]
            x, low, high = (point[key] for key in ("median_ms", "q1_ms", "q3_ms"))
            ax.errorbar(
                x,
                point["pooled_depth_rmse_m"],
                xerr=[[x - low], [high - x]],
                fmt=marker,
                color=color,
                capsize=3,
                markersize=7,
                label=name,
            )
        ax.set(
            xscale="log",
            xlabel="400×50 h/q workload median (ms, log scale)",
            ylabel="Depth RMSE over all 20,000 points (m)",
            title=case_id,
        )
        ax.set_ylim(bottom=-0.002)
        ax.grid(axis="x", alpha=0.2)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=8)
    fig.suptitle("Measured cost and depth accuracy on the frozen common grid")
    caption_lines = "\n".join(
        textwrap.fill(paragraph, width=185) for paragraph in caption.splitlines()
    )
    fig.text(0.02, 0.025, caption_lines, fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, 0.2, 1, 0.94))
    artifacts = []
    try:
        for destination in destinations:
            with destination.open("xb") as handle:
                fig.savefig(
                    handle, format=destination.suffix[1:], dpi=160, bbox_inches="tight"
                )
            artifacts.append(
                {"path": str(destination.relative_to(ROOT)), "sha256": sha(destination)}
            )
    finally:
        plt.close(fig)
    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("metrics", "timings", "output-dir", "evidence-output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    metrics_path, timing_path = project_path(args.metrics), project_path(args.timings)
    output_dir, evidence_path = (
        project_path(args.output_dir),
        project_path(args.evidence_output),
    )
    exports = [output_dir / f"timing_accuracy.{suffix}" for suffix in ("png", "pdf")]
    destinations = [*exports, evidence_path]
    require(len(set(destinations)) == 3, "output destinations must be distinct")
    for destination in destinations:
        require(not destination.exists(), f"destination already exists: {destination}")
    metrics, timings = read_json(metrics_path), read_json(timing_path)
    protocol, linked_sources, manifest_resolution = validate_provenance(
        metrics, timings
    )
    points = points_from_saved(metrics, timings, protocol)
    caption, context = caption_from_saved(points, timings)
    inputs = list(dict.fromkeys([metrics_path, timing_path, *linked_sources]))
    source_artifacts = [
        {"path": str(path.relative_to(ROOT)), "sha256": sha(path)} for path in inputs
    ]
    for destination in destinations:
        destination.parent.mkdir(parents=True, exist_ok=True)
    artifacts = render(points, caption, exports)
    evidence = {
        "schema_version": 1,
        "figure_id": "timing_accuracy",
        "kind": "timing_and_accuracy",
        "purpose": "fresh_saved_core_timing_accuracy_rebuild",
        "state": "completed",
        "artifacts": artifacts,
        "run_ids": sorted({point["run_id"] for point in points}),
        "points": points,
        "caption": caption,
        "derived_timing_context": context,
        "provenance": {
            key: metrics["provenance"][key]
            for key in ("protocol_sha256", "protocol_file_sha256", "manifest_sha256")
        },
        "protocol_id": metrics["protocol_id"],
        "manifest_resolution": manifest_resolution,
        "source_artifacts": source_artifacts,
        "plot_source_sha256": {
            str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))
        },
        "pooled_depth_rmse_rule": (
            "sqrt(mean(per_time_whole_h_rmse^2)); "
            "50 exact times with 400 equal-weight points each"
        ),
        "scope": (
            "Depth error versus resident complete h/q output cost; "
            "no generalization or hardware-independent speed claim"
        ),
        "performance_measurements_executed_by_renderer": False,
    }
    with evidence_path.open("x", encoding="utf-8") as handle:
        json.dump(evidence, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "figure": "timing_accuracy",
                "points": len(points),
                "evidence": str(evidence_path.relative_to(ROOT)),
            }
        )
    )


if __name__ == "__main__":
    main()
