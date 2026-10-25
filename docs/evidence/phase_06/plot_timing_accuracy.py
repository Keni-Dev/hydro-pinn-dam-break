"""Regenerate the final timing/accuracy figure from saved Phase 06 evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/aifox-phase06-mpl")
os.environ.setdefault("MPL_IGNORE_SYSTEM_FONTS", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timings", type=Path, default=Path("results/timings.json"))
    parser.add_argument("--metrics", type=Path, default=Path("results/metrics.json"))
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("docs/evidence/phase_06/production_summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    parser.add_argument(
        "--evidence-output",
        type=Path,
        default=Path("docs/evidence/phase_06/timing_figure_evidence.json"),
    )
    args = parser.parse_args()
    timing_path = ROOT / args.timings
    metrics_path = ROOT / args.metrics
    summary_path = ROOT / args.summary
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    timing = json.loads(timing_path.read_text())
    summary = json.loads(summary_path.read_text())
    assert timing["state"] == "completed"
    by_run = {run["run_id"]: run for run in summary["runs"]}
    categories = {
        "analytical_reference_generation": ("Exact reference", "#222222", "D"),
        "numerical_integration_and_reconstruction": ("Fresh N=400", "#5D3A9B", "s"),
        "resident_classical_cache_reconstruction": ("Cached N=400", "#0072B2", "P"),
        "resident_network_physical_forward": ("PINN", "#D55E00", "o"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    sources, figure_points = set(), []
    for case_id, ax in zip(("E1", "E2"), axes, strict=True):
        for row in timing["costs"]:
            category = row["category"]
            if row["case_id"] != case_id or category not in categories:
                continue
            if row.get("cells") not in (None, 400):
                continue
            name, color, marker = categories[category]
            if category == "resident_network_physical_forward":
                name = f"{row['variant']} seed {row['seed']}"
                color = "#D55E00" if row["variant"] == "baseline" else "#009E73"
                marker = {
                    ("baseline", 11): "o",
                    ("adaptive", 11): "^",
                    ("baseline", 22): "s",
                    ("adaptive", 22): "v",
                    ("baseline", 33): "D",
                    ("adaptive", 33): "P",
                }[(row["variant"], row["seed"])]
            error = by_run[row["run_id"]]["pooled_all_20000_points"]["h_m"]["rmse"]
            x, low, high = (1000 * row[key] for key in ("median_s", "q1_s", "q3_s"))
            ax.errorbar(
                x,
                error,
                xerr=[[x - low], [high - x]],
                fmt=marker,
                color=color,
                capsize=3,
                markersize=7,
                label=name,
            )
            sources.add(row["run_id"])
            figure_points.append(
                {
                    "case_id": case_id,
                    "run_id": row["run_id"],
                    "category": category,
                    "median_ms": x,
                    "q1_ms": low,
                    "q3_ms": high,
                    "pooled_depth_rmse_m": error,
                }
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
    fig.text(
        0.02,
        0.035,
        "Horizontal bars: Q1–Q3 of 30 samples after 10 warmups. "
        "CPU float64 solver/reference versus CPU float32 PINN.\n"
        "Resident inputs/model/cache; training, loading and host export excluded. "
        "Other host load was uncontrolled; observed one-minute load 16.95–20.37 "
        "on 10 logical CPUs.",
        fontsize=8,
        va="bottom",
    )
    fig.tight_layout(rect=(0, 0.15, 1, 0.94))
    artifacts = []
    for suffix in ("png", "pdf"):
        path = output_dir / f"timing_accuracy.{suffix}"
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=160, bbox_inches="tight")
        artifacts.append({"path": str(path), "sha256": sha(path)})
    plt.close(fig)
    evidence = {
        "figure_id": "timing_accuracy",
        "kind": "timing_and_accuracy",
        "artifacts": artifacts,
        "run_ids": sorted(sources),
        "points": figure_points,
        "plot_source_sha256": {
            str(Path(__file__).relative_to(ROOT)): sha(Path(__file__))
        },
        "source_artifacts": [
            {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}
            for path in (metrics_path, timing_path, summary_path)
        ],
        "scope": (
            "Depth error versus resident complete h/q output cost; "
            "no generalization or hardware-independent speed claim"
        ),
    }
    target = ROOT / args.evidence_output
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x") as handle:
        json.dump(evidence, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"figure": "timing_accuracy", "points": len(figure_points)}))


if __name__ == "__main__":
    main()
