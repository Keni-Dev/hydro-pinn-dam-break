"""Reanalyze saved sensitivity outputs; never run a solver or original gate.

Use the companion scientific-input inventory. All output goes to a fresh
explicit directory. The original prepare/execute preservation guards remain
unchanged and are not claimed to have passed by this saved-data route.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True
GATE = Path("results/extensions/inverse_sensitivity/gate_v01")
DEFAULT_INPUTS = Path(__file__).with_name("inverse_sensitivity_saved_inputs_v01.json")


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            result.update(chunk)
    return result.hexdigest()


def input_path(root: Path, relative: str) -> Path:
    name = Path(relative)
    if name.is_absolute() or ".." in name.parts:
        raise ValueError(f"unsafe inventory path: {relative}")
    path = root / name
    if any(p.is_symlink() for p in (path, *path.parents) if p != root.parent):
        raise ValueError(f"symlink input not accepted: {relative}")
    if not path.resolve().is_relative_to(root) or not path.is_file():
        raise ValueError(f"missing input within project: {relative}")
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def compare_tree(actual: Any, expected: Any) -> dict[str, Any]:
    differences = []
    max_absolute_difference = 0.0
    numeric_leaves = 0

    def visit(a: Any, b: Any, path: str) -> None:
        nonlocal max_absolute_difference, numeric_leaves
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                differences.append({"path": path, "reason": "mapping keys differ"})
            for key in sorted(set(a) & set(b)):
                visit(a[key], b[key], f"{path}/{key}")
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                differences.append({"path": path, "reason": "sequence lengths differ"})
            for index, (left, right) in enumerate(zip(a, b, strict=False)):
                visit(left, right, f"{path}/{index}")
        elif (
            isinstance(a, (int, float))
            and not isinstance(a, bool)
            and isinstance(b, (int, float))
            and not isinstance(b, bool)
        ):
            numeric_leaves += 1
            if not math.isfinite(a) or not math.isfinite(b):
                differences.append({"path": path, "reason": "nonfinite number"})
            else:
                max_absolute_difference = max(max_absolute_difference, abs(a - b))
                if a != b:
                    differences.append({"path": path, "actual": a, "expected": b})
        elif a != b or type(a) is not type(b):
            differences.append({"path": path, "reason": "value or type differs"})

    visit(actual, expected, "")
    return {
        "exact_match": not differences,
        "numeric_leaves_compared": numeric_leaves,
        "maximum_absolute_numeric_difference": max_absolute_difference,
        "difference_count": len(differences),
        "first_differences": differences[:20],
        "comparison_rule": (
            "exact saved-value equality; no tolerance selected from results"
        ),
    }


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def reanalyze(root: Path, output: Path, inventory_path: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    output = output.absolute()
    if output.exists():
        raise FileExistsError("output directory already exists; use a new destination")
    for parent in output.parents:
        if parent.is_symlink():
            raise ValueError("symlink output parent not accepted")
    if output.resolve().is_relative_to(root):
        relative_output = output.resolve().relative_to(root)
        if (
            not relative_output.parts
            or relative_output.parts[0] != "results"
            or relative_output.parts[:2] == ("results", "extensions")
        ):
            raise ValueError("in-project outputs must use a new results directory")
    inventory = read_json(inventory_path)
    if inventory.get("purpose") != "saved_sensitivity_scientific_inputs_only":
        raise ValueError("wrong saved-input inventory purpose")
    expected = {row["path"]: row for row in inventory["files"]}
    if len(expected) != len(inventory["files"]):
        raise ValueError("duplicate input inventory paths")
    verified = {}
    for relative, row in expected.items():
        path = input_path(root, relative)
        actual = digest(path)
        if actual != row["sha256"] or path.stat().st_size != row["bytes"]:
            raise ValueError(f"scientific input hash/size mismatch: {relative}")
        verified[relative] = actual

    # The project modules must come from the supplied root, not an editable
    # installation elsewhere. Imports define functions; none launches a solve.
    sys.path.insert(0, str(root / "src"))
    gate = importlib.import_module("hydro_pinn.extensions.inverse_sensitivity")
    friction = importlib.import_module("hydro_pinn.extensions.friction_reference")
    for module, relative in [
        (gate, "src/hydro_pinn/extensions/inverse_sensitivity.py"),
        (friction, "src/hydro_pinn/extensions/friction_reference.py"),
    ]:
        if Path(module.__file__).resolve() != input_path(root, relative).resolve():
            raise ValueError("project module already loaded from a different root")
    import numpy as np

    protocol = read_json(root / GATE / "protocol.json")
    if (
        digest(root / GATE / "protocol.json")
        != (root / GATE / "protocol.sha256").read_text().strip()
    ):
        raise ValueError("stored protocol digest differs")
    c = protocol["config"]
    gate._validate_config(c)
    operator_sha = hashlib.sha256(
        (
            inspect.getsource(friction.conservative_project)
            + inspect.getsource(gate.extract_gauge_means)
        ).encode()
    ).hexdigest()
    if operator_sha != protocol["operator_source_sha256"]:
        raise ValueError("canonical conservative-extraction operator differs")
    matrix = read_json(root / GATE / "matrix.json")
    records = {(r["candidate_id"], r["id"]): r for r in matrix["records"]}
    if len(protocol["matrix"]) != 15 or len(records) != 15:
        raise ValueError("expected five candidates at three native levels")
    if matrix["budget_exceeded"] or matrix["wall_s"] >= matrix["budget_s"]:
        raise ValueError("saved gate did not complete within its declared budget")
    with np.load(root / GATE / "frozen_axes.npz", allow_pickle=False) as data:
        times = data["t_s"].copy()
        edges = data["common_cell_edges_m"].copy()
        np.testing.assert_array_equal(data["time_row_indices"], c["time_row_indices"])
        np.testing.assert_array_equal(
            data["gauge_cell_indices"], c["gauge_common_cell_indices"]
        )
    np.testing.assert_array_equal(times, protocol["full_t_s"])
    responses = {n: {} for n in c["candidate_ids"]}
    native_checks = []
    for entry in protocol["matrix"]:
        record = records[(entry["candidate_id"], entry["id"])]
        relative = entry["path"]
        if record["state"] not in ("completed", "reused"):
            raise ValueError("saved matrix has an incomplete record")
        if relative not in verified or verified[relative] != record["outputs_sha256"]:
            raise ValueError("native payload differs from the accepted matrix")
        with np.load(input_path(root, relative), allow_pickle=False) as data:
            native = {key: data[key].copy() for key in data.files}
        np.testing.assert_array_equal(native["t_s"], times)
        np.testing.assert_array_equal(
            native["cell_edges_m"], np.linspace(-5, 5, entry["cells"] + 1)
        )
        means = gate.extract_gauge_means(
            native, edges, c["time_row_indices"], c["gauge_common_cell_indices"]
        )
        np.testing.assert_array_equal(means, record["gauge_means_m"])
        responses[entry["candidate_id"]][entry["id"]] = means
        diagnostics = gate._diagnostics(native, entry)
        native_checks.append(
            {
                "candidate_id": entry["candidate_id"],
                "level": entry["id"],
                "outputs_sha256": verified[relative],
                "diagnostics_comparison": compare_tree(
                    diagnostics, record["diagnostics"]
                ),
            }
        )
    means = np.array(
        [
            [responses[n][level["id"]] for level in gate.LEVELS]
            for n in c["candidate_ids"]
        ]
    )
    with np.load(root / GATE / "gauge_means.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["n_values"], c["n_values"])
        np.testing.assert_array_equal(data["means_m"], means)
        np.testing.assert_array_equal(
            data["selected_t_s"], times[c["time_row_indices"]]
        )
        np.testing.assert_array_equal(data["common_cell_edges_m"], edges)
    analysis = gate.analyze(responses, c)
    comparison = compare_tree(analysis, read_json(root / GATE / "analysis.json"))
    unchanged = all(
        digest(input_path(root, relative)) == value
        for relative, value in verified.items()
    )
    passed = (
        comparison["exact_match"]
        and unchanged
        and all(row["diagnostics_comparison"]["exact_match"] for row in native_checks)
    )
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "state": "passed_saved_sensitivity_reanalysis"
        if passed
        else "failed_saved_sensitivity_reanalysis",
        "purpose": "saved_output_projection_and_arithmetic_reanalysis",
        "scientific_input_inventory_sha256": digest(inventory_path),
        "helper_source_sha256": digest(Path(__file__)),
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__},
        "verified_inputs_sha256": verified,
        "native_records_reanalyzed": 15,
        "responses_recomputed": int(means.size),
        "pair_screens_recomputed": len(analysis["pairs"]),
        "analysis_comparison": comparison,
        "native_budget_checks": native_checks,
        "source_inputs_unchanged": unchanged,
        "outcome": analysis["outcome"],
        "local_adjusted_distances": [
            p["d_adj"]
            for p in analysis["pairs"]
            if [p["a"], p["b"]] in analysis["truth_local_pairs"]
        ],
        "numerical_solver_executed": False,
        "training_executed": False,
        "new_timing_measurements": False,
        "legacy_preservation_guard_rechecked": False,
        "legacy_prepare_execute_made_portable": False,
        "scope": (
            "Recomputes saved native-output projection, diagnostics and pair "
            "statistics using unchanged canonical functions. No PDE solve, "
            "inverse fit, noise draws, or original prepare/execute is called."
        ),
        "limit": (
            "Historical source/archive preservation bindings are not re-certified. "
            "This route does not authorize or replace a fresh guard-bound gate "
            "execution, and adds no physical identifiability or recovery claim."
        ),
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "analysis_recomputed.json", analysis)
    with (output / "gauge_means_recomputed.npz").open("xb") as stream:
        np.savez_compressed(
            stream,
            n_values=np.array(c["n_values"]),
            means_m=means,
            selected_t_s=times[c["time_row_indices"]],
            common_cell_edges_m=edges,
        )
    write_json(output / "reanalysis_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-inventory", type=Path, default=DEFAULT_INPUTS)
    args = parser.parse_args()
    report = reanalyze(args.root, args.output_dir, args.input_inventory)
    print(
        json.dumps(
            {
                key: report[key]
                for key in [
                    "state",
                    "native_records_reanalyzed",
                    "responses_recomputed",
                    "pair_screens_recomputed",
                    "outcome",
                    "local_adjusted_distances",
                    "source_inputs_unchanged",
                ]
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["state"].startswith("passed_") else 1)


if __name__ == "__main__":
    main()
