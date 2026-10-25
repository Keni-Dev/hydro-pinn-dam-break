"""Analytical E1 label partitions, stored separately from frozen final axes.

These generated idealized solution values are not field observations. Loading
training labels never opens the validation file or evaluates the final axes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hydro_pinn.config import load_config
from hydro_pinn.records import RunRecord, write_json_atomic
from hydro_pinn.references import REFERENCE_SOURCES, evaluate_reference

DEFAULT_LABELS = {
    "seed": 110211,
    "training_count": 256,
    "validation_count": 128,
    "training_window_s": [0.0, 0.35],
    "validation_window_s": [0.35, 0.40],
    "final_window_s": [0.40, 0.50],
    "final_n_x": 400,
    "final_n_t": 11,
}


def file_sha256(path: str | Path) -> str:
    """Hash persisted file bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha256(array: NDArray[Any]) -> str:
    """Hash an array's exact dtype, shape and C-order bytes."""
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"dtype": value.dtype.str, "shape": list(value.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    )
    digest.update(b"\n")
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _path(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def _strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")

    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(), parse_constant=reject, object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError("hybrid configuration must be an object")
    return value


def _options(config: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    options = copy.deepcopy(DEFAULT_LABELS)
    supplied = config.get("labels", {})
    if not isinstance(supplied, dict) or supplied.keys() - options.keys():
        raise ValueError("unknown or invalid hybrid labels settings")
    options.update(copy.deepcopy(supplied))
    for key in ("training_count", "validation_count", "final_n_x", "final_n_t"):
        value = options[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"labels.{key} must be a positive integer")
    if options["final_n_t"] < 2:
        raise ValueError("final_n_t must include both final-window endpoints")
    seed = options["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("label seed must be an integer in [0,2**32)")
    windows = []
    for key in ("training_window_s", "validation_window_s", "final_window_s"):
        values = np.asarray(options[key], dtype=float)
        if (
            values.shape != (2,)
            or not np.isfinite(values).all()
            or not 0 <= values[0] < values[1] <= case["final_time_s"]
        ):
            raise ValueError(f"invalid labels.{key}")
        windows.append(values)
    if not (windows[0][1] <= windows[1][0] and windows[1][1] <= windows[2][0]):
        raise ValueError(
            "label training, validation and final windows must be disjoint"
        )
    return options


def _draw_coordinates(
    rng: np.random.Generator,
    count: int,
    domain: list[float],
    window: list[float],
    *,
    exclude_lower: bool,
) -> NDArray[np.float64]:
    # nextafter excludes domain endpoints and (for training) t=0 exactly.
    left, right = domain
    start, stop = window
    x = rng.uniform(np.nextafter(left, right), right, count)
    lower = np.nextafter(start, stop) if exclude_lower else start
    t = rng.uniform(lower, stop, count)
    return np.column_stack((x, t))


def _validate_memberships(
    train: NDArray[Any],
    validation: NDArray[Any],
    x_m: NDArray[Any],
    t_s: NDArray[Any],
    options: dict[str, Any],
    domain: list[float],
) -> None:
    left, right = domain
    for name, values, count, window, lower_inclusive in (
        (
            "training",
            train,
            options["training_count"],
            options["training_window_s"],
            False,
        ),
        (
            "validation",
            validation,
            options["validation_count"],
            options["validation_window_s"],
            True,
        ),
    ):
        if values.shape != (count, 2) or not np.isfinite(values).all():
            raise ValueError(f"{name} coordinates shape/finiteness mismatch")
        start, stop = window
        lower_ok = values[:, 1] >= start if lower_inclusive else values[:, 1] > start
        if not (
            np.all(left < values[:, 0])
            and np.all(values[:, 0] < right)
            and np.all(lower_ok)
            and np.all(values[:, 1] < stop)
        ):
            raise ValueError(f"{name} coordinates violate partition membership")
        if np.unique(values, axis=0).shape[0] != count:
            raise ValueError(f"duplicate {name} coordinates")
    final = np.column_stack((np.tile(x_m, t_s.size), np.repeat(t_s, x_m.size)))
    combined = np.concatenate((train, validation, final))
    if np.unique(combined, axis=0).shape[0] != combined.shape[0]:
        raise ValueError("duplicate or overlapping partition coordinates")
    start, stop = options["final_window_s"]
    if (
        x_m.shape != (options["final_n_x"],)
        or t_s.shape != (options["final_n_t"],)
        or not np.isfinite(final).all()
        or np.any(x_m <= left)
        or np.any(x_m >= right)
        or not np.all(np.diff(x_m) > 0)
        or not np.all(np.diff(t_s) > 0)
        or t_s[0] != start
        or t_s[-1] != stop
    ):
        raise ValueError("final axes violate frozen partition membership")


def _paired_values(case: dict[str, Any], xt: NDArray[Any]) -> NDArray[np.float64]:
    # The public reference API returns a rectangle; only paired diagonal values
    # are saved. This bounded intermediate work is included in generation cost.
    solution = evaluate_reference(case, xt[:, 0], xt[:, 1])
    return np.column_stack((np.diag(solution.h_m), np.diag(solution.q_m2_s)))


def generate_labels(
    config_path: str | Path | dict[str, Any], run_id: str, root: Path = Path(".")
) -> Path:
    """Create a new analytical-label run; never generate final reference values."""
    root = Path(root).absolute()
    if isinstance(config_path, dict):
        config = copy.deepcopy(config_path)
        config_input: list[dict[str, Any]] = []
    else:
        source = _path(root, config_path)
        config = _strict_json(source)
        config_input = [{"path": str(source), "sha256": file_sha256(source)}]
    case_path = _path(root, config.get("case", "configs/cases/e1_wet.json"))
    case = load_config(case_path)
    if case["case_id"] != "E1" or case["reference"]["kind"] != "wet_bed":
        raise ValueError("the selected hybrid comparison requires analytical E1")
    options = _options(config, case)
    resolved = {
        "extension": "hybrid_analytical_labels_v1",
        "case": case,
        "labels": options,
        "request": {
            "seed": options["seed"],
            "device": "cpu",
            "dtype": "float64",
            "run_id": run_id,
        },
        "provenance": {
            "input_files": config_input
            + [{"path": str(case_path), "sha256": file_sha256(case_path)}]
        },
        "data_partitions": {
            "training": "solution labels, t in (0,0.35) s",
            "validation": "separate solution labels, t in [0.35,0.40) s",
            "final": "axes only; labels withheld until final evaluation",
        },
    }
    run = RunRecord.create(
        root / "runs",
        run_id,
        resolved,
        project_root=root,
        metadata={
            "stage": "hybrid_label_generation",
            "data_origin": "analytical_generated",
        },
    )
    with run:
        started = time.perf_counter()
        rng = np.random.Generator(np.random.PCG64(options["seed"]))
        train = _draw_coordinates(
            rng,
            options["training_count"],
            case["domain_m"],
            options["training_window_s"],
            exclude_lower=True,
        )
        validation = _draw_coordinates(
            rng,
            options["validation_count"],
            case["domain_m"],
            options["validation_window_s"],
            exclude_lower=False,
        )
        left, right = case["domain_m"]
        # Preserve the evaluator's exact arithmetic as well as the same centres.
        x_m = (
            left
            + (np.arange(options["final_n_x"], dtype=float) + 0.5)
            * (right - left)
            / options["final_n_x"]
        )
        t_s = np.linspace(*options["final_window_s"], options["final_n_t"])
        _validate_memberships(train, validation, x_m, t_s, options, case["domain_m"])
        coordinate_s = time.perf_counter() - started
        label_started = time.perf_counter()
        train_hq = _paired_values(case, train)
        validation_hq = _paired_values(case, validation)
        label_s = time.perf_counter() - label_started
        np.savez(run.path / "train_labels.npz", xt_m_s=train, h_q=train_hq)
        np.savez(
            run.path / "validation_labels.npz", xt_m_s=validation, h_q=validation_hq
        )
        np.savez(run.path / "final_axes.npz", x_m=x_m, t_s=t_s)
        partitions = {}
        for name, filename, coordinates, values, interval in (
            ("training", "train_labels.npz", train, train_hq, "(lower,upper)"),
            (
                "validation",
                "validation_labels.npz",
                validation,
                validation_hq,
                "[lower,upper)",
            ),
        ):
            partitions[name] = {
                "file": filename,
                "count": int(coordinates.shape[0]),
                "coordinate_sha256": array_sha256(coordinates),
                "value_sha256": array_sha256(values),
                "file_sha256": file_sha256(run.path / filename),
                "time_window_s": options[f"{name}_window_s"],
                "time_membership": interval,
                "coordinate_columns": ["x_m", "t_s"],
                "value_columns": ["h_m", "q_m2_s"],
            }
        final_coords = np.column_stack(
            (np.tile(x_m, t_s.size), np.repeat(t_s, x_m.size))
        )
        partitions["final"] = {
            "file": "final_axes.npz",
            "count": int(final_coords.shape[0]),
            "x_sha256": array_sha256(x_m),
            "t_sha256": array_sha256(t_s),
            "coordinate_sha256": array_sha256(final_coords),
            "file_sha256": file_sha256(run.path / "final_axes.npz"),
            "time_window_s": options["final_window_s"],
            "time_membership": "[lower,upper]",
            "reference_values_generated": False,
        }
        manifest = {
            "schema_version": 1,
            "extension": "hybrid_analytical_labels_v1",
            "run_id": run_id,
            "data_origin": "analytical_generated",
            "field_observations": False,
            "case": case,
            "selection": options,
            "units": {"x_m": "m", "t_s": "s", "h_m": "m", "q_m2_s": "m^2/s"},
            "scales": {
                **case["scales"],
                "Q_m2_s": case["scales"]["H_m"] * case["scales"]["Uc_m_s"],
            },
            "label_rng": {
                "algorithm": "numpy.PCG64",
                "seed": options["seed"],
                "independent_of_training_rng": True,
                "draw_order": "training x,t then validation x,t",
            },
            "source": {
                "kind": "idealized E1 wet-bed analytical reference",
                "module": "hydro_pinn.references",
                "source_sha256": file_sha256(
                    Path(__file__).parents[1] / "references.py"
                ),
                "generator_sha256": file_sha256(__file__),
                "case_sha256": file_sha256(case_path),
                "versioned_references": copy.deepcopy(REFERENCE_SOURCES),
                "numerical_uncertainty": {
                    "representation": (
                        "CPU float64 arithmetic and bracketed wet-star root"
                    ),
                    "root_absolute_tolerance_m": case["reference"]["root_tolerance_m"],
                    "measurement_uncertainty": None,
                    "measurement_uncertainty_reason": "generated analytical values",
                },
            },
            "partitions": partitions,
            "partition_checks": {
                "unique": True,
                "pairwise_disjoint": True,
                "within_declared_windows": True,
            },
            "coverage": {
                "label_training": "random interior coordinates below 0.35 s",
                "physics_collocation": "may cover the full PDE domain and 0<t<=0.5 s",
                "final_holdout": "solution-label holdout only; same physical case",
            },
            "cost_s": {
                "coordinate_generation_and_checks": coordinate_s,
                "analytical_label_evaluation": label_s,
                "through_array_save": time.perf_counter() - started,
            },
            "generation_workload": {
                "saved_label_count": int(train_hq.shape[0] + validation_hq.shape[0]),
                "reference_rectangle_intermediate_points": int(
                    train_hq.shape[0] ** 2 + validation_hq.shape[0] ** 2
                ),
                "final_reference_points": 0,
            },
        }
        write_json_atomic(run.path / "label_manifest.json", manifest)
        run.complete(
            stopping_reason="analytical labels and disjoint final axes saved",
            elapsed_timings_s=manifest["cost_s"],
            iterations=0,
            function_evaluations=2,
        )
    return run.path


def load_label_manifest(label_dir: str | Path) -> dict[str, Any]:
    """Read partition metadata without opening validation or final array files."""
    label_dir = Path(label_dir)
    status = _strict_json(label_dir / "status.json")
    if status.get("state") != "completed":
        raise ValueError("label run is not completed")
    manifest = _strict_json(label_dir / "label_manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("extension") != "hybrid_analytical_labels_v1"
        or manifest.get("field_observations") is not False
    ):
        raise ValueError("invalid hybrid analytical-label manifest")
    return manifest


def load_training_labels(label_dir: str | Path) -> dict[str, NDArray[np.float64]]:
    """Load only training arrays and verify their saved membership/value hashes."""
    label_dir = Path(label_dir)
    manifest = load_label_manifest(label_dir)
    partition = manifest["partitions"]["training"]
    path = label_dir / "train_labels.npz"
    if file_sha256(path) != partition["file_sha256"]:
        raise ValueError("training label file hash mismatch")
    with np.load(path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"xt_m_s", "h_q"}:
            raise ValueError("unexpected training label payload")
        result = {key: np.array(arrays[key], copy=True) for key in arrays.files}
    xt, hq = result["xt_m_s"], result["h_q"]
    if (
        xt.shape != (partition["count"], 2)
        or hq.shape != xt.shape
        or xt.dtype != np.float64
        or hq.dtype != np.float64
        or not np.isfinite(xt).all()
        or not np.isfinite(hq).all()
        or not np.all(hq[:, 0] > 0)
    ):
        raise ValueError("invalid physical training label arrays")
    if (
        array_sha256(xt) != partition["coordinate_sha256"]
        or array_sha256(hq) != partition["value_sha256"]
    ):
        raise ValueError("training label coordinate/value hash mismatch")
    return result


def verify_label_integrity(label_dir: str | Path) -> dict[str, Any]:
    """Read back all partition artifacts separately, for verification/evaluation."""
    label_dir = Path(label_dir)
    manifest = load_label_manifest(label_dir)
    train = load_training_labels(label_dir)
    for name in ("validation", "final"):
        partition = manifest["partitions"][name]
        if file_sha256(label_dir / partition["file"]) != partition["file_sha256"]:
            raise ValueError(f"{name} file hash mismatch")
    with np.load(label_dir / "validation_labels.npz", allow_pickle=False) as arrays:
        validation = np.array(arrays["xt_m_s"], copy=True)
        hq = np.array(arrays["h_q"], copy=True)
    with np.load(label_dir / "final_axes.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != {"x_m", "t_s"}:
            raise ValueError("final axes artifact contains reference values")
        x_m, t_s = (
            np.array(arrays["x_m"], copy=True),
            np.array(arrays["t_s"], copy=True),
        )
    part = manifest["partitions"]["validation"]
    if (
        array_sha256(validation) != part["coordinate_sha256"]
        or array_sha256(hq) != part["value_sha256"]
    ):
        raise ValueError("validation coordinate/value hash mismatch")
    final = manifest["partitions"]["final"]
    if array_sha256(x_m) != final["x_sha256"] or array_sha256(t_s) != final["t_sha256"]:
        raise ValueError("final axis hash mismatch")
    _validate_memberships(
        train["xt_m_s"],
        validation,
        x_m,
        t_s,
        manifest["selection"],
        manifest["case"]["domain_m"],
    )
    return {
        "unique": True,
        "pairwise_disjoint": True,
        "within_declared_windows": True,
        "all_partition_file_hashes_match": True,
        "final_reference_values_absent": True,
    }
