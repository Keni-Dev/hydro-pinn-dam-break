"""Frozen raw-query operators and guarded analysis for the genuine 2D PINN.

``run_raw_queries`` computes no reference error, balance, parity or region score.
The pilot and future full fit use its identical no-dedup query/I/O workload.
Quality analysis requires a separately completed 10,000-update full run.
"""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from hydro_pinn.physics import Scales

BATCH_SIZE = 20000
FIELD_NAMES = ("h_m", "qx_m2_s", "qy_m2_s")
ALL_FIELDS = (*FIELD_NAMES, "u_cell_ratio_m_s", "v_cell_ratio_m_s")
WALLS = ("left", "right", "bottom", "top")
NORMAL_COMPONENT = {"left": 1, "right": 1, "bottom": 2, "top": 2}
OUTWARD_SIGN = {"left": -1.0, "right": 1.0, "bottom": -1.0, "top": 1.0}
EXPECTED_COUNTS = {
    "Gauss3": 2880000,
    "Gauss5": 8000000,
    "volume": 1958400,
    "boundary": 801280,
    "total": 13639680,
}
EVALUATION_INPUT_PATHS = {
    "selected_reference": (
        "results/extensions/two_dimensional/reference_v01/ref_rect320"
        "_c0225_v01/common_reference.npz"
    ),
    "gradient_region": (
        "results/extensions/two_dimensional/reference_v01/r6_analysis"
        "_v01/selected_reference_gradient_region.npz"
    ),
    "sensitivity": (
        "results/extensions/two_dimensional/reference_v01/r6_analysis"
        "_v01/signed_differences_and_sensitivity.npz"
    ),
    "reference_norms": (
        "results/extensions/two_dimensional/reference_v01/r6_analysis_v01/norms.json"
    ),
}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, value: dict) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def _save(path: Path, arrays: dict) -> None:
    with path.open("xb") as handle:
        np.savez_compressed(handle, **arrays)


def _load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def _schema(arrays: dict) -> dict:
    return {
        key: {"dtype": str(value.dtype), "shape": list(value.shape)}
        for key, value in arrays.items()
    }


def _scales(value: Scales | dict) -> Scales:
    return value if isinstance(value, Scales) else Scales(**value)


def _scale_dict(scales: Scales) -> dict:
    return {key: getattr(scales, key) for key in ("H_m", "Lc_m", "Uc_m_s", "Tc_s")}


def midpoint_axes(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Square SI midpoint geometry; weights are area, not point-mean weights."""
    edges = np.linspace(-5.0, 5.0, n + 1, dtype=np.float64)
    centres = (edges[:-1] + edges[1:]) / 2
    return centres, edges, np.full((n, n), 100.0 / n**2, dtype=np.float64)


def output_times() -> np.ndarray:
    return 0.005 * np.arange(51, dtype=np.float64)


def mass_time_schedule(nodes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact frozen replacements, with no near-coincident union/interpolation."""
    if nodes not in (501, 1001):
        raise ValueError("mass schedule requires 501 or 1001 nodes")
    times = 0.25 * np.arange(nodes, dtype=np.float64) / (nodes - 1)
    indices = np.arange(51, dtype=np.int64) * ((nodes - 1) // 50)
    reported = output_times()
    times[indices] = reported
    if not np.all(np.diff(times) > 0) or not np.array_equal(times[indices], reported):
        raise ValueError("mass replacement times are not strictly ordered/exact")
    if times[0] != 0 or times[-1] != 0.25:
        raise ValueError("mass schedule endpoints differ")
    return times, indices, reported


def execution_representations(
    points_si64: np.ndarray, scales: Scales | dict
) -> dict[str, np.ndarray]:
    """Use the model's actual SIcast32-then-Torch-division sequence."""
    scales = _scales(scales)
    points = np.asarray(points_si64, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("physical query points require finite float64(N,3)")
    physical32 = torch.as_tensor(points, dtype=torch.float32, device="cpu")
    model32 = torch.stack(
        (
            physical32[:, 0] / scales.Lc_m,
            physical32[:, 1] / scales.Lc_m,
            physical32[:, 2] / scales.Tc_s,
        ),
        dim=1,
    )
    return {
        "si64": points,
        "scaled64": points / np.array([scales.Lc_m, scales.Lc_m, scales.Tc_s]),
        "si32_physical": physical32.numpy(),
        "model_nd32": model32.numpy(),
    }


def _axis_representations(name: str, axis: np.ndarray, scale: float) -> dict:
    physical32 = torch.as_tensor(axis, dtype=torch.float32, device="cpu")
    return {
        f"{name}_si64": axis,
        f"{name}_scaled64": axis / scale,
        f"{name}_si32_physical": physical32.numpy(),
        f"{name}_model_nd32": (physical32 / scale).numpy(),
    }


def _hashers(count: int) -> dict:
    return {
        key: hashlib.sha256(f"{key}|rows={count}|x,y,t|C|canonical-zero".encode())
        for key in ("si64", "scaled64", "si32_physical", "model_nd32")
    }


def _update_hashers(hashers: dict, values: dict) -> None:
    for key, h in hashers.items():
        # Signed zero is canonicalized only for membership/hash equality.
        # Actual physical model inputs are otherwise unchanged.
        a = np.ascontiguousarray(values[key]).copy()
        a[a == 0] = 0
        h.update(a.tobytes(order="C"))


def _stream_count(operator: dict, category: str) -> int:
    if category == "gauss":
        return (
            len(operator["t_s"])
            * operator["x_nodes_m"].size
            * operator["y_nodes_m"].size
        )
    if category == "volume":
        return len(operator["output_t_s"]) * len(operator["x_m"]) * len(operator["y_m"])
    if category == "edge":
        return len(operator["edge_t_s"]) * len(operator["tangent_m"])
    raise ValueError(f"unknown query category: {category}")


def iter_query_points(
    operator: dict, category: str, wall: str | None = None, batch_size: int = BATCH_SIZE
) -> Iterator[dict]:
    """Lazy C-order time/y/x/Gauss-y/Gauss-x or wall/time/tangent batches.

    No coordinate reuse occurs between rules, volume passes, operators or walls.
    Returned float64 SI coordinates are cast32 only at the physical-state call.
    """
    if isinstance(batch_size, bool) or batch_size < 1:
        raise ValueError("query batch size must be positive")
    if category == "edge" and wall not in WALLS:
        raise ValueError("edge query requires a named wall")
    count = _stream_count(operator, category)
    for start in range(0, count, batch_size):
        index = np.arange(start, min(start + batch_size, count), dtype=np.int64)
        if category == "gauss":
            q = len(operator["legendre_nodes"])
            nx, ny = operator["x_nodes_m"].shape[0], operator["y_nodes_m"].shape[0]
            ixg, iyg = index % q, (index // q) % q
            ix, iy = (index // (q * q)) % nx, (index // (q * q * nx)) % ny
            it = index // (q * q * nx * ny)
            points = np.column_stack(
                (
                    operator["x_nodes_m"][ix, ixg],
                    operator["y_nodes_m"][iy, iyg],
                    operator["t_s"][it],
                )
            )
            weights = (
                operator["legendre_weights"][ixg]
                * operator["legendre_weights"][iyg]
                / 4
            )
            yield {
                "points_si64": points,
                "flat_indices": index,
                "cell_indices": index // (q * q),
                "point_weights": weights,
            }
        elif category == "volume":
            nx, ny = len(operator["x_m"]), len(operator["y_m"])
            ix, iy, it = index % nx, (index // nx) % ny, index // (nx * ny)
            yield {
                "points_si64": np.column_stack(
                    (
                        operator["x_m"][ix],
                        operator["y_m"][iy],
                        operator["output_t_s"][it],
                    )
                ),
                "flat_indices": index,
            }
        else:
            n = len(operator["tangent_m"])
            tangent, times = (
                operator["tangent_m"][index % n],
                operator["edge_t_s"][index // n],
            )
            fixed = np.full(len(index), -5.0 if wall in ("left", "bottom") else 5.0)
            points = np.column_stack(
                (
                    fixed if wall in ("left", "right") else tangent,
                    tangent if wall in ("left", "right") else fixed,
                    times,
                )
            )
            yield {"points_si64": points, "flat_indices": index}


def gauss_cell_means(values: np.ndarray, legendre_weights: np.ndarray) -> np.ndarray:
    """Pure small manufactured-check helper: values(...,gy,gx,component)."""
    values, weights = (
        np.asarray(values, dtype=np.float64),
        np.asarray(legendre_weights, dtype=np.float64),
    )
    if values.shape[-3:-1] != (len(weights), len(weights)):
        raise ValueError("Gauss value shape does not match both rule axes")
    return np.einsum("...ijc,i,j->...c", values, weights, weights) / 4


def accumulate_gauss_batch(
    accumulator: np.ndarray,
    cell_indices: np.ndarray,
    states64: np.ndarray,
    point_weights: np.ndarray,
) -> None:
    """Accumulate all conserved states in frozen point order, including split cells."""
    np.add.at(accumulator, cell_indices, states64 * point_weights[:, None])


def _gauss_operator(q: int, scales: Scales) -> dict:
    centres, edges, areas = midpoint_axes(80)
    nodes, weights = np.polynomial.legendre.leggauss(q)
    physical = centres[:, None] + 0.5 * np.diff(edges)[:, None] * nodes[None, :]
    times = output_times()[1:]
    arrays = {
        "x_m": centres,
        "y_m": centres.copy(),
        "x_edges_m": edges,
        "y_edges_m": edges.copy(),
        "cell_area_m2": areas,
        "t_s": times,
        "legendre_nodes": nodes,
        "legendre_weights": weights,
        "x_nodes_m": physical,
        "y_nodes_m": physical.copy(),
    }
    arrays.update(_axis_representations("x_nodes", physical, scales.Lc_m))
    arrays.update(_axis_representations("y_nodes", physical.copy(), scales.Lc_m))
    arrays.update(_axis_representations("t", times, scales.Tc_s))
    return arrays


def _mass_operator(n: int, nodes: int, scales: Scales) -> dict:
    centres, edges, areas = midpoint_axes(n)
    edge_times, indices, reported = mass_time_schedule(nodes)
    arrays = {
        "x_m": centres,
        "y_m": centres.copy(),
        "x_edges_m": edges,
        "y_edges_m": edges.copy(),
        "cell_area_m2": areas,
        "output_t_s": reported,
        "edge_t_s": edge_times,
        "output_indices": indices,
        "tangent_m": centres.copy(),
        "length_weights_m": np.full(n, 10.0 / n, dtype=np.float64),
        "wall_normal_coordinate_m": np.array([-5.0, 5.0, -5.0, 5.0]),
        "wall_outward_sign": np.array([-1.0, 1.0, -1.0, 1.0]),
        "wall_normal_component": np.array([1, 1, 2, 2], dtype=np.int64),
        "trapezoid_interval_s": np.diff(edge_times),
    }
    prefix_weights = np.zeros((51, nodes), dtype=np.float64)
    for row, stop in enumerate(indices):
        half_widths = np.diff(edge_times)[:stop] / 2
        prefix_weights[row, :stop] += half_widths
        prefix_weights[row, 1 : stop + 1] += half_widths
    arrays["output_prefix_trapezoid_weights_s"] = prefix_weights
    for name, a, scale in (
        ("x", centres, scales.Lc_m),
        ("y", centres.copy(), scales.Lc_m),
        ("tangent", centres.copy(), scales.Lc_m),
        ("output_t", reported, scales.Tc_s),
        ("edge_t", edge_times, scales.Tc_s),
        ("wall_normal", arrays["wall_normal_coordinate_m"], scales.Lc_m),
    ):
        arrays.update(_axis_representations(name, a, scale))
    for name in ("edge_t_si32_physical", "edge_t_model_nd32"):
        if not np.all(np.diff(arrays[name]) > 0):
            raise ValueError("CPU32 mass schedule is not strictly increasing")
    if not np.array_equal(
        arrays["edge_t_model_nd32"][indices], arrays["output_t_model_nd32"]
    ):
        raise ValueError("CPU32 output nodes differ from replaced mass time nodes")
    return arrays


def _output_schema(kind: str, operator: dict) -> dict:
    if kind == "gauss":
        arrays = {
            key: operator[key]
            for key in ("x_m", "y_m", "x_edges_m", "y_edges_m", "cell_area_m2", "t_s")
        }
        schema = _schema(arrays)
        schema["state_mean_h_qx_qy"] = {"dtype": "float64", "shape": [50, 80, 80, 3]}
        return schema
    n, nt = len(operator["x_m"]), len(operator["edge_t_s"])
    if kind == "volume":
        schema = _schema(
            {
                key: operator[key]
                for key in ("x_m", "y_m", "x_edges_m", "y_edges_m", "cell_area_m2")
            }
        )
        schema["t_s"] = {"dtype": "float64", "shape": [51]}
        schema["h_m"] = {"dtype": "float64", "shape": [51, n, n]}
        return schema
    schema = _schema(
        {
            "t_s": operator["edge_t_s"],
            "tangent_m": operator["tangent_m"],
            "length_weights_m": operator["length_weights_m"],
            "wall_normal_coordinate_m": operator["wall_normal_coordinate_m"],
            "wall_outward_sign": operator["wall_outward_sign"],
            "wall_normal_component": operator["wall_normal_component"],
            "output_indices": operator["output_indices"],
            "output_t_s": operator["output_t_s"],
        }
    )
    for key in ("left_qx_m2_s", "right_qx_m2_s", "bottom_qy_m2_s", "top_qy_m2_s"):
        schema[key] = {"dtype": "float64", "shape": [nt, n]}
    return schema


def evaluation_definitions() -> dict:
    return {
        "raw_edge_convention": (
            "Wall order left/right/bottom/top; saved left_qx/right_qx and "
            "bottom_qy/top_qy retain signed global coordinate components. "
            "Saved coordinates, selectors and outward signs reconstruct "
            "the normals; outward signs are applied only during mass analysis."
        ),
        "Gauss": (
            "Mean=1/4 sum(wy*wx*physical conserved state), float64 accumu"
            "lation; no second area factor; no mean point velocities."
        ),
        "query_order": (
            "Gauss3 then5; each base/spatial/temporal volume then left/ri"
            "ght/bottom/top; time,y,x,Gauss-y,Gauss-x or wall,time,tangen"
            "t"
        ),
        "query_execution": (
            "CPUfloat32 SIcast input to physical_state, all3 states compu"
            "ted; predictions promoted64 before any SI weight/reduction/s"
            "torage; batch20000, no dedup"
        ),
        "hash_representation": (
            "C-contiguous rows ordered x,y,t; dtype-preserving bytes with"
            " signed zero canonicalized; per-stream SHA prefix names repr"
            "esentation/row count/order"
        ),
        "regions": (
            "whole, frozen dilated reference-gradient mask, complement; u"
            "se every cell including low neural depth/large error"
        ),
        "per_time": (
            "A_t=sum_R A; MAE=sum_R(A abs(error))/A_t; RMSE=sqrt(sum_R(A error²)/A_t)"
        ),
        "pooled_area_means": (
            "mean_t(MAE_t),sqrt(mean_t(RMSE_t²)); equal positive-time wei"
            "ghts, no omitted empty times"
        ),
        "integral_norms": (
            "L1=sum_R(A abs(error)); L2=sqrt(sum_R(A error²)); pooled L1="
            "mean_t L1,pooled L2=sqrt(mean_t L2²)"
        ),
        "pooled_integral_relative_L2_percent": (
            "100sqrt(sum_t,R(A error²)/sum_t,R(A ref²)); null zero refere"
            "nce or any empty region, no epsilon; not a ratio of area-nor"
            "malized pooled RMS when A_t varies"
        ),
        "velocities": (
            "u=qxbar/hbar,v=qybar/hbar for neural and reference allcells;"
            " cell ratios, no point-velocity or mean-of-point-velocity cl"
            "aim"
        ),
        "sensitivity": (
            "Gauss5-minus3 operator discrepancy; frozen reference Dspace,"
            "Dtime,Dsum conserved fields only, sensitivity indicators not"
            " certified bounds or velocity indicators"
        ),
        "parity": (
            "h evenx/eveny; qx and u oddx/eveny; qy and v evenx/oddy; sav"
            "e maxabs/RMS SI defects, no x-y interchange expectation"
        ),
        "mass": (
            "M=sum(h_midpoint*area), B=M-11.8+Iout, initialbiasB0, additi"
            "onal=B-B0; -leftqx,+rightqx,-bottomqy,+topqy"
        ),
        "mass_absolute": (
            "Each Qabs=sum(length*abs(normal q)), cumulative trapezoid; s"
            "igned and absolute all4 separate, maxabs normalq saved"
        ),
        "mass_sensitivity": (
            "spatial and temporal deltaM,deltaIout,deltaB separate; spati"
            "al changes volume+edge spatial rules, temporal repeats base "
            "volume queries and changes edge time nodes; indicators not b"
            "ounds"
        ),
        "maximum_qualification": (
            "All maxima are over saved samples, not continuous extrema; n"
            "o momentum budget"
        ),
        "full_only": (
            "Quality metrics require completed full10000update status/his"
            "tory/checkpoint readback; pilot raw queries cannot call anal"
            "ysis"
        ),
    }


def prepare_operators(cfg: dict, directory: str | Path, scales: Scales | dict) -> dict:
    """Save/freeze every operator, execution-axis representation and stream hash.

    This setup generates/hashes coordinates without a model call. The caller
    freezes the returned files with initial tensors and experiment memberships.
    """
    directory, scales = Path(directory), _scales(scales)
    query = cfg["query"]
    expected = {
        "batch_size": BATCH_SIZE,
        "common_n": 80,
        "gauss_orders": [3, 5],
        "positive_times": 50,
        "positive_dt_s": 0.005,
        "mass": [
            {"id": "base", "n": 80, "time_nodes": 501},
            {"id": "spatial", "n": 160, "time_nodes": 501},
            {"id": "temporal", "n": 80, "time_nodes": 1001},
        ],
    }
    if query != expected:
        raise ValueError("query configuration differs from frozen no-dedup v01")
    if _scale_dict(scales) != {
        "H_m": 1.0,
        "Lc_m": 10.0,
        "Uc_m_s": float(np.sqrt(9.81)),
        "Tc_s": 10.0 / float(np.sqrt(9.81)),
    }:
        raise ValueError("query scales differ from H1/L10 gravity identities")
    directory.mkdir(parents=True, exist_ok=True)
    entries, files = [], {}
    for q in (3, 5):
        entries.append(
            {
                "id": f"gauss{q}",
                "kind": "gauss",
                "file": f"gauss{q}_operator.npz",
                "operator": _gauss_operator(q, scales),
            }
        )
    for spec in query["mass"]:
        name = f"mass_{spec['id']}"
        entries.append(
            {
                "id": name,
                "kind": "mass",
                "file": f"{name}_operator.npz",
                "operator": _mass_operator(spec["n"], spec["time_nodes"], scales),
            }
        )
    manifest_entries, counts = [], {key: 0 for key in EXPECTED_COUNTS}
    for entry in entries:
        operator, kind = entry["operator"], entry["kind"]
        _save(directory / entry["file"], operator)
        files[entry["file"]] = _sha(directory / entry["file"])
        streams = (
            [("gauss", None)]
            if kind == "gauss"
            else [("volume", None), *[("edge", wall) for wall in WALLS]]
        )
        stream_records = []
        for category, wall in streams:
            count = _stream_count(operator, category)
            h, actual = _hashers(count), 0
            for batch in iter_query_points(operator, category, wall):
                representations = execution_representations(
                    batch["points_si64"], scales
                )
                _update_hashers(h, representations)
                actual += len(batch["points_si64"])
            if actual != count:
                raise RuntimeError("prepared query stream count mismatch")
            group = (
                f"Gauss{entry['id'][-1]}"
                if category == "gauss"
                else ("volume" if category == "volume" else "boundary")
            )
            counts[group] += count
            counts["total"] += count
            stream_records.append(
                {
                    "category": category,
                    "wall": wall,
                    "count": count,
                    "batch_size": BATCH_SIZE,
                    "batch_count": (count + BATCH_SIZE - 1) // BATCH_SIZE,
                    "last_batch_count": (count - 1) % BATCH_SIZE + 1,
                    "input_sha256": {
                        key: value.hexdigest() for key, value in h.items()
                    },
                }
            )
        outputs = (
            {f"{entry['id']}_means.npz": _output_schema("gauss", operator)}
            if kind == "gauss"
            else {
                f"{entry['id']}_volume.npz": _output_schema("volume", operator),
                f"{entry['id']}_edges.npz": _output_schema("edge", operator),
            }
        )
        manifest_entries.append(
            {
                "id": entry["id"],
                "kind": kind,
                "file": entry["file"],
                "operator_schema": _schema(operator),
                "streams": stream_records,
                "output_schemas": outputs,
            }
        )
    if counts != EXPECTED_COUNTS:
        raise RuntimeError("actual frozen query count differs from13,639,680")
    manifest = {
        "schema_version": 1,
        "query": query,
        "scales": _scale_dict(scales),
        "files": files,
        "operators": manifest_entries,
        "query_counts": counts,
        "component_order": list(FIELD_NAMES),
        "definitions": evaluation_definitions(),
        "reference_archive_provenance_sha256": (
            "5346679a435eebb1198d69ebc3208d92b250b725433fc5a7e2529b9b205809b5"
        ),
    }
    _write(directory / "operators_manifest.json", manifest)
    return {
        **manifest,
        "files": {
            **files,
            "operators_manifest.json": _sha(directory / "operators_manifest.json"),
        },
    }


def final_gauss_axes(directory: str | Path) -> Iterator[tuple]:
    """Factorized strict-membership support, including each rule's common centres."""
    directory = Path(directory)
    for q in (3, 5):
        a = _load(directory / f"gauss{q}_operator.npz")
        yield q, a["x_nodes_m"].ravel(), a["y_nodes_m"].ravel(), a["t_s"]


def _model_identity(model) -> str:
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes(order="C"))
    return h.hexdigest()


def _validated_operators(directory: Path) -> dict:
    manifest = _read(directory / "operators_manifest.json")
    if (
        manifest["query_counts"] != EXPECTED_COUNTS
        or manifest["query"]["batch_size"] != BATCH_SIZE
    ):
        raise ValueError("raw-query operator count/batch contract mismatch")
    for name, expected in manifest["files"].items():
        if Path(name).name != name or _sha(directory / name) != expected:
            raise ValueError(f"frozen query operator changed: {name}")
    return manifest


def _query_stream(model, operator: dict, spec: dict, scales: Scales, callback) -> dict:
    count = spec["count"]
    h, actual = _hashers(count), 0
    times = {
        "generation": 0.0,
        "input_conversion_and_hashing": 0.0,
        "forward_and_float64_transfer": 0.0,
        "reduction": 0.0,
    }
    batches = iter_query_points(operator, spec["category"], spec["wall"], BATCH_SIZE)
    while True:
        now = time.perf_counter()
        try:
            batch = next(batches)
        except StopIteration:
            times["generation"] += time.perf_counter() - now
            break
        times["generation"] += time.perf_counter() - now
        now = time.perf_counter()
        representations = execution_representations(batch["points_si64"], scales)
        _update_hashers(h, representations)
        points32 = torch.from_numpy(representations["si32_physical"])
        times["input_conversion_and_hashing"] += time.perf_counter() - now
        now = time.perf_counter()
        with torch.no_grad():
            state = model.physical_state(points32)
        if (
            state.shape != (len(points32), 3)
            or state.dtype != torch.float32
            or state.device.type != "cpu"
            or not bool(torch.isfinite(state).all())
            or bool((state[:, 0] <= 0).any())
        ):
            raise RuntimeError("raw query returned invalid CPU32 conserved states")
        states64 = state.detach().numpy().astype(np.float64)
        times["forward_and_float64_transfer"] += time.perf_counter() - now
        now = time.perf_counter()
        callback(batch, states64)
        times["reduction"] += time.perf_counter() - now
        actual += len(state)
    observed = {key: value.hexdigest() for key, value in h.items()}
    if actual != count or observed != spec["input_sha256"]:
        raise RuntimeError("actual raw query membership/hash/count differs from freeze")
    return {
        "category": spec["category"],
        "wall": spec["wall"],
        "actual_states": actual,
        "input_sha256": observed,
        "cost_s": times,
        "batch_count": spec["batch_count"],
        "last_batch_count": spec["last_batch_count"],
    }


def _assert_schema(arrays: dict, expected: dict) -> None:
    if _schema(arrays) != expected:
        raise RuntimeError("raw output key/dtype/shape schema differs from freeze")
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise RuntimeError("nonfinite saved raw query output")


def run_raw_queries(model, protocol_dir: str | Path, output_dir: str | Path) -> dict:
    """ONE full no-dedup pass. This function performs no quality analysis."""
    start = time.perf_counter()
    directory, output = Path(protocol_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError("raw-query output directory must be new")
    manifest = _validated_operators(directory)
    scales = _scales(manifest["scales"])
    p = next(model.parameters())
    if (
        p.dtype != torch.float32
        or p.device.type != "cpu"
        or _scale_dict(model.scales) != _scale_dict(scales)
    ):
        raise ValueError("raw queries require matching CPU32 H1/L10 model")
    output.mkdir(parents=True, exist_ok=False)
    all_streams, output_files, category_counts = (
        [],
        {},
        {key: 0 for key in EXPECTED_COUNTS},
    )
    operator_costs = {}
    io_cost = 0.0
    for entry in manifest["operators"]:
        operator_start = time.perf_counter()
        a = _load(directory / entry["file"])
        if _schema(a) != entry["operator_schema"]:
            raise ValueError("saved operator schema differs from freeze")
        if entry["kind"] == "gauss":
            result = np.zeros((50 * 80 * 80, 3), dtype=np.float64)
            receipt = _query_stream(
                model,
                a,
                entry["streams"][0],
                scales,
                lambda batch, state, target=result: accumulate_gauss_batch(
                    target, batch["cell_indices"], state, batch["point_weights"]
                ),
            )
            all_streams.append({"operator": entry["id"], **receipt})
            category_counts[f"Gauss{entry['id'][-1]}"] += receipt["actual_states"]
            arrays = {
                key: a[key]
                for key in (
                    "x_m",
                    "y_m",
                    "x_edges_m",
                    "y_edges_m",
                    "cell_area_m2",
                    "t_s",
                )
            }
            arrays["state_mean_h_qx_qy"] = result.reshape(50, 80, 80, 3)
            filename = f"{entry['id']}_means.npz"
            _assert_schema(arrays, entry["output_schemas"][filename])
            now = time.perf_counter()
            _save(output / filename, arrays)
            io_cost += time.perf_counter() - now
            output_files[filename] = _sha(output / filename)
        else:
            n, nt = len(a["x_m"]), len(a["edge_t_s"])
            depth = np.empty(51 * n * n, dtype=np.float64)
            receipt = _query_stream(
                model,
                a,
                entry["streams"][0],
                scales,
                lambda batch, state, target=depth: target.__setitem__(
                    batch["flat_indices"], state[:, 0]
                ),
            )
            all_streams.append({"operator": entry["id"], **receipt})
            category_counts["volume"] += receipt["actual_states"]
            arrays = {
                key: a[key]
                for key in ("x_m", "y_m", "x_edges_m", "y_edges_m", "cell_area_m2")
            }
            arrays.update(t_s=a["output_t_s"], h_m=depth.reshape(51, n, n))
            filename = f"{entry['id']}_volume.npz"
            _assert_schema(arrays, entry["output_schemas"][filename])
            now = time.perf_counter()
            _save(output / filename, arrays)
            io_cost += time.perf_counter() - now
            output_files[filename] = _sha(output / filename)
            del depth, arrays
            edges = {
                "t_s": a["edge_t_s"],
                "tangent_m": a["tangent_m"],
                "length_weights_m": a["length_weights_m"],
                "wall_normal_coordinate_m": a["wall_normal_coordinate_m"],
                "wall_outward_sign": a["wall_outward_sign"],
                "wall_normal_component": a["wall_normal_component"],
                "output_indices": a["output_indices"],
                "output_t_s": a["output_t_s"],
            }
            for spec in entry["streams"][1:]:
                wall = spec["wall"]
                normal = np.empty(nt * n, dtype=np.float64)
                component = NORMAL_COMPONENT[wall]
                receipt = _query_stream(
                    model,
                    a,
                    spec,
                    scales,
                    lambda batch, state, target=normal, c=component: target.__setitem__(
                        batch["flat_indices"], state[:, c]
                    ),
                )
                all_streams.append({"operator": entry["id"], **receipt})
                category_counts["boundary"] += receipt["actual_states"]
                edges[f"{wall}_{'qx' if component == 1 else 'qy'}_m2_s"] = (
                    normal.reshape(nt, n)
                )
            filename = f"{entry['id']}_edges.npz"
            _assert_schema(edges, entry["output_schemas"][filename])
            now = time.perf_counter()
            _save(output / filename, edges)
            io_cost += time.perf_counter() - now
            output_files[filename] = _sha(output / filename)
        operator_costs[entry["id"]] = time.perf_counter() - operator_start
    category_counts["total"] = sum(
        v for key, v in category_counts.items() if key != "total"
    )
    if category_counts != EXPECTED_COUNTS:
        raise RuntimeError("actual complete raw query workload count mismatch")
    scientific_bytes = sum((output / name).stat().st_size for name in output_files)
    report = {
        "state": "completed_raw_queries_no_quality_analysis",
        "actual_query_counts": category_counts,
        "actual_query_count": category_counts["total"],
        "expected_query_count": EXPECTED_COUNTS["total"],
        "streams": all_streams,
        "output_sha256": output_files,
        "operators_manifest_sha256": _sha(directory / "operators_manifest.json"),
        "model_state_sha256": _model_identity(model),
        "quality_metrics_computed": False,
        "cost_s": {"compressed_output_io": io_cost},
        "operator_cost_s": operator_costs,
        "raw_output_bytes": scientific_bytes,
        "output_bytes": scientific_bytes,
        "byte_scope": (
            "eight compressed scientific NPZ files; JSON receipt bytes ex"
            "cluded and counted by experiment outer inventory"
        ),
        "wall_s_before_receipt": time.perf_counter() - start,
        "peak_process_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if platform.system() == "Darwin" else 1024)
        ),
        "memory_limit": (
            "process-lifetime peak; instantaneous allocator and system pr"
            "essure unmeasured"
        ),
    }
    _write(output / "raw_query_report.json", report)
    return report


def trapezoid_prefix(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    values, times = (
        np.asarray(values, dtype=np.float64),
        np.asarray(times, dtype=np.float64),
    )
    if len(values) != len(times) or not np.all(np.diff(times) > 0):
        raise ValueError("prefix quadrature requires aligned strictly increasing times")
    widths = np.diff(times).reshape((-1,) + (1,) * (values.ndim - 1))
    increments = 0.5 * (values[1:] + values[:-1]) * widths
    return np.concatenate(
        (np.zeros_like(values[:1]), np.cumsum(increments, axis=0)), axis=0
    )


def outward_mass_rates(normal_fields: dict, length_weights: np.ndarray) -> dict:
    weights = np.asarray(length_weights, dtype=np.float64)
    return {
        "signed": {
            wall: OUTWARD_SIGN[wall] * np.sum(normal_fields[wall] * weights, axis=1)
            for wall in WALLS
        },
        "absolute": {
            wall: np.sum(np.abs(normal_fields[wall]) * weights, axis=1)
            for wall in WALLS
        },
        "max_abs_normal": {
            wall: np.max(np.abs(normal_fields[wall]), axis=1) for wall in WALLS
        },
    }


def region_metrics(
    predicted: np.ndarray,
    reference: np.ndarray,
    cell_areas: np.ndarray,
    mask: np.ndarray,
    t_s: np.ndarray,
    field_names=None,
) -> dict:
    """Changing-region area means and separately named pooled integral ratios."""
    p, r, a, m = (
        np.asarray(predicted),
        np.asarray(reference),
        np.asarray(cell_areas),
        np.asarray(mask, dtype=bool),
    )
    names = tuple(field_names or (ALL_FIELDS if p.shape[-1] == 5 else FIELD_NAMES))
    if (
        p.shape != r.shape
        or p.ndim != 4
        or p.shape[-1] != len(names)
        or a.shape != p.shape[1:3]
        or m.shape != p.shape[:3]
        or len(t_s) != len(p)
        or np.any(a <= 0)
        or not all(np.isfinite(v).all() for v in (p, r, a))
    ):
        raise ValueError(
            "regional metrics require aligned finite fields/positive areas"
        )
    w = m[..., None] * a[None, ..., None]
    area = np.sum(m * a[None], axis=(1, 2))
    error = p - r
    l1, e2, r2 = [np.sum(w * v, axis=(1, 2)) for v in (np.abs(error), error**2, r**2)]

    def row(index: int, field: int) -> dict:
        if area[index] == 0:
            return {
                "area_mean_MAE": None,
                "area_mean_RMSE": None,
                "integral_L1": None,
                "integral_L2": None,
                "relative_L2_percent": None,
                "reason": "empty_region",
                "relative_reason": "empty_region",
            }
        return {
            "area_mean_MAE": float(l1[index, field] / area[index]),
            "area_mean_RMSE": float(np.sqrt(e2[index, field] / area[index])),
            "integral_L1": float(l1[index, field]),
            "integral_L2": float(np.sqrt(e2[index, field])),
            "relative_L2_percent": float(
                100 * np.sqrt(e2[index, field] / r2[index, field])
            )
            if r2[index, field] > 0
            else None,
            "reason": None,
            "relative_reason": None if r2[index, field] > 0 else "zero_reference_norm",
        }

    per_time = [
        {
            "t_s": float(t_s[i]),
            "region_area_m2": float(area[i]),
            "fields": {name: row(i, j) for j, name in enumerate(names)},
        }
        for i in range(len(p))
    ]
    pooled = {}
    for j, name in enumerate(names):
        if np.any(area == 0):
            pooled[name] = {
                "area_mean_MAE": None,
                "area_mean_RMSE": None,
                "integral_L1": None,
                "integral_L2": None,
                "pooled_integral_relative_L2_percent": None,
                "reason": "empty_region_at_one_or_more_times_no_omission",
                "relative_reason": "empty_region_at_one_or_more_times_no_omission",
            }
        else:
            denominator = float(np.sum(r2[:, j]))
            pooled[name] = {
                "area_mean_MAE": float(np.mean(l1[:, j] / area)),
                "area_mean_RMSE": float(np.sqrt(np.mean(e2[:, j] / area))),
                "integral_L1": float(np.mean(l1[:, j])),
                "integral_L2": float(np.sqrt(np.mean(e2[:, j]))),
                "pooled_integral_relative_L2_percent": float(
                    100 * np.sqrt(np.sum(e2[:, j]) / denominator)
                )
                if denominator > 0
                else None,
                "relative_reason": None if denominator > 0 else "zero_reference_norm",
                "reason": None,
            }
    return {
        "per_time": per_time,
        "pooled_equal_time": pooled,
        "empty_region_times": np.flatnonzero(area == 0).tolist(),
        "pooled_relative_definition": (
            "ratio of summed regional area-weighted squared integrals, no"
            "t ratio of time-averaged area-normalized RMS values"
        ),
        "units": {
            name: {
                "area_mean": "m"
                if name == "h_m"
                else ("m^2/s" if name in FIELD_NAMES else "m/s"),
                "integral_L1": "m^3"
                if name == "h_m"
                else ("m^4/s" if name in FIELD_NAMES else "m^3/s"),
                "integral_L2": "m^2"
                if name == "h_m"
                else ("m^3/s" if name in FIELD_NAMES else "m^2/s"),
            }
            for name in names
        },
    }


def _with_cell_ratios(states: np.ndarray) -> np.ndarray:
    if np.any(states[..., 0] <= 0):
        raise ValueError("cell ratios require positive mean depth; no cell exclusion")
    return np.concatenate((states, states[..., 1:] / states[..., :1]), axis=-1)


def parity_metrics(states: np.ndarray, times: np.ndarray) -> dict:
    values = _with_cell_ratios(states)
    result = {}
    for i, name in enumerate(ALL_FIELDS):
        sx, sy = (1, 1) if i == 0 else ((-1, 1) if i in (1, 3) else (1, -1))
        a = values[..., i]
        result[name] = {}
        for axis, sign, reflected in (
            ("x", sx, a[:, :, ::-1]),
            ("y", sy, a[:, ::-1, :]),
        ):
            defect = a - sign * reflected
            result[name][axis] = {
                "expected_reflection_sign": sign,
                "per_time_max_absolute": np.max(np.abs(defect), axis=(1, 2)).tolist(),
                "per_time_RMS": np.sqrt(np.mean(defect**2, axis=(1, 2))).tolist(),
                "sampled_max_absolute": float(np.max(np.abs(defect))),
                "RMS_over_time_and_cells": float(np.sqrt(np.mean(defect**2))),
            }
    return {"t_s": times.tolist(), "fields": result, "axis_exchange_expected": False}


def _full_context(run: Path, root: Path) -> tuple[dict, dict, object]:
    from hydro_pinn.extensions.experiment_2d import (
        load_checkpoint,
        verify_protocol,
        verify_training_readback,
    )

    summary = _read(run / "summary.json")
    if (
        summary.get("mode") != "full"
        or summary.get("actual_updates") != 10000
        or _read(run / "status.json").get("state") != "completed"
    ):
        raise ValueError(
            "quality analysis requires completed full10000updates; pilots refused"
        )
    protocol = verify_protocol(run / "protocol.json", root)
    verify_training_readback(run, protocol, summary)
    if _sha(run / "checkpoint.pt") != summary["checkpoint_sha256"]:
        raise ValueError("selected checkpoint bytes changed")
    return protocol, summary, load_checkpoint(run / "checkpoint.pt", protocol)


def _mass_analysis(raw: Path, output: Path, name: str) -> dict:
    volume, edges = (
        _load(raw / f"mass_{name}_volume.npz"),
        _load(raw / f"mass_{name}_edges.npz"),
    )
    if not np.array_equal(volume["t_s"], edges["output_t_s"]) or not np.array_equal(
        edges["t_s"][edges["output_indices"]], volume["t_s"]
    ):
        raise ValueError("raw mass output-node alignment mismatch")
    mass = np.sum(volume["h_m"] * volume["cell_area_m2"], axis=(1, 2))
    normal = {
        wall: edges[f"{wall}_{'qx' if NORMAL_COMPONENT[wall] == 1 else 'qy'}_m2_s"]
        for wall in WALLS
    }
    rates = outward_mass_rates(normal, edges["length_weights_m"])
    signed = {
        wall: trapezoid_prefix(rates["signed"][wall], edges["t_s"]) for wall in WALLS
    }
    absolute = {
        wall: trapezoid_prefix(rates["absolute"][wall], edges["t_s"]) for wall in WALLS
    }
    integral = sum(signed.values())[edges["output_indices"]]
    balance = mass - 11.8 + integral
    arrays = {
        "output_t_s": volume["t_s"],
        "edge_t_s": edges["t_s"],
        "volume_m3": mass,
        "outward_integral_at_outputs_m3": integral,
        "corrected_balance_m3": balance,
        "bias_removed_balance_m3": balance - balance[0],
        "initial_bias_m3": np.asarray(balance[0], dtype=np.float64),
    }
    for wall in WALLS:
        arrays.update(
            {
                f"{wall}_signed_rate_m3_s": rates["signed"][wall],
                f"{wall}_absolute_rate_m3_s": rates["absolute"][wall],
                f"{wall}_signed_integral_m3": signed[wall],
                f"{wall}_absolute_integral_m3": absolute[wall],
                f"{wall}_max_abs_normal_q_m2_s": rates["max_abs_normal"][wall],
            }
        )
    _save(output / f"mass_{name}_analysis.npz", arrays)
    return arrays


def analyze_raw(
    run_dir: str | Path,
    raw_dir: str | Path,
    output_dir: str | Path,
    *,
    root: str | Path = Path("."),
) -> Path:
    """Guarded saved-raw quality analysis; impossible to call successfully on pilot."""
    root, run = Path(root).resolve(), Path(run_dir)
    run = run if run.is_absolute() else root / run
    protocol, summary, model = _full_context(run, root)
    raw, output = Path(raw_dir), Path(output_dir)
    raw = raw if raw.is_absolute() else root / raw
    output = output if output.is_absolute() else root / output
    if output.exists():
        raise FileExistsError("quality output directory must be new")
    manifest = _validated_operators(run)
    raw_report = _read(raw / "raw_query_report.json")
    if (
        raw_report != summary["raw_queries"]
        or raw_report["quality_metrics_computed"] is not False
        or raw_report["actual_query_counts"] != EXPECTED_COUNTS
        or raw_report["model_state_sha256"] != _model_identity(model)
        or raw_report["operators_manifest_sha256"]
        != _sha(run / "operators_manifest.json")
    ):
        raise ValueError("raw query report/checkpoint/operator mismatch")
    for filename, expected in raw_report["output_sha256"].items():
        if _sha(raw / filename) != expected:
            raise ValueError(f"raw neural output changed: {filename}")
    for entry in manifest["operators"]:
        for filename, schema in entry["output_schemas"].items():
            _assert_schema(_load(raw / filename), schema)
    inputs = {}
    for key, binding in protocol["evaluation_inputs"].items():
        path = root / binding["path"]
        if _sha(path) != binding["sha256"]:
            raise ValueError(f"frozen required evaluation input changed: {key}")
        inputs[key] = path
    reference = _load(inputs["selected_reference"])
    region, sensitivity = _load(inputs["gradient_region"]), _load(inputs["sensitivity"])
    primary, higher = _load(raw / "gauss3_means.npz"), _load(raw / "gauss5_means.npz")
    times, areas = primary["t_s"], primary["cell_area_m2"]
    if not np.array_equal(reference["t_s"][1:], times) or not np.array_equal(
        region["t_s"], times
    ):
        raise ValueError("reference/gradient/neural time axes mismatch")
    for key in ("x_m", "y_m", "x_edges_m", "y_edges_m", "cell_area_m2"):
        if not np.array_equal(reference[key], primary[key]) or not np.array_equal(
            primary[key], higher[key]
        ):
            raise ValueError(f"reference/neural common mean geometry mismatch: {key}")
    ref = np.stack([reference[key][1:] for key in FIELD_NAMES], axis=-1)
    if np.any(ref[..., 0] <= 0):
        raise ValueError("accepted wet reference has nonpositive common mean depth")
    p, h = primary["state_mean_h_qx_qy"], higher["state_mean_h_qx_qy"]
    mask = region["dilated_mask"]
    regions = {
        "whole": np.ones_like(mask, dtype=bool),
        "reference_gradient": mask,
        "complement": ~mask,
    }
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    metrics = {
        name: {
            reg: region_metrics(
                _with_cell_ratios(state),
                _with_cell_ratios(ref),
                areas,
                selection,
                times,
            )
            for reg, selection in regions.items()
        }
        for name, state in (("Gauss3", p), ("Gauss5", h))
    }
    metrics["Gauss5_minus_Gauss3_operator_discrepancy"] = {
        reg: region_metrics(
            _with_cell_ratios(h), _with_cell_ratios(p), areas, selection, times
        )
        for reg, selection in regions.items()
    }
    _write(
        output / "field_metrics.json",
        {
            "definitions": evaluation_definitions(),
            "metrics": metrics,
            "reference_minimum_mean_depth_m": float(np.min(ref[..., 0])),
            "reference_cells_included": (
                "all; no threshold or neural-output-dependent exclusions"
            ),
        },
    )
    saved = {
        key: primary[key]
        for key in ("x_m", "y_m", "x_edges_m", "y_edges_m", "cell_area_m2", "t_s")
    }
    saved.update(
        reference_state_h_qx_qy=ref,
        gauss3_state_h_qx_qy=p,
        gauss5_state_h_qx_qy=h,
        gauss3_minus_reference=p - ref,
        gauss5_minus_reference=h - ref,
        gauss5_minus_gauss3=h - p,
        reference_gradient_dilated_mask=mask,
    )
    indicator_report = {}
    for name in ("Dspace", "Dtime", "Dsum"):
        indicator = sensitivity[name][1:]
        if indicator.shape != ref.shape or not np.isfinite(indicator).all():
            raise ValueError("reference conserved sensitivity shape/values invalid")
        saved[f"reference_{name}"] = indicator
        indicator_report[name] = {
            reg: region_metrics(
                indicator, np.zeros_like(indicator), areas, selection, times
            )
            for reg, selection in regions.items()
        }
    _save(output / "comparison_fields_and_reference_sensitivity.npz", saved)
    _write(
        output / "reference_sensitivity_amplitudes.json",
        {
            "SI_amplitudes": indicator_report,
            "conserved_fields_only": True,
            "relative_reason": (
                "indicator amplitudes have zero comparison baseline; no relat"
                "ive amplitude or velocity-indicator claim"
            ),
            "primary_error_greater_than_Dsum_counts_by_field": np.sum(
                np.abs(p - ref) > saved["reference_Dsum"], axis=(0, 1, 2)
            ).tolist(),
            "positive_cell_times_per_field": int(np.prod(ref.shape[:-1])),
            "qualification": (
                "Error/indicator comparisons are descriptive; Dsum is not a b"
                "ound or interval with coverage"
            ),
        },
    )
    _write(
        output / "parity.json",
        {
            name: parity_metrics(state, times)
            for name, state in (
                ("Gauss3", p),
                ("Gauss5", h),
                ("numerical_reference", ref),
            )
        },
    )
    masses = {
        name: _mass_analysis(raw, output, name)
        for name in ("base", "spatial", "temporal")
    }
    deltas = {"output_t_s": masses["base"]["output_t_s"]}
    for name in ("spatial", "temporal"):
        for field, source in (
            ("deltaM_m3", "volume_m3"),
            ("deltaIout_m3", "outward_integral_at_outputs_m3"),
            ("deltaB_m3", "corrected_balance_m3"),
        ):
            deltas[f"{name}_{field}"] = masses[name][source] - masses["base"][source]
    _save(output / "mass_spatial_and_temporal_differences.npz", deltas)
    mass_summary = {}
    for name, arrays in masses.items():
        mass_summary[name] = {
            "initial_bias_m3": float(arrays["initial_bias_m3"]),
            "sampled_max_abs_corrected_balance_m3": float(
                np.max(np.abs(arrays["corrected_balance_m3"]))
            ),
            "sampled_max_abs_bias_removed_balance_m3": float(
                np.max(np.abs(arrays["bias_removed_balance_m3"]))
            ),
            "final_corrected_balance_m3": float(arrays["corrected_balance_m3"][-1]),
            "edges": {
                wall: {
                    "final_signed_outward_integral_m3": float(
                        arrays[f"{wall}_signed_integral_m3"][-1]
                    ),
                    "final_absolute_normal_integral_m3": float(
                        arrays[f"{wall}_absolute_integral_m3"][-1]
                    ),
                    "sampled_max_absolute_normal_q_m2_s": float(
                        np.max(arrays[f"{wall}_max_abs_normal_q_m2_s"])
                    ),
                }
                for wall in WALLS
            },
        }
    _write(
        output / "mass_summary.json",
        {
            "operators": mass_summary,
            "sampled_max_abs_differences_m3": {
                key: float(np.max(np.abs(value)))
                for key, value in deltas.items()
                if key != "output_t_s"
            },
            "limits": (
                "Spatial changes volume and edge spatial rules; temporal "
                "changes the time rule and repeats the base volume pass. "
                "These differences are not error bounds. Maxima are sampled."
            ),
        },
    )
    _write(
        output / "analysis_receipt.json",
        {
            "state": "completed_full10000update_quality_analysis",
            "protocol_sha256": summary["protocol_sha256"],
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "raw_query_report_sha256": _sha(raw / "raw_query_report.json"),
            "evaluation_input_bindings": protocol["evaluation_inputs"],
            "analysis_wall_s_before_receipt": time.perf_counter() - started,
            "definitions": evaluation_definitions(),
            "source_sha256": _sha(Path(__file__)),
            "output_sha256": {
                path.name: _sha(path) for path in output.iterdir() if path.is_file()
            },
        },
    )
    return output


def evaluate_experiment(
    run_dir: str | Path, output_dir: str | Path, root: str | Path = Path(".")
) -> Path:
    """Future full-only analysis of the one raw pass already saved by training.

    The separately authorized extracted selected-checkpoint replay performs the
    second forward/operator pass. Evaluation does not add an uncounted third.
    """
    root = Path(root).resolve()
    run, output = root / run_dir, root / output_dir
    protocol, summary, _ = _full_context(run, root)
    if output.exists():
        raise FileExistsError("full evaluation directory must be new")
    output.mkdir(parents=True, exist_ok=False)
    raw_report = _read(run / "raw_queries" / "raw_query_report.json")
    analyze_raw(run, run / "raw_queries", output / "analysis", root=root)
    _write(
        output / "evaluation_receipt.json",
        {
            "state": "completed",
            "protocol_sha256": summary["protocol_sha256"],
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "actual_raw_queries": raw_report,
            "evaluation_inputs": protocol["evaluation_inputs"],
            "additional_neural_queries": 0,
            "raw_query_directory": str((run / "raw_queries").relative_to(root)),
            "scope": (
                "single-seed genuine2D numerical-reference comparison; provis"
                "ional reference with material sensitivity"
            ),
        },
    )
    return output
