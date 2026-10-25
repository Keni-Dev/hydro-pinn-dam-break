"""Fixed PCG64 physics-only coordinate partitions and exact IC/BC targets.

All NPZ arrays are float64 with shape (points,2). Coordinate columns are
(x/Lc,t/Tc); target columns are (h/H,q/(H*Uc)). ``interior_xt_nd`` has no
matching labels. Only initial and boundary targets are present. Training uses
seeded random points; validation PDE points use its separate declared grid and
times, with deterministic IC/BC points. The dam point is excluded from IC sets.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hydro_pinn.config import validate_config, validate_request
from hydro_pinn.physics import Scales

type Samples = dict[str, NDArray[np.float64]]
SAMPLE_KEYS = frozenset(
    {
        "interior_xt_nd",
        "initial_xt_nd",
        "initial_hq_nd",
        "boundary_xt_nd",
        "boundary_hq_nd",
    }
)


def _inputs(
    case: dict[str, Any], training: dict[str, Any], seed: int
) -> tuple[dict[str, Any], dict[str, Any], Scales]:
    checked_case, checked_training = validate_config(case), validate_config(training)
    if (
        checked_case["config_type"] != "case"
        or checked_training["config_type"] != "training"
    ):
        raise ValueError("sampling requires case and training configurations")
    validate_request(
        {"seed": seed, "device": "cpu", "dtype": "float64", "run_id": None}
    )
    validation = checked_training["validation"]
    if validation["grid"]["domain_m"] != checked_case["domain_m"]:
        raise ValueError("validation and case domains must match")
    if validation["times_s"][-1] > checked_case["final_time_s"]:
        raise ValueError("validation times must stay inside the physical horizon")
    return checked_case, checked_training, Scales.from_case(checked_case)


def _targets(case: dict[str, Any], scales: Scales, x_nd: NDArray) -> NDArray:
    dam_nd = case["dam_position_m"] / scales.Lc_m
    depth = np.where(x_nd < dam_nd, case["left_depth_m"], case["right_depth_m"])
    return np.column_stack((depth / scales.H_m, np.zeros_like(depth)))


def _strict_uniform(
    rng: np.random.Generator, lower: float, upper: float, count: int
) -> NDArray:
    """Open-interval coordinates, including protection against rounded endpoints."""
    values = lower + (upper - lower) * rng.random(count)
    return np.clip(values, np.nextafter(lower, upper), np.nextafter(upper, lower))


def generate_training_samples(
    case: dict[str, Any], training: dict[str, Any], seed: int
) -> Samples:
    """Generate fixed PCG64 points; retain exact discontinuous rest-state targets.

    IC counts split floor(N/2) on the left and the rest on the right, then are
    shuffled with the same generator. This guarantees both branches for N>=2
    without smoothing the jump. Boundary times and interior times are positive.
    """
    case, training, scales = _inputs(case, training, seed)
    counts = training["sampling"]
    initial_count = counts["initial_points"]
    if initial_count < 2:
        raise ValueError("initial_points must be >=2 to sample both sides of the dam")
    rng = np.random.Generator(np.random.PCG64(seed))
    a, b = np.array(case["domain_m"]) / scales.Lc_m
    dam = case["dam_position_m"] / scales.Lc_m
    horizon = case["final_time_s"] / scales.Tc_s
    interior = np.column_stack(
        (
            _strict_uniform(rng, a, b, counts["interior_points"]),
            _strict_uniform(rng, 0, horizon, counts["interior_points"]),
        )
    )
    initial_x = np.concatenate(
        (
            _strict_uniform(rng, a, dam, initial_count // 2),
            _strict_uniform(rng, dam, b, initial_count - initial_count // 2),
        )
    )
    rng.shuffle(initial_x)
    initial = np.column_stack((initial_x, np.zeros(initial_count)))
    per_side = counts["boundary_points_per_side"]
    boundary = np.vstack(
        (
            np.column_stack(
                (np.full(per_side, a), _strict_uniform(rng, 0, horizon, per_side))
            ),
            np.column_stack(
                (np.full(per_side, b), _strict_uniform(rng, 0, horizon, per_side))
            ),
        )
    )
    samples = _validated_samples(
        {
            "interior_xt_nd": interior,
            "initial_xt_nd": initial,
            "initial_hq_nd": _targets(case, scales, initial_x),
            "boundary_xt_nd": boundary,
            "boundary_hq_nd": _targets(case, scales, boundary[:, 0]),
        }
    )
    validation = generate_validation_samples(case, training, seed)
    coordinate_keys = ("interior_xt_nd", "initial_xt_nd", "boundary_xt_nd")
    training_points = {tuple(row) for key in coordinate_keys for row in samples[key]}
    validation_points = {
        tuple(row) for key in coordinate_keys for row in validation[key]
    }
    if not training_points.isdisjoint(validation_points):
        raise ValueError("training and validation coordinate partitions collide")
    return samples


def generate_validation_samples(
    case: dict[str, Any], training: dict[str, Any], seed: int
) -> Samples:
    """Use the declared nonfinal validation grid with exact IC/BC targets.

    Validation is deterministic for its configuration; seed is validated for a
    uniform caller interface but does not randomize the declared grid. IC uses
    those cell centres except any dam point. BC uses the configured positive
    validation times at both outer endpoints. No analytical interior labels
    are evaluated or stored.
    """
    case, training, scales = _inputs(case, training, seed)
    validation = training["validation"]
    cells = validation["grid"]["cells"]
    a, b = np.array(case["domain_m"], dtype=np.float64) / scales.Lc_m
    x = a + (np.arange(cells, dtype=np.float64) + 0.5) * (b - a) / cells
    times = np.array(validation["times_s"], dtype=np.float64) / scales.Tc_s
    xx, tt = np.meshgrid(x, times)
    interior = np.column_stack((xx.ravel(), tt.ravel()))
    dam = case["dam_position_m"] / scales.Lc_m
    initial_x = x[x != dam]
    if not np.any(initial_x < dam) or not np.any(initial_x > dam):
        raise ValueError(
            "validation grid must provide initial points on both dam sides"
        )
    initial = np.column_stack((initial_x, np.zeros_like(initial_x)))
    boundary = np.vstack(
        (
            np.column_stack((np.full(len(times), a), times)),
            np.column_stack((np.full(len(times), b), times)),
        )
    )
    return _validated_samples(
        {
            "interior_xt_nd": interior,
            "initial_xt_nd": initial,
            "initial_hq_nd": _targets(case, scales, initial_x),
            "boundary_xt_nd": boundary,
            "boundary_hq_nd": _targets(case, scales, boundary[:, 0]),
        }
    )


def _validated_samples(samples: Samples) -> Samples:
    if set(samples) != SAMPLE_KEYS:
        raise ValueError("sample schema requires IC/BC targets and no interior labels")
    for name, value in samples.items():
        if not isinstance(value, np.ndarray) or value.dtype != np.float64:
            raise ValueError(f"{name} must be a float64 NumPy array")
        if value.ndim != 2 or value.shape[1] != 2 or len(value) == 0:
            raise ValueError(f"{name} must have nonempty shape (points,2)")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
    for component in ("initial", "boundary"):
        if samples[f"{component}_xt_nd"].shape != samples[f"{component}_hq_nd"].shape:
            raise ValueError(
                f"{component} coordinates and targets must have matching shapes"
            )
    if np.any(samples["interior_xt_nd"][:, 1] <= 0):
        raise ValueError("interior times must be strictly positive")
    if np.any(samples["initial_xt_nd"][:, 1] != 0):
        raise ValueError("initial times must equal zero")
    if np.any(samples["initial_hq_nd"][:, 0] < 0) or np.any(
        samples["boundary_hq_nd"][:, 0] < 0
    ):
        raise ValueError("IC/BC target depths must be nonnegative")
    return samples


def save_samples(path: str | Path, samples: Samples) -> None:
    """Save one new NPZ atomically; refuse to replace a saved partition."""
    samples = _validated_samples(samples)
    destination = Path(path)
    if destination.is_symlink() or any(
        parent.is_symlink() for parent in destination.absolute().parents
    ):
        raise ValueError("sample output must not follow symlinks")
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, suffix=".npz.tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            np.savez(handle, **samples)
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def load_samples(path: str | Path) -> Samples:
    """Load non-pickle arrays and validate the exact physics-only NPZ schema."""
    with np.load(path, allow_pickle=False) as arrays:
        samples = {name: arrays[name] for name in arrays.files}
    return _validated_samples(samples)
