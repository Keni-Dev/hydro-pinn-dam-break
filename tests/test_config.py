"""Configuration failures that could invalidate a later scientific comparison."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest

from hydro_pinn.config import (
    ConfigError,
    load_config,
    resolve_run_config,
    validate_config,
    validate_request,
    validate_run_id,
)

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / "configs/cases/e1_wet.json"
DRY_CASE = ROOT / "configs/cases/e2_dry.json"
BASELINE = ROOT / "configs/training/baseline.json"
ADAPTIVE = ROOT / "configs/training/adaptive.json"
EVALUATION = ROOT / "configs/evaluation.json"
VERIFICATION = ROOT / "configs/verification.json"
SMOKE = ROOT / "configs/smoke.json"


def changed(document: dict, path: str, value: object) -> dict:
    result = copy.deepcopy(document)
    current = result
    components = path.split(".")
    for component in components[:-1]:
        current = current[component]
    current[components[-1]] = value
    return result


@pytest.mark.parametrize(
    "path", [CASE, DRY_CASE, BASELINE, ADAPTIVE, EVALUATION, VERIFICATION, SMOKE]
)
def test_checked_in_configs_are_valid(path: Path) -> None:
    data = load_config(path)
    assert validate_config(data) == data


def test_dry_depth_is_exact_zero_and_not_the_network_floor() -> None:
    dry = load_config(DRY_CASE)
    training = load_config(BASELINE)
    assert dry["right_depth_m"] == 0.0
    assert dry["boundary"]["right_h_m"] == 0.0
    assert dry["reference"]["kind"] == "ritter"
    assert training["network"]["depth_floor_nd"] == 1e-6
    assert dry["grid"]["cfl"] == load_config(CASE)["grid"]["cfl"] == 0.45


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("left_depth_m", 0, "left_depth_m"),
        ("left_depth_m", -1, "left_depth_m"),
        ("right_depth_m", -1, "right_depth_m"),
        ("right_depth_m", 1, "right_depth_m"),
        ("right_depth_m", 0, "wet_bed requires"),
        ("gravity_m_s2", 0, "gravity_m_s2"),
        ("gravity_m_s2", True, "not a boolean"),
        ("final_time_s", 0, "final_time_s"),
        ("final_time_s", -0.5, "final_time_s"),
        ("final_time_s", 1.0, "boundary contact"),
        ("domain_m", [-5, -5], "domain_m"),
        ("domain_m", [5, -5], "domain_m"),
        ("domain_m", [0, 0, 1], "domain_m"),
        ("domain_m", [-5, True], "not a boolean"),
        ("dam_position_m", 5, "dam_position_m"),
        ("dam_position_m", 0.001, "align with a cell face"),
        ("left_velocity_m_s", 1, "core rest-state"),
        ("grid.cell_counts", [100, 201], "must be even"),
        ("grid.cell_counts", [100, "200"], "must be an integer"),
        ("grid.cell_counts", [100, 100], "strictly increasing"),
        ("grid.cfl", 0, "CFL"),
        ("grid.cfl", 1.1, "CFL"),
        ("grid.state_location", "point_value", "cell_average"),
        ("scales.H_m", 0, "H_m"),
        ("scales.Uc_m_s", 1, "sqrt"),
        ("scales.Tc_s", 1, "Lc_m/Uc_m_s"),
        ("boundary.right_h_m", 0, "initial state"),
        ("assumptions.friction", "manning", "none"),
        ("array_units", {"h_m": "cm"}, "array_units"),
        ("schema_version", True, "schema_version"),
    ],
)
def test_invalid_case_fields_report_the_condition(
    path: str, value: object, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        validate_config(changed(load_config(CASE), path, value))


def test_ritter_rejects_a_wet_downstream_state() -> None:
    with pytest.raises(ConfigError, match="Ritter requires exact hR=0"):
        validate_config(changed(load_config(DRY_CASE), "right_depth_m", 0.001))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 10**1000])
def test_nonfinite_or_unrepresentable_case_numbers_are_rejected(value: float) -> None:
    with pytest.raises(ConfigError, match="finite"):
        validate_config(changed(load_config(CASE), "gravity_m_s2", value))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"gravity_m_s2": 1e-300, "scales.H_m": 1e-300},
            "case.scales.Uc_m_s.*positive",
        ),
        (
            {
                "gravity_m_s2": 1e100,
                "final_time_s": 1e-100,
                "scales.H_m": 1e300,
            },
            "case.scales.Uc_m_s.*finite",
        ),
        (
            {
                "gravity_m_s2": 1e-300,
                "scales.Uc_m_s": 1e-150,
                "scales.Lc_m": 1e308,
            },
            "case.scales.Tc_s.*finite",
        ),
        (
            {
                "gravity_m_s2": 1e308,
                "final_time_s": 1e-200,
                "scales.Uc_m_s": 1e154,
                "scales.Lc_m": 5e-324,
                "scales.Tc_s": 5e-324,
            },
            "case.scales.Tc_s.*positive",
        ),
    ],
)
def test_finite_inputs_cannot_overflow_or_underflow_derived_scales(
    updates: dict[str, float], message: str
) -> None:
    document = load_config(CASE)
    for path, value in updates.items():
        assert math.isfinite(value) and value > 0
        document = changed(document, path, value)
    with pytest.raises(ConfigError, match=message):
        validate_config(document)


@pytest.mark.parametrize("source", [CASE, EVALUATION])
def test_finite_domain_endpoints_cannot_overflow_the_span(source: Path) -> None:
    document = load_config(source)
    path = "domain_m" if source == CASE else "grid.domain_m"
    with pytest.raises(ConfigError, match="domain_m width.*finite"):
        validate_config(changed(document, path, [-1e308, 1e308]))


def test_native_cell_spacing_cannot_underflow_to_zero() -> None:
    document = load_config(CASE)
    document["domain_m"] = [-5e-320, 5e-320]
    document["final_time_s"] = 1e-322
    document["grid"]["cell_counts"] = [10**10]
    with pytest.raises(ConfigError, match="derived dx_m.*positive"):
        validate_config(document)


def test_unknown_and_missing_fields_are_not_silently_ignored() -> None:
    unknown = load_config(CASE)
    unknown["final_tim_s"] = 0.5
    with pytest.raises(ConfigError, match="unknown fields: final_tim_s"):
        validate_config(unknown)
    missing = load_config(CASE)
    del missing["final_time_s"]
    with pytest.raises(ConfigError, match="missing required fields: final_time_s"):
        validate_config(missing)


def test_validation_returns_an_independent_copy() -> None:
    case = load_config(CASE)
    result = validate_config(case)
    result["grid"]["cell_counts"].append(1600)
    assert case["grid"]["cell_counts"] == [100, 200, 400, 800]


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ('{"config_type":"case","config_type":"training"}', "duplicate"),
        ('{"config_type":"case","schema_version":1,"x":NaN}', "nonstandard JSON"),
        ('{"config_type":"case","schema_version":1,"x":Infinity}', "nonstandard JSON"),
        ('{"config_type":"unknown","schema_version":1}', "config_type"),
        ("[]", "must be an object"),
        ("null", "must be an object"),
        ("{", "Expecting"),
    ],
)
def test_malformed_and_ambiguous_json_is_rejected(
    tmp_path: Path, contents: str, message: str
) -> None:
    path = tmp_path / "bad.json"
    path.write_text(contents)
    with pytest.raises(ConfigError, match=message):
        load_config(path)


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        ".",
        "..",
        "../escape",
        "a/b",
        "a\\b",
        "/tmp/run",
        "trailing.",
        " spaces",
        "hí",
        "a" * 97,
        "CON",
        "lpt1",
    ],
)
def test_run_ids_cannot_address_paths_or_reserved_names(run_id: str) -> None:
    with pytest.raises(ConfigError, match="run_id"):
        validate_run_id(run_id)


def test_run_id_policy_agrees_with_record_writer() -> None:
    from hydro_pinn.records import validate_run_id as validate_record_id

    for run_id in (
        "e1-reference-001",
        "E2_dry_seed11",
        "a",
        "2026-run",
        "CON_foo",
        "lpt1_seed22",
    ):
        assert validate_run_id(run_id) == validate_record_id(run_id)
    for run_id in ("CON", "lpt1", "../bad", "AUX", "a/b"):
        with pytest.raises(ValueError):
            validate_run_id(run_id)
        with pytest.raises(ValueError):
            validate_record_id(run_id)


@pytest.mark.parametrize(
    "run_id", ["e1-reference-001", "E2_dry_seed11", "a", "2026-run"]
)
def test_portable_run_ids_are_accepted(run_id: str) -> None:
    assert validate_run_id(run_id) == run_id


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed", -1, "request.seed"),
        ("seed", True, "request.seed"),
        ("seed", 2**32, "request.seed"),
        ("device", "cuda:hello", "request.device"),
        ("dtype", "fp64", "request.dtype"),
        ("run_id", "../bad", "request.run_id"),
    ],
)
def test_invalid_request_fields(field: str, value: object, message: str) -> None:
    data = {"seed": 11, "device": "cpu", "dtype": "float64", "run_id": "test"}
    data[field] = value
    with pytest.raises(ConfigError, match=message):
        validate_request(data)


def test_mps_float64_is_rejected_without_an_implicit_fallback() -> None:
    with pytest.raises(ConfigError, match="MPS does not support float64"):
        resolve_run_config(CASE, device="mps", dtype="float64")
    resolved = resolve_run_config(CASE, device="mps", dtype="float32")
    assert resolved["request"]["device"] == "mps"


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("grid.cells", 0, "grid.cells"),
        ("grid.cells", True, "grid.cells"),
        ("times_s", [0, 0.5], "positive"),
        ("times_s", [0.5, 0.1], "strictly increasing"),
        ("times_s", [0.1, 0.1], "duplicates"),
        ("profile_times_s", [0.499], "must occur"),
        ("velocity_wet_threshold_m", 0, "positive"),
        ("mass_quadrature.volume.cells", [2000, 3000], "double"),
        ("mass_quadrature.volume.include_t0", False, "True"),
        ("mass_quadrature.boundary_flux.nodes", [1001, 2002], "double"),
        (
            "mass_quadrature.boundary_flux.spatial_locations",
            "cell_centres",
            "domain_endpoints",
        ),
    ],
)
def test_evaluation_coordinates_and_quadrature_are_validated(
    path: str, value: object, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        validate_config(changed(load_config(EVALUATION), path, value))


def test_fixed_reference_and_evaluation_partitions_remain_distinct() -> None:
    verification = load_config(VERIFICATION)
    evaluation = load_config(EVALUATION)
    assert verification["partition"] == "implementation_verification"
    assert verification["grid"]["cells"] == 129
    assert verification["times_s"] == [0.037, 0.113, 0.229, 0.347, 0.463, 0.5]
    assert evaluation["grid"]["cells"] == 400
    assert evaluation["times_s"] == [i / 100 for i in range(1, 51)]
    assert evaluation["mass_quadrature"]["volume"]["cells"] == [2000, 4000]
    assert evaluation["mass_quadrature"]["boundary_flux"]["nodes"] == [1001, 2001]


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("network.hidden_layers", [64, False], "integer"),
        ("network.depth_floor_nd", 0, "positive"),
        ("sampling.interior_labels", True, "False"),
        ("sampling.interior_points", 0, "integer"),
        ("optimizer.steps", 0, "integer"),
        ("optimizer.learning_rate", 0, "positive"),
        ("weighting.lambda_data", 1, "physics-only"),
        ("weighting.ema_decay", 1, "decay"),
        ("weighting.target_bounds", [100, 0.01], "increasing"),
        ("seeds", [11, 11], "duplicates"),
        ("validation.config_type", "case", "evaluation"),
    ],
)
def test_training_settings_cannot_change_scientific_conventions_silently(
    path: str, value: object, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        validate_config(changed(load_config(BASELINE), path, value))


def test_baseline_and_adaptive_only_change_the_weighting_mode() -> None:
    baseline = load_config(BASELINE)
    adaptive = load_config(ADAPTIVE)
    assert baseline["weighting"]["mode"] == "fixed"
    assert adaptive["weighting"]["mode"] == "gradient_norm_ema"
    adaptive["weighting"]["mode"] = baseline["weighting"]["mode"]
    assert adaptive == baseline
    resolved_baseline = resolve_run_config(CASE, BASELINE, EVALUATION, seed=11)
    resolved_adaptive = resolve_run_config(CASE, ADAPTIVE, EVALUATION, seed=11)
    resolved_adaptive["training"]["weighting"]["mode"] = "fixed"
    for key in ("case", "training", "evaluation", "request"):
        assert resolved_baseline[key] == resolved_adaptive[key]


def test_resolution_records_exact_input_bytes_and_does_not_modify_files() -> None:
    paths = [CASE, BASELINE, EVALUATION]
    before = {path: path.read_bytes() for path in paths}
    resolved = resolve_run_config(
        CASE, BASELINE, EVALUATION, seed=11, run_id="e1-seed11"
    )
    assert resolved["request"] == {
        "seed": 11,
        "device": "cpu",
        "dtype": "float64",
        "run_id": "e1-seed11",
    }
    for entry in resolved["provenance"]["input_files"]:
        path = Path(entry["path"])
        assert entry["sha256"] == hashlib.sha256(before[path]).hexdigest()
        assert path.read_bytes() == before[path]


def test_smoke_then_explicit_overrides_have_recorded_precedence() -> None:
    resolved = resolve_run_config(
        CASE,
        BASELINE,
        EVALUATION,
        seed=22,
        run_id="smoke22",
        smoke_path=SMOKE,
        overrides={"training": {"optimizer": {"steps": 30}}},
    )
    assert resolved["training"]["optimizer"]["steps"] == 30
    assert resolved["training"]["sampling"]["interior_points"] == 128
    assert resolved["evaluation"]["partition"] == "smoke_evaluation"
    assert (
        resolved["training"]["validation"]["times_s"]
        != resolved["evaluation"]["times_s"]
    )
    updates = [
        entry
        for entry in resolved["provenance"]["overrides"]
        if entry["path"] == "training.optimizer.steps"
    ]
    assert [
        (entry["before"], entry["after"], entry["source"]) for entry in updates
    ] == [(10000, 20, "smoke"), (20, 30, "explicit_override")]
    assert resolved["provenance"]["precedence"] == [
        "input_files",
        "smoke",
        "explicit_overrides",
        "request_arguments",
    ]


@pytest.mark.parametrize(
    "overrides",
    [
        {"training": {"optimizer": {"steps": 201}}},
        {"training": {"sampling": {"interior_points": 1001}}},
        {"training": {"lbfgs": {"enabled": True}}},
    ],
)
def test_explicit_override_cannot_escape_the_smoke_budget(overrides: dict) -> None:
    with pytest.raises(ConfigError, match="smoke budget"):
        resolve_run_config(
            CASE, BASELINE, EVALUATION, seed=11, smoke_path=SMOKE, overrides=overrides
        )


def test_smoke_evaluation_dispatches_by_declared_type_not_filename(
    tmp_path: Path,
) -> None:
    path = tmp_path / "renamed.json"
    path.write_bytes(SMOKE.read_bytes())
    resolved = resolve_run_config(CASE, evaluation_path=path)
    assert resolved["evaluation"]["partition"] == "smoke_evaluation"


@pytest.mark.parametrize(
    "overrides",
    [
        {"trainng": {}},
        {"case": {"final_tim_s": 0.4}},
        {"evaluation": {"grid": {"cells": 100}}},
    ],
)
def test_overrides_cannot_create_undeclared_fields_or_missing_sections(
    overrides: dict,
) -> None:
    with pytest.raises(ConfigError, match="overrides|unknown field"):
        resolve_run_config(CASE, overrides=overrides)


def test_resolved_evaluation_must_fit_case_domain_and_horizon() -> None:
    with pytest.raises(ConfigError, match="must match"):
        resolve_run_config(
            CASE,
            evaluation_path=EVALUATION,
            overrides={"evaluation": {"grid": {"domain_m": [-4, 4]}}},
        )
    with pytest.raises(ConfigError, match="must not exceed"):
        resolve_run_config(
            CASE, evaluation_path=EVALUATION, overrides={"case": {"final_time_s": 0.4}}
        )


def test_training_requires_explicit_seed_and_disjoint_validation() -> None:
    with pytest.raises(ConfigError, match="explicit seed"):
        resolve_run_config(CASE, BASELINE, EVALUATION)
    validation = load_config(EVALUATION)
    validation["partition"] = "validation"
    with pytest.raises(ConfigError, match="overlap"):
        resolve_run_config(
            CASE,
            BASELINE,
            EVALUATION,
            seed=11,
            overrides={"training": {"validation": validation}},
        )


def test_derived_scales_are_stored_without_replacing_physical_depths() -> None:
    resolved = resolve_run_config(DRY_CASE, BASELINE, VERIFICATION, seed=33)
    scales = resolved["case"]["scales"]
    assert scales["Uc_m_s"] == math.sqrt(9.81)
    assert scales["Tc_s"] == 5 / math.sqrt(9.81)
    assert resolved["case"]["right_depth_m"] == 0
    json.dumps(resolved, allow_nan=False)
