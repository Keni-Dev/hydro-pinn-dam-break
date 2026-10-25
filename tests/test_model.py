"""PINN architecture, explicit physical scales, and depth-transform gradients."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from hydro_pinn.config import load_config
from hydro_pinn.model import ModelError, ShallowWaterPINN

ROOT = Path(__file__).resolve().parents[1]


def inputs() -> tuple[dict, dict]:
    return (
        load_config(ROOT / "configs/cases/e1_wet.json"),
        load_config(ROOT / "configs/training/baseline.json"),
    )


def test_architecture_scaled_state_and_physical_conversion() -> None:
    case, training = inputs()
    # Nonunit H makes both H and H*Uc factors visible independently of core H=1.
    case["scales"]["H_m"] = 4.0
    case["scales"]["Uc_m_s"] = (case["gravity_m_s2"] * 4) ** 0.5
    case["scales"]["Tc_s"] = case["scales"]["Lc_m"] / case["scales"]["Uc_m_s"]
    torch.manual_seed(11)
    model = ShallowWaterPINN(case, training)
    layers = [layer for layer in model.network if isinstance(layer, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in layers] == [
        (2, 64),
        (64, 64),
        (64, 64),
        (64, 64),
        (64, 2),
    ]
    assert sum(isinstance(layer, nn.Tanh) for layer in model.network) == 4
    physical_coords = torch.tensor(
        [[-3.0, 0.1], [0.0, 0.25], [4.0, 0.5]], dtype=torch.float64, requires_grad=True
    )
    nd_coords = physical_coords / torch.tensor(
        [model.scales.Lc_m, model.scales.Tc_s], dtype=torch.float64
    )
    nd = model(nd_coords)
    raw = model.network(nd_coords)
    assert nd.shape == (3, 2) and nd.dtype == torch.float64
    assert torch.isfinite(nd).all() and (nd[:, 0] >= model.floor).all()
    torch.testing.assert_close(
        nd[:, 0], torch.nn.functional.softplus(raw[:, 0]) + 1e-6, rtol=0, atol=0
    )
    torch.testing.assert_close(nd[:, 1], nd[:, 0] * raw[:, 1], rtol=0, atol=0)
    physical = model.physical_state(physical_coords)
    torch.testing.assert_close(physical[:, 0], nd[:, 0] * 4, rtol=1e-14, atol=1e-14)
    torch.testing.assert_close(
        physical[:, 1], nd[:, 1] * model.scales.Qc_m2_s, rtol=1e-14, atol=1e-14
    )
    gradients = torch.autograd.grad(physical.sum(), physical_coords)[0]
    assert gradients.shape == physical_coords.shape and torch.isfinite(gradients).all()


def test_depth_floor_and_softplus_derivative_do_not_change_dry_targets() -> None:
    case, training = inputs()
    case = load_config(ROOT / "configs/cases/e2_dry.json")
    training["network"]["hidden_layers"] = [2]
    model = ShallowWaterPINN(case, training)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.network[-1].bias.copy_(torch.tensor([0.0, 2.0], dtype=torch.float64))
    result = model(torch.zeros((1, 2), dtype=torch.float64))
    torch.testing.assert_close(
        result[0, 0], torch.log(torch.tensor(2.0, dtype=torch.float64)) + 1e-6
    )
    derivative = torch.autograd.grad(result[0, 0], model.network[-1].bias)[0]
    torch.testing.assert_close(
        derivative, torch.tensor([0.5, 0], dtype=torch.float64), rtol=0, atol=0
    )
    with torch.no_grad():
        model.network[-1].bias[0] = -1000
    near_dry = model(torch.zeros((1, 2), dtype=torch.float64))
    assert near_dry[0, 0].item() == 1e-6
    assert near_dry[0, 1].item() == 2e-6
    assert case["right_depth_m"] == 0.0


def test_model_rejects_implicit_input_cast_and_invalid_configuration() -> None:
    case, training = inputs()
    model = ShallowWaterPINN(case, training)
    with pytest.raises(ModelError, match="dtype and device"):
        model(torch.zeros((1, 2), dtype=torch.float32))
    with pytest.raises(ModelError, match="finite"):
        model(torch.tensor([[float("nan"), 0]], dtype=torch.float64))
    invalid = copy.deepcopy(training)
    invalid["network"]["hidden_layers"] = [0]
    with pytest.raises(ValueError, match="hidden_layers"):
        ShallowWaterPINN(case, invalid)
    with pytest.raises(ValueError, match="MPS does not support float64"):
        ShallowWaterPINN(case, training, device="mps", dtype="float64")
    float32_model = ShallowWaterPINN(case, training, dtype="float32")
    assert (
        float32_model(torch.zeros((1, 2), dtype=torch.float32)).dtype == torch.float32
    )
