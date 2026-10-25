"""Scaled 1D PINN with a disclosed positive nondimensional depth floor.

``forward`` accepts (x/Lc,t/Tc) and returns (h/H,q/(H*Uc)). The two raw
network outputs are depth logit and u/Uc. ``physical_state`` accepts metres and
seconds and returns SI h,q. Neither path changes the caller's dtype or device.
"""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hydro_pinn.config import validate_config, validate_request
from hydro_pinn.physics import Scales


class ModelError(ValueError):
    """Model configuration, input, or output violates its declared contract."""


class ShallowWaterPINN(nn.Module):
    """Tanh MLP with h*=softplus(zh)+floor and q*=h*u*.

    The caller owns the initialization seed. The architecture is taken exactly
    from the validated training configuration; both weighting variants can use
    the same state dictionary. A positive network floor does not alter E2's
    prescribed hR=0 target or establish conservation/solution accuracy.
    """

    def __init__(
        self,
        case: dict[str, Any],
        training: dict[str, Any],
        *,
        device: str = "cpu",
        dtype: str = "float64",
    ) -> None:
        super().__init__()
        self.case = validate_config(case)
        self.training_config = validate_config(training)
        if self.case["config_type"] != "case":
            raise ModelError("model requires a case configuration")
        if self.training_config["config_type"] != "training":
            raise ModelError("model requires a training configuration")
        validate_request(
            {"seed": None, "device": device, "dtype": dtype, "run_id": None}
        )
        if device == "cuda" and not torch.cuda.is_available():
            raise ModelError(
                "requested CUDA backend is unavailable; no fallback applied"
            )
        if device == "mps" and not torch.backends.mps.is_available():
            raise ModelError(
                "requested MPS backend is unavailable; no fallback applied"
            )
        self.scales = Scales.from_case(self.case)
        self.floor = self.training_config["network"]["depth_floor_nd"]
        widths = [2, *self.training_config["network"]["hidden_layers"], 2]
        torch_dtype = {"float32": torch.float32, "float64": torch.float64}[dtype]
        layers: list[nn.Module] = []
        for index, (input_width, output_width) in enumerate(
            zip(widths[:-1], widths[1:], strict=True)
        ):
            layers.append(
                nn.Linear(input_width, output_width, device=device, dtype=torch_dtype)
            )
            if index < len(widths) - 2:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def _check_coordinates(self, coordinates: Tensor) -> None:
        if not isinstance(coordinates, Tensor):
            raise ModelError(
                "coordinates must be a torch Tensor; conversion is explicit"
            )
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ModelError("coordinates must have shape (points,2), ordered (x,t)")
        parameter = next(self.parameters())
        if (
            coordinates.dtype != parameter.dtype
            or coordinates.device != parameter.device
        ):
            raise ModelError("coordinates must match the model dtype and device")
        if not bool(torch.isfinite(coordinates).all()):
            raise ModelError("coordinates must contain only finite values")

    def forward(self, coords_nd: Tensor) -> Tensor:
        """Return nondimensional h*,q* while preserving derivative graphs."""
        self._check_coordinates(coords_nd)
        raw = self.network(coords_nd)
        depth = F.softplus(raw[:, 0]) + self.floor
        state = torch.stack((depth, depth * raw[:, 1]), dim=-1)
        if not bool(torch.isfinite(state).all()):
            raise ModelError("network produced a nonfinite state")
        return state

    def physical_state(self, coords_m_s: Tensor) -> Tensor:
        """Map physical x[m],t[s] to physical h[m],q[m²/s per unit width]."""
        self._check_coordinates(coords_m_s)
        coords_nd = torch.stack(
            (
                self.scales.x_to_scaled(coords_m_s[:, 0]),
                self.scales.t_to_scaled(coords_m_s[:, 1]),
            ),
            dim=-1,
        )
        return self.scales.state_to_physical(self(coords_nd))
