"""Independent x,y,t PINN with positive depth and unconstrained u,v."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from hydro_pinn.physics import Scales


class PlanarPINN(nn.Module):
    """Scaled tanh MLP; the architecture does not impose y-invariance.

    Forward inputs are (x/Lc,y/Lc,t/Tc); outputs are (h*,h*u*,h*v*).
    The caller owns the initialization seed. CPU dtype is explicit.
    """

    def __init__(
        self,
        scales: Scales,
        hidden_layers: tuple[int, ...] = (64, 64, 64, 64),
        *,
        dtype: torch.dtype = torch.float32,
        depth_floor_nd: float = 1e-6,
    ) -> None:
        super().__init__()
        if not hidden_layers or any(width < 1 for width in hidden_layers):
            raise ValueError("hidden layers require positive widths")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("planar model requires CPU float32 or float64")
        if not 0 < depth_floor_nd < 1:
            raise ValueError("depth floor must be positive and below one")
        self.scales = scales
        self.depth_floor_nd = depth_floor_nd
        widths = (3, *hidden_layers, 3)
        layers: list[nn.Module] = []
        for index, (first, second) in enumerate(
            zip(widths[:-1], widths[1:], strict=True)
        ):
            layers.append(nn.Linear(first, second, device="cpu", dtype=dtype))
            if index < len(widths) - 2:
                layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def _check(self, coords: Tensor) -> None:
        parameter = next(self.parameters())
        if (
            not isinstance(coords, Tensor)
            or coords.ndim != 2
            or coords.shape[1] != 3
            or coords.dtype != parameter.dtype
            or coords.device != parameter.device
        ):
            raise ValueError(
                "coordinates require (N,3) x,y,t matching model dtype/device"
            )
        if not bool(torch.isfinite(coords).all()):
            raise ValueError("coordinates must be finite")

    def forward(self, coords_nd: Tensor) -> Tensor:
        self._check(coords_nd)
        raw = self.network(coords_nd)
        h = F.softplus(raw[:, 0]) + self.depth_floor_nd
        state = torch.stack((h, h * raw[:, 1], h * raw[:, 2]), dim=1)
        if not bool(torch.isfinite(state).all()):
            raise ValueError("model produced nonfinite state")
        return state

    def physical_state(self, coords_m_s: Tensor) -> Tensor:
        """Return SI h [m], qx/qy [m²/s] from SI x,y [m], t [s]."""
        self._check(coords_m_s)
        scales = self.scales
        coords_nd = torch.stack(
            (
                coords_m_s[:, 0] / scales.Lc_m,
                coords_m_s[:, 1] / scales.Lc_m,
                coords_m_s[:, 2] / scales.Tc_s,
            ),
            dim=1,
        )
        state = self(coords_nd)
        factors = state.new_tensor((scales.H_m, scales.Qc_m2_s, scales.Qc_m2_s))
        return state * factors
