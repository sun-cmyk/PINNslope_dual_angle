"""Dual-angle PINNslope model and training utilities."""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from pinnslope.Arch_Adaptive import Network
from pinnslope.ArchPE import NetworkPE


Tensor = torch.Tensor


def map_slope_outputs(
    raw_slopes: Tensor,
    shifts: Tensor,
    scales: Tensor,
) -> Tensor:
    """Map two unconstrained outputs to user-defined slope intervals.

    Each output follows ``sigma = sigmoid(raw) * scale + shift``. A
    negative scale reverses the interval, which is useful for constraining
    one output to negative slopes.
    """
    if raw_slopes.ndim != 2 or raw_slopes.shape[1] != 2:
        raise ValueError("raw_slopes must have shape (n_points, 2)")
    if shifts.numel() != 2 or scales.numel() != 2:
        raise ValueError("shifts and scales must each contain two values")

    shifts = shifts.to(device=raw_slopes.device, dtype=raw_slopes.dtype)
    scales = scales.to(device=raw_slopes.device, dtype=raw_slopes.dtype)
    return torch.sigmoid(raw_slopes) * scales.reshape(1, 2) + shifts.reshape(1, 2)


class DualAnglePINN(nn.Module):
    """Two-network PINN with two wavefield and two slope outputs."""

    def __init__(
        self,
        wavefield_hidden: Sequence[int] = (512, 512, 512, 512),
        slope_hidden: Sequence[int] = (8, 8, 8, 8),
        positional_encoding: Sequence[float] = (8, 32, 2),
        slope_shifts: Sequence[float] = (0.0, 0.0),
        slope_scales: Sequence[float] = (1.0, -1.0),
        activation: str = "Tanh",
        layer_type: str = "linear",
        device: str = "cpu",
        wavefield_network: Optional[nn.Module] = None,
        slope_network: Optional[nn.Module] = None,
    ):
        super().__init__()
        if len(wavefield_hidden) == 0 or len(slope_hidden) == 0:
            raise ValueError("hidden layer definitions cannot be empty")
        if len(positional_encoding) != 3:
            raise ValueError("positional_encoding must contain x, t, and scale values")
        if len(slope_shifts) != 2 or len(slope_scales) != 2:
            raise ValueError("two slope shifts and two slope scales are required")

        pos_enc = list(positional_encoding)
        encoded_inputs = int(2 * pos_enc[0] + 2 * pos_enc[1])
        self.wavefield_network = wavefield_network or NetworkPE(
            encoded_inputs,
            2,
            list(wavefield_hidden),
            lay=layer_type,
            act=activation,
            PosEnc=pos_enc,
            device=device,
        )
        self.slope_network = slope_network or Network(
            2,
            2,
            list(slope_hidden),
            lay=layer_type,
            act=activation,
        )
        self.register_buffer(
            "slope_shifts", torch.tensor(slope_shifts, dtype=torch.float32)
        )
        self.register_buffer(
            "slope_scales", torch.tensor(slope_scales, dtype=torch.float32)
        )
        self.to(device)

    def wavefields(self, coordinates: Tensor) -> Tensor:
        wavefields = self.wavefield_network(coordinates)
        if wavefields.ndim != 2 or wavefields.shape[1] != 2:
            raise ValueError("wavefield network must return shape (n_points, 2)")
        return wavefields

    def slopes(self, coordinates: Tensor) -> Tensor:
        raw_slopes = self.slope_network(coordinates)
        return map_slope_outputs(raw_slopes, self.slope_shifts, self.slope_scales)

    def forward(self, coordinates: Tensor) -> Tuple[Tensor, Tensor]:
        return self.wavefields(coordinates), self.slopes(coordinates)


@dataclass
class DualAngleLosses:
    total: Tensor
    data: Tensor
    physics: Tensor
    physics_1: Tensor
    physics_2: Tensor

    def detach(self) -> "DualAngleLosses":
        return DualAngleLosses(
            total=self.total.detach(),
            data=self.data.detach(),
            physics=self.physics.detach(),
            physics_1=self.physics_1.detach(),
            physics_2=self.physics_2.detach(),
        )

    def as_floats(self) -> Dict[str, float]:
        return {
            "total": self.total.item(),
            "data": self.data.item(),
            "physics": self.physics.item(),
            "physics_1": self.physics_1.item(),
            "physics_2": self.physics_2.item(),
        }


def dual_angle_loss(
    model: DualAnglePINN,
    grid_coordinates: Tensor,
    trace_coordinates: Tensor,
    trace_values: Tensor,
    data_weight: float = 1000.0,
    data_criterion: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
) -> DualAngleLosses:
    """Compute two plane-wave residuals and the summed-wavefield data loss."""
    _validate_coordinates("grid_coordinates", grid_coordinates)
    _validate_coordinates("trace_coordinates", trace_coordinates)
    if grid_coordinates.device != trace_coordinates.device:
        raise ValueError("grid and trace coordinates must be on the same device")

    grid_coordinates = grid_coordinates.detach().requires_grad_(True)
    grid_wavefields = model.wavefields(grid_coordinates)

    gradient_1 = torch.autograd.grad(
        grid_wavefields[:, 0],
        grid_coordinates,
        grad_outputs=torch.ones_like(grid_wavefields[:, 0]),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient_2 = torch.autograd.grad(
        grid_wavefields[:, 1],
        grid_coordinates,
        grad_outputs=torch.ones_like(grid_wavefields[:, 1]),
        create_graph=True,
        retain_graph=True,
    )[0]

    slopes = model.slopes(grid_coordinates)
    residual_1 = gradient_1[:, 0] + slopes[:, 0] * gradient_1[:, 1]
    residual_2 = gradient_2[:, 0] + slopes[:, 1] * gradient_2[:, 1]
    physics_1 = residual_1.square().mean()
    physics_2 = residual_2.square().mean()
    physics = physics_1 + physics_2

    predicted_traces = model.wavefields(trace_coordinates).sum(dim=1, keepdim=True)
    trace_values = trace_values.to(
        device=predicted_traces.device, dtype=predicted_traces.dtype
    ).reshape(-1, 1)
    if trace_values.shape[0] != predicted_traces.shape[0]:
        raise ValueError("trace_values must contain one value per trace coordinate")
    criterion = data_criterion or F.l1_loss
    data = criterion(predicted_traces, trace_values)
    total = physics + float(data_weight) * data

    return DualAngleLosses(
        total=total,
        data=data,
        physics=physics,
        physics_1=physics_1,
        physics_2=physics_2,
    )


def train_dual_angle_step(
    model: DualAnglePINN,
    wavefield_optimizer: torch.optim.Optimizer,
    slope_optimizer: torch.optim.Optimizer,
    grid_coordinates: Tensor,
    trace_coordinates: Tensor,
    trace_values: Tensor,
    data_weight: float = 1000.0,
    data_criterion: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
) -> DualAngleLosses:
    """Run one complete optimization step for both networks."""
    model.train()
    wavefield_optimizer.zero_grad(set_to_none=True)
    slope_optimizer.zero_grad(set_to_none=True)
    losses = dual_angle_loss(
        model=model,
        grid_coordinates=grid_coordinates,
        trace_coordinates=trace_coordinates,
        trace_values=trace_values,
        data_weight=data_weight,
        data_criterion=data_criterion,
    )
    losses.total.backward()
    wavefield_optimizer.step()
    slope_optimizer.step()
    return losses.detach()


def _validate_coordinates(name: str, coordinates: Tensor) -> None:
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("{} must have shape (n_points, 2)".format(name))
    if coordinates.shape[0] == 0:
        raise ValueError("{} cannot be empty".format(name))
