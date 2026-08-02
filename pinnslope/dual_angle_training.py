"""Data preparation and experiment utilities for dual-angle interpolation."""

import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from pinnslope.dual_angle import DualAnglePINN, train_dual_angle_step


Tensor = torch.Tensor


def normalize_slope_interval(
    bounds: Sequence[float],
    units: str,
    x_extent: float,
    t_extent: float,
) -> Tuple[float, float]:
    """Convert physical ``dt/dx`` slopes to normalized-coordinate slopes."""
    if len(bounds) != 2 or bounds[0] >= bounds[1]:
        raise ValueError("slope bounds must contain two increasing values")
    if units == "normalized":
        return float(bounds[0]), float(bounds[1])
    if units != "physical":
        raise ValueError("slope units must be 'normalized' or 'physical'")
    if x_extent <= 0.0 or t_extent <= 0.0:
        raise ValueError("coordinate extents must be positive")
    factor = x_extent / t_extent
    return float(bounds[0] * factor), float(bounds[1] * factor)


def load_gather(
    path: str,
    orientation: str = "x-t",
    x_slice: Tuple[Optional[int], Optional[int]] = (None, None),
    t_slice: Tuple[Optional[int], Optional[int]] = (None, None),
) -> np.ndarray:
    """Load a finite two-dimensional seismic gather as ``(nx, nt)``."""
    gather = np.load(path)
    if gather.ndim != 2:
        raise ValueError("input gather must be a two-dimensional array")
    if orientation == "t-x":
        gather = gather.T
    elif orientation != "x-t":
        raise ValueError("orientation must be 'x-t' or 't-x'")
    gather = gather[slice(*x_slice), slice(*t_slice)]
    if gather.size == 0 or min(gather.shape) < 2:
        raise ValueError("cropping must leave at least two samples on each axis")
    if not np.isfinite(gather).all():
        raise ValueError("input gather contains NaN or infinite values")
    return np.asarray(gather, dtype=np.float32)


def prepare_interpolation_data(
    gather: np.ndarray,
    trace_step: int,
    device: str,
) -> Dict[str, Tensor]:
    """Build normalized coordinates, observed samples, and evaluation masks."""
    if gather.ndim != 2:
        raise ValueError("gather must have shape (nx, nt)")
    if trace_step < 1:
        raise ValueError("trace_step must be positive")

    nx, nt = gather.shape
    x = torch.linspace(0.0, 1.0, nx, device=device)
    t = torch.linspace(0.0, 1.0, nt, device=device)
    x_mesh = x.reshape(-1, 1).expand(nx, nt)
    t_mesh = t.reshape(1, -1).expand(nx, nt)
    coordinate_grid = torch.stack((x_mesh, t_mesh), dim=2)

    observed_trace_indices = torch.arange(0, nx, trace_step, device=device)
    observed_coordinates = coordinate_grid[observed_trace_indices].reshape(-1, 2)
    values = torch.as_tensor(gather, dtype=torch.float32, device=device)
    observed_values = values[observed_trace_indices].reshape(-1, 1)
    missing_trace_mask = torch.ones(nx, dtype=torch.bool, device=device)
    missing_trace_mask[observed_trace_indices] = False

    return {
        "full_coordinates": coordinate_grid.reshape(-1, 2),
        "observed_coordinates": observed_coordinates,
        "observed_values": observed_values,
        "target": values,
        "observed_trace_indices": observed_trace_indices,
        "missing_trace_mask": missing_trace_mask,
    }


def train_epoch(
    model: DualAnglePINN,
    wavefield_optimizer: torch.optim.Optimizer,
    slope_optimizer: torch.optim.Optimizer,
    full_coordinates: Tensor,
    observed_coordinates: Tensor,
    observed_values: Tensor,
    grid_batch_size: int,
    trace_batch_size: int,
    data_weight: float,
    generator: torch.Generator,
) -> Dict[str, float]:
    """Train once over a random permutation of all collocation points."""
    if grid_batch_size < 1 or trace_batch_size < 1:
        raise ValueError("batch sizes must be positive")

    permutation = torch.randperm(
        full_coordinates.shape[0],
        generator=generator,
        device=full_coordinates.device,
    )
    totals: Dict[str, float] = {
        "total": 0.0,
        "data": 0.0,
        "physics": 0.0,
        "physics_1": 0.0,
        "physics_2": 0.0,
    }
    steps = int(math.ceil(full_coordinates.shape[0] / grid_batch_size))

    for step in range(steps):
        grid_indices = permutation[step * grid_batch_size : (step + 1) * grid_batch_size]
        trace_indices = torch.randint(
            observed_coordinates.shape[0],
            (trace_batch_size,),
            generator=generator,
            device=observed_coordinates.device,
        )
        losses = train_dual_angle_step(
            model=model,
            wavefield_optimizer=wavefield_optimizer,
            slope_optimizer=slope_optimizer,
            grid_coordinates=full_coordinates[grid_indices],
            trace_coordinates=observed_coordinates[trace_indices],
            trace_values=observed_values[trace_indices],
            data_weight=data_weight,
        )
        for name, value in losses.as_floats().items():
            totals[name] += value

    return {name: value / steps for name, value in totals.items()}


def predict_in_batches(
    model: DualAnglePINN,
    coordinates: Tensor,
    batch_size: int,
) -> Tuple[Tensor, Tensor]:
    """Evaluate wavefields and slopes without retaining a full autograd graph."""
    wavefields: List[Tensor] = []
    slopes: List[Tensor] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, coordinates.shape[0], batch_size):
            wavefield_batch, slope_batch = model(coordinates[start : start + batch_size])
            wavefields.append(wavefield_batch.cpu())
            slopes.append(slope_batch.cpu())
    return torch.cat(wavefields), torch.cat(slopes)


def save_checkpoint(
    path: Path,
    epoch: int,
    model: DualAnglePINN,
    wavefield_optimizer: torch.optim.Optimizer,
    slope_optimizer: torch.optim.Optimizer,
    history: Sequence[Dict[str, float]],
    config: Dict[str, object],
) -> None:
    """Atomically save all state required to resume training."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "wavefield_optimizer": wavefield_optimizer.state_dict(),
            "slope_optimizer": slope_optimizer.state_dict(),
            "history": list(history),
            "config": config,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def load_checkpoint(
    path: str,
    model: DualAnglePINN,
    wavefield_optimizer: torch.optim.Optimizer,
    slope_optimizer: torch.optim.Optimizer,
    device: str,
) -> Tuple[int, List[Dict[str, float]]]:
    """Restore model and optimizer state, returning the next epoch number."""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    wavefield_optimizer.load_state_dict(checkpoint["wavefield_optimizer"])
    slope_optimizer.load_state_dict(checkpoint["slope_optimizer"])
    return int(checkpoint["epoch"]) + 1, list(checkpoint.get("history", []))


def write_history(path: Path, history: Iterable[Dict[str, float]]) -> None:
    """Write one averaged loss record per epoch."""
    records = list(history)
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def write_json(path: Path, value: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)


def export_predictions(
    output_directory: Path,
    model: DualAnglePINN,
    full_coordinates: Tensor,
    target: Tensor,
    missing_trace_mask: Tensor,
    shape: Tuple[int, int],
    inference_batch_size: int,
) -> Dict[str, float]:
    """Save component fields, slopes, reconstruction, and interpolation metrics."""
    output_directory.mkdir(parents=True, exist_ok=True)
    wavefields, slopes = predict_in_batches(model, full_coordinates, inference_batch_size)
    wavefields = wavefields.reshape(shape[0], shape[1], 2).numpy()
    slopes = slopes.reshape(shape[0], shape[1], 2).numpy()
    reconstruction = wavefields.sum(axis=2)
    target_array = target.detach().cpu().numpy()
    missing_mask = missing_trace_mask.detach().cpu().numpy()

    np.save(output_directory / "phi_1.npy", wavefields[:, :, 0])
    np.save(output_directory / "phi_2.npy", wavefields[:, :, 1])
    np.save(output_directory / "slope_1.npy", slopes[:, :, 0])
    np.save(output_directory / "slope_2.npy", slopes[:, :, 1])
    np.save(output_directory / "reconstruction.npy", reconstruction)

    observed_mask = ~missing_mask
    metrics = {
        "full_mse": float(np.mean((reconstruction - target_array) ** 2)),
        "observed_mse": float(
            np.mean((reconstruction[observed_mask] - target_array[observed_mask]) ** 2)
        ),
        "missing_mse": float(
            np.mean((reconstruction[missing_mask] - target_array[missing_mask]) ** 2)
        )
        if missing_mask.any()
        else float("nan"),
        "slope_1_min": float(slopes[:, :, 0].min()),
        "slope_1_max": float(slopes[:, :, 0].max()),
        "slope_2_min": float(slopes[:, :, 1].min()),
        "slope_2_max": float(slopes[:, :, 1].max()),
    }
    write_json(output_directory / "metrics.json", metrics)
    return metrics
