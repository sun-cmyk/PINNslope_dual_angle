"""Small CPU-friendly dual-angle interpolation example."""

import argparse
import math

import torch

from pinnslope.dual_angle import DualAnglePINN, train_dual_angle_step


def analytic_components(coordinates):
    x = coordinates[:, 0]
    t = coordinates[:, 1]
    phi_1 = torch.sin(2.0 * math.pi * 5.0 * (t - 0.35 * x))
    phi_2 = 0.7 * torch.sin(2.0 * math.pi * 4.0 * (t + 0.30 * x))
    return torch.stack((phi_1, phi_2), dim=1)


def coordinate_grid(nx, nt, device):
    x = torch.linspace(0.0, 1.0, nx, device=device)
    t = torch.linspace(0.0, 1.0, nt, device=device)
    x_mesh = x.reshape(-1, 1).expand(nx, nt)
    t_mesh = t.reshape(1, -1).expand(nx, nt)
    return torch.stack((x_mesh, t_mesh), dim=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    torch.manual_seed(42)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    grid = coordinate_grid(nx=31, nt=51, device=device)
    full_coordinates = grid.reshape(-1, 2)
    trace_coordinates = grid[::4].reshape(-1, 2)
    trace_values = analytic_components(trace_coordinates).sum(dim=1, keepdim=True)

    model = DualAnglePINN(
        wavefield_hidden=(64, 64, 64),
        slope_hidden=(8, 8, 8, 8),
        positional_encoding=(4, 8, 2),
        slope_shifts=(0.05, -0.8),
        slope_scales=(0.75, 0.75),
        device=device,
    )
    wavefield_optimizer = torch.optim.Adam(
        model.wavefield_network.parameters(), lr=1e-3
    )
    slope_optimizer = torch.optim.Adam(model.slope_network.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        grid_indices = torch.randint(
            full_coordinates.shape[0], (512,), device=full_coordinates.device
        )
        trace_indices = torch.randint(
            trace_coordinates.shape[0], (256,), device=trace_coordinates.device
        )
        losses = train_dual_angle_step(
            model=model,
            wavefield_optimizer=wavefield_optimizer,
            slope_optimizer=slope_optimizer,
            grid_coordinates=full_coordinates[grid_indices],
            trace_coordinates=trace_coordinates[trace_indices],
            trace_values=trace_values[trace_indices],
            data_weight=100.0,
        )
        if epoch % 25 == 0 or epoch == args.epochs - 1:
            print("epoch {:4d}: {}".format(epoch, losses.as_floats()))

    model.eval()
    with torch.no_grad():
        wavefields, slopes = model(full_coordinates)
        reconstruction = wavefields.sum(dim=1)
        target = analytic_components(full_coordinates).sum(dim=1)
        mse = torch.mean((reconstruction - target).square()).item()
        slope_min = slopes.min(dim=0).values.tolist()
        slope_max = slopes.max(dim=0).values.tolist()

    print("full-grid reconstruction MSE: {:.6e}".format(mse))
    print("predicted slope ranges: min={}, max={}".format(slope_min, slope_max))


if __name__ == "__main__":
    main()
