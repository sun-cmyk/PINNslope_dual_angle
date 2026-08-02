"""Plot exported dual-angle interpolation results."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pinnslope.dual_angle_training import load_gather


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def seismic_limits(data):
    limit = float(np.percentile(np.abs(data), 99.0))
    return -limit, limit


def add_image(axis, data, title, extent, cmap, limits=None):
    image = axis.imshow(
        data.T,
        aspect="auto",
        origin="upper",
        extent=extent,
        cmap=cmap,
        vmin=None if limits is None else limits[0],
        vmax=None if limits is None else limits[1],
    )
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("t")
    return image


def main():
    args = parse_arguments()
    run_directory = Path(args.run_dir)
    output_directory = Path(args.output_dir) if args.output_dir else run_directory / "figures"
    output_directory.mkdir(parents=True, exist_ok=True)

    with (run_directory / "config.json").open(encoding="utf-8") as stream:
        config = json.load(stream)
    with (run_directory / "metrics.json").open(encoding="utf-8") as stream:
        metrics = json.load(stream)

    target = load_gather(
        config["data"],
        orientation=config["orientation"],
        x_slice=(config["x_start"], config["x_stop"]),
        t_slice=(config["t_start"], config["t_stop"]),
    )
    reconstruction = np.load(run_directory / "reconstruction.npy")
    phi_1 = np.load(run_directory / "phi_1.npy")
    phi_2 = np.load(run_directory / "phi_2.npy")
    slope_1 = np.load(run_directory / "slope_1.npy")
    slope_2 = np.load(run_directory / "slope_2.npy")

    nx, nt = target.shape
    x_min = config["x_start"] * config["dx"]
    x_max = (config["x_start"] + nx - 1) * config["dx"]
    t_min = config["t_start"] * config["dt"]
    t_max = (config["t_start"] + nt - 1) * config["dt"]
    extent = (x_min, x_max, t_max, t_min)
    amplitude_limits = seismic_limits(target)

    observed = np.full_like(target, np.nan)
    observed[:: config["trace_step"]] = target[:: config["trace_step"]]
    seismic_cmap = plt.get_cmap("gray").copy()
    seismic_cmap.set_bad("white")

    figure, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    add_image(axes[0, 0], target, "Dense reference", extent, seismic_cmap, amplitude_limits)
    add_image(axes[0, 1], observed, "Observed traces", extent, seismic_cmap, amplitude_limits)
    add_image(
        axes[1, 0],
        reconstruction,
        "Dual-angle reconstruction",
        extent,
        seismic_cmap,
        amplitude_limits,
    )
    error_limits = seismic_limits(reconstruction - target)
    error_image = add_image(
        axes[1, 1],
        reconstruction - target,
        "Reconstruction error",
        extent,
        "RdBu_r",
        error_limits,
    )
    figure.colorbar(error_image, ax=axes[1, 1], shrink=0.8)
    figure.suptitle(
        "Dual-angle interpolation | missing-trace MSE={:.4e}".format(
            metrics["missing_mse"]
        )
    )
    figure.savefig(output_directory / "interpolation_overview.png", dpi=args.dpi)
    plt.close(figure)

    component_limits = seismic_limits(np.concatenate((phi_1.ravel(), phi_2.ravel())))
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    component_1 = add_image(axes[0], phi_1, "Wavefield component phi_1", extent, "gray", component_limits)
    add_image(axes[1], phi_2, "Wavefield component phi_2", extent, "gray", component_limits)
    figure.colorbar(component_1, ax=axes, shrink=0.8)
    figure.savefig(output_directory / "wavefield_components.png", dpi=args.dpi)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    slope_image_1 = add_image(axes[0], slope_1, "Positive slope sigma_1", extent, "viridis")
    slope_image_2 = add_image(axes[1], slope_2, "Negative slope sigma_2", extent, "viridis")
    figure.colorbar(slope_image_1, ax=axes[0], shrink=0.8)
    figure.colorbar(slope_image_2, ax=axes[1], shrink=0.8)
    figure.savefig(output_directory / "estimated_slopes.png", dpi=args.dpi)
    plt.close(figure)

    history = np.genfromtxt(run_directory / "losses.csv", delimiter=",", names=True)
    figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
    axis.semilogy(history["epoch"], history["total"], label="total")
    axis.semilogy(history["epoch"], history["data"], label="data")
    axis.semilogy(history["epoch"], history["physics"], label="physics")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Loss")
    axis.set_title("Training losses")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.savefig(output_directory / "training_losses.png", dpi=args.dpi)
    plt.close(figure)

    print(output_directory)


if __name__ == "__main__":
    main()
