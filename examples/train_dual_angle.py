"""Run a checkpointed dual-angle seismic interpolation experiment."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from pinnslope.dual_angle import DualAnglePINN
from pinnslope.dual_angle_training import (
    export_predictions,
    load_checkpoint,
    load_gather,
    normalize_slope_interval,
    prepare_interpolation_data,
    save_checkpoint,
    train_epoch,
    write_history,
    write_json,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/synth_data/data_dense.npy")
    parser.add_argument("--orientation", choices=("x-t", "t-x"), default="x-t")
    parser.add_argument("--output-dir", default="runs/synth_dual_angle")
    parser.add_argument("--x-start", type=int, default=0)
    parser.add_argument("--x-stop", type=int, default=119)
    parser.add_argument("--t-start", type=int, default=40)
    parser.add_argument("--t-stop", type=int, default=251)
    parser.add_argument("--dx", type=float, default=10.0)
    parser.add_argument("--dt", type=float, default=0.004)
    parser.add_argument("--trace-step", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--grid-batch-size", type=int, default=1000)
    parser.add_argument("--trace-batch-size", type=int, default=0)
    parser.add_argument("--data-weight", type=float, default=1000.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--positive-slope", type=float, nargs=2, default=(0.05, 0.8))
    parser.add_argument("--negative-slope", type=float, nargs=2, default=(-0.8, -0.05))
    parser.add_argument(
        "--slope-units", choices=("normalized", "physical"), default="normalized"
    )
    parser.add_argument("--wavefield-width", type=int, default=512)
    parser.add_argument("--wavefield-depth", type=int, default=4)
    parser.add_argument("--slope-width", type=int, default=8)
    parser.add_argument("--slope-depth", type=int, default=4)
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--inference-batch-size", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", default=None)
    return parser.parse_args()


def validate_arguments(args):
    if args.epochs < 1:
        raise ValueError("epochs must be positive")
    if args.dx <= 0.0 or args.dt <= 0.0:
        raise ValueError("dx and dt must be positive")
    if args.positive_slope[0] >= args.positive_slope[1]:
        raise ValueError("positive slope bounds must be increasing")
    if args.negative_slope[0] >= args.negative_slope[1]:
        raise ValueError("negative slope bounds must be increasing")
    if args.positive_slope[0] < 0.0 or args.negative_slope[1] > 0.0:
        raise ValueError("positive/negative slope intervals must not cross zero")
    positive_integer_names = (
        "trace_step",
        "grid_batch_size",
        "wavefield_width",
        "wavefield_depth",
        "slope_width",
        "slope_depth",
        "checkpoint_every",
        "log_every",
        "inference_batch_size",
    )
    for name in positive_integer_names:
        if getattr(args, name) < 1:
            raise ValueError("{} must be positive".format(name))


def main():
    args = parse_arguments()
    validate_arguments(args)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    gather = load_gather(
        args.data,
        orientation=args.orientation,
        x_slice=(args.x_start, args.x_stop),
        t_slice=(args.t_start, args.t_stop),
    )
    data = prepare_interpolation_data(gather, args.trace_step, device)
    trace_batch_size = args.trace_batch_size or max(
        1, data["observed_coordinates"].shape[0] // 2
    )

    x_extent = (gather.shape[0] - 1) * args.dx
    t_extent = (gather.shape[1] - 1) * args.dt
    positive_min, positive_max = normalize_slope_interval(
        args.positive_slope, args.slope_units, x_extent, t_extent
    )
    negative_min, negative_max = normalize_slope_interval(
        args.negative_slope, args.slope_units, x_extent, t_extent
    )
    model = DualAnglePINN(
        wavefield_hidden=(args.wavefield_width,) * args.wavefield_depth,
        slope_hidden=(args.slope_width,) * args.slope_depth,
        positional_encoding=(8, 32, 2),
        slope_shifts=(positive_min, negative_min),
        slope_scales=(positive_max - positive_min, negative_max - negative_min),
        device=device,
    )
    wavefield_optimizer = torch.optim.Adam(
        model.wavefield_network.parameters(), lr=args.learning_rate
    )
    slope_optimizer = torch.optim.Adam(
        model.slope_network.parameters(), lr=args.learning_rate
    )

    output_directory = Path(args.output_dir)
    checkpoint_directory = output_directory / "checkpoints"
    output_directory.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(
        {
            "data": str(Path(args.data).resolve()),
            "device": device,
            "gather_shape": list(gather.shape),
            "trace_batch_size": trace_batch_size,
            "coordinate_order": ["x", "t"],
            "coordinates_normalized": True,
            "x_extent": x_extent,
            "t_extent": t_extent,
            "normalized_positive_slope": [positive_min, positive_max],
            "normalized_negative_slope": [negative_min, negative_max],
        }
    )
    latest_checkpoint = output_directory / "checkpoints" / "latest.pt"
    if latest_checkpoint.exists() and not args.resume:
        raise RuntimeError(
            "output directory already contains a checkpoint; pass --resume or use "
            "a different --output-dir"
        )
    write_json(output_directory / "config.json", config)

    start_epoch = 0
    history = []
    if args.resume:
        start_epoch, history = load_checkpoint(
            args.resume,
            model,
            wavefield_optimizer,
            slope_optimizer,
            device,
        )

    generator_device = "cuda" if device.startswith("cuda") else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(args.seed)
    started = time.time()
    print(json.dumps(config, indent=2, sort_keys=True), flush=True)

    for epoch in range(start_epoch, args.epochs):
        losses = train_epoch(
            model=model,
            wavefield_optimizer=wavefield_optimizer,
            slope_optimizer=slope_optimizer,
            full_coordinates=data["full_coordinates"],
            observed_coordinates=data["observed_coordinates"],
            observed_values=data["observed_values"],
            grid_batch_size=args.grid_batch_size,
            trace_batch_size=trace_batch_size,
            data_weight=args.data_weight,
            generator=generator,
        )
        record = {"epoch": epoch, **losses}
        history.append(record)

        if epoch % args.log_every == 0 or epoch == args.epochs - 1:
            elapsed_minutes = (time.time() - started) / 60.0
            print(
                "epoch={:04d} total={:.6e} data={:.6e} physics={:.6e} "
                "elapsed_min={:.2f}".format(
                    epoch,
                    losses["total"],
                    losses["data"],
                    losses["physics"],
                    elapsed_minutes,
                ),
                flush=True,
            )

        should_checkpoint = (
            (epoch + 1) % args.checkpoint_every == 0 or epoch == args.epochs - 1
        )
        if should_checkpoint:
            numbered_path = checkpoint_directory / "epoch_{:04d}.pt".format(epoch + 1)
            save_checkpoint(
                numbered_path,
                epoch,
                model,
                wavefield_optimizer,
                slope_optimizer,
                history,
                config,
            )
            save_checkpoint(
                checkpoint_directory / "latest.pt",
                epoch,
                model,
                wavefield_optimizer,
                slope_optimizer,
                history,
                config,
            )
            write_history(output_directory / "losses.csv", history)

    metrics = export_predictions(
        output_directory=output_directory,
        model=model,
        full_coordinates=data["full_coordinates"],
        target=data["target"],
        missing_trace_mask=data["missing_trace_mask"],
        shape=gather.shape,
        inference_batch_size=args.inference_batch_size,
    )
    print("metrics={}".format(json.dumps(metrics, sort_keys=True)), flush=True)


if __name__ == "__main__":
    main()
