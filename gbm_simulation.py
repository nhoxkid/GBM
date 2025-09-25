#!/usr/bin/env python3
"""CLI for PyTorch-based regime-switching GBM simulation."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch

from gbm import (
    GBMSimulator,
    build_default_config,
    summarize_terminal_distribution,
)
from gbm.visualization import (
    animate_paths,
    animate_streaming_paths,
    plot_sample_paths,
    plot_terminal_distribution,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate regime-switching GBM paths with PyTorch acceleration and visualization.",
    )
    parser.add_argument("--paths", type=int, default=10000, help="Number of Monte Carlo paths.")
    parser.add_argument("--steps", type=int, default=252, help="Number of time steps per path.")
    parser.add_argument("--horizon", type=float, default=1.0, help="Time horizon in years.")
    parser.add_argument("--s0", type=float, default=100.0, help="Initial asset price.")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed.")
    parser.add_argument(
        "--initial-state",
        type=int,
        default=0,
        help="Starting Markov state index (0-based).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="PyTorch device: auto, cpu, cuda, or explicit device string.",
    )
    parser.add_argument(
        "--precision",
        choices=("float32", "float64"),
        default="float64",
        help="Floating point precision for simulation.",
    )
    parser.add_argument(
        "--plot-paths",
        type=int,
        default=12,
        help="Number of sample paths to include in the line chart.",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("figures"),
        help="Directory where plots are saved (if not disabled).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving plot images to disk.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively after simulation.",
    )
    parser.add_argument(
        "--hist-bins",
        type=int,
        default=60,
        help="Number of bins for the terminal distribution histogram.",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Animate a subset of paths (streaming uses real-time updates).",
    )
    parser.add_argument(
        "--animation-paths",
        type=int,
        default=6,
        help="Number of paths to animate.",
    )
    parser.add_argument(
        "--animation-interval",
        type=int,
        default=40,
        help="Animation frame interval in milliseconds.",
    )
    parser.add_argument(
        "--animation-file",
        type=Path,
        default=None,
        help="Optional path to save the animation (gif/mp4).",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable real-time streaming simulation (drives animation as it computes).",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device


def precision_to_dtype(precision: str) -> torch.dtype:
    return torch.float64 if precision == "float64" else torch.float32


def manual_seed_or_random(generator: torch.Generator, seed: Optional[int]) -> None:
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.manual_seed(torch.seed())


def fmt(value: float) -> str:
    return f"{value:.6f}"


def maybe_save_animation(anim, path: Path) -> None:
    suffix = path.suffix.lower()
    writer = "pillow" if suffix == ".gif" else "ffmpeg"
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / path.name
    try:
        anim.save(str(target.resolve()), writer=writer, dpi=150)
        print(f"\nSaved animation: {target}")
    except (RuntimeError, ValueError) as exc:
        print(f"\nFailed to save animation ({exc}).")


def main() -> None:

    args = parse_args()
    device = resolve_device(args.device)
    dtype = precision_to_dtype(args.precision)

    generator = torch.Generator(device=device)
    manual_seed_or_random(generator, args.seed)

    config = build_default_config(device=device, dtype=dtype)
    simulator = GBMSimulator(config=config, generator=generator)

    result = None
    animation_obj = None
    animation_fig = None

    if args.stream:
        stream = simulator.simulate_stream(
            n_paths=args.paths,
            n_steps=args.steps,
            horizon=args.horizon,
            s0=args.s0,
            initial_state=args.initial_state,
        )
        want_animation = args.animate or args.animation_file is not None or args.show
        if want_animation:
            animation_obj, animation_fig, _ = animate_streaming_paths(
                stream,
                num_paths=args.animation_paths,
                interval_ms=args.animation_interval,
            )
        else:
            stream.ensure_complete()

        if args.animation_file is not None and animation_obj is not None:
            maybe_save_animation(animation_obj, args.animation_file)

        if args.show and animation_fig is not None:
            plt.show()
        elif animation_fig is not None and not args.show:
            plt.close(animation_fig)

        stream.ensure_complete()
        result = stream.to_result()
    else:
        result = simulator.simulate(
            n_paths=args.paths,
            n_steps=args.steps,
            horizon=args.horizon,
            s0=args.s0,
            initial_state=args.initial_state,
        )

        want_animation = args.animate or args.animation_file is not None
        if want_animation or args.show:
            animation_obj, animation_fig, _ = animate_paths(
                result,
                num_paths=args.animation_paths,
                interval_ms=args.animation_interval,
            )

        if args.animation_file is not None and animation_obj is not None:
            maybe_save_animation(animation_obj, args.animation_file)

    summary = summarize_terminal_distribution(result.prices)

    print("Simulation complete")
    print(f"Device: {device}")
    print(f"Precision: {args.precision}")
    print(f"Paths: {args.paths}")
    print(f"Steps: {args.steps}")
    print(f"Horizon: {fmt(args.horizon)} years")
    print(f"Initial price: {fmt(args.s0)}")
    print("")
    print("Regime drift:", config.drift.cpu().numpy())
    print("Regime volatility:", config.volatility.cpu().numpy())
    print("State labels:", config.state_labels)
    print("")
    print(f"Terminal mean: {fmt(summary.mean)}")
    print(f"Terminal standard deviation: {fmt(summary.standard_deviation)}")
    print(f"5th percentile: {fmt(summary.quantile_05)}")
    print(f"95th percentile: {fmt(summary.quantile_95)}")
    print(
        "95% CI for mean: ("
        f"{fmt(summary.confidence_interval[0])}, {fmt(summary.confidence_interval[1])})"
    )

    figures: list[plt.Figure] = []
    if not args.no_save or args.show:
        fig_paths, _ = plot_sample_paths(result, num_paths=args.plot_paths)
        fig_hist, _ = plot_terminal_distribution(
            result.prices[:, -1],
            bins=args.hist_bins,
        )
        figures.extend([fig_paths, fig_hist])

        if not args.no_save:
            args.save_dir.mkdir(parents=True, exist_ok=True)
            path_chart = args.save_dir / "gbm_paths.png"
            path_hist = args.save_dir / "gbm_terminal_hist.png"
            fig_paths.savefig(path_chart, dpi=150, bbox_inches="tight")
            fig_hist.savefig(path_hist, dpi=150, bbox_inches="tight")
            print("")
            print("Saved:")
            print(f"  {path_chart}")
            print(f"  {path_hist}")

    if args.show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)
        if animation_fig is not None and not args.stream:
            plt.close(animation_fig)


if __name__ == "__main__":
    main()




