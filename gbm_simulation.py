#!/usr/bin/env python3
"""CLI and GUI launcher for regime-switching GBM simulations."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional
import sys

import matplotlib.pyplot as plt
import torch

from gbm import (
    GBMSimulator,
    attach_randomness,
    build_default_config,
    summarize_terminal_distribution,
)
from gbm.visualization import (
    animate_paths,
    animate_streaming_paths,
    plot_sample_paths,
    plot_terminal_distribution,
)
from gbm.ui.interactive import run_interactive_wizard

DEFAULT_DRIFT_STD = (0.015, 0.010, 0.020)
DEFAULT_VOLATILITY_CV = (0.12, 0.18, 0.25)


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
    parser.add_argument(
        "--endless",
        action="store_true",
        help="Run streaming simulation without a fixed horizon (Ctrl+C to stop).",
    )
    parser.add_argument(
        "--drift-std",
        type=float,
        nargs="+",
        default=None,
        help="Per-state drift standard deviations for stochastic calibration.",
    )
    parser.add_argument(
        "--volatility-cv",
        type=float,
        nargs="+",
        default=None,
        help="Per-state volatility coefficients of variation (scale of randomness).",
    )
    parser.add_argument(
        "--deterministic-params",
        action="store_true",
        help="Disable stochastic parameter sampling for drift and volatility.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch a rich interactive CLI wizard to choose simulation options.",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Force the PyQt6 GUI configurator (falls back to CLI if PyQt6 is unavailable).",
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


def expand_sequence(raw: Optional[list[float]], default: tuple[float, ...], length: int, name: str) -> list[float]:
    """Expand or validate a per-state sequence."""
    if raw is None:
        values = list(default)
    else:
        if len(raw) == 1:
            values = [float(raw[0])] * length
        elif len(raw) == length:
            values = [float(v) for v in raw]
        else:
            raise ValueError(f"Expected 1 or {length} values for {name}, got {len(raw)}.")
    return values


def maybe_save_animation(anim, path: Path, log) -> Optional[Path]:
    suffix = path.suffix.lower()
    writer = "pillow" if suffix == ".gif" else "ffmpeg"
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / path.name
    try:
        anim.save(str(target.resolve()), writer=writer, dpi=150)
        log(f"Saved animation: {target}")
        return target
    except (RuntimeError, ValueError) as exc:
        log(f"Failed to save animation ({exc}).")
        return None


def _coerce_path(value: Optional[Path | str]) -> Optional[Path]:
    if value in (None, ""):
        return None
    if isinstance(value, Path):
        return value
    return Path(value)


def execute_simulation(args: argparse.Namespace, *, suppress_output: bool = False) -> dict:
    messages: list[str] = []
    saved_paths: list[Path] = []
    animation_saved: Optional[Path] = None

    def log(message: str = "") -> None:
        messages.append(message)
        if not suppress_output:
            print(message)

    device = resolve_device(args.device)
    dtype = precision_to_dtype(args.precision)

    generator = torch.Generator(device=device)
    manual_seed_or_random(generator, args.seed)

    config = build_default_config(device=device, dtype=dtype)

    if args.deterministic_params:
        drift_std = None
        vol_cv = None
    else:
        n_states = config.transition_matrix.shape[0]
        drift_std = expand_sequence(args.drift_std, DEFAULT_DRIFT_STD, n_states, "drift standard deviation")
        vol_cv = expand_sequence(args.volatility_cv, DEFAULT_VOLATILITY_CV, n_states, "volatility coefficient of variation")
        attach_randomness(config, drift_std=drift_std, volatility_cv=vol_cv)

    simulator = GBMSimulator(config=config, generator=generator)

    animation_obj = None
    animation_fig = None
    figures: list[plt.Figure] = []

    animation_file = _coerce_path(args.animation_file)
    save_dir = args.save_dir.expanduser()

    if args.stream:
        if args.endless and not args.show:
            raise SystemExit("--endless requires --show to visualise the unbounded simulation.")

        stream = simulator.simulate_stream(
            n_paths=args.paths,
            n_steps=args.steps,
            horizon=args.horizon,
            s0=args.s0,
            initial_state=args.initial_state,
            infinite=args.endless,
        )

        if args.endless and animation_file is not None:
            log("Infinite streaming cannot be exported to a file; ignoring --animation-file.")
        stream_animation_file = None if args.endless else animation_file
        animation_target_stream = args.show or (stream_animation_file is not None)

        if args.animate and not animation_target_stream:
            log("Animation requested without --show or --animation-file; skipping animation.")

        if animation_target_stream:
            animation_obj, animation_fig, _ = animate_streaming_paths(
                stream,
                num_paths=args.animation_paths,
                interval_ms=args.animation_interval,
            )
        elif not args.endless:
            stream.ensure_complete()

        if stream_animation_file is not None and animation_obj is not None:
            animation_saved = maybe_save_animation(animation_obj, stream_animation_file, log)

        if args.show and animation_fig is not None:
            plt.show()
        elif animation_fig is not None and not args.show:
            plt.close(animation_fig)

        if not args.endless:
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

        animation_target_batch = args.show or animation_file is not None
        if args.animate and not animation_target_batch:
            log("Animation requested without --show or --animation-file; skipping animation.")
        if animation_target_batch:
            animation_obj, animation_fig, _ = animate_paths(
                result,
                num_paths=args.animation_paths,
                interval_ms=args.animation_interval,
            )
            if animation_file is not None:
                animation_saved = maybe_save_animation(animation_obj, animation_file, log)
        if not args.show and animation_fig is not None:
            plt.close(animation_fig)

    summary = summarize_terminal_distribution(result.prices)

    log("Simulation complete")
    log(f"Device: {device}")
    log(f"Precision: {args.precision}")
    log(f"Paths: {args.paths}")
    log(f"Steps: {args.steps}")
    log(f"Horizon: {fmt(args.horizon)} years")
    log(f"Initial price: {fmt(args.s0)}")
    log("")
    log("Regime drift: " + str(config.drift.cpu().numpy()))
    log("Regime volatility: " + str(config.volatility.cpu().numpy()))
    log("State labels: " + str(config.state_labels))
    randomness = getattr(config, "randomness", None)
    if randomness is not None:
        log("Drift std (per state): " + str(randomness.drift_std.cpu().numpy()))
        log("Volatility CV (per state): " + str(randomness.volatility_cv.cpu().numpy()))
    log("")
    log(f"Terminal mean: {fmt(summary.mean)}")
    log(f"Terminal standard deviation: {fmt(summary.standard_deviation)}")
    log(f"5th percentile: {fmt(summary.quantile_05)}")
    log(f"95th percentile: {fmt(summary.quantile_95)}")
    log(
        "95% CI for mean: ("
        f"{fmt(summary.confidence_interval[0])}, {fmt(summary.confidence_interval[1])})"
    )

    if not args.no_save or args.show:
        fig_paths, ax_paths = plot_sample_paths(result, num_paths=args.plot_paths)
        fig_hist, ax_hist = plot_terminal_distribution(
            result.prices[:, -1],
            bins=args.hist_bins,
        )
        figures.extend([fig_paths, fig_hist])

        if not args.no_save:
            save_dir.mkdir(parents=True, exist_ok=True)
            path_chart = save_dir / "gbm_paths.png"
            path_hist = save_dir / "gbm_terminal_hist.png"
            fig_paths.savefig(path_chart, dpi=150, bbox_inches="tight")
            fig_hist.savefig(path_hist, dpi=150, bbox_inches="tight")
            saved_paths.extend([path_chart, path_hist])
            log("")
            log("Saved:")
            log(f"  {path_chart}")
            log(f"  {path_hist}")

    if args.show:
        plt.show()
    else:
        for fig in figures:
            plt.close(fig)

    return {
        "config": config,
        "result": result,
        "summary": summary,
        "messages": messages,
        "saved_paths": saved_paths,
        "animation_file": animation_saved,
    }


def _qt_available() -> bool:
    try:
        import PyQt6  # noqa: F401
    except ImportError:
        return False
    return True


def main() -> None:
    args = parse_args()

    want_gui = args.gui or (len(sys.argv) == 1 and _qt_available())
    if want_gui:
        try:
            from gbm.ui.qt import launch_qt_interface
        except ImportError:
            if args.gui:
                raise SystemExit("PyQt6 is required for the GUI. Install it with 'pip install PyQt6'.")
        else:
            launch_qt_interface(
                args,
                default_drift_std=DEFAULT_DRIFT_STD,
                default_vol_cv=DEFAULT_VOLATILITY_CV,
                runner=lambda namespace: execute_simulation(namespace, suppress_output=True),
            )
            return

    auto_interactive = len(sys.argv) == 1 and sys.stdin.isatty() and sys.stdout.isatty()
    if getattr(args, "interactive", False) or auto_interactive:
        args = run_interactive_wizard(
            args,
            default_drift_std=DEFAULT_DRIFT_STD,
            default_vol_cv=DEFAULT_VOLATILITY_CV,
        )

    execute_simulation(args)


if __name__ == "__main__":
    main()
