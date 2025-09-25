"""Visualization utilities for GBM simulations."""
from __future__ import annotations

from typing import Tuple

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch

from .results import SimulationResult
from .streaming import StreamingSimulation


__all__ = (
    "plot_sample_paths",
    "plot_terminal_distribution",
    "animate_paths",
    "animate_streaming_paths",
)


def plot_sample_paths(
    result: SimulationResult,
    *,
    num_paths: int = 10,
    ax: plt.Axes | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    num_paths = min(num_paths, result.prices.shape[0])
    time = result.time_grid.detach().cpu().numpy()
    sample = result.prices[:num_paths].detach().cpu().numpy()
    for path in sample:
        ax.plot(time, path, linewidth=1.1, alpha=0.8)
    ax.set_xlabel("Time")
    ax.set_ylabel("Asset price")
    ax.set_title("Sample GBM paths with regime switching")
    ax.grid(True, alpha=0.2)
    return fig, ax


def plot_terminal_distribution(
    terminal_prices: torch.Tensor,
    *,
    bins: int = 60,
    ax: plt.Axes | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    data = terminal_prices.detach().cpu().numpy()
    ax.hist(data, bins=bins, alpha=0.75, color="#1f77b4", edgecolor="black")
    ax.set_xlabel("Terminal price")
    ax.set_ylabel("Frequency")
    ax.set_title("Terminal distribution (Monte Carlo)")
    ax.grid(True, alpha=0.2)
    return fig, ax


def animate_paths(
    result: SimulationResult,
    *,
    num_paths: int = 6,
    interval_ms: int = 40,
) -> tuple[animation.FuncAnimation, plt.Figure, plt.Axes]:
    num_paths = min(num_paths, result.prices.shape[0])
    time = result.time_grid.detach().cpu().numpy()
    sample = result.prices[:num_paths].detach().cpu().numpy()

    fig, ax = plt.subplots()
    lines = [ax.plot([], [], linewidth=1.2, alpha=0.85)[0] for _ in range(num_paths)]

    ax.set_xlabel("Time")
    ax.set_ylabel("Asset price")
    ax.set_title("GBM path animation")
    ax.grid(True, alpha=0.2)

    def init() -> list[plt.Line2D]:
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame: int) -> list[plt.Line2D]:
        slice_time = time[: frame + 1]
        for idx, line in enumerate(lines):
            line.set_data(slice_time, sample[idx, : frame + 1])
        y_min = float(np.min(sample[:, : frame + 1]))
        y_max = float(np.max(sample[:, : frame + 1]))
        if np.isclose(y_min, y_max):
            y_min -= 1.0
            y_max += 1.0
        margin = 0.05 * (y_max - y_min)
        span = slice_time[-1] - slice_time[0]
        if span == 0:
            span = 1.0
        pad = 0.05 * span
        ax.set_xlim(slice_time[0] - pad, slice_time[-1] + pad)
        ax.set_ylim(y_min - margin, y_max + margin)
        return lines

    frames = range(result.prices.shape[1])
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frames,
        interval=interval_ms,
        blit=False,
        repeat=False,
    )
    return anim, fig, ax


def animate_streaming_paths(
    stream: StreamingSimulation,
    *,
    num_paths: int = 6,
    interval_ms: int = 40,
) -> tuple[animation.FuncAnimation, plt.Figure, plt.Axes]:
    actual = stream.set_visual_count(num_paths)

    fig, ax = plt.subplots()
    lines = [ax.plot([], [], linewidth=1.2, alpha=0.85)[0] for _ in range(actual)]

    ax.set_xlabel("Time")
    ax.set_ylabel("Asset price")
    ax.set_title("Real-time GBM path animation")
    ax.grid(True, alpha=0.2)

    def init() -> list[plt.Line2D]:
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame: int) -> list[plt.Line2D]:
        time_slice, sample = stream.get_visual_snapshot(frame)
        available = sample.shape[0]
        for idx, line in enumerate(lines):
            if idx < available:
                line.set_data(time_slice, sample[idx])
                line.set_visible(True)
            else:
                line.set_visible(False)
        if available:
            y_min = float(np.min(sample))
            y_max = float(np.max(sample))
            if np.isclose(y_min, y_max):
                y_min -= 1.0
                y_max += 1.0
            margin = 0.05 * (y_max - y_min)
            span = time_slice[-1] - time_slice[0]
            if span == 0:
                span = 1.0
            pad = 0.05 * span
            ax.set_xlim(time_slice[0] - pad, time_slice[-1] + pad)
            ax.set_ylim(y_min - margin, y_max + margin)
        return lines

    frame_source = count(0) if getattr(stream, "infinite", False) else range(stream.max_steps + 1)
    anim = animation.FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=frame_source,
        interval=interval_ms,
        blit=False,
        repeat=False,
        save_count=None if getattr(stream, "infinite", False) else stream.max_steps + 1,
    )
    return anim, fig, ax

