"""Visualization utilities for GBM simulations."""
from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt
import torch

from .simulator import SimulationResult


def plot_sample_paths(
    result: SimulationResult,
    *,
    num_paths: int = 10,
    ax: plt.Axes | None = None,
) -> Tuple[plt.Figure, plt.Axes]:
    created = False
    if ax is None:
        fig, ax = plt.subplots()
        created = True
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
    created = False
    if ax is None:
        fig, ax = plt.subplots()
        created = True
    else:
        fig = ax.figure

    data = terminal_prices.detach().cpu().numpy()
    ax.hist(data, bins=bins, alpha=0.75, color="#1f77b4", edgecolor="black")
    ax.set_xlabel("Terminal price")
    ax.set_ylabel("Frequency")
    ax.set_title("Terminal distribution (Monte Carlo)")
    ax.grid(True, alpha=0.2)
    return fig, ax
