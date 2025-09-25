"""Monte Carlo summary statistics."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass
class MonteCarloSummary:
    mean: float
    standard_deviation: float
    quantile_05: float
    quantile_95: float
    confidence_interval: tuple[float, float]


def summarize_terminal_distribution(prices: torch.Tensor) -> MonteCarloSummary:
    terminal = prices[:, -1]
    mean = terminal.mean()
    std = terminal.std(unbiased=True)
    quantiles = torch.quantile(
        terminal,
        torch.tensor([0.05, 0.95], device=terminal.device, dtype=terminal.dtype),
    )
    stderr = std / math.sqrt(prices.shape[0])
    ci_low = mean - 1.96 * stderr
    ci_high = mean + 1.96 * stderr
    return MonteCarloSummary(
        mean=float(mean.cpu()),
        standard_deviation=float(std.cpu()),
        quantile_05=float(quantiles[0].cpu()),
        quantile_95=float(quantiles[1].cpu()),
        confidence_interval=(float(ci_low.cpu()), float(ci_high.cpu())),
    )
