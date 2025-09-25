"""State-dependent parameter distributions for GBM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .config import _as_tensor, RegimeConfig


@dataclass(frozen=True)
class RegimeRandomness:
    """Randomness specifications for state-conditional drift and volatility."""

    drift_std: torch.Tensor
    volatility_cv: torch.Tensor
    min_volatility: float = 1e-6

    def __post_init__(self) -> None:
        if torch.any(self.drift_std < 0):
            raise ValueError("Drift standard deviations must be non-negative.")
        if torch.any(self.volatility_cv < 0):
            raise ValueError("Volatility coefficients of variation must be non-negative.")
        if self.min_volatility <= 0:
            raise ValueError("Minimum volatility must be positive.")

    @classmethod
    def from_sequences(
        cls,
        *,
        drift_std: Sequence[float],
        volatility_cv: Sequence[float],
        device: torch.device,
        dtype: torch.dtype,
        min_volatility: float = 1e-6,
    ) -> "RegimeRandomness":
        return cls(
            drift_std=torch.as_tensor(drift_std, device=device, dtype=dtype),
            volatility_cv=torch.as_tensor(volatility_cv, device=device, dtype=dtype),
            min_volatility=min_volatility,
        )


def attach_randomness(
    config: RegimeConfig,
    drift_std: Sequence[float] | torch.Tensor,
    volatility_cv: Sequence[float] | torch.Tensor,
    *,
    min_volatility: float = 1e-4,
) -> RegimeConfig:
    """Return a copy of the configuration with randomness parameters attached."""
    drift_std_tensor = _as_tensor(
        drift_std,
        dtype=config.dtype,
        device=config.device,
    )
    vol_cv_tensor = _as_tensor(
        volatility_cv,
        dtype=config.dtype,
        device=config.device,
    )
    randomness = RegimeRandomness(
        drift_std=drift_std_tensor,
        volatility_cv=vol_cv_tensor,
        min_volatility=min_volatility,
    )
    object.__setattr__(config, "randomness", randomness)
    return config


def sample_parameters(
    config: RegimeConfig,
    states: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample drift and volatility conditional on the current regime state."""
    base_drift = config.drift.index_select(0, states)
    base_vol = config.volatility.index_select(0, states)
    randomness: RegimeRandomness | None = getattr(config, "randomness", None)

    if randomness is None:
        return base_drift, base_vol

    drift_std = randomness.drift_std.index_select(0, states)
    vol_cv = randomness.volatility_cv.index_select(0, states)

    drift_noise = torch.randn(
        base_drift.shape,
        device=config.device,
        dtype=config.dtype,
        generator=generator,
    ) * drift_std
    drift_sample = base_drift + drift_noise

    # Convert coefficient of variation to log-normal parameters.
    variance_factor = vol_cv.pow(2) + 1.0
    sigma_ln = torch.sqrt(torch.log(torch.clamp(variance_factor, min=1.0)))
    mu_ln = torch.log(torch.clamp(base_vol, min=randomness.min_volatility)) - 0.5 * sigma_ln.pow(2)
    lognormal_noise = torch.exp(
        torch.randn(
            base_vol.shape,
            device=config.device,
            dtype=config.dtype,
            generator=generator,
        )
        * sigma_ln
        + mu_ln
    )
    volatility_sample = torch.clamp(lognormal_noise, min=randomness.min_volatility)

    return drift_sample, volatility_sample
