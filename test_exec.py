import argparse
from types import SimpleNamespace
from pathlib import Path

from gbm_simulation import execute_simulation

ns = SimpleNamespace(
    paths=10000,
    steps=252,
    horizon=1.0,
    s0=100.0,
    seed=42,
    initial_state=0,
    device='auto',
    precision='float64',
    plot_paths=12,
    save_dir=Path('figures'),
    no_save=True,
    show=False,
    hist_bins=60,
    animate=False,
    animation_paths=6,
    animation_interval=40,
    animation_file=None,
    stream=False,
    endless=False,
    drift_std=None,
    volatility_cv=None,
    deterministic_params=False,
    interactive=False,
    gui=False,
)

res = execute_simulation(ns, suppress_output=True)
summary = res['summary']
print(summary)
print(res['result'].prices.shape)
print(res['result'].prices[:, -1].mean(), res['result'].prices[:, -1].std())
