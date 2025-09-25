"""GBM simulation package with regime switching."""
from .config import RegimeConfig, build_default_config
from .markov import RegimeMarkovChain
from .distributions import attach_randomness
from .results import SimulationFrame, SimulationResult
from .simulator import GBMSimulator
from .statistics import MonteCarloSummary, summarize_terminal_distribution
from .streaming import StreamingSimulation

__all__ = [
    "RegimeConfig",
    "RegimeMarkovChain",
    "GBMSimulator",
    "SimulationResult",
    "SimulationFrame",
    "StreamingSimulation",
    "attach_randomness",
    "MonteCarloSummary",
    "summarize_terminal_distribution",
    "build_default_config",
]

