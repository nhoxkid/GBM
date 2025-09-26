# GBM Monte Carlo Studio

A PyTorch-powered laboratory for regime-switching geometric Brownian motion (GBM).
The app supports fast Monte Carlo batches, real-time streaming, and a PyQt6 front end
styled after modern Apple UI patterns.

## Requirements
- like duck 
- Python 3.10 or newer
- PyTorch 2.x with CUDA if you want GPU acceleration
- Matplotlib, NumPy
- PyQt6 for the desktop interface

```bash
pip install -r requirements.txt  # create one or install packages individually
```

## Quick Start

### CLI run

```bash
python3 gbm_simulation.py --paths 10000 --steps 252 --show
```

The CLI prints Monte Carlo summary statistics and optionally opens Matplotlib plots.

### GUI run

```bash
python3 gbm_simulation.py --gui
```

You will be greeted by the GBM Monte Carlo Studio window. Use **Run Simulation** for
classic Monte Carlo batches or **Stream in Real Time** for incremental Ito updates.

## UI Field Reference

| Control | Purpose | Should do | Avoid |
| --- | --- | --- | --- |
| Monte Carlo paths | Number of simulated trajectories. | Keep 1e3-5e4 for quick iteration; scale higher on GPU. | Millions of paths on CPU (slow). |
| Time steps | Discrete Ito steps per trajectory. | Match trading days (252) or refine for intraday studies. | Zero or very coarse values (loss of accuracy). |
| Time horizon (years) | Total span of the simulation. | Use fractions for months (for example 0.5). | Non-positive values; Ito requires dt > 0. |
| Initial price | Starting asset level S0. | Match your asset's current price. | Leaving at zero unless modeling defaults. |
| Random seed | Deterministic RNG. | Set for reproducible studies; blank for fresh randomness. | Non-integer values. |
| Initial state index | Starting Markov regime. | 0 = bull, 1 = neutral, 2 = bear. | Index >= number of regimes. |
| Device | PyTorch device selection. | auto is safest; cuda when you know GPU is ready. | Forcing cuda without CUDA drivers. |
| Precision | Simulation dtype. | float64 for accuracy, float32 for speed. | Switching mid-run; restart instead. |
| Stream paths in real time | Enable incremental evaluation. | Pair with Stream in Real Time button and watch live paths. | Expecting animation export while streaming endlessly. |
| Run without fixed horizon | Infinite stream flag. | Use for live demos; stop with Stop Streaming. | Assuming it will auto-stop or save files. |
| Capture animation frames for static runs | Record Matplotlib animation when not streaming. | Combine with --show or an animation file path. | Enabling without a display or save target (skip warning). |
| Show Matplotlib windows | Display static figures. | Enable for manual inspection. | Forgetting to disable when only filesystem output is needed. |
| Paths in static view | Number of trajectories in static plot. | Keep <= 100 for legible graphs. | Huge counts that hide detail. |
| Histogram bins | Terminal distribution resolution. | 40-80 bins typically; increase for very large samples. | >200 bins with few paths (noisy). |
| Paths in live stream | Lines rendered during streaming. | Pick a small subset (5-20) for smooth animation. | Requesting more than simulated paths. |
| Frame interval (ms) | QTimer cadence for streaming. | 30-100 ms keeps animation smooth. | <10 ms on weaker CPUs (stutter). |
| Save static figures to disk | Toggle filesystem export. | Default writes to figures/. | Disabling and expecting PNG outputs. |
| Save directory | Destination folder. | Choose writable path; relative paths resolve from repo root. | Read-only or network paths without access. |
| Animation file | File to store rendered animation. | .gif uses Pillow, .mp4 uses ffmpeg. | Supplying when endless streaming is enabled (ignored). |
| Enable stochastic drift/volatility | Turn on per-state randomness. | Provide drift std and volatility CV lists (1 value broadcasts). | Mismatched counts; must be 1 or number of regimes. |
| Drift std per state | Spread of normal sampling around drift. | Example: 0.015 0.010 0.020. | Negative values. |
| Volatility CV per state | Coefficient of variation for lognormal vol sampling. | Example: 0.12 0.18 0.25. | Very large values (>2) causing explosive variance. |

## Streaming Notes

- Streaming uses a dedicated `StreamingSimulation` that steps the Ito process with a Qt timer.
  The canvas redraws each frame and the controller emits status messages.
- Endless streams never finish automatically. Use Stop Streaming or close the window to
  release resources.
- Saving animations while streaming is only possible for finite horizons.

## Monte Carlo Output

The CLI (and GUI output pane) reports:

- Drift/volatility per regime, including stochastic parameters when enabled.
- Terminal mean, standard deviation, 5th/95th percentiles, and a 95% confidence interval for the mean.
- Saved asset-path and histogram PNGs if exporting is on.

## Packaging as an EXE (PyInstaller)

1. Create an isolated environment and install requirements plus PyInstaller.
2. Run:
   ```bash
   pyinstaller gbm_simulation.py \
     --name "GBMStudio" \
     --noconsole \
     --collect-all PyQt6 \
     --collect-all matplotlib \
     --collect-all torch \
     --hidden-import matplotlib.backends.backend_qtagg \
     --add-data "gbm;gbm"
   ```
3. Distribute the folder at `dist/GBMStudio/`; the executable relies on bundled Qt plugins.
4. Edit the generated spec file to set an icon, splash screen, or tweak resources as needed.

## Contributing

- The codebase is split into `gbm/` for reusable logic and `gbm/ui/` for the front end.
- Inline comments explain each public helper, and the runtime module aggregates configuration
  so both CLI and GUI share the same setup.
- Preferred workflow: fork, create a feature branch, run `python3 -m compileall gbm gbm_simulation.py`
  before submitting pull requests.

