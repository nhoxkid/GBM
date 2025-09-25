"""Rich-powered interactive wizard for configuring GBM simulations."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.theme import Theme

_THEME = Theme(
    {
        "accent": "bright_cyan",
        "muted": "grey70",
        "warning": "gold1",
        "success": "spring_green2",
    }
)

_console = Console(theme=_THEME)


def _prompt_int(message: str, default: int, *, minimum: Optional[int] = None) -> int:
    while True:
        response = Prompt.ask(message, default=str(default), console=_console)
        try:
            value = int(response)
        except ValueError:
            _console.print("[warning]Please enter a whole number.[/warning]")
            continue
        if minimum is not None and value < minimum:
            _console.print(f"[warning]Value must be at least {minimum}.[/warning]")
            continue
        return value


def _prompt_float(message: str, default: float, *, minimum: Optional[float] = None) -> float:
    while True:
        response = Prompt.ask(message, default=f"{default}", console=_console)
        try:
            value = float(response)
        except ValueError:
            _console.print("[warning]Please enter a numeric value.[/warning]")
            continue
        if minimum is not None and value < minimum:
            _console.print(f"[warning]Value must be at least {minimum}.[/warning]")
            continue
        return value


def _prompt_optional_int(message: str, default: Optional[int]) -> Optional[int]:
    default_label = "none" if default is None else str(default)
    response = Prompt.ask(message, default=default_label, console=_console)
    if response.strip().lower() in {"", "none", "null"}:
        return None
    try:
        return int(response)
    except ValueError:
        _console.print("[warning]Invalid integer; falling back to default.[/warning]")
        return default


def _prompt_choice(message: str, choices: Iterable[str], default: str) -> str:
    choice_list = list(choices)
    response = Prompt.ask(
        message,
        choices=choice_list,
        default=default,
        console=_console,
        show_choices=True,
    )
    return response


def _prompt_float_sequence(
    message: str,
    default: Iterable[float],
    length: int,
) -> list[float]:
    default_str = ", ".join(f"{value:.4f}" for value in default)
    while True:
        response = Prompt.ask(message, default=default_str, console=_console)
        tokens = [token for token in response.replace(",", " ").split() if token]
        try:
            values = [float(token) for token in tokens] if tokens else list(default)
        except ValueError:
            _console.print("[warning]Please provide numeric values separated by spaces or commas.[/warning]")
            continue
        if len(values) == 1:
            return [values[0]] * length
        if len(values) != length:
            _console.print(f"[warning]Enter either 1 value or {length} values.[/warning]")
            continue
        return values


def _summarise_configuration(data: dict[str, str]) -> None:
    table = Table(title="Configuration Summary", show_lines=False, expand=True)
    table.add_column("Setting", style="accent", no_wrap=True)
    table.add_column("Value", style="muted")
    for key, value in data.items():
        table.add_row(key, value)
    _console.print(table)


def run_interactive_wizard(
    args: argparse.Namespace,
    *,
    default_drift_std: Iterable[float],
    default_vol_cv: Iterable[float],
) -> argparse.Namespace:
    _console.print(Panel.fit("[accent bold]GBM Monte Carlo Configurator[/accent bold]", border_style="accent"))
    _console.print(
        "Use the prompts below to tailor the simulation. Press [accent]<enter>[/accent] to accept defaults.",
        style="muted",
    )

    default_drift_std = list(default_drift_std)
    default_vol_cv = list(default_vol_cv)
    n_states = len(default_drift_std) or 1

    paths = _prompt_int("Monte Carlo paths", args.paths, minimum=1)
    steps = _prompt_int("Time steps", args.steps, minimum=1)
    horizon = _prompt_float("Time horizon (years)", args.horizon, minimum=1e-6)
    s0 = _prompt_float("Initial asset price", args.s0, minimum=0.0)
    seed = _prompt_optional_int("Random seed (or 'none')", args.seed)
    initial_state = _prompt_int("Initial Markov state index", args.initial_state, minimum=0)

    device = _prompt_choice("Computation device", ["auto", "cpu", "cuda"], args.device)
    precision = _prompt_choice("Floating point precision", ["float64", "float32"], args.precision)

    stream = Confirm.ask("Stream paths in real time?", default=args.stream, console=_console)
    endless = False
    if stream:
        endless = Confirm.ask(
            "Run streaming without a fixed horizon (Ctrl+C to stop)?",
            default=args.endless,
            console=_console,
        )

    animate = Confirm.ask("Animate sample paths?", default=args.animate or stream, console=_console)
    animation_paths = args.animation_paths
    animation_interval = args.animation_interval
    if animate:
        animation_paths = _prompt_int(
            "Paths to animate",
            min(args.animation_paths, paths),
            minimum=1,
        )
        animation_interval = _prompt_int(
            "Animation frame interval (ms)",
            args.animation_interval,
            minimum=1,
        )

    show = Confirm.ask("Show plots/animation window?", default=args.show or animate, console=_console)

    save_plots = Confirm.ask("Save static plots to disk?", default=not args.no_save, console=_console)
    if save_plots:
        save_dir = Path(Prompt.ask("Directory for saved plots", default=str(args.save_dir), console=_console)).expanduser()
        no_save = False
    else:
        save_dir = args.save_dir
        no_save = True

    deterministic_params = not Confirm.ask(
        "Let drift/volatility fluctuate stochastically per state?",
        default=not args.deterministic_params,
        console=_console,
    )

    drift_std_values: Optional[list[float]] = None
    vol_cv_values: Optional[list[float]] = None
    if not deterministic_params:
        drift_std_values = _prompt_float_sequence(
            "Drift standard deviation by state",
            default_drift_std,
            n_states,
        )
        vol_cv_values = _prompt_float_sequence(
            "Volatility coefficient of variation by state",
            default_vol_cv,
            n_states,
        )

    animation_file: Optional[Path] = None
    if animate:
        save_animation = Confirm.ask(
            "Export animation to file?",
            default=args.animation_file is not None,
            console=_console,
        )
        if save_animation:
            default_anim = args.animation_file or (save_dir / "gbm_animation.gif")
            animation_file = Path(
                Prompt.ask("Animation output path", default=str(default_anim), console=_console)
            ).expanduser()

    hist_bins = _prompt_int("Histogram bins", args.hist_bins, minimum=1)
    plot_paths = _prompt_int("Paths to draw in static chart", args.plot_paths, minimum=1)

    summary_data = {
        "Paths": f"{paths}",
        "Steps": f"{steps}",
        "Horizon": f"{horizon}",
        "Initial price": f"{s0}",
        "Device": device,
        "Precision": precision,
        "Streaming": "Yes" if stream else "No",
        "Endless": "Yes" if stream and endless else "No",
        "Animate": "Yes" if animate else "No",
        "Show window": "Yes" if show else "No",
        "Save plots": "Yes" if not no_save else "No",
    }
    _summarise_configuration(summary_data)

    return argparse.Namespace(
        paths=paths,
        steps=steps,
        horizon=horizon,
        s0=s0,
        seed=seed,
        initial_state=initial_state,
        device=device,
        precision=precision,
        plot_paths=plot_paths,
        save_dir=save_dir,
        no_save=no_save,
        show=show,
        hist_bins=hist_bins,
        animate=animate,
        animation_paths=animation_paths,
        animation_interval=animation_interval,
        animation_file=animation_file,
        stream=stream,
        endless=endless,
        drift_std=drift_std_values,
        volatility_cv=vol_cv_values,
        deterministic_params=deterministic_params,
        interactive=False,
        gui=False,
    )




