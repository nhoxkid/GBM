"""Matplotlib canvas helpers for the Qt interface."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from gbm.results import SimulationResult


class SimulationCanvas(FigureCanvasQTAgg):
    """Reusable canvas that can switch between streaming and static views."""

    def __init__(self, *, parent=None) -> None:  # type: ignore[override]
        figure = Figure(figsize=(10.0, 5.6), constrained_layout=True)
        super().__init__(figure)
        if parent is not None:
            self.setParent(parent)

        grid = figure.add_gridspec(1, 2, width_ratios=(3, 2))
        self.ax_stream = figure.add_subplot(grid[0, 0])
        self.ax_hist = figure.add_subplot(grid[0, 1])
        self.ax_hist.set_visible(False)

        self._lines: list = []
        self._theme = "light"
        self._apply_axis_style()
        self.figure = figure

    # ------------------------------------------------------------------
    # Theme management
    # ------------------------------------------------------------------
    def set_theme(self, theme: str) -> None:
        if theme not in {"light", "dark"}:
            raise ValueError("Theme must be 'light' or 'dark'.")
        self._theme = theme
        self._apply_axis_style()

    def _apply_axis_style(self) -> None:
        if self._theme == "dark":
            face = "#1d1d1f"
            axis_face = "#2c2c34"
            fg = "#f5f5f7"
            grid_color = "#3a3a46"
        else:
            face = "#f5f5f7"
            axis_face = "#ffffff"
            fg = "#1d1d1f"
            grid_color = "#d0d0d4"

        self.figure.set_facecolor(face)
        for ax in (self.ax_stream, self.ax_hist):
            ax.set_facecolor(axis_face)
            ax.tick_params(colors=fg, which="both")
            ax.xaxis.label.set_color(fg)
            ax.yaxis.label.set_color(fg)
            ax.title.set_color(fg)
            ax.grid(True, color=grid_color, alpha=0.3, linewidth=0.8)
            ax.spines["top"].set_color(grid_color)
            ax.spines["right"].set_color(grid_color)
            ax.spines["bottom"].set_color(grid_color)
            ax.spines["left"].set_color(grid_color)
        self.draw_idle()

    # ------------------------------------------------------------------
    # Streaming helpers
    # ------------------------------------------------------------------
    def prepare_streaming(self, *, title: str, line_count: int) -> None:
        self.ax_stream.clear()
        self.ax_stream.set_title(title)
        self.ax_stream.set_xlabel("Time")
        self.ax_stream.set_ylabel("Asset price")
        self.ax_stream.grid(True, alpha=0.3)
        self._lines = [self.ax_stream.plot([], [], linewidth=1.4, alpha=0.9)[0] for _ in range(line_count)]

        self.ax_hist.clear()
        self.ax_hist.set_visible(False)
        self._apply_axis_style()

        if line_count == 0:
            self.ax_stream.text(
                0.5,
                0.5,
                "No paths selected for streaming",
                transform=self.ax_stream.transAxes,
                ha="center",
                va="center",
                fontsize=12,
            )
        self.draw_idle()

    def update_streaming(self, time_slice: np.ndarray, sample: np.ndarray) -> None:
        available = sample.shape[0]
        for idx, line in enumerate(self._lines):
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
            if margin == 0:
                margin = 1.0
            x_start = float(time_slice[0])
            x_end = float(time_slice[-1])
            span = x_end - x_start
            if np.isclose(span, 0.0):
                span = 1.0
            pad = 0.05 * span
            self.ax_stream.set_xlim(x_start - pad, x_end + pad)
            self.ax_stream.set_ylim(y_min - margin, y_max + margin)

        self.draw_idle()

    # ------------------------------------------------------------------
    # Static view helpers
    # ------------------------------------------------------------------
    def show_static_result(
        self,
        result: SimulationResult,
        *,
        num_paths: int,
        hist_bins: int,
        highlight_states: bool = False,
    ) -> None:
        time = result.time_grid.detach().cpu().numpy()
        prices = result.prices.detach().cpu().numpy()
        paths_to_plot = min(num_paths, prices.shape[0])

        self.ax_stream.clear()
        self.ax_stream.set_title("Sample GBM paths")
        self.ax_stream.set_xlabel("Time")
        self.ax_stream.set_ylabel("Asset price")

        selection = prices[:paths_to_plot]
        for idx, path in enumerate(selection):
            alpha = 0.85 if idx == 0 else 0.75
            self.ax_stream.plot(time, path, linewidth=1.1, alpha=alpha)

        if highlight_states and hasattr(result, "states"):
            # Draw faint state background bands to emphasise regime changes.
            states = result.states[0].detach().cpu().numpy()
            last_state = states[0]
            segment_start = 0
            palette = (
                "#6e8efb",
                "#a777e3",
                "#f7b500",
                "#fa8072",
                "#3ccfcf",
            )
            for idx_state, state in enumerate(states[1:], start=1):
                if state != last_state:
                    color = palette[last_state % len(palette)]
                    self.ax_stream.axvspan(
                        time[segment_start],
                        time[idx_state],
                        color=color,
                        alpha=0.08,
                    )
                    segment_start = idx_state
                    last_state = int(state)
            color = palette[last_state % len(palette)]
            self.ax_stream.axvspan(
                time[segment_start],
                time[-1],
                color=color,
                alpha=0.08,
            )

        terminal_prices = result.prices[:, -1].detach().cpu().numpy()
        self.ax_hist.clear()
        self.ax_hist.set_visible(True)
        self.ax_hist.hist(
            terminal_prices,
            bins=hist_bins,
            alpha=0.8,
            color="#0071e3" if self._theme == "light" else "#78c8ff",
            edgecolor="#1d1d1f" if self._theme == "light" else "#f5f5f7",
        )
        self.ax_hist.set_title("Terminal distribution")
        self.ax_hist.set_xlabel("Terminal price")
        self.ax_hist.set_ylabel("Frequency")

        self._apply_axis_style()
        self.draw_idle()

    def show_empty_message(self, message: str) -> None:
        self.ax_stream.clear()
        self.ax_hist.clear()
        self.ax_hist.set_visible(False)
        self.ax_stream.text(
            0.5,
            0.5,
            message,
            transform=self.ax_stream.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )
        self._apply_axis_style()
        self.draw_idle()

    def freeze_view(self) -> None:
        for line in self._lines:
            line.set_animated(False)
        self.draw_idle()

    # ------------------------------------------------------------------
    # Convenience utilities
    # ------------------------------------------------------------------
    def export_png(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.figure.savefig(path, dpi=160, bbox_inches="tight")
