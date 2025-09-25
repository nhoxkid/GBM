"""PyQt6 graphical interface for GBM simulation configuration."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable, Optional
import sys

from gbm.runtime import (
    DEFAULT_DRIFT_STD,
    DEFAULT_VOLATILITY_CV,
    create_simulation_context,
    fmt,
)
from gbm import summarize_terminal_distribution

try:
    from PyQt6 import QtCore, QtWidgets
except ImportError:  # pragma: no cover - handled in caller
    QtCore = None  # type: ignore
    QtWidgets = None  # type: ignore


def _to_path(text: str) -> Optional[Path]:
    value = text.strip()
    if not value:
        return None
    return Path(value).expanduser()


def _parse_float_list(text: str) -> Optional[list[float]]:
    cleaned = text.replace(",", " ").strip()
    if not cleaned:
        return None
    try:
        return [float(token) for token in cleaned.split()]
    except ValueError:
        raise ValueError("Please enter numeric values separated by spaces or commas.")


if QtWidgets is not None:
    from gbm.ui.qt_canvas import SimulationCanvas
    from gbm.ui.qt_theme import apply_theme
    from gbm.ui.qt_worker import SimulationWorker
    from gbm.ui.qt_stream import StreamingController


    class SimulationWindow(QtWidgets.QMainWindow):  # type: ignore[misc]
        """Main window for configuring and running a simulation."""

        def __init__(
            self,
            base_args: argparse.Namespace,
            *,
            default_drift_std: Iterable[float],
            default_vol_cv: Iterable[float],
            runner: Callable[[argparse.Namespace], dict],
        ) -> None:
            super().__init__()
            self.setWindowTitle("GBM Monte Carlo Studio")
            self.resize(1180, 760)

            self.runner = runner
            self._base_args = base_args
            self._default_drift_std = list(default_drift_std)
            self._default_vol_cv = list(default_vol_cv)
            self._worker_thread: QtCore.QThread | None = None
            self._worker: SimulationWorker | None = None
            self._stream_controller: StreamingController | None = None
            self._last_namespace: argparse.Namespace | None = None
            self._last_result = None
            self._current_theme = "light"

            self._build_ui()
            self._apply_defaults()
            self._change_theme("Light")

        # ------------------------------------------------------------------
        # UI construction helpers
        # ------------------------------------------------------------------
        def _build_ui(self) -> None:
            central = QtWidgets.QWidget(self)
            layout = QtWidgets.QVBoxLayout(central)
            layout.setContentsMargins(18, 18, 18, 18)
            layout.setSpacing(18)

            # Hero header ----------------------------------------------------
            header = QtWidgets.QWidget(central)
            header_layout = QtWidgets.QVBoxLayout(header)
            header_layout.setContentsMargins(0, 0, 0, 0)
            title = QtWidgets.QLabel("GBM Monte Carlo Studio", header)
            title.setStyleSheet("font-size: 28px; font-weight: 700;")
            subtitle = QtWidgets.QLabel(
                "Design, stream, and analyse regime-switching geometric Brownian motion",
                header,
            )
            subtitle.setStyleSheet("color: #6e6e73; font-size: 15px;")
            header_layout.addWidget(title)
            header_layout.addWidget(subtitle)

            theme_bar = QtWidgets.QHBoxLayout()
            theme_bar.addStretch(1)
            theme_label = QtWidgets.QLabel("Theme")
            self.theme_combo = QtWidgets.QComboBox()
            self.theme_combo.addItems(["Light", "Dark"])
            self.theme_combo.currentTextChanged.connect(self._change_theme)
            theme_bar.addWidget(theme_label)
            theme_bar.addWidget(self.theme_combo)
            theme_bar.addStretch(1)
            header_layout.addLayout(theme_bar)

            layout.addWidget(header)

            # Main body -----------------------------------------------------
            body = QtWidgets.QHBoxLayout()
            body.setSpacing(16)

            form_area = QtWidgets.QScrollArea()
            form_area.setWidgetResizable(True)
            form_container = QtWidgets.QWidget()
            form_layout = QtWidgets.QGridLayout(form_container)
            form_layout.setHorizontalSpacing(14)
            form_layout.setVerticalSpacing(12)
            row = 0

            def add_section(title_text: str) -> None:
                nonlocal row
                label = QtWidgets.QLabel(title_text)
                label.setProperty("class", "section-title")
                label.setStyleSheet("font-size: 16px; font-weight: 600;")
                form_layout.addWidget(label, row, 0, 1, 2)
                row += 1

            def add_row(label_text: str, widget: QtWidgets.QWidget) -> None:
                nonlocal row
                label = QtWidgets.QLabel(label_text)
                form_layout.addWidget(label, row, 0)
                form_layout.addWidget(widget, row, 1)
                row += 1

            add_section("Simulation setup")
            self.paths_spin = QtWidgets.QSpinBox()
            self.paths_spin.setRange(1, 10_000_000)
            add_row("Monte Carlo paths", self.paths_spin)

            self.steps_spin = QtWidgets.QSpinBox()
            self.steps_spin.setRange(1, 200_000)
            add_row("Time steps", self.steps_spin)

            self.horizon_spin = QtWidgets.QDoubleSpinBox()
            self.horizon_spin.setDecimals(5)
            self.horizon_spin.setRange(1e-6, 2000.0)
            add_row("Time horizon (years)", self.horizon_spin)

            self.s0_spin = QtWidgets.QDoubleSpinBox()
            self.s0_spin.setDecimals(6)
            self.s0_spin.setRange(0.0, 1_000_000.0)
            add_row("Initial price", self.s0_spin)

            self.seed_edit = QtWidgets.QLineEdit()
            self.seed_edit.setPlaceholderText("none")
            add_row("Random seed", self.seed_edit)

            self.initial_state_spin = QtWidgets.QSpinBox()
            self.initial_state_spin.setRange(0, 10_000)
            add_row("Initial state index", self.initial_state_spin)

            self.device_combo = QtWidgets.QComboBox()
            self.device_combo.addItems(["auto", "cpu", "cuda"])
            add_row("Device", self.device_combo)

            self.precision_combo = QtWidgets.QComboBox()
            self.precision_combo.addItems(["float64", "float32"])
            add_row("Precision", self.precision_combo)

            add_section("Visualisation & streaming")
            self.stream_checkbox = QtWidgets.QCheckBox("Stream paths in real time")
            self.stream_checkbox.stateChanged.connect(self._toggle_stream_dependent)
            form_layout.addWidget(self.stream_checkbox, row, 0, 1, 2)
            row += 1

            self.endless_checkbox = QtWidgets.QCheckBox("Run without fixed horizon (Ctrl+C to stop)")
            form_layout.addWidget(self.endless_checkbox, row, 0, 1, 2)
            row += 1

            self.animate_checkbox = QtWidgets.QCheckBox("Capture animation frames for static runs")
            form_layout.addWidget(self.animate_checkbox, row, 0, 1, 2)
            row += 1

            self.show_checkbox = QtWidgets.QCheckBox("Show Matplotlib windows (static only)")
            form_layout.addWidget(self.show_checkbox, row, 0, 1, 2)
            row += 1

            self.plot_paths_spin = QtWidgets.QSpinBox()
            self.plot_paths_spin.setRange(1, 10_000)
            add_row("Paths in static view", self.plot_paths_spin)

            self.hist_bins_spin = QtWidgets.QSpinBox()
            self.hist_bins_spin.setRange(1, 500)
            add_row("Histogram bins", self.hist_bins_spin)

            self.anim_paths_spin = QtWidgets.QSpinBox()
            self.anim_paths_spin.setRange(1, 5000)
            add_row("Paths in live stream", self.anim_paths_spin)

            self.anim_interval_spin = QtWidgets.QSpinBox()
            self.anim_interval_spin.setRange(1, 1000)
            add_row("Frame interval (ms)", self.anim_interval_spin)

            add_section("Outputs")
            self.save_plots_checkbox = QtWidgets.QCheckBox("Save static figures to disk")
            form_layout.addWidget(self.save_plots_checkbox, row, 0, 1, 2)
            row += 1

            save_dir_layout = QtWidgets.QHBoxLayout()
            self.save_dir_edit = QtWidgets.QLineEdit()
            save_dir_button = QtWidgets.QPushButton("Browse…")
            save_dir_button.clicked.connect(self._choose_save_dir)
            save_dir_layout.addWidget(self.save_dir_edit)
            save_dir_layout.addWidget(save_dir_button)
            save_dir_container = QtWidgets.QWidget()
            save_dir_container.setLayout(save_dir_layout)
            add_row("Save directory", save_dir_container)

            self.animation_file_edit = QtWidgets.QLineEdit()
            self.animation_file_edit.setPlaceholderText("Optional animation export path")
            add_row("Animation file", self.animation_file_edit)

            add_section("Randomness")
            self.randomness_checkbox = QtWidgets.QCheckBox("Enable stochastic drift/volatility")
            self.randomness_checkbox.stateChanged.connect(self._toggle_randomness_inputs)
            form_layout.addWidget(self.randomness_checkbox, row, 0, 1, 2)
            row += 1

            self.drift_std_edit = QtWidgets.QLineEdit()
            add_row("Drift std per state", self.drift_std_edit)

            self.vol_cv_edit = QtWidgets.QLineEdit()
            add_row("Volatility CV per state", self.vol_cv_edit)

            form_layout.setRowStretch(row, 1)
            form_container.setLayout(form_layout)
            form_area.setWidget(form_container)
            body.addWidget(form_area, stretch=1)

            # Visualisation panel ------------------------------------------
            visual_panel = QtWidgets.QWidget()
            visual_layout = QtWidgets.QVBoxLayout(visual_panel)
            visual_layout.setContentsMargins(0, 0, 0, 0)
            visual_layout.setSpacing(12)

            self.canvas = SimulationCanvas(parent=visual_panel)
            self.canvas.setMinimumHeight(420)
            visual_layout.addWidget(self.canvas, stretch=3)

            self.summary_label = QtWidgets.QLabel("Configure a simulation to begin.")
            self.summary_label.setWordWrap(True)
            self.summary_label.setStyleSheet("font-size: 13px; line-height: 1.4;")
            visual_layout.addWidget(self.summary_label)

            self.output_text = QtWidgets.QPlainTextEdit()
            self.output_text.setReadOnly(True)
            self.output_text.setPlaceholderText("Simulation output will appear here…")
            visual_layout.addWidget(self.output_text, stretch=2)

            body.addWidget(visual_panel, stretch=2)
            layout.addLayout(body, stretch=1)

            # Action bar ----------------------------------------------------
            action_bar = QtWidgets.QHBoxLayout()
            self.run_button = QtWidgets.QPushButton("Run Simulation")
            self.run_button.clicked.connect(self._run_simulation)
            action_bar.addWidget(self.run_button)

            self.stream_button = QtWidgets.QPushButton("Stream in Real Time")
            self.stream_button.clicked.connect(self._run_stream_now)
            action_bar.addWidget(self.stream_button)

            self.stop_button = QtWidgets.QPushButton("Stop Streaming")
            self.stop_button.clicked.connect(self._stop_streaming)
            self.stop_button.setEnabled(False)
            action_bar.addWidget(self.stop_button)

            self.export_button = QtWidgets.QPushButton("Export Snapshot")
            self.export_button.clicked.connect(self._export_snapshot)
            action_bar.addStretch(1)
            action_bar.addWidget(self.export_button)

            layout.addLayout(action_bar)

            self.status_label = QtWidgets.QLabel("")
            layout.addWidget(self.status_label)

            self.setCentralWidget(central)

        # ------------------------------------------------------------------
        # Form preparation
        # ------------------------------------------------------------------
        def _apply_defaults(self) -> None:
            args = self._base_args
            self.paths_spin.setValue(args.paths)
            self.steps_spin.setValue(args.steps)
            self.horizon_spin.setValue(float(args.horizon))
            self.s0_spin.setValue(float(args.s0))
            if args.seed is not None:
                self.seed_edit.setText(str(args.seed))
            self.initial_state_spin.setValue(args.initial_state)
            self.device_combo.setCurrentText(args.device)
            self.precision_combo.setCurrentText(args.precision)
            self.stream_checkbox.setChecked(args.stream)
            self.endless_checkbox.setChecked(args.endless)
            self.animate_checkbox.setChecked(args.animate)
            self.show_checkbox.setChecked(args.show)
            self.save_plots_checkbox.setChecked(not args.no_save)
            self.plot_paths_spin.setValue(args.plot_paths)
            self.hist_bins_spin.setValue(args.hist_bins)
            self.anim_paths_spin.setValue(args.animation_paths)
            self.anim_interval_spin.setValue(args.animation_interval)
            self.save_dir_edit.setText(str(args.save_dir))
            if args.animation_file is not None:
                self.animation_file_edit.setText(str(args.animation_file))
            self.randomness_checkbox.setChecked(not args.deterministic_params)
            self.drift_std_edit.setText(", ".join(f"{value:.4f}" for value in self._default_drift_std))
            self.vol_cv_edit.setText(", ".join(f"{value:.4f}" for value in self._default_vol_cv))
            self._toggle_stream_dependent()
            self._toggle_randomness_inputs()

        def _toggle_stream_dependent(self) -> None:
            streaming = self.stream_checkbox.isChecked()
            self.endless_checkbox.setEnabled(streaming)
            if not streaming:
                self.endless_checkbox.setChecked(False)

        def _toggle_randomness_inputs(self) -> None:
            enabled = self.randomness_checkbox.isChecked()
            self.drift_std_edit.setEnabled(enabled)
            self.vol_cv_edit.setEnabled(enabled)

        # ------------------------------------------------------------------
        # Actions
        # ------------------------------------------------------------------
        def _choose_save_dir(self) -> None:
            directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select save directory")
            if directory:
                self.save_dir_edit.setText(directory)

        def _change_theme(self, label: str) -> None:
            theme = label.lower()
            app = QtWidgets.QApplication.instance()
            if app is not None:
                apply_theme(app, theme)
            self._current_theme = theme
            self.canvas.set_theme(theme)

        def _run_simulation(self) -> None:
            namespace = self._build_namespace()
            if namespace is None:
                return
            self._execute_namespace(namespace)

        def _run_stream_now(self) -> None:
            namespace = self._build_namespace()
            if namespace is None:
                return
            namespace.stream = True
            self.stream_checkbox.blockSignals(True)
            self.stream_checkbox.setChecked(True)
            self.stream_checkbox.blockSignals(False)
            self._execute_namespace(namespace)

        def _execute_namespace(self, namespace: argparse.Namespace) -> None:
            self._reset_state("Running simulation…")
            self.run_button.setEnabled(False)
            self.stream_button.setEnabled(False)
            self.stop_button.setEnabled(False)

            if namespace.stream:
                self._start_streaming(namespace)
            else:
                self._start_static(namespace)

        def _start_static(self, namespace: argparse.Namespace) -> None:
            namespace.show = False
            self._last_namespace = namespace
            self._worker_thread = QtCore.QThread(self)
            self._worker = SimulationWorker(namespace, self.runner)
            self._worker.moveToThread(self._worker_thread)
            self._worker_thread.started.connect(self._worker.run)
            self._worker.finished.connect(self._on_static_finished)
            self._worker.failed.connect(self._on_static_failed)
            self._worker.finished.connect(self._cleanup_worker)
            self._worker.failed.connect(self._cleanup_worker)
            self._worker_thread.start()

        def _start_streaming(self, namespace: argparse.Namespace) -> None:
            self.stop_button.setEnabled(True)
            self.output_text.clear()
            self.canvas.show_empty_message("Preparing streaming simulation…")

            try:
                context = create_simulation_context(
                    device=namespace.device,
                    precision=namespace.precision,
                    seed=namespace.seed,
                    deterministic_params=namespace.deterministic_params,
                    drift_std=namespace.drift_std,
                    volatility_cv=namespace.volatility_cv,
                    save_dir=namespace.save_dir,
                )
            except Exception as exc:
                QtWidgets.QMessageBox.critical(self, "Failed to prepare stream", str(exc))
                self.run_button.setEnabled(True)
                self.stream_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                return
            simulator = context.simulator
            stream = simulator.simulate_stream(
                n_paths=namespace.paths,
                n_steps=namespace.steps,
                horizon=namespace.horizon,
                s0=namespace.s0,
                initial_state=namespace.initial_state,
                infinite=namespace.endless,
            )

            if namespace.endless and namespace.animation_file is not None:
                self._append_message(
                    "Infinite streaming cannot be exported as an animation; ignoring the file path."
                )

            controller = StreamingController(
                stream,
                self.canvas,
                interval_ms=namespace.animation_interval,
                parent=self,
            )
            controller.message.connect(self._append_message)
            controller.completed.connect(lambda result: self._finish_streaming(result, namespace, context))
            controller.frame_advanced.connect(self._update_stream_status)
            self._stream_controller = controller

            actual = controller.start(namespace.animation_paths)
            self._append_message(f"Streaming {actual} path(s) in real time…")
            self.status_label.setText("Streaming in progress")
            self._last_namespace = namespace

        def _stop_streaming(self) -> None:
            if self._stream_controller is not None:
                self._stream_controller.stop()
            self.stop_button.setEnabled(False)

        def _finish_streaming(self, result, namespace: argparse.Namespace, context) -> None:
            summary = summarize_terminal_distribution(result.prices)
            self.canvas.show_static_result(
                result,
                num_paths=namespace.plot_paths,
                hist_bins=namespace.hist_bins,
            )
            self._last_result = result
            self._update_summary(summary, context.config)
            self._append_message("Streaming summary computed.")
            self.run_button.setEnabled(True)
            self.stream_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_label.setText("Streaming finished")

        def _on_static_finished(self, payload: dict) -> None:
            self.run_button.setEnabled(True)
            self.stream_button.setEnabled(True)
            self.status_label.setText("Simulation complete")
            result = payload.get("result")
            if result is not None and self._last_namespace is not None:
                self.canvas.show_static_result(
                    result,
                    num_paths=self._last_namespace.plot_paths,
                    hist_bins=self._last_namespace.hist_bins,
                )
                self._last_result = result
            summary = payload.get("summary")
            config = payload.get("config")
            if summary is not None and config is not None:
                self._update_summary(summary, config)
            messages = payload.get("messages", [])
            text = "\n".join(messages)
            saved_paths = payload.get("saved_paths", [])
            animation_file = payload.get("animation_file")
            if saved_paths or animation_file:
                text += "\n\nSaved outputs:\n"
                for path in saved_paths:
                    text += f"- {path}\n"
                if animation_file:
                    text += f"- {animation_file}\n"
            self.output_text.setPlainText(text)

        def _on_static_failed(self, error: str) -> None:
            self.run_button.setEnabled(True)
            self.stream_button.setEnabled(True)
            QtWidgets.QMessageBox.critical(self, "Simulation failed", error)
            self.status_label.setText("Simulation failed")

        def _cleanup_worker(self) -> None:
            if self._worker_thread is not None:
                self._worker_thread.quit()
                self._worker_thread.wait()
            self._worker_thread = None
            self._worker = None

        def _update_stream_status(self, step: int) -> None:
            self.status_label.setText(f"Streaming step {step}")

        def _append_message(self, message: str) -> None:
            existing = self.output_text.toPlainText()
            if existing:
                existing += "\n"
            self.output_text.setPlainText(existing + message)
            self.output_text.verticalScrollBar().setValue(self.output_text.verticalScrollBar().maximum())

        def _export_snapshot(self) -> None:
            if self._last_result is None:
                QtWidgets.QMessageBox.information(self, "No data", "Run a simulation before exporting.")
                return
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Export current view",
                filter="PNG (*.png)"
            )
            if filename:
                path = Path(filename)
                self.canvas.export_png(path)
                self._append_message(f"Snapshot exported to {path}")

        def _update_summary(self, summary, config) -> None:
            lines = [
                f"Terminal mean: {fmt(summary.mean)}",
                f"Terminal standard deviation: {fmt(summary.standard_deviation)}",
                f"5th percentile: {fmt(summary.quantile_05)}",
                f"95th percentile: {fmt(summary.quantile_95)}",
                (
                    "95% CI for mean: ("
                    f"{fmt(summary.confidence_interval[0])}, {fmt(summary.confidence_interval[1])})"
                ),
            ]
            randomness = getattr(config, "randomness", None)
            if randomness is not None:
                lines.append(
                    "Random drift std: "
                    + ", ".join(f"{value:.4f}" for value in randomness.drift_std.cpu().numpy())
                )
                lines.append(
                    "Random volatility CV: "
                    + ", ".join(f"{value:.4f}" for value in randomness.volatility_cv.cpu().numpy())
                )
            self.summary_label.setText("\n".join(lines))

        def _reset_state(self, status: str) -> None:
            self.status_label.setText(status)
            self.output_text.clear()
            self._last_result = None

        def _build_namespace(self) -> Optional[argparse.Namespace]:
            try:
                drift_std = _parse_float_list(self.drift_std_edit.text()) if self.randomness_checkbox.isChecked() else None
                vol_cv = _parse_float_list(self.vol_cv_edit.text()) if self.randomness_checkbox.isChecked() else None
            except ValueError as exc:
                QtWidgets.QMessageBox.warning(self, "Invalid input", str(exc))
                return None

            seed_text = self.seed_edit.text().strip()
            seed_value: Optional[int]
            if seed_text:
                try:
                    seed_value = int(seed_text)
                except ValueError:
                    QtWidgets.QMessageBox.warning(self, "Invalid input", "Seed must be an integer or left blank.")
                    return None
            else:
                seed_value = None

            save_dir = Path(self.save_dir_edit.text()).expanduser()
            animation_file = _to_path(self.animation_file_edit.text())

            namespace = argparse.Namespace(
                paths=self.paths_spin.value(),
                steps=self.steps_spin.value(),
                horizon=float(self.horizon_spin.value()),
                s0=float(self.s0_spin.value()),
                seed=seed_value,
                initial_state=self.initial_state_spin.value(),
                device=self.device_combo.currentText(),
                precision=self.precision_combo.currentText(),
                plot_paths=self.plot_paths_spin.value(),
                save_dir=save_dir,
                no_save=not self.save_plots_checkbox.isChecked(),
                show=self.show_checkbox.isChecked(),
                hist_bins=self.hist_bins_spin.value(),
                animate=self.animate_checkbox.isChecked(),
                animation_paths=self.anim_paths_spin.value(),
                animation_interval=self.anim_interval_spin.value(),
                animation_file=animation_file,
                stream=self.stream_checkbox.isChecked(),
                endless=self.endless_checkbox.isChecked(),
                drift_std=drift_std,
                volatility_cv=vol_cv,
                deterministic_params=not self.randomness_checkbox.isChecked(),
                interactive=False,
                gui=True,
            )
            return namespace

else:
    class SimulationWindow:  # pragma: no cover - placeholder when PyQt is absent
        pass


def launch_qt_interface(
    base_args: argparse.Namespace,
    *,
    default_drift_std: Iterable[float],
    default_vol_cv: Iterable[float],
    runner: Callable[[argparse.Namespace], dict],
) -> None:
    if QtWidgets is None:
        raise ImportError("PyQt6 is not available in this environment.")

    app = QtWidgets.QApplication.instance()
    owns_app = False
    if app is None:
        app = QtWidgets.QApplication(sys.argv[:1])  # type: ignore[name-defined]
        owns_app = True

    window = SimulationWindow(
        base_args,
        default_drift_std=default_drift_std,
        default_vol_cv=default_vol_cv,
        runner=runner,
    )
    window.show()

    if owns_app:
        app.exec()
