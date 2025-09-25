"""PyQt6 graphical interface for GBM simulation configuration."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Iterable, Optional
import sys

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
            self.resize(720, 720)
            self.runner = runner
            self._default_drift_std = list(default_drift_std)
            self._default_vol_cv = list(default_vol_cv)
            self._base_args = base_args

            self._build_ui()
            self._apply_defaults()

        def _build_ui(self) -> None:
            central = QtWidgets.QWidget(self)
            layout = QtWidgets.QVBoxLayout(central)

            form_widget = QtWidgets.QWidget(central)
            form_layout = QtWidgets.QGridLayout(form_widget)
            row = 0

            def add_row(label: str, widget: QtWidgets.QWidget) -> None:
                nonlocal row
                form_layout.addWidget(QtWidgets.QLabel(label), row, 0)
                form_layout.addWidget(widget, row, 1)
                row += 1

            self.paths_spin = QtWidgets.QSpinBox()
            self.paths_spin.setRange(1, 10_000_000)
            add_row("Monte Carlo paths", self.paths_spin)

            self.steps_spin = QtWidgets.QSpinBox()
            self.steps_spin.setRange(1, 100_000)
            add_row("Time steps", self.steps_spin)

            self.horizon_spin = QtWidgets.QDoubleSpinBox()
            self.horizon_spin.setDecimals(4)
            self.horizon_spin.setRange(1e-6, 1000.0)
            add_row("Time horizon (years)", self.horizon_spin)

            self.s0_spin = QtWidgets.QDoubleSpinBox()
            self.s0_spin.setDecimals(4)
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

            self.stream_checkbox = QtWidgets.QCheckBox("Stream in real time")
            self.stream_checkbox.stateChanged.connect(self._toggle_stream_dependent)
            add_row("Streaming", self.stream_checkbox)

            self.endless_checkbox = QtWidgets.QCheckBox("Run without fixed horizon (Ctrl+C to stop)")
            add_row("Endless streaming", self.endless_checkbox)

            self.animate_checkbox = QtWidgets.QCheckBox("Animate paths")
            self.animate_checkbox.stateChanged.connect(self._toggle_animation_controls)
            add_row("Animation", self.animate_checkbox)

            self.show_checkbox = QtWidgets.QCheckBox("Show Matplotlib windows")
            add_row("Show windows", self.show_checkbox)

            self.save_plots_checkbox = QtWidgets.QCheckBox("Save plots to disk")
            add_row("Save plots", self.save_plots_checkbox)

            self.randomness_checkbox = QtWidgets.QCheckBox("Enable stochastic drift/volatility")
            add_row("Stochastic parameters", self.randomness_checkbox)

            self.drift_std_edit = QtWidgets.QLineEdit()
            add_row("Drift std per state", self.drift_std_edit)

            self.vol_cv_edit = QtWidgets.QLineEdit()
            add_row("Volatility CV per state", self.vol_cv_edit)

            self.plot_paths_spin = QtWidgets.QSpinBox()
            self.plot_paths_spin.setRange(1, 10_000)
            add_row("Paths in static plot", self.plot_paths_spin)

            self.hist_bins_spin = QtWidgets.QSpinBox()
            self.hist_bins_spin.setRange(1, 500)
            add_row("Histogram bins", self.hist_bins_spin)

            self.anim_paths_spin = QtWidgets.QSpinBox()
            self.anim_paths_spin.setRange(1, 5_000)
            add_row("Paths to animate", self.anim_paths_spin)

            self.anim_interval_spin = QtWidgets.QSpinBox()
            self.anim_interval_spin.setRange(1, 1000)
            add_row("Animation interval (ms)", self.anim_interval_spin)

            save_dir_layout = QtWidgets.QHBoxLayout()
            self.save_dir_edit = QtWidgets.QLineEdit()
            save_dir_button = QtWidgets.QPushButton("Browse...")
            save_dir_button.clicked.connect(self._choose_save_dir)
            save_dir_layout.addWidget(self.save_dir_edit)
            save_dir_layout.addWidget(save_dir_button)
            save_dir_container = QtWidgets.QWidget()
            save_dir_container.setLayout(save_dir_layout)
            add_row("Save directory", save_dir_container)

            anim_file_layout = QtWidgets.QHBoxLayout()
            self.anim_file_edit = QtWidgets.QLineEdit()
            anim_file_button = QtWidgets.QPushButton("Browse...")
            anim_file_button.clicked.connect(self._choose_animation_file)
            anim_file_layout.addWidget(self.anim_file_edit)
            anim_file_layout.addWidget(anim_file_button)
            anim_file_container = QtWidgets.QWidget()
            anim_file_container.setLayout(anim_file_layout)
            add_row("Animation file", anim_file_container)

            layout.addWidget(form_widget)

            button_row = QtWidgets.QHBoxLayout()
            self.run_button = QtWidgets.QPushButton("Run Simulation")
            self.run_button.clicked.connect(self._run_simulation)
            button_row.addWidget(self.run_button)

            self.status_label = QtWidgets.QLabel("")
            button_row.addWidget(self.status_label)
            button_container = QtWidgets.QWidget()
            button_container.setLayout(button_row)
            layout.addWidget(button_container)

            self.output_text = QtWidgets.QTextEdit()
            self.output_text.setReadOnly(True)
            self.output_text.setPlaceholderText("Simulation output will appear here...")
            layout.addWidget(self.output_text, stretch=1)

            self.setCentralWidget(central)

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
            self.animate_checkbox.setChecked(args.animate or args.stream)
            self.show_checkbox.setChecked(args.show or args.animate or args.stream)
            self.save_plots_checkbox.setChecked(not args.no_save)
            self.randomness_checkbox.setChecked(not args.deterministic_params)
            self.drift_std_edit.setText(", ".join(f"{value:.4f}" for value in self._default_drift_std))
            self.vol_cv_edit.setText(", ".join(f"{value:.4f}" for value in self._default_vol_cv))
            self.plot_paths_spin.setValue(args.plot_paths)
            self.hist_bins_spin.setValue(args.hist_bins)
            self.anim_paths_spin.setValue(args.animation_paths)
            self.anim_interval_spin.setValue(args.animation_interval)
            self.save_dir_edit.setText(str(args.save_dir))
            if args.animation_file is not None:
                self.anim_file_edit.setText(str(args.animation_file))
            self._toggle_stream_dependent()
            self._toggle_animation_controls()

        def _toggle_stream_dependent(self) -> None:
            enabled = self.stream_checkbox.isChecked()
            self.endless_checkbox.setEnabled(enabled)
            if not enabled:
                self.endless_checkbox.setChecked(False)

        def _toggle_animation_controls(self) -> None:
            enabled = self.animate_checkbox.isChecked()
            for widget in (self.anim_paths_spin, self.anim_interval_spin, self.anim_file_edit):
                widget.setEnabled(enabled)

        def _choose_save_dir(self) -> None:
            directory = QtWidgets.QFileDialog.getExistingDirectory(self, "Select save directory")
            if directory:
                self.save_dir_edit.setText(directory)

        def _choose_animation_file(self) -> None:
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Select animation file",
                filter="GIF (*.gif);;MP4 (*.mp4)"
            )
            if filename:
                self.anim_file_edit.setText(filename)

        def _build_namespace(self) -> Optional[argparse.Namespace]:
            try:
                drift_std = _parse_float_list(self.drift_std_edit.text())
                vol_cv = _parse_float_list(self.vol_cv_edit.text())
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
                save_dir=_to_path(self.save_dir_edit.text()) or Path("figures"),
                no_save=not self.save_plots_checkbox.isChecked(),
                show=self.show_checkbox.isChecked(),
                hist_bins=self.hist_bins_spin.value(),
                animate=self.animate_checkbox.isChecked(),
                animation_paths=self.anim_paths_spin.value(),
                animation_interval=self.anim_interval_spin.value(),
                animation_file=_to_path(self.anim_file_edit.text()),
                stream=self.stream_checkbox.isChecked(),
                endless=self.endless_checkbox.isChecked(),
                drift_std=drift_std,
                volatility_cv=vol_cv,
                deterministic_params=not self.randomness_checkbox.isChecked(),
                interactive=False,
                gui=True,
            )
            return namespace

        def _run_simulation(self) -> None:
            namespace = self._build_namespace()
            if namespace is None:
                return

            self.run_button.setEnabled(False)
            self.status_label.setText("Running...")
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
            try:
                result = self.runner(namespace)
            except Exception as exc:  # pragma: no cover - UI path
                QtWidgets.QApplication.restoreOverrideCursor()
                self.run_button.setEnabled(True)
                self.status_label.setText("Error")
                QtWidgets.QMessageBox.critical(self, "Simulation failed", str(exc))
                return

            QtWidgets.QApplication.restoreOverrideCursor()
            self.run_button.setEnabled(True)
            self.status_label.setText("Finished")

            messages = result.get("messages", [])
            self.output_text.setPlainText("\n".join(messages))

            saved = result.get("saved_paths", [])
            animation_file = result.get("animation_file")
            if saved or animation_file:
                note_lines = ["Output saved:"]
                for path in saved:
                    note_lines.append(f"- {path}")
                if animation_file:
                    note_lines.append(f"- {animation_file}")
                QtWidgets.QMessageBox.information(self, "Simulation complete", "\n".join(note_lines))

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
