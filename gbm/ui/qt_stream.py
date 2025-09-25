"""Real-time streaming controller for the Qt interface."""
from __future__ import annotations

from PyQt6 import QtCore

from gbm.streaming import StreamingSimulation
from gbm.ui.qt_canvas import SimulationCanvas


class StreamingController(QtCore.QObject):
    completed = QtCore.pyqtSignal(object)  # Emits SimulationResult
    message = QtCore.pyqtSignal(str)
    frame_advanced = QtCore.pyqtSignal(int)

    def __init__(
        self,
        simulation: StreamingSimulation,
        canvas: SimulationCanvas,
        *,
        interval_ms: int,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._simulation = simulation
        self._canvas = canvas
        self._timer = QtCore.QTimer(self)
        self._timer.setInterval(max(1, interval_ms))
        self._timer.timeout.connect(self._advance)
        self._active = False
        self._visual_count = 0

    def start(self, paths_to_show: int) -> int:
        actual = self._simulation.set_visual_count(paths_to_show)
        self._visual_count = actual
        title = "Real-time GBM streaming" if not self._simulation.infinite else "Infinite GBM streaming"
        self._canvas.prepare_streaming(title=title, line_count=actual)
        time_slice, sample = self._simulation.get_visual_snapshot(0)
        if actual:
            self._canvas.update_streaming(time_slice, sample)
        self._timer.start()
        self._active = True
        self.message.emit("Streaming started.")
        return actual

    def stop(self) -> None:
        if not self._active:
            return
        self._timer.stop()
        self._active = False
        self.message.emit("Streaming stopped.")
        self.completed.emit(self._simulation.to_result())

    def _advance(self) -> None:
        try:
            self._simulation.step()
        except StopIteration:
            self._timer.stop()
            self._active = False
            self.message.emit("Streaming completed.")
            self.completed.emit(self._simulation.to_result())
            return
        except Exception as exc:  # pragma: no cover - UI runtime errors
            self._timer.stop()
            self._active = False
            self.message.emit(f"Streaming error: {exc}")
            return

        current = self._simulation.current_step
        time_slice, sample = self._simulation.get_visual_snapshot(current)
        if self._visual_count:
            self._canvas.update_streaming(time_slice, sample)
        self.frame_advanced.emit(current)

    def is_active(self) -> bool:
        return self._active

