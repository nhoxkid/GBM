"""Worker utilities for running simulations without freezing the UI."""
from __future__ import annotations

from typing import Any, Callable

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot


class SimulationWorker(QObject):
    """Lift the blocking simulation call onto a background thread."""
    finished = pyqtSignal(dict)
    failed = pyqtSignal(str)

    def __init__(self, args: Any, runner: Callable[[Any], dict]) -> None:
        super().__init__()
        self._args = args
        self._runner = runner

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = self._runner(self._args)
        except Exception as exc:  # pragma: no cover - UI execution path
            self.failed.emit(str(exc))
        else:
            self.finished.emit(result)
