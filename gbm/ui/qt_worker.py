"""Worker utilities for running simulations without freezing the UI."""
from __future__ import annotations

from PyQt6 import QtCore


class SimulationWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, args, runner) -> None:
        super().__init__()
        self._args = args
        self._runner = runner

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            result = self._runner(self._args)
        except Exception as exc:  # pragma: no cover - UI execution path
            self.failed.emit(str(exc))
        else:
            self.finished.emit(result)

