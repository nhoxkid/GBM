"""Worker utilities for running simulations without freezing the UI."""
from __future__ import annotations

from typing import Any, Callable

try:  # pragma: no cover - optional dependency in headless CI
    from PyQt6 import QtCore
except ImportError:  # pragma: no cover - provide a stub for type checkers
    QtCore = None  # type: ignore[assignment]


if QtCore is None:  # pragma: no cover - keep explicit failure message
    raise ImportError("PyQt6 is required to run the Qt worker module.")


class SimulationWorker(QtCore.QObject):
    """Lift the blocking simulation call onto a background thread."""
    finished = QtCore.pyqtSignal(dict)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, args: Any, runner: Callable[[Any], dict]) -> None:
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
