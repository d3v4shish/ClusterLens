from __future__ import annotations

from collections.abc import Callable
from itertools import count

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot

from infra.cancel import Cancelled, raise_if_cancelled
from infra.qt_diagnostics import append_qt_diagnostic

_JOB_COUNTER = count(1)


class _AsyncJobSignalRelay(QObject):
    def __init__(self, job: "AsyncJob"):
        super().__init__()
        self._job = job

    def _emit_safe(self, signal_name: str, *args) -> None:
        try:
            signal = getattr(self._job, signal_name)
            signal.emit(*args)
        except RuntimeError:
            # The owning AsyncJob can be deleted during shutdown while queued
            # relay events are still draining back onto the GUI thread.
            return

    @pyqtSlot()
    def relay_started(self) -> None:
        self._emit_safe("started")

    @pyqtSlot(int, str)
    def relay_progress(self, value: int, status: str) -> None:
        self._emit_safe("progress", int(value), str(status))

    @pyqtSlot(object)
    def relay_completed(self, result: object) -> None:
        self._emit_safe("completed", result)

    @pyqtSlot(str)
    def relay_failed(self, message: str) -> None:
        self._emit_safe("failed", str(message))

    @pyqtSlot()
    def relay_cancelled(self) -> None:
        self._emit_safe("cancelled")


class AsyncJob(QObject):
    """
    Minimal worker abstraction for running a callable on a QThread.

    Contract:
      fn(progress_cb, cancel_check) -> object
    """

    started = pyqtSignal()
    progress = pyqtSignal(int, str)
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    _worker_started = pyqtSignal()
    _worker_progress = pyqtSignal(int, str)
    _worker_completed = pyqtSignal(object)
    _worker_failed = pyqtSignal(str)
    _worker_cancelled = pyqtSignal()

    def __init__(self, fn: Callable[[Callable[[int, str], None], Callable[[], bool]], object]):
        super().__init__()
        self._fn = fn
        self._cancel_requested = False
        self._debug_name = f"AsyncJob:{next(_JOB_COUNTER)}"
        self._signal_relay = _AsyncJobSignalRelay(self)
        self._worker_started.connect(
            self._signal_relay.relay_started,
            Qt.ConnectionType.QueuedConnection,
        )
        self._worker_progress.connect(
            self._signal_relay.relay_progress,
            Qt.ConnectionType.QueuedConnection,
        )
        self._worker_completed.connect(
            self._signal_relay.relay_completed,
            Qt.ConnectionType.QueuedConnection,
        )
        self._worker_failed.connect(
            self._signal_relay.relay_failed,
            Qt.ConnectionType.QueuedConnection,
        )
        self._worker_cancelled.connect(
            self._signal_relay.relay_cancelled,
            Qt.ConnectionType.QueuedConnection,
        )

    def cancel(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        self._worker_started.emit()
        try:
            result = self._fn(self._emit_progress, self._is_cancelled)
        except Cancelled:
            self._worker_cancelled.emit()
        except Exception as exc:
            self._worker_failed.emit(str(exc))
        else:
            self._worker_completed.emit(result)

    def _emit_progress(self, value: int, status: str) -> None:
        self._worker_progress.emit(int(value), str(status))

    def _is_cancelled(self) -> bool:
        return bool(self._cancel_requested)


__all__ = ["AsyncJob", "start_job_in_thread", "raise_if_cancelled", "Cancelled"]


def start_job_in_thread(job: AsyncJob) -> QThread:
    """
    Starts a job in its own QThread and returns the thread.
    Caller owns lifecycle references to keep them alive.
    """

    thread = QThread()
    thread.setObjectName(job._debug_name)
    job.moveToThread(thread)
    thread.started.connect(job.run)
    thread.started.connect(lambda: append_qt_diagnostic(f"[ThreadStart] {thread.objectName()}"))

    def _stop_thread(*_args) -> None:
        thread.quit()

    job._worker_completed.connect(_stop_thread)
    job._worker_failed.connect(_stop_thread)
    job._worker_cancelled.connect(_stop_thread)
    thread.finished.connect(lambda: append_qt_diagnostic(f"[ThreadFinish] {thread.objectName()}"))
    thread.finished.connect(job.deleteLater)
    thread.finished.connect(job._signal_relay.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread
