from __future__ import annotations

from collections.abc import Callable
from itertools import count
from time import monotonic, sleep

from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot

from infra.cancel import Cancelled, raise_if_cancelled
from infra.qt_diagnostics import append_qt_diagnostic

_JOB_COUNTER = count(1)
_DETACHED_ASYNC_REFS: list[tuple[object | None, object | None]] = []


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
        if not self._emit_worker_signal("_worker_started"):
            return
        try:
            result = self._fn(self._emit_progress, self._is_cancelled)
        except Cancelled:
            append_qt_diagnostic(f"[JobCancelled] {self._debug_name}")
            self._emit_worker_signal("_worker_cancelled")
        except Exception as exc:
            append_qt_diagnostic(f"[JobFailed] {self._debug_name} error={exc}")
            self._emit_worker_signal("_worker_failed", str(exc))
        else:
            append_qt_diagnostic(f"[JobCompleted] {self._debug_name}")
            self._emit_worker_signal("_worker_completed", result)

    def _emit_progress(self, value: int, status: str) -> None:
        if not self._emit_worker_signal("_worker_progress", int(value), str(status)):
            raise Cancelled()

    def _emit_worker_signal(self, signal_name: str, *args) -> bool:
        try:
            signal = getattr(self, signal_name)
            signal.emit(*args)
        except RuntimeError:
            # Qt may delete the QObject during shutdown while a worker is
            # unwinding. Treat that as cancelled instead of crashing the app.
            return False
        return True

    def _is_cancelled(self) -> bool:
        return bool(self._cancel_requested)


def wait_for_thread_shutdown(
    thread,
    *,
    timeout_ms: int = 2500,
    quit_thread: bool = True,
    poll_interval_s: float = 0.01,
) -> bool:
    if thread is None:
        return True
    try:
        if not thread.isRunning():
            return True
    except RuntimeError:
        return True
    except Exception:
        return False
    try:
        if quit_thread:
            thread.quit()
    except RuntimeError:
        return True
    except Exception:
        pass
    timeout_ms = max(0, int(timeout_ms))
    if not isinstance(thread, QThread):
        try:
            return bool(thread.wait(timeout_ms))
        except RuntimeError:
            return True
        except Exception:
            return False
    deadline = monotonic() + (timeout_ms / 1000.0)
    poll_interval_s = min(0.05, max(0.001, float(poll_interval_s)))
    while monotonic() < deadline:
        try:
            if not thread.isRunning():
                return True
        except RuntimeError:
            return True
        except Exception:
            return False
        sleep(min(poll_interval_s, max(0.0, deadline - monotonic())))
    try:
        return not thread.isRunning()
    except RuntimeError:
        return True
    except Exception:
        return False


def _release_detached_async_refs(thread=None) -> None:
    global _DETACHED_ASYNC_REFS
    if thread is None:
        return
    _DETACHED_ASYNC_REFS = [
        (job, retained_thread)
        for job, retained_thread in _DETACHED_ASYNC_REFS
        if retained_thread is not thread
    ]


def detach_running_async_job(job: object | None, thread) -> bool:
    if thread is None:
        return False
    try:
        if not thread.isRunning():
            return False
    except RuntimeError:
        return False
    except Exception:
        return False
    if not any(retained_thread is thread for _retained_job, retained_thread in _DETACHED_ASYNC_REFS):
        _DETACHED_ASYNC_REFS.append((job, thread))
    if job is not None:
        for signal_name in ("started", "progress", "completed", "failed", "cancelled"):
            signal = getattr(job, signal_name, None)
            if signal is None:
                continue
            try:
                signal.disconnect()
            except Exception:
                pass
    finished = getattr(thread, "finished", None)
    if finished is not None:
        try:
            finished.disconnect()
        except Exception:
            pass
        try:
            finished.connect(
                lambda thread=thread: _release_detached_async_refs(thread),
                Qt.ConnectionType.QueuedConnection,
            )
        except Exception:
            pass
        for cleanup in (
            getattr(job, "deleteLater", None),
            getattr(getattr(job, "_signal_relay", None), "deleteLater", None),
            getattr(thread, "deleteLater", None),
        ):
            if cleanup is None:
                continue
            try:
                finished.connect(cleanup)
            except Exception:
                pass
    return True


__all__ = [
    "AsyncJob",
    "start_job_in_thread",
    "wait_for_thread_shutdown",
    "detach_running_async_job",
    "raise_if_cancelled",
    "Cancelled",
]


class _AsyncJobThread(QThread):
    def __init__(self, job: AsyncJob):
        super().__init__()
        self._job = job
        self.setObjectName(job._debug_name)

    def run(self) -> None:
        append_qt_diagnostic(f"[ThreadStart] {self.objectName()}")
        self._job.run()
        append_qt_diagnostic(f"[ThreadRunReturn] {self.objectName()}")


def start_job_in_thread(job: AsyncJob) -> QThread:
    """
    Starts a job in its own QThread and returns the thread.
    Caller owns lifecycle references to keep them alive.
    """

    thread = _AsyncJobThread(job)
    job.moveToThread(thread)
    thread_name = thread.objectName()
    thread.finished.connect(lambda name=thread_name: append_qt_diagnostic(f"[ThreadFinish] {name}"))
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread
