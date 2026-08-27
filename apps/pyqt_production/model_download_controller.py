from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, pyqtSignal

from app.services.model_downloads import ModelDownloadItem, normalize_model_download_items
from apps.shared.runtime_support import RuntimeLayout
from infra.atomic_io import atomic_write_text


LOGGER = logging.getLogger(__name__)


class ModelDownloadController(QObject):
    """Runs all model acquisition in one visible, cancellable worker process."""

    started = pyqtSignal(tuple)
    progress = pyqtSignal(int, str)
    completed = pyqtSignal(dict)
    failed = pyqtSignal(str)
    cancelled = pyqtSignal()
    running_changed = pyqtSignal(bool)

    def __init__(self, runtime_layout: RuntimeLayout, parent=None) -> None:
        super().__init__(parent)
        self.runtime_layout = runtime_layout
        self.repo_root = Path(__file__).resolve().parents[2]
        # Preserve the active virtual environment instead of resolving through
        # its python symlink to an interpreter without project dependencies.
        self.worker_program = os.path.abspath(sys.executable)
        self._process: QProcess | None = None
        self._request_file: Path | None = None
        self._items: tuple[ModelDownloadItem, ...] = ()
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._result_payload: dict[str, object] | None = None
        self._error_message = ""
        self._cancel_requested = False
        self._terminal_emitted = False

    def is_running(self) -> bool:
        return self._process is not None

    @property
    def active_items(self) -> tuple[ModelDownloadItem, ...]:
        return self._items

    def start(self, items: list[ModelDownloadItem] | tuple[ModelDownloadItem, ...]) -> bool:
        if self.is_running():
            return False
        normalized = normalize_model_download_items(items)
        if not normalized:
            self.failed.emit("The model download request is empty.")
            return False

        self._items = normalized
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._result_payload = None
        self._error_message = ""
        self._cancel_requested = False
        self._terminal_emitted = False

        request_dir = self.runtime_layout.cache_dir / "tmp"
        request_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self._request_file = request_dir / f"model_download_request_{stamp}.json"
        payload = {
            "items": [
                {"model_name": item.model_name, "require_text": bool(item.require_text)}
                for item in normalized
            ]
        }
        atomic_write_text(self._request_file, json.dumps(payload, indent=2))

        process = QProcess(self)
        process.setProgram(self.worker_program)
        process.setArguments(self._worker_arguments(self._request_file))
        process.setWorkingDirectory(str(self.repo_root))
        process.setProcessEnvironment(self._worker_environment())
        process.readyReadStandardOutput.connect(lambda process=process: self._read_stdout(process))
        process.readyReadStandardError.connect(lambda process=process: self._read_stderr(process))
        process.finished.connect(
            lambda exit_code, exit_status, process=process: self._on_finished(process, exit_code, exit_status)
        )
        process.errorOccurred.connect(lambda error, process=process: self._on_error(process, error))
        self._process = process
        LOGGER.info(
            "Starting model download worker | program=%s args=%s items=%s",
            self.worker_program,
            process.arguments(),
            [item.key for item in normalized],
        )
        process.start()
        self.started.emit(tuple(item.key for item in normalized))
        self.running_changed.emit(True)
        return True

    def cancel(self) -> None:
        process = self._process
        if process is None:
            return
        self._cancel_requested = True
        self.progress.emit(-1, "Cancelling model download; cached partial files will be reused on retry...")
        try:
            process.kill()
        except RuntimeError:
            return

    def shutdown(self, timeout_ms: int = 2500) -> bool:
        process = self._process
        if process is None:
            return True
        self._cancel_requested = True
        try:
            process.kill()
            finished = bool(process.waitForFinished(int(timeout_ms)))
            if finished and process is self._process:
                self._on_finished(process, process.exitCode(), process.exitStatus())
            return finished
        except (RuntimeError, TypeError):
            return False

    def _worker_arguments(self, request_file: Path) -> list[str]:
        if bool(getattr(sys, "frozen", False)):
            return ["--worker", "--model-request-json", str(request_file)]
        return ["-m", "apps.pyqt_production.worker", "--model-request-json", str(request_file)]

    def _worker_environment(self) -> QProcessEnvironment:
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("CLUSTERLENS_RUNTIME_ROOT", str(self.runtime_layout.root))
        environment.insert("IMAGE_CLUSTERING_APP_DIR", str(self.runtime_layout.root))
        environment.insert("IMAGE_CLUSTERING_MODEL_ASSETS_DIR", str(self.runtime_layout.model_assets_dir))
        environment.insert("PYTHONUNBUFFERED", "1")
        for key, value in os.environ.items():
            if key.startswith("QT_") or key in {"PATH", "PYTHONPATH"}:
                environment.insert(key, value)
        return environment

    def _read_stdout(self, process: QProcess) -> None:
        try:
            text = bytes(process.readAllStandardOutput()).decode("utf-8", errors="replace")
        except RuntimeError:
            return
        if process is not self._process:
            return
        self._stdout_buffer += text
        while "\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split("\n", 1)
            self._handle_stdout_line(line)

    def _handle_stdout_line(self, line: str) -> None:
        line = str(line or "").strip()
        if not line:
            return
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            LOGGER.warning("Ignoring non-JSON model worker output: %s", line)
            return
        event_type = str(event.get("type") or "")
        body = dict(event.get("payload") or {})
        if event_type == "progress":
            self.progress.emit(int(body.get("value", -1) or -1), str(body.get("status") or ""))
        elif event_type == "result":
            self._result_payload = body
        elif event_type == "error":
            self._error_message = str(body.get("message") or "Model download worker failed.")
            if body.get("traceback"):
                LOGGER.error("Model download worker traceback:\n%s", body.get("traceback"))

    def _read_stderr(self, process: QProcess) -> None:
        try:
            text = bytes(process.readAllStandardError()).decode("utf-8", errors="replace")
        except RuntimeError:
            return
        if process is not self._process:
            return
        self._stderr_buffer += text
        if text.strip():
            LOGGER.warning("Model download worker stderr: %s", text.rstrip())

    def _on_error(self, process: QProcess, error) -> None:  # noqa: ANN001
        if not self._qobject_alive():
            return
        if process is not self._process:
            return
        self._error_message = self._error_message or (
            f"Model download process error: {error} | program={self.worker_program} | cwd={self.repo_root}"
        )
        LOGGER.error(self._error_message)
        if error == QProcess.ProcessError.FailedToStart:
            # Qt does not guarantee a finished signal after FailedToStart.
            # Finalize on the event loop so Jobs never remains permanently
            # stuck in a running state.
            QTimer.singleShot(0, lambda process=process: self._finalize_failed_start(process))

    def _finalize_failed_start(self, process: QProcess) -> None:
        if process is not self._process:
            return
        self._on_finished(process, -1, QProcess.ExitStatus.CrashExit)

    def _on_finished(self, process: QProcess, exit_code: int, exit_status) -> None:  # noqa: ANN001
        if not self._qobject_alive():
            return
        if process is not self._process:
            try:
                process.deleteLater()
            except RuntimeError:
                pass
            return
        self._read_stdout(process)
        self._read_stderr(process)
        if self._stdout_buffer.strip():
            self._handle_stdout_line(self._stdout_buffer)
        self._stdout_buffer = ""
        self._process = None
        self.running_changed.emit(False)
        self._remove_request_file()
        try:
            process.deleteLater()
        except RuntimeError:
            pass
        if self._terminal_emitted:
            return
        self._terminal_emitted = True
        if self._cancel_requested:
            self.cancelled.emit()
        elif self._result_payload is not None and int(exit_code) == 0:
            self.completed.emit(dict(self._result_payload))
        else:
            message = self._error_message or self._stderr_buffer.strip()
            if not message:
                message = f"Model download worker exited with code {exit_code} ({exit_status})."
            self.failed.emit(message)
        self._items = ()

    def _qobject_alive(self) -> bool:
        try:
            self.thread()
        except RuntimeError:
            return False
        return True

    def _remove_request_file(self) -> None:
        request_file = self._request_file
        self._request_file = None
        if request_file is None:
            return
        try:
            request_file.unlink(missing_ok=True)
        except OSError:
            LOGGER.warning("Could not remove model download request file: %s", request_file, exc_info=True)
