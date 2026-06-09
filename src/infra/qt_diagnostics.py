from __future__ import annotations

from datetime import datetime
from pathlib import Path
from threading import Lock

from PyQt6.QtCore import QtMsgType, qInstallMessageHandler

from .settings import get_settings

_INSTALL_LOCK = Lock()
_APPEND_LOCK = Lock()


def _diagnostic_path() -> Path:
    settings = get_settings()
    path = Path(settings.log_file).with_name("qt_diagnostics.log")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def append_qt_diagnostic(message: str) -> None:
    line = f"{datetime.now().isoformat(timespec='milliseconds')} {message}\n"
    with _APPEND_LOCK:
        with _diagnostic_path().open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.flush()


def install_qt_message_handler() -> None:
    if getattr(install_qt_message_handler, "_installed", False):
        return
    with _INSTALL_LOCK:
        if getattr(install_qt_message_handler, "_installed", False):
            return

        def _handler(msg_type, context, message) -> None:
            del context
            type_name = {
                QtMsgType.QtDebugMsg: "DEBUG",
                QtMsgType.QtInfoMsg: "INFO",
                QtMsgType.QtWarningMsg: "WARNING",
                QtMsgType.QtCriticalMsg: "CRITICAL",
                QtMsgType.QtFatalMsg: "FATAL",
            }.get(msg_type, "UNKNOWN")
            append_qt_diagnostic(f"[Qt/{type_name}] {message}")

        qInstallMessageHandler(_handler)
        install_qt_message_handler._installed = True
