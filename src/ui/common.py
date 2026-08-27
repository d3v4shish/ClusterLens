from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QToolButton, QWidget

from infra.logging_config import get_logger

Log = get_logger(__name__)


class HelpIconButton(QToolButton):
    def __init__(self, tooltip_text: str, parent=None, *, help_key: str = "") -> None:
        super().__init__(parent)
        self.setText("i")
        self.setToolTip(str(tooltip_text or ""))
        self.setAccessibleName(f"Help: {help_key.replace('_', ' ') or 'more information'}")
        self.setAccessibleDescription(str(tooltip_text or ""))
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAutoRaise(True)
        self.setFixedSize(32, 32)
        self.setProperty("helpIcon", True)
        if help_key:
            self.setObjectName(f"helpIcon_{help_key}")


def build_help_inline(
    primary_widget,
    tooltip_text: str,
    *,
    help_key: str = "",
    primary_stretch: int = 0,
    parent=None,
) -> QWidget:
    _ = help_key
    _ = primary_stretch
    _ = parent
    primary_widget.setToolTip(str(tooltip_text or ""))
    return primary_widget


def build_help_label(text: str, tooltip_text: str, *, help_key: str = "", parent=None) -> QWidget:
    label = QLabel(str(text or ""), parent)
    if help_key:
        label.setObjectName(f"helpLabel_{help_key}")
    label.setToolTip(str(tooltip_text or ""))
    return label
