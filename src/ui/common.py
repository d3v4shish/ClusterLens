from __future__ import annotations

from PyQt6.QtCore import QPoint, QRect, QSize, Qt
from PyQt6.QtWidgets import QApplication, QLayout, QLabel, QTabBar, QToolButton, QToolTip, QWidget

from infra.logging_config import get_logger

Log = get_logger(__name__)


class KeyboardTabBar(QTabBar):
    """A tab bar that uses arrows within tabs and Tab to leave the control."""

    def focusNextPrevChild(self, forward: bool) -> bool:  # noqa: N802 - Qt API override
        candidate = self.nextInFocusChain() if forward else self.previousInFocusChain()
        while candidate is not self:
            if (
                candidate.isEnabled()
                and candidate.isVisibleTo(self.window())
                and bool(candidate.focusPolicy() & Qt.FocusPolicy.TabFocus)
            ):
                candidate.setFocus(
                    Qt.FocusReason.TabFocusReason if forward else Qt.FocusReason.BacktabFocusReason
                )
                if QApplication.focusWidget() is candidate:
                    return True
            candidate = candidate.nextInFocusChain() if forward else candidate.previousInFocusChain()
        return False


class ResponsiveFlowLayout(QLayout):
    """Lay compact controls left-to-right and wrap only when space runs out.

    This is deliberately small and widget-agnostic.  It lets desktop-wide
    action strips use their available space without forcing narrow windows to
    clip controls or reserve empty grid cells.
    """

    def __init__(self, parent=None, *, spacing: int = 6) -> None:
        super().__init__(parent)
        self._items: list[object] = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(max(0, int(spacing)))

    def __del__(self) -> None:
        while self.takeAt(0) is not None:
            pass

    def addItem(self, item) -> None:  # noqa: N802 - Qt API override
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802 - Qt API override
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):  # noqa: N802 - Qt API override
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):  # noqa: N802 - Qt API override
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt API override
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt API override
        return self._layout(QRect(0, 0, max(0, int(width)), 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 - Qt API override
        super().setGeometry(rect)
        self._layout(rect, test_only=False)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API override
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt API override
        size = QSize()
        for item in self._items:
            if item is not None and not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def _layout(self, rect: QRect, *, test_only: bool) -> int:
        margins = self.contentsMargins()
        available = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        point = QPoint(available.x(), available.y())
        line_height = 0
        spacing = max(0, self.spacing())
        for item in self._items:
            if item is None or item.isEmpty():
                continue
            hint = item.sizeHint().expandedTo(item.minimumSize())
            width = min(max(0, hint.width()), max(0, available.width()))
            if line_height and point.x() + width > available.right() + 1:
                point.setX(available.x())
                point.setY(point.y() + line_height + spacing)
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(point, QSize(width, hint.height())))
            point.setX(point.x() + width + spacing)
            line_height = max(line_height, hint.height())
        return max(0, point.y() + line_height - rect.y() + margins.bottom())


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
        self.setFixedSize(20, 20)
        self.setProperty("helpIcon", True)
        self.clicked.connect(self._show_help)
        if help_key:
            self.setObjectName(f"helpIcon_{help_key}")

    def _show_help(self) -> None:
        QToolTip.showText(self.mapToGlobal(self.rect().bottomLeft()), self.toolTip(), self)


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
