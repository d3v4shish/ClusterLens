from __future__ import annotations

from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtWidgets import QApplication


ULTRA_DARK_QSS = """
/* Matte black ultra-dark theme */
QWidget {
  background: #0B0B0B;
  color: #EDEDED;
  font-size: 12px;
}

QToolTip {
  background: #111111;
  color: #EDEDED;
  border: 1px solid #2A2A2A;
  padding: 6px;
}

QGroupBox {
  border: 1px solid #2A2A2A;
  border-radius: 8px;
  margin-top: 10px;
  padding: 10px;
  background: #0E0E0E;
}
QGroupBox::title {
  subcontrol-origin: margin;
  left: 10px;
  padding: 0 6px 0 6px;
  color: #CFCFCF;
}

QLabel {
  background: transparent;
}

QLineEdit, QTextEdit, QPlainTextEdit, QListWidget, QTreeView, QTableView, QComboBox, QSpinBox, QDoubleSpinBox {
  background: #111111;
  border: 1px solid #2A2A2A;
  border-radius: 8px;
  padding: 6px;
  selection-background-color: #1B3A66;
  selection-color: #FFFFFF;
}

QComboBox::drop-down {
  border: 0px;
}
QComboBox::down-arrow {
  width: 10px;
  height: 10px;
}

QPushButton {
  background: #151515;
  border: 1px solid #2A2A2A;
  border-radius: 8px;
  padding: 6px 10px;
}
QPushButton:hover {
  background: #1A1A1A;
  border: 1px solid #3A3A3A;
}
QPushButton:pressed {
  background: #101010;
  border: 1px solid #4A4A4A;
}
QPushButton:checked {
  background: #1B3A66;
  border: 1px solid #3C6FB6;
  color: #FFFFFF;
}
QPushButton:disabled {
  background: #101010;
  color: #777777;
  border: 1px solid #222222;
}

QToolButton[helpIcon="true"] {
  background: #111111;
  color: #CFCFCF;
  border: 1px solid #2E2E2E;
  border-radius: 8px;
  padding: 0px;
  font-size: 10px;
  font-weight: 700;
}
QToolButton[helpIcon="true"]:hover {
  background: #1A1A1A;
  border: 1px solid #3A3A3A;
  color: #FFFFFF;
}
QToolButton[helpIcon="true"]:pressed {
  background: #101010;
  border: 1px solid #4A4A4A;
}

/* Navigation buttons in the top toolbar */
QPushButton[nav="true"] {
  padding: 7px 12px;
  font-weight: 600;
}
QPushButton[nav="true"]:checked {
  background: #1B3A66;
  border: 1px solid #3C6FB6;
}

QPushButton[paneToggle="true"] {
  padding: 4px 8px;
  font-size: 11px;
  color: #B9B9B9;
}
QPushButton[paneToggle="true"]:checked {
  background: #1B3A66;
  border: 1px solid #3C6FB6;
  color: #FFFFFF;
}

QCheckBox {
  spacing: 8px;
}

QTabWidget::pane {
  border: 1px solid #2A2A2A;
  border-radius: 8px;
  top: -1px;
  background: #0E0E0E;
}
QTabBar::tab {
  background: #101010;
  border: 1px solid #2A2A2A;
  border-bottom: none;
  border-top-left-radius: 8px;
  border-top-right-radius: 8px;
  padding: 6px 10px;
  margin-right: 4px;
  color: #CFCFCF;
}
QTabBar::tab:selected {
  background: #0E0E0E;
  color: #FFFFFF;
  border: 1px solid #3A3A3A;
}
QTabBar::tab:hover {
  border: 1px solid #3A3A3A;
}

/* Workspace tabs in the top toolbar */
QTabBar#workspaceTabs::tab {
  background: #101010;
  border: 1px solid #2A2A2A;
  border-radius: 10px;
  padding: 7px 12px;
  margin-right: 6px;
  color: #CFCFCF;
  font-weight: 600;
}
QTabBar#workspaceTabs::tab:selected {
  background: #1B3A66;
  border: 1px solid #3C6FB6;
  color: #FFFFFF;
}
QTabBar#workspaceTabs::tab:hover {
  border: 1px solid #3A3A3A;
}

/* Search mode tabs promoted into the top toolbar */
QTabBar#workspaceModeTabs::tab {
  background: #0E0E0E;
  border: 1px solid #242424;
  border-radius: 9px;
  padding: 6px 11px;
  margin-right: 6px;
  color: #BEBEBE;
  font-weight: 600;
}
QTabBar#workspaceModeTabs::tab:selected {
  background: #151515;
  border: 1px solid #3A3A3A;
  color: #FFFFFF;
}
QTabBar#workspaceModeTabs::tab:hover {
  border: 1px solid #3A3A3A;
}

QHeaderView::section {
  background: #101010;
  color: #CFCFCF;
  padding: 6px;
  border: 1px solid #2A2A2A;
}

QScrollBar:vertical, QScrollBar:horizontal {
  background: #0B0B0B;
  border: 1px solid #1A1A1A;
  margin: 0px;
}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
  background: #2A2A2A;
  border-radius: 6px;
  min-height: 18px;
  min-width: 18px;
}
QScrollBar::handle:hover {
  background: #3A3A3A;
}
QScrollBar::add-line, QScrollBar::sub-line {
  height: 0px;
  width: 0px;
}
QScrollBar::add-page, QScrollBar::sub-page {
  background: none;
}

QProgressBar {
  border: 1px solid #2A2A2A;
  border-radius: 8px;
  text-align: center;
  background: #101010;
}
QProgressBar::chunk {
  background: #1F6FEB;
  border-radius: 8px;
}

QWidget#workspaceFooter {
  background: #0D0D0D;
  border-top: 1px solid #1A1A1A;
}
QWidget#workspaceFooter QLabel {
  color: #AFAFAF;
  font-size: 11px;
}
QWidget#workspaceFooter QProgressBar {
  border: 1px solid #202020;
  border-radius: 4px;
  background: #101010;
}
QWidget#workspaceFooter QProgressBar::chunk {
  border-radius: 4px;
}

QSplitter::handle {
  background: #111111;
}
QSplitter::handle:hover {
  background: #242424;
}
QSplitter::handle:horizontal {
  width: 4px;
  margin: 8px 0;
  border-radius: 2px;
}
QSplitter::handle:vertical {
  height: 4px;
  margin: 0 8px;
  border-radius: 2px;
}

QMenu {
  background: #111111;
  color: #EDEDED;
  border: 1px solid #2A2A2A;
}
QMenu::item:selected {
  background: #1B3A66;
}
"""


def apply_ultra_dark(app: QApplication) -> None:
    # Ensure consistent widget rendering across platforms.
    try:
        app.setStyle("Fusion")
    except Exception:
        pass

    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor("#0B0B0B"))
    palette.setColor(QPalette.ColorRole.WindowText, QColor("#EDEDED"))
    palette.setColor(QPalette.ColorRole.Base, QColor("#111111"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor("#0E0E0E"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor("#111111"))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor("#EDEDED"))
    palette.setColor(QPalette.ColorRole.Text, QColor("#EDEDED"))
    palette.setColor(QPalette.ColorRole.Button, QColor("#151515"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor("#EDEDED"))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#1B3A66"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)
    app.setStyleSheet(ULTRA_DARK_QSS)
