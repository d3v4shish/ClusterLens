from __future__ import annotations

from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtWidgets import QApplication


# Authoritative UI design tokens. Custom-painted widgets import these values so
# the palette does not drift away from the stylesheet.
COLORS = {
    "canvas": "#0B0D10",
    "surface": "#11151A",
    "surface_raised": "#171C22",
    "surface_sunken": "#0F1318",
    "surface_selected": "#102A46",
    "surface_checked": "#102218",
    "overlay": "#000000A0",
    "border": "#303844",
    "border_strong": "#465366",
    "text": "#F2F5F8",
    "text_muted": "#B5BEC9",
    "accent": "#2F81F7",
    "accent_hover": "#4893FA",
    "focus": "#7DB7FF",
    "success": "#238636",
    "success_bright": "#3FB950",
    "warning": "#D29922",
    "info": "#58A6FF",
    "danger": "#DA3633",
    "danger_hover": "#F85149",
    "danger_surface": "#3A1313",
    "danger_text": "#FFB4B4",
    "identity_pending": "#A371F7",
}

METRICS = {
    "body_font_px": 13,
    "helper_font_px": 11,
    "control_min_height_px": 32,
    "radius_px": 8,
    "space_px": 8,
}


ULTRA_DARK_QSS = """
/* ClusterLens dark design system */
QWidget {
  background: #0B0D10;
  color: #F2F5F8;
  font-size: 13px;
}

QToolTip {
  background: #171C22;
  color: #F2F5F8;
  border: 1px solid #465366;
  padding: 6px;
}

QGroupBox {
  border: 1px solid #303844;
  border-radius: 8px;
  margin-top: 10px;
  padding: 10px;
  background: #11151A;
}
QGroupBox::title {
  subcontrol-origin: margin;
  left: 10px;
  padding: 0 6px 0 6px;
  color: #D4DBE4;
}

QLabel {
  background: transparent;
}

QLineEdit, QTextEdit, QPlainTextEdit, QListWidget, QTreeView, QTableView, QComboBox, QSpinBox, QDoubleSpinBox {
  background: #11151A;
  border: 1px solid #303844;
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
  background: #171C22;
  border: 1px solid #303844;
  border-radius: 8px;
  padding: 6px 10px;
  min-height: 20px;
}
QGroupBox[role="danger"] { border-color: #DA3633; background: #1B1010; }
QGroupBox[role="danger"]::title { color: #FFAAA6; }
QPushButton:hover {
  background: #202731;
  border: 1px solid #465366;
}
QPushButton:pressed {
  background: #101010;
  border: 1px solid #4A4A4A;
}
QPushButton:checked {
  background: #1F5FAF;
  border: 1px solid #7DB7FF;
  color: #FFFFFF;
}
QPushButton:disabled {
  background: #101010;
  color: #777777;
  border: 1px solid #222222;
}

QPushButton[kind="primary"] {
  background: #2F81F7;
  border: 1px solid #7DB7FF;
  color: #FFFFFF;
  font-weight: 650;
}
QPushButton[kind="primary"]:hover { background: #4893FA; }
QPushButton[kind="danger"] {
  background: #491515;
  border: 1px solid #DA3633;
  color: #FFD7D5;
  font-weight: 650;
}
QPushButton[kind="danger"]:hover { background: #6A1F1F; border-color: #F85149; }
QPushButton[kind="quiet"] { background: transparent; border-color: transparent; color: #B5BEC9; }
QPushButton[state="success"] { border-color: #238636; color: #A7F3B4; }
QPushButton[state="warning"] { border-color: #D29922; color: #F0C674; }
QPushButton[state="error"] { border-color: #DA3633; color: #FFAAA6; }

QPushButton:focus, QToolButton:focus, QLineEdit:focus, QTextEdit:focus,
QPlainTextEdit:focus, QListWidget:focus, QListView:focus, QTreeView:focus,
QTableView:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
QCheckBox:focus, QTabBar::tab:focus {
  border: 2px solid #7DB7FF;
}

QToolButton[helpIcon="true"] {
  background: #111111;
  color: #CFCFCF;
  border: 1px solid #2E2E2E;
  border-radius: 8px;
  padding: 0px;
  font-size: 11px;
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
  font-size: 12px;
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
  color: #B5BEC9;
  font-size: 12px;
}

QWidget#applicationHeader {
  background: #11151A;
  border: 1px solid #252D38;
  border-radius: 10px;
}
QWidget#workspaceCanvas { background: #0B0D10; }
QWidget#sidebarSurface, QWidget#detailsSurface {
  background: #0F1318;
  border: 1px solid #252D38;
  border-radius: 8px;
}
QWidget#emptyStateCard {
  background: #11151A;
  border: 1px solid #303844;
  border-radius: 12px;
}
QLabel[role="title"] { font-size: 20px; font-weight: 700; color: #F2F5F8; }
QLabel[role="section"] { font-size: 15px; font-weight: 650; color: #F2F5F8; }
QLabel[role="helper"] { font-size: 11px; color: #B5BEC9; }
QLabel[state="warning"] { color: #F0C674; }
QLabel[state="error"] { color: #FF8A86; }
QLabel[state="success"] { color: #7EE787; }
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
    palette.setColor(QPalette.ColorRole.Window, QColor(COLORS["canvas"]))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Base, QColor(COLORS["surface"]))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(COLORS["surface_raised"]))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(COLORS["surface_raised"]))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.Button, QColor(COLORS["surface_raised"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(COLORS["text"]))
    palette.setColor(QPalette.ColorRole.BrightText, QColor("#FFFFFF"))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(COLORS["accent"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(palette)
    app.setStyleSheet(ULTRA_DARK_QSS)
