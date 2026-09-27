from __future__ import annotations

from functools import lru_cache
import re

from PyQt6.QtCore import QObject, Qt, pyqtSignal
from PyQt6.QtGui import QPalette, QColor
from PyQt6.QtWidgets import QApplication


# Authoritative UI design tokens. Custom-painted widgets import these values so
# the palette does not drift away from the stylesheet.
DARK_COLORS = {
    "canvas": "#121416",
    "surface": "#181B1E",
    "surface_raised": "#202428",
    "surface_sunken": "#0F1113",
    "surface_selected": "#293A47",
    "surface_checked": "#203126",
    "overlay": "#00000099",
    "border": "#30363B",
    "border_strong": "#454D54",
    "text": "#E8EAED",
    "text_muted": "#A9B0B7",
    "accent": "#466C88",
    "accent_hover": "#3D617A",
    "focus": "#79A7CC",
    "success": "#5F8A6B",
    "success_bright": "#7DA98A",
    "warning": "#B58A4B",
    "info": "#6E95B7",
    "danger": "#A94D4D",
    "danger_hover": "#913C3C",
    "danger_surface": "#352124",
    "danger_text": "#F2C4C4",
    "identity_pending": "#8E7BA4",
    "viewer_canvas": "#121416",
}

LIGHT_COLORS = {
    "canvas": "#F2F3F2",
    "surface": "#FBFCFB",
    "surface_raised": "#F0F2F1",
    "surface_sunken": "#E7E9E8",
    "surface_selected": "#DDE7ED",
    "surface_checked": "#E2ECE4",
    "overlay": "#1D252BB8",
    "border": "#C9CED0",
    "border_strong": "#ABB3B7",
    "text": "#252A2D",
    "text_muted": "#5E666D",
    "accent": "#456A86",
    "accent_hover": "#395B73",
    "focus": "#315F82",
    "success": "#4D7659",
    "success_bright": "#487052",
    "warning": "#896631",
    "info": "#456A86",
    "danger": "#A65252",
    "danger_hover": "#8E4141",
    "danger_surface": "#F4E5E5",
    "danger_text": "#6E2929",
    "identity_pending": "#705A89",
    "viewer_canvas": "#F2F3F2",
}

# Keep this dictionary object stable: custom-painted widgets import it directly.
# Theme changes mutate it in place so those consumers see the current tokens.
COLORS = dict(DARK_COLORS)

APP_THEME_MODES = ("system", "light", "dark")
VIEWER_BACKDROP_MODES = ("adaptive_neutral", "theme", "black", "middle_gray", "light_gray")
VIEWER_BACKDROP_COLORS = {
    "black": "#050607",
    "middle_gray": "#30343A",
    "light_gray": "#E6E8EB",
}

_theme_revision = 0
_theme_manager: "ThemeManager | None" = None

# The application stylesheet references these palette roles instead of baking
# resolved theme hex values into every selector.  A changed theme can therefore
# update QPalette without forcing Qt to parse and repolish the complete widget
# tree through QApplication.setStyleSheet().  Geometry-changing density/text
# scale updates still rebuild the stylesheet deliberately.
_QSS_PALETTE_ROLES = {
    "canvas": (QPalette.ColorRole.Window, "window"),
    "surface": (QPalette.ColorRole.Base, "base"),
    "surface_raised": (QPalette.ColorRole.Button, "button"),
    "surface_sunken": (QPalette.ColorRole.AlternateBase, "alternate-base"),
    "surface_selected": (QPalette.ColorRole.Highlight, "highlight"),
    "surface_checked": (QPalette.ColorRole.Light, "light"),
    "border": (QPalette.ColorRole.Mid, "mid"),
    "border_strong": (QPalette.ColorRole.Dark, "dark"),
    "text": (QPalette.ColorRole.WindowText, "window-text"),
    "text_muted": (QPalette.ColorRole.PlaceholderText, "placeholder-text"),
    "accent": (QPalette.ColorRole.Link, "link"),
    "accent_hover": (QPalette.ColorRole.LinkVisited, "link-visited"),
    "focus": (QPalette.ColorRole.Accent, "accent"),
    "success": (QPalette.ColorRole.Midlight, "midlight"),
    "success_bright": (QPalette.ColorRole.BrightText, "bright-text"),
    "warning": (QPalette.ColorRole.ToolTipText, "tool-tip-text"),
    "danger": (QPalette.ColorRole.Shadow, "shadow"),
    "danger_hover": (QPalette.ColorRole.Shadow, "shadow"),
    "danger_surface": (QPalette.ColorRole.ToolTipBase, "tool-tip-base"),
    "danger_text": (QPalette.ColorRole.WindowText, "window-text"),
}

METRICS = {
    "body_font_px": 14,
    "helper_font_px": 12,
    "card_title_font_px": 14,
    "section_font_px": 16,
    "title_font_px": 20,
    "control_min_height_px": 34,
    "radius_px": 6,
    "space_px": 8,
}

# Use a compact desktop width for finite values and let text-heavy fields use
# the available grid width.  Layout builders call this deliberately rather
# than relying on a blanket rule that would make every combo box equally wide.
FIELD_WIDTHS = {
    "numeric": 96,
    "short": 160,
    "medium": 240,
}


def apply_field_size(widget, size: str) -> None:
    """Apply the shared input-width contract without changing widget behavior."""

    normalized = str(size or "medium").strip().lower()
    widget.setProperty("fieldSize", normalized)
    width = FIELD_WIDTHS.get(normalized)
    if width is None:
        return
    widget.setMinimumWidth(width)


ULTRA_DARK_QSS = """
/* ClusterLens dark design system */
QWidget {
  background: #0B0D10;
  color: #F2F5F8;
  font-size: 14px;
}

QToolTip {
  background: #171C22;
  color: #F2F5F8;
  border: 1px solid #465366;
  padding: 6px;
}

QGroupBox {
  border: 1px solid #3B4757;
  border-radius: 8px;
  margin-top: 12px;
  padding: 11px 10px 10px 10px;
  background: #111820;
}
QGroupBox::title {
  subcontrol-origin: margin;
  left: 11px;
  padding: 0 6px 0 6px;
  background: #111820;
  color: #DCE8F7;
  font-weight: 650;
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

QListView[sidebarList="true"], QListWidget[sidebarList="true"] {
  background: #11151A;
  border: 1px solid #303844;
  border-radius: 8px;
  padding: 2px;
  outline: 0;
}
QListView[sidebarList="true"]::item, QListWidget[sidebarList="true"]::item {
  border-bottom: 1px solid #202933;
  padding: 2px 4px;
}
QListView[sidebarList="true"]::item:selected, QListWidget[sidebarList="true"]::item:selected {
  background: #173554;
  color: #FFFFFF;
}
QListView[sidebarList="true"]::item:hover, QListWidget[sidebarList="true"]::item:hover {
  background: #16202C;
}

QComboBox::drop-down {
  border: 0px;
}
QComboBox::down-arrow {
  width: 10px;
  height: 10px;
}

QPushButton {
  background: #151C25;
  border: 1px solid #3B4757;
  border-radius: 8px;
  padding: 6px 10px;
  min-height: 20px;
}
QGroupBox[role="danger"] { border-color: #DA3633; background: #1B1010; }
QGroupBox[role="danger"]::title { color: #FFAAA6; }
QPushButton:hover {
  background: #202A36;
  border: 1px solid #5B6B80;
}
QPushButton:pressed {
  background: #101010;
  border: 1px solid #4A4A4A;
}
QPushButton:checked {
  background: #1F5FAF;
  border: 1px solid #58A6FF;
  color: #FFFFFF;
}
QPushButton:disabled {
  background: #101010;
  color: #777777;
  border: 1px solid #222222;
}

QPushButton[kind="primary"] {
  background: #1F6FEB;
  border: 1px solid #58A6FF;
  color: #FFFFFF;
  font-weight: 650;
}
QPushButton[kind="primary"]:hover { background: #4893FA; }
QPushButton[kind="primary"]:disabled {
  background: #132237;
  border: 1px solid #29496B;
  color: #71839A;
}
QPushButton[kind="secondary"] {
  background: #111B28;
  border: 1px solid #456A91;
  color: #D7E8FA;
  font-weight: 600;
}
QPushButton[kind="secondary"]:hover { background: #172A40; border-color: #6FA6DD; }
QPushButton[kind="secondary"]:disabled {
  background: #101721;
  border: 1px solid #263A4E;
  color: #71839A;
}
QToolButton[kind="secondary"] {
  background: #111B28;
  border: 1px solid #456A91;
  border-radius: 8px;
  color: #D7E8FA;
  font-weight: 600;
  padding: 6px 10px;
}
QToolButton[kind="secondary"]:hover { background: #172A40; border-color: #6FA6DD; }
QToolButton[kind="secondary"]:disabled { background: #101721; border-color: #263A4E; color: #71839A; }
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
  border: 2px solid #58A6FF;
}

QToolButton[helpIcon="true"] {
  background: #111111;
  color: #CFCFCF;
  border: 1px solid #2E2E2E;
  border-radius: 8px;
  padding: 0px;
  font-size: 12px;
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
  font-size: 13px;
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
  border: 1px solid #344153;
  border-radius: 8px;
  top: -1px;
  background: #0E0E0E;
}
QTabBar::tab {
  background: #0F141B;
  border: 1px solid #293545;
  border-bottom: none;
  border-top-left-radius: 8px;
  border-top-right-radius: 8px;
  padding: 6px 10px;
  margin-right: 4px;
  color: #B9C5D3;
  font-weight: 600;
}
QTabBar::tab:selected {
  background: #173554;
  color: #FFFFFF;
  border: 1px solid #5D9FE5;
  font-weight: 700;
}
QTabBar::tab:hover {
  background: #16202C;
  border: 1px solid #5B6B80;
}

/* Faces task navigation is primary navigation, not an ordinary button row. */
QTabBar#facesTaskNavigation::tab {
  background: #101721;
  border: 1px solid #2E3D4F;
  border-radius: 7px;
  padding: 6px 8px;
  margin: 1px;
  color: #C4D0DE;
  font-weight: 650;
}
QTabBar#facesTaskNavigation::tab:selected {
  background: #17416C;
  border: 1px solid #66AEF5;
  color: #FFFFFF;
  font-weight: 700;
}
QTabBar#facesTaskNavigation::tab:hover { background: #1A2C40; border-color: #6C91B8; }

/* Result-view tabs are subordinate to task navigation but retain a clear state. */
QTabBar#faceResultViewTabs::tab {
  background: #0E131A;
  border: 1px solid #2B3949;
  border-radius: 6px;
  padding: 6px 12px;
  margin: 2px 3px;
  color: #B9C7D6;
  font-weight: 600;
}
QTabBar#faceResultViewTabs::tab:selected {
  background: #153354;
  border: 1px solid #5794D2;
  color: #FFFFFF;
  font-weight: 700;
}
QTabBar#faceResultViewTabs::tab:hover { background: #182536; border-color: #5A7899; }

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

QTableView[clusterComparisonTable="true"] {
  background: #11151A;
  border: 1px solid #303844;
  border-radius: 8px;
  padding: 0px;
  selection-background-color: #173554;
}
QTableView[clusterComparisonTable="true"]::item { padding: 0px; }
QHeaderView[clusterComparisonHeader="true"]::section {
  background: #171C22;
  color: #DCE8F7;
  border: 0px;
  border-bottom: 1px solid #465366;
  padding: 8px 12px;
  font-weight: 650;
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
  font-size: 13px;
}
QLabel#footerFolderChip {
  color: #DCEBFF;
  background: #17365D;
  border: 1px solid #2D6FAD;
  border-radius: 4px;
  padding: 2px 6px;
}
QLabel#selectedFolderRoot {
  color: #BFD7F2;
  background: #141E2A;
  border: 1px solid #2A4057;
  border-radius: 3px;
  padding: 4px 6px;
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
QLabel[role="section"] { font-size: 16px; font-weight: 650; color: #F2F5F8; }
QLabel[role="helper"] { font-size: 12px; color: #B5BEC9; }
QDialog QLabel[role="dialogTitle"] { font-size: 20px; font-weight: 700; color: #F2F5F8; }
QDialog QGroupBox { background: #111820; }
QPushButton[helpIcon="true"] { min-width: 20px; max-width: 20px; min-height: 20px; max-height: 20px; padding: 0; color: #BFD7F2; }
QPushButton#facesStatusStrip {
  background: transparent;
  border: 0;
  border-radius: 3px;
  padding: 3px 2px;
  min-height: 18px;
  text-align: left;
  font-size: 14px;
  font-weight: 600;
  color: #DCEBFF;
}
QPushButton#facesStatusStrip:hover { background: #17365D; color: #FFFFFF; }
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


_LIGHT_QSS_REPLACEMENTS = {
    "#0B0D10": "#F5F7FA",
    "#0B0B0B": "#F5F7FA",
    "#0D0D0D": "#F5F7FA",
    "#0E0E0E": "#F5F7FA",
    "#0E131A": "#FFFFFF",
    "#0F141B": "#FFFFFF",
    "#111111": "#FFFFFF",
    "#11151A": "#FFFFFF",
    "#111820": "#FFFFFF",
    "#0F1318": "#E7ECF2",
    "#171C22": "#EEF2F6",
    "#151C25": "#EEF2F6",
    "#101010": "#E7ECF2",
    "#132237": "#E7ECF2",
    "#141E2A": "#E7ECF2",
    "#151515": "#EEF2F6",
    "#111B28": "#E7ECF2",
    "#101721": "#E7ECF2",
    "#102218": "#E2F3E6",
    "#1B1010": "#FFEBE9",
    "#3A1313": "#FFEBE9",
    "#491515": "#FFEBE9",
    "#6A1F1F": "#FFD7D5",
    "#202A36": "#DCE3EA",
    "#16202C": "#E7ECF2",
    "#172A40": "#DCEBFF",
    "#182536": "#DCEBFF",
    "#1A2C40": "#DCEBFF",
    "#102A46": "#DCEBFF",
    "#173554": "#DCEBFF",
    "#1B3A66": "#C7DFFF",
    "#1F5FAF": "#0969DA",
    "#1F6FEB": "#0969DA",
    "#153354": "#0969DA",
    "#17365D": "#0969DA",
    "#17416C": "#0969DA",
    "#2F81F7": "#0969DA",
    "#4893FA": "#0550AE",
    "#7DB7FF": "#0550AE",
    "#58A6FF": "#0550AE",
    "#303844": "#C5CFDA",
    "#1A1A1A": "#D8E0E8",
    "#202020": "#D8E0E8",
    "#242424": "#D8E0E8",
    "#252D38": "#C5CFDA",
    "#263A4E": "#C5CFDA",
    "#293545": "#C5CFDA",
    "#29496B": "#A9BDD2",
    "#2A4057": "#A9BDD2",
    "#2B3949": "#C5CFDA",
    "#2D6FAD": "#0550AE",
    "#2E2E2E": "#C5CFDA",
    "#2E3D4F": "#C5CFDA",
    "#344153": "#B7C2CE",
    "#3A3A3A": "#B7C2CE",
    "#3C6FB6": "#0550AE",
    "#3B4757": "#B7C2CE",
    "#4A4A4A": "#8C99A8",
    "#465366": "#8C99A8",
    "#5B6B80": "#768596",
    "#456A91": "#768596",
    "#5794D2": "#0550AE",
    "#5A7899": "#768596",
    "#5D9FE5": "#0550AE",
    "#66AEF5": "#0550AE",
    "#6C91B8": "#768596",
    "#6FA6DD": "#0550AE",
    "#2A2A2A": "#C5CFDA",
    "#202933": "#D8E0E8",
    "#222222": "#C5CFDA",
    "#F2F5F8": "#17202A",
    "#EDEDED": "#17202A",
    "#CFCFCF": "#526170",
    "#B9B9B9": "#526170",
    "#B9C5D3": "#526170",
    "#B9C7D6": "#526170",
    "#BEBEBE": "#526170",
    "#BFD7F2": "#0550AE",
    "#C4D0DE": "#263442",
    "#DCE8F7": "#263442",
    "#D7E8FA": "#263442",
    "#B5BEC9": "#526170",
    "#777777": "#697887",
    "#71839A": "#697887",
    "#FFAAA6": "#82071E",
    "#FFB4B4": "#82071E",
    "#FFD7D5": "#82071E",
    "#FF8A86": "#CF222E",
    "#F85149": "#CF222E",
    "#DA3633": "#CF222E",
    "#238636": "#1A7F37",
    "#7EE787": "#1A7F37",
    "#A7F3B4": "#1A7F37",
    "#D29922": "#9A6700",
    "#F0C674": "#7D4E00",
}


def _normalized_text_scale(value: object) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        return 100
    return normalized if normalized in {100, 125, 150, 200} else 100


@lru_cache(maxsize=8)
def stylesheet_for_theme(theme: str, *, compact: bool = True, text_scale: int = 100) -> str:
    resolved = "light" if str(theme).lower() == "light" else "dark"
    # Render once from the dark semantic values, then replace every dynamic
    # token with its palette role.  The resulting selectors are identical for
    # dark and light themes apart from their descriptive comment.
    colors = DARK_COLORS
    stylesheet = ULTRA_DARK_QSS
    if resolved == "light":
        stylesheet = stylesheet.replace("ClusterLens dark design system", "ClusterLens light design system")
    replacements = _legacy_qss_replacements(colors)
    pattern = re.compile("|".join(re.escape(source) for source in replacements))
    scale = _normalized_text_scale(text_scale)
    rendered = pattern.sub(lambda match: replacements[match.group(0)], stylesheet) + _matte_density_qss(
        colors,
        compact=compact,
        text_scale=scale,
    )
    if scale != 100:
        rendered = re.sub(
            r"font-size:\s*(\d+)px",
            lambda match: f"font-size: {max(1, round(int(match.group(1)) * scale / 100))}px",
            rendered,
        )
    for token, (_role, role_name) in _QSS_PALETTE_ROLES.items():
        rendered = rendered.replace(colors[token], f"palette({role_name})")
    return rendered


def _resolved_palette(colors: dict[str, str]) -> QPalette:
    palette = QPalette()
    for token, (role, _role_name) in _QSS_PALETTE_ROLES.items():
        palette.setColor(role, QColor(colors[token]))
    # Preserve conventional native-widget roles that are not referenced by
    # the application QSS.  Some platform dialogs and accessibility probes
    # read these directly.
    palette.setColor(QPalette.ColorRole.WindowText, QColor(colors["text"]))
    palette.setColor(QPalette.ColorRole.Text, QColor(colors["text"]))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(colors["text"]))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(colors["text"]))
    palette.setColor(QPalette.ColorRole.Shadow, QColor(colors["danger"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(colors["text_muted"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, QColor(colors["text_muted"]))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(colors["text_muted"]))
    return palette


def _legacy_qss_replacements(colors: dict[str, str]) -> dict[str, str]:
    """Translate retained component selectors onto the current semantic palette."""

    return {
        "#0B0D10": colors["canvas"], "#0B0B0B": colors["canvas"], "#0D0D0D": colors["canvas"],
        "#0E0E0E": colors["canvas"], "#0E131A": colors["surface"], "#0F141B": colors["surface"],
        "#111111": colors["surface"], "#11151A": colors["surface"], "#111820": colors["surface"],
        "#0F1318": colors["surface_sunken"], "#171C22": colors["surface_raised"],
        "#151C25": colors["surface_raised"], "#101010": colors["surface_sunken"],
        "#132237": colors["surface_sunken"], "#141E2A": colors["surface_sunken"],
        "#151515": colors["surface_raised"], "#111B28": colors["surface_raised"],
        "#101721": colors["surface_sunken"], "#102218": colors["surface_checked"],
        "#1B1010": colors["danger_surface"], "#3A1313": colors["danger_surface"],
        "#491515": colors["danger_surface"], "#6A1F1F": colors["danger"],
        "#202A36": colors["surface_raised"], "#16202C": colors["surface_raised"],
        "#172A40": colors["surface_selected"], "#182536": colors["surface_selected"],
        "#1A2C40": colors["surface_selected"], "#102A46": colors["surface_selected"],
        "#173554": colors["surface_selected"], "#1B3A66": colors["surface_selected"],
        "#1F5FAF": colors["accent"], "#1F6FEB": colors["accent"], "#153354": colors["accent"],
        "#17365D": colors["accent"], "#17416C": colors["accent"], "#2F81F7": colors["accent"],
        "#4893FA": colors["accent_hover"], "#7DB7FF": colors["accent_hover"],
        "#58A6FF": colors["focus"], "#303844": colors["border"],
        "#1A1A1A": colors["border"], "#202020": colors["border"], "#242424": colors["border"],
        "#252D38": colors["border"], "#263A4E": colors["border"], "#293545": colors["border"],
        "#29496B": colors["border_strong"], "#2A4057": colors["border_strong"],
        "#2B3949": colors["border"], "#2D6FAD": colors["accent"], "#2E2E2E": colors["border"],
        "#2E3D4F": colors["border"], "#344153": colors["border_strong"],
        "#3A3A3A": colors["border_strong"], "#3C6FB6": colors["accent"],
        "#3B4757": colors["border_strong"], "#4A4A4A": colors["border_strong"],
        "#465366": colors["border_strong"], "#5B6B80": colors["border_strong"],
        "#456A91": colors["border_strong"], "#5794D2": colors["accent"],
        "#5A7899": colors["border_strong"], "#5D9FE5": colors["accent"],
        "#66AEF5": colors["focus"], "#6C91B8": colors["border_strong"],
        "#6FA6DD": colors["accent_hover"], "#2A2A2A": colors["border"],
        "#202933": colors["border"], "#222222": colors["border"],
        "#F2F5F8": colors["text"], "#EDEDED": colors["text"], "#FFFFFF": colors["text"],
        "#CFCFCF": colors["text_muted"], "#B9B9B9": colors["text_muted"],
        "#B9C5D3": colors["text_muted"], "#B9C7D6": colors["text_muted"],
        "#BEBEBE": colors["text_muted"], "#BFD7F2": colors["text"],
        "#C4D0DE": colors["text"], "#DCE8F7": colors["text"], "#D7E8FA": colors["text"],
        "#B5BEC9": colors["text_muted"], "#777777": colors["text_muted"],
        "#71839A": colors["text_muted"], "#FFAAA6": colors["danger_text"],
        "#FFB4B4": colors["danger_text"], "#FFD7D5": colors["danger_text"],
        "#FF8A86": colors["danger_text"], "#F85149": colors["danger_hover"],
        "#DA3633": colors["danger"], "#238636": colors["success"],
        "#7EE787": colors["success_bright"], "#A7F3B4": colors["success_bright"],
        "#D29922": colors["warning"], "#F0C674": colors["warning"],
    }


def _matte_density_qss(colors: dict[str, str], *, compact: bool, text_scale: int = 100) -> str:
    """Final shared overrides: flat, dense, and independent of legacy selectors."""

    control_padding = "5px 7px" if compact else "7px 9px"
    button_padding = "5px 9px" if compact else "7px 11px"
    group_padding = "8px" if compact else "10px"
    normalized_text_scale = _normalized_text_scale(text_scale)
    single_line_min_height = max(22, round(22 * normalized_text_scale / 100))
    header_control_min_height = single_line_min_height
    return """
/* Matte density and semantic hierarchy overrides. */
QWidget { color: %(text)s; font-size: 14px; }
QToolTip { background: %(surface_raised)s; color: %(text)s; border: 1px solid %(border_strong)s; padding: 6px 8px; }
QGroupBox { background: %(surface)s; border: 1px solid %(border)s; border-radius: 6px; margin-top: 10px; padding: %(group_padding)s; }
QGroupBox::title { background: %(surface)s; color: %(text)s; font-weight: 600; left: 8px; padding: 0 4px; }
QLineEdit, QTextEdit, QPlainTextEdit, QListWidget, QListView, QTreeView, QTableView, QComboBox, QSpinBox, QDoubleSpinBox {
  background: %(surface)s; border: 1px solid %(border)s; border-radius: 5px; padding: %(control_padding)s; min-height: 22px;
  selection-background-color: %(surface_selected)s; selection-color: %(text)s;
}
QLineEdit { min-height: %(single_line_min_height)dpx; }
QLineEdit:hover, QTextEdit:hover, QPlainTextEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: %(border_strong)s; }
QPushButton, QToolButton { background: %(surface_raised)s; border: 1px solid %(border)s; border-radius: 5px; padding: %(button_padding)s; min-height: 22px; }
QPushButton:hover, QToolButton:hover { background: %(surface_selected)s; border-color: %(border_strong)s; }
QPushButton:checked, QToolButton:checked { background: %(surface_selected)s; border-color: %(accent)s; }
QPushButton[kind="primary"] { background: %(accent)s; border-color: %(accent)s; color: #FFFFFF; font-weight: 600; }
QPushButton[kind="primary"]:hover { background: %(accent_hover)s; border-color: %(accent_hover)s; }
QPushButton[kind="secondary"], QToolButton[kind="secondary"] { background: %(surface_raised)s; border-color: %(border_strong)s; color: %(text)s; font-weight: 500; }
QPushButton[kind="quiet"], QToolButton[kind="quiet"] { background: transparent; border-color: transparent; color: %(text_muted)s; }
QPushButton[kind="danger"] { background: %(danger_surface)s; border-color: %(danger)s; color: %(danger_text)s; font-weight: 600; }
QPushButton[kind="danger"]:hover { background: %(danger)s; color: #FFFFFF; }
QPushButton:disabled, QToolButton:disabled { background: %(surface_sunken)s; border-color: %(border)s; color: %(text_muted)s; }
QPushButton:focus, QToolButton:focus, QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus, QListWidget:focus, QListView:focus, QTreeView:focus, QTableView:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QCheckBox:focus, QTabBar::tab:focus { border: 2px solid %(focus)s; }
QToolButton[iconOnly="true"] { min-width: 22px; max-width: 22px; min-height: 22px; max-height: 22px; padding: 4px; }
QToolButton[helpIcon="true"], QPushButton[helpIcon="true"] { min-width: 20px; max-width: 20px; min-height: 20px; max-height: 20px; border-radius: 10px; padding: 0; color: %(text_muted)s; }
QTabWidget::pane { background: %(surface)s; border: 1px solid %(border)s; border-radius: 6px; }
QTabBar::tab, QTabBar#workspaceTabs::tab, QTabBar#workspaceModeTabs::tab, QTabBar#facesTaskNavigation::tab, QTabBar#faceResultViewTabs::tab { background: transparent; border: 1px solid transparent; border-radius: 5px; color: %(text_muted)s; padding: 5px 9px; margin: 1px 2px; font-weight: 500; }
QTabBar::tab:selected, QTabBar#workspaceTabs::tab:selected, QTabBar#workspaceModeTabs::tab:selected, QTabBar#facesTaskNavigation::tab:selected, QTabBar#faceResultViewTabs::tab:selected { background: %(surface_selected)s; border-color: %(accent)s; color: %(text)s; font-weight: 600; }
QTabBar::tab:hover, QTabBar#workspaceTabs::tab:hover, QTabBar#workspaceModeTabs::tab:hover, QTabBar#facesTaskNavigation::tab:hover, QTabBar#faceResultViewTabs::tab:hover { background: %(surface_raised)s; border-color: %(border_strong)s; }
QHeaderView::section { background: %(surface_raised)s; color: %(text)s; border: 0; border-bottom: 1px solid %(border)s; padding: 6px 8px; font-weight: 600; }
QScrollBar:vertical, QScrollBar:horizontal { background: transparent; border: 0; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal { background: %(border_strong)s; border-radius: 4px; min-height: 18px; min-width: 18px; }
QProgressBar { background: %(surface_sunken)s; border: 1px solid %(border)s; border-radius: 4px; }
QProgressBar::chunk { background: %(accent)s; border-radius: 3px; }
QProgressBar[state="finished"] { border-color: %(success)s; }
QProgressBar[state="failed"] { border-color: %(danger)s; }
QProgressBar[state="cancelled"], QProgressBar[state="cancelling"] { border-color: %(warning)s; }
QProgressBar[state="queued"] { border-style: dashed; border-color: %(border_strong)s; }
QWidget#applicationHeader { background: %(surface)s; border-color: %(border)s; border-radius: 6px; }
QWidget#applicationHeader QPushButton, QWidget#applicationHeader QToolButton { min-height: %(header_control_min_height)dpx; }
QWidget#workspaceCanvas { background: %(canvas)s; }
QWidget#sidebarSurface, QWidget#detailsSurface { background: %(surface_sunken)s; border-color: %(border)s; border-radius: 6px; }
QWidget#emptyStateCard { background: %(surface)s; border-color: %(border)s; border-radius: 6px; }
QWidget#workspaceFooter { background: %(canvas)s; border-top: 1px solid %(border)s; }
QWidget#workspaceFooter QLabel { color: %(text_muted)s; }
QWidget#workspaceFooter QLabel#footerFolderChip { color: %(text)s; background: %(surface_selected)s; border: 1px solid %(border_strong)s; }
QLabel[role="title"], QDialog QLabel[role="dialogTitle"] { font-size: 20px; font-weight: 600; color: %(text)s; }
QLabel[role="section"] { font-size: 16px; font-weight: 600; color: %(text)s; }
QLabel[role="helper"] { font-size: 12px; color: %(text_muted)s; }
QLabel[state="warning"] { color: %(warning)s; }
QLabel[state="error"] { color: %(danger_text)s; }
QLabel[state="success"] { color: %(success_bright)s; }
QMenu { background: %(surface_raised)s; color: %(text)s; border: 1px solid %(border_strong)s; }
QMenu::item { padding: 5px 22px 5px 9px; }
QMenu::item:selected { background: %(surface_selected)s; }
""" % {
        **colors,
        "control_padding": control_padding,
        "button_padding": button_padding,
        "group_padding": group_padding,
        "header_control_min_height": header_control_min_height,
        "single_line_min_height": single_line_min_height,
    }


def theme_revision() -> int:
    return _theme_revision


def current_theme() -> str:
    return "light" if COLORS.get("canvas") == LIGHT_COLORS["canvas"] else "dark"


def _resolved_system_theme(app: QApplication) -> str:
    try:
        scheme = app.styleHints().colorScheme()
        if scheme == Qt.ColorScheme.Light:
            return "light"
        if scheme == Qt.ColorScheme.Dark:
            return "dark"
    except (AttributeError, RuntimeError):
        pass
    return "dark"


def apply_app_theme(app: QApplication, theme: str) -> str:
    """Apply one resolved app theme without changing saved preferences."""

    global _theme_revision
    resolved = "light" if str(theme).lower() == "light" else "dark"
    compact = bool(app.property("clusterlens_compact_density"))
    text_scale = _normalized_text_scale(app.property("clusterlens_text_scale"))
    if app.property("clusterlens_resolved_theme") == resolved and bool(app.styleSheet()):
        return resolved
    COLORS.clear()
    COLORS.update(LIGHT_COLORS if resolved == "light" else DARK_COLORS)
    if not bool(app.property("clusterlens_fusion_style")):
        try:
            app.setStyle("Fusion")
            app.setProperty("clusterlens_fusion_style", True)
        except Exception:
            pass

    app.setPalette(_resolved_palette(COLORS))
    stylesheet_signature = (compact, text_scale)
    if app.property("clusterlens_stylesheet_signature") != stylesheet_signature or not app.styleSheet():
        app.setStyleSheet(stylesheet_for_theme(resolved, compact=compact, text_scale=text_scale))
        app.setProperty("clusterlens_stylesheet_signature", stylesheet_signature)
    app.setProperty("clusterlens_resolved_theme", resolved)
    _theme_revision += 1
    try:
        from ui.icons import refresh_themed_icons

        refresh_themed_icons(app)
    except (ImportError, RuntimeError):
        pass
    return resolved


def apply_app_density(app: QApplication, compact: bool) -> bool:
    """Apply the saved spacing mode without touching the resolved color theme."""

    normalized = bool(compact)
    if app.property("clusterlens_compact_density") == normalized:
        return False
    app.setProperty("clusterlens_compact_density", normalized)
    resolved = str(app.property("clusterlens_resolved_theme") or current_theme())
    text_scale = _normalized_text_scale(app.property("clusterlens_text_scale"))
    COLORS.clear()
    COLORS.update(LIGHT_COLORS if resolved == "light" else DARK_COLORS)
    app.setStyleSheet(stylesheet_for_theme(resolved, compact=normalized, text_scale=text_scale))
    app.setProperty("clusterlens_stylesheet_signature", (normalized, text_scale))
    return True


def apply_app_text_scale(app: QApplication, text_scale: int) -> bool:
    """Apply one supported text scale without changing theme or density."""

    normalized = _normalized_text_scale(text_scale)
    if _normalized_text_scale(app.property("clusterlens_text_scale")) == normalized and bool(app.styleSheet()):
        return False
    app.setProperty("clusterlens_text_scale", normalized)
    resolved = str(app.property("clusterlens_resolved_theme") or current_theme())
    compact = bool(app.property("clusterlens_compact_density"))
    app.setStyleSheet(stylesheet_for_theme(resolved, compact=compact, text_scale=normalized))
    app.setProperty("clusterlens_stylesheet_signature", (compact, normalized))
    return True


class ThemeManager(QObject):
    theme_changed = pyqtSignal(str)
    viewer_backdrop_changed = pyqtSignal(str)
    text_scale_changed = pyqtSignal(int)

    def __init__(self, app: QApplication) -> None:
        super().__init__(app)
        self._app = app
        self._store = None
        self.theme_preference = "system"
        self.resolved_theme = "dark"
        self.viewer_backdrop = "adaptive_neutral"
        self.text_scale = 100
        try:
            app.styleHints().colorSchemeChanged.connect(self._system_color_scheme_changed)
        except (AttributeError, RuntimeError):
            pass

    def configure(self, store: object | None) -> None:
        self._store = store
        theme = self._read("appearance/theme", "system")
        backdrop = self._read("appearance/viewer_backdrop", "adaptive_neutral")
        try:
            text_scale = int(self._read("appearance/text_scale", "100"))
        except ValueError:
            text_scale = 100
        self.set_text_scale(text_scale, persist=False)
        self.set_preferences(theme, backdrop, persist=False)

    def _read(self, key: str, default: str) -> str:
        if self._store is None:
            return default
        try:
            value = self._store.value(key, default)
        except (AttributeError, TypeError):
            return default
        return str(value or default)

    def set_preferences(self, theme: str, backdrop: str, *, persist: bool = False) -> None:
        normalized_theme = str(theme).lower()
        if normalized_theme not in APP_THEME_MODES:
            normalized_theme = "system"
        normalized_backdrop = str(backdrop).lower()
        if normalized_backdrop not in VIEWER_BACKDROP_MODES:
            normalized_backdrop = "adaptive_neutral"
        old_backdrop = self.viewer_backdrop
        self.theme_preference = normalized_theme
        self.viewer_backdrop = normalized_backdrop
        if persist and self._store is not None:
            self._store.setValue("appearance/theme", normalized_theme)
            self._store.setValue("appearance/viewer_backdrop", normalized_backdrop)
            self._store.sync()
        self._apply_resolved_theme()
        if old_backdrop != normalized_backdrop:
            self.viewer_backdrop_changed.emit(normalized_backdrop)

    def set_viewer_backdrop(self, backdrop: str, *, persist: bool = True) -> None:
        normalized = str(backdrop).lower()
        if normalized not in VIEWER_BACKDROP_MODES:
            normalized = "adaptive_neutral"
        if normalized == self.viewer_backdrop:
            return
        self.viewer_backdrop = normalized
        if persist and self._store is not None:
            self._store.setValue("appearance/viewer_backdrop", normalized)
            self._store.sync()
        self.viewer_backdrop_changed.emit(normalized)

    def set_text_scale(self, text_scale: int, *, persist: bool = True) -> None:
        normalized = _normalized_text_scale(text_scale)
        changed = normalized != self.text_scale
        self.text_scale = normalized
        if persist and self._store is not None:
            self._store.setValue("appearance/text_scale", normalized)
            self._store.sync()
        stylesheet_changed = apply_app_text_scale(self._app, normalized)
        if changed or stylesheet_changed:
            self.text_scale_changed.emit(normalized)

    def _apply_resolved_theme(self) -> None:
        resolved = _resolved_system_theme(self._app) if self.theme_preference == "system" else self.theme_preference
        changed = resolved != self.resolved_theme
        if changed or self._app.property("clusterlens_resolved_theme") != resolved:
            self.resolved_theme = apply_app_theme(self._app, resolved)
        if changed:
            self.theme_changed.emit(self.resolved_theme)

    def _system_color_scheme_changed(self, *_args) -> None:
        if self.theme_preference == "system":
            self._apply_resolved_theme()


def install_theme_manager(app: QApplication, store: object | None = None) -> ThemeManager:
    global _theme_manager
    if _theme_manager is None or _theme_manager._app is not app:
        _theme_manager = ThemeManager(app)
    _theme_manager.configure(store)
    return _theme_manager


def get_theme_manager() -> ThemeManager | None:
    return _theme_manager


def apply_ultra_dark(app: QApplication) -> None:
    """Compatibility entry point retained for older launchers and tests."""

    apply_app_theme(app, "dark")
