from __future__ import annotations

from PyQt6.QtCore import QByteArray, QSize
from PyQt6.QtGui import QIcon, QPixmap


# Palette-aware SVG sources. They are rendered at the requested device scale by
# Qt, so the same source remains sharp across supported DPI settings.
_PATHS = {
    "workspace": '<rect x="4" y="4" width="7" height="7" rx="1"/><rect x="13" y="4" width="7" height="7" rx="1"/><rect x="4" y="13" width="7" height="7" rx="1"/><rect x="13" y="13" width="7" height="7" rx="1"/>',
    "folder": '<path d="M3 6h7l2 2h9v11H3z"/><path d="M3 6V4h7l2 2" fill="none"/>',
    "search": '<circle cx="10" cy="10" r="6" fill="none"/><path d="m14.5 14.5 6 6" fill="none"/>',
    "scan": '<path d="M4 9V4h5M15 4h5v5M20 15v5h-5M9 20H4v-5" fill="none"/><circle cx="12" cy="12" r="4" fill="none"/>',
    "settings": '<circle cx="12" cy="12" r="3" fill="none"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3M5 5l2 2M17 17l2 2M19 5l-2 2M7 17l-2 2" fill="none"/>',
    "jobs": '<path d="M5 4h14v16H5z" fill="none"/><path d="M8 9h8M8 13h8M8 17h5" fill="none"/>',
    "health": '<path d="M3 12h4l2-5 4 10 2-5h6" fill="none"/>',
    "recovery": '<path d="M6 8H2V4" fill="none"/><path d="M3 8a9 9 0 1 1 1 10" fill="none"/>',
    "reveal": '<path d="M2 12s4-6 10-6 10 6 10 6-4 6-10 6S2 12 2 12z" fill="none"/><circle cx="12" cy="12" r="3" fill="none"/>',
    "retry": '<path d="M6 8H2V4M3 8a9 9 0 0 1 16-2" fill="none"/><path d="M18 16h4v4m-1-4a9 9 0 0 1-16 2" fill="none"/>',
    "restore": '<path d="M4 4h16v16H4z" fill="none"/><path d="M8 11h8M8 11l3-3M8 11l3 3" fill="none"/>',
    "warning": '<path d="M12 3 22 21H2z" fill="none"/><path d="M12 9v5M12 18h.01" fill="none"/>',
    "delete": '<path d="M5 7h14M9 7V4h6v3M7 7l1 14h8l1-14" fill="none"/>',
}


def themed_icon(name: str, *, color: str = "#F2F5F8", size: int = 24) -> QIcon:
    paths = _PATHS.get(str(name), _PATHS["workspace"])
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" '
        f'width="{int(size)}" height="{int(size)}" fill="{color}" stroke="{color}" '
        f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{paths}</svg>'
    )
    pixmap = QPixmap()
    pixmap.loadFromData(QByteArray(svg.encode("utf-8")), "SVG")
    return QIcon(pixmap)


def apply_icon(widget, name: str, *, color: str = "#F2F5F8", size: int = 18) -> None:
    widget.setIcon(themed_icon(name, color=color, size=size))
    widget.setIconSize(QSize(size, size))
