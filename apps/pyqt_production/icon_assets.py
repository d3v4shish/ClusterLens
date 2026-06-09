from __future__ import annotations

import sys
from pathlib import Path

from PyQt6.QtGui import QIcon


def production_icon_path() -> Path | None:
    module_dir = Path(__file__).resolve().parent
    candidates = [module_dir / "assets" / "app_icon.png"]
    meipass = getattr(sys, "_MEIPASS", "")
    if meipass:
        candidates.insert(0, Path(meipass) / "apps" / "pyqt_production" / "assets" / "app_icon.png")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def production_app_icon() -> QIcon:
    path = production_icon_path()
    if path is None:
        return QIcon()
    return QIcon(str(path))
