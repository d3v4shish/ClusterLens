from __future__ import annotations

import os
import sys
from pathlib import Path

from PyQt6.QtGui import QIcon

LINUX_DESKTOP_FILE_ID = "io.clusterlens.ClusterLens"


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


def configure_linux_desktop_entry(
    app,
    *,
    executable_path: str | Path | None = None,
    install_desktop_entry: bool | None = None,
) -> Path | None:
    """Expose a stable Linux desktop id and optionally create a user desktop entry.

    On Linux, especially Wayland, the shell usually resolves the taskbar icon
    from the app id / desktop-file id instead of the raw executable or Qt
    window icon. Packaged builds install their entry automatically. Source
    launches only set the desktop id and never mutate the user's applications
    directory unless an installer explicitly opts in.
    """

    if not sys.platform.startswith("linux"):
        return None
    try:
        app.setDesktopFileName(LINUX_DESKTOP_FILE_ID)
    except Exception:
        pass
    should_install = bool(getattr(sys, "frozen", False)) if install_desktop_entry is None else bool(install_desktop_entry)
    if not should_install:
        return None
    return install_linux_desktop_entry(executable_path=executable_path)


def install_linux_desktop_entry(*, executable_path: str | Path | None = None) -> Path | None:
    if not sys.platform.startswith("linux"):
        return None
    icon_path = production_icon_path()
    executable = Path(executable_path or sys.executable).resolve()
    if icon_path is None or not executable.exists():
        return None
    data_home = Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
    desktop_dir = data_home / "applications"
    desktop_path = desktop_dir / f"{LINUX_DESKTOP_FILE_ID}.desktop"
    desktop_dir.mkdir(parents=True, exist_ok=True)
    text = _desktop_entry_text(executable=executable, icon_path=icon_path.resolve())
    try:
        if desktop_path.exists() and desktop_path.read_text(encoding="utf-8") == text:
            return desktop_path
        desktop_path.write_text(text, encoding="utf-8")
        desktop_path.chmod(0o644)
    except OSError:
        return None
    return desktop_path


def _desktop_entry_text(*, executable: Path, icon_path: Path) -> str:
    return "\n".join(
        [
            "[Desktop Entry]",
            "Type=Application",
            "Name=ClusterLens",
            "Comment=Cluster, inspect, and manage image collections",
            f"Exec={_desktop_exec_value(executable)} %F",
            f"Icon={icon_path}",
            "Terminal=false",
            "Categories=Graphics;Photography;Utility;",
            "StartupNotify=true",
            "StartupWMClass=ClusterLens",
            "",
        ]
    )


def _desktop_exec_value(path: Path) -> str:
    text = str(path)
    if not any(ch.isspace() or ch in {'"', "\\", "'"} for ch in text):
        return text
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'
