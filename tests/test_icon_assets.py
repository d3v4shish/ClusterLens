from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import Mock, patch

from apps.pyqt_production.icon_assets import LINUX_DESKTOP_FILE_ID, configure_linux_desktop_entry


def test_source_launch_sets_desktop_id_without_installing() -> None:
    app = Mock()

    with (
        patch.object(sys, "platform", "linux"),
        patch("apps.pyqt_production.icon_assets.install_linux_desktop_entry") as install,
    ):
        result = configure_linux_desktop_entry(app, executable_path=Path(sys.executable))

    assert result is None
    app.setDesktopFileName.assert_called_once_with(LINUX_DESKTOP_FILE_ID)
    install.assert_not_called()


def test_explicit_desktop_install_uses_requested_executable() -> None:
    app = Mock()
    expected = Path("/tmp/io.clusterlens.ClusterLens.desktop")
    executable = Path("/opt/clusterlens/clusterlens")

    with (
        patch.object(sys, "platform", "linux"),
        patch("apps.pyqt_production.icon_assets.install_linux_desktop_entry", return_value=expected) as install,
    ):
        result = configure_linux_desktop_entry(
            app,
            executable_path=executable,
            install_desktop_entry=True,
        )

    assert result == expected
    install.assert_called_once_with(executable_path=executable)
