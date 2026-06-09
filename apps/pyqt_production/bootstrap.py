from __future__ import annotations

import sys
from pathlib import Path

from apps.pyqt_production.identity import LEGACY_PRODUCTION_APP_IDS, PRODUCTION_APP_ID
from apps.shared.runtime_support import RuntimeLayout, activate_runtime_root


APP_NAME = PRODUCTION_APP_ID


def bootstrap_runtime() -> tuple[RuntimeLayout, Path]:
    repo_root = Path(__file__).resolve().parents[2]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    layout = activate_runtime_root(APP_NAME, legacy_app_names=LEGACY_PRODUCTION_APP_IDS)
    return layout, repo_root
