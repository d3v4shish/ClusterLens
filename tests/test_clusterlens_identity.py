from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from apps.pyqt_production.identity import LEGACY_PRODUCTION_APP_ID, PRODUCTION_APP_ID
from apps.shared.runtime_support import activate_runtime_root


def test_clusterlens_runtime_root_copies_missing_legacy_runtime_files(tmp_path: Path) -> None:
    legacy_root = tmp_path / LEGACY_PRODUCTION_APP_ID
    legacy_tag_db = legacy_root / "cache" / "image_tags.sqlite3"
    legacy_model = legacy_root / "model_assets" / "fast_preview" / "model.onnx"
    for path in (legacy_tag_db, legacy_model):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"legacy")

    with patch.dict(os.environ, {"XDG_DATA_HOME": str(tmp_path)}, clear=False):
        os.environ.pop("IMAGE_CLUSTERING_APP_DIR", None)
        layout = activate_runtime_root(PRODUCTION_APP_ID, legacy_app_names=(LEGACY_PRODUCTION_APP_ID,))

    assert layout.root == tmp_path / PRODUCTION_APP_ID
    assert (layout.cache_dir / "image_tags.sqlite3").read_bytes() == b"legacy"
    assert (layout.model_assets_dir / "fast_preview" / "model.onnx").read_bytes() == b"legacy"
    assert (layout.root / "runtime_identity_migration.json").exists()
