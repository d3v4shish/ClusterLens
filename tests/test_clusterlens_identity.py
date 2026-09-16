from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

from apps.pyqt_production.identity import LEGACY_PRODUCTION_APP_ID, PRODUCTION_APP_ID
from apps.shared.runtime_support import activate_runtime_root, candidate_runtime_roots
from infra import settings as settings_mod


def test_clusterlens_runtime_root_copies_missing_legacy_runtime_files(tmp_path: Path) -> None:
    legacy_root = tmp_path / LEGACY_PRODUCTION_APP_ID
    legacy_tag_db = legacy_root / "cache" / "image_tags.sqlite3"
    legacy_model = legacy_root / "model_assets" / "fast_preview" / "model.onnx"
    for path in (legacy_tag_db, legacy_model):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"legacy")

    with patch.dict(
        os.environ,
        {
            "CLUSTERLENS_RUNTIME_ROOT": "",
            "IMAGE_CLUSTERING_APP_DIR": "",
            "XDG_DATA_HOME": str(tmp_path),
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
        },
        clear=False,
    ):
        layout = activate_runtime_root(PRODUCTION_APP_ID, legacy_app_names=(LEGACY_PRODUCTION_APP_ID,))

    assert layout.root == tmp_path / PRODUCTION_APP_ID
    assert (layout.cache_dir / "image_tags.sqlite3").read_bytes() == b"legacy"
    assert (layout.model_assets_dir / "fast_preview" / "model.onnx").read_bytes() == b"legacy"
    assert (layout.root / "runtime_identity_migration.json").exists()


def test_clusterlens_runtime_override_is_shared_by_local_and_packaged_processes(tmp_path: Path) -> None:
    canonical_root = tmp_path / "canonical-runtime"
    legacy_root = tmp_path / "legacy-runtime"
    with patch.dict(
        os.environ,
        {
            "CLUSTERLENS_RUNTIME_ROOT": str(canonical_root),
            "IMAGE_CLUSTERING_APP_DIR": str(legacy_root),
        },
        clear=False,
    ):
        settings_mod._RUNTIME_BASE_DIR = None
        layout = activate_runtime_root(PRODUCTION_APP_ID)
        assert layout.root == canonical_root
        assert settings_mod.get_runtime_base_dir() == canonical_root
        assert os.environ["CLUSTERLENS_RUNTIME_ROOT"] == str(canonical_root)
        assert os.environ["IMAGE_CLUSTERING_APP_DIR"] == str(canonical_root)
    settings_mod._RUNTIME_BASE_DIR = None


def test_configured_data_home_is_used_when_no_process_override_exists(tmp_path: Path) -> None:
    configured_root = tmp_path / "relocated-runtime"
    config_path = tmp_path / "config" / PRODUCTION_APP_ID / "data_home.json"
    config_path.parent.mkdir(parents=True)
    config_path.write_text('{"data_home": "' + str(configured_root) + '"}', encoding="utf-8")

    with patch.dict(
        os.environ,
        {
            "CLUSTERLENS_RUNTIME_ROOT": "",
            "IMAGE_CLUSTERING_APP_DIR": "",
            "XDG_CONFIG_HOME": str(tmp_path / "config"),
            "XDG_DATA_HOME": str(tmp_path / "data"),
        },
        clear=False,
    ):
        assert candidate_runtime_roots(PRODUCTION_APP_ID)[0] == configured_root
