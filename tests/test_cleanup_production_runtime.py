from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "cleanup_production_runtime.py"
SPEC = importlib.util.spec_from_file_location("cleanup_production_runtime", SCRIPT_PATH)
cleanup = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = cleanup
SPEC.loader.exec_module(cleanup)


def test_default_is_dry_run_and_preserves_files(tmp_path: Path) -> None:
    root = tmp_path / "ClusterLens"
    cache = root / "cache"
    embedding_index = cache / "embedding_indexes" / "vectors.npy"
    embedding_db = cache / "embeddings.sqlite3"
    logs = root / "logs" / "app.log"
    model_asset = root / "model_assets" / "model.onnx"
    for path in (embedding_index, embedding_db, logs, model_asset):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"content")

    assert cleanup.main(["--runtime-root", str(root)]) == 0

    assert embedding_index.exists()
    assert embedding_db.exists()
    assert logs.exists()
    assert model_asset.exists()


def test_yes_deletes_cache_targets_and_preserves_user_data(tmp_path: Path) -> None:
    root = tmp_path / "ClusterLens"
    cache = root / "cache"
    embedding_index = cache / "embedding_indexes" / "vectors.npy"
    embedding_db = cache / "embeddings.sqlite3"
    temp_file = cache / "tmp" / "request.json"
    partial_download = cache / "huggingface" / "model.partial"
    tag_db = root / "image_tags.sqlite3"
    logs = root / "logs" / "app.log"
    model_asset = root / "model_assets" / "model.onnx"
    for path in (embedding_index, embedding_db, temp_file, partial_download, tag_db, logs, model_asset):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"content")

    assert cleanup.main(["--runtime-root", str(root), "--yes"]) == 0

    assert not embedding_index.exists()
    assert not embedding_db.exists()
    assert not temp_file.exists()
    assert not partial_download.exists()
    assert tag_db.exists()
    assert logs.exists()
    assert model_asset.exists()


def test_optional_logs_scope_deletes_logs(tmp_path: Path) -> None:
    root = tmp_path / "ClusterLens"
    log_file = root / "logs" / "app.log"
    log_file.parent.mkdir(parents=True)
    log_file.write_bytes(b"log")

    assert cleanup.main(["--runtime-root", str(root), "--logs", "--yes"]) == 0

    assert not log_file.exists()


def test_all_user_data_requires_second_confirmation(tmp_path: Path) -> None:
    root = tmp_path / "ClusterLens"
    data = root / "image_tags.sqlite3"
    data.parent.mkdir(parents=True)
    data.write_bytes(b"data")

    assert cleanup.main(["--runtime-root", str(root), "--all-user-data", "--yes"]) == 2
    assert root.exists()

    assert (
        cleanup.main(
            [
                "--runtime-root",
                str(root),
                "--all-user-data",
                "--i-understand-this-deletes-user-data",
                "--yes",
            ]
        )
        == 0
    )
    assert not root.exists()


def test_custom_runtime_root_requires_force(tmp_path: Path) -> None:
    root = tmp_path / "custom-runtime"
    temp_file = root / "cache" / "tmp" / "request.json"
    temp_file.parent.mkdir(parents=True)
    temp_file.write_bytes(b"temp")

    assert cleanup.main(["--runtime-root", str(root), "--yes"]) == 2
    assert temp_file.exists()

    assert cleanup.main(["--runtime-root", str(root), "--force-runtime-root", "--yes"]) == 0
    assert not temp_file.exists()


def test_symlink_cache_file_is_not_followed_or_deleted(tmp_path: Path) -> None:
    root = tmp_path / "ClusterLens"
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    link = root / "cache" / "danger.tmp"
    link.parent.mkdir(parents=True)
    link.symlink_to(outside)

    assert cleanup.main(["--runtime-root", str(root), "--yes"]) == 0

    assert link.exists()
    assert outside.exists()
