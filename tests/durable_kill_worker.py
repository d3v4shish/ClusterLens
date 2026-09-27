#!/usr/bin/env python3
"""Disposable child used by durable recovery tests.

Each scenario terminates this process with SIGKILL at an exact production
checkpoint.  The parent test owns fixture creation and recovery assertions, so
this helper never touches a user profile or source tree.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.services.cache_maintenance import CacheMaintenanceService
from app.services.data_home import DataHomeManager
from app.services.embedding_index import EmbeddingIndexService
from app.services.face_model_installer import FaceModelInstaller
from app.services.face_search import DetectedFace, FaceIndexService
from app.services.face_storage_recovery import FaceStorageRemovalService
from app.services.gallery_actions import GalleryActionService
from app.services.image_tags import ImageTagService
from app.services.library_catalog import CatalogAsset, LibraryCatalogService
from app.services.photo_metadata import PhotoEditDraft, PhotoEditService
from app.services.result_cache import ResultCacheService
from app.services.saved_searches import SavedSearchService
from infra import atomic_io


def _kill() -> None:
    os.kill(os.getpid(), signal.SIGKILL)
    raise AssertionError("SIGKILL unexpectedly returned")


def _kill_atomic_target(target: Path, boundary: str, wanted: Path) -> None:
    original = atomic_io._atomic_write_checkpoint

    def checkpoint(candidate: Path, phase: str) -> None:
        if Path(candidate) == wanted and phase == boundary:
            _kill()
        original(candidate, phase)

    atomic_io._atomic_write_checkpoint = checkpoint


def _photo_sidecar(root: Path, boundary: str) -> None:
    image = root / "photo.jpg"
    sidecar = root / "photo.jpg.clusterlens.json"
    _kill_atomic_target(sidecar, boundary, sidecar)
    PhotoEditService().save_draft(image, PhotoEditDraft(title="replacement"))


def _saved_search(root: Path, boundary: str) -> None:
    path = root / "saved_searches.json"
    search_id = (root / "search_id.txt").read_text(encoding="utf-8").strip()
    _kill_atomic_target(path, boundary, path)
    SavedSearchService(path).rename_search(search_id, "Replacement")


def _result_cache(root: Path, boundary: str) -> None:
    service = ResultCacheService()
    service.cache_dir = root
    path = root / "fault-matrix.json"
    _kill_atomic_target(path, boundary, path)
    service.save("fault-matrix", {0: ["new.jpg"]}, {"generation": "new"})


def _embedding_index(root: Path, boundary: str) -> None:
    image = root / "photo.jpg"
    service = EmbeddingIndexService()
    service.index_dir = root / "indexes"
    snapshot = service.build_snapshot_key([str(image)])
    manifest = service._prefix(snapshot, "clip", "").with_suffix(".manifest.json")
    _kill_atomic_target(manifest, boundary, manifest)

    def checkpoint(name: str, _manifest: Path) -> None:
        if name == boundary:
            _kill()

    service._index_generation_checkpoint = checkpoint  # type: ignore[method-assign]
    service.save_index(
        snapshot,
        "clip",
        [(str(image), np.asarray([0.8, 0.9], dtype=np.float32))],
    )


def _image_tag(root: Path, boundary: str) -> None:
    service = ImageTagService(db_path=root / "tags.sqlite3")

    def checkpoint(name: str) -> None:
        if name == boundary:
            _kill()

    service._tag_write_checkpoint = checkpoint  # type: ignore[method-assign]
    service.rename_tag("Old", "New")


def _gallery_move(root: Path, boundary: str) -> None:
    config_path = root / "gallery_paths.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    service = GalleryActionService(
        audit_log_path=Path(config.get("audit_log_path", root / "logs" / "operations.jsonl")),
        journal_path=Path(config.get("journal_path", root / "logs" / "operations.sqlite3")),
        temp_dir=Path(config.get("temp_dir", root / "temporary")),
    )

    def checkpoint(name: str, _source: Path, _target: Path) -> None:
        if name == boundary:
            _kill()

    service._file_mutation_checkpoint = checkpoint  # type: ignore[method-assign]
    service.move_to_directory(
        [str(config.get("source_path", root / "source.jpg"))],
        str(config.get("destination", root / "destination")),
    )


def _face_model(root: Path, boundary: str) -> None:
    installer = FaceModelInstaller(settings=SimpleNamespace(cache_dir=root))
    target = installer.bundle_dir("yunet_2026may")
    staging = target.parent / ".yunet_2026may.kill.installing"

    def checkpoint(name: str, *_args: object) -> None:
        if name == boundary:
            _kill()

    installer._promotion_checkpoint = checkpoint  # type: ignore[method-assign]
    installer._promote_bundle_directory(staging, target)


class _FixedFaceDetector:
    def detect_faces(self, image_path: str) -> list[DetectedFace]:
        return [
            DetectedFace(
                image_path=image_path,
                bbox=(0, 0, 36, 36),
                confidence=0.99,
                crop=Image.new("RGB", (36, 36), (255, 255, 255)),
            )
        ]


class _FixedFaceEmbedder:
    def embed_faces(self, face_crops: list[Image.Image]) -> np.ndarray:
        return np.asarray([[1.0, 0.0, 0.0] for _ in face_crops], dtype=np.float32)


def _face_index(root: Path, boundary: str) -> None:
    service = FaceIndexService(
        detection_service=_FixedFaceDetector(),
        embedding_service=_FixedFaceEmbedder(),
        db_path=root / "faces.sqlite3",
    )

    def checkpoint(name: str) -> None:
        if name == boundary:
            _kill()

    service._face_index_write_checkpoint = checkpoint  # type: ignore[method-assign]
    service.index_paths([str(root / "face.jpg")])


def _library_catalog(root: Path, boundary: str) -> None:
    service = LibraryCatalogService(db_path=root / "catalog.sqlite3")
    root_id = (root / "catalog_root_id.txt").read_text(encoding="utf-8").strip()
    image = root / "catalog-photo.jpg"
    asset = CatalogAsset(str(image), root_id, "", image.stem, "", "", 64, 48, 1, ".jpg")

    def checkpoint(name: str) -> None:
        if name == boundary:
            _kill()

    service._catalog_write_checkpoint = checkpoint  # type: ignore[method-assign]
    service._upsert_assets(((asset, 1),))


def _embedded_metadata(root: Path, boundary: str) -> None:
    service = GalleryActionService(temp_dir=root / "temporary")

    def checkpoint(name: str, *_args: object) -> None:
        if name == boundary:
            _kill()

    service._metadata_rewrite_checkpoint = checkpoint  # type: ignore[method-assign]
    service.write_exif_comments([str(root / "metadata.jpg")], "replacement")


def _rebuildable_cache(root: Path, boundary: str) -> None:
    cache_root = root / "cache"
    settings = SimpleNamespace(
        cache_dir=cache_root,
        embedding_cache_db=cache_root / "embeddings.sqlite3",
        thumbnail_cache_dir=cache_root / "thumbnails",
    )
    service = CacheMaintenanceService(settings=settings)

    def checkpoint(name: str, *_args: object) -> None:
        if name == boundary:
            _kill()

    service._cache_directory_clear_checkpoint = checkpoint  # type: ignore[method-assign]
    service._clear_rebuildable_directory(
        "cluster_results/",
        cache_root / "cluster_results",
    )


def _sqlite_cache(root: Path, boundary: str) -> None:
    service = CacheMaintenanceService(settings=SimpleNamespace(cache_dir=root, base_dir=root))

    def checkpoint(name: str, *_args: object) -> None:
        if name == boundary:
            _kill()

    service._sqlite_cache_clear_checkpoint = checkpoint  # type: ignore[method-assign]
    service.clear_sqlite_cache_bundle(root / "embeddings.sqlite3")


def _data_home_backup(root: Path, boundary: str) -> None:
    manager = DataHomeManager("ClusterLens", root / "runtime", config_path=root / "config.json")

    def checkpoint(name: str, *_args: object) -> None:
        if name == boundary:
            _kill()

    manager._backup_checkpoint = checkpoint  # type: ignore[method-assign]
    manager.create_backup(root / "backups")


def _data_home_relocation(root: Path, boundary: str) -> None:
    manager = DataHomeManager("ClusterLens", root / "runtime", config_path=root / "control" / "data_home.json")

    def checkpoint(name: str, *_args: object) -> None:
        if name == boundary:
            _kill()

    manager._relocation_checkpoint = checkpoint  # type: ignore[method-assign]
    manager.relocate(root / "new-runtime")


def _face_storage(root: Path, boundary: str) -> None:
    service = FaceStorageRemovalService(root)

    def checkpoint(name: str) -> None:
        if name == boundary:
            _kill()

    service.remove(root / "face_search_human.db", checkpoint=checkpoint)


SCENARIOS = {
    "data_home_backup": _data_home_backup,
    "data_home_relocation": _data_home_relocation,
    "embedded_metadata": _embedded_metadata,
    "embedding_index": _embedding_index,
    "face_index": _face_index,
    "face_model": _face_model,
    "face_storage": _face_storage,
    "gallery_move": _gallery_move,
    "image_tag": _image_tag,
    "library_catalog": _library_catalog,
    "photo_sidecar": _photo_sidecar,
    "rebuildable_cache": _rebuildable_cache,
    "result_cache": _result_cache,
    "saved_search": _saved_search,
    "sqlite_cache": _sqlite_cache,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario", choices=sorted(SCENARIOS))
    parser.add_argument("boundary")
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    marker = root / "child_started.json"
    marker.write_text(
        json.dumps({"pid": os.getpid(), "scenario": args.scenario, "boundary": args.boundary}),
        encoding="utf-8",
    )
    SCENARIOS[args.scenario](root, args.boundary)
    return 97


if __name__ == "__main__":
    raise SystemExit(main())
