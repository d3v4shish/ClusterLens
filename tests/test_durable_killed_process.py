from __future__ import annotations

import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from app.services.cache_maintenance import CacheMaintenanceService
from app.services.data_home import DataHomeManager
from app.services.embedding_index import EmbeddingIndexService
from app.services.face_model_installer import FaceModelInstaller
from app.services.face_search import FaceIndexService
from app.services.face_storage_recovery import FaceStorageRemovalService
from app.services.gallery_actions import GalleryActionService
from app.services.image_tags import ImageTagService
from app.services.library_catalog import CatalogQuery, LibraryCatalogService
from app.services.photo_metadata import PhotoEditDraft, PhotoEditService
from app.services.result_cache import ResultCacheService
from app.services.saved_searches import SavedSearchService


REPO_ROOT = Path(__file__).resolve().parents[1]
WORKER = REPO_ROOT / "tests" / "durable_kill_worker.py"
REQUIRES_SIGKILL = pytest.mark.skipif(
    os.name != "posix" or not hasattr(signal, "SIGKILL"),
    reason="actual SIGKILL recovery evidence requires a POSIX host",
)


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_data_home(root: Path) -> None:
    (root / "cache" / "thumbnails").mkdir(parents=True)
    (root / "cache" / "thumbnails" / "one.webp").write_bytes(b"preview")
    (root / "logs").mkdir()
    (root / "logs" / "app.log").write_text("line\n", encoding="utf-8")
    (root / "support").mkdir()
    (root / "support" / "recovery.json").write_text("{}\n", encoding="utf-8")


def _run_killed_child(root: Path, scenario: str, boundary: str) -> int:
    environment = dict(os.environ)
    source_root = str(REPO_ROOT / "src")
    python_paths = source_root + os.pathsep + str(REPO_ROOT)
    environment["PYTHONPATH"] = python_paths + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(
        [sys.executable, str(WORKER), scenario, boundary, str(root)],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert completed.returncode == -signal.SIGKILL, completed.stderr
    marker = json.loads((root / "child_started.json").read_text(encoding="utf-8"))
    assert marker["scenario"] == scenario
    assert marker["boundary"] == boundary
    assert int(marker["pid"]) > 0
    return int(completed.returncode)


def _run_completed_child(root: Path, scenario: str) -> None:
    environment = dict(os.environ)
    source_root = str(REPO_ROOT / "src")
    python_paths = source_root + os.pathsep + str(REPO_ROOT)
    environment["PYTHONPATH"] = python_paths + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(
        [sys.executable, str(WORKER), scenario, "no_checkpoint", str(root)],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert completed.returncode == 97, completed.stderr


def _record_evidence(
    evidence_id: str,
    *,
    operation: str,
    boundary: str,
    child_exit_code: int,
    pre_state: object,
    crash_state: object,
    recovered_state: object,
    recovery_attempts: int = 2,
) -> None:
    path_text = os.environ.get("CLUSTERLENS_DURABLE_KILL_EVIDENCE", "").strip()
    if not path_text:
        return
    record = {
        "evidence_id": evidence_id,
        "operation": operation,
        "boundary": boundary,
        "fault": "actual_sigkill",
        "signal": "SIGKILL",
        "child_exit_code": child_exit_code,
        "pre_state_sha256": _digest(pre_state),
        "crash_state_sha256": _digest(crash_state),
        "recovered_state_sha256": _digest(recovered_state),
        "recovery_attempts": recovery_attempts,
        "result": "PASS",
    }
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


@REQUIRES_SIGKILL
@pytest.mark.parametrize(
    ("boundary", "expected_title"),
    (("before_replace", "Original"), ("after_replace", "replacement")),
)
def test_photo_sidecar_actual_sigkill_has_exact_atomic_state(
    tmp_path: Path,
    boundary: str,
    expected_title: str,
) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"immutable-source-photo")
    service = PhotoEditService()
    service.save_draft(image, PhotoEditDraft(title="Original"))
    sidecar = tmp_path / "photo.jpg.clusterlens.json"
    pre = {"source": _file_sha(image), "sidecar": _file_sha(sidecar), "title": "Original"}

    exit_code = _run_killed_child(tmp_path, "photo_sidecar", boundary)
    crash = {
        "source": _file_sha(image),
        "sidecar": _file_sha(sidecar),
        "title": service.load_draft(image).title,
    }
    assert crash["source"] == pre["source"]
    assert crash["title"] == expected_title

    service.save_draft(image, PhotoEditDraft(title="replacement"))
    first = service.load_draft(image).title
    service.save_draft(image, PhotoEditDraft(title="replacement"))
    recovered = {
        "source": _file_sha(image),
        "sidecar": _file_sha(sidecar),
        "title": service.load_draft(image).title,
    }
    assert first == recovered["title"] == "replacement"
    assert recovered["source"] == pre["source"]
    _record_evidence(
        f"photo_sidecar_save:{boundary}:actual_sigkill",
        operation="photo_sidecar_save",
        boundary=boundary,
        child_exit_code=exit_code,
        pre_state=pre,
        crash_state=crash,
        recovered_state=recovered,
    )


@REQUIRES_SIGKILL
@pytest.mark.parametrize(
    ("boundary", "expected_name"),
    (("before_replace", "Original"), ("after_replace", "Replacement")),
)
def test_saved_search_actual_sigkill_has_exact_atomic_state(
    tmp_path: Path,
    boundary: str,
    expected_name: str,
) -> None:
    path = tmp_path / "saved_searches.json"
    service = SavedSearchService(path)
    record = service.save_search("Original", "unknown_people", {"mode": "unknown"})
    (tmp_path / "search_id.txt").write_text(record.search_id, encoding="utf-8")
    pre = {"file": _file_sha(path), "name": "Original"}

    exit_code = _run_killed_child(tmp_path, "saved_search", boundary)
    crash_name = SavedSearchService(path).get_search(record.search_id).name
    crash = {"file": _file_sha(path), "name": crash_name}
    assert crash_name == expected_name

    SavedSearchService(path).rename_search(record.search_id, "Replacement")
    first = SavedSearchService(path).get_search(record.search_id).name
    SavedSearchService(path).rename_search(record.search_id, "Replacement")
    recovered = {"file": _file_sha(path), "name": SavedSearchService(path).get_search(record.search_id).name}
    assert first == recovered["name"] == "Replacement"
    _record_evidence(
        f"saved_search_mutation:{boundary}:actual_sigkill",
        operation="saved_search_mutation",
        boundary=boundary,
        child_exit_code=exit_code,
        pre_state=pre,
        crash_state=crash,
        recovered_state=recovered,
    )


@REQUIRES_SIGKILL
@pytest.mark.parametrize(
    ("boundary", "expected_generation"),
    (("before_replace", "old"), ("after_replace", "new")),
)
def test_result_cache_actual_sigkill_has_exact_atomic_state(
    tmp_path: Path,
    boundary: str,
    expected_generation: str,
) -> None:
    service = ResultCacheService()
    service.cache_dir = tmp_path
    service.save("fault-matrix", {0: ["old.jpg"]}, {"generation": "old"})
    path = tmp_path / "fault-matrix.json"
    pre = {"file": _file_sha(path), "generation": "old"}

    exit_code = _run_killed_child(tmp_path, "result_cache", boundary)
    loaded = service.load("fault-matrix")
    assert loaded is not None
    crash = {"file": _file_sha(path), "generation": loaded[1]["generation"]}
    assert crash["generation"] == expected_generation

    service.save("fault-matrix", {0: ["new.jpg"]}, {"generation": "new"})
    first = service.load("fault-matrix")
    service.save("fault-matrix", {0: ["new.jpg"]}, {"generation": "new"})
    second = service.load("fault-matrix")
    assert first == second == ({0: ["new.jpg"]}, {"generation": "new"})
    recovered = {"file": _file_sha(path), "generation": second[1]["generation"]}
    _record_evidence(
        f"clustering_result_cache:{boundary}:actual_sigkill",
        operation="clustering_result_cache",
        boundary=boundary,
        child_exit_code=exit_code,
        pre_state=pre,
        crash_state=crash,
        recovered_state=recovered,
    )


@REQUIRES_SIGKILL
@pytest.mark.parametrize(
    ("boundary", "expected_tag"),
    (("before_commit", "Old"), ("after_commit", "New")),
)
def test_image_tag_actual_sigkill_has_exact_transaction_state(
    tmp_path: Path,
    boundary: str,
    expected_tag: str,
) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"photo")
    database = tmp_path / "tags.sqlite3"
    service = ImageTagService(db_path=database)
    service._replace_rows_for_path(str(image), [("old", "Old", "user")])

    def state() -> dict[str, object]:
        with sqlite3.connect(database) as connection:
            rows = connection.execute("SELECT tag_norm, display_tag FROM image_tags ORDER BY tag_norm").fetchall()
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        return {"rows": rows, "integrity": integrity, "source": _file_sha(image)}

    pre = state()
    exit_code = _run_killed_child(tmp_path, "image_tag", boundary)
    crash = state()
    assert crash["integrity"] == "ok"
    assert crash["rows"] == [(expected_tag.casefold(), expected_tag)]

    restarted = ImageTagService(db_path=database)
    restarted.rename_tag("Old", "New")
    first = state()
    restarted.rename_tag("Old", "New")
    recovered = state()
    assert first == recovered
    assert recovered["rows"] == [("new", "New")]
    _record_evidence(
        f"image_tag_mutation:{boundary}:actual_sigkill",
        operation="image_tag_mutation",
        boundary=boundary,
        child_exit_code=exit_code,
        pre_state=pre,
        crash_state=crash,
        recovered_state=recovered,
    )


@REQUIRES_SIGKILL
@pytest.mark.parametrize(
    ("boundary", "committed"),
    (
        ("operation_intent_committed", False),
        ("before_filesystem_mutation", False),
        ("after_filesystem_mutation", True),
        ("file_terminal_recorded", True),
    ),
)
def test_gallery_move_actual_sigkill_reconciles_twice(
    tmp_path: Path,
    boundary: str,
    committed: bool,
) -> None:
    source = tmp_path / "source.jpg"
    source.write_bytes(b"source")
    destination = tmp_path / "destination"
    service = GalleryActionService(
        audit_log_path=tmp_path / "logs" / "operations.jsonl",
        journal_path=tmp_path / "logs" / "operations.sqlite3",
        temp_dir=tmp_path / "temporary",
    )

    def state() -> dict[str, object]:
        targets = sorted(destination.glob("*")) if destination.is_dir() else []
        return {
            "source_exists": source.is_file(),
            "source_sha256": _file_sha(source) if source.is_file() else "",
            "targets": [(path.name, _file_sha(path)) for path in targets if path.is_file()],
        }

    pre = state()
    exit_code = _run_killed_child(tmp_path, "gallery_move", boundary)
    crash = state()
    assert crash["source_exists"] is (not committed)
    assert bool(crash["targets"]) is committed

    restarted = GalleryActionService(
        audit_log_path=tmp_path / "logs" / "operations.jsonl",
        journal_path=tmp_path / "logs" / "operations.sqlite3",
        temp_dir=tmp_path / "temporary",
    )
    first = restarted.recover_incomplete_operations()
    after_first = state()
    second = restarted.recover_incomplete_operations()
    recovered = state()
    assert after_first == recovered
    assert first["recovered_operations"] or boundary == "file_terminal_recorded"
    assert not second["recovered_operations"]
    assert not first["failures"] and not second["failures"]
    _record_evidence(
        f"gallery_file_mutations:{boundary}:actual_sigkill",
        operation="gallery_file_mutations",
        boundary=boundary,
        child_exit_code=exit_code,
        pre_state=pre,
        crash_state=crash,
        recovered_state=recovered,
    )


@REQUIRES_SIGKILL
@pytest.mark.parametrize("boundary", ("previous_staged", "target_published", "previous_discarded"))
def test_face_model_actual_sigkill_recovers_verified_bundle_twice(tmp_path: Path, boundary: str) -> None:
    installer = FaceModelInstaller(settings=SimpleNamespace(cache_dir=tmp_path))
    target = installer.bundle_dir("yunet_2026may")
    target.mkdir(parents=True)
    installer._copy_catalog_metadata("yunet_2026may", target)
    old_payload = target / "detector.onnx"
    old_payload.write_bytes(b"old-verified-model")
    installer._write_install_record(old_payload, bundle_id="yunet_2026may")
    staging = target.parent / ".yunet_2026may.kill.installing"
    staging.mkdir()
    installer._copy_catalog_metadata("yunet_2026may", staging)
    new_payload = staging / "detector.onnx"
    new_payload.write_bytes(b"new-verified-model")
    installer._write_install_record(new_payload, bundle_id="yunet_2026may")

    def state() -> dict[str, object]:
        previous = sorted(target.parent.glob(".yunet_2026may.*.previous"))
        return {
            "target": _file_sha(target / "detector.onnx") if (target / "detector.onnx").is_file() else "",
            "staging": _file_sha(staging / "detector.onnx") if (staging / "detector.onnx").is_file() else "",
            "previous": [_file_sha(path / "detector.onnx") for path in previous],
        }

    pre = state()
    exit_code = _run_killed_child(tmp_path, "face_model", boundary)
    crash = state()
    assert crash != pre

    restarted = FaceModelInstaller(settings=SimpleNamespace(cache_dir=tmp_path))
    first = restarted.recover_managed_models()
    after_first = state()
    second = restarted.recover_managed_models()
    recovered = state()
    assert not first.failures and not second.failures
    assert after_first == recovered
    assert (target / "detector.onnx").read_bytes() == b"new-verified-model"
    assert not staging.exists()
    assert not list(target.parent.glob(".yunet_2026may.*.previous"))
    _record_evidence(
        f"face_model_promotion:{boundary}:actual_sigkill",
        operation="face_model_promotion",
        boundary=boundary,
        child_exit_code=exit_code,
        pre_state=pre,
        crash_state=crash,
        recovered_state=recovered,
    )


@REQUIRES_SIGKILL
@pytest.mark.parametrize("boundary", ("manifest_removed", "before_replace", "after_replace"))
def test_embedding_index_actual_sigkill_rebuilds_one_ready_generation(tmp_path: Path, boundary: str) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"fixture")
    service = EmbeddingIndexService()
    service.index_dir = tmp_path / "indexes"
    service.index_dir.mkdir()
    snapshot = service.build_snapshot_key([str(image)])
    old = [(str(image), np.asarray([0.1, 0.2], dtype=np.float32))]
    new = [(str(image), np.asarray([0.8, 0.9], dtype=np.float32))]
    paths = service.save_index(snapshot, "clip", old)
    manifest = Path(paths["manifest_path"])

    def state() -> dict[str, object]:
        ready = service._index_is_valid(
            Path(paths["vector_path"]),
            Path(paths["path_map_path"]),
            Path(paths["faiss_index_path"]) if paths["faiss_index_path"] else None,
            manifest,
            expected_paths=[str(image)],
            expected_dimension=2,
        )
        vectors = np.load(paths["vector_path"], allow_pickle=False).tolist()
        return {"ready": ready, "vectors": vectors, "manifest": _file_sha(manifest) if manifest.is_file() else ""}

    pre = state()
    exit_code = _run_killed_child(tmp_path, "embedding_index", boundary)
    crash = state()
    assert crash["ready"] is (boundary == "after_replace")

    repaired, reused = service.ensure_index(snapshot, "clip", new)
    first = state()
    repaired_again, reused_again = service.ensure_index(snapshot, "clip", new)
    recovered = state()
    assert Path(repaired["manifest_path"]).is_file()
    assert repaired_again == repaired
    assert reused is (boundary == "after_replace")
    assert reused_again
    assert first == recovered
    assert recovered["ready"] is True
    assert recovered["vectors"] == [[pytest.approx(0.8), pytest.approx(0.9)]]
    _record_evidence(
        f"embedding_index_generation:{boundary}:actual_sigkill",
        operation="embedding_index_generation",
        boundary=boundary,
        child_exit_code=exit_code,
        pre_state=pre,
        crash_state=crash,
        recovered_state=recovered,
    )


@REQUIRES_SIGKILL
@pytest.mark.parametrize(
    ("boundary", "expected_faces"),
    (("before_commit", 0), ("after_commit", 1)),
)
def test_face_index_actual_sigkill_has_exact_transaction_state(
    tmp_path: Path,
    boundary: str,
    expected_faces: int,
) -> None:
    image = tmp_path / "face.jpg"
    Image.new("RGB", (48, 48), (20, 30, 40)).save(image)
    database = tmp_path / "faces.sqlite3"

    def state() -> dict[str, object]:
        service = FaceIndexService(db_path=database)
        with sqlite3.connect(database) as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        return {
            "faces": service.count_indexed_faces(include_tiny_faces=True),
            "integrity": integrity,
            "source": _file_sha(image),
        }

    pre = {"faces": 0, "source": _file_sha(image)}
    exit_code = _run_killed_child(tmp_path, "face_index", boundary)
    crash = state()
    assert crash["faces"] == expected_faces
    assert crash["integrity"] == "ok"
    assert crash["source"] == pre["source"]

    _run_completed_child(tmp_path, "face_index")
    first = state()
    _run_completed_child(tmp_path, "face_index")
    recovered = state()
    assert first == recovered
    assert recovered["faces"] == 1
    _record_evidence(
        f"face_index_batch:{boundary}:actual_sigkill",
        operation="face_index_batch",
        boundary=boundary,
        child_exit_code=exit_code,
        pre_state=pre,
        crash_state=crash,
        recovered_state=recovered,
    )


@REQUIRES_SIGKILL
@pytest.mark.parametrize(
    ("boundary", "expected_assets"),
    (("before_commit", 0), ("after_commit", 1)),
)
def test_library_catalog_actual_sigkill_has_exact_transaction_state(
    tmp_path: Path,
    boundary: str,
    expected_assets: int,
) -> None:
    database = tmp_path / "catalog.sqlite3"
    service = LibraryCatalogService(db_path=database)
    root = service.register_root(tmp_path)
    (tmp_path / "catalog_root_id.txt").write_text(root.root_id, encoding="utf-8")

    def state() -> dict[str, object]:
        reopened = LibraryCatalogService(db_path=database)
        with sqlite3.connect(database) as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        return {
            "assets": reopened.query_assets(CatalogQuery(limit=10)).total_count,
            "integrity": integrity,
        }

    pre = state()
    exit_code = _run_killed_child(tmp_path, "library_catalog", boundary)
    crash = state()
    assert crash["assets"] == expected_assets
    assert crash["integrity"] == "ok"

    _run_completed_child(tmp_path, "library_catalog")
    first = state()
    _run_completed_child(tmp_path, "library_catalog")
    recovered = state()
    assert first == recovered
    assert recovered["assets"] == 1
    _record_evidence(
        f"library_catalog_batch:{boundary}:actual_sigkill",
        operation="library_catalog_batch",
        boundary=boundary,
        child_exit_code=exit_code,
        pre_state=pre,
        crash_state=crash,
        recovered_state=recovered,
    )


@REQUIRES_SIGKILL
@pytest.mark.parametrize(
    ("boundary", "expect_replaced"),
    (("rewrite_intent_committed", False), ("before_replace", False), ("after_replace", True)),
)
def test_embedded_metadata_actual_sigkill_has_exact_replace_state(
    tmp_path: Path,
    boundary: str,
    expect_replaced: bool,
) -> None:
    image = tmp_path / "metadata.jpg"
    Image.new("RGB", (32, 32), (50, 80, 110)).save(image)
    original_sha = _file_sha(image)

    def state() -> dict[str, object]:
        with Image.open(image) as reopened:
            reopened.verify()
        with Image.open(image) as reopened:
            comment = str(reopened.getexif().get(37510, ""))
        return {"source": _file_sha(image), "comment": comment}

    pre = state()
    exit_code = _run_killed_child(tmp_path, "embedded_metadata", boundary)
    crash = state()
    assert (crash["source"] != original_sha) is expect_replaced
    assert ("replacement" in str(crash["comment"])) is expect_replaced

    GalleryActionService(temp_dir=tmp_path / "temporary").write_exif_comment(str(image), "replacement")
    first = state()
    GalleryActionService(temp_dir=tmp_path / "temporary").write_exif_comment(str(image), "replacement")
    recovered = state()
    assert first["comment"] == recovered["comment"]
    assert "replacement" in str(recovered["comment"])
    _record_evidence(
        f"embedded_metadata_rewrite:{boundary}:actual_sigkill",
        operation="embedded_metadata_rewrite",
        boundary=boundary,
        child_exit_code=exit_code,
        pre_state=pre,
        crash_state=crash,
        recovered_state=recovered,
    )


@REQUIRES_SIGKILL
@pytest.mark.parametrize(
    ("boundary", "source_exists", "staging_exists"),
    (
        ("before_detach", True, False),
        ("after_detach", False, True),
        ("after_recreate", True, True),
    ),
)
def test_rebuildable_cache_actual_sigkill_retries_detached_tree_twice(
    tmp_path: Path,
    boundary: str,
    source_exists: bool,
    staging_exists: bool,
) -> None:
    cache_root = tmp_path / "cache"
    target = cache_root / "cluster_results"
    target.mkdir(parents=True)
    (target / "payload.bin").write_bytes(b"derived")
    settings = SimpleNamespace(
        cache_dir=cache_root,
        embedding_cache_db=cache_root / "embeddings.sqlite3",
        thumbnail_cache_dir=cache_root / "thumbnails",
    )
    service = CacheMaintenanceService(settings=settings)
    staging = service._cache_clear_staging_path(target)

    def state() -> dict[str, object]:
        return {
            "source_exists": target.is_dir(),
            "source_files": sorted(path.name for path in target.iterdir()) if target.is_dir() else [],
            "staging_exists": staging.is_dir(),
            "staging_files": sorted(path.name for path in staging.iterdir()) if staging.is_dir() else [],
        }

    pre = state()
    exit_code = _run_killed_child(tmp_path, "rebuildable_cache", boundary)
    crash = state()
    assert crash["source_exists"] is source_exists
    assert crash["staging_exists"] is staging_exists

    service._clear_rebuildable_directory("cluster_results/", target)
    first = state()
    service._clear_rebuildable_directory("cluster_results/", target)
    recovered = state()
    assert first == recovered == {
        "source_exists": True,
        "source_files": [],
        "staging_exists": False,
        "staging_files": [],
    }
    _record_evidence(
        f"rebuildable_cache_clear:{boundary}:actual_sigkill",
        operation="rebuildable_cache_clear",
        boundary=boundary,
        child_exit_code=exit_code,
        pre_state=pre,
        crash_state=crash,
        recovered_state=recovered,
    )


@REQUIRES_SIGKILL
@pytest.mark.parametrize(
    ("boundary", "main_exists"),
    (("before_remove_main", True), ("after_remove_main", False)),
)
def test_sqlite_cache_actual_sigkill_preserves_or_removes_complete_main(
    tmp_path: Path,
    boundary: str,
    main_exists: bool,
) -> None:
    database = tmp_path / "embeddings.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA journal_mode=WAL;")
        connection.execute("CREATE TABLE cached(value TEXT)")
        connection.execute("INSERT INTO cached(value) VALUES ('ready')")
    service = CacheMaintenanceService(settings=SimpleNamespace(cache_dir=tmp_path, base_dir=tmp_path))

    def state() -> dict[str, object]:
        wal_exists = database.with_name(database.name + "-wal").exists()
        shm_exists = database.with_name(database.name + "-shm").exists()
        value = ""
        integrity = "removed"
        if database.is_file():
            with sqlite3.connect(database) as connection:
                value = connection.execute("SELECT value FROM cached").fetchone()[0]
                integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
        return {
            "main_exists": database.is_file(),
            "wal_exists": wal_exists,
            "shm_exists": shm_exists,
            "value": value,
            "integrity": integrity,
        }

    pre = state()
    exit_code = _run_killed_child(tmp_path, "sqlite_cache", boundary)
    crash = state()
    assert crash["main_exists"] is main_exists
    assert not crash["wal_exists"] and not crash["shm_exists"]
    if main_exists:
        assert crash["value"] == "ready" and crash["integrity"] == "ok"

    service.clear_sqlite_cache_bundle(database)
    first = state()
    service.clear_sqlite_cache_bundle(database)
    recovered = state()
    assert first == recovered
    assert recovered["main_exists"] is False
    _record_evidence(
        f"sqlite_cache_clear:{boundary}:actual_sigkill",
        operation="sqlite_cache_clear",
        boundary=boundary,
        child_exit_code=exit_code,
        pre_state=pre,
        crash_state=crash,
        recovered_state=recovered,
    )


@REQUIRES_SIGKILL
@pytest.mark.parametrize(
    ("boundary", "expected_status", "published"),
    (
        ("before_publish", "verified", False),
        ("after_publish", "published", True),
        ("journal_finalized", "complete", True),
    ),
)
def test_data_home_backup_actual_sigkill_recovers_publication_twice(
    tmp_path: Path,
    boundary: str,
    expected_status: str,
    published: bool,
) -> None:
    source = tmp_path / "runtime"
    _seed_data_home(source)
    manager = DataHomeManager("ClusterLens", source, config_path=tmp_path / "config.json")
    pre = {"source": _file_sha(source / "cache" / "thumbnails" / "one.webp")}

    exit_code = _run_killed_child(tmp_path, "data_home_backup", boundary)
    entries = manager.backup_recovery_journals()
    journal_path = next((source / "support" / "data_home_backups").glob("*.json"))
    if expected_status == "complete":
        assert not entries
        entry = manager.migration_preview(journal_path)
        entry["journal_path"] = str(journal_path)
    else:
        assert len(entries) == 1
        entry = entries[0]
    backup_root = Path(str(entry["backup_root"]))
    staging = Path(str(entry["staging_root"]))
    crash = {
        "status": entry["status"],
        "backup_exists": backup_root.is_dir(),
        "staging_exists": staging.is_dir(),
        "source": _file_sha(source / "cache" / "thumbnails" / "one.webp"),
    }
    assert crash["status"] == expected_status
    assert crash["backup_exists"] is published
    assert crash["staging_exists"] is (not published)
    assert crash["source"] == pre["source"]

    first = manager.discard_backup_staging(str(entry["journal_path"]))
    second = manager.discard_backup_staging(str(entry["journal_path"]))
    recovered = {
        "first": first["outcome"],
        "second": second["outcome"],
        "backup_exists": backup_root.is_dir(),
        "staging_exists": staging.exists(),
        "source": _file_sha(source / "cache" / "thumbnails" / "one.webp"),
    }
    expected_outcome = "published" if published else "discarded"
    assert first["outcome"] == second["outcome"] == expected_outcome
    assert recovered["source"] == pre["source"]
    if published:
        assert manager.verify_backup(backup_root) == (True, ())
    _record_evidence(
        f"data_home_backup:{boundary}:actual_sigkill",
        operation="data_home_backup",
        boundary=boundary,
        child_exit_code=exit_code,
        pre_state=pre,
        crash_state=crash,
        recovered_state=recovered,
    )


@REQUIRES_SIGKILL
@pytest.mark.parametrize(
    ("boundary", "expected_status", "target_exists", "config_exists"),
    (
        ("staging_verified", "verified", False, False),
        ("target_switched", "switched", True, False),
        ("active_pointer_committed", "switched", True, True),
    ),
)
def test_data_home_relocation_actual_sigkill_resumes_switch_twice(
    tmp_path: Path,
    boundary: str,
    expected_status: str,
    target_exists: bool,
    config_exists: bool,
) -> None:
    source = tmp_path / "runtime"
    target = tmp_path / "new-runtime"
    config = tmp_path / "control" / "data_home.json"
    _seed_data_home(source)
    manager = DataHomeManager("ClusterLens", source, config_path=config)
    pre = {"source": _file_sha(source / "cache" / "thumbnails" / "one.webp"), "config": config.exists()}

    exit_code = _run_killed_child(tmp_path, "data_home_relocation", boundary)
    journal = next((source / "support" / "data_home_migrations").glob("*.json"))
    crash = {
        "status": manager.migration_preview(journal)["status"],
        "target_exists": target.is_dir(),
        "config_exists": config.exists(),
        "source": _file_sha(source / "cache" / "thumbnails" / "one.webp"),
    }
    assert crash["status"] == expected_status
    assert crash["target_exists"] is target_exists
    assert crash["config_exists"] is config_exists
    assert crash["source"] == pre["source"]

    first = manager.resume_migration(journal)
    second = manager.resume_migration(journal)
    recovered = {
        "first_target": first.target_root,
        "second_target": second.target_root,
        "status": manager.migration_preview(journal)["status"],
        "target": _file_sha(target / "cache" / "thumbnails" / "one.webp"),
        "source": _file_sha(source / "cache" / "thumbnails" / "one.webp"),
        "config": config.read_text(encoding="utf-8"),
    }
    assert first.target_root == second.target_root == str(target)
    assert recovered["status"] == "complete"
    assert recovered["target"] == recovered["source"] == pre["source"]
    assert str(target) in str(recovered["config"])
    _record_evidence(
        f"data_home_relocation:{boundary}:actual_sigkill",
        operation="data_home_relocation",
        boundary=boundary,
        child_exit_code=exit_code,
        pre_state=pre,
        crash_state=crash,
        recovered_state=recovered,
    )


@REQUIRES_SIGKILL
@pytest.mark.parametrize(
    ("boundary", "manifest_boundary", "committed"),
    (
        ("journal_prepared", "journal_prepared", False),
        ("before_rename:face_search_human.db", "before_each_rename", False),
        ("after_rename:face_search_human.db", "after_each_rename", False),
        ("after_commit", "after_commit", True),
    ),
)
def test_face_storage_actual_sigkill_restores_or_retains_quarantine_twice(
    tmp_path: Path,
    boundary: str,
    manifest_boundary: str,
    committed: bool,
) -> None:
    database = tmp_path / "face_search_human.db"
    manifest: dict[str, str] = {}
    for index, path in enumerate(FaceStorageRemovalService.managed_paths(database), start=1):
        path.write_bytes(f"managed-{index}".encode("utf-8"))
        manifest[path.name] = _file_sha(path)

    def state() -> dict[str, object]:
        return {
            "managed": {
                name: _file_sha(tmp_path / name) if (tmp_path / name).is_file() else ""
                for name in sorted(manifest)
            },
            "journals": len(list((tmp_path / ".face-storage-recovery").glob("*/journal.json"))),
        }

    pre = state()
    exit_code = _run_killed_child(tmp_path, "face_storage", boundary)
    crash = state()

    service = FaceStorageRemovalService(tmp_path)
    first = service.recover_incomplete()
    after_first = state()
    second = service.recover_incomplete()
    recovered = state()
    assert after_first == recovered
    assert not first.failures and not second.failures
    if committed:
        assert all(not value for value in recovered["managed"].values())
        assert len(first.committed_operations) == 1
        assert first == second
    else:
        assert recovered["managed"] == manifest
        assert len(first.restored_operations) == 1
        assert not second.restored_operations and not second.committed_operations
    _record_evidence(
        f"face_storage_removal:{boundary}:actual_sigkill",
        operation="face_storage_removal",
        boundary=boundary,
        child_exit_code=exit_code,
        pre_state=pre,
        crash_state=crash,
        recovered_state=recovered,
    )
