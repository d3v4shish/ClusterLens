from __future__ import annotations

import errno
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from app.services.embedding_index import EmbeddingIndexService
from app.services.face_model_installer import FaceModelInstaller
from app.services.gallery_actions import GalleryActionService
from app.services.image_tags import ImageTagService
from app.services.photo_metadata import PhotoEditDraft, PhotoEditService
from app.services.result_cache import ResultCacheService
from app.services.saved_searches import SavedSearchService
from infra import atomic_io
from infra.cancel import Cancelled


class SimulatedProcessExit(BaseException):
    """Bypass normal Exception handlers at one deterministic commit seam."""


def _raise_at(target: Path, phase: str, wanted: Path, fault: BaseException) -> None:
    if Path(target) == wanted and phase == "before_replace":
        raise fault


@pytest.mark.parametrize(
    "fault",
    (
        PermissionError(errno.EACCES, "permission denied"),
        OSError(errno.ENOSPC, "disk full"),
        SimulatedProcessExit("before sidecar commit"),
    ),
)
def test_photo_sidecar_faults_preserve_source_and_prior_sidecar(tmp_path: Path, fault: BaseException) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"source-photo")
    sidecar = tmp_path / "photo.jpg.clusterlens.json"
    sidecar.write_text('{"version": 2, "keep": true}\n', encoding="utf-8")
    original_source = image.read_bytes()
    original_sidecar = sidecar.read_bytes()

    with patch.object(
        atomic_io,
        "_atomic_write_checkpoint",
        side_effect=lambda target, phase: _raise_at(target, phase, sidecar, fault),
    ), pytest.raises(type(fault)):
        PhotoEditService().save_draft(image, PhotoEditDraft(title="replacement"))

    assert image.read_bytes() == original_source
    assert sidecar.read_bytes() == original_sidecar
    assert not list(tmp_path.glob(".*.partial"))


@pytest.mark.parametrize(
    "fault",
    (
        PermissionError(errno.EACCES, "permission denied"),
        OSError(errno.ENOSPC, "disk full"),
        SimulatedProcessExit("before saved-search commit"),
    ),
)
def test_saved_search_faults_preserve_the_complete_prior_revision(tmp_path: Path, fault: BaseException) -> None:
    path = tmp_path / "saved_searches.json"
    service = SavedSearchService(path)
    record = service.save_search("Original", "unknown_people", {"mode": "unknown"})
    before = path.read_bytes()

    with patch.object(
        atomic_io,
        "_atomic_write_checkpoint",
        side_effect=lambda target, phase: _raise_at(target, phase, path, fault),
    ), pytest.raises(type(fault)):
        service.rename_search(record.search_id, "Replacement")

    assert path.read_bytes() == before
    assert SavedSearchService(path).get_search(record.search_id).name == "Original"


@pytest.mark.parametrize(
    "fault",
    (
        OSError(errno.EIO, "I/O failure"),
        PermissionError(errno.EACCES, "permission denied"),
        OSError(errno.ENOSPC, "disk full"),
        SimulatedProcessExit("before result-cache commit"),
    ),
)
def test_result_cache_faults_preserve_the_prior_complete_result(tmp_path: Path, fault: BaseException) -> None:
    service = ResultCacheService()
    service.cache_dir = tmp_path
    key = "fault-matrix"
    service.save(key, {0: ["old.jpg"]}, {"generation": "old"})
    path = tmp_path / f"{key}.json"
    before = path.read_bytes()

    with patch.object(
        atomic_io,
        "_atomic_write_checkpoint",
        side_effect=lambda target, phase: _raise_at(target, phase, path, fault),
    ), pytest.raises(type(fault)):
        service.save(key, {0: ["new.jpg"]}, {"generation": "new"})

    assert path.read_bytes() == before
    assert service.load(key) == ({0: ["old.jpg"]}, {"generation": "old"})


def test_embedding_index_process_exit_before_manifest_is_not_a_ready_generation(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"fixture")
    service = EmbeddingIndexService()
    service.index_dir = tmp_path / "indexes"
    service.index_dir.mkdir()
    snapshot = service.build_snapshot_key([str(image)])
    original = [(str(image), np.asarray([0.1, 0.2], dtype=np.float32))]
    replacement = [(str(image), np.asarray([0.8, 0.9], dtype=np.float32))]
    paths = service.save_index(snapshot, "clip", original)
    manifest = Path(paths["manifest_path"])

    with patch.object(
        atomic_io,
        "_atomic_write_checkpoint",
        side_effect=lambda target, phase: _raise_at(
            target,
            phase,
            manifest,
            SimulatedProcessExit("before manifest commit"),
        ),
    ), pytest.raises(SimulatedProcessExit):
        service.save_index(snapshot, "clip", replacement)

    assert not manifest.exists()
    repaired, reused = service.ensure_index(snapshot, "clip", replacement)
    assert not reused
    assert Path(repaired["manifest_path"]).is_file()
    np.testing.assert_allclose(
        np.load(repaired["vector_path"], allow_pickle=False),
        np.asarray([[0.8, 0.9]], dtype=np.float32),
    )


@pytest.mark.parametrize(
    "fault",
    (
        PermissionError(errno.EACCES, "permission denied"),
        OSError(errno.ENOSPC, "disk full"),
    ),
)
def test_embedded_metadata_write_faults_preserve_the_original_image(tmp_path: Path, fault: BaseException) -> None:
    image = tmp_path / "photo.jpg"
    Image.new("RGB", (32, 32), (50, 80, 110)).save(image)
    before = image.read_bytes()
    service = GalleryActionService(temp_dir=tmp_path / "temporary")

    with patch.object(
        service,
        "_metadata_rewrite_checkpoint",
        side_effect=lambda name, *_args: (_ for _ in ()).throw(fault) if name == "before_replace" else None,
    ), pytest.raises(type(fault)):
        service.write_exif_comment(str(image), "replacement")

    assert image.read_bytes() == before
    assert not list(tmp_path.glob("*.ic_tmp*"))


@pytest.mark.parametrize(
    ("boundary", "expect_replaced"),
    (("before_replace", False), ("after_replace", True)),
)
def test_embedded_metadata_process_exit_has_an_exact_commit_state(
    tmp_path: Path,
    boundary: str,
    expect_replaced: bool,
) -> None:
    image = tmp_path / f"{boundary}.jpg"
    Image.new("RGB", (32, 32), (50, 80, 110)).save(image)
    before = image.read_bytes()
    service = GalleryActionService(temp_dir=tmp_path / "temporary")

    def exit_at(name: str, *_args) -> None:
        if name == boundary:
            raise SimulatedProcessExit(name)

    with patch.object(service, "_metadata_rewrite_checkpoint", side_effect=exit_at), pytest.raises(SimulatedProcessExit):
        service.write_exif_comment(str(image), "replacement")

    assert (image.read_bytes() != before) is expect_replaced
    with Image.open(image) as reopened:
        reopened.verify()
    if expect_replaced:
        with Image.open(image) as reopened:
            assert "replacement" in str(reopened.getexif().get(37510, ""))


@pytest.mark.parametrize(
    "fault",
    (
        Cancelled("cancel before promotion"),
        PermissionError(errno.EACCES, "permission denied"),
        OSError(errno.ENOSPC, "disk full"),
    ),
)
def test_face_model_precommit_fault_restores_the_previous_bundle(tmp_path: Path, fault: BaseException) -> None:
    installer = FaceModelInstaller(settings=SimpleNamespace(cache_dir=tmp_path))
    parent = tmp_path / "face_model_assets" / "human" / "detectors"
    target = parent / "bundle"
    staging = parent / ".bundle.test.installing"
    target.mkdir(parents=True)
    staging.mkdir()
    (target / "payload.bin").write_bytes(b"old")
    (staging / "payload.bin").write_bytes(b"new")

    def fail_after_previous_is_staged(name: str, *_args) -> None:
        if name == "previous_staged":
            raise fault

    with patch.object(installer, "_promotion_checkpoint", side_effect=fail_after_previous_is_staged), pytest.raises(type(fault)):
        installer._promote_bundle_directory(staging, target)

    assert (target / "payload.bin").read_bytes() == b"old"
    assert (staging / "payload.bin").read_bytes() == b"new"
    assert not list(parent.glob(".bundle.*.previous"))


def test_gallery_cancel_after_commit_reports_the_committed_file(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    source.write_bytes(b"source")
    service = GalleryActionService(
        audit_log_path=tmp_path / "logs" / "operations.jsonl",
        journal_path=tmp_path / "logs" / "operations.sqlite3",
        temp_dir=tmp_path / "temporary",
    )
    committed = False

    def checkpoint(name: str, *_args) -> None:
        nonlocal committed
        if name == "after_filesystem_mutation":
            committed = True

    service._file_mutation_checkpoint = checkpoint  # type: ignore[method-assign]
    result = service.move_to_directory(
        [str(source)],
        str(tmp_path / "destination"),
        cancel_check=lambda: committed,
    )

    assert result.cancelled
    assert len(result.changed_paths) == 1
    assert not source.exists()
    assert Path(result.changed_paths[0][1]).read_bytes() == b"source"


def test_gallery_process_exit_before_mutation_recovers_without_source_change(tmp_path: Path) -> None:
    source = tmp_path / "source.jpg"
    source.write_bytes(b"source")
    service = GalleryActionService(
        audit_log_path=tmp_path / "logs" / "operations.jsonl",
        journal_path=tmp_path / "logs" / "operations.sqlite3",
        temp_dir=tmp_path / "temporary",
    )

    def exit_before(name: str, *_args) -> None:
        if name == "before_filesystem_mutation":
            raise SimulatedProcessExit(name)

    service._file_mutation_checkpoint = exit_before  # type: ignore[method-assign]
    with pytest.raises(SimulatedProcessExit):
        service.move_to_directory([str(source)], str(tmp_path / "destination"))

    restarted = GalleryActionService(
        audit_log_path=tmp_path / "logs" / "operations.jsonl",
        journal_path=tmp_path / "logs" / "operations.sqlite3",
        temp_dir=tmp_path / "temporary",
    )
    first = restarted.recover_incomplete_operations()
    second = restarted.recover_incomplete_operations()
    assert source.read_bytes() == b"source"
    assert not list((tmp_path / "destination").glob("*"))
    assert len(first["recovered_operations"]) == 1
    assert not second["recovered_operations"]


@pytest.mark.parametrize(
    "fault",
    (
        PermissionError(errno.EACCES, "permission denied"),
        OSError(errno.ENOSPC, "disk full"),
    ),
)
def test_gallery_write_faults_leave_the_source_and_report_failure(tmp_path: Path, fault: BaseException) -> None:
    source = tmp_path / "source.jpg"
    source.write_bytes(b"source")
    service = GalleryActionService(
        audit_log_path=tmp_path / "logs" / "operations.jsonl",
        journal_path=tmp_path / "logs" / "operations.sqlite3",
        temp_dir=tmp_path / "temporary",
    )

    with patch.object(service, "_safe_move", side_effect=fault):
        result = service.move_to_directory([str(source)], str(tmp_path / "destination"))

    assert source.read_bytes() == b"source"
    assert not result.changed_paths
    assert len(result.failures) == 1
    assert not list((tmp_path / "destination").glob("*"))


@pytest.mark.parametrize(
    "fault",
    (
        PermissionError(errno.EACCES, "permission denied"),
        sqlite3.OperationalError("database or disk is full"),
        SimulatedProcessExit("before tag commit"),
    ),
)
def test_image_tag_precommit_faults_roll_back_the_complete_transaction(tmp_path: Path, fault: BaseException) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"photo")
    database = tmp_path / "tags.sqlite3"
    service = ImageTagService(db_path=database)
    service._replace_rows_for_path(str(image), [("old", "Old", "user")])

    def fail_before(name: str) -> None:
        if name == "before_commit":
            raise fault

    service._tag_write_checkpoint = fail_before  # type: ignore[method-assign]
    with pytest.raises(type(fault)):
        service.rename_tag("Old", "New")

    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT tag_norm, display_tag FROM image_tags").fetchall()
        assert rows == [("old", "Old")]
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)


def test_image_tag_process_exit_after_commit_retains_the_complete_transaction(tmp_path: Path) -> None:
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"photo")
    database = tmp_path / "tags.sqlite3"
    service = ImageTagService(db_path=database)
    service._replace_rows_for_path(str(image), [("old", "Old", "user")])

    def exit_after(name: str) -> None:
        if name == "after_commit":
            raise SimulatedProcessExit(name)

    service._tag_write_checkpoint = exit_after  # type: ignore[method-assign]
    with pytest.raises(SimulatedProcessExit):
        service.rename_tag("Old", "New")

    with sqlite3.connect(database) as connection:
        rows = connection.execute("SELECT tag_norm, display_tag FROM image_tags").fetchall()
        assert rows == [("new", "New")]
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
