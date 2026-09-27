from __future__ import annotations

import errno
from pathlib import Path

import pytest

from app.services.data_home import DataHomeManager
from infra.cancel import Cancelled


def _seed_data_home(root: Path) -> None:
    (root / "cache" / "thumbnails").mkdir(parents=True)
    (root / "cache" / "thumbnails" / "one.webp").write_bytes(b"preview")
    (root / "logs").mkdir()
    (root / "logs" / "app.log").write_text("line\n", encoding="utf-8")
    (root / "support").mkdir()
    (root / "support" / "recovery.json").write_text("{}\n", encoding="utf-8")


def test_backup_is_checksummed_and_detects_tampering(tmp_path: Path) -> None:
    source = tmp_path / "runtime"
    _seed_data_home(source)
    manager = DataHomeManager("ClusterLens", source, config_path=tmp_path / "config.json")

    backup = manager.create_backup(tmp_path / "backups")
    valid, failures = manager.verify_backup(backup.backup_root)

    assert valid
    assert not failures
    copied = Path(backup.backup_root) / "cache" / "thumbnails" / "one.webp"
    copied.write_bytes(b"tampered")
    valid, failures = manager.verify_backup(backup.backup_root)
    assert not valid
    assert any("Checksum mismatch" in failure for failure in failures)


def test_relocation_verifies_then_switches_on_next_launch_without_touching_source(tmp_path: Path) -> None:
    source = tmp_path / "runtime"
    target = tmp_path / "new-runtime"
    config = tmp_path / "control" / "data_home.json"
    _seed_data_home(source)
    manager = DataHomeManager("ClusterLens", source, config_path=config)

    result = manager.relocate(target, source_roots=(str(tmp_path / "photos"),))

    assert result.restart_required
    assert (target / ".clusterlens-data-home").is_file()
    assert (target / "support" / "data_home_migrations" / f"{result.migration_id}.json").is_file()
    assert (target / "cache" / "thumbnails" / "one.webp").read_bytes() == b"preview"
    assert (source / "cache" / "thumbnails" / "one.webp").read_bytes() == b"preview"
    assert '"data_home"' in config.read_text(encoding="utf-8")
    preview = manager.migration_preview(result.journal_path)
    assert preview["status"] == "complete"


def test_relocation_refuses_a_photo_source_or_nonempty_destination(tmp_path: Path) -> None:
    source = tmp_path / "runtime"
    _seed_data_home(source)
    manager = DataHomeManager("ClusterLens", source, config_path=tmp_path / "config.json")
    photos = tmp_path / "photos"
    photos.mkdir()
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="photo source"):
        manager.relocate(photos / "managed", source_roots=(str(photos),))
    with pytest.raises(ValueError, match="empty"):
        manager.relocate(occupied)


def test_interrupted_relocation_can_resume_or_roll_back_without_source_mutation(tmp_path: Path) -> None:
    source = tmp_path / "runtime"
    target = tmp_path / "new-runtime"
    _seed_data_home(source)
    manager = DataHomeManager("ClusterLens", source, config_path=tmp_path / "config.json")
    checks = 0

    def cancel_after_staging() -> bool:
        nonlocal checks
        checks += 1
        return checks > 10

    with pytest.raises(Exception):
        manager.relocate(target, cancel_check=cancel_after_staging)

    journal = next((source / "support" / "data_home_migrations").glob("*.json"))
    preview = manager.migration_preview(journal)
    assert preview["status"] == "cancelled"
    assert not target.exists()
    assert (source / "cache" / "thumbnails" / "one.webp").read_bytes() == b"preview"

    resumed = manager.resume_migration(journal)
    assert resumed.target_root == str(target)
    assert manager.migration_preview(journal)["status"] == "complete"
    assert (target / "cache" / "thumbnails" / "one.webp").read_bytes() == b"preview"

    failed_journal = source / "support" / "data_home_migrations" / "manual-failed.json"
    staging = target.parent / ".discard.manual-failed.staging"
    staging.mkdir()
    (staging / "temporary").write_text("safe to discard", encoding="utf-8")
    failed_journal.write_text(
        '{"id":"manual-failed","source_root":"' + str(source) + '","target_root":"' + str(target.parent / "discard")
        + '","staging_root":"' + str(staging) + '","status":"failed"}',
        encoding="utf-8",
    )
    assert manager.rollback_migration(failed_journal)
    assert not staging.exists()
    assert manager.migration_preview(failed_journal)["status"] == "rolled_back"


def test_backup_refuses_a_destination_inside_the_active_data_home(tmp_path: Path) -> None:
    source = tmp_path / "runtime"
    _seed_data_home(source)
    manager = DataHomeManager("ClusterLens", source, config_path=tmp_path / "config.json")

    with pytest.raises(ValueError, match="outside"):
        manager.create_backup(source / "backups")


def test_backup_publication_is_atomic_across_process_exit_boundaries(tmp_path: Path, monkeypatch) -> None:
    class ProcessExit(BaseException):
        pass

    for boundary, expected_published in (("before_publish", False), ("after_publish", True)):
        case = tmp_path / boundary
        source = case / "runtime"
        backups = case / "backups"
        _seed_data_home(source)
        manager = DataHomeManager("ClusterLens", source, config_path=case / "config.json")

        def exit_at_checkpoint(name: str, _staging: Path, _backup_root: Path) -> None:
            if name == boundary:
                raise ProcessExit(name)

        monkeypatch.setattr(manager, "_backup_checkpoint", exit_at_checkpoint)
        with pytest.raises(ProcessExit):
            manager.create_backup(backups)

        published = list(backups.glob("clusterlens-backup-*"))
        assert bool(published) is expected_published
        if published:
            valid, failures = manager.verify_backup(published[0])
            assert valid
            assert not failures
        assert (source / "cache" / "thumbnails" / "one.webp").read_bytes() == b"preview"


def test_backup_copy_permission_failure_never_publishes_a_backup(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "runtime"
    backups = tmp_path / "backups"
    _seed_data_home(source)
    manager = DataHomeManager("ClusterLens", source, config_path=tmp_path / "config.json")

    def deny_copy(*_args, **_kwargs) -> None:
        raise PermissionError("read-only destination fixture")

    monkeypatch.setattr("app.services.data_home.shutil.copy2", deny_copy)
    with pytest.raises(PermissionError, match="read-only destination"):
        manager.create_backup(backups)

    assert not list(backups.glob("clusterlens-backup-*"))
    assert (source / "cache" / "thumbnails" / "one.webp").read_bytes() == b"preview"


def test_backup_cancel_before_commit_is_discardable_and_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "runtime"
    backups = tmp_path / "backups"
    _seed_data_home(source)
    manager = DataHomeManager("ClusterLens", source, config_path=tmp_path / "config.json")
    checks = 0

    def cancel_during_copy() -> bool:
        nonlocal checks
        checks += 1
        return checks > 2

    with pytest.raises(Cancelled):
        manager.create_backup(backups, cancel_check=cancel_during_copy)

    assert not list(backups.glob("clusterlens-backup-*"))
    entry = manager.backup_recovery_journals()[0]
    first = manager.discard_backup_staging(str(entry["journal_path"]))
    second = manager.discard_backup_staging(str(entry["journal_path"]))
    assert first["outcome"] == second["outcome"] == "discarded"
    assert manager.backup_recovery_journals() == ()
    assert (source / "cache" / "thumbnails" / "one.webp").read_bytes() == b"preview"


def test_backup_cancel_after_publication_retains_and_reports_complete_backup(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "runtime"
    backups = tmp_path / "backups"
    _seed_data_home(source)
    manager = DataHomeManager("ClusterLens", source, config_path=tmp_path / "config.json")
    cancelled = False

    def cancel_after_publish(name: str, _staging: Path, _backup: Path) -> None:
        nonlocal cancelled
        if name == "after_publish":
            cancelled = True

    monkeypatch.setattr(manager, "_backup_checkpoint", cancel_after_publish)
    backup = manager.create_backup(backups, cancel_check=lambda: cancelled)

    assert cancelled
    assert manager.verify_backup(backup.backup_root) == (True, ())
    assert manager.backup_recovery_journals() == ()
    assert manager.migration_preview(backup.journal_path)["status"] == "complete"


@pytest.mark.parametrize(
    "fault",
    (
        OSError(errno.EIO, "injected backup I/O error"),
        FileNotFoundError(errno.ENOENT, "source vanished during backup"),
    ),
)
def test_backup_copy_fault_never_publishes_and_recovery_discards_staging(
    tmp_path: Path,
    monkeypatch,
    fault: OSError,
) -> None:
    source = tmp_path / "runtime"
    backups = tmp_path / "backups"
    _seed_data_home(source)
    manager = DataHomeManager("ClusterLens", source, config_path=tmp_path / "config.json")

    def fail_copy(*_args, **_kwargs) -> None:
        raise fault

    monkeypatch.setattr("app.services.data_home.shutil.copy2", fail_copy)
    with pytest.raises(type(fault)):
        manager.create_backup(backups)

    assert not list(backups.glob("clusterlens-backup-*"))
    entry = manager.backup_recovery_journals()[0]
    first = manager.discard_backup_staging(str(entry["journal_path"]))
    second = manager.discard_backup_staging(str(entry["journal_path"]))
    assert first["outcome"] == second["outcome"] == "discarded"


def test_incomplete_backup_staging_is_discoverable_and_discard_is_idempotent(tmp_path: Path, monkeypatch) -> None:
    class ProcessExit(BaseException):
        pass

    source = tmp_path / "runtime"
    backups = tmp_path / "backups"
    _seed_data_home(source)
    manager = DataHomeManager("ClusterLens", source, config_path=tmp_path / "config.json")

    def exit_before_publish(name: str, _staging: Path, _backup_root: Path) -> None:
        if name == "before_publish":
            raise ProcessExit(name)

    monkeypatch.setattr(manager, "_backup_checkpoint", exit_before_publish)
    with pytest.raises(ProcessExit):
        manager.create_backup(backups)

    entries = manager.backup_recovery_journals()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["status"] == "verified"
    staging = Path(str(entry["staging_root"]))
    assert staging.is_dir()
    assert not Path(str(entry["backup_root"])).exists()

    restarted = DataHomeManager("ClusterLens", source, config_path=tmp_path / "config.json")
    first = restarted.discard_backup_staging(str(entry["journal_path"]))
    second = restarted.discard_backup_staging(str(entry["journal_path"]))
    assert first["outcome"] == second["outcome"] == "discarded"
    assert not staging.exists()
    assert restarted.backup_recovery_journals() == ()
    assert (source / "cache" / "thumbnails" / "one.webp").read_bytes() == b"preview"


def test_backup_recovery_retains_and_finalizes_an_already_published_backup(tmp_path: Path, monkeypatch) -> None:
    class ProcessExit(BaseException):
        pass

    source = tmp_path / "runtime"
    backups = tmp_path / "backups"
    _seed_data_home(source)
    manager = DataHomeManager("ClusterLens", source, config_path=tmp_path / "config.json")

    def exit_after_publish(name: str, _staging: Path, _backup_root: Path) -> None:
        if name == "after_publish":
            raise ProcessExit(name)

    monkeypatch.setattr(manager, "_backup_checkpoint", exit_after_publish)
    with pytest.raises(ProcessExit):
        manager.create_backup(backups)

    entry = manager.backup_recovery_journals()[0]
    backup_root = Path(str(entry["backup_root"]))
    assert entry["status"] == "published"
    assert backup_root.is_dir()

    restarted = DataHomeManager("ClusterLens", source, config_path=tmp_path / "config.json")
    first = restarted.discard_backup_staging(str(entry["journal_path"]))
    second = restarted.discard_backup_staging(str(entry["journal_path"]))
    assert first["outcome"] == second["outcome"] == "published"
    assert backup_root.is_dir()
    assert restarted.verify_backup(backup_root) == (True, ())
    assert restarted.backup_recovery_journals() == ()


def test_backup_discard_refuses_unmarked_or_mismatched_staging(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "runtime"
    backups = tmp_path / "backups"
    _seed_data_home(source)
    manager = DataHomeManager("ClusterLens", source, config_path=tmp_path / "config.json")

    def deny_copy(*_args, **_kwargs) -> None:
        raise PermissionError("stop after marker")

    monkeypatch.setattr("app.services.data_home.shutil.copy2", deny_copy)
    with pytest.raises(PermissionError):
        manager.create_backup(backups)
    entry = manager.backup_recovery_journals()[0]
    staging = Path(str(entry["staging_root"]))
    marker = staging / ".clusterlens-data-home-backup"
    external_journal = tmp_path / "external-backup.json"
    external_journal.write_text(Path(str(entry["journal_path"])).read_text(encoding="utf-8"), encoding="utf-8")
    with pytest.raises(ValueError, match="outside the active Data Home"):
        manager.discard_backup_staging(external_journal)
    marker.write_text('{"app_name":"Other","backup_id":"wrong"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="does not match"):
        manager.discard_backup_staging(str(entry["journal_path"]))
    assert staging.is_dir()
    assert (source / "cache" / "thumbnails" / "one.webp").read_bytes() == b"preview"


def test_backup_recovery_can_discard_a_pre_staging_failure_record(tmp_path: Path) -> None:
    source = tmp_path / "runtime"
    backups = tmp_path / "backups"
    _seed_data_home(source)
    manager = DataHomeManager("ClusterLens", source, config_path=tmp_path / "config.json")
    backup_id = "clusterlens-backup-pre-staging"
    journal = source / "support" / "data_home_backups" / f"{backup_id}.json"
    journal.parent.mkdir(parents=True)
    journal.write_text(
        "{"
        f'"kind":"backup","id":"{backup_id}","destination_root":"{backups}",'
        f'"backup_root":"{backups / backup_id}",'
        f'"staging_root":"{backups / ("." + backup_id + ".staging")}",'
        '"status":"failed","error":"destination unavailable"}',
        encoding="utf-8",
    )

    result = manager.discard_backup_staging(journal)

    assert result["outcome"] == "discarded"
    assert manager.migration_preview(journal)["status"] == "discarded"
    assert manager.backup_recovery_journals() == ()


def test_verified_relocation_resume_rejects_corrupt_staging_and_never_recopies_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class ProcessExit(BaseException):
        pass

    source = tmp_path / "runtime"
    target = tmp_path / "new-runtime"
    _seed_data_home(source)
    manager = DataHomeManager("ClusterLens", source, config_path=tmp_path / "config.json")

    def exit_before_publish(_staging: Path, _target: Path) -> None:
        raise ProcessExit()

    monkeypatch.setattr(manager, "_publish_staging", exit_before_publish)
    with pytest.raises(ProcessExit):
        manager.relocate(target)

    journal = next((source / "support" / "data_home_migrations").glob("*.json"))
    record = manager.migration_preview(journal)
    assert record["status"] == "verified"
    staging = Path(str(record["staging_root"]))
    staged_photo = staging / "cache" / "thumbnails" / "one.webp"
    staged_photo.write_bytes(b"corrupt")
    (source / "cache" / "thumbnails" / "one.webp").write_bytes(b"changed-source")

    restarted = DataHomeManager("ClusterLens", source, config_path=tmp_path / "config.json")
    with pytest.raises(ValueError, match="staging is corrupt"):
        restarted.resume_migration(journal)

    assert not target.exists()
    assert staged_photo.read_bytes() == b"corrupt"
    assert (source / "cache" / "thumbnails" / "one.webp").read_bytes() == b"changed-source"


def test_incomplete_relocation_resume_refuses_a_vanished_active_data_home(tmp_path: Path) -> None:
    source = tmp_path / "runtime"
    target = tmp_path / "new-runtime"
    _seed_data_home(source)
    manager = DataHomeManager("ClusterLens", source, config_path=tmp_path / "config.json")
    checks = 0

    def cancel_during_copy() -> bool:
        nonlocal checks
        checks += 1
        return checks > 2

    with pytest.raises(Exception):
        manager.relocate(target, cancel_check=cancel_during_copy)
    journal = next((source / "support" / "data_home_migrations").glob("*.json"))
    vanished = tmp_path / "vanished-runtime"
    source.rename(vanished)

    restarted = DataHomeManager("ClusterLens", source, config_path=tmp_path / "config.json")
    moved_journal = next((vanished / "support" / "data_home_migrations").glob("*.json"))
    with pytest.raises(ValueError, match="active Data Home is unavailable"):
        restarted.resume_migration(moved_journal)

    assert not target.exists()


def test_relocation_recovers_process_exit_after_directory_switch_idempotently(tmp_path: Path, monkeypatch) -> None:
    class ProcessExit(BaseException):
        pass

    source = tmp_path / "runtime"
    target = tmp_path / "new-runtime"
    config = tmp_path / "control" / "data_home.json"
    _seed_data_home(source)
    manager = DataHomeManager("ClusterLens", source, config_path=config)
    original_publish = manager._publish_staging

    def publish_then_exit(staging: Path, published: Path) -> None:
        original_publish(staging, published)
        raise ProcessExit()

    monkeypatch.setattr(manager, "_publish_staging", publish_then_exit)
    with pytest.raises(ProcessExit):
        manager.relocate(target)

    journal = next((source / "support" / "data_home_migrations").glob("*.json"))
    assert manager.migration_preview(journal)["status"] == "verified"
    assert (target / "cache" / "thumbnails" / "one.webp").read_bytes() == b"preview"
    assert not config.exists()

    restarted = DataHomeManager("ClusterLens", source, config_path=config)
    first = restarted.resume_migration(journal)
    second = restarted.resume_migration(journal)
    assert first.target_root == second.target_root == str(target)
    assert restarted.migration_preview(journal)["status"] == "complete"
    assert str(target) in config.read_text(encoding="utf-8")
    assert (source / "cache" / "thumbnails" / "one.webp").read_bytes() == b"preview"


def test_relocation_cancel_after_directory_switch_finishes_the_committed_pointer(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "runtime"
    target = tmp_path / "new-runtime"
    config = tmp_path / "control" / "data_home.json"
    _seed_data_home(source)
    manager = DataHomeManager("ClusterLens", source, config_path=config)
    original_publish = manager._publish_staging
    cancelled = False

    def publish_then_cancel(staging: Path, published: Path) -> None:
        nonlocal cancelled
        original_publish(staging, published)
        cancelled = True

    monkeypatch.setattr(manager, "_publish_staging", publish_then_cancel)
    result = manager.relocate(target, cancel_check=lambda: cancelled)

    assert cancelled
    assert result.target_root == str(target)
    assert (target / "cache" / "thumbnails" / "one.webp").read_bytes() == b"preview"
    assert str(target) in config.read_text(encoding="utf-8")
    assert manager.migration_preview(result.journal_path)["status"] == "complete"


@pytest.mark.parametrize(
    "fault",
    (
        OSError(errno.EIO, "injected relocation I/O error"),
        PermissionError(errno.EACCES, "permission denied"),
        OSError(errno.ENOSPC, "disk full"),
    ),
)
def test_relocation_publish_fault_preserves_old_home_and_resumes_idempotently(
    tmp_path: Path,
    monkeypatch,
    fault: OSError,
) -> None:
    case = tmp_path / str(fault.errno)
    source = case / "runtime"
    target = case / "new-runtime"
    config = case / "control" / "data_home.json"
    _seed_data_home(source)
    manager = DataHomeManager("ClusterLens", source, config_path=config)

    def fail_publish(_staging: Path, _target: Path) -> None:
        raise fault

    monkeypatch.setattr(manager, "_publish_staging", fail_publish)
    with pytest.raises(type(fault)):
        manager.relocate(target)

    assert (source / "cache" / "thumbnails" / "one.webp").read_bytes() == b"preview"
    assert not target.exists()
    assert not config.exists()
    journal = next((source / "support" / "data_home_migrations").glob("*.json"))

    restarted = DataHomeManager("ClusterLens", source, config_path=config)
    first = restarted.resume_migration(journal)
    second = restarted.resume_migration(journal)
    assert first.target_root == second.target_root == str(target)
    assert (target / "cache" / "thumbnails" / "one.webp").read_bytes() == b"preview"
    assert str(target) in config.read_text(encoding="utf-8")
