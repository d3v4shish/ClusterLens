from __future__ import annotations

from pathlib import Path

import pytest

from app.services.data_home import DataHomeManager


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
