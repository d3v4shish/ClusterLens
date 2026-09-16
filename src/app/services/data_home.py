from __future__ import annotations

"""Safe, checksummed management of the ClusterLens Data Home.

Source folders are deliberately absent from this module.  It manages only the
runtime directory that ClusterLens owns: caches, indexes, logs, journals,
models, reports, and backups.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
from typing import Callable
from uuid import uuid4

from apps.shared.runtime_support import data_home_config_path
from infra.atomic_io import atomic_write_text
from infra.cancel import raise_if_cancelled


MANAGED_TOP_LEVEL = ("cache", "logs", "crash", "benchmarks", "model_assets", "support")
MARKER_NAME = ".clusterlens-data-home"


@dataclass(frozen=True)
class DataHomeInventory:
    root: str
    files: int
    bytes: int
    categories: dict[str, int]


@dataclass(frozen=True)
class DataHomeBackup:
    backup_root: str
    manifest_path: str
    files: int
    bytes: int


@dataclass(frozen=True)
class DataHomeMigration:
    migration_id: str
    source_root: str
    target_root: str
    journal_path: str
    files: int
    bytes: int
    restart_required: bool = True


class DataHomeManager:
    def __init__(self, app_name: str, current_root: str | Path, *, config_path: str | Path | None = None) -> None:
        self.app_name = str(app_name or "ClusterLens")
        self.current_root = Path(current_root).expanduser().absolute()
        self.config_path = Path(config_path) if config_path is not None else data_home_config_path(self.app_name)

    def inventory(self, *, progress_callback=None, cancel_check=None) -> DataHomeInventory:
        categories: dict[str, int] = {}
        files = 0
        total_bytes = 0
        for index, name in enumerate(MANAGED_TOP_LEVEL):
            raise_if_cancelled(cancel_check)
            size, count = self._tree_size(self.current_root / name, cancel_check=cancel_check)
            categories[name] = size
            files += count
            total_bytes += size
            self._progress(progress_callback, index + 1, len(MANAGED_TOP_LEVEL), f"Scanning {name}")
        return DataHomeInventory(str(self.current_root), files, total_bytes, categories)

    def create_backup(self, destination: str | Path, *, progress_callback=None, cancel_check=None) -> DataHomeBackup:
        destination_root = Path(destination).expanduser().absolute()
        self._validate_backup_destination(destination_root)
        backup_root = destination_root / f"clusterlens-backup-{self._stamp()}-{uuid4().hex[:8]}"
        backup_root.mkdir(parents=True, exist_ok=False)
        manifest = self._copy_managed_tree(backup_root, progress_callback=progress_callback, cancel_check=cancel_check)
        manifest_path = backup_root / "manifest.json"
        atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        return DataHomeBackup(str(backup_root), str(manifest_path), len(manifest["files"]), int(manifest["bytes"]))

    def verify_backup(self, backup_root: str | Path, *, progress_callback=None, cancel_check=None) -> tuple[bool, tuple[str, ...]]:
        root = Path(backup_root).expanduser().absolute()
        manifest_path = root / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as exc:
            return False, (f"Cannot read backup manifest: {exc}",)
        failures: list[str] = []
        files = list(manifest.get("files", []) or [])
        for index, entry in enumerate(files):
            raise_if_cancelled(cancel_check)
            relative = str(entry.get("path", "") or "")
            candidate = root / relative
            if not self._safe_relative(candidate, root):
                failures.append(f"Unsafe manifest path: {relative}")
            elif not candidate.is_file():
                failures.append(f"Missing: {relative}")
            elif self._digest(candidate, cancel_check=cancel_check) != str(entry.get("sha256", "")):
                failures.append(f"Checksum mismatch: {relative}")
            self._progress(progress_callback, index + 1, max(1, len(files)), f"Verifying {relative}")
        return not failures, tuple(failures)

    def relocate(
        self,
        target: str | Path,
        *,
        source_roots: tuple[str, ...] = (),
        progress_callback=None,
        cancel_check=None,
    ) -> DataHomeMigration:
        target_root = Path(target).expanduser().absolute()
        self._validate_relocation_target(target_root, source_roots)
        migration_id = f"data-home-{self._stamp()}-{uuid4().hex[:8]}"
        journal_path = self.current_root / "support" / "data_home_migrations" / f"{migration_id}.json"
        staging = target_root.parent / f".{target_root.name}.{migration_id}.staging"
        record = {
            "id": migration_id,
            "source_root": str(self.current_root),
            "target_root": str(target_root),
            "staging_root": str(staging),
            "status": "copying",
            "created_at_utc": self._timestamp(),
        }
        self._write_journal(journal_path, record)
        return self._continue_relocation(
            record,
            journal_path=journal_path,
            source_roots=source_roots,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )

    def resume_migration(
        self,
        journal_path: str | Path,
        *,
        source_roots: tuple[str, ...] = (),
        progress_callback=None,
        cancel_check=None,
    ) -> DataHomeMigration:
        """Resume an interrupted derived-data migration from its safe staging area.

        The source Data Home remains authoritative until verification and the
        final directory switch complete.  Existing staged files are only
        reused when their source and destination checksums agree.
        """
        path = Path(journal_path).expanduser().absolute()
        record = self.migration_preview(path)
        if record.get("status") not in {"copying", "failed", "cancelled", "verified", "rollback_cancelled"}:
            raise ValueError("Only an incomplete Data Home migration can be resumed.")
        if Path(str(record.get("source_root", ""))).absolute() != self.current_root:
            raise ValueError("This migration belongs to a different active Data Home.")
        target_root = Path(str(record.get("target_root", ""))).expanduser().absolute()
        staging = self._staging_from_record(record, target_root)
        self._validate_relocation_target(target_root, source_roots)
        if not staging.exists() or not staging.is_dir():
            raise ValueError("The recoverable migration staging area is unavailable.")
        record.pop("error", None)
        record["status"] = "copying"
        record["resumed_at_utc"] = self._timestamp()
        self._write_journal(path, record)
        return self._continue_relocation(
            record,
            journal_path=path,
            source_roots=source_roots,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            reuse_staging=True,
        )

    def rollback_migration(self, journal_path: str | Path, *, progress_callback=None, cancel_check=None) -> bool:
        """Remove only a verified app-owned staging directory for an incomplete move."""
        path = Path(journal_path).expanduser().absolute()
        record = self.migration_preview(path)
        if record.get("status") not in {"copying", "failed", "cancelled", "verified", "rollback_cancelled"}:
            return False
        target_root = Path(str(record.get("target_root", ""))).expanduser().absolute()
        staging = self._staging_from_record(record, target_root)
        try:
            if staging.exists():
                if staging.is_symlink() or not staging.is_dir():
                    raise ValueError("Refusing to remove an unsafe migration staging path.")
                self._remove_staging_tree(staging, progress_callback=progress_callback, cancel_check=cancel_check)
            record.pop("error", None)
            record["status"] = "rolled_back"
            record["rolled_back_at_utc"] = self._timestamp()
            self._write_journal(path, record)
            return True
        except Exception as exc:
            record.update({"status": "rollback_cancelled" if self._is_cancelled(cancel_check) else "failed", "error": str(exc)})
            self._write_journal(path, record)
            raise

    def recovery_journals(self) -> tuple[dict[str, object], ...]:
        journal_dir = self.current_root / "support" / "data_home_migrations"
        if not journal_dir.exists():
            return ()
        entries: list[dict[str, object]] = []
        for path in sorted(journal_dir.glob("*.json"), reverse=True):
            entry = self.migration_preview(path)
            entry["journal_path"] = str(path)
            entries.append(entry)
        return tuple(entries)

    def _continue_relocation(
        self,
        record: dict[str, object],
        *,
        journal_path: Path,
        source_roots: tuple[str, ...],
        progress_callback,
        cancel_check,
        reuse_staging: bool = False,
    ) -> DataHomeMigration:
        migration_id = str(record["id"])
        target_root = Path(str(record["target_root"])).expanduser().absolute()
        staging = self._staging_from_record(record, target_root)
        try:
            if reuse_staging:
                self._validate_staging_path(staging, target_root, migration_id)
            else:
                staging.mkdir(parents=True, exist_ok=False)
            manifest = self._copy_managed_tree(
                staging,
                progress_callback=progress_callback,
                cancel_check=cancel_check,
                reuse_existing=reuse_staging,
            )
            atomic_write_text(staging / MARKER_NAME, json.dumps({"app_name": self.app_name, "migration_id": migration_id}) + "\n")
            atomic_write_text(staging / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            valid, failures = self.verify_backup(staging, progress_callback=progress_callback, cancel_check=cancel_check)
            if not valid:
                raise ValueError("Verification failed: " + "; ".join(failures[:5]))
            record.update({"status": "verified", "files": len(manifest["files"]), "bytes": manifest["bytes"]})
            self._write_journal(journal_path, record)
            raise_if_cancelled(cancel_check)
            os.replace(staging, target_root)
            record["status"] = "switched"
            self._write_journal(journal_path, record)
            self._write_active_data_home(target_root, migration_id)
            record["status"] = "complete"
            self._write_journal(journal_path, record)
            self._write_journal(target_root / "support" / "data_home_migrations" / f"{migration_id}.json", record)
            return DataHomeMigration(migration_id, str(self.current_root), str(target_root), str(journal_path), len(manifest["files"]), int(manifest["bytes"]))
        except Exception as exc:
            record.update({"status": "cancelled" if self._is_cancelled(cancel_check) else "failed", "error": str(exc)})
            self._write_journal(journal_path, record)
            raise

    def migration_preview(self, journal_path: str | Path) -> dict[str, object]:
        try:
            return dict(json.loads(Path(journal_path).read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as exc:
            return {"status": "unavailable", "error": str(exc)}

    def _copy_managed_tree(
        self,
        destination: Path,
        *,
        progress_callback=None,
        cancel_check=None,
        reuse_existing: bool = False,
    ) -> dict[str, object]:
        entries = list(self._managed_files(cancel_check=cancel_check))
        manifest_files: list[dict[str, object]] = []
        total_bytes = 0
        for index, source in enumerate(entries):
            raise_if_cancelled(cancel_check)
            relative = source.relative_to(self.current_root)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            reusing_verified_file = reuse_existing and self._same_file(source, target, cancel_check=cancel_check)
            if not reusing_verified_file:
                shutil.copy2(source, target, follow_symlinks=False)
            size = int(target.stat().st_size)
            total_bytes += size
            manifest_files.append({"path": str(relative), "bytes": size, "sha256": self._digest(target, cancel_check=cancel_check)})
            self._progress(progress_callback, index + 1, max(1, len(entries)), f"Copying {relative}")
        return {
            "schema": 1,
            "app_name": self.app_name,
            "created_at_utc": self._timestamp(),
            "files": manifest_files,
            "bytes": total_bytes,
        }

    def _managed_files(self, *, cancel_check=None):
        for name in MANAGED_TOP_LEVEL:
            root = self.current_root / name
            if not root.exists():
                continue
            for path in sorted(root.rglob("*"), key=lambda value: str(value).casefold()):
                raise_if_cancelled(cancel_check)
                if path.is_symlink():
                    raise ValueError(f"Data Home contains a symlink and cannot be migrated safely: {path}")
                if path.is_file():
                    yield path

    def _remove_staging_tree(self, staging: Path, *, progress_callback=None, cancel_check=None) -> None:
        entries = sorted(staging.rglob("*"), key=lambda value: (len(value.parts), str(value)), reverse=True)
        total = max(1, len(entries))
        for index, entry in enumerate(entries):
            raise_if_cancelled(cancel_check)
            if entry.is_symlink() or entry.is_file():
                entry.unlink(missing_ok=True)
            elif entry.is_dir():
                entry.rmdir()
            self._progress(progress_callback, index + 1, total, f"Removing staged {entry.relative_to(staging)}")
        raise_if_cancelled(cancel_check)
        staging.rmdir()

    def _tree_size(self, root: Path, *, cancel_check=None) -> tuple[int, int]:
        bytes_count = 0
        file_count = 0
        if not root.exists():
            return bytes_count, file_count
        for path in root.rglob("*"):
            raise_if_cancelled(cancel_check)
            if path.is_symlink() or not path.is_file():
                continue
            try:
                bytes_count += int(path.stat().st_size)
                file_count += 1
            except OSError:
                continue
        return bytes_count, file_count

    def _validate_relocation_target(self, target: Path, source_roots: tuple[str, ...]) -> None:
        if target == self.current_root:
            raise ValueError("The new Data Home is already active.")
        if target.exists() and any(target.iterdir()):
            raise ValueError("Choose a new, empty Data Home directory.")
        if target.parent == target or target == Path.home().resolve():
            raise ValueError("Refusing an unsafe Data Home target.")
        for root in source_roots:
            source = Path(root).expanduser().absolute()
            if self._contains(source, target) or self._contains(target, source):
                raise ValueError("Data Home must not be inside a selected photo source, or contain one.")

    def _validate_backup_destination(self, destination: Path) -> None:
        if destination.exists() and not destination.is_dir():
            raise ValueError("Backup destination must be a directory.")
        if destination.parent == destination:
            raise ValueError("Refusing to create a backup at a filesystem root.")
        if self._contains(self.current_root, destination):
            raise ValueError("Backup destination must be outside the active Data Home.")
        destination.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _same_file(source: Path, target: Path, *, cancel_check=None) -> bool:
        if not target.is_file() or target.is_symlink():
            return False
        try:
            if source.stat().st_size != target.stat().st_size:
                return False
        except OSError:
            return False
        return DataHomeManager._digest(source, cancel_check=cancel_check) == DataHomeManager._digest(target, cancel_check=cancel_check)

    @staticmethod
    def _staging_from_record(record: dict[str, object], target_root: Path) -> Path:
        migration_id = str(record.get("id", "") or "")
        expected = target_root.parent / f".{target_root.name}.{migration_id}.staging"
        recorded = Path(str(record.get("staging_root", "") or expected)).expanduser().absolute()
        if recorded != expected:
            raise ValueError("Migration journal has an unsafe staging path.")
        return expected

    @staticmethod
    def _validate_staging_path(staging: Path, target_root: Path, migration_id: str) -> None:
        expected = target_root.parent / f".{target_root.name}.{migration_id}.staging"
        if staging != expected or staging.is_symlink() or not staging.is_dir():
            raise ValueError("Migration staging directory is unavailable or unsafe.")

    def _write_active_data_home(self, target: Path, migration_id: str) -> None:
        payload = {"schema": 1, "data_home": str(target), "migration_id": migration_id, "updated_at_utc": self._timestamp()}
        atomic_write_text(self.config_path, json.dumps(payload, indent=2, sort_keys=True) + "\n")

    @staticmethod
    def _contains(parent: Path, child: Path) -> bool:
        try:
            child.resolve(strict=False).relative_to(parent.resolve(strict=False))
            return True
        except ValueError:
            return False

    @staticmethod
    def _safe_relative(candidate: Path, root: Path) -> bool:
        try:
            candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
            return True
        except ValueError:
            return False

    @staticmethod
    def _digest(path: Path, *, cancel_check=None) -> str:
        digest = sha256()
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                raise_if_cancelled(cancel_check)
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _progress(callback, current: int, total: int, text: str) -> None:
        if callback is not None:
            callback(int(current * 100 / max(1, total)), text)

    @staticmethod
    def _is_cancelled(cancel_check) -> bool:
        try:
            return bool(cancel_check and cancel_check())
        except Exception:
            return False

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    @staticmethod
    def _stamp() -> str:
        return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    @staticmethod
    def _write_journal(path: Path, record: dict[str, object]) -> None:
        atomic_write_text(path, json.dumps(record, indent=2, sort_keys=True) + "\n")
