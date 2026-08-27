from __future__ import annotations

import json
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .runtime_support import RuntimeLayout


RUNTIME_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class RuntimeMigrationResult:
    previous_version: int
    current_version: int
    actions: tuple[str, ...]
    failures: tuple[str, ...]


class RuntimeMigrationService:
    def __init__(self, layout: RuntimeLayout) -> None:
        self.layout = layout
        self.state_db_path = layout.root / "runtime_migrations.sqlite3"
        self.state_path = layout.root / "runtime_migrations.json"

    def run(self) -> RuntimeMigrationResult:
        previous_version = self._read_schema_version()
        actions: list[str] = []
        failures: list[str] = []
        if previous_version < 1:
            self._ensure_standard_directories(actions, failures)
            if not failures:
                self._backup_mutable_databases(actions, failures)
        current_version = RUNTIME_SCHEMA_VERSION if not failures else previous_version
        self._write_state(previous_version, current_version, actions, failures)
        return RuntimeMigrationResult(
            previous_version=previous_version,
            current_version=current_version,
            actions=tuple(actions),
            failures=tuple(failures),
        )

    def _ensure_standard_directories(self, actions: list[str], failures: list[str]) -> None:
        for path in (
            self.layout.logs_dir,
            self.layout.cache_dir,
            self.layout.crash_dir,
            self.layout.benchmarks_dir,
            self.layout.model_assets_dir,
            self.layout.support_dir,
            self.layout.cache_dir / "tmp",
        ):
            try:
                path.mkdir(parents=True, exist_ok=True)
                actions.append(f"ensured:{path.relative_to(self.layout.root)}")
            except OSError as exc:
                failures.append(f"{path}: {exc}")

    def _backup_mutable_databases(self, actions: list[str], failures: list[str]) -> None:
        backup_dir = self.layout.support_dir / "migration_backups" / f"v{RUNTIME_SCHEMA_VERSION}"
        db_paths = [self.layout.cache_dir / "image_tags.sqlite3"]
        db_paths.extend(sorted(self.layout.cache_dir.glob("face_search*.db")))
        # Tags and face identity/label databases are user-authored and
        # non-rebuildable. Embedding indexes are rebuildable and may be large,
        # so migration does not duplicate them at startup.
        for db_path in db_paths:
            if not db_path.exists():
                continue
            try:
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup_path = backup_dir / db_path.name
                if not backup_path.exists():
                    shutil.copy2(db_path, backup_path)
                    actions.append(f"backup:{db_path.name}")
            except OSError as exc:
                failures.append(f"{db_path.name}: {exc}")

    def _read_schema_version(self) -> int:
        sqlite_version = self._read_sqlite_schema_version()
        if sqlite_version is not None:
            return sqlite_version
        state = self._read_legacy_json_state()
        return int(state.get("schema_version") or 0)

    def _read_sqlite_schema_version(self) -> int | None:
        if not self.state_db_path.exists():
            return None
        try:
            with sqlite3.connect(self.state_db_path) as conn:
                self._ensure_state_schema(conn)
                row = conn.execute(
                    "select max(version) from schema_migrations where succeeded = 1"
                ).fetchone()
        except sqlite3.Error:
            return None
        value = row[0] if row else None
        return int(value or 0)

    def _read_legacy_json_state(self) -> dict[str, object]:
        if not self.state_path.exists():
            return {}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    def _write_state(self, previous_version: int, current_version: int, actions: list[str], failures: list[str]) -> None:
        self._write_sqlite_state(previous_version, current_version, actions, failures)
        payload = {
            "schema_version": current_version,
            "previous_version": previous_version,
            "target_version": RUNTIME_SCHEMA_VERSION,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "actions": actions,
            "failures": failures,
        }
        tmp_path = self.state_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.replace(self.state_path)

    def _write_sqlite_state(self, previous_version: int, current_version: int, actions: list[str], failures: list[str]) -> None:
        self.layout.root.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.state_db_path) as conn:
            self._ensure_state_schema(conn)
            conn.commit()
            conn.execute("pragma synchronous=FULL")
            succeeded = int(not failures)
            conn.execute("begin immediate")
            try:
                conn.execute(
                    """
                    insert into schema_migrations
                    (version, previous_version, target_version, succeeded, updated_at_utc, actions_json, failures_json)
                    values (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        current_version if succeeded else previous_version,
                        previous_version,
                        RUNTIME_SCHEMA_VERSION,
                        succeeded,
                        datetime.now(timezone.utc).isoformat(),
                        json.dumps(actions, sort_keys=True),
                        json.dumps(failures, sort_keys=True),
                    ),
                )
                conn.commit()
            except Exception:
                conn.rollback()
                raise

    @staticmethod
    def _ensure_state_schema(conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            create table if not exists schema_migrations (
                id integer primary key autoincrement,
                version integer not null,
                previous_version integer not null,
                target_version integer not null,
                succeeded integer not null,
                updated_at_utc text not null,
                actions_json text not null,
                failures_json text not null
            )
            """
        )
