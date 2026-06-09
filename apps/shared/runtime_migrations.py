from __future__ import annotations

import json
import shutil
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
        self.state_path = layout.root / "runtime_migrations.json"

    def run(self) -> RuntimeMigrationResult:
        state = self._read_state()
        previous_version = int(state.get("schema_version") or 0)
        actions: list[str] = []
        failures: list[str] = []
        if previous_version < 1:
            self._ensure_standard_directories(actions, failures)
            self._backup_mutable_databases(actions, failures)
        self._write_state(previous_version, actions, failures)
        return RuntimeMigrationResult(
            previous_version=previous_version,
            current_version=RUNTIME_SCHEMA_VERSION,
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
        # Tags are user-authored/non-rebuildable. Embeddings are rebuildable and
        # can be very large, so migration does not duplicate them at startup.
        for db_name in ("image_tags.sqlite3",):
            db_path = self.layout.cache_dir / db_name
            if not db_path.exists():
                continue
            try:
                backup_dir.mkdir(parents=True, exist_ok=True)
                backup_path = backup_dir / db_name
                if not backup_path.exists():
                    shutil.copy2(db_path, backup_path)
                    actions.append(f"backup:{db_name}")
            except OSError as exc:
                failures.append(f"{db_name}: {exc}")

    def _read_state(self) -> dict[str, object]:
        if not self.state_path.exists():
            return {}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    def _write_state(self, previous_version: int, actions: list[str], failures: list[str]) -> None:
        payload = {
            "schema_version": RUNTIME_SCHEMA_VERSION,
            "previous_version": previous_version,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "actions": actions,
            "failures": failures,
        }
        self.state_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
