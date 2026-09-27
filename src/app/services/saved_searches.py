from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock, RLock
from uuid import uuid4

from infra.atomic_io import atomic_write_text
from infra.settings import get_settings


_LOCKS_GUARD = Lock()
_PATH_LOCKS: dict[str, RLock] = {}


def _path_lock(path: Path) -> RLock:
    key = str(path.expanduser().resolve(strict=False))
    with _LOCKS_GUARD:
        return _PATH_LOCKS.setdefault(key, RLock())


@dataclass(frozen=True)
class SavedSearch:
    search_id: str
    name: str
    search_type: str
    payload: dict[str, object]
    created_at: str
    updated_at: str


class SavedSearchService:
    def __init__(self, file_path: str | Path | None = None) -> None:
        self.file_path = Path(file_path) if file_path is not None else get_settings().base_dir / "saved_searches.json"
        self._lock = _path_lock(self.file_path)

    def list_searches(self, search_type: str | None = None) -> list[SavedSearch]:
        with self._lock:
            records = self._load()
            if search_type:
                wanted = str(search_type).strip()
                records = [record for record in records if record.search_type == wanted]
            return sorted(records, key=lambda record: (record.name.lower(), record.search_type, record.created_at))

    def get_search(self, search_id: str) -> SavedSearch | None:
        wanted = str(search_id or "").strip()
        with self._lock:
            for record in self._load():
                if record.search_id == wanted:
                    return record
        return None

    def save_search(self, name: str, search_type: str, payload: dict[str, object]) -> SavedSearch:
        clean_name = str(name or "").strip()
        clean_type = str(search_type or "").strip()
        if not clean_name:
            raise ValueError("Saved search name is required.")
        if not clean_type:
            raise ValueError("Saved search type is required.")
        if not isinstance(payload, dict):
            raise ValueError("Saved search payload must be a dictionary.")
        now = _utc_timestamp()
        record = SavedSearch(
            search_id=uuid4().hex,
            name=clean_name,
            search_type=clean_type,
            payload=_json_dict(payload),
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            records = self._load()
            records.append(record)
            self._save(records)
        return record

    def rename_search(self, search_id: str, new_name: str) -> SavedSearch:
        wanted = str(search_id or "").strip()
        clean_name = str(new_name or "").strip()
        if not clean_name:
            raise ValueError("Saved search name is required.")
        with self._lock:
            records = self._load()
            updated: SavedSearch | None = None
            now = _utc_timestamp()
            rewritten: list[SavedSearch] = []
            for record in records:
                if record.search_id == wanted:
                    updated = SavedSearch(
                        search_id=record.search_id,
                        name=clean_name,
                        search_type=record.search_type,
                        payload=dict(record.payload),
                        created_at=record.created_at,
                        updated_at=now,
                    )
                    rewritten.append(updated)
                else:
                    rewritten.append(record)
            if updated is None:
                raise KeyError(f"Saved search not found: {wanted}")
            self._save(rewritten)
        return updated

    def delete_search(self, search_id: str) -> bool:
        wanted = str(search_id or "").strip()
        with self._lock:
            records = self._load()
            kept = [record for record in records if record.search_id != wanted]
            if len(kept) == len(records):
                return False
            self._save(kept)
            return True

    def _load(self) -> list[SavedSearch]:
        if not self.file_path.exists():
            return []
        try:
            raw = json.loads(self.file_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Saved-search file is unreadable: {exc}") from exc
        if not isinstance(raw, dict):
            raise ValueError("Saved-search file must contain a JSON object.")
        items = raw.get("searches", []) if isinstance(raw, dict) else []
        records: list[SavedSearch] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            search_id = str(item.get("search_id") or item.get("id") or "").strip()
            name = str(item.get("name") or "").strip()
            search_type = str(item.get("search_type") or item.get("type") or "").strip()
            if not search_id or not name or not search_type:
                continue
            payload = item.get("payload", {})
            records.append(
                SavedSearch(
                    search_id=search_id,
                    name=name,
                    search_type=search_type,
                    payload=_json_dict(payload if isinstance(payload, dict) else {}),
                    created_at=str(item.get("created_at") or ""),
                    updated_at=str(item.get("updated_at") or ""),
                )
            )
        return records

    def _save(self, records: list[SavedSearch]) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "searches": [
                {
                    "search_id": record.search_id,
                    "name": record.name,
                    "search_type": record.search_type,
                    "payload": _json_dict(record.payload),
                    "created_at": record.created_at,
                    "updated_at": record.updated_at,
                }
                for record in records
            ],
        }
        atomic_write_text(self.file_path, json.dumps(payload, indent=2, sort_keys=True))


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_dict(payload: dict[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(dict(payload), sort_keys=True))
