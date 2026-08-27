from __future__ import annotations

import json
import os
from pathlib import Path


RECENT_FOLDERS_SETTING_KEY = "workspace/recent_folders_v1"
RECENT_FOLDERS_VERSION = 1


class RecentFolderHistory:
    """Small QSettings-backed MRU shared by the folder and Faces panes."""

    def __init__(self, store, *, key: str = RECENT_FOLDERS_SETTING_KEY, limit: int = 10) -> None:
        self.store = store
        self.key = str(key)
        self.limit = max(1, int(limit))
        self._paths = self._load()
        # Rewriting the small payload also prunes missing folders and upgrades
        # malformed/older values without needing a separate migration.
        self._save()

    @staticmethod
    def _canonical_path(path: str) -> str:
        value = str(path or "").strip()
        if not value:
            return ""
        try:
            return str(Path(value).expanduser().resolve(strict=False))
        except (OSError, RuntimeError):
            return str(Path(value).expanduser().absolute())

    @staticmethod
    def _comparison_key(path: str) -> str:
        return os.path.normcase(os.path.normpath(str(path or "")))

    @classmethod
    def _valid_directory(cls, path: str) -> str:
        canonical = cls._canonical_path(path)
        return canonical if canonical and Path(canonical).is_dir() else ""

    def _load(self) -> list[str]:
        try:
            raw = self.store.value(self.key, "")
        except Exception:
            raw = ""
        try:
            payload = json.loads(str(raw or "")) if raw else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        source_paths = payload.get("paths", []) if isinstance(payload, dict) else []
        if not isinstance(source_paths, list):
            source_paths = []
        valid: list[str] = []
        seen: set[str] = set()
        for candidate in source_paths:
            path = self._valid_directory(str(candidate or ""))
            key = self._comparison_key(path)
            if not path or key in seen:
                continue
            seen.add(key)
            valid.append(path)
            if len(valid) >= self.limit:
                break
        return valid

    def _save(self) -> None:
        payload = {"version": RECENT_FOLDERS_VERSION, "paths": list(self._paths)}
        self.store.setValue(self.key, json.dumps(payload, separators=(",", ":")))

    def paths(self) -> list[str]:
        return list(self._paths)

    def record(self, path: str) -> bool:
        canonical = self._valid_directory(path)
        if not canonical:
            return False
        key = self._comparison_key(canonical)
        updated = [canonical]
        updated.extend(item for item in self._paths if self._comparison_key(item) != key)
        updated = updated[: self.limit]
        if updated == self._paths:
            return False
        self._paths = updated
        self._save()
        return True

    def remove(self, path: str) -> bool:
        key = self._comparison_key(self._canonical_path(path))
        updated = [item for item in self._paths if self._comparison_key(item) != key]
        if updated == self._paths:
            return False
        self._paths = updated
        self._save()
        return True

    def clear(self) -> bool:
        if not self._paths:
            return False
        self._paths = []
        self._save()
        return True
