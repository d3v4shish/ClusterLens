from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from .runtime_support import RuntimeLayout


def export_support_bundle(
    layout: RuntimeLayout,
    *,
    bundle_name: str,
    metadata: dict[str, object],
    redact: bool = True,
) -> Path:
    bundle_path = layout.support_dir / f"{bundle_name}.zip"
    replacements = _redaction_replacements(layout)
    with ZipFile(bundle_path, "w", compression=ZIP_DEFLATED) as archive:
        safe_metadata = _redact_value(metadata, replacements) if redact else metadata
        archive.writestr("metadata.json", json.dumps(safe_metadata, indent=2, sort_keys=True, default=str))
        for root in (layout.logs_dir, layout.crash_dir):
            if not root.exists():
                continue
            for file_path in root.rglob("*"):
                if file_path.is_file():
                    archive_name = str(file_path.relative_to(layout.root)).replace("\\", "/")
                    if redact and _looks_text_file(file_path):
                        text = file_path.read_text(encoding="utf-8", errors="replace")
                        archive.writestr(archive_name, _redact_text(text, replacements))
                    else:
                        archive.write(file_path, archive_name)
    return bundle_path


def _redaction_replacements(layout: RuntimeLayout) -> list[tuple[str, str]]:
    candidates = [
        (layout.root, "<runtime-root>"),
        (layout.cache_dir, "<cache-dir>"),
        (layout.logs_dir, "<logs-dir>"),
        (layout.crash_dir, "<crash-dir>"),
        (Path.home(), "<user-home>"),
    ]
    replacements: list[tuple[str, str]] = []
    for path, marker in candidates:
        try:
            resolved = str(path.resolve())
        except OSError:
            resolved = str(path.absolute())
        replacements.append((resolved, marker))
        replacements.append((resolved.replace("\\", "/"), marker))
    username = os.environ.get("USERNAME")
    if username:
        replacements.append((username, "<user>"))
    replacements.sort(key=lambda item: len(item[0]), reverse=True)
    return replacements


def _redact_value(value: Any, replacements: list[tuple[str, str]]) -> Any:
    if isinstance(value, dict):
        return {str(key): _redact_value(item, replacements) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_value(item, replacements) for item in value]
    if isinstance(value, tuple):
        return [_redact_value(item, replacements) for item in value]
    if isinstance(value, str):
        return _redact_text(value, replacements)
    return value


def _redact_text(text: str, replacements: list[tuple[str, str]]) -> str:
    redacted = str(text)
    for needle, marker in replacements:
        if needle:
            redacted = redacted.replace(needle, marker)
    return redacted


def _looks_text_file(path: Path) -> bool:
    return path.suffix.lower() in {"", ".log", ".txt", ".json", ".md", ".csv", ".yaml", ".yml"}
