from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from uuid import uuid4

from infra.atomic_io import atomic_write_text


MAX_PROTOCOL_LINE_CHARS = 2 * 1024 * 1024
MAX_PROTOCOL_BUFFER_CHARS = 4 * 1024 * 1024
MAX_DIAGNOSTIC_CHARS = 256 * 1024
MAX_RESULT_BYTES = 512 * 1024 * 1024
TRUNCATION_MARKER = "\n[ClusterLens diagnostic output truncated; newest data retained]\n"

_LOG_WRITER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="clusterlens-process-logs")


def prepare_protocol_line(event_type: str, payload: dict[str, object]) -> bytes:
    """Serialize a daemon request outside the Qt thread."""

    return (json.dumps({"type": str(event_type), "payload": payload}) + "\n").encode("utf-8")


def read_result_payload(path: Path, *, allowed_directory: Path | None = None) -> dict[str, object]:
    """Read and validate a file-backed worker result outside the Qt thread."""

    path = Path(path).resolve()
    if allowed_directory is not None:
        allowed = Path(allowed_directory).resolve()
        if path.parent != allowed:
            raise ValueError("Worker result path is outside the managed handoff directory.")
    size = path.stat().st_size
    if size > MAX_RESULT_BYTES:
        raise ValueError(f"Worker result exceeds the {MAX_RESULT_BYTES}-byte safety limit.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Worker result payload must be a JSON object.")
    return value


def write_result_payload(directory: Path, prefix: str, payload: dict[str, object]) -> Path:
    """Persist a large worker result and return its handoff path."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}_{uuid4().hex}.json"
    atomic_write_text(path, json.dumps(payload))
    return path


def bounded_diagnostic_append(current: str, text: str) -> str:
    combined = str(current or "") + str(text or "")
    if len(combined) <= MAX_DIAGNOSTIC_CHARS:
        return combined
    keep = max(0, MAX_DIAGNOSTIC_CHARS - len(TRUNCATION_MARKER))
    return TRUNCATION_MARKER + combined[-keep:]


def write_process_logs_async(
    stdout_path: Path | None,
    stdout_text: str,
    stderr_path: Path | None,
    stderr_text: str,
):
    """Persist bounded diagnostic snapshots without filesystem work on Qt."""

    snapshots = (
        (stdout_path, str(stdout_text or "")),
        (stderr_path, str(stderr_text or "")),
    )

    def _write() -> None:
        for path, text in snapshots:
            if path is None:
                continue
            try:
                # Runtime teardown may remove a disposable log directory after
                # the terminal signal. Diagnostics must never recreate it.
                if not path.parent.is_dir():
                    continue
                path.write_text(text, encoding="utf-8")
            except OSError:
                continue

    return _LOG_WRITER.submit(_write)


def cleanup_paths_async(*paths: Path | None):
    targets = tuple(Path(path) for path in paths if path is not None)

    def _remove() -> None:
        for path in targets:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue

    return _LOG_WRITER.submit(_remove)
