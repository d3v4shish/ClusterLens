from __future__ import annotations

import os
import tempfile
from collections.abc import Callable
from pathlib import Path


def atomic_write_bytes(path: str | Path, payload: bytes) -> None:
    _atomic_replace(Path(path), lambda handle: handle.write(payload), binary=True)


def atomic_write_text(path: str | Path, payload: str, *, encoding: str = "utf-8") -> None:
    _atomic_replace(
        Path(path),
        lambda handle: handle.write(payload),
        binary=False,
        encoding=encoding,
    )


def atomic_write_with(path: str | Path, writer: Callable[[Path], None]) -> None:
    """Write a path-based format to a sibling temp file, then atomically publish it."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".partial", dir=target.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        writer(temporary)
        _sync_path(temporary)
        _atomic_write_checkpoint(target, "before_replace")
        os.replace(temporary, target)
        _atomic_write_checkpoint(target, "after_replace")
        _sync_directory(target.parent)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _atomic_replace(
    target: Path,
    writer: Callable[[object], object],
    *,
    binary: bool,
    encoding: str = "utf-8",
) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "wb" if binary else "w"
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".partial", dir=target.parent)
    temporary = Path(temporary_name)
    try:
        open_kwargs = {} if binary else {"encoding": encoding}
        with os.fdopen(fd, mode, **open_kwargs) as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_write_checkpoint(target, "before_replace")
        os.replace(temporary, target)
        _atomic_write_checkpoint(target, "after_replace")
        _sync_directory(target.parent)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _atomic_write_checkpoint(_target: Path, _phase: str) -> None:
    """Deterministic fault-injection seam around the atomic commit point."""

    return


def _sync_path(path: Path) -> None:
    try:
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    except OSError:
        pass


def _sync_directory(path: Path) -> None:
    """Persist the directory entry where the platform supports directory fsync."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except (AttributeError, OSError):
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)
