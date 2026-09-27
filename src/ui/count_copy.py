from __future__ import annotations


def _plural(singular: str) -> str:
    return f"{singular[:-1]}ies" if singular.endswith("y") else f"{singular}s"


def counted(value: int, singular: str) -> str:
    count = max(0, int(value))
    return f"{count:,} {singular if count == 1 else _plural(singular)}"


def showing(value: int, singular: str, *, total: int | None = None, matching: bool = False) -> str:
    shown = max(0, int(value))
    if total is None:
        return f"Showing {counted(shown, singular)}"
    available = max(0, int(total))
    adjective = " matching" if matching else ""
    unit = singular if available == 1 else _plural(singular)
    return f"Showing {shown:,} of {available:,}{adjective} {unit}"


def loaded_and_showing(loaded: int, visible: int, singular: str) -> str:
    return f"Loaded {counted(loaded, singular)} · Showing {counted(visible, singular)}"


def loading(completed: int, total: int, singular: str, *, subject: str = "metadata") -> str:
    available = max(0, int(total))
    unit = singular if available == 1 else _plural(singular)
    return f"Loading {subject} {max(0, int(completed)):,}/{available:,} {unit}"


def thumbnail_progress(completed: int, total: int, *, done: bool = False, qualifier: str = "") -> str:
    prefix = "Thumbnails ready" if done else "Loading thumbnails"
    suffix = f" {str(qualifier).strip()}" if str(qualifier).strip() else ""
    return f"{prefix} {max(0, int(completed)):,}/{max(0, int(total)):,}{suffix}"
