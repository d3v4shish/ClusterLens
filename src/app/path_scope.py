from __future__ import annotations

import ntpath
import os
from pathlib import Path
import re


_WINDOWS_ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")
SQL_LIKE_ESCAPE = "!"


def _uses_windows_path_rules(value: str) -> bool:
    return bool(_WINDOWS_ABSOLUTE_PATH.match(str(value or "").strip()))


def normalize_scoped_path(value: str, *, resolve_symlinks: bool = True) -> str:
    """Return a stable path key for scope comparisons.

    ClusterLens is Linux-first, but imported databases can contain Windows paths.
    Windows-looking values therefore use ``ntpath`` semantics while native paths
    are normalized and, by default, resolved to prevent symlinks escaping a
    selected folder.
    """

    text = str(value or "").strip()
    if not text:
        return ""
    if _uses_windows_path_rules(text):
        return ntpath.normcase(ntpath.normpath(text))
    expanded = Path(text).expanduser()
    if resolve_symlinks:
        try:
            expanded = expanded.resolve(strict=False)
        except (OSError, RuntimeError):
            expanded = Path(os.path.abspath(os.path.normpath(str(expanded))))
    elif not expanded.is_absolute():
        expanded = Path(os.path.abspath(str(expanded)))
    return os.path.normcase(os.path.normpath(str(expanded)))


def path_is_within_scope(candidate: str, root: str, *, resolve_symlinks: bool = True) -> bool:
    """Return whether *candidate* is *root* or a real descendant of it."""

    candidate_text = str(candidate or "").strip()
    root_text = str(root or "").strip()
    if not candidate_text or not root_text:
        return False
    candidate_windows = _uses_windows_path_rules(candidate_text)
    root_windows = _uses_windows_path_rules(root_text)
    if candidate_windows != root_windows:
        return False
    normalized_candidate = normalize_scoped_path(candidate_text, resolve_symlinks=resolve_symlinks)
    normalized_root = normalize_scoped_path(root_text, resolve_symlinks=resolve_symlinks)
    if not normalized_candidate or not normalized_root:
        return False
    path_module = ntpath if candidate_windows else os.path
    try:
        return path_module.commonpath([normalized_candidate, normalized_root]) == normalized_root
    except ValueError:
        return False


def escape_sql_like(value: str, *, escape: str = SQL_LIKE_ESCAPE) -> str:
    """Escape a literal value for a SQLite ``LIKE ... ESCAPE`` parameter."""

    text = str(value or "")
    return text.replace(escape, escape + escape).replace("%", escape + "%").replace("_", escape + "_")


def folder_scope_sql(column: str, folder: str) -> tuple[str, list[str]]:
    """Build an exact-or-descendant SQLite predicate for a stored path column."""

    target = str(folder or "").strip().rstrip("/\\")
    if not target:
        return "1=1", []
    escaped = escape_sql_like(target)
    clause = (
        f"({column} = ? "
        f"OR {column} LIKE ? ESCAPE '{SQL_LIKE_ESCAPE}' "
        f"OR {column} LIKE ? ESCAPE '{SQL_LIKE_ESCAPE}')"
    )
    return clause, [target, f"{escaped}/%", f"{escaped}\\%"]
