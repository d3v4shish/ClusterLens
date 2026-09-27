from __future__ import annotations

import ntpath
import os
from pathlib import Path
import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable


_WINDOWS_ABSOLUTE_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")
SQL_LIKE_ESCAPE = "!"


@dataclass(frozen=True)
class PathScope:
    """A canonical, non-overlapping set of local folder roots.

    A scope intentionally represents folders rather than an eagerly discovered
    list of files.  This keeps queries cheap and makes every expensive source
    walk an explicit, cancellable background operation.
    """

    roots: tuple[str, ...] = ()

    @classmethod
    def from_paths(
        cls,
        paths: Iterable[str] | None,
        *,
        resolve_symlinks: bool = True,
    ) -> "PathScope":
        candidates = {
            normalize_scoped_path(str(path), resolve_symlinks=resolve_symlinks)
            for path in (paths or ())
            if str(path or "").strip()
        }
        # Process ancestors first.  An active parent already covers a child,
        # so keeping both would cause duplicate source walks and ambiguous UI.
        ordered = sorted(candidates, key=lambda value: (value.count("/") + value.count("\\"), len(value), value.casefold()))
        roots: list[str] = []
        for candidate in ordered:
            if not candidate:
                continue
            if any(path_is_within_scope(candidate, root) for root in roots):
                continue
            roots.append(candidate)
        return cls(tuple(roots))

    @property
    def is_empty(self) -> bool:
        return not self.roots

    @property
    def primary_root(self) -> str:
        return self.roots[0] if self.roots else ""

    @property
    def signature(self) -> str:
        payload = "\n".join(sorted(self.roots, key=str.casefold)).encode("utf-8")
        return sha256(payload).hexdigest()

    def contains(self, candidate: str) -> bool:
        return any(path_is_within_scope(candidate, root) for root in self.roots)

    def summary(self) -> str:
        count = len(self.roots)
        if count == 0:
            return "No active roots"
        if count == 1:
            return self.roots[0]
        return f"{count} active roots"


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
    return normalized_path_is_within_scope(normalized_candidate, normalized_root)


def normalized_path_is_within_scope(candidate: str, root: str) -> bool:
    """Compare two already canonical local path keys without filesystem work."""

    normalized_candidate = str(candidate or "").strip()
    normalized_root = str(root or "").strip()
    if not normalized_candidate or not normalized_root:
        return False
    candidate_windows = _uses_windows_path_rules(normalized_candidate)
    root_windows = _uses_windows_path_rules(normalized_root)
    if candidate_windows != root_windows:
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


def roots_scope_sql(column: str, roots: Iterable[str] | PathScope | None) -> tuple[str, list[str]]:
    """Build a safe exact-or-descendant predicate for multiple root paths.

    An empty active scope intentionally matches nothing.  A global view must
    omit this predicate explicitly, which prevents an unavailable scope from
    silently becoming an all-files query.
    """

    scope = roots if isinstance(roots, PathScope) else PathScope.from_paths(roots)
    if scope.is_empty:
        return "0=1", []
    clauses: list[str] = []
    args: list[str] = []
    for root in scope.roots:
        clause, values = folder_scope_sql(column, root)
        clauses.append(clause)
        args.extend(values)
    return f"({' OR '.join(clauses)})", args
