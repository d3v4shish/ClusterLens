from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


SOURCE_FILTER_IGNORE_THUMBNAILS = "source_filters/ignore_thumbnail_like"
SOURCE_FILTER_MIN_WIDTH = "source_filters/min_width"
SOURCE_FILTER_MIN_HEIGHT = "source_filters/min_height"
SOURCE_FILTER_MIN_FILE_SIZE_BYTES = "source_filters/min_file_size_bytes"

_THUMBNAIL_TOKENS = frozenset({"thumb", "thumbs", "thumbnail", "thumbnails"})
_THUMBNAIL_TOKEN_SPLIT = re.compile(r"[^a-z0-9]+")
_COMPACT_THUMBNAIL_TOKEN = re.compile(r"^(?:thumb|thumbs|thumbnail|thumbnails)\d+(?:x\d+)?$")


@dataclass(frozen=True)
class SourceAdmissionDecision:
    """The result of applying the global, source-read-only admission rules."""

    admitted: bool
    reason: str = ""


@dataclass(frozen=True)
class SourceAdmissionPolicy:
    """Rules that decide whether a discovered image may enter a workspace.

    The policy intentionally never edits or deletes source files.  It is used
    at scan boundaries, before Gallery, clustering, Faces, similarity, or the
    managed Library catalog receive a candidate path.
    """

    ignore_thumbnail_like: bool = False
    minimum_width: int = 0
    minimum_height: int = 0
    minimum_file_size_bytes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "minimum_width", max(0, int(self.minimum_width)))
        object.__setattr__(self, "minimum_height", max(0, int(self.minimum_height)))
        object.__setattr__(self, "minimum_file_size_bytes", max(0, int(self.minimum_file_size_bytes)))

    @property
    def needs_dimensions(self) -> bool:
        return self.minimum_width > 0 or self.minimum_height > 0

    @classmethod
    def from_global_settings(cls) -> "SourceAdmissionPolicy":
        """Load the persisted production policy without coupling callers to Qt UI.

        QSettings is opened only once for each discovery/catalog operation;
        individual files never read settings.  The explicit organization and
        application names match the production settings store and keep CLI and
        worker processes on the same policy.
        """

        try:
            from PyQt6.QtCore import QSettings

            from infra.settings import get_production_settings_registry

            return cls.from_settings_store(QSettings("ClusterLens", "ClusterLens"))
        except Exception:
            # Source discovery remains available in lightweight/CLI test
            # environments even when Qt settings are not initialized.
            return cls()

    @classmethod
    def from_settings_store(cls, store: object) -> "SourceAdmissionPolicy":
        """Read a policy from an already-owned production settings store."""

        from infra.settings import get_production_settings_registry

        registry = get_production_settings_registry()
        return cls(
            ignore_thumbnail_like=bool(registry.get(store, SOURCE_FILTER_IGNORE_THUMBNAILS, False)),
            minimum_width=int(registry.get(store, SOURCE_FILTER_MIN_WIDTH, 0)),
            minimum_height=int(registry.get(store, SOURCE_FILTER_MIN_HEIGHT, 0)),
            minimum_file_size_bytes=int(registry.get(store, SOURCE_FILTER_MIN_FILE_SIZE_BYTES, 0)),
        )

    def decide(self, path: str | Path, *, stat_result: os.stat_result | None = None) -> SourceAdmissionDecision:
        candidate = Path(path)
        if self.ignore_thumbnail_like and is_thumbnail_like_path(candidate):
            return SourceAdmissionDecision(False, "thumbnail-like name")

        try:
            stat = stat_result if stat_result is not None else candidate.stat()
        except OSError:
            return SourceAdmissionDecision(False, "unreadable file")
        if int(stat.st_size) < self.minimum_file_size_bytes:
            return SourceAdmissionDecision(False, "below minimum file size")

        if not self.needs_dimensions:
            return SourceAdmissionDecision(True)
        try:
            with Image.open(candidate) as image:
                width, height = image.size
        except Exception:
            return SourceAdmissionDecision(False, "unreadable image dimensions")
        if width < self.minimum_width or height < self.minimum_height:
            return SourceAdmissionDecision(False, "below minimum dimensions")
        return SourceAdmissionDecision(True)


def is_thumbnail_like_path(path: str | Path) -> bool:
    """Return true for deliberate thumbnail names without broad substring matches.

    A photo such as ``thumbprint.jpg`` remains eligible.  Conventional names
    such as ``IMG_1234_thumb.jpg`` and files placed in a ``.thumbnails``
    directory are excluded when the user enables the rule.
    """

    candidate = Path(path)
    for part in candidate.parts[:-1]:
        normalized = str(part).casefold().lstrip(".")
        if normalized in _THUMBNAIL_TOKENS:
            return True
    tokens = tuple(token for token in _THUMBNAIL_TOKEN_SPLIT.split(candidate.stem.casefold()) if token)
    return any(
        token in _THUMBNAIL_TOKENS or _COMPACT_THUMBNAIL_TOKEN.fullmatch(token) is not None
        for token in tokens
    )
