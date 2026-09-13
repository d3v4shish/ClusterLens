"""Validate the deterministic synthetic photo fixture used for release evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class ReleaseFixtureValidationError(ValueError):
    """The requested fixture no longer matches its immutable manifest."""


def verified_photo_fixture_metadata(photo_folder: str | Path) -> dict[str, object] | None:
    """Return fixture provenance, or ``None`` for an ordinary user folder.

    A sibling manifest opts a folder into strict validation. This makes the
    release path reproducible without imposing release-fixture rules on normal
    local photo browsing.
    """
    folder = Path(photo_folder).expanduser().resolve()
    manifest_path = folder.parent / "fixture_manifest.json"
    if not manifest_path.is_file():
        return None
    manifest = _read_verified_manifest(manifest_path)
    photos = manifest.get("photos")
    if not isinstance(photos, dict):
        raise ReleaseFixtureValidationError("fixture manifest does not describe photos")
    expected_root = (manifest_path.parent / str(photos.get("root") or "")).resolve()
    if folder != expected_root:
        raise ReleaseFixtureValidationError(
            f"fixture manifest photo root is {expected_root}, not requested folder {folder}"
        )
    if bool(manifest.get("private_source_photos", True)):
        raise ReleaseFixtureValidationError("release fixture must not contain private source photos")
    expected_count = int(photos.get("count") or 0)
    expected_digest = str(photos.get("aggregate_sha256") or "")
    actual_count, actual_digest = _photo_aggregate(folder)
    if actual_count != expected_count:
        raise ReleaseFixtureValidationError(
            f"fixture photo count mismatch: expected {expected_count}, found {actual_count}"
        )
    if not expected_digest or actual_digest != expected_digest:
        raise ReleaseFixtureValidationError("fixture photo aggregate checksum does not match its manifest")
    return {
        "fixture_manifest_path": str(manifest_path),
        "fixture_manifest_sha256": str(manifest["manifest_sha256"]),
        "fixture_photo_count": actual_count,
        "fixture_photo_aggregate_sha256": actual_digest,
        "fixture_private_source_photos": False,
    }


def _read_verified_manifest(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseFixtureValidationError(f"could not read fixture manifest: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReleaseFixtureValidationError("fixture manifest must be a JSON object")
    expected = str(payload.get("manifest_sha256") or "")
    digest_source = dict(payload)
    digest_source.pop("manifest_sha256", None)
    actual = hashlib.sha256(
        json.dumps(digest_source, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if not expected or expected != actual:
        raise ReleaseFixtureValidationError("fixture manifest checksum does not match its contents")
    return payload


def _photo_aggregate(photo_root: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    # Generator order is the globally unique synthetic filename, not directory
    # traversal order (files are distributed among group directories).
    paths = sorted((item for item in photo_root.rglob("*") if item.is_file()), key=lambda item: item.name)
    for path in paths:
        digest.update(path.relative_to(photo_root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(_sha256_file(path).encode("ascii"))
        digest.update(b"\n")
    return len(paths), digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
