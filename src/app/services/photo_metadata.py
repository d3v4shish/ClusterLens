from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import ExifTags, Image

from infra.atomic_io import atomic_write_text

if TYPE_CHECKING:
    from .face_search import FaceDetectionService


@dataclass(frozen=True)
class PhotoMetadata:
    image_path: str
    width: int
    height: int
    file_size: int
    modified_at: str
    camera: str
    exif: dict[str, str] = field(default_factory=dict)
    hashes: dict[str, str] = field(default_factory=dict)
    face_boxes: list[dict[str, object]] = field(default_factory=list)
    context: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class PhotoEditDraft:
    """User-curated metadata kept separate from a source image by default."""

    title: str = ""
    description: str = ""
    rating: int = 0
    tags: tuple[str, ...] = ()
    creator: str = ""
    copyright: str = ""
    captured_at: str = ""
    location: str = ""
    custom_fields: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: object) -> "PhotoEditDraft":
        values = payload if isinstance(payload, dict) else {}
        tags = tuple(
            dict.fromkeys(
                str(value).strip()
                for value in list(values.get("tags", []) or [])
                if str(value).strip()
            )
        )
        custom = {
            str(key).strip(): str(value).strip()
            for key, value in dict(values.get("custom_fields", {}) or {}).items()
            if str(key).strip() and str(value).strip()
        }
        return cls(
            title=str(values.get("title", "") or "").strip(),
            description=str(values.get("description", "") or "").strip(),
            rating=max(0, min(5, int(values.get("rating", 0) or 0))),
            tags=tags,
            creator=str(values.get("creator", "") or "").strip(),
            copyright=str(values.get("copyright", "") or "").strip(),
            captured_at=str(values.get("captured_at", "") or "").strip(),
            location=str(values.get("location", "") or "").strip(),
            custom_fields=custom,
        )

    def as_payload(self) -> dict[str, object]:
        return {
            "title": self.title,
            "description": self.description,
            "rating": int(self.rating),
            "tags": list(self.tags),
            "creator": self.creator,
            "copyright": self.copyright,
            "captured_at": self.captured_at,
            "location": self.location,
            "custom_fields": dict(sorted(self.custom_fields.items())),
        }


class PhotoEditService:
    """Staged metadata edits with an explicit, separately requested embed step."""

    def __init__(self, metadata_sidecars: "MetadataSidecarService | None" = None) -> None:
        # Do not construct MetadataSidecarService by default: it also owns a
        # tag service whose database setup belongs in an explicit worker path.
        self.metadata_sidecars = metadata_sidecars

    def _sidecar_path(self, image_path: str | Path) -> Path:
        if self.metadata_sidecars is not None:
            return self.metadata_sidecars.sidecar_path_for_image(image_path)
        image = Path(image_path)
        return image.parent / f"{image.name}.clusterlens.json"

    def load_draft(self, image_path: str | Path) -> PhotoEditDraft:
        sidecar = self._sidecar_path(image_path)
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
        except (OSError, ValueError, TypeError):
            payload = {}
        values = dict(payload) if isinstance(payload, dict) else {}
        try:
            return PhotoEditDraft.from_payload(values.get("photo_edits", {}))
        except (TypeError, ValueError):
            return PhotoEditDraft()

    def save_draft(self, image_path: str | Path, draft: PhotoEditDraft) -> str:
        image = Path(image_path)
        if not image.is_file():
            raise FileNotFoundError(image)
        sidecar = self._sidecar_path(image)
        payload = _load_sidecar_for_update(sidecar)
        try:
            version = int(payload.get("version", 1) or 1)
        except (TypeError, ValueError):
            version = 1
        payload["version"] = max(2, version)
        payload["image_path"] = str(image)
        payload["photo_edits"] = draft.as_payload()
        atomic_write_text(sidecar, json.dumps(payload, indent=2, sort_keys=True))
        return str(sidecar)

    @staticmethod
    def can_embed(image_path: str | Path) -> bool:
        return Path(image_path).suffix.casefold() in {".jpg", ".jpeg", ".tif", ".tiff"}

    def embed_draft(self, image_path: str | Path, draft: PhotoEditDraft, *, progress_callback=None, cancel_check=None):
        if not self.can_embed(image_path):
            raise ValueError("Embedding metadata is available only for writable JPEG and TIFF files. The sidecar remains safe for this file.")
        from .gallery_actions import GalleryActionService

        return GalleryActionService().write_exif_drafts(
            [str(image_path)],
            draft.as_payload(),
            progress_callback=progress_callback,
            cancel_check=cancel_check,
        )


class PhotoMetadataService:
    def __init__(self, face_detection_service: FaceDetectionService | None = None) -> None:
        self.face_detection_service = face_detection_service

    def _face_detection(self) -> FaceDetectionService:
        if self.face_detection_service is None:
            from .face_search import FaceDetectionService

            self.face_detection_service = FaceDetectionService()
        return self.face_detection_service

    def get_metadata(
        self,
        image_path: str,
        context: dict[str, object] | None = None,
        *,
        include_hashes: bool = True,
        include_face_boxes: bool = False,
    ) -> PhotoMetadata:
        path = Path(image_path)
        stat = path.stat()
        exif = {}
        camera = ""
        width = 0
        height = 0
        hashes = {"phash": "", "dhash": "", "whash": ""}
        try:
            with Image.open(path) as image:
                width, height = image.size
                image_exif = image.getexif()
                if image_exif:
                    for key, value in image_exif.items():
                        tag = str(ExifTags.TAGS.get(key, key))
                        exif[tag] = str(value)
                    # Pillow keeps the most useful capture-time fields in
                    # nested Exif/GPS IFDs for many JPEGs. Include those
                    # scalar values in the normal metadata payload so the
                    # standard inspector can show their dates without an
                    # advanced-only raw dump.
                    for ifd_tag, tag_names in ((34665, ExifTags.TAGS), (34853, ExifTags.GPSTAGS)):
                        try:
                            nested = image_exif.get_ifd(ifd_tag)
                        except (AttributeError, KeyError, TypeError, ValueError):
                            nested = {}
                        for key, value in dict(nested or {}).items():
                            tag = str(tag_names.get(key, key))
                            if tag not in exif or not exif[tag]:
                                exif[tag] = str(value)
                    camera = exif.get("Model", "")
                if include_hashes:
                    import imagehash

                    rgb = image.convert("RGB")
                    hashes = {
                        "phash": str(imagehash.phash(rgb, hash_size=8)),
                        "dhash": str(imagehash.dhash(rgb, hash_size=8)),
                        "whash": str(imagehash.whash(rgb, hash_size=8)),
                    }
        except Exception:
            width = 0
            height = 0
            camera = ""
            exif = {}
            hashes = {"phash": "", "dhash": "", "whash": ""}
        face_boxes = []
        if include_face_boxes:
            face_boxes = [
                {
                    "bbox": face.bbox,
                    "confidence": round(float(face.confidence), 6),
                }
                for face in self._face_detection().detect_faces(str(path))
            ]
        return PhotoMetadata(
            image_path=str(path),
            width=width,
            height=height,
            file_size=int(stat.st_size),
            modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(sep=" ", timespec="seconds"),
            camera=camera,
            exif=exif,
            hashes=hashes,
            face_boxes=face_boxes,
            context=context or {},
        )


class MetadataSidecarService:
    def __init__(self, *, tag_service=None, face_service=None) -> None:
        if tag_service is None:
            from .image_tags import ImageTagService

            tag_service = ImageTagService()
        self.tag_service = tag_service
        self.face_service = face_service

    @staticmethod
    def sidecar_path_for_image(image_path: str | Path, output_dir: str | Path | None = None) -> Path:
        image = Path(image_path)
        base_dir = Path(output_dir) if output_dir is not None else image.parent
        return base_dir / f"{image.name}.clusterlens.json"

    def export_sidecars(
        self,
        image_paths: list[str],
        *,
        output_dir: str | Path | None = None,
        ratings: dict[str, int] | None = None,
        captions: dict[str, str] | None = None,
        people: dict[str, list[dict[str, object]]] | None = None,
        profile_refs: dict[str, list[dict[str, object]]] | None = None,
    ) -> dict[str, str]:
        paths = [str(Path(path)) for path in image_paths if str(path or "").strip()]
        tags_by_path = self.tag_service.load_tags_for_paths(paths, import_missing_exif=False)
        ratings = dict(ratings or {})
        captions = dict(captions or {})
        people = dict(people or {})
        profile_refs = dict(profile_refs or {})
        written: dict[str, str] = {}
        for image_path in paths:
            sidecar_path = self.sidecar_path_for_image(image_path, output_dir)
            sidecar_path.parent.mkdir(parents=True, exist_ok=True)
            existing = _load_sidecar_for_update(sidecar_path)
            payload = {
                "version": 1,
                "image_path": image_path,
                "tags": list(tags_by_path.get(str(Path(image_path)), tags_by_path.get(image_path, ()))),
                "rating": int(ratings.get(image_path, ratings.get(str(Path(image_path)), 0)) or 0),
                "caption": str(captions.get(image_path, captions.get(str(Path(image_path)), "")) or ""),
                "people_labels": list(people.get(image_path, people.get(str(Path(image_path)), [])) or []),
                "profile_refs": list(profile_refs.get(image_path, profile_refs.get(str(Path(image_path)), [])) or []),
            }
            if isinstance(existing, dict) and isinstance(existing.get("photo_edits"), dict):
                try:
                    existing_version = int(existing.get("version", 1) or 1)
                except (TypeError, ValueError):
                    existing_version = 1
                payload["version"] = max(2, existing_version)
                payload["photo_edits"] = dict(existing["photo_edits"])
            atomic_write_text(sidecar_path, json.dumps(payload, indent=2, sort_keys=True))
            written[image_path] = str(sidecar_path)
        return written

    def import_sidecars(
        self,
        sidecar_paths: list[str],
        *,
        apply_tags: bool = True,
        mirror_to_exif: bool = False,
    ) -> dict[str, dict[str, object]]:
        imported: dict[str, dict[str, object]] = {}
        for sidecar_path in [Path(path) for path in sidecar_paths if str(path or "").strip()]:
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            image_path = str(payload.get("image_path") or "").strip()
            if not image_path:
                continue
            tags = [str(tag).strip() for tag in list(payload.get("tags", []) or []) if str(tag).strip()]
            if apply_tags and tags:
                self.tag_service.apply_tag_edit([image_path], add_tags=tags, mirror_to_exif=mirror_to_exif)
            imported[image_path] = {
                "tags": tags,
                "rating": int(payload.get("rating", 0) or 0),
                "caption": str(payload.get("caption", "") or ""),
                "people_labels": list(payload.get("people_labels", []) or []),
                "profile_refs": list(payload.get("profile_refs", []) or []),
            }
        return imported


def _load_sidecar_for_update(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Existing metadata sidecar is unreadable and was not overwritten: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Existing metadata sidecar is not an object and was not overwritten: {path}")
    return dict(payload)
