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
            payload = {
                "version": 1,
                "image_path": image_path,
                "tags": list(tags_by_path.get(str(Path(image_path)), tags_by_path.get(image_path, ()))),
                "rating": int(ratings.get(image_path, ratings.get(str(Path(image_path)), 0)) or 0),
                "caption": str(captions.get(image_path, captions.get(str(Path(image_path)), "")) or ""),
                "people_labels": list(people.get(image_path, people.get(str(Path(image_path)), [])) or []),
                "profile_refs": list(profile_refs.get(image_path, profile_refs.get(str(Path(image_path)), [])) or []),
            }
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
