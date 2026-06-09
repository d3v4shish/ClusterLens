from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import ExifTags, Image

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
