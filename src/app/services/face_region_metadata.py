"""Read and write independently named face regions in photo metadata.

The interoperability format is the Metadata Working Group XMP Region schema.
ClusterLens also stores a compact mirror in its own XMP namespace (and, for
embedded JPEG files, in the existing EXIF UserComment key/value store) so a
round trip remains possible if a third-party application strips MWG fields.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
import json
from pathlib import Path
import struct
from typing import Iterable
from xml.etree import ElementTree as ET

from PIL import Image

from infra.atomic_io import atomic_write_text


XMP_META_NS = "adobe:ns:meta/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
MWG_RS_NS = "http://www.metadataworkinggroup.com/schemas/regions/"
ST_AREA_NS = "http://ns.adobe.com/xmp/sType/Area#"
ST_DIM_NS = "http://ns.adobe.com/xap/1.0/sType/Dimensions#"
CLUSTERLENS_NS = "https://clusterlens.app/ns/face-regions/1.0/"
XMP_PACKET_HEADER = b"http://ns.adobe.com/xap/1.0/\x00"
EXIF_MIRROR_KEY = "clusterlens_face_regions_v1"
LEGACY_PERSON_KEY = "ic_person_name"
IOU_MATCH_THRESHOLD = 0.75

ET.register_namespace("x", XMP_META_NS)
ET.register_namespace("rdf", RDF_NS)
ET.register_namespace("mwg-rs", MWG_RS_NS)
ET.register_namespace("stArea", ST_AREA_NS)
ET.register_namespace("stDim", ST_DIM_NS)
ET.register_namespace("clusterlens", CLUSTERLENS_NS)


@dataclass(frozen=True)
class FaceRegion:
    """A normalized image region. ``name`` is empty for an unlabeled face."""

    x: float
    y: float
    width: float
    height: float
    name: str = ""
    region_type: str = "Face"

    def normalized(self) -> "FaceRegion":
        x = min(1.0, max(0.0, float(self.x)))
        y = min(1.0, max(0.0, float(self.y)))
        width = min(1.0 - x, max(0.0, float(self.width)))
        height = min(1.0 - y, max(0.0, float(self.height)))
        return FaceRegion(x, y, width, height, str(self.name or "").strip(), str(self.region_type or "Face").strip() or "Face")

    def as_dict(self) -> dict[str, object]:
        value = self.normalized()
        return {
            "x": round(value.x, 8),
            "y": round(value.y, 8),
            "width": round(value.width, 8),
            "height": round(value.height, 8),
            "name": value.name,
            "type": value.region_type,
        }


@dataclass(frozen=True)
class FaceRegionUpdate:
    """Set the name for a normalized face rectangle; ``None`` removes it."""

    x: float
    y: float
    width: float
    height: float
    name: str | None

    def region(self) -> FaceRegion:
        return FaceRegion(self.x, self.y, self.width, self.height).normalized()


@dataclass(frozen=True)
class FaceRegionDocument:
    image_path: str
    regions: tuple[FaceRegion, ...] = ()
    storage: str = "none"
    legacy_name: str = ""

    @property
    def named_regions(self) -> tuple[FaceRegion, ...]:
        return tuple(region for region in self.regions if str(region.name or "").strip())

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(sorted({str(region.name).strip() for region in self.named_regions}, key=str.casefold))


@dataclass(frozen=True)
class FaceRegionWriteResult:
    image_path: str
    storage: str
    regions: tuple[FaceRegion, ...]
    error: str = ""

    @property
    def succeeded(self) -> bool:
        return not bool(self.error)


def _tag(namespace: str, name: str) -> str:
    return f"{{{namespace}}}{name}"


def _float_attr(element: ET.Element, namespace: str, name: str) -> float | None:
    try:
        value = element.attrib.get(_tag(namespace, name), element.attrib.get(name, ""))
        if value in {None, ""}:
            value = element.findtext(_tag(namespace, name), default="")
        return float(value)
    except (TypeError, ValueError):
        return None


def _region_iou(left: FaceRegion, right: FaceRegion) -> float:
    a = left.normalized()
    b = right.normalized()
    x1 = max(a.x, b.x)
    y1 = max(a.y, b.y)
    x2 = min(a.x + a.width, b.x + b.width)
    y2 = min(a.y + a.height, b.y + b.height)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a.width * a.height) + (b.width * b.height) - intersection
    return intersection / union if union > 0 else 0.0


class FaceRegionMetadataService:
    """Losslessly scoped face-region metadata operations.

    JPEG receives an embedded XMP packet. A pre-existing ``.xmp`` sidecar is
    authoritative and continues to be used. Other formats use a sidecar so
    their source pixels/containers are never rewritten by this feature.
    """

    def __init__(self, *, action_service=None) -> None:
        # Injection keeps the metadata layer deterministic in isolated tests
        # and lets callers share the existing journal/audit service.
        self._action_service = action_service

    def sidecar_path_for_image(self, image_path: str | Path) -> Path:
        return Path(image_path).with_suffix(".xmp")

    def read(self, image_path: str | Path) -> FaceRegionDocument:
        path = Path(image_path)
        sidecar = self.sidecar_path_for_image(path)
        legacy_name = self._read_legacy_name(path)
        if sidecar.is_file():
            try:
                regions = self._regions_from_packet(sidecar.read_bytes())
                return FaceRegionDocument(str(path), tuple(regions), "sidecar", legacy_name)
            except (OSError, ValueError, ET.ParseError):
                return FaceRegionDocument(str(path), (), "sidecar", legacy_name)
        if path.suffix.lower() in {".jpg", ".jpeg"}:
            try:
                packet = self._read_jpeg_xmp(path)
                if packet:
                    return FaceRegionDocument(str(path), tuple(self._regions_from_packet(packet)), "embedded", legacy_name)
            except (OSError, ValueError, ET.ParseError):
                pass
        mirrored = self._read_exif_mirror(path)
        if mirrored:
            return FaceRegionDocument(str(path), tuple(mirrored), "exif-mirror", legacy_name)
        return FaceRegionDocument(str(path), (), "none", legacy_name)

    def update(self, image_path: str | Path, updates: Iterable[FaceRegionUpdate]) -> FaceRegionWriteResult:
        path = Path(image_path)
        document = self.read(path)
        regions = list(document.regions)
        for update in updates:
            target = update.region()
            match_index = self._unique_best_match(regions, target)
            updated = FaceRegion(target.x, target.y, target.width, target.height, "" if update.name is None else str(update.name or "").strip())
            if match_index is None:
                regions.append(updated)
            else:
                existing = regions[match_index]
                regions[match_index] = FaceRegion(existing.x, existing.y, existing.width, existing.height, updated.name, existing.region_type).normalized()
        normalized = tuple(region.normalized() for region in regions)
        try:
            storage = self._write(path, normalized, preferred_storage=document.storage)
            return FaceRegionWriteResult(str(path), storage, normalized)
        except Exception as exc:  # the caller needs a per-photo failure, not a batch abort
            return FaceRegionWriteResult(str(path), document.storage, tuple(document.regions), str(exc))

    def replace(self, image_path: str | Path, regions: Iterable[FaceRegion]) -> FaceRegionWriteResult:
        path = Path(image_path)
        normalized = tuple(region.normalized() for region in regions)
        try:
            storage = self._write(path, normalized, preferred_storage=self.read(path).storage)
            return FaceRegionWriteResult(str(path), storage, normalized)
        except Exception as exc:
            return FaceRegionWriteResult(str(path), "none", (), str(exc))

    def names_for_paths(self, image_paths: Iterable[str | Path]) -> dict[str, tuple[str, ...]]:
        return {str(Path(path)): self.read(path).names for path in image_paths if str(path or "").strip()}

    @staticmethod
    def normalized_region_for_bbox(bbox: tuple[int, int, int, int], image_size: tuple[int, int]) -> FaceRegion:
        width, height = (max(1, int(image_size[0])), max(1, int(image_size[1])))
        left, top, right, bottom = (int(value) for value in bbox)
        return FaceRegion(left / width, top / height, max(0, right - left) / width, max(0, bottom - top) / height).normalized()

    @staticmethod
    def bbox_for_region(region: FaceRegion, image_size: tuple[int, int]) -> tuple[int, int, int, int]:
        width, height = (max(1, int(image_size[0])), max(1, int(image_size[1])))
        value = region.normalized()
        left = int(round(value.x * width))
        top = int(round(value.y * height))
        right = int(round((value.x + value.width) * width))
        bottom = int(round((value.y + value.height) * height))
        return (left, top, max(left + 1, right), max(top + 1, bottom))

    @staticmethod
    def best_matching_region(regions: Iterable[FaceRegion], target: FaceRegion) -> FaceRegion | None:
        values = list(regions)
        scores = sorted(((_region_iou(value, target), index, value) for index, value in enumerate(values)), reverse=True)
        if not scores or scores[0][0] < IOU_MATCH_THRESHOLD:
            return None
        if len(scores) > 1 and abs(scores[0][0] - scores[1][0]) < 1e-9:
            return None
        return scores[0][2]

    def _unique_best_match(self, regions: list[FaceRegion], target: FaceRegion) -> int | None:
        scores = sorted(((_region_iou(value, target), index) for index, value in enumerate(regions)), reverse=True)
        if not scores or scores[0][0] < IOU_MATCH_THRESHOLD:
            return None
        if len(scores) > 1 and abs(scores[0][0] - scores[1][0]) < 1e-9:
            return None
        return int(scores[0][1])

    def _write(self, path: Path, regions: tuple[FaceRegion, ...], *, preferred_storage: str) -> str:
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            image_size = tuple(int(value) for value in image.size)
        sidecar = self.sidecar_path_for_image(path)
        if preferred_storage == "sidecar" or sidecar.exists() or path.suffix.lower() not in {".jpg", ".jpeg"}:
            packet = self._build_packet(
                regions,
                existing_packet=sidecar.read_bytes() if sidecar.exists() else b"",
                image_size=image_size,
            )
            atomic_write_text(sidecar, packet.decode("utf-8"))
            return "sidecar"
        existing = self._read_jpeg_xmp(path)
        packet = self._build_packet(regions, existing_packet=existing, image_size=image_size)
        # The pre-existing EXIF writer may rebuild the JPEG and discard APP1
        # XMP.  Run it first, then install the complete packet below.  The
        # standard XMP document remains the successful write if a legacy
        # mirror cannot be written (for example an image format/Pillow limit).
        try:
            self._write_exif_mirror(path, regions)
        except Exception:
            pass
        self._write_jpeg_xmp(path, packet)
        return "embedded"

    def _regions_from_packet(self, packet: bytes) -> list[FaceRegion]:
        if not packet:
            return []
        root = ET.fromstring(packet.decode("utf-8", errors="strict"))
        regions: list[FaceRegion] = []
        for area in root.findall(f".//{_tag(MWG_RS_NS, 'Area')}"):
            x = _float_attr(area, ST_AREA_NS, "x")
            y = _float_attr(area, ST_AREA_NS, "y")
            width = _float_attr(area, ST_AREA_NS, "w")
            height = _float_attr(area, ST_AREA_NS, "h")
            if None in {x, y, width, height}:
                continue
            parent = next((candidate for candidate in root.iter() if area in list(candidate)), None)
            name = ""
            region_type = "Face"
            if parent is not None:
                name = str(parent.attrib.get(_tag(MWG_RS_NS, "Name"), parent.attrib.get("Name", "")) or "").strip()
                region_type = str(parent.attrib.get(_tag(MWG_RS_NS, "Type"), parent.attrib.get("Type", "Face")) or "Face").strip()
                if not name:
                    name = str(parent.findtext(_tag(MWG_RS_NS, "Name"), default="") or "").strip()
                if region_type == "Face":
                    region_type = str(parent.findtext(_tag(MWG_RS_NS, "Type"), default="Face") or "Face").strip()
            regions.append(FaceRegion(float(x), float(y), float(width), float(height), name, region_type).normalized())
        if regions:
            return regions
        # A valid ClusterLens mirror is deliberately accepted only if there
        # were no standard MWG regions in the packet.
        mirror = root.find(f".//{_tag(CLUSTERLENS_NS, 'FaceRegionsJson')}")
        if mirror is not None and str(mirror.text or "").strip():
            return self._decode_mirror(str(mirror.text or ""))
        return []

    def _build_packet(
        self,
        regions: tuple[FaceRegion, ...],
        *,
        existing_packet: bytes = b"",
        image_size: tuple[int, int] = (1, 1),
    ) -> bytes:
        try:
            root = ET.fromstring(existing_packet.decode("utf-8", errors="strict")) if existing_packet else None
        except (ET.ParseError, UnicodeDecodeError):
            root = None
        if root is None:
            root = ET.Element(_tag(XMP_META_NS, "xmpmeta"))
            rdf = ET.SubElement(root, _tag(RDF_NS, "RDF"))
            description = ET.SubElement(rdf, _tag(RDF_NS, "Description"))
        else:
            rdf = root.find(f".//{_tag(RDF_NS, 'RDF')}")
            if rdf is None:
                rdf = ET.SubElement(root, _tag(RDF_NS, "RDF"))
            description = next(
                (
                    candidate
                    for candidate in rdf.findall(_tag(RDF_NS, "Description"))
                    if candidate.find(_tag(MWG_RS_NS, "Regions")) is not None
                ),
                None,
            )
            if description is None:
                description = rdf.find(_tag(RDF_NS, "Description"))
            if description is None:
                description = ET.SubElement(rdf, _tag(RDF_NS, "Description"))
        for child in list(description):
            if child.tag in {_tag(MWG_RS_NS, "Regions"), _tag(CLUSTERLENS_NS, "FaceRegionsJson")}:
                description.remove(child)
        regions_node = ET.SubElement(description, _tag(MWG_RS_NS, "Regions"))
        regions_node.set(_tag(RDF_NS, "parseType"), "Resource")
        dimensions = ET.SubElement(regions_node, _tag(MWG_RS_NS, "AppliedToDimensions"))
        dimensions.set(_tag(ST_DIM_NS, "w"), str(max(1, int(image_size[0]))))
        dimensions.set(_tag(ST_DIM_NS, "h"), str(max(1, int(image_size[1]))))
        dimensions.set(_tag(ST_DIM_NS, "unit"), "pixel")
        region_list = ET.SubElement(regions_node, _tag(MWG_RS_NS, "RegionList"))
        bag = ET.SubElement(region_list, _tag(RDF_NS, "Bag"))
        for region in regions:
            item = ET.SubElement(bag, _tag(RDF_NS, "li"))
            item_description = ET.SubElement(item, _tag(RDF_NS, "Description"))
            value = region.normalized()
            if value.name:
                item_description.set(_tag(MWG_RS_NS, "Name"), value.name)
            item_description.set(_tag(MWG_RS_NS, "Type"), value.region_type or "Face")
            area = ET.SubElement(item_description, _tag(MWG_RS_NS, "Area"))
            area.set(_tag(ST_AREA_NS, "x"), f"{value.x:.8f}")
            area.set(_tag(ST_AREA_NS, "y"), f"{value.y:.8f}")
            area.set(_tag(ST_AREA_NS, "w"), f"{value.width:.8f}")
            area.set(_tag(ST_AREA_NS, "h"), f"{value.height:.8f}")
            area.set(_tag(ST_AREA_NS, "unit"), "normalized")
        mirror = ET.SubElement(description, _tag(CLUSTERLENS_NS, "FaceRegionsJson"))
        mirror.text = self._encode_mirror(regions)
        packet_header = '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'.encode("utf-8")
        return packet_header + ET.tostring(root, encoding="utf-8", xml_declaration=False) + b"\n<?xpacket end=\"w\"?>"

    @staticmethod
    def _encode_mirror(regions: Iterable[FaceRegion]) -> str:
        payload = json.dumps([region.as_dict() for region in regions], sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return base64.urlsafe_b64encode(payload).decode("ascii")

    @staticmethod
    def _decode_mirror(value: str) -> list[FaceRegion]:
        try:
            raw = base64.urlsafe_b64decode(str(value).encode("ascii"))
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeError, json.JSONDecodeError):
            return []
        if not isinstance(payload, list):
            return []
        regions: list[FaceRegion] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                regions.append(
                    FaceRegion(
                        float(item.get("x", 0.0)),
                        float(item.get("y", 0.0)),
                        float(item.get("width", 0.0)),
                        float(item.get("height", 0.0)),
                        str(item.get("name", "") or ""),
                        str(item.get("type", "Face") or "Face"),
                    ).normalized()
                )
            except (TypeError, ValueError):
                continue
        return regions

    def _read_legacy_name(self, path: Path) -> str:
        try:
            from app.services.gallery_actions import GalleryActionService

            return str(GalleryActionService.read_exif_metadata_value(str(path), LEGACY_PERSON_KEY) or "").strip()
        except Exception:
            return ""

    def _read_exif_mirror(self, path: Path) -> list[FaceRegion]:
        try:
            from app.services.gallery_actions import GalleryActionService

            return self._decode_mirror(GalleryActionService.read_exif_metadata_value(str(path), EXIF_MIRROR_KEY))
        except Exception:
            return []

    def _write_exif_mirror(self, path: Path, regions: tuple[FaceRegion, ...]) -> None:
        from app.services.gallery_actions import GalleryActionService

        action_service = self._action_service or GalleryActionService()
        result = action_service.write_exif_metadata_pairs(
            [str(path)],
            EXIF_MIRROR_KEY,
            self._encode_mirror(regions),
        )
        if result.failures or str(path) not in set(result.affected_paths):
            detail = "; ".join(str(item) for item in result.failures) or "EXIF mirror write did not affect the image."
            raise OSError(detail)

    @staticmethod
    def _read_jpeg_xmp(path: Path) -> bytes:
        raw = path.read_bytes()
        for marker, payload_start, payload_end, segment_start, segment_end in FaceRegionMetadataService._jpeg_segments(raw):
            if marker == 0xE1 and raw[payload_start:payload_end].startswith(XMP_PACKET_HEADER):
                return raw[payload_start + len(XMP_PACKET_HEADER) : payload_end]
        return b""

    @staticmethod
    def _write_jpeg_xmp(path: Path, packet: bytes) -> None:
        payload = XMP_PACKET_HEADER + bytes(packet)
        if len(payload) + 2 > 0xFFFF:
            raise ValueError("Face-region XMP packet is too large for embedded JPEG metadata.")
        raw = path.read_bytes()
        if not raw.startswith(b"\xff\xd8"):
            raise ValueError("Embedded face-region XMP requires a JPEG file.")
        retained: list[bytes] = [raw[:2]]
        cursor = 2
        for marker, payload_start, payload_end, segment_start, segment_end in FaceRegionMetadataService._jpeg_segments(raw):
            if marker == 0xDA:
                retained.append(raw[segment_start:])
                cursor = len(raw)
                break
            if marker == 0xE1 and raw[payload_start:payload_end].startswith(XMP_PACKET_HEADER):
                cursor = segment_end
                continue
            retained.append(raw[segment_start:segment_end])
            cursor = segment_end
        if cursor < len(raw):
            retained.append(raw[cursor:])
        segment = b"\xff\xe1" + struct.pack(">H", len(payload) + 2) + payload
        output = retained[0] + segment + b"".join(retained[1:])
        temporary = path.with_name(f".{path.stem}.face_regions_tmp{path.suffix}")
        try:
            temporary.write_bytes(output)
            with Image.open(temporary) as image:
                image.verify()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _jpeg_segments(raw: bytes):
        if not raw.startswith(b"\xff\xd8"):
            raise ValueError("Not a JPEG file")
        position = 2
        length = len(raw)
        while position + 1 < length:
            if raw[position] != 0xFF:
                raise ValueError("Invalid JPEG segment marker")
            segment_start = position
            while position < length and raw[position] == 0xFF:
                position += 1
            if position >= length:
                break
            marker = raw[position]
            position += 1
            if marker == 0xD9:
                yield marker, position, position, segment_start, position
                break
            if marker == 0xDA:
                if position + 2 > length:
                    raise ValueError("Truncated JPEG scan header")
                size = struct.unpack(">H", raw[position : position + 2])[0]
                segment_end = position + size
                if segment_end > length:
                    raise ValueError("Truncated JPEG scan header")
                yield marker, position + 2, segment_end, segment_start, segment_end
                break
            if marker in {0x01, *range(0xD0, 0xD8)}:
                yield marker, position, position, segment_start, position
                continue
            if position + 2 > length:
                raise ValueError("Truncated JPEG metadata segment")
            size = struct.unpack(">H", raw[position : position + 2])[0]
            if size < 2:
                raise ValueError("Invalid JPEG metadata segment length")
            segment_end = position + size
            if segment_end > length:
                raise ValueError("Truncated JPEG metadata segment")
            yield marker, position + 2, segment_end, segment_start, segment_end
            position = segment_end
