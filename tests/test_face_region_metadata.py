from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from app.services.face_region_metadata import FaceRegion, FaceRegionMetadataService, FaceRegionUpdate
from app.services.face_search import FaceIndexRecord, FaceIndexService
from app.services.gallery_actions import GalleryActionService


@pytest.fixture(autouse=True)
def _isolated_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Keep metadata tests independent from user application state."""

    import infra.settings as settings_module

    monkeypatch.setenv("CLUSTERLENS_RUNTIME_ROOT", str(tmp_path / "runtime"))
    previous = settings_module._RUNTIME_BASE_DIR
    settings_module._RUNTIME_BASE_DIR = None
    try:
        yield
    finally:
        settings_module._RUNTIME_BASE_DIR = previous


def _service(tmp_path: Path) -> FaceRegionMetadataService:
    return FaceRegionMetadataService(
        action_service=GalleryActionService(
            audit_log_path=tmp_path / "audit.jsonl",
            journal_path=tmp_path / "journal.sqlite3",
            temp_dir=tmp_path / "temporary",
        )
    )


def _face_index(tmp_path: Path) -> FaceIndexService:
    # These tests write supplied index records only, so no detector/model is
    # initialized. The placeholder services keep the fixture local and fast.
    return FaceIndexService(
        detection_service=object(),
        embedding_service=object(),
        db_path=tmp_path / "faces.sqlite3",
    )


def _save_two_indexed_faces(service: FaceIndexService, image_path: Path) -> None:
    service.save_face_records(
        [
            FaceIndexRecord(str(image_path), 0, (0, 0, 40, 50), 0.99, np.array([1.0, 0.0], dtype=np.float32)),
            FaceIndexRecord(str(image_path), 1, (60, 0, 100, 50), 0.99, np.array([0.0, 1.0], dtype=np.float32)),
        ],
        mtime_ns=image_path.stat().st_mtime_ns,
        file_size=image_path.stat().st_size,
        assess_quality=False,
    )


def test_jpeg_round_trip_preserves_independent_face_region_names(tmp_path: Path) -> None:
    image_path = tmp_path / "two-people.jpg"
    Image.new("RGB", (100, 100), "white").save(image_path)
    service = _service(tmp_path)

    result = service.update(
        image_path,
        [
            FaceRegionUpdate(0.10, 0.15, 0.25, 0.30, "Alice"),
            FaceRegionUpdate(0.60, 0.20, 0.20, 0.25, "Bob"),
        ],
    )

    assert result.succeeded
    assert result.storage == "embedded"
    assert [region.name for region in service.read(image_path).regions] == ["Alice", "Bob"]
    packet = FaceRegionMetadataService._read_jpeg_xmp(image_path)
    assert b"mwg-rs:AppliedToDimensions" in packet
    assert b"mwg-rs:RegionList" in packet
    # The app mirror is written alongside the interoperable MWG XMP packet.
    assert GalleryActionService.read_exif_metadata_value(str(image_path), "clusterlens_face_regions_v1")

    renamed = service.update(image_path, [FaceRegionUpdate(0.10, 0.15, 0.25, 0.30, "Cara")])

    assert renamed.succeeded
    assert [region.name for region in service.read(image_path).regions] == ["Cara", "Bob"]


def test_unlabel_keeps_the_face_region_box(tmp_path: Path) -> None:
    image_path = tmp_path / "face.jpg"
    Image.new("RGB", (80, 60), "white").save(image_path)
    service = _service(tmp_path)
    service.update(image_path, [FaceRegionUpdate(0.25, 0.20, 0.30, 0.45, "Alice")])

    result = service.update(image_path, [FaceRegionUpdate(0.25, 0.20, 0.30, 0.45, None)])
    document = service.read(image_path)

    assert result.succeeded
    assert len(document.regions) == 1
    assert document.regions[0].name == ""
    assert document.regions[0].width == 0.30


def test_non_jpeg_uses_standard_xmp_sidecar_without_rewriting_source(tmp_path: Path) -> None:
    image_path = tmp_path / "face.png"
    Image.new("RGB", (50, 50), "white").save(image_path)
    original = image_path.read_bytes()
    service = _service(tmp_path)

    result = service.update(image_path, [FaceRegionUpdate(0.1, 0.1, 0.4, 0.4, "Alice")])

    assert result.succeeded
    assert result.storage == "sidecar"
    assert image_path.read_bytes() == original
    assert image_path.with_suffix(".xmp").is_file()
    assert service.read(image_path).names == ("Alice",)


def test_sidecar_takes_precedence_over_embedded_regions(tmp_path: Path) -> None:
    image_path = tmp_path / "face.jpg"
    Image.new("RGB", (50, 50), "white").save(image_path)
    service = _service(tmp_path)
    service.update(image_path, [FaceRegionUpdate(0.1, 0.1, 0.4, 0.4, "Alice")])
    image_path.with_suffix(".xmp").write_text(
        service._build_packet((FaceRegion(0.1, 0.1, 0.4, 0.4, "Bob"),)).decode("utf-8"),
        encoding="utf-8",
    )

    assert service.read(image_path).storage == "sidecar"
    assert service.read(image_path).names == ("Bob",)


def test_face_region_merge_preserves_unrelated_xmp_fields(tmp_path: Path) -> None:
    image_path = tmp_path / "preserve.jpg"
    Image.new("RGB", (50, 50), "white").save(image_path)
    service = _service(tmp_path)
    original = b'''<?xpacket begin="\xef\xbb\xbf"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <rdf:RDF><rdf:Description><dc:title>Keep this title</dc:title></rdf:Description></rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>'''
    FaceRegionMetadataService._write_jpeg_xmp(image_path, original)

    result = service.update(image_path, [FaceRegionUpdate(0.1, 0.1, 0.4, 0.4, "Alice")])

    assert result.succeeded
    assert b"Keep this title" in FaceRegionMetadataService._read_jpeg_xmp(image_path)


def test_reads_standard_mwg_regions_that_use_child_value_elements(tmp_path: Path) -> None:
    image_path = tmp_path / "external-standard.jpg"
    Image.new("RGB", (50, 50), "white").save(image_path)
    packet = b'''<?xpacket begin="\xef\xbb\xbf"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:mwg-rs="http://www.metadataworkinggroup.com/schemas/regions/" xmlns:stArea="http://ns.adobe.com/xmp/sType/Area#">
  <rdf:RDF><rdf:Description><mwg-rs:Regions rdf:parseType="Resource"><mwg-rs:RegionList><rdf:Bag><rdf:li rdf:parseType="Resource"><mwg-rs:Name>External Alice</mwg-rs:Name><mwg-rs:Type>Face</mwg-rs:Type><mwg-rs:Area rdf:parseType="Resource"><stArea:x>0.1</stArea:x><stArea:y>0.2</stArea:y><stArea:w>0.3</stArea:w><stArea:h>0.4</stArea:h><stArea:unit>normalized</stArea:unit></mwg-rs:Area></rdf:li></rdf:Bag></mwg-rs:RegionList></mwg-rs:Regions></rdf:Description></rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>'''
    FaceRegionMetadataService._write_jpeg_xmp(image_path, packet)

    document = _service(tmp_path).read(image_path)

    assert document.storage == "embedded"
    assert document.names == ("External Alice",)
    assert document.regions[0].width == 0.3


def test_legacy_image_person_name_is_read_only_fallback(tmp_path: Path) -> None:
    image_path = tmp_path / "legacy.jpg"
    Image.new("RGB", (50, 50), "white").save(image_path)
    action_service = GalleryActionService(
        audit_log_path=tmp_path / "audit.jsonl",
        journal_path=tmp_path / "journal.sqlite3",
        temp_dir=tmp_path / "temporary",
    )
    action_service.write_exif_metadata_pair(str(image_path), "ic_person_name", "Legacy Alice")

    document = _service(tmp_path).read(image_path)

    assert document.regions == ()
    assert document.legacy_name == "Legacy Alice"


def test_missing_image_reports_a_scoped_write_failure(tmp_path: Path) -> None:
    result = _service(tmp_path).update(tmp_path / "missing.jpg", [FaceRegionUpdate(0.1, 0.1, 0.4, 0.4, "Alice")])

    assert not result.succeeded
    assert "missing.jpg" in result.error


def test_indexed_face_mutations_are_region_scoped_and_persisted(tmp_path: Path) -> None:
    image_path = tmp_path / "mixed_people.jpg"
    Image.new("RGB", (100, 100), "white").save(image_path)
    service = _face_index(tmp_path)
    _save_two_indexed_faces(service, image_path)

    first = service.label_indexed_faces_with_metadata("Alice", [(str(image_path), 0)], similarity_threshold=0.5)
    second = service.label_indexed_faces_with_metadata("Bob", [(str(image_path), 1)], similarity_threshold=0.5)

    assert first.affected_refs == ((str(image_path), 0),)
    assert second.affected_refs == ((str(image_path), 1),)
    assert [region.name for region in _service(tmp_path).read(image_path).regions] == ["Alice", "Bob"]

    renamed = service.rename_face_regions_in_images("Alice", "Cara", [str(image_path)])
    unlabelled = service.unlabel_face_regions_in_images("Cara", [str(image_path)])

    assert renamed.affected_refs == ((str(image_path), 0),)
    assert unlabelled.affected_refs == ((str(image_path), 0),)
    assert [region.name for region in _service(tmp_path).read(image_path).regions] == ["", "Bob"]
    assert service.load_face_record(str(image_path), 0).person_name == ""
    assert service.load_face_record(str(image_path), 1).person_name == "Bob"


def test_explicit_face_region_name_overrides_automatic_quality_gate(tmp_path: Path) -> None:
    image_path = tmp_path / "low-quality-but-confirmed.jpg"
    Image.new("RGB", (100, 100), "white").save(image_path)
    service = _face_index(tmp_path)
    service.save_face_records(
        [
            FaceIndexRecord(
                image_path=str(image_path),
                face_index=0,
                face_bbox=(0, 0, 40, 50),
                face_confidence=0.99,
                embedding=np.array([1.0, 0.0], dtype=np.float32),
                quality_status="reject",
                quality_score=0.10,
            )
        ],
        mtime_ns=image_path.stat().st_mtime_ns,
        file_size=image_path.stat().st_size,
        assess_quality=False,
    )

    named = service.label_indexed_faces_with_metadata("Lilly", [(str(image_path), 0)], similarity_threshold=0.5)

    assert named.affected_refs == ((str(image_path), 0),)
    assert _service(tmp_path).read(image_path).names == ("Lilly",)
    assert service.load_face_record(str(image_path), 0).person_name == "Lilly"

    unlabelled = service.unlabel_indexed_faces_with_metadata([(str(image_path), 0)])

    assert unlabelled.affected_refs == ((str(image_path), 0),)
    assert _service(tmp_path).read(image_path).names == ()
    assert service.load_face_record(str(image_path), 0).person_name == ""


def test_reindex_keeps_database_name_when_xmp_conflicts(tmp_path: Path) -> None:
    image_path = tmp_path / "conflict.jpg"
    Image.new("RGB", (100, 100), "white").save(image_path)
    _service(tmp_path).update(image_path, [FaceRegionUpdate(0.0, 0.0, 0.4, 0.5, "External Alice")])
    service = _face_index(tmp_path)
    _save_two_indexed_faces(service, image_path)
    service.label_indexed_faces_immediately("Database Bob", [(str(image_path), 0)], similarity_threshold=0.5)

    _save_two_indexed_faces(service, image_path)

    assert service.load_face_record(str(image_path), 0).person_name == "Database Bob"
    assert _service(tmp_path).read(image_path).regions[0].name == "External Alice"
