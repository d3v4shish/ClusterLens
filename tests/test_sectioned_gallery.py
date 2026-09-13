from __future__ import annotations

import os
from unittest.mock import patch

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from ui.sectioned_gallery import GallerySection, SectionedGalleryModel


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_sectioned_gallery_keeps_photos_unique_and_groups_in_order() -> None:
    model = SectionedGalleryModel()
    model.set_column_count(2)
    model.set_sections(
        [
            GallerySection("largest", ("/a.jpg", "/b.jpg", "/c.jpg"), title="Photo group"),
            GallerySection("duplicate", ("/c.jpg", "/d.jpg"), title="Photo group"),
            GallerySection("other", ("/e.jpg",), kind="other", title="Other photos"),
        ]
    )

    assert model.all_paths() == ["/a.jpg", "/b.jpg", "/c.jpg", "/d.jpg", "/e.jpg"]
    assert model.rowCount() == 7  # header + two rows, header + one row, header + one row


def test_sectioned_gallery_collapse_and_group_selection_survive_reflow() -> None:
    model = SectionedGalleryModel()
    model.set_sections([GallerySection("group", tuple(f"/{index}.jpg" for index in range(5)))])
    model.toggle_checked_section("group")
    model.toggle_collapsed("group")
    assert model.rowCount() == 1
    assert model.checked_section_ids() == {"group"}

    model.set_column_count(2)
    assert model.rowCount() == 1
    model.toggle_collapsed("group")
    assert model.rowCount() == 4
    assert model.paths_for_sections({"group"}) == [f"/{index}.jpg" for index in range(5)]


def test_sectioned_gallery_expands_and_collapses_every_group() -> None:
    model = SectionedGalleryModel()
    model.set_column_count(2)
    model.set_sections(
        [
            GallerySection("first", ("/a.jpg", "/b.jpg")),
            GallerySection("second", ("/c.jpg", "/d.jpg")),
        ]
    )

    assert model.set_all_collapsed(True)
    assert model.rowCount() == 2
    assert not model.set_all_collapsed(True)
    assert model.set_all_collapsed(False)
    assert model.rowCount() == 4


def test_sectioned_gallery_exposes_header_and_photo_roles() -> None:
    model = SectionedGalleryModel()
    model.set_sections([GallerySection("group", ("/photo.jpg",), title="Photo group")])
    header = model.index(0, 0)
    photo = model.index(1, 0)

    assert model.data(header, model.HeaderRole).section_id == "group"
    assert model.data(photo, model.PathRole) == "/photo.jpg"
    assert model.flags(header) == Qt.ItemFlag.ItemIsEnabled


def test_sectioned_gallery_reflows_after_its_table_receives_a_real_width() -> None:
    app = QApplication.instance() or QApplication([])
    from ui.sectioned_gallery import SectionedGallery

    gallery = SectionedGallery()
    gallery.resize(1120, 700)
    gallery.show()
    app.processEvents()

    expected_columns = max(1, gallery.table.viewport().width() // max(120, gallery._tile_size + 18))
    assert gallery._model.columnCount() == expected_columns
    gallery.close()


def test_expanding_a_group_queues_its_thumbnails_without_waiting_for_layout() -> None:
    app = QApplication.instance() or QApplication([])
    from ui.sectioned_gallery import SectionedGallery

    gallery = SectionedGallery()
    paths = tuple(f"/{index}.jpg" for index in range(60))
    gallery._model.set_sections([GallerySection("group", paths)])
    gallery._model.toggle_collapsed("group")
    queued: list[tuple[str, ...]] = []
    gallery._queue_paths = lambda values: queued.append(tuple(values))  # type: ignore[method-assign]
    gallery._queue_visible_loads_after_layout = lambda: None  # type: ignore[method-assign]

    gallery._toggle_collapsed_section("group")

    assert queued == [paths[:48]]
    gallery.close()
    app.processEvents()


def test_group_header_hover_opens_a_photo_first_preview() -> None:
    app = QApplication.instance() or QApplication([])
    from ui.sectioned_gallery import SectionedGallery

    gallery = SectionedGallery()
    section = GallerySection("group", ("/a.jpg", "/b.jpg", "/c.jpg"), title="Photo group")
    gallery._model.set_sections([section])
    gallery._queue_paths = lambda _paths: None  # type: ignore[method-assign]

    gallery._update_hover_preview(gallery._model.index(0, 0))

    assert gallery._hover_section_id == "group"
    assert gallery._hover_popup.title.text() == "Photo group · 3 photos"
    gallery._hide_hover_preview()
    gallery.close()
    app.processEvents()


def test_photos_gallery_opens_inspector_with_face_region_editing_when_index_is_ready() -> None:
    app = QApplication.instance() or QApplication([])
    from ui.sectioned_gallery import SectionedGallery

    gallery = SectionedGallery()
    face_service = object()
    callback_paths: list[str] = []
    gallery.face_service_provider = lambda: face_service
    gallery.face_edit_saved_callback = callback_paths.append
    gallery._model.set_sections([GallerySection("group", ("/a.jpg", "/b.jpg"))])
    captured: dict[str, object] = {}

    class _Dialog:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def exec(self) -> int:
            return 0

    with patch("ui.sectioned_gallery.PhotoInspectorDialog", _Dialog):
        gallery._open_photo(gallery._model.index(1, 1))

    assert captured["image_paths"] == ["/a.jpg", "/b.jpg"]
    assert captured["start_index"] == 1
    assert captured["face_service"] is face_service
    assert captured["allow_face_edit"] is True
    captured["face_edit_saved_callback"]("/a.jpg")
    assert callback_paths == ["/a.jpg"]
    gallery.close()
    app.processEvents()
