from __future__ import annotations

import os
from unittest.mock import patch

from PyQt6.QtCore import QItemSelectionModel, Qt
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


def test_sectioned_gallery_nested_headers_keep_only_the_latest_month_open() -> None:
    model = SectionedGalleryModel()
    model.set_column_count(2)
    model.set_sections(
        [
            GallerySection(
                "year:2026",
                (),
                kind="timeline_year",
                title="2026",
                children=(
                    GallerySection("month:2026-03", ("/march-a.jpg", "/march-b.jpg"), kind="timeline_month", title="March 2026"),
                    GallerySection("month:2026-02", ("/february.jpg",), kind="timeline_month", title="February 2026"),
                ),
            ),
            GallerySection(
                "year:2025",
                (),
                kind="timeline_year",
                title="2025",
                children=(GallerySection("month:2025-12", ("/december.jpg",), kind="timeline_month", title="December 2025"),),
            ),
        ]
    )
    model.set_collapsed_sections({"month:2026-02", "year:2025", "month:2025-12"})

    assert model.all_paths() == ["/march-a.jpg", "/march-b.jpg", "/february.jpg", "/december.jpg"]
    assert model.paths_for_sections({"year:2026"}) == ["/march-a.jpg", "/march-b.jpg", "/february.jpg"]
    assert [model.data(model.index(row, 0), model.HeaderRole).section_id for row in range(model.rowCount()) if model.data(model.index(row, 0), model.HeaderRole)] == [
        "year:2026",
        "month:2026-03",
        "month:2026-02",
        "year:2025",
    ]
    assert model.rowCount() == 5  # 2026 header, March header/grid, February header, 2025 header


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


def test_sectioned_gallery_requests_lazy_face_tools_before_opening_editor() -> None:
    app = QApplication.instance() or QApplication([])
    from ui.sectioned_gallery import SectionedGallery

    gallery = SectionedGallery()
    requested: list[str] = []
    ready_callbacks: list[object] = []
    opened: list[tuple[str, bool]] = []

    def _request(path: str, ready, _failed) -> None:
        requested.append(path)
        ready_callbacks.append(ready)

    gallery.face_edit_request_handler = _request
    gallery._open_photo_for_path = lambda path, *, allow_face_edit: opened.append((path, allow_face_edit))  # type: ignore[method-assign]

    gallery._request_face_edit("/photo.jpg")

    assert requested == ["/photo.jpg"]
    assert gallery.status_label.text() == "Preparing face tools for this photo…"
    ready_callbacks[0]()
    assert opened == [("/photo.jpg", True)]
    gallery.close()
    app.processEvents()


def test_large_section_preparation_discards_stale_worker_result() -> None:
    app = QApplication.instance() or QApplication([])
    from threading import Event
    from time import monotonic, sleep
    from unittest.mock import patch

    from ui.sectioned_gallery import SectionedGallery

    gallery = SectionedGallery()
    gallery.ASYNC_PREPARE_PATH_THRESHOLD = 1
    gallery._queue_visible_loads = lambda: None  # type: ignore[method-assign]
    first_started = Event()
    release_first = Event()
    original_prepare = SectionedGalleryModel.prepare_sections
    calls = 0

    def _delayed_prepare(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            first_started.set()
            release_first.wait(timeout=5.0)
        return original_prepare(*args, **kwargs)

    def _wait_for(predicate) -> None:
        deadline = monotonic() + 5.0
        while monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            sleep(0.01)
        assert predicate()

    try:
        with patch.object(SectionedGalleryModel, "prepare_sections", side_effect=_delayed_prepare):
            gallery.set_sections([GallerySection("old", ("/old.jpg",))])
            _wait_for(first_started.is_set)
            gallery.set_sections([GallerySection("new", ("/new.jpg",))])
            _wait_for(lambda: gallery._model.all_paths() == ["/new.jpg"])
            release_first.set()
            _wait_for(lambda: gallery._section_prepare_thread is None)
            assert gallery._model.all_paths() == ["/new.jpg"]
    finally:
        release_first.set()
        gallery.close()
        app.processEvents()


def test_large_section_resize_prepares_reflow_off_the_ui_thread() -> None:
    app = QApplication.instance() or QApplication([])
    from time import monotonic, sleep

    from ui.sectioned_gallery import SectionedGallery

    gallery = SectionedGallery()
    gallery.ASYNC_PREPARE_PATH_THRESHOLD = 1
    gallery._queue_visible_loads = lambda: None  # type: ignore[method-assign]
    paths = tuple(f"/photo-{index}.jpg" for index in range(2_200))

    def _wait_for(predicate) -> None:
        deadline = monotonic() + 5.0
        while monotonic() < deadline:
            app.processEvents()
            if predicate():
                return
            sleep(0.01)
        assert predicate()

    try:
        gallery.resize(1_120, 700)
        gallery.show()
        app.processEvents()
        gallery.set_sections([GallerySection("large", paths)])
        _wait_for(lambda: gallery._model.all_paths() == list(paths))

        selected_path = paths[100]
        row, column = gallery._model._path_locations[selected_path]  # noqa: SLF001 - model owns stable path locations
        gallery.table.selectionModel().select(
            gallery._model.index(row, column),
            QItemSelectionModel.SelectionFlag.Select,
        )
        old_columns = gallery._model.columnCount()

        gallery.resize(500, 700)
        app.processEvents()
        expected_columns = max(1, gallery.table.viewport().width() // max(120, gallery._tile_size + 18))
        assert expected_columns != old_columns
        assert gallery._section_prepare_job is not None
        _wait_for(lambda: gallery._model.columnCount() == expected_columns)
        _wait_for(
            lambda: selected_path
            in {
                gallery._model.path_at(index)
                for index in gallery.table.selectionModel().selectedIndexes()
            }
        )
        assert gallery._model.all_paths() == list(paths)
    finally:
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


def test_viewport_thumbnail_progress_uses_distinct_units_and_workspace_owner() -> None:
    app = QApplication.instance() or QApplication([])
    from ui.job_manager import JobManager
    from ui.sectioned_gallery import SectionedGallery

    manager = JobManager()
    gallery = SectionedGallery()
    gallery.set_job_manager(manager, origin="Library")
    try:
        gallery._begin_viewport_task(1, "/photos/a.jpg", "thumbnail")
        gallery._begin_viewport_task(1, "/photos/b.jpg", "thumbnail")
        job_id = gallery._viewport_job_id
        assert job_id is not None
        assert manager.get(job_id).origin == "Library"
        assert gallery.status_label.text() == "Loading thumbnails 0/2 visible"

        gallery._complete_viewport_task(1, "/photos/a.jpg", "thumbnail")
        gallery._complete_viewport_task(1, "/photos/b.jpg", "thumbnail")

        assert gallery.status_label.text() == "Thumbnails ready 2/2."
        assert "Loaded" not in gallery.status_label.text()
        assert manager.get(job_id).status == "finished"

        gallery._begin_viewport_task(1, "/photos/a.jpg", "label")
        assert gallery._viewport_job_id is None
        assert gallery._viewport_tasks == set()
    finally:
        gallery.close()
        app.processEvents()
