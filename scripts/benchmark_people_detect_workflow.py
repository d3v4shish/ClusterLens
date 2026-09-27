#!/usr/bin/env python3
"""Exercise the production People / Detect review workflow at a fixed scale.

This deterministic, temporary fixture seeds 11,284 source-image paths and
14,477 indexed faces, then drives the real ``SearchPane`` through its normal
Qt jobs.  It keeps model inference and crop decoding out of scope on purpose:
those have independent benchmark fixtures.  The output separates SQLite and
review-metadata work from the UI publication and viewport-tail handoff.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import logging
import os
import pstats
import sys
import time
import tracemalloc
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event

# This script intentionally runs without a desktop session.  It must happen
# before importing PyQt through SearchPane.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_path in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from PyQt6.QtWidgets import QApplication

from app.services.face_search import FACE_REVIEW_PAGE_MAX_IMAGES, FaceIndexService
from infra.cancel import Cancelled
from ui.search_pane import SearchPane

# The product services log each deterministic page and tile batch at INFO.
# Keep the benchmark report machine-readable while preserving warnings/errors.
logging.getLogger("ui.search_pane").setLevel(logging.WARNING)
logging.getLogger("app.services.face_search").setLevel(logging.WARNING)


# A valid 1x1 PNG.  Every fixture source image contains the same deterministic
# bytes; the benchmark is about review state and UI publication, not decoding.
_FIXTURE_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\rIDATx\x9cc````\x00\x00\x00\x05\x00\x01"
    b"\xa5\xf6E@\x00\x00\x00\x00IEND\xaeB`\x82"
)


class _GatedFaceIndexService(FaceIndexService):
    """A real review service that can pause one background page on demand."""

    def __init__(self, *args, gate_background_pages: bool, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gate_background_pages = bool(gate_background_pages)
        self.background_page_started = Event()
        self.release_background_pages = Event()
        self.cancel_observed = Event()
        self.page_offsets: list[int] = []

    def load_folder_review_image_page(self, directory: str, *, offset: int = 0, cancel_check=None, **kwargs):
        page_offset = int(offset)
        self.page_offsets.append(page_offset)
        if self.gate_background_pages and page_offset >= FACE_REVIEW_PAGE_MAX_IMAGES:
            self.background_page_started.set()
            while not self.release_background_pages.wait(0.005):
                if callable(cancel_check) and cancel_check():
                    self.cancel_observed.set()
                    raise Cancelled()
        return super().load_folder_review_image_page(
            directory,
            offset=page_offset,
            cancel_check=cancel_check,
            **kwargs,
        )


class _EventPump:
    """Pump Qt while recording the largest observed gap between iterations."""

    def __init__(self, app: QApplication) -> None:
        self.app = app
        self._last_tick = time.perf_counter()
        self.max_gap_ms = 0.0

    def pump(self) -> None:
        now = time.perf_counter()
        self.max_gap_ms = max(self.max_gap_ms, (now - self._last_tick) * 1000.0)
        self._last_tick = now
        self.app.processEvents()

    def wait_until(self, predicate, *, timeout_s: float, label: str) -> None:
        deadline = time.perf_counter() + float(timeout_s)
        while time.perf_counter() < deadline:
            self.pump()
            if predicate():
                return
            time.sleep(0.001)
        self.pump()
        if not predicate():
            raise RuntimeError(f"Timed out while waiting for {label}")

    def settle(self, *, duration_s: float) -> None:
        deadline = time.perf_counter() + float(duration_s)
        while time.perf_counter() < deadline:
            self.pump()
            time.sleep(0.001)


def _create_source_fixture(root: Path, photos: int) -> list[str]:
    source_root = root / "photos"
    source_root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for index in range(photos):
        path = source_root / f"review-{index:05d}.png"
        path.write_bytes(_FIXTURE_PNG)
        paths.append(str(path))
    return paths


def _seed_index(service: FaceIndexService, paths: list[str], faces: int) -> None:
    if faces < len(paths):
        raise ValueError("faces must be at least the number of photos")
    extra_faces = faces - len(paths)
    face_rows: list[tuple[object, ...]] = []
    scan_rows: list[tuple[object, ...]] = []
    for image_index, path in enumerate(paths):
        face_count = 1 + int(image_index < extra_faces)
        scan_rows.append((path, 1, 1, face_count, 100, 100))
        for face_index in range(face_count):
            face_rows.append(
                (
                    path,
                    face_index,
                    "[10, 10, 50, 60]",
                    0.99,
                    b"fixture-embedding",
                    "clean",
                    1.0,
                    "[]",
                    1,
                    0,
                    0,
                    1,
                    1,
                )
            )
    with service._connect() as connection:  # Fixture setup only; workflow uses public APIs.
        connection.executemany(
            """
            INSERT INTO face_index(
                image_path, face_index, bbox_json, face_confidence, embedding,
                quality_status, quality_score, quality_reasons_json,
                quality_revision, is_tiny, hidden, mtime_ns, file_size
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            face_rows,
        )
        connection.executemany(
            """
            INSERT INTO face_scan_images(
                image_path, mtime_ns, file_size, face_count, image_width, image_height
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            scan_rows,
        )


def _new_pane(service: FaceIndexService, root: Path) -> SearchPane:
    pane = SearchPane(
        enabled_tabs=["Face Library"],
        external_results=False,
        face_service_global=service,
        face_service_session=service,
    )
    pane.current_scope_roots_provider = lambda: (str(root),)
    pane.show()
    pane.face_folder_path.setText(str(root))
    return pane


def _start_detect_review(pane: SearchPane) -> None:
    pane._request_face_library_refresh(
        refresh_people=False,
        reason="benchmark People Detect workflow",
        force_refresh=True,
    )


def _run_cancel_case(
    app: QApplication,
    root: Path,
    paths: list[str],
    faces: int,
) -> dict[str, float | int]:
    service = _GatedFaceIndexService(db_path=root / "cancel.sqlite3", gate_background_pages=True)
    _seed_index(service, paths, faces)
    pane = _new_pane(service, root)
    pump = _EventPump(app)
    try:
        _start_detect_review(pane)
        pump.wait_until(
            lambda: len(pane.results_gallery.images) == min(len(paths), FACE_REVIEW_PAGE_MAX_IMAGES)
            and service.background_page_started.is_set(),
            timeout_s=30.0,
            label="the cancellable initial review page",
        )
        visible_before_cancel = list(pane.results_gallery.images)
        cancel_started = time.perf_counter()
        pane._cancel_face_review_stream(replaced=True)
        pump.wait_until(
            service.cancel_observed.is_set,
            timeout_s=5.0,
            label="background cache cancellation acknowledgement",
        )
        cancel_acknowledged_ms = (time.perf_counter() - cancel_started) * 1000.0
        service.release_background_pages.set()
        pump.settle(duration_s=0.1)
        if list(pane.results_gallery.images) != visible_before_cancel:
            raise RuntimeError("Cancelled background review published stale gallery paths")
        if len(pane._face_review_all_images) != len(visible_before_cancel):
            raise RuntimeError("Cancelled background review appended stale review metadata")
        if pane.face_detected_faces_model.rowCount() != 0:
            raise RuntimeError("Cancelled background review populated hidden Faces tiles")
        return {
            "cancel_acknowledged_ms": round(cancel_acknowledged_ms, 3),
            "visible_photos_retained": len(visible_before_cancel),
            "max_event_loop_gap_ms": round(pump.max_gap_ms, 3),
        }
    finally:
        service.release_background_pages.set()
        pane.close()
        pump.settle(duration_s=0.02)


def _run_workflow(
    app: QApplication,
    root: Path,
    photos: int,
    faces: int,
    *,
    trace_metadata_memory: bool,
    profile_faces_publication: bool = False,
) -> dict[str, object]:
    paths = _create_source_fixture(root, photos)
    service = _GatedFaceIndexService(db_path=root / "faces.sqlite3", gate_background_pages=True)
    _seed_index(service, paths, faces)
    pane = _new_pane(service, root / "photos")
    pump = _EventPump(app)
    try:
        if trace_metadata_memory:
            tracemalloc.start()
        refresh_started = time.perf_counter()
        _start_detect_review(pane)
        pump.wait_until(
            lambda: len(pane.results_gallery.images) == min(photos, FACE_REVIEW_PAGE_MAX_IMAGES)
            and service.background_page_started.is_set(),
            timeout_s=30.0,
            label="first visible Detect photos and background-cache handoff",
        )
        first_visible_photos_ms = (time.perf_counter() - refresh_started) * 1000.0
        if pane.face_detected_faces_model.rowCount() != 0:
            raise RuntimeError("Hidden Faces model was populated while Photos remained active")
        if not pane._face_review_stream_in_progress:
            raise RuntimeError("Large review stopped streaming before its metadata cache completed")

        cache_release_started = time.perf_counter()
        service.release_background_pages.set()
        pump.wait_until(
            lambda: pane._face_review_cache_complete
            and not pane._face_review_cache_in_progress
            and len(pane._face_review_all_images) == photos,
            timeout_s=60.0,
            label="all Detect review metadata",
        )
        cache_metadata_ms = (time.perf_counter() - cache_release_started) * 1000.0
        retained_bytes = peak_bytes = 0
        if trace_metadata_memory:
            retained_bytes, peak_bytes = tracemalloc.get_traced_memory()
            tracemalloc.stop()
        if pane.face_detected_faces_model.rowCount() != 0:
            raise RuntimeError("Background metadata cache populated hidden Faces tiles")

        faces_publish_started = time.perf_counter()
        faces_profile = cProfile.Profile()
        if profile_faces_publication:
            faces_profile.enable()
        pane.face_review_results_tabs.setCurrentWidget(pane.face_detected_faces_panel)
        pump.wait_until(
            lambda: pane.face_detected_faces_model.rowCount() == faces
            and not pane._face_detected_publish_in_progress,
            timeout_s=60.0,
            label="visible Faces-tab publication",
        )
        if profile_faces_publication:
            faces_profile.disable()
        faces_publish_ms = (time.perf_counter() - faces_publish_started) * 1000.0
        faces_profile_hotspots: list[dict[str, object]] = []
        if profile_faces_publication:
            for (filename, line, function), (_cc, calls, own, cumulative, _callers) in pstats.Stats(
                faces_profile
            ).stats.items():
                faces_profile_hotspots.append(
                    {
                        "function": f"{Path(filename).name}:{line}:{function}",
                        "calls": int(calls),
                        "own_seconds": round(float(own), 6),
                        "cumulative_seconds": round(float(cumulative), 6),
                    }
                )
            faces_profile_hotspots.sort(key=lambda item: float(item["cumulative_seconds"]), reverse=True)
            faces_profile_hotspots = faces_profile_hotspots[:20]
        gallery_before_tail = list(pane.results_gallery.images)
        if len(gallery_before_tail) != min(photos, FACE_REVIEW_PAGE_MAX_IMAGES):
            raise RuntimeError("Opening Faces changed the bounded Photos working set")

        pane.face_review_results_tabs.setCurrentWidget(pane.results_gallery)
        tail_reveal_started = time.perf_counter()
        pane._on_results_gallery_visible_paths_changed([pane.results_gallery.images[-1]])
        expected_tail_size = min(photos, FACE_REVIEW_PAGE_MAX_IMAGES * 2)
        pump.wait_until(
            lambda: len(pane.results_gallery.images) == expected_tail_size,
            timeout_s=15.0,
            label="one cached Photos tail page",
        )
        tail_reveal_ms = (time.perf_counter() - tail_reveal_started) * 1000.0
        if pane.face_detected_faces_model.rowCount() != faces:
            raise RuntimeError("Photos tail handoff invalidated the already published Faces model")
        if service.page_offsets.count(FACE_REVIEW_PAGE_MAX_IMAGES) != 1:
            raise RuntimeError("Cached tail reveal re-queried the second review page")

        cancellation = _run_cancel_case(app, root, paths, faces)
        return {
            "fixture": {
                "photos": photos,
                "faces": faces,
                "source": "temporary-1px-png-files-and-seeded-sqlite",
            },
            "page_size": FACE_REVIEW_PAGE_MAX_IMAGES,
            "first_visible_photos_ms": round(first_visible_photos_ms, 3),
            "background_metadata_cache_ms": round(cache_metadata_ms, 3),
            "faces_tab_publication_ms": round(faces_publish_ms, 3),
            "faces_publication_profile_hotspots": faces_profile_hotspots,
            "cached_tail_reveal_ms": round(tail_reveal_ms, 3),
            "retained_python_bytes_after_metadata_cache": int(retained_bytes),
            "peak_python_bytes_through_metadata_cache": int(peak_bytes),
            "observed_max_event_loop_gap_ms": round(pump.max_gap_ms, 3),
            "visible_photos_before_tail": len(gallery_before_tail),
            "visible_photos_after_one_tail": len(pane.results_gallery.images),
            "metadata_reviewed_photos": len(pane._face_review_all_images),
            "published_face_tiles": int(pane.face_detected_faces_model.rowCount()),
            "service_page_offsets": list(service.page_offsets),
            "cancellation": cancellation,
            "scope": {
                "included": "SQLite source selection, candidate snapshot, page queries, Qt Photos publication, metadata cache, Faces tile model publication, cached tail reveal, cancellation acknowledgement",
                "excluded": "model inference, embedding, full-resolution thumbnail/crop decoding; those are benchmarked independently",
            },
        }
    finally:
        service.release_background_pages.set()
        if trace_metadata_memory and tracemalloc.is_tracing():
            tracemalloc.stop()
        pane.close()
        pump.settle(duration_s=0.02)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generated People / Detect Qt workflow benchmark.")
    parser.add_argument("--photos", type=int, default=11_284)
    parser.add_argument("--faces", type=int, default=14_477)
    args = parser.parse_args()
    if args.photos < 2 or args.faces < args.photos:
        parser.error("photos must be at least 2 and faces must be at least photos")

    app = QApplication.instance() or QApplication([])
    with TemporaryDirectory(prefix="clusterlens-people-detect-workflow-") as temporary:
        temporary_root = Path(temporary)
        report = _run_workflow(
            app,
            temporary_root / "timed",
            int(args.photos),
            int(args.faces),
            trace_metadata_memory=False,
        )
        memory_report = _run_workflow(
            app,
            temporary_root / "memory",
            int(args.photos),
            int(args.faces),
            trace_metadata_memory=True,
        )
        profile_report = _run_workflow(
            app,
            temporary_root / "profile",
            int(args.photos),
            int(args.faces),
            trace_metadata_memory=False,
            profile_faces_publication=True,
        )
        report["retained_python_bytes_after_metadata_cache"] = int(
            memory_report["retained_python_bytes_after_metadata_cache"]
        )
        report["peak_python_bytes_through_metadata_cache"] = int(
            memory_report["peak_python_bytes_through_metadata_cache"]
        )
        report["memory_measurement"] = "separate traced workflow; timings are from the untraced workflow"
        report["faces_publication_profile_hotspots"] = profile_report["faces_publication_profile_hotspots"]
        report["profile_measurement"] = "separate profiled workflow; profiled timing is not reported"
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
