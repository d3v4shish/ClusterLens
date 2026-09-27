#!/usr/bin/env python3
"""Measure worker preparation and Qt commit for a large sectioned gallery.

The fixture is generated in memory: it opens no source image, thumbnail,
database, model, cache, network, or user-runtime data.  It measures only the
presentation hierarchy used by Library Timeline and duplicate previews.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO_ROOT = Path(__file__).resolve().parents[1]
for import_path in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if import_path not in sys.path:
        sys.path.insert(0, import_path)

from PyQt6.QtCore import QItemSelectionModel
from PyQt6.QtWidgets import QApplication

from apps.shared.profile_support import profile_call
from ui.sectioned_gallery import GallerySection, SectionedGallery, SectionedGalleryModel


class _EventPump:
    def __init__(self, app: QApplication) -> None:
        self.app = app
        self.last_tick = time.perf_counter()
        self.max_gap_ms = 0.0

    def pump(self) -> None:
        now = time.perf_counter()
        self.max_gap_ms = max(self.max_gap_ms, (now - self.last_tick) * 1000.0)
        self.last_tick = now
        self.app.processEvents()

    def wait_until(self, predicate, *, timeout_s: float, label: str) -> None:
        deadline = time.perf_counter() + timeout_s
        while time.perf_counter() < deadline:
            self.pump()
            if predicate():
                return
            time.sleep(0.001)
        self.pump()
        if not predicate():
            raise RuntimeError(f"Timed out waiting for {label}")


def _sections(paths: int, section_size: int) -> tuple[GallerySection, ...]:
    sections: list[GallerySection] = []
    for section_index, start in enumerate(range(0, paths, section_size)):
        end = min(paths, start + section_size)
        section_paths = tuple(f"/generated/section-{section_index:04d}/photo-{index:06d}.jpg" for index in range(start, end))
        sections.append(GallerySection(f"section:{section_index:04d}", section_paths, title=f"Section {section_index:04d}"))
    return tuple(sections)


def _percentiles(samples: list[float]) -> dict[str, object]:
    ordered = sorted(samples)
    return {
        "samples": [round(value, 3) for value in samples],
        "p50": round(statistics.median(samples), 3),
        "p95": round(ordered[max(0, int(round((len(ordered) - 1) * 0.95)))], 3),
    }


def _prepare_once(sections: tuple[GallerySection, ...], columns: int) -> int:
    prepared = SectionedGalleryModel.prepare_sections(sections, columns=columns)
    return len(prepared.all_paths)


def _measure_worker_preparation(sections: tuple[GallerySection, ...], *, columns: int, repeats: int) -> tuple[dict[str, object], list[dict[str, object]]]:
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = _prepare_once(sections, columns)
        samples.append((time.perf_counter() - started) * 1000.0)
        if result != sum(len(section.paths) for section in sections):
            raise RuntimeError("Prepared section hierarchy lost generated paths")
    _result, profile = profile_call(lambda: _prepare_once(sections, columns))
    return _percentiles(samples), profile.hotspots


def _measure_qt_publication(
    app: QApplication,
    sections: tuple[GallerySection, ...],
    *,
    expected_paths: int,
) -> dict[str, object]:
    gallery = SectionedGallery()
    gallery.resize(1280, 720)
    gallery.show()
    gallery._queue_visible_loads = lambda: None  # type: ignore[method-assign]
    app.processEvents()
    pump = _EventPump(app)
    commits: list[float] = []
    original_publish = gallery._publish_prepared_sections

    def _measured_publish(*args, **kwargs):
        started = time.perf_counter()
        result = original_publish(*args, **kwargs)
        commits.append((time.perf_counter() - started) * 1000.0)
        return result

    gallery._publish_prepared_sections = _measured_publish  # type: ignore[method-assign]
    try:
        started = time.perf_counter()
        gallery.set_sections(list(sections), status="Publishing generated section hierarchy")
        pump.wait_until(
            lambda: len(gallery._model.all_paths()) == expected_paths and bool(commits),
            timeout_s=30.0,
            label="the worker-prepared 100k section hierarchy",
        )
        end_to_end_ms = (time.perf_counter() - started) * 1000.0
        # The selection survives when the same stable path appears in a
        # hierarchy replacement. This uses the public QItemSelectionModel.
        selected_path = gallery._model.all_paths()[0]
        location = gallery._model._path_locations[selected_path]  # noqa: SLF001 - stable model identity under test
        index = gallery._model.index(*location)
        gallery.table.selectionModel().select(index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        gallery.set_sections(list(sections), status="Replacing generated section hierarchy")
        pump.wait_until(
            lambda: len(commits) == 2 and selected_path in {
                gallery._model.path_at(item)
                for item in gallery.table.selectionModel().selectedIndexes()
            },
            timeout_s=30.0,
            label="selection-preserving hierarchy replacement",
        )
        return {
            "end_to_end_worker_to_ui_ms": round(end_to_end_ms, 3),
            "ui_commit_ms": round(commits[0], 3),
            "selection_preserved_on_replacement": True,
            "max_event_loop_gap_ms": round(pump.max_gap_ms, 3),
            "rows": gallery._model.rowCount(),
            "columns": gallery._model.columnCount(),
        }
    finally:
        gallery.close()
        pump.pump()


def _measure_replacement_memory(sections: tuple[GallerySection, ...]) -> dict[str, object]:
    """Trace repeated full/filtered publication without retaining generations."""

    model = SectionedGalleryModel()
    full_samples: list[int] = []
    filtered_samples: list[int] = []
    peaks: list[int] = []
    tracemalloc.start()
    try:
        for cycle in range(6):
            filtered = cycle % 2 == 1
            cycle_sections = sections[::2] if filtered else sections
            prepared = SectionedGalleryModel.prepare_sections(
                cycle_sections,
                columns=6 if cycle % 3 else 5,
            )
            model.apply_prepared_sections(prepared)
            del prepared
            gc.collect()
            retained, peak = tracemalloc.get_traced_memory()
            expected = sum(len(section.paths) for section in cycle_sections)
            if len(model.all_paths()) != expected:
                raise RuntimeError("Repeated section publication changed membership")
            target = filtered_samples if filtered else full_samples
            target.append(int(retained))
            peaks.append(int(peak))
    finally:
        tracemalloc.stop()
    return {
        "cycles": 6,
        "pattern": "full/filtered section replacement with 5/6-column reflow",
        "full_retained_python_bytes": full_samples,
        "filtered_retained_python_bytes": filtered_samples,
        "full_first_to_last_growth_bytes": full_samples[-1] - full_samples[0],
        "filtered_first_to_last_growth_bytes": filtered_samples[-1] - filtered_samples[0],
        "peak_python_bytes": max(peaks),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generated 100k SectionedGallery publication benchmark.")
    parser.add_argument("--paths", type=int, default=100_000)
    parser.add_argument("--section-size", type=int, default=1_000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.paths < 1 or args.section_size < 1 or args.repeats < 1:
        parser.error("paths, section-size, and repeats must be positive")

    app = QApplication.instance() or QApplication([])
    sections = _sections(int(args.paths), int(args.section_size))
    expected_paths = sum(len(section.paths) for section in sections)
    worker_preparation, hotspots = _measure_worker_preparation(sections, columns=5, repeats=int(args.repeats))
    qt_publication = _measure_qt_publication(app, sections, expected_paths=expected_paths)
    replacement_memory = _measure_replacement_memory(sections)
    if expected_paths != int(args.paths) or qt_publication["rows"] <= 0:
        raise RuntimeError("Generated section fixture did not publish the requested hierarchy")
    print(
        json.dumps(
            {
                "operation": "sectioned_gallery_worker_publication",
                "fixture": {
                    "paths": int(args.paths),
                    "sections": len(sections),
                    "section_size": int(args.section_size),
                    "source": "generated-in-memory-no-io",
                },
                "worker_preparation_ms": worker_preparation,
                "ui_publication": qt_publication,
                "replacement_memory": replacement_memory,
                "profile_hotspots": hotspots,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
