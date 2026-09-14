#!/usr/bin/env python3
"""Measure deterministic publication of the transient sectioned Faces grid.

This benchmark intentionally creates no photos, thumbnails, databases, models, or
GPU work. It measures only Qt-model row construction for an already computed
similarity arrangement.
"""

from __future__ import annotations

import argparse
import json
import os
from statistics import median
from time import perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication

from ui.search_pane import FaceTileItem, FaceTileSection, SectionedFaceTileModel


def _items(count: int) -> tuple[FaceTileItem, ...]:
    return tuple(
        FaceTileItem(
            image_path=f"/fixture/face-{index:05d}.jpg",
            face_index=0,
            bbox=(10, 10, 74, 82),
            title=f"Unlabeled\nface-{index:05d}.jpg",
            tooltip=f"/fixture/face-{index:05d}.jpg",
            saved_face_index=0,
        )
        for index in range(max(1, int(count)))
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--faces", type=int, default=4096)
    parser.add_argument("--columns", type=int, default=6)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    app = QApplication.instance() or QApplication([])
    items = _items(args.faces)
    sections = [FaceTileSection("similar-1", "Similar group 1", items)]
    samples_ms: list[float] = []
    row_count = 0
    for _ in range(max(1, int(args.repeats))):
        model = SectionedFaceTileModel(lambda _item, _size: None, lambda _item, _size: None, QSize(88, 88))
        started = perf_counter()
        model.set_sections(sections)
        model.set_column_count(max(1, int(args.columns)))
        app.processEvents()
        samples_ms.append(round((perf_counter() - started) * 1000.0, 3))
        row_count = int(model.rowCount())
    print(
        json.dumps(
            {
                "operation": "sectioned_faces_model_publication",
                "fixture": {"faces": len(items), "columns": max(1, int(args.columns)), "sections": 1},
                "row_count": row_count,
                "repeats": len(samples_ms),
                "samples_ms": samples_ms,
                "median_ms": round(float(median(samples_ms)), 3),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
