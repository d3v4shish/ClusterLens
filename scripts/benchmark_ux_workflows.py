#!/usr/bin/env python3
"""Seeded 500/10,000-row virtual UI publication baseline.

No files, catalog, model, thumbnail, or user runtime data are opened.  This
measures only model-backed publication and a visible first-page query, the
common path used by Gallery, Faces, Names, Tags, and Library.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from apps.shared.profile_support import current_memory_rss_mb, profile_call
from ui.list_models import ListEntry, PagedListEntryModel


def _items(count: int) -> list[ListEntry]:
    return [ListEntry(title=f"Photo {index:05d}", subtitle=f"Generated item {index}", payload=index) for index in range(count)]


def _sample(count: int, repeats: int) -> dict[str, object]:
    items = _items(count)
    first_content_samples: list[float] = []
    publish_samples: list[float] = []
    for _ in range(repeats):
        model = PagedListEntryModel(page_size=500)
        started = time.perf_counter()
        model.set_source_items(items)
        first_content_samples.append((time.perf_counter() - started) * 1000.0)
        started = time.perf_counter()
        while model.canFetchMore():
            model.fetchMore()
        publish_samples.append((time.perf_counter() - started) * 1000.0)
        assert model.rowCount() == count
        model.deleteLater()
    _result, profile = profile_call(lambda: _profile_once(items))
    return {
        "assets": count,
        "first_content_ms": _percentiles(first_content_samples),
        "full_publish_ms": _percentiles(publish_samples),
        "queue_delay_ms": 0.0,
        "cache_behavior": "not-applicable: generated in-memory rows",
        "io_behavior": "none",
        "cpu_gpu_fallback": "not-applicable: no compute backend",
        "profile_hotspots": profile.hotspots,
    }


def _profile_once(items: list[ListEntry]) -> int:
    model = PagedListEntryModel(page_size=500)
    model.set_source_items(items)
    while model.canFetchMore():
        model.fetchMore()
    return model.rowCount()


def _percentiles(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "p50": round(float(statistics.median(ordered)), 3),
        "p95": round(float(ordered[max(0, int(round((len(ordered) - 1) * 0.95)))]), 3),
        "samples": [round(value, 3) for value in samples],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3 for p50/p95 evidence")
    _app = QApplication.instance() or QApplication([])
    result = {
        "operation": "virtual_ui_publication",
        "fixture": "seeded generated in-memory model rows",
        "rss_mb": current_memory_rss_mb(),
        "runs": [_sample(500, args.repeats), _sample(10_000, args.repeats)],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
