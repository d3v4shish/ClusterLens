#!/usr/bin/env python3
"""Deterministic autocomplete regression benchmark.

The benchmark does not query a database or user media.  It measures the shared
name/entity picker against generated saved names so a future UI change cannot
accidentally turn a normal type-ahead interaction into an unbounded Qt-thread
operation.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from apps.shared.profile_support import profile_call
from ui.entity_picker import EntityPicker


def _names(count: int) -> list[str]:
    return [f"Person {index:05d}" for index in range(count)]


def _run_once(names: list[str], query: str) -> tuple[float, int]:
    picker = EntityPicker(entity_label="person name")
    started = time.perf_counter()
    picker.set_choices(names)
    picker._update_suggestions(query)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    matches = picker._name_model.rowCount()
    picker.deleteLater()
    return elapsed_ms, int(matches)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark shared EntityPicker autocomplete.")
    parser.add_argument("--names", type=int, default=10_000)
    parser.add_argument("--query", default="Person 09")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if args.names < 1 or args.repeats < 1:
        parser.error("names and repeats must be positive")

    _app = QApplication.instance() or QApplication([])
    names = _names(args.names)
    samples: list[float] = []
    match_count = 0
    for _ in range(args.repeats):
        elapsed_ms, match_count = _run_once(names, str(args.query))
        samples.append(elapsed_ms)
    _result, profile = profile_call(lambda: _run_once(names, str(args.query)))

    if match_count < 1:
        raise RuntimeError("Generated EntityPicker fixture returned no matches")
    print(
        json.dumps(
            {
                "operation": "entity_picker_saved_name_filter",
                "fixture": {"names": args.names, "query": str(args.query), "source": "generated-in-memory"},
                "repeats": args.repeats,
                "timings_ms": [round(value, 3) for value in samples],
                "median_ms": round(statistics.median(samples), 3),
                "match_count": match_count,
                "profile_hotspots": profile.hotspots,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
