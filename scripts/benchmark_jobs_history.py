#!/usr/bin/env python3
"""Measure bounded incremental refresh of the 500-row Jobs history table.

The fixture is entirely in memory. It creates no source media, database,
models, cache, settings, or network connections. A JobManager signal drives
the real JobsDialog refresh path so the samples include changed-row detection
and the Qt table update that a normal progress callback uses.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from ui.job_manager import JobManager
from ui.job_widgets import JobsDialog


def _percentile(samples: list[float], percentile: float) -> float:
    ordered = sorted(samples)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(description="Generated 500-row Jobs history refresh benchmark.")
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--updates", type=int, default=100)
    args = parser.parse_args()
    rows = max(10, int(args.rows))
    updates = max(1, int(args.updates))
    app = QApplication.instance() or QApplication([])
    manager = JobManager(max_history=rows)
    active_ids: list[int] = []
    for index in range(rows):
        job_id = manager.register_job(
            f"Generated task {index:03d}",
            cancel_fn=(lambda: None) if index % 10 == 0 else None,
            origin="Benchmark",
            foreground=index % 2 == 0,
        )
        if index % 4 == 0:
            active_ids.append(job_id)
            manager.update(job_id, progress=index % 100, text=f"{index}/500 generated assets")
        else:
            manager.finish(job_id)
    dialog = JobsDialog(manager)
    dialog.resize(1280, 720)
    dialog.show()
    app.processEvents()
    try:
        initial_started = time.perf_counter()
        dialog.refresh()
        initial_ms = (time.perf_counter() - initial_started) * 1000.0
        if dialog.table.rowCount() != rows:
            raise RuntimeError("Jobs dialog did not retain the generated 500-row history")
        if dialog.table.columnWidth(6) < 260:
            raise RuntimeError("Jobs details column is not readable at the required minimum width")

        samples: list[float] = []
        for update_index in range(updates):
            job_id = active_ids[update_index % len(active_ids)]
            started = time.perf_counter()
            manager.update(job_id, progress=update_index % 101, text=f"refresh {update_index:03d}")
            samples.append((time.perf_counter() - started) * 1000.0)
            app.processEvents()
        current_row = dialog._job_ids.index(active_ids[0])
        detail = dialog.table.item(current_row, 6)
        if detail is None or not detail.text().startswith("refresh"):
            raise RuntimeError("Changed job detail was not incrementally refreshed")
        print(
            json.dumps(
                {
                    "operation": "jobs_history_incremental_refresh",
                    "fixture": {"rows": rows, "updates": updates, "source": "generated-in-memory-no-io"},
                    "initial_refresh_ms": round(initial_ms, 3),
                    "changed_row_refresh_ms": {
                        "p50": round(statistics.median(samples), 3),
                        "p95": round(_percentile(samples, 0.95), 3),
                        "samples": [round(sample, 3) for sample in samples],
                    },
                    "details_column_width": dialog.table.columnWidth(6),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        dialog.close()
        app.processEvents()


if __name__ == "__main__":
    raise SystemExit(main())
