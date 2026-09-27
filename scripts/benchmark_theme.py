#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
import tracemalloc

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import PYQT_VERSION_STR, QT_VERSION_STR
from PyQt6.QtGui import QColor, QImage
from PyQt6.QtWidgets import QApplication

from ui.theme import apply_app_theme
from ui.zoomable_image import adaptive_neutral_backdrop


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return float(ordered[index])


def _image(color: QColor) -> QImage:
    image = QImage(320, 240, QImage.Format.Format_RGBA8888)
    image.fill(color)
    return image


def main() -> int:
    parser = argparse.ArgumentParser(description="Deterministic ClusterLens theme and viewer-canvas microbenchmark")
    parser.add_argument("--iterations", type=int, default=250)
    args = parser.parse_args()
    iterations = max(10, int(args.iterations))
    app = QApplication.instance() or QApplication([])
    samples = (
        _image(QColor("#050607")),
        _image(QColor("#B0B0B0")),
        _image(QColor("#FFFFFF")),
        _image(QColor(255, 255, 255, 0)),
    )

    for image in samples:
        adaptive_neutral_backdrop(image)
    apply_app_theme(app, "dark")

    tracemalloc.start()
    classify_ms: list[float] = []
    classifications: list[str | None] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        classifications = [adaptive_neutral_backdrop(image) for image in samples]
        classify_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    _current, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    switch_ms: list[float] = []
    for index in range(iterations):
        started = time.perf_counter_ns()
        apply_app_theme(app, "light" if index % 2 == 0 else "dark")
        switch_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)
    apply_app_theme(app, "dark")

    # Reapplying an already resolved theme is a common Settings/shell path.
    # It must not rebuild the global stylesheet or refresh every icon.
    reapply_ms: list[float] = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        apply_app_theme(app, "dark")
        reapply_ms.append((time.perf_counter_ns() - started) / 1_000_000.0)

    result = {
        "benchmark": "theme-v1",
        "inputs": {
            "classifier_images": len(samples),
            "classifier_sample_size": "32x32 border",
            "iterations": iterations,
        },
        "environment": {
            "python": platform.python_version(),
            "qt": QT_VERSION_STR,
            "pyqt": PYQT_VERSION_STR,
            "platform": platform.system().lower(),
        },
        "viewer_classifier": {
            "classifications": classifications,
            "batch_median_ms": round(statistics.median(classify_ms), 4),
            "batch_p95_ms": round(_percentile(classify_ms, 0.95), 4),
            "per_image_median_ms": round(statistics.median(classify_ms) / len(samples), 4),
            "python_peak_bytes": int(peak_bytes),
        },
        "theme_switch": {
            "median_ms": round(statistics.median(switch_ms), 4),
            "p95_ms": round(_percentile(switch_ms, 0.95), 4),
        },
        "theme_reapply_noop": {
            "median_ms": round(statistics.median(reapply_ms), 4),
            "p95_ms": round(_percentile(reapply_ms, 0.95), 4),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
