from __future__ import annotations

import cProfile
import importlib.util
import json
import os
import platform
import pstats
import sys
import time
from pathlib import Path
from threading import Event

from PIL import Image
from PyQt6 import sip
from PyQt6.QtCore import QCoreApplication, QEvent, QSettings, QThread, QTimer, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QAbstractScrollArea, QApplication


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from apps.pyqt_production.identity import (  # noqa: E402
    PRODUCTION_QSETTINGS_APP,
    PRODUCTION_QSETTINGS_ORG,
)
from apps.shared.runtime_support import activate_runtime_root  # noqa: E402
from infra import settings as settings_mod  # noqa: E402
from ui.async_job import AsyncJob, start_job_in_thread  # noqa: E402
from ui.theme import apply_app_theme  # noqa: E402
from ui.work_coordinator import JobSpec  # noqa: E402


BENCHMARK_SCRIPT = REPO_ROOT / "scripts" / "benchmark_production_workloads.py"
APP = QApplication.instance() or QApplication([])


def _load_workloads():
    spec = importlib.util.spec_from_file_location("ux31_production_workloads", BENCHMARK_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _wait_for(predicate, *, timeout_s: float = 10.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        APP.processEvents()
        if predicate():
            return
        QThread.yieldCurrentThread()
    APP.processEvents()
    if not predicate():
        raise AssertionError("timed out waiting for event-gated UI workload state")


def _drain_deletes() -> None:
    for _ in range(4):
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        APP.processEvents()


def test_production_ui_stays_operable_during_full_resolution_mixed_workload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workloads = _load_workloads()
    report_value = str(os.environ.get("CLUSTERLENS_UX31_REPORT") or "").strip()
    report_path = Path(report_value).expanduser().resolve() if report_value else None
    if report_path is not None and report_path.exists():
        raise AssertionError(f"refusing to overwrite UX-31 report: {report_path}")
    runtime_root = tmp_path / "runtime"
    fixture_root = tmp_path / "photos"
    paths = workloads._create_fixture(
        fixture_root,
        photos=4,
        width=3840,
        height=2160,
        seed=20260925,
    )
    monkeypatch.setenv("CLUSTERLENS_RUNTIME_ROOT", str(runtime_root))
    monkeypatch.setenv("IMAGE_CLUSTERING_APP_DIR", str(runtime_root))
    monkeypatch.setenv("CLUSTERLENS_PACKAGED_LAUNCH_SMOKE", "ux31-mixed-ui")
    settings_mod._RUNTIME_BASE_DIR = None
    store = QSettings(PRODUCTION_QSETTINGS_ORG, PRODUCTION_QSETTINGS_APP)
    settings_snapshot = {key: store.value(key) for key in store.allKeys()}
    store.clear()
    store.setValue("setup/completed", True)
    store.setValue("runtime/preferred_mode", "cpu")
    store.setValue("workspace/default_view", "library")
    store.sync()

    from app.services.thumbnails import ThumbnailService
    from app.services import face_search as _face_search  # noqa: F401 - preload outside measured work
    from apps.pyqt_production.app import ProductionClusterApp

    window = ProductionClusterApp(activate_runtime_root("Ux31MixedUiWorkload"))
    window.resize(1280, 720)
    window.show()
    window.source_pane.set_active_roots((str(fixture_root),))
    _wait_for(lambda: window._gallery_discovery_job is None)
    APP.processEvents()
    process_before = workloads._process_snapshot()
    navigation = (
        (window.library_workspace_button, "library"),
        (window.organize_workspace_button, "organize"),
        (window.people_workspace_button, "people"),
        (window.tools_workspace_button, "tools"),
    )
    cold_navigation_actions: list[dict[str, object]] = []
    for button, expected_group in navigation:
        cold_started = time.perf_counter()
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        APP.processEvents()
        assert window._workspace_group(window._active_workspace) == expected_group
        cold_navigation_actions.append(
            {
                "action": f"cold_navigate_{expected_group}",
                "elapsed_ms": (time.perf_counter() - cold_started) * 1000.0,
            }
        )

    release = Event()
    thumbnail_checkpoint = Event()
    thumbnail_release = Event()
    started = {name: Event() for name in ("crop", "catalog", "thumbnail")}
    jobs: list[AsyncJob] = []
    threads = []
    results: dict[str, object] = {}

    def _await_release(name: str, cancel_check) -> None:
        started[name].set()
        if not release.wait(10.0):
            raise RuntimeError(f"{name} workload release was not signalled")
        if cancel_check():
            return

    def _crop(progress, cancel_check):
        _await_release("crop", cancel_check)
        progress(5, "Decoding full-resolution photos")
        result = workloads._crop_pass(paths)
        progress(100, "Full-resolution crops ready")
        return result

    def _catalog(progress, cancel_check):
        _await_release("catalog", cancel_check)
        progress(5, "Scanning generated catalog")
        result = workloads._catalog_pass(paths, runtime_root / "concurrent-catalog")
        progress(100, "Catalog ready")
        return result

    def _thumbnail(progress, cancel_check):
        _await_release("thumbnail", cancel_check)
        service = ThumbnailService(qimage_cache_size=32)
        first = workloads._thumbnail_pass(paths[:1], service)
        progress(25, "First full-resolution thumbnail ready")
        thumbnail_checkpoint.set()
        if not thumbnail_release.wait(10.0):
            raise RuntimeError("thumbnail cancellation checkpoint was not released")
        progress(50, "Continuing thumbnail decode")
        tail = workloads._thumbnail_pass(paths[1:], service)
        return {"first": first, "tail": tail}

    def _submit(name: str, spec: JobSpec, fn) -> int:
        job = AsyncJob(fn)
        jobs.append(job)
        job.completed.connect(lambda value, name=name: results.__setitem__(name, value))

        def _launch(_use_cpu_fallback: bool) -> None:
            thread = start_job_in_thread(job)
            threads.append(thread)

        return window.work_coordinator.submit_async_job(spec, job, _launch)

    crop_id = _submit(
        "crop",
        JobSpec(
            "Preparing full-resolution face crops",
            origin="People",
            foreground=False,
            source_reads=tuple(str(path) for path in paths),
        ),
        _crop,
    )
    catalog_id = _submit(
        "catalog",
        JobSpec(
            "Refreshing generated catalog",
            origin="Library",
            foreground=False,
            io_bound=True,
            source_reads=(str(fixture_root),),
            data_home_write=True,
        ),
        _catalog,
    )
    thumbnail_id = _submit(
        "thumbnail",
        JobSpec(
            "Loading full-resolution thumbnails",
            origin="Organize",
            foreground=True,
            io_bound=True,
            source_reads=tuple(str(path) for path in paths),
        ),
        _thumbnail,
    )

    try:
        _wait_for(lambda: all(event.is_set() for event in started.values()))
        assert [window.job_manager.get(job_id).status for job_id in (crop_id, catalog_id, thumbnail_id)] == [
            "running",
            "running",
            "running",
        ]

        heartbeat_gaps_ms: list[float] = []
        previous_heartbeat = time.perf_counter()

        def _heartbeat() -> None:
            nonlocal previous_heartbeat
            now = time.perf_counter()
            heartbeat_gaps_ms.append((now - previous_heartbeat) * 1000.0)
            previous_heartbeat = now

        heartbeat = QTimer()
        heartbeat.setInterval(5)
        heartbeat.timeout.connect(_heartbeat)
        heartbeat.start()
        release.set()
        _wait_for(thumbnail_checkpoint.is_set)

        ui_actions: list[dict[str, object]] = []
        profiled_ui_actions: dict[str, list[dict[str, object]]] = {}

        def _profile_entries(profile: cProfile.Profile) -> list[dict[str, object]]:
            entries = []
            for (filename, line, function), (_cc, calls, own, cumulative, _callers) in pstats.Stats(
                profile
            ).stats.items():
                entries.append(
                    {
                        "function": f"{Path(filename).name}:{line}:{function}",
                        "calls": int(calls),
                        "own_seconds": round(float(own), 6),
                        "cumulative_seconds": round(float(cumulative), 6),
                    }
                )
            return sorted(
                entries,
                key=lambda item: float(item["cumulative_seconds"]),
                reverse=True,
            )[:20]

        def _record_action(label: str, action, *, profile_key: str = "") -> None:
            profile = cProfile.Profile() if profile_key else None
            if profile is not None:
                profile.enable()
            action_started = time.perf_counter()
            action()
            APP.processEvents()
            if profile is not None:
                profile.disable()
                profiled_ui_actions[profile_key] = _profile_entries(profile)
            ui_actions.append(
                {
                    "action": str(label),
                    "elapsed_ms": (time.perf_counter() - action_started) * 1000.0,
                }
            )

        visible_scroll_area_counts: list[int] = []
        for cycle in range(3):
            for button, expected_group in navigation:
                _record_action(
                    f"navigate_{expected_group}",
                    lambda button=button: QTest.mouseClick(button, Qt.MouseButton.LeftButton),
                    profile_key=(
                        "first_active_navigate_library"
                        if cycle == 0 and expected_group == "library"
                        else ""
                    ),
                )
                assert window._workspace_group(window._active_workspace) == expected_group
            _record_action(
                "resize",
                lambda cycle=cycle: window.resize(
                    1920 if cycle % 2 else 1280,
                    1080 if cycle % 2 else 720,
                ),
            )
            if window.theme_manager is not None:
                _record_action(
                    "theme_switch",
                    lambda cycle=cycle: window.theme_manager.set_preferences(
                        "light" if cycle % 2 else "dark",
                        "theme",
                    ),
                )

            def _scroll(cycle: int = cycle) -> None:
                visible_scroll_areas = [
                    scroll_area
                    for scroll_area in window.findChildren(QAbstractScrollArea)
                    if scroll_area.isVisibleTo(window)
                ]
                visible_scroll_area_counts.append(len(visible_scroll_areas))
                for scroll_area in visible_scroll_areas:
                    bar = scroll_area.verticalScrollBar()
                    bar.setValue(bar.maximum() if cycle % 2 else bar.minimum())

            _record_action("scroll_visible_surfaces", _scroll)

        _record_action(
            "open_jobs_shortcut",
            lambda: QTest.keyClick(window, Qt.Key.Key_J, Qt.KeyboardModifier.ControlModifier),
        )
        assert window.jobs_widget._jobs_dialog is not None
        assert window.jobs_widget._jobs_dialog.isVisible()
        assert window.jobs_widget.cancel_btn.isVisible()
        _record_action(
            "cancel_foreground_job",
            lambda: QTest.mouseClick(window.jobs_widget.cancel_btn, Qt.MouseButton.LeftButton),
        )
        assert window.job_manager.get(thumbnail_id).status == "cancelling"
        thumbnail_release.set()

        _wait_for(
            lambda: all(
                window.job_manager.get(job_id).status in {"finished", "failed", "cancelled"}
                for job_id in (crop_id, catalog_id, thumbnail_id)
            ),
            timeout_s=30.0,
        )
        heartbeat.stop()

        assert window.job_manager.get(crop_id).status == "finished"
        assert window.job_manager.get(catalog_id).status == "finished"
        assert window.job_manager.get(thumbnail_id).status == "cancelled"
        assert set(results) == {"crop", "catalog"}
        assert ui_actions
        assert heartbeat_gaps_ms

        serial_crop = workloads._crop_pass(paths)
        serial_catalog = workloads._catalog_pass(paths, runtime_root / "serial-catalog")
        assert results["crop"]["digest"] == serial_crop["digest"]
        assert results["catalog"]["digest"] == serial_catalog["digest"]
        assert results["catalog"]["assets"] == len(paths)

        for thread in threads:
            if sip.isdeleted(thread):
                continue
            assert thread.wait(5_000)
            assert not thread.isRunning()
        assert not window.work_coordinator._queued
        assert not window.work_coordinator._running
        assert window.job_manager.active_jobs() == []
        if report_path is not None:
            theme_profile = cProfile.Profile()
            theme_profile.enable()
            if window.theme_manager is not None:
                next_theme = "light" if window.theme_manager.resolved_theme == "dark" else "dark"
                window.theme_manager.set_preferences(next_theme, "theme")
            APP.processEvents()
            theme_profile.disable()
            theme_hotspots = _profile_entries(theme_profile)
            process_after = workloads._process_snapshot()
            report = {
                "report_version": 1,
                "operation": "ux31_production_ui_mixed_workload",
                "status": "PASS",
                "environment": {
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "qt_platform": APP.platformName(),
                },
                "fixture": {
                    "photos": len(paths),
                    "width": 3840,
                    "height": 2160,
                    "seed": 20260925,
                    "manifest": workloads._fixture_manifest(paths),
                },
                "coverage": {
                    "coordinated_workers": 3,
                    "workload_modules_preloaded": True,
                    "workspace_navigation_cycles": 3,
                    "workspace_groups": ["library", "organize", "people", "tools"],
                    "logical_sizes": [[1280, 720], [1920, 1080]],
                    "themes": ["dark", "light"],
                    "jobs_keyboard_shortcut": "Ctrl+J",
                    "cancellation_surface": "visible footer cancel button",
                    "max_visible_scroll_areas_exercised_per_cycle": max(
                        visible_scroll_area_counts,
                        default=0,
                    ),
                },
                "measurements": {
                    "ui_actions": [
                        {
                            "action": item["action"],
                            "elapsed_ms": round(float(item["elapsed_ms"]), 3),
                        }
                        for item in ui_actions
                    ],
                    "cold_navigation_actions": [
                        {
                            "action": item["action"],
                            "elapsed_ms": round(float(item["elapsed_ms"]), 3),
                        }
                        for item in cold_navigation_actions
                    ],
                    "cold_navigation_max_ms": round(
                        max(float(item["elapsed_ms"]) for item in cold_navigation_actions),
                        3,
                    ),
                    "ui_action_max_ms": round(
                        max(float(item["elapsed_ms"]) for item in ui_actions),
                        3,
                    ),
                    "heartbeat_samples": len(heartbeat_gaps_ms),
                    "heartbeat_max_gap_ms": round(max(heartbeat_gaps_ms), 3),
                    "heartbeat_kind": "offscreen Qt timer; not native keyboard/compositor latency",
                    "process_resource_delta": workloads._snapshot_delta(process_before, process_after),
                    "final_rss_bytes": process_after.get("rss_bytes"),
                    "peak_queue_depth": 3,
                    "profiled_theme_switch_hotspots": theme_hotspots,
                    "profiled_ui_action_hotspots": profiled_ui_actions,
                },
                "results": {
                    "crop_status": window.job_manager.get(crop_id).status,
                    "catalog_status": window.job_manager.get(catalog_id).status,
                    "thumbnail_status": window.job_manager.get(thumbnail_id).status,
                    "crop_digest": results["crop"]["digest"],
                    "catalog_digest": results["catalog"]["digest"],
                    "serial_crop_equal": True,
                    "serial_catalog_equal": True,
                    "final_active_jobs": 0,
                    "final_queued_resources": 0,
                    "final_running_resources": 0,
                },
                "limitations": [
                    "offscreen Qt only",
                    "CPU decode/crop/catalog only; no detector/embedder or GPU inference",
                    "face-search/Torch modules preloaded outside the measured workload",
                    "generated media rather than an approved photographic fixture",
                    "single bounded journey rather than repeated fresh-process qualification",
                ],
            }
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                f"{json.dumps(report, indent=2, sort_keys=True)}\n",
                encoding="utf-8",
            )
    finally:
        release.set()
        thumbnail_release.set()
        for job in jobs:
            job.cancel()
        for thread in threads:
            try:
                if sip.isdeleted(thread):
                    continue
                thread.quit()
                thread.wait(5_000)
            except RuntimeError:
                pass
        if not sip.isdeleted(window):
            window.close()
        _drain_deletes()
        apply_app_theme(APP, "dark")
        store.clear()
        for key, value in settings_snapshot.items():
            store.setValue(key, value)
        store.sync()
        settings_mod._RUNTIME_BASE_DIR = None
