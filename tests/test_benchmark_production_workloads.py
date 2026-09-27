from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "benchmark_production_workloads.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("benchmark_production_workloads", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_manifest_is_deterministic(tmp_path):
    benchmark = _load_module()

    first = benchmark._create_fixture(tmp_path / "first", photos=2, width=64, height=64, seed=17)
    second = benchmark._create_fixture(tmp_path / "second", photos=2, width=64, height=64, seed=17)

    assert benchmark._fixture_manifest(first) == benchmark._fixture_manifest(second)
    assert [path.stat().st_mtime_ns for path in first] == [path.stat().st_mtime_ns for path in second]
    assert [int(path.stat().st_mtime) for path in first] == [
        benchmark.FIXTURE_MTIME_EPOCH_S,
        benchmark.FIXTURE_MTIME_EPOCH_S + 1,
    ]


def test_aggregate_omits_unstable_tail_percentile():
    benchmark = _load_module()
    samples = [{"phase": {"elapsed_ms": value}} for value in (5, 1, 4, 2, 3)]

    aggregate = benchmark._aggregate(samples, ("phase", "elapsed_ms"))

    assert aggregate["samples_ms"] == [5.0, 1.0, 4.0, 2.0, 3.0]
    assert aggregate["median_ms"] == 3.0
    assert aggregate["range_ms"] == [1.0, 5.0]
    assert aggregate["p95_ms"] is None
    assert "insufficient" in aggregate["p95_status"]


def test_aggregate_reports_tail_percentile_only_with_enough_samples():
    benchmark = _load_module()
    samples = [{"phase": {"elapsed_ms": value}} for value in range(20)]

    aggregate = benchmark._aggregate(samples, ("phase", "elapsed_ms"))

    assert aggregate["p95_ms"] == 18.0
    assert aggregate["p95_status"] == "REPORTED_FROM_20_FRESH_PROCESS_SAMPLES"


def test_catalog_digest_is_independent_of_temporary_root():
    benchmark = _load_module()
    first_root = Path("/tmp/first-fixture")
    second_root = Path("/tmp/second-fixture")
    relative_paths = (Path("a.jpg"), Path("nested/b.jpg"))
    first = [SimpleNamespace(image_path=str(first_root / path)) for path in relative_paths]
    second = [SimpleNamespace(image_path=str(second_root / path)) for path in relative_paths]

    assert benchmark._catalog_membership_digest(first, first_root) == benchmark._catalog_membership_digest(
        second, second_root
    )


def test_tiny_fresh_process_workload_reports_required_evidence(tmp_path):
    report_path = tmp_path / "report.json"
    runtime_root = tmp_path / "runtime"
    environment = os.environ.copy()
    environment.update(
        {
            "CLUSTERLENS_RUNTIME_ROOT": str(runtime_root),
            "IMAGE_CLUSTERING_APP_DIR": str(runtime_root),
            "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache"),
            "QT_QPA_PLATFORM": "offscreen",
            "XDG_CONFIG_HOME": str(tmp_path / "settings"),
            "PYTHONPATH": os.pathsep.join((str(REPO_ROOT / "src"), str(REPO_ROOT))),
        }
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCRIPT_PATH),
            "--photos",
            "1",
            "--width",
            "64",
            "--height",
            "64",
            "--runs",
            "1",
            "--seed",
            "17",
            "--report",
            str(report_path),
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert json.loads(completed.stdout) == report
    assert report["operation"] == "ux28_full_resolution_and_mixed_workload_baseline"
    assert len(report["untraced_runs"]) == 1
    trace = report["traced_run"]["trace"]
    assert trace["peak_python_bytes"] > 0
    assert len(trace["crop_cycle_retained_python_bytes"]) == 5
    assert len(trace["crop_cycle_rss_bytes"]) == 5
    assert report["profiled_run"]["profile_hotspots"]
    assert report["untraced_runs"][0]["mixed"]["queue_depths"] == {
        "full_frame_crop_buffers_per_worker": 1,
        "max_concurrent_workloads": 3,
        "submitted_workloads": 3,
        "thumbnail_memory_cache_items": 32,
    }
    mixed = report["untraced_runs"][0]["mixed"]
    assert mixed["input_round_trip_samples"] > 0
    assert mixed["resource_delta"]["cpu_seconds"] >= 0
    assert mixed["peak_rss_bytes"] > 0
    assert report["untraced_runs"][0]["process_peak_rss_bytes"] > 0
    assert report["untraced_runs"][0]["environment"]["gpu"]["status"] == "NOT_USED"
    assert report["aggregates"]["mixed_complete"]["p95_ms"] is None
    assert report["deterministic_result_digest"]
