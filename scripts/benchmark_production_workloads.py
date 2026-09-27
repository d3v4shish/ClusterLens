#!/usr/bin/env python3
"""Benchmark generated full-resolution media and mixed production workloads.

The default parent process creates one fixed, local JPEG corpus and launches
five fresh untraced child processes plus separate traced and profiled children.
Each child gets an empty ClusterLens runtime. No user media, model, network, or
persistent application state is read.
"""

from __future__ import annotations

import argparse
import cProfile
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import pstats
import resource
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_SEED = 20260925
DEFAULT_PHOTOS = 8
DEFAULT_WIDTH = 3840
DEFAULT_HEIGHT = 2160
FIXTURE_MTIME_EPOCH_S = 1_700_000_000


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photos", type=int, default=DEFAULT_PHOTOS)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--runs", type=int, default=5, help="Fresh untraced child processes.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--report", type=Path, help="Optional new JSON report path.")
    parser.add_argument("--child-mode", choices=("untraced", "trace", "profile"), help=argparse.SUPPRESS)
    parser.add_argument("--fixture-dir", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--run-index", type=int, default=0, help=argparse.SUPPRESS)
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if min(args.photos, args.width, args.height, args.runs) < 1:
        parser.error("photos, width, height, and runs must be positive")
    if args.width < 64 or args.height < 64:
        parser.error("fixture dimensions must be at least 64x64")
    if bool(args.child_mode) != bool(args.fixture_dir):
        parser.error("internal child mode and fixture directory must be supplied together")
    if args.report is not None and args.report.expanduser().exists():
        parser.error(f"refusing to overwrite report: {args.report.expanduser()}")


def _create_fixture(root: Path, *, photos: int, width: int, height: int, seed: int) -> list[Path]:
    root.mkdir(parents=True, exist_ok=False)
    x = np.arange(width, dtype=np.uint16)[None, :]
    y = np.arange(height, dtype=np.uint16)[:, None]
    paths: list[Path] = []
    for index in range(photos):
        # Formula-generated pixels are deterministic without holding the whole
        # corpus in memory or depending on a random-number implementation.
        red = np.broadcast_to((x + seed + index * 17) % 256, (height, width))
        green = np.broadcast_to((y + seed // 3 + index * 29) % 256, (height, width))
        blue = (x // 3 + y // 5 + seed // 7 + index * 41) % 256
        array = np.empty((height, width, 3), dtype=np.uint8)
        array[:, :, 0] = red
        array[:, :, 1] = green
        array[:, :, 2] = blue
        path = root / f"full-resolution-{index:03d}.jpg"
        Image.fromarray(array, mode="RGB").save(path, "JPEG", quality=90, subsampling=0)
        fixed_time = FIXTURE_MTIME_EPOCH_S + index
        os.utime(path, (fixed_time, fixed_time))
        paths.append(path)
    return paths


def _fixture_manifest(paths: list[Path]) -> dict[str, object]:
    digest = hashlib.sha256()
    total_bytes = 0
    for path in paths:
        data = path.read_bytes()
        total_bytes += len(data)
        digest.update(path.name.encode("utf-8"))
        digest.update(data)
    return {
        "sha256": digest.hexdigest(),
        "bytes": total_bytes,
        "files": len(paths),
    }


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile))))
    return round(float(ordered[rank]), 3)


def _process_snapshot() -> dict[str, float | int | None]:
    rss_bytes = None
    read_bytes = None
    write_bytes = None
    read_chars = None
    write_chars = None
    try:
        resident_pages = int(Path("/proc/self/statm").read_text(encoding="ascii").split()[1])
        rss_bytes = resident_pages * int(os.sysconf("SC_PAGE_SIZE"))
    except (OSError, ValueError, IndexError):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        if sys.platform.startswith("linux"):
            rss_bytes = int(usage.ru_maxrss) * 1024
    try:
        io_values = {}
        for line in Path("/proc/self/io").read_text(encoding="ascii").splitlines():
            key, value = line.split(":", 1)
            io_values[key] = int(value.strip())
        read_bytes = io_values.get("read_bytes")
        write_bytes = io_values.get("write_bytes")
        read_chars = io_values.get("rchar")
        write_chars = io_values.get("wchar")
    except (OSError, ValueError):
        pass
    return {
        "cpu_seconds": float(time.process_time()),
        "rss_bytes": rss_bytes,
        "read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "read_chars": read_chars,
        "write_chars": write_chars,
    }


def _peak_rss_bytes() -> int | None:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform.startswith("linux"):
        return int(usage.ru_maxrss) * 1024
    if sys.platform == "darwin":
        return int(usage.ru_maxrss)
    return None


def _snapshot_delta(before: dict[str, object], after: dict[str, object]) -> dict[str, float | int | None]:
    output: dict[str, float | int | None] = {}
    for key in ("cpu_seconds", "rss_bytes", "read_bytes", "write_bytes", "read_chars", "write_chars"):
        first = before.get(key)
        second = after.get(key)
        output[key] = None if first is None or second is None else round(float(second) - float(first), 6)
    return output


def _qimage_digest(image, digest) -> None:
    digest.update(int(image.width()).to_bytes(4, "little", signed=False))
    digest.update(int(image.height()).to_bytes(4, "little", signed=False))
    bits = image.bits()
    digest.update(bits.asstring(image.sizeInBytes()))


def _thumbnail_pass(paths: list[Path], service, *, size: int = 320, first_event=None) -> dict[str, object]:
    digest = hashlib.sha256()
    started = time.perf_counter()
    first_content_ms = None
    for index, path in enumerate(paths):
        image = service.load_qimage(str(path), size)
        if image.isNull():
            raise RuntimeError(f"thumbnail decode failed: {path}")
        _qimage_digest(image, digest)
        if index == 0:
            first_content_ms = (time.perf_counter() - started) * 1000.0
            if first_event is not None:
                first_event.set()
    return {
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "first_content_ms": round(float(first_content_ms or 0.0), 3),
        "digest": digest.hexdigest(),
        "items": len(paths),
    }


def _crop_boxes(width: int, height: int) -> tuple[tuple[int, int, int, int], ...]:
    edge = max(32, min(width, height) // 4)
    return (
        (width // 8, height // 8, width // 8 + edge, height // 8 + edge),
        (width // 2 - edge // 2, height // 4, width // 2 + edge // 2, height // 4 + edge),
        (width - width // 8 - edge, height // 3, width - width // 8, height // 3 + edge),
        (width // 3, height - height // 8 - edge, width // 3 + edge, height - height // 8),
    )


def _crop_landmarks(bbox: tuple[int, int, int, int]) -> tuple[tuple[float, float], ...]:
    x1, y1, x2, y2 = bbox
    width = float(x2 - x1)
    height = float(y2 - y1)
    return (
        (x1 + width * 0.30, y1 + height * 0.35),
        (x1 + width * 0.70, y1 + height * 0.35),
        (x1 + width * 0.50, y1 + height * 0.55),
        (x1 + width * 0.35, y1 + height * 0.75),
        (x1 + width * 0.65, y1 + height * 0.75),
    )


def _crop_pass(paths: list[Path]) -> dict[str, object]:
    from app.services.face_search import _DetectedFaceCropper, _face_crop_quality_metrics

    digest = hashlib.sha256()
    started = time.perf_counter()
    crop_count = 0
    for path in paths:
        with Image.open(path) as source:
            rgb = source.convert("RGB")
            cropper = _DetectedFaceCropper(rgb)
            for box in _crop_boxes(*rgb.size):
                crop = cropper.crop(box, _crop_landmarks(box))
                quality = _face_crop_quality_metrics(crop)
                digest.update(crop.tobytes())
                digest.update(json.dumps([round(value, 6) for value in quality]).encode("ascii"))
                crop_count += 1
    return {
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "digest": digest.hexdigest(),
        "crops": crop_count,
    }


def _catalog_membership_digest(items, root: Path) -> str:
    members = [Path(item.image_path).relative_to(root).as_posix() for item in items]
    return hashlib.sha256("\n".join(members).encode("utf-8")).hexdigest()


def _catalog_pass(paths: list[Path], runtime_root: Path) -> dict[str, object]:
    from app.services.library_catalog import CatalogQuery, LibraryCatalogService

    catalog = LibraryCatalogService(db_path=runtime_root / "mixed-catalog.sqlite3")
    root = catalog.register_root(paths[0].parent)
    started = time.perf_counter()
    result = catalog.scan_root(root.root_id)
    page = catalog.query_assets(CatalogQuery(root_ids=(root.root_id,), limit=len(paths)))
    return {
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "assets": int(page.total_count),
        "scan": {key: int(value) for key, value in result.items()},
        "digest": _catalog_membership_digest(page.items, paths[0].parent),
    }


def _mixed_pass(app, paths: list[Path], runtime_root: Path) -> dict[str, object]:
    from PyQt6.QtCore import QCoreApplication, QEvent, QObject, QTimer
    from app.services.thumbnails import ThumbnailService

    input_event_type = QEvent.Type(QEvent.registerEventType())
    input_posted: list[float] = []
    input_round_trips: list[float] = []

    class InputProbe(QObject):
        def event(self, event) -> bool:
            if event.type() == input_event_type and input_posted:
                input_round_trips.append((time.perf_counter() - input_posted.pop(0)) * 1000.0)
                return True
            return super().event(event)

    input_probe = InputProbe()

    service = ThumbnailService(qimage_cache_size=max(32, len(paths)))
    first_content = threading.Event()
    callback_latencies: list[float] = []
    pump_gaps: list[float] = []
    before = _process_snapshot()
    peak_rss = int(before.get("rss_bytes") or 0)
    started = time.perf_counter()
    previous_pump = started
    first_content_ms = None
    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="ux28-mixed") as pool:
        futures = {
            "thumbnails": pool.submit(_thumbnail_pass, paths, service, first_event=first_content),
            "crops": pool.submit(_crop_pass, paths),
            "catalog": pool.submit(_catalog_pass, paths, runtime_root),
        }
        while not all(future.done() for future in futures.values()):
            scheduled = time.perf_counter()
            QTimer.singleShot(0, lambda when=scheduled: callback_latencies.append((time.perf_counter() - when) * 1000.0))
            input_posted.append(time.perf_counter())
            QCoreApplication.postEvent(input_probe, QEvent(input_event_type))
            app.processEvents()
            now = time.perf_counter()
            pump_gaps.append((now - previous_pump) * 1000.0)
            previous_pump = now
            if first_content_ms is None and first_content.is_set():
                first_content_ms = (now - started) * 1000.0
            snapshot = _process_snapshot()
            peak_rss = max(peak_rss, int(snapshot.get("rss_bytes") or 0))
            time.sleep(0.001)
        app.processEvents()
        results = {name: future.result() for name, future in futures.items()}
    after = _process_snapshot()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "elapsed_ms": round(elapsed_ms, 3),
        "first_content_ms": round(float(first_content_ms or elapsed_ms), 3),
        "ui_callback_samples": len(callback_latencies),
        "ui_callback_p50_ms": _percentile(callback_latencies, 0.50),
        "ui_callback_p95_ms": _percentile(callback_latencies, 0.95) if len(callback_latencies) >= 20 else None,
        "ui_callback_max_ms": round(max(callback_latencies, default=0.0), 3),
        "input_round_trip_samples": len(input_round_trips),
        "input_round_trip_p50_ms": _percentile(input_round_trips, 0.50),
        "input_round_trip_p95_ms": _percentile(input_round_trips, 0.95) if len(input_round_trips) >= 20 else None,
        "input_round_trip_max_ms": round(max(input_round_trips, default=0.0), 3),
        "input_round_trip_kind": "synthetic posted Qt event; not native keyboard/compositor latency",
        "event_pump_p95_ms": _percentile(pump_gaps, 0.95) if len(pump_gaps) >= 20 else None,
        "event_pump_max_ms": round(max(pump_gaps, default=0.0), 3),
        "resource_delta": _snapshot_delta(before, after),
        "peak_rss_bytes": peak_rss,
        "queue_depths": {
            "submitted_workloads": 3,
            "max_concurrent_workloads": 3,
            "thumbnail_memory_cache_items": service.qimage_cache_size,
            "full_frame_crop_buffers_per_worker": 1,
        },
        "results": results,
    }


def _cancellation_pass(paths: list[Path]) -> dict[str, object]:
    checkpoint = threading.Event()
    release = threading.Event()
    cancel = threading.Event()
    acknowledged = threading.Event()

    def worker() -> int:
        completed = 0
        for path in paths:
            with Image.open(path) as source:
                source.convert("RGB").crop(_crop_boxes(*source.size)[0]).load()
            completed += 1
            if completed == 1:
                checkpoint.set()
                release.wait(10.0)
            if cancel.is_set():
                acknowledged.set()
                return completed
        return completed

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="ux28-cancel") as pool:
        future = pool.submit(worker)
        if not checkpoint.wait(10.0):
            raise RuntimeError("cancellation fixture did not reach its checkpoint")
        started = time.perf_counter()
        cancel.set()
        release.set()
        if not acknowledged.wait(10.0):
            raise RuntimeError("cancellation fixture did not acknowledge cancellation")
        acknowledged_ms = (time.perf_counter() - started) * 1000.0
        completed = future.result(timeout=10.0)
        safe_drain_ms = (time.perf_counter() - started) * 1000.0
    return {
        "acknowledgement_ms": round(acknowledged_ms, 3),
        "safe_drain_ms": round(safe_drain_ms, 3),
        "completed_before_cancel": completed,
        "committed_outputs": 0,
    }


def _profile_hotspots(profile: cProfile.Profile, *, limit: int = 20) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for (filename, line, function), (_cc, calls, own, cumulative, _callers) in pstats.Stats(profile).stats.items():
        entries.append(
            {
                "function": f"{Path(filename).name}:{line}:{function}",
                "calls": int(calls),
                "own_seconds": round(float(own), 6),
                "cumulative_seconds": round(float(cumulative), 6),
            }
        )
    return sorted(entries, key=lambda item: float(item["cumulative_seconds"]), reverse=True)[:limit]


def _dependency_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("clusterlens", "PyQt6", "Pillow", "numpy", "psutil", "onnxruntime", "torch"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _gpu_evidence() -> dict[str, object]:
    evidence: dict[str, object] = {
        "workload": "CPU-only; GPU memory is not applicable to this fixture",
        "status": "NOT_USED",
        "driver": "NOT_AVAILABLE",
    }
    try:
        import torch

        evidence["torch_cuda_available"] = bool(torch.cuda.is_available())
        evidence["torch_cuda_runtime"] = str(torch.version.cuda or "not-built")
        if torch.cuda.is_available():
            evidence["device"] = str(torch.cuda.get_device_name(0))
            evidence["allocated_bytes"] = int(torch.cuda.memory_allocated(0))
            evidence["reserved_bytes"] = int(torch.cuda.memory_reserved(0))
            try:
                evidence["driver"] = str(torch._C._cuda_getDriverVersion())
            except (AttributeError, RuntimeError):
                evidence["driver"] = "UNAVAILABLE_FROM_RUNTIME"
    except Exception as exc:
        evidence["torch_cuda_available"] = False
        evidence["reason"] = f"{type(exc).__name__}: {exc}"
    return evidence


def _environment(app) -> dict[str, object]:
    screen = app.primaryScreen()
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            if line.casefold().startswith("model name"):
                cpu = line.split(":", 1)[-1].strip()
                break
    except OSError:
        pass
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu,
        "logical_cpu_count": os.cpu_count(),
        "qt_platform": app.platformName(),
        "display": None
        if screen is None
        else {
            "device_pixel_ratio": float(screen.devicePixelRatio()),
            "logical_dpi": float(screen.logicalDotsPerInch()),
        },
        "dependencies": _dependency_versions(),
        "gpu": _gpu_evidence(),
    }


def _child(args: argparse.Namespace) -> dict[str, object]:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    fixture_dir = args.fixture_dir.expanduser().resolve()
    paths = sorted(fixture_dir.glob("*.jpg"))
    if len(paths) != args.photos:
        raise RuntimeError(f"fixture contains {len(paths)} photos, expected {args.photos}")
    runtime_root = Path(os.environ["CLUSTERLENS_RUNTIME_ROOT"]).resolve()
    runtime_root.mkdir(parents=True, exist_ok=True)
    for value in (REPO_ROOT, REPO_ROOT / "src"):
        if str(value) not in sys.path:
            sys.path.insert(0, str(value))

    from PyQt6.QtWidgets import QApplication
    # Preload the production modules outside the traced/profiled interval so
    # allocation and CPU evidence describe the workload rather than imports.
    from app.services import face_search as _face_search  # noqa: F401
    from app.services import library_catalog as _library_catalog  # noqa: F401
    from app.services.thumbnails import ThumbnailService

    app = QApplication.instance() or QApplication([])
    profile = cProfile.Profile()
    if args.child_mode == "trace":
        tracemalloc.start()
    if args.child_mode == "profile":
        profile.enable()
    process_before = _process_snapshot()
    service = ThumbnailService(qimage_cache_size=max(32, len(paths)))
    cold_thumbnail = _thumbnail_pass(paths, service)
    cold_crop = _crop_pass(paths)
    warm_thumbnail = _thumbnail_pass(paths, service)
    warm_crop = _crop_pass(paths)
    mixed = _mixed_pass(app, paths, runtime_root)
    cancellation = _cancellation_pass(paths)
    process_after = _process_snapshot()
    if args.child_mode == "profile":
        profile.disable()
    trace = None
    if args.child_mode == "trace":
        crop_cycle_retained: list[int] = []
        crop_cycle_rss: list[int | None] = []
        for _cycle in range(5):
            cycle_result = _crop_pass(paths)
            if cycle_result["digest"] != cold_crop["digest"]:
                raise RuntimeError("repeated crop cycle changed its result")
            gc.collect()
            retained, _peak = tracemalloc.get_traced_memory()
            crop_cycle_retained.append(int(retained))
            crop_cycle_rss.append(_process_snapshot()["rss_bytes"])
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        trace = {
            "retained_python_bytes": int(current),
            "peak_python_bytes": int(peak),
            "crop_cycle_retained_python_bytes": crop_cycle_retained,
            "crop_cycle_rss_bytes": crop_cycle_rss,
            "crop_cycle_retained_growth_bytes": crop_cycle_retained[-1] - crop_cycle_retained[0],
            "crop_cycle_rss_growth_bytes": (
                None
                if crop_cycle_rss[0] is None or crop_cycle_rss[-1] is None
                else int(crop_cycle_rss[-1]) - int(crop_cycle_rss[0])
            ),
        }
    digest_payload = {
        "cold_thumbnail": cold_thumbnail["digest"],
        "warm_thumbnail": warm_thumbnail["digest"],
        "cold_crop": cold_crop["digest"],
        "warm_crop": warm_crop["digest"],
        "mixed_thumbnail": mixed["results"]["thumbnails"]["digest"],
        "mixed_crop": mixed["results"]["crops"]["digest"],
        "mixed_catalog": mixed["results"]["catalog"]["digest"],
    }
    if cold_thumbnail["digest"] != warm_thumbnail["digest"] or cold_crop["digest"] != warm_crop["digest"]:
        raise RuntimeError("cold and warm media results differ")
    return {
        "mode": args.child_mode,
        "run_index": args.run_index,
        "cache_state": {
            "cold": "new empty per-process runtime and thumbnail cache",
            "warm": "same process, source corpus, ThumbnailService, and generated cache",
        },
        "cold": {"thumbnail": cold_thumbnail, "full_resolution_crop": cold_crop},
        "warm": {"thumbnail": warm_thumbnail, "full_resolution_crop": warm_crop},
        "mixed": mixed,
        "cancellation": cancellation,
        "process_resource_delta": _snapshot_delta(process_before, process_after),
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "trace": trace,
        "profile_hotspots": _profile_hotspots(profile) if args.child_mode == "profile" else [],
        "result_digest": hashlib.sha256(json.dumps(digest_payload, sort_keys=True).encode("utf-8")).hexdigest(),
        "result_components": digest_payload,
        "environment": _environment(app),
    }


def _run_child(
    args: argparse.Namespace,
    *,
    fixture_dir: Path,
    mode: str,
    run_index: int,
    parent_runtime: Path,
) -> dict[str, object]:
    child_runtime = parent_runtime / f"{mode}-{run_index:02d}"
    child_runtime.mkdir(parents=True, exist_ok=False)
    command = [
        sys.executable,
        "-B",
        str(SCRIPT_PATH),
        "--photos",
        str(args.photos),
        "--width",
        str(args.width),
        "--height",
        str(args.height),
        "--runs",
        str(args.runs),
        "--seed",
        str(args.seed),
        "--child-mode",
        mode,
        "--fixture-dir",
        str(fixture_dir),
        "--run-index",
        str(run_index),
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "QT_QPA_PLATFORM": "offscreen",
            "CLUSTERLENS_RUNTIME_ROOT": str(child_runtime),
            "IMAGE_CLUSTERING_APP_DIR": str(child_runtime),
            "XDG_CONFIG_HOME": str(child_runtime / "settings"),
            "XDG_DATA_HOME": str(child_runtime / "data"),
            "PYTHONPYCACHEPREFIX": str(child_runtime / "pycache"),
            "PYTHONPATH": os.pathsep.join((str(REPO_ROOT / "src"), str(REPO_ROOT))),
        }
    )
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{mode} child {run_index} failed with exit {completed.returncode}:\n{completed.stderr[-4000:]}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{mode} child emitted invalid JSON: {completed.stdout[-4000:]}") from exc


def _aggregate(samples: list[dict[str, object]], path: tuple[str, ...]) -> dict[str, object]:
    values = []
    for sample in samples:
        value: object = sample
        for key in path:
            value = value[key]  # type: ignore[index]
        values.append(float(value))
    sufficient_tail_samples = len(values) >= 20
    return {
        "samples_ms": [round(value, 3) for value in values],
        "median_ms": round(statistics.median(values), 3),
        "range_ms": [round(min(values), 3), round(max(values), 3)],
        "p95_ms": _percentile(values, 0.95) if sufficient_tail_samples else None,
        "p95_status": (
            f"REPORTED_FROM_{len(values)}_FRESH_PROCESS_SAMPLES"
            if sufficient_tail_samples
            else f"NOT_REPORTED: {len(values)} fresh-process samples are insufficient for a stable tail percentile"
        ),
    }


def _parent(args: argparse.Namespace) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="clusterlens-ux28-") as temporary:
        root = Path(temporary)
        fixture_dir = root / "fixture"
        paths = _create_fixture(
            fixture_dir,
            photos=args.photos,
            width=args.width,
            height=args.height,
            seed=args.seed,
        )
        fixture = {
            "kind": "generated fixed full-resolution JPEG corpus",
            "photos": args.photos,
            "width": args.width,
            "height": args.height,
            "pixels_per_photo": args.width * args.height,
            "aligned_crops_per_photo": 4,
            "seed": args.seed,
            "source": "formula-generated local media; no user photos, model, or network",
            **_fixture_manifest(paths),
        }
        runtime = root / "runs"
        runtime.mkdir()
        untraced = [
            _run_child(args, fixture_dir=fixture_dir, mode="untraced", run_index=index, parent_runtime=runtime)
            for index in range(args.runs)
        ]
        traced = _run_child(args, fixture_dir=fixture_dir, mode="trace", run_index=0, parent_runtime=runtime)
        profiled = _run_child(args, fixture_dir=fixture_dir, mode="profile", run_index=0, parent_runtime=runtime)
        digests = {str(sample["result_digest"]) for sample in untraced}
        if len(digests) != 1:
            raise RuntimeError("fresh-process workload results are not deterministic")
        return {
            "report_version": 1,
            "operation": "ux28_full_resolution_and_mixed_workload_baseline",
            "fixture": fixture,
            "method": {
                "fresh_untraced_processes": args.runs,
                "separate_traced_processes": 1,
                "separate_profiled_processes": 1,
                "serial_phases": ["cold media", "warm media", "mixed contention", "cooperative cancellation"],
                "mixed_workers": ["thumbnail decode/cache", "full-resolution face crop/quality", "catalog scan/query"],
            },
            "aggregates": {
                "cold_thumbnail": _aggregate(untraced, ("cold", "thumbnail", "elapsed_ms")),
                "warm_thumbnail": _aggregate(untraced, ("warm", "thumbnail", "elapsed_ms")),
                "cold_full_resolution_crop": _aggregate(untraced, ("cold", "full_resolution_crop", "elapsed_ms")),
                "warm_full_resolution_crop": _aggregate(untraced, ("warm", "full_resolution_crop", "elapsed_ms")),
                "mixed_first_content": _aggregate(untraced, ("mixed", "first_content_ms")),
                "mixed_complete": _aggregate(untraced, ("mixed", "elapsed_ms")),
                "cancellation_acknowledgement": _aggregate(untraced, ("cancellation", "acknowledgement_ms")),
                "cancellation_safe_drain": _aggregate(untraced, ("cancellation", "safe_drain_ms")),
            },
            "untraced_runs": untraced,
            "traced_run": traced,
            "profiled_run": profiled,
            "deterministic_result_digest": next(iter(digests)),
            "limitations": [
                "CPU-only generated media; no detector/embedder model or GPU work is performed.",
                "Offscreen Qt measures event-pump responsiveness without native compositor latency.",
                "Filesystem I/O counters may reflect the host page cache; cold means an empty application cache, not a dropped OS cache.",
                f"{args.runs} fresh-process samples report median/range; aggregate p95 requires at least 20 samples.",
            ],
        }


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    _validate_args(args, parser)
    payload = _child(args) if args.child_mode else _parent(args)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        target = args.report.expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
