#!/usr/bin/env python3
"""Qualify warm-worker lifecycle with a real, explicitly supplied ONNX asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw
from PyQt6.QtCore import QCoreApplication, QEventLoop


REPO_ROOT = Path(__file__).resolve().parents[1]
for import_path in (REPO_ROOT / "src", REPO_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

EXIT_NOT_RUN = 3
EXECUTION_MODES = ("cpu", "cuda")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _linux_process_memory(pid: int) -> dict[str, int | None]:
    result: dict[str, int | None] = {"rss_bytes": None, "peak_rss_bytes": None}
    try:
        lines = Path(f"/proc/{int(pid)}/status").read_text(encoding="utf-8").splitlines()
    except OSError:
        return result
    for line in lines:
        if line.startswith("VmRSS:"):
            result["rss_bytes"] = int(line.split()[1]) * 1024
        elif line.startswith("VmHWM:"):
            result["peak_rss_bytes"] = int(line.split()[1]) * 1024
    return result


def _nvidia_compute_process_memory(pid: int) -> int | None:
    """Return this process's current VRAM allocation in bytes, when observable."""

    if int(pid) <= 0:
        return None
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode != 0:
        return None
    for raw_line in completed.stdout.splitlines():
        fields = [field.strip() for field in raw_line.split(",", 1)]
        if len(fields) != 2:
            continue
        try:
            candidate_pid = int(fields[0])
            used_mib = int(fields[1])
        except ValueError:
            continue
        if candidate_pid == int(pid):
            return used_mib * 1024 * 1024
    return None


def _cuda_runtime_issue() -> str:
    try:
        import onnxruntime as ort
        import torch
    except Exception as exc:
        return f"CUDA qualification runtime could not import Torch/ONNX Runtime: {exc}"
    if not bool(torch.cuda.is_available()):
        return "Torch CUDA is unavailable in the qualification runtime"
    if "CUDAExecutionProvider" not in set(ort.get_available_providers()):
        return "CUDAExecutionProvider is unavailable in the qualification runtime"
    return ""


def _wait_until(app: QCoreApplication, predicate, *, timeout_s: float) -> bool:
    deadline = time.monotonic() + max(0.1, float(timeout_s))
    while time.monotonic() < deadline:
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
        if predicate():
            return True
        time.sleep(0.005)
    app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
    return bool(predicate())


def _create_fixture(root: Path) -> list[str]:
    paths: list[str] = []
    colors = ((220, 30, 30), (30, 210, 60), (25, 70, 220), (210, 180, 25))
    for index, color in enumerate(colors):
        path = root / f"fixture-{index:02d}.png"
        image = Image.new("RGB", (320, 240), color)
        draw = ImageDraw.Draw(image)
        draw.rectangle((24 + index * 7, 30, 180, 190), outline=(255, 255, 255), width=8)
        draw.ellipse((130, 55 + index * 5, 280, 205), outline=(0, 0, 0), width=7)
        image.save(path, format="PNG", optimize=False)
        paths.append(str(path))
    return paths


def _validate_asset(model_asset_dir: Path) -> tuple[dict[str, object] | None, str]:
    metadata_path = model_asset_dir / "metadata.json"
    model_path = model_asset_dir / "model.onnx"
    if not metadata_path.is_file() or not model_path.is_file():
        return None, "model.onnx and metadata.json are required"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as exc:
        return None, f"invalid metadata.json: {exc}"
    expected_sha = str(metadata.get("sha256") or "").strip().lower()
    actual_sha = _sha256_file(model_path)
    if expected_sha != actual_sha:
        return None, f"model checksum mismatch: expected {expected_sha or '<missing>'}, got {actual_sha}"
    if str(metadata.get("model_name") or "").strip() != "fast_preview":
        return None, "the warm-worker qualifier currently requires the fast_preview asset"
    return dict(metadata), ""


def _normalized_membership(payload: dict[str, object], fixture_root: Path) -> dict[str, list[list[str]]]:
    normalized: dict[str, list[list[str]]] = {}
    for comparison_key, clusters in sorted(dict(payload.get("clusters_by_key") or {}).items()):
        groups: list[list[str]] = []
        for members in dict(clusters or {}).values():
            relative_members = []
            for member in members:
                try:
                    relative_members.append(str(Path(str(member)).resolve().relative_to(fixture_root.resolve())))
                except ValueError:
                    relative_members.append(Path(str(member)).name)
            groups.append(sorted(relative_members))
        normalized[str(comparison_key)] = sorted(groups)
    return normalized


def qualify(
    model_asset_dir: Path,
    *,
    cycles: int = 2,
    timeout_s: float = 60.0,
    execution_mode: str = "cpu",
) -> tuple[dict[str, object], int]:
    model_asset_dir = model_asset_dir.expanduser().resolve()
    execution_mode = str(execution_mode).strip().lower()
    metadata, asset_error = _validate_asset(model_asset_dir)
    report: dict[str, object] = {
        "report_version": 2,
        "operation": "warm_worker_real_model_qualification",
        "status": "FAIL",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "execution_mode": execution_mode,
        },
        "asset_dir": str(model_asset_dir),
        "cycles": [],
        "failures": [],
    }
    if metadata is None:
        report["status"] = "NOT_RUN"
        report["not_run_reasons"] = [asset_error]
        return report, EXIT_NOT_RUN
    if execution_mode not in EXECUTION_MODES:
        report["status"] = "NOT_RUN"
        report["not_run_reasons"] = [
            f"unsupported execution mode {execution_mode!r}; expected one of {', '.join(EXECUTION_MODES)}"
        ]
        return report, EXIT_NOT_RUN
    if execution_mode == "cuda":
        cuda_issue = _cuda_runtime_issue()
        if cuda_issue:
            report["status"] = "NOT_RUN"
            report["not_run_reasons"] = [cuda_issue]
            return report, EXIT_NOT_RUN
        report["vram_measurement"] = {
            "method": "nvidia-smi per-compute-process used_memory after workload while worker is warm",
            "release_contract": "the exact worker PID is absent from the compute-process table after shutdown",
        }
    report["asset"] = {
        "model_name": metadata.get("model_name"),
        "signature": metadata.get("signature"),
        "sha256": metadata.get("sha256"),
        "size_bytes": int((model_asset_dir / "model.onnx").stat().st_size),
        "source_url": metadata.get("source_url"),
        "license": metadata.get("license"),
    }

    from apps.pyqt_production.session_controller import ClusteringSessionController
    from apps.pyqt_production.worker_protocol import ProductionClusterRequest
    from apps.shared.runtime_support import activate_runtime_root

    app = QCoreApplication.instance() or QCoreApplication([])
    previous_runtime = os.environ.get("CLUSTERLENS_RUNTIME_ROOT")
    previous_legacy_runtime = os.environ.get("IMAGE_CLUSTERING_APP_DIR")
    controller = None
    try:
        with tempfile.TemporaryDirectory(prefix="clusterlens-warm-worker-") as temporary:
            temporary_root = Path(temporary)
            runtime_root = temporary_root / "runtime"
            fixture_root = temporary_root / "photos"
            fixture_root.mkdir()
            os.environ["CLUSTERLENS_RUNTIME_ROOT"] = str(runtime_root)
            os.environ["IMAGE_CLUSTERING_APP_DIR"] = str(runtime_root)
            layout = activate_runtime_root("ClusterLensWarmWorkerQualification")
            shutil.copytree(model_asset_dir, layout.model_assets_dir / "fast_preview")
            image_paths = _create_fixture(fixture_root)
            request = ProductionClusterRequest(
                directory=str(fixture_root),
                embedding_models=["fast_preview"],
                num_clusters=2,
                clustering_backends=["cosine-kmeans"],
                recursive=False,
                similarity_mode="cosine",
                outlier_policy="assign",
                use_onnx=True,
                reuse_result_cache=False,
                use_embedding_cache_lookup=False,
                source_paths=image_paths,
                preferred_execution_mode=execution_mode,
                allow_model_downloads=False,
                generate_cluster_explanations=False,
            )
            controller = ClusteringSessionController(layout)
            completed: list[dict[str, object]] = []
            failures: list[str] = []
            controller.completed.connect(lambda payload: completed.append(dict(payload)))
            controller.failed.connect(failures.append)

            previous_pid = 0
            expected_membership = None
            for cycle_index in range(max(2, int(cycles))):
                completed_count = len(completed)
                failure_count = len(failures)
                controller.set_keep_worker_warm(True)
                started = time.perf_counter()
                if not controller.start(request):
                    raise RuntimeError(f"cycle {cycle_index + 1}: controller rejected start")
                terminal = _wait_until(
                    app,
                    lambda: len(completed) > completed_count or len(failures) > failure_count,
                    timeout_s=timeout_s,
                )
                if not terminal:
                    raise RuntimeError(f"cycle {cycle_index + 1}: workload timed out")
                if len(failures) > failure_count:
                    raise RuntimeError(f"cycle {cycle_index + 1}: {failures[-1]}")
                if not controller.is_worker_warm() or controller._process is None:
                    raise RuntimeError(f"cycle {cycle_index + 1}: worker did not remain idle and warm")
                process = controller._process
                pid = int(process.processId())
                if pid <= 0 or (previous_pid and pid == previous_pid):
                    raise RuntimeError(f"cycle {cycle_index + 1}: process did not restart with a new PID")
                payload = completed[-1]
                membership = _normalized_membership(payload, fixture_root)
                if expected_membership is None:
                    expected_membership = membership
                elif membership != expected_membership:
                    raise RuntimeError(f"cycle {cycle_index + 1}: clustering membership changed after restart")
                metrics = dict(payload.get("metrics") or {})
                memory = _linux_process_memory(pid)
                vram_while_idle = _nvidia_compute_process_memory(pid) if execution_mode == "cuda" else None
                if execution_mode == "cuda" and vram_while_idle is None:
                    raise RuntimeError(
                        f"cycle {cycle_index + 1}: worker PID {pid} has no observable CUDA allocation while warm"
                    )
                cycle_record = {
                    "cycle": cycle_index + 1,
                    "pid": pid,
                    "workload_ms": round((time.perf_counter() - started) * 1000.0, 3),
                    "rss_bytes_while_idle": memory["rss_bytes"],
                    "peak_rss_bytes": memory["peak_rss_bytes"],
                    "vram_bytes_while_idle": vram_while_idle,
                    "model_load_time_s": metrics.get("model_load_time_s"),
                    "embedding_inference_time_s": metrics.get("inference_time_s"),
                    "embedding_backend": metrics.get("embedding_backend"),
                    "effective_mode": metrics.get("effective_mode"),
                    "onnx_provider": metrics.get("onnx_provider"),
                    "membership": membership,
                }
                stop_started = time.perf_counter()
                controller.set_keep_worker_warm(False)
                if not _wait_until(app, lambda: controller._process is None, timeout_s=10.0):
                    raise RuntimeError(f"cycle {cycle_index + 1}: worker did not exit after warm mode was disabled")
                cycle_record["shutdown_ms"] = round((time.perf_counter() - stop_started) * 1000.0, 3)
                cycle_record["process_absent_after_shutdown"] = not Path(f"/proc/{pid}").exists()
                if not cycle_record["process_absent_after_shutdown"]:
                    raise RuntimeError(f"cycle {cycle_index + 1}: worker process still exists after shutdown")
                if execution_mode == "cuda":
                    cycle_record["gpu_allocation_absent_after_shutdown"] = (
                        _nvidia_compute_process_memory(pid) is None
                    )
                    if not cycle_record["gpu_allocation_absent_after_shutdown"]:
                        raise RuntimeError(
                            f"cycle {cycle_index + 1}: worker PID {pid} retains a CUDA allocation after shutdown"
                        )
                report["cycles"].append(cycle_record)
                previous_pid = pid

            report["membership_digest"] = hashlib.sha256(
                json.dumps(expected_membership, sort_keys=True).encode("utf-8")
            ).hexdigest()
            report["status"] = "PASS"
            return report, 0
    except Exception as exc:
        report["failures"] = [f"{type(exc).__name__}: {exc}"]
        return report, 1
    finally:
        if controller is not None:
            controller.set_keep_worker_warm(False)
            controller.shutdown(1000)
            _wait_until(app, lambda: controller._process is None, timeout_s=3.0)
        if previous_runtime is None:
            os.environ.pop("CLUSTERLENS_RUNTIME_ROOT", None)
        else:
            os.environ["CLUSTERLENS_RUNTIME_ROOT"] = previous_runtime
        if previous_legacy_runtime is None:
            os.environ.pop("IMAGE_CLUSTERING_APP_DIR", None)
        else:
            os.environ["IMAGE_CLUSTERING_APP_DIR"] = previous_legacy_runtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-asset-dir", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=2)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--execution-mode", choices=EXECUTION_MODES, default="cpu")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report_path = args.report.expanduser().resolve()
    if report_path.exists():
        parser.error(f"refusing to overwrite report: {report_path}")
    report, exit_code = qualify(
        args.model_asset_dir,
        cycles=max(2, args.cycles),
        timeout_s=max(1.0, args.timeout_seconds),
        execution_mode=args.execution_mode,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    report_path.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
