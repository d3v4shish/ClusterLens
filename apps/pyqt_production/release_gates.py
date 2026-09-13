from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from apps.pyqt_production.bootstrap import bootstrap_runtime
from apps.shared.runtime_migrations import RUNTIME_SCHEMA_VERSION, RuntimeMigrationService
from apps.shared.runtime_support import configure_rotating_logging, install_crash_handlers


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


@dataclass(frozen=True)
class GateResult:
    name: str
    status: str
    details: str


STARTUP_IMPORT_MAX_SECONDS = 1.0
STARTUP_WINDOW_MAX_SECONDS = 1.5
STARTUP_RSS_MAX_MIB = 256.0


def _release_subprocess_environment(*, offscreen: bool = False) -> dict[str, str]:
    environment = os.environ.copy()
    existing_pythonpath = str(environment.get("PYTHONPATH") or "").strip()
    python_paths = [str(REPO_ROOT), str(SRC_ROOT)]
    if existing_pythonpath:
        python_paths.append(existing_pythonpath)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    if offscreen:
        environment["QT_QPA_PLATFORM"] = "offscreen"
    return environment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run production release-readiness gates.")
    parser.add_argument("--folder", default="", help="Optional real image folder for a smoke benchmark.")
    parser.add_argument("--report-dir", default="benchmarks/release_gates")
    parser.add_argument("--allow-downloads", action="store_true", help="Allow model downloads during optional smoke runs.")
    args = parser.parse_args(argv)

    report_dir = Path(args.report_dir).resolve()
    runtime_root = report_dir / "runtime" / "pyqt_production_release_gates"
    os.environ["IMAGE_CLUSTERING_APP_DIR"] = str(runtime_root)
    layout, _repo_root = bootstrap_runtime()
    configure_rotating_logging(layout)
    install_crash_handlers(layout)

    from app.services.clustering_options import clustering_model_names
    from app.services.model_assets import ModelAssetService

    gates: list[GateResult] = []
    migrations = RuntimeMigrationService(layout).run()
    gates.append(
        GateResult(
            "runtime_migrations",
            "PASS" if not migrations.failures and migrations.current_version == RUNTIME_SCHEMA_VERSION else "FAIL",
            f"previous={migrations.previous_version} current={migrations.current_version} actions={len(migrations.actions)} failures={len(migrations.failures)}",
        )
    )
    gates.extend(_path_gates(layout))
    gates.append(_authoritative_ui_gate())
    gates.append(_ui_ux_acceptance_gate(report_dir))
    gates.append(_startup_performance_gate(report_dir))

    asset_service = ModelAssetService(runtime_model_assets_dir=layout.model_assets_dir)
    inventory = asset_service.model_inventory(clustering_model_names(scope="production"))
    invalid = [item.model_name for item in inventory if item.validation_status == "invalid"]
    unverified = [item.model_name for item in inventory if item.packaged and item.validation_status == "unverified"]
    bundled = [item.model_name for item in inventory if item.packaged]
    fallback = asset_service.choose_bundled_fallback_model()
    gates.append(
        GateResult(
            "model_assets",
            "FAIL" if invalid or unverified or not fallback else "PASS",
            f"bundled={bundled or '-'} fallback={fallback or '-'} invalid={invalid or '-'} unverified={unverified or '-'}",
        )
    )
    gates.append(
        GateResult(
            "offline_model_policy",
            "PASS" if fallback else "FAIL",
            "At least one bundled fallback exists." if fallback else "No bundled fallback; declining downloads cannot run.",
        )
    )

    if args.folder:
        gates.append(_fixture_manifest_gate(args.folder))
        gates.append(_benchmark_smoke_gate(args.folder, report_dir, args.allow_downloads, fallback))
    else:
        gates.append(GateResult("benchmark_smoke", "SKIP", "Pass --folder to run a real-image smoke benchmark."))

    failed = any(gate.status == "FAIL" for gate in gates)
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_root": str(layout.root),
        "gates": [asdict(gate) for gate in gates],
        "status": "FAIL" if failed else "PASS",
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "pyqt_production_release_gates.json"
    md_path = report_dir / "pyqt_production_release_gates.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    md_path.write_text(_markdown(payload), encoding="utf-8")
    print(json.dumps({"json_report": str(json_path), "markdown_report": str(md_path), "status": payload["status"]}, indent=2))
    return 1 if failed else 0


def _path_gates(layout) -> list[GateResult]:
    gates = []
    for name, path in (
        ("logs_writable", layout.logs_dir),
        ("cache_writable", layout.cache_dir),
        ("crash_writable", layout.crash_dir),
        ("support_writable", layout.support_dir),
    ):
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".release_gate_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            gates.append(GateResult(name, "PASS", str(path)))
        except OSError as exc:
            gates.append(GateResult(name, "FAIL", f"{path}: {exc}"))
    return gates


def _benchmark_smoke_gate(folder: str, report_dir: Path, allow_downloads: bool, fallback_model: str | None) -> GateResult:
    try:
        from apps.pyqt_production import benchmark

        if not allow_downloads and not fallback_model:
            return GateResult("benchmark_smoke", "FAIL", "No bundled fallback model is available for the smoke benchmark.")
        model_name = fallback_model or "fast_preview"
        args = [
            "--bench",
            "cluster",
            "--folder",
            folder,
            "--models",
            model_name,
            "--backends",
            "cosine-kmeans",
            "--passes",
            "cold",
            "--report-dir",
            str(report_dir / "smoke_benchmark"),
            "--use-onnx",
        ]
        if allow_downloads:
            args.append("--allow-downloads")
        exit_code = benchmark.main(args)
        return GateResult(
            "benchmark_smoke",
            "PASS" if exit_code == 0 else "FAIL",
            f"exit_code={exit_code} model={model_name}",
        )
    except Exception as exc:
        return GateResult("benchmark_smoke", "FAIL", str(exc))


def _fixture_manifest_gate(folder: str) -> GateResult:
    try:
        from apps.shared.release_fixture import verified_photo_fixture_metadata

        metadata = verified_photo_fixture_metadata(folder)
        if metadata is None:
            return GateResult("fixture_manifest", "SKIP", "No sibling fixture manifest; folder is treated as a user-supplied smoke input.")
        return GateResult(
            "fixture_manifest",
            "PASS",
            f"photos={metadata['fixture_photo_count']} manifest={metadata['fixture_manifest_path']}",
        )
    except Exception as exc:
        return GateResult("fixture_manifest", "FAIL", str(exc))


def _authoritative_ui_gate() -> GateResult:
    production_source = (REPO_ROOT / "apps" / "pyqt_production" / "app.py").read_text(encoding="utf-8")
    prohibited_imports = (
        "apps.pyqt_production.ui",
        "from .ui",
    )
    violations = [token for token in prohibited_imports if token in production_source]
    compatibility_root = REPO_ROOT / "apps" / "pyqt_production" / "ui"
    independent_copies = []
    for path in compatibility_root.glob("*.py"):
        if path.name == "__init__.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "from ui." not in text or len(text.splitlines()) > 5:
            independent_copies.append(path.name)
    passed = not violations and not independent_copies
    return GateResult(
        "authoritative_ui",
        "PASS" if passed else "FAIL",
        f"prohibited_imports={violations or '-'} independent_compatibility_widgets={independent_copies or '-'}",
    )


def _ui_ux_acceptance_gate(report_dir: Path) -> GateResult:
    log_path = report_dir / "ui_ux_acceptance.log"
    environment = _release_subprocess_environment(offscreen=True)
    command = [
        sys.executable,
        "-m",
        "unittest",
        "-q",
        "tests.test_ui_ux_acceptance",
        "tests.test_production_safety",
    ]
    try:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(
            f"$ {' '.join(command)}\n\nSTDOUT\n{completed.stdout}\n\nSTDERR\n{completed.stderr}",
            encoding="utf-8",
        )
        return GateResult(
            "ui_ux_acceptance",
            "PASS" if completed.returncode == 0 else "FAIL",
            f"exit_code={completed.returncode} log={log_path}",
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return GateResult("ui_ux_acceptance", "FAIL", str(exc))


def _startup_performance_gate(report_dir: Path) -> GateResult:
    report_path = report_dir / "startup_performance.json"
    marker = "__CLUSTERLENS_STARTUP__="
    script = f"""
import json
import os
import resource
import sys
import time
from pathlib import Path

started = time.perf_counter()
from apps.pyqt_production.app import ProductionClusterApp, RUNTIME_LAYOUT
from PyQt6.QtWidgets import QApplication
import_seconds = time.perf_counter() - started

app = QApplication.instance() or QApplication([])
window_started = time.perf_counter()
window = ProductionClusterApp(RUNTIME_LAYOUT)
window_seconds = time.perf_counter() - window_started
try:
    rss_pages = int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1])
    rss_mib = float(rss_pages * int(os.sysconf("SC_PAGE_SIZE"))) / (1024.0 * 1024.0)
except (OSError, ValueError, IndexError):
    rss_mib = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
heavy_modules = [
    name for name in ("torch", "onnxruntime", "app.services.face_search", "ui.search_pane")
    if name in sys.modules
]
window.close()
app.processEvents()
print({marker!r} + json.dumps({{
    "import_seconds": import_seconds,
    "window_seconds": window_seconds,
    "rss_mib": rss_mib,
    "eager_heavy_modules": heavy_modules,
}}))
"""
    environment = _release_subprocess_environment(offscreen=True)
    environment["PYTHONHASHSEED"] = "0"
    try:
        completed = subprocess.run(
            [sys.executable, "-c", script],
            cwd=REPO_ROOT,
            env=environment,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        marker_line = next(
            (line for line in reversed(completed.stdout.splitlines()) if line.startswith(marker)),
            "",
        )
        if completed.returncode != 0 or not marker_line:
            details = (completed.stderr or completed.stdout or "startup probe produced no result").strip()[-1000:]
            return GateResult("startup_performance", "FAIL", details)
        metrics = json.loads(marker_line[len(marker) :])
        metrics["thresholds"] = {
            "import_seconds": STARTUP_IMPORT_MAX_SECONDS,
            "window_seconds": STARTUP_WINDOW_MAX_SECONDS,
            "rss_mib": STARTUP_RSS_MAX_MIB,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
        passed = bool(
            float(metrics["import_seconds"]) <= STARTUP_IMPORT_MAX_SECONDS
            and float(metrics["window_seconds"]) <= STARTUP_WINDOW_MAX_SECONDS
            and float(metrics["rss_mib"]) <= STARTUP_RSS_MAX_MIB
            and not list(metrics["eager_heavy_modules"])
        )
        return GateResult(
            "startup_performance",
            "PASS" if passed else "FAIL",
            (
                f"import={float(metrics['import_seconds']):.3f}s/{STARTUP_IMPORT_MAX_SECONDS:.1f}s "
                f"window={float(metrics['window_seconds']):.3f}s/{STARTUP_WINDOW_MAX_SECONDS:.1f}s "
                f"rss={float(metrics['rss_mib']):.1f}MiB/{STARTUP_RSS_MAX_MIB:.0f}MiB "
                f"eager={list(metrics['eager_heavy_modules']) or '-'} report={report_path}"
            ),
        )
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as exc:
        return GateResult("startup_performance", "FAIL", str(exc))


def _markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Production Release Gates",
        "",
        f"Status: **{payload.get('status')}**",
        f"Runtime root: `{payload.get('runtime_root')}`",
        "",
        "| Gate | Status | Details |",
        "| --- | --- | --- |",
    ]
    for gate in payload.get("gates", []):
        if isinstance(gate, dict):
            lines.append(f"| {gate.get('name')} | {gate.get('status')} | {str(gate.get('details')).replace('|', '/')} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
