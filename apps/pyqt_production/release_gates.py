from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from apps.pyqt_production.bootstrap import bootstrap_runtime
from apps.shared.runtime_migrations import RuntimeMigrationService
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
            "PASS" if not migrations.failures else "WARN",
            f"previous={migrations.previous_version} current={migrations.current_version} actions={len(migrations.actions)} failures={len(migrations.failures)}",
        )
    )
    gates.extend(_path_gates(layout))

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
