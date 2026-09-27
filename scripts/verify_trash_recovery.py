"""Verify ClusterLens Trash recovery against disposable generated image files."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
TRASH_DIR_NAME = "ClusterLens Trash"
for import_path in (REPO_ROOT, SRC_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-dir", type=Path, required=True, help="Empty directory where disposable photos are created.")
    parser.add_argument("--report-dir", type=Path, required=True, help="Directory for a new JSON evidence report.")
    args = parser.parse_args(argv)
    fixture_dir = _require_empty_directory(args.fixture_dir)
    report_dir = args.report_dir.expanduser().resolve()
    report_path = report_dir / "trash_recovery.json"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite existing report: {report_path}")
    report_dir.mkdir(parents=True, exist_ok=True)

    from app.services.gallery_actions import CLUSTERLENS_TRASH_DIR_NAME, GalleryActionService

    sources = fixture_dir / "photos"
    sources.mkdir()
    originals = {
        "collision.png": "#ba4a00",
        "partial.png": "#1f618d",
        "restart.png": "#196f3d",
        "restore_conflict.png": "#7d3c98",
    }
    for name, color in originals.items():
        _write_image(sources / name, color)
    source_manifest_before = _file_manifest(sources)
    audit_log = report_dir / "runtime" / "logs" / "file_operations.jsonl"
    service = GalleryActionService(
        audit_log_path=audit_log,
        temp_dir=report_dir / "runtime" / "cache" / "tmp" / "file_ops",
    )
    trash_dir = sources / CLUSTERLENS_TRASH_DIR_NAME

    _write_image(trash_dir / "collision.png", "#d35400")
    collision = service.move_to_trash([str(sources / "collision.png")])
    _require(len(collision.changed_paths) == 1 and not collision.failures, "collision move did not succeed")
    collision_target = Path(collision.changed_paths[0][1])
    _require(collision_target != trash_dir / "collision.png", "collision move overwrote an existing trash file")

    partial = service.move_to_trash([str(sources / "partial.png"), str(sources / "missing.png")])
    _require(len(partial.changed_paths) == 1, "partial operation did not preserve its successful move")
    _require(len(partial.failures) == 1 and "source file does not exist" in partial.failures[0], "partial failure was not actionable")

    restart_move = service.move_to_trash([str(sources / "restart.png")])
    _require(len(restart_move.changed_paths) == 1 and not restart_move.failures, "restart fixture move did not succeed")
    restarted_service = GalleryActionService(audit_log_path=audit_log, temp_dir=service.temp_dir)
    restart_restore = restarted_service.restore_changed_paths(list(restart_move.changed_paths))
    _require(not restart_restore.failures and (sources / "restart.png").is_file(), "restart recovery did not restore its file")

    conflict_move = restarted_service.move_to_trash([str(sources / "restore_conflict.png")])
    _require(len(conflict_move.changed_paths) == 1 and not conflict_move.failures, "conflict fixture move did not succeed")
    _write_image(sources / "restore_conflict.png", "#f4d03f")
    skipped = restarted_service.restore_changed_paths(list(conflict_move.changed_paths), conflict_policy="skip")
    _require(len(skipped.failures) == 1 and "already exists" in skipped.failures[0], "skip policy did not preserve replacement")
    skipped_preserves_both = Path(conflict_move.changed_paths[0][1]).is_file()
    _require(skipped_preserves_both, "skip policy removed the recoverable original")
    unique = restarted_service.restore_changed_paths(list(conflict_move.changed_paths), conflict_policy="unique_name")
    _require(len(unique.changed_paths) == 1 and not unique.failures, "unique-name restore did not succeed")
    unique_target = Path(unique.changed_paths[0][1])
    _require(unique_target != sources / "restore_conflict.png" and unique_target.is_file(), "unique restore overwrote replacement")

    current_manifest = _file_manifest(sources)
    _require(current_manifest["restart.png"] == source_manifest_before["restart.png"], "restart recovery changed image content")
    _require(current_manifest["restore_conflict.png"] != source_manifest_before["restore_conflict.png"], "replacement was unexpectedly overwritten")
    _require(_sha256_file(unique_target) == source_manifest_before["restore_conflict.png"], "unique restore lost original content")
    journal_entries = restarted_service.read_audit_entries(limit=50)
    _require(len(journal_entries) >= 7, "journal lost operations after restart")

    payload = {
        "report_version": "1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": "disposable-real-file-trash-recovery",
        "validation": "PASS",
        "fixture_dir": str(fixture_dir),
        "audit_log": str(audit_log),
        "journal_path": str(restarted_service.journal_path),
        "checks": {
            "trash_collision_uses_unique_name": str(collision_target.relative_to(trash_dir)),
            "partial_failure_preserves_success": len(partial.changed_paths) == 1 and len(partial.failures) == 1,
            "restart_recovery_restores_original": current_manifest["restart.png"] == source_manifest_before["restart.png"],
            "skip_policy_preserves_both": skipped_preserves_both,
            "unique_policy_preserves_both": unique_target != sources / "restore_conflict.png",
            "journal_operation_count": len(journal_entries),
        },
        "network": "disabled by workload",
    }
    report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"json_report": str(report_path), "validation": payload["validation"]}, indent=2))
    return 0


def _require_empty_directory(path: Path) -> Path:
    target = path.expanduser().resolve()
    if target.exists() and any(target.iterdir()):
        raise ValueError(f"fixture directory must be empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    return target


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _write_image(path: Path, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (16, 12), color).save(path, format="PNG")


def _file_manifest(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)).replace("\\", "/"): _sha256_file(path)
        for path in sorted(root.rglob("*.png"))
        if TRASH_DIR_NAME not in path.parts
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
