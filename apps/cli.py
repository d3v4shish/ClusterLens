from __future__ import annotations

"""Clean-native command line companion for inspection and managed-data care.

The CLI deliberately does not expose source-changing actions.  Face labels,
metadata edits, renames, move/trash/restore, model installation, and indexing
remain GUI-only because they need visual review and the Jobs safety surface.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def _prepare_imports(data_home: str | None) -> None:
    if str(SRC_ROOT) not in sys.path:
        sys.path.insert(0, str(SRC_ROOT))
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    if data_home:
        os.environ["CLUSTERLENS_RUNTIME_ROOT"] = str(Path(data_home).expanduser().absolute())
        os.environ["IMAGE_CLUSTERING_APP_DIR"] = os.environ["CLUSTERLENS_RUNTIME_ROOT"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-home", help="Inspect this Data Home instead of the configured location.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("status", help="Show the active Data Home and shared root scope.")
    scope = commands.add_parser("scope", help="Show the persisted shared active roots.")
    scope.add_argument("action", choices=("show",), nargs="?", default="show")

    catalog = commands.add_parser("catalog", help="Read the managed Library catalog without scanning media.")
    catalog_sub = catalog.add_subparsers(dest="catalog_command", required=True)
    search = catalog_sub.add_parser("search", help="Search registered catalog rows.")
    search.add_argument("text")
    search.add_argument("--limit", type=int, default=80)

    duplicates = commands.add_parser("duplicates", help="Build a read-only exact/near/burst review report.")
    duplicates.add_argument("--limit", type=int, default=200)

    storage = commands.add_parser("storage", help="Inspect, back up, or verify only ClusterLens-managed data.")
    storage_sub = storage.add_subparsers(dest="storage_command", required=True)
    storage_sub.add_parser("inventory", help="List managed Data Home sizes.")
    backup = storage_sub.add_parser("backup", help="Create a checksummed managed-data backup.")
    backup.add_argument("--destination", required=True, help="Existing directory that receives a new backup folder.")
    verify = storage_sub.add_parser("verify", help="Verify a checksummed backup.")
    verify.add_argument("--backup", required=True)
    recovery = storage_sub.add_parser("recovery", help="List Data Home migration journals.")
    recovery.add_argument("action", choices=("list",), nargs="?", default="list")

    benchmark = commands.add_parser("benchmark", help="Run an existing isolated deterministic benchmark.")
    benchmark.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to scripts/benchmark.sh.")
    return parser


def _runtime_root(data_home: str | None) -> Path:
    if data_home:
        return Path(data_home).expanduser().absolute()
    from apps.shared.runtime_support import candidate_runtime_roots

    return candidate_runtime_roots("ClusterLens")[0].expanduser().absolute()


def _active_roots() -> tuple[str, ...]:
    from apps.pyqt_production.identity import production_settings_store
    from app.path_scope import PathScope

    raw_value = production_settings_store().value("workspace/active_roots", "")
    try:
        values = json.loads(str(raw_value or "[]"))
    except (TypeError, ValueError):
        values = []
    return PathScope.from_paths(values if isinstance(values, list) else ()).roots


def _plain_or_json(value: object, *, json_mode: bool) -> int:
    if json_mode:
        print(json.dumps(value, indent=2, sort_keys=True, default=str))
        return 0
    if isinstance(value, dict):
        for key, item in value.items():
            print(f"{key}: {item}")
    elif isinstance(value, (list, tuple)):
        for item in value:
            print(item)
    else:
        print(value)
    return 0


def _data_home_manager(root: Path):
    from app.services.data_home import DataHomeManager

    return DataHomeManager("ClusterLens", root)


def run(args: argparse.Namespace) -> int:
    root = _runtime_root(args.data_home)
    if args.command in {"status", "scope"}:
        roots = _active_roots()
        return _plain_or_json(
            {"data_home": str(root), "active_roots": list(roots), "active_root_count": len(roots)},
            json_mode=args.json,
        )
    if args.command == "storage":
        manager = _data_home_manager(root)
        if args.storage_command == "inventory":
            inventory = manager.inventory()
            return _plain_or_json(
                {"data_home": inventory.root, "files": inventory.files, "bytes": inventory.bytes, "categories": inventory.categories},
                json_mode=args.json,
            )
        if args.storage_command == "backup":
            backup = manager.create_backup(args.destination)
            return _plain_or_json(backup.__dict__, json_mode=args.json)
        if args.storage_command == "verify":
            valid, failures = manager.verify_backup(args.backup)
            _plain_or_json({"valid": valid, "failures": list(failures)}, json_mode=args.json)
            return 0 if valid else 1
        journal_dir = root / "support" / "data_home_migrations"
        entries = [manager.migration_preview(path) for path in sorted(journal_dir.glob("*.json"), reverse=True)] if journal_dir.exists() else []
        return _plain_or_json(entries, json_mode=args.json)
    if args.command == "catalog":
        from app.services.library_catalog import CatalogQuery, LibraryCatalogService

        page = LibraryCatalogService().query_assets(
            CatalogQuery(text=str(args.text), scope_paths=_active_roots(), limit=max(1, min(1000, int(args.limit))))
        )
        payload = {
            "total_count": page.total_count,
            "items": [
                {
                    "path": item.image_path,
                    "captured_at": item.captured_at,
                    "capture_source": item.capture_source,
                    "capture_sequence": item.capture_sequence,
                    "camera": item.camera,
                }
                for item in page.items
            ],
        }
        return _plain_or_json(payload, json_mode=args.json)
    if args.command == "duplicates":
        from app.services.duplicate_review import DuplicateReviewService

        groups = DuplicateReviewService().build_groups(scope_paths=_active_roots())[: max(1, min(1000, int(args.limit)))]
        payload = [
            {
                "kind": group.kind,
                "keeper_path": group.keeper_path,
                "summary": group.summary,
                "members": [candidate.image_path for candidate in group.members],
            }
            for group in groups
        ]
        return _plain_or_json(payload, json_mode=args.json)
    if args.command == "benchmark":
        command = ["bash", str(REPO_ROOT / "scripts" / "benchmark.sh"), *list(args.args or ())]
        return int(subprocess.run(command, cwd=REPO_ROOT, check=False).returncode)
    raise ValueError(f"Unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _prepare_imports(args.data_home)
    try:
        return run(args)
    except (OSError, ValueError) as exc:
        print(f"ClusterLens CLI: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
