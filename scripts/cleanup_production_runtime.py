from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from apps.pyqt_production.identity import LEGACY_PRODUCTION_APP_IDS, PRODUCTION_APP_ID  # noqa: E402


APP_NAME = PRODUCTION_APP_ID
VALID_RUNTIME_ROOT_NAMES = {APP_NAME, "clusterlens", *LEGACY_PRODUCTION_APP_IDS}
RUNTIME_MARKER_NAMES = {".image-clustering-runtime", ".clusterlens-runtime"}

CACHE_DIR_TARGETS = (
    "tmp",
    "embedding_indexes",
    "thumbnails",
    "thumbnail_cache",
    "result_cache",
)
CACHE_FILE_NAMES = {
    "embeddings.sqlite3",
    "embeddings.sqlite",
    "embeddings.db",
}
CACHE_SUFFIX_TARGETS = (
    ".npy",
    ".faiss",
    ".tmp",
    ".temp",
    ".partial",
    ".incomplete",
    ".lock",
    ".sqlite-wal",
    ".sqlite-shm",
    ".sqlite3-wal",
    ".sqlite3-shm",
    ".db-wal",
    ".db-shm",
)


@dataclass(frozen=True)
class CleanupTarget:
    path: Path
    reason: str
    kind: str
    size_bytes: int


@dataclass(frozen=True)
class CleanupPlan:
    runtime_root: Path
    targets: tuple[CleanupTarget, ...]
    skipped_symlinks: tuple[Path, ...]

    @property
    def total_bytes(self) -> int:
        return sum(target.size_bytes for target in self.targets)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    runtime_root = resolve_runtime_root(args.runtime_root)
    if not runtime_root.exists():
        print(f"Runtime root does not exist: {runtime_root}")
        return 0

    safety_error = validate_runtime_root(runtime_root, force=bool(args.force_runtime_root))
    if safety_error:
        print(f"Refusing cleanup: {safety_error}", file=sys.stderr)
        return 2

    if args.all_user_data and not args.i_understand_this_deletes_user_data:
        print(
            "--all-user-data requires --i-understand-this-deletes-user-data.",
            file=sys.stderr,
        )
        return 2

    plan = build_cleanup_plan(runtime_root, args)
    print_plan(plan, delete=bool(args.yes))

    if not args.yes:
        print("Dry run only. Re-run with --yes to delete the listed targets.")
        return 0

    failures = delete_targets(plan.targets)
    if failures:
        print("Cleanup completed with failures:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("Cleanup completed.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Safely clean rebuildable ClusterLens production runtime files. "
            "The default mode is a dry run."
        )
    )
    parser.add_argument("--runtime-root", help="Override the production runtime root.")
    parser.add_argument("--yes", action="store_true", help="Delete targets instead of performing a dry run.")
    parser.add_argument("--force-runtime-root", action="store_true", help="Allow a runtime root without the app name/marker.")
    parser.add_argument("--logs", action="store_true", help="Also delete runtime logs.")
    parser.add_argument("--crash", action="store_true", help="Also delete crash records.")
    parser.add_argument("--support", action="store_true", help="Also delete support bundles.")
    parser.add_argument("--benchmarks", action="store_true", help="Also delete benchmark output.")
    parser.add_argument("--model-assets", action="store_true", help="Also delete installed model assets.")
    parser.add_argument("--all-user-data", action="store_true", help="Delete the entire runtime root.")
    parser.add_argument(
        "--i-understand-this-deletes-user-data",
        action="store_true",
        help="Required with --all-user-data.",
    )
    return parser


def resolve_runtime_root(runtime_root_arg: str | None) -> Path:
    if runtime_root_arg:
        return Path(runtime_root_arg).expanduser().absolute()

    env_root = os.environ.get("IMAGE_CLUSTERING_APP_DIR")
    if env_root:
        return Path(env_root).expanduser().absolute()

    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return (Path(base) / APP_NAME).expanduser().absolute()
        return (Path.home() / "AppData" / "Local" / APP_NAME).absolute()

    if sys.platform == "darwin":
        return (Path.home() / "Library" / "Application Support" / APP_NAME).absolute()

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        return (Path(xdg_data_home) / APP_NAME).expanduser().absolute()
    return (Path.home() / ".local" / "share" / APP_NAME).absolute()


def validate_runtime_root(runtime_root: Path, *, force: bool) -> str | None:
    try:
        root = runtime_root.resolve()
    except OSError as exc:
        return f"could not resolve runtime root {runtime_root}: {exc}"

    if str(root) in {"", "."}:
        return "runtime root is empty."
    if root.parent == root:
        return f"runtime root is a filesystem root: {root}"
    try:
        home = Path.home().resolve()
    except OSError:
        home = None
    if home is not None and root == home:
        return f"runtime root is the user home directory: {root}"
    try:
        repo_root = REPO_ROOT.resolve()
    except OSError:
        repo_root = None
    if repo_root is not None and root == repo_root:
        return f"runtime root is the repository root: {root}"
    if any(part == ".git" for part in root.parts):
        return f"runtime root is inside a .git directory: {root}"

    if force:
        return None
    if root.name in VALID_RUNTIME_ROOT_NAMES:
        return None
    if any((root / marker).exists() for marker in RUNTIME_MARKER_NAMES):
        return None
    return (
        f"runtime root must be named one of {sorted(VALID_RUNTIME_ROOT_NAMES)} "
        "or contain a runtime marker; pass --force-runtime-root for test/custom roots."
    )


def build_cleanup_plan(runtime_root: Path, args: argparse.Namespace) -> CleanupPlan:
    runtime_root = runtime_root.absolute()
    if args.all_user_data:
        target = make_target(runtime_root, "all user data")
        targets = (target,) if target is not None else ()
        skipped = (runtime_root,) if runtime_root.is_symlink() else ()
        return CleanupPlan(runtime_root=runtime_root, targets=targets, skipped_symlinks=skipped)

    targets: list[CleanupTarget] = []
    skipped_symlinks: list[Path] = []

    cache_dir = runtime_root / "cache"
    for relative in CACHE_DIR_TARGETS:
        add_target(cache_dir / relative, f"rebuildable cache directory: cache/{relative}", targets, skipped_symlinks)

    for name in CACHE_FILE_NAMES:
        add_target(cache_dir / name, f"rebuildable embedding cache database: cache/{name}", targets, skipped_symlinks)

    for path in walk_files(cache_dir):
        if path.name in CACHE_FILE_NAMES or path.suffix.lower() in CACHE_SUFFIX_TARGETS:
            add_target(path, f"rebuildable cache file: {path.name}", targets, skipped_symlinks)

    add_target(cache_dir / "huggingface" / ".locks", "Hugging Face download locks", targets, skipped_symlinks)

    optional_dirs = (
        ("logs", args.logs, "runtime logs"),
        ("crash", args.crash, "crash records"),
        ("support", args.support, "support bundles"),
        ("benchmarks", args.benchmarks, "benchmark output"),
        ("model_assets", args.model_assets, "installed model assets"),
    )
    for dirname, enabled, reason in optional_dirs:
        if enabled:
            add_target(runtime_root / dirname, reason, targets, skipped_symlinks)

    return CleanupPlan(
        runtime_root=runtime_root,
        targets=tuple(minimize_targets(targets)),
        skipped_symlinks=tuple(dedupe_paths(skipped_symlinks)),
    )


def add_target(path: Path, reason: str, targets: list[CleanupTarget], skipped_symlinks: list[Path]) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_symlink():
        skipped_symlinks.append(path)
        return
    target = make_target(path, reason)
    if target is not None:
        targets.append(target)


def make_target(path: Path, reason: str) -> CleanupTarget | None:
    if not path.exists() and not path.is_symlink():
        return None
    if path.is_symlink():
        return None
    kind = "dir" if path.is_dir() else "file"
    return CleanupTarget(path=path.absolute(), reason=reason, kind=kind, size_bytes=path_size(path))


def walk_files(root: Path) -> tuple[Path, ...]:
    if not root.exists() or root.is_symlink():
        return ()
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = [dirname for dirname in dirnames if not (current / dirname).is_symlink()]
        for filename in filenames:
            files.append(current / filename)
    return tuple(files)


def path_size(path: Path) -> int:
    if path.is_symlink():
        return 0
    try:
        if path.is_file():
            return path.stat().st_size
        if not path.is_dir():
            return path.stat().st_size
    except OSError:
        return 0

    total = 0
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = [dirname for dirname in dirnames if not (current / dirname).is_symlink()]
        for filename in filenames:
            candidate = current / filename
            if candidate.is_symlink():
                continue
            try:
                total += candidate.stat().st_size
            except OSError:
                continue
    return total


def minimize_targets(targets: list[CleanupTarget]) -> tuple[CleanupTarget, ...]:
    deduped: dict[Path, CleanupTarget] = {}
    for target in targets:
        deduped[target.path] = target
    sorted_targets = sorted(deduped.values(), key=lambda target: (len(target.path.parts), str(target.path)))

    kept: list[CleanupTarget] = []
    for target in sorted_targets:
        if any(is_relative_to(target.path, kept_target.path) for kept_target in kept):
            continue
        kept.append(target)
    return tuple(kept)


def dedupe_paths(paths: list[Path]) -> tuple[Path, ...]:
    return tuple(sorted({path.absolute() for path in paths}, key=str))


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return path != parent
    except ValueError:
        return False


def print_plan(plan: CleanupPlan, *, delete: bool) -> None:
    mode = "delete" if delete else "dry-run"
    print(f"Runtime root: {plan.runtime_root}")
    print(f"Mode: {mode}")
    print(f"Targets: {len(plan.targets)}")
    print(f"Total bytes: {plan.total_bytes}")
    for target in plan.targets:
        print(f"  - [{target.kind}] {target.path} ({target.reason}, {target.size_bytes} bytes)")
    if plan.skipped_symlinks:
        print("Skipped symlinks:")
        for path in plan.skipped_symlinks:
            print(f"  - {path}")
    if not plan.targets:
        print("No cleanup targets found.")


def delete_targets(targets: tuple[CleanupTarget, ...]) -> list[str]:
    failures: list[str] = []
    for target in targets:
        path = target.path
        if path.is_symlink():
            failures.append(f"refusing to delete symlink: {path}")
            continue
        try:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as exc:
            failures.append(f"{path}: {exc}")
    return failures


if __name__ == "__main__":
    raise SystemExit(main())
