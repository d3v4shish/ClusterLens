from __future__ import annotations

import faulthandler
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import traceback
from dataclasses import dataclass
from logging.handlers import RotatingFileHandler
from pathlib import Path


CLUSTERLENS_RUNTIME_ROOT_ENV = "CLUSTERLENS_RUNTIME_ROOT"
LEGACY_RUNTIME_ROOT_ENV = "IMAGE_CLUSTERING_APP_DIR"


def runtime_root_override() -> str:
    return str(
        os.environ.get(CLUSTERLENS_RUNTIME_ROOT_ENV)
        or os.environ.get(LEGACY_RUNTIME_ROOT_ENV)
        or ""
    ).strip()


@dataclass(frozen=True)
class RuntimeLayout:
    app_name: str
    root: Path
    logs_dir: Path
    cache_dir: Path
    crash_dir: Path
    benchmarks_dir: Path
    model_assets_dir: Path
    support_dir: Path
    app_log: Path
    qt_diagnostics_log: Path
    crash_log: Path
    last_crash_json: Path


def candidate_runtime_roots(app_name: str) -> list[Path]:
    roots: list[Path] = []
    override = runtime_root_override()
    if override:
        roots.append(Path(override))
    configured = configured_data_home(app_name)
    if configured is not None:
        roots.append(configured)
    roots.extend(_platform_runtime_roots(app_name))
    roots.append(Path.cwd() / ".runtime" / app_name)
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in roots:
        key = os.path.normcase(os.path.abspath(str(candidate)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def data_home_config_path(app_name: str) -> Path:
    """Return the small control-plane file, deliberately outside Data Home.

    A relocation cannot keep its active-location pointer only inside the
    directory being moved.  The config file contains no photo data and lets a
    failed new Data Home fall back to the platform default on the next launch.
    """
    if os.name == "nt":
        base = Path(os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Roaming")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / app_name / "data_home.json"


def configured_data_home(app_name: str) -> Path | None:
    config_path = data_home_config_path(app_name)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        value = str(payload.get("data_home", "") or "").strip()
        if not value:
            return None
        return Path(value).expanduser()
    except (OSError, ValueError, TypeError):
        return None


def _platform_runtime_roots(app_name: str) -> list[Path]:
    roots: list[Path] = []
    if os.name == "nt":
        local_appdata = os.environ.get("LOCALAPPDATA")
        if local_appdata:
            roots.append(Path(local_appdata) / app_name)
        appdata = os.environ.get("APPDATA")
        if appdata:
            roots.append(Path(appdata) / app_name)
        return roots

    home = _home_path()
    if sys.platform == "darwin":
        if home is not None:
            roots.append(home / "Library" / "Application Support" / app_name)
        return roots

    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        roots.append(Path(xdg_data_home) / app_name)
    elif home is not None:
        roots.append(home / ".local" / "share" / app_name)
    return roots


def _home_path() -> Path | None:
    raw_home = os.path.expanduser("~")
    if not raw_home or raw_home == "~":
        return None
    return Path(raw_home)


def resolve_runtime_root(app_name: str) -> Path:
    last_error: Exception | None = None
    for candidate in candidate_runtime_roots(app_name):
        try:
            (candidate / "logs").mkdir(parents=True, exist_ok=True)
            (candidate / "cache").mkdir(parents=True, exist_ok=True)
            probe = candidate / "cache" / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return candidate
        except OSError as exc:
            last_error = exc
            continue
    raise PermissionError(f"Unable to create a writable runtime root for {app_name}: {last_error}")


def activate_runtime_root(app_name: str, *, legacy_app_names: tuple[str, ...] = ()) -> RuntimeLayout:
    root = resolve_runtime_root(app_name)
    migrate_legacy_runtime_roots(root, legacy_runtime_roots(legacy_app_names))
    os.environ[CLUSTERLENS_RUNTIME_ROOT_ENV] = str(root)
    os.environ["IMAGE_CLUSTERING_APP_DIR"] = str(root)
    logs_dir = root / "logs"
    cache_dir = root / "cache"
    crash_dir = root / "crash"
    benchmarks_dir = root / "benchmarks"
    model_assets_dir = root / "model_assets"
    support_dir = root / "support"
    for path in (logs_dir, cache_dir, crash_dir, benchmarks_dir, model_assets_dir, support_dir):
        path.mkdir(parents=True, exist_ok=True)
    return RuntimeLayout(
        app_name=app_name,
        root=root,
        logs_dir=logs_dir,
        cache_dir=cache_dir,
        crash_dir=crash_dir,
        benchmarks_dir=benchmarks_dir,
        model_assets_dir=model_assets_dir,
        support_dir=support_dir,
        app_log=logs_dir / "app.log",
        qt_diagnostics_log=logs_dir / "qt_diagnostics.log",
        crash_log=crash_dir / "crash.log",
        last_crash_json=crash_dir / "last_crash.json",
    )


def legacy_runtime_roots(app_names: tuple[str, ...]) -> tuple[Path, ...]:
    roots: list[Path] = []
    for app_name in app_names:
        roots.extend(_platform_runtime_roots(app_name))
        roots.append(Path.cwd() / ".runtime" / app_name)
    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in roots:
        key = os.path.normcase(os.path.abspath(str(candidate)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return tuple(deduped)


def migrate_legacy_runtime_roots(root: Path, legacy_roots: tuple[Path, ...]) -> None:
    marker = root / "runtime_identity_migration.json"
    if marker.exists():
        return

    actions: list[str] = []
    failures: list[str] = []
    source_root = None
    try:
        resolved_root = root.resolve()
    except OSError:
        resolved_root = root.absolute()

    for candidate in legacy_roots:
        if not candidate.exists() or candidate.is_symlink():
            continue
        try:
            resolved_candidate = candidate.resolve()
        except OSError:
            resolved_candidate = candidate.absolute()
        if resolved_candidate == resolved_root:
            continue
        source_root = candidate
        _copy_missing_tree(candidate, root, actions, failures)
        break

    if source_root is None and not actions and not failures:
        return

    payload = {
        "target_root": str(root),
        "source_root": str(source_root) if source_root is not None else "",
        "actions": actions,
        "failures": failures,
    }
    try:
        marker.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except OSError:
        pass


def _copy_missing_tree(source: Path, target: Path, actions: list[str], failures: list[str]) -> None:
    for dirpath, dirnames, filenames in os.walk(source, followlinks=False):
        current = Path(dirpath)
        dirnames[:] = [dirname for dirname in dirnames if not (current / dirname).is_symlink()]
        try:
            relative = current.relative_to(source)
        except ValueError:
            continue
        target_dir = target / relative
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            failures.append(f"mkdir:{target_dir}:{exc}")
            continue
        for filename in filenames:
            source_file = current / filename
            if source_file.is_symlink():
                continue
            target_file = target_dir / filename
            if target_file.exists():
                continue
            try:
                shutil.copy2(source_file, target_file)
                actions.append(f"copied:{source_file.relative_to(source)}")
            except OSError as exc:
                failures.append(f"copy:{source_file}:{exc}")


def configure_rotating_logging(
    layout: RuntimeLayout,
    logger_name: str = "",
    *,
    force: bool = False,
) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    root = logging.getLogger()
    configured_for = getattr(configure_rotating_logging, "_configured_for", "")
    if (
        not force
        and configured_for == str(layout.app_log)
        and _has_file_handler(root, layout.app_log)
    ):
        return logger
    root.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    file_handler = RotatingFileHandler(layout.app_log, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)
    configure_rotating_logging._configured = True
    configure_rotating_logging._configured_for = str(layout.app_log)
    return logger


def _has_file_handler(logger: logging.Logger, path: Path) -> bool:
    expected = _safe_resolve(path)
    for handler in logger.handlers:
        base_filename = getattr(handler, "baseFilename", "")
        if not base_filename:
            continue
        if _safe_resolve(Path(base_filename)) == expected:
            return True
    return False


def _safe_resolve(path: Path) -> str:
    try:
        return str(path.resolve()).casefold()
    except OSError:
        return str(path.absolute()).casefold()


def flush_logging_handlers() -> None:
    for handler in logging.getLogger().handlers:
        try:
            handler.flush()
        except Exception:
            pass


def install_crash_handlers(layout: RuntimeLayout) -> None:
    if getattr(install_crash_handlers, "_installed_for", None) == str(layout.root):
        return
    crash_stream = layout.crash_log.open("a", encoding="utf-8")
    faulthandler.enable(crash_stream)

    def _write_crash_summary(kind: str, exc_type, exc_value, exc_tb) -> None:
        traceback_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        payload = {
            "kind": kind,
            "type": getattr(exc_type, "__name__", str(exc_type)),
            "message": str(exc_value),
            "traceback": traceback_text,
        }
        layout.last_crash_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with layout.crash_log.open("a", encoding="utf-8") as handle:
            handle.write(f"[{kind}] {payload['type']}: {payload['message']}\n")
            handle.write(traceback_text)
            handle.write("\n")

    def _sys_hook(exc_type, exc_value, exc_tb) -> None:
        _write_crash_summary("sys", exc_type, exc_value, exc_tb)
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    def _thread_hook(args: threading.ExceptHookArgs) -> None:
        _write_crash_summary("thread", args.exc_type, args.exc_value, args.exc_traceback)
        threading.__excepthook__(args)

    sys.excepthook = _sys_hook
    threading.excepthook = _thread_hook
    install_crash_handlers._installed_for = str(layout.root)


def open_path_in_shell(path: str | Path) -> None:
    resolved = Path(path).resolve()
    if os.name == "nt":
        os.startfile(str(resolved))  # type: ignore[attr-defined]
        return
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(resolved)])
        return
    opener = shutil.which("xdg-open")
    if opener:
        subprocess.Popen([opener, str(resolved)])
        return
    gio = shutil.which("gio")
    if gio:
        subprocess.Popen([gio, "open", str(resolved)])
        return
    raise OSError(f"Opening paths is unsupported on this platform: {resolved}")
