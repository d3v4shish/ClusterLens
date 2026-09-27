from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from app.services.model_assets import (
    HF_CACHE_REQUIRED_FILES,
    HF_CACHE_TEXT_REQUIRED_FILES,
    HF_MODEL_REPOSITORIES,
    HF_MODEL_REVISIONS,
    ModelAssetService,
    _complete_hf_cache_snapshot,
)
from infra.atomic_io import atomic_write_text
from infra.cancel import raise_if_cancelled
from infra.performance import select_performance_profile
from infra.runtime import RuntimeCapabilityService
from infra.settings import AppSettings, configure_model_cache_environment, get_settings


MODEL_DOWNLOAD_LOCK_STALE_SECONDS = 24 * 60 * 60

# Loading a Hugging Face model through separate model/processor/tokenizer calls
# can resolve different revisions while a repository is being updated.  Download
# the files needed by ClusterLens as one snapshot first, so offline readiness
# and the runtime agree on the same revision.
HF_SNAPSHOT_REPOSITORIES = HF_MODEL_REPOSITORIES
HF_SNAPSHOT_ALLOW_PATTERNS = (
    "*.json",
    "*.txt",
    "*.model",
    "*.safetensors",
)


@dataclass(frozen=True)
class ModelDownloadItem:
    model_name: str
    require_text: bool = False

    @property
    def key(self) -> str:
        return f"{self.model_name}:text={int(self.require_text)}"


@dataclass(frozen=True)
class ModelDownloadResult:
    requested: tuple[str, ...]
    downloaded: tuple[str, ...]
    reused: tuple[str, ...]
    cache_bytes_before: int
    cache_bytes_after: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ModelCacheRecoveryResult:
    repaired_revisions: tuple[str, ...]
    complete_models: tuple[str, ...]


def normalize_model_download_items(items: list[ModelDownloadItem] | tuple[ModelDownloadItem, ...]) -> tuple[ModelDownloadItem, ...]:
    requirements: dict[str, bool] = {}
    order: list[str] = []
    for item in items:
        model_name = str(item.model_name or "").strip().lower()
        if not model_name:
            continue
        if model_name not in requirements:
            order.append(model_name)
            requirements[model_name] = False
        requirements[model_name] = bool(requirements[model_name] or item.require_text)
    return tuple(ModelDownloadItem(model_name, requirements[model_name]) for model_name in order)


class ModelDownloadService:
    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()
        self.assets = ModelAssetService(settings=self.settings)
        self.lock_root = Path(self.settings.cache_dir) / "model_download_locks"

    def recover_cached_models(self) -> ModelCacheRecoveryResult:
        """Make complete persisted Hugging Face snapshots active again.

        Downloads are stored in the application cache, but a process can stop
        after snapshot creation and before the ``refs/main`` pointer update.
        Readiness checks can still find the files while the runtime resolves a
        stale reference. Repair that local pointer without downloading or
        deleting anything.
        """
        configure_model_cache_environment(self.settings)
        repaired: list[str] = []
        complete: list[str] = []
        hub_root = Path(self.settings.cache_dir) / "huggingface" / "hub"
        for model_name, repository in HF_SNAPSHOT_REPOSITORIES.items():
            repo_dir = hub_root / f"models--{repository.replace('/', '--')}"
            base_groups = HF_CACHE_REQUIRED_FILES.get(model_name, ())
            text_groups = base_groups + HF_CACHE_TEXT_REQUIRED_FILES.get(model_name, ())
            snapshot = _complete_hf_cache_snapshot(repo_dir, text_groups)
            if snapshot is None:
                snapshot = _complete_hf_cache_snapshot(repo_dir, base_groups)
            if snapshot is None:
                continue
            complete.append(model_name)
            snapshots_dir = repo_dir / "snapshots"
            try:
                snapshot.relative_to(snapshots_dir)
            except ValueError:
                continue
            refs_dir = repo_dir / "refs"
            refs_dir.mkdir(parents=True, exist_ok=True)
            ref_path = refs_dir / "main"
            try:
                current = ref_path.read_text(encoding="utf-8").strip()
            except OSError:
                current = ""
            if current != snapshot.name:
                atomic_write_text(ref_path, snapshot.name)
                repaired.append(model_name)
        return ModelCacheRecoveryResult(
            repaired_revisions=tuple(repaired),
            complete_models=tuple(complete),
        )

    def acquire(
        self,
        items: list[ModelDownloadItem] | tuple[ModelDownloadItem, ...],
        *,
        progress_callback=None,
        cancel_check=None,
    ) -> ModelDownloadResult:
        requested_items = normalize_model_download_items(items)
        requested = tuple(item.key for item in requested_items)
        before = sum(self.assets.local_cache_size(item.model_name) for item in requested_items)
        downloaded: list[str] = []
        reused: list[str] = []
        total = max(1, len(requested_items))

        for index, item in enumerate(requested_items):
            raise_if_cancelled(cancel_check)
            start_progress = int(index * 100 / total)
            if self.assets.model_available_without_download(item.model_name, require_text=item.require_text):
                reused.append(item.key)
                if progress_callback:
                    progress_callback(start_progress, f"Using cached {self._label(item)}")
                continue

            lease = SharedDownloadLease(
                self.lock_root,
                item.model_name,
                wait_message="Waiting for the existing model download to finish...",
            )
            lease.acquire(progress_callback=progress_callback, cancel_check=cancel_check)
            try:
                raise_if_cancelled(cancel_check)
                if self.assets.model_available_without_download(item.model_name, require_text=item.require_text):
                    reused.append(item.key)
                    if progress_callback:
                        progress_callback(start_progress, f"Another process completed {self._label(item)}; using cache")
                    continue
                if progress_callback:
                    progress_callback(start_progress, f"Downloading {self._label(item)}")
                self._load_and_verify(item)
                raise_if_cancelled(cancel_check)
                downloaded.append(item.key)
                if progress_callback:
                    progress_callback(int((index + 1) * 100 / total), f"Cached {self._label(item)}")
            finally:
                lease.release()

        after = sum(self.assets.local_cache_size(item.model_name) for item in requested_items)
        return ModelDownloadResult(
            requested=requested,
            downloaded=tuple(downloaded),
            reused=tuple(reused),
            cache_bytes_before=int(before),
            cache_bytes_after=int(after),
        )

    def _load_and_verify(self, item: ModelDownloadItem) -> None:
        from ml.embeddings import ModelManager

        self._download_huggingface_snapshot(item)
        runtime_service = RuntimeCapabilityService()
        manager = ModelManager(
            use_onnx=False,
            execution_policy=runtime_service.select_policy("cpu"),
            runtime_service=runtime_service,
            performance_profile=select_performance_profile("low_memory"),
            allow_model_downloads=True,
            load_text_tokenizer=bool(item.require_text),
        )
        try:
            bundle = manager.get_bundle(item.model_name, use_onnx=False)
            if item.require_text and bundle.text_tokenizer is None:
                raise RuntimeError(f"The text tokenizer for {item.model_name} was not installed.")
        finally:
            del manager
            gc.collect()

        self._consolidate_huggingface_snapshot(item)
        if not self.assets.model_available_without_download(item.model_name, require_text=item.require_text):
            asset_kind = "image weights and text tokenizer" if item.require_text else "image weights"
            raise RuntimeError(f"Downloaded {item.model_name}, but its {asset_kind} did not pass the local cache readiness check.")

    def _download_huggingface_snapshot(self, item: ModelDownloadItem) -> None:
        """Populate one complete revision for models stored on Hugging Face."""
        repository = HF_SNAPSHOT_REPOSITORIES.get(str(item.model_name or "").strip().lower())
        if not repository:
            return
        configure_model_cache_environment(self.settings)
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=repository,
            revision=HF_MODEL_REVISIONS[str(item.model_name or "").strip().lower()],
            cache_dir=str(Path(self.settings.cache_dir) / "huggingface" / "hub"),
            allow_patterns=HF_SNAPSHOT_ALLOW_PATTERNS,
        )

    def _consolidate_huggingface_snapshot(self, item: ModelDownloadItem) -> None:
        """Make a coherent offline snapshot after Hugging Face resolves split revisions.

        Some repositories publish model weights and tokenizer/configuration files
        from different cache revisions. Transformers can load that combination
        online, but an offline run resolves one revision through ``refs/main``.
        Link the verified artifacts into one local snapshot once the runtime has
        loaded them successfully.
        """
        model_name = str(item.model_name or "").strip().lower()
        repository = HF_SNAPSHOT_REPOSITORIES.get(model_name)
        if not repository:
            return
        repo_dir = Path(self.settings.cache_dir) / "huggingface" / "hub" / f"models--{repository.replace('/', '--')}"
        snapshots_dir = repo_dir / "snapshots"
        if not snapshots_dir.is_dir():
            return

        preferred_revision = ""
        try:
            preferred_revision = (repo_dir / "refs" / "main").read_text(encoding="utf-8").strip()
        except OSError:
            pass
        source_snapshots = [path for path in snapshots_dir.iterdir() if path.is_dir() and not path.name.startswith("clusterlens-complete-")]
        source_snapshots.sort(key=lambda path: (path.name != preferred_revision, path.name))
        artifacts: dict[str, Path] = {}
        for snapshot in source_snapshots:
            for path in snapshot.iterdir():
                if path.name not in artifacts and path.is_file():
                    artifacts[path.name] = path
        if not artifacts:
            return

        fingerprint = hashlib.sha256(
            "\n".join(f"{name}:{path.resolve()}" for name, path in sorted(artifacts.items())).encode("utf-8")
        ).hexdigest()[:16]
        target_name = f"clusterlens-complete-{fingerprint}"
        target_dir = snapshots_dir / target_name
        if not target_dir.exists():
            staging_dir = snapshots_dir / f".{target_name}-{uuid4().hex}.tmp"
            try:
                staging_dir.mkdir()
                for name, source in artifacts.items():
                    target = staging_dir / name
                    if source.is_symlink():
                        target.symlink_to(os.readlink(source))
                    else:
                        try:
                            os.link(source, target)
                        except OSError:
                            shutil.copy2(source, target)
                staging_dir.replace(target_dir)
            except Exception:
                shutil.rmtree(staging_dir, ignore_errors=True)
                raise
        (repo_dir / "refs").mkdir(parents=True, exist_ok=True)
        atomic_write_text(repo_dir / "refs" / "main", target_name)

    @staticmethod
    def _label(item: ModelDownloadItem) -> str:
        suffix = " image weights and text tokenizer" if item.require_text else " image weights"
        return f"{item.model_name}{suffix}"


class SharedDownloadLease:
    """Cross-process lease for a named cache acquisition.

    Image-only and image+text requests intentionally share a model key so two
    processes cannot initialize the same weights concurrently. Face-model
    archives use the same primitive with their content hash as the key.
    """

    def __init__(self, lock_root: Path, key: str, *, wait_message: str = "Waiting for an existing download...") -> None:
        safe_name = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in str(key or "download")
        )
        self.lock_dir = Path(lock_root) / f"{safe_name}.lock"
        self.owner_path = self.lock_dir / "owner.json"
        self.wait_message = str(wait_message or "Waiting for an existing download...")
        self.token = uuid4().hex
        self.acquired = False

    def acquire(self, *, progress_callback=None, cancel_check=None) -> None:
        self.lock_dir.parent.mkdir(parents=True, exist_ok=True)
        last_notice = 0.0
        while True:
            raise_if_cancelled(cancel_check)
            try:
                self.lock_dir.mkdir()
            except FileExistsError:
                if self._is_stale():
                    try:
                        shutil.rmtree(self.lock_dir)
                    except FileNotFoundError:
                        pass
                    except OSError:
                        time.sleep(0.25)
                    continue
                now = time.monotonic()
                if progress_callback and now - last_notice >= 1.0:
                    progress_callback(-1, self.wait_message)
                    last_notice = now
                time.sleep(0.25)
                continue
            self.acquired = True
            try:
                atomic_write_text(
                    self.owner_path,
                    json.dumps(
                        {
                            "pid": os.getpid(),
                            "process_start": _process_start_token(os.getpid()),
                            "token": self.token,
                            "created_at": time.time(),
                        }
                    ),
                )
            except Exception:
                self.acquired = False
                shutil.rmtree(self.lock_dir, ignore_errors=True)
                raise
            return

    def release(self) -> None:
        if not self.acquired:
            return
        self.acquired = False
        try:
            payload = json.loads(self.owner_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
        if str(payload.get("token") or "") != self.token:
            return
        try:
            shutil.rmtree(self.lock_dir)
        except FileNotFoundError:
            pass

    def _is_stale(self) -> bool:
        try:
            stat = self.lock_dir.stat()
        except FileNotFoundError:
            return True
        age = max(0.0, time.time() - float(stat.st_mtime))
        try:
            payload = json.loads(self.owner_path.read_text(encoding="utf-8"))
            pid = int(payload.get("pid") or 0)
            expected_start = str(payload.get("process_start") or "")
        except Exception:
            return age > 10.0
        if not _process_is_alive(pid):
            return True
        actual_start = _process_start_token(pid)
        if expected_start and actual_start and expected_start != actual_start:
            return True
        return age > MODEL_DOWNLOAD_LOCK_STALE_SECONDS


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _process_start_token(pid: int) -> str:
    """Return a Linux process birth token so PID reuse cannot strand a lease."""
    try:
        payload = Path(f"/proc/{int(pid)}/stat").read_text(encoding="utf-8")
        closing = payload.rfind(")")
        if closing < 0:
            return ""
        fields_after_name = payload[closing + 2 :].split()
        # /proc/<pid>/stat field 22 (starttime); fields_after_name begins at 3.
        return str(fields_after_name[19])
    except (OSError, ValueError, IndexError):
        return ""
