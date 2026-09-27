from __future__ import annotations

import hashlib
import http.client
import importlib.machinery
import importlib
import json
import os
import shutil
import sys
import tempfile
import time
import types
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from infra.cancel import raise_if_cancelled
from infra.atomic_io import atomic_write_text, atomic_write_with
from infra.settings import AppSettings, get_settings
from app.services.model_downloads import SharedDownloadLease


RECOMMENDED_MODEL_ARCHIVE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip"
RECOMMENDED_MODEL_ARCHIVE_SHA256 = "d98264bd8f2dc75cbc2ddce2a14e636e02bb857b3051c234b737bf3b614edca9"
EDGE_MODEL_ARCHIVE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip"
EDGE_MODEL_ARCHIVE_SHA256 = "d85a87f503f691807cd8bb97128bdf7a0660326cd9cd02657127fa978bab8b5e"
LATEST_GPU_MODEL_ARCHIVE_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/antelopev2.zip"
LATEST_GPU_MODEL_ARCHIVE_SHA256 = "8e182f14fc6e80b3bfa375b33eb6cff7ee05d8ef7633e738d1c89021dcf0c5c5"
YOLO_WEIGHTS_URL = "https://drive.google.com/uc?export=download&id=18oenL6tjFkdR1f5IgpYeQfDFqU4w3jEr"
YOLO_WEIGHTS_SHA256 = "794c94da54630f2ca66167fea25530c68133c61a2b14131b073c0d4064934e50"
YOLO_SOURCE_COMMIT = "152c688d551aefb973b7b589fb0691c93dab3564"
YOLO_SOURCE_ARCHIVE_URL = f"https://github.com/deepcam-cn/yolov5-face/archive/{YOLO_SOURCE_COMMIT}.zip"
YOLO_SOURCE_ARCHIVE_SHA256 = "ac8064e43357b3c3b8979cd55455bd595b95818c4de7cd3bb882994e2a4746d7"
YUNET_2026MAY_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2026may.onnx"
YUNET_2026MAY_SHA256 = "ebafce4e3c118d6554634be5c27ab333b4c047a9a8c3faf1d7cf93101c22f0f0"
YUNET_2023MAR_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
YUNET_2023MAR_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
YUNET_2023MAR_INT8BQ_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar_int8bq.onnx"
YUNET_2023MAR_INT8BQ_SHA256 = "49f000ec501fef24739071fc7e68267d32209045b6822c0c72dce1da25726f10"
SCRFD_10G_KPS_URL = "https://huggingface.co/kunkunlin1221/face-detection_scrfd-10g-gnkps/resolve/main/scrfd_10g_gnkps_fp32.onnx?download=true"
SCRFD_10G_KPS_SHA256 = "2112d066c1dce6cc648670e69cf90561b9287bb1945153f3b461a487131255b9"
SCRFD_34GF_KPS_URL = "https://huggingface.co/immich-app/scrfd_34g_gnkps/resolve/main/detection/model.onnx?download=true"
SCRFD_34GF_KPS_SHA256 = "aa19f0e7f4d120d4cf990086639ab74a0136adceaebd232e0dc4745e0cfd4257"
ADAFACE_R100_URL = "https://huggingface.co/Evn9172/cvlface_adaface_ir101_webface12m_onnx/resolve/main/adaface_ir101.onnx?download=true"
ADAFACE_R100_SHA256 = "6168ed3961838a871e706e8e5ebbf963de2821ecb3f64da9d5c3fd373cc80c65"
SFACE_2021DEC_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
SFACE_2021DEC_SHA256 = "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79"
SFACE_2021DEC_INT8BQ_URL = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_recognition_sface/face_recognition_sface_2021dec_int8bq.onnx"
SFACE_2021DEC_INT8BQ_SHA256 = "fb143eea07838aa532d1c95df5f69899974ea0140e1fba05e94204be13ed74ee"

_BUNDLE_LAYOUTS: dict[str, tuple[str, str, str]] = {
    "adaface_r100": ("human", "embedders", "AdaFace R100 embedder"),
    "arcface_r100_glint360k": ("human", "embedders", "ArcFace R100 Glint360K embedder"),
    "mobilefacenet_arcface": ("human", "embedders", "MobileFaceNet ArcFace embedder"),
    "scrfd_10g_kps": ("human", "detectors", "SCRFD 10G detector"),
    "scrfd_34gf_kps": ("human", "detectors", "SCRFD 34GF detector"),
    "scrfd_2.5g_kps": ("human", "detectors", "SCRFD 2.5G detector"),
    "scrfd_500m_kps": ("human", "detectors", "SCRFD 500M detector"),
    "arcface_r50": ("human", "embedders", "ArcFace R50 embedder"),
    "sface_2021dec": ("human", "embedders", "SFace 2021 Dec embedder"),
    "sface_2021dec_int8bq": ("human", "embedders", "SFace 2021 Dec INT8-BQ embedder"),
    "yolo5face_n": ("human", "detectors", "YOLO5Face Nano detector"),
    "yunet_2026may": ("human", "detectors", "YuNet 2026 May detector"),
    "yunet_2023mar": ("human", "detectors", "YuNet 2023 Mar detector"),
    "yunet_2023mar_int8bq": ("human", "detectors", "YuNet 2023 Mar INT8-BQ detector"),
}

_CATALOG_METADATA_PATHS: dict[str, str] = {
    "adaface_r100": "human/embedders/adaface_r100/metadata.json",
    "arcface_r100_glint360k": "human/embedders/arcface_r100_glint360k/metadata.json",
    "mobilefacenet_arcface": "human/embedders/mobilefacenet_arcface/metadata.json",
    "scrfd_10g_kps": "human/detectors/scrfd_10g_kps/metadata.json",
    "scrfd_34gf_kps": "human/detectors/scrfd_34gf_kps/metadata.json",
    "scrfd_2.5g_kps": "human/detectors/scrfd_2.5g_kps/metadata.json",
    "scrfd_500m_kps": "human/detectors/scrfd_500m_kps/metadata.json",
    "arcface_r50": "human/embedders/arcface_r50/metadata.json",
    "sface_2021dec": "human/embedders/sface_2021dec/metadata.json",
    "sface_2021dec_int8bq": "human/embedders/sface_2021dec_int8bq/metadata.json",
    "yolo5face_n": "human/detectors/yolo5face_n/metadata.json",
    "yunet_2026may": "human/detectors/yunet_2026may/metadata.json",
    "yunet_2023mar": "human/detectors/yunet_2023mar/metadata.json",
    "yunet_2023mar_int8bq": "human/detectors/yunet_2023mar_int8bq/metadata.json",
}

DOWNLOAD_IO_TIMEOUT_SECONDS = 15
DOWNLOAD_RETRY_ATTEMPTS = 4

# A managed bundle can be reconstructed without network access when its
# verified source archive remains in ``face_model_downloads``.  This mapping is
# deliberately limited to deterministic, pre-existing download recipes; it
# never attempts to fetch a missing model during startup recovery.
_RECOVERY_RECIPES: dict[str, tuple[tuple[str, str, str | None], ...]] = {
    "scrfd_2.5g_kps": ((RECOMMENDED_MODEL_ARCHIVE_SHA256, "buffalo_m.zip", "det_2.5g.onnx"),),
    "arcface_r50": ((RECOMMENDED_MODEL_ARCHIVE_SHA256, "buffalo_m.zip", "w600k_r50.onnx"),),
    "scrfd_500m_kps": ((EDGE_MODEL_ARCHIVE_SHA256, "buffalo_s.zip", "det_500m.onnx"),),
    "mobilefacenet_arcface": ((EDGE_MODEL_ARCHIVE_SHA256, "buffalo_s.zip", "w600k_mbf.onnx"),),
    "scrfd_10g_kps": (
        (LATEST_GPU_MODEL_ARCHIVE_SHA256, "antelopev2.zip", "scrfd_10g_bnkps.onnx"),
        (SCRFD_10G_KPS_SHA256, "scrfd_10g_gnkps_fp32.onnx", None),
    ),
    "arcface_r100_glint360k": ((LATEST_GPU_MODEL_ARCHIVE_SHA256, "antelopev2.zip", "glintr100.onnx"),),
    "scrfd_34gf_kps": ((SCRFD_34GF_KPS_SHA256, "scrfd_34g_gnkps.onnx", None),),
    "adaface_r100": ((ADAFACE_R100_SHA256, "adaface_ir101.onnx", None),),
    "sface_2021dec": ((SFACE_2021DEC_SHA256, "face_recognition_sface_2021dec.onnx", None),),
    "sface_2021dec_int8bq": ((SFACE_2021DEC_INT8BQ_SHA256, "face_recognition_sface_2021dec_int8bq.onnx", None),),
    "yunet_2026may": ((YUNET_2026MAY_SHA256, "face_detection_yunet_2026may.onnx", None),),
    "yunet_2023mar": ((YUNET_2023MAR_SHA256, "face_detection_yunet_2023mar.onnx", None),),
    "yunet_2023mar_int8bq": ((YUNET_2023MAR_INT8BQ_SHA256, "face_detection_yunet_2023mar_int8bq.onnx", None),),
}


@dataclass(frozen=True)
class FaceModelInventoryItem:
    bundle_id: str
    display_name: str
    kind: str
    installed: bool
    size_bytes: int
    bundle_dir: str
    source_url: str
    license: str
    status: str


@dataclass(frozen=True)
class FaceModelRecoveryResult:
    restored: tuple[str, ...]
    promoted_interrupted: tuple[str, ...]
    unavailable: tuple[str, ...]
    failures: tuple[str, ...]


def face_model_runtime_root_dir(settings: AppSettings | None = None) -> Path:
    app_settings = settings or get_settings()
    return Path(app_settings.cache_dir) / "face_model_assets"


class FaceModelInstaller:
    def __init__(self, settings: AppSettings | None = None) -> None:
        self.settings = settings or get_settings()

    def runtime_root(self) -> Path:
        root = face_model_runtime_root_dir(self.settings)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def download_cache_dir(self) -> Path:
        root = Path(self.settings.cache_dir) / "face_model_downloads"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def state_path(self) -> Path:
        """Return the durable record of managed face-model intent.

        The record intentionally lives beside, not inside, the installed
        bundle tree.  A damaged or removed ``face_model_assets`` directory can
        therefore still be reconstructed from retained verified downloads.
        """
        return Path(self.settings.cache_dir) / "face_model_state.json"

    def recover_managed_models(self) -> FaceModelRecoveryResult:
        """Repair interrupted or missing managed bundles without networking.

        A user explicitly deleting a model is remembered and never undone.
        Missing bundles with no retained verified source are left absent and
        reported to the caller so the UI can truthfully show ``Install``.
        """
        self.runtime_root()
        lease = SharedDownloadLease(
            Path(self.settings.cache_dir) / "model_download_locks",
            "face-model-recovery",
            wait_message="Waiting for another ClusterLens process to reconcile face models...",
        )
        lease.acquire()
        try:
            state = self._read_state()
            bundles = state["bundles"]
            restored: list[str] = []
            unavailable: list[str] = []
            failures: list[str] = []
            promoted = self._recover_interrupted_promotions(bundles, failures)

            for bundle_id in _BUNDLE_LAYOUTS:
                entry = bundles.get(bundle_id)
                if self._bundle_installed(bundle_id):
                    if not isinstance(entry, dict) or bool(entry.get("deleted")):
                        bundles[bundle_id] = {"deleted": False, "last_seen_at": int(time.time())}
                    continue
                if not isinstance(entry, dict) or bool(entry.get("deleted")):
                    continue
                try:
                    if self._restore_bundle_from_download_cache(bundle_id):
                        restored.append(bundle_id)
                        bundles[bundle_id] = {"deleted": False, "last_seen_at": int(time.time())}
                    else:
                        unavailable.append(bundle_id)
                except Exception as exc:
                    failures.append(f"{bundle_id}: {exc}")

            self._write_state(state)
            return FaceModelRecoveryResult(
                restored=tuple(restored),
                promoted_interrupted=tuple(promoted),
                unavailable=tuple(unavailable),
                failures=tuple(failures),
            )
        finally:
            lease.release()

    def clear_download_cache(self, progress=None, cancel_check=None) -> tuple[tuple[str, ...], int, tuple[str, ...]]:
        """Clear reusable face archives without racing another app process."""
        cache_root = self.download_cache_dir()
        lease = SharedDownloadLease(
            Path(self.settings.cache_dir) / "model_download_locks",
            "face-cache-maintenance",
            wait_message="Waiting for another ClusterLens process to finish its face-model download...",
        )
        lease.acquire(progress_callback=progress, cancel_check=cancel_check)
        removed: list[str] = []
        failures: list[str] = []
        freed_bytes = 0
        try:
            entries = tuple(sorted(cache_root.iterdir(), key=lambda path: path.name.casefold()))
            total = max(1, len(entries))
            for index, path in enumerate(entries):
                raise_if_cancelled(cancel_check)
                if progress is not None:
                    progress(int(index * 100 / total), f"Deleting cached face download {path.name}")
                try:
                    if path.is_symlink() or path.is_file():
                        try:
                            freed_bytes += max(0, int(path.stat().st_size))
                        except OSError:
                            pass
                        path.unlink(missing_ok=True)
                        removed.append(str(path))
                    else:
                        failures.append(f"Unexpected directory was preserved: {path}")
                except OSError as exc:
                    failures.append(f"{path}: {exc}")
            if progress is not None:
                progress(100, "Reusable face download cache cleared")
        finally:
            lease.release()
        return tuple(removed), int(freed_bytes), tuple(failures)

    def bundle_dir(self, bundle_id: str) -> Path:
        mode, plural_kind, _label = self._bundle_layout(bundle_id)
        return self.runtime_root() / mode / plural_kind / bundle_id

    def inventory(self) -> tuple[FaceModelInventoryItem, ...]:
        # Settings may be opened without a complete application startup (for
        # example in a test or a packaged maintenance entry point). Reconcile
        # here as well so a recoverable bundle is never presented as missing.
        self.recover_managed_models()
        items: list[FaceModelInventoryItem] = []
        for bundle_id, (_mode, plural_kind, label) in _BUNDLE_LAYOUTS.items():
            bundle_dir = self.bundle_dir(bundle_id)
            filename = "detector.onnx" if plural_kind == "detectors" else "embedder.onnx"
            payload_path = bundle_dir / filename
            metadata = self._catalog_metadata(bundle_id)
            payload_installed, verified = self._installed_payload_state(payload_path)
            installed = payload_installed and (bundle_dir / "metadata.json").is_file()
            size_bytes = self._path_size(payload_path)
            status = "ready (verified)" if installed and verified else ("ready" if installed else "install required")
            items.append(
                FaceModelInventoryItem(
                    bundle_id=bundle_id,
                    display_name=label,
                    kind="detector" if plural_kind == "detectors" else "embedder",
                    installed=installed,
                    size_bytes=size_bytes,
                    bundle_dir=str(bundle_dir),
                    source_url=str(metadata.get("source_url") or ""),
                    license=str(metadata.get("license") or ""),
                    status=status,
                )
            )
        return tuple(items)

    def install_recommended(self, progress=None, cancel_check=None) -> tuple[str, ...]:
        target_ids = ("scrfd_2.5g_kps", "arcface_r50")
        if all(self._bundle_installed(bundle_id) for bundle_id in target_ids):
            progress and progress(100, "Recommended face models already installed; reused managed cache")
            return target_ids
        archive_path, temp_root = self._download_to_temp(
            RECOMMENDED_MODEL_ARCHIVE_URL,
            RECOMMENDED_MODEL_ARCHIVE_SHA256,
            "buffalo_m.zip",
            progress=progress,
            cancel_check=cancel_check,
            progress_prefix="Downloading InsightFace buffalo_m",
        )
        installed: list[str] = []
        try:
            raise_if_cancelled(cancel_check)
            progress and progress(65, "Installing SCRFD 2.5G detector")
            if not self._bundle_installed("scrfd_2.5g_kps"):
                self._install_zip_member(
                    archive_path,
                    member_name="det_2.5g.onnx",
                    bundle_id="scrfd_2.5g_kps",
                    output_name="detector.onnx",
                )
            installed.append("scrfd_2.5g_kps")
            raise_if_cancelled(cancel_check)
            progress and progress(82, "Installing ArcFace R50 embedder")
            if not self._bundle_installed("arcface_r50"):
                self._install_zip_member(
                    archive_path,
                    member_name="w600k_r50.onnx",
                    bundle_id="arcface_r50",
                    output_name="embedder.onnx",
                )
            installed.append("arcface_r50")
            progress and progress(100, "Installed SCRFD 2.5G and ArcFace R50")
            return tuple(installed)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def install_edge(self, progress=None, cancel_check=None) -> tuple[str, ...]:
        target_ids = ("scrfd_500m_kps", "mobilefacenet_arcface")
        if all(self._bundle_installed(bundle_id) for bundle_id in target_ids):
            progress and progress(100, "Edge face models already installed; reused managed cache")
            return target_ids
        archive_path, temp_root = self._download_to_temp(
            EDGE_MODEL_ARCHIVE_URL,
            EDGE_MODEL_ARCHIVE_SHA256,
            "buffalo_s.zip",
            progress=progress,
            cancel_check=cancel_check,
            progress_prefix="Downloading InsightFace buffalo_s",
        )
        installed: list[str] = []
        try:
            raise_if_cancelled(cancel_check)
            progress and progress(65, "Installing SCRFD 500M detector")
            if not self._bundle_installed("scrfd_500m_kps"):
                self._install_zip_member(
                    archive_path,
                    member_name="det_500m.onnx",
                    bundle_id="scrfd_500m_kps",
                    output_name="detector.onnx",
                )
            installed.append("scrfd_500m_kps")
            raise_if_cancelled(cancel_check)
            progress and progress(82, "Installing MobileFaceNet ArcFace embedder")
            if not self._bundle_installed("mobilefacenet_arcface"):
                self._install_zip_member(
                    archive_path,
                    member_name="w600k_mbf.onnx",
                    bundle_id="mobilefacenet_arcface",
                    output_name="embedder.onnx",
                )
            installed.append("mobilefacenet_arcface")
            progress and progress(100, "Installed SCRFD 500M and MobileFaceNet ArcFace")
            return tuple(installed)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def install_yolo(self, progress=None, cancel_check=None) -> tuple[str, ...]:
        if self._bundle_installed("yolo5face_n"):
            progress and progress(100, "YOLO5Face Nano already installed; reused managed cache")
            return ("yolo5face_n",)
        weights_path, weights_temp_root = self._download_to_temp(
            YOLO_WEIGHTS_URL,
            YOLO_WEIGHTS_SHA256,
            "yolov5n-face.pt",
            progress=progress,
            cancel_check=cancel_check,
            progress_prefix="Downloading YOLO5Face weights",
        )
        source_path, source_temp_root = self._download_to_temp(
            YOLO_SOURCE_ARCHIVE_URL,
            YOLO_SOURCE_ARCHIVE_SHA256,
            f"yolov5-face-{YOLO_SOURCE_COMMIT}.zip",
            progress=progress,
            cancel_check=cancel_check,
            progress_prefix="Downloading YOLO5Face source",
        )
        tmp_dir = Path(self.settings.cache_dir) / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        install_root = Path(tempfile.mkdtemp(prefix="face-model-yolo-install-", dir=str(tmp_dir)))
        try:
            raise_if_cancelled(cancel_check)
            progress and progress(70, "Exporting YOLO5Face ONNX")
            source_dir = self._extract_source_archive(source_path, install_root)
            output_path = install_root / "yolov5n-face.onnx"
            self._export_yolo_face_onnx(source_dir, weights_path, output_path, cancel_check=cancel_check)
            raise_if_cancelled(cancel_check)
            progress and progress(88, "Installing YOLO5Face detector")
            self._install_payload_file(output_path, bundle_id="yolo5face_n", output_name="detector.onnx")
            progress and progress(100, "Installed YOLO5Face Nano detector")
            return ("yolo5face_n",)
        finally:
            shutil.rmtree(weights_temp_root, ignore_errors=True)
            shutil.rmtree(source_temp_root, ignore_errors=True)
            shutil.rmtree(install_root, ignore_errors=True)

    def install_yunet(self, progress=None, cancel_check=None) -> tuple[str, ...]:
        downloads = (
            ("yunet_2026may", YUNET_2026MAY_URL, YUNET_2026MAY_SHA256, "face_detection_yunet_2026may.onnx", "YuNet 2026 May detector"),
            ("yunet_2023mar", YUNET_2023MAR_URL, YUNET_2023MAR_SHA256, "face_detection_yunet_2023mar.onnx", "YuNet 2023 Mar detector"),
            ("yunet_2023mar_int8bq", YUNET_2023MAR_INT8BQ_URL, YUNET_2023MAR_INT8BQ_SHA256, "face_detection_yunet_2023mar_int8bq.onnx", "YuNet 2023 Mar INT8-BQ detector"),
        )
        installed: list[str] = []
        for index, (bundle_id, url, sha256, filename, label) in enumerate(downloads, start=1):
            raise_if_cancelled(cancel_check)
            if self._bundle_installed(bundle_id):
                installed.append(bundle_id)
                progress and progress(
                    max(1, int((index / max(1, len(downloads))) * 100)),
                    f"Using installed {label}",
                )
                continue
            progress and progress(max(1, int(((index - 1) / max(1, len(downloads))) * 100)), f"Downloading {label}")
            payload_path, temp_root = self._download_to_temp(
                url,
                sha256,
                filename,
                progress=progress,
                cancel_check=cancel_check,
                progress_prefix=f"Downloading {label}",
            )
            try:
                raise_if_cancelled(cancel_check)
                progress and progress(max(1, int(((index - 0.35) / max(1, len(downloads))) * 100)), f"Installing {label}")
                self._install_payload_file(payload_path, bundle_id=bundle_id, output_name="detector.onnx")
                installed.append(bundle_id)
            finally:
                shutil.rmtree(temp_root, ignore_errors=True)
        progress and progress(100, "Installed YuNet detector family")
        return tuple(installed)

    def install_accuracy(self, progress=None, cancel_check=None) -> tuple[str, ...]:
        return self._install_direct_face_bundles(
            (
                ("scrfd_10g_kps", SCRFD_10G_KPS_URL, SCRFD_10G_KPS_SHA256, "scrfd_10g_gnkps_fp32.onnx", "detector.onnx", "SCRFD 10G detector"),
                ("adaface_r100", ADAFACE_R100_URL, ADAFACE_R100_SHA256, "adaface_ir101.onnx", "embedder.onnx", "AdaFace R100 embedder"),
            ),
            progress=progress,
            cancel_check=cancel_check,
            completion_label="Installed SCRFD 10G and AdaFace R100",
        )

    def install_max_accuracy(self, progress=None, cancel_check=None) -> tuple[str, ...]:
        return self._install_direct_face_bundles(
            (
                ("scrfd_34gf_kps", SCRFD_34GF_KPS_URL, SCRFD_34GF_KPS_SHA256, "scrfd_34g_gnkps.onnx", "detector.onnx", "SCRFD 34GF detector"),
                ("adaface_r100", ADAFACE_R100_URL, ADAFACE_R100_SHA256, "adaface_ir101.onnx", "embedder.onnx", "AdaFace R100 embedder"),
            ),
            progress=progress,
            cancel_check=cancel_check,
            completion_label="Installed SCRFD 34GF and AdaFace R100",
        )

    def install_latest_gpu(self, progress=None, cancel_check=None) -> tuple[str, ...]:
        target_ids = ("scrfd_10g_kps", "arcface_r100_glint360k")
        if all(self._bundle_installed(bundle_id) for bundle_id in target_ids):
            progress and progress(100, "Latest GPU face models already installed; reused managed cache")
            return target_ids
        archive_path, temp_root = self._download_to_temp(
            LATEST_GPU_MODEL_ARCHIVE_URL,
            LATEST_GPU_MODEL_ARCHIVE_SHA256,
            "antelopev2.zip",
            progress=progress,
            cancel_check=cancel_check,
            progress_prefix="Downloading InsightFace antelopev2",
        )
        installed: list[str] = []
        try:
            raise_if_cancelled(cancel_check)
            progress and progress(65, "Installing SCRFD 10G detector")
            if not self._bundle_installed("scrfd_10g_kps"):
                self._install_zip_member(
                    archive_path,
                    member_name="scrfd_10g_bnkps.onnx",
                    bundle_id="scrfd_10g_kps",
                    output_name="detector.onnx",
                )
            installed.append("scrfd_10g_kps")
            raise_if_cancelled(cancel_check)
            progress and progress(82, "Installing ArcFace R100 Glint360K embedder")
            if not self._bundle_installed("arcface_r100_glint360k"):
                self._install_zip_member(
                    archive_path,
                    member_name="glintr100.onnx",
                    bundle_id="arcface_r100_glint360k",
                    output_name="embedder.onnx",
                )
            installed.append("arcface_r100_glint360k")
            progress and progress(100, "Installed SCRFD 10G and ArcFace R100 Glint360K")
            return tuple(installed)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def install_sface(self, progress=None, cancel_check=None) -> tuple[str, ...]:
        return self._install_direct_face_bundles(
            (
                (
                    "sface_2021dec",
                    SFACE_2021DEC_URL,
                    SFACE_2021DEC_SHA256,
                    "face_recognition_sface_2021dec.onnx",
                    "embedder.onnx",
                    "SFace 2021 Dec embedder",
                ),
                (
                    "sface_2021dec_int8bq",
                    SFACE_2021DEC_INT8BQ_URL,
                    SFACE_2021DEC_INT8BQ_SHA256,
                    "face_recognition_sface_2021dec_int8bq.onnx",
                    "embedder.onnx",
                    "SFace 2021 Dec INT8-BQ embedder",
                ),
            ),
            progress=progress,
            cancel_check=cancel_check,
            completion_label="Installed SFace 2021 Dec embedders",
        )

    def install_opencv_cpu(self, progress=None, cancel_check=None) -> tuple[str, ...]:
        def _mapped(start: int, span: int):
            if progress is None:
                return None
            return lambda value, status: progress(
                start + int(max(0, min(100, int(value))) * span / 100),
                status,
            )

        detectors = self.install_yunet(_mapped(0, 50), cancel_check)
        raise_if_cancelled(cancel_check)
        embedders = self.install_sface(_mapped(50, 50), cancel_check)
        progress and progress(100, "Installed OpenCV CPU face detector and embedder family")
        return tuple(dict.fromkeys((*detectors, *embedders)))

    def delete_installed_model(self, bundle_id: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        bundle_dir = self.bundle_dir(bundle_id)
        if not bundle_dir.exists():
            self._set_bundle_deleted(bundle_id)
            return (), ()
        try:
            shutil.rmtree(bundle_dir)
        except OSError as exc:
            return (), (f"{bundle_dir}: {exc}",)
        self._set_bundle_deleted(bundle_id)
        return (str(bundle_dir),), ()

    def bundled_catalog_root(self) -> Path:
        # PyInstaller extracts/collects data files under sys._MEIPASS. In an
        # onedir build this is the executable's `_internal` directory, while
        # walking up from this module reaches the directory beside `_internal`.
        meipass = getattr(sys, "_MEIPASS", "")
        if meipass:
            return Path(meipass) / "face_model_assets"
        return Path(__file__).resolve().parents[3] / "face_model_assets"

    def _catalog_metadata(self, bundle_id: str) -> dict[str, object]:
        relative_path = _CATALOG_METADATA_PATHS[bundle_id]
        metadata_path = self.bundled_catalog_root() / relative_path
        if not metadata_path.exists():
            return {}
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return dict(payload) if isinstance(payload, dict) else {}

    def _bundle_layout(self, bundle_id: str) -> tuple[str, str, str]:
        if bundle_id not in _BUNDLE_LAYOUTS:
            raise ValueError(f"Unknown face model bundle id: {bundle_id}")
        return _BUNDLE_LAYOUTS[bundle_id]

    def _download_to_temp(
        self,
        url: str,
        sha256: str,
        filename: str,
        *,
        progress=None,
        cancel_check=None,
        progress_prefix: str,
    ) -> tuple[Path, Path]:
        expected_sha = str(sha256 or "").strip().lower()
        safe_filename = Path(filename).name
        cache_key = expected_sha or hashlib.sha256(str(url).encode("utf-8")).hexdigest()
        download_cache = self.download_cache_dir()
        cache_path = download_cache / f"{cache_key[:16]}-{safe_filename}"
        partial_path = cache_path.with_name(f"{cache_path.name}.partial")
        verification_path = cache_path.with_name(f"{cache_path.name}.verified.json")
        lock_root = Path(self.settings.cache_dir) / "model_download_locks"
        maintenance_lease = SharedDownloadLease(
            lock_root,
            "face-cache-maintenance",
            wait_message="Waiting for face-model cache maintenance to finish...",
        )
        lease = SharedDownloadLease(
            lock_root,
            f"face-{cache_key}",
            wait_message=f"Waiting for another process to finish {safe_filename}...",
        )
        maintenance_lease.acquire(progress_callback=progress, cancel_check=cancel_check)
        try:
            lease.acquire(progress_callback=progress, cancel_check=cancel_check)
            try:
                raise_if_cancelled(cancel_check)
                if not self._verified_download_cache_entry(cache_path, verification_path, expected_sha):
                    cache_path.unlink(missing_ok=True)
                    verification_path.unlink(missing_ok=True)
                    self._download_resumable(
                        url,
                        partial_path,
                        progress=progress,
                        cancel_check=cancel_check,
                        progress_prefix=progress_prefix,
                    )
                    raise_if_cancelled(cancel_check)
                    actual_sha = self._sha256_path(partial_path, cancel_check=cancel_check)
                    if expected_sha and actual_sha != expected_sha:
                        partial_path.unlink(missing_ok=True)
                        raise RuntimeError(
                            f"Checksum mismatch for {safe_filename}: expected {expected_sha}, got {actual_sha}."
                        )
                    atomic_write_with(cache_path, lambda temporary: shutil.copyfile(partial_path, temporary))
                    partial_path.unlink(missing_ok=True)
                    stat = cache_path.stat()
                    atomic_write_text(
                        verification_path,
                        json.dumps(
                            {
                                "sha256": actual_sha,
                                "size_bytes": int(stat.st_size),
                                "mtime_ns": int(stat.st_mtime_ns),
                                "source_url": str(url),
                            },
                            indent=2,
                            sort_keys=True,
                        ),
                    )
                elif progress is not None:
                    progress(60, f"Using cached {safe_filename}")

                raise_if_cancelled(cancel_check)
                tmp_dir = Path(self.settings.cache_dir) / "tmp"
                tmp_dir.mkdir(parents=True, exist_ok=True)
                temp_root = Path(tempfile.mkdtemp(prefix="face-model-download-", dir=str(tmp_dir)))
                target_path = temp_root / safe_filename
                try:
                    try:
                        os.link(cache_path, target_path)
                    except OSError:
                        shutil.copyfile(cache_path, target_path)
                except Exception:
                    shutil.rmtree(temp_root, ignore_errors=True)
                    raise
                return target_path, temp_root
            finally:
                lease.release()
        finally:
            maintenance_lease.release()

    def _download_resumable(
        self,
        url: str,
        partial_path: Path,
        *,
        progress=None,
        cancel_check=None,
        progress_prefix: str,
    ) -> None:
        last_error: Exception | None = None
        for attempt in range(1, DOWNLOAD_RETRY_ATTEMPTS + 1):
            raise_if_cancelled(cancel_check)
            resume_bytes = self._path_size(partial_path)
            headers = {"User-Agent": "ClusterLens/1.0"}
            if resume_bytes > 0:
                headers["Range"] = f"bytes={resume_bytes}-"
            request = urllib.request.Request(url, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=DOWNLOAD_IO_TIMEOUT_SECONDS) as response:
                    status = int(
                        getattr(response, "status", 0)
                        or getattr(response, "getcode", lambda: 200)()
                        or 200
                    )
                    append = resume_bytes > 0 and status == 206
                    if not append:
                        resume_bytes = 0
                    content_bytes = int(response.headers.get("Content-Length") or 0)
                    total_bytes = resume_bytes + content_bytes if content_bytes > 0 else 0
                    received = resume_bytes
                    mode = "ab" if append else "wb"
                    with partial_path.open(mode) as handle:
                        while True:
                            raise_if_cancelled(cancel_check)
                            chunk = response.read(256 * 1024)
                            if not chunk:
                                break
                            handle.write(chunk)
                            received += len(chunk)
                            if progress is not None:
                                if total_bytes > 0:
                                    percent = min(60, max(1, int((received / total_bytes) * 60.0)))
                                    progress(percent, f"{progress_prefix} ({received // (1024 * 1024)} MB)")
                                else:
                                    progress(10, f"{progress_prefix} ({received // (1024 * 1024)} MB)")
                        handle.flush()
                        os.fsync(handle.fileno())
                    if content_bytes > 0 and received < total_bytes:
                        raise ConnectionError(
                            f"incomplete response: received {received - resume_bytes} of {content_bytes} bytes"
                        )
                return
            except urllib.error.HTTPError as exc:
                # A completed partial can receive 416 when the previous run
                # ended after its last byte but before checksum publication.
                if int(getattr(exc, "code", 0) or 0) == 416 and partial_path.is_file():
                    return
                last_error = exc
            except (TimeoutError, urllib.error.URLError, ConnectionError, http.client.HTTPException) as exc:
                last_error = exc

            if attempt >= DOWNLOAD_RETRY_ATTEMPTS:
                break
            if progress is not None:
                progress(
                    -1,
                    f"{progress_prefix}: connection interrupted; resuming "
                    f"(attempt {attempt + 1}/{DOWNLOAD_RETRY_ATTEMPTS})",
                )
            # Check cancellation during retry backoff instead of blocking the
            # worker thread in one long sleep.
            for _step in range(4):
                raise_if_cancelled(cancel_check)
                time.sleep(0.25)

        raise RuntimeError(
            f"{progress_prefix} failed after {DOWNLOAD_RETRY_ATTEMPTS} attempts; "
            f"the partial download was kept for resume: {last_error}"
        )

    @staticmethod
    def _sha256_path(path: Path, *, cancel_check=None) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            while True:
                raise_if_cancelled(cancel_check)
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest().lower()

    def _verified_download_cache_entry(self, path: Path, metadata_path: Path, expected_sha: str) -> bool:
        try:
            stat = path.stat()
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            if int(stat.st_size) <= 0:
                return False
            if int(payload.get("size_bytes") or -1) != int(stat.st_size):
                return False
            if int(payload.get("mtime_ns") or -1) != int(stat.st_mtime_ns):
                return False
            saved_sha = str(payload.get("sha256") or "").strip().lower()
            return bool(saved_sha and (not expected_sha or saved_sha == expected_sha))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def _install_zip_member(self, archive_path: Path, *, member_name: str, bundle_id: str, output_name: str) -> None:
        with zipfile.ZipFile(archive_path) as archive:
            member = self._find_zip_member(archive, member_name)
            if member is None:
                raise RuntimeError(f"{archive_path.name} does not contain {member_name}.")
            self._install_bundle_atomically(
                bundle_id,
                output_name,
                lambda output_path: self._copy_zip_member(archive, member, output_path),
            )

    def _install_payload_file(self, payload_path: Path, *, bundle_id: str, output_name: str) -> None:
        self._install_bundle_atomically(
            bundle_id,
            output_name,
            lambda output_path: shutil.copyfile(payload_path, output_path),
        )

    def _copy_catalog_metadata(self, bundle_id: str, bundle_dir: Path) -> None:
        relative_path = _CATALOG_METADATA_PATHS[bundle_id]
        source_path = self.bundled_catalog_root() / relative_path
        if not source_path.exists():
            raise RuntimeError(f"Bundled metadata is missing for {bundle_id}: {source_path}")
        atomic_write_text(bundle_dir / "metadata.json", source_path.read_text(encoding="utf-8"))

    def _install_direct_face_bundles(
        self,
        bundles: tuple[tuple[str, str, str, str, str, str], ...],
        *,
        progress=None,
        cancel_check=None,
        completion_label: str,
    ) -> tuple[str, ...]:
        installed: list[str] = []
        total = max(1, len(bundles))
        for index, (bundle_id, url, sha256, filename, output_name, label) in enumerate(bundles, start=1):
            raise_if_cancelled(cancel_check)
            if self._bundle_installed(bundle_id):
                installed.append(bundle_id)
                progress and progress(
                    max(1, int((index / total) * 100)),
                    f"Using installed {label}",
                )
                continue
            start_progress = max(1, int(((index - 1) / total) * 100))
            progress and progress(start_progress, f"Downloading {label}")
            payload_path, temp_root = self._download_to_temp(
                url,
                sha256,
                filename,
                progress=progress,
                cancel_check=cancel_check,
                progress_prefix=f"Downloading {label}",
            )
            try:
                raise_if_cancelled(cancel_check)
                mid_progress = max(1, int(((index - 0.35) / total) * 100))
                progress and progress(mid_progress, f"Installing {label}")
                self._install_payload_file(payload_path, bundle_id=bundle_id, output_name=output_name)
                installed.append(bundle_id)
            finally:
                shutil.rmtree(temp_root, ignore_errors=True)
        progress and progress(100, completion_label)
        return tuple(installed)

    def _bundle_installed(self, bundle_id: str) -> bool:
        _mode, plural_kind, _label = self._bundle_layout(bundle_id)
        filename = "detector.onnx" if plural_kind == "detectors" else "embedder.onnx"
        bundle_dir = self.bundle_dir(bundle_id)
        if not (bundle_dir / "metadata.json").is_file():
            return False
        installed, _verified = self._installed_payload_state(bundle_dir / filename)
        return installed

    @staticmethod
    def _copy_zip_member(archive: zipfile.ZipFile, member: str, output_path: Path) -> None:
        with archive.open(member, "r") as source, output_path.open("wb") as target:
            shutil.copyfileobj(source, target)

    def _install_bundle_atomically(self, bundle_id: str, output_name: str, write_payload) -> None:
        """Publish a complete face bundle as one directory rename.

        Readers see either the previous complete bundle or the new verified
        bundle.  The recovery pass can promote a completed staging directory
        if the process stops between the two renames.
        """
        target_dir = self.bundle_dir(bundle_id)
        parent = target_dir.parent
        parent.mkdir(parents=True, exist_ok=True)
        staging_dir = parent / f".{bundle_id}.{uuid4().hex}.installing"
        staging_dir.mkdir()
        try:
            self._copy_catalog_metadata(bundle_id, staging_dir)
            payload_path = staging_dir / output_name
            atomic_write_with(payload_path, write_payload)
            self._write_install_record(payload_path, bundle_id=bundle_id)
            if not self._bundle_dir_is_complete(staging_dir, bundle_id, verify_sha=True):
                raise RuntimeError(f"Staged {bundle_id} bundle did not pass its integrity check.")
            self._promote_bundle_directory(staging_dir, target_dir)
            self._set_bundle_present(bundle_id)
        except Exception:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise

    def _promote_bundle_directory(self, staging_dir: Path, target_dir: Path) -> None:
        previous_dir: Path | None = None
        try:
            if target_dir.exists():
                previous_dir = target_dir.parent / f".{target_dir.name}.{uuid4().hex}.previous"
                target_dir.replace(previous_dir)
                self._promotion_checkpoint("previous_staged", staging_dir, target_dir, previous_dir)
            staging_dir.replace(target_dir)
            self._promotion_checkpoint("target_published", staging_dir, target_dir, previous_dir)
        except Exception:
            if previous_dir is not None and previous_dir.exists() and not target_dir.exists():
                previous_dir.replace(target_dir)
            raise
        if previous_dir is not None:
            shutil.rmtree(previous_dir, ignore_errors=True)
            self._promotion_checkpoint("previous_discarded", staging_dir, target_dir, previous_dir)

    def _promotion_checkpoint(
        self,
        _name: str,
        _staging_dir: Path,
        _target_dir: Path,
        _previous_dir: Path | None,
    ) -> None:
        """Deterministic fault-injection seam at durable promotion boundaries."""

        return

    @staticmethod
    def _discard_previous_bundle_dirs(target_dir: Path) -> None:
        for previous in target_dir.parent.glob(f".{target_dir.name}.*.previous"):
            if previous.is_symlink() or not previous.is_dir():
                continue
            shutil.rmtree(previous)

    def _write_install_record(self, payload_path: Path, *, bundle_id: str = "") -> None:
        stat = payload_path.stat()
        record_path = payload_path.with_name("install.json")
        atomic_write_text(
            record_path,
            json.dumps(
                {
                    "bundle_id": str(bundle_id or ""),
                    "payload": payload_path.name,
                    "sha256": self._sha256_path(payload_path),
                    "size_bytes": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                    "installed_at": int(time.time()),
                },
                indent=2,
                sort_keys=True,
            ),
        )

    @staticmethod
    def _installed_payload_state(payload_path: Path) -> tuple[bool, bool]:
        try:
            stat = payload_path.stat()
            if not payload_path.is_file() or int(stat.st_size) <= 0:
                return False, False
        except OSError:
            return False, False
        record_path = payload_path.with_name("install.json")
        if not record_path.exists():
            # Preserve compatibility with manually installed and older managed
            # bundles. New installs always receive a verified record.
            return True, False
        try:
            payload = json.loads(record_path.read_text(encoding="utf-8"))
            verified = bool(
                str(payload.get("payload") or "") == payload_path.name
                and str(payload.get("sha256") or "")
                and int(payload.get("size_bytes") or -1) == int(stat.st_size)
                and int(payload.get("mtime_ns") or -1) == int(stat.st_mtime_ns)
            )
            return verified, verified
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False, False

    def _bundle_dir_is_complete(self, bundle_dir: Path, bundle_id: str, *, verify_sha: bool) -> bool:
        _mode, plural_kind, _label = self._bundle_layout(bundle_id)
        output_name = "detector.onnx" if plural_kind == "detectors" else "embedder.onnx"
        payload_path = bundle_dir / output_name
        if not (bundle_dir / "metadata.json").is_file():
            return False
        installed, verified = self._installed_payload_state(payload_path)
        if not installed:
            return False
        if not verify_sha:
            return True
        if not verified:
            return False
        try:
            record = json.loads((bundle_dir / "install.json").read_text(encoding="utf-8"))
            expected_sha = str(record.get("sha256") or "").strip().lower()
            return bool(expected_sha and self._sha256_path(payload_path) == expected_sha)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False

    def _recover_interrupted_promotions(self, bundles: dict[str, object], failures: list[str]) -> list[str]:
        promoted: list[str] = []
        for bundle_id in _BUNDLE_LAYOUTS:
            entry = bundles.get(bundle_id)
            if isinstance(entry, dict) and bool(entry.get("deleted")):
                continue
            target_dir = self.bundle_dir(bundle_id)
            if self._bundle_dir_is_complete(target_dir, bundle_id, verify_sha=False):
                if self._bundle_dir_is_complete(target_dir, bundle_id, verify_sha=True):
                    try:
                        self._discard_previous_bundle_dirs(target_dir)
                    except OSError as exc:
                        failures.append(f"{bundle_id}: unable to clean an interrupted previous bundle: {exc}")
                continue
            parent = target_dir.parent
            if not parent.is_dir():
                continue
            candidates = [
                *sorted(parent.glob(f".{bundle_id}.*.installing"), key=lambda path: path.name, reverse=True),
                *sorted(parent.glob(f".{bundle_id}.*.previous"), key=lambda path: path.name, reverse=True),
            ]
            source_dir = next(
                (path for path in candidates if self._bundle_dir_is_complete(path, bundle_id, verify_sha=True)),
                None,
            )
            if source_dir is None:
                continue
            try:
                if target_dir.exists():
                    corrupt_dir = parent / f".{bundle_id}.{uuid4().hex}.corrupt"
                    target_dir.replace(corrupt_dir)
                source_dir.replace(target_dir)
                if self._bundle_dir_is_complete(target_dir, bundle_id, verify_sha=True):
                    self._discard_previous_bundle_dirs(target_dir)
                promoted.append(bundle_id)
            except OSError as exc:
                failures.append(f"{bundle_id}: unable to promote an interrupted install: {exc}")
        return promoted

    def _restore_bundle_from_download_cache(self, bundle_id: str) -> bool:
        _mode, plural_kind, _label = self._bundle_layout(bundle_id)
        output_name = "detector.onnx" if plural_kind == "detectors" else "embedder.onnx"
        for expected_sha, filename, archive_member in _RECOVERY_RECIPES.get(bundle_id, ()):
            cached_path = self._verified_recovery_download(expected_sha, filename)
            if cached_path is None:
                continue
            if archive_member is None:
                self._install_payload_file(cached_path, bundle_id=bundle_id, output_name=output_name)
            else:
                self._install_zip_member(
                    cached_path,
                    member_name=archive_member,
                    bundle_id=bundle_id,
                    output_name=output_name,
                )
            return True
        return False

    def _verified_recovery_download(self, expected_sha: str, filename: str) -> Path | None:
        cache_path = self.download_cache_dir() / f"{str(expected_sha)[:16]}-{Path(filename).name}"
        metadata_path = cache_path.with_name(f"{cache_path.name}.verified.json")
        if self._verified_download_cache_entry(cache_path, metadata_path, expected_sha):
            return cache_path
        if not cache_path.is_file():
            return None
        try:
            actual_sha = self._sha256_path(cache_path)
            if actual_sha != str(expected_sha).strip().lower():
                return None
            stat = cache_path.stat()
            atomic_write_text(
                metadata_path,
                json.dumps(
                    {
                        "sha256": actual_sha,
                        "size_bytes": int(stat.st_size),
                        "mtime_ns": int(stat.st_mtime_ns),
                        "source_url": "",
                    },
                    indent=2,
                    sort_keys=True,
                ),
            )
            return cache_path
        except OSError:
            return None

    def _read_state(self) -> dict[str, object]:
        try:
            payload = json.loads(self.state_path().read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            payload = {}
        bundles = payload.get("bundles") if isinstance(payload, dict) else None
        return {
            "version": 1,
            "bundles": dict(bundles) if isinstance(bundles, dict) else {},
        }

    def _write_state(self, state: dict[str, object]) -> None:
        atomic_write_text(self.state_path(), json.dumps(state, indent=2, sort_keys=True))

    def _set_bundle_present(self, bundle_id: str) -> None:
        state = self._read_state()
        bundles = state["bundles"]
        assert isinstance(bundles, dict)
        bundles[bundle_id] = {"deleted": False, "last_seen_at": int(time.time())}
        self._write_state(state)

    def _set_bundle_deleted(self, bundle_id: str) -> None:
        self._bundle_layout(bundle_id)
        state = self._read_state()
        bundles = state["bundles"]
        assert isinstance(bundles, dict)
        bundles[bundle_id] = {"deleted": True, "last_seen_at": int(time.time())}
        self._write_state(state)

    @staticmethod
    def _find_zip_member(archive: zipfile.ZipFile, member_name: str) -> str | None:
        target = str(member_name).strip().lower()
        for member in archive.namelist():
            if Path(member).name.strip().lower() == target:
                return member
        return None

    @staticmethod
    def _extract_source_archive(source_archive: Path, install_root: Path) -> Path:
        with zipfile.ZipFile(source_archive) as archive:
            resolved_root = install_root.resolve()
            for member in archive.infolist():
                target = (install_root / member.filename).resolve()
                try:
                    target.relative_to(resolved_root)
                except ValueError as exc:
                    raise RuntimeError(f"Unsafe path in face model source archive: {member.filename}") from exc
                unix_mode = int(member.external_attr >> 16)
                if (unix_mode & 0o170000) == 0o120000:
                    raise RuntimeError(f"Symlinks are not allowed in face model source archives: {member.filename}")
                archive.extract(member, install_root)
        for child in install_root.iterdir():
            if child.is_dir() and (child / "export.py").exists():
                return child
        raise RuntimeError("Failed to locate the extracted YOLO5Face source tree.")

    @staticmethod
    def _export_yolo_face_onnx(source_dir: Path, weights_path: Path, output_path: Path, *, cancel_check=None) -> None:
        raise_if_cancelled(cancel_check)
        if importlib.util.find_spec("onnx") is None:
            raise RuntimeError("YOLO export requires the 'onnx' Python package to be installed.")
        stubbed_modules: dict[str, object | None] = {}
        if importlib.util.find_spec("matplotlib") is None:
            matplotlib_module = types.ModuleType("matplotlib")
            pyplot_module = types.ModuleType("matplotlib.pyplot")
            matplotlib_module.__spec__ = importlib.machinery.ModuleSpec("matplotlib", loader=None)
            pyplot_module.__spec__ = importlib.machinery.ModuleSpec("matplotlib.pyplot", loader=None)
            matplotlib_module.rc = lambda *_args, **_kwargs: None
            matplotlib_module.use = lambda *_args, **_kwargs: None
            pyplot_module.rcParams = {}
            matplotlib_module.pyplot = pyplot_module
            stubbed_modules["matplotlib"] = sys.modules.get("matplotlib")
            stubbed_modules["matplotlib.pyplot"] = sys.modules.get("matplotlib.pyplot")
            sys.modules["matplotlib"] = matplotlib_module
            sys.modules["matplotlib.pyplot"] = pyplot_module
        for module_name in ("pandas", "seaborn", "thop"):
            if importlib.util.find_spec(module_name) is not None:
                continue
            stubbed_modules[module_name] = sys.modules.get(module_name)
            stub = types.ModuleType(module_name)
            stub.__spec__ = importlib.machinery.ModuleSpec(module_name, loader=None)
            if module_name == "thop":
                stub.profile = lambda *_args, **_kwargs: (0.0, 0.0)
                stub.clever_format = lambda values, _format=None: values
            sys.modules[module_name] = stub
        sys.path.insert(0, str(source_dir))
        try:
            import torch
            import torch.nn as nn

            import models  # type: ignore
            from models.experimental import attempt_load  # type: ignore
            from utils.activations import Hardswish, SiLU  # type: ignore
            from utils.general import check_img_size  # type: ignore
        finally:
            if sys.path and sys.path[0] == str(source_dir):
                sys.path.pop(0)
        sys.path.insert(0, str(source_dir))
        try:
            model = attempt_load(str(weights_path), map_location=torch.device("cpu"))
            delattr(model.model[-1], "anchor_grid")
            model.model[-1].anchor_grid = [torch.zeros(1)] * 3
            model.model[-1].export_cat = True
            model.eval()
            gs = int(max(model.stride))
            image_size = [check_img_size(value, gs) for value in (640, 640)]
            dummy = torch.zeros(1, 3, *image_size)
            for _name, module in model.named_modules():
                module._non_persistent_buffers_set = set()
                if isinstance(module, models.common.Conv):
                    if isinstance(module.act, nn.Hardswish):
                        module.act = Hardswish()
                    elif isinstance(module.act, nn.SiLU):
                        module.act = SiLU()
                if isinstance(module, models.common.ShuffleV2Block):
                    for index, branch_module in enumerate(module.branch1):
                        if isinstance(branch_module, nn.SiLU):
                            module.branch1[index] = SiLU()
                    for index, branch_module in enumerate(module.branch2):
                        if isinstance(branch_module, nn.SiLU):
                            module.branch2[index] = SiLU()
            _ = model(dummy)
            model.fuse()
            torch.onnx.export(
                model,
                dummy,
                str(output_path),
                verbose=False,
                opset_version=12,
                input_names=["input"],
                output_names=["output"],
            )
        finally:
            if sys.path and sys.path[0] == str(source_dir):
                sys.path.pop(0)
            for module_name, previous in stubbed_modules.items():
                if previous is None:
                    sys.modules.pop(module_name, None)
                else:
                    sys.modules[module_name] = previous

    @staticmethod
    def _path_size(path: Path) -> int:
        if not path.exists():
            return 0
        if path.is_file():
            try:
                return int(path.stat().st_size)
            except OSError:
                return 0
        total = 0
        for child in path.rglob("*"):
            try:
                if child.is_file():
                    total += int(child.stat().st_size)
            except OSError:
                continue
        return total
