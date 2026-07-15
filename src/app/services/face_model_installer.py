from __future__ import annotations

import hashlib
import importlib.machinery
import importlib
import json
import shutil
import sys
import tempfile
import types
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from infra.cancel import raise_if_cancelled
from infra.settings import AppSettings, get_settings


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

    def bundle_dir(self, bundle_id: str) -> Path:
        mode, plural_kind, _label = self._bundle_layout(bundle_id)
        return self.runtime_root() / mode / plural_kind / bundle_id

    def inventory(self) -> tuple[FaceModelInventoryItem, ...]:
        items: list[FaceModelInventoryItem] = []
        for bundle_id, (_mode, plural_kind, label) in _BUNDLE_LAYOUTS.items():
            bundle_dir = self.bundle_dir(bundle_id)
            filename = "detector.onnx" if plural_kind == "detectors" else "embedder.onnx"
            payload_path = bundle_dir / filename
            metadata = self._catalog_metadata(bundle_id)
            installed = payload_path.exists()
            size_bytes = self._path_size(payload_path)
            status = "ready" if installed else "install required"
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
            self._install_zip_member(
                archive_path,
                member_name="det_2.5g.onnx",
                bundle_id="scrfd_2.5g_kps",
                output_name="detector.onnx",
            )
            installed.append("scrfd_2.5g_kps")
            raise_if_cancelled(cancel_check)
            progress and progress(82, "Installing ArcFace R50 embedder")
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
            self._install_zip_member(
                archive_path,
                member_name="det_500m.onnx",
                bundle_id="scrfd_500m_kps",
                output_name="detector.onnx",
            )
            installed.append("scrfd_500m_kps")
            raise_if_cancelled(cancel_check)
            progress and progress(82, "Installing MobileFaceNet ArcFace embedder")
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
            self._install_zip_member(
                archive_path,
                member_name="scrfd_10g_bnkps.onnx",
                bundle_id="scrfd_10g_kps",
                output_name="detector.onnx",
            )
            installed.append("scrfd_10g_kps")
            raise_if_cancelled(cancel_check)
            progress and progress(82, "Installing ArcFace R100 Glint360K embedder")
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

    def delete_installed_model(self, bundle_id: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        bundle_dir = self.bundle_dir(bundle_id)
        if not bundle_dir.exists():
            return (), ()
        try:
            shutil.rmtree(bundle_dir)
        except OSError as exc:
            return (), (f"{bundle_dir}: {exc}",)
        return (str(bundle_dir),), ()

    def bundled_catalog_root(self) -> Path:
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
        tmp_dir = Path(self.settings.cache_dir) / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        temp_root = Path(tempfile.mkdtemp(prefix="face-model-download-", dir=str(tmp_dir)))
        target_path = temp_root / filename
        request = urllib.request.Request(url, headers={"User-Agent": "ClusterLens/1.0"})
        digest = hashlib.sha256()
        with urllib.request.urlopen(request, timeout=300) as response, target_path.open("wb") as handle:
            total_bytes = int(response.headers.get("Content-Length") or 0)
            received = 0
            while True:
                raise_if_cancelled(cancel_check)
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                received += len(chunk)
                if progress is not None:
                    if total_bytes > 0:
                        percent = min(60, max(1, int((received / total_bytes) * 60.0)))
                        progress(percent, f"{progress_prefix} ({received // (1024 * 1024)} MB)")
                    else:
                        progress(10, progress_prefix)
        actual_sha = digest.hexdigest().lower()
        expected_sha = str(sha256 or "").strip().lower()
        if expected_sha and actual_sha != expected_sha:
            target_path.unlink(missing_ok=True)
            raise RuntimeError(f"Checksum mismatch for {filename}: expected {expected_sha}, got {actual_sha}.")
        return target_path, temp_root

    def _install_zip_member(self, archive_path: Path, *, member_name: str, bundle_id: str, output_name: str) -> None:
        with zipfile.ZipFile(archive_path) as archive:
            member = self._find_zip_member(archive, member_name)
            if member is None:
                raise RuntimeError(f"{archive_path.name} does not contain {member_name}.")
            bundle_dir = self.bundle_dir(bundle_id)
            bundle_dir.mkdir(parents=True, exist_ok=True)
            temp_output = bundle_dir / f"{output_name}.part"
            with archive.open(member, "r") as source, temp_output.open("wb") as target:
                shutil.copyfileobj(source, target)
            final_output = bundle_dir / output_name
            temp_output.replace(final_output)
            self._copy_catalog_metadata(bundle_id, bundle_dir)

    def _install_payload_file(self, payload_path: Path, *, bundle_id: str, output_name: str) -> None:
        bundle_dir = self.bundle_dir(bundle_id)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        temp_output = bundle_dir / f"{output_name}.part"
        shutil.copyfile(payload_path, temp_output)
        final_output = bundle_dir / output_name
        temp_output.replace(final_output)
        self._copy_catalog_metadata(bundle_id, bundle_dir)

    def _copy_catalog_metadata(self, bundle_id: str, bundle_dir: Path) -> None:
        relative_path = _CATALOG_METADATA_PATHS[bundle_id]
        source_path = self.bundled_catalog_root() / relative_path
        if not source_path.exists():
            raise RuntimeError(f"Bundled metadata is missing for {bundle_id}: {source_path}")
        shutil.copyfile(source_path, bundle_dir / "metadata.json")

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
            archive.extractall(install_root)
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
