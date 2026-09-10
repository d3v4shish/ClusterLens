import sys
import unittest
import os
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic, sleep
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image
from PyQt6.QtCore import QEvent, QItemSelectionModel, QModelIndex, QPoint, QSize, Qt, QSettings, QThread
from PyQt6.QtGui import QImage, QPainter, QPixmap
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QAbstractItemView, QGridLayout, QMenu, QMessageBox, QScrollArea, QSplitter, QStyleOptionViewItem, QTabBar, QToolButton, QVBoxLayout

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app.selection import SelectionTarget
from app.services.cluster_explanations import ClusterExplanation
from app.services.cluster_explanations import ClusterExplanation
from app.services.cluster_meanings import ClusterMeaning, ClusterMeaningLabel
from app.services.cache_maintenance import CacheClearResult, CacheUsageSummary, GeneratedStorageSummary
from app.services.clustering_pipeline import ClusteringRequest, MultiBackendClusteringResult
from app.services.face_search import BUILTIN_HUMAN_DETECTOR_ID, BUILTIN_HUMAN_EMBEDDER_ID, EditableFaceInput, FaceAlbumGroupSummary, FaceAlbumRecord, FaceClusterIdentitySuggestion, FaceClusterMember, FaceClusteringComparisonResult, FaceFolderReviewImage, FaceLabelAcceptanceBatch, FaceLabelAssignment, FaceScanImageRecord, FaceSearchRequest, FaceSearchResult, IndexedFaceRecord, PersonProfile, PersonPrototypeFace
from app.services.face_search import NamedPhotoSummary
from app.services.image_tags import ClusterTagSummary, ImageTagService
from app.services.saved_searches import SavedSearchService
from app.services.similarity_search import SearchResult
from infra.settings import get_settings
from infra.runtime import ExecutionPolicy, RuntimeCapabilities, RuntimeCapabilityService
import main as main_module
from main import ClusterGalleryApp
from ui.async_job import AsyncJob, start_job_in_thread, wait_for_thread_shutdown
from ui.cluster_pane import CLUSTER_BASIS_EMPTY_TEXT, ClusterPane
from ui.common import HelpIconButton
from ui.gallery_model import GalleryImageModel, GalleryItemDelegate, MAX_FACE_BOXES_PER_TILE
from ui.gallery_pane import (
    GalleryPane,
    MAX_PENDING_UI_ITEMS_PER_FLUSH,
    MAX_PREFETCH_THUMBNAIL_REQUESTS_PER_CYCLE,
    MAX_VISIBLE_THUMBNAIL_REQUESTS_PER_CYCLE,
)
from ui.job_manager import JobManager
from ui.list_models import ListEntryModel
from ui.names_pane import NamesPane
from ui.photo_inspector_dialog import EditableFaceDraft, PhotoInspectorDialog
from ui.search_pane import FaceResultGroup, FaceTileItem, FaceTileListModel, SearchPane
from ui.selection_details_pane import SelectionDetailsPane
from ui.settings_dialog import SettingsDialog
from ui.theme import ULTRA_DARK_QSS
from ui.runtime_widgets import RuntimeBadge
from ui.zoomable_image import ZoomableImageView


APP = QApplication.instance() or QApplication([])


class UiSmokeTests(unittest.TestCase):
    def setUp(self):
        settings = get_settings()
        self._app_settings_store = QSettings(settings.app_name, settings.app_name)
        self._saved_settings = {
            key: self._app_settings_store.value(key, None)
            for key in [
                "workspace/default_view",
                "workspace/faces_mode",
                "workspace/recent_folders_v1",
                "faces/default_mode",
                "faces/model_root",
                "faces/animal_model_root",
                "faces/default_detector/human",
                "faces/default_detector/dog",
                "faces/default_detector/cat",
                "faces/default_embedder/human",
                "faces/default_embedder/dog",
                "faces/default_embedder/cat",
                "faces/detector_score_threshold/human",
                "faces/detector_score_threshold/dog",
                "faces/detector_score_threshold/cat",
                "faces/max_detections/human",
                "faces/max_detections/dog",
                "faces/max_detections/cat",
                "clusteringpane/state",
                "facespane/state",
                "window/main_splitter",
                "window/cluster_right_splitter",
                "sourcepane/selected_directory",
                "runtime/allow_gpu_warmup",
                "runtime/preferred_mode",
                "tutorial/app_first_run_dismissed",
            ]
        }
        self._app_settings_store.setValue("workspace/default_view", "clustering")
        self._app_settings_store.remove("workspace/faces_mode")
        self._app_settings_store.remove("workspace/recent_folders_v1")
        self._app_settings_store.remove("faces/default_mode")
        self._app_settings_store.remove("faces/model_root")
        self._app_settings_store.remove("faces/animal_model_root")
        self._app_settings_store.remove("clusteringpane/state")
        self._app_settings_store.remove("facespane/state")
        self._app_settings_store.remove("window/main_splitter")
        self._app_settings_store.remove("window/cluster_right_splitter")
        self._app_settings_store.remove("sourcepane/selected_directory")
        self._app_settings_store.remove("tutorial/app_first_run_dismissed")
        self._app_settings_store.setValue("runtime/allow_gpu_warmup", False)
        self._app_settings_store.setValue("runtime/preferred_mode", "cpu")

    def tearDown(self):
        for key, value in self._saved_settings.items():
            if value is None:
                self._app_settings_store.remove(key)
            else:
                self._app_settings_store.setValue(key, value)

    @staticmethod
    def _wait_until(predicate, *, timeout_s: float = 5.0) -> bool:
        deadline = monotonic() + timeout_s
        while monotonic() < deadline:
            APP.processEvents()
            if predicate():
                return True
            sleep(0.01)
        APP.processEvents()
        return bool(predicate())

    @staticmethod
    def _list_view_count(view) -> int:
        model = view.model()
        if model is None:
            return 0
        return int(model.rowCount())

    @staticmethod
    def _list_view_text(view, row: int) -> str:
        model = view.model()
        if model is None:
            return ""
        index = model.index(int(row), 0)
        return str(model.data(index, Qt.ItemDataRole.DisplayRole) or "")

    @staticmethod
    def _list_view_payload(view, row: int):
        model = view.model()
        if model is None:
            return None
        index = model.index(int(row), 0)
        return index.data(Qt.ItemDataRole.UserRole + 1) if index.isValid() else None

    @staticmethod
    def _current_list_view_text(view) -> str:
        index = view.currentIndex()
        return str(index.data(Qt.ItemDataRole.DisplayRole) or "") if index.isValid() else ""

    @staticmethod
    def _select_list_view_row(view, row: int, *, append: bool = False) -> None:
        model = view.model()
        selection_model = view.selectionModel()
        if model is None or selection_model is None:
            return
        index = model.index(int(row), 0)
        flags = QItemSelectionModel.SelectionFlag.Select
        if not append:
            selection_model.clearSelection()
        selection_model.select(index, flags)
        view.scrollTo(index)
        APP.processEvents()

    @staticmethod
    def _selected_list_view_count(view) -> int:
        selection_model = view.selectionModel()
        if selection_model is None:
            return 0
        return len(selection_model.selectedIndexes())

    def test_main_configures_linux_dialog_fallback_by_default(self):
        with patch.object(main_module.sys, "platform", "linux"), patch.object(
            main_module.QApplication, "setAttribute"
        ) as set_attribute, patch.dict(os.environ, {}, clear=False):
            os.environ.pop("CLUSTERLENS_USE_NATIVE_FILE_DIALOGS", None)
            main_module._configure_linux_dialog_fallback()

        set_attribute.assert_called_once_with(Qt.ApplicationAttribute.AA_DontUseNativeDialogs, True)

    def test_main_skips_linux_dialog_fallback_when_native_dialogs_are_forced(self):
        with patch.object(main_module.sys, "platform", "linux"), patch.object(
            main_module.QApplication, "setAttribute"
        ) as set_attribute, patch.dict(os.environ, {"CLUSTERLENS_USE_NATIVE_FILE_DIALOGS": "1"}, clear=False):
            main_module._configure_linux_dialog_fallback()

        set_attribute.assert_not_called()

    class _FakeAsyncThread:
        def __init__(self, *, running: bool = True, wait_result: bool = True):
            self.running = running
            self.wait_result = wait_result
            self.quit_calls = 0
            self.wait_calls: list[int] = []
            self.disconnect_calls = 0

        def isRunning(self):
            return self.running

        def quit(self):
            self.quit_calls += 1

        @property
        def finished(self):
            return self

        def connect(self, _callback, *_args):
            return None

        def disconnect(self, *_args):
            self.disconnect_calls += 1

        def wait(self, timeout_ms: int):
            self.wait_calls.append(int(timeout_ms))
            self.running = False
            return self.wait_result

    class _FakeAsyncJob:
        def __init__(self):
            self.cancel_calls = 0

        def cancel(self):
            self.cancel_calls += 1

    class _FakeWorkerThread(_FakeAsyncThread):
        def __init__(self):
            super().__init__()
            self.cancel_calls = 0

        def cancel(self):
            self.cancel_calls += 1

    class _StickyLoaderThread(_FakeAsyncThread):
        def __init__(self):
            super().__init__(running=True, wait_result=False)
            self._finished_callbacks = []

        @property
        def finished(self):
            return self

        def connect(self, callback):
            self._finished_callbacks.append(callback)

        def wait(self, timeout_ms: int):
            self.wait_calls.append(int(timeout_ms))
            return False

    class _PollingAsyncQThread(QThread):
        def __init__(self):
            super().__init__()
            self.stop_event = Event()
            self.wait_calls: list[int] = []

        def wait(self, timeout_ms: int = 0):
            self.wait_calls.append(int(timeout_ms))
            return super().wait(timeout_ms)

        def run(self):
            while not self.stop_event.is_set():
                sleep(0.01)

    class _FakePipeline:
        def run(self, request, progress_callback, cancel_check):
            _ = request
            self.cancel_check_seen = bool(cancel_check())
            progress_callback(33, "working")
            return (
                MultiBackendClusteringResult(
                    clusters_by_key={"fake::cosine-kmeans": {0: ["a.jpg", "b.jpg"]}},
                    membership_by_image={},
                    metrics_by_key={"fake::cosine-kmeans": {"backend": "kmeans"}},
                ),
                {"image_count": 2},
            )

    class _FakeFaceLibraryService:
        def __init__(self, *, profiles=None, records=None, review_paths=None, pending_assignments=None, prototype_faces_by_name=None, duplicate_warnings=None, audit_events=None, ready=True, readiness_message="Human mode uses built-in face models."):
            self.profiles = list(profiles or [])
            self.records = list(records or [])
            self.review_paths = list(review_paths or [])
            self.pending_assignments = list(pending_assignments or [])
            self.audit_events = list(audit_events or [])
            self.prototype_faces_by_name = {
                str(name): list(values)
                for name, values in dict(prototype_faces_by_name or {}).items()
            }
            self.duplicate_warnings_map = {
                str(name): tuple(values)
                for name, values in dict(duplicate_warnings or {}).items()
            }
            self.ready = bool(ready)
            self._readiness_message = str(readiness_message)
            self.index_directory_calls: list[dict[str, object]] = []
            self.index_paths_calls: list[dict[str, object]] = []
            self.load_person_profiles_calls: list[dict[str, object]] = []
            self.load_person_prototype_faces_calls: list[dict[str, object]] = []
            self.load_indexed_faces_calls: list[dict[str, object]] = []
            self.load_folder_review_images_calls: list[dict[str, object]] = []
            self.load_face_album_groups_calls: list[dict[str, object]] = []
            self.load_face_album_members_calls: list[dict[str, object]] = []
            self.load_face_metadata_many_calls: list[list[tuple[str, int]]] = []
            self.load_pending_face_labels_calls: list[dict[str, object]] = []
            self.count_indexed_faces_calls: list[dict[str, object]] = []
            self.merge_person_labels_calls: list[tuple[str, str]] = []
            self.clear_person_labels_calls: list[str] = []
            self.save_image_faces_calls: list[dict[str, object]] = []
            self.search_faces_calls: list[dict[str, object]] = []
            self.search_query_faces_calls: list[dict[str, object]] = []
            self.search_similar_face_calls: list[dict[str, object]] = []
            self.search_similar_faces_calls: list[dict[str, object]] = []
            self.search_by_person_name_calls: list[dict[str, object]] = []
            self.label_indexed_faces_calls: list[dict[str, object]] = []
            self.label_indexed_faces_immediately_calls: list[dict[str, object]] = []
            self.cluster_faces_compare_calls: list[dict[str, object]] = []
            self.find_face_records_by_people_calls: list[dict[str, object]] = []
            self.find_face_records_with_named_people_calls: list[dict[str, object]] = []
            self.find_face_records_with_unknown_people_calls: list[dict[str, object]] = []
            self.find_face_records_with_primary_person_calls: list[dict[str, object]] = []
            self.search_face_query_calls: list[dict[str, object]] = []
            self.remove_person_prototype_face_calls: list[tuple[str, str, int]] = []
            self.pin_person_prototype_face_calls: list[tuple[str, str, int]] = []
            self.merge_person_identities_calls: list[tuple[str, str]] = []
            self.save_person_profile_calls: list[dict[str, object]] = []
            self.hide_face_calls: list[tuple[str, int]] = []
            self.import_identity_paths: list[str] = []
            self.import_identity_payloads: list[dict[str, object]] = []
            self.import_identity_modes: list[str] = []
            self.export_identity_data_calls = 0
            self.preview_identity_payloads: list[dict[str, object]] = []
            self.set_face_recognition_enabled_calls: list[bool] = []
            self.prepare_face_data_purge_calls = 0
            self.purge_face_data_calls: list[bool] = []
            self._last_pending_accept_batch: FaceLabelAcceptanceBatch | None = None

        def is_ready(self):
            return bool(self.ready)

        def readiness_message(self):
            return str(self._readiness_message)

        def assess_face_bbox(self, image_path, bbox, *, confidence=1.0):
            _ = image_path
            x1, y1, x2, y2 = [int(value) for value in bbox]
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            if float(confidence) < 0.3 or min(width, height) < 22:
                return "reject", 0.12, ("low_detector_score",)
            if float(confidence) < 0.58 or min(width, height) < 34:
                return "review", 0.54, ("review_confidence",)
            return "clean", 0.92, ()

        @staticmethod
        def _is_visible_record(record: IndexedFaceRecord, *, include_tiny_faces: bool) -> bool:
            if include_tiny_faces:
                return True
            x1, y1, x2, y2 = [int(value) for value in record.face_bbox]
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            return min(width, height) >= 28 and (width * height) >= 900

        def index_directory(self, directory, *, recursive=True, progress_callback=None, cancel_check=None):
            self.index_directory_calls.append({"directory": directory, "recursive": bool(recursive)})
            _ = cancel_check
            if callable(progress_callback):
                progress_callback(100, "done")
            image_count = len({record.image_path for record in self.records})
            return {
                "faces_indexed": len(self.records),
                "images_done": image_count,
                "images_total": image_count,
            }

        def index_paths(self, paths, *, progress_callback=None, cancel_check=None, force=False):
            self.index_paths_calls.append({"paths": list(paths or []), "force": bool(force)})
            _ = cancel_check
            if callable(progress_callback):
                progress_callback(100, "done")
            return {
                "faces_indexed": len(self.records),
                "images_done": len(self.records),
                "images_total": len(self.records),
            }

        def load_person_profiles(self, *, folder_prefix="", candidate_paths=None, limit=200, include_tiny_faces=True, include_hidden=False):
            self.load_person_profiles_calls.append(
                {
                    "folder_prefix": folder_prefix,
                    "candidate_paths": candidate_paths,
                    "limit": int(limit),
                    "include_tiny_faces": bool(include_tiny_faces),
                    "include_hidden": bool(include_hidden),
                }
            )
            return list(self.profiles)[: int(limit)]

        def load_person_prototype_faces(self, person_name, *, limit=100, include_tiny_faces=True):
            self.load_person_prototype_faces_calls.append(
                {
                    "person_name": str(person_name),
                    "limit": int(limit),
                    "include_tiny_faces": bool(include_tiny_faces),
                }
            )
            return list(self.prototype_faces_by_name.get(str(person_name), []))[: int(limit)]

        def save_person_profile(
            self,
            person_name,
            *,
            notes="",
            tags=None,
            cover_face_ref=None,
            favorite=None,
            birth_date=None,
            hidden=None,
        ):
            self.save_person_profile_calls.append(
                {
                    "person_name": str(person_name),
                    "notes": str(notes),
                    "tags": list(tags or []),
                    "cover_face_ref": cover_face_ref,
                    "favorite": favorite,
                    "birth_date": birth_date,
                    "hidden": hidden,
                }
            )

        def hide_face(self, image_path, face_index):
            self.hide_face_calls.append((str(image_path), int(face_index)))

        def export_identity_data(self):
            self.export_identity_data_calls += 1
            face_rows = [
                {
                    "image_path": str(record.image_path),
                    "face_index": int(record.face_index),
                    "bbox": list(record.face_bbox),
                    "face_confidence": float(record.face_confidence),
                    "embedding": np.asarray(record.embedding, dtype=np.float32).astype(float).tolist(),
                    "quality_status": str(record.quality_status or "clean"),
                    "quality_score": float(record.quality_score),
                    "quality_reasons": list(record.quality_reasons or ()),
                    "quality_revision": 1,
                    "mtime_ns": 0,
                    "file_size": 0,
                    "hidden": bool(getattr(record, "hidden", False)),
                }
                for record in self.records
            ]
            face_scan_images = []
            for image_path in sorted({str(record.image_path) for record in self.records}):
                count = len([record for record in self.records if str(record.image_path) == image_path])
                face_scan_images.append(
                    {
                        "image_path": image_path,
                        "mtime_ns": 0,
                        "file_size": 0,
                        "face_count": count,
                        "image_width": 100,
                        "image_height": 100,
                        "indexed_at": "",
                    }
                )
            labels = [
                {
                    "image_path": str(record.image_path),
                    "face_index": int(record.face_index),
                    "person_name": str(record.person_name),
                    "confidence": float(record.label_confidence),
                }
                for record in self.records
                if str(record.person_name or "").strip()
            ]
            pending_labels = [
                {
                    "image_path": str(assignment.image_path),
                    "face_index": int(assignment.face_index),
                    "person_name": str(assignment.person_name),
                    "confidence": float(assignment.confidence),
                    "source": str(assignment.source or ""),
                    "created_at": str(assignment.created_at or ""),
                }
                for assignment in self.pending_assignments
            ]
            identities = []
            for profile in self.profiles:
                prototype_refs = [
                    {
                        "image_path": str(record.image_path),
                        "face_index": int(record.face_index),
                        "pinned": False,
                        "sort_order": index,
                    }
                    for index, record in enumerate(self.records)
                    if str(record.person_name or "") == str(profile.person_name)
                ]
                identities.append(
                    {
                        "person_name": str(profile.person_name),
                        "notes": str(profile.notes or ""),
                        "tags": list(profile.tags or ()),
                        "cover_image_path": str(profile.cover_image_path or ""),
                        "cover_face_index": int(profile.cover_face_index),
                        "favorite": bool(profile.favorite),
                        "birth_date": str(profile.birth_date or ""),
                        "hidden": bool(profile.hidden),
                        "similarity_threshold": float(profile.similarity_threshold),
                        "example_count": int(profile.example_count),
                        "prototype_embedding": [],
                        "prototype_refs": prototype_refs,
                    }
                )
            return {
                "version": 1,
                "mode": "human",
                "identities": identities,
                "face_index": face_rows,
                "face_scan_images": face_scan_images,
                "labels": labels,
                "pending_labels": pending_labels,
                "corrections": [],
            }

        def preview_identity_import(self, payload):
            self.preview_identity_payloads.append(dict(payload or {}))
            identities = list((payload or {}).get("identities", []) or [])
            labels = list((payload or {}).get("labels", []) or [])
            return {
                "counts": {
                    "identities": len(identities),
                    "labels": len(labels),
                    "prototypes": sum(len(list(item.get("prototype_refs", []) or [])) for item in identities),
                    "hidden_items": sum(1 for item in identities if bool(item.get("hidden", False))),
                    "corrections": len(list((payload or {}).get("corrections", []) or [])),
                },
                "conflicts": [{"type": "identity", "person_name": "Alice"}] if identities else [],
                "conflict_count": 1 if identities else 0,
            }

        def import_identity_data(self, payload, *, mode="merge"):
            self.import_identity_payloads.append(dict(payload or {}))
            self.import_identity_modes.append(str(mode))

        def import_identity_data_from_file(self, input_path, *, mode="merge"):
            path = str(input_path)
            self.import_identity_paths.append(path)
            self.import_identity_data(json.loads(Path(path).read_text(encoding="utf-8")), mode=mode)

        def set_face_recognition_enabled(self, enabled):
            self.set_face_recognition_enabled_calls.append(bool(enabled))

        def prepare_face_data_purge(self):
            self.prepare_face_data_purge_calls += 1
            return {
                "table_counts": {"face_index": len(self.records), "person_profiles": len(self.profiles)},
                "ann_files": ["/tmp/faces.faiss"],
                "model_cache_targets": ["/tmp/face-model-cache"],
            }

        def purge_face_data(self, *, confirm=False):
            self.purge_face_data_calls.append(bool(confirm))
            if not confirm:
                raise ValueError("confirm required")
            self.records = []
            self.profiles = []
            self.prototype_faces_by_name = {}
            self.pending_assignments = []
            return {
                "removed_ann_files": ["/tmp/faces.faiss"],
                "removed_model_cache_targets": ["/tmp/face-model-cache"],
            }

        def load_face_action_audit(self, *, limit=50):
            return list(self.audit_events)[: int(limit)]

        def age_at_photo(self, person_name, image_path):
            _ = (person_name, image_path)
            return 24

        def identity_duplicate_warnings(self, *, similarity_threshold=0.92):
            _ = similarity_threshold
            return dict(self.duplicate_warnings_map)

        def count_indexed_faces(self, *, folder_prefix="", candidate_paths=None, include_tiny_faces=True):
            self.count_indexed_faces_calls.append(
                {
                    "folder_prefix": folder_prefix,
                    "candidate_paths": candidate_paths,
                    "include_tiny_faces": bool(include_tiny_faces),
                }
            )
            records = [
                record
                for record in self.records
                if self._is_visible_record(record, include_tiny_faces=bool(include_tiny_faces))
            ]
            return len(records)

        def load_indexed_faces(self, *, folder_prefix="", candidate_paths=None, limit=200, include_tiny_faces=True):
            self.load_indexed_faces_calls.append(
                {
                    "folder_prefix": folder_prefix,
                    "candidate_paths": candidate_paths,
                    "limit": int(limit),
                    "include_tiny_faces": bool(include_tiny_faces),
                }
            )
            records = [
                record
                for record in self.records
                if self._is_visible_record(record, include_tiny_faces=bool(include_tiny_faces))
            ]
            return records[: int(limit)]

        def load_folder_review_images(self, directory, *, recursive=True, candidate_paths=None, include_tiny_faces=True):
            self.load_folder_review_images_calls.append(
                {
                    "directory": str(directory),
                    "recursive": bool(recursive),
                    "candidate_paths": candidate_paths,
                    "include_tiny_faces": bool(include_tiny_faces),
                }
            )
            paths = list(dict.fromkeys([record.image_path for record in self.records] + list(self.review_paths)))
            review_images: list[FaceFolderReviewImage] = []
            for path in paths:
                all_records = [record for record in self.records if record.image_path == path]
                visible_records = [
                    record
                    for record in all_records
                    if self._is_visible_record(record, include_tiny_faces=bool(include_tiny_faces))
                ]
                if visible_records:
                    status = "detected"
                elif all_records:
                    status = "tiny_hidden"
                else:
                    status = "not_scanned"
                review_images.append(
                    FaceFolderReviewImage(
                        image_path=path,
                        review_status=status,
                        visible_faces=tuple(visible_records),
                        total_face_count=len(all_records),
                        hidden_face_count=max(0, len(all_records) - len(visible_records)),
                        image_width=100,
                        image_height=100,
                    )
                )
            return review_images

        def load_image_faces(self, image_path, *, include_tiny_faces=True):
            records = [record for record in self.records if record.image_path == str(image_path)]
            if include_tiny_faces:
                return list(records)
            return [record for record in records if self._is_visible_record(record, include_tiny_faces=False)]

        def load_face_record(self, image_path, face_index):
            for record in self.records:
                if str(record.image_path) == str(image_path) and int(record.face_index) == int(face_index):
                    return record
            return None

        def load_face_metadata_many(self, face_refs, *, include_hidden=False):
            _ = include_hidden
            refs = [
                (str(image_path), int(face_index))
                for image_path, face_index in list(face_refs or [])
                if str(image_path).strip()
            ]
            self.load_face_metadata_many_calls.append(list(refs))
            payload: dict[tuple[str, int], tuple[str, str]] = {}
            for image_path, face_index in refs:
                record = self.load_face_record(image_path, face_index)
                if record is None:
                    continue
                payload[(image_path, face_index)] = (
                    str(record.person_name or ""),
                    str(record.quality_status or "clean").strip().lower(),
                )
            return payload

        def _album_records(self, *, include_tiny_faces=True, folder_prefix="", candidate_paths=None):
            allowed_paths = {str(path) for path in list(candidate_paths or []) if str(path).strip()}
            album_records: list[FaceAlbumRecord] = []
            records_by_ref: dict[tuple[str, int], IndexedFaceRecord] = {}
            for record in self.records:
                if folder_prefix and not str(record.image_path).startswith(str(folder_prefix)):
                    continue
                if allowed_paths and str(record.image_path) not in allowed_paths:
                    continue
                if not self._is_visible_record(record, include_tiny_faces=bool(include_tiny_faces)):
                    continue
                person_name = str(record.person_name or "").strip()
                if bool(getattr(record, "hidden", False)):
                    group_kind = "hidden"
                elif person_name:
                    group_kind = "named"
                else:
                    group_kind = "unlabeled"
                group_id = f"person:{person_name}" if group_kind == "named" else group_kind
                album_records.append(
                    FaceAlbumRecord(
                        group_id=group_id,
                        group_kind=group_kind,
                        image_path=str(record.image_path),
                        face_index=int(record.face_index),
                        face_bbox=tuple(record.face_bbox),
                        face_confidence=float(record.face_confidence),
                        person_name=person_name,
                        label_confidence=float(record.label_confidence),
                        quality_status=str(record.quality_status or "clean"),
                        quality_score=float(record.quality_score),
                        quality_reasons=tuple(record.quality_reasons or ()),
                        hidden=bool(getattr(record, "hidden", False)),
                        display_name=person_name or "Unlabeled",
                    )
                )
                records_by_ref[(str(record.image_path), int(record.face_index))] = record
            for assignment in self.pending_assignments:
                ref = (str(assignment.image_path), int(assignment.face_index))
                record = records_by_ref.get(ref)
                if record is None:
                    continue
                album_records.append(
                    FaceAlbumRecord(
                        group_id="pending",
                        group_kind="pending",
                        image_path=str(record.image_path),
                        face_index=int(record.face_index),
                        face_bbox=tuple(record.face_bbox),
                        face_confidence=float(record.face_confidence),
                        person_name=str(assignment.person_name or ""),
                        label_confidence=float(assignment.confidence),
                        quality_status=str(record.quality_status or "clean"),
                        quality_score=float(record.quality_score),
                        quality_reasons=tuple(record.quality_reasons or ()),
                        display_name=str(assignment.person_name or "") or "Pending label",
                        source=str(assignment.source or ""),
                        created_at=str(assignment.created_at or ""),
                    )
                )
            return album_records

        def load_face_album_members(self, *, folder_prefix="", candidate_paths=None, include_tiny_faces=True):
            self.load_face_album_members_calls.append(
                {
                    "folder_prefix": str(folder_prefix),
                    "candidate_paths": candidate_paths,
                    "include_tiny_faces": bool(include_tiny_faces),
                }
            )
            return self._album_records(
                include_tiny_faces=bool(include_tiny_faces),
                folder_prefix=str(folder_prefix),
                candidate_paths=candidate_paths,
            )

        def load_face_album_groups(self, *, folder_prefix="", candidate_paths=None, include_tiny_faces=True):
            self.load_face_album_groups_calls.append(
                {
                    "folder_prefix": str(folder_prefix),
                    "candidate_paths": candidate_paths,
                    "include_tiny_faces": bool(include_tiny_faces),
                }
            )
            groups_by_id: dict[str, list[FaceAlbumRecord]] = {}
            for record in self._album_records(
                include_tiny_faces=bool(include_tiny_faces),
                folder_prefix=str(folder_prefix),
                candidate_paths=candidate_paths,
            ):
                groups_by_id.setdefault(str(record.group_id), []).append(record)
            summaries: list[FaceAlbumGroupSummary] = []
            for group_id, records in groups_by_id.items():
                first = records[0]
                face_count = len(records)
                photo_count = len({str(item.image_path) for item in records})
                if first.group_kind == "named":
                    prefix = "[Favorite] " if any(str(profile.person_name) == str(first.person_name) and bool(profile.favorite) for profile in self.profiles) else ""
                    summaries.append(
                        FaceAlbumGroupSummary(
                            group_id=group_id,
                            group_kind="named",
                            title=f"{prefix}{first.person_name} | {face_count} face(s)",
                            summary=f"{first.person_name}: {face_count} face(s) across {photo_count} photo(s).",
                            face_count=face_count,
                            photo_count=photo_count,
                            person_name=str(first.person_name),
                        )
                    )
                elif first.group_kind == "pending":
                    summaries.append(
                        FaceAlbumGroupSummary(
                            group_id=group_id,
                            group_kind="pending",
                            title=f"Pending Labels | {face_count} face(s)",
                            summary=f"{face_count} pending face assignment(s) across {photo_count} photo(s).",
                            face_count=face_count,
                            photo_count=photo_count,
                        )
                    )
                elif first.group_kind == "hidden":
                    summaries.append(
                        FaceAlbumGroupSummary(
                            group_id=group_id,
                            group_kind="hidden",
                            title=f"Hidden / Rejected | {face_count} face(s)",
                            summary=f"{face_count} hidden face(s) across {photo_count} photo(s).",
                            face_count=face_count,
                            photo_count=photo_count,
                        )
                    )
                else:
                    summaries.append(
                        FaceAlbumGroupSummary(
                            group_id=group_id,
                            group_kind="unlabeled",
                            title=f"Unlabeled | {face_count} face(s)",
                            summary=f"{face_count} unlabeled face(s) across {photo_count} photo(s).",
                            face_count=face_count,
                            photo_count=photo_count,
                        )
                    )
            return sorted(summaries, key=lambda item: (0 if item.group_kind == "named" else 1, item.title.lower()))

        def save_image_faces(self, image_path, face_boxes, preserve_labels=True):
            self.save_image_faces_calls.append(
                {
                    "image_path": str(image_path),
                    "preserve_labels": bool(preserve_labels),
                    "face_count": len(list(face_boxes or [])),
                }
            )
            prior_records = [record for record in self.records if record.image_path == str(image_path)]
            next_records: list[IndexedFaceRecord] = []
            for index, face in enumerate(list(face_boxes or [])):
                matched = next(
                    (
                        record
                        for record in prior_records
                        if tuple(record.face_bbox) == tuple(face.bbox)
                    ),
                    None,
                )
                next_records.append(
                    IndexedFaceRecord(
                        image_path=str(image_path),
                        face_index=int(index),
                        face_bbox=tuple(int(value) for value in face.bbox),
                        face_confidence=float(face.confidence or 1.0),
                        embedding=np.zeros((512,), dtype=np.float32),
                        person_name=str(matched.person_name if matched is not None else ""),
                        label_confidence=float(matched.label_confidence if matched is not None else 0.0),
                        quality_status=str((matched.quality_status if matched is not None else self.assess_face_bbox(image_path, tuple(int(value) for value in face.bbox), confidence=float(face.confidence or 1.0))[0])),
                        quality_score=float(matched.quality_score if matched is not None else self.assess_face_bbox(image_path, tuple(int(value) for value in face.bbox), confidence=float(face.confidence or 1.0))[1]),
                        quality_reasons=tuple(matched.quality_reasons if matched is not None else self.assess_face_bbox(image_path, tuple(int(value) for value in face.bbox), confidence=float(face.confidence or 1.0))[2]),
                    )
                )
            self.records = [record for record in self.records if record.image_path != str(image_path)] + next_records
            return list(next_records)

        def list_face_labels(self, *, include_tiny_faces=True):
            return [
                SimpleNamespace(
                    image_path=str(record.image_path),
                    face_index=int(record.face_index),
                    person_name=str(record.person_name or ""),
                    confidence=float(record.label_confidence or 0.0),
                    face_bbox=tuple(record.face_bbox),
                )
                for record in self.records
                if self._is_visible_record(record, include_tiny_faces=bool(include_tiny_faces)) and str(record.person_name or "").strip()
            ]

        def _replace_record_identity(self, record, person_name, *, confidence):
            return IndexedFaceRecord(
                image_path=str(record.image_path),
                face_index=int(record.face_index),
                face_bbox=tuple(record.face_bbox),
                face_confidence=float(record.face_confidence),
                embedding=np.asarray(record.embedding, dtype=np.float32),
                person_name=str(person_name or ""),
                label_confidence=float(confidence),
                quality_status=str(record.quality_status or "clean"),
                quality_score=float(record.quality_score),
                quality_reasons=tuple(record.quality_reasons or ()),
                hidden=bool(getattr(record, "hidden", False)),
                image_mtime=float(getattr(record, "image_mtime", 0.0) or 0.0),
            )

        def label_indexed_faces(self, person_name, face_refs, *, similarity_threshold=0.72):
            self.label_indexed_faces_calls.append(
                {
                    "person_name": str(person_name),
                    "face_refs": [(str(image_path), int(face_index)) for image_path, face_index in list(face_refs or [])],
                    "similarity_threshold": float(similarity_threshold),
                }
            )
            refs = [(str(image_path), int(face_index)) for image_path, face_index in list(face_refs or [])]
            for offset, (image_path, face_index) in enumerate(refs, start=1):
                self.pending_assignments.append(
                    FaceLabelAssignment(
                        person_name=str(person_name),
                        image_path=image_path,
                        face_index=int(face_index),
                        face_bbox=(0, 0, 0, 0),
                        confidence=1.0,
                        source="manual_selected_faces",
                        proposal_id=10_000 + offset,
                    )
                )
            return SimpleNamespace(person_name=str(person_name))

        def label_indexed_faces_immediately(
            self,
            person_name,
            face_refs,
            *,
            similarity_threshold=0.72,
            source="manual_cluster_name",
        ):
            refs = [(str(image_path), int(face_index)) for image_path, face_index in list(face_refs or [])]
            self.label_indexed_faces_immediately_calls.append(
                {
                    "person_name": str(person_name),
                    "face_refs": list(refs),
                    "similarity_threshold": float(similarity_threshold),
                    "source": str(source),
                }
            )
            ref_set = set(refs)
            self.records = [
                self._replace_record_identity(record, person_name, confidence=1.0)
                if (str(record.image_path), int(record.face_index)) in ref_set
                else record
                for record in self.records
            ]
            self.pending_assignments = [
                assignment
                for assignment in self.pending_assignments
                if (str(assignment.image_path), int(assignment.face_index)) not in ref_set
            ]
            return SimpleNamespace(person_name=str(person_name))

        def _search_results(self):
            return [
                FaceSearchResult(
                    image_path=str(record.image_path),
                    face_index=int(record.face_index),
                    score=round(0.95 - (0.05 * index), 4),
                    phash_distance=-1,
                    face_bbox=tuple(record.face_bbox),
                    face_confidence=float(record.face_confidence),
                    model_name="fake",
                    match_reason="fake-search",
                )
                for index, record in enumerate(self.records)
            ]

        def search_faces(self, request: FaceSearchRequest):
            self.search_faces_calls.append({"request": request})
            return self._search_results()[: max(1, int(request.top_k))]

        def search_query_faces(self, query_face_image, query_face_bboxes, *, top_k=30, min_face_score=0.35, candidate_paths=None, include_tiny_faces=True):
            self.search_query_faces_calls.append(
                {
                    "query_face_image": str(query_face_image),
                    "query_face_bboxes": [tuple(int(value) for value in bbox) for bbox in list(query_face_bboxes or [])],
                    "top_k": int(top_k),
                    "min_face_score": float(min_face_score),
                    "candidate_paths": candidate_paths,
                    "include_tiny_faces": bool(include_tiny_faces),
                }
            )
            return self._search_results()[: max(1, int(top_k))]

        def search_similar_face(self, image_path, face_index, *, top_k=30, min_score=0.35, folder_prefix="", candidate_paths=None, include_tiny_faces=True):
            self.search_similar_face_calls.append(
                {
                    "image_path": str(image_path),
                    "face_index": int(face_index),
                    "top_k": int(top_k),
                    "min_score": float(min_score),
                    "folder_prefix": str(folder_prefix),
                    "candidate_paths": candidate_paths,
                    "include_tiny_faces": bool(include_tiny_faces),
                }
            )
            return self._search_results()[: max(1, int(top_k))]

        def search_similar_faces(self, face_refs, *, top_k=30, min_score=0.35, folder_prefix="", candidate_paths=None, include_tiny_faces=True):
            self.search_similar_faces_calls.append(
                {
                    "face_refs": [(str(image_path), int(face_index)) for image_path, face_index in list(face_refs or [])],
                    "top_k": int(top_k),
                    "min_score": float(min_score),
                    "folder_prefix": str(folder_prefix),
                    "candidate_paths": candidate_paths,
                    "include_tiny_faces": bool(include_tiny_faces),
                }
            )
            return self._search_results()[: max(1, int(top_k))]

        def search_by_person_name(self, person_name, *, top_k=60, min_score=None, folder_prefix="", candidate_paths=None, include_tiny_faces=True):
            self.search_by_person_name_calls.append(
                {
                    "person_name": str(person_name),
                    "top_k": int(top_k),
                    "min_score": None if min_score is None else float(min_score),
                    "folder_prefix": str(folder_prefix),
                    "candidate_paths": candidate_paths,
                    "include_tiny_faces": bool(include_tiny_faces),
                }
            )
            return self._search_results()[: max(1, int(top_k))]

        def find_face_records_by_people(self, person_names, *, require_all=True, candidate_paths=None, include_tiny_faces=True):
            self.find_face_records_by_people_calls.append(
                {
                    "person_names": [str(value) for value in list(person_names or [])],
                    "require_all": bool(require_all),
                    "candidate_paths": candidate_paths,
                    "include_tiny_faces": bool(include_tiny_faces),
                }
            )
            wanted = {str(value).strip().lower() for value in list(person_names or []) if str(value).strip()}
            return [
                record
                for record in self.records
                if str(record.person_name or "").strip().lower() in wanted
            ]

        def find_face_records_with_named_people(self, *, candidate_paths=None, include_tiny_faces=True):
            self.find_face_records_with_named_people_calls.append(
                {
                    "candidate_paths": candidate_paths,
                    "include_tiny_faces": bool(include_tiny_faces),
                }
            )
            return [record for record in self.records if str(record.person_name or "").strip()]

        def find_face_records_with_unknown_people(self, *, candidate_paths=None, include_tiny_faces=True):
            self.find_face_records_with_unknown_people_calls.append(
                {
                    "candidate_paths": candidate_paths,
                    "include_tiny_faces": bool(include_tiny_faces),
                }
            )
            return [record for record in self.records if not str(record.person_name or "").strip()]

        def find_face_records_with_primary_person(self, person_name, *, candidate_paths=None, include_tiny_faces=True):
            self.find_face_records_with_primary_person_calls.append(
                {
                    "person_name": str(person_name),
                    "candidate_paths": candidate_paths,
                    "include_tiny_faces": bool(include_tiny_faces),
                }
            )
            target = str(person_name or "").strip().lower()
            return [
                record
                for record in self.records
                if str(record.person_name or "").strip().lower() == target
            ][:1]

        def search_face_query(self, query, *, candidate_paths=None, include_tiny_faces=True, include_hidden=False):
            self.search_face_query_calls.append(
                {
                    "query": str(query),
                    "candidate_paths": candidate_paths,
                    "include_tiny_faces": bool(include_tiny_faces),
                    "include_hidden": bool(include_hidden),
                }
            )
            return list(self.records)

        def load_pending_face_labels(self, *, folder_prefix="", candidate_paths=None, include_tiny_faces=True, include_hidden=False):
            self.load_pending_face_labels_calls.append(
                {
                    "folder_prefix": str(folder_prefix),
                    "candidate_paths": list(candidate_paths or []) if candidate_paths is not None else None,
                    "include_tiny_faces": bool(include_tiny_faces),
                    "include_hidden": bool(include_hidden),
                }
            )
            return list(self.pending_assignments)

        def accept_pending_face_labels(self, proposal_ids):
            accepted = [assignment for assignment in self.pending_assignments if int(assignment.proposal_id) in {int(value) for value in proposal_ids}]
            self.pending_assignments = [assignment for assignment in self.pending_assignments if int(assignment.proposal_id) not in {int(value) for value in proposal_ids}]
            return accepted

        def accept_pending_face_labels_batch(self, proposal_ids):
            accepted = tuple(self.accept_pending_face_labels(proposal_ids))
            batch = FaceLabelAcceptanceBatch(accepted=accepted, previous_labels=())
            self._last_pending_accept_batch = batch
            return batch

        def reject_pending_face_labels(self, proposal_ids):
            before = len(self.pending_assignments)
            ids = {int(value) for value in proposal_ids}
            self.pending_assignments = [assignment for assignment in self.pending_assignments if int(assignment.proposal_id) not in ids]
            return before - len(self.pending_assignments)

        def undo_last_pending_face_acceptance(self):
            batch = self._last_pending_accept_batch
            if batch is None:
                return []
            self.pending_assignments = list(batch.accepted) + list(self.pending_assignments)
            self._last_pending_accept_batch = None
            return list(batch.accepted)

        def cluster_faces(
            self,
            num_clusters=12,
            min_face_score=0.0,
            *,
            backend="hdbscan",
            backend_options=None,
            folder_prefix="",
            candidate_paths=None,
            face_refs=None,
            include_tiny_faces=True,
        ):
            _ = (num_clusters, min_face_score, backend, backend_options, folder_prefix, candidate_paths, face_refs, include_tiny_faces)
            return {
                0: [
                    FaceClusterMember(
                        image_path=record.image_path,
                        face_index=int(record.face_index),
                        face_bbox=tuple(record.face_bbox),
                        face_confidence=float(record.face_confidence),
                        person_name=str(record.person_name or ""),
                        quality_status=str(record.quality_status or "clean"),
                    )
                    for record in self.records
                ]
            }

        def cluster_faces_compare(
            self,
            num_clusters=12,
            min_face_score=0.0,
            *,
            backends=None,
            outlier_policy="isolate",
            backend_options_by_backend=None,
            folder_prefix="",
            candidate_paths=None,
            face_refs=None,
            include_tiny_faces=True,
        ):
            self.cluster_faces_compare_calls.append(
                {
                    "num_clusters": int(num_clusters),
                    "min_face_score": float(min_face_score),
                    "backends": list(backends or []),
                    "outlier_policy": str(outlier_policy),
                    "backend_options_by_backend": dict(backend_options_by_backend or {}),
                    "folder_prefix": str(folder_prefix),
                    "candidate_paths": candidate_paths,
                    "face_refs": face_refs,
                    "include_tiny_faces": bool(include_tiny_faces),
                }
            )
            selected_backends = [str(value) for value in (backends or ["hdbscan"])]
            return FaceClusteringComparisonResult(
                clusters_by_key={
                    backend: self.cluster_faces(
                        num_clusters=num_clusters,
                        min_face_score=min_face_score,
                        backend=backend,
                        backend_options=dict((backend_options_by_backend or {}).get(str(backend), {}) or {}),
                        folder_prefix=folder_prefix,
                        candidate_paths=candidate_paths,
                        face_refs=face_refs,
                        include_tiny_faces=include_tiny_faces,
                    )
                    for backend in selected_backends
                },
                membership_by_face_ref={},
                metrics_by_key={backend: {"requested_backend": backend, "cluster_quality_score": 0.5} for backend in selected_backends},
                explanations_by_key={},
                identity_suggestions_by_key={},
            )

        def merge_person_labels(self, source, target):
            self.merge_person_labels_calls.append((str(source), str(target)))

        def merge_person_identities(self, source, target):
            source = str(source)
            target = str(target)
            self.merge_person_identities_calls.append((source, target))
            source_faces = list(self.prototype_faces_by_name.pop(source, []))
            target_faces = list(self.prototype_faces_by_name.get(target, []))
            merged_faces: list[PersonPrototypeFace] = []
            seen_refs: set[tuple[str, int]] = set()
            for face in list(target_faces) + list(source_faces):
                ref = (str(face.image_path), int(face.face_index))
                if ref in seen_refs:
                    continue
                seen_refs.add(ref)
                merged_faces.append(
                    PersonPrototypeFace(
                        person_name=target,
                        image_path=str(face.image_path),
                        face_index=int(face.face_index),
                        face_bbox=tuple(face.face_bbox),
                        face_confidence=float(face.face_confidence),
                        quality_status=str(face.quality_status),
                        quality_score=float(face.quality_score),
                        quality_reasons=tuple(face.quality_reasons),
                        pinned=bool(face.pinned and not any(item.pinned for item in merged_faces)),
                        label_person_name=str(face.label_person_name),
                        label_confidence=float(face.label_confidence),
                    )
                )
            if merged_faces and not any(face.pinned for face in merged_faces):
                first = merged_faces[0]
                merged_faces[0] = PersonPrototypeFace(
                    person_name=first.person_name,
                    image_path=first.image_path,
                    face_index=first.face_index,
                    face_bbox=first.face_bbox,
                    face_confidence=first.face_confidence,
                    quality_status=first.quality_status,
                    quality_score=first.quality_score,
                    quality_reasons=first.quality_reasons,
                    pinned=True,
                    label_person_name=first.label_person_name,
                    label_confidence=first.label_confidence,
                )
            self.prototype_faces_by_name[target] = merged_faces
            self.profiles = [profile for profile in self.profiles if str(profile.person_name) != source]

        def clear_person_labels(self, person_name):
            self.clear_person_labels_calls.append(str(person_name))

        def remove_person_prototype_face(self, person_name, image_path, face_index):
            name = str(person_name)
            path = str(image_path)
            index = int(face_index)
            self.remove_person_prototype_face_calls.append((name, path, index))
            faces = list(self.prototype_faces_by_name.get(name, []))
            faces = [face for face in faces if (str(face.image_path), int(face.face_index)) != (path, index)]
            if faces and not any(face.pinned for face in faces):
                first = faces[0]
                faces[0] = PersonPrototypeFace(
                    person_name=first.person_name,
                    image_path=first.image_path,
                    face_index=first.face_index,
                    face_bbox=first.face_bbox,
                    face_confidence=first.face_confidence,
                    quality_status=first.quality_status,
                    quality_score=first.quality_score,
                    quality_reasons=first.quality_reasons,
                    pinned=True,
                    label_person_name=first.label_person_name,
                    label_confidence=first.label_confidence,
                )
            self.prototype_faces_by_name[name] = faces
            for idx, profile in enumerate(list(self.profiles)):
                if str(profile.person_name) == name:
                    self.profiles[idx] = PersonProfile(
                        person_name=profile.person_name,
                        similarity_threshold=profile.similarity_threshold,
                        example_count=max(1, len(faces)),
                        labeled_count=profile.labeled_count,
                        visible_face_count=profile.visible_face_count,
                        notes=profile.notes,
                        tags=profile.tags,
                        cover_image_path=str(faces[0].image_path if faces else ""),
                        cover_face_index=int(faces[0].face_index if faces else 0),
                    )
                    break

        def pin_person_prototype_face(self, person_name, image_path, face_index):
            name = str(person_name)
            path = str(image_path)
            index = int(face_index)
            self.pin_person_prototype_face_calls.append((name, path, index))
            updated: list[PersonPrototypeFace] = []
            for face in list(self.prototype_faces_by_name.get(name, [])):
                updated.append(
                    PersonPrototypeFace(
                        person_name=face.person_name,
                        image_path=face.image_path,
                        face_index=face.face_index,
                        face_bbox=face.face_bbox,
                        face_confidence=face.face_confidence,
                        quality_status=face.quality_status,
                        quality_score=face.quality_score,
                        quality_reasons=face.quality_reasons,
                        pinned=(str(face.image_path), int(face.face_index)) == (path, index),
                        label_person_name=face.label_person_name,
                        label_confidence=face.label_confidence,
                    )
                )
            updated.sort(key=lambda face: (not face.pinned, str(face.image_path), int(face.face_index)))
            self.prototype_faces_by_name[name] = updated

    class _FakeFacePipelineService(_FakeFaceLibraryService):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.configure_calls: list[dict[str, object]] = []
            self.configure_quality_calls: list[dict[str, object]] = []
            self.configure_cascade_calls: list[dict[str, object]] = []
            self.configure_recognition_calls: list[dict[str, object]] = []

        def configure_detector_options(self, *, score_threshold=None, max_detections=None):
            self.configure_calls.append(
                {
                    "score_threshold": score_threshold,
                    "max_detections": max_detections,
                }
            )

        def configure_quality_options(self, *, profile_id=None, thresholds=None):
            self.configure_quality_calls.append(
                {
                    "profile_id": profile_id,
                    "thresholds": dict(thresholds or {}),
                }
            )

        def configure_cascade_options(self, *, detector_policy=None, fallback_detector_id=None, verifier_mode=None):
            self.configure_cascade_calls.append(
                {
                    "detector_policy": detector_policy,
                    "fallback_detector_id": fallback_detector_id,
                    "verifier_mode": verifier_mode,
                }
            )

        def configure_recognition_options(
            self,
            *,
            search_quality_min=None,
            cluster_quality_min=None,
            prototype_quality_min=None,
            recognition_min_score=None,
            auto_label_min_score=None,
            rerank_policy=None,
            rerank_top_n=None,
        ):
            self.configure_recognition_calls.append(
                {
                    "search_quality_min": search_quality_min,
                    "cluster_quality_min": cluster_quality_min,
                    "prototype_quality_min": prototype_quality_min,
                    "recognition_min_score": recognition_min_score,
                    "auto_label_min_score": auto_label_min_score,
                    "rerank_policy": rerank_policy,
                    "rerank_top_n": rerank_top_n,
                }
            )

        def assess_face_bbox(self, image_path, bbox, *, confidence=1.0):
            _ = image_path
            x1, y1, x2, y2 = [int(value) for value in bbox]
            width = max(0, x2 - x1)
            height = max(0, y2 - y1)
            if float(confidence) < 0.3 or min(width, height) < 22:
                return "reject", 0.12, ("low_detector_score",)
            if float(confidence) < 0.58 or min(width, height) < 34:
                return "review", 0.54, ("review_confidence",)
            return "clean", 0.92, ()

        def detect_faces_for_image(self, image_path, *, include_tiny_faces=False):
            _ = include_tiny_faces
            faces = []
            for record in self.records:
                if str(record.image_path) != str(image_path):
                    continue
                faces.append(
                    SimpleNamespace(
                        image_path=str(image_path),
                        bbox=tuple(record.face_bbox),
                        confidence=float(record.face_confidence),
                    )
                )
            return faces

        def load_image_faces(self, image_path, *, include_tiny_faces=True):
            return [
                record
                for record in self.load_indexed_faces(limit=1000, include_tiny_faces=include_tiny_faces)
                if str(record.image_path) == str(image_path)
            ]

        def save_image_faces(self, image_path, face_inputs, *, preserve_labels=True):
            image_path = str(image_path)
            prior = {
                tuple(record.face_bbox): record
                for record in self.records
                if str(record.image_path) == image_path
            }
            self.records = [record for record in self.records if str(record.image_path) != image_path]
            for face_index, face_input in enumerate(list(face_inputs or [])):
                if isinstance(face_input, EditableFaceInput):
                    bbox = tuple(face_input.bbox)
                    confidence = float(face_input.confidence)
                else:
                    bbox = tuple(getattr(face_input, "bbox", (0, 0, 0, 0)))
                    confidence = float(getattr(face_input, "confidence", 1.0))
                previous = prior.get(bbox)
                status, score, reasons = self.assess_face_bbox(image_path, bbox, confidence=confidence)
                self.records.append(
                    IndexedFaceRecord(
                        image_path=image_path,
                        face_index=int(face_index),
                        face_bbox=bbox,
                        face_confidence=confidence,
                        embedding=np.zeros((4,), dtype=np.float32),
                        person_name=str(previous.person_name if preserve_labels and previous is not None else ""),
                        label_confidence=float(previous.label_confidence if preserve_labels and previous is not None else 0.0),
                        quality_status=str(previous.quality_status if previous is not None else status),
                        quality_score=float(previous.quality_score if previous is not None else score),
                        quality_reasons=tuple(previous.quality_reasons if previous is not None else reasons),
                    )
                )
            return self.load_image_faces(image_path, include_tiny_faces=True)

    @staticmethod
    def _face_record(
        path: str,
        face_index: int,
        *,
        person_name: str = "",
        confidence: float = 0.97,
        bbox: tuple[int, int, int, int] = (10, 12, 42, 54),
        quality_status: str | None = None,
        quality_reasons: tuple[str, ...] | None = None,
    ) -> IndexedFaceRecord:
        x1, y1, x2, y2 = [int(value) for value in bbox]
        width = max(0, x2 - x1)
        height = max(0, y2 - y1)
        resolved_status = quality_status
        resolved_reasons = tuple(quality_reasons or ())
        if resolved_status is None:
            if float(confidence) < 0.3 or min(width, height) < 22:
                resolved_status = "reject"
                resolved_reasons = ("low_detector_score",)
            elif float(confidence) < 0.58 or min(width, height) < 34:
                resolved_status = "review"
                resolved_reasons = ("review_confidence",)
            else:
                resolved_status = "clean"
        return IndexedFaceRecord(
            image_path=path,
            face_index=face_index,
            face_bbox=bbox,
            face_confidence=confidence,
            embedding=np.zeros((4,), dtype=np.float32),
            person_name=person_name,
            label_confidence=0.0,
            quality_status=str(resolved_status or "clean"),
            quality_score=1.0 if str(resolved_status or "clean") == "clean" else 0.5,
            quality_reasons=resolved_reasons,
        )

    @staticmethod
    def _person_profile(
        name: str,
        *,
        visible_face_count: int,
        notes: str = "profile note",
        similarity_threshold: float = 0.72,
        example_count: int = 2,
        labeled_count: int = 3,
        cover_image_path: str = "",
        cover_face_index: int = 0,
        favorite: bool = False,
        birth_date: str = "",
        hidden: bool = False,
    ) -> PersonProfile:
        return PersonProfile(
            person_name=name,
            similarity_threshold=float(similarity_threshold),
            example_count=int(example_count),
            labeled_count=int(labeled_count),
            visible_face_count=visible_face_count,
            notes=notes,
            tags=("friend",),
            cover_image_path=str(cover_image_path),
            cover_face_index=int(cover_face_index),
            favorite=bool(favorite),
            birth_date=str(birth_date),
            hidden=bool(hidden),
        )

    def test_cluster_pane_exposes_selected_cluster_target(self):
        pane = ClusterPane()
        pane.update_clusters({"clip::graph": {3: ["a.jpg", "b.jpg"]}})
        pane.on_cluster_selected(pane._grid_model.index(0, 0))
        target = pane.current_selection_target()
        self.assertIsNotNone(target)
        self.assertEqual("cluster", target.kind)
        self.assertEqual(("a.jpg", "b.jpg"), target.paths)

    def test_cluster_pane_select_default_cluster_uses_first_sorted_cluster(self):
        pane = ClusterPane()
        pane.update_clusters(
            {
                "dino::cosine-kmeans": {2: ["b.jpg"], 5: ["e.jpg"]},
                "clip::graph": {7: ["g.jpg"], 3: ["c.jpg", "d.jpg"]},
            }
        )
        selection = pane.select_default_cluster()
        target = pane.current_selection_target()
        self.assertEqual(("clip::graph", 3), selection)
        self.assertIsNotNone(target)
        self.assertEqual(("c.jpg", "d.jpg"), target.paths)

    def test_cluster_pane_renders_cluster_tag_summary(self):
        pane = ClusterPane()
        pane.update_clusters(
            {"clip::graph": {3: ["a.jpg", "b.jpg"]}},
            cluster_summaries={
                "clip::graph": {
                    3: ClusterTagSummary(
                        tag_counts={"selfie": 2},
                        top_tags=(("selfie", 2),),
                        tagged_image_count=2,
                        unique_tag_count=1,
                    )
                }
            },
        )
        pane.on_cluster_selected(pane._grid_model.index(0, 0))
        tooltip = pane._grid_model.data(pane._grid_model.index(0, 0), Qt.ItemDataRole.ToolTipRole)
        self.assertIn("selfie (2)", pane.summary_label.text())
        self.assertIn("Tagged images: 2/2", tooltip)

    def test_cluster_pane_advanced_mode_shows_cluster_basis_for_selected_cluster(self):
        pane = ClusterPane()
        pane.set_basic_mode(False)
        pane.update_clusters(
            {"clip::graph": {3: ["a.jpg", "b.jpg"]}},
            cluster_summaries={
                "clip::graph": {
                    3: ClusterTagSummary(
                        tag_counts={"selfie": 2},
                        top_tags=(("selfie", 2),),
                        tagged_image_count=2,
                        unique_tag_count=1,
                    )
                }
            },
            cluster_explanations={
                "clip::graph": {
                    3: ClusterExplanation(
                        cluster_id=3,
                        cluster_size=2,
                        is_outlier=False,
                        cohesion_mean=0.91,
                        cohesion_median=0.92,
                        cohesion_min=0.87,
                        cohesion_max=0.95,
                        nearest_cluster_id=4,
                        nearest_cluster_similarity=0.63,
                        separation_margin=0.28,
                        representative_images_note="Preview list is already ordered by centroid similarity.",
                        cluster_quality_score=0.81,
                        similarity_mode="semantic",
                        similarity_space="semantic projection | input 768 -> PCA 50 | normalized",
                        input_dimension=768,
                        prepared_dimension=50,
                        pca_components=50,
                    )
                }
            },
        )
        pane.on_cluster_selected(pane._grid_model.index(0, 0))
        self.assertFalse(pane.basis_label.isHidden())
        self.assertIn("Size: 2 image(s)", pane.basis_label.text())
        self.assertIn("Similarity Space: semantic projection | input 768 -> PCA 50 | normalized", pane.basis_label.text())
        self.assertIn("Nearest Competing Cluster: 4 @ 0.630", pane.basis_label.text())
        self.assertIn("Tag Evidence: Top tags: selfie (2)", pane.basis_label.text())

    def test_cluster_pane_advanced_mode_shows_cluster_meaning_for_selected_cluster(self):
        pane = ClusterPane()
        pane.set_basic_mode(False)
        pane.update_clusters(
            {"clip::semantic::graph": {3: ["beach_trip.jpg", "ocean_day.jpg"]}},
            cluster_summaries={
                "clip::semantic::graph": {
                    3: ClusterTagSummary(
                        tag_counts={"beach": 2},
                        top_tags=(("beach", 2),),
                        tagged_image_count=2,
                        unique_tag_count=1,
                    )
                }
            },
            cluster_explanations={
                "clip::semantic::graph": {
                    3: ClusterExplanation(
                        cluster_id=3,
                        cluster_size=2,
                        is_outlier=False,
                        cohesion_mean=0.91,
                        cohesion_min=0.86,
                        separation_margin=0.22,
                        representative_images_note="Preview list is already ordered by centroid similarity.",
                        member_cohesion_scores=(0.94, 0.86),
                    )
                }
            },
            cluster_meanings={
                "clip::semantic::graph": {
                    3: ClusterMeaning(
                        cluster_id=3,
                        labels=(
                            ClusterMeaningLabel("beach", "a photo of a beach", 0.34),
                            ClusterMeaningLabel("ocean", "a photo of the ocean", 0.31),
                            ClusterMeaningLabel("sunset", "a photo of a sunset", 0.22),
                        ),
                        confidence="High",
                        explanation_model="clip",
                        image_count_used=2,
                        filename_terms=(("trip", 1),),
                        status="ok",
                    )
                }
            },
        )

        pane.on_cluster_selected(pane._grid_model.index(0, 0))

        self.assertFalse(pane.meaning_label.isHidden())
        self.assertIn("Cluster Verdict: Strong visual theme", pane.meaning_label.text())
        self.assertIn("Likely About: beach / ocean / sunset", pane.meaning_label.text())
        self.assertIn("Confidence: High", pane.meaning_label.text())
        self.assertIn("Why Clustered: CLIP image-feature vectors are close to this cluster center", pane.meaning_label.text())
        self.assertIn("Why Named: CLIP matched representative images to text prompts", pane.meaning_label.text())
        self.assertIn("Suggested Action: Safe to move/copy as a group; review before delete", pane.meaning_label.text())
        self.assertIn("Source: Clustered by CLIP; Named by CLIP native text-image model", pane.meaning_label.text())
        self.assertIn("Tags: beach (2)", pane.meaning_label.text())
        self.assertIn("Reliability: High", pane.meaning_label.text())
        self.assertFalse(pane.shape_widget.isHidden())
        self.assertEqual((0.94, 0.86), pane.shape_widget._scores)

    def test_cluster_pane_dino_meaning_identifies_clip_as_sidecar(self):
        pane = ClusterPane()
        pane.set_basic_mode(False)
        pane.update_clusters(
            {"dino::semantic::hdbscan": {5: ["dog_park.jpg", "dog_run.jpg"]}},
            cluster_explanations={
                "dino::semantic::hdbscan": {
                    5: ClusterExplanation(
                        cluster_id=5,
                        cluster_size=2,
                        is_outlier=False,
                        cohesion_mean=0.88,
                        cohesion_min=0.74,
                        separation_margin=0.18,
                        representative_images_note="Preview list is already ordered by centroid similarity.",
                        member_cohesion_scores=(0.91, 0.74),
                    )
                }
            },
            cluster_meanings={
                "dino::semantic::hdbscan": {
                    5: ClusterMeaning(
                        cluster_id=5,
                        labels=(ClusterMeaningLabel("animals", "a photo of animals or pets", 0.29),),
                        confidence="Medium",
                        explanation_model="clip",
                        image_count_used=2,
                        filename_terms=(("dog", 2),),
                        status="ok",
                    )
                }
            },
        )

        pane.on_cluster_selected(pane._grid_model.index(0, 0))

        self.assertIn("Likely About: animals", pane.meaning_label.text())
        self.assertIn("Why Clustered: DINO image-feature vectors are close to this cluster center", pane.meaning_label.text())
        self.assertIn("Why Named: CLIP matched representative images to text prompts", pane.meaning_label.text())
        self.assertIn("Source: Clustered by DINO; Named by CLIP sidecar", pane.meaning_label.text())
        self.assertNotIn("Named by DINO", pane.meaning_label.text())
        self.assertEqual((0.91, 0.74), pane.shape_widget._scores)

    def test_cluster_pane_missing_meaning_uses_safe_unavailable_explanation(self):
        pane = ClusterPane()
        pane.set_basic_mode(False)
        pane.update_clusters(
            {"convnext::cosine::graph": {2: ["a.jpg", "b.jpg"]}},
            cluster_summaries={
                "convnext::cosine::graph": {
                    2: ClusterTagSummary(
                        tag_counts={"travel": 2},
                        top_tags=(("travel", 2),),
                        tagged_image_count=2,
                        unique_tag_count=1,
                    )
                }
            },
            cluster_explanations={
                "convnext::cosine::graph": {
                    2: ClusterExplanation(
                        cluster_id=2,
                        cluster_size=2,
                        is_outlier=False,
                        cohesion_mean=0.72,
                        cohesion_min=0.49,
                        separation_margin=0.03,
                        representative_images_note="Preview list is already ordered by centroid similarity.",
                        member_cohesion_scores=(0.72, 0.49),
                    )
                }
            },
        )

        pane.on_cluster_selected(pane._grid_model.index(0, 0))

        self.assertIn("Likely About: Unavailable", pane.meaning_label.text())
        self.assertIn("Why Clustered: ConvNeXt image-feature vectors are close to this cluster center", pane.meaning_label.text())
        self.assertIn("Why Named: Text naming unavailable", pane.meaning_label.text())
        self.assertIn("cannot name this cluster from text-model evidence", pane.meaning_label.text())
        self.assertIn("Source: Clustered by ConvNeXt; text naming unavailable", pane.meaning_label.text())
        self.assertIn("Suggested Action: Use as a starting point only", pane.meaning_label.text())

    def test_cluster_pane_advanced_basis_reports_similarity_sibling_difference(self):
        pane = ClusterPane()
        pane.set_basic_mode(False)
        pane.update_clusters(
            {
                "clip::semantic::hdbscan": {3: ["a.jpg", "b.jpg", "c.jpg"]},
                "clip::cosine::hdbscan": {7: ["a.jpg", "b.jpg", "d.jpg"], 8: ["e.jpg"]},
            },
            cluster_explanations={
                "clip::semantic::hdbscan": {
                    3: ClusterExplanation(
                        cluster_id=3,
                        cluster_size=3,
                        is_outlier=False,
                        representative_images_note="Preview list is already ordered by centroid similarity.",
                        similarity_mode="semantic",
                        similarity_space="semantic projection | input 768 -> PCA 50 | normalized",
                    )
                }
            },
        )

        semantic_column = pane._grid_model._columns.index("clip::semantic::hdbscan")
        pane.on_cluster_selected(pane._grid_model.index(0, semantic_column))

        self.assertIn("Similarity Difference: compared with clip::cosine::hdbscan", pane.basis_label.text())
        self.assertIn("Best Match: Cluster 7 | overlap 0.500", pane.basis_label.text())
        self.assertIn("Moved Out: 1 image(s)", pane.basis_label.text())
        self.assertIn("Moved In: 1 image(s)", pane.basis_label.text())

    def test_cluster_pane_basic_mode_hides_cluster_basis_panel(self):
        pane = ClusterPane()
        pane.set_basic_mode(True)
        self.assertTrue(pane.details_scroll.isHidden())
        self.assertTrue(pane.meaning_section_label.isHidden())
        self.assertTrue(pane.meaning_label.isHidden())
        self.assertTrue(pane.shape_section_label.isHidden())
        self.assertTrue(pane.shape_widget.isHidden())
        self.assertTrue(pane.basis_section_label.isHidden())
        self.assertTrue(pane.basis_label.isHidden())
        pane.set_basic_mode(False)
        self.assertFalse(pane.details_scroll.isHidden())
        self.assertGreaterEqual(pane.cluster_table.minimumHeight(), 360)

    def test_cluster_pane_reset_restores_cluster_basis_empty_state(self):
        pane = ClusterPane()
        pane.set_basic_mode(False)
        pane.update_clusters(
            {"clip::graph": {3: ["a.jpg", "b.jpg"]}},
            cluster_explanations={
                "clip::graph": {
                    3: ClusterExplanation(
                        cluster_id=3,
                        cluster_size=2,
                        is_outlier=False,
                        representative_images_note="Preview list is already ordered by centroid similarity.",
                        member_cohesion_scores=(0.9, 0.8),
                    )
                }
            },
            cluster_meanings={
                "clip::graph": {
                    3: ClusterMeaning(
                        cluster_id=3,
                        labels=(ClusterMeaningLabel("beach", "a photo of a beach", 0.3),),
                        confidence="Medium",
                        explanation_model="clip",
                        image_count_used=2,
                        status="ok",
                    )
                }
            },
        )
        pane.on_cluster_selected(pane._grid_model.index(0, 0))
        self.assertNotEqual(CLUSTER_BASIS_EMPTY_TEXT, pane.basis_label.text())
        self.assertIn("Likely About: beach", pane.meaning_label.text())
        self.assertEqual((0.9, 0.8), pane.shape_widget._scores)
        pane.update_clusters({})
        self.assertEqual(CLUSTER_BASIS_EMPTY_TEXT, pane.basis_label.text())
        self.assertEqual((), pane.shape_widget._scores)
        self.assertIn("Select a cluster", pane.meaning_label.text())

    def test_cluster_pane_hover_popup_opens_on_hover_and_hides_on_empty_target(self):
        with TemporaryDirectory() as tmp:
            paths = []
            for index in range(3):
                image_path = Path(tmp) / f"hover_{index}.png"
                Image.new("RGB", (48, 48), (index * 50, 40, 80)).save(image_path)
                paths.append(str(image_path))
            pane = ClusterPane()
            pane.resize(760, 520)
            pane.update_clusters({"clip::graph": {3: paths}})
            pane.show()
            APP.processEvents()

            index = pane._grid_model.index(0, 0)
            cell_rect = pane.cluster_table.visualRect(index)
            QTest.mouseMove(pane.cluster_table.viewport(), cell_rect.center())
            if not self._wait_until(lambda: pane._hover_popup.isVisible(), timeout_s=1.0):
                pane._update_hover_popup_for_index(index)

            self.assertTrue(self._wait_until(lambda: pane._hover_popup.isVisible()))
            self.assertIn("clip::graph | Cluster 3", pane._hover_popup.title_label.text())

            pane._update_hover_popup_for_index(QModelIndex())
            APP.processEvents()

            self.assertFalse(pane._hover_popup.isVisible())
            pane.close()

    def test_cluster_pane_hover_popup_updates_existing_widget_for_new_cluster(self):
        with TemporaryDirectory() as tmp:
            image_a = Path(tmp) / "a.png"
            image_b = Path(tmp) / "b.png"
            Image.new("RGB", (48, 48), (200, 20, 30)).save(image_a)
            Image.new("RGB", (48, 48), (20, 100, 220)).save(image_b)
            pane = ClusterPane()
            pane.resize(760, 520)
            pane.update_clusters({"clip::graph": {3: [str(image_a)], 4: [str(image_b)]}})
            pane.show()
            APP.processEvents()

            first_index = pane._grid_model.index(0, 0)
            second_index = pane._grid_model.index(1, 0)
            popup = pane._hover_popup

            pane._update_hover_popup_for_index(first_index)
            APP.processEvents()
            self.assertTrue(self._wait_until(lambda: popup.isVisible() and "Cluster 3" in popup.title_label.text()))

            pane._update_hover_popup_for_index(second_index)
            APP.processEvents()
            self.assertTrue(self._wait_until(lambda: "Cluster 4" in popup.title_label.text()))
            self.assertIs(popup, pane._hover_popup)
            self.assertTrue(popup.isVisible())
            pane.close()

    def test_cluster_pane_hover_popup_discards_stale_async_preview_results(self):
        with TemporaryDirectory() as tmp:
            slow_image = Path(tmp) / "slow.png"
            Image.new("RGB", (48, 48), (255, 0, 0)).save(slow_image)
            fast_paths = []
            for index in range(11):
                image_path = Path(tmp) / f"fast_{index}.png"
                Image.new("RGB", (48, 48), (0, 20 + index, 255)).save(image_path)
                fast_paths.append(str(image_path))
            pane = ClusterPane()
            pane.resize(760, 520)
            pane.update_clusters({"clip::graph": {3: [str(slow_image)], 4: fast_paths}})
            pane.show()
            APP.processEvents()

            first_index = pane._grid_model.index(0, 0)
            second_index = pane._grid_model.index(1, 0)

            def _fake_sheet(_service, image_paths, size, *, max_items=9, columns=3, cancel_check=None):
                _ = (max_items, columns, cancel_check)
                width = int(size.width()) if hasattr(size, "width") else int(size)
                height = int(size.height()) if hasattr(size, "height") else int(size)
                if Path(image_paths[0]).name.startswith("slow"):
                    sleep(0.2)
                    color = Qt.GlobalColor.red
                else:
                    color = Qt.GlobalColor.blue
                image = QImage(width, height, QImage.Format.Format_RGB32)
                image.fill(color)
                return image

            with patch("ui.cluster_pane.ThumbnailService.build_contact_sheet_qimage", new=_fake_sheet):
                pane._update_hover_popup_for_index(first_index)
                APP.processEvents()
                pane._update_hover_popup_for_index(second_index)
                APP.processEvents()
                self.assertTrue(
                    self._wait_until(
                        lambda: (
                            pane._hover_popup.image_label.pixmap() is not None
                            and not pane._hover_popup.image_label.pixmap().isNull()
                            and pane._hover_popup.image_label.pixmap().toImage().pixelColor(5, 5).blue() == 255
                        )
                    )
                )

            self.assertIn("Cluster 4", pane._hover_popup.title_label.text())
            self.assertIn("+2 more", pane._hover_popup.note_label.text())
            pane.close()

    def test_cluster_pane_hover_popup_hides_on_viewport_leave(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "leave.png"
            Image.new("RGB", (48, 48), (140, 90, 30)).save(image_path)
            pane = ClusterPane()
            pane.resize(760, 520)
            pane.update_clusters({"clip::graph": {3: [str(image_path)]}})
            pane.show()
            APP.processEvents()

            index = pane._grid_model.index(0, 0)
            pane._update_hover_popup_for_index(index)
            APP.processEvents()
            self.assertTrue(self._wait_until(lambda: pane._hover_popup.isVisible()))

            APP.sendEvent(pane.cluster_table.viewport(), QEvent(QEvent.Type.Leave))
            APP.processEvents()

            self.assertFalse(pane._hover_popup.isVisible())
            pane.close()

    def test_gallery_select_current_group_checks_target_paths(self):
        pane = GalleryPane()
        pane.images = ["a.jpg", "b.jpg", "c.jpg"]
        pane.model.set_images(pane.images)
        pane.set_action_target_provider(
            lambda: SelectionTarget(paths=("b.jpg", "c.jpg"), kind="cluster", label="Cluster 9")
        )
        pane.select_current_group()
        self.assertEqual(["b.jpg", "c.jpg"], pane.model.checked_paths())
        pane.select_current_group()
        self.assertEqual([], pane.model.checked_paths())

    def test_delete_selection_requires_confirmation(self):
        pane = GalleryPane()
        pane.images = ["a.jpg", "b.jpg"]
        pane.model.set_images(pane.images)
        pane.set_action_target_provider(
            lambda: SelectionTarget(paths=("a.jpg", "b.jpg"), kind="cluster", label="Cluster 4")
        )
        with patch("ui.gallery_pane.confirmBox", return_value=False), patch.object(pane, "_start_action_job") as start_job:
            pane.slotDeleteSelect()
        start_job.assert_not_called()

    def test_gallery_copy_selected_paths_uses_current_target(self):
        pane = GalleryPane()
        pane.images = ["a.jpg", "b.jpg", "c.jpg"]
        pane.model.set_images(pane.images)
        pane.set_action_target_provider(
            lambda: SelectionTarget(paths=("b.jpg", "c.jpg"), kind="cluster", label="Cluster 9")
        )

        pane.slotCopySelectedPaths()

        self.assertEqual("b.jpg\nc.jpg", APP.clipboard().text())
        self.assertIn("Copied 2 path(s)", pane.status_label.text())
        pane.close()

    def test_gallery_selected_only_tag_target_never_falls_back_to_cluster(self):
        pane = GalleryPane()
        pane.images = ["a.jpg", "b.jpg", "c.jpg"]
        pane.model.set_images(pane.images)
        pane.set_action_target_provider(
            lambda: SelectionTarget(paths=("b.jpg", "c.jpg"), kind="cluster", label="Cluster 9")
        )

        self.assertEqual(("b.jpg", "c.jpg"), pane._selection_for_actions().paths)
        self.assertIsNone(pane._selection_for_explicit_gallery_actions())
        self.assertIn("will not use the current cluster", pane.status_label.text())

        selection = pane.list_view.selectionModel()
        selection.select(
            pane.model.index(0, 0),
            QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
        )
        target = pane._selection_for_explicit_gallery_actions()
        self.assertIsNotNone(target)
        self.assertEqual(("a.jpg",), target.paths)
        self.assertEqual("selected_gallery_images", target.kind)

        selection.clearSelection()
        pane.model.setData(pane.model.index(2, 0), Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole)
        target = pane._selection_for_explicit_gallery_actions()
        self.assertIsNotNone(target)
        self.assertEqual(("c.jpg",), target.paths)
        self.assertEqual("selected_images", target.kind)
        pane.close()

    def test_gallery_selected_only_tag_button_uses_explicit_target(self):
        pane = GalleryPane()
        target = SelectionTarget(paths=("a.jpg",), kind="selected_gallery_images", label="Selected Gallery Photos (1)")
        with patch.object(pane, "_selection_for_explicit_gallery_actions", return_value=target) as selection, patch.object(
            pane, "_edit_tags_for_target"
        ) as edit_tags:
            pane.slotEditSelectedOnlyTags()

        selection.assert_called_once()
        edit_tags.assert_called_once_with(target)
        pane.close()

    def test_gallery_action_bar_uses_compact_menu_groups(self):
        pane = GalleryPane()

        self.assertEqual(
            ["Write EXIF to current target (0)", "Tag current target (0)", "Export sidecars", "Import sidecars"],
            [action.text() for action in pane.metadata_menu.actions()],
        )
        self.assertEqual(
            ["Copy current target (0)", "Move current target (0)", "Move current target to ClusterLens Trash (0)"],
            [action.text() for action in pane.file_ops_menu.actions()],
        )
        self.assertEqual(
            ["Copy paths (0)", "Export paths (0)", "Retry failed thumbnails (0)"],
            [action.text() for action in pane.more_menu.actions()],
        )
        self.assertFalse(hasattr(pane, "tags_button"))
        self.assertFalse(hasattr(pane, "copy_button"))
        self.assertEqual("Metadata", pane.metadata_menu.title())
        self.assertEqual("File actions", pane.file_ops_menu.title())
        self.assertEqual("Utilities", pane.more_menu.title())
        self.assertTrue(pane.actions_menu_button.toolTip())
        pane.close()

    def test_gallery_export_selected_paths_writes_text_file(self):
        pane = GalleryPane()
        pane.images = ["a.jpg", "b.jpg"]
        pane.model.set_images(pane.images)
        pane.set_action_target_provider(
            lambda: SelectionTarget(paths=("a.jpg", "b.jpg"), kind="cluster", label="Cluster 2")
        )
        with TemporaryDirectory() as tmp:
            export_path = Path(tmp) / "paths.txt"
            with patch(
                "ui.gallery_pane.QFileDialog.getSaveFileName",
                return_value=(str(export_path), "Text Files (*.txt)"),
            ), patch("ui.gallery_pane.infoBox") as info_box:
                pane.slotExportSelectedPaths()
                self.assertEqual("a.jpg\nb.jpg\n", export_path.read_text(encoding="utf-8"))
                self.assertIn("Exported 2 path(s)", pane.status_label.text())
                info_box.assert_called_once()
        pane.close()

    def test_gallery_metadata_sidecar_export_import_actions_round_trip_without_source_writes(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "photo.jpg"
            Image.new("RGB", (32, 32), (20, 40, 60)).save(image_path)
            image_key = str(image_path)
            tag_service = ImageTagService(db_path=root / "tags.sqlite3")
            tag_service.apply_tag_edit([image_key], add_tags=["family"], mirror_to_exif=False)
            before_bytes = image_path.read_bytes()
            before_mtime = image_path.stat().st_mtime_ns
            sidecar_dir = root / "sidecars"
            sidecar_path = sidecar_dir / f"{image_path.name}.clusterlens.json"
            pane = GalleryPane()
            pane.image_tag_service = tag_service
            pane.images = [image_key]
            pane.model.set_images(pane.images)
            pane.set_action_target_provider(
                lambda: SelectionTarget(paths=(image_key,), kind="gallery", label="Visible Images")
            )

            def _run_sync(_label, fn, on_completed):
                on_completed(fn(lambda *_args: None, lambda: False))

            with patch(
                "ui.gallery_pane.QFileDialog.getExistingDirectory",
                return_value=str(sidecar_dir),
            ), patch("ui.gallery_pane.infoBox"), patch.object(pane, "_start_action_job", side_effect=_run_sync):
                pane.slotExportMetadataSidecars()

            self.assertTrue(sidecar_path.exists())
            self.assertIn("Exported 1 metadata sidecar", pane.status_label.text())
            tag_service.delete_tag("family")
            self.assertEqual((), tag_service.load_tags_for_paths([image_key], import_missing_exif=False)[image_key])

            with patch(
                "ui.gallery_pane.QFileDialog.getOpenFileNames",
                return_value=([str(sidecar_path)], "ClusterLens Sidecars (*.clusterlens.json *.json)"),
            ), patch("ui.gallery_pane.infoBox"), patch.object(pane, "_start_action_job", side_effect=_run_sync):
                pane.slotImportMetadataSidecars()

            self.assertEqual(("family",), tag_service.load_tags_for_paths([image_key], import_missing_exif=False)[image_key])
            self.assertIn("Imported 1 metadata sidecar", pane.status_label.text())
            self.assertEqual(before_bytes, image_path.read_bytes())
            self.assertEqual(before_mtime, image_path.stat().st_mtime_ns)
            pane.close()

    def test_main_window_starts_with_faces_workspace_available(self):
        window = ClusterGalleryApp()
        window.show()
        APP.processEvents()
        self.assertTrue(window.feature_modules["clustering"].enabled)
        self.assertFalse(window.feature_modules["face_search"].enabled)
        self.assertIsNotNone(window.clustering_pane)
        self.assertIsNotNone(window.gallery_pane)
        self.assertIsNotNone(window.faces_pane)
        self.assertIsNotNone(window.names_pane)
        self.assertIsNotNone(window.footer_bar)
        self.assertEqual("basic", window._clustering_mode)
        self.assertEqual("clustering", window._active_workspace)
        self.assertTrue(window.basic_mode_button.isChecked())
        self.assertFalse(window.advanced_mode_button.isChecked())
        self.assertTrue(window.clustering_workspace_button.isChecked())
        self.assertFalse(window.faces_workspace_button.isChecked())
        self.assertFalse(window.names_workspace_button.isChecked())
        self.assertTrue(window.source_pane.basic_action_section.isVisible())
        self.assertTrue(window.source_pane.basic_run_button.isVisible())
        self.assertFalse(window.source_pane.basic_cancel_button.isVisible())
        self.assertFalse(window.clustering_pane.isVisible())
        self.assertFalse(hasattr(window.gallery_pane, "tags_button"))
        self.assertFalse(window.gallery_pane.metadata_menu.menuAction().isVisible())
        self.assertFalse(window.gallery_pane.selected_tags_button.isVisible())
        self.assertEqual("Compare", window.details_toggle.text())
        self.assertFalse(window.source_toggle.isVisible())
        self.assertFalse(window.controls_toggle.isVisible())
        self.assertFalse(window.details_toggle.isVisible())
        sizes = window.main_splitter.sizes()
        self.assertEqual(2, len(sizes))
        self.assertGreaterEqual(sizes[0], 220)
        self.assertGreater(sizes[1], sizes[0])
        self.assertEqual(["All Faces", "Folder Review", "Face Search", "Identities"], window.faces_pane.tab_labels())
        window.close()

    def test_names_workspace_is_a_top_level_page_with_one_names_sidebar(self):
        window = ClusterGalleryApp()
        refresh_calls: list[bool] = []
        window.names_pane.refresh_names = lambda: refresh_calls.append(True)
        window.show()
        APP.processEvents()

        window.set_active_workspace("names")
        APP.processEvents()

        self.assertEqual("names", window._active_workspace)
        self.assertTrue(window.names_workspace_button.isChecked())
        self.assertFalse(window.clustering_workspace_button.isChecked())
        self.assertFalse(window.faces_workspace_button.isChecked())
        self.assertIs(window.workspace_stack.currentWidget(), window.names_pane)
        self.assertEqual([True], refresh_calls)
        self.assertFalse(window.source_pane.isVisible())
        self.assertFalse(window.basic_mode_button.isVisible())
        self.assertFalse(window.advanced_mode_button.isVisible())
        self.assertFalse(window.source_toggle.isVisible())
        self.assertFalse(window.controls_toggle.isVisible())
        self.assertFalse(window.details_toggle.isVisible())
        self.assertEqual("Feature: Names", window.feature_label.text())
        window.close()

    def test_names_pane_lists_durable_names_and_unique_photo_paths(self):
        class _NameService:
            db_path = "/tmp/names-test.sqlite3"

            def __init__(self):
                self.recovery_calls = 0

            def recover_legacy_manual_face_labels(self):
                self.recovery_calls += 1
                return SimpleNamespace(promoted_count=0, duplicate_count=0, conflict_count=0)

            def list_named_photo_summaries(self):
                return [
                    NamedPhotoSummary("Alice", face_count=3, photo_count=2),
                    NamedPhotoSummary("Bob", face_count=1, photo_count=1),
                ]

            def list_named_photo_paths(self, person_name):
                return {
                    "Alice": ["/photos/alice-a.jpg", "/photos/alice-b.jpg"],
                    "Bob": ["/photos/bob.jpg"],
                }.get(str(person_name), [])

        service = _NameService()
        pane = NamesPane(lambda: service)
        pane.show()
        try:
            pane.refresh_names()
            self.assertTrue(self._wait_until(lambda: self._list_view_count(pane.names_list) == 2))
            self.assertTrue(self._wait_until(lambda: pane.gallery.images == ["/photos/alice-a.jpg", "/photos/alice-b.jpg"]))
            self.assertEqual(1, service.recovery_calls)
            self.assertIn("Alice", self._list_view_text(pane.names_list, 0))
            self.assertEqual("Alice · 2 unique photo(s)", pane.status_label.text())
            pane.search_field.setText("Bob")
            APP.processEvents()
            self.assertTrue(self._wait_until(lambda: pane.gallery.images == ["/photos/bob.jpg"]))
            self.assertEqual("Bob · 1 unique photo(s)", pane.status_label.text())
        finally:
            pane.shutdown_jobs(timeout_ms=500)
            pane.close()

    def test_names_pane_selected_image_actions_are_scoped_to_the_active_name(self):
        class _NameService:
            db_path = "/tmp/names-actions-test.sqlite3"

            def __init__(self):
                self.calls: list[tuple] = []

            def recover_legacy_manual_face_labels(self):
                return SimpleNamespace(promoted_count=0, duplicate_count=0, conflict_count=0)

            def list_named_photo_summaries(self):
                return [
                    NamedPhotoSummary("Alice", face_count=1, photo_count=1),
                    NamedPhotoSummary("Bob", face_count=1, photo_count=1),
                ]

            def list_named_photo_paths(self, person_name):
                return {
                    "Alice": ["/photos/mixed-a.jpg", "/photos/mixed-b.jpg"],
                    "Bob": ["/photos/mixed-a.jpg", "/photos/mixed-b.jpg"],
                }.get(str(person_name), [])

            def label_unlabeled_faces_in_images(self, person_name, image_paths):
                self.calls.append(("name", person_name, tuple(image_paths)))
                return 1

            def rename_labeled_faces_in_images(self, source_name, target_name, image_paths):
                self.calls.append(("rename", source_name, target_name, tuple(image_paths)))
                return 1

            def unlabel_labeled_faces_in_images(self, person_name, image_paths):
                self.calls.append(("unlabel", person_name, tuple(image_paths)))
                return 1

        service = _NameService()
        pane = NamesPane(lambda: service)
        changes: list[bool] = []
        pane.face_labels_changed.connect(lambda: changes.append(True))
        pane.show()
        try:
            pane.refresh_names()
            self.assertTrue(
                self._wait_until(lambda: pane.gallery.images == ["/photos/mixed-a.jpg", "/photos/mixed-b.jpg"])
            )

            def select_photos() -> None:
                selection_model = pane.gallery.list_view.selectionModel()
                first = pane.gallery.model.index(0, 0)
                second = pane.gallery.model.index(1, 0)
                selection_model.select(
                    first,
                    QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows,
                )
                selection_model.select(
                    second,
                    QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
                )
                APP.processEvents()

            select_photos()
            actions = {item[0]: item for item in pane._names_context_menu_actions("/photos/mixed-a.jpg")}
            self.assertTrue(actions["Rename Selected…"][3])
            self.assertTrue(actions["Unlabel Selected"][3])
            self.assertNotIn("Name Selected…", actions)
            self.assertIsNotNone(pane.gallery.context_menu_action_provider)
            pane.set_read_only_mode(True)
            self.assertTrue(all(not item[3] for item in pane._names_context_menu_actions("/photos/mixed-a.jpg")))
            pane.set_read_only_mode(False)

            def choose_rename_action(*_args):
                return next(
                    action
                    for menu in pane.gallery.findChildren(QMenu)
                    for action in menu.actions()
                    if action.text() == "Rename Selected…"
                )

            with (
                patch("ui.names_pane.QInputDialog.getText", return_value=("Cara", True)),
                patch("ui.gallery_pane.QMenu.exec", side_effect=choose_rename_action),
            ):
                pane.gallery.on_context_menu(pane.gallery.list_view.visualRect(pane.gallery.model.index(0, 0)).center())
            self.assertTrue(
                self._wait_until(
                    lambda: service.calls[:1]
                    == [("rename", "Alice", "Cara", ("/photos/mixed-a.jpg", "/photos/mixed-b.jpg"))]
                )
            )
            self.assertTrue(self._wait_until(lambda: pane._mutation_job is None))
            self.assertTrue(self._wait_until(lambda: pane._refresh_job is None))

            self.assertTrue(self._wait_until(lambda: pane.names_list.currentIndex().data(ListEntryModel.PayloadRole) == "Alice"))
            self.assertTrue(
                self._wait_until(lambda: pane.gallery.images == ["/photos/mixed-a.jpg", "/photos/mixed-b.jpg"])
            )
            self.assertTrue(self._wait_until(lambda: pane._photos_job is None))
            select_photos()
            with patch("ui.names_pane.confirmBox", return_value=True):
                actions["Unlabel Selected"][2]()
            self.assertTrue(
                self._wait_until(
                    lambda: service.calls[1:2] == [("unlabel", "Alice", ("/photos/mixed-a.jpg", "/photos/mixed-b.jpg"))]
                )
            )
            self.assertTrue(self._wait_until(lambda: pane._mutation_job is None))
            self.assertEqual(2, len(changes))
        finally:
            pane.shutdown_jobs(timeout_ms=500)
            pane.close()

    def test_runtime_warmup_is_deferred_outside_clustering_workspace(self):
        progress_calls: list[object] = []
        overlay_updates: list[bool] = []
        fake_window = SimpleNamespace(
            _active_workspace="faces",
            _warmup_state="",
            footer_bar=SimpleNamespace(
                set_progress=lambda value, *_args: progress_calls.append(value),
                set_status=lambda _text: None,
            ),
            _update_metrics_overlay=lambda: overlay_updates.append(True),
        )
        ClusterGalleryApp.maybe_warm_runtime(fake_window)
        self.assertEqual("deferred", fake_window._warmup_state)
        self.assertEqual([None], progress_calls)
        self.assertEqual([True], overlay_updates)

    def test_runtime_warmup_skips_when_selected_model_requires_download(self):
        progress_calls: list[object] = []
        status_calls: list[str] = []
        overlay_updates: list[bool] = []
        fake_window = SimpleNamespace(
            _active_workspace="clustering",
            _warmup_state="",
            _warmup_cache=set(),
            _warmup_thread=None,
            settings=SimpleNamespace(allow_gpu_warmup=True),
            settings_store=SimpleNamespace(value=lambda _key, default=None, _type=None: True),
            execution_policy=SimpleNamespace(effective_mode="cuda", onnx_provider="CUDAExecutionProvider"),
            clustering_pane=SimpleNamespace(
                selected_embedding_models=lambda: ["dino"],
                onnx_checkbox=SimpleNamespace(isChecked=lambda: False),
            ),
            embedding_service=SimpleNamespace(
                model_manager=SimpleNamespace(
                    model_asset_service=SimpleNamespace(model_available_without_download=lambda _name: False)
                )
            ),
            footer_bar=SimpleNamespace(
                set_progress=lambda value, *_args: progress_calls.append(value),
                set_status=lambda text: status_calls.append(str(text)),
            ),
            _update_metrics_overlay=lambda: overlay_updates.append(True),
        )

        with patch.object(main_module, "start_job_in_thread") as start_job:
            ClusterGalleryApp.maybe_warm_runtime(fake_window)

        self.assertEqual("skipped (dino requires download)", fake_window._warmup_state)
        self.assertEqual([None], progress_calls)
        self.assertTrue(any("Warm-up skipped for dino" in text for text in status_calls))
        self.assertEqual([True], overlay_updates)
        start_job.assert_not_called()

    def test_face_review_gallery_chunk_publish_uses_nonzero_timer_delay(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        paths = [f"/photos/{index}.jpg" for index in range(120)]
        delays: list[int] = []

        with patch.object(pane.results_gallery, "append_images"), patch.object(
            pane, "_publish_results"
        ), patch("ui.search_pane.QTimer.singleShot") as single_shot:
            single_shot.side_effect = lambda delay_ms, _callback: delays.append(int(delay_ms))
            pane._publish_results_gallery_paths(paths)

        self.assertEqual([1], delays)
        pane.close()

    def test_face_tile_queue_scheduling_uses_shared_refresh_timer(self):
        with patch.object(SearchPane, "_start_face_tile_loader_threads"):
            pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        expected_delay = max(1, int(pane.settings.gallery_flush_interval_ms))

        with patch.object(pane._face_tile_refresh_timer, "start") as start_timer, patch("ui.search_pane.QTimer.singleShot") as single_shot:
            pane._schedule_selected_face_tile_loads()
            pane.face_review_results_tabs.setCurrentWidget(pane.face_detected_faces_panel)
            pane._schedule_detected_face_tile_loads()

            original_tab = pane.face_review_results_tabs.currentWidget()
            pane.face_review_results_tabs.setCurrentWidget(pane.face_results_panel)
            pane._schedule_face_result_tile_loads()
            pane.face_review_results_tabs.setCurrentWidget(original_tab)

        delays = [call.args[0] for call in start_timer.call_args_list]
        self.assertGreaterEqual(len(delays), 3)
        self.assertTrue(all(delay == expected_delay for delay in delays))
        single_shot.assert_not_called()
        pane.close()

    def test_user_flows_launcher_opens_bundled_guide_and_reports_missing_file(self):
        window = ClusterGalleryApp()
        window.show()
        APP.processEvents()
        try:
            self.assertEqual("User Flows", window.user_flows_button.text())
            self.assertTrue(window.user_flows_button.isVisible())
            guide_path = window.user_flows_guide_path()
            self.assertEqual("USER_FLOWS.md", guide_path.name)
            self.assertTrue(guide_path.exists())
            opened_paths: list[str] = []

            def _record_open(url):
                opened_paths.append(url.toLocalFile())
                return True

            with patch("main.QDesktopServices.openUrl", side_effect=_record_open):
                window.open_user_flows()
            self.assertEqual([str(guide_path)], opened_paths)
            self.assertIn("Opened user flows", window.footer_bar.status_label.text())

            missing_path = guide_path.parent / "missing-user-flows.md"
            window.user_flows_guide_path = lambda: missing_path  # type: ignore[method-assign]
            with patch("main.QDesktopServices.openUrl") as open_url, patch("main.errorBox") as error_box:
                window.open_user_flows()
            open_url.assert_not_called()
            error_box.assert_called_once()
            self.assertEqual("User flows unavailable", error_box.call_args.args[0])
            self.assertIn(str(missing_path), error_box.call_args.args[1])
        finally:
            window.close()

    def test_app_first_run_tutorial_is_dismissible_persistent_and_reopenable(self):
        window = ClusterGalleryApp()
        window.show()
        APP.processEvents()
        try:
            self.assertEqual("Tutorial", window.tutorial_button.text())
            self.assertEqual("Dismiss Tutorial", window.startup_tutorial_dismiss_button.text())
            self.assertTrue(window.startup_tutorial_panel.isVisible())
            tutorial_text = window.startup_tutorial_label.text()
            for expected in ("source folder", "clustering", "Faces", "Settings", "storage", "source images"):
                self.assertIn(expected, tutorial_text)

            window.startup_tutorial_dismiss_button.click()
            APP.processEvents()
            self.assertFalse(window.startup_tutorial_panel.isVisible())
            self.assertTrue(self._app_settings_store.value("tutorial/app_first_run_dismissed", False, bool))
        finally:
            window.close()

        second_window = ClusterGalleryApp()
        second_window.show()
        APP.processEvents()
        try:
            self.assertFalse(second_window.startup_tutorial_panel.isVisible())
            second_window.tutorial_button.click()
            APP.processEvents()
            self.assertTrue(second_window.startup_tutorial_panel.isVisible())
        finally:
            second_window.close()

    def test_toolbar_toggles_can_hide_side_panes(self):
        window = ClusterGalleryApp()
        window.show()
        APP.processEvents()
        window.set_clustering_mode("advanced")
        APP.processEvents()
        starting_gallery_width = window.gallery_pane.width()
        window.source_toggle.setChecked(False)
        window.controls_toggle.setChecked(False)
        window.details_toggle.setChecked(False)
        APP.processEvents()
        self.assertFalse(window.source_pane.isVisible())
        self.assertFalse(window.clustering_pane.isVisible())
        self.assertFalse(window.cluster_right_splitter.isVisible())
        self.assertGreater(window.gallery_pane.width(), starting_gallery_width)
        window.source_toggle.setChecked(True)
        window.controls_toggle.setChecked(True)
        window.details_toggle.setChecked(True)
        APP.processEvents()
        self.assertTrue(window.source_pane.isVisible())
        self.assertTrue(window.clustering_pane.isVisible())
        self.assertTrue(window.cluster_right_splitter.isVisible())
        window.close()

    def test_main_run_clustering_uses_tag_filtered_source_paths(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_a = root / "a.jpg"
            image_b = root / "b.jpg"
            image_c = root / "c.jpg"
            for image_path in [image_a, image_b, image_c]:
                Image.new("RGB", (24, 24), (10, 20, 30)).save(image_path)
            window = ClusterGalleryApp()
            window.clustering_pane.cluster_spinbox.setValue(2)
            window.source_pane.set_selected_directory(str(root))
            window.image_tag_service.apply_tag_edit([str(image_a), str(image_b)], add_tags=["selfie"], mirror_to_exif=False)
            window.clustering_pane.tag_filter_field.setText("selfie")
            window.clustering_pane.tag_match_combobox.setCurrentText("Any")
            with patch.object(window, "_start_clustering_request") as start_request:
                window.run_clustering()
            request = start_request.call_args.args[0]
            self.assertEqual("tag_filter", start_request.call_args.kwargs["run_origin"])
            self.assertEqual([str(image_a), str(image_b)], request.source_paths)
            window.close()

    def test_main_run_clustering_handles_tag_filter_exceptions(self):
        window = ClusterGalleryApp()
        window.source_pane.set_selected_directory("D:/broken")
        window.clustering_pane.tag_filter_field.setText("selfie")
        with patch.object(
            window.pipeline.discovery_service,
            "discover_result",
            return_value=SimpleNamespace(paths=["D:/broken/a.jpg"]),
        ), patch.object(window.image_tag_service, "select_paths_by_tags", side_effect=ValueError("bad exif image")), patch(
            "main.errorBox"
        ) as error_box, patch.object(window, "_start_clustering_request") as start_request:
            window.run_clustering()
        start_request.assert_not_called()
        error_box.assert_called_once()
        self.assertIn("bad exif image", error_box.call_args.args[1])
        self.assertTrue(window.footer_bar.status_label.text().startswith("Clustering coul"))
        window.close()

    def test_clustering_options_persist_embedding_cache_lookup_checkbox(self):
        window = ClusterGalleryApp()
        self.assertTrue(window.clustering_pane.embedding_cache_lookup_checkbox.isChecked())
        window.clustering_pane.embedding_cache_lookup_checkbox.setChecked(False)
        state = window.clustering_pane.export_state()
        self.assertFalse(state["use_embedding_cache_lookup"])
        window.clustering_pane.embedding_cache_lookup_checkbox.setChecked(True)
        window.clustering_pane.apply_state(state)
        self.assertFalse(window.clustering_pane.embedding_cache_lookup_checkbox.isChecked())
        window.close()

    def test_clustering_options_similarity_modes_are_multi_select_without_near_duplicate(self):
        window = ClusterGalleryApp()
        self.assertEqual({"semantic", "cosine"}, set(window.clustering_pane.similarity_checkboxes.keys()))
        self.assertNotIn("near_duplicate", window.clustering_pane.similarity_checkboxes)

        window.clustering_pane.apply_state({"similarity_mode": "near_duplicate"})
        self.assertEqual(["semantic"], window.clustering_pane.selected_similarity_modes())

        window.clustering_pane.apply_state({"similarity_modes": ["semantic", "cosine"]})
        state = window.clustering_pane.export_state()
        self.assertEqual(["semantic", "cosine"], state["similarity_modes"])
        window.close()

    def test_main_build_clustering_request_carries_embedding_cache_lookup_flag(self):
        window = ClusterGalleryApp()
        window.clustering_pane.apply_state({"similarity_modes": ["semantic"]})
        window.clustering_pane.embedding_cache_lookup_checkbox.setChecked(False)
        request = window._build_clustering_request()
        self.assertFalse(request.use_embedding_cache_lookup)
        self.assertEqual(["semantic"], request.similarity_modes)
        self.assertFalse(request.generate_cluster_meanings)
        window.set_clustering_mode("advanced")
        advanced_request = window._build_clustering_request()
        self.assertTrue(advanced_request.generate_cluster_meanings)
        self.assertEqual("auto", advanced_request.cluster_meaning_model)
        window.close()

    def test_main_recluster_selected_cluster_uses_selected_paths(self):
        window = ClusterGalleryApp()
        window.clustering_pane.cluster_spinbox.setValue(2)
        window.cluster_pane.update_clusters({"clip::graph": {3: ["a.jpg", "b.jpg"]}})
        window.cluster_pane.on_cluster_selected(window.cluster_pane._grid_model.index(0, 0))
        with patch.object(window, "_start_clustering_request") as start_request:
            window.recluster_selected_cluster()
        request = start_request.call_args.args[0]
        self.assertEqual("recluster", start_request.call_args.kwargs["run_origin"])
        self.assertEqual(["a.jpg", "b.jpg"], request.source_paths)
        window.close()

    def test_main_on_clustering_finished_defers_tag_refresh(self):
        window = ClusterGalleryApp()
        window.clustering_pane.set_running(True)
        window.source_pane.set_running(True)
        window._clustering_job_id = window.job_manager.register_job("Clustering")
        result = MultiBackendClusteringResult(
            clusters_by_key={"clip::graph": {3: ["a.jpg", "b.jpg"]}},
            membership_by_image={},
            metrics_by_key={"clip::graph": {"backend": "graph"}},
        )
        with patch.object(window, "_start_cluster_tag_context_refresh") as start_refresh:
            window.on_clustering_finished(result, {"image_count": 2})
        start_refresh.assert_called_once()
        self.assertTrue(window.clustering_pane.cluster_button.isEnabled())
        self.assertFalse(window.source_pane.basic_run_button.isHidden())
        self.assertTrue(window.source_pane.basic_cancel_button.isHidden())
        self.assertEqual([], window.job_manager.active_jobs())
        window.close()

    def test_main_performance_dashboard_updates_from_fake_job_metrics(self):
        window = ClusterGalleryApp()
        job_id = window.job_manager.register_job("Indexing faces")
        window.last_run_metrics = {
            "model_load_time_s": 0.25,
            "cache_hits": 8,
            "cache_misses": 2,
            "skipped_unchanged": 13,
            "detector_calls": 6,
            "embedder_calls": 5,
            "ann_state": "ready",
        }

        window._update_metrics_overlay()

        text = window.footer_bar.performance_dashboard_label.toolTip()
        self.assertIn("Indexing faces", text)
        self.assertIn("Cancel: running", text)
        self.assertIn("Model: loaded", text)
        self.assertIn("Cache: 8 hit/2 miss", text)
        self.assertIn("Skipped: 13 unchanged", text)
        self.assertIn("Face calls: detector=6, embedder=5", text)
        self.assertIn("ANN: ready", text)

        window.job_manager.finish(job_id, status="cancelled")
        window.last_run_metrics["status"] = "cancelled"
        window._update_metrics_overlay()

        self.assertIn("Cancel: cancelled", window.footer_bar.performance_dashboard_label.toolTip())
        window.close()

    def test_clustering_worker_callbacks_arrive_on_gui_thread(self):
        window = ClusterGalleryApp()
        window.pipeline = self._FakePipeline()
        request = ClusteringRequest(
            directory=".",
            embedding_models=["dino"],
            num_clusters=2,
            clustering_backends=["cosine-kmeans"],
            recursive=False,
            similarity_mode="cosine",
            outlier_policy="assign",
            use_onnx=False,
            reuse_result_cache=False,
            use_embedding_cache_lookup=False,
            source_paths=None,
            performance_profile="balanced",
        )
        gui_thread = APP.thread()
        seen: list[tuple[str, bool]] = []
        completed: list[bool] = []

        def progress_slot(value: int, status: str) -> None:
            seen.append((f"progress:{value}:{status}", QThread.currentThread() is gui_thread))

        def finished_slot(result, metrics) -> None:
            _ = result
            _ = metrics
            seen.append(("completed", QThread.currentThread() is gui_thread))
            completed.append(True)

        with patch.object(window.clustering_pane, "update_progress", side_effect=progress_slot), patch.object(
            window.source_pane, "update_progress", side_effect=progress_slot
        ), patch.object(window, "_on_clustering_progress", side_effect=progress_slot), patch.object(
            window, "on_clustering_finished", side_effect=finished_slot
        ):
            window._start_clustering_request(request, run_origin="folder")
            deadline = monotonic() + 5.0
            while monotonic() < deadline and not completed:
                APP.processEvents()
                sleep(0.01)
            APP.processEvents()

        self.assertTrue(completed)
        self.assertTrue(seen)
        self.assertTrue(all(on_gui_thread for _label, on_gui_thread in seen), seen)
        window.close()

    def test_async_job_callbacks_arrive_on_gui_thread(self):
        gui_thread = APP.thread()
        seen: list[tuple[str, bool]] = []
        completed: list[object] = []

        def _thread_running(thread) -> bool:
            try:
                return bool(thread.isRunning())
            except RuntimeError:
                return False

        def _run(progress, cancel_check):
            self.assertFalse(cancel_check())
            progress(25, "working")
            return "done"

        job = AsyncJob(_run)
        job.started.connect(lambda: seen.append(("started", QThread.currentThread() is gui_thread)))
        job.progress.connect(lambda value, text: seen.append((f"progress:{value}:{text}", QThread.currentThread() is gui_thread)))
        job.completed.connect(lambda result: (seen.append(("completed", QThread.currentThread() is gui_thread)), completed.append(result)))
        thread = start_job_in_thread(job)

        deadline = monotonic() + 5.0
        while monotonic() < deadline and (_thread_running(thread) or not completed):
            APP.processEvents()
            sleep(0.01)
        APP.processEvents()

        self.assertEqual(["done"], completed)
        self.assertTrue(seen)
        self.assertTrue(all(on_gui_thread for _label, on_gui_thread in seen), seen)
        self.assertFalse(_thread_running(thread))

    def test_async_job_cancel_after_function_return_is_not_reported_completed(self):
        completed: list[object] = []
        cancelled: list[bool] = []
        job = AsyncJob(lambda _progress, _cancel_check: "ignored cancellation")
        job.completed.connect(completed.append)
        job.cancelled.connect(lambda: cancelled.append(True))

        job.cancel()
        job.run()
        self.assertTrue(self._wait_until(lambda: bool(cancelled), timeout_s=1.0))

        self.assertEqual([], completed)
        self.assertEqual([True], cancelled)
        job.dispose()

    def test_job_manager_cancels_once_preserves_terminal_state_and_bounds_history(self):
        cancel_calls: list[bool] = []
        manager = JobManager(max_history=10)
        job_id = manager.register_job("Download model", cancel_fn=lambda: cancel_calls.append(True))
        manager.update(job_id, progress=25, text="Downloading", cache_status="cache miss")

        manager.cancel(job_id)
        manager.cancel(job_id)
        manager.finish(job_id, status="cancelled", cache_status="partial cache retained")
        manager.update(job_id, progress=99, text="late update")
        manager.finish(job_id, status="failed", error="late failure")

        state = manager.get(job_id)
        self.assertEqual([True], cancel_calls)
        self.assertEqual("cancelled", state.status)
        self.assertEqual(25, state.progress)
        self.assertEqual("Downloading", state.text)
        self.assertEqual("partial cache retained", state.cache_status)

        for index in range(20):
            history_id = manager.register_job(f"History {index}")
            manager.finish(history_id)
        self.assertEqual(10, len(manager.history(limit=100)))

    def test_main_cancel_cluster_tag_refresh_retains_live_thread_refs(self):
        window = ClusterGalleryApp()
        job = self._FakeAsyncJob()
        thread = self._FakeAsyncThread()
        window._tag_context_job = job
        window._tag_context_thread = thread
        generation = window._tag_context_generation

        window._cancel_cluster_tag_context_refresh()

        self.assertEqual(1, job.cancel_calls)
        self.assertIsNone(window._tag_context_job)
        self.assertIsNone(window._tag_context_thread)
        self.assertEqual(generation + 1, window._tag_context_generation)
        self.assertIn((job, thread), window._retained_async_refs)

        window._release_async_refs(job, thread)
        self.assertEqual([], window._retained_async_refs)
        window.close()

    def test_wait_for_thread_shutdown_polls_python_qthreads(self):
        thread = self._PollingAsyncQThread()
        thread.start()
        self.assertTrue(self._wait_until(thread.isRunning, timeout_s=1.0))

        thread.stop_event.set()
        ready = wait_for_thread_shutdown(thread, timeout_ms=500)

        self.assertTrue(ready)
        self.assertEqual([], thread.wait_calls)
        self.assertTrue(self._wait_until(lambda: not thread.isRunning(), timeout_s=1.0))

    def test_main_shutdown_background_jobs_waits_for_threads(self):
        window = ClusterGalleryApp()
        warmup_job = self._FakeAsyncJob()
        warmup_thread = self._FakeAsyncThread()
        tag_job = self._FakeAsyncJob()
        tag_thread = self._FakeAsyncThread()
        worker = self._FakeWorkerThread()
        window._warmup_job = warmup_job
        window._warmup_thread = warmup_thread
        window._tag_context_job = tag_job
        window._tag_context_thread = tag_thread
        window.worker = worker

        with patch.object(window.gallery_pane, "shutdown_jobs", return_value=True) as shutdown_jobs, patch.object(
            window.faces_pane, "shutdown_jobs", return_value=True
        ) as faces_shutdown_jobs:
            ready_to_close = window._shutdown_background_jobs(timeout_ms=25)

        self.assertTrue(ready_to_close)
        self.assertEqual(1, warmup_job.cancel_calls)
        self.assertEqual(1, tag_job.cancel_calls)
        self.assertEqual(1, worker.cancel_calls)
        self.assertEqual([25], warmup_thread.wait_calls)
        self.assertEqual([25], tag_thread.wait_calls)
        self.assertEqual([25], worker.wait_calls)
        shutdown_jobs.assert_called_once_with(timeout_ms=25)
        faces_shutdown_jobs.assert_called_once_with(timeout_ms=25)
        window.close()

    def test_visible_clustering_ui_exposes_tooltips_without_help_icons(self):
        window = ClusterGalleryApp()
        window.set_clustering_mode("advanced")
        window.show()
        APP.processEvents()
        self.assertTrue(window.clustering_pane.cluster_button.isEnabled())
        self.assertFalse(window.clustering_pane.cancel_button.isEnabled())
        self.assertEqual([], window.clustering_pane.findChildren(QToolButton))
        self.assertEqual([], window.cluster_pane.findChildren(QToolButton))
        self.assertEqual([], window.gallery_pane.findChildren(QToolButton))
        self.assertTrue(window.clustering_pane.tag_filter_field.toolTip())
        self.assertTrue(window.gallery_pane.selected_tags_button.toolTip())
        self.assertTrue(window.gallery_pane.tags_checked_group_action.toolTip())
        self.assertTrue(window.gallery_pane.delete_selection_action.toolTip())
        self.assertTrue(window.gallery_pane.copy_paths_action.toolTip())
        self.assertTrue(window.gallery_pane.actions_menu_button.toolTip())
        self.assertTrue(window.cluster_pane.recluster_button.toolTip())
        window.close()

    def test_help_tooltips_explain_tag_filter_tags_and_recluster(self):
        window = ClusterGalleryApp()
        window.set_clustering_mode("advanced")
        window.show()
        APP.processEvents()

        tag_filter_tooltip = window.clustering_pane.tag_filter_field.toolTip()
        self.assertIn("selected folder", tag_filter_tooltip)
        self.assertIn("Any means", tag_filter_tooltip)
        self.assertIn("All means", tag_filter_tooltip)

        tags_tooltip = window.gallery_pane.tags_checked_group_action.toolTip()
        self.assertIn("selected photos", tags_tooltip)
        self.assertIn("current group", tags_tooltip)
        self.assertIn("EXIF", tags_tooltip)
        selected_tags_tooltip = window.gallery_pane.selected_tags_button.toolTip()
        self.assertIn("never falls back", selected_tags_tooltip)
        self.assertIn("selected gallery tiles", selected_tags_tooltip)
        self.assertIn("whole selected cluster", window.gallery_pane.delete_selection_action.toolTip())
        self.assertIn("copy paths", window.gallery_pane.actions_menu_button.toolTip())

        recluster_tooltip = window.cluster_pane.recluster_button.toolTip()
        self.assertIn("selected cluster", recluster_tooltip)
        self.assertIn("main clustering settings", recluster_tooltip)

        window.close()

    def test_gallery_clear_memory_caches_empties_pixmap_and_thumbnail_caches(self):
        pane = GalleryPane()
        pane.images = ["a.jpg"]
        pane.model.set_images(pane.images)
        image = QImage(12, 12, QImage.Format.Format_RGB32)
        image.fill(Qt.GlobalColor.white)
        pane.model.set_image(0, image)
        pane.loaded_indexes.add(0)
        pane.thumbnail_service._qimage_cache[("a.jpg", pane.image_size)] = QImage(12, 12, QImage.Format.Format_RGB32)
        pane.clear_memory_caches()
        pane.refresh_timer.stop()
        self.assertEqual({}, dict(pane.thumbnail_service._qimage_cache))
        self.assertEqual(set(), pane.loaded_indexes)
        self.assertIsNone(pane.model.data(pane.model.index(0, 0), pane.model.PixmapRole))
        pane.close()

    def test_gallery_uses_batched_layout_for_large_face_review_sets(self):
        pane = GalleryPane()

        self.assertEqual(pane.list_view.LayoutMode.Batched, pane.list_view.layoutMode())
        self.assertEqual(24, pane.list_view.batchSize())

        pane.close()

    def test_gallery_delegate_paints_face_boxes_with_clipping_and_limit(self):
        model = GalleryImageModel()
        model.set_images(["/photos/a.jpg"])
        image = QImage(128, 128, QImage.Format.Format_RGB32)
        image.fill(Qt.GlobalColor.white)
        model.set_image(0, image)
        model.set_face_boxes_by_path(
            {
                "/photos/a.jpg": [
                    (-5.0, -2.0, 4.0, 3.0),
                    *[(0.01 * index, 0.02, 0.05 + (0.01 * index), 0.10) for index in range(MAX_FACE_BOXES_PER_TILE + 8)],
                ]
            }
        )
        delegate = GalleryItemDelegate(128)
        option = QStyleOptionViewItem()
        option.rect.setRect(0, 0, delegate.card_width, delegate.card_height)
        index = model.index(0, 0)
        canvas = QImage(delegate.card_width, delegate.card_height, QImage.Format.Format_ARGB32)
        canvas.fill(Qt.GlobalColor.black)
        painter = QPainter(canvas)
        try:
            delegate.paint(painter, option, index)
        finally:
            painter.end()

        self.assertFalse(canvas.isNull())
        self.assertEqual(MAX_FACE_BOXES_PER_TILE + 9, len(index.data(GalleryImageModel.FaceBoxesRole)))

    def test_gallery_loader_callbacks_arrive_on_gui_thread(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample.jpg"
            Image.new("RGB", (64, 64), (120, 30, 40)).save(image_path)
            pane = GalleryPane()
            pane.max_thumbnail_workers = 1
            pane.resize(900, 700)
            pane.show()
            APP.processEvents()
            gui_thread = APP.thread()
            seen: list[bool] = []
            loaded: list[int] = []
            original_queue = pane.queueImageForGallery

            def queue_slot(generation, index, path, qimage):
                seen.append(QThread.currentThread() is gui_thread)
                loaded.append(int(index))
                return original_queue(generation, index, path, qimage)

            pane.queueImageForGallery = queue_slot
            pane.update_gallery_with_options(images=[str(image_path)], clear_pixmaps=True, reset_scroll=True)
            pane.load_visible_images()
            deadline = monotonic() + 5.0
            while monotonic() < deadline and not loaded:
                APP.processEvents()
                sleep(0.01)
            APP.processEvents()

            self.assertTrue(loaded)
            self.assertTrue(seen)
            self.assertTrue(all(seen), seen)
            pane.cancel_loader()
            pane.close()

    def test_gallery_incremental_append_hides_empty_state_and_shows_results(self):
        pane = GalleryPane()
        pane.resize(900, 700)
        pane.show()
        pane.update_gallery_with_options([], clear_pixmaps=True, reset_scroll=True)
        APP.processEvents()
        self.assertTrue(pane.empty_state.isVisible())
        self.assertFalse(pane.list_view.isVisible())

        pane.append_images(["/photos/one.jpg", "/photos/two.jpg"])
        APP.processEvents()

        self.assertEqual(2, pane.model.rowCount())
        self.assertFalse(pane.empty_state.isVisible())
        self.assertTrue(pane.list_view.isVisible())
        pane.close()

    def test_gallery_cancel_loader_retains_still_running_threads(self):
        pane = GalleryPane()
        thread = self._StickyLoaderThread()
        pane.loader_threads = [thread]

        pane.cancel_loader()

        self.assertEqual([], pane.loader_threads)
        self.assertIn(thread, pane._retained_loader_threads)
        self.assertEqual([2500], thread.wait_calls)
        pane._release_loader_thread(thread)
        self.assertEqual([], pane._retained_loader_threads)
        pane.close()

    def test_gallery_flush_pending_ui_items_batches_large_updates(self):
        pane = GalleryPane()
        pane.images = [f"/photos/{index}.jpg" for index in range(MAX_PENDING_UI_ITEMS_PER_FLUSH + 5)]
        pane.model.set_images(pane.images)
        qimage = QImage(8, 8, QImage.Format.Format_RGB32)
        qimage.fill(Qt.GlobalColor.white)
        pane.pending_ui_items = [
            (index, image_path, qimage)
            for index, image_path in enumerate(pane.images)
        ]
        pane.first_paint_start = monotonic()
        pane.flush_pending_ui_items()

        self.assertEqual(MAX_PENDING_UI_ITEMS_PER_FLUSH, len(pane.loaded_indexes))
        self.assertEqual(5, len(pane.pending_ui_items))
        self.assertTrue(pane.flush_timer.isActive())
        self.assertIsInstance(pane.model.data(pane.model.index(0, 0), pane.model.PixmapRole), QImage)
        pane.flush_timer.stop()
        pane.close()

    def test_gallery_load_visible_images_limits_requested_batches(self):
        pane = GalleryPane()
        pane.images = [f"/photos/{index}.jpg" for index in range(200)]
        visible_indexes = list(range(100))
        prefetch_indexes = list(range(100, 180))
        enqueue_calls: list[tuple[list[int], int]] = []

        with (
            patch.object(pane, "_visible_and_prefetch_indexes", return_value=(visible_indexes, prefetch_indexes)),
            patch.object(pane, "_ensure_loader"),
            patch.object(
                pane.loader_queue,
                "enqueue",
                side_effect=lambda indexes, priority: enqueue_calls.append((list(indexes), int(priority))),
            ),
        ):
            pane.load_visible_images()

        self.assertEqual(
            [
                (list(range(MAX_VISIBLE_THUMBNAIL_REQUESTS_PER_CYCLE)), 0),
                (
                    list(
                        range(
                            100,
                            100 + MAX_PREFETCH_THUMBNAIL_REQUESTS_PER_CYCLE,
                        )
                    ),
                    1,
                ),
            ],
            enqueue_calls,
        )
        self.assertEqual(
            set(range(MAX_VISIBLE_THUMBNAIL_REQUESTS_PER_CYCLE)).union(
                range(100, 100 + MAX_PREFETCH_THUMBNAIL_REQUESTS_PER_CYCLE)
            ),
            pane.pending_indexes,
        )
        pane.close()

    def test_gallery_load_visible_images_skips_already_pending_indexes(self):
        pane = GalleryPane()
        pane.images = [f"/photos/{index}.jpg" for index in range(40)]
        visible_indexes = list(range(20))
        prefetch_indexes = list(range(20, 40))
        pane.pending_indexes.update({2, 3, 21, 22})
        enqueue_calls: list[tuple[list[int], int]] = []

        with (
            patch.object(pane, "_visible_and_prefetch_indexes", return_value=(visible_indexes, prefetch_indexes)),
            patch.object(pane, "_ensure_loader"),
            patch.object(
                pane.loader_queue,
                "enqueue",
                side_effect=lambda indexes, priority: enqueue_calls.append((list(indexes), int(priority))),
            ),
        ):
            pane.load_visible_images()

        self.assertNotIn(2, enqueue_calls[0][0])
        self.assertNotIn(3, enqueue_calls[0][0])
        self.assertNotIn(21, enqueue_calls[1][0])
        self.assertNotIn(22, enqueue_calls[1][0])
        self.assertEqual({2, 3, 21, 22}.union(enqueue_calls[0][0]).union(enqueue_calls[1][0]), pane.pending_indexes)
        pane.close()

    def test_settings_dialog_exposes_storage_tab_and_cache_actions(self):
        store = QSettings("ClusterLensTests", "SettingsDialogStorage")
        dialog = SettingsDialog(
            store,
            RuntimeCapabilityService(),
            describe_rebuildable_caches=lambda: CacheUsageSummary(
                cache_root="D:/cache",
                target_bytes={"embeddings.sqlite3": 1024, "thumbnails/": 2048},
                total_bytes=3072,
            ),
            describe_generated_storage=lambda: GeneratedStorageSummary(
                runtime_root="D:/ClusterLens",
                config_location="D:/config/ClusterLens.ini",
                cache_root="D:/cache",
                target_bytes={
                    "logs": 10,
                    "thumbnails": 20,
                    "rebuildable_caches": 30,
                    "face_databases": 40,
                    "ann_files": 50,
                    "model_caches": 60,
                    "temp_files": 70,
                },
                target_paths={
                    "logs": ("D:/ClusterLens/logs",),
                    "thumbnails": ("D:/cache/thumbnails",),
                    "rebuildable_caches": ("D:/cache/embeddings.sqlite3",),
                    "face_databases": ("D:/cache/face_search_human.db",),
                    "ann_files": ("D:/cache/face_search_human.db.faiss",),
                    "model_caches": ("D:/cache/huggingface",),
                    "temp_files": ("D:/cache/tmp",),
                },
                total_bytes=280,
            ),
            prepare_rebuildable_cache_clear=lambda: None,
            clear_rebuildable_caches=lambda: CacheClearResult(
                cleared_targets=("embeddings.sqlite3", "thumbnails/"),
                freed_bytes=3072,
                failures=(),
            ),
            clear_runtime_temp_files=lambda: CacheClearResult(cleared_targets=("tmp/",), freed_bytes=70, failures=()),
            clear_face_storage=lambda: CacheClearResult(cleared_targets=("face_search_human.db",), freed_bytes=40, failures=()),
            clear_model_caches=lambda: CacheClearResult(cleared_targets=("huggingface",), freed_bytes=60, failures=()),
            clear_logs=lambda: CacheClearResult(cleared_targets=("logs/app.log",), freed_bytes=10, failures=()),
            can_clear_rebuildable_caches=lambda: True,
        )
        dialog.show()
        APP.processEvents()
        deadline = monotonic() + 1.5
        while dialog._cache_usage_thread is not None and monotonic() < deadline:
            dialog._cache_usage_thread.wait(50)
            APP.processEvents()
        self.assertEqual("Storage", dialog.tabs.tabText(2))
        self.assertEqual("Refresh Cache Usage", dialog.refresh_cache_usage_button.text())
        self.assertEqual("Clear Rebuildable Caches", dialog.clear_cache_button.text())
        self.assertEqual("Clear Temp Files", dialog.clear_runtime_temp_button.text())
        self.assertEqual("Clear Face DBs / ANN", dialog.clear_face_storage_button.text())
        self.assertEqual("Clear Model Caches", dialog.clear_model_caches_button.text())
        self.assertEqual("Clear Logs", dialog.clear_logs_button.text())
        self.assertTrue(dialog.clear_cache_button.isEnabled())
        self.assertTrue(dialog.clear_runtime_temp_button.isEnabled())
        self.assertTrue(dialog.clear_face_storage_button.isEnabled())
        self.assertTrue(dialog.clear_model_caches_button.isEnabled())
        self.assertTrue(dialog.clear_logs_button.isEnabled())
        self.assertTrue(dialog.runtime_root_label.text())
        self.assertTrue(dialog.config_location_label.text())
        self.assertTrue(dialog.cache_root_label.text())
        generated_text = dialog.generated_storage_text.toPlainText()
        for expected in ("Runtime root", "Config location", "Logs", "Thumbnails", "Face DBs", "ANN files", "Model caches", "Temp files"):
            self.assertIn(expected, generated_text)
        dialog.close()

    def test_settings_dialog_allows_faces_as_default_workspace(self):
        store = QSettings("ClusterLensTests", "SettingsDialogFacesWorkspace")
        store.setValue("workspace/default_view", "faces")
        store.setValue("faces/default_mode", "dog")
        store.setValue("faces/model_root", "/models/pets")
        store.setValue("faces/default_detector/human", BUILTIN_HUMAN_DETECTOR_ID)
        store.setValue("faces/default_embedder/human", BUILTIN_HUMAN_EMBEDDER_ID)
        store.setValue("faces/detector_score_threshold/human", 0.35)
        store.setValue("faces/max_detections/human", 22)
        dialog = SettingsDialog(
            store,
            RuntimeCapabilityService(),
            describe_rebuildable_caches=lambda: CacheUsageSummary(
                cache_root="D:/cache",
                target_bytes={"embeddings.sqlite3": 1024},
                total_bytes=1024,
            ),
            prepare_rebuildable_cache_clear=lambda: None,
            clear_rebuildable_caches=lambda: CacheClearResult(
                cleared_targets=("embeddings.sqlite3",),
                freed_bytes=1024,
                failures=(),
            ),
            can_clear_rebuildable_caches=lambda: True,
        )
        self.assertEqual("faces", dialog.default_workspace.currentText())
        self.assertIn("faces", [dialog.default_workspace.itemText(i) for i in range(dialog.default_workspace.count())])
        self.assertEqual("dog", dialog.default_face_mode.currentText())
        self.assertEqual("/models/pets", dialog.face_model_root.text())
        self.assertTrue(dialog.face_model_status.text())
        self.assertTrue(dialog.default_face_detector.count() >= 1)
        self.assertTrue(dialog.default_face_embedder.count() >= 1)
        deadline = monotonic() + 1.5
        while dialog._cache_usage_thread is not None and monotonic() < deadline:
            dialog._cache_usage_thread.wait(50)
            APP.processEvents()
        dialog.shutdown_jobs(timeout_ms=50)
        dialog.close()

    def test_settings_dialog_exposes_face_model_install_actions(self):
        store = QSettings("ClusterLensTests", "SettingsDialogFaceModelActions")
        dialog = SettingsDialog(
            store,
            RuntimeCapabilityService(),
            describe_rebuildable_caches=lambda: CacheUsageSummary(
                cache_root="D:/cache",
                target_bytes={"embeddings.sqlite3": 1024, "face_model_assets/": 2048},
                total_bytes=3072,
            ),
            prepare_rebuildable_cache_clear=lambda: None,
            clear_rebuildable_caches=lambda: CacheClearResult(
                cleared_targets=("embeddings.sqlite3", "face_model_assets/"),
                freed_bytes=3072,
                failures=(),
            ),
            can_clear_rebuildable_caches=lambda: True,
        )
        self.assertEqual("Refresh Face Models", dialog.refresh_face_models_button.text())
        self.assertEqual("Install Recommended", dialog.install_recommended_face_models_button.text())
        self.assertEqual("Install Edge", dialog.install_edge_face_models_button.text())
        self.assertEqual("Install Accuracy", dialog.install_accuracy_face_models_button.text())
        self.assertEqual("Install Latest GPU", dialog.install_latest_gpu_face_models_button.text())
        self.assertEqual("Install Max Accuracy", dialog.install_max_accuracy_face_models_button.text())
        self.assertEqual("Install SFace", dialog.install_sface_face_models_button.text())
        self.assertEqual("Install YOLO", dialog.install_yolo_face_model_button.text())
        self.assertEqual("Install YuNet", dialog.install_yunet_face_model_button.text())
        self.assertEqual("Delete Installed Face Model", dialog.delete_face_model_button.text())
        self.assertTrue(dialog.face_model_cache_label.text())
        status_text = dialog.face_model_status.text()
        self.assertIn("Human choices", status_text)
        self.assertIn("Dog:", status_text)
        self.assertIn("Cat:", status_text)
        self.assertGreaterEqual(dialog.delete_face_model_combo.count(), 3)
        deadline = monotonic() + 1.5
        while dialog._cache_usage_thread is not None and monotonic() < deadline:
            dialog._cache_usage_thread.wait(50)
            APP.processEvents()
        dialog.shutdown_jobs(timeout_ms=50)
        dialog.close()

    def test_settings_dialog_tabs_are_scroll_wrapped(self):
        store = QSettings("ClusterLensTests", "SettingsDialogScrollTabs")
        dialog = SettingsDialog(store, RuntimeCapabilityService())
        dialog.show()
        APP.processEvents()

        self.assertEqual("Runtime", dialog.tabs.tabText(0))
        self.assertEqual("Workspace", dialog.tabs.tabText(1))
        self.assertEqual("Storage", dialog.tabs.tabText(2))
        for index in range(3):
            self.assertEqual("QScrollArea", dialog.tabs.widget(index).__class__.__name__)

        dialog.shutdown_jobs(timeout_ms=50)
        dialog.close()

    def test_default_workspace_setting_can_start_in_faces_view(self):
        self._app_settings_store.setValue("workspace/default_view", "faces")
        window = ClusterGalleryApp()
        window.show()
        APP.processEvents()

        self.assertEqual("faces", window._active_workspace)
        self.assertTrue(window.faces_workspace_button.isChecked())
        self.assertTrue(window.basic_mode_button.isVisible())
        self.assertTrue(window.advanced_mode_button.isVisible())
        self.assertTrue(window.basic_mode_button.isChecked())
        self.assertFalse(window.advanced_mode_button.isChecked())
        self.assertTrue(window.source_toggle.isVisible())
        self.assertFalse(window.controls_toggle.isVisible())
        self.assertFalse(window.details_toggle.isVisible())
        self.assertFalse(window.source_pane.basic_action_section.isVisible())
        self.assertEqual("Feature: Faces | Basic", window.feature_label.text())
        self.assertIs(window.workspace_stack.currentWidget(), window.faces_pane)
        window.close()

    def test_default_face_mode_setting_selects_faces_mode_on_startup(self):
        self._app_settings_store.setValue("workspace/default_view", "faces")
        self._app_settings_store.setValue("faces/default_mode", "dog")
        window = ClusterGalleryApp()
        window.show()
        APP.processEvents()

        self.assertEqual("dog", window.faces_pane.current_face_mode())
        self.assertIn("Dog", window.faces_pane.face_mode_status_label.text())
        window.close()

    def test_stored_faces_ui_mode_is_restored_on_startup(self):
        self._app_settings_store.setValue("workspace/default_view", "faces")
        self._app_settings_store.setValue("workspace/faces_mode", "advanced")
        window = ClusterGalleryApp()
        window.show()
        APP.processEvents()

        self.assertEqual("advanced", window._faces_mode)
        self.assertTrue(window.advanced_mode_button.isChecked())
        self.assertFalse(window.basic_mode_button.isChecked())
        self.assertEqual("advanced", window.faces_pane.current_ui_mode())
        self.assertEqual("Feature: Faces | Advanced", window.feature_label.text())
        window.close()

    def test_faces_workspace_switch_hides_clustering_only_controls(self):
        window = ClusterGalleryApp()
        window.show()
        APP.processEvents()

        window.set_active_workspace("faces")
        APP.processEvents()
        self.assertEqual("faces", window._active_workspace)
        self.assertTrue(window.faces_workspace_button.isChecked())
        self.assertFalse(window.clustering_workspace_button.isChecked())
        self.assertTrue(window.source_toggle.isVisible())
        self.assertFalse(window.controls_toggle.isVisible())
        self.assertFalse(window.details_toggle.isVisible())
        self.assertTrue(window.basic_mode_button.isVisible())
        self.assertTrue(window.advanced_mode_button.isVisible())
        self.assertTrue(window.basic_mode_button.isChecked())
        self.assertFalse(window.advanced_mode_button.isChecked())
        self.assertEqual("Feature: Faces | Basic", window.feature_label.text())
        self.assertEqual(["All Faces", "Folder Review", "Face Search", "Identities"], window.faces_pane.tab_labels())

        window.set_active_workspace("clustering")
        APP.processEvents()
        self.assertEqual("clustering", window._active_workspace)
        self.assertTrue(window.basic_mode_button.isVisible())
        self.assertTrue(window.advanced_mode_button.isVisible())
        window.close()

    def test_faces_workspace_switch_loads_saved_album_and_folder_cache(self):
        window = ClusterGalleryApp()
        calls: list[str] = []
        window.faces_pane.ensure_faces_workspace_loaded = lambda: calls.append("workspace")

        window.set_active_workspace("faces")
        APP.processEvents()

        self.assertEqual(["workspace"], calls)
        window.close()

    def test_faces_workspace_loader_primes_album_and_selected_folder_cache(self):
        pane = SearchPane(enabled_tabs=["All Faces", "Folder Review", "Face Search"], external_results=False)
        calls: list[tuple[str, dict[str, object]]] = []
        pane.ensure_face_album_loaded = lambda **kwargs: calls.append(("album", dict(kwargs)))  # type: ignore[method-assign]
        pane.ensure_face_library_loaded = lambda **kwargs: calls.append(("folder", dict(kwargs)))  # type: ignore[method-assign]

        pane.ensure_faces_workspace_loaded()

        self.assertEqual(
            [("album", {"allow_inactive": True}), ("folder", {"allow_inactive": True})],
            calls,
        )
        pane.close()

    def test_all_faces_tab_is_the_default_faces_home_and_loads_album_groups(self):
        pane = SearchPane(enabled_tabs=["All Faces", "Faces In Folder", "Face Search"], external_results=False)
        service = self._FakeFaceLibraryService(
            profiles=[self._person_profile("Alice", visible_face_count=1)],
            records=[
                self._face_record("/photos/a.jpg", 0, person_name="Alice"),
                self._face_record("/photos/b.jpg", 0),
            ],
            pending_assignments=[
                FaceLabelAssignment(
                    proposal_id=1,
                    person_name="Alice",
                    image_path="/photos/b.jpg",
                    face_index=0,
                    face_bbox=(10, 10, 48, 52),
                    confidence=0.82,
                    pending=True,
                )
            ],
        )
        pane.face_service_global = service
        pane.show()
        APP.processEvents()
        try:
            self.assertEqual(["All Faces", "Folder Review", "Face Search"], pane.tab_labels())
            self.assertEqual(0, pane.active_tab_index())

            pane.ensure_face_album_loaded()
            self.assertTrue(
                self._wait_until(
                    lambda: bool(service.load_face_album_groups_calls)
                    and bool(service.load_face_album_members_calls)
                    and pane._results_kind == "face_album"
                    and self._list_view_count(pane.face_results_groups_list) >= 2
                )
            )
            self.assertGreaterEqual(self._list_view_count(pane.face_results_groups_list), 2)
            self.assertIn("saved face library", pane._face_album_summary_text.lower())
        finally:
            pane.close()

    def test_all_faces_refresh_does_not_trigger_folder_review_load(self):
        pane = SearchPane(enabled_tabs=["All Faces", "Faces In Folder", "Face Search"], external_results=False)
        service = self._FakeFaceLibraryService(
            records=[self._face_record("/photos/a.jpg", 0, person_name="Alice")],
        )
        pane.face_service_global = service
        pane.show()
        APP.processEvents()
        try:
            self.assertEqual(0, pane.active_tab_index())
            pane._on_show_tiny_detections_changed()
            self.assertTrue(
                self._wait_until(
                    lambda: bool(service.load_face_album_groups_calls)
                    and bool(service.load_face_album_members_calls)
                )
            )
            self.assertEqual([], service.load_folder_review_images_calls)
        finally:
            pane.close()

    def test_faces_in_folder_can_upload_session_faces_into_global_db(self):
        source_service = self._FakeFaceLibraryService(
            profiles=[self._person_profile("Alice", visible_face_count=1)],
            records=[
                self._face_record("/photos/a.jpg", 0, person_name="Alice"),
                self._face_record("/other/outside.jpg", 0, person_name="Bob"),
            ],
            pending_assignments=[
                FaceLabelAssignment(
                    proposal_id=1,
                    person_name="Alice",
                    image_path="/photos/a.jpg",
                    face_index=0,
                    face_bbox=(10, 10, 48, 52),
                    confidence=0.91,
                    pending=True,
                )
            ],
        )
        target_service = self._FakeFaceLibraryService()
        pane = SearchPane(
            enabled_tabs=["All Faces", "Faces In Folder", "Face Search"],
            external_results=False,
            face_service_global=target_service,
            face_service_session=source_service,
        )
        pane.face_db_scope.setCurrentIndex(pane.face_db_scope.findData("session"))
        pane.face_folder_path.setText("/photos")
        try:
            pane._upload_face_folder_to_global_db()
            self.assertTrue(self._wait_until(lambda: bool(target_service.import_identity_payloads)))
            payload = target_service.import_identity_payloads[-1]

            self.assertEqual("merge", target_service.import_identity_modes[-1])
            self.assertEqual({"/photos/a.jpg"}, {str(item["image_path"]) for item in payload["face_index"]})
            self.assertEqual({"/photos/a.jpg"}, {str(item["image_path"]) for item in payload["face_scan_images"]})
            self.assertEqual({"/photos/a.jpg"}, {str(item["image_path"]) for item in payload["labels"]})
            self.assertEqual({"/photos/a.jpg"}, {str(item["image_path"]) for item in payload["pending_labels"]})
            self.assertEqual(["Alice"], [str(item["person_name"]) for item in payload["identities"]])
        finally:
            pane.close()

    def test_faces_workspace_mode_switch_is_separate_from_clustering_mode(self):
        window = ClusterGalleryApp()
        window.show()
        APP.processEvents()

        window.set_clustering_mode("advanced")
        APP.processEvents()
        self.assertEqual("advanced", window._clustering_mode)

        window.set_active_workspace("faces")
        APP.processEvents()
        self.assertEqual("basic", window._faces_mode)
        self.assertTrue(window.basic_mode_button.isChecked())
        self.assertFalse(window.advanced_mode_button.isChecked())

        window.set_faces_mode("advanced")
        APP.processEvents()
        self.assertEqual("advanced", window._faces_mode)
        self.assertEqual("Feature: Faces | Advanced", window.feature_label.text())
        self.assertEqual("advanced", window.faces_pane.current_ui_mode())

        window.set_active_workspace("clustering")
        APP.processEvents()
        self.assertEqual("advanced", window._clustering_mode)
        self.assertTrue(window.advanced_mode_button.isChecked())
        self.assertFalse(window.basic_mode_button.isChecked())

        window.set_active_workspace("faces")
        APP.processEvents()
        self.assertEqual("advanced", window._faces_mode)
        self.assertTrue(window.advanced_mode_button.isChecked())
        self.assertFalse(window.basic_mode_button.isChecked())
        window.close()

    def test_mode_switch_toggles_basic_and_advanced_workspace_visibility(self):
        window = ClusterGalleryApp()
        window.show()
        APP.processEvents()
        self.assertEqual("basic", window._clustering_mode)
        self.assertTrue(window.source_pane.basic_action_section.isVisible())
        self.assertFalse(window.clustering_pane.isVisible())
        self.assertFalse(hasattr(window.gallery_pane, "tags_button"))
        self.assertFalse(window.gallery_pane.metadata_menu.menuAction().isVisible())
        self.assertFalse(window.gallery_pane.selected_tags_button.isVisible())
        self.assertEqual("Compare", window.details_toggle.text())

        window.set_clustering_mode("advanced")
        APP.processEvents()
        self.assertEqual("advanced", window._clustering_mode)
        self.assertFalse(window.source_pane.basic_action_section.isVisible())
        self.assertTrue(window.clustering_pane.isVisible())
        self.assertFalse(hasattr(window.gallery_pane, "tags_button"))
        self.assertTrue(window.gallery_pane.metadata_menu.menuAction().isVisible())
        self.assertTrue(window.gallery_pane.selected_tags_button.isVisible())
        self.assertTrue(window.source_toggle.isVisible())
        self.assertTrue(window.controls_toggle.isVisible())
        self.assertTrue(window.details_toggle.isVisible())
        window.close()

    def test_basic_source_actions_track_running_state_across_mode_switch(self):
        window = ClusterGalleryApp()
        window.show()
        APP.processEvents()

        self.assertTrue(window.source_pane.basic_run_button.isVisible())
        self.assertFalse(window.source_pane.basic_cancel_button.isVisible())

        class _RunningWorker:
            @staticmethod
            def isRunning():
                return True

        window.worker = _RunningWorker()
        window.clustering_pane.set_running(True)
        window.source_pane.set_running(True)
        window.set_clustering_mode("advanced")
        APP.processEvents()
        self.assertFalse(window.source_pane.basic_action_section.isVisible())
        self.assertTrue(window.clustering_pane.cancel_button.isEnabled())

        window.set_clustering_mode("basic")
        APP.processEvents()
        self.assertTrue(window.source_pane.basic_action_section.isVisible())
        self.assertFalse(window.source_pane.basic_run_button.isVisible())
        self.assertTrue(window.source_pane.basic_cancel_button.isVisible())
        window.close()

    def test_basic_mode_run_uses_current_advanced_settings(self):
        window = ClusterGalleryApp()
        window.clustering_pane.cluster_spinbox.setValue(7)
        window.clustering_pane.embedding_cache_lookup_checkbox.setChecked(False)
        window.clustering_pane.tag_filter_field.setText("family")
        window.set_clustering_mode("basic")
        request = window._build_clustering_request()
        self.assertEqual(7, request.num_clusters)
        self.assertFalse(request.use_embedding_cache_lookup)
        self.assertFalse(request.generate_cluster_meanings)
        self.assertEqual("family", window.clustering_pane.tag_filter_field.text())
        window.close()

    def test_switching_back_to_advanced_preserves_in_session_advanced_state(self):
        window = ClusterGalleryApp()
        window.set_clustering_mode("advanced")
        APP.processEvents()
        window.clustering_pane.cluster_spinbox.setValue(9)
        window.details_toggle.setChecked(False)
        APP.processEvents()
        window.set_clustering_mode("basic")
        APP.processEvents()
        window.set_clustering_mode("advanced")
        APP.processEvents()
        self.assertEqual(9, window.clustering_pane.cluster_spinbox.value())
        self.assertFalse(window.details_toggle.isChecked())
        self.assertFalse(window.cluster_right_splitter.isVisible())
        window.close()

    def test_open_face_results_in_main_gallery_switches_workspace_and_captures_context(self):
        window = ClusterGalleryApp()
        window.show()
        APP.processEvents()
        window.set_active_workspace("faces")
        window.faces_pane.results_gallery.inspector_context_provider = lambda path: {"face_search": {"image_path": path}}

        window._open_face_results_in_main_gallery(["a.jpg", "b.jpg", "a.jpg"])

        self.assertEqual("clustering", window._active_workspace)
        self.assertEqual(["a.jpg", "b.jpg"], list(window.gallery_pane.images))
        self.assertEqual({"face_search": {"image_path": "a.jpg"}}, window._main_gallery_context_overrides["a.jpg"])
        self.assertEqual({}, window.gallery_pane.membership_context)
        window.close()

    def test_faces_workspace_does_not_inherit_clustering_scope(self):
        window = ClusterGalleryApp()

        self.assertIsNone(window.faces_pane.current_scope_paths_provider)
        self.assertEqual("Limit search to current folder", window.faces_pane.search_only_current_folder.text())

        window.close()

    def test_face_library_scan_and_refresh_ignore_hidden_candidate_scope(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        service = self._FakeFaceLibraryService(
            profiles=[self._person_profile("Alice", visible_face_count=1)],
            records=[
                self._face_record("/photos/a.jpg", 0),
                self._face_record("/photos/a.jpg", 1),
            ],
        )
        pane.face_service_global = service
        pane.current_scope_paths_provider = lambda: ["/hidden/selected.jpg"]
        pane.face_folder_path.setText("/photos")
        pane._start_job = lambda _label, run, done: done(run(lambda *_args: None, lambda: False))

        pane._scan_face_folder()

        self.assertTrue(
            self._wait_until(
                lambda: bool(service.load_folder_review_images_calls)
                and bool(service.load_person_profiles_calls)
                and pane.face_named_people_list.count() + pane.face_unlabeled_groups_list.count() == 2
            )
        )
        self.assertEqual([{"directory": "/photos", "recursive": True}], service.index_directory_calls)
        self.assertEqual([], service.index_paths_calls)
        self.assertEqual("/photos", service.load_folder_review_images_calls[-1]["directory"])
        self.assertEqual([], service.load_folder_review_images_calls[-1]["candidate_paths"])
        self.assertEqual("/photos", service.load_person_profiles_calls[-1]["folder_prefix"])
        self.assertIsNone(service.load_person_profiles_calls[-1]["candidate_paths"])
        self.assertEqual([], service.load_indexed_faces_calls)
        self.assertEqual([], service.count_indexed_faces_calls)
        self.assertEqual(1, pane.face_named_people_list.count())
        self.assertEqual(1, pane.face_unlabeled_groups_list.count())
        selected_photo_help = pane.findChild(HelpIconButton, "helpIcon_selected_photo_faces")
        self.assertIsNotNone(selected_photo_help)
        self.assertTrue(selected_photo_help.toolTip())
        self.assertIn("/photos", pane.face_model_summary_label.toolTip())
        self.assertTrue(pane.face_scan_button.toolTip())
        self.assertTrue(pane.face_refresh_faces_button.toolTip())

        pane.close()

    def test_face_library_unlabeled_group_click_selects_faces_and_updates_results(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        service = self._FakeFaceLibraryService(
            profiles=[self._person_profile("Alice", visible_face_count=1)],
            records=[
                self._face_record("/photos/a.jpg", 0),
                self._face_record("/photos/a.jpg", 1),
                self._face_record("/photos/b.jpg", 0, person_name="Alice"),
            ],
        )
        pane.face_service_global = service
        pane.face_folder_path.setText("/photos")

        pane._refresh_scanned_faces()
        pane._refresh_face_people()
        pane.face_label_name.setText("stale")
        pane.face_profile_notes.setText("stale")

        unlabeled_item = pane.face_unlabeled_groups_list.item(0)

        self.assertIsNotNone(unlabeled_item)
        self.assertEqual("Reload People List", pane.face_refresh_people_button.text())
        self.assertEqual(1, pane.face_named_people_list.count())

        pane._on_person_clicked(unlabeled_item)

        self.assertEqual(2, self._selected_list_view_count(pane.face_scanned_list))
        self.assertEqual(["/photos/a.jpg", "/photos/b.jpg"], list(pane.results_gallery.images))
        self.assertEqual("/photos/a.jpg", pane._face_review_selected_path)
        self.assertEqual("a.jpg", pane.face_selected_photo_label.text())
        self.assertEqual("", pane.face_label_name.text())
        self.assertEqual("", pane.face_profile_notes.text())
        self.assertIn("Selected 2 unlabeled face(s)", pane.face_scanned_summary.text())
        self.assertIn("Loaded unlabeled group", pane.face_people_summary.text())

        pane.close()

    def test_face_library_named_profile_click_still_populates_fields(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        service = self._FakeFaceLibraryService(
            profiles=[self._person_profile("Alice", visible_face_count=1, notes="saved note")],
            records=[self._face_record("/photos/b.jpg", 0, person_name="Alice")],
        )
        pane.face_service_global = service
        pane.face_folder_path.setText("/photos")

        pane._refresh_face_people()
        pane._on_person_clicked(pane.face_named_people_list.item(0))

        self.assertEqual("Alice", pane.face_name_query.text())
        self.assertEqual("Alice", pane.face_label_name.text())
        self.assertEqual("saved note", pane.face_profile_notes.text())
        self.assertEqual("friend", pane.face_profile_tags.text())
        self.assertIn("Loaded saved identity 'Alice'", pane.face_people_summary.text())

        pane.close()

    def test_face_selection_clears_stale_identity_when_no_face_is_selected(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        service = self._FakeFaceLibraryService(
            records=[self._face_record("/photos/a.jpg", 0, person_name="Milly")],
        )
        pane.face_service_global = service
        pane.face_folder_path.setText("/photos")

        pane._refresh_scanned_faces()
        self._select_list_view_row(pane.face_scanned_list, 0)

        self.assertEqual("Milly", pane.face_name_query.text())
        self.assertEqual("Milly", pane.face_label_name.text())
        pane.face_scanned_list.selectionModel().clearSelection()
        APP.processEvents()

        self.assertEqual("", pane.face_name_query.text())
        self.assertEqual("", pane.face_label_name.text())
        self.assertIn("selected 0", pane.face_selected_faces_context_label.text())
        pane.close()

    def test_detected_face_selection_focuses_its_source_photo_and_handles_multi_photo_selection(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        service = self._FakeFaceLibraryService(
            records=[
                self._face_record("/photos/a.jpg", 0, person_name="Milly"),
                self._face_record("/photos/b.jpg", 0, person_name="Zoe"),
            ],
        )
        pane.face_service_global = service
        pane.face_folder_path.setText("/photos")
        pane._refresh_scanned_faces()
        APP.processEvents()

        detected_model = pane.face_detected_faces_model
        rows_by_path = {
            detected_model.item_at(row).image_path: row
            for row in range(detected_model.rowCount())
            if detected_model.item_at(row) is not None
        }
        selected_b = detected_model.index(rows_by_path["/photos/b.jpg"], 0)
        selected_a = detected_model.index(rows_by_path["/photos/a.jpg"], 0)
        selection_model = pane.face_detected_faces_list.selectionModel()
        selection_model.select(selected_b, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        selection_model.setCurrentIndex(selected_b, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        APP.processEvents()

        self.assertEqual("/photos/b.jpg", pane._face_review_selected_path)
        self.assertEqual("b.jpg", pane.face_selected_photo_label.text())
        self.assertEqual(1, self._selected_list_view_count(pane.face_scanned_list))
        self.assertEqual("Zoe", pane.face_label_name.text())

        selection_model.select(selected_a, QItemSelectionModel.SelectionFlag.Select)
        APP.processEvents()

        self.assertEqual("/photos/b.jpg", pane._face_review_selected_path)
        self.assertEqual("b.jpg", pane.face_selected_photo_label.text())
        self.assertEqual(2, self._selected_list_view_count(pane.face_detected_faces_list))
        self.assertEqual(0, self._selected_list_view_count(pane.face_scanned_list))
        self.assertEqual("", pane.face_name_query.text())
        self.assertEqual("", pane.face_label_name.text())
        self.assertIn("2 selected faces · 2 photos", pane.face_selected_faces_context_label.text())
        pane.close()

    def test_face_workspace_guidance_and_action_labels_are_visible(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)

        self.assertEqual("Detect Faces", pane.face_scan_button.text())
        self.assertEqual("Reload People List", pane.face_refresh_people_button.text())
        self.assertEqual("Refresh Folder Review", pane.face_refresh_faces_button.text())
        self.assertEqual("Name", pane.face_review_sort.itemText(0))
        self.assertEqual("Detected Faces", pane.face_review_sort.itemText(1))
        self.assertEqual("Best Detected Face", pane.face_review_sort.itemText(2))
        self.assertEqual("Name Selected Face(s)", pane.face_save_name_button.text())
        self.assertEqual("Save/Update Profile", pane.face_save_profile_button.text())
        self.assertEqual("Find Photos of Selected Face", pane.face_search_selected_button.text())
        self.assertEqual("Find Photos by Saved Name", pane.face_search_by_name_button.text())
        self.assertEqual("Find Similar", pane.face_detected_find_button.text())
        self.assertEqual("Cluster Selected", pane.face_detected_cluster_selected_button.text())
        self.assertEqual("Cluster Visible", pane.face_detected_cluster_visible_button.text())
        self.assertEqual("Name Selected", pane.face_detected_name_button.text())
        self.assertEqual("Index Current Folder", pane.face_index_button.text())
        self.assertEqual("Name Selected Faces", pane.face_results_name_button.text())
        self.assertEqual("Name Selected Clusters", pane.face_results_name_clusters_button.text())
        self.assertEqual("Queue Suggested Identity", pane.face_results_queue_suggestion_button.text())
        self.assertEqual("Reject Suggestion", pane.face_results_reject_suggestion_button.text())
        self.assertEqual("Keep Cluster Unlabeled", pane.face_results_keep_unknown_button.text())
        self.assertEqual("Create New Cluster From Selected Faces", pane.face_results_split_button.text())
        self.assertEqual("Merge Selected Clusters", pane.face_results_merge_button.text())
        self.assertEqual("Review Pending Labels", pane.face_results_review_pending_button.text())
        self.assertEqual("Actions", pane.face_results_actions_menu_button.text())
        self.assertEqual("Advanced Actions", pane.face_results_advanced_actions_menu_button.text())
        self.assertFalse(pane.face_results_queue_suggestion_button.isVisible())
        self.assertFalse(pane.face_results_split_button.isVisible())
        self.assertEqual(
            [
                "Queue Suggested Identity",
                "Reject Suggestion",
                "Keep Cluster Unlabeled",
                "Show Photo",
                "Open Inspector",
                "Review Pending Labels",
                "Export Cluster Results",
            ],
            [action.text() for action in pane.face_results_actions_menu.actions() if not action.isSeparator()],
        )
        self.assertEqual(
            [
                "Create New Cluster From Selected Faces",
                "Merge Selected Clusters",
                "Recluster Selected Cluster",
                "Run Current Backend Again",
            ],
            [action.text() for action in pane.face_results_advanced_actions_menu.actions()],
        )
        self.assertTrue(pane.face_results_actions_menu_button.toolTip())
        self.assertTrue(pane.face_results_name_button.toolTip())
        self.assertEqual("Find Similar From Selection", pane.face_search_selected_card_button.text())
        self.assertEqual("Find by Face", pane.face_search_button.text())
        self.assertEqual("Find by Name + Similar", pane.face_find_name_button.text())

        self.assertEqual("Find Photos by Saved Name", pane.face_search_by_name_card_button.text())
        self.assertEqual("Save Named Examples", pane.face_label_button.text())
        self.assertEqual("Auto-Apply Saved Names", pane.face_propagate_button.text())
        self.assertEqual("Show Named Faces", pane.face_list_button.text())
        self.assertEqual("Merge Two Names", pane.face_merge_button.text())
        self.assertEqual("Remove Name From Faces", pane.face_clear_button.text())
        self.assertEqual(["hdbscan"], pane.current_face_cluster_backends())
        self.assertEqual("isolate", pane.current_face_cluster_outlier_policy())
        self.assertEqual("Advanced backend override", pane.face_cluster_backend_override_checkbox.text())
        self.assertEqual("Run Current Backend Again", pane.face_results_compare_again_button.text())
        self.assertEqual("Grouped Photos", pane.face_results_groups_label.text())
        self.assertEqual("Named Photos", pane.face_results_merged_groups_label.text())
        self.assertEqual(
            ["Named Photos", "Grouped Photos"],
            [pane.face_result_view_tabs.tabText(index) for index in range(pane.face_result_view_tabs.count())],
        )
        self.assertEqual("Show Tiny Detections", pane.show_tiny_detections_checkbox.text())
        self.assertGreaterEqual(pane.face_model_profile_combo.count(), 4)
        self.assertTrue(pane.face_detector_combo.toolTip())
        self.assertTrue(pane.face_embedder_combo.toolTip())
        self.assertGreaterEqual(pane.face_detector_combo.count(), 4)
        self.assertGreaterEqual(pane.face_embedder_combo.count(), 4)
        self.assertEqual("human", pane.current_face_mode())
        self.assertTrue(pane.face_search_quick_start_label.text())
        self.assertTrue(pane.face_merge_button.toolTip())
        self.assertTrue(pane.face_clear_button.toolTip())
        self.assertTrue(pane.show_tiny_detections_checkbox.toolTip())
        self.assertTrue(pane.face_mode_status_label.text())
        self.assertTrue(pane.face_model_summary_label.text())
        self.assertEqual("Advanced Pipeline…", pane.face_choose_pipeline_button.text())
        self.assertEqual("Install / Manage Face Models", pane.face_model_settings_button.text())
        self.assertEqual("primary", pane.face_scan_button.property("kind"))
        self.assertEqual("primary", pane.face_detected_name_button.property("kind"))
        self.assertEqual("primary", pane.face_results_name_button.property("kind"))
        self.assertEqual("secondary", pane.face_results_name_clusters_button.property("kind"))
        self.assertEqual("secondary", pane.face_choose_pipeline_button.property("kind"))
        self.assertEqual("secondary", pane.face_model_settings_button.property("kind"))
        self.assertEqual("primary", pane.face_search_button.property("kind"))
        self.assertEqual("primary", pane.face_find_name_button.property("kind"))
        self.assertEqual("facesTaskNavigation", pane.task_navigation.objectName())
        self.assertEqual("faceResultViewTabs", pane.face_result_view_tabs.objectName())
        self.assertIn("QTabBar#facesTaskNavigation::tab:selected", ULTRA_DARK_QSS)
        self.assertIn("QTabBar#faceResultViewTabs::tab:selected", ULTRA_DARK_QSS)
        self.assertIn('QPushButton[kind="secondary"]', ULTRA_DARK_QSS)
        self.assertFalse(hasattr(pane, "face_model_status_dashboard_label"))
        self.assertFalse(hasattr(pane, "face_model_details_button"))
        self.assertGreaterEqual(pane.face_detector_policy_combo.count(), 5)
        self.assertEqual("Face Groups", pane.face_review_results_tabs.tabText(2))
        self.assertGreaterEqual(pane.face_verifier_mode_combo.count(), 3)
        self.assertGreaterEqual(pane.face_search_quality_min_combo.count(), 3)
        self.assertGreaterEqual(pane.face_rerank_policy_combo.count(), 2)
        self.assertEqual(Qt.Orientation.Horizontal, pane.workspace_splitter.orientation())
        self.assertEqual(QSize(72, 72), pane.face_scanned_list.iconSize())
        self.assertIn(pane._face_library_layout_mode, {"compact", "wide"})
        self.assertIn(pane._face_search_layout_mode, {"compact", "wide"})
        self.assertIsInstance(pane.face_source_fields_grid, QVBoxLayout)
        self.assertEqual(0, pane.face_source_fields_grid.count())
        self.assertTrue(pane.face_db_scope.isHidden())
        self.assertTrue(pane.face_folder_override_row.isHidden())
        self.assertIsInstance(pane.face_scan_actions_grid, QGridLayout)
        self.assertIs(pane.face_scan_actions_grid.itemAtPosition(0, 0).widget(), pane.face_scan_button)
        self.assertIs(pane.face_scan_actions_grid.itemAtPosition(0, 1).widget(), pane.face_refresh_faces_button)
        self.assertGreater(len(pane.findChildren(HelpIconButton)), 0)
        self.assertNotEqual(-1, pane.face_quality_page_layout.indexOf(pane.face_review_quality_field))
        self.assertNotEqual(-1, pane.face_quality_page_layout.indexOf(pane.face_review_reason_field))
        self.assertIsInstance(pane.face_profile_fields_grid, QVBoxLayout)
        self.assertIsInstance(pane.face_find_fields_grid, QGridLayout)
        self.assertIsInstance(pane.face_find_name_fields_grid, QGridLayout)
        self.assertIs(pane.face_find_fields_grid.itemAtPosition(1, 0).widget(), pane.face_query_detect_button)
        self.assertIs(pane.face_find_fields_grid.itemAtPosition(1, 1).widget(), pane.face_index_button)
        self.assertIs(pane.face_find_fields_grid.itemAtPosition(4, 0).widget(), pane.face_search_button)
        self.assertIs(pane.face_find_name_fields_grid.itemAtPosition(0, 1).widget(), pane.face_find_name_button)
        self.assertIsInstance(pane.face_save_fields_grid, QVBoxLayout)
        self.assertIsInstance(pane.face_manage_fields_grid, QVBoxLayout)
        self.assertEqual(-1, pane.face_library_grid.indexOf(pane.face_people_group))
        self.assertNotEqual(-1, pane.face_library_selection_grid.indexOf(pane.face_people_group))
        self.assertNotEqual(-1, pane.face_library_selection_grid.indexOf(pane.face_profile_group))
        self.assertEqual(-1, pane.face_search_grid.indexOf(pane.face_people_group))
        self.assertEqual(-1, pane.face_search_grid.indexOf(pane.face_profile_group))
        self.assertTrue(pane.face_search_quick_group.isHidden())
        self.assertEqual(-1, pane.face_search_grid.indexOf(pane.face_search_quick_group))
        self.assertNotEqual(-1, pane.face_search_grid.indexOf(pane.face_search_selected_group))
        self.assertNotEqual(-1, pane.face_search_grid.indexOf(pane.face_people_query_group))
        self.assertEqual(["Detect Faces", "Review & Name"], [pane.face_library_tabs.tabText(index) for index in range(pane.face_library_tabs.count())])
        self.assertFalse(pane.face_library_tabs.tabBar().isTabVisible(1))
        self.assertIs(pane.face_library_grid.itemAtPosition(0, 0).widget(), pane.face_source_group)
        self.assertIs(pane.face_library_selection_grid.itemAtPosition(0, 0).widget(), pane.face_scanned_group)
        self.assertIs(pane.face_library_selection_grid.itemAtPosition(1, 0).widget(), pane.face_people_group)
        self.assertIs(pane.face_library_selection_grid.itemAtPosition(2, 0).widget(), pane.face_profile_group)

        pane.close()

    def test_all_faces_controls_are_compact_and_documented(self):
        pane = SearchPane(enabled_tabs=["All Faces", "Face Library", "Face Search"], external_results=False)
        try:
            self.assertEqual("All Faces", pane.tabs.tabText(0))
            self.assertFalse(hasattr(pane, "face_album_summary"))
            self.assertFalse(hasattr(pane, "face_album_scope_summary"))
            self.assertIsInstance(pane.face_album_help_button, HelpIconButton)
            self.assertTrue(pane.face_album_help_button.toolTip())
            self.assertTrue(pane.face_album_refresh_button.toolTip())
            self.assertTrue(pane.face_album_load_more_groups_button.toolTip())
            self.assertTrue(pane.face_album_load_more_faces_button.toolTip())
        finally:
            pane.close()

    def test_cuda_runtime_badge_stays_ready_when_application_health_needs_attention(self):
        badge = RuntimeBadge()
        try:
            capabilities = RuntimeCapabilities(
                torch_version="2.2.2+cu121",
                torch_cuda_build=True,
                torch_cuda_available=True,
                cuda_device_count=1,
                cuda_device_name="NVIDIA GeForce RTX 4090",
                onnx_available=True,
                onnx_version="1.18.0",
                onnx_providers=("CUDAExecutionProvider", "CPUExecutionProvider"),
            )
            policy = ExecutionPolicy(
                preferred_mode="cuda",
                effective_mode="cuda",
                torch_device="cuda",
                onnx_provider="CUDAExecutionProvider",
                reason="Using CUDA Torch on NVIDIA GeForce RTX 4090.",
            )
            badge.update_runtime(capabilities, policy)
            badge.set_health("warning", ["Crash report present"])

            self.assertIn("CUDA | NVIDIA GeForce RTX 4090", badge.text())
            self.assertTrue(badge.text().endswith("· Ready"))
            self.assertIn("do not affect GPU availability", badge.toolTip())
            self.assertIn("Crash report present", badge.toolTip())
        finally:
            badge.deleteLater()

    def test_find_by_name_uses_saved_identity_similarity_search(self):
        service = self._FakeFaceLibraryService(
            records=[self._face_record("/photos/alice.jpg", 0, person_name="")]
        )
        pane = SearchPane(
            enabled_tabs=["Face Library", "Face Search"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane.show()
        try:
            pane.face_find_name_query.setText("Alice")
            pane.face_find_name_button.click()

            self.assertTrue(
                self._wait_until(
                    lambda: bool(service.search_by_person_name_calls)
                    and bool(pane._face_result_groups)
                    and pane._face_result_groups[0].title == "Name + similar: Alice"
                )
            )
            self.assertEqual("Alice", service.search_by_person_name_calls[-1]["person_name"])
            self.assertIn("saved-name and similar-face", pane.status_label.text())
            self.assertIn("visually similar faces", pane.face_results_summary.text())
        finally:
            pane.close()

    def test_read_only_mode_allows_face_indexing_and_keeps_curated_edits_blocked(self):
        service = self._FakeFaceLibraryService()
        service.db_path = "/tmp/clusterlens-global-faces.db"
        pane = SearchPane(
            face_service_global=service,
            face_service_session=service,
            enabled_tabs=["Face Library", "Face Search"],
            external_results=False,
        )
        original_label_tooltip = pane.face_label_button.toolTip()

        pane.set_read_only_mode(True)
        APP.processEvents()

        self.assertTrue(pane.face_scan_button.isEnabled())
        self.assertTrue(pane.face_index_button.isEnabled())
        self.assertFalse(bool(pane.face_scan_button.property("mutationAction")))
        self.assertFalse(bool(pane.face_index_button.property("mutationAction")))
        self.assertFalse(bool(pane.face_rescan_selected_button.property("mutationAction")))
        self.assertFalse(pane.face_label_button.isEnabled())
        self.assertIn("read-only mode", pane.face_label_button.toolTip())
        self.assertFalse(hasattr(pane, "face_database_path_label"))

        # Later readiness refreshes must not bypass the read-only guard.
        pane._update_face_mode_status()
        self.assertFalse(pane.face_label_button.isEnabled())

        pane.set_read_only_mode(False)
        APP.processEvents()
        self.assertTrue(pane.face_label_button.isEnabled())
        self.assertEqual(original_label_tooltip, pane.face_label_button.toolTip())
        pane.close()

    def test_missing_face_model_scan_action_opens_cancellable_model_setup(self):
        service = self._FakeFaceLibraryService(
            ready=False,
            readiness_message="FaceNet weights are not installed.",
        )
        service.db_path = "/tmp/clusterlens-global-faces.db"
        pane = SearchPane(
            face_service_global=service,
            face_service_session=service,
            enabled_tabs=["Face Library", "Face Search"],
            external_results=False,
        )
        pane.face_folder_path.setText("/photos")
        install_requests: list[str] = []
        pane.install_face_model_requested.connect(install_requests.append)

        pane.set_read_only_mode(True)
        APP.processEvents()

        self.assertTrue(pane.face_scan_button.isEnabled())
        self.assertTrue(pane.face_index_button.isEnabled())
        self.assertIn("FaceNet weights are not installed", pane.face_scan_button.toolTip())
        pane._scan_face_folder()

        self.assertEqual([], service.index_directory_calls)
        self.assertEqual(["facenet"], install_requests)
        self.assertIn("face model setup required", pane.status_label.text().lower())
        pane.close()

    def test_main_cluster_faces_uses_selected_backend_and_hdbscan_options(self):
        service = self._FakeFacePipelineService(
            records=[
                self._face_record("/photos/a.jpg", 0, bbox=(0, 0, 40, 40), person_name="Alice"),
                self._face_record("/photos/b.jpg", 0, bbox=(0, 0, 40, 40), person_name="Bob"),
            ]
        )
        pane = SearchPane(
            enabled_tabs=["Face Library", "Face Search"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane._start_job = lambda _label, run, done: done(run(lambda *_args: None, lambda: False))
        emitted: list[dict] = []
        pane.face_clusters_ready.connect(lambda payload: emitted.append(dict(payload)))
        try:
            pane.face_cluster_backend.setCurrentIndex(pane.face_cluster_backend.findData("graph"))
            pane.face_hdbscan_min_cluster_size_spin.setValue(7)
            pane.face_hdbscan_min_samples_spin.setValue(3)
            pane.face_hdbscan_cluster_selection_epsilon_spin.setValue(0.14)
            pane.face_hdbscan_allow_single_cluster_checkbox.setChecked(True)
            pane._cluster_faces()
            self.assertEqual(["hdbscan"], service.cluster_faces_compare_calls[-1]["backends"])
            self.assertEqual(
                {
                    "hdbscan": {
                        "min_cluster_size": 7,
                        "min_samples": 3,
                        "cluster_selection_epsilon": 0.14,
                        "allow_single_cluster": True,
                    }
                },
                service.cluster_faces_compare_calls[-1]["backend_options_by_backend"],
            )
            self.assertIn("hdbscan", emitted[-1])
            self.assertIn("HDBSCAN", pane.status_label.text())
        finally:
            pane.close()

    def test_face_cluster_backend_override_allows_non_hdbscan_backend(self):
        service = self._FakeFacePipelineService(
            records=[
                self._face_record("/photos/a.jpg", 0, bbox=(0, 0, 40, 40), person_name="Alice"),
                self._face_record("/photos/b.jpg", 0, bbox=(0, 0, 40, 40), person_name="Bob"),
            ]
        )
        pane = SearchPane(
            enabled_tabs=["Face Library", "Face Search"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane._start_job = lambda _label, run, done: done(run(lambda *_args: None, lambda: False))
        try:
            pane.face_cluster_backend_override_checkbox.setChecked(True)
            pane.face_cluster_backend.setCurrentIndex(pane.face_cluster_backend.findData("graph"))
            pane._cluster_faces()
            self.assertEqual(["graph"], service.cluster_faces_compare_calls[-1]["backends"])
            self.assertEqual({}, service.cluster_faces_compare_calls[-1]["backend_options_by_backend"])
            self.assertIn("Graph", pane.status_label.text())
        finally:
            pane.close()

    def test_animal_provider_failure_does_not_fall_back_to_human_face_service(self):
        human_service = self._FakeFacePipelineService()
        dog_service = self._FakeFacePipelineService(ready=False, readiness_message="Dog model missing.")
        dog_service.mode = "dog"
        pane = SearchPane(
            enabled_tabs=["Face Library", "Face Search"],
            external_results=False,
            face_service_global=human_service,
            face_service_session=human_service,
            face_services_global={"dog": dog_service},
            face_services_session={"dog": dog_service},
        )
        pane.face_service_provider = lambda *_args: (_ for _ in ()).throw(RuntimeError("provider failed"))
        try:
            pane.set_active_face_mode("dog", refresh=False)
            service = pane._active_face_service()
            self.assertIs(service, dog_service)
            self.assertIsNot(service, human_service)
            self.assertEqual([], human_service.index_paths_calls)
            self.assertEqual([], human_service.load_indexed_faces_calls)
        finally:
            pane.close()

    def test_profile_controls_save_favorite_birth_date_hidden_and_identity_summary(self):
        service = self._FakeFacePipelineService(
            profiles=[
                self._person_profile(
                    "Alice",
                    visible_face_count=1,
                    cover_image_path="/photos/a.jpg",
                    cover_face_index=0,
                    favorite=True,
                    birth_date="2001-02-03",
                    hidden=True,
                )
            ],
            records=[self._face_record("/photos/a.jpg", 0, person_name="Alice")],
            prototype_faces_by_name={
                "Alice": [
                    PersonPrototypeFace(
                        person_name="Alice",
                        image_path="/photos/a.jpg",
                        face_index=0,
                        face_bbox=(10, 12, 42, 54),
                        face_confidence=0.97,
                        pinned=True,
                    )
                ]
            },
        )
        pane = SearchPane(
            enabled_tabs=["Face Library", "Face Search", "Identities"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane._start_job = lambda _label, run, done: done(run(lambda *_args: None, lambda: False))
        try:
            pane._apply_face_people_data(service.profiles, [], folder="/photos", include_tiny_faces=True)
            pane._on_person_clicked(pane.face_named_people_list.item(0))
            self.assertTrue(pane.face_profile_favorite.isChecked())
            self.assertTrue(pane.face_profile_hidden.isChecked())
            self.assertEqual("2001-02-03", pane.face_profile_birth_date.text())
            pane.face_profile_notes.setText("updated")
            pane.face_profile_favorite.setChecked(False)
            pane.face_profile_hidden.setChecked(True)
            pane.face_profile_birth_date.setText("2001-02-04")
            pane._save_person_profile()
            self.assertEqual("Alice", service.save_person_profile_calls[-1]["person_name"])
            self.assertFalse(service.save_person_profile_calls[-1]["favorite"])
            self.assertEqual("2001-02-04", service.save_person_profile_calls[-1]["birth_date"])
            self.assertTrue(service.save_person_profile_calls[-1]["hidden"])

            pane._refresh_face_identities(force_reload=True)
            self._select_list_view_row(pane.face_identity_list, 0)
            pane._refresh_selected_face_identity()
            self.assertIn("DOB 2001-02-03", pane.face_identity_summary.text())
            self.assertIn("age at cover 24", pane.face_identity_summary.text())
            self.assertIn("[Favorite]", self._list_view_text(pane.face_identity_list, 0))
        finally:
            pane.close()

    def test_face_cluster_compare_results_show_details_and_membership(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        pane.show()
        try:
            result = FaceClusteringComparisonResult(
                clusters_by_key={
                    "cosine-kmeans": {
                        0: [
                            FaceClusterMember("/photos/a.jpg", 0, (10, 10, 50, 50), 0.98, "Alice", "clean"),
                            FaceClusterMember("/photos/b.jpg", 0, (12, 12, 48, 48), 0.96, "", "clean"),
                        ]
                    }
                },
                membership_by_face_ref={
                    ("/photos/a.jpg", 0): {
                        "cosine-kmeans": {
                            "backend": "cosine-kmeans",
                            "cluster_id": 0,
                            "cluster_size": 2,
                            "rank": 1,
                            "outlier": False,
                            "cluster_quality_score": 0.64,
                        }
                    }
                },
                metrics_by_key={"cosine-kmeans": {"requested_backend": "cosine-kmeans", "cluster_quality_score": 0.64}},
                explanations_by_key={
                    "cosine-kmeans": {
                        0: ClusterExplanation(
                            cluster_id=0,
                            cluster_size=2,
                            is_outlier=False,
                            cohesion_mean=0.91,
                            cohesion_min=0.86,
                            nearest_cluster_id=1,
                            separation_margin=0.12,
                            cluster_quality_score=0.64,
                        )
                    }
                },
                identity_suggestions_by_key={
                    "cosine-kmeans": {
                        0: FaceClusterIdentitySuggestion("Alice", 2, 2, 0.92, 0.88, 0.95)
                    }
                },
            )
            pane._show_face_cluster_results(result, summary="Compared 1 backend.", compare_refs=[("/photos/a.jpg", 0), ("/photos/b.jpg", 0)])
            self.assertTrue(self._wait_until(lambda: self._list_view_count(pane.face_results_groups_list) == 1))
            self.assertTrue(self._wait_until(lambda: self._list_view_count(pane.face_results_merged_groups_list) == 1))

            self.assertEqual(1, self._list_view_count(pane.face_results_groups_list))
            self.assertEqual(1, self._list_view_count(pane.face_results_merged_groups_list))
            self.assertIn("Cosine KMeans", self._list_view_text(pane.face_results_groups_list, 0))
            self.assertIn("Alice", self._list_view_text(pane.face_results_merged_groups_list, 0))
            self.assertIn("Likely identity: Alice", pane.face_results_cluster_suggestion.text())
            self.assertIn("Quality 0.640", pane.face_results_cluster_explanation.text())
            pane.face_results_list.selectionModel().select(
                pane.face_results_model.index(0, 0),
                QItemSelectionModel.SelectionFlag.ClearAndSelect,
            )
            APP.processEvents()
            self.assertEqual(1, pane.face_results_membership_table.rowCount())
            self.assertTrue(pane.face_results_recluster_button.isEnabled())
            self.assertTrue(pane.face_results_compare_again_button.isEnabled())
            self.assertTrue(pane.face_results_queue_suggestion_button.isEnabled())
            self.assertTrue(pane.face_results_review_pending_button.isEnabled())
        finally:
            pane.close()

    def test_grouped_photos_lists_every_saved_name_in_a_cluster(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        pane.show()
        try:
            result = FaceClusteringComparisonResult(
                clusters_by_key={
                    "hdbscan": {
                        4: [
                            FaceClusterMember("/photos/alice.jpg", 0, (10, 10, 50, 50), 0.98, "Alice", "clean"),
                            FaceClusterMember("/photos/bob.jpg", 0, (12, 12, 48, 48), 0.96, "Bob", "clean"),
                        ]
                    }
                },
                membership_by_face_ref={},
                metrics_by_key={"hdbscan": {"requested_backend": "hdbscan"}},
                explanations_by_key={},
                identity_suggestions_by_key={},
            )
            pane._show_face_cluster_results(result, summary="Two named faces.")

            self.assertTrue(self._wait_until(lambda: self._list_view_count(pane.face_results_groups_list) == 1))
            self.assertIn("Alice, Bob", self._list_view_text(pane.face_results_groups_list, 0))
        finally:
            pane.close()

    def test_face_result_cluster_name_writes_labels_exif_and_refreshes_named_groups(self):
        service = self._FakeFacePipelineService(
            records=[
                self._face_record("/photos/a.jpg", 0, bbox=(10, 10, 50, 50), person_name=""),
                self._face_record("/photos/b.jpg", 0, bbox=(12, 12, 48, 48), person_name=""),
            ]
        )
        pane = SearchPane(
            enabled_tabs=["Face Library", "Face Search"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane._start_job = lambda _label, run, done: done(run(lambda *_args: None, lambda: False))
        pane._refresh_pending_face_labels = lambda *args, **kwargs: None
        pane.refresh_face_library = lambda *args, **kwargs: None
        pane._maybe_refresh_global_face_album = lambda *args, **kwargs: None
        pane._refresh_face_identities = lambda *args, **kwargs: None
        exif_calls: list[dict[str, object]] = []

        class _FakeActionService:
            def write_exif_metadata_pairs(self, image_paths, key, value, progress_callback=None, cancel_check=None):
                _ = (progress_callback, cancel_check)
                exif_calls.append(
                    {
                        "image_paths": list(image_paths),
                        "key": str(key),
                        "value": str(value),
                    }
                )
                return SimpleNamespace(affected_paths=list(image_paths), failures=[])

        pane.results_gallery.action_service = _FakeActionService()
        pane.show()
        try:
            result = FaceClusteringComparisonResult(
                clusters_by_key={
                    "hdbscan": {
                        0: [
                            FaceClusterMember("/photos/a.jpg", 0, (10, 10, 50, 50), 0.98, "", "clean"),
                            FaceClusterMember("/photos/b.jpg", 0, (12, 12, 48, 48), 0.96, "", "clean"),
                        ]
                    }
                },
                membership_by_face_ref={},
                metrics_by_key={"hdbscan": {"requested_backend": "hdbscan", "cluster_quality_score": 0.64}},
                explanations_by_key={},
                identity_suggestions_by_key={},
            )
            pane._show_face_cluster_results(result, summary="Compared 1 backend.", compare_refs=[("/photos/a.jpg", 0), ("/photos/b.jpg", 0)])
            self.assertTrue(self._wait_until(lambda: self._list_view_count(pane.face_results_groups_list) == 1))
            self.assertEqual(0, self._list_view_count(pane.face_results_merged_groups_list))

            with patch("ui.search_pane.QInputDialog.getText", return_value=("Charlie", True)):
                pane._name_selected_face_result_group_immediately()

            self.assertEqual("Charlie", service.records[0].person_name)
            self.assertEqual("Charlie", service.records[1].person_name)
            self.assertEqual(
                [
                    {
                        "image_paths": ["/photos/a.jpg", "/photos/b.jpg"],
                        "key": "ic_person_name",
                        "value": "Charlie",
                    }
                ],
                exif_calls,
            )
            self.assertTrue(self._wait_until(lambda: self._list_view_count(pane.face_results_merged_groups_list) == 1))
            self.assertTrue(self._wait_until(lambda: "Charlie" in self._list_view_text(pane.face_results_groups_list, 0)))
            self.assertIn("Cluster 0", self._list_view_text(pane.face_results_groups_list, 0))
            self.assertIn("Charlie", self._list_view_text(pane.face_results_groups_list, 0))
            self.assertIn("Charlie", self._list_view_text(pane.face_results_merged_groups_list, 0))
            self.assertEqual("Charlie", pane.person_name.text())
        finally:
            pane.close()

    def test_face_result_bulk_cluster_name_writes_labels_exif_and_refreshes_all_raw_groups(self):
        service = self._FakeFacePipelineService(
            records=[
                self._face_record("/photos/a.jpg", 0, bbox=(10, 10, 50, 50), person_name=""),
                self._face_record("/photos/b.jpg", 0, bbox=(12, 12, 48, 48), person_name=""),
                self._face_record("/photos/c.jpg", 0, bbox=(14, 14, 46, 46), person_name=""),
                self._face_record("/photos/d.jpg", 0, bbox=(16, 16, 44, 44), person_name=""),
            ]
        )
        pane = SearchPane(
            enabled_tabs=["Face Library", "Face Search"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane._start_job = lambda _label, run, done: done(run(lambda *_args: None, lambda: False))
        pane.refresh_face_library = lambda *args, **kwargs: None
        pane._maybe_refresh_global_face_album = lambda *args, **kwargs: None
        pane._refresh_face_identities = lambda *args, **kwargs: None
        exif_calls: list[dict[str, object]] = []

        class _FakeActionService:
            def write_exif_metadata_pairs(self, image_paths, key, value, progress_callback=None, cancel_check=None):
                _ = (progress_callback, cancel_check)
                exif_calls.append(
                    {
                        "image_paths": list(image_paths),
                        "key": str(key),
                        "value": str(value),
                    }
                )
                return SimpleNamespace(affected_paths=list(image_paths), failures=[])

        pane.results_gallery.action_service = _FakeActionService()
        pane.show()
        try:
            result = FaceClusteringComparisonResult(
                clusters_by_key={
                    "hdbscan": {
                        0: [
                            FaceClusterMember("/photos/a.jpg", 0, (10, 10, 50, 50), 0.98, "", "clean"),
                            FaceClusterMember("/photos/b.jpg", 0, (12, 12, 48, 48), 0.96, "", "clean"),
                        ],
                        1: [
                            FaceClusterMember("/photos/c.jpg", 0, (14, 14, 46, 46), 0.95, "", "clean"),
                            FaceClusterMember("/photos/d.jpg", 0, (16, 16, 44, 44), 0.94, "", "clean"),
                        ],
                    }
                },
                membership_by_face_ref={},
                metrics_by_key={"hdbscan": {"requested_backend": "hdbscan", "cluster_quality_score": 0.64}},
                explanations_by_key={},
                identity_suggestions_by_key={},
            )
            pane._show_face_cluster_results(
                result,
                summary="Compared 1 backend.",
                compare_refs=[("/photos/a.jpg", 0), ("/photos/b.jpg", 0), ("/photos/c.jpg", 0), ("/photos/d.jpg", 0)],
            )
            self.assertTrue(self._wait_until(lambda: self._list_view_count(pane.face_results_groups_list) == 2))

            selection_model = pane.face_results_groups_list.selectionModel()
            first_index = pane.face_results_groups_model.index(0, 0)
            second_index = pane.face_results_groups_model.index(1, 0)
            selection_model.select(first_index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
            pane.face_results_groups_list.setCurrentIndex(first_index)
            selection_model.select(second_index, QItemSelectionModel.SelectionFlag.Select)
            APP.processEvents()

            self.assertEqual(2, self._selected_list_view_count(pane.face_results_groups_list))
            self.assertTrue(pane.face_results_name_clusters_button.isEnabled())

            with patch("ui.search_pane.QInputDialog.getText", return_value=("Charlie", True)):
                pane._name_selected_face_result_groups_immediately()

            self.assertEqual(1, len(service.label_indexed_faces_immediately_calls))
            self.assertEqual("Charlie", service.label_indexed_faces_immediately_calls[0]["person_name"])
            self.assertEqual(
                [("/photos/a.jpg", 0), ("/photos/b.jpg", 0), ("/photos/c.jpg", 0), ("/photos/d.jpg", 0)],
                service.label_indexed_faces_immediately_calls[0]["face_refs"],
            )
            self.assertTrue(all(record.person_name == "Charlie" for record in service.records[:4]))
            self.assertEqual(
                [
                    {
                        "image_paths": ["/photos/a.jpg", "/photos/b.jpg", "/photos/c.jpg", "/photos/d.jpg"],
                        "key": "ic_person_name",
                        "value": "Charlie",
                    }
                ],
                exif_calls,
            )
            self.assertTrue(self._wait_until(lambda: self._list_view_count(pane.face_results_merged_groups_list) == 1))
            self.assertIn("Charlie", self._list_view_text(pane.face_results_groups_list, 0))
            self.assertIn("Charlie", self._list_view_text(pane.face_results_groups_list, 1))
            self.assertIn("Charlie", self._list_view_text(pane.face_results_merged_groups_list, 0))
        finally:
            pane.close()

    def test_face_result_name_selected_button_still_labels_selected_tiles_not_raw_groups(self):
        service = self._FakeFacePipelineService(
            records=[
                self._face_record("/photos/a.jpg", 0, bbox=(10, 10, 50, 50), person_name=""),
                self._face_record("/photos/b.jpg", 0, bbox=(12, 12, 48, 48), person_name=""),
            ]
        )
        pane = SearchPane(
            enabled_tabs=["Face Library", "Face Search"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane._start_job = lambda _label, run, done: done(run(lambda *_args: None, lambda: False))
        pane._refresh_pending_face_labels = lambda *args, **kwargs: None
        pane.refresh_face_library = lambda *args, **kwargs: None
        pane._maybe_refresh_global_face_album = lambda *args, **kwargs: None
        pane.show()
        try:
            result = FaceClusteringComparisonResult(
                clusters_by_key={
                    "hdbscan": {
                        0: [
                            FaceClusterMember("/photos/a.jpg", 0, (10, 10, 50, 50), 0.98, "", "clean"),
                            FaceClusterMember("/photos/b.jpg", 0, (12, 12, 48, 48), 0.96, "", "clean"),
                        ]
                    }
                },
                membership_by_face_ref={},
                metrics_by_key={"hdbscan": {"requested_backend": "hdbscan", "cluster_quality_score": 0.64}},
                explanations_by_key={},
                identity_suggestions_by_key={},
            )
            pane._show_face_cluster_results(result, summary="Compared 1 backend.", compare_refs=[("/photos/a.jpg", 0), ("/photos/b.jpg", 0)])
            self.assertTrue(self._wait_until(lambda: self._list_view_count(pane.face_results_list) == 2))

            tile_index = pane.face_results_model.index(0, 0)
            pane.face_results_list.selectionModel().select(
                tile_index,
                QItemSelectionModel.SelectionFlag.ClearAndSelect,
            )
            APP.processEvents()

            self.assertTrue(pane.face_results_name_button.isEnabled())
            with patch("ui.search_pane.QInputDialog.getText", return_value=("Dana", True)):
                pane._prepare_name_selected_face_results()

            self.assertEqual(0, len(service.label_indexed_faces_calls))
            self.assertEqual(1, len(service.label_indexed_faces_immediately_calls))
            self.assertEqual("Dana", service.label_indexed_faces_immediately_calls[0]["person_name"])
            self.assertEqual([("/photos/a.jpg", 0)], service.label_indexed_faces_immediately_calls[0]["face_refs"])
            self.assertEqual("manual_selected_faces", service.label_indexed_faces_immediately_calls[0]["source"])
            self.assertIn("Saved 'Dana' on 1 face(s) from Grouped Photos.", pane.status_label.text())
        finally:
            pane.close()

    def test_face_result_tile_context_menu_preserves_multi_selection_and_exposes_batch_naming(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        pane.show()
        try:
            result = FaceClusteringComparisonResult(
                clusters_by_key={
                    "hdbscan": {
                        0: [
                            FaceClusterMember("/photos/a.jpg", 0, (10, 10, 50, 50), 0.98, "", "clean"),
                            FaceClusterMember("/photos/b.jpg", 0, (12, 12, 48, 48), 0.96, "", "clean"),
                        ]
                    }
                },
                membership_by_face_ref={},
                metrics_by_key={"hdbscan": {"requested_backend": "hdbscan", "cluster_quality_score": 0.64}},
                explanations_by_key={},
                identity_suggestions_by_key={},
            )
            pane._show_face_cluster_results(result, summary="Compared 1 backend.", compare_refs=[("/photos/a.jpg", 0), ("/photos/b.jpg", 0)])
            self.assertTrue(self._wait_until(lambda: self._list_view_count(pane.face_results_list) == 2))

            selection_model = pane.face_results_list.selectionModel()
            first_index = pane.face_results_model.index(0, 0)
            second_index = pane.face_results_model.index(1, 0)
            selection_model.select(first_index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
            selection_model.select(second_index, QItemSelectionModel.SelectionFlag.Select)
            APP.processEvents()

            point = pane.face_results_list.visualRect(first_index).center()
            returned_index = pane._prepare_face_result_context_selection(point)
            menu = pane._build_face_results_context_menu()

            self.assertTrue(returned_index.isValid())
            self.assertEqual(2, self._selected_list_view_count(pane.face_results_list))
            self.assertEqual(["Name Selected Faces…"], [action.text() for action in menu.actions()])
            self.assertTrue(menu.actions()[0].isEnabled())
            menu.deleteLater()
        finally:
            pane.close()

    def test_prepare_face_result_group_context_selection_preserves_multi_selection_for_selected_row(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        pane.show()
        try:
            result = FaceClusteringComparisonResult(
                clusters_by_key={
                    "hdbscan": {
                        0: [FaceClusterMember("/photos/a.jpg", 0, (10, 10, 50, 50), 0.98, "", "clean")],
                        1: [FaceClusterMember("/photos/b.jpg", 0, (12, 12, 48, 48), 0.96, "", "clean")],
                    }
                },
                membership_by_face_ref={},
                metrics_by_key={"hdbscan": {"requested_backend": "hdbscan", "cluster_quality_score": 0.64}},
                explanations_by_key={},
                identity_suggestions_by_key={},
            )
            pane._show_face_cluster_results(result, summary="Compared 1 backend.", compare_refs=[("/photos/a.jpg", 0), ("/photos/b.jpg", 0)])
            self.assertTrue(self._wait_until(lambda: self._list_view_count(pane.face_results_groups_list) == 2))

            selection_model = pane.face_results_groups_list.selectionModel()
            first_index = pane.face_results_groups_model.index(0, 0)
            second_index = pane.face_results_groups_model.index(1, 0)
            selection_model.select(first_index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
            pane.face_results_groups_list.setCurrentIndex(first_index)
            selection_model.select(second_index, QItemSelectionModel.SelectionFlag.Select)
            APP.processEvents()

            selected_before = self._selected_list_view_count(pane.face_results_groups_list)
            point = pane.face_results_groups_list.visualRect(first_index).center()
            returned_index = pane._prepare_face_result_group_context_selection(point)
            APP.processEvents()

            self.assertTrue(returned_index.isValid())
            self.assertEqual(0, returned_index.row())
            self.assertEqual(selected_before, self._selected_list_view_count(pane.face_results_groups_list))
            self.assertEqual(2, self._selected_list_view_count(pane.face_results_groups_list))
        finally:
            pane.close()

    def test_face_result_hover_preview_popup_shows_for_raw_and_named_groups(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        pane.show()
        try:
            result = FaceClusteringComparisonResult(
                clusters_by_key={
                    "cosine-kmeans": {
                        0: [
                            FaceClusterMember("/photos/a.jpg", 0, (10, 10, 50, 50), 0.98, "Alice", "clean"),
                            FaceClusterMember("/photos/b.jpg", 0, (12, 12, 48, 48), 0.96, "", "clean"),
                        ]
                    }
                },
                membership_by_face_ref={},
                metrics_by_key={"cosine-kmeans": {"requested_backend": "cosine-kmeans", "cluster_quality_score": 0.64}},
                explanations_by_key={},
                identity_suggestions_by_key={},
            )
            pane._show_face_cluster_results(result, summary="Compared 1 backend.", compare_refs=[("/photos/a.jpg", 0), ("/photos/b.jpg", 0)])
            self.assertTrue(self._wait_until(lambda: self._list_view_count(pane.face_results_groups_list) == 1))
            self.assertTrue(self._wait_until(lambda: self._list_view_count(pane.face_results_merged_groups_list) == 1))

            raw_index = pane.face_results_groups_model.index(0, 0)
            pane._update_face_result_hover_popup_for_index("raw", pane.face_results_groups_list, raw_index)
            self.assertTrue(pane._face_result_hover_popup.isVisible())
            self.assertIn("Cluster 0", pane._face_result_hover_popup.title_label.text())

            named_index = pane.face_results_merged_groups_model.index(0, 0)
            pane._update_face_result_hover_popup_for_index("merged_name", pane.face_results_merged_groups_list, named_index)
            self.assertTrue(pane._face_result_hover_popup.isVisible())
            self.assertIn("Alice", pane._face_result_hover_popup.title_label.text())
        finally:
            pane.close()

    def test_face_result_recluster_uses_selected_subset_when_available(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        pane.show()
        try:
            result = FaceClusteringComparisonResult(
                clusters_by_key={
                    "cosine-kmeans": {
                        0: [
                            FaceClusterMember("/photos/a.jpg", 0, (10, 10, 50, 50), 0.98, "Alice", "clean"),
                            FaceClusterMember("/photos/b.jpg", 0, (12, 12, 48, 48), 0.96, "", "clean"),
                            FaceClusterMember("/photos/c.jpg", 0, (14, 14, 46, 46), 0.95, "", "clean"),
                        ]
                    }
                },
                membership_by_face_ref={},
                metrics_by_key={"cosine-kmeans": {"requested_backend": "cosine-kmeans", "cluster_quality_score": 0.64}},
                explanations_by_key={},
                identity_suggestions_by_key={},
            )
            pane._show_face_cluster_results(result, summary="Compared 1 backend.", compare_refs=[("/photos/a.jpg", 0), ("/photos/b.jpg", 0), ("/photos/c.jpg", 0)])
            self.assertTrue(self._wait_until(lambda: self._list_view_count(pane.face_results_groups_list) == 1))

            selection_model = pane.face_results_list.selectionModel()
            selection_model.select(
                pane.face_results_model.index(0, 0),
                QItemSelectionModel.SelectionFlag.ClearAndSelect,
            )
            selection_model.select(
                pane.face_results_model.index(1, 0),
                QItemSelectionModel.SelectionFlag.Select,
            )
            APP.processEvents()

            captured: dict[str, object] = {}

            def _capture(refs, *, source_label):
                captured["refs"] = list(refs)
                captured["source_label"] = str(source_label)

            pane._cluster_face_refs = _capture
            pane._recluster_selected_face_result_group()

            self.assertEqual([("/photos/a.jpg", 0), ("/photos/b.jpg", 0)], captured["refs"])
            self.assertIn("selected subset", str(captured["source_label"]))
        finally:
            pane.close()

    def test_face_result_can_queue_suggestion_and_open_pending_review(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        pane.show()
        try:
            result = FaceClusteringComparisonResult(
                clusters_by_key={
                    "cosine-kmeans": {
                        0: [
                            FaceClusterMember("/photos/a.jpg", 0, (10, 10, 50, 50), 0.98, "Alice", "clean"),
                            FaceClusterMember("/photos/b.jpg", 0, (12, 12, 48, 48), 0.96, "", "clean"),
                        ]
                    }
                },
                membership_by_face_ref={},
                metrics_by_key={"cosine-kmeans": {"requested_backend": "cosine-kmeans", "cluster_quality_score": 0.64}},
                explanations_by_key={},
                identity_suggestions_by_key={
                    "cosine-kmeans": {
                        0: FaceClusterIdentitySuggestion("Alice", 2, 2, 0.92, 0.88, 0.95)
                    }
                },
            )
            pane._show_face_cluster_results(result, summary="Compared 1 backend.", compare_refs=[("/photos/a.jpg", 0), ("/photos/b.jpg", 0)])
            self.assertTrue(self._wait_until(lambda: self._list_view_count(pane.face_results_groups_list) == 1))

            captured: dict[str, object] = {}

            def _capture_save(refs, *, source_label, person_name_override=None):
                captured["refs"] = list(refs)
                captured["source_label"] = str(source_label)
                captured["person_name_override"] = person_name_override

            refresh_calls: list[bool] = []

            pane._save_face_refs_name = _capture_save
            pane._refresh_pending_face_labels = lambda: refresh_calls.append(True)
            pane.tabs.setCurrentIndex(0)

            pane._queue_selected_face_result_group_suggestion()
            self.assertEqual([("/photos/a.jpg", 0), ("/photos/b.jpg", 0)], captured["refs"])
            self.assertEqual("Alice", captured["person_name_override"])
            self.assertIn("suggestion", str(captured["source_label"]))

            pane._open_face_result_pending_review()
            self.assertEqual("Face Search", pane.tabs.tabText(pane.tabs.currentIndex()))
            self.assertTrue(pane.face_pending_items_toggle.isChecked())
            self.assertEqual(1, len(refresh_calls))
        finally:
            pane.close()

    def test_face_result_can_reject_suggestion_and_mark_unknown(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        pane.show()
        try:
            result = FaceClusteringComparisonResult(
                clusters_by_key={
                    "cosine-kmeans": {
                        0: [
                            FaceClusterMember("/photos/a.jpg", 0, (10, 10, 50, 50), 0.98, "Alice", "clean"),
                            FaceClusterMember("/photos/b.jpg", 0, (12, 12, 48, 48), 0.96, "", "clean"),
                        ]
                    }
                },
                membership_by_face_ref={},
                metrics_by_key={"cosine-kmeans": {"requested_backend": "cosine-kmeans", "cluster_quality_score": 0.64}},
                explanations_by_key={},
                identity_suggestions_by_key={
                    "cosine-kmeans": {
                        0: FaceClusterIdentitySuggestion("Alice", 2, 2, 0.92, 0.88, 0.95)
                    }
                },
            )
            pane._show_face_cluster_results(result, summary="Compared 1 backend.", compare_refs=[("/photos/a.jpg", 0), ("/photos/b.jpg", 0)])
            self.assertTrue(self._wait_until(lambda: self._list_view_count(pane.face_results_groups_list) == 1))

            pane._reject_selected_face_result_group_suggestion()
            self.assertTrue(self._wait_until(lambda: "dismissed" in pane.face_results_cluster_suggestion.text().lower()))
            self.assertIn("dismissed", pane.face_results_cluster_suggestion.text().lower())
            self.assertFalse(pane.face_results_queue_suggestion_button.isEnabled())

            pane._mark_selected_face_result_group_unknown()
            self.assertTrue(self._wait_until(lambda: "intentionally kept unlabeled" in pane.face_results_cluster_suggestion.text().lower()))
            self.assertIn("intentionally kept unlabeled", pane.face_results_cluster_suggestion.text().lower())
            self.assertIn("kept unlabeled", pane.face_results_cluster_summary.text().lower())
        finally:
            pane.close()

    def test_face_result_can_split_and_merge_cluster_groups(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        pane.show()
        try:
            result = FaceClusteringComparisonResult(
                clusters_by_key={
                    "cosine-kmeans": {
                        0: [
                            FaceClusterMember("/photos/a.jpg", 0, (10, 10, 50, 50), 0.98, "Alice", "clean"),
                            FaceClusterMember("/photos/b.jpg", 0, (12, 12, 48, 48), 0.96, "", "clean"),
                            FaceClusterMember("/photos/c.jpg", 0, (14, 14, 46, 46), 0.95, "", "clean"),
                        ],
                        1: [
                            FaceClusterMember("/photos/d.jpg", 0, (16, 16, 44, 44), 0.94, "", "clean"),
                        ],
                    }
                },
                membership_by_face_ref={},
                metrics_by_key={"cosine-kmeans": {"requested_backend": "cosine-kmeans", "cluster_quality_score": 0.64}},
                explanations_by_key={},
                identity_suggestions_by_key={},
            )
            pane._show_face_cluster_results(result, summary="Compared 1 backend.", compare_refs=[("/photos/a.jpg", 0), ("/photos/b.jpg", 0), ("/photos/c.jpg", 0), ("/photos/d.jpg", 0)])
            self.assertTrue(self._wait_until(lambda: self._list_view_count(pane.face_results_groups_list) == 2))

            selection_model = pane.face_results_list.selectionModel()
            selection_model.select(
                pane.face_results_model.index(0, 0),
                QItemSelectionModel.SelectionFlag.ClearAndSelect,
            )
            selection_model.select(
                pane.face_results_model.index(1, 0),
                QItemSelectionModel.SelectionFlag.Select,
            )
            APP.processEvents()
            pane._split_selected_face_result_group()
            self.assertTrue(self._wait_until(lambda: self._list_view_count(pane.face_results_groups_list) == 3))

            self.assertEqual(3, self._list_view_count(pane.face_results_groups_list))
            self.assertIn("Split", self._current_list_view_text(pane.face_results_groups_list))
            self.assertEqual(2, pane.face_results_model.rowCount())

            self._select_list_view_row(pane.face_results_groups_list, 0)
            self._select_list_view_row(pane.face_results_groups_list, 1, append=True)
            APP.processEvents()
            pane._merge_selected_face_result_groups()
            self.assertTrue(self._wait_until(lambda: self._list_view_count(pane.face_results_groups_list) == 2))

            self.assertEqual(2, self._list_view_count(pane.face_results_groups_list))
            self.assertIn("Manual Merge", self._current_list_view_text(pane.face_results_groups_list))
            self.assertEqual(3, pane.face_results_model.rowCount())
        finally:
            pane.close()

    def test_face_cluster_results_order_members_by_rank_then_quality(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        pane.show()
        try:
            result = FaceClusteringComparisonResult(
                clusters_by_key={
                    "cosine-kmeans": {
                        0: [
                            FaceClusterMember("/photos/c.jpg", 0, (14, 14, 46, 46), 0.90, "", "reject"),
                            FaceClusterMember("/photos/a.jpg", 0, (10, 10, 50, 50), 0.98, "Alice", "clean"),
                            FaceClusterMember("/photos/b.jpg", 0, (12, 12, 48, 48), 0.96, "", "review"),
                        ]
                    }
                },
                membership_by_face_ref={
                    ("/photos/a.jpg", 0): {"cosine-kmeans": {"rank": 1}},
                    ("/photos/b.jpg", 0): {"cosine-kmeans": {"rank": 2}},
                    ("/photos/c.jpg", 0): {"cosine-kmeans": {"rank": 3}},
                },
                metrics_by_key={"cosine-kmeans": {"requested_backend": "cosine-kmeans", "cluster_quality_score": 0.64}},
                explanations_by_key={},
                identity_suggestions_by_key={},
            )
            pane._show_face_cluster_results(result, summary="Compared 1 backend.", compare_refs=[("/photos/a.jpg", 0), ("/photos/b.jpg", 0), ("/photos/c.jpg", 0)])
            self.assertTrue(self._wait_until(lambda: pane.face_results_model.rowCount() == 3))

            first = pane.face_results_model.item_at(0)
            second = pane.face_results_model.item_at(1)
            third = pane.face_results_model.item_at(2)
            self.assertEqual("/photos/a.jpg", first.image_path)
            self.assertEqual("/photos/b.jpg", second.image_path)
            self.assertEqual("/photos/c.jpg", third.image_path)
        finally:
            pane.close()

    def test_pending_face_review_preview_threshold_accept_and_undo(self):
        assignments = [
            FaceLabelAssignment("Alice", "/photos/a.jpg", 0, (10, 10, 50, 50), 0.95, proposal_id=1, source="cluster"),
            FaceLabelAssignment("Alice", "/photos/b.jpg", 0, (12, 12, 48, 48), 0.70, proposal_id=2, source="cluster"),
            FaceLabelAssignment("Bob", "/photos/c.jpg", 0, (14, 14, 46, 46), 0.88, proposal_id=3, source="cluster"),
        ]
        service = self._FakeFaceLibraryService(pending_assignments=assignments)
        pane = SearchPane(
            enabled_tabs=["Face Library", "Face Search"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane.show()
        try:
            pane._start_job = lambda _label, run, done: done(run(lambda *_args: None, lambda: False))
            pane._set_pending_face_assignments(list(assignments))
            self._select_list_view_row(pane.face_pending_list, 0)
            APP.processEvents()
            self.assertEqual(1, pane.face_pending_preview_model.rowCount())
            self.assertIn("Alice", pane.face_pending_preview_summary.text())

            pane.face_pending_accept_threshold.setValue(0.90)
            pane._accept_pending_face_labels_above_threshold()
            self.assertEqual(2, self._list_view_count(pane.face_pending_list))
            self.assertTrue(pane.face_pending_undo_button.isEnabled())

            pane._undo_last_pending_face_acceptance()
            self.assertEqual(3, self._list_view_count(pane.face_pending_list))
            self.assertFalse(pane.face_pending_undo_button.isEnabled())
        finally:
            pane.close()

    def test_pending_face_review_can_accept_and_reject_current_cluster(self):
        assignments = [
            FaceLabelAssignment("Alice", "/photos/a.jpg", 0, (10, 10, 50, 50), 0.95, proposal_id=1, source="cluster"),
            FaceLabelAssignment("Alice", "/photos/b.jpg", 0, (12, 12, 48, 48), 0.70, proposal_id=2, source="cluster"),
            FaceLabelAssignment("Bob", "/photos/c.jpg", 0, (14, 14, 46, 46), 0.88, proposal_id=3, source="cluster"),
        ]
        service = self._FakeFaceLibraryService(pending_assignments=assignments)
        pane = SearchPane(
            enabled_tabs=["Face Library", "Face Search"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane.show()
        try:
            pane._start_job = lambda _label, run, done: done(run(lambda *_args: None, lambda: False))
            pane._set_pending_face_assignments(list(assignments))
            result = FaceClusteringComparisonResult(
                clusters_by_key={
                    "cosine-kmeans": {
                        0: [
                            FaceClusterMember("/photos/a.jpg", 0, (10, 10, 50, 50), 0.98, "Alice", "clean"),
                            FaceClusterMember("/photos/b.jpg", 0, (12, 12, 48, 48), 0.96, "", "clean"),
                        ]
                    }
                },
                membership_by_face_ref={},
                metrics_by_key={"cosine-kmeans": {"requested_backend": "cosine-kmeans", "cluster_quality_score": 0.64}},
                explanations_by_key={},
                identity_suggestions_by_key={},
            )
            pane._show_face_cluster_results(result, summary="Compared 1 backend.", compare_refs=[("/photos/a.jpg", 0), ("/photos/b.jpg", 0)])
            APP.processEvents()

            pane._accept_current_cluster_pending_face_labels()
            self.assertEqual(1, self._list_view_count(pane.face_pending_list))
            self.assertEqual("Bob", self._list_view_payload(pane.face_pending_list, 0).person_name)

            pane._undo_last_pending_face_acceptance()
            self.assertEqual(3, self._list_view_count(pane.face_pending_list))

            pane._reject_current_cluster_pending_face_labels()
            self.assertEqual(1, self._list_view_count(pane.face_pending_list))
            remaining = self._list_view_payload(pane.face_pending_list, 0)
            self.assertEqual("/photos/c.jpg", remaining.image_path)
        finally:
            pane.close()

    def test_face_library_review_filters_and_hide_restore_rejected(self):
        service = self._FakeFaceLibraryService(
            records=[
                self._face_record("/photos/clean.jpg", 0, quality_status="clean"),
                self._face_record("/photos/review.jpg", 0, quality_status="review", quality_reasons=("blurred_crop",)),
                self._face_record("/photos/reject.jpg", 0, quality_status="reject", quality_reasons=("low_detector_score",)),
            ],
        )
        pane = SearchPane(
            enabled_tabs=["Face Library"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane.face_folder_path.setText("/photos")
        pane._refresh_scanned_faces()
        try:
            self.assertTrue(
                self._wait_until(
                    lambda: len(pane.results_gallery.images) == 3
                    and pane.face_detected_faces_model.rowCount() == 3
                )
            )
            self.assertEqual(3, len(pane.results_gallery.images))
            self.assertEqual(3, pane.face_detected_faces_model.rowCount())

            pane.face_review_quality_filter.setCurrentIndex(pane.face_review_quality_filter.findData("suspicious"))
            self.assertTrue(
                self._wait_until(
                    lambda: len(pane.results_gallery.images) == 2
                    and pane.face_detected_faces_model.rowCount() == 2
                )
            )
            self.assertEqual(2, len(pane.results_gallery.images))
            self.assertEqual(2, pane.face_detected_faces_model.rowCount())

            pane.face_review_reason_filter.setCurrentIndex(pane.face_review_reason_filter.findData("low_detector_score"))
            self.assertTrue(
                self._wait_until(
                    lambda: len(pane.results_gallery.images) == 1
                    and list(pane.results_gallery.images)[0] == "/photos/reject.jpg"
                )
            )
            self.assertEqual(1, len(pane.results_gallery.images))
            self.assertEqual("/photos/reject.jpg", pane.results_gallery.images[0])

            pane.face_review_quality_filter.setCurrentIndex(pane.face_review_quality_filter.findData("all"))
            pane.face_review_reason_filter.setCurrentIndex(pane.face_review_reason_filter.findData(""))
            self.assertTrue(self._wait_until(lambda: len(pane.results_gallery.images) == 3))
            pane._hide_rejected_face_review_faces()
            self.assertTrue(
                self._wait_until(
                    lambda: pane.face_detected_faces_model.rowCount() == 2
                    and "/photos/reject.jpg" not in list(pane.results_gallery.images)
                )
            )
            self.assertEqual(2, pane.face_detected_faces_model.rowCount())
            self.assertNotIn("/photos/reject.jpg", list(pane.results_gallery.images))

            pane._restore_rejected_face_review_faces()
            self.assertTrue(
                self._wait_until(
                    lambda: pane.face_detected_faces_model.rowCount() == 3
                    and "/photos/reject.jpg" in list(pane.results_gallery.images)
                )
            )
            self.assertEqual(3, pane.face_detected_faces_model.rowCount())
            self.assertIn("/photos/reject.jpg", list(pane.results_gallery.images))
        finally:
            pane.close()

    def test_face_library_rescan_selected_and_suspicious_images(self):
        service = self._FakeFacePipelineService(
            records=[
                self._face_record("/photos/clean.jpg", 0, quality_status="clean"),
                self._face_record("/photos/review.jpg", 0, quality_status="review", quality_reasons=("blurred_crop",)),
                self._face_record("/photos/reject.jpg", 0, quality_status="reject", quality_reasons=("low_detector_score",)),
            ],
        )
        pane = SearchPane(
            enabled_tabs=["Face Library"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane.face_folder_path.setText("/photos")
        pane._start_job = lambda _label, run, done: done(run(lambda *_args: None, lambda: False))
        refresh_calls: list[dict[str, object]] = []
        pane._request_face_library_refresh = lambda **kwargs: refresh_calls.append(dict(kwargs))
        try:
            pane._face_review_all_images = [
                FaceFolderReviewImage(
                    image_path="/photos/clean.jpg",
                    review_status="detected",
                    visible_faces=(self._face_record("/photos/clean.jpg", 0, quality_status="clean"),),
                    total_face_count=1,
                    hidden_face_count=0,
                    image_width=100,
                    image_height=100,
                ),
                FaceFolderReviewImage(
                    image_path="/photos/review.jpg",
                    review_status="detected",
                    visible_faces=(
                        self._face_record(
                            "/photos/review.jpg",
                            0,
                            quality_status="review",
                            quality_reasons=("blurred_crop",),
                        ),
                    ),
                    total_face_count=1,
                    hidden_face_count=0,
                    image_width=100,
                    image_height=100,
                ),
                FaceFolderReviewImage(
                    image_path="/photos/reject.jpg",
                    review_status="detected",
                    visible_faces=(
                        self._face_record(
                            "/photos/reject.jpg",
                            0,
                            quality_status="reject",
                            quality_reasons=("low_detector_score",),
                        ),
                    ),
                    total_face_count=1,
                    hidden_face_count=0,
                    image_width=100,
                    image_height=100,
                ),
            ]
            pane._face_review_images = list(pane._face_review_all_images)
            pane._selected_face_review_image_paths = lambda: ["/photos/clean.jpg", "/photos/review.jpg"]
            pane._rescan_selected_face_review_images()
            self.assertEqual(["/photos/clean.jpg", "/photos/review.jpg"], service.index_paths_calls[-1]["paths"])
            self.assertTrue(refresh_calls)

            self.assertEqual(["/photos/review.jpg", "/photos/reject.jpg"], pane._suspicious_face_review_image_paths())
            pane._rescan_suspicious_face_review_images()
            self.assertEqual(["/photos/review.jpg", "/photos/reject.jpg"], service.index_paths_calls[-1]["paths"])
        finally:
            pane.close()

    def test_face_library_rescan_with_fallback_detector_uses_temporary_policy(self):
        service = self._FakeFacePipelineService(
            records=[self._face_record("/photos/a.jpg", 0, quality_status="review")],
        )
        pane = SearchPane(
            enabled_tabs=["Face Library", "Face Search"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane.face_folder_path.setText("/photos")
        pane._start_job = lambda _label, run, done: done(run(lambda *_args: None, lambda: False))
        pane._request_face_library_refresh = lambda **kwargs: None
        try:
            pane._selected_face_review_image_paths = lambda: ["/photos/a.jpg"]
            for row in range(pane.face_fallback_detector_combo.count()):
                backend = pane.face_fallback_detector_combo.itemData(row)
                if backend and backend != pane.current_face_detector_id():
                    pane.face_fallback_detector_combo.setCurrentIndex(row)
                    break
            pane._rescan_selected_face_review_images_with_fallback()
            self.assertEqual(["/photos/a.jpg"], service.index_paths_calls[-1]["paths"])
            self.assertGreaterEqual(len(service.configure_cascade_calls), 2)
            self.assertEqual("union_then_verify", service.configure_cascade_calls[-2]["detector_policy"])
        finally:
            pane.close()

    def test_face_query_photo_detects_multiple_faces_and_searches_selected_examples(self):
        service = self._FakeFacePipelineService(
            records=[
                self._face_record("/photos/query.jpg", 0, bbox=(4, 6, 30, 34), confidence=0.95),
                self._face_record("/photos/query.jpg", 1, bbox=(40, 10, 74, 46), confidence=0.92),
                self._face_record("/photos/result.jpg", 0, bbox=(8, 10, 44, 52), confidence=0.88),
            ],
        )
        pane = SearchPane(
            enabled_tabs=["Face Library", "Face Search"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane._start_job = lambda _label, run, done: done(run(lambda *_args: None, lambda: False))
        try:
            pane.face_query_path.setText("/photos/query.jpg")
            pane.face_recognition_mode_combo.setCurrentIndex(pane.face_recognition_mode_combo.findData("strict"))
            pane.face_min_score.setValue(0.10)
            pane._detect_query_photo_faces()
            APP.processEvents()

            self.assertEqual(2, self._list_view_count(pane.face_query_faces_list))
            self._select_list_view_row(pane.face_query_faces_list, 0)
            self._select_list_view_row(pane.face_query_faces_list, 1, append=True)
            pane._search_faces()

            self.assertEqual(
                [(4, 6, 30, 34), (40, 10, 74, 46)],
                service.search_query_faces_calls[-1]["query_face_bboxes"],
            )
            self.assertGreaterEqual(service.search_query_faces_calls[-1]["min_face_score"], 0.6)
        finally:
            pane.close()

    def test_selected_face_search_uses_multi_example_when_multiple_faces_selected(self):
        service = self._FakeFacePipelineService(
            records=[
                self._face_record("/photos/a.jpg", 0, bbox=(8, 10, 44, 52), confidence=0.93),
                self._face_record("/photos/a.jpg", 1, bbox=(48, 12, 84, 56), confidence=0.91),
                self._face_record("/photos/b.jpg", 0, bbox=(10, 14, 50, 58), confidence=0.89),
            ],
        )
        pane = SearchPane(
            enabled_tabs=["Face Library", "Face Search"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane._start_job = lambda _label, run, done: done(run(lambda *_args: None, lambda: False))
        try:
            pane.face_folder_path.setText("/photos")
            pane._refresh_scanned_faces()
            self.assertTrue(self._wait_until(lambda: self._list_view_count(pane.face_scanned_list) >= 2))
            pane.face_recognition_mode_combo.setCurrentIndex(pane.face_recognition_mode_combo.findData("strict"))
            pane.face_browser_min_score.setValue(0.10)
            self._select_list_view_row(pane.face_scanned_list, 0)
            self._select_list_view_row(pane.face_scanned_list, 1, append=True)
            pane._search_selected_face()

            self.assertEqual(
                [("/photos/a.jpg", 0), ("/photos/a.jpg", 1)],
                service.search_similar_faces_calls[-1]["face_refs"],
            )
            self.assertGreaterEqual(service.search_similar_faces_calls[-1]["min_score"], 0.6)
        finally:
            pane.close()

    def test_face_results_sort_and_filters_update_visible_tiles(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp) / "people"
            folder.mkdir()
            older = folder / "older.jpg"
            newer = folder / "newer.jpg"
            Image.new("RGB", (64, 64), (20, 30, 40)).save(older)
            Image.new("RGB", (64, 64), (30, 40, 50)).save(newer)
            older_ts = 1_700_000_000
            newer_ts = 1_710_000_000
            os.utime(older, (older_ts, older_ts))
            os.utime(newer, (newer_ts, newer_ts))
            service = self._FakeFacePipelineService(
                records=[
                    self._face_record(str(older), 0, bbox=(8, 10, 44, 52), person_name="Alice", quality_status="clean"),
                    self._face_record(str(newer), 0, bbox=(12, 14, 48, 56), person_name="", quality_status="review"),
                ],
            )
            pane = SearchPane(
                enabled_tabs=["Face Library", "Face Search", "Identities"],
                external_results=False,
                face_service_global=service,
                face_service_session=service,
            )
            try:
                pane._show_face_match_results(
                    [
                        FaceSearchResult(str(older), 0, 0.91, -1, (8, 10, 44, 52), 0.95, "fake", "older"),
                        FaceSearchResult(str(newer), 0, 0.72, -1, (12, 14, 48, 56), 0.88, "fake", "newer"),
                    ],
                    title="Matches",
                    summary="Synthetic face results.",
                    match_label="query",
                )
                self.assertTrue(self._wait_until(lambda: pane.face_results_model.rowCount() == 2))
                self.assertEqual(2, pane.face_results_model.rowCount())

                pane.face_results_quality_filter_combo.setCurrentIndex(pane.face_results_quality_filter_combo.findData("review"))
                self.assertTrue(
                    self._wait_until(
                        lambda: pane.face_results_model.rowCount() == 1
                        and pane.face_results_model.item_at(0) is not None
                        and pane.face_results_model.item_at(0).image_path == str(newer)
                    )
                )
                self.assertEqual(1, pane.face_results_model.rowCount())
                self.assertEqual(str(newer), pane.face_results_model.item_at(0).image_path)

                pane.face_results_quality_filter_combo.setCurrentIndex(pane.face_results_quality_filter_combo.findData("all"))
                pane.face_results_label_filter_combo.setCurrentIndex(pane.face_results_label_filter_combo.findData("named"))
                self.assertTrue(
                    self._wait_until(
                        lambda: pane.face_results_model.rowCount() == 1
                        and pane.face_results_model.item_at(0) is not None
                        and pane.face_results_model.item_at(0).image_path == str(older)
                    )
                )
                self.assertEqual(1, pane.face_results_model.rowCount())
                self.assertEqual(str(older), pane.face_results_model.item_at(0).image_path)

                pane.face_results_label_filter_combo.setCurrentIndex(pane.face_results_label_filter_combo.findData("all"))
                pane.face_results_folder_filter.setText("people")
                pane.face_results_date_from.setText("2023-01-01")
                pane.face_results_date_to.setText("2024-12-31")
                self.assertTrue(self._wait_until(lambda: pane.face_results_model.rowCount() == 2))
                self.assertEqual(2, pane.face_results_model.rowCount())

                pane.face_results_sort_combo.setCurrentIndex(pane.face_results_sort_combo.findData("newest"))
                self.assertTrue(
                    self._wait_until(
                        lambda: pane.face_results_model.rowCount() == 2
                        and pane.face_results_model.item_at(0) is not None
                        and pane.face_results_model.item_at(0).image_path == str(newer)
                    )
                )
                self.assertEqual(str(newer), pane.face_results_model.item_at(0).image_path)
            finally:
                pane.close()

    def test_people_queries_show_cooccurrence_named_unknown_and_primary_results(self):
        service = self._FakeFacePipelineService(
            records=[
                self._face_record("/photos/group.jpg", 0, bbox=(0, 0, 60, 60), person_name="Alice", quality_status="clean"),
                self._face_record("/photos/group.jpg", 1, bbox=(62, 0, 94, 32), person_name="Bob", quality_status="clean"),
                self._face_record("/photos/unknown.jpg", 0, bbox=(0, 0, 48, 48), person_name="", quality_status="review"),
            ],
        )
        pane = SearchPane(
            enabled_tabs=["Face Library", "Face Search"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane._start_job = lambda _label, run, done: done(run(lambda *_args: None, lambda: False))
        try:
            pane.face_people_query_names.setText("Alice, Bob")
            pane._show_photos_with_people_query()
            self.assertTrue(self._wait_until(lambda: pane.face_results_model.rowCount() == 2))
            self.assertEqual(["Alice", "Bob"], service.find_face_records_by_people_calls[-1]["person_names"])
            self.assertEqual(2, pane.face_results_model.rowCount())

            pane._show_photos_with_any_named_people()
            self.assertTrue(self._wait_until(lambda: pane.face_results_model.rowCount() == 2))
            self.assertEqual(2, pane.face_results_model.rowCount())

            pane._show_photos_with_unknown_people()
            self.assertTrue(self._wait_until(lambda: pane.face_results_model.rowCount() == 1))
            self.assertEqual(1, pane.face_results_model.rowCount())

            pane.face_primary_person_query.setText("Alice")
            pane._show_photos_with_primary_person()
            self.assertTrue(self._wait_until(lambda: pane.face_results_model.rowCount() == 1))
            self.assertEqual("Alice", service.find_face_records_with_primary_person_calls[-1]["person_name"])
            self.assertEqual(1, pane.face_results_model.rowCount())
        finally:
            pane.close()

    def test_face_query_builder_generates_grammar_and_dispatches_search(self):
        service = self._FakeFacePipelineService(
            records=[
                self._face_record("/photos/group.jpg", 0, bbox=(0, 0, 60, 60), person_name="Alice", quality_status="clean"),
                self._face_record("/photos/unknown.jpg", 0, bbox=(0, 0, 48, 48), person_name="", quality_status="review"),
            ],
        )
        pane = SearchPane(
            enabled_tabs=["Face Library", "Face Search"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane._start_job = lambda _label, run, done: done(run(lambda *_args: None, lambda: False))
        try:
            pane.face_query_builder_faces_state.setCurrentText("Has faces")
            pane.face_query_builder_face_count.setValue(3)
            pane.face_query_builder_person.setText("Alice")
            pane.face_query_builder_partial.setText("Ali")
            pane.face_query_builder_people.setText("Alice, Bob")
            pane.face_query_builder_people_mode.setCurrentText("All")
            pane.face_query_builder_unknown.setChecked(True)
            pane.face_query_builder_hidden.setChecked(True)

            query = pane._build_face_query_from_controls()
            self.assertIn("faces:3", query)
            self.assertIn('person:"Alice"', query)
            self.assertIn('people:"Ali"', query)
            self.assertIn('person:"Alice&Bob"', query)
            self.assertIn("unknown:true", query)
            self.assertIn("hidden:true", query)

            pane._run_face_query_builder()
            self.assertEqual(query, service.search_face_query_calls[-1]["query"])
            self.assertTrue(service.search_face_query_calls[-1]["include_hidden"])
            self.assertTrue(self._wait_until(lambda: pane.face_results_model.rowCount() == 2))
            self.assertEqual(2, pane.face_results_model.rowCount())
        finally:
            pane.close()

    def test_saved_searches_save_run_rename_delete_and_persist(self):
        with TemporaryDirectory() as tmp:
            store_path = Path(tmp) / "saved_searches.json"
            saved_service = SavedSearchService(store_path)
            service = self._FakeFacePipelineService(
                records=[
                    self._face_record("/photos/group.jpg", 0, bbox=(0, 0, 60, 60), person_name="Alice", quality_status="clean"),
                    self._face_record("/photos/unknown.jpg", 0, bbox=(0, 0, 48, 48), person_name="", quality_status="review"),
                ],
            )
            pane = SearchPane(
                enabled_tabs=["Face Library", "Face Search"],
                external_results=False,
                face_service_global=service,
                face_service_session=service,
                saved_search_service=saved_service,
            )
            pane._start_job = lambda _label, run, done: done(run(lambda *_args: None, lambda: False))
            cluster_payloads: list[dict[str, object]] = []
            pane.saved_clustering_filter_requested.connect(lambda payload: cluster_payloads.append(dict(payload)))
            pane.clustering_filter_state_provider = lambda: {"tags": ["family", "selfie"], "tag_match": "All"}

            def set_kind(value: str) -> None:
                row = pane.saved_search_kind.findData(value)
                self.assertGreaterEqual(row, 0)
                pane.saved_search_kind.setCurrentIndex(row)

            try:
                set_kind("clustering_filter")
                pane.saved_search_name.setText("Family tags")
                pane._save_current_saved_search()
                pane._run_selected_saved_search()
                self.assertEqual({"tags": ["family", "selfie"], "tag_match": "All"}, cluster_payloads[-1])

                set_kind("face_query")
                pane.saved_search_name.setText("Alice hidden query")
                pane.face_query_builder_person.setText("Alice")
                pane.face_query_builder_hidden.setChecked(True)
                pane._save_current_saved_search()
                pane._run_selected_saved_search()
                self.assertIn('person:"Alice"', service.search_face_query_calls[-1]["query"])
                self.assertTrue(service.search_face_query_calls[-1]["include_hidden"])

                set_kind("people_together")
                pane.saved_search_name.setText("Alice and Bob")
                pane.face_people_query_names.setText("Alice, Bob")
                pane._save_current_saved_search()
                pane._run_selected_saved_search()
                self.assertEqual(["Alice", "Bob"], service.find_face_records_by_people_calls[-1]["person_names"])

                set_kind("unknown_people")
                pane.saved_search_name.setText("Unknown queue")
                pane._save_current_saved_search()
                pane._run_selected_saved_search()
                self.assertTrue(service.find_face_records_with_unknown_people_calls)

                set_kind("hidden_faces")
                pane.saved_search_name.setText("Hidden queue")
                pane._save_current_saved_search()
                pane._run_selected_saved_search()
                self.assertEqual("hidden:true", service.search_face_query_calls[-1]["query"])
                self.assertTrue(service.search_face_query_calls[-1]["include_hidden"])

                saved_count = pane.saved_searches_list.count()
                pane.saved_search_name.setText("Hidden queue renamed")
                pane._rename_selected_saved_search()
                self.assertIn("Hidden queue renamed", pane.saved_searches_list.currentItem().text())
                pane._delete_selected_saved_search()
                self.assertEqual(saved_count - 1, pane.saved_searches_list.count())
                persisted_count = pane.saved_searches_list.count()
            finally:
                pane.close()

            restored = SearchPane(
                enabled_tabs=["Face Library", "Face Search"],
                external_results=False,
                face_service_global=service,
                face_service_session=service,
                saved_search_service=SavedSearchService(store_path),
            )
            try:
                self.assertEqual(persisted_count, restored.saved_searches_list.count())
                self.assertTrue(any("Family tags" in restored.saved_searches_list.item(row).text() for row in range(restored.saved_searches_list.count())))
            finally:
                restored.close()

    def test_duplicate_review_records_and_exports_decisions_without_moving_files(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            query = root / "query.jpg"
            exact = root / "exact.jpg"
            near = root / "near.jpg"
            for path in (query, exact, near):
                Image.new("RGB", (24, 24), (10, 20, 30)).save(path)
            before = {path: path.stat().st_mtime_ns for path in (query, exact, near)}
            pane = SearchPane(enabled_tabs=["Duplicate Search"], external_results=False)
            try:
                pane._populate_duplicate_review(
                    [
                        SearchResult(str(near), 0.88, 3, "clip", "duplicate", {}, hash_distance=3, hash_backend="phash"),
                        SearchResult(str(exact), 0.99, 0, "clip", "duplicate", {}, hash_distance=0, hash_backend="phash"),
                    ],
                    str(query),
                )
                self.assertEqual(2, pane.duplicate_review_list.count())
                self.assertIn("Exact", pane.duplicate_review_list.item(0).text())
                self.assertIn(str(query), pane.duplicate_review_summary.text())

                pane.duplicate_review_tag.setText("duplicate")
                pane._record_duplicate_decision("tag")
                with patch("ui.search_pane.confirmBox", return_value=True):
                    pane._record_duplicate_decision("trash")
                self.assertEqual(["tag", "trash"], [item["action"] for item in pane._duplicate_decisions])
                self.assertIn("Source files were not moved", pane.status_label.text())

                out_path = root / "duplicate_decisions.json"
                with patch("ui.search_pane.QFileDialog.getSaveFileName", return_value=(str(out_path), "JSON Files (*.json)")):
                    pane._export_duplicate_decisions()
                payload = json.loads(out_path.read_text(encoding="utf-8"))
                self.assertEqual(2, len(payload["decisions"]))
                self.assertFalse(any(item["source_files_changed"] for item in payload["decisions"]))
                self.assertEqual(before, {path: path.stat().st_mtime_ns for path in (query, exact, near)})
            finally:
                pane.close()

    def test_faces_walkthrough_visibility_and_state_round_trip(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        try:
            pane.show()
            APP.processEvents()
            self.assertFalse(pane.face_walkthrough_panel.isHidden())
            pane._set_face_walkthrough_visible(False)
            self.assertTrue(pane.face_walkthrough_panel.isHidden())
            state = pane.export_state()
        finally:
            pane.close()

        restored = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        try:
            restored.show()
            APP.processEvents()
            restored.apply_state(state)
            self.assertTrue(restored.face_walkthrough_panel.isHidden())
            restored._set_face_walkthrough_visible(True)
            self.assertFalse(restored.face_walkthrough_panel.isHidden())
        finally:
            restored.close()

    def test_face_db_usage_and_delete_actions_are_wired(self):
        with TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "faces.sqlite3"
            faiss_path = Path(f"{db_path}.faiss")
            meta_path = Path(f"{db_path}.faiss.json")
            db_path.write_bytes(b"db")
            faiss_path.write_bytes(b"index")
            meta_path.write_text("{}", encoding="utf-8")
            audit_path = Path(tmp) / "face_action_audit.jsonl"
            service = self._FakeFacePipelineService()
            service.db_path = db_path
            pane = SearchPane(
                enabled_tabs=["Face Library", "Face Search", "Identities"],
                external_results=False,
                face_service_global=service,
                face_service_session=service,
            )
            try:
                pane._face_ui_action_audit_path = lambda: audit_path  # type: ignore[method-assign]
                pane._refresh_face_db_usage_label()
                self.assertIn("Face data usage", pane.face_db_usage_label.text())
                rebuild_calls: list[str] = []
                pane._index_faces = lambda: rebuild_calls.append("rebuild")  # type: ignore[method-assign]
                pane._rebuild_current_face_index()
                self.assertEqual(["rebuild"], rebuild_calls)
                with patch("ui.search_pane.confirmBox", return_value=True):
                    pane._delete_active_face_db()
                self.assertFalse(db_path.exists())
                self.assertFalse(faiss_path.exists())
                self.assertFalse(meta_path.exists())
                audit = json.loads(audit_path.read_text(encoding="utf-8").splitlines()[-1])
                self.assertEqual("delete_active_face_db", audit["action"])
                self.assertEqual(str(db_path), audit["target"])
            finally:
                pane.close()

    def test_face_identity_import_recognition_and_purge_actions_are_wired(self):
        with TemporaryDirectory() as tmp:
            import_path = Path(tmp) / "identities.json"
            import_payload = {
                "version": 1,
                "identities": [{"person_name": "Alice", "similarity_threshold": 0.75}],
            }
            import_path.write_text(json.dumps(import_payload), encoding="utf-8")
            service = self._FakeFacePipelineService(
                profiles=[self._person_profile("Alice", visible_face_count=1)],
                records=[self._face_record("/photos/a.jpg", 0, person_name="Alice")],
            )
            service.db_path = Path(tmp) / "faces.sqlite3"
            pane = SearchPane(
                enabled_tabs=["Face Library", "Face Search", "Identities"],
                external_results=False,
                face_service_global=service,
                face_service_session=service,
            )
            try:
                self.assertEqual("Import Identities", pane.face_import_identities_button.text())
                self.assertEqual("Disable Face Recognition", pane.face_disable_recognition_button.text())
                self.assertEqual("Enable Face Recognition", pane.face_enable_recognition_button.text())
                self.assertEqual("Purge Face Data", pane.face_purge_data_button.text())

                with patch("ui.search_pane.QFileDialog.getOpenFileName", return_value=(str(import_path), "JSON Files (*.json)")), patch(
                    "ui.search_pane.QMessageBox.question",
                    return_value=QMessageBox.StandardButton.Yes,
                ), patch("ui.search_pane.infoBox"):
                    pane._import_face_identities()
                self.assertEqual([import_payload], service.preview_identity_payloads)
                self.assertEqual(import_payload, service.import_identity_payloads[-1])
                self.assertEqual(["merge"], service.import_identity_modes)

                with patch("ui.search_pane.QFileDialog.getOpenFileName", return_value=(str(import_path), "JSON Files (*.json)")), patch(
                    "ui.search_pane.QMessageBox.question",
                    return_value=QMessageBox.StandardButton.Cancel,
                ), patch("ui.search_pane.infoBox"):
                    pane._import_face_identities()
                self.assertEqual(["merge"], service.import_identity_modes)
                self.assertIn("cancelled", pane.status_label.text().lower())

                with patch("ui.search_pane.QFileDialog.getOpenFileName", return_value=(str(import_path), "JSON Files (*.json)")), patch(
                    "ui.search_pane.QMessageBox.question",
                    return_value=QMessageBox.StandardButton.No,
                ), patch("ui.search_pane.infoBox"):
                    pane._import_face_identities()
                self.assertEqual(["merge", "replace"], service.import_identity_modes)

                with patch("ui.search_pane.confirmBox", return_value=True):
                    pane._set_face_recognition_enabled(False)
                    pane._set_face_recognition_enabled(True)
                self.assertEqual([False, True], service.set_face_recognition_enabled_calls)

                with patch("ui.search_pane.confirmBox", return_value=True), patch(
                    "ui.search_pane.QInputDialog.getText", return_value=("PURGE", True)
                ):
                    pane._purge_face_data()
                self.assertEqual(1, service.prepare_face_data_purge_calls)
                self.assertEqual([True], service.purge_face_data_calls)
                self.assertEqual([], service.records)
                self.assertIn("Purged face data", pane.status_label.text())
            finally:
                pane.close()

    def test_face_action_audit_renders_recent_events_and_supported_undo_hint(self):
        service = self._FakeFacePipelineService(
            audit_events=[
                SimpleNamespace(
                    action="accept_pending_face_labels",
                    target="Alice",
                    created_at="2026-07-02 09:00:00",
                    reversible=True,
                    details={"accepted_count": 2},
                ),
                SimpleNamespace(
                    action="purge_face_data",
                    target="human",
                    created_at="2026-07-02 09:01:00",
                    reversible=False,
                    details={},
                ),
            ]
        )
        pane = SearchPane(
            enabled_tabs=["Face Library", "Face Search", "Identities"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        try:
            self.assertEqual("Refresh Action Audit", pane.face_action_audit_refresh_button.text())
            pane._refresh_face_action_audit()
            rows = [pane.face_action_audit_list.item(index).text() for index in range(pane.face_action_audit_list.count())]
            self.assertEqual(2, len(rows))
            self.assertIn("accept_pending_face_labels", rows[0])
            self.assertIn("Alice", rows[0])
            self.assertIn("Undo Last Accept", rows[0])
            self.assertIn("purge_face_data", rows[1])
            self.assertNotIn("Undo Last Accept", rows[1])
        finally:
            pane.close()

    def test_pending_face_label_audit_metadata_renders(self):
        assignments = [
            FaceLabelAssignment(
                person_name="Alice",
                image_path="/photos/a.jpg",
                face_index=0,
                face_bbox=(8, 10, 44, 52),
                confidence=0.91,
                source="cluster_suggestion",
                pending=True,
                proposal_id=11,
            )
        ]
        service = self._FakeFacePipelineService(pending_assignments=assignments)
        pane = SearchPane(
            enabled_tabs=["Face Library", "Face Search"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane._start_job = lambda _label, run, done: done(run(lambda *_args: None, lambda: False))
        try:
            pane._refresh_pending_face_labels()
            self.assertEqual(1, self._list_view_count(pane.face_pending_list))
            self._select_list_view_row(pane.face_pending_list, 0)
            pane._refresh_pending_face_preview()
            self.assertIn("cluster_suggestion", pane.face_label_audit_label.text())
            self.assertIn("Alice", pane.face_label_audit_label.text())
            self.assertIn("0.9100", pane.face_label_audit_label.text())
        finally:
            pane.close()

    def test_face_cluster_results_and_identities_can_be_exported(self):
        with TemporaryDirectory() as tmp:
            export_clusters = Path(tmp) / "clusters.json"
            export_identities = Path(tmp) / "identities.json"
            service = self._FakeFacePipelineService(
                profiles=[
                    self._person_profile(
                        "Alice",
                        similarity_threshold=0.72,
                        example_count=2,
                        labeled_count=3,
                        visible_face_count=3,
                        cover_image_path="/photos/a.jpg",
                        cover_face_index=0,
                    )
                ],
                records=[self._face_record("/photos/a.jpg", 0, person_name="Alice")],
                prototype_faces_by_name={
                    "Alice": [
                        PersonPrototypeFace(
                            person_name="Alice",
                            image_path="/photos/a.jpg",
                            face_index=0,
                            face_bbox=(8, 10, 44, 52),
                            face_confidence=0.97,
                            pinned=True,
                        )
                    ]
                },
            )
            pane = SearchPane(
                enabled_tabs=["Face Library", "Face Search"],
                external_results=False,
                face_service_global=service,
                face_service_session=service,
            )
            try:
                pane._show_face_result_groups(
                    [
                        FaceResultGroup(
                            group_id="cluster:1",
                            title="Cluster 1",
                            summary="One cluster",
                            items=(
                                FaceTileItem(
                                    image_path="/photos/a.jpg",
                                    face_index=0,
                                    bbox=(8, 10, 44, 52),
                                    title="Alice\na.jpg",
                                    tooltip="cluster",
                                    status="clean",
                                    saved_face_index=0,
                                ),
                            ),
                            comparison_key="hdbscan",
                            cluster_id=1,
                        )
                    ],
                    summary="Cluster export source",
                    photo_paths=["/photos/a.jpg"],
                    overlay_by_path={"/photos/a.jpg": "Alice"},
                    context_by_path={"/photos/a.jpg": {}},
                    kind="face_clusters",
                )
                with patch("ui.search_pane.QFileDialog.getSaveFileName", return_value=(str(export_clusters), "JSON Files (*.json)")):
                    pane._export_face_cluster_results()
                cluster_payload = json.loads(export_clusters.read_text(encoding="utf-8"))
                self.assertEqual("hdbscan", cluster_payload["groups"][0]["comparison_key"])
                self.assertEqual("/photos/a.jpg", cluster_payload["groups"][0]["items"][0]["image_path"])

                with patch("ui.search_pane.QFileDialog.getSaveFileName", return_value=(str(export_identities), "JSON Files (*.json)")):
                    pane._export_face_identities()
                identity_payload = json.loads(export_identities.read_text(encoding="utf-8"))
                self.assertEqual("Alice", identity_payload["identities"][0]["person_name"])
                self.assertEqual("/photos/a.jpg", identity_payload["identities"][0]["prototype_refs"][0]["image_path"])
            finally:
                pane.close()

    def test_face_identities_tab_shows_profiles_duplicate_warning_and_prototype_faces(self):
        service = self._FakeFacePipelineService(
            profiles=[
                self._person_profile(
                    "Alice",
                    visible_face_count=2,
                    cover_image_path="/photos/a.jpg",
                    cover_face_index=0,
                )
            ],
            records=[
                self._face_record("/photos/a.jpg", 0, person_name="Alice"),
                self._face_record("/photos/b.jpg", 1, person_name="Alice", bbox=(12, 14, 48, 58)),
            ],
            prototype_faces_by_name={
                "Alice": [
                    PersonPrototypeFace(
                        person_name="Alice",
                        image_path="/photos/a.jpg",
                        face_index=0,
                        face_bbox=(10, 12, 42, 54),
                        face_confidence=0.97,
                        pinned=True,
                    ),
                    PersonPrototypeFace(
                        person_name="Alice",
                        image_path="/photos/b.jpg",
                        face_index=1,
                        face_bbox=(12, 14, 48, 58),
                        face_confidence=0.95,
                        pinned=False,
                    ),
                ]
            },
            duplicate_warnings={"Alice": (("Alicia", 0.98),)},
        )
        pane = SearchPane(
            enabled_tabs=["Face Library", "Face Search", "Identities"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        try:
            pane._refresh_face_identities(force_reload=True)
            self.assertEqual(1, self._list_view_count(pane.face_identity_list))
            self._select_list_view_row(pane.face_identity_list, 0)
            pane._refresh_selected_face_identity()
            self.assertIn("Alice", pane.face_identity_summary.text())
            self.assertIn("Alicia", pane.face_identity_duplicate_warning.text())
            self.assertEqual(2, pane.face_identity_prototype_model.rowCount())
        finally:
            pane.close()

    def test_face_identity_selection_reuses_cached_prototype_faces_until_forced_reload(self):
        service = self._FakeFacePipelineService(
            profiles=[
                self._person_profile(
                    "Alice",
                    visible_face_count=1,
                    cover_image_path="/photos/a.jpg",
                    cover_face_index=0,
                )
            ],
            prototype_faces_by_name={
                "Alice": [
                    PersonPrototypeFace(
                        person_name="Alice",
                        image_path="/photos/a.jpg",
                        face_index=0,
                        face_bbox=(10, 12, 42, 54),
                        face_confidence=0.97,
                        pinned=True,
                    )
                ]
            },
        )
        pane = SearchPane(
            enabled_tabs=["Face Library", "Face Search", "Identities"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        try:
            pane._refresh_face_identities(force_reload=True)
            self.assertEqual(1, len(service.load_person_prototype_faces_calls))

            pane._refresh_selected_face_identity()
            pane._refresh_selected_face_identity()
            self.assertEqual(1, len(service.load_person_prototype_faces_calls))

            pane._refresh_face_identities(force_reload=True)
            self.assertEqual(2, len(service.load_person_prototype_faces_calls))
        finally:
            pane.close()

    def test_face_identity_prototype_can_focus_parent_photo(self):
        service = self._FakeFacePipelineService(
            profiles=[self._person_profile("Alice", visible_face_count=1, cover_image_path="/photos/a.jpg", cover_face_index=0)],
            prototype_faces_by_name={
                "Alice": [
                    PersonPrototypeFace(
                        person_name="Alice",
                        image_path="/photos/a.jpg",
                        face_index=0,
                        face_bbox=(10, 12, 42, 54),
                        face_confidence=0.97,
                        pinned=True,
                    )
                ]
            },
        )
        pane = SearchPane(
            enabled_tabs=["Face Library", "Face Search", "Identities"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        try:
            focused: list[str] = []
            pane._focus_face_review_image = lambda image_path: focused.append(str(image_path)) or True
            pane._refresh_face_identities(force_reload=True)
            self._select_list_view_row(pane.face_identity_list, 0)
            pane._refresh_selected_face_identity()
            pane._focus_selected_identity_prototype_photo()
            self.assertEqual(["/photos/a.jpg"], focused)
        finally:
            pane.close()

    def test_face_identities_can_pin_remove_and_merge_prototype_examples(self):
        service = self._FakeFacePipelineService(
            profiles=[
                self._person_profile("Alice", visible_face_count=2, example_count=2, cover_image_path="/photos/a.jpg", cover_face_index=0),
                self._person_profile("Bob", visible_face_count=1, example_count=1, cover_image_path="/photos/c.jpg", cover_face_index=0),
            ],
            records=[
                self._face_record("/photos/a.jpg", 0, person_name="Alice"),
                self._face_record("/photos/b.jpg", 1, person_name="Alice", bbox=(12, 14, 48, 58)),
                self._face_record("/photos/c.jpg", 0, person_name="Bob", bbox=(16, 18, 52, 62)),
            ],
            prototype_faces_by_name={
                "Alice": [
                    PersonPrototypeFace(
                        person_name="Alice",
                        image_path="/photos/a.jpg",
                        face_index=0,
                        face_bbox=(10, 12, 42, 54),
                        face_confidence=0.97,
                        pinned=True,
                    ),
                    PersonPrototypeFace(
                        person_name="Alice",
                        image_path="/photos/b.jpg",
                        face_index=1,
                        face_bbox=(12, 14, 48, 58),
                        face_confidence=0.95,
                        pinned=False,
                    ),
                ],
                "Bob": [
                    PersonPrototypeFace(
                        person_name="Bob",
                        image_path="/photos/c.jpg",
                        face_index=0,
                        face_bbox=(16, 18, 52, 62),
                        face_confidence=0.96,
                        pinned=True,
                    )
                ],
            },
        )
        pane = SearchPane(
            enabled_tabs=["Face Library", "Face Search", "Identities"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        try:
            pane.refresh_face_library = lambda *args, **kwargs: None
            pane._refresh_face_identities(force_reload=True)
            self._select_list_view_row(pane.face_identity_list, 0)
            pane._refresh_selected_face_identity()
            second_index = pane.face_identity_prototype_model.index(1, 0)
            pane.face_identity_prototype_list.selectionModel().select(
                second_index,
                QItemSelectionModel.SelectionFlag.ClearAndSelect,
            )
            pane._pin_selected_identity_prototype_face()
            self.assertEqual(("Alice", "/photos/b.jpg", 1), service.pin_person_prototype_face_calls[-1])
            pane._refresh_face_identities(force_reload=True)
            self._select_list_view_row(pane.face_identity_list, 0)
            pane._refresh_selected_face_identity()
            first_item = pane.face_identity_prototype_model.item_at(0)
            self.assertIsNotNone(first_item)
            self.assertEqual("/photos/b.jpg", first_item.image_path)

            selected_index = pane.face_identity_prototype_model.index(1, 0)
            pane.face_identity_prototype_list.selectionModel().select(
                selected_index,
                QItemSelectionModel.SelectionFlag.ClearAndSelect,
            )
            with patch("ui.search_pane.confirmBox", return_value=True):
                pane._remove_selected_identity_prototype_faces()
            self.assertEqual(("Alice", "/photos/a.jpg", 0), service.remove_person_prototype_face_calls[-1])

            pane.face_identity_merge_target.setText("Bob")
            with patch("ui.search_pane.confirmBox", return_value=True):
                pane._merge_selected_identity_into_target()
            self.assertEqual(("Alice", "Bob"), service.merge_person_identities_calls[-1])
            pane._refresh_face_identities(force_reload=True)
            self.assertEqual(1, self._list_view_count(pane.face_identity_list))
            self.assertEqual("Bob", str(self._list_view_payload(pane.face_identity_list, 0)))
        finally:
            pane.close()

    def test_face_pipeline_profile_selector_updates_detector_and_embedder(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)

        self.assertGreaterEqual(pane.face_model_profile_combo.count(), 6)
        pane.face_model_profile_combo.setCurrentIndex(pane.face_model_profile_combo.findData("balanced"))
        APP.processEvents()
        self.assertNotEqual(-1, pane.face_detector_combo.findData("yunet_2026may"))
        self.assertEqual("scrfd_2.5g_kps", pane.current_face_detector_id())
        self.assertEqual("arcface_r50", pane.current_face_embedder_id())
        self.assertAlmostEqual(0.35, pane.current_face_detector_score_threshold(), places=3)
        self.assertEqual(50, pane.current_face_max_detections())
        self.assertRegex(pane.face_model_summary_label.text(), r" — (CPU|GPU)$")
        self.assertGreaterEqual(pane.face_quality_profile_combo.count(), 3)

        pane.face_model_profile_combo.setCurrentIndex(pane.face_model_profile_combo.findData("latest_gpu"))
        APP.processEvents()
        self.assertEqual("scrfd_10g_kps", pane.current_face_detector_id())
        self.assertEqual("arcface_r100_glint360k", pane.current_face_embedder_id())

        pane.face_model_profile_combo.setCurrentIndex(pane.face_model_profile_combo.findData("opencv_cpu"))
        APP.processEvents()
        self.assertEqual("yunet_2026may", pane.current_face_detector_id())
        self.assertEqual("sface_2021dec", pane.current_face_embedder_id())

        pane.face_detector_combo.setCurrentIndex(pane.face_detector_combo.findData(BUILTIN_HUMAN_DETECTOR_ID))
        APP.processEvents()
        self.assertEqual("custom", pane.face_model_profile_combo.currentData())

        pane.close()

    def test_face_mode_switch_keeps_model_setup_actions_available_for_missing_animal_models(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        human_service = self._FakeFaceLibraryService(readiness_message="Human mode uses built-in face models.")
        dog_service = self._FakeFaceLibraryService(ready=False, readiness_message="Missing Dog model bundle.")
        pane.face_service_global = human_service
        pane.set_face_services(
            face_services_global={"human": human_service, "dog": dog_service},
            face_services_session={"human": human_service, "dog": dog_service},
        )

        pane.set_active_face_mode("dog")
        APP.processEvents()

        self.assertEqual("dog", pane.current_face_mode())
        self.assertIs(dog_service, pane._active_face_service())
        self.assertTrue(pane.face_scan_button.isEnabled())
        self.assertTrue(pane.face_index_button.isEnabled())
        self.assertFalse(pane.face_search_button.isEnabled())
        self.assertFalse(pane.face_label_button.isEnabled())
        self.assertIn("Missing Dog model bundle", pane.face_scan_button.toolTip())
        self.assertIn("Dog: unavailable.", pane.face_mode_status_label.text())
        self.assertFalse(hasattr(pane, "face_model_status_dashboard_label"))

        pane.set_active_face_mode("human")
        APP.processEvents()

        self.assertTrue(pane.face_scan_button.isEnabled())
        self.assertTrue(pane.face_index_button.isEnabled())
        self.assertEqual("human", pane.current_face_mode())
        pane.close()

    def test_face_workspace_layout_keeps_search_cards_readable_in_task_pane(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        pane.show()
        pane.resize(1400, 900)
        pane.workspace_splitter.setSizes([520, 880])
        pane.workspace_splitter.requested_sizes = [900, 500]
        pane._refresh_face_grid_layouts(force=True)
        APP.processEvents()

        source_index = pane.face_library_grid.indexOf(pane.face_source_group)
        people_index = pane.face_library_selection_grid.indexOf(pane.face_people_group)
        scanned_index = pane.face_library_selection_grid.indexOf(pane.face_scanned_group)
        profile_index = pane.face_library_selection_grid.indexOf(pane.face_profile_group)
        self.assertEqual((0, 0, 1, 2), pane.face_library_grid.getItemPosition(source_index))
        self.assertEqual((0, 0, 1, 2), pane.face_library_selection_grid.getItemPosition(scanned_index))
        self.assertEqual((1, 0, 1, 2), pane.face_library_selection_grid.getItemPosition(people_index))
        self.assertEqual((2, 0, 1, 2), pane.face_library_selection_grid.getItemPosition(profile_index))

        selected_index = pane.face_search_grid.indexOf(pane.face_search_selected_group)
        find_index = pane.face_search_grid.indexOf(pane.face_find_group)
        manage_index = pane.face_search_grid.indexOf(pane.face_manage_group)
        save_index = pane.face_search_grid.indexOf(pane.face_save_group)
        pending_index = pane.face_search_grid.indexOf(pane.face_pending_group)
        people_query_index = pane.face_search_grid.indexOf(pane.face_people_query_group)
        self.assertEqual("compact", pane._face_search_layout_mode)
        self.assertEqual(-1, pane.face_search_grid.indexOf(pane.face_search_quick_group))
        self.assertEqual((0, 0, 1, 2), pane.face_search_grid.getItemPosition(selected_index))
        self.assertEqual((1, 0, 1, 2), pane.face_search_grid.getItemPosition(find_index))
        self.assertEqual((3, 0, 1, 2), pane.face_search_grid.getItemPosition(manage_index))
        self.assertEqual((4, 0, 1, 2), pane.face_search_grid.getItemPosition(save_index))
        self.assertEqual((5, 0, 1, 2), pane.face_search_grid.getItemPosition(pending_index))
        self.assertEqual((6, 0, 1, 2), pane.face_search_grid.getItemPosition(people_query_index))
        selected_actions = pane.face_search_selected_actions_grid
        self.assertEqual((0, 0, 1, 2), selected_actions.getItemPosition(0))
        self.assertEqual((1, 0, 1, 2), selected_actions.getItemPosition(1))
        pane._refresh_face_search_action_layout("wide")
        self.assertEqual((0, 0, 1, 1), selected_actions.getItemPosition(0))
        self.assertEqual((0, 1, 1, 1), selected_actions.getItemPosition(1))
        pane._refresh_face_search_action_layout("compact")

        pane.workspace_splitter.setSizes([340, 1060])
        pane._refresh_face_grid_layouts(force=True)
        APP.processEvents()

        source_index = pane.face_library_grid.indexOf(pane.face_source_group)
        people_index = pane.face_library_selection_grid.indexOf(pane.face_people_group)
        scanned_index = pane.face_library_selection_grid.indexOf(pane.face_scanned_group)
        profile_index = pane.face_library_selection_grid.indexOf(pane.face_profile_group)
        selected_index = pane.face_search_grid.indexOf(pane.face_search_selected_group)
        find_index = pane.face_search_grid.indexOf(pane.face_find_group)
        manage_index = pane.face_search_grid.indexOf(pane.face_manage_group)
        save_index = pane.face_search_grid.indexOf(pane.face_save_group)
        pending_index = pane.face_search_grid.indexOf(pane.face_pending_group)
        people_query_index = pane.face_search_grid.indexOf(pane.face_people_query_group)
        self.assertEqual((0, 0, 1, 2), pane.face_library_grid.getItemPosition(source_index))
        self.assertEqual((0, 0, 1, 2), pane.face_library_selection_grid.getItemPosition(scanned_index))
        self.assertEqual((1, 0, 1, 2), pane.face_library_selection_grid.getItemPosition(people_index))
        self.assertEqual((2, 0, 1, 2), pane.face_library_selection_grid.getItemPosition(profile_index))
        self.assertEqual(-1, pane.face_search_grid.indexOf(pane.face_search_quick_group))
        self.assertEqual((0, 0, 1, 2), pane.face_search_grid.getItemPosition(selected_index))
        self.assertEqual((1, 0, 1, 2), pane.face_search_grid.getItemPosition(find_index))
        self.assertEqual((3, 0, 1, 2), pane.face_search_grid.getItemPosition(manage_index))
        self.assertEqual((4, 0, 1, 2), pane.face_search_grid.getItemPosition(save_index))
        self.assertEqual((5, 0, 1, 2), pane.face_search_grid.getItemPosition(pending_index))
        self.assertEqual((6, 0, 1, 2), pane.face_search_grid.getItemPosition(people_query_index))

        pane.close()

    def test_face_library_review_sort_orders_detected_before_no_faces_and_not_scanned(self):
        service = self._FakeFaceLibraryService()
        pane = SearchPane(
            enabled_tabs=["Face Library"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        review_images = [
            FaceFolderReviewImage(
                image_path="/photos/z_not_scanned.jpg",
                review_status="not_scanned",
                visible_faces=(),
                total_face_count=0,
                hidden_face_count=0,
                image_width=100,
                image_height=100,
            ),
            FaceFolderReviewImage(
                image_path="/photos/a_no_faces.jpg",
                review_status="no_faces",
                visible_faces=(),
                total_face_count=0,
                hidden_face_count=0,
                image_width=100,
                image_height=100,
            ),
            FaceFolderReviewImage(
                image_path="/photos/m_two_faces.jpg",
                review_status="detected",
                visible_faces=(
                    self._face_record("/photos/m_two_faces.jpg", 0, confidence=0.80, bbox=(0, 0, 40, 50)),
                    self._face_record("/photos/m_two_faces.jpg", 1, confidence=0.70, bbox=(0, 0, 30, 50)),
                ),
                total_face_count=2,
                hidden_face_count=0,
                image_width=100,
                image_height=100,
            ),
            FaceFolderReviewImage(
                image_path="/photos/b_best_face.jpg",
                review_status="detected",
                visible_faces=(
                    self._face_record("/photos/b_best_face.jpg", 0, confidence=0.95, bbox=(0, 0, 80, 90)),
                ),
                total_face_count=1,
                hidden_face_count=0,
                image_width=100,
                image_height=100,
            ),
        ]

        pane._apply_face_folder_review(review_images, folder="/photos")

        self.assertEqual(
            [
                "/photos/b_best_face.jpg",
                "/photos/m_two_faces.jpg",
            ],
            list(pane.results_gallery.images),
        )

        pane.face_review_sort.setCurrentIndex(pane.face_review_sort.findData("detected_faces"))
        APP.processEvents()

        self.assertEqual(
            [
                "/photos/m_two_faces.jpg",
                "/photos/b_best_face.jpg",
            ],
            list(pane.results_gallery.images),
        )

        pane.face_review_sort.setCurrentIndex(pane.face_review_sort.findData("best_detected_face"))
        APP.processEvents()

        self.assertEqual(
            [
                "/photos/b_best_face.jpg",
                "/photos/m_two_faces.jpg",
            ],
            list(pane.results_gallery.images),
        )

        pane.close()

    def test_face_library_empty_states_show_without_folder(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        service = self._FakeFaceLibraryService()
        pane.face_service_global = service

        pane._refresh_face_people()
        pane._refresh_scanned_faces()

        self.assertEqual(0, pane.face_named_people_list.count())
        self.assertEqual(0, pane.face_unlabeled_groups_list.count())
        self.assertEqual(0, self._list_view_count(pane.face_scanned_list))
        self.assertIn("No folder selected", pane.face_people_summary.text())
        self.assertIn("No folder selected", pane.face_scanned_summary.text())
        self.assertEqual([], service.load_person_profiles_calls)
        self.assertEqual([], service.load_indexed_faces_calls)

        pane.close()

    def test_face_library_folder_review_gallery_uses_full_folder_and_face_boxes(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        service = self._FakeFaceLibraryService(
            records=[self._face_record("/photos/a.jpg", 0, bbox=(10, 10, 50, 60))],
            review_paths=["/photos/not_scanned.jpg"],
        )
        pane.face_service_global = service
        pane.face_folder_path.setText("/photos")

        pane._refresh_scanned_faces()

        self.assertEqual(["/photos/a.jpg"], list(pane.results_gallery.images))
        self.assertIn("Loaded 2 image(s)", pane.face_library_review_summary.text())
        first_index = pane.results_gallery.model.index(0, 0)
        self.assertTrue(first_index.data(GalleryImageModel.FaceBoxesRole))
        pane.face_photo_filter.setCurrentIndex(pane.face_photo_filter.findData("all"))
        APP.processEvents()
        self.assertEqual(["/photos/a.jpg", "/photos/not_scanned.jpg"], list(pane.results_gallery.images))
        second_index = pane.results_gallery.model.index(1, 0)
        self.assertEqual("Not scanned", second_index.data(GalleryImageModel.OverlayRole))
        self.assertEqual("a.jpg", pane.face_selected_photo_label.text())

        pane.close()

    def test_face_library_refresh_shows_indexed_results_then_augments_with_discovery_paths(self):
        indexed_path = "/photos/a.jpg"
        not_scanned_path = "/photos/not_scanned.jpg"
        service = self._FakeFaceLibraryService(
            records=[self._face_record(indexed_path, 0, bbox=(10, 10, 50, 60))],
            review_paths=[not_scanned_path],
        )
        service.db_path = Path("/tmp/indexed-review.db")
        pane = SearchPane(
            enabled_tabs=["Face Library"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane.face_folder_path.setText("/photos")
        pane._iter_face_review_db_candidates = lambda **_kwargs: [  # type: ignore[method-assign]
            {
                "db_path": Path("/tmp/indexed-review.db"),
                "runtime_root": Path("/tmp/runtime-indexed"),
                "rank": 0,
                "same_runtime": True,
                "same_pipeline": True,
            }
        ]
        pane._face_review_db_metrics = lambda _db_path, _folder: {  # type: ignore[method-assign]
            "indexed_image_count": 1,
            "face_image_count": 1,
            "face_count": 1,
            "db_mtime_ns": 1,
        }

        def _load_scan_image_records(*, folder_prefix="", candidate_paths=None):
            _ = candidate_paths
            if folder_prefix and not indexed_path.startswith(str(folder_prefix)):
                return []
            return [
                FaceScanImageRecord(
                    image_path=indexed_path,
                    mtime_ns=1,
                    file_size=1,
                    face_count=1,
                    image_width=100,
                    image_height=100,
                    indexed_at="now",
                )
            ]

        def _load_folder_review_images(directory, *, recursive=True, candidate_paths=None, include_tiny_faces=True):
            _ = (directory, recursive)
            allowed = None if candidate_paths is None else {str(path) for path in list(candidate_paths or []) if str(path).strip()}
            review_images: list[FaceFolderReviewImage] = []
            for path in [indexed_path, not_scanned_path]:
                if allowed is not None and path not in allowed:
                    continue
                all_records = [record for record in service.records if record.image_path == path]
                visible_records = [
                    record
                    for record in all_records
                    if service._is_visible_record(record, include_tiny_faces=bool(include_tiny_faces))
                ]
                if visible_records:
                    status = "detected"
                elif all_records:
                    status = "tiny_hidden"
                else:
                    status = "not_scanned"
                review_images.append(
                    FaceFolderReviewImage(
                        image_path=path,
                        review_status=status,
                        visible_faces=tuple(visible_records),
                        total_face_count=len(all_records),
                        hidden_face_count=max(0, len(all_records) - len(visible_records)),
                        image_width=100,
                        image_height=100,
                    )
                )
            return review_images

        discovery_calls: list[tuple[str, bool]] = []

        class _Discovery:
            @staticmethod
            def discover_result(directory, recursive=True, progress_callback=None, cancel_check=None):
                _ = (directory, recursive, cancel_check)
                discovery_calls.append((str(directory), bool(recursive)))
                if callable(progress_callback):
                    progress_callback(-1, "Scanning folder review...")
                sleep(0.15)
                return SimpleNamespace(paths=(not_scanned_path,), complete=True, warning_count=0)

        service.load_scan_image_records = _load_scan_image_records  # type: ignore[method-assign]
        service.load_folder_review_images = _load_folder_review_images  # type: ignore[method-assign]
        service.discovery_service = _Discovery()
        try:
            pane._request_face_library_refresh(refresh_people=False, reason="test", force_refresh=False)
            self.assertTrue(
                self._wait_until(lambda: list(pane.results_gallery.images) == [indexed_path], timeout_s=1.0)
            )
            self.assertIn("run scan folder for faces to discover newly added photos", pane.face_library_review_summary.text().lower())
            self.assertTrue(
                self._wait_until(
                    lambda: (
                        getattr(pane, "_face_refresh_thread", None) is None
                        and not list(getattr(pane, "_retained_face_refresh_refs", []) or [])
                    ),
                    timeout_s=1.0,
                )
            )
            self.assertEqual([], discovery_calls)
            self.assertEqual([indexed_path], list(pane.results_gallery.images))
        finally:
            pane.close()

    def test_face_library_refresh_skips_discovery_when_no_indexed_review_exists(self):
        service = self._FakeFaceLibraryService()
        service.db_path = Path("/tmp/active-empty.db")
        pane = SearchPane(
            enabled_tabs=["Face Library"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane.face_folder_path.setText("/photos")
        pane._iter_face_review_db_candidates = lambda **_kwargs: [  # type: ignore[method-assign]
            {
                "db_path": Path("/tmp/active-empty.db"),
                "runtime_root": Path("/tmp/runtime-empty"),
                "rank": 0,
                "same_runtime": True,
                "same_pipeline": True,
            }
        ]
        pane._face_review_db_metrics = lambda _db_path, _folder: {  # type: ignore[method-assign]
            "indexed_image_count": 0,
            "face_image_count": 0,
            "face_count": 0,
            "db_mtime_ns": 0,
        }

        def _load_scan_image_records(*, folder_prefix="", candidate_paths=None):
            _ = (folder_prefix, candidate_paths)
            return []

        service.load_scan_image_records = _load_scan_image_records  # type: ignore[method-assign]
        try:
            pane._request_face_library_refresh(refresh_people=False, reason="empty-review", force_refresh=False)
            self.assertTrue(
                self._wait_until(
                    lambda: "no indexed face review data is available yet" in pane.face_library_review_summary.text().lower(),
                    timeout_s=1.0,
                )
            )
            self.assertEqual([], list(pane.results_gallery.images))
            self.assertEqual([], service.load_folder_review_images_calls[-1]["candidate_paths"])
            self.assertEqual("/tmp/active-empty.db", str(pane._face_review_source.db_path))
        finally:
            pane.close()

    def test_face_library_refresh_prefers_populated_alternate_review_source(self):
        indexed_path = "/photos/a.jpg"
        active_service = self._FakeFaceLibraryService()
        active_service.db_path = Path("/tmp/active-empty.db")
        alternate_service = self._FakeFaceLibraryService(
            records=[self._face_record(indexed_path, 0, bbox=(10, 10, 50, 60), person_name="Alice")],
        )
        alternate_service.db_path = Path("/tmp/alternate-populated.db")
        pane = SearchPane(
            enabled_tabs=["Face Library"],
            external_results=False,
            face_service_global=active_service,
            face_service_session=active_service,
        )
        pane.face_folder_path.setText("/photos")

        active_service.load_scan_image_records = lambda **_kwargs: []  # type: ignore[method-assign]
        alternate_service.load_scan_image_records = (  # type: ignore[method-assign]
            lambda **_kwargs: [
                FaceScanImageRecord(
                    image_path=indexed_path,
                    mtime_ns=1,
                    file_size=1,
                    face_count=1,
                    image_width=100,
                    image_height=100,
                    indexed_at="now",
                )
            ]
        )
        pane._iter_face_review_db_candidates = lambda **_kwargs: [  # type: ignore[method-assign]
            {
                "db_path": Path("/tmp/active-empty.db"),
                "runtime_root": Path("/tmp/runtime-a"),
                "rank": 0,
                "same_runtime": True,
                "same_pipeline": True,
            },
            {
                "db_path": Path("/tmp/alternate-populated.db"),
                "runtime_root": Path("/tmp/runtime-b"),
                "rank": 3,
                "same_runtime": False,
                "same_pipeline": False,
            },
        ]
        pane._face_review_db_metrics = lambda db_path, _folder: {  # type: ignore[method-assign]
            "indexed_image_count": 0 if str(db_path) == "/tmp/active-empty.db" else 1,
            "face_image_count": 0 if str(db_path) == "/tmp/active-empty.db" else 1,
            "face_count": 0 if str(db_path) == "/tmp/active-empty.db" else 1,
            "db_mtime_ns": 1 if str(db_path) == "/tmp/active-empty.db" else 2,
        }
        pane._face_review_service_for_db = (  # type: ignore[method-assign]
            lambda db_path, *, active_service: alternate_service if str(db_path) == "/tmp/alternate-populated.db" else active_service
        )
        try:
            pane._request_face_library_refresh(refresh_people=False, reason="alternate-review-source", force_refresh=False)
            self.assertTrue(
                self._wait_until(
                    lambda: list(pane.results_gallery.images) == [indexed_path],
                    timeout_s=1.0,
                )
            )
            self.assertEqual([], active_service.load_folder_review_images_calls)
            self.assertEqual([indexed_path], alternate_service.load_folder_review_images_calls[-1]["candidate_paths"])
            self.assertEqual("/tmp/alternate-populated.db", str(pane._face_review_source.db_path))
            self.assertIn("reusing a populated review db", pane._face_review_source.selection_reason.lower())
        finally:
            pane.close()

    def test_face_library_refresh_releases_completed_refresh_thread_refs(self):
        indexed_path = "/photos/a.jpg"
        not_scanned_path = "/photos/not_scanned.jpg"
        service = self._FakeFaceLibraryService(
            records=[self._face_record(indexed_path, 0, bbox=(10, 10, 50, 60))],
            review_paths=[not_scanned_path],
        )
        service.db_path = Path("/tmp/thread-release-review.db")
        pane = SearchPane(
            enabled_tabs=["Face Library"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane.face_folder_path.setText("/photos")
        pane._iter_face_review_db_candidates = lambda **_kwargs: [  # type: ignore[method-assign]
            {
                "db_path": Path("/tmp/thread-release-review.db"),
                "runtime_root": Path("/tmp/runtime-thread"),
                "rank": 0,
                "same_runtime": True,
                "same_pipeline": True,
            }
        ]
        pane._face_review_db_metrics = lambda _db_path, _folder: {  # type: ignore[method-assign]
            "indexed_image_count": 1,
            "face_image_count": 1,
            "face_count": 1,
            "db_mtime_ns": 1,
        }

        def _load_scan_image_records(*, folder_prefix="", candidate_paths=None):
            _ = candidate_paths
            if folder_prefix and not indexed_path.startswith(str(folder_prefix)):
                return []
            return [
                FaceScanImageRecord(
                    image_path=indexed_path,
                    mtime_ns=1,
                    file_size=1,
                    face_count=1,
                    image_width=100,
                    image_height=100,
                    indexed_at="now",
                )
            ]

        def _load_folder_review_images(directory, *, recursive=True, candidate_paths=None, include_tiny_faces=True):
            _ = (directory, recursive)
            allowed = None if candidate_paths is None else {str(path) for path in list(candidate_paths or []) if str(path).strip()}
            review_images: list[FaceFolderReviewImage] = []
            for path in [indexed_path, not_scanned_path]:
                if allowed is not None and path not in allowed:
                    continue
                all_records = [record for record in service.records if record.image_path == path]
                visible_records = [
                    record
                    for record in all_records
                    if service._is_visible_record(record, include_tiny_faces=bool(include_tiny_faces))
                ]
                review_images.append(
                    FaceFolderReviewImage(
                        image_path=path,
                        review_status="detected" if visible_records else "not_scanned",
                        visible_faces=tuple(visible_records),
                        total_face_count=len(all_records),
                        hidden_face_count=max(0, len(all_records) - len(visible_records)),
                        image_width=100,
                        image_height=100,
                    )
                )
            return review_images

        class _Discovery:
            @staticmethod
            def discover_result(directory, recursive=True, progress_callback=None, cancel_check=None):
                _ = (directory, recursive, cancel_check)
                if callable(progress_callback):
                    progress_callback(-1, "Scanning folder review...")
                sleep(0.05)
                return SimpleNamespace(paths=(not_scanned_path,), complete=True, warning_count=0)

        service.load_scan_image_records = _load_scan_image_records  # type: ignore[method-assign]
        service.load_folder_review_images = _load_folder_review_images  # type: ignore[method-assign]
        service.discovery_service = _Discovery()
        try:
            pane._request_face_library_refresh(refresh_people=False, reason="test-thread-release", force_refresh=False)
            self.assertTrue(
                self._wait_until(
                    lambda: list(pane.results_gallery.images) == [indexed_path],
                    timeout_s=1.0,
                )
            )
            self.assertTrue(
                self._wait_until(
                    lambda: (
                        getattr(pane, "_face_refresh_thread", None) is None
                        and not list(getattr(pane, "_retained_face_refresh_refs", []) or [])
                    ),
                    timeout_s=1.0,
                )
            )
        finally:
            pane.close()

    def test_face_library_uses_saved_quality_metadata_for_review_counts(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        service = self._FakeFaceLibraryService(
            records=[
                self._face_record(
                    "/photos/a.jpg",
                    0,
                    bbox=(10, 10, 50, 60),
                    quality_status="review",
                    quality_reasons=("soft_focus",),
                )
            ],
        )
        pane.face_service_global = service
        pane.face_folder_path.setText("/photos")

        pane._refresh_scanned_faces()
        index = pane.results_gallery.model.index(0, 0)

        self.assertIn("need review", index.data(GalleryImageModel.SubtitleRole))
        self.assertIn("still need manual review", pane.face_library_review_summary.text())
        self.assertIn("[Review]", self._list_view_text(pane.face_detected_faces_list, 0))

        pane.close()

    def test_face_library_draft_edits_update_gallery_boxes_and_mark_draft_state(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        service = self._FakeFaceLibraryService(
            records=[self._face_record("/photos/a.jpg", 0, bbox=(10, 10, 50, 60))],
        )
        pane.face_service_global = service
        pane.face_folder_path.setText("/photos")

        pane._refresh_scanned_faces()
        pane._on_face_review_drafts_updated(
            "/photos/a.jpg",
            [EditableFaceDraft(bbox=(20, 22, 60, 66), confidence=0.91, source="manual")],
            True,
        )

        first_index = pane.results_gallery.model.index(0, 0)
        self.assertEqual("draft", first_index.data(GalleryImageModel.FaceBoxStateRole))
        self.assertEqual("1 face", first_index.data(GalleryImageModel.OverlayRole))
        self.assertIn("Unsaved face edits", first_index.data(GalleryImageModel.SubtitleRole))
        self.assertIn("unsaved face edits", pane.face_scanned_summary.text().lower())

        boxes = first_index.data(GalleryImageModel.FaceBoxesRole)
        self.assertEqual(((0.2, 0.22, 0.6, 0.66),), tuple(tuple(round(float(value), 2) for value in box) for box in boxes))
        self.assertEqual(1, self._list_view_count(pane.face_detected_faces_list))
        self.assertFalse(pane.face_review_results_tabs.tabBar().isHidden())

        pane.close()

    def test_face_library_migrated_review_paths_reload_affected_gallery_thumbnails(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        pane.show()
        try:
            review_images = [
                FaceFolderReviewImage(
                    image_path="/photos/a.jpg",
                    review_status="detected",
                    visible_faces=(self._face_record("/photos/a.jpg", 0, bbox=(10, 10, 50, 60)),),
                    total_face_count=1,
                    hidden_face_count=0,
                    image_width=100,
                    image_height=100,
                ),
                FaceFolderReviewImage(
                    image_path="/photos/b.jpg",
                    review_status="detected",
                    visible_faces=(self._face_record("/photos/b.jpg", 0, bbox=(15, 15, 55, 65)),),
                    total_face_count=1,
                    hidden_face_count=0,
                    image_width=100,
                    image_height=100,
                ),
            ]
            pane._apply_face_folder_review(review_images, folder="/photos")
            APP.processEvents()

            with patch.object(pane.results_gallery, "invalidate_image_paths") as invalidate, patch.object(
                pane.results_gallery, "update_gallery_with_options"
            ) as update_gallery:
                pane._apply_face_folder_review(
                    review_images,
                    folder="/photos",
                    migrated_image_paths=["/photos/b.jpg"],
                )
                APP.processEvents()
                self.assertEqual(0, self._list_view_count(pane.face_results_groups_list))

            invalidate.assert_called_once_with(["/photos/b.jpg"], reload_visible=True)
            update_gallery.assert_not_called()
        finally:
            pane.close()

    def test_face_tile_cache_key_depends_on_requested_size_and_explicit_invalidation(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        try:
            with TemporaryDirectory() as tmp:
                image_path = Path(tmp) / "face.jpg"
                Image.new("RGB", (120, 120), (40, 80, 120)).save(image_path)
                item = SimpleNamespace(image_path=str(image_path), bbox=(10, 12, 64, 80))
                with patch.object(Path, "stat", side_effect=AssertionError("stat should not run")):
                    key_72 = pane._face_tile_cache_key_for_item(item, QSize(72, 72))
                    key_88 = pane._face_tile_cache_key_for_item(item, QSize(88, 88))
                self.assertIsNotNone(key_72)
                self.assertIsNotNone(key_88)
                self.assertNotEqual(key_72, key_88)

                sleep(0.02)
                Image.new("RGB", (120, 120), (120, 80, 40)).save(image_path)
                with patch.object(Path, "stat", side_effect=AssertionError("stat should not run")):
                    key_after_write = pane._face_tile_cache_key_for_item(item, QSize(72, 72))
                self.assertEqual(key_72, key_after_write)

                pane._clear_face_tile_caches(str(image_path))
                with patch.object(Path, "stat", side_effect=AssertionError("stat should not run")):
                    key_after_invalidate = pane._face_tile_cache_key_for_item(item, QSize(72, 72))
                self.assertNotEqual(key_72, key_after_invalidate)
        finally:
            pane.close()

    def test_shared_face_tile_load_keeps_qimage_apply_path_off_qpixmap(self):
        with patch.object(SearchPane, "_start_face_tile_loader_threads"):
            pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        try:
            item = FaceTileItem(
                image_path="/photos/face.jpg",
                face_index=0,
                bbox=(10, 12, 64, 80),
                title="Unlabeled\nface.jpg",
                tooltip="/photos/face.jpg",
                saved_face_index=0,
            )
            pane.face_detected_faces_model.set_items([item])
            key = pane._face_tile_cache_key_for_item(item, pane.face_detected_faces_model.requested_icon_size())
            self.assertIsNotNone(key)

            image = QImage(88, 88, QImage.Format.Format_ARGB32_Premultiplied)
            image.fill(Qt.GlobalColor.red)
            with patch.object(QPixmap, "fromImage", side_effect=AssertionError("QPixmap conversion should stay off the shared apply path")), patch(
                "ui.search_pane.LOGGER.info"
            ) as logger_info, patch("ui.search_pane.append_qt_diagnostic") as append_diagnostic:
                pane._on_face_tile_loaded(key, image, 0.001)

            cached = pane._face_thumb_cache.get(key)
            self.assertIsInstance(cached, QImage)
            self.assertFalse(cached.isNull())
            index = pane.face_detected_faces_model.index(0, 0)
            model_image = pane.face_detected_faces_model.data(index, FaceTileListModel.ImageRole)
            self.assertIsInstance(model_image, QImage)
            self.assertFalse(model_image.isNull())
            logger_info.assert_not_called()
            append_diagnostic.assert_not_called()
        finally:
            pane.close()

    def test_face_tile_refresh_timer_coalesces_multiple_schedule_requests(self):
        with patch.object(SearchPane, "_start_face_tile_loader_threads"):
            pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        try:
            items = [
                FaceTileItem(
                    image_path=f"/photos/{index:03d}.jpg",
                    face_index=0,
                    bbox=(10, 10, 50, 60),
                    title=f"Face {index}",
                    tooltip=f"/photos/{index:03d}.jpg",
                )
                for index in range(8)
            ]
            pane.face_detected_faces_model.set_items(items)

            with patch.object(
                pane,
                "_iter_visible_face_tile_views",
                return_value=[("detected_faces", pane.face_detected_faces_list, pane.face_detected_faces_model)],
            ), patch.object(
                pane,
                "_visible_and_prefetch_face_tile_rows",
                return_value=(list(range(len(items))), []),
            ), patch.object(
                pane._face_tile_request_queue,
                "clear",
                wraps=pane._face_tile_request_queue.clear,
            ) as clear_queue:
                pane._schedule_face_tile_refresh()
                pane._schedule_face_tile_refresh()
                pane._schedule_face_tile_refresh()

                self.assertTrue(self._wait_until(lambda: clear_queue.call_count > 0))
                self.assertEqual(1, clear_queue.call_count)
                self.assertEqual(len(items), len(pane._face_tile_request_queue._queued_keys))
        finally:
            pane.close()

    def test_face_tile_queue_skips_cached_rows_when_spending_visible_budget(self):
        with patch.object(SearchPane, "_start_face_tile_loader_threads"):
            pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        try:
            items = [
                FaceTileItem(
                    image_path=f"/photos/{index:03d}.jpg",
                    face_index=0,
                    bbox=(10, 10, 50, 60),
                    title=f"Face {index}",
                    tooltip=f"/photos/{index:03d}.jpg",
                )
                for index in range(60)
            ]
            pane.face_detected_faces_model.set_items(items)
            cached_image = QImage(88, 88, QImage.Format.Format_ARGB32_Premultiplied)
            cached_image.fill(Qt.GlobalColor.blue)
            requested_size = pane.face_detected_faces_model.requested_icon_size()
            for item in items[:48]:
                key = pane._face_tile_cache_key_for_item(item, requested_size)
                self.assertIsNotNone(key)
                pane._face_thumb_cache[key] = cached_image

            with patch.object(
                pane,
                "_visible_and_prefetch_face_tile_rows",
                return_value=(list(range(len(items))), []),
            ):
                pane._queue_visible_face_tiles(
                    pane.face_detected_faces_list,
                    pane.face_detected_faces_model,
                    source="detected_faces",
                )

            queued_paths = {str(key[0]) for key in pane._face_tile_request_queue._queued_keys}
            self.assertEqual({f"/photos/{index:03d}.jpg" for index in range(48, 60)}, queued_paths)
        finally:
            pane.close()

    def test_face_tile_refresh_drops_stale_queued_work_on_viewport_change(self):
        with patch.object(SearchPane, "_start_face_tile_loader_threads"):
            pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        try:
            items = [
                FaceTileItem(
                    image_path=f"/photos/{index:03d}.jpg",
                    face_index=0,
                    bbox=(10, 10, 50, 60),
                    title=f"Face {index}",
                    tooltip=f"/photos/{index:03d}.jpg",
                )
                for index in range(48)
            ]
            pane.face_detected_faces_model.set_items(items)

            with patch.object(
                pane,
                "_iter_visible_face_tile_views",
                return_value=[("detected_faces", pane.face_detected_faces_list, pane.face_detected_faces_model)],
            ), patch.object(
                pane,
                "_visible_and_prefetch_face_tile_rows",
                side_effect=[
                    (list(range(12)), []),
                    (list(range(24, 36)), []),
                ],
            ):
                pane._flush_face_tile_refresh()
                first_keys = set(pane._face_tile_request_queue._queued_keys)
                pane._flush_face_tile_refresh()
                second_keys = set(pane._face_tile_request_queue._queued_keys)

            self.assertEqual(12, len(first_keys))
            self.assertEqual(12, len(second_keys))
            self.assertNotEqual(first_keys, second_keys)
            self.assertTrue(first_keys.isdisjoint(second_keys))
        finally:
            pane.close()

    def test_face_tile_load_failure_stabilizes_tile_and_prevents_requeue(self):
        with patch.object(SearchPane, "_start_face_tile_loader_threads"):
            pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        try:
            item = FaceTileItem(
                image_path="/photos/unreadable.jpg",
                face_index=0,
                bbox=(10, 10, 50, 60),
                title="Unreadable\nunreadable.jpg",
                tooltip="/photos/unreadable.jpg",
            )
            pane.face_detected_faces_model.set_items([item])
            requested_size = pane.face_detected_faces_model.requested_icon_size()
            key = pane._face_tile_cache_key_for_item(item, requested_size)
            self.assertIsNotNone(key)
            pane._face_tile_pending_keys.add(key)
            pane._face_tile_inflight_keys.add(key)

            pane._on_face_tile_load_failed(key, "permission denied")

            self.assertIn(key, pane._face_tile_failed_keys)
            self.assertEqual("permission denied", pane._face_tile_failure_messages[key])
            self.assertNotIn(key, pane._face_tile_pending_keys)
            self.assertNotIn(key, pane._face_tile_inflight_keys)
            self.assertFalse(pane._queue_face_tile_load(item, requested_size))

            with patch.object(
                pane,
                "_visible_and_prefetch_face_tile_rows",
                return_value=([0], []),
            ), patch.object(pane, "_start_face_tile_loader_threads"):
                pane._flush_face_tile_refresh()

            self.assertEqual(set(), pane._face_tile_request_queue._queued_keys)
            image = pane.face_detected_faces_model.data(
                pane.face_detected_faces_model.index(0, 0),
                FaceTileListModel.ImageRole,
            )
            self.assertIsInstance(image, QImage)
            self.assertFalse(image.isNull())
        finally:
            pane.close()

    def test_face_tile_cache_invalidation_clears_failures_and_allows_retry(self):
        with patch.object(SearchPane, "_start_face_tile_loader_threads"):
            pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        try:
            item = FaceTileItem(
                image_path="/photos/unreadable.jpg",
                face_index=0,
                bbox=(10, 10, 50, 60),
                title="Unreadable\nunreadable.jpg",
                tooltip="/photos/unreadable.jpg",
            )
            pane.face_detected_faces_model.set_items([item])
            requested_size = pane.face_detected_faces_model.requested_icon_size()
            key = pane._face_tile_cache_key_for_item(item, requested_size)
            self.assertIsNotNone(key)

            pane._on_face_tile_load_failed(key, "permission denied")
            self.assertIn(key, pane._face_tile_failed_keys)
            pane._clear_face_tile_caches(str(item.image_path))
            self.assertNotIn(key, pane._face_tile_failed_keys)
            self.assertTrue(pane._queue_face_tile_load(item, requested_size))

            pane._face_tile_request_queue.clear()
            current_key = pane._face_tile_cache_key_for_item(item, requested_size)
            self.assertIsNotNone(current_key)
            pane._on_face_tile_load_failed(current_key, "permission denied")
            self.assertIn(current_key, pane._face_tile_failed_keys)
            pane._clear_face_tile_caches()
            self.assertEqual(set(), pane._face_tile_failed_keys)
            self.assertEqual({}, pane._face_tile_failure_messages)
        finally:
            pane.close()

    def test_detected_faces_summary_reports_unavailable_tiles(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        try:
            item = FaceTileItem(
                image_path="/photos/unreadable.jpg",
                face_index=0,
                bbox=(10, 10, 50, 60),
                title="Unreadable\nunreadable.jpg",
                tooltip="/photos/unreadable.jpg",
            )
            pane.face_detected_faces_model.set_items([item])
            pane._refresh_detected_faces_summary()
            key = pane._face_tile_cache_key_for_item(item, pane.face_detected_faces_model.requested_icon_size())
            self.assertIsNotNone(key)

            pane._on_face_tile_load_failed(key, "permission denied")

            summary = pane.face_detected_faces_summary.text()
            self.assertIn("showing 1 face tile(s)", summary.lower())
            self.assertIn("unavailable: 1 tile(s)", summary.lower())
        finally:
            pane.close()

    def test_detected_face_tiles_queue_only_when_detected_faces_tab_is_active(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        pane.show()
        try:
            review_images = [
                FaceFolderReviewImage(
                    image_path="/photos/one.jpg",
                    review_status="detected",
                    visible_faces=(
                        self._face_record("/photos/one.jpg", 0, confidence=0.95, bbox=(10, 10, 40, 40)),
                        self._face_record("/photos/one.jpg", 1, confidence=0.80, bbox=(45, 12, 80, 48)),
                    ),
                    total_face_count=2,
                    hidden_face_count=0,
                    image_width=100,
                    image_height=100,
                ),
            ]
            queued: list[tuple[str, tuple[int, int]]] = []
            pane._queue_face_tile_load = lambda item, size, priority=100: queued.append((item.image_path, (size.width(), size.height())))  # type: ignore[method-assign]
            pane._apply_face_folder_review(review_images, folder="/photos")
            APP.processEvents()
            queued.clear()
            APP.processEvents()

            pane.face_review_results_tabs.setCurrentIndex(0)
            pane._refresh_detected_faces_review()
            APP.processEvents()
            self.assertEqual([], [entry for entry in queued if entry[1] == (88, 88)])

            pane.face_review_results_tabs.setCurrentIndex(1)
            self.assertTrue(self._wait_until(lambda: len([entry for entry in queued if entry[1] == (88, 88)]) > 0))
            self.assertTrue(all(size == (88, 88) for _path, size in queued if size == (88, 88)))
        finally:
            pane.close()

    def test_large_face_review_defers_aux_tabs_until_opened(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        pane.show()
        try:
            review_images = [
                FaceFolderReviewImage(
                    image_path=f"/photos/{index:03d}.jpg",
                    review_status="detected",
                    visible_faces=(self._face_record(f"/photos/{index:03d}.jpg", 0, bbox=(10, 10, 50, 60)),),
                    total_face_count=1,
                    hidden_face_count=0,
                    image_width=100,
                    image_height=100,
                )
                for index in range(257)
            ]
            pane._face_review_all_images = list(review_images)
            pane._face_review_images = list(review_images)
            pane._face_review_by_path = {item.image_path: item for item in review_images}
            pane._face_review_folder = "/photos"
            pane._face_result_source_kind = "faces_review"
            pane.face_review_results_tabs.setCurrentWidget(pane.results_gallery)

            pane._refresh_detected_faces_review()
            pane._sync_face_review_result_groups()

            self.assertEqual(0, pane.face_detected_faces_model.rowCount())
            self.assertIn("deferred for large reviews", pane.face_detected_faces_summary.text().lower())
            self.assertEqual(0, self._list_view_count(pane.face_results_groups_list))
            self.assertIn("no groups or matches yet", pane.face_results_summary.text().lower())

            pane.face_review_results_tabs.setCurrentWidget(pane.face_detected_faces_panel)
            self.assertTrue(self._wait_until(lambda: pane.face_detected_faces_model.rowCount() == 257))

            pane.face_review_results_tabs.setCurrentWidget(pane.face_results_panel)
            APP.processEvents()
            self.assertEqual(0, self._list_view_count(pane.face_results_groups_list))
        finally:
            pane.close()

    def test_face_photo_filter_defaults_to_faces_and_group_navigation_is_explicit(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        pane.show()
        try:
            review_images = [
                FaceFolderReviewImage(
                    image_path="/photos/a.jpg",
                    review_status="detected",
                    visible_faces=(self._face_record("/photos/a.jpg", 0),),
                    total_face_count=1,
                    hidden_face_count=0,
                    image_width=100,
                    image_height=100,
                ),
                FaceFolderReviewImage(
                    image_path="/photos/b.jpg",
                    review_status="detected",
                    visible_faces=(self._face_record("/photos/b.jpg", 0),),
                    total_face_count=1,
                    hidden_face_count=0,
                    image_width=100,
                    image_height=100,
                ),
                FaceFolderReviewImage(
                    image_path="/photos/empty.jpg",
                    review_status="no_faces",
                    visible_faces=(),
                    total_face_count=0,
                    hidden_face_count=0,
                    image_width=100,
                    image_height=100,
                ),
            ]
            pane._apply_face_folder_review(review_images, folder="/photos")
            APP.processEvents()
            self.assertEqual("with_faces", pane.face_photo_filter.currentData())
            self.assertEqual(["/photos/a.jpg", "/photos/b.jpg"], pane.results_gallery.images)

            pane.face_photo_filter.setCurrentIndex(pane.face_photo_filter.findData("no_faces"))
            APP.processEvents()
            self.assertEqual(["/photos/empty.jpg"], pane.results_gallery.images)
            pane.face_photo_filter.setCurrentIndex(pane.face_photo_filter.findData("with_faces"))
            APP.processEvents()
            self.assertEqual(["/photos/a.jpg", "/photos/b.jpg"], pane.results_gallery.images)

            face_item = FaceTileItem(
                image_path="/photos/a.jpg",
                face_index=0,
                bbox=(10, 10, 40, 40),
                title="Match",
                tooltip="Match",
            )
            group = FaceResultGroup(
                group_id="match:1",
                title="Match Group",
                summary="One matching face",
                items=(face_item,),
            )
            pane._face_result_groups = [group]
            pane._face_result_group_by_id = {group.group_id: group}
            pane._face_result_selected_group_id = group.group_id
            pane._face_result_selected_group_id_by_kind["raw"] = group.group_id
            pane._face_result_active_group_kind = "raw"
            pane._face_result_photo_paths_by_group_id = {group.group_id: ["/photos/a.jpg"]}
            pane._refresh_current_face_result_group()
            self.assertEqual(["/photos/a.jpg", "/photos/b.jpg"], pane.results_gallery.images)

            pane._show_current_face_result_group_photos()
            APP.processEvents()
            self.assertEqual(["/photos/a.jpg"], pane.results_gallery.images)
            self.assertTrue(pane.face_photos_back_button.isVisible())

            pane._restore_folder_photos()
            APP.processEvents()
            self.assertEqual(["/photos/a.jpg", "/photos/b.jpg"], pane.results_gallery.images)
        finally:
            pane.close()

    def test_large_detected_faces_publish_streams_partial_rows_before_completion(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        pane.show()
        try:
            review_images = [
                FaceFolderReviewImage(
                    image_path=f"/photos/{index:03d}.jpg",
                    review_status="detected",
                    visible_faces=(self._face_record(f"/photos/{index:03d}.jpg", 0, bbox=(10, 10, 50, 60)),),
                    total_face_count=1,
                    hidden_face_count=0,
                    image_width=100,
                    image_height=100,
                )
                for index in range(257)
            ]

            pane._apply_face_folder_review(review_images, folder="/photos")
            APP.processEvents()
            pane.face_review_results_tabs.setCurrentWidget(pane.face_detected_faces_panel)

            initial_count = pane.face_detected_faces_model.rowCount()
            self.assertGreater(initial_count, 0)
            self.assertLess(initial_count, 257)
            self.assertTrue(pane._face_detected_publish_in_progress)
            self.assertIn("publishing", pane.face_detected_faces_summary.text().lower())

            self.assertTrue(self._wait_until(lambda: pane.face_detected_faces_model.rowCount() == 257))
            self.assertFalse(pane._face_detected_publish_in_progress)
            self.assertIn("showing 257 face tile(s)", pane.face_detected_faces_summary.text().lower())
        finally:
            pane.close()

    def test_cluster_visible_uses_full_visible_scope_while_detected_faces_publish_is_in_progress(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        pane.show()
        try:
            review_images = [
                FaceFolderReviewImage(
                    image_path=f"/photos/{index:03d}.jpg",
                    review_status="detected",
                    visible_faces=(self._face_record(f"/photos/{index:03d}.jpg", 0, bbox=(10, 10, 50, 60)),),
                    total_face_count=1,
                    hidden_face_count=0,
                    image_width=100,
                    image_height=100,
                )
                for index in range(257)
            ]

            pane._apply_face_folder_review(review_images, folder="/photos")
            APP.processEvents()
            pane.face_review_results_tabs.setCurrentWidget(pane.face_detected_faces_panel)

            self.assertGreater(pane.face_detected_faces_model.rowCount(), 0)
            self.assertLess(pane.face_detected_faces_model.rowCount(), 257)
            self.assertTrue(pane._face_detected_publish_in_progress)
            self.assertTrue(pane.face_detected_cluster_visible_button.isEnabled())

            captured_refs: list[tuple[str, int]] = []
            pane._cluster_face_refs = lambda refs, source_label="": captured_refs.extend(list(refs))  # type: ignore[method-assign]
            pane._cluster_visible_detected_faces()

            self.assertEqual(257, len(captured_refs))
        finally:
            pane.close()

    def test_large_face_review_publish_is_lazy_on_first_open(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        pane.resize(900, 700)
        pane.show()
        try:
            review_images = [
                FaceFolderReviewImage(
                    image_path=f"/photos/{index:03d}.jpg",
                    review_status="detected",
                    visible_faces=(self._face_record(f"/photos/{index:03d}.jpg", 0, bbox=(10, 10, 50, 60)),),
                    total_face_count=1,
                    hidden_face_count=0,
                    image_width=100,
                    image_height=100,
                )
                for index in range(257)
            ]

            pane._apply_face_folder_review(review_images, folder="/photos")
            APP.processEvents()

            selected_path = review_images[0].image_path
            deferred_path = review_images[95].image_path

            self.assertTrue(pane._face_review_lazy_publish_enabled)
            self.assertIn(selected_path, pane._face_review_hydrated_paths)
            self.assertNotIn(deferred_path, pane._face_review_hydrated_paths)
            self.assertLess(len(pane._face_review_hydrated_paths), len(review_images))
            self.assertEqual(
                [],
                list(
                    (
                        pane._face_review_context_by_path.get(deferred_path, {})
                        .get("indexed_faces", {})
                        .get("faces", [])
                    )
                ),
            )
            self.assertEqual(1, self._list_view_count(pane.face_scanned_list))
        finally:
            pane.close()

    def test_large_face_review_visible_paths_hydrate_incrementally(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        pane.resize(900, 700)
        pane.show()
        try:
            review_images = [
                FaceFolderReviewImage(
                    image_path=f"/photos/{index:03d}.jpg",
                    review_status="detected",
                    visible_faces=(self._face_record(f"/photos/{index:03d}.jpg", 0, bbox=(10, 10, 50, 60)),),
                    total_face_count=1,
                    hidden_face_count=0,
                    image_width=100,
                    image_height=100,
                )
                for index in range(257)
            ]

            pane._apply_face_folder_review(review_images, folder="/photos")
            APP.processEvents()

            deferred_path = review_images[95].image_path
            self.assertNotIn(deferred_path, pane._face_review_hydrated_paths)

            pane._request_face_review_hydration([deferred_path])
            self.assertTrue(self._wait_until(lambda: deferred_path in pane._face_review_hydrated_paths))
            self.assertEqual(
                1,
                len(
                    list(
                        (
                            pane._face_review_context_by_path.get(deferred_path, {})
                            .get("indexed_faces", {})
                            .get("faces", [])
                        )
                    )
                ),
            )
        finally:
            pane.close()

    def test_large_face_review_visible_path_hydration_drops_stale_scroll_work(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        pane.resize(900, 700)
        pane.show()
        try:
            review_images = [
                FaceFolderReviewImage(
                    image_path=f"/photos/{index:03d}.jpg",
                    review_status="detected",
                    visible_faces=(self._face_record(f"/photos/{index:03d}.jpg", 0, bbox=(10, 10, 50, 60)),),
                    total_face_count=1,
                    hidden_face_count=0,
                    image_width=100,
                    image_height=100,
                )
                for index in range(257)
            ]

            pane._apply_face_folder_review(review_images, folder="/photos")
            APP.processEvents()

            stale_path = review_images[72].image_path
            current_path = review_images[95].image_path
            self.assertNotIn(stale_path, pane._face_review_hydrated_paths)
            self.assertNotIn(current_path, pane._face_review_hydrated_paths)

            pane._queue_face_review_path_hydration([stale_path], replace_existing=True)
            pane._queue_face_review_path_hydration([current_path], replace_existing=True)
            pane._flush_face_review_hydration_paths()

            self.assertTrue(self._wait_until(lambda: current_path in pane._face_review_hydrated_paths))
            self.assertNotIn(stale_path, pane._face_review_hydrated_paths)
        finally:
            pane.close()

    def test_face_review_refresh_reuses_sorted_images_without_resorting(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        pane.show()
        try:
            review_images = [
                FaceFolderReviewImage(
                    image_path="/photos/c.jpg",
                    review_status="detected",
                    visible_faces=(self._face_record("/photos/c.jpg", 0, confidence=0.70, bbox=(10, 10, 40, 40)),),
                    total_face_count=1,
                    hidden_face_count=0,
                    image_width=100,
                    image_height=100,
                ),
                FaceFolderReviewImage(
                    image_path="/photos/a.jpg",
                    review_status="detected",
                    visible_faces=(self._face_record("/photos/a.jpg", 0, confidence=0.95, bbox=(10, 10, 60, 60)),),
                    total_face_count=1,
                    hidden_face_count=0,
                    image_width=100,
                    image_height=100,
                ),
                FaceFolderReviewImage(
                    image_path="/photos/b.jpg",
                    review_status="detected",
                    visible_faces=(self._face_record("/photos/b.jpg", 0, confidence=0.80, bbox=(10, 10, 50, 50)),),
                    total_face_count=1,
                    hidden_face_count=0,
                    image_width=100,
                    image_height=100,
                ),
            ]
            calls: list[tuple[int, str]] = []
            original = SearchPane.__dict__["_sort_face_review_images_for_mode"].__func__

            def _counting_sort(cls, review_images_arg, *, mode):
                calls.append((len(list(review_images_arg or [])), str(mode)))
                return original(cls, review_images_arg, mode=mode)

            with patch.object(SearchPane, "_sort_face_review_images_for_mode", new=classmethod(_counting_sort)):
                pane._apply_face_folder_review(review_images, folder="/photos")
                APP.processEvents()
                self.assertEqual([(3, "name")], calls)

                pane._refresh_detected_faces_review()
                pane._sync_face_review_result_groups(force=True)
                APP.processEvents()

            self.assertEqual([(3, "name")], calls)
        finally:
            pane.close()

    def test_zoomable_image_emits_face_move_only_on_release(self):
        view = ZoomableImageView()
        try:
            view.resize(320, 320)
            view.show()
            pixmap = QPixmap(200, 200)
            pixmap.fill(Qt.GlobalColor.white)
            view.set_pixmap(pixmap)
            view.set_face_boxes([(40, 40, 100, 100)])
            APP.processEvents()

            moved: list[tuple[int, tuple[int, int, int, int]]] = []
            view.face_box_moved.connect(lambda index, bbox: moved.append((int(index), tuple(int(v) for v in bbox))))

            label = view.widget()
            self.assertIsNotNone(label)
            pixmap_label = label.pixmap()
            self.assertIsNotNone(pixmap_label)
            center_x = pixmap_label.width() // 2
            center_y = pixmap_label.height() // 2
            start_point = label.mapTo(view.viewport(), label.rect().topLeft())
            press = start_point + QPoint(center_x - 30, center_y - 30)
            move = start_point + QPoint(center_x - 10, center_y - 20)

            QTest.mousePress(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, press)
            QTest.mouseMove(view.viewport(), move)
            APP.processEvents()
            self.assertEqual([], moved)
            QTest.mouseRelease(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, move)
            self.assertEqual(1, len(moved))
        finally:
            view.close()

    def test_face_library_detected_faces_tab_can_delete_parent_face_draft(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        service = self._FakeFaceLibraryService(
            records=[
                self._face_record("/photos/a.jpg", 0, bbox=(10, 10, 50, 60)),
                self._face_record("/photos/b.jpg", 0, bbox=(15, 15, 55, 65)),
            ],
        )
        pane.face_service_global = service
        pane.face_folder_path.setText("/photos")

        pane._refresh_scanned_faces()
        self.assertEqual(2, self._list_view_count(pane.face_detected_faces_list))
        self._select_list_view_row(pane.face_detected_faces_list, 0)
        pane._remove_selected_detected_face_tiles()
        APP.processEvents()

        self.assertEqual(1, self._list_view_count(pane.face_detected_faces_list))
        first_index = pane.results_gallery.model.index(0, 0)
        self.assertEqual("draft", first_index.data(GalleryImageModel.FaceBoxStateRole))
        self.assertEqual("No faces detected", first_index.data(GalleryImageModel.OverlayRole))

        pane.close()

    def test_detected_faces_context_menu_selects_clicked_item_and_exposes_actions(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        service = self._FakeFaceLibraryService(
            records=[
                self._face_record("/photos/a.jpg", 0, person_name="Alice", bbox=(10, 10, 50, 60)),
                self._face_record("/photos/b.jpg", 0, person_name="Bob", bbox=(15, 15, 55, 65)),
            ],
        )
        pane.face_service_global = service
        pane.face_folder_path.setText("/photos")
        pane.show()
        pane._refresh_scanned_faces()
        pane.face_review_results_tabs.setCurrentIndex(1)
        APP.processEvents()

        index = pane.face_detected_faces_model.index(1, 0)
        rect = pane.face_detected_faces_list.visualRect(index)
        pane._prepare_detected_faces_context_selection(rect.center())
        menu = pane._build_detected_faces_context_menu()

        self.assertEqual(1, self._selected_list_view_count(pane.face_detected_faces_list))
        action_map = {
            action.text(): action
            for action in menu.actions()
            if not action.isSeparator()
        }
        self.assertTrue(action_map["Find Similar"].isEnabled())
        self.assertTrue(action_map["Name Face..."].isEnabled())
        self.assertTrue(action_map["Find Photos by This Name"].isEnabled())
        self.assertTrue(action_map["Jump To Photo"].isEnabled())
        self.assertTrue(action_map["Open Inspector"].isEnabled())
        self.assertTrue(action_map["Remove Face"].isEnabled())
        self.assertNotIn("Cluster Selected Faces", action_map)
        pane.close()

    def test_detected_faces_context_menu_name_and_saved_name_search_use_existing_flow(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        service = self._FakeFaceLibraryService(
            records=[
                self._face_record("/photos/a.jpg", 0, bbox=(10, 10, 50, 60)),
            ],
        )
        pane.face_service_global = service
        pane.face_folder_path.setText("/photos")
        pane._refresh_scanned_faces()
        self._select_list_view_row(pane.face_detected_faces_list, 0)
        APP.processEvents()

        save_calls: list[tuple[list[tuple[str, int]], dict[str, object]]] = []
        search_calls: list[str] = []
        pane._save_face_refs_name = lambda refs, **kwargs: save_calls.append((list(refs), dict(kwargs)))  # type: ignore[method-assign]
        pane._search_by_name = lambda: search_calls.append(pane.face_name_query.text())  # type: ignore[method-assign]

        with patch("ui.search_pane.QInputDialog.getText", return_value=("Alice", True)):
            menu = pane._build_detected_faces_context_menu()
            action_map = {
                action.text(): action
                for action in menu.actions()
                if not action.isSeparator()
            }
            action_map["Name Face..."].trigger()
            action_map["Find Photos by This Name"].trigger()

        self.assertEqual([([("/photos/a.jpg", 0)], {"source_label": "Detected Faces", "person_name_override": "Alice"})], save_calls)
        self.assertEqual(["Alice"], search_calls)
        pane.close()

    def test_detected_faces_context_menu_keeps_multi_selection_and_names_every_selected_face(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        service = self._FakeFaceLibraryService(
            records=[
                self._face_record("/photos/a.jpg", 0, bbox=(10, 10, 50, 60)),
                self._face_record("/photos/b.jpg", 0, bbox=(15, 15, 55, 65)),
                self._face_record("/photos/c.jpg", 0, bbox=(20, 20, 60, 70)),
            ],
        )
        pane.face_service_global = service
        pane.face_folder_path.setText("/photos")
        pane.show()
        pane._refresh_scanned_faces()
        pane.face_review_results_tabs.setCurrentIndex(1)
        APP.processEvents()

        view = pane.face_detected_faces_list
        self.assertEqual(QAbstractItemView.SelectionMode.ExtendedSelection, view.selectionMode())
        first = pane.face_detected_faces_model.index(0, 0)
        second = pane.face_detected_faces_model.index(1, 0)
        third = pane.face_detected_faces_model.index(2, 0)
        QTest.mouseClick(
            view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            view.visualRect(first).center(),
        )
        QTest.mouseClick(
            view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ShiftModifier,
            view.visualRect(third).center(),
        )
        QTest.mouseClick(
            view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.ControlModifier,
            view.visualRect(second).center(),
        )
        APP.processEvents()

        pane._prepare_detected_faces_context_selection(view.visualRect(first).center())
        self.assertEqual(2, self._selected_list_view_count(view))
        menu = pane._build_detected_faces_context_menu()
        action_map = {action.text(): action for action in menu.actions() if not action.isSeparator()}
        self.assertTrue(action_map["Name Selected Faces..."].isEnabled())
        self.assertTrue(action_map["Cluster Selected Faces"].isEnabled())
        self.assertTrue(action_map["Remove Selected Faces"].isEnabled())
        self.assertNotIn("Find Similar", action_map)
        self.assertNotIn("Jump To Photo", action_map)
        self.assertNotIn("Open Inspector", action_map)

        save_calls: list[tuple[list[tuple[str, int]], dict[str, object]]] = []
        pane._save_face_refs_name = lambda refs, **kwargs: save_calls.append((list(refs), dict(kwargs)))  # type: ignore[method-assign]
        with patch("ui.search_pane.QInputDialog.getText", return_value=("Alice", True)):
            action_map["Name Selected Faces..."].trigger()

        self.assertEqual(1, len(save_calls))
        saved_refs, saved_kwargs = save_calls[0]
        self.assertEqual({("/photos/a.jpg", 0), ("/photos/c.jpg", 0)}, set(saved_refs))
        self.assertEqual(
            {"source_label": "Detected Faces", "person_name_override": "Alice"},
            saved_kwargs,
        )
        pane.close()

    def test_face_library_auto_clean_folder_review_removes_obvious_junk(self):
        service = self._FakeFaceLibraryService(
            records=[
                self._face_record("/photos/a.jpg", 0, confidence=0.92, bbox=(10, 10, 52, 64)),
                self._face_record("/photos/a.jpg", 1, confidence=0.20, bbox=(2, 2, 30, 38)),
                self._face_record("/photos/b.jpg", 0, confidence=0.52, bbox=(8, 8, 38, 46)),
            ],
        )
        pane = SearchPane(
            enabled_tabs=["Face Library"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane.face_folder_path.setText("/photos")

        pane._refresh_scanned_faces()
        pane._auto_clean_face_review_folder()
        APP.processEvents()

        row = list(pane.results_gallery.images).index("/photos/a.jpg")
        index = pane.results_gallery.model.index(row, 0)
        self.assertEqual("draft", index.data(GalleryImageModel.FaceBoxStateRole))
        self.assertIn("updated 1 image(s), removed 1 obvious junk", pane.status_label.text())
        self.assertIn("still need manual review", pane.face_library_review_summary.text())

        pane.close()

    def test_face_library_auto_clean_image_keeps_borderline_faces_for_review(self):
        service = self._FakeFaceLibraryService(
            records=[self._face_record("/photos/a.jpg", 0, confidence=0.52, bbox=(8, 8, 38, 46))],
        )
        pane = SearchPane(
            enabled_tabs=["Face Library"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane.face_folder_path.setText("/photos")

        pane._refresh_scanned_faces()
        pane._auto_clean_face_review_image("/photos/a.jpg")
        APP.processEvents()

        index = pane.results_gallery.model.index(0, 0)
        self.assertEqual("saved", index.data(GalleryImageModel.FaceBoxStateRole))
        self.assertIn("need review", index.data(GalleryImageModel.SubtitleRole))
        self.assertIn("remove or adjust them manually", pane.status_label.text())

        pane.close()

    def test_face_library_inspector_auto_clean_callback_accepts_current_drafts(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        try:
            with TemporaryDirectory() as tmp:
                image_path = Path(tmp) / "a.jpg"
                Image.new("RGB", (160, 160), "white").save(image_path)
                drafts = [
                    EditableFaceDraft(
                        bbox=(24, 24, 112, 128),
                        confidence=0.92,
                        source="detected",
                        person_name="",
                        face_index=0,
                    ),
                    EditableFaceDraft(
                        bbox=(2, 2, 16, 18),
                        confidence=0.20,
                        source="detected",
                        person_name="",
                        face_index=1,
                    ),
                ]

                cleaned, metrics = pane.results_gallery.face_auto_clean_callback(str(image_path), drafts)

                self.assertIsInstance(cleaned, list)
                self.assertEqual(2, len(cleaned) + int(metrics["removed"]))
                self.assertEqual(len(cleaned), int(metrics["kept"]))
                self.assertIn("review", metrics)
        finally:
            pane.close()

    def test_face_library_remove_all_and_reset_face_draft_for_image(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        service = self._FakeFaceLibraryService(
            records=[self._face_record("/photos/a.jpg", 0, bbox=(10, 10, 50, 60))],
        )
        pane.face_service_global = service
        pane.face_folder_path.setText("/photos")

        pane._refresh_scanned_faces()

        with patch("ui.search_pane.confirmBox", return_value=True):
            pane._remove_all_face_review_boxes_for_image("/photos/a.jpg")
        APP.processEvents()

        index = pane.results_gallery.model.index(0, 0)
        self.assertEqual("draft", index.data(GalleryImageModel.FaceBoxStateRole))
        self.assertEqual("No faces detected", index.data(GalleryImageModel.OverlayRole))

        with patch("ui.search_pane.confirmBox", return_value=True):
            pane._reset_face_review_draft_for_image("/photos/a.jpg")
        APP.processEvents()

        self.assertEqual("saved", index.data(GalleryImageModel.FaceBoxStateRole))
        self.assertEqual("1 face", index.data(GalleryImageModel.OverlayRole))

        pane.close()

    def test_face_library_hides_tiny_detections_by_default_and_can_show_them(self):
        service = self._FakeFaceLibraryService(
            records=[
                self._face_record("/photos/large.jpg", 0, bbox=(10, 12, 52, 64)),
                self._face_record("/photos/tiny.jpg", 0, bbox=(10, 12, 30, 30)),
            ],
        )
        pane = SearchPane(
            enabled_tabs=["Face Library"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane.face_folder_path.setText("/photos")

        pane._refresh_face_people()
        pane._refresh_scanned_faces()

        self.assertEqual(1, self._list_view_count(pane.face_scanned_list))
        self.assertIn("1 tiny detection(s) hidden", pane.face_library_review_summary.text())
        self.assertIn("1 tiny detection(s) hidden", pane.face_people_summary.text())
        self.assertFalse(pane.show_tiny_detections_checkbox.isChecked())

        pane.show_tiny_detections_checkbox.setChecked(True)
        self.assertTrue(
            self._wait_until(
                lambda: bool(service.load_folder_review_images_calls)
                and bool(service.load_folder_review_images_calls[-1]["include_tiny_faces"])
                and "tiny detection(s) hidden" not in pane.face_library_review_summary.text(),
                timeout_s=2.0,
            )
        )

        self.assertEqual(1, self._list_view_count(pane.face_scanned_list))
        self.assertNotIn("tiny detection(s) hidden", pane.face_library_review_summary.text())
        self.assertTrue(pane.show_tiny_detections_checkbox.isChecked())
        self.assertTrue(service.load_folder_review_images_calls[-1]["include_tiny_faces"])

        pane.close()

    def test_face_library_reports_when_only_tiny_detections_exist(self):
        service = self._FakeFaceLibraryService(
            records=[self._face_record("/photos/tiny.jpg", 0, bbox=(10, 12, 30, 30))],
        )
        pane = SearchPane(
            enabled_tabs=["Face Library"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane.face_folder_path.setText("/photos")

        pane._refresh_face_people()
        pane._refresh_scanned_faces()

        self.assertEqual(0, pane.face_named_people_list.count())
        self.assertEqual(0, pane.face_unlabeled_groups_list.count())
        self.assertEqual(0, self._list_view_count(pane.face_scanned_list))
        self.assertIn("Only tiny detections were found", pane.face_people_summary.text())
        self.assertIn("tiny detection(s) hidden", pane.face_library_review_summary.text())
        self.assertIn("tiny detection(s) hidden", pane.face_scanned_summary.text())
        self.assertIn("No photos match With Faces", pane.results_gallery.empty_state_title.text())

        pane.show_tiny_detections_checkbox.setChecked(True)
        self.assertTrue(
            self._wait_until(
                lambda: self._list_view_count(pane.face_scanned_list) == 1
                and "Loaded 0 saved identities and 1 unlabeled group(s)." in pane.face_people_summary.text(),
                timeout_s=2.0,
            )
        )

        self.assertEqual(1, self._list_view_count(pane.face_scanned_list))
        self.assertIn("Loaded 0 saved identities and 1 unlabeled group(s).", pane.face_people_summary.text())

        pane.close()

    def test_face_state_round_trips_show_tiny_toggle_review_sort_and_splitter(self):
        service = self._FakeFaceLibraryService()
        pane = SearchPane(
            enabled_tabs=["Face Library", "Face Search"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        pane.show()
        APP.processEvents()
        self.assertFalse(pane.face_advanced_toggle.isChecked())
        self.assertFalse(pane.face_quality_manual_toggle.isChecked())
        pane.configure_face_pipeline_options(
            "",
            {
                "human": {
                    "detector_id": BUILTIN_HUMAN_DETECTOR_ID,
                    "embedder_id": BUILTIN_HUMAN_EMBEDDER_ID,
                    "score_threshold": 0.33,
                    "max_detections": 17,
                },
                "dog": {
                    "detector_id": "dog_det",
                    "embedder_id": "dog_emb",
                    "fallback_detector_id": "dog_det",
                    "detector_policy": "consensus",
                    "verifier_mode": "animal_face_verifier",
                    "score_threshold": 0.44,
                    "max_detections": 23,
                    "search_quality_min": "review",
                    "cluster_quality_min": "reject",
                    "prototype_quality_min": "review",
                    "recognition_min_score": 0.41,
                    "auto_label_min_score": 0.77,
                    "rerank_policy": "quality_score",
                    "rerank_top_n": 17,
                },
            },
            refresh=False,
        )
        pane.set_active_face_mode("dog")
        pane.show_tiny_detections_checkbox.setChecked(True)
        pane.face_review_sort.setCurrentIndex(pane.face_review_sort.findData("best_detected_face"))
        pane.face_cluster_backend_override_checkbox.setChecked(True)
        pane.face_cluster_backend.setCurrentIndex(pane.face_cluster_backend.findData("hdbscan"))
        pane.face_cluster_outlier_policy_combo.setCurrentIndex(pane.face_cluster_outlier_policy_combo.findData("keep"))
        pane.face_hdbscan_min_cluster_size_spin.setValue(9)
        pane.face_hdbscan_min_samples_spin.setValue(4)
        pane.face_hdbscan_cluster_selection_epsilon_spin.setValue(0.21)
        pane.face_hdbscan_allow_single_cluster_checkbox.setChecked(True)
        pane.face_detector_score_spin.setValue(0.41)
        pane.face_max_detections_spin.setValue(19)
        pane.face_quality_profile_combo.setCurrentIndex(pane.face_quality_profile_combo.findData("high_precision"))
        pane.face_quality_reject_confidence_spin.setValue(0.39)
        pane.face_quality_min_face_side_px_spin.setValue(41)
        pane.face_quality_landmarks_policy_combo.setCurrentIndex(pane.face_quality_landmarks_policy_combo.findData("require"))
        pane.face_detector_policy_combo.setCurrentIndex(pane.face_detector_policy_combo.findData("union_then_verify"))
        pane.face_fallback_detector_combo.setCurrentIndex(pane.face_fallback_detector_combo.findData("dog_det"))
        pane.face_verifier_mode_combo.setCurrentIndex(pane.face_verifier_mode_combo.findData("animal_face_verifier"))
        pane.face_search_quality_min_combo.setCurrentIndex(pane.face_search_quality_min_combo.findData("review"))
        pane.face_cluster_quality_min_combo.setCurrentIndex(pane.face_cluster_quality_min_combo.findData("reject"))
        pane.face_prototype_quality_min_combo.setCurrentIndex(pane.face_prototype_quality_min_combo.findData("review"))
        pane.face_recognition_min_score_spin.setValue(0.43)
        pane.face_auto_label_min_score_spin.setValue(0.79)
        pane.face_rerank_policy_combo.setCurrentIndex(pane.face_rerank_policy_combo.findData("quality_score"))
        pane.face_rerank_top_n_spin.setValue(31)
        pane.set_ui_mode("advanced")
        pane.face_library_tabs.setCurrentIndex(1)
        pane.face_advanced_toggle.click()
        pane.face_advanced_tabs.setCurrentIndex(1)
        pane.face_quality_manual_toggle.click()
        pane.face_search_selected_options_toggle.click()
        pane.face_find_options_toggle.click()
        pane.face_save_options_toggle.click()
        pane.face_manage_options_toggle.click()
        pane.face_pending_items_toggle.click()
        pane.workspace_splitter.setSizes([410, 990])
        APP.processEvents()

        state = pane.export_state()

        restored = SearchPane(
            enabled_tabs=["Face Library", "Face Search"],
            external_results=False,
            face_service_global=service,
            face_service_session=service,
        )
        restored.show()
        restored.apply_state(state)
        APP.processEvents()
        splitter_sizes = [int(value) for value in state["splitter_sizes"]]

        self.assertTrue(state["show_tiny_detections"])
        self.assertEqual("dog", state["face_mode"])
        self.assertEqual("advanced", state["face_ui_mode"])
        self.assertEqual(0, state["face_library_tab"])
        self.assertEqual("best_detected_face", state["face_review_sort"])
        self.assertEqual("hdbscan", state["face_cluster_backend"])
        self.assertEqual(["hdbscan"], state["face_cluster_backends"])
        self.assertTrue(state["face_cluster_backend_override_enabled"])
        self.assertEqual("keep", state["face_cluster_outlier_policy"])
        self.assertEqual(9, state["face_hdbscan_min_cluster_size"])
        self.assertEqual(4, state["face_hdbscan_min_samples"])
        self.assertAlmostEqual(0.21, float(state["face_hdbscan_cluster_selection_epsilon"]), places=3)
        self.assertTrue(state["face_hdbscan_allow_single_cluster"])
        self.assertEqual(2, len(splitter_sizes))
        self.assertGreater(splitter_sizes[0], 250)
        self.assertGreater(splitter_sizes[1], 0)
        self.assertTrue(restored.show_tiny_detections_checkbox.isChecked())
        self.assertEqual("dog", restored.current_face_mode())
        self.assertEqual("advanced", restored.current_ui_mode())
        self.assertEqual(0, restored.face_library_tabs.currentIndex())
        self.assertEqual("best_detected_face", restored.face_review_sort.currentData())
        self.assertTrue(restored.face_cluster_backend_override_checkbox.isChecked())
        self.assertEqual("hdbscan", restored.face_cluster_backend.currentData())
        self.assertEqual(["hdbscan"], restored.current_face_cluster_backends())
        self.assertEqual("keep", restored.current_face_cluster_outlier_policy())
        self.assertEqual(9, restored.face_hdbscan_min_cluster_size_spin.value())
        self.assertEqual(4, restored.face_hdbscan_min_samples_spin.value())
        self.assertAlmostEqual(0.21, restored.face_hdbscan_cluster_selection_epsilon_spin.value(), places=3)
        self.assertTrue(restored.face_hdbscan_allow_single_cluster_checkbox.isChecked())
        self.assertEqual("dog_det", restored.current_face_detector_id())
        self.assertEqual("dog_emb", restored.current_face_embedder_id())
        self.assertAlmostEqual(0.41, restored.current_face_detector_score_threshold(), places=3)
        self.assertEqual(19, restored.current_face_max_detections())
        self.assertEqual("high_precision", restored.current_face_quality_profile_id())
        self.assertAlmostEqual(0.39, float(restored.current_face_quality_thresholds()["reject_confidence"]), places=3)
        self.assertEqual(41, int(float(restored.current_face_quality_thresholds()["min_face_side_px"])))
        self.assertEqual("require", str(restored.current_face_quality_thresholds()["landmarks_policy"]))
        self.assertEqual("union_then_verify", restored.current_face_detector_policy())
        self.assertEqual("dog_det", restored.current_face_fallback_detector_id())
        self.assertEqual("animal_face_verifier", restored.current_face_verifier_mode())
        self.assertEqual("review", restored.current_face_search_quality_min())
        self.assertEqual("reject", restored.current_face_cluster_quality_min())
        self.assertEqual("review", restored.current_face_prototype_quality_min())
        self.assertAlmostEqual(0.43, restored.current_face_recognition_min_score(), places=3)
        self.assertAlmostEqual(0.79, restored.current_face_auto_label_min_score(), places=3)
        self.assertEqual("quality_score", restored.current_face_rerank_policy())
        self.assertEqual(31, restored.current_face_rerank_top_n())
        self.assertEqual(splitter_sizes, [int(value) for value in restored.workspace_splitter.sizes()])
        self.assertTrue(restored.face_advanced_toggle.isChecked())
        self.assertEqual(1, restored.face_advanced_tabs.currentIndex())
        self.assertTrue(restored.face_quality_manual_toggle.isChecked())
        self.assertFalse(hasattr(restored, "face_scanned_details_toggle"))
        self.assertFalse(restored.face_scanned_list.isVisible())
        self.assertTrue(restored.face_search_selected_options_toggle.isChecked())
        self.assertTrue(restored.face_find_options_toggle.isChecked())
        self.assertTrue(restored.face_save_options_toggle.isChecked())
        self.assertTrue(restored.face_manage_options_toggle.isChecked())
        self.assertTrue(restored.face_pending_items_toggle.isChecked())

        pane.close()
        restored.close()

    def test_face_advanced_pipeline_is_collapsed_by_default(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        pane.show()
        APP.processEvents()

        self.assertEqual("basic", pane.current_ui_mode())
        self.assertTrue(pane.face_pipeline_summary_group.isVisible())
        self.assertFalse(pane.face_advanced_group.isVisible())
        self.assertFalse(pane.face_manage_group.isVisible())
        self.assertFalse(pane.face_people_query_group.isVisible())
        self.assertFalse(pane.face_advanced_toggle.isChecked())
        self.assertFalse(pane.face_advanced_panel.isVisible())
        self.assertFalse(pane.face_quality_manual_toggle.isChecked())
        self.assertFalse(pane.face_quality_manual_panel.isVisible())
        self.assertFalse(hasattr(pane, "face_model_details_button"))
        self.assertFalse(hasattr(pane, "face_scanned_details_toggle"))
        pane.face_library_tabs.setCurrentIndex(1)
        APP.processEvents()
        self.assertTrue(pane.face_scanned_list.isVisible())
        self.assertFalse(pane.face_search_selected_options_toggle.isChecked())
        self.assertFalse(pane.face_search_selected_options_panel.isVisible())
        self.assertFalse(pane.face_find_options_toggle.isChecked())
        self.assertFalse(pane.face_find_options_panel.isVisible())
        self.assertFalse(pane.face_save_options_toggle.isChecked())
        self.assertFalse(pane.face_save_options_panel.isVisible())
        self.assertFalse(pane.face_manage_options_toggle.isChecked())
        self.assertFalse(pane.face_manage_options_panel.isVisible())
        self.assertFalse(pane.face_pending_items_toggle.isChecked())
        self.assertFalse(pane.face_pending_items_panel.isVisible())

        pane.reveal_face_pipeline_controls()
        APP.processEvents()
        self.assertIsNotNone(pane._face_pipeline_dialog)
        self.assertTrue(pane._face_pipeline_dialog.isVisible())
        self.assertFalse(pane.face_advanced_group.isVisible())
        self.assertEqual(0, pane.face_advanced_tabs.currentIndex())
        pane._face_pipeline_dialog.reject()
        APP.processEvents()

        pane.close()

    def test_face_advanced_mode_has_no_detached_legacy_controls(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        pane.show()
        pane.set_ui_mode("advanced")
        APP.processEvents()

        for name in (
            "face_db_scope_field",
            "face_upload_to_global_button",
            "face_database_path_label",
            "face_review_source_summary",
            "face_model_status_dashboard_label",
        ):
            self.assertFalse(hasattr(pane, name), name)
        self.assertTrue(pane.face_db_scope.isHidden())
        self.assertIs(pane.face_db_scope.parentWidget(), pane.face_library_tabs.widget(0))

        pane.close()

    def test_faces_basic_navigation_and_status_strip_follow_user_tasks(self):
        pane = SearchPane(
            enabled_tabs=["All Faces", "Folder Review", "Face Search", "Identities"],
            external_results=False,
            supported_face_modes=["human"],
        )
        pane.face_folder_path.setText("/photos/family")
        pane.show()
        APP.processEvents()
        try:
            self.assertIsInstance(pane.task_navigation, QTabBar)
            self.assertEqual(
                ["All Faces", "Detect", "Find"],
                [pane.task_navigation.tabText(index) for index in range(pane.task_navigation.count())],
            )
            self.assertLessEqual(pane.task_navigation.height(), 40)
            self.assertIn("MTCNN", pane.face_model_summary_label.text())
            self.assertRegex(pane.face_model_summary_label.text(), r" — (CPU|GPU)$")
            self.assertIn("Global library", pane.face_model_summary_label.toolTip())
            active_service = pane._active_face_service()
            original_policy = active_service.execution_policy
            active_service.execution_policy = SimpleNamespace(
                effective_mode="cuda", torch_device="cuda", onnx_provider="CUDAExecutionProvider"
            )
            pane._active_face_service = lambda: active_service  # type: ignore[method-assign]
            pane._update_face_status_strip()
            self.assertIn("GPU", pane.face_model_summary_label.text())
            self.assertIn("CUDAExecutionProvider", pane.face_model_summary_label.toolTip())
            active_service.execution_policy = original_policy

            pane.task_navigation.setCurrentIndex(1)
            APP.processEvents()
            self.assertEqual("Folder Review", pane.tabs.tabText(pane.tabs.currentIndex()))
            self.assertIn("Folder: family", pane.face_model_summary_label.toolTip())
            self.assertEqual("Detect Faces", pane.face_scan_button.text())

            pane.task_navigation.setCurrentIndex(2)
            APP.processEvents()
            self.assertEqual("Face Search", pane.tabs.tabText(pane.tabs.currentIndex()))
            self.assertFalse(pane.face_library_tabs.tabBar().isTabVisible(1))
            self.assertNotIn("Identities", [pane.task_navigation.tabText(index) for index in range(pane.task_navigation.count())])
        finally:
            pane.close()

    def test_faces_sidebar_uses_one_parent_scroll_owner(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        pane.show()
        APP.processEvents()

        self.assertIsInstance(pane.sidebar_scroll, QScrollArea)
        self.assertIs(pane.sidebar_scroll.widget(), pane.sidebar_content_panel)
        self.assertIs(pane.sidebar_content_layout.itemAt(0).widget(), pane.global_controls_panel)
        self.assertIs(pane.sidebar_content_layout.itemAt(1).widget(), pane.task_navigation)
        self.assertIs(pane.sidebar_content_layout.itemAt(2).widget(), pane.tabs)

        for index, label in enumerate(pane.tab_labels()):
            self.assertIn(label, {"Face Library", "Face Search", "Identities"})
            self.assertNotIsInstance(pane.tabs.widget(index), QScrollArea)

        self.assertFalse(pane.face_settings_group.isVisible())
        pane.face_detector_score_spin.setValue(pane.face_detector_score_spin.value() + 0.01)
        APP.processEvents()
        self.assertFalse(pane.face_settings_group.isVisible())

        pane.set_ui_mode("advanced")
        pane.open_face_pipeline_dialog()
        pane.face_advanced_tabs.setCurrentIndex(1)
        pane.face_quality_manual_toggle.click()
        APP.processEvents()

        self.assertFalse(pane.face_advanced_group.isVisible())
        self.assertIsNotNone(pane._face_pipeline_dialog)
        self.assertTrue(pane._face_pipeline_dialog.isVisible())
        self.assertTrue(pane.face_quality_manual_panel.isVisible())
        self.assertIs(pane.sidebar_scroll.widget(), pane.sidebar_content_panel)

        pane._face_pipeline_dialog.reject()
        pane.close()

    def test_faces_workspace_stays_responsive_at_compact_width(self):
        pane = SearchPane(
            enabled_tabs=["All Faces", "Folder Review", "Face Search", "Identities"],
            external_results=False,
            supported_face_modes=["human"],
        )
        pane.resize(720, 900)
        pane.show()
        APP.processEvents()

        self.assertEqual(720, pane.width())
        self.assertLessEqual(pane.minimumSizeHint().width(), 720)
        self.assertGreaterEqual(pane.gallery_panel.width(), 400)
        self.assertLessEqual(pane.sidebar_content_panel.minimumSizeHint().width(), pane.sidebar_scroll.viewport().width())
        self.assertEqual(0, pane.sidebar_scroll.verticalScrollBar().maximum())
        self.assertTrue(pane.face_workspace_scope_group.isVisible())
        self.assertIn("CPU", pane.face_model_summary_label.text())
        self.assertIn("—", pane.face_model_summary_label.text())

        album_page = pane.tabs.widget(0)
        for button in (
            pane.face_album_refresh_button,
            pane.face_album_load_more_groups_button,
            pane.face_album_load_more_faces_button,
        ):
            self.assertGreater(button.width(), 0)
            self.assertLessEqual(button.geometry().right(), album_page.width())

        results_tab_bar = pane.face_review_results_tabs.tabBar()
        self.assertTrue(results_tab_bar.usesScrollButtons())
        self.assertLessEqual(results_tab_bar.tabRect(2).right(), results_tab_bar.width())

        pane.tabs.setCurrentIndex(2)
        APP.processEvents()
        self.assertEqual(0, pane.sidebar_scroll.verticalScrollBar().maximum())
        self.assertLessEqual(pane.task_navigation.height(), 40)
        pane.set_ui_mode("advanced")
        self.assertTrue(self._wait_until(lambda: pane.sidebar_scroll.verticalScrollBar().maximum() > 0, timeout_s=0.5))
        pane.set_ui_mode("basic")
        pane.tabs.setCurrentIndex(0)
        self.assertTrue(
            self._wait_until(lambda: pane.sidebar_scroll.verticalScrollBar().maximum() == 0, timeout_s=0.5)
        )

        pane.task_panel_toggle.setChecked(False)
        APP.processEvents()
        self.assertFalse(pane.sidebar_panel.isVisible())
        self.assertEqual(pane.workspace_splitter.width(), pane.gallery_panel.width())
        pane.task_panel_toggle.setChecked(True)
        APP.processEvents()
        self.assertTrue(pane.sidebar_panel.isVisible())
        self.assertGreaterEqual(pane.gallery_panel.width(), 400)

        pane.close()

    def test_face_pipeline_provider_uses_selected_detector_and_embedder(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        service = self._FakeFacePipelineService()
        calls: list[tuple[str, str, str, str]] = []

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            detector_dir = root / "human" / "detectors" / "yolo_face"
            embedder_dir = root / "human" / "embedders" / "arcface_small"
            detector_dir.mkdir(parents=True, exist_ok=True)
            embedder_dir.mkdir(parents=True, exist_ok=True)
            (detector_dir / "detector.onnx").write_bytes(b"detector")
            (embedder_dir / "embedder.onnx").write_bytes(b"embedder")
            (detector_dir / "metadata.json").write_text(
                '{"display_name":"YOLO Face","backend_family":"yolo","supported_modes":["human"],"input_name":"images","input_size":[320,320],"output_boxes_name":"boxes","output_scores_name":"scores","mean":[0,0,0],"std":[255,255,255]}',
                encoding="utf-8",
            )
            (embedder_dir / "metadata.json").write_text(
                '{"display_name":"ArcFace Small","supported_modes":["human"],"input_name":"input","input_size":[160,160],"output_name":"embedding","mean":[0.5,0.5,0.5],"std":[0.5,0.5,0.5]}',
                encoding="utf-8",
            )

            pane.set_face_service_provider(
                lambda scope, mode, detector_id, embedder_id: (
                    calls.append((scope, mode, detector_id, embedder_id)) or service
                )
            )
            pane.configure_face_pipeline_options(
                str(root),
                {
                    "human": {
                        "detector_id": "yolo_face",
                        "embedder_id": "arcface_small",
                        "fallback_detector_id": BUILTIN_HUMAN_DETECTOR_ID,
                        "detector_policy": "rescue_on_low_confidence",
                        "verifier_mode": "human_face_verifier",
                        "score_threshold": 0.52,
                        "max_detections": 13,
                        "search_quality_min": "review",
                        "cluster_quality_min": "clean",
                        "prototype_quality_min": "review",
                        "recognition_min_score": 0.44,
                        "auto_label_min_score": 0.81,
                        "rerank_policy": "quality_score",
                        "rerank_top_n": 9,
                    }
                },
                refresh=False,
            )
            pane.set_active_face_mode("human", refresh=False)

            active = pane._active_face_service()

            self.assertIs(active, service)
            self.assertEqual(("global", "human", "yolo_face", "arcface_small"), calls[-1])
            self.assertEqual(0.52, service.configure_calls[-1]["score_threshold"])
            self.assertEqual(13, service.configure_calls[-1]["max_detections"])
            self.assertEqual("balanced", service.configure_quality_calls[-1]["profile_id"])
            self.assertIn("reject_confidence", service.configure_quality_calls[-1]["thresholds"])
            self.assertEqual("rescue_on_low_confidence", service.configure_cascade_calls[-1]["detector_policy"])
            self.assertEqual(BUILTIN_HUMAN_DETECTOR_ID, service.configure_cascade_calls[-1]["fallback_detector_id"])
            self.assertEqual("human_face_verifier", service.configure_cascade_calls[-1]["verifier_mode"])
            self.assertEqual("review", service.configure_recognition_calls[-1]["search_quality_min"])
            self.assertEqual(0.44, service.configure_recognition_calls[-1]["recognition_min_score"])
            self.assertEqual("quality_score", service.configure_recognition_calls[-1]["rerank_policy"])
            self.assertEqual(9, service.configure_recognition_calls[-1]["rerank_top_n"])

        pane.close()

    def test_face_pipeline_changes_stay_pending_until_apply(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        service = self._FakeFacePipelineService()
        calls: list[tuple[str, str, str, str]] = []
        pane.set_face_service_provider(
            lambda scope, mode, detector_id, embedder_id: (
                calls.append((scope, mode, detector_id, embedder_id)) or service
            )
        )
        pane.configure_face_pipeline_options("", {"human": {"detector_id": BUILTIN_HUMAN_DETECTOR_ID, "embedder_id": BUILTIN_HUMAN_EMBEDDER_ID}}, refresh=False)
        pane.set_active_face_mode("human", refresh=False)

        pane._active_face_service()
        self.assertEqual(("global", "human", BUILTIN_HUMAN_DETECTOR_ID, BUILTIN_HUMAN_EMBEDDER_ID), calls[-1])
        self.assertFalse(pane.face_apply_settings_button.isEnabled())

        pane.face_detector_score_spin.setValue(0.47)
        APP.processEvents()

        self.assertTrue(pane.face_apply_settings_button.isEnabled())
        self.assertIn("Pending face setting changes", pane.face_settings_dirty_label.text())
        pane._active_face_service()
        self.assertEqual(("global", "human", BUILTIN_HUMAN_DETECTOR_ID, BUILTIN_HUMAN_EMBEDDER_ID), calls[-1])

        pane._apply_pending_face_settings()
        pane._active_face_service()

        self.assertEqual(0.47, service.configure_calls[-1]["score_threshold"])
        self.assertFalse(pane.face_apply_settings_button.isEnabled())
        self.assertIn("applied", pane.face_settings_dirty_label.text().lower())
        pane.close()

    def test_uninstalled_face_profile_requires_download_before_apply(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        pane.configure_face_pipeline_options(
            "",
            {
                "human": {
                    "detector_id": BUILTIN_HUMAN_DETECTOR_ID,
                    "embedder_id": BUILTIN_HUMAN_EMBEDDER_ID,
                }
            },
            refresh=False,
        )
        pane.face_model_profile_combo.setCurrentIndex(pane.face_model_profile_combo.findData("latest_gpu"))
        APP.processEvents()

        with (
            patch("ui.search_pane.errorBox") as error,
            patch.object(pane, "refresh_face_library"),
            patch.object(pane, "refresh_face_album"),
        ):
            applied_result = pane._apply_pending_face_settings()

        applied = pane._applied_face_pipeline_prefs("human")
        self.assertFalse(applied_result)
        self.assertEqual(BUILTIN_HUMAN_DETECTOR_ID, applied["detector_id"])
        self.assertEqual(BUILTIN_HUMAN_EMBEDDER_ID, applied["embedder_id"])
        self.assertEqual("scrfd_10g_kps", pane.current_face_detector_id())
        self.assertEqual("arcface_r100_glint360k", pane.current_face_embedder_id())
        self.assertIn("not downloaded", str(error.call_args.args[0]).lower())
        pane.close()

    def test_face_sidebar_global_controls_are_scroll_wrapped(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        pane.show()
        APP.processEvents()

        self.assertEqual("QScrollArea", pane.global_controls_scroll.__class__.__name__)
        self.assertEqual(Qt.ScrollBarPolicy.ScrollBarAlwaysOff, pane.global_controls_scroll.horizontalScrollBarPolicy())
        self.assertEqual(Qt.ScrollBarPolicy.ScrollBarAsNeeded, pane.global_controls_scroll.verticalScrollBarPolicy())

        pane.close()

    def test_shutdown_jobs_cancels_face_refresh_thread(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        job = self._FakeAsyncJob()
        pane._face_refresh_job = job
        thread = self._FakeAsyncThread(running=True, wait_result=True)
        pane._face_refresh_thread = thread
        retained_job = self._FakeAsyncJob()
        retained_thread = self._FakeAsyncThread(running=True, wait_result=True)
        pane._retained_face_refresh_refs = [(retained_job, retained_thread)]

        ready = pane.shutdown_jobs(timeout_ms=75)

        self.assertTrue(ready)
        self.assertEqual(1, job.cancel_calls)
        self.assertEqual(1, retained_job.cancel_calls)
        self.assertEqual(1, thread.quit_calls)
        self.assertEqual(1, retained_thread.quit_calls)
        self.assertEqual([75], thread.wait_calls)
        self.assertEqual([75], retained_thread.wait_calls)
        self.assertIsNone(pane._face_refresh_thread)
        self.assertEqual([], pane._retained_face_refresh_refs)
        pane.close()

    def test_shutdown_jobs_detaches_stuck_face_refresh_thread(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        job = self._FakeAsyncJob()
        thread = self._StickyLoaderThread()
        pane._face_refresh_job = job
        pane._face_refresh_thread = thread
        pane._retained_face_refresh_refs = [(job, thread)]

        with patch("ui.search_pane.detach_running_async_job", return_value=True) as detach:
            ready = pane.shutdown_jobs(timeout_ms=75)

        self.assertTrue(ready)
        self.assertEqual(1, job.cancel_calls)
        self.assertEqual(1, thread.quit_calls)
        self.assertEqual([75], thread.wait_calls)
        detach.assert_called_once_with(job, thread)
        self.assertIsNone(pane._face_refresh_thread)
        self.assertEqual([], pane._retained_face_refresh_refs)
        pane.close()

    def test_shutdown_jobs_cancels_face_review_publish_threads(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        job = self._FakeAsyncJob()
        thread = self._FakeAsyncThread(running=True, wait_result=True)
        retained_job = self._FakeAsyncJob()
        retained_thread = self._FakeAsyncThread(running=True, wait_result=True)
        pane._face_review_publish_job = job
        pane._face_review_publish_thread = thread
        pane._retained_face_review_publish_refs = [(retained_job, retained_thread)]

        ready = pane.shutdown_jobs(timeout_ms=75)

        self.assertTrue(ready)
        self.assertEqual(1, job.cancel_calls)
        self.assertEqual(1, retained_job.cancel_calls)
        self.assertEqual(1, thread.quit_calls)
        self.assertEqual(1, retained_thread.quit_calls)
        self.assertEqual([75], thread.wait_calls)
        self.assertEqual([75], retained_thread.wait_calls)
        self.assertIsNone(pane._face_review_publish_thread)
        self.assertEqual([], pane._retained_face_review_publish_refs)
        pane.close()

    def test_shutdown_jobs_cancels_face_review_hydration_threads(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        job = self._FakeAsyncJob()
        thread = self._FakeAsyncThread(running=True, wait_result=True)
        retained_job = self._FakeAsyncJob()
        retained_thread = self._FakeAsyncThread(running=True, wait_result=True)
        pane._face_review_hydration_job = job
        pane._face_review_hydration_thread = thread
        pane._retained_face_review_hydration_refs = [(retained_job, retained_thread)]

        ready = pane.shutdown_jobs(timeout_ms=75)

        self.assertTrue(ready)
        self.assertEqual(1, job.cancel_calls)
        self.assertEqual(1, retained_job.cancel_calls)
        self.assertEqual(1, thread.quit_calls)
        self.assertEqual(1, retained_thread.quit_calls)
        self.assertEqual([75], thread.wait_calls)
        self.assertEqual([75], retained_thread.wait_calls)
        self.assertIsNone(pane._face_review_hydration_thread)
        self.assertEqual([], pane._retained_face_review_hydration_refs)
        pane.close()

    def test_shutdown_jobs_cancels_face_album_refresh_and_publish_threads(self):
        pane = SearchPane(enabled_tabs=["Face Library"], external_results=False)
        refresh_job = self._FakeAsyncJob()
        refresh_thread = self._FakeAsyncThread(running=True, wait_result=True)
        retained_refresh_job = self._FakeAsyncJob()
        retained_refresh_thread = self._FakeAsyncThread(running=True, wait_result=True)
        publish_job = self._FakeAsyncJob()
        publish_thread = self._FakeAsyncThread(running=True, wait_result=True)
        retained_publish_job = self._FakeAsyncJob()
        retained_publish_thread = self._FakeAsyncThread(running=True, wait_result=True)
        pane._face_album_refresh_job = refresh_job
        pane._face_album_refresh_thread = refresh_thread
        pane._retained_face_album_refresh_refs = [(retained_refresh_job, retained_refresh_thread)]
        pane._face_album_publish_job = publish_job
        pane._face_album_publish_thread = publish_thread
        pane._retained_face_album_publish_refs = [(retained_publish_job, retained_publish_thread)]

        ready = pane.shutdown_jobs(timeout_ms=75)

        self.assertTrue(ready)
        self.assertEqual(1, refresh_job.cancel_calls)
        self.assertEqual(1, retained_refresh_job.cancel_calls)
        self.assertEqual(1, publish_job.cancel_calls)
        self.assertEqual(1, retained_publish_job.cancel_calls)
        self.assertEqual([75], refresh_thread.wait_calls)
        self.assertEqual([75], retained_refresh_thread.wait_calls)
        self.assertEqual([75], publish_thread.wait_calls)
        self.assertEqual([75], retained_publish_thread.wait_calls)
        self.assertIsNone(pane._face_album_refresh_thread)
        self.assertIsNone(pane._face_album_publish_thread)
        self.assertEqual([], pane._retained_face_album_refresh_refs)
        self.assertEqual([], pane._retained_face_album_publish_refs)
        pane.close()

    def test_face_pipeline_falls_back_to_ready_managed_detector_when_saved_detector_is_not_installed(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        bundled_root = Path(__file__).resolve().parents[1] / "face_model_assets"
        with TemporaryDirectory() as tmp:
            managed_root = Path(tmp)
            detector_dir = managed_root / "human" / "detectors" / "scrfd_2.5g_kps"
            embedder_dir = managed_root / "human" / "embedders" / "arcface_r50"
            detector_dir.mkdir(parents=True, exist_ok=True)
            embedder_dir.mkdir(parents=True, exist_ok=True)
            (detector_dir / "detector.onnx").write_bytes(b"detector")
            (embedder_dir / "embedder.onnx").write_bytes(b"embedder")
            (detector_dir / "metadata.json").write_text(
                (bundled_root / "human" / "detectors" / "scrfd_2.5g_kps" / "metadata.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            (embedder_dir / "metadata.json").write_text(
                (bundled_root / "human" / "embedders" / "arcface_r50" / "metadata.json").read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            with patch("app.services.face_search.face_model_runtime_root_dir", return_value=managed_root):
                pane.configure_face_pipeline_options(
                    "",
                    {
                        "human": {
                            "detector_id": "scrfd_500m_kps",
                            "embedder_id": "arcface_r50",
                            "score_threshold": 0.35,
                            "max_detections": 50,
                        }
                    },
                    refresh=False,
                )

            self.assertEqual("scrfd_2.5g_kps", pane.current_face_detector_id())
            self.assertEqual("arcface_r50", pane.current_face_embedder_id())

        pane.close()

    def test_face_search_shows_shared_selected_face_context(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        service = self._FakeFaceLibraryService(
            records=[
                self._face_record("/photos/a.jpg", 0),
                self._face_record("/photos/a.jpg", 1),
            ],
        )
        pane.face_service_global = service
        pane.face_folder_path.setText("/photos")

        pane._refresh_scanned_faces()
        self._select_list_view_row(pane.face_scanned_list, 0)
        pane._on_scanned_face_selection_changed()
        pane.set_active_tab_index(1)

        self.assertIn("a.jpg", pane.face_selected_faces_context_label.text())
        self.assertIn("selected 1", pane.face_selected_faces_context_label.text())
        self.assertTrue(pane.face_search_selected_button.isEnabled())
        self.assertTrue(pane.face_save_name_button.isEnabled())

        pane.close()

    def test_face_management_actions_require_confirmation(self):
        pane = SearchPane(enabled_tabs=["Face Library", "Face Search"], external_results=False)
        service = self._FakeFaceLibraryService(
            profiles=[self._person_profile("Alice", visible_face_count=1)],
            records=[self._face_record("/photos/a.jpg", 0, person_name="Alice")],
        )
        pane.face_service_global = service
        pane.face_folder_path.setText("/photos")
        pane.person_name.setText("Alice")
        pane.merge_source_person.setText("Alice A")
        pane.merge_target_person.setText("Alice")
        pane._start_job = lambda _label, run, done: done(run(lambda *_args: None, lambda: False))

        with patch("ui.search_pane.confirmBox", return_value=False):
            pane._merge_people()
            pane._clear_person()

        self.assertEqual([], service.merge_person_labels_calls)
        self.assertEqual([], service.clear_person_labels_calls)

        with patch("ui.search_pane.confirmBox", return_value=True):
            pane._merge_people()
            pane._clear_person()

        self.assertEqual([("Alice A", "Alice")], service.merge_person_labels_calls)
        self.assertEqual(["Alice"], service.clear_person_labels_calls)

        pane.close()

    def test_selection_details_shutdown_jobs_cancels_active_and_retained_threads(self):
        pane = SelectionDetailsPane()
        active_job = self._FakeAsyncJob()
        active_thread = self._FakeAsyncThread()
        retained_job = self._FakeAsyncJob()
        retained_thread = self._FakeAsyncThread()
        pane._active_job = active_job
        pane._active_thread = active_thread
        pane._retained_async_refs = [(retained_job, retained_thread)]

        ready_to_close = pane.shutdown_jobs(timeout_ms=15)

        self.assertTrue(ready_to_close)
        self.assertEqual(1, active_job.cancel_calls)
        self.assertEqual(1, retained_job.cancel_calls)
        self.assertEqual([15], active_thread.wait_calls)
        self.assertEqual([15], retained_thread.wait_calls)
        self.assertEqual([], pane._retained_async_refs)
        pane.close()

    def test_photo_inspector_shutdown_jobs_cancels_active_and_preview_threads(self):
        dialog = PhotoInspectorDialog(image_paths=[], start_index=0)
        active_job = self._FakeAsyncJob()
        active_thread = self._FakeAsyncThread()
        preview_job = self._FakeAsyncJob()
        preview_thread = self._FakeAsyncThread()
        face_edit_job = self._FakeAsyncJob()
        face_edit_thread = self._FakeAsyncThread()
        retained_job = self._FakeAsyncJob()
        retained_thread = self._FakeAsyncThread()
        dialog._active_job = active_job
        dialog._active_thread = active_thread
        dialog._preview_job = preview_job
        dialog._preview_thread = preview_thread
        dialog._face_edit_job = face_edit_job
        dialog._face_edit_thread = face_edit_thread
        dialog._retained_async_refs = [(retained_job, retained_thread)]

        ready_to_close = dialog.shutdown_jobs(timeout_ms=15)

        self.assertTrue(ready_to_close)
        self.assertEqual(1, active_job.cancel_calls)
        self.assertEqual(1, preview_job.cancel_calls)
        self.assertEqual(1, face_edit_job.cancel_calls)
        self.assertEqual(1, retained_job.cancel_calls)
        self.assertEqual([15], active_thread.wait_calls)
        self.assertEqual([15], preview_thread.wait_calls)
        self.assertEqual([15], face_edit_thread.wait_calls)
        self.assertEqual([15], retained_thread.wait_calls)
        self.assertEqual([], dialog._retained_async_refs)
        dialog.close()

    def test_photo_inspector_basic_mode_hides_metadata_and_shows_path_summary(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "sample-photo.jpg"
            image_path.write_bytes(b"stub")
            with patch.object(PhotoInspectorDialog, "_load_preview"), patch.object(
                PhotoInspectorDialog, "_prefetch_neighbors"
            ):
                dialog = PhotoInspectorDialog(image_paths=[str(image_path)], display_mode="basic")
                dialog.show()
                APP.processEvents()

                self.assertEqual(image_path.name, dialog.name_label.text())
                self.assertEqual(str(image_path.parent), dialog.folder_label.text())
                self.assertEqual(image_path.name, dialog.name_label.toolTip())
                self.assertEqual(str(image_path.parent), dialog.folder_label.toolTip())
                self.assertEqual(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, dialog.name_label.alignment())
                self.assertEqual(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, dialog.folder_label.alignment())
                self.assertIn("Metadata is hidden", dialog.state_label.text())
                self.assertFalse(dialog.metadata_summary_label.isVisible())
                self.assertFalse(dialog.info_text.isVisible())

                dialog.close()

    def test_photo_inspector_uses_horizontal_splitter_with_details_on_right(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "long-photo-name-for-alignment-check.jpg"
            image_path.write_bytes(b"stub")
            with patch.object(PhotoInspectorDialog, "_load_preview"), patch.object(
                PhotoInspectorDialog, "_prefetch_neighbors"
            ):
                dialog = PhotoInspectorDialog(image_paths=[str(image_path)], display_mode="basic")
                dialog.resize(1200, 800)
                dialog.show()
                APP.processEvents()

                self.assertIsInstance(dialog.content_splitter, QSplitter)
                self.assertEqual(Qt.Orientation.Horizontal, dialog.content_splitter.orientation())
                self.assertEqual(0, dialog.content_splitter.indexOf(dialog.image_panel))
                self.assertEqual(1, dialog.content_splitter.indexOf(dialog.details_panel))
                self.assertGreaterEqual(dialog.details_panel.minimumWidth(), 380)
                self.assertLessEqual(dialog.details_panel.maximumWidth(), 520)
                self.assertGreater(dialog.details_panel.mapTo(dialog, dialog.details_panel.rect().topLeft()).x(), dialog.image_panel.mapTo(dialog, dialog.image_panel.rect().topLeft()).x())
                self.assertEqual("Name", dialog.name_title_label.text())
                self.assertEqual("Folder", dialog.folder_title_label.text())
                self.assertEqual(image_path.name, dialog.name_label.toolTip())

                dialog.close()

    def test_photo_inspector_advanced_mode_keeps_metadata_on_right_panel(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "advanced-photo.jpg"
            image_path.write_bytes(b"stub")
            with patch.object(PhotoInspectorDialog, "_load_preview"), patch.object(
                PhotoInspectorDialog, "_load_metadata_async"
            ), patch.object(PhotoInspectorDialog, "_prefetch_neighbors"):
                dialog = PhotoInspectorDialog(image_paths=[str(image_path)], display_mode="advanced")
                dialog.show()
                APP.processEvents()

                self.assertTrue(dialog.metadata_summary_label.isVisible())
                self.assertTrue(dialog.info_text.isVisible())
                self.assertEqual(1, dialog.content_splitter.indexOf(dialog.details_panel))
                self.assertIn("Loading metadata", dialog.metadata_summary_label.text())

                dialog.close()

    def test_photo_inspector_face_editor_can_add_remove_and_save_current_image_faces(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "editable-photo.jpg"
            Image.new("RGB", (96, 96), (20, 30, 40)).save(image_path)
            service = self._FakeFaceLibraryService(
                records=[
                    self._face_record(
                        str(image_path),
                        0,
                        person_name="Alice",
                        bbox=(8, 10, 44, 52),
                    )
                ]
            )
            saved_paths: list[str] = []

            def _sync_face_job(_label, fn, on_completed):
                on_completed(fn(lambda *_args: None, lambda: False))

            def _auto_clean(path, drafts):
                self.assertEqual(str(image_path), str(path))
                return drafts[:1], {"kept": 1, "removed": 1, "review": 0}

            with patch.object(PhotoInspectorDialog, "_load_preview"), patch.object(
                PhotoInspectorDialog, "_load_metadata_async"
            ), patch.object(PhotoInspectorDialog, "_prefetch_neighbors"):
                dialog = PhotoInspectorDialog(
                    image_paths=[str(image_path)],
                    display_mode="basic",
                    context={
                        "face_review": {"status": "detected", "total_face_count": 1, "hidden_face_count": 0},
                        "indexed_faces": {
                            "count": 1,
                            "faces": [
                                {
                                    "face_index": 0,
                                    "bbox": (8, 10, 44, 52),
                                    "confidence": 0.97,
                                    "person_name": "Alice",
                                }
                            ],
                        },
                    },
                    face_service=service,
                    allow_face_edit=True,
                    face_edit_saved_callback=lambda path: saved_paths.append(str(path)),
                    face_auto_clean_callback=_auto_clean,
                )
                dialog._start_face_edit_job = _sync_face_job  # type: ignore[method-assign]
                dialog.show()
                APP.processEvents()

                self.assertTrue(dialog.face_editor_panel.isVisible())
                self.assertEqual(1, self._list_view_count(dialog.image_faces_list))
                self.assertEqual("Selected face: none", dialog.face_selected_details_label.text())

                dialog._on_preview_face_box_drawn((48, 12, 84, 56))
                APP.processEvents()
                self.assertEqual(2, self._list_view_count(dialog.image_faces_list))

                dialog._auto_clean_faces()
                APP.processEvents()
                self.assertEqual(1, self._list_view_count(dialog.image_faces_list))

                dialog._on_preview_face_box_drawn((52, 16, 88, 60))
                APP.processEvents()
                self.assertEqual(2, self._list_view_count(dialog.image_faces_list))

                dialog.image_faces_list.selectionModel().clearSelection()
                self._select_list_view_row(dialog.image_faces_list, 0)
                dialog._on_face_draft_selection_changed()
                self.assertIn("Selected face #1", dialog.face_selected_details_label.text())
                dialog._remove_selected_faces()
                APP.processEvents()
                self.assertEqual(1, self._list_view_count(dialog.image_faces_list))

                with patch("ui.photo_inspector_dialog.confirmBox", return_value=True):
                    dialog._remove_all_faces()
                APP.processEvents()
                self.assertEqual(0, self._list_view_count(dialog.image_faces_list))

                dialog._on_preview_face_box_drawn((48, 12, 84, 56))
                APP.processEvents()
                dialog._save_face_edits()
                APP.processEvents()
                self.assertEqual([str(image_path)], saved_paths)
                self.assertEqual(1, len(service.load_image_faces(str(image_path), include_tiny_faces=True)))
                self.assertEqual((48, 12, 84, 56), service.load_image_faces(str(image_path), include_tiny_faces=True)[0].face_bbox)

                dialog.close()

    def test_photo_inspector_face_editor_supports_resize_nudge_history_duplicate_and_split(self):
        with TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "editable-ops-photo.jpg"
            Image.new("RGB", (96, 96), (20, 30, 40)).save(image_path)
            service = self._FakeFaceLibraryService(
                records=[
                    self._face_record(
                        str(image_path),
                        0,
                        person_name="Alice",
                        bbox=(8, 10, 44, 52),
                    )
                ]
            )

            with patch.object(PhotoInspectorDialog, "_load_preview"), patch.object(
                PhotoInspectorDialog, "_load_metadata_async"
            ), patch.object(PhotoInspectorDialog, "_prefetch_neighbors"):
                dialog = PhotoInspectorDialog(
                    image_paths=[str(image_path)],
                    display_mode="basic",
                    context={
                        "face_review": {"status": "detected", "total_face_count": 1, "hidden_face_count": 0},
                        "indexed_faces": {
                            "count": 1,
                            "faces": [
                                {
                                    "face_index": 0,
                                    "bbox": (8, 10, 44, 52),
                                    "confidence": 0.97,
                                    "person_name": "Alice",
                                }
                            ],
                        },
                    },
                    face_service=service,
                    allow_face_edit=True,
                )
                dialog.show()
                APP.processEvents()

                self.assertEqual([(8.0, 10.0, 44.0, 52.0)], list(dialog.preview_view._face_boxes))

                self._select_list_view_row(dialog.image_faces_list, 0)
                dialog._on_face_draft_selection_changed()
                dialog._on_preview_face_box_resized(0, (6, 8, 48, 56))
                APP.processEvents()
                self.assertEqual((6, 8, 48, 56), dialog._face_drafts_by_path[str(image_path)][0].bbox)
                self.assertEqual((6.0, 8.0, 48.0, 56.0), dialog.preview_view._face_boxes[0])

                dialog._duplicate_selected_face()
                APP.processEvents()
                self.assertEqual(2, self._list_view_count(dialog.image_faces_list))
                duplicated_bbox = dialog._face_drafts_by_path[str(image_path)][1].bbox
                self.assertNotEqual((6, 8, 48, 56), duplicated_bbox)

                dialog._undo_face_edit()
                APP.processEvents()
                self.assertEqual(1, self._list_view_count(dialog.image_faces_list))

                dialog._redo_face_edit()
                APP.processEvents()
                self.assertEqual(2, self._list_view_count(dialog.image_faces_list))

                dialog.image_faces_list.selectionModel().clearSelection()
                self._select_list_view_row(dialog.image_faces_list, 1)
                dialog._on_face_draft_selection_changed()
                dialog._split_selected_face()
                APP.processEvents()
                self.assertEqual(3, self._list_view_count(dialog.image_faces_list))

                dialog.image_faces_list.selectionModel().clearSelection()
                self._select_list_view_row(dialog.image_faces_list, 0)
                dialog._on_face_draft_selection_changed()
                before_nudge = dialog._face_drafts_by_path[str(image_path)][0].bbox
                QTest.keyClick(dialog, Qt.Key.Key_Right)
                APP.processEvents()
                after_nudge = dialog._face_drafts_by_path[str(image_path)][0].bbox
                self.assertEqual(before_nudge[0] + 1, after_nudge[0])
                self.assertEqual(before_nudge[2] + 1, after_nudge[2])

                dialog._undo_face_edit()
                APP.processEvents()
                self.assertEqual(before_nudge, dialog._face_drafts_by_path[str(image_path)][0].bbox)

                dialog.close()

    def test_photo_inspector_arrow_shortcuts_navigate_when_details_has_focus(self):
        with TemporaryDirectory() as tmp:
            paths = []
            for name in ["a.jpg", "b.jpg", "c.jpg"]:
                path = Path(tmp) / name
                path.write_bytes(b"stub")
                paths.append(path)
            with patch.object(PhotoInspectorDialog, "_load_preview"), patch.object(
                PhotoInspectorDialog, "_load_metadata_async"
            ), patch.object(PhotoInspectorDialog, "_prefetch_neighbors"):
                dialog = PhotoInspectorDialog(
                    image_paths=[str(path) for path in paths],
                    start_index=1,
                    display_mode="advanced",
                )
                dialog.show()
                APP.processEvents()

                dialog.info_text.setFocus()
                APP.processEvents()
                QTest.keyClick(dialog.info_text, Qt.Key.Key_Left)
                APP.processEvents()
                self.assertEqual(0, dialog._index)
                self.assertIn(paths[0].name, dialog.windowTitle())

                dialog.info_text.setFocus()
                APP.processEvents()
                QTest.keyClick(dialog.info_text, Qt.Key.Key_Right)
                APP.processEvents()
                self.assertEqual(1, dialog._index)
                self.assertIn(paths[1].name, dialog.windowTitle())

                dialog.close()

    def test_photo_inspector_metadata_render_emphasizes_key_fields(self):
        metadata = SimpleNamespace(
            image_path="D:/images/demo.jpg",
            width=1920,
            height=1080,
            file_size=3145728,
            modified_at="2026-05-07 12:00:00",
            camera="Pixel",
            exif={"ISO": "100", "LensModel": "Wide"},
            hashes={"phash": "abc"},
            face_boxes=[],
            context={"tags": ["travel", "sunset"], "membership": {"clip::graph": {"cluster_id": 3}}},
        )

        summary_html = PhotoInspectorDialog._render_metadata_summary(metadata)
        details_html = PhotoInspectorDialog._render_metadata(metadata)

        self.assertIn("<b>Dimensions</b>", summary_html)
        self.assertIn("<b>Size</b>", summary_html)
        self.assertIn("travel, sunset", summary_html)
        self.assertIn("<h3>EXIF</h3>", details_html)
        self.assertIn("Cluster Context", details_html)
        self.assertIn("LensModel", details_html)

    def test_gallery_shutdown_jobs_cancels_active_and_retained_action_threads(self):
        pane = GalleryPane()
        active_job = self._FakeAsyncJob()
        active_thread = self._FakeAsyncThread()
        retained_job = self._FakeAsyncJob()
        retained_thread = self._FakeAsyncThread()
        pane._active_action_job = active_job
        pane._active_action_thread = active_thread
        pane._retained_action_refs = [(retained_job, retained_thread)]

        ready_to_close = pane.shutdown_jobs(timeout_ms=15)

        self.assertTrue(ready_to_close)
        self.assertEqual(1, active_job.cancel_calls)
        self.assertEqual(1, retained_job.cancel_calls)
        self.assertEqual([15], active_thread.wait_calls)
        self.assertEqual([15], retained_thread.wait_calls)
        self.assertEqual([], pane._retained_action_refs)
        pane.close()

    def test_search_pane_shutdown_jobs_cancels_active_thread(self):
        pane = SearchPane(enabled_tabs=[], external_results=True)
        active_job = self._FakeAsyncJob()
        active_thread = self._FakeAsyncThread()
        pane._active_job = active_job
        pane._active_thread = active_thread

        ready_to_close = pane.shutdown_jobs(timeout_ms=15)

        self.assertTrue(ready_to_close)
        self.assertEqual(1, active_job.cancel_calls)
        self.assertEqual([15], active_thread.wait_calls)
        self.assertIsNone(pane._active_job)
        self.assertIsNone(pane._active_thread)
        pane.close()

    def test_settings_dialog_shutdown_jobs_cancels_async_threads(self):
        dialog = SettingsDialog(
            QSettings("NOC", "SettingsShutdown"),
            RuntimeCapabilityService(),
            describe_rebuildable_caches=lambda: CacheUsageSummary(
                cache_root="D:/cache",
                target_bytes={"embeddings.sqlite3": 1024},
                total_bytes=1024,
            ),
            prepare_rebuildable_cache_clear=lambda: None,
            clear_rebuildable_caches=lambda: CacheClearResult(
                cleared_targets=("embeddings.sqlite3",),
                freed_bytes=1024,
                failures=(),
            ),
            can_clear_rebuildable_caches=lambda: True,
        )
        deadline = monotonic() + 1.5
        while dialog._cache_usage_thread is not None and monotonic() < deadline:
            dialog._cache_usage_thread.wait(50)
            APP.processEvents()
        usage_job = self._FakeAsyncJob()
        usage_thread = self._FakeAsyncThread()
        clear_job = self._FakeAsyncJob()
        clear_thread = self._FakeAsyncThread()
        verify_job = self._FakeAsyncJob()
        verify_thread = self._FakeAsyncThread()
        face_model_job = self._FakeAsyncJob()
        face_model_thread = self._FakeAsyncThread()
        dialog._cache_usage_job = usage_job
        dialog._cache_usage_thread = usage_thread
        dialog._cache_clear_job = clear_job
        dialog._cache_clear_thread = clear_thread
        dialog._verify_job = verify_job
        dialog._verify_thread = verify_thread
        dialog._face_model_job = face_model_job
        dialog._face_model_thread = face_model_thread

        ready_to_close = dialog.shutdown_jobs(timeout_ms=15)

        self.assertTrue(ready_to_close)
        self.assertEqual(1, usage_job.cancel_calls)
        self.assertEqual(1, clear_job.cancel_calls)
        self.assertEqual(1, verify_job.cancel_calls)
        self.assertEqual(1, face_model_job.cancel_calls)
        self.assertEqual([15], usage_thread.wait_calls)
        self.assertEqual([15], clear_thread.wait_calls)
        self.assertEqual([15], verify_thread.wait_calls)
        self.assertEqual([15], face_model_thread.wait_calls)
        self.assertIsNone(dialog._cache_usage_thread)
        self.assertIsNone(dialog._cache_clear_thread)
        self.assertIsNone(dialog._verify_thread)
        self.assertIsNone(dialog._face_model_thread)
        dialog.close()


if __name__ == "__main__":
    unittest.main()
