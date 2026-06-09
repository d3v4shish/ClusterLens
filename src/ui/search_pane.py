from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from PIL import Image
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import QSize, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QGuiApplication, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFileDialog,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.services.face_search import FaceIndexService, FaceLabelRequest, FaceSearchRequest, IndexedFaceRecord
from app.services.similarity_search import SearchResult, SimilaritySearchRequest, SimilaritySearchService
from infra.settings import get_settings
from ui.error_mbox import errorBox, infoBox
from ui.async_job import AsyncJob, start_job_in_thread
from ui.gallery_pane import GalleryPane

try:
    from ui.job_manager import JobManager
except Exception:  # pragma: no cover
    JobManager = None  # type: ignore[assignment]


class PasteAwareLineEdit(QLineEdit):
    def __init__(self, on_paste, parent=None):
        super().__init__(parent)
        self._on_paste = on_paste

    def insertFromMimeData(self, source) -> None:  # type: ignore[override]
        try:
            if self._on_paste is not None and self._on_paste(source):
                return
        except Exception:
            pass
        super().insertFromMimeData(source)


class SearchPane(QWidget):
    open_in_gallery_requested = pyqtSignal(list)
    append_to_gallery_requested = pyqtSignal(list)
    face_clusters_ready = pyqtSignal(dict)
    result_selected = pyqtSignal(str, list, int, object)
    active_tab_changed = pyqtSignal(int)
    results_ready = pyqtSignal(list, object, object, object, str)

    def __init__(
        self,
        parent=None,
        *,
        search_service_global=None,
        search_service_session=None,
        face_service_global=None,
        face_service_session=None,
        enabled_tabs: list[str] | None = None,
        external_results: bool = False,
    ):
        super().__init__(parent)
        self.settings = get_settings()
        self.current_directory_provider = None
        self.current_scope_paths_provider = None
        self.use_onnx_provider = None  # injected callable() -> bool
        self.job_manager: JobManager | None = None  # injected by main window
        self.search_service_global = search_service_global or SimilaritySearchService()
        self.search_service_session = search_service_session or SimilaritySearchService()
        self.search_service = self.search_service_global
        self.face_service_global = face_service_global or FaceIndexService()
        self.face_service_session = face_service_session or FaceIndexService(db_path=(self.settings.cache_dir / "face_search_session.db"), reset_db=True)
        self.face_service = self.face_service_global
        self._active_thread = None
        self._active_job = None
        self._active_job_id: int | None = None
        self._thread_jobs: dict[object, object | None] = {}
        self._action_buttons: list[QPushButton] = []
        self._temp_query_files: dict[int, str] = {}
        self._auto_reindexed: set[tuple[str, str]] = set()
        self._last_results = []
        self._result_by_path: dict[str, SearchResult] = {}
        self._face_thumb_cache: dict[tuple[str, int, tuple[int, int, int, int]], QIcon] = {}
        self._results_kind = "none"  # similarity|faces|face_clusters|labels|none
        self._enabled_tabs = {str(name).strip().lower() for name in (enabled_tabs or []) if str(name).strip()}
        self._external_results = bool(external_results)
        self.main_layout = QVBoxLayout(self)
        self._build_global_controls()
        self.tabs = QTabWidget()
        self.results_gallery = GalleryPane(self)
        self.results_gallery.review_action_mode = "add"
        self.results_list = QListWidget()
        self.status_label = QLabel("Search index idle.")
        if not self._enabled_tabs or "image search" in self._enabled_tabs:
            self._build_image_tab()
        if not self._enabled_tabs or "text search" in self._enabled_tabs:
            self._build_text_tab()
        if not self._enabled_tabs or "duplicate search" in self._enabled_tabs:
            self._build_duplicate_tab()
        if not self._enabled_tabs or "face library" in self._enabled_tabs:
            self._build_face_library_tab()
        if not self._enabled_tabs or "face search" in self._enabled_tabs:
            self._build_face_tab()
        self.tabs.currentChanged.connect(self.active_tab_changed.emit)
        self.main_layout.addWidget(self.tabs)
        self.status_label.setWordWrap(True)
        self.main_layout.addWidget(self.status_label)
        self.results_toolbar = self._build_results_toolbar()
        self.main_layout.addWidget(self.results_toolbar)
        self._set_results_kind("none")
        self._setup_results_gallery()
        self.main_layout.addWidget(self.results_gallery)
        # Keep the text list for debugging, but default to the gallery UX.
        self.results_list.hide()
        self.main_layout.addWidget(self.results_list)
        self.set_external_results_mode(self._external_results)

    def _active_search_service(self, scope_combo) -> SimilaritySearchService:
        selected = "global"
        try:
            if scope_combo is not None:
                selected = str(scope_combo.currentText()).lower()
        except Exception:
            selected = "global"
        return self.search_service_session if "session" in selected else self.search_service_global

    def _active_face_service(self) -> FaceIndexService:
        scope = getattr(self, "face_db_scope", None)
        selected = "global"
        try:
            if scope is not None:
                selected = str(scope.currentText()).lower()
        except Exception:
            selected = "global"
        return self.face_service_session if "session" in selected else self.face_service_global

    def _build_global_controls(self) -> None:
        row = QHBoxLayout()
        self.search_only_current_folder = QCheckBox("Limit to active scope")
        self.search_only_current_folder.setChecked(True)
        row.addWidget(self.search_only_current_folder)
        row.addStretch(1)
        self.main_layout.addLayout(row)

    def _use_onnx(self) -> bool:
        try:
            if callable(self.use_onnx_provider):
                return bool(self.use_onnx_provider())
        except Exception:
            pass
        return False

    def _model_id(self, combo) -> str:
        try:
            data = combo.currentData()
            if data:
                return str(data)
        except Exception:
            pass
        try:
            return str(combo.currentText())
        except Exception:
            return ""


    def _build_results_toolbar(self) -> QWidget:
        widget = QWidget(self)
        bar = QHBoxLayout(widget)
        bar.setContentsMargins(0, 0, 0, 0)
        self.results_open_btn = QPushButton("Open In Main Gallery")
        self.results_append_btn = QPushButton("Append To Main Gallery")
        self.results_export_btn = QPushButton("Export Paths")
        bar.addWidget(self.results_open_btn)
        bar.addWidget(self.results_append_btn)
        bar.addWidget(self.results_export_btn)
        bar.addSpacing(14)
        bar.addWidget(QLabel("Min score"))
        self.results_min_score = QDoubleSpinBox()
        self.results_min_score.setRange(0.0, 1.0)
        self.results_min_score.setSingleStep(0.01)
        self.results_min_score.setValue(0.0)
        bar.addWidget(self.results_min_score)
        bar.addSpacing(10)
        bar.addWidget(QLabel("Sort"))
        self.results_sort = QComboBox()
        self.results_sort.addItems(["Score (desc)", "Name", "Folder", "Hash dist"])
        bar.addWidget(self.results_sort)
        bar.addStretch(1)

        self.results_open_btn.clicked.connect(self._open_results_in_main_gallery)
        self.results_append_btn.clicked.connect(self._append_results_to_main_gallery)
        self.results_export_btn.clicked.connect(self._export_results_paths)
        self.results_min_score.valueChanged.connect(lambda _v: self._apply_results_view())
        self.results_sort.currentIndexChanged.connect(lambda _i: self._apply_results_view())
        return widget

    def _set_results_kind(self, kind: str) -> None:
        self._results_kind = str(kind)
        is_similarity = self._results_kind == "similarity"
        try:
            self.results_min_score.setEnabled(is_similarity)
            self.results_sort.setEnabled(is_similarity)
        except Exception:
            pass

    def _setup_results_gallery(self) -> None:
        # Hide gallery actions for search results.
        try:
            self.results_gallery.set_action_visibility(
                show_actions=False,
                show_metadata_actions=False,
                show_file_actions=False,
            )
        except Exception:
            pass
        for widget in [
            getattr(self.results_gallery, "select_all_button", None),
            getattr(self.results_gallery, "exif_button", None),
            getattr(self.results_gallery, "selected_tags_button", None),
            getattr(self.results_gallery, "metadata_menu_button", None),
            getattr(self.results_gallery, "file_ops_menu_button", None),
            getattr(self.results_gallery, "more_menu_button", None),
            getattr(self.results_gallery, "move_button", None),
            getattr(self.results_gallery, "delete_button", None),
            getattr(self.results_gallery, "retry_failed_button", None),
            getattr(self.results_gallery, "status_label", None),
            getattr(self.results_gallery, "progress_bar", None),
        ]:
            if widget is not None:
                widget.hide()
        from PyQt6.QtWidgets import QListView

        self.results_gallery.list_view.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self.results_gallery.image_selected.connect(self._on_results_image_selected)

    def _set_busy(self, busy: bool, status: str = "") -> None:
        for button in self._action_buttons:
            button.setEnabled(not busy)
        if status:
            self.status_label.setText(status)

    def _start_job(self, label: str, fn, on_completed) -> None:
        # QThread objects are deleted via deleteLater() in the helper; keep our reference safe.
        if self._active_thread is not None:
            try:
                if self._active_thread.isRunning():
                    errorBox("Busy", "Another operation is already running in SearchPane.")
                    return
            except RuntimeError:
                self._active_thread = None
                self._active_job = None

        job = AsyncJob(fn)

        if self.job_manager is not None:
            self._active_job_id = self.job_manager.register_job(label, cancel_fn=job.cancel)
            job.progress.connect(lambda value, text: self.job_manager.update(self._active_job_id or -1, progress=value, text=text))
        # Always reflect progress text in the pane status for better UX.
        job.progress.connect(lambda _value, text: self.status_label.setText(str(text)))

        def _on_failed(message: str) -> None:
            self._set_busy(False, "Search index idle.")
            if self.job_manager is not None and self._active_job_id is not None:
                self.job_manager.finish(self._active_job_id, status="failed", error=message)
                self._active_job_id = None
            errorBox(f"{label} failed", message)

        def _on_cancelled() -> None:
            self._set_busy(False, "Cancelled.")
            if self.job_manager is not None and self._active_job_id is not None:
                self.job_manager.finish(self._active_job_id, status="cancelled")
                self._active_job_id = None

        def _on_completed(result) -> None:
            self._set_busy(False)
            if self.job_manager is not None and self._active_job_id is not None:
                self.job_manager.finish(self._active_job_id, status="finished")
                self._active_job_id = None
            on_completed(result)

        job.failed.connect(_on_failed)
        job.cancelled.connect(_on_cancelled)
        job.completed.connect(_on_completed)
        self._active_job = job
        thread = start_job_in_thread(job)
        self._thread_jobs[thread] = job
        thread.finished.connect(
            self._on_async_thread_finished,
            Qt.ConnectionType.QueuedConnection,
        )
        self._active_thread = thread
        self._set_busy(True, f"{label} running...")
        if self.job_manager is not None and self._active_job_id is not None:
            self.job_manager.update(self._active_job_id, progress=None, text="Running")

    def _release_finished_thread(self, thread) -> None:
        if thread is None:
            return
        job = self._thread_jobs.pop(thread, None)
        if self._active_thread is thread:
            self._active_thread = None
            if self._active_job is job:
                self._active_job = None

    @pyqtSlot()
    def _on_async_thread_finished(self) -> None:
        self._release_finished_thread(self.sender())

    def shutdown_jobs(self, *, timeout_ms: int = 2500) -> bool:
        ready_to_close = True
        if self._active_job is not None:
            try:
                self._active_job.cancel()
            except Exception:
                pass
        if self._active_thread is not None:
            try:
                if self._active_thread.isRunning():
                    self._active_thread.quit()
                    ready_to_close = bool(self._active_thread.wait(timeout_ms)) and ready_to_close
            except Exception:
                ready_to_close = False
        try:
            ready_to_close = self.results_gallery.shutdown_jobs(timeout_ms=timeout_ms) and ready_to_close
        except Exception:
            ready_to_close = False
        if ready_to_close:
            self._active_job = None
            self._active_thread = None
            self._thread_jobs = {}
        return ready_to_close

    def closeEvent(self, event) -> None:
        if not self.shutdown_jobs():
            event.ignore()
            return
        return super().closeEvent(event)

    def _open_results_in_main_gallery(self) -> None:
        paths = list(getattr(self.results_gallery, "images", []) or [])
        if paths:
            self.open_in_gallery_requested.emit(paths)

    def _append_results_to_main_gallery(self) -> None:
        paths = list(getattr(self.results_gallery, "images", []) or [])
        if paths:
            self.append_to_gallery_requested.emit(paths)

    def _on_results_image_selected(self, image_path: str) -> None:
        paths = list(getattr(self.results_gallery, "images", []) or [])
        if not paths or image_path not in paths:
            return
        self.result_selected.emit(image_path, paths, paths.index(image_path), self.results_gallery.inspector_context_provider)

    def _export_results_paths(self) -> None:
        paths = list(getattr(self.results_gallery, "images", []) or [])
        if not paths:
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Export Result Paths", "", "Text Files (*.txt)")
        if not out_path:
            return
        try:
            Path(out_path).write_text("\n".join(paths), encoding="utf-8")
            infoBox("Export complete", f"Wrote {len(paths)} paths.")
        except Exception as exc:
            errorBox("Export failed", str(exc))

    def _apply_results_view(self) -> None:
        if getattr(self, "_results_kind", "none") != "similarity":
            return
        results = list(self._last_results or [])
        if not results:
            self._result_by_path = {}
            self.results_gallery.update_gallery([])
            try:
                self.results_gallery.model.set_overlays_by_path({})
            except Exception:
                pass
            self.results_gallery.inspector_context_provider = None
            self._publish_results([], lambda _p: {}, {}, {}, self._results_kind)
            return

        # De-duplicate by path while preserving the initial sort order from the service.
        seen: set[str] = set()
        unique: list[SearchResult] = []
        by_path: dict[str, SearchResult] = {}
        for r in results:
            by_path.setdefault(r.image_path, r)
            if r.image_path in seen:
                continue
            seen.add(r.image_path)
            unique.append(r)
        self._result_by_path = by_path

        min_score = float(self.results_min_score.value()) if hasattr(self, "results_min_score") else 0.0
        filtered = [r for r in unique if float(getattr(r, "score", 0.0)) >= min_score]

        sort_mode = self.results_sort.currentText() if hasattr(self, "results_sort") else "Score (desc)"
        if sort_mode == "Name":
            filtered.sort(key=lambda r: (Path(r.image_path).name.lower(), r.image_path))
        elif sort_mode == "Folder":
            filtered.sort(key=lambda r: (str(Path(r.image_path).parent).lower(), Path(r.image_path).name.lower(), r.image_path))
        elif sort_mode == "Hash dist":
            filtered.sort(key=lambda r: ((r.hash_distance if r.hash_distance >= 0 else 9999), -float(r.score), r.image_path))
        else:
            # Score (desc): keep service ordering.
            pass

        paths = [r.image_path for r in filtered]
        self.results_gallery.update_gallery(paths)

        overlay_by_path: dict[str, str] = {}
        for r in filtered:
            parts = [f"{r.model_name} {float(r.score):.3f}"]
            if int(getattr(r, "hash_distance", -1)) >= 0:
                parts.append(f"{r.hash_backend}:{int(r.hash_distance)}")
            if float(getattr(r, "orb_score", 0.0)) > 0.0:
                parts.append(f"orb:{float(r.orb_score):.2f}")
            overlay_by_path[r.image_path] = " | ".join(parts)
        try:
            self.results_gallery.model.set_overlays_by_path(overlay_by_path)
        except Exception:
            pass

        def _ctx(path: str) -> dict[str, object]:
            r = self._result_by_path.get(path)
            if r is None:
                return {}
            return {
                "search": {
                    "score": float(r.score),
                    "model": r.model_name,
                    "hash_backend": r.hash_backend,
                    "hash_distance": int(r.hash_distance),
                    "orb_score": float(r.orb_score),
                    "match_reason": r.match_reason,
                }
            }

        self.results_gallery.inspector_context_provider = _ctx
        self._publish_results(paths, _ctx, overlay_by_path, {}, self._results_kind)

    def set_external_results_mode(self, enabled: bool) -> None:
        self._external_results = bool(enabled)
        self.results_open_btn.setVisible(not self._external_results)
        self.results_append_btn.setVisible(not self._external_results)
        self.results_gallery.setVisible(not self._external_results)
        self.results_list.setVisible(False)

    def _current_scope_paths(self) -> list[str] | None:
        try:
            if callable(self.current_scope_paths_provider):
                paths = self.current_scope_paths_provider()
                if paths:
                    return [str(path) for path in paths if str(path or "").strip()]
        except Exception:
            pass
        return None

    def _publish_results(self, paths: list[str], context_provider, overlays: dict[str, str] | None, subtitles: dict[str, str] | None, kind: str) -> None:
        self.results_ready.emit(list(paths or []), context_provider, dict(overlays or {}), dict(subtitles or {}), str(kind or "none"))

    def _build_image_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)
        self.image_query_path = PasteAwareLineEdit(lambda mime: self._handle_paste_mime(self.image_query_path, mime, multi_append=False), self)
        self.image_db_scope = QComboBox()
        self.image_db_scope.addItems(["Global DB", "Session only"])
        self.image_index_folder = QLineEdit()
        self.image_index_folder.setPlaceholderText("Folder to index (blank = current folder)")
        self.image_model = QComboBox()
        self.image_model.addItem("clip (Torch)", "clip")
        self.image_model.addItem("siglip (Torch)", "siglip")
        self.image_model.addItem("convnext (ONNX/Torch)", "convnext")
        self.image_model.addItem("resnet (ONNX/Torch)", "resnet")
        self.image_model.addItem("phash_embedding (Torch)", "phash_embedding")
        self.image_top_k = QSpinBox()
        self.image_top_k.setRange(1, 500)
        self.image_top_k.setValue(30)
        self.image_min_score = QDoubleSpinBox()
        self.image_min_score.setRange(-1.0, 1.0)
        self.image_min_score.setSingleStep(0.05)
        self.image_min_score.setValue(0.25)
        self.image_folder_filter = QLineEdit()
        form.addRow("Store index in", self.image_db_scope)
        form.addRow("Index Folder", self._folder_row(self.image_index_folder))
        form.addRow("Query Image", self._path_row(self.image_query_path))
        form.addRow("Model", self.image_model)
        form.addRow("Top-K", self.image_top_k)
        form.addRow("Min Score", self.image_min_score)
        form.addRow("Folder Filter", self.image_folder_filter)
        buttons = QHBoxLayout()
        index_button = QPushButton("Index Current Folder")
        search_button = QPushButton("Search Similar Images")
        index_button.clicked.connect(lambda: self._index_images_for(self._model_id(self.image_model), self.image_index_folder.text().strip(), self.image_db_scope))
        search_button.clicked.connect(self._search_by_image)
        self._action_buttons.extend([index_button, search_button])
        buttons.addWidget(index_button)
        buttons.addWidget(search_button)
        form.addRow(buttons)
        self.tabs.addTab(tab, "Image Search")

    def _build_text_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)
        self.text_query = QLineEdit()
        self.text_model = QComboBox()
        self.text_model.addItem("clip (Torch)", "clip")
        self.text_model.addItem("siglip (Torch)", "siglip")
        self.text_top_k = QSpinBox()
        self.text_top_k.setRange(1, 500)
        self.text_top_k.setValue(30)
        self.text_min_score = QDoubleSpinBox()
        self.text_min_score.setRange(-1.0, 1.0)
        self.text_min_score.setSingleStep(0.05)
        self.text_min_score.setValue(0.2)
        form.addRow("Prompt", self.text_query)
        form.addRow("Model", self.text_model)
        form.addRow("Top-K", self.text_top_k)
        form.addRow("Min Score", self.text_min_score)
        buttons = QHBoxLayout()
        index_button = QPushButton("Index Current Folder")
        search_button = QPushButton("Search By Text")
        index_button.clicked.connect(lambda: self._index_images_for(self._model_id(self.text_model), self._current_directory(), None))
        search_button.clicked.connect(self._search_by_text)
        self._action_buttons.extend([index_button, search_button])
        buttons.addWidget(index_button)
        buttons.addWidget(search_button)
        form.addRow(buttons)
        self.tabs.addTab(tab, "Text Search")

    def _build_duplicate_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)
        self.duplicate_query_path = PasteAwareLineEdit(lambda mime: self._handle_paste_mime(self.duplicate_query_path, mime, multi_append=False), self)
        self.duplicate_db_scope = QComboBox()
        self.duplicate_db_scope.addItems(["Global DB", "Session only"])
        self.duplicate_index_folder = QLineEdit()
        self.duplicate_index_folder.setPlaceholderText("Folder to index (blank = current folder)")
        self.duplicate_model = QComboBox()
        self.duplicate_model.addItem("phash_embedding (Torch)", "phash_embedding")
        self.duplicate_model.addItem("clip (Torch)", "clip")
        self.duplicate_model.addItem("siglip (Torch)", "siglip")
        self.duplicate_model.addItem("convnext (ONNX/Torch)", "convnext")
        self.hash_backend = QComboBox()
        self.hash_backend.addItems(["phash", "dhash", "whash"])
        self.ann_backend = QComboBox()
        self.ann_backend.addItems(["flat", "hnsw", "ivf_pq"])
        self.orb_rerank_checkbox = QCheckBox("ORB rerank")
        self.duplicate_top_k = QSpinBox()
        self.duplicate_top_k.setRange(1, 500)
        self.duplicate_top_k.setValue(50)
        self.hash_distance = QSpinBox()
        self.hash_distance.setRange(0, 64)
        self.hash_distance.setValue(12)
        self.ivf_nlist = QSpinBox()
        self.ivf_nlist.setRange(1, 4096)
        self.ivf_nlist.setValue(64)
        self.ivf_nprobe = QSpinBox()
        self.ivf_nprobe.setRange(1, 1024)
        self.ivf_nprobe.setValue(8)
        self.ivf_pq_m = QSpinBox()
        self.ivf_pq_m.setRange(1, 128)
        self.ivf_pq_m.setValue(8)
        self.duplicate_min_score = QDoubleSpinBox()
        self.duplicate_min_score.setRange(-1.0, 1.0)
        self.duplicate_min_score.setSingleStep(0.05)
        self.duplicate_min_score.setValue(0.1)
        form.addRow("Store index in", self.duplicate_db_scope)
        form.addRow("Index Folder", self._folder_row(self.duplicate_index_folder))
        form.addRow("Query Image", self._path_row(self.duplicate_query_path))
        form.addRow("Rerank Model", self.duplicate_model)
        form.addRow("Hash Backend", self.hash_backend)
        form.addRow("ANN Backend", self.ann_backend)
        form.addRow("Top-K", self.duplicate_top_k)
        form.addRow("Hash Distance", self.hash_distance)
        form.addRow("Min Score", self.duplicate_min_score)
        form.addRow("IVF nlist", self.ivf_nlist)
        form.addRow("IVF nprobe", self.ivf_nprobe)
        form.addRow("IVF PQ m", self.ivf_pq_m)
        form.addRow(self.orb_rerank_checkbox)
        buttons = QHBoxLayout()
        index_button = QPushButton("Index Current Folder")
        search_button = QPushButton("Search Duplicates")
        index_button.clicked.connect(lambda: self._index_images_for(self._model_id(self.duplicate_model), self.duplicate_index_folder.text().strip(), self.duplicate_db_scope))
        search_button.clicked.connect(self._search_duplicates)
        self._action_buttons.extend([index_button, search_button])
        buttons.addWidget(index_button)
        buttons.addWidget(search_button)
        form.addRow(buttons)
        self.tabs.addTab(tab, "Duplicate Search")

    def _build_face_library_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)

        self.face_db_scope = QComboBox()
        self.face_db_scope.addItems(["Global DB", "Session only"])

        self.face_folder_path = QLineEdit()
        self.face_folder_path.setPlaceholderText("Folder to scan for faces (optional)")

        self.face_people_list = QListWidget()
        self.face_people_list.setMaximumHeight(150)

        self.face_name_query = QLineEdit()
        self.face_name_query.setPlaceholderText("Person name (must be saved as a prototype)")

        self.face_name_top_k = QSpinBox()
        self.face_name_top_k.setRange(1, 500)
        self.face_name_top_k.setValue(60)

        self.face_name_min_score = QDoubleSpinBox()
        self.face_name_min_score.setRange(0.0, 1.0)
        self.face_name_min_score.setSingleStep(0.01)
        self.face_name_min_score.setValue(0.0)

        self.face_name_folder_filter = QLineEdit()
        self.face_name_folder_filter.setPlaceholderText("Folder filter (blank = all)")

        self.face_label_name = QLineEdit()
        self.face_label_name.setPlaceholderText("Assign this name to the selected face(s)")
        self.face_profile_notes = QLineEdit()
        self.face_profile_notes.setPlaceholderText("Profile notes")
        self.face_profile_tags = QLineEdit()
        self.face_profile_tags.setPlaceholderText("Comma-separated tags")

        self.face_library_label_threshold = QDoubleSpinBox()
        self.face_library_label_threshold.setRange(0.0, 1.0)
        self.face_library_label_threshold.setSingleStep(0.01)
        self.face_library_label_threshold.setValue(0.72)

        self.face_browser_top_k = QSpinBox()
        self.face_browser_top_k.setRange(1, 500)
        self.face_browser_top_k.setValue(40)

        self.face_browser_min_score = QDoubleSpinBox()
        self.face_browser_min_score.setRange(0.0, 1.0)
        self.face_browser_min_score.setSingleStep(0.01)
        self.face_browser_min_score.setValue(0.45)

        self.face_browser_limit = QSpinBox()
        self.face_browser_limit.setRange(25, 5000)
        self.face_browser_limit.setSingleStep(25)
        self.face_browser_limit.setValue(400)

        self.face_scanned_summary = QLabel("No scanned faces loaded.")
        self.face_scanned_summary.setWordWrap(True)

        self.face_scanned_list = QListWidget()
        self.face_scanned_list.setViewMode(QListView.ViewMode.IconMode)
        self.face_scanned_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.face_scanned_list.setMovement(QListView.Movement.Static)
        self.face_scanned_list.setWrapping(True)
        self.face_scanned_list.setWordWrap(True)
        self.face_scanned_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.face_scanned_list.setIconSize(QSize(112, 112))
        self.face_scanned_list.setGridSize(QSize(148, 176))
        self.face_scanned_list.setSpacing(10)
        self.face_scanned_list.itemSelectionChanged.connect(self._on_scanned_face_selection_changed)

        folder_row = QWidget()
        folder_layout = QHBoxLayout(folder_row)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_face_folder)
        folder_layout.addWidget(self.face_folder_path)
        folder_layout.addWidget(browse_btn)

        form.addRow("Store faces in", self.face_db_scope)
        form.addRow("Folder", folder_row)

        buttons = QHBoxLayout()
        scan_btn = QPushButton("Scan Faces")
        refresh_btn = QPushButton("Refresh People")
        refresh_faces_btn = QPushButton("Refresh Scanned Faces")
        scan_btn.clicked.connect(self._scan_face_folder)
        refresh_btn.clicked.connect(self._refresh_face_people)
        refresh_faces_btn.clicked.connect(self._refresh_scanned_faces)
        self._action_buttons.extend([scan_btn, refresh_btn, refresh_faces_btn])
        buttons.addWidget(scan_btn)
        buttons.addWidget(refresh_btn)
        buttons.addWidget(refresh_faces_btn)
        form.addRow(buttons)

        form.addRow(QLabel("Saved people (prototypes)"))
        form.addRow(self.face_people_list)

        name_row = QWidget()
        name_layout = QHBoxLayout(name_row)
        name_layout.setContentsMargins(0, 0, 0, 0)
        search_name_btn = QPushButton("Search By Name")
        search_name_btn.clicked.connect(self._search_by_name)
        self._action_buttons.append(search_name_btn)
        name_layout.addWidget(self.face_name_query)
        name_layout.addWidget(search_name_btn)

        form.addRow("Name", name_row)
        form.addRow("Top-K", self.face_name_top_k)
        form.addRow("Min score (0=use saved threshold)", self.face_name_min_score)
        form.addRow("Folder Filter", self.face_name_folder_filter)
        form.addRow("Selected Face Name", self.face_label_name)
        form.addRow("Profile Notes", self.face_profile_notes)
        form.addRow("Profile Tags", self.face_profile_tags)
        form.addRow("Label Threshold", self.face_library_label_threshold)
        form.addRow("Face Match Top-K", self.face_browser_top_k)
        form.addRow("Face Match Min Score", self.face_browser_min_score)
        form.addRow("Shown Faces", self.face_browser_limit)

        face_actions = QHBoxLayout()
        save_name_btn = QPushButton("Save Name From Selected")
        save_profile_btn = QPushButton("Save Profile")
        search_selected_btn = QPushButton("Search Selected Face")
        save_name_btn.clicked.connect(self._save_selected_face_name)
        save_profile_btn.clicked.connect(self._save_person_profile)
        search_selected_btn.clicked.connect(self._search_selected_face)
        self._action_buttons.extend([save_name_btn, save_profile_btn, search_selected_btn])
        face_actions.addWidget(save_name_btn)
        face_actions.addWidget(save_profile_btn)
        face_actions.addWidget(search_selected_btn)
        form.addRow(face_actions)

        form.addRow(self.face_scanned_summary)
        form.addRow(self.face_scanned_list)

        self.face_people_list.itemClicked.connect(self._on_person_clicked)
        self.tabs.addTab(tab, "Face Library")

    def _browse_face_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Folder")
        if directory:
            self.face_folder_path.setText(str(directory))

    def _scan_face_folder(self) -> None:
        directory = (self.face_folder_path.text().strip() or self._current_directory())
        scope_paths = self._current_scope_paths()
        if not directory and not scope_paths:
            errorBox("No folder selected", "Choose a folder path or select a folder in the sidebar first.")
            return
        service = self._active_face_service()

        def _run(progress, cancel_check):
            progress(0, "0/0 images, faces=0")
            if scope_paths:
                return service.index_paths(scope_paths, progress_callback=progress, cancel_check=cancel_check)
            return service.index_directory(directory, recursive=True, progress_callback=progress, cancel_check=cancel_check)

        def _done(metrics: dict) -> None:
            faces = int((metrics or {}).get("faces_indexed", 0))
            done = int((metrics or {}).get("images_done", 0))
            total = int((metrics or {}).get("images_total", 0))
            source_label = "working set" if scope_paths else directory
            self.status_label.setText(f"Indexed {faces} faces from {done}/{total} images in {source_label}.")
            self._refresh_face_people()
            self._refresh_scanned_faces()

        self._start_job("Indexing faces", _run, _done)

    def _refresh_face_people(self) -> None:
        service = self._active_face_service()
        folder = self.face_folder_path.text().strip()
        scope_paths = self._current_scope_paths()
        if not folder and self.search_only_current_folder.isChecked():
            folder = self._current_directory()
        try:
            profiles = service.load_person_profiles(
                folder_prefix=folder,
                candidate_paths=scope_paths,
                limit=int(self.face_browser_limit.value()),
            )
        except Exception as exc:
            errorBox("Load failed", str(exc))
            return
        self.face_people_list.clear()
        for profile in profiles:
            if (folder or scope_paths) and int(getattr(profile, "visible_face_count", 0)) <= 0:
                continue
            tags_text = ", ".join(profile.tags[:3]) if profile.tags else "no tags"
            notes_text = str(profile.notes or "").strip()
            summary = f"{profile.person_name} | examples={profile.example_count} | labeled={profile.labeled_count} | visible={profile.visible_face_count} | thr={profile.similarity_threshold:.2f}"
            if notes_text:
                summary += f" | {notes_text[:36]}"
            summary += f" | tags={tags_text}"
            item = QListWidgetItem(summary)
            item.setData(Qt.ItemDataRole.UserRole, profile)
            self.face_people_list.addItem(item)

    def _refresh_scanned_faces(self) -> None:
        service = self._active_face_service()
        folder = self.face_folder_path.text().strip()
        scope_paths = self._current_scope_paths()
        if not folder and self.search_only_current_folder.isChecked():
            folder = self._current_directory()
        try:
            records = service.load_indexed_faces(
                folder_prefix=folder,
                limit=int(self.face_browser_limit.value()),
                candidate_paths=scope_paths,
            )
        except Exception as exc:
            errorBox("Load failed", str(exc))
            return
        self.face_scanned_list.clear()
        for record in records:
            label = record.person_name or "Unlabeled"
            item = QListWidgetItem(self._make_face_thumbnail_icon(record), f"{label}\n{Path(record.image_path).name} #{int(record.face_index) + 1}")
            item.setData(Qt.ItemDataRole.UserRole, record)
            item.setTextAlignment(int(Qt.AlignmentFlag.AlignHCenter))
            item.setToolTip(
                f"{record.image_path}\nface #{int(record.face_index) + 1}\n"
                f"bbox={tuple(record.face_bbox)}\n"
                f"face_conf={float(record.face_confidence):.3f}"
            )
            self.face_scanned_list.addItem(item)
        image_count = len({record.image_path for record in records})
        scope_text = f" in {folder}" if folder else ""
        self.face_scanned_summary.setText(f"Loaded {len(records)} scanned faces from {image_count} image(s){scope_text}.")
        self._show_indexed_face_images(records)

    def _show_indexed_face_images(self, records: list[IndexedFaceRecord]) -> None:
        grouped: dict[str, list[IndexedFaceRecord]] = {}
        for record in records:
            grouped.setdefault(record.image_path, []).append(record)
        self._set_results_kind("faces")
        self.results_list.clear()
        if not grouped:
            self.results_gallery.update_gallery([])
            try:
                self.results_gallery.model.set_overlays_by_path({})
            except Exception:
                pass
            self.results_gallery.inspector_context_provider = lambda _p: {}
            self._publish_results([], lambda _p: {}, {}, {}, self._results_kind)
            return

        paths = list(grouped.keys())
        overlay_by_path: dict[str, str] = {}
        subtitle_by_path: dict[str, str] = {}
        ctx_by_path: dict[str, dict[str, object]] = {}
        for image_path, image_records in grouped.items():
            labels = [record.person_name for record in image_records if record.person_name]
            unique_labels = list(dict.fromkeys(labels))
            overlay_by_path[image_path] = f"{len(image_records)} face{'s' if len(image_records) != 1 else ''}"
            subtitle_by_path[image_path] = ", ".join(unique_labels[:2]) if unique_labels else "unlabeled"
            ctx_by_path[image_path] = {
                "indexed_faces": {
                    "count": len(image_records),
                    "faces": [
                        {
                            "face_index": int(record.face_index),
                            "bbox": tuple(record.face_bbox),
                            "confidence": float(record.face_confidence),
                            "person_name": str(record.person_name or ""),
                        }
                        for record in image_records
                    ],
                }
            }
            self.results_list.addItem(
                f"{Path(image_path).name} | faces={len(image_records)} | names={subtitle_by_path[image_path]} | {image_path}"
            )
        self.results_gallery.update_gallery(paths)
        try:
            self.results_gallery.model.set_overlays_by_path(overlay_by_path, subtitle_by_path)
        except Exception:
            pass
        self.results_gallery.inspector_context_provider = lambda p: ctx_by_path.get(p, {})
        self._publish_results(paths, self.results_gallery.inspector_context_provider, overlay_by_path, subtitle_by_path, self._results_kind)

    def _selected_scanned_faces(self) -> list[IndexedFaceRecord]:
        records: list[IndexedFaceRecord] = []
        for item in self.face_scanned_list.selectedItems():
            record = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(record, IndexedFaceRecord):
                records.append(record)
        return records

    def _make_face_thumbnail_icon(self, record: IndexedFaceRecord) -> QIcon:
        key = (record.image_path, int(record.face_index), tuple(record.face_bbox))
        cached = self._face_thumb_cache.get(key)
        if cached is not None:
            return cached
        try:
            with Image.open(record.image_path) as image:
                rgb = image.convert("RGB")
                x1, y1, x2, y2 = [int(v) for v in record.face_bbox]
                pad_x = max(8, (x2 - x1) // 6)
                pad_y = max(8, (y2 - y1) // 6)
                crop = rgb.crop(
                    (
                        max(0, x1 - pad_x),
                        max(0, y1 - pad_y),
                        min(rgb.width, x2 + pad_x),
                        min(rgb.height, y2 + pad_y),
                    )
                )
                crop.thumbnail((112, 112), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (112, 112), (18, 18, 18))
                offset = ((112 - crop.width) // 2, (112 - crop.height) // 2)
                canvas.paste(crop, offset)
                icon = QIcon(QPixmap.fromImage(ImageQt(canvas)))
        except Exception:
            icon = QIcon()
        self._face_thumb_cache[key] = icon
        return icon

    def _on_scanned_face_selection_changed(self) -> None:
        records = self._selected_scanned_faces()
        if not records:
            return
        first = records[0]
        if first.person_name:
            self.face_label_name.setText(first.person_name)
            self.face_name_query.setText(first.person_name)
            if hasattr(self, "person_name"):
                try:
                    self.person_name.setText(first.person_name)
                except Exception:
                    pass
        if first.person_name:
            self._populate_profile_fields(first.person_name)
        self.face_scanned_summary.setText(
            f"Selected {len(records)} face(s). First: {Path(first.image_path).name} #{int(first.face_index) + 1} | "
            f"bbox={tuple(first.face_bbox)} | conf={float(first.face_confidence):.3f}"
        )

    def _save_selected_face_name(self) -> None:
        records = self._selected_scanned_faces()
        if not records:
            errorBox("No face selected", "Select one or more scanned faces first.")
            return
        person_name = self.face_label_name.text().strip()
        if not person_name:
            errorBox("Missing name", "Enter the person name for the selected face(s).")
            return
        refs = [(record.image_path, int(record.face_index)) for record in records]
        threshold = float(self.face_library_label_threshold.value())

        def _run(progress, cancel_check):
            progress(-1, "Saving selected face label...")
            _ = cancel_check
            return self._active_face_service().label_indexed_faces(
                person_name,
                refs,
                similarity_threshold=threshold,
            )

        def _done(person) -> None:
            self.face_name_query.setText(person.person_name)
            self.face_label_name.setText(person.person_name)
            if hasattr(self, "person_name"):
                try:
                    self.person_name.setText(person.person_name)
                except Exception:
                    pass
            tags = [tag.strip() for tag in self.face_profile_tags.text().split(",") if tag.strip()]
            cover_ref = (records[0].image_path, int(records[0].face_index)) if records else None
            self._active_face_service().save_person_profile(
                person.person_name,
                notes=self.face_profile_notes.text().strip(),
                tags=tags,
                cover_face_ref=cover_ref,
            )
            self.status_label.setText(
                f"Saved '{person.person_name}' from {person.example_count} selected face(s)."
            )
            self._refresh_face_people()
            self._refresh_scanned_faces()

        self._start_job("Saving selected face label", _run, _done)

    def _search_selected_face(self) -> None:
        records = self._selected_scanned_faces()
        if len(records) != 1:
            errorBox("Select one face", "Select exactly one scanned face to search across photos.")
            return
        record = records[0]
        folder = self.face_name_folder_filter.text().strip()
        scope_paths = self._current_scope_paths()
        if not folder and self.search_only_current_folder.isChecked():
            folder = self._current_directory()

        def _run(progress, cancel_check):
            progress(-1, "Searching selected face...")
            _ = cancel_check
            return self._active_face_service().search_similar_face(
                record.image_path,
                int(record.face_index),
                top_k=int(self.face_browser_top_k.value()),
                min_score=float(self.face_browser_min_score.value()),
                folder_prefix=folder,
                candidate_paths=scope_paths,
            )

        def _done(results) -> None:
            self._set_results_kind("faces")
            self.results_list.clear()
            preview_paths = []
            overlay_by_path: dict[str, str] = {}
            ctx_by_path: dict[str, dict[str, object]] = {}
            label = record.person_name or self.face_label_name.text().strip() or "selected face"
            for result in results:
                preview_paths.append(result.image_path)
                overlay_by_path.setdefault(result.image_path, f"{label} {float(result.score):.3f} | conf:{float(result.face_confidence):.2f}")
                ctx_by_path.setdefault(
                    result.image_path,
                    {
                        "face_search": {
                            "score": float(result.score),
                            "bbox": tuple(result.face_bbox),
                            "confidence": float(result.face_confidence),
                            "query": {
                                "image_path": record.image_path,
                                "face_index": int(record.face_index),
                                "bbox": tuple(record.face_bbox),
                            },
                        }
                    },
                )
                self.results_list.addItem(
                    f"{Path(result.image_path).name} | score={result.score:.4f} | bbox={result.face_bbox} | conf={result.face_confidence:.3f} | {result.image_path}"
                )
            self.status_label.setText(f"Found {len(results)} matches for the selected face.")
            unique_paths = list(dict.fromkeys(preview_paths))
            self.results_gallery.update_gallery(unique_paths)
            try:
                self.results_gallery.model.set_overlays_by_path(overlay_by_path)
            except Exception:
                pass
            self.results_gallery.inspector_context_provider = lambda p: ctx_by_path.get(p, {})
            self._publish_results(unique_paths, self.results_gallery.inspector_context_provider, overlay_by_path, {}, self._results_kind)

        self._start_job("Searching selected face", _run, _done)

    def _save_person_profile(self) -> None:
        person_name = self.face_label_name.text().strip() or self.face_name_query.text().strip()
        if not person_name:
            errorBox("Missing name", "Enter or select a person name first.")
            return
        records = self._selected_scanned_faces()
        cover_ref = (records[0].image_path, int(records[0].face_index)) if records else None
        tags = [tag.strip() for tag in self.face_profile_tags.text().split(",") if tag.strip()]

        def _run(progress, cancel_check):
            progress(-1, "Saving profile...")
            _ = cancel_check
            self._active_face_service().save_person_profile(
                person_name,
                notes=self.face_profile_notes.text().strip(),
                tags=tags,
                cover_face_ref=cover_ref,
            )
            return True

        def _done(_ok: bool) -> None:
            self.status_label.setText(f"Saved profile for {person_name}.")
            self._refresh_face_people()

        self._start_job("Saving person profile", _run, _done)

    def _populate_profile_fields(self, person_name: str) -> None:
        service = self._active_face_service()
        try:
            profiles = service.load_person_profiles(limit=max(1, int(self.face_browser_limit.value() or 1)))
        except Exception:
            return
        for profile in profiles:
            if str(profile.person_name) != str(person_name):
                continue
            self.face_profile_notes.setText(str(profile.notes or ""))
            self.face_profile_tags.setText(", ".join(profile.tags))
            break

    def _on_person_clicked(self, item) -> None:
        profile = item.data(Qt.ItemDataRole.UserRole)
        if profile is not None and hasattr(profile, "person_name"):
            name = str(profile.person_name or "").strip()
        else:
            text = str(item.text() or "")
            name = text.split("|", 1)[0].strip()
        if name:
            self.face_name_query.setText(name)
            self.face_label_name.setText(name)
            if hasattr(self, "person_name"):
                try:
                    self.person_name.setText(name)
                except Exception:
                    pass
            if profile is not None:
                try:
                    self.face_profile_notes.setText(str(getattr(profile, "notes", "") or ""))
                    self.face_profile_tags.setText(", ".join(getattr(profile, "tags", ()) or ()))
                except Exception:
                    pass

    def _search_by_name(self) -> None:
        service = self._active_face_service()
        name = self.face_name_query.text().strip()
        folder = self.face_name_folder_filter.text().strip()
        scope_paths = self._current_scope_paths()
        if not folder and self.search_only_current_folder.isChecked():
            folder = self._current_directory()
        min_score = float(self.face_name_min_score.value())
        min_score_arg = None if min_score <= 0.0 else min_score

        def _run(progress, cancel_check):
            progress(-1, "Searching by name...")
            _ = cancel_check
            return service.search_by_person_name(
                name,
                top_k=int(self.face_name_top_k.value()),
                min_score=min_score_arg,
                folder_prefix=folder,
                candidate_paths=scope_paths,
            )

        def _done(results) -> None:
            self._set_results_kind("faces")
            self.results_list.clear()
            preview_paths = []
            overlay_by_path: dict[str, str] = {}
            ctx_by_path: dict[str, dict[str, object]] = {}
            for result in results:
                preview_paths.append(result.image_path)
                overlay_by_path.setdefault(result.image_path, f"{name} {float(result.score):.3f} | conf:{float(result.face_confidence):.2f}")
                ctx_by_path.setdefault(
                    result.image_path,
                    {"face_name": {"person_name": str(name), "score": float(result.score), "bbox": tuple(result.face_bbox), "confidence": float(result.face_confidence)}},
                )
                self.results_list.addItem(
                    f"{Path(result.image_path).name} | score={result.score:.4f} | bbox={result.face_bbox} | conf={result.face_confidence:.3f} | {result.image_path}"
                )
            self.status_label.setText(f"Found {len(results)} matches for '{name}'.")
            unique_paths = list(dict.fromkeys(preview_paths))
            self.results_gallery.update_gallery(unique_paths)
            try:
                self.results_gallery.model.set_overlays_by_path(overlay_by_path)
            except Exception:
                pass
            self.results_gallery.inspector_context_provider = lambda p: ctx_by_path.get(p, {})
            self._publish_results(unique_paths, self.results_gallery.inspector_context_provider, overlay_by_path, {}, self._results_kind)

        self._start_job("Searching faces by name", _run, _done)

    def _build_face_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)
        self.face_query_path = PasteAwareLineEdit(lambda mime: self._handle_paste_mime(self.face_query_path, mime, multi_append=False), self)
        self.face_examples = QLineEdit()
        self.person_name = QLineEdit()
        self.face_top_k = QSpinBox()
        self.face_top_k.setRange(1, 500)
        self.face_top_k.setValue(30)
        self.face_min_score = QDoubleSpinBox()
        self.face_min_score.setRange(-1.0, 1.0)
        self.face_min_score.setSingleStep(0.05)
        self.face_min_score.setValue(0.35)
        self.face_label_threshold = QDoubleSpinBox()
        self.face_label_threshold.setRange(0.0, 1.0)
        self.face_label_threshold.setSingleStep(0.01)
        self.face_label_threshold.setValue(0.72)
        self.face_cluster_count = QSpinBox()
        self.face_cluster_count.setRange(2, 200)
        self.face_cluster_count.setValue(12)
        self.merge_source_person = QLineEdit()
        self.merge_target_person = QLineEdit()
        form.addRow("Query Face Image", self._path_row(self.face_query_path))
        form.addRow("Few-shot Examples", self._path_row(self.face_examples, multi_append=True, enable_paste=False))
        form.addRow("Person Name", self.person_name)
        form.addRow("Top-K", self.face_top_k)
        form.addRow("Min Face Score", self.face_min_score)
        form.addRow("Label Threshold", self.face_label_threshold)
        form.addRow("Face Clusters", self.face_cluster_count)
        form.addRow("Merge Source", self.merge_source_person)
        form.addRow("Merge Target", self.merge_target_person)
        buttons = QHBoxLayout()
        index_button = QPushButton("Index Faces")
        search_button = QPushButton("Find Same Person")
        cluster_button = QPushButton("Cluster Faces")
        label_button = QPushButton("Save Person Label")
        propagate_button = QPushButton("Auto-Propagate")
        list_button = QPushButton("List Labels")
        merge_button = QPushButton("Merge Names")
        clear_button = QPushButton("Clear Person")
        index_button.clicked.connect(self._index_faces)
        search_button.clicked.connect(self._search_faces)
        cluster_button.clicked.connect(self._cluster_faces)
        label_button.clicked.connect(self._label_person)
        propagate_button.clicked.connect(self._auto_propagate_labels)
        list_button.clicked.connect(self._list_face_labels)
        merge_button.clicked.connect(self._merge_people)
        clear_button.clicked.connect(self._clear_person)
        for button in [index_button, search_button, cluster_button, label_button, propagate_button, list_button, merge_button, clear_button]:
            buttons.addWidget(button)
        self._action_buttons.extend([index_button, search_button, cluster_button, label_button, propagate_button, list_button, merge_button, clear_button])
        form.addRow(buttons)
        self.tabs.addTab(tab, "Face Search")

    def _folder_row(self, line_edit: QLineEdit) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(lambda: self._browse_folder(line_edit))
        layout.addWidget(line_edit)
        layout.addWidget(browse_button)
        return row

    def _path_row(self, line_edit: QLineEdit, multi_append: bool = False, enable_paste: bool = True) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(lambda: self._browse_image(line_edit, multi_append=multi_append))
        paste_button = QPushButton("Paste")
        paste_button.clicked.connect(lambda: self._paste_query_image(line_edit, multi_append=multi_append))
        layout.addWidget(line_edit)
        layout.addWidget(browse_button)
        if enable_paste:
            layout.addWidget(paste_button)
        return row

    def _browse_folder(self, line_edit: QLineEdit) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Folder")
        if directory:
            line_edit.setText(str(directory))

    def _browse_image(self, line_edit: QLineEdit, multi_append: bool = False) -> None:
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Query Images")
        if not file_paths:
            return
        if multi_append:
            current = [path.strip() for path in line_edit.text().split(";") if path.strip()]
            current.extend(file_paths)
            line_edit.setText(";".join(dict.fromkeys(current)))
        else:
            line_edit.setText(file_paths[0])

    def _handle_paste_mime(self, line_edit: QLineEdit, mime, multi_append: bool) -> bool:
        """
        Returns True if we handled paste (so the line edit should not paste plain text).
        """
        if mime is None:
            return False
        # Prefer real clipboard image data if present, else fall back to urls/text.
        if self._try_paste_image(line_edit, mime, multi_append=multi_append):
            return True
        if self._try_paste_urls(line_edit, mime, multi_append=multi_append):
            return True
        return False

    def _paste_query_image(self, line_edit: QLineEdit, multi_append: bool = False) -> None:
        clipboard = QGuiApplication.clipboard()
        mime = clipboard.mimeData()
        if mime is None:
            errorBox("Paste failed", "Clipboard is empty.")
            return
        if self._try_paste_image(line_edit, mime, multi_append=multi_append, clipboard=clipboard):
            return
        if self._try_paste_urls(line_edit, mime, multi_append=multi_append):
            return
        if mime.hasText():
            text = (mime.text() or "").strip().strip('"')
            if not text:
                errorBox("Paste failed", "Clipboard text is empty.")
                return
            if multi_append:
                current = [path.strip() for path in line_edit.text().split(";") if path.strip()]
                current.append(text)
                line_edit.setText(";".join(dict.fromkeys(current)))
            else:
                line_edit.setText(text)
            return
        errorBox("Paste failed", "Clipboard does not contain an image or a file path.")

    def _try_paste_urls(self, line_edit: QLineEdit, mime, multi_append: bool) -> bool:
        if not mime.hasUrls():
            return False
        paths = [url.toLocalFile() for url in mime.urls() if url.isLocalFile()]
        paths = [path for path in paths if path]
        if not paths:
            return False
        if multi_append:
            current = [path.strip() for path in line_edit.text().split(";") if path.strip()]
            current.extend(paths)
            line_edit.setText(";".join(dict.fromkeys(current)))
        else:
            line_edit.setText(paths[0])
        return True

    def _try_paste_image(self, line_edit: QLineEdit, mime, multi_append: bool, clipboard=None) -> bool:
        clipboard = clipboard or QGuiApplication.clipboard()
        # On Windows, mime.hasImage() can be false for bitmaps copied from some apps.
        qimage = clipboard.image()
        if qimage.isNull():
            pixmap = clipboard.pixmap()
            if not pixmap.isNull():
                qimage = pixmap.toImage()
        if qimage.isNull() and getattr(mime, "formats", None):
            for fmt in mime.formats():
                if not fmt.lower().startswith("image/"):
                    continue
                data = mime.data(fmt)
                if data:
                    from PyQt6.QtGui import QImage

                    candidate = QImage.fromData(data)
                    if not candidate.isNull():
                        qimage = candidate
                        break
        if qimage.isNull():
            return False
        out_dir = self.settings.cache_dir / "pasted_queries"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"pasted_{uuid4().hex}.png"
        if not qimage.save(str(out_path), "PNG"):
            return False
        # Track temp file so we can clean up if user pastes repeatedly.
        key = id(line_edit)
        prev = self._temp_query_files.get(key)
        if prev and prev != str(out_path):
            try:
                Path(prev).unlink(missing_ok=True)
            except Exception:
                pass
        self._temp_query_files[key] = str(out_path)
        if multi_append:
            current = [path.strip() for path in line_edit.text().split(";") if path.strip()]
            current.append(str(out_path))
            line_edit.setText(";".join(dict.fromkeys(current)))
        else:
            line_edit.setText(str(out_path))
        return True

    def context_for_path(self, image_path: str) -> dict[str, object]:
        provider = getattr(self.results_gallery, "inspector_context_provider", None)
        if callable(provider):
            try:
                return dict(provider(image_path) or {})
            except Exception:
                return {}
        return {}

    def tab_labels(self) -> list[str]:
        return [self.tabs.tabText(index) for index in range(self.tabs.count())]

    def active_tab_index(self) -> int:
        return int(self.tabs.currentIndex())

    def set_active_tab_index(self, index: int) -> None:
        if 0 <= int(index) < self.tabs.count() and self.tabs.currentIndex() != int(index):
            self.tabs.setCurrentIndex(int(index))

    def set_tab_bar_visible(self, visible: bool) -> None:
        tab_bar = self.tabs.tabBar()
        if tab_bar is not None:
            tab_bar.setVisible(bool(visible))

    def export_state(self) -> dict[str, object]:
        def _combo_id(combo: QComboBox | None) -> str:
            if combo is None:
                return ""
            try:
                data = combo.currentData()
                if data:
                    return str(data)
            except Exception:
                pass
            try:
                return str(combo.currentText())
            except Exception:
                return ""

        return {
            "active_tab": int(self.tabs.currentIndex()),
            "search_only_current_folder": bool(self.search_only_current_folder.isChecked()),
            "image_model": _combo_id(getattr(self, "image_model", None)),
            "text_model": _combo_id(getattr(self, "text_model", None)),
            "duplicate_model": _combo_id(getattr(self, "duplicate_model", None)),
        }

    def apply_state(self, state: dict[str, object] | None) -> None:
        if not state:
            return
        try:
            self.tabs.setCurrentIndex(int(state.get("active_tab", 0)))
        except Exception:
            pass
        self.search_only_current_folder.setChecked(bool(state.get("search_only_current_folder", True)))
        self._set_combo_to_id(getattr(self, "image_model", None), state.get("image_model"))
        self._set_combo_to_id(getattr(self, "text_model", None), state.get("text_model"))
        self._set_combo_to_id(getattr(self, "duplicate_model", None), state.get("duplicate_model"))

    @staticmethod
    def _set_combo_to_id(combo: QComboBox | None, value: object) -> None:
        if combo is None or value is None:
            return
        target = str(value)
        if not target:
            return
        # Prefer matching userData (model id), fall back to label text.
        try:
            for i in range(combo.count()):
                data = combo.itemData(i)
                if data and str(data) == target:
                    combo.setCurrentIndex(i)
                    return
        except Exception:
            pass
        try:
            combo.setCurrentText(target)
        except Exception:
            pass

    def _current_directory(self) -> str:
        if self.current_directory_provider is None:
            return ""
        return self.current_directory_provider()

    def _index_images_for(self, embedding_model: str, directory: str, scope_combo) -> None:
        directory = (str(directory or "").strip() or self._current_directory())
        scope_paths = self._current_scope_paths()
        if not directory and not scope_paths:
            errorBox("No folder selected", "Choose a folder in the file pane first.")
            return

        def _run(progress, cancel_check):
            progress(-1, "Indexing...")
            service = self._active_search_service(scope_combo)
            if scope_paths:
                return service.index_paths(
                    scope_paths,
                    embedding_model=embedding_model,
                    directory_hint=directory,
                    use_onnx=self._use_onnx(),
                    progress_callback=progress,
                    cancel_check=cancel_check,
                )
            return service.index_directory(
                directory,
                embedding_model=embedding_model,
                recursive=True,
                use_onnx=self._use_onnx(),
                progress_callback=progress,
                cancel_check=cancel_check,
            )

        def _done(metrics: dict) -> None:
            source_label = "working set" if scope_paths else directory
            self.status_label.setText(f"Indexed {metrics.get('indexed_images', 0)} images with {embedding_model} for {source_label}.")
            infoBox("Indexing complete", self.status_label.text())

        self._start_job("Indexing images", _run, _done)

    def _search_by_image(self) -> None:
        folder = self.image_folder_filter.text().strip()
        scope_paths = self._current_scope_paths()
        if not folder and self.search_only_current_folder.isChecked():
            # Prefer the tab's index folder when set, else fall back to current folder.
            folder = (getattr(self, "image_index_folder", None).text().strip() if hasattr(self, "image_index_folder") else "") or self._current_directory()
        filters = {"folder": folder} if folder else {}
        self._run_similarity_search(
            SimilaritySearchRequest(
                query_image_path=self.image_query_path.text().strip(),
                search_mode="image",
                top_k=self.image_top_k.value(),
                min_score=self.image_min_score.value(),
                embedding_model=self._model_id(self.image_model),
                use_onnx=self._use_onnx(),
                filters=filters,
                candidate_paths=scope_paths,
            ),
            scope_combo=self.image_db_scope,
        )

    def _search_by_text(self) -> None:
        filters = {}
        scope_paths = self._current_scope_paths()
        if self.search_only_current_folder.isChecked():
            folder = self._current_directory()
            if folder:
                filters = {"folder": folder}
        self._run_similarity_search(
            SimilaritySearchRequest(
                query_text=self.text_query.text().strip(),
                search_mode="text",
                top_k=self.text_top_k.value(),
                min_score=self.text_min_score.value(),
                embedding_model=self._model_id(self.text_model),
                use_onnx=self._use_onnx(),
                filters=filters,
                candidate_paths=scope_paths,
            )
        )

    def _search_duplicates(self) -> None:
        filters = {}
        scope_paths = self._current_scope_paths()
        if self.search_only_current_folder.isChecked():
            folder = (getattr(self, "duplicate_index_folder", None).text().strip() if hasattr(self, "duplicate_index_folder") else "") or self._current_directory()
            if folder:
                filters = {"folder": folder}
        self._run_similarity_search(
            SimilaritySearchRequest(
                query_image_path=self.duplicate_query_path.text().strip(),
                search_mode="duplicate",
                top_k=self.duplicate_top_k.value(),
                min_score=self.duplicate_min_score.value(),
                max_phash_distance=self.hash_distance.value(),
                embedding_model=self._model_id(self.duplicate_model),
                hash_backend=self.hash_backend.currentText(),
                hash_distance=self.hash_distance.value(),
                orb_rerank=self.orb_rerank_checkbox.isChecked(),
                ann_backend=self.ann_backend.currentText(),
                nlist=self.ivf_nlist.value(),
                nprobe=self.ivf_nprobe.value(),
                pq_m=self.ivf_pq_m.value(),
                use_onnx=self._use_onnx(),
                filters=filters,
                candidate_paths=scope_paths,
            ),
            scope_combo=self.duplicate_db_scope,
        )

    def _run_similarity_search(self, request: SimilaritySearchRequest, *, scope_combo=None) -> None:

        def _run(progress, cancel_check):
            try:
                progress(-1, "Searching...")
                service = self._active_search_service(scope_combo)
                return service.search(request, cancel_check=cancel_check)
            except Exception as exc:
                message = str(exc)
                if "incompatible dimensions" not in message:
                    raise
                directory = str((request.filters or {}).get("folder") or "").strip() or self._current_directory()
                if request.candidate_paths:
                    service = self._active_search_service(scope_combo)
                    service.index_paths(
                        request.candidate_paths,
                        embedding_model=request.embedding_model,
                        directory_hint=directory,
                        use_onnx=request.use_onnx,
                        progress_callback=progress,
                        cancel_check=cancel_check,
                    )
                    progress(-1, "Retrying search...")
                    service = self._active_search_service(scope_combo)
                    return service.search(request, cancel_check=cancel_check)
                if not directory:
                    raise
                key = (request.embedding_model, directory)
                if key in self._auto_reindexed:
                    raise
                self._auto_reindexed.add(key)
                # Heal old/corrupted indexes by re-indexing once, then retry the search.
                service = self._active_search_service(scope_combo)
                service.index_directory(
                    directory,
                    embedding_model=request.embedding_model,
                    recursive=True,
                    use_onnx=request.use_onnx,
                    progress_callback=progress,
                    cancel_check=cancel_check,
                )
                progress(-1, "Retrying search...")
                service = self._active_search_service(scope_combo)
                return service.search(request, cancel_check=cancel_check)

        def _done(results) -> None:
            self._set_results_kind("similarity")
            self._last_results = list(results or [])
            self.results_list.clear()
            preview_paths = []
            for result in results:
                preview_paths.append(result.image_path)
                hash_text = "-" if result.hash_distance < 0 else f"{result.hash_backend}:{result.hash_distance}"
                self.results_list.addItem(
                    f"{Path(result.image_path).name} | score={result.score:.4f} | hash={hash_text} | orb={result.orb_score:.3f} | {result.match_reason} | {result.image_path}"
                )
            self.status_label.setText(f"Found {len(results)} results.")
            self._apply_results_view()

        self._start_job("Searching images", _run, _done)

    def _index_faces(self) -> None:
        directory = self._current_directory()
        scope_paths = self._current_scope_paths()
        if not directory and not scope_paths:
            errorBox("No folder selected", "Choose a folder in the file pane first.")
            return

        def _run(progress, cancel_check):
            progress(0, "0/0 images, faces=0")
            if scope_paths:
                return self._active_face_service().index_paths(scope_paths, progress_callback=progress, cancel_check=cancel_check)
            return self._active_face_service().index_directory(directory, recursive=True, progress_callback=progress, cancel_check=cancel_check)

        def _done(metrics: dict) -> None:
            faces = int((metrics or {}).get("faces_indexed", 0))
            done = int((metrics or {}).get("images_done", 0))
            total = int((metrics or {}).get("images_total", 0))
            source_label = "working set" if scope_paths else directory
            self.status_label.setText(f"Indexed {faces} faces from {done}/{total} images in {source_label}.")
            infoBox("Face indexing complete", self.status_label.text())

        self._start_job("Indexing faces", _run, _done)

    def _search_faces(self) -> None:
        scope_paths = self._current_scope_paths()

        request = FaceSearchRequest(
            query_face_image=self.face_query_path.text().strip(),
            top_k=self.face_top_k.value(),
            min_face_score=self.face_min_score.value(),
            candidate_paths=scope_paths,
        )

        def _run(progress, cancel_check):
            progress(-1, "Searching faces...")
            _ = cancel_check
            return self._active_face_service().search_faces(request)

        def _done(results) -> None:
            self._set_results_kind("faces")
            self.results_list.clear()
            preview_paths = []
            overlay_by_path: dict[str, str] = {}
            ctx_by_path: dict[str, dict[str, object]] = {}
            for result in results:
                preview_paths.append(result.image_path)
                overlay_by_path.setdefault(result.image_path, f"face {float(result.score):.3f} | conf:{float(result.face_confidence):.2f}")
                ctx_by_path.setdefault(
                    result.image_path,
                    {
                        "face_search": {
                            "score": float(result.score),
                            "bbox": tuple(result.face_bbox),
                            "confidence": float(result.face_confidence),
                        }
                    },
                )
                self.results_list.addItem(
                    f"{Path(result.image_path).name} | score={result.score:.4f} | bbox={result.face_bbox} | conf={result.face_confidence:.3f} | {result.image_path}"
                )
            self.status_label.setText(f"Found {len(results)} face matches.")
            unique_paths = list(dict.fromkeys(preview_paths))
            self.results_gallery.update_gallery(unique_paths)
            try:
                self.results_gallery.model.set_overlays_by_path(overlay_by_path)
            except Exception:
                pass
            self.results_gallery.inspector_context_provider = lambda p: ctx_by_path.get(p, {})
            self._publish_results(unique_paths, self.results_gallery.inspector_context_provider, overlay_by_path, {}, self._results_kind)

        self._start_job("Searching faces", _run, _done)

    def _cluster_faces(self) -> None:

        num_clusters = self.face_cluster_count.value()
        min_score = self.face_min_score.value()
        scope_paths = self._current_scope_paths()
        folder = self._current_directory() if self.search_only_current_folder.isChecked() else ""

        def _run(progress, cancel_check):
            progress(-1, "Clustering faces...")
            _ = cancel_check
            return self._active_face_service().cluster_faces(
                num_clusters=num_clusters,
                min_face_score=min_score,
                folder_prefix=folder,
                candidate_paths=scope_paths,
            )

        def _done(clusters) -> None:
            self._set_results_kind("face_clusters")
            self.results_list.clear()
            preview_paths = []
            overlay_by_path: dict[str, str] = {}
            ctx_by_path: dict[str, dict[str, object]] = {}
            for cluster_id, image_paths in sorted(clusters.items()):
                unique_paths = sorted(set(image_paths))
                preview_paths.extend(unique_paths)
                for path in unique_paths:
                    overlay_by_path.setdefault(path, f"face cluster {cluster_id}")
                    ctx_by_path.setdefault(path, {"face_cluster": {"cluster_id": int(cluster_id)}})
                self.results_list.addItem(f"Cluster {cluster_id}: {len(unique_paths)} images | {', '.join(Path(path).name for path in unique_paths[:5])}")
            self.status_label.setText(f"Generated {len(clusters)} face clusters.")
            unique_paths = list(dict.fromkeys(preview_paths))
            self.results_gallery.update_gallery(unique_paths)
            try:
                self.results_gallery.model.set_overlays_by_path(overlay_by_path)
            except Exception:
                pass
            self.results_gallery.inspector_context_provider = lambda p: ctx_by_path.get(p, {})
            self._publish_results(unique_paths, self.results_gallery.inspector_context_provider, overlay_by_path, {}, self._results_kind)
            self.face_clusters_ready.emit(clusters)

        self._start_job("Clustering faces", _run, _done)

    def _label_person(self) -> None:
        examples = [path.strip() for path in self.face_examples.text().split(";") if path.strip()]
        if self.face_query_path.text().strip():
            examples.append(self.face_query_path.text().strip())
        examples = list(dict.fromkeys(examples))

        request = FaceLabelRequest(
            person_name=self.person_name.text().strip(),
            example_image_paths=examples,
            similarity_threshold=self.face_label_threshold.value(),
        )

        def _run(progress, cancel_check):
            progress(-1, "Saving prototype...")
            _ = cancel_check
            return self._active_face_service().label_face_examples(request)

        def _done(person) -> None:
            tags = [tag.strip() for tag in self.face_profile_tags.text().split(",") if tag.strip()]
            self._active_face_service().save_person_profile(
                person.person_name,
                notes=self.face_profile_notes.text().strip(),
                tags=tags,
            )
            self.status_label.setText(f"Saved prototype for {person.person_name} from {person.example_count} examples.")
            infoBox("Face label saved", self.status_label.text())
            self._refresh_face_people()

        self._start_job("Saving face label", _run, _done)

    def _auto_propagate_labels(self) -> None:

        def _run(progress, cancel_check):
            progress(-1, "Auto-propagating...")
            _ = cancel_check
            return self._active_face_service().auto_propagate_labels()

        def _done(assignments) -> None:
            self._set_results_kind("labels")
            self.results_list.clear()
            preview_paths = []
            overlay_by_path: dict[str, str] = {}
            ctx_by_path: dict[str, dict[str, object]] = {}
            for assignment in assignments:
                preview_paths.append(assignment.image_path)
                overlay_by_path.setdefault(assignment.image_path, str(assignment.person_name))
                ctx_by_path.setdefault(
                    assignment.image_path,
                    {
                        "face_label": {
                            "person_name": str(assignment.person_name),
                            "confidence": float(assignment.confidence),
                            "bbox": tuple(assignment.face_bbox),
                        }
                    },
                )
                self.results_list.addItem(
                    f"{assignment.person_name} | {Path(assignment.image_path).name} | face={assignment.face_index} | bbox={assignment.face_bbox} | conf={assignment.confidence:.4f}"
                )
            self.status_label.setText(f"Assigned {len(assignments)} face labels.")
            unique_paths = list(dict.fromkeys(preview_paths))
            self.results_gallery.update_gallery(unique_paths)
            try:
                self.results_gallery.model.set_overlays_by_path(overlay_by_path)
            except Exception:
                pass
            self.results_gallery.inspector_context_provider = lambda p: ctx_by_path.get(p, {})
            self._publish_results(unique_paths, self.results_gallery.inspector_context_provider, overlay_by_path, {}, self._results_kind)

        self._start_job("Auto-propagating labels", _run, _done)

    def _list_face_labels(self) -> None:

        def _run(progress, cancel_check):
            progress(-1, "Loading labels...")
            _ = cancel_check
            return self._active_face_service().list_face_labels()

        def _done(labels) -> None:
            self._set_results_kind("labels")
            self.results_list.clear()
            preview_paths = []
            overlay_by_path: dict[str, str] = {}
            ctx_by_path: dict[str, dict[str, object]] = {}
            for assignment in labels:
                preview_paths.append(assignment.image_path)
                overlay_by_path.setdefault(assignment.image_path, str(assignment.person_name))
                ctx_by_path.setdefault(
                    assignment.image_path,
                    {
                        "face_label": {
                            "person_name": str(assignment.person_name),
                            "confidence": float(assignment.confidence),
                            "bbox": tuple(assignment.face_bbox),
                        }
                    },
                )
                self.results_list.addItem(
                    f"{assignment.person_name} | {Path(assignment.image_path).name} | face={assignment.face_index} | conf={assignment.confidence:.4f} | {assignment.image_path}"
                )
            self.status_label.setText(f"Loaded {len(labels)} saved labels.")
            unique_paths = list(dict.fromkeys(preview_paths))
            self.results_gallery.update_gallery(unique_paths)
            try:
                self.results_gallery.model.set_overlays_by_path(overlay_by_path)
            except Exception:
                pass
            self.results_gallery.inspector_context_provider = lambda p: ctx_by_path.get(p, {})
            self._publish_results(unique_paths, self.results_gallery.inspector_context_provider, overlay_by_path, {}, self._results_kind)

        self._start_job("Loading labels", _run, _done)

    def _merge_people(self) -> None:

        source = self.merge_source_person.text().strip()
        target = self.merge_target_person.text().strip()

        def _run(progress, cancel_check):
            progress(-1, "Merging labels...")
            _ = cancel_check
            self._active_face_service().merge_person_labels(source, target)
            return True

        def _done(_ok: bool) -> None:
            self._refresh_face_people()
            self._refresh_scanned_faces()
            self._list_face_labels()

        self._start_job("Merging labels", _run, _done)

    def _clear_person(self) -> None:

        person_name = self.person_name.text().strip()

        def _run(progress, cancel_check):
            progress(-1, "Clearing person...")
            _ = cancel_check
            self._active_face_service().clear_person_labels(person_name)
            return True

        def _done(_ok: bool) -> None:
            self._refresh_face_people()
            self._refresh_scanned_faces()
            self._list_face_labels()

        self._start_job("Clearing person", _run, _done)






