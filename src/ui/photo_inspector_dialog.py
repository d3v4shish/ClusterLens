from __future__ import annotations

import html
import json
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, QItemSelectionModel, QSize, Qt, pyqtSlot
from PyQt6.QtGui import QIcon, QImageReader, QKeyEvent, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import QAbstractItemView, QDialog, QGridLayout, QHBoxLayout, QLabel, QListView, QPushButton, QSizePolicy, QSplitter, QTextBrowser, QVBoxLayout, QWidget

from app.services.face_types import EditableFaceInput
from app.services.photo_metadata import PhotoMetadata, PhotoMetadataService
from ui.async_job import AsyncJob, raise_if_cancelled, start_job_in_thread, wait_for_thread_shutdown
from ui.error_mbox import confirmBox, errorBox
from ui.list_models import ListEntry, ListEntryModel
from ui.zoomable_image import ZoomableImageView
from PIL import Image, ImageOps
from PIL.ImageQt import ImageQt

if TYPE_CHECKING:
    from app.services.face_search import FaceIndexService, IndexedFaceRecord


@dataclass(frozen=True)
class EditableFaceDraft:
    bbox: tuple[int, int, int, int]
    confidence: float
    source: str
    person_name: str = ""
    face_index: int = -1


class PhotoInspectorDialog(QDialog):
    def __init__(
        self,
        image_path: str | None = None,
        context: dict[str, object] | None = None,
        *,
        image_paths: list[str] | None = None,
        start_index: int = 0,
        context_provider: Callable[[str], dict[str, object]] | None = None,
        metadata_service: PhotoMetadataService | None = None,
        display_mode: str = "advanced",
        face_service: FaceIndexService | None = None,
        allow_face_edit: bool = False,
        face_draft_provider: Callable[[str], tuple[list[EditableFaceDraft], bool] | None] | None = None,
        face_draft_updated_callback: Callable[[str, list[EditableFaceDraft], bool], None] | None = None,
        face_edit_saved_callback: Callable[[str], None] | None = None,
        face_auto_clean_callback: Callable[[str, list[EditableFaceDraft]], tuple[list[EditableFaceDraft], dict[str, int]]] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.metadata_service = metadata_service or PhotoMetadataService()
        self.face_service = face_service
        self.face_draft_provider = face_draft_provider
        self.face_draft_updated_callback = face_draft_updated_callback
        self.face_edit_saved_callback = face_edit_saved_callback
        self.face_auto_clean_callback = face_auto_clean_callback
        self._allow_face_edit = bool(allow_face_edit and face_service is not None)
        self._active_thread = None
        self._active_job = None
        self._preview_thread = None
        self._preview_job = None
        self._face_edit_thread = None
        self._face_edit_job = None
        self._retained_async_refs: list[tuple[object | None, object | None]] = []
        self._thread_jobs: dict[object, object | None] = {}
        self._request_id = 0
        self._full_res_loaded = False
        self._context_provider = context_provider
        self._base_context: dict[str, object] = dict(context or {})
        self._display_mode = "advanced" if str(display_mode).strip().lower() == "advanced" else "basic"
        self._shortcuts: list[QShortcut] = []
        self._face_drafts_by_path: dict[str, list[EditableFaceDraft]] = {}
        self._face_original_drafts_by_path: dict[str, list[EditableFaceDraft]] = {}
        self._face_editor_dirty_paths: set[str] = set()
        self._face_undo_by_path: dict[str, list[list[EditableFaceDraft]]] = {}
        self._face_redo_by_path: dict[str, list[list[EditableFaceDraft]]] = {}
        self._current_context: dict[str, object] = {}
        self._current_image_path = ""
        self.setWindowTitle("Photo Inspector")
        self.resize(1200, 800)
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(10, 10, 10, 10)

        self._image_paths = list(image_paths or ([] if image_path is None else [image_path]))
        self._index = int(start_index) if self._image_paths else 0
        self._index = max(0, min(self._index, max(0, len(self._image_paths) - 1)))

        self.content_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.content_splitter.setChildrenCollapsible(False)
        self.image_panel = QWidget(self.content_splitter)
        image_layout = QVBoxLayout(self.image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_layout.setSpacing(8)

        self.preview_view = ZoomableImageView(self)
        self.preview_view.setMinimumSize(640, 480)
        self.preview_view.zoom_changed.connect(self._on_zoom_changed)
        image_layout.addWidget(self.preview_view, stretch=1)

        nav = QHBoxLayout()
        nav.setContentsMargins(0, 0, 0, 0)
        self.prev_button = QPushButton("<")
        self.next_button = QPushButton(">")
        self.fit_button = QPushButton("Fit")
        self.index_label = QLabel("")
        self.index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav.addWidget(self.prev_button)
        nav.addWidget(self.next_button)
        nav.addWidget(self.fit_button)
        nav.addWidget(self.index_label, stretch=1)
        image_layout.addLayout(nav)

        self.details_panel = QWidget(self.content_splitter)
        self.details_panel.setMinimumWidth(380)
        self.details_panel.setMaximumWidth(520)
        details_layout = QVBoxLayout(self.details_panel)
        details_layout.setContentsMargins(12, 0, 0, 0)
        details_layout.setSpacing(8)

        self.identity_panel = QWidget(self.details_panel)
        identity_layout = QGridLayout(self.identity_panel)
        identity_layout.setContentsMargins(0, 0, 0, 0)
        identity_layout.setHorizontalSpacing(10)
        identity_layout.setVerticalSpacing(4)
        self.name_title_label = QLabel("Name")
        self.folder_title_label = QLabel("Folder")
        for title_label in (self.name_title_label, self.folder_title_label):
            title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
            title_label.setMinimumWidth(54)

        self.name_label = QLabel("")
        name_font = self.name_label.font()
        name_font.setBold(True)
        name_font.setPointSize(max(name_font.pointSize(), 11))
        self.name_label.setFont(name_font)
        self.name_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.name_label.setWordWrap(True)
        self.name_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.folder_label = QLabel("")
        self.folder_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.folder_label.setWordWrap(True)
        self.folder_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        identity_layout.addWidget(self.name_title_label, 0, 0)
        identity_layout.addWidget(self.name_label, 0, 1)
        identity_layout.addWidget(self.folder_title_label, 1, 0)
        identity_layout.addWidget(self.folder_label, 1, 1)
        identity_layout.setColumnStretch(1, 1)
        details_layout.addWidget(self.identity_panel)

        self.state_label = QLabel("")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.state_label.setWordWrap(True)
        self.state_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        details_layout.addWidget(self.state_label)

        self.face_editor_panel = QWidget(self.details_panel)
        face_editor_layout = QVBoxLayout(self.face_editor_panel)
        face_editor_layout.setContentsMargins(0, 0, 0, 0)
        face_editor_layout.setSpacing(6)

        self.face_editor_summary_label = QLabel("Face editing is only available for Face Library review images.")
        self.face_editor_summary_label.setWordWrap(True)
        face_editor_layout.addWidget(self.face_editor_summary_label)

        face_action_grid = QGridLayout()
        face_action_grid.setContentsMargins(0, 0, 0, 0)
        face_action_grid.setHorizontalSpacing(6)
        face_action_grid.setVerticalSpacing(6)
        self.face_rescan_button = QPushButton("Auto-Scan This Image")
        self.face_draw_button = QPushButton("Draw Face Box")
        self.face_remove_button = QPushButton("Remove Selected Faces")
        self.face_duplicate_button = QPushButton("Duplicate Selected")
        self.face_split_button = QPushButton("Split Selected")
        self.face_undo_button = QPushButton("Undo")
        self.face_redo_button = QPushButton("Redo")
        self.face_remove_all_button = QPushButton("Remove All")
        self.face_auto_clean_button = QPushButton("Auto-Clean This Image")
        self.face_reset_button = QPushButton("Reset")
        self.face_save_button = QPushButton("Save Face Edits")
        face_action_buttons = [
            self.face_rescan_button,
            self.face_draw_button,
            self.face_remove_button,
            self.face_duplicate_button,
            self.face_split_button,
            self.face_undo_button,
            self.face_redo_button,
            self.face_remove_all_button,
            self.face_auto_clean_button,
            self.face_reset_button,
            self.face_save_button,
        ]
        for button_index, button in enumerate(face_action_buttons):
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            face_action_grid.addWidget(button, button_index // 2, button_index % 2)
        face_editor_layout.addLayout(face_action_grid)

        self.face_editor_helper_label = QLabel(
            "Use Auto-Scan for this image only, or draw a box directly on the photo when a face was missed. "
            "Select bad detections below and remove them before saving."
        )
        self.face_editor_helper_label.setWordWrap(True)
        face_editor_layout.addWidget(self.face_editor_helper_label)

        self.image_faces_model = ListEntryModel(self)
        self.image_faces_list = QListView()
        self.image_faces_list.setViewMode(QListView.ViewMode.IconMode)
        self.image_faces_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.image_faces_list.setMovement(QListView.Movement.Static)
        self.image_faces_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.image_faces_list.setIconSize(QSize(72, 72))
        self.image_faces_list.setGridSize(QSize(102, 120))
        self.image_faces_list.setSpacing(8)
        self.image_faces_list.setUniformItemSizes(True)
        self.image_faces_list.setMaximumHeight(240)
        self.image_faces_list.setModel(self.image_faces_model)
        face_editor_layout.addWidget(self.image_faces_list)

        self.face_selected_details_label = QLabel("Selected face: none")
        self.face_selected_details_label.setWordWrap(True)
        self.face_selected_details_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        face_editor_layout.addWidget(self.face_selected_details_label)
        details_layout.addWidget(self.face_editor_panel)

        self.metadata_summary_label = QLabel("")
        self.metadata_summary_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.metadata_summary_label.setWordWrap(True)
        self.metadata_summary_label.setTextFormat(Qt.TextFormat.RichText)
        self.metadata_summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        details_layout.addWidget(self.metadata_summary_label)

        self.info_text = QTextBrowser()
        self.info_text.setReadOnly(True)
        self.info_text.installEventFilter(self)
        self.info_text.document().setDefaultStyleSheet(
            """
            h3 { margin: 8px 0 4px 0; }
            p { margin: 0 0 6px 0; }
            table { width: 100%; border-collapse: collapse; }
            td.key { font-weight: 600; padding: 2px 12px 2px 0; vertical-align: top; }
            td.value { padding: 2px 0; vertical-align: top; }
            pre {
                margin: 4px 0 8px 0;
                padding: 6px;
                white-space: pre-wrap;
                font-family: Consolas, "Courier New", monospace;
            }
            """
        )
        self.info_text.setMinimumHeight(220)
        details_layout.addWidget(self.info_text, stretch=1)

        self.content_splitter.addWidget(self.image_panel)
        self.content_splitter.addWidget(self.details_panel)
        self.content_splitter.setStretchFactor(0, 1)
        self.content_splitter.setStretchFactor(1, 0)
        self.content_splitter.setSizes([780, 420])
        self.main_layout.addWidget(self.content_splitter, stretch=1)

        self.prev_button.clicked.connect(lambda: self._step(-1))
        self.next_button.clicked.connect(lambda: self._step(1))
        self.fit_button.clicked.connect(self.preview_view.fit_to_window)
        self.face_rescan_button.clicked.connect(self._auto_scan_current_image_faces)
        self.face_draw_button.clicked.connect(self._toggle_draw_face_box)
        self.face_remove_button.clicked.connect(self._remove_selected_faces)
        self.face_duplicate_button.clicked.connect(self._duplicate_selected_face)
        self.face_split_button.clicked.connect(self._split_selected_face)
        self.face_undo_button.clicked.connect(self._undo_face_edit)
        self.face_redo_button.clicked.connect(self._redo_face_edit)
        self.face_remove_all_button.clicked.connect(self._remove_all_faces)
        self.face_auto_clean_button.clicked.connect(self._auto_clean_faces)
        self.face_reset_button.clicked.connect(self._reset_face_drafts)
        self.face_save_button.clicked.connect(self._save_face_edits)
        selection_model = self.image_faces_list.selectionModel()
        if selection_model is not None:
            selection_model.selectionChanged.connect(lambda *_args: self._on_face_draft_selection_changed())
        self.preview_view.face_box_clicked.connect(self._on_preview_face_box_clicked)
        self.preview_view.face_box_drawn.connect(self._on_preview_face_box_drawn)
        self.preview_view.face_box_moved.connect(self._on_preview_face_box_moved)
        self.preview_view.face_box_resized.connect(self._on_preview_face_box_resized)
        self._install_shortcuts()
        self._apply_display_mode()

        self._update_nav_state()
        if self._image_paths:
            self.load_metadata(self._image_paths[self._index], dict(self._base_context))

    def load_metadata(self, image_path: str, context: dict[str, object]) -> None:
        self._request_id += 1
        request_id = self._request_id
        self._full_res_loaded = False
        self._update_identity(image_path)
        self._update_nav_state()
        self._update_window_title(image_path)

        if self._active_job is not None:
            self._retain_async_refs(self._active_job, self._active_thread)
            self._active_job.cancel()
            self._active_job = None
            self._active_thread = None
        if self._preview_job is not None:
            self._retain_async_refs(self._preview_job, self._preview_thread)
            self._preview_job.cancel()
            self._preview_job = None
            self._preview_thread = None

        if self._context_provider is not None:
            try:
                context = dict(context)
                context.update(self._context_provider(image_path) or {})
            except Exception:
                pass
        self._current_context = dict(context or {})
        self._current_image_path = str(image_path)

        self.preview_view.set_face_boxes(self._context_face_boxes(context), normalized=False, dirty=False)
        self.preview_view.set_selected_face_indexes(())
        self.preview_view.set_draw_mode(False)
        self.preview_view.set_pixmap(None)
        self._set_loading_state(face_message=self._context_face_message(context))
        self._load_face_editor_state(image_path, self._current_context)

        self._load_preview(image_path, request_id=request_id, full_res=False)
        if self._display_mode == "advanced":
            self._load_metadata_async(image_path, context, request_id=request_id)
        self._prefetch_neighbors()

    def _load_metadata_async(self, image_path: str, context: dict[str, object], request_id: int) -> None:
        def _run(_progress, cancel_check):
            raise_if_cancelled(cancel_check)
            metadata = self.metadata_service.get_metadata(
                image_path,
                context=context,
                include_hashes=True,
                include_face_boxes=False,
            )
            raise_if_cancelled(cancel_check)
            return metadata

        job = AsyncJob(_run)

        def _on_completed(metadata: PhotoMetadata) -> None:
            if request_id != self._request_id:
                return
            self.metadata_summary_label.setText(self._render_metadata_summary(metadata))
            self.info_text.setHtml(self._render_metadata(metadata))
            self._set_state_text(self._context_face_message(metadata.context))

        def _on_failed(message: str) -> None:
            if request_id != self._request_id:
                return
            escaped_message = html.escape(message)
            self.metadata_summary_label.setText(f"<b>Metadata</b><br>{escaped_message}")
            self.info_text.setHtml(f"<p><b>Failed to load metadata.</b></p><p>{escaped_message}</p>")
            self._set_state_text("Advanced view. Metadata failed to load for this image.")

        def _on_cancelled() -> None:
            if request_id != self._request_id:
                return
            self.metadata_summary_label.setText("<b>Metadata</b><br>Cancelled.")
            self.info_text.setHtml("<p><b>Metadata load cancelled.</b></p>")
            self._set_state_text("Advanced view. Metadata loading was cancelled.")

        job.completed.connect(_on_completed)
        job.failed.connect(_on_failed)
        job.cancelled.connect(_on_cancelled)
        self._active_job = job
        thread = start_job_in_thread(job)
        self._thread_jobs[thread] = job
        thread.finished.connect(
            lambda thread=thread: self._on_async_thread_finished(thread),
            Qt.ConnectionType.QueuedConnection,
        )
        self._active_thread = thread

    def _load_preview(self, image_path: str, *, request_id: int, full_res: bool) -> None:
        target_size = self.preview_view.viewport().size()

        def _run(_progress, cancel_check):
            raise_if_cancelled(cancel_check)
            reader = QImageReader(image_path)
            reader.setAutoTransform(True)
            if not full_res and target_size.width() > 0 and target_size.height() > 0:
                # Preserve aspect ratio: QImageReader.setScaledSize() scales to the exact size provided.
                # Use transformed size when available, otherwise rotated images can be scaled incorrectly.
                original = reader.transformedSize() if hasattr(reader, "transformedSize") else reader.size()
                if original.width() > 0 and original.height() > 0:
                    scale = min(
                        target_size.width() / max(1, original.width()),
                        target_size.height() / max(1, original.height()),
                    )
                    scaled = QSize(
                        max(1, int(original.width() * scale)),
                        max(1, int(original.height() * scale)),
                    )
                    reader.setScaledSize(scaled)
            image = reader.read()
            if image.isNull():
                raise ValueError(reader.errorString() or "Could not decode image")
            raise_if_cancelled(cancel_check)
            return image

        job = AsyncJob(_run)

        def _on_completed(image) -> None:
            if request_id != self._request_id:
                return
            pixmap = QPixmap.fromImage(image)
            self.preview_view.set_pixmap(pixmap if not pixmap.isNull() else None, preserve_zoom=full_res)
            if full_res:
                self._full_res_loaded = True

        def _on_failed(_message: str) -> None:
            if request_id != self._request_id:
                return
            self.preview_view.set_pixmap(None)

        job.completed.connect(_on_completed)
        job.failed.connect(_on_failed)
        self._preview_job = job
        thread = start_job_in_thread(job)
        self._thread_jobs[thread] = job
        thread.finished.connect(
            lambda thread=thread: self._on_async_thread_finished(thread),
            Qt.ConnectionType.QueuedConnection,
        )
        self._preview_thread = thread

    @staticmethod
    def _thread_is_running(thread) -> bool:
        if thread is None:
            return False
        try:
            return bool(thread.isRunning())
        except Exception:
            return False

    def _retain_async_refs(self, job: object | None, thread: object | None) -> None:
        if not self._thread_is_running(thread):
            return
        if any(existing_thread is thread for _existing_job, existing_thread in self._retained_async_refs):
            return
        self._retained_async_refs.append((job, thread))

    def _release_async_refs(self, job: object | None, thread: object | None) -> None:
        self._thread_jobs.pop(thread, None)
        self._retained_async_refs = [
            (existing_job, existing_thread)
            for existing_job, existing_thread in self._retained_async_refs
            if existing_thread is not thread
        ]
        if self._active_thread is thread:
            self._active_thread = None
            if self._active_job is job:
                self._active_job = None
        if self._preview_thread is thread:
            self._preview_thread = None
            if self._preview_job is job:
                self._preview_job = None
        if self._face_edit_thread is thread:
            self._face_edit_thread = None
            if self._face_edit_job is job:
                self._face_edit_job = None

    def _handle_finished_thread(self, thread) -> None:
        if thread is None:
            return
        self._release_async_refs(self._thread_jobs.get(thread), thread)

    def _on_async_thread_finished(self, thread=None) -> None:
        self._handle_finished_thread(thread)

    def _current_editable_path(self) -> str:
        if self._image_paths and 0 <= self._index < len(self._image_paths):
            return str(self._image_paths[self._index])
        return str(self._current_image_path or "")

    def _face_edit_enabled_for_context(self, context: dict[str, object] | None) -> bool:
        return bool(self._allow_face_edit and isinstance((context or {}).get("face_review"), dict))

    def _load_face_editor_state(self, image_path: str, context: dict[str, object]) -> None:
        enabled = self._face_edit_enabled_for_context(context)
        self.face_editor_panel.setVisible(enabled)
        if not enabled:
            self.preview_view.set_selected_face_indexes(())
            self.preview_view.set_draw_mode(False)
            return
        path = str(image_path or "")
        self._face_undo_by_path.setdefault(path, [])
        self._face_redo_by_path.setdefault(path, [])
        if path not in self._face_drafts_by_path:
            shared_state = None
            if callable(self.face_draft_provider):
                try:
                    shared_state = self.face_draft_provider(path)
                except Exception:
                    shared_state = None
            if isinstance(shared_state, tuple) and len(shared_state) == 2:
                shared_drafts, shared_dirty = shared_state
                drafts = list(shared_drafts or [])
                if shared_dirty:
                    self._face_original_drafts_by_path[path] = list(self._load_indexed_face_drafts(path, context))
                else:
                    self._face_original_drafts_by_path[path] = list(drafts)
                self._face_drafts_by_path[path] = list(drafts)
                if shared_dirty:
                    self._face_editor_dirty_paths.add(path)
                else:
                    self._face_editor_dirty_paths.discard(path)
            else:
                drafts = self._load_indexed_face_drafts(path, context)
                self._face_original_drafts_by_path[path] = list(drafts)
                self._face_drafts_by_path[path] = list(drafts)
        self._refresh_face_editor_ui(path)

    def _load_indexed_face_drafts(self, image_path: str, context: dict[str, object]) -> list[EditableFaceDraft]:
        records: list[IndexedFaceRecord] = []
        if self.face_service is not None:
            try:
                records = list(self.face_service.load_image_faces(image_path, include_tiny_faces=True))
            except Exception:
                records = []
        drafts: list[EditableFaceDraft] = []
        if records:
            for record in records:
                drafts.append(
                    EditableFaceDraft(
                        bbox=tuple(int(value) for value in record.face_bbox),
                        confidence=float(record.face_confidence or 1.0),
                        source="indexed",
                        person_name=str(record.person_name or ""),
                        face_index=int(record.face_index),
                    )
                )
            return drafts
        if not drafts:
            indexed = context.get("indexed_faces")
            if isinstance(indexed, dict) and isinstance(indexed.get("faces"), list):
                for index, face in enumerate(indexed.get("faces", [])):
                    if not isinstance(face, dict):
                        continue
                    bbox = face.get("bbox")
                    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                        continue
                    drafts.append(
                        EditableFaceDraft(
                            bbox=tuple(int(value) for value in bbox),
                            confidence=float(face.get("confidence", 1.0) or 1.0),
                            source="indexed",
                            person_name=str(face.get("person_name") or ""),
                            face_index=int(face.get("face_index", index)),
                        )
                    )
        return drafts

    def _refresh_face_editor_ui(self, image_path: str) -> None:
        if not self._face_edit_enabled_for_context(self._current_context):
            return
        if image_path != self._current_editable_path():
            return
        drafts = list(self._face_drafts_by_path.get(image_path, []))
        selected_indexes = self._selected_face_draft_indexes()
        selection_model = self.image_faces_list.selectionModel()
        if selection_model is not None:
            selection_model.blockSignals(True)
        entries: list[ListEntry] = []
        for index, draft in enumerate(drafts):
            title = draft.person_name or "Unlabeled"
            source = draft.source.replace("_", " ")
            entries.append(
                ListEntry(
                    title=f"{title}\nFace #{index + 1}",
                    payload=int(index),
                    icon=self._make_face_thumbnail_icon(image_path, draft.bbox),
                    tooltip=(
                        f"{Path(image_path).name}\nface #{index + 1}\nsource={source}\n"
                        f"bbox={tuple(draft.bbox)}\nconf={float(draft.confidence):.3f}"
                    ),
                )
            )
        self.image_faces_model.set_items(entries)
        for index in selected_indexes:
            self._select_face_row(int(index), append=True)
        if selection_model is not None:
            selection_model.blockSignals(False)
        self.preview_view.set_face_boxes(
            [draft.bbox for draft in drafts],
            normalized=False,
            dirty=image_path in self._face_editor_dirty_paths,
        )
        self.preview_view.set_selected_face_indexes(selected_indexes)
        self._update_draw_button_text()
        self.face_undo_button.setEnabled(bool(self._face_undo_by_path.get(image_path)))
        self.face_redo_button.setEnabled(bool(self._face_redo_by_path.get(image_path)))
        selected_count = len(selected_indexes)
        self.face_remove_button.setEnabled(selected_count > 0)
        self.face_duplicate_button.setEnabled(selected_count == 1)
        self.face_split_button.setEnabled(selected_count == 1)
        if drafts:
            dirty = image_path in self._face_editor_dirty_paths
            dirty_text = " Unsaved edits pending." if dirty else ""
            self.face_editor_summary_label.setText(
                f"{len(drafts)} face(s) loaded for {Path(image_path).name}. "
                "Select bad scans below to remove them, or draw a box on the photo for a missed face."
                f"{dirty_text}"
            )
        else:
            dirty = image_path in self._face_editor_dirty_paths
            suffix = " Unsaved edits pending." if dirty else ""
            self.face_editor_summary_label.setText(
                f"No saved faces for {Path(image_path).name}. Use Auto-Scan This Image or Draw Face Box to add one.{suffix}"
            )
        self._update_selected_face_details(image_path)

    @staticmethod
    def _face_drafts_equal(left: list[EditableFaceDraft], right: list[EditableFaceDraft]) -> bool:
        return [
            (tuple(item.bbox), round(float(item.confidence), 6), str(item.person_name or ""))
            for item in left
        ] == [
            (tuple(item.bbox), round(float(item.confidence), 6), str(item.person_name or ""))
            for item in right
        ]

    def _mark_face_editor_dirty(self, image_path: str) -> None:
        current = list(self._face_drafts_by_path.get(image_path, []))
        original = list(self._face_original_drafts_by_path.get(image_path, []))
        if self._face_drafts_equal(current, original):
            self._face_editor_dirty_paths.discard(image_path)
        else:
            self._face_editor_dirty_paths.add(image_path)
        self._publish_face_drafts(image_path)

    def _publish_face_drafts(self, image_path: str) -> None:
        if not callable(self.face_draft_updated_callback):
            return
        try:
            self.face_draft_updated_callback(
                image_path,
                list(self._face_drafts_by_path.get(image_path, [])),
                image_path in self._face_editor_dirty_paths,
            )
        except Exception:
            pass

    @staticmethod
    def _clone_face_drafts(drafts: list[EditableFaceDraft]) -> list[EditableFaceDraft]:
        return [
            EditableFaceDraft(
                bbox=tuple(int(value) for value in draft.bbox),
                confidence=float(draft.confidence or 1.0),
                source=str(draft.source or "manual"),
                person_name=str(draft.person_name or ""),
                face_index=int(draft.face_index),
            )
            for draft in list(drafts or [])
        ]

    def _push_face_edit_history(self, image_path: str) -> None:
        current = self._clone_face_drafts(list(self._face_drafts_by_path.get(image_path, [])))
        undo_stack = self._face_undo_by_path.setdefault(image_path, [])
        if undo_stack and self._face_drafts_equal(undo_stack[-1], current):
            return
        undo_stack.append(current)
        self._face_redo_by_path[image_path] = []

    def _apply_face_edit_drafts(
        self,
        image_path: str,
        drafts: list[EditableFaceDraft],
        *,
        selected_indexes: list[int] | None = None,
    ) -> None:
        self._face_drafts_by_path[image_path] = self._clone_face_drafts(drafts)
        self._mark_face_editor_dirty(image_path)
        self._refresh_face_editor_ui(image_path)
        if selected_indexes is None:
            return
        selection_model = self.image_faces_list.selectionModel()
        if selection_model is not None:
            selection_model.blockSignals(True)
            selection_model.clearSelection()
        for index in selected_indexes:
            self._select_face_row(int(index), append=True)
        if selection_model is not None:
            selection_model.blockSignals(False)
        self.preview_view.set_selected_face_indexes(selected_indexes)
        self._update_selected_face_details(image_path)

    def _select_face_row(self, row: int, *, append: bool = False) -> None:
        if row < 0 or row >= self.image_faces_model.rowCount():
            return
        index = self.image_faces_model.index(int(row), 0)
        if not index.isValid():
            return
        selection_model = self.image_faces_list.selectionModel()
        if selection_model is None:
            return
        flags = QItemSelectionModel.SelectionFlag.Select
        if not append:
            flags |= QItemSelectionModel.SelectionFlag.ClearAndSelect
        selection_model.select(index, flags)
        selection_model.setCurrentIndex(index, flags)
        self.image_faces_list.setCurrentIndex(index)
        self.image_faces_list.scrollTo(index)

    def _selected_face_draft_indexes(self) -> list[int]:
        indexes: list[int] = []
        selection_model = self.image_faces_list.selectionModel()
        if selection_model is None:
            return indexes
        for model_index in selection_model.selectedIndexes():
            try:
                indexes.append(int(model_index.data(ListEntryModel.PayloadRole)))
            except Exception:
                continue
        return sorted(set(indexes))

    def _on_face_draft_selection_changed(self) -> None:
        self.preview_view.set_selected_face_indexes(self._selected_face_draft_indexes())
        self._update_selected_face_details(self._current_editable_path())

    def _update_selected_face_details(self, image_path: str) -> None:
        selected = self._selected_face_draft_indexes()
        drafts = list(self._face_drafts_by_path.get(image_path, []))
        if not selected:
            self.face_selected_details_label.setText("Selected face: none")
            return
        first_index = int(selected[0])
        if first_index < 0 or first_index >= len(drafts):
            self.face_selected_details_label.setText("Selected face: none")
            return
        draft = drafts[first_index]
        source = str(draft.source or "manual").replace("_", " ")
        label = draft.person_name or "Unlabeled"
        self.face_selected_details_label.setText(
            f"Selected face #{first_index + 1}: {label} | source={source} | "
            f"bbox={tuple(int(value) for value in draft.bbox)} | conf={float(draft.confidence):.3f}"
        )

    def _on_preview_face_box_clicked(self, index: int) -> None:
        if index < 0 or index >= self.image_faces_model.rowCount():
            return
        selection_model = self.image_faces_list.selectionModel()
        if selection_model is not None:
            selection_model.blockSignals(True)
            selection_model.clearSelection()
            self._select_face_row(int(index))
            selection_model.blockSignals(False)
        self.preview_view.set_selected_face_indexes([int(index)])

    def _update_draw_button_text(self) -> None:
        if self.preview_view.draw_mode_enabled():
            self.face_draw_button.setText("Cancel Drawing")
        else:
            self.face_draw_button.setText("Draw Face Box")

    def _toggle_draw_face_box(self) -> None:
        enabled = not self.preview_view.draw_mode_enabled()
        self.preview_view.set_draw_mode(enabled)
        self._update_draw_button_text()
        if enabled:
            self.face_editor_summary_label.setText("Drag a box directly on the photo, then release to add the face draft.")
        else:
            self._refresh_face_editor_ui(self._current_editable_path())

    def _on_preview_face_box_drawn(self, bbox: tuple[int, int, int, int]) -> None:
        image_path = self._current_editable_path()
        self._push_face_edit_history(image_path)
        drafts = list(self._face_drafts_by_path.get(image_path, []))
        drafts.append(
            EditableFaceDraft(
                bbox=tuple(int(value) for value in bbox),
                confidence=1.0,
                source="manual",
            )
        )
        self._apply_face_edit_drafts(image_path, drafts)
        if self.image_faces_model.rowCount() > 0:
            last_row = self.image_faces_model.rowCount() - 1
            selection_model = self.image_faces_list.selectionModel()
            if selection_model is not None:
                selection_model.blockSignals(True)
                selection_model.clearSelection()
                self._select_face_row(last_row)
                selection_model.blockSignals(False)
            self.preview_view.set_selected_face_indexes([last_row])

    def _on_preview_face_box_moved(self, index: int, bbox: tuple[int, int, int, int]) -> None:
        image_path = self._current_editable_path()
        drafts = list(self._face_drafts_by_path.get(image_path, []))
        if index < 0 or index >= len(drafts):
            return
        self._push_face_edit_history(image_path)
        draft = drafts[index]
        drafts[index] = EditableFaceDraft(
            bbox=tuple(int(value) for value in bbox),
            confidence=float(draft.confidence or 1.0),
            source=str(draft.source or "manual"),
            person_name=str(draft.person_name or ""),
            face_index=int(draft.face_index),
        )
        self._apply_face_edit_drafts(image_path, drafts, selected_indexes=[int(index)])

    def _on_preview_face_box_resized(self, index: int, bbox: tuple[int, int, int, int]) -> None:
        image_path = self._current_editable_path()
        drafts = list(self._face_drafts_by_path.get(image_path, []))
        if index < 0 or index >= len(drafts):
            return
        self._push_face_edit_history(image_path)
        draft = drafts[index]
        drafts[index] = EditableFaceDraft(
            bbox=tuple(int(value) for value in bbox),
            confidence=float(draft.confidence or 1.0),
            source=str(draft.source or "manual"),
            person_name=str(draft.person_name or ""),
            face_index=int(draft.face_index),
        )
        self._apply_face_edit_drafts(image_path, drafts, selected_indexes=[int(index)])

    def _reset_face_drafts(self) -> None:
        image_path = self._current_editable_path()
        if image_path in self._face_editor_dirty_paths:
            if not confirmBox("Discard face edits?", "Reset the current image face draft back to the saved indexed faces?", parent=self):
                return
        self._push_face_edit_history(image_path)
        self._face_drafts_by_path[image_path] = self._clone_face_drafts(list(self._face_original_drafts_by_path.get(image_path, [])))
        self._face_editor_dirty_paths.discard(image_path)
        self._face_redo_by_path[image_path] = []
        self.preview_view.set_draw_mode(False)
        self._publish_face_drafts(image_path)
        self._refresh_face_editor_ui(image_path)

    def _remove_selected_faces(self) -> None:
        image_path = self._current_editable_path()
        selected = self._selected_face_draft_indexes()
        if not selected:
            errorBox("No faces selected", "Select one or more faces in the Image Faces panel first.")
            return
        self._push_face_edit_history(image_path)
        drafts = list(self._face_drafts_by_path.get(image_path, []))
        for index in sorted(selected, reverse=True):
            if 0 <= index < len(drafts):
                drafts.pop(index)
        next_selection = []
        if drafts:
            next_selection = [min(selected[0], len(drafts) - 1)]
        self._apply_face_edit_drafts(image_path, drafts, selected_indexes=next_selection)

    def _duplicate_selected_face(self) -> None:
        image_path = self._current_editable_path()
        selected = self._selected_face_draft_indexes()
        if len(selected) != 1:
            errorBox("Select one face", "Select exactly one face to duplicate.")
            return
        drafts = list(self._face_drafts_by_path.get(image_path, []))
        index = int(selected[0])
        if index < 0 or index >= len(drafts):
            return
        self._push_face_edit_history(image_path)
        draft = drafts[index]
        x1, y1, x2, y2 = [int(value) for value in draft.bbox]
        offset = max(4, min(12, max(2, (x2 - x1) // 8)))
        duplicated = EditableFaceDraft(
            bbox=(x1 + offset, y1 + offset, x2 + offset, y2 + offset),
            confidence=float(draft.confidence or 1.0),
            source="manual",
            person_name=str(draft.person_name or ""),
            face_index=-1,
        )
        drafts.insert(index + 1, duplicated)
        self._apply_face_edit_drafts(image_path, drafts, selected_indexes=[index + 1])

    def _split_selected_face(self) -> None:
        image_path = self._current_editable_path()
        selected = self._selected_face_draft_indexes()
        if len(selected) != 1:
            errorBox("Select one face", "Select exactly one face to split.")
            return
        drafts = list(self._face_drafts_by_path.get(image_path, []))
        index = int(selected[0])
        if index < 0 or index >= len(drafts):
            return
        draft = drafts[index]
        x1, y1, x2, y2 = [int(value) for value in draft.bbox]
        width = max(2, x2 - x1)
        if width < 6:
            errorBox("Face too small", "The selected face box is too narrow to split cleanly.")
            return
        self._push_face_edit_history(image_path)
        midpoint = x1 + width // 2
        gap = 2
        left_box = (x1, y1, max(x1 + 2, midpoint - gap), y2)
        right_box = (min(x2 - 2, midpoint + gap), y1, x2, y2)
        replacement = [
            EditableFaceDraft(
                bbox=left_box,
                confidence=float(draft.confidence or 1.0),
                source="manual",
                person_name=str(draft.person_name or ""),
                face_index=-1,
            ),
            EditableFaceDraft(
                bbox=right_box,
                confidence=float(draft.confidence or 1.0),
                source="manual",
                person_name=str(draft.person_name or ""),
                face_index=-1,
            ),
        ]
        drafts[index : index + 1] = replacement
        self._apply_face_edit_drafts(image_path, drafts, selected_indexes=[index, index + 1])

    def _undo_face_edit(self) -> None:
        image_path = self._current_editable_path()
        undo_stack = self._face_undo_by_path.get(image_path, [])
        if not undo_stack:
            return
        current = self._clone_face_drafts(list(self._face_drafts_by_path.get(image_path, [])))
        previous = undo_stack.pop()
        self._face_redo_by_path.setdefault(image_path, []).append(current)
        self._apply_face_edit_drafts(image_path, previous)

    def _redo_face_edit(self) -> None:
        image_path = self._current_editable_path()
        redo_stack = self._face_redo_by_path.get(image_path, [])
        if not redo_stack:
            return
        current = self._clone_face_drafts(list(self._face_drafts_by_path.get(image_path, [])))
        next_state = redo_stack.pop()
        self._face_undo_by_path.setdefault(image_path, []).append(current)
        self._apply_face_edit_drafts(image_path, next_state)

    def _nudge_selected_faces(self, delta_x: int, delta_y: int) -> bool:
        image_path = self._current_editable_path()
        selected = self._selected_face_draft_indexes()
        drafts = list(self._face_drafts_by_path.get(image_path, []))
        if not selected or not drafts:
            return False
        self._push_face_edit_history(image_path)
        updated = False
        for index in selected:
            if index < 0 or index >= len(drafts):
                continue
            draft = drafts[index]
            x1, y1, x2, y2 = [int(value) for value in draft.bbox]
            drafts[index] = EditableFaceDraft(
                bbox=(x1 + int(delta_x), y1 + int(delta_y), x2 + int(delta_x), y2 + int(delta_y)),
                confidence=float(draft.confidence or 1.0),
                source=str(draft.source or "manual"),
                person_name=str(draft.person_name or ""),
                face_index=int(draft.face_index),
            )
            updated = True
        if not updated:
            return False
        self._apply_face_edit_drafts(image_path, drafts, selected_indexes=selected)
        return True

    def _remove_all_faces(self) -> None:
        image_path = self._current_editable_path()
        drafts = list(self._face_drafts_by_path.get(image_path, []))
        if not drafts:
            return
        if not confirmBox(
            "Remove all faces?",
            f"Clear every face box from {Path(image_path).name}? This stays as an unsaved draft until you save it.",
            parent=self,
        ):
            return
        self._push_face_edit_history(image_path)
        self._apply_face_edit_drafts(image_path, [], selected_indexes=[])

    def _auto_clean_faces(self) -> None:
        image_path = self._current_editable_path()
        drafts = list(self._face_drafts_by_path.get(image_path, []))
        if not drafts or not callable(self.face_auto_clean_callback):
            return
        try:
            cleaned, metrics = self.face_auto_clean_callback(image_path, list(drafts))
        except Exception as exc:
            errorBox("Auto-clean failed", str(exc))
            return
        if int(metrics.get("removed", 0)) <= 0:
            self.face_editor_summary_label.setText(
                f"Auto-clean found no obvious junk in {Path(image_path).name}. "
                f"{int(metrics.get('review', 0))} face(s) still need manual review."
            )
            return
        self._push_face_edit_history(image_path)
        self._apply_face_edit_drafts(image_path, list(cleaned or []))
        self.face_editor_summary_label.setText(
            f"Auto-clean removed {int(metrics.get('removed', 0))} obvious junk detection(s) from {Path(image_path).name}. "
            f"{int(metrics.get('review', 0))} face(s) still need manual review."
        )

    def _make_face_thumbnail_icon(self, image_path: str, bbox: tuple[int, int, int, int]) -> QIcon:
        thumb_size = 72
        try:
            with Image.open(image_path) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
                x1, y1, x2, y2 = [int(value) for value in bbox]
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
                crop.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
                canvas = Image.new("RGB", (thumb_size, thumb_size), (18, 18, 18))
                canvas.paste(crop, ((thumb_size - crop.width) // 2, (thumb_size - crop.height) // 2))
                return QIcon(QPixmap.fromImage(ImageQt(canvas)))
        except Exception:
            return QIcon()

    def _set_face_edit_busy(self, busy: bool, message: str = "") -> None:
        for button in [
            self.face_rescan_button,
            self.face_draw_button,
            self.face_remove_button,
            self.face_duplicate_button,
            self.face_split_button,
            self.face_undo_button,
            self.face_redo_button,
            self.face_remove_all_button,
            self.face_auto_clean_button,
            self.face_reset_button,
            self.face_save_button,
        ]:
            button.setEnabled(not busy)
        if message:
            self.face_editor_summary_label.setText(message)

    def _start_face_edit_job(self, label: str, fn, on_completed) -> None:
        if self._face_edit_thread is not None:
            try:
                if self._face_edit_thread.isRunning():
                    errorBox("Busy", "Another face edit operation is already running for this image.")
                    return
            except RuntimeError:
                self._face_edit_thread = None
                self._face_edit_job = None
        job = AsyncJob(fn)

        def _on_completed(result) -> None:
            self._set_face_edit_busy(False)
            on_completed(result)

        def _on_failed(message: str) -> None:
            self._set_face_edit_busy(False)
            errorBox(f"{label} failed", message)

        def _on_cancelled() -> None:
            self._set_face_edit_busy(False, "Face edit operation cancelled.")

        job.completed.connect(_on_completed)
        job.failed.connect(_on_failed)
        job.cancelled.connect(_on_cancelled)
        self._face_edit_job = job
        thread = start_job_in_thread(job)
        self._thread_jobs[thread] = job
        thread.finished.connect(
            lambda thread=thread: self._on_async_thread_finished(thread),
            Qt.ConnectionType.QueuedConnection,
        )
        self._face_edit_thread = thread
        self._set_face_edit_busy(True, f"{label} running...")

    def _set_face_drafts_for_path(self, image_path: str, drafts: list[EditableFaceDraft]) -> None:
        self._face_drafts_by_path[image_path] = list(drafts)
        self._mark_face_editor_dirty(image_path)
        if image_path == self._current_editable_path():
            self._refresh_face_editor_ui(image_path)

    def _auto_scan_current_image_faces(self) -> None:
        image_path = self._current_editable_path()
        if not image_path or self.face_service is None:
            return
        if image_path in self._face_editor_dirty_paths:
            if not confirmBox("Replace draft?", "Auto-Scan This Image will replace the current unsaved face draft. Continue?", parent=self):
                return

        def _run(progress, cancel_check):
            progress(-1, "Scanning current image for faces...")
            _ = cancel_check
            return self.face_service.detect_faces_for_image(image_path, include_tiny_faces=True)

        def _done(faces) -> None:
            self._push_face_edit_history(image_path)
            drafts = [
                EditableFaceDraft(
                    bbox=tuple(int(value) for value in face.bbox),
                    confidence=float(face.confidence or 1.0),
                    source="auto",
                )
                for face in list(faces or [])
            ]
            self._set_face_drafts_for_path(image_path, drafts)
            if image_path == self._current_editable_path():
                self.face_editor_summary_label.setText(
                    f"Auto-scan loaded {len(drafts)} face(s) for {Path(image_path).name}. Remove bad scans or draw another box before saving."
                )

        self._start_face_edit_job("Auto-scan current image", _run, _done)

    def _save_face_edits(self) -> None:
        image_path = self._current_editable_path()
        if not image_path or self.face_service is None:
            return
        drafts = list(self._face_drafts_by_path.get(image_path, []))
        payload = [
            EditableFaceInput(
                bbox=tuple(int(value) for value in draft.bbox),
                confidence=float(draft.confidence or 1.0),
            )
            for draft in drafts
        ]

        def _run(progress, cancel_check):
            progress(-1, "Saving face edits for this image...")
            _ = cancel_check
            return self.face_service.save_image_faces(image_path, payload, preserve_labels=True)

        def _done(records) -> None:
            refreshed = [
                EditableFaceDraft(
                    bbox=tuple(int(value) for value in record.face_bbox),
                    confidence=float(record.face_confidence or 1.0),
                    source="indexed",
                    person_name=str(record.person_name or ""),
                    face_index=int(record.face_index),
                )
                for record in list(records or [])
            ]
            self._face_original_drafts_by_path[image_path] = list(refreshed)
            self._face_drafts_by_path[image_path] = list(refreshed)
            self._face_editor_dirty_paths.discard(image_path)
            self._face_undo_by_path[image_path] = []
            self._face_redo_by_path[image_path] = []
            self._publish_face_drafts(image_path)
            if image_path == self._current_editable_path():
                self._refresh_face_editor_ui(image_path)
            if callable(self.face_edit_saved_callback):
                try:
                    self.face_edit_saved_callback(image_path)
                except Exception:
                    pass
            if image_path == self._current_editable_path() and refreshed:
                self.face_editor_summary_label.setText(
                    f"Saved {len(refreshed)} face(s) for {Path(image_path).name}."
                )
            elif image_path == self._current_editable_path():
                self.face_editor_summary_label.setText(
                    f"Saved face edits for {Path(image_path).name}. This image now has no indexed faces."
                )

        self._start_face_edit_job("Save face edits", _run, _done)

    def shutdown_jobs(self, *, timeout_ms: int = 2500) -> bool:
        ready_to_close = True
        active_pairs = [
            (self._active_job, self._active_thread),
            (self._preview_job, self._preview_thread),
            (self._face_edit_job, self._face_edit_thread),
            *self._retained_async_refs,
        ]
        for job, thread in active_pairs:
            if job is not None:
                try:
                    job.cancel()
                except Exception:
                    pass
            thread_finished = True
            if thread is not None:
                try:
                    thread_finished = wait_for_thread_shutdown(thread, timeout_ms=timeout_ms)
                except RuntimeError:
                    thread_finished = True
            ready_to_close = bool(thread_finished) and ready_to_close
        self._active_job = None
        self._active_thread = None
        self._preview_job = None
        self._preview_thread = None
        self._face_edit_job = None
        self._face_edit_thread = None
        if ready_to_close:
            self._retained_async_refs = []
            self._thread_jobs = {}
        return ready_to_close

    def _on_zoom_changed(self, zoom: float) -> None:
        if zoom <= 1.01 or self._full_res_loaded or not self._image_paths:
            return
        self._load_preview(self._image_paths[self._index], request_id=self._request_id, full_res=True)

    def _prefetch_neighbors(self) -> None:
        # Keep this lightweight: touching the filesystem cache through QImageReader is enough.
        if not self._image_paths:
            return
        neighbor_indexes = [idx for idx in {self._index - 1, self._index + 1} if 0 <= idx < len(self._image_paths)]
        for neighbor_index in neighbor_indexes:
            try:
                reader = QImageReader(self._image_paths[neighbor_index])
                reader.setAutoTransform(True)
                _ = reader.size()
            except Exception:
                pass

    def closeEvent(self, event) -> None:
        if not self.shutdown_jobs():
            event.ignore()
            return
        return super().closeEvent(event)

    def _install_shortcuts(self) -> None:
        shortcut_specs = [
            ("Left", lambda: self._nudge_selected_faces(-1, 0) or self._step(-1)),
            ("Right", lambda: self._nudge_selected_faces(1, 0) or self._step(1)),
            ("Up", lambda: self._nudge_selected_faces(0, -1)),
            ("Down", lambda: self._nudge_selected_faces(0, 1)),
            ("Shift+Left", lambda: self._nudge_selected_faces(-10, 0)),
            ("Shift+Right", lambda: self._nudge_selected_faces(10, 0)),
            ("Shift+Up", lambda: self._nudge_selected_faces(0, -10)),
            ("Shift+Down", lambda: self._nudge_selected_faces(0, 10)),
            ("A", lambda: self._step(-1)),
            ("D", lambda: self._step(1)),
            ("F", self.preview_view.fit_to_window),
            ("Ctrl+Z", self._undo_face_edit),
            ("Ctrl+Y", self._redo_face_edit),
        ]
        for key, handler in shortcut_specs:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            shortcut.activated.connect(handler)
            self._shortcuts.append(shortcut)

    def _apply_display_mode(self) -> None:
        advanced = self._display_mode == "advanced"
        self.metadata_summary_label.setVisible(advanced)
        self.info_text.setVisible(advanced)

    def _update_identity(self, image_path: str) -> None:
        path = Path(image_path)
        image_name = path.name or str(path)
        self.name_label.setText(image_name)
        self.name_label.setToolTip(image_name)
        folder_text = str(path.parent) if str(path.parent) else "."
        self.folder_label.setText(folder_text)
        self.folder_label.setToolTip(folder_text)

    def _update_window_title(self, image_path: str) -> None:
        path = Path(image_path)
        self.setWindowTitle(
            f"Photo Inspector - {path.name or path} ({self._index + 1}/{max(1, len(self._image_paths))})"
        )

    def _set_state_text(self, message: str) -> None:
        self.state_label.setText(message)

    def _set_loading_state(self, *, face_message: str = "") -> None:
        if self._display_mode == "basic":
            self.metadata_summary_label.clear()
            self.info_text.clear()
            message = str(face_message or "").strip()
            if not message or message.startswith("Advanced view."):
                message = "Basic view. Left/Right arrows browse the current gallery group. Metadata is hidden in basic mode."
            self._set_state_text(message)
            return
        self.metadata_summary_label.setText("<b>Important Details</b><br>Loading metadata...")
        self.info_text.setHtml("<p>Loading metadata...</p>")
        self._set_state_text(face_message or "Advanced view. Left/Right arrows browse the current gallery group.")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        return super().keyPressEvent(event)

    def eventFilter(self, watched, event) -> bool:
        if watched is getattr(self, "info_text", None) and event.type() == QEvent.Type.KeyPress:
            modifiers = event.modifiers()
            if modifiers in (Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.KeypadModifier):
                if event.key() == Qt.Key.Key_Left:
                    self._step(-1)
                    return True
                if event.key() == Qt.Key.Key_Right:
                    self._step(1)
                    return True
        return super().eventFilter(watched, event)

    def _step(self, delta: int) -> None:
        if not self._image_paths:
            return
        new_index = self._index + int(delta)
        if new_index < 0 or new_index >= len(self._image_paths):
            return
        self._index = new_index
        self.load_metadata(self._image_paths[self._index], dict(self._base_context))

    def _update_nav_state(self) -> None:
        total = len(self._image_paths)
        if total <= 1:
            self.prev_button.setEnabled(False)
            self.next_button.setEnabled(False)
            self.index_label.setText("")
            return
        self.prev_button.setEnabled(self._index > 0)
        self.next_button.setEnabled(self._index < total - 1)
        self.index_label.setText(f"{self._index + 1} / {total}")

    @staticmethod
    def _format_file_size(file_size: int) -> str:
        value = float(max(0, int(file_size)))
        units = ["B", "KB", "MB", "GB", "TB"]
        for unit in units:
            if value < 1024.0 or unit == units[-1]:
                if unit == "B":
                    return f"{int(value)} {unit}"
                return f"{value:.1f} {unit}"
            value /= 1024.0
        return f"{int(file_size)} B"

    @staticmethod
    def _context_face_boxes(context: dict[str, object] | None) -> list[tuple[float, float, float, float]]:
        if not isinstance(context, dict):
            return []
        indexed = context.get("indexed_faces")
        if isinstance(indexed, dict):
            faces = indexed.get("faces")
            if isinstance(faces, list):
                boxes = []
                for face in faces:
                    if not isinstance(face, dict):
                        continue
                    bbox = face.get("bbox")
                    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                        boxes.append(tuple(float(value) for value in bbox))
                if boxes:
                    return boxes
        for key in ("face_search", "face_name", "face_label"):
            payload = context.get(key)
            if not isinstance(payload, dict):
                continue
            bbox = payload.get("bbox")
            if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                return [tuple(float(value) for value in bbox)]
        return []

    @classmethod
    def _context_face_message(cls, context: dict[str, object] | None) -> str:
        boxes = cls._context_face_boxes(context)
        review = context.get("face_review") if isinstance(context, dict) else None
        if not boxes:
            if isinstance(review, dict):
                status = str(review.get("status") or "").strip()
                if status == "no_faces":
                    return "This photo was scanned and no faces were detected."
                if status == "not_scanned":
                    return "This photo has not been scanned in the active face database."
                if status == "tiny_hidden":
                    return "This photo only has tiny detections hidden by the current filter."
            return "Advanced view. Left/Right arrows browse the current gallery group."
        names: list[str] = []
        if isinstance(context, dict):
            indexed = context.get("indexed_faces")
            if isinstance(indexed, dict) and isinstance(indexed.get("faces"), list):
                for face in indexed.get("faces", []):
                    if not isinstance(face, dict):
                        continue
                    name = str(face.get("person_name") or "").strip()
                    if name and name not in names:
                        names.append(name)
        if names:
            return f"Showing {len(boxes)} detected face(s). Names: {', '.join(names[:3])}."
        return f"Showing {len(boxes)} detected face(s) on the full photo."

    @classmethod
    def _render_metadata_summary(cls, metadata: PhotoMetadata) -> str:
        camera = html.escape(metadata.camera or "Unknown")
        tags = metadata.context.get("tags") if isinstance(metadata.context, dict) else None
        tags_line = ""
        if isinstance(tags, (list, tuple)) and tags:
            tags_line = f"<br><b>Tags</b> {html.escape(', '.join(str(tag) for tag in tags))}"
        return (
            "<b>Important Details</b><br>"
            f"<b>Dimensions</b> {metadata.width} x {metadata.height}"
            f" &nbsp; | &nbsp; <b>Size</b> {cls._format_file_size(metadata.file_size)}"
            f" &nbsp; | &nbsp; <b>Modified</b> {html.escape(metadata.modified_at)}"
            f"<br><b>Camera</b> {camera}"
            f"{tags_line}"
        )

    @classmethod
    def _render_metadata(cls, metadata: PhotoMetadata) -> str:
        exif_rows = []
        for key, value in sorted(metadata.exif.items()):
            exif_rows.append(
                "<tr>"
                f"<td class='key'>{html.escape(str(key))}</td>"
                f"<td class='value'>{html.escape(str(value))}</td>"
                "</tr>"
            )
        exif_html = (
            "<table>{rows}</table>".format(rows="".join(exif_rows))
            if exif_rows
            else "<p>No EXIF metadata was found for this image.</p>"
        )
        hashes_json = html.escape(json.dumps(metadata.hashes, indent=2, default=str))
        faces_json = html.escape(json.dumps(metadata.face_boxes, indent=2, default=str))
        context_json = html.escape(json.dumps(metadata.context, indent=2, default=str))
        image_path = html.escape(metadata.image_path)
        return (
            "<h3>File</h3>"
            f"<table>"
            f"<tr><td class='key'>Path</td><td class='value'>{image_path}</td></tr>"
            f"<tr><td class='key'>Dimensions</td><td class='value'>{metadata.width} x {metadata.height}</td></tr>"
            f"<tr><td class='key'>Size</td><td class='value'>{cls._format_file_size(metadata.file_size)}</td></tr>"
            f"<tr><td class='key'>Modified</td><td class='value'>{html.escape(metadata.modified_at)}</td></tr>"
            f"<tr><td class='key'>Camera</td><td class='value'>{html.escape(metadata.camera or 'Unknown')}</td></tr>"
            f"</table>"
            "<h3>EXIF</h3>"
            f"{exif_html}"
            "<h3>Hashes</h3>"
            f"<pre>{hashes_json}</pre>"
            "<h3>Cluster Context</h3>"
            f"<pre>{context_json}</pre>"
            "<h3>Face Boxes</h3>"
            f"<pre>{faces_json}</pre>"
        )
