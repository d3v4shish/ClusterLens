from __future__ import annotations

import html
import json
from dataclasses import dataclass
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, QItemSelectionModel, QRect, QSize, Qt, pyqtSlot
from PyQt6.QtGui import QIcon, QImage, QImageReader, QKeyEvent, QKeySequence, QPainter, QPixmap, QShortcut
from PyQt6.QtWidgets import QAbstractItemView, QComboBox, QDialog, QFormLayout, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QListView, QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSizePolicy, QSpinBox, QSplitter, QTabWidget, QTextBrowser, QToolButton, QVBoxLayout, QWidget

from app.services.face_types import EditableFaceInput
from app.services.photo_metadata import PhotoEditDraft, PhotoEditService, PhotoMetadata, PhotoMetadataService
from ui.async_job import AsyncJob, raise_if_cancelled, start_job_in_thread, wait_for_thread_shutdown
from ui.error_mbox import confirmBox, errorBox
from ui.entity_picker import EntityPicker, EntityPickerDialog
from ui.common import HelpIconButton, KeyboardTabBar, ResponsiveFlowLayout
from ui.job_manager import JobManager
from ui.list_models import ListEntry, ListEntryModel
from ui.icons import apply_icon
from ui.theme import get_theme_manager
from ui.work_coordinator import JobSpec, WorkCoordinator
from ui.zoomable_image import ZoomableImageView, adaptive_neutral_backdrop
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
        allow_metadata_edit: bool = True,
        allow_file_rename: bool = True,
        rename_current_callback: Callable[[str], None] | None = None,
        job_manager: JobManager | None = None,
        work_coordinator: WorkCoordinator | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.metadata_service = metadata_service or PhotoMetadataService()
        self.photo_edit_service = PhotoEditService()
        self.face_service = face_service
        self.face_draft_provider = face_draft_provider
        self.face_draft_updated_callback = face_draft_updated_callback
        self.face_edit_saved_callback = face_edit_saved_callback
        self.face_auto_clean_callback = face_auto_clean_callback
        self._allow_metadata_edit = bool(allow_metadata_edit)
        self._allow_file_rename = bool(allow_file_rename)
        self.rename_current_callback = rename_current_callback
        self.job_manager = job_manager
        self.work_coordinator = work_coordinator
        self._coordinated_job_ids: dict[AsyncJob, int] = {}
        # Keep the Faces surface visible in every inspector route.  A missing
        # service explains its readiness state instead of making face regions
        # appear to be unsupported by that workspace.
        self._allow_face_edit = bool(allow_face_edit)
        self._active_thread = None
        self._active_job = None
        self._preview_thread = None
        self._preview_job = None
        self._face_draft_thread = None
        self._face_draft_job = None
        self._face_thumbnail_thread = None
        self._face_thumbnail_job = None
        self._face_thumbnail_request_signature: tuple[object, ...] | None = None
        self._face_thumbnail_cache: dict[tuple[str, tuple[int, int, int, int]], QIcon] = {}
        self._prefetch_thread = None
        self._prefetch_job = None
        self._face_edit_thread = None
        self._face_edit_job = None
        self._face_name_suggestion_thread = None
        self._face_name_suggestion_job = None
        self._metadata_edit_thread = None
        self._metadata_edit_job = None
        self._retained_async_refs: list[tuple[object | None, object | None]] = []
        self._thread_jobs: dict[object, object | None] = {}
        self._progress_jobs: dict[int, dict[str, object]] = {}
        self._progress_sequence = 0
        self._request_id = 0
        self._full_res_loaded = False
        self._full_res_requested = False
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
        self._metadata_baseline_draft = PhotoEditDraft()
        self._metadata_populating = False
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

        self.viewer_actions = QWidget(self.image_panel)
        nav = ResponsiveFlowLayout(self.viewer_actions, spacing=6)
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.fit_button = QPushButton("Fit")
        self.actual_size_button = QPushButton("1:1")
        self.zoom_out_button = QPushButton("−")
        self.zoom_in_button = QPushButton("+")
        self.viewer_background_combo = QComboBox()
        self.viewer_background_combo.addItem("Adaptive", "adaptive_neutral")
        self.viewer_background_combo.addItem("Theme", "theme")
        self.viewer_background_combo.addItem("Black", "black")
        self.viewer_background_combo.addItem("Middle gray", "middle_gray")
        self.viewer_background_combo.addItem("Light gray", "light_gray")
        self.viewer_background_combo.setAccessibleName("Photo viewer background")
        self.viewer_background_combo.setToolTip(
            "Choose only the photo canvas background. Adaptive samples the loaded preview border; app chrome is unchanged."
        )
        theme_manager = get_theme_manager()
        viewer_backdrop = theme_manager.viewer_backdrop if theme_manager is not None else "adaptive_neutral"
        backdrop_index = self.viewer_background_combo.findData(viewer_backdrop)
        self.viewer_background_combo.setCurrentIndex(max(0, backdrop_index))
        self.viewer_background_combo.currentIndexChanged.connect(self._viewer_background_changed)
        if theme_manager is not None:
            theme_manager.viewer_backdrop_changed.connect(self._sync_viewer_background_combo)
        self.full_screen_button = QPushButton("Full screen")
        self.face_regions_button = QPushButton("Face regions")
        self.capture_time_button = QPushButton("Set capture time…")
        self.fit_button.setAccessibleName("Fit image to viewer")
        self.actual_size_button.setAccessibleName("Show image at actual size")
        self.zoom_out_button.setAccessibleName("Zoom out")
        self.zoom_in_button.setAccessibleName("Zoom in")
        self.full_screen_button.setAccessibleName("Toggle full-screen viewer")
        self.face_regions_button.setAccessibleName("Show or hide face regions")
        self.capture_time_button.setAccessibleName("Set photo capture time")
        self.face_regions_button.setCheckable(True)
        self.face_regions_button.setChecked(True)
        self.zoom_label = QLabel("Fit")
        self.prev_button.setToolTip("Previous photo (Left or A)")
        self.next_button.setToolTip("Next photo (Right or D)")
        self.fit_button.setToolTip("Fit the image to the viewer (F)")
        self.actual_size_button.setToolTip("Show source pixels at 1:1 (1)")
        self.zoom_out_button.setToolTip("Zoom out (−)")
        self.zoom_in_button.setToolTip("Zoom in (+)")
        self.full_screen_button.setToolTip("Toggle full-screen viewer (F11)")
        self.face_regions_button.setToolTip("Show or hide detected and saved face-region boxes on this photo.")
        self.capture_time_button.setToolTip(
            "Enter a capture date and time for this photo, then Save sidecar. "
            "Timeline treats a valid saved value as your explicit correction after Refresh dates."
        )
        apply_icon(self.face_regions_button, "faces")
        apply_icon(self.capture_time_button, "metadata")
        self.zoom_label.setToolTip("Mouse-wheel zoom is centered on the pointer. Middle-drag pans a zoomed photo.")
        self.index_label = QLabel("")
        self.index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav.addWidget(self.prev_button)
        nav.addWidget(self.next_button)
        nav.addWidget(self.fit_button)
        nav.addWidget(self.actual_size_button)
        nav.addWidget(self.zoom_out_button)
        nav.addWidget(self.zoom_in_button)
        nav.addWidget(self.zoom_label)
        nav.addWidget(self.viewer_background_combo)
        nav.addWidget(self.index_label)
        nav.addWidget(self.face_regions_button)
        nav.addWidget(self.capture_time_button)
        nav.addWidget(self.full_screen_button)
        image_layout.addWidget(self.viewer_actions)

        self.details_panel = QWidget(self.content_splitter)
        text_scale = int(getattr(theme_manager, "text_scale", 100)) if theme_manager is not None else 100
        if text_scale >= 200:
            details_minimum_width = 600
            details_maximum_width = 660
        elif text_scale >= 150:
            details_minimum_width = 480
            details_maximum_width = 580
        elif text_scale >= 125:
            details_minimum_width = 420
            details_maximum_width = 540
        else:
            details_minimum_width = 380
            details_maximum_width = 520
        self.details_panel.setMinimumWidth(details_minimum_width)
        self.details_panel.setMaximumWidth(details_maximum_width)
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
        self.name_label.setProperty("role", "section")
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

        self.operation_progress_bar = QProgressBar(self.details_panel)
        self.operation_progress_bar.setTextVisible(False)
        self.operation_progress_bar.setAccessibleName("Photo inspector operation progress")
        self.operation_progress_bar.hide()
        details_layout.addWidget(self.operation_progress_bar)

        self.face_editor_panel = QWidget(self.details_panel)
        face_editor_layout = QVBoxLayout(self.face_editor_panel)
        face_editor_layout.setContentsMargins(0, 0, 0, 0)
        face_editor_layout.setSpacing(6)

        self.face_editor_summary_label = QLabel("Face regions are available when face indexing is ready for this photo.")
        self.face_editor_summary_label.setProperty("role", "helper")
        self.face_editor_summary_label.setWordWrap(True)
        face_editor_layout.addWidget(self.face_editor_summary_label)

        self.face_actions = QWidget(self.face_editor_panel)
        face_action_grid = ResponsiveFlowLayout(self.face_actions, spacing=6)
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
        self.face_remove_button.setToolTip("Remove only the selected staged face regions.")
        self.face_duplicate_button.setToolTip("Duplicate one selected region so two nearby faces can be mapped separately.")
        self.face_split_button.setToolTip("Split one selected region into two staged face boxes.")
        self.face_auto_clean_button.setToolTip("Remove obvious low-quality detections from the staged draft.")
        self.face_reset_button.setToolTip("Discard staged face-region changes and restore the saved index.")
        self.face_remove_all_button.setToolTip("Stage removal of every face region from this photo.")
        face_action_buttons = [
            self.face_rescan_button,
            self.face_draw_button,
            self.face_undo_button,
            self.face_redo_button,
        ]
        for button in face_action_buttons:
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            face_action_grid.addWidget(button)
        self.face_save_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        face_action_grid.addWidget(self.face_save_button)
        self.face_more_button = QToolButton(self.face_editor_panel)
        self.face_more_button.setText("More face actions")
        self.face_more_button.setAccessibleName("Open additional face-region actions")
        self.face_more_button.setToolTip("Duplicate, split, clean, reset, or remove face regions.")
        self.face_more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.face_more_menu = QMenu(self.face_more_button)
        self.face_more_actions = {
            self.face_remove_button: self.face_more_menu.addAction("Remove selected faces"),
            self.face_duplicate_button: self.face_more_menu.addAction("Duplicate selected face"),
            self.face_split_button: self.face_more_menu.addAction("Split selected face"),
            self.face_auto_clean_button: self.face_more_menu.addAction("Auto-clean this image"),
            self.face_reset_button: self.face_more_menu.addAction("Discard face draft"),
            self.face_remove_all_button: self.face_more_menu.addAction("Remove all face regions"),
        }
        self.face_more_menu.insertSeparator(self.face_more_actions[self.face_remove_all_button])
        for button, action in self.face_more_actions.items():
            button.hide()
            action.setToolTip(button.toolTip())
            action.triggered.connect(button.click)
        self.face_more_menu.aboutToShow.connect(self._sync_face_overflow_actions)
        self.face_more_button.setMenu(self.face_more_menu)
        face_action_grid.addWidget(self.face_more_button)
        face_editor_layout.addWidget(self.face_actions)

        self.face_name_group = QGroupBox("Selected face regions", self.face_editor_panel)
        self.face_name_group.setToolTip(
            "Name, rename, or unlabel only the selected saved face regions. "
            "Other face regions in the photo are unchanged."
        )
        face_name_layout = QVBoxLayout(self.face_name_group)
        face_name_layout.setContentsMargins(8, 6, 8, 8)
        face_name_layout.setSpacing(6)
        face_name_row = QHBoxLayout()
        face_name_row.setContentsMargins(0, 0, 0, 0)
        face_name_label = QLabel("&Name", self.face_name_group)
        face_name_label.setMinimumWidth(42)
        self.face_name_input = EntityPicker(self.face_name_group, allow_create=True, entity_label="person name")
        self.face_name_input.setPlaceholderText("Name for the selected face region(s)")
        self.face_name_input.setClearButtonEnabled(True)
        self.face_name_input.setMinimumHeight(32)
        self.face_name_input.setToolTip(
            "Enter the name to apply to the selected saved face regions. "
            "An explicit name is saved even when automatic matching quality is low."
        )
        face_name_label.setBuddy(self.face_name_input)
        face_name_row.addWidget(face_name_label)
        face_name_row.addWidget(self.face_name_input, stretch=1)
        face_name_layout.addLayout(face_name_row)

        self.face_name_selected_button = QPushButton("Apply Name", self.face_name_group)
        self.face_rename_selected_button = QPushButton("Rename", self.face_name_group)
        self.face_unlabel_selected_button = QPushButton("Unlabel", self.face_name_group)
        self.face_name_selected_button.setAccessibleName("Name selected face regions")
        self.face_rename_selected_button.setAccessibleName("Rename selected face regions")
        self.face_unlabel_selected_button.setAccessibleName("Unlabel selected face regions")
        self.face_name_selected_button.setToolTip("Write the entered name to every selected saved face region and its XMP/EXIF metadata.")
        self.face_rename_selected_button.setToolTip("Choose a current name from the selected regions, then rename only those regions.")
        self.face_unlabel_selected_button.setToolTip("Choose a current name from the selected regions, then remove it only from those regions.")
        self.face_name_actions = QWidget(self.face_name_group)
        face_name_actions = ResponsiveFlowLayout(self.face_name_actions, spacing=6)
        for button in (
            (self.face_name_selected_button, self.face_rename_selected_button, self.face_unlabel_selected_button)
        ):
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
            button.setMinimumWidth(100)
            button.setMinimumHeight(32)
            face_name_actions.addWidget(button)
        face_name_layout.addWidget(self.face_name_actions)
        face_editor_layout.addWidget(self.face_name_group)

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
        self.metadata_summary_label.setProperty("role", "helper")
        self.metadata_summary_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.metadata_summary_label.setWordWrap(True)
        self.metadata_summary_label.setTextFormat(Qt.TextFormat.RichText)
        self.metadata_summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        details_layout.addWidget(self.metadata_summary_label)

        self.metadata_editor_group = QGroupBox("Edit photo metadata", self.details_panel)
        self.metadata_editor_group.setToolTip(
            "Save keeps these curated values in a reversible ClusterLens sidecar. "
            "Embed into original is a separate, journaled action for supported JPEG and TIFF files."
        )
        metadata_form = QFormLayout(self.metadata_editor_group)
        metadata_form.setContentsMargins(8, 6, 8, 8)
        metadata_form.setSpacing(6)
        self.metadata_title_input = QLineEdit(self.metadata_editor_group)
        self.metadata_description_input = QPlainTextEdit(self.metadata_editor_group)
        self.metadata_description_input.setTabChangesFocus(True)
        self.metadata_description_input.setMaximumHeight(62)
        self.metadata_rating_input = QSpinBox(self.metadata_editor_group)
        self.metadata_rating_input.setRange(0, 5)
        self.metadata_tags_input = QLineEdit(self.metadata_editor_group)
        self.metadata_creator_input = QLineEdit(self.metadata_editor_group)
        self.metadata_copyright_input = QLineEdit(self.metadata_editor_group)
        self.metadata_captured_at_input = QLineEdit(self.metadata_editor_group)
        self.metadata_location_input = QLineEdit(self.metadata_editor_group)
        self.metadata_custom_input = QPlainTextEdit(self.metadata_editor_group)
        self.metadata_custom_input.setTabChangesFocus(True)
        self.metadata_custom_input.setMaximumHeight(56)
        self.metadata_tags_input.setPlaceholderText("family, travel, portrait")
        self.metadata_captured_at_input.setPlaceholderText("YYYY-MM-DD HH:MM:SS")
        self.metadata_custom_input.setPlaceholderText("key: value (one per line)")
        for widget, accessible_name in (
            (self.metadata_title_input, "Photo title"),
            (self.metadata_description_input, "Photo description"),
            (self.metadata_rating_input, "Photo rating"),
            (self.metadata_tags_input, "Photo keywords"),
            (self.metadata_creator_input, "Photo creator"),
            (self.metadata_copyright_input, "Photo copyright"),
            (self.metadata_captured_at_input, "Photo capture date and time"),
            (self.metadata_location_input, "Photo location"),
            (self.metadata_custom_input, "Advanced photo metadata"),
        ):
            widget.setAccessibleName(accessible_name)
        for widget, tooltip in (
            (self.metadata_title_input, "A short photo title."),
            (self.metadata_description_input, "A longer caption or description."),
            (self.metadata_rating_input, "Your 0–5 rating, saved safely in the sidecar."),
            (self.metadata_tags_input, "Comma-separated keywords; existing tags remain unchanged until a dedicated tag action is used."),
            (self.metadata_creator_input, "Creator/artist credit."),
            (self.metadata_copyright_input, "Copyright notice."),
            (
                self.metadata_captured_at_input,
                "Capture date and time. Use ISO format, for example 2026-09-16 14:30:00. "
                "A valid value from 1991 through the current year is a manual Timeline correction after Refresh dates.",
            ),
            (self.metadata_location_input, "A human-readable location."),
            (self.metadata_custom_input, "Advanced textual fields. Unsafe binary and maker-note EXIF fields are intentionally read-only."),
        ):
            widget.setToolTip(tooltip)
        metadata_form.addRow("Title", self.metadata_title_input)
        metadata_form.addRow("Description", self.metadata_description_input)
        metadata_form.addRow("Rating", self.metadata_rating_input)
        metadata_form.addRow("Keywords", self.metadata_tags_input)
        metadata_form.addRow("Creator", self.metadata_creator_input)
        metadata_form.addRow("Copyright", self.metadata_copyright_input)
        metadata_form.addRow("Capture time", self.metadata_captured_at_input)
        metadata_form.addRow("Location", self.metadata_location_input)
        metadata_form.addRow("Advanced", self.metadata_custom_input)
        self.metadata_actions = QWidget(self.metadata_editor_group)
        metadata_actions = ResponsiveFlowLayout(self.metadata_actions, spacing=6)
        self.metadata_save_button = QPushButton("Save sidecar", self.metadata_editor_group)
        self.metadata_discard_button = QPushButton("Discard changes", self.metadata_editor_group)
        self.metadata_embed_button = QPushButton("Embed into original", self.metadata_editor_group)
        self.rename_file_button = QPushButton("Rename file…", self.metadata_editor_group)
        self.metadata_help_button = HelpIconButton(
            "Save sidecar is reversible and leaves the source photo untouched. "
            "Embed into original writes only supported textual fields to JPEG/TIFF after confirmation. "
            "Raw maker-note EXIF values remain read-only for safety.",
            self.metadata_editor_group,
            help_key="photo_metadata",
        )
        self.metadata_save_button.setToolTip("Save a reversible metadata sidecar without modifying the photo.")
        self.metadata_discard_button.setToolTip("Restore every staged field to its last loaded or saved value.")
        self.metadata_embed_button.setToolTip("Explicitly embed supported fields into this JPEG/TIFF original. The operation is journaled and cancellable before the atomic write.")
        self.rename_file_button.setToolTip("Preview a collision-safe rename for this photo. No filename changes until confirmation.")
        apply_icon(self.metadata_save_button, "metadata")
        apply_icon(self.metadata_embed_button, "metadata")
        apply_icon(self.rename_file_button, "rename")
        metadata_actions.addWidget(self.metadata_save_button)
        metadata_actions.addWidget(self.metadata_discard_button)
        metadata_actions.addWidget(self.metadata_embed_button)
        metadata_actions.addWidget(self.rename_file_button)
        metadata_actions.addWidget(self.metadata_help_button)
        metadata_form.addRow(self.metadata_actions)
        details_layout.addWidget(self.metadata_editor_group)

        self.info_text = QTextBrowser()
        self.info_text.setReadOnly(True)
        self.info_text.setTabChangesFocus(True)
        self.info_text.setAccessibleName("Photo EXIF metadata")
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

        # The inspector used to stack every editor into one tall rail. Keep
        # the image dominant and make each kind of information predictable.
        for widget in (
            self.identity_panel,
            self.face_editor_panel,
            self.metadata_summary_label,
            self.metadata_editor_group,
            self.info_text,
        ):
            details_layout.removeWidget(widget)
        self.inspector_tabs = QTabWidget(self.details_panel)
        self.inspector_tabs.setTabBar(KeyboardTabBar(self.inspector_tabs))
        self.inspector_tabs.setAccessibleName("Photo inspector panels")
        self.inspector_tabs.tabBar().setAccessibleName("Photo inspector panels")
        self.info_page = QWidget(self.inspector_tabs)
        info_layout = QVBoxLayout(self.info_page)
        info_layout.setContentsMargins(0, 8, 0, 0)
        info_layout.addWidget(self.identity_panel)
        info_layout.addWidget(self.metadata_summary_label)
        info_layout.addStretch(1)
        self.inspector_tabs.addTab(self.info_page, "Info")

        self.people_page = QScrollArea(self.inspector_tabs)
        self.people_page.setWidgetResizable(True)
        self.people_page.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        people_content = QWidget(self.people_page)
        people_layout = QVBoxLayout(people_content)
        people_layout.setContentsMargins(0, 8, 0, 0)
        self.people_unavailable_label = QLabel(
            "Face regions will appear here when face tools are ready for this photo.",
            people_content,
        )
        self.people_unavailable_label.setWordWrap(True)
        people_layout.addWidget(self.people_unavailable_label)
        people_layout.addWidget(self.face_editor_panel, stretch=1)
        self.people_page.setWidget(people_content)
        self.inspector_tabs.addTab(self.people_page, "People")

        self.metadata_page = QScrollArea(self.inspector_tabs)
        self.metadata_page.setWidgetResizable(True)
        self.metadata_page.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        metadata_content = QWidget(self.metadata_page)
        metadata_layout = QVBoxLayout(metadata_content)
        metadata_layout.setContentsMargins(0, 8, 0, 0)
        metadata_layout.addWidget(self.metadata_editor_group)
        metadata_layout.addStretch(1)
        self.metadata_page.setWidget(metadata_content)
        self.inspector_tabs.addTab(self.metadata_page, "Metadata")

        self.exif_page = QWidget(self.inspector_tabs)
        exif_layout = QVBoxLayout(self.exif_page)
        exif_layout.setContentsMargins(0, 8, 0, 0)
        exif_search_row = QHBoxLayout()
        self.exif_search_field = QLineEdit(self.exif_page)
        self.exif_search_field.setPlaceholderText("Find an EXIF or metadata field")
        self.exif_search_field.setAccessibleName("Search complete EXIF metadata")
        self.exif_find_button = QPushButton("Find next", self.exif_page)
        self.exif_find_button.setToolTip("Find the next matching field or value in the complete metadata table.")
        self.exif_search_field.returnPressed.connect(self._find_next_exif_value)
        self.exif_find_button.clicked.connect(self._find_next_exif_value)
        exif_search_row.addWidget(self.exif_search_field, stretch=1)
        exif_search_row.addWidget(self.exif_find_button)
        exif_layout.addLayout(exif_search_row)
        exif_layout.addWidget(self.info_text, stretch=1)
        self.inspector_tabs.addTab(self.exif_page, "EXIF")
        details_layout.addWidget(self.inspector_tabs, stretch=1)
        inspector_focus_order = (
            self.inspector_tabs.tabBar(),
            self.prev_button,
            self.next_button,
            self.fit_button,
            self.actual_size_button,
            self.zoom_out_button,
            self.zoom_in_button,
            self.viewer_background_combo,
            self.face_regions_button,
            self.capture_time_button,
            self.full_screen_button,
        )
        for current, following in zip(inspector_focus_order, inspector_focus_order[1:]):
            QWidget.setTabOrder(current, following)
        metadata_focus_order = (
            self.metadata_custom_input,
            self.metadata_save_button,
            self.metadata_discard_button,
            self.metadata_embed_button,
            self.rename_file_button,
            self.metadata_help_button,
        )
        for current, following in zip(metadata_focus_order, metadata_focus_order[1:]):
            QWidget.setTabOrder(current, following)

        self.content_splitter.addWidget(self.image_panel)
        self.content_splitter.addWidget(self.details_panel)
        self.content_splitter.setStretchFactor(0, 1)
        self.content_splitter.setStretchFactor(1, 0)
        self.content_splitter.setSizes([780, 420])
        self.main_layout.addWidget(self.content_splitter, stretch=1)

        self.prev_button.clicked.connect(lambda: self._step(-1))
        self.next_button.clicked.connect(lambda: self._step(1))
        self.fit_button.clicked.connect(self.preview_view.fit_to_window)
        self.actual_size_button.clicked.connect(self.preview_view.actual_size)
        self.zoom_out_button.clicked.connect(self.preview_view.zoom_out)
        self.zoom_in_button.clicked.connect(self.preview_view.zoom_in)
        self.full_screen_button.clicked.connect(self._toggle_full_screen)
        self.face_regions_button.toggled.connect(self.preview_view.set_face_boxes_visible)
        self.capture_time_button.clicked.connect(self._focus_capture_time_editor)
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
        self.face_name_selected_button.clicked.connect(self._name_selected_faces)
        self.face_rename_selected_button.clicked.connect(self._rename_selected_faces)
        self.face_unlabel_selected_button.clicked.connect(self._unlabel_selected_faces)
        self.metadata_save_button.clicked.connect(self._save_metadata_sidecar)
        self.metadata_discard_button.clicked.connect(self._discard_metadata_changes)
        self.metadata_embed_button.clicked.connect(self._embed_metadata_into_original)
        self.rename_file_button.clicked.connect(self._rename_current_file)
        for field in (
            self.metadata_title_input,
            self.metadata_tags_input,
            self.metadata_creator_input,
            self.metadata_copyright_input,
            self.metadata_captured_at_input,
            self.metadata_location_input,
        ):
            field.textChanged.connect(self._metadata_inputs_changed)
        self.metadata_description_input.textChanged.connect(self._metadata_inputs_changed)
        self.metadata_custom_input.textChanged.connect(self._metadata_inputs_changed)
        self.metadata_rating_input.valueChanged.connect(self._metadata_inputs_changed)
        selection_model = self.image_faces_list.selectionModel()
        if selection_model is not None:
            selection_model.selectionChanged.connect(lambda *_args: self._on_face_draft_selection_changed())
        self.preview_view.face_box_clicked.connect(self._on_preview_face_box_clicked)
        self.preview_view.face_box_drawn.connect(self._on_preview_face_box_drawn)
        self.preview_view.face_box_moved.connect(self._on_preview_face_box_moved)
        self.preview_view.face_box_resized.connect(self._on_preview_face_box_resized)
        self._install_shortcuts()
        self._apply_display_mode()
        self._request_face_name_suggestions()

        self._update_nav_state()
        if self._image_paths:
            self.load_metadata(self._image_paths[self._index], dict(self._base_context))

    def load_metadata(self, image_path: str, context: dict[str, object]) -> None:
        self._request_id += 1
        request_id = self._request_id
        self._full_res_loaded = False
        self._full_res_requested = False
        self._update_identity(image_path)
        self._update_nav_state()
        self._update_window_title(image_path)

        if self._active_job is not None:
            self._retain_async_refs(self._active_job, self._active_thread)
            self._cancel_operation_job(self._active_job)
            self._active_job = None
            self._active_thread = None
        if self._preview_job is not None:
            self._retain_async_refs(self._preview_job, self._preview_thread)
            self._cancel_operation_job(self._preview_job)
            self._preview_job = None
            self._preview_thread = None
        if self._face_draft_job is not None:
            self._retain_async_refs(self._face_draft_job, self._face_draft_thread)
            self._cancel_operation_job(self._face_draft_job)
            self._face_draft_job = None
            self._face_draft_thread = None
        if self._face_thumbnail_job is not None:
            self._retain_async_refs(self._face_thumbnail_job, self._face_thumbnail_thread)
            self._cancel_operation_job(self._face_thumbnail_job)
            self._face_thumbnail_job = None
            self._face_thumbnail_thread = None
        self._face_thumbnail_request_signature = None
        if self._prefetch_job is not None:
            self._retain_async_refs(self._prefetch_job, self._prefetch_thread)
            self._cancel_operation_job(self._prefetch_job)
            self._prefetch_job = None
            self._prefetch_thread = None

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
        self.preview_view.set_adaptive_backdrop(None)
        self.preview_view.set_pixmap(None)
        self._set_loading_state(face_message=self._context_face_message(context))
        self._load_face_editor_state(image_path, self._current_context)

        self._load_preview(image_path, request_id=request_id, full_res=False)
        # Curated metadata is available in every inspector.  Advanced mode
        # only controls the raw diagnostic table below it.
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
            draft = self.photo_edit_service.load_draft(image_path)
            raise_if_cancelled(cancel_check)
            return metadata, draft

        job = AsyncJob(_run)

        def _on_completed(result) -> None:
            if request_id != self._request_id:
                return
            metadata, draft = result
            self.metadata_summary_label.setText(self._render_metadata_summary(metadata, draft))
            self.info_text.setHtml(self._render_metadata(metadata))
            self._populate_metadata_editor(draft)
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
        self._start_operation_worker(
            job,
            "Loading photo metadata",
            foreground=False,
            job_attribute="_active_job",
            thread_attribute="_active_thread",
            source_reads=(image_path, str(self.photo_edit_service._sidecar_path(image_path))),
        )

    def _populate_metadata_editor(self, draft: PhotoEditDraft, *, update_baseline: bool = True) -> None:
        self._metadata_populating = True
        try:
            self.metadata_title_input.setText(draft.title)
            self.metadata_description_input.setPlainText(draft.description)
            self.metadata_rating_input.setValue(int(draft.rating))
            self.metadata_tags_input.setText(", ".join(draft.tags))
            self.metadata_creator_input.setText(draft.creator)
            self.metadata_copyright_input.setText(draft.copyright)
            self.metadata_captured_at_input.setText(draft.captured_at)
            self.metadata_location_input.setText(draft.location)
            self.metadata_custom_input.setPlainText(
                "\n".join(f"{key}: {value}" for key, value in sorted(draft.custom_fields.items()))
            )
        finally:
            self._metadata_populating = False
        if update_baseline:
            self._metadata_baseline_draft = draft
        path = self._current_editable_path()
        editable = bool(self._allow_metadata_edit and path)
        self.metadata_editor_group.setEnabled(editable)
        self.capture_time_button.setEnabled(editable)
        self.metadata_embed_button.setEnabled(editable and self.photo_edit_service.can_embed(path))
        self.rename_file_button.setEnabled(bool(self._allow_file_rename and callable(self.rename_current_callback) and path))
        self._metadata_inputs_changed()

    def _metadata_inputs_changed(self, *_args) -> None:
        if self._metadata_populating:
            return
        dirty = self._metadata_draft_from_inputs() != self._metadata_baseline_draft
        self.metadata_save_button.setVisible(dirty)
        self.metadata_discard_button.setVisible(dirty)

    def _metadata_is_dirty(self) -> bool:
        return self._metadata_draft_from_inputs() != self._metadata_baseline_draft

    def _discard_metadata_changes(self) -> None:
        self._populate_metadata_editor(self._metadata_baseline_draft, update_baseline=False)
        self._set_state_text("Discarded staged metadata changes. No source or sidecar file was changed.")

    def _confirm_leave_metadata_changes(self) -> bool:
        if not self._metadata_is_dirty():
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Unsaved metadata changes")
        box.setText("This photo has staged metadata changes.")
        box.setInformativeText(
            "Save them to the reversible sidecar, discard them, or stay on this photo."
        )
        box.setStandardButtons(
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QMessageBox.StandardButton.Save)
        choice = box.exec()
        if choice == QMessageBox.StandardButton.Discard:
            self._discard_metadata_changes()
            return True
        if choice == QMessageBox.StandardButton.Save:
            self._save_metadata_sidecar()
            self._set_state_text("Saving metadata. Navigate or close again after the Job completes.")
        return False

    def _discard_face_changes(self) -> None:
        image_path = self._current_editable_path()
        if not image_path:
            return
        self._face_drafts_by_path[image_path] = self._clone_face_drafts(
            self._face_original_drafts_by_path.get(image_path, [])
        )
        self._face_editor_dirty_paths.discard(image_path)
        self._face_undo_by_path[image_path] = []
        self._face_redo_by_path[image_path] = []
        self.preview_view.set_draw_mode(False)
        self._publish_face_drafts(image_path)
        self._refresh_face_editor_ui(image_path)
        self._set_state_text("Discarded staged face-region changes. The saved face index was not changed.")

    def _confirm_leave_face_changes(self) -> bool:
        image_path = self._current_editable_path()
        if not image_path or image_path not in self._face_editor_dirty_paths:
            return True
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Unsaved face-region changes")
        box.setText("This photo has staged face-region changes.")
        box.setInformativeText("Save them to the face index, discard them, or stay on this photo.")
        box.setStandardButtons(
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel
        )
        box.setDefaultButton(QMessageBox.StandardButton.Save)
        choice = box.exec()
        if choice == QMessageBox.StandardButton.Discard:
            self._discard_face_changes()
            return True
        if choice == QMessageBox.StandardButton.Save:
            self._save_face_edits()
            self._set_state_text("Saving face regions. Navigate or close again after the Job completes.")
        return False

    def _confirm_leave_changes(self) -> bool:
        return self._confirm_leave_metadata_changes() and self._confirm_leave_face_changes()

    def _metadata_draft_from_inputs(self) -> PhotoEditDraft:
        custom: dict[str, str] = {}
        for line in self.metadata_custom_input.toPlainText().splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            normalized_key = key.strip()
            normalized_value = value.strip()
            if normalized_key and normalized_value:
                custom[normalized_key] = normalized_value
        tags = tuple(
            dict.fromkeys(
                value.strip()
                for value in self.metadata_tags_input.text().split(",")
                if value.strip()
            )
        )
        return PhotoEditDraft(
            title=self.metadata_title_input.text().strip(),
            description=self.metadata_description_input.toPlainText().strip(),
            rating=int(self.metadata_rating_input.value()),
            tags=tags,
            creator=self.metadata_creator_input.text().strip(),
            copyright=self.metadata_copyright_input.text().strip(),
            captured_at=self.metadata_captured_at_input.text().strip(),
            location=self.metadata_location_input.text().strip(),
            custom_fields=custom,
        )

    def _start_metadata_edit_job(self, label: str, fn, on_completed) -> None:
        if self._metadata_edit_job is not None:
            return

        def _run(progress, cancel_check):
            raise_if_cancelled(cancel_check)
            result = fn(progress, cancel_check)
            raise_if_cancelled(cancel_check)
            return result

        job = AsyncJob(_run)
        self.metadata_editor_group.setEnabled(False)

        def _done(result) -> None:
            self._metadata_edit_job = None
            self._metadata_edit_thread = None
            self._populate_metadata_editor(self._metadata_draft_from_inputs())
            on_completed(result)

        def _failed(message: str) -> None:
            self._metadata_edit_job = None
            self._metadata_edit_thread = None
            self._populate_metadata_editor(self._metadata_draft_from_inputs(), update_baseline=False)
            self._set_state_text(f"Metadata operation failed: {message}")

        def _cancelled() -> None:
            self._metadata_edit_job = None
            self._metadata_edit_thread = None
            self._populate_metadata_editor(self._metadata_draft_from_inputs(), update_baseline=False)
            self._set_state_text("Metadata operation cancelled. No unfinished write was applied.")

        job.completed.connect(_done)
        job.failed.connect(_failed)
        job.cancelled.connect(_cancelled)
        path = self._current_editable_path()
        sidecar = str(self.photo_edit_service._sidecar_path(path)) if path else ""
        source_writes = (path,) if label == "Embedding photo metadata" else (sidecar,)
        self._start_operation_worker(
            job,
            label,
            foreground=True,
            job_attribute="_metadata_edit_job",
            thread_attribute="_metadata_edit_thread",
            source_writes=tuple(value for value in source_writes if value),
        )

    def _save_metadata_sidecar(self) -> None:
        path = self._current_editable_path()
        if not path or not self._allow_metadata_edit:
            return
        draft = self._metadata_draft_from_inputs()

        def _run(progress, cancel_check):
            progress(-1, "Saving reversible metadata sidecar…")
            raise_if_cancelled(cancel_check)
            return self.photo_edit_service.save_draft(path, draft)

        self._start_metadata_edit_job(
            "Saving photo metadata sidecar",
            _run,
            lambda sidecar: self._set_state_text(
                f"Saved reversible metadata sidecar: {Path(str(sidecar)).name}"
                + (
                    ". Refresh Timeline dates to use this capture-time correction."
                    if draft.captured_at
                    else ""
                )
            ),
        )

    def _focus_capture_time_editor(self) -> None:
        """Make the common manual Timeline correction direct in every viewer route."""

        if not self.capture_time_button.isEnabled():
            self._set_state_text("Capture time cannot be changed while this photo is read-only.")
            return
        self.inspector_tabs.setCurrentWidget(self.metadata_page)
        self.metadata_captured_at_input.setFocus(Qt.FocusReason.ShortcutFocusReason)
        self.metadata_captured_at_input.selectAll()
        self._set_state_text(
            "Enter YYYY-MM-DD HH:MM:SS, then choose Save sidecar. "
            "In Timeline, Refresh dates applies a valid 1991–current-year value."
        )

    def _embed_metadata_into_original(self) -> None:
        path = self._current_editable_path()
        if not path or not self._allow_metadata_edit or not self.photo_edit_service.can_embed(path):
            return
        if not confirmBox(
            "Embed metadata into original?",
            "This writes supported textual fields into the original image. Your sidecar remains available, and the operation is recorded in Safety & Recovery.",
            parent=self,
        ):
            return
        draft = self._metadata_draft_from_inputs()

        def _run(progress, cancel_check):
            return self.photo_edit_service.embed_draft(path, draft, progress_callback=progress, cancel_check=cancel_check)

        self._start_metadata_edit_job(
            "Embedding photo metadata",
            _run,
            lambda result: self._set_state_text(
                f"Embedded metadata for {len(getattr(result, 'affected_paths', []) or [])} photo(s)."
            ),
        )

    def _rename_current_file(self) -> None:
        path = self._current_editable_path()
        if not path or not self._allow_file_rename or not callable(self.rename_current_callback):
            return
        self.rename_current_callback(path)

    def _request_face_name_suggestions(self) -> None:
        """Populate the shared name picker without reading SQLite on the UI thread."""
        service = self.face_service
        loader = getattr(service, "list_known_person_names", None) if service is not None else None
        if not callable(loader) or self._face_name_suggestion_job is not None:
            return

        def _run(_progress, cancel_check):
            raise_if_cancelled(cancel_check)
            try:
                names = loader(cancel_check=cancel_check)
            except TypeError:
                names = loader()
            raise_if_cancelled(cancel_check)
            return [str(name or "").strip() for name in list(names or ())]

        job = AsyncJob(_run)

        def _completed(names) -> None:
            self.face_name_input.set_choices(names)

        job.completed.connect(_completed)
        self._start_operation_worker(
            job,
            "Loading saved person names",
            foreground=False,
            job_attribute="_face_name_suggestion_job",
            thread_attribute="_face_name_suggestion_thread",
            data_home_read=True,
        )

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
            backdrop = None if full_res else adaptive_neutral_backdrop(image)
            return image, backdrop

        job = AsyncJob(_run)
        label = "Loading full-resolution preview" if full_res else "Loading photo preview"

        def _on_completed(result) -> None:
            if request_id != self._request_id:
                return
            image, backdrop = result
            if not full_res:
                self.preview_view.set_adaptive_backdrop(backdrop)
            pixmap = QPixmap.fromImage(image)
            self.preview_view.set_pixmap(pixmap if not pixmap.isNull() else None, preserve_zoom=full_res)
            if full_res:
                self._full_res_loaded = True

        def _on_failed(_message: str) -> None:
            if request_id != self._request_id:
                return
            if full_res:
                self._full_res_requested = False
                return
            self.preview_view.set_pixmap(None)

        job.completed.connect(_on_completed)
        job.failed.connect(_on_failed)
        self._start_operation_worker(
            job,
            label,
            foreground=False,
            job_attribute="_preview_job",
            thread_attribute="_preview_thread",
            source_reads=(image_path,),
        )

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
        if self._face_draft_thread is thread:
            self._face_draft_thread = None
            if self._face_draft_job is job:
                self._face_draft_job = None
        if self._face_thumbnail_thread is thread:
            self._face_thumbnail_thread = None
            if self._face_thumbnail_job is job:
                self._face_thumbnail_job = None
        if self._prefetch_thread is thread:
            self._prefetch_thread = None
            if self._prefetch_job is job:
                self._prefetch_job = None
        if self._face_edit_thread is thread:
            self._face_edit_thread = None
            if self._face_edit_job is job:
                self._face_edit_job = None
        if self._face_name_suggestion_thread is thread:
            self._face_name_suggestion_thread = None
            if self._face_name_suggestion_job is job:
                self._face_name_suggestion_job = None
        if self._metadata_edit_thread is thread:
            self._metadata_edit_thread = None
            if self._metadata_edit_job is job:
                self._metadata_edit_job = None

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
        _ = context
        return bool(self._allow_face_edit)

    def _load_face_editor_state(self, image_path: str, context: dict[str, object]) -> None:
        enabled = self._face_edit_enabled_for_context(context)
        self.face_editor_panel.setVisible(enabled)
        self.people_unavailable_label.setVisible(not enabled)
        if not enabled:
            self.preview_view.set_selected_face_indexes(())
            self.preview_view.set_draw_mode(False)
            return
        if self.face_service is None:
            self.face_editor_summary_label.setText(
                "Face-region tools are not ready for this workspace yet. Existing regions remain visible on the photo."
            )
            for button in (
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
                self.face_name_selected_button,
                self.face_rename_selected_button,
                self.face_unlabel_selected_button,
            ):
                button.setEnabled(False)
            return
        for button in (
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
            self.face_name_selected_button,
            self.face_rename_selected_button,
            self.face_unlabel_selected_button,
        ):
            button.setEnabled(True)
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
                self._face_original_drafts_by_path[path] = list(drafts)
                self._face_drafts_by_path[path] = list(drafts)
                if shared_dirty:
                    self._face_editor_dirty_paths.add(path)
                else:
                    self._face_editor_dirty_paths.discard(path)
            else:
                drafts = self._drafts_from_context(context)
                self._face_original_drafts_by_path[path] = list(drafts)
                self._face_drafts_by_path[path] = list(drafts)
        self._refresh_face_editor_ui(path)
        self.face_editor_summary_label.setText(f"Loading saved face regions for {Path(path).name}…")
        self._load_indexed_face_drafts_async(path, context, request_id=self._request_id)

    @staticmethod
    def _drafts_from_context(context: dict[str, object]) -> list[EditableFaceDraft]:
        indexed = context.get("indexed_faces")
        if not isinstance(indexed, dict) or not isinstance(indexed.get("faces"), list):
            return []
        drafts: list[EditableFaceDraft] = []
        for index, face in enumerate(indexed.get("faces", [])):
            if not isinstance(face, dict):
                continue
            bbox = face.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            try:
                drafts.append(
                    EditableFaceDraft(
                        bbox=tuple(int(value) for value in bbox),
                        confidence=float(face.get("confidence", 1.0) or 1.0),
                        source="indexed",
                        person_name=str(face.get("person_name") or ""),
                        face_index=int(face.get("face_index", index)),
                    )
                )
            except (TypeError, ValueError):
                continue
        return drafts

    def _load_indexed_face_drafts_async(self, image_path: str, context: dict[str, object], *, request_id: int) -> None:
        """Read the database and XMP face regions away from the Qt UI thread."""
        if self.face_service is None:
            return

        def _run(progress, cancel_check):
            progress(-1, "Reading saved face regions…")
            raise_if_cancelled(cancel_check)
            drafts = self._load_indexed_face_drafts(image_path, context)
            raise_if_cancelled(cancel_check)
            return drafts

        job = AsyncJob(_run)

        def _completed(drafts: object) -> None:
            if request_id != self._request_id or image_path != self._current_editable_path():
                return
            # A user may have edited the context drafts while the database/XMP
            # read was in flight.  Never replace that edit or its selection
            # with a late passive refresh.
            if image_path in self._face_editor_dirty_paths:
                return
            loaded = list(drafts or []) or self._drafts_from_context(context)
            self._face_original_drafts_by_path[image_path] = list(loaded)
            self._face_drafts_by_path[image_path] = list(loaded)
            self._refresh_face_editor_ui(image_path)

        def _failed(_message: str) -> None:
            if request_id == self._request_id and image_path == self._current_editable_path():
                self._refresh_face_editor_ui(image_path)

        job.completed.connect(_completed)
        job.failed.connect(_failed)
        self._start_operation_worker(
            job,
            "Reading saved face regions",
            foreground=False,
            job_attribute="_face_draft_job",
            thread_attribute="_face_draft_thread",
            source_reads=(image_path,),
            data_home_read=True,
        )

    def _load_indexed_face_drafts(self, image_path: str, context: dict[str, object]) -> list[EditableFaceDraft]:
        records: list[IndexedFaceRecord] = []
        if self.face_service is not None:
            try:
                records = list(self.face_service.load_image_faces(image_path, include_tiny_faces=True))
            except Exception:
                records = []
        drafts: list[EditableFaceDraft] = []
        metadata_regions = []
        try:
            from app.services.face_region_metadata import FaceRegionMetadataService

            with Image.open(image_path) as image:
                image_size = image.size
            region_service = FaceRegionMetadataService()
            metadata_regions = list(region_service.read(image_path).regions)
        except Exception:
            region_service = None
            image_size = (1, 1)
        if records:
            for record in records:
                metadata_name = ""
                if region_service is not None:
                    candidate = region_service.normalized_region_for_bbox(record.face_bbox, image_size)
                    matched = region_service.best_matching_region(metadata_regions, candidate)
                    if matched is not None:
                        metadata_name = str(matched.name or "").strip()
                drafts.append(
                    EditableFaceDraft(
                        bbox=tuple(int(value) for value in record.face_bbox),
                        confidence=float(record.face_confidence or 1.0),
                        source="indexed",
                        person_name=str(record.person_name or metadata_name or ""),
                        face_index=int(record.face_index),
                    )
                )
            if region_service is not None:
                indexed_regions = [region_service.normalized_region_for_bbox(record.face_bbox, image_size) for record in records]
                for region in metadata_regions:
                    if region_service.best_matching_region(indexed_regions, region) is None:
                        drafts.append(
                            EditableFaceDraft(
                                bbox=region_service.bbox_for_region(region, image_size),
                                confidence=1.0,
                                source="xmp",
                                person_name=str(region.name or ""),
                                face_index=-1,
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
        if not drafts and region_service is not None:
            for region in metadata_regions:
                drafts.append(
                    EditableFaceDraft(
                        bbox=region_service.bbox_for_region(region, image_size),
                        confidence=1.0,
                        source="xmp",
                        person_name=str(region.name or ""),
                        face_index=-1,
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
                    icon=self._face_thumbnail_cache.get((image_path, tuple(draft.bbox))),
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
        selected_saved = [draft for index, draft in enumerate(drafts) if index in selected_indexes and int(draft.face_index) >= 0]
        self.face_name_selected_button.setEnabled(bool(selected_saved))
        self.face_rename_selected_button.setEnabled(bool(selected_saved and any(str(draft.person_name or "").strip() for draft in selected_saved)))
        self.face_unlabel_selected_button.setEnabled(bool(selected_saved and any(str(draft.person_name or "").strip() for draft in selected_saved)))
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
        self._load_face_thumbnail_icons_async(image_path, drafts, request_id=self._request_id)
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
            self.face_name_selected_button.setEnabled(False)
            self.face_rename_selected_button.setEnabled(False)
            self.face_unlabel_selected_button.setEnabled(False)
            self.face_selected_details_label.setText("Selected face: none")
            return
        first_index = int(selected[0])
        if first_index < 0 or first_index >= len(drafts):
            self.face_name_selected_button.setEnabled(False)
            self.face_rename_selected_button.setEnabled(False)
            self.face_unlabel_selected_button.setEnabled(False)
            self.face_selected_details_label.setText("Selected face: none")
            return
        draft = drafts[first_index]
        selected_drafts = [drafts[index] for index in selected if 0 <= index < len(drafts)]
        saved_drafts = [item for item in selected_drafts if int(item.face_index) >= 0]
        named_drafts = [item for item in saved_drafts if str(item.person_name or "").strip()]
        self.face_name_selected_button.setEnabled(bool(saved_drafts))
        self.face_rename_selected_button.setEnabled(bool(named_drafts))
        self.face_unlabel_selected_button.setEnabled(bool(named_drafts))
        if len(selected_drafts) == 1 and str(draft.person_name or "").strip():
            self.face_name_input.setText(str(draft.person_name))
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
        self._update_selected_face_details(self._current_editable_path())

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

    def _load_face_thumbnail_icons_async(
        self,
        image_path: str,
        drafts: list[EditableFaceDraft],
        *,
        request_id: int,
    ) -> None:
        """Decode face crops in a worker so face-list refresh never blocks paint."""
        missing = [
            tuple(draft.bbox)
            for draft in drafts
            if (image_path, tuple(draft.bbox)) not in self._face_thumbnail_cache
        ]
        if not missing:
            return
        signature = (int(request_id), str(image_path), tuple(missing))
        if signature == self._face_thumbnail_request_signature:
            return
        if self._face_thumbnail_job is not None:
            self._retain_async_refs(self._face_thumbnail_job, self._face_thumbnail_thread)
            self._cancel_operation_job(self._face_thumbnail_job)
        self._face_thumbnail_request_signature = signature

        def _run(progress, cancel_check):
            progress(-1, "Loading face-region previews…")
            reader = QImageReader(image_path)
            reader.setAutoTransform(True)
            image = reader.read()
            if image.isNull():
                raise ValueError(reader.errorString() or "Could not decode face-region previews")
            thumb_size = 72
            results: list[tuple[tuple[int, int, int, int], QImage]] = []
            for index, bbox in enumerate(missing, start=1):
                raise_if_cancelled(cancel_check)
                x1, y1, x2, y2 = [int(value) for value in bbox]
                pad_x = max(8, (x2 - x1) // 6)
                pad_y = max(8, (y2 - y1) // 6)
                left = max(0, x1 - pad_x)
                top = max(0, y1 - pad_y)
                right = min(image.width(), x2 + pad_x)
                bottom = min(image.height(), y2 + pad_y)
                if right <= left or bottom <= top:
                    continue
                crop = image.copy(QRect(left, top, right - left, bottom - top))
                scaled = crop.scaled(
                    thumb_size,
                    thumb_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                canvas = QImage(thumb_size, thumb_size, QImage.Format.Format_ARGB32_Premultiplied)
                canvas.fill(Qt.GlobalColor.black)
                painter = QPainter(canvas)
                painter.drawImage((thumb_size - scaled.width()) // 2, (thumb_size - scaled.height()) // 2, scaled)
                painter.end()
                results.append((bbox, canvas))
                progress(int(index * 100 / max(1, len(missing))), f"Loading face-region previews {index}/{len(missing)}")
            return results

        job = AsyncJob(_run)

        def _completed(payload: object) -> None:
            if request_id != self._request_id or image_path != self._current_editable_path():
                return
            for bbox, image in list(payload or []):
                if isinstance(image, QImage) and not image.isNull():
                    self._face_thumbnail_cache[(image_path, tuple(bbox))] = QIcon(QPixmap.fromImage(image))
            for row, draft in enumerate(list(self._face_drafts_by_path.get(image_path, []))):
                icon = self._face_thumbnail_cache.get((image_path, tuple(draft.bbox)))
                if icon is not None:
                    self.image_faces_model.set_item_icon(row, icon)

        job.completed.connect(_completed)
        self._start_operation_worker(
            job,
            "Loading face-region previews",
            foreground=False,
            job_attribute="_face_thumbnail_job",
            thread_attribute="_face_thumbnail_thread",
            source_reads=(image_path,),
        )

    def _set_face_edit_busy(self, busy: bool, message: str = "") -> None:
        for control in [
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
            self.face_name_input,
            self.face_name_selected_button,
            self.face_rename_selected_button,
            self.face_unlabel_selected_button,
            self.face_more_button,
        ]:
            control.setEnabled(not busy)
        if not busy:
            self._refresh_face_editor_ui(self._current_editable_path())
        if message:
            self.face_editor_summary_label.setText(message)

    def _sync_face_overflow_actions(self) -> None:
        for button, action in self.face_more_actions.items():
            action.setEnabled(button.isEnabled())

    def _start_face_edit_job(self, label: str, fn, on_completed) -> None:
        if self._face_edit_job is not None:
            try:
                if self._face_edit_thread is None or self._face_edit_thread.isRunning():
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
        image_path = self._current_editable_path()
        mutating = label != "Auto-scan current image"
        self._start_operation_worker(
            job,
            label,
            foreground=True,
            job_attribute="_face_edit_job",
            thread_attribute="_face_edit_thread",
            source_reads=(image_path,) if not mutating else (),
            source_writes=(image_path,) if mutating else (),
            data_home_write=mutating,
            model_cache_read=not mutating,
        )
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

    def _selected_saved_face_refs(self, *, person_name: str = "") -> list[tuple[str, int]]:
        image_path = self._current_editable_path()
        drafts = list(self._face_drafts_by_path.get(image_path, []))
        source = str(person_name or "").strip()
        refs: list[tuple[str, int]] = []
        for index in self._selected_face_draft_indexes():
            if not (0 <= index < len(drafts)):
                continue
            draft = drafts[index]
            if int(draft.face_index) < 0:
                continue
            if source and str(draft.person_name or "").strip() != source:
                continue
            refs.append((image_path, int(draft.face_index)))
        return list(dict.fromkeys(refs))

    def _refresh_saved_face_drafts(self, image_path: str) -> None:
        """Refresh persisted regions without performing database I/O in Qt's event loop."""
        self._face_editor_dirty_paths.discard(image_path)
        self._load_indexed_face_drafts_async(
            str(image_path),
            dict(self._current_context),
            request_id=self._request_id,
        )

    def _name_selected_faces(self) -> None:
        refs = self._selected_saved_face_refs()
        if not refs or self.face_service is None:
            errorBox("No saved face region", "Save a detected or drawn face box, then select it to name it.")
            return
        name = self.face_name_input.text().strip()
        if not name:
            dialog = EntityPickerDialog(
                "Name selected face regions",
                "Choose a saved person or create a new person for the selected face region(s).",
                choices=self.face_name_input.choices,
                entity_label="person name",
                parent=self,
            )
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            name = dialog.selected_value()
            if not name:
                return
        image_path = self._current_editable_path()

        def _run(progress, cancel_check):
            _ = cancel_check
            progress(-1, "Writing selected face-region names...")
            label_with_metadata = getattr(self.face_service, "label_indexed_faces_with_metadata", None)
            if callable(label_with_metadata):
                return label_with_metadata(name, refs, source="photo_inspector")
            return self.face_service.label_indexed_faces_immediately(name, refs, source="photo_inspector")

        def _done(result) -> None:
            person = getattr(result, "person", result)
            if person is None:
                self.face_editor_summary_label.setText("Could not write the selected face-region name.")
                return
            self.face_name_input.setText(str(person.person_name))
            self._refresh_saved_face_drafts(image_path)
            self.face_editor_summary_label.setText(f"Saved '{person.person_name}' on {len(getattr(result, 'affected_refs', refs) or refs)} face region(s).")
            if callable(self.face_edit_saved_callback):
                self.face_edit_saved_callback(image_path)

        self._start_face_edit_job("Name selected face regions", _run, _done)

    def _choose_selected_source_name(self, *, title: str) -> str:
        image_path = self._current_editable_path()
        drafts = list(self._face_drafts_by_path.get(image_path, []))
        names = sorted(
            {
                str(drafts[index].person_name or "").strip()
                for index in self._selected_face_draft_indexes()
                if 0 <= index < len(drafts) and str(drafts[index].person_name or "").strip()
            },
            key=str.casefold,
        )
        if not names:
            return ""
        if len(names) == 1:
            return names[0]
        dialog = EntityPickerDialog(
            title,
            "Choose the current person name on the selected face region(s).",
            choices=names,
            allow_create=False,
            entity_label="person name",
            parent=self,
        )
        return dialog.selected_value() if dialog.exec() == QDialog.DialogCode.Accepted else ""

    def _rename_selected_faces(self) -> None:
        if not self._selected_saved_face_refs() or self.face_service is None:
            return
        source = self._choose_selected_source_name(title="Rename selected face regions")
        if not source:
            return
        refs = self._selected_saved_face_refs(person_name=source)
        if not refs:
            return
        dialog = EntityPickerDialog(
            "Rename selected face regions",
            f"Choose a saved person or create the new name for {source}.",
            choices=self.face_name_input.choices,
            initial=source,
            entity_label="person name",
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        target = dialog.selected_value()
        if not target or target == source:
            return
        image_path = self._current_editable_path()

        def _run(progress, cancel_check):
            _ = cancel_check
            progress(-1, "Renaming selected face regions...")
            label_with_metadata = getattr(self.face_service, "label_indexed_faces_with_metadata", None)
            if callable(label_with_metadata):
                return label_with_metadata(target, refs, source="photo_inspector_rename")
            return self.face_service.label_indexed_faces_immediately(target, refs, source="photo_inspector_rename")

        def _done(_result) -> None:
            self.face_name_input.setText(target)
            self._refresh_saved_face_drafts(image_path)
            self.face_editor_summary_label.setText(f"Renamed selected '{source}' face region(s) to '{target}'.")
            if callable(self.face_edit_saved_callback):
                self.face_edit_saved_callback(image_path)

        self._start_face_edit_job("Rename selected face regions", _run, _done)

    def _unlabel_selected_faces(self) -> None:
        if not self._selected_saved_face_refs() or self.face_service is None:
            return
        source = self._choose_selected_source_name(title="Unlabel selected face regions")
        if not source:
            return
        refs = self._selected_saved_face_refs(person_name=source)
        if not refs:
            return
        if not confirmBox("Unlabel selected face regions", f"Remove '{source}' only from the selected face region(s)?", parent=self):
            return
        image_path = self._current_editable_path()

        def _run(progress, cancel_check):
            _ = cancel_check
            progress(-1, "Unlabeling selected face regions...")
            unlabel = getattr(self.face_service, "unlabel_indexed_faces_with_metadata", None)
            return unlabel(refs, source="photo_inspector_unlabel") if callable(unlabel) else self.face_service.unlabel_labeled_faces_in_images(source, [image_path])

        def _done(_result) -> None:
            self.face_name_input.clear()
            self._refresh_saved_face_drafts(image_path)
            self.face_editor_summary_label.setText(f"Unlabeled selected '{source}' face region(s).")
            if callable(self.face_edit_saved_callback):
                self.face_edit_saved_callback(image_path)

        self._start_face_edit_job("Unlabel selected face regions", _run, _done)

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
            (self._face_draft_job, self._face_draft_thread),
            (self._face_thumbnail_job, self._face_thumbnail_thread),
            (self._prefetch_job, self._prefetch_thread),
            (self._face_edit_job, self._face_edit_thread),
            (self._face_name_suggestion_job, self._face_name_suggestion_thread),
            (self._metadata_edit_job, self._metadata_edit_thread),
            *self._retained_async_refs,
            *[(job, thread) for thread, job in list(self._thread_jobs.items())],
        ]
        for job, thread in active_pairs:
            if job is not None:
                self._cancel_operation_job(job)
            thread_finished = True
            if thread is not None:
                try:
                    thread_finished = wait_for_thread_shutdown(thread, timeout_ms=timeout_ms)
                except RuntimeError:
                    thread_finished = True
            ready_to_close = bool(thread_finished) and ready_to_close
        if ready_to_close:
            if self.work_coordinator is not None:
                for coordinated_job_id in tuple(self._coordinated_job_ids.values()):
                    self.work_coordinator.finish(coordinated_job_id, status="cancelled")
            self._coordinated_job_ids.clear()
            # Threads can finish before their queued terminal relay reaches the
            # GUI event loop. Disconnect inspector observers while the dialog
            # is still alive so those callbacks cannot touch deleted controls.
            disconnected_jobs: set[int] = set()
            for job, _thread in active_pairs:
                if job is None or id(job) in disconnected_jobs:
                    continue
                disconnected_jobs.add(id(job))
                for signal_name in ("started", "progress", "completed", "failed", "cancelled"):
                    signal = getattr(job, signal_name, None)
                    if signal is None:
                        continue
                    try:
                        signal.disconnect()
                    except (TypeError, RuntimeError):
                        pass
        self._active_job = None
        self._active_thread = None
        self._preview_job = None
        self._preview_thread = None
        self._face_draft_job = None
        self._face_draft_thread = None
        self._face_thumbnail_job = None
        self._face_thumbnail_thread = None
        self._face_thumbnail_request_signature = None
        self._prefetch_job = None
        self._prefetch_thread = None
        self._face_edit_job = None
        self._face_edit_thread = None
        self._face_name_suggestion_job = None
        self._face_name_suggestion_thread = None
        self._metadata_edit_job = None
        self._metadata_edit_thread = None
        if ready_to_close:
            self._retained_async_refs = []
            self._thread_jobs = {}
            self._progress_jobs = {}
        return ready_to_close

    def _viewer_background_changed(self, _index: int) -> None:
        mode = str(self.viewer_background_combo.currentData() or "adaptive_neutral")
        manager = get_theme_manager()
        if manager is not None:
            manager.set_viewer_backdrop(mode, persist=True)
        else:
            self.preview_view.set_backdrop_mode(mode)

    def _sync_viewer_background_combo(self, mode: str) -> None:
        index = self.viewer_background_combo.findData(str(mode))
        if index < 0 or index == self.viewer_background_combo.currentIndex():
            return
        self.viewer_background_combo.blockSignals(True)
        self.viewer_background_combo.setCurrentIndex(index)
        self.viewer_background_combo.blockSignals(False)

    def _on_zoom_changed(self, zoom: float) -> None:
        self.zoom_label.setText(f"{int(round(float(zoom) * 100.0))}%")
        if zoom <= 1.01 or self._full_res_loaded or self._full_res_requested or not self._image_paths:
            return
        self._full_res_requested = True
        self._load_preview(self._image_paths[self._index], request_id=self._request_id, full_res=True)

    def _toggle_full_screen(self) -> None:
        if self.isFullScreen():
            self.showNormal()
            self.full_screen_button.setText("Full screen")
        else:
            self.showFullScreen()
            self.full_screen_button.setText("Exit full screen")

    def _prefetch_neighbors(self) -> None:
        if not self._image_paths:
            return
        paths = [
            self._image_paths[index]
            for index in {self._index - 1, self._index + 1}
            if 0 <= index < len(self._image_paths)
        ]
        if not paths:
            return

        def _run(progress, cancel_check):
            progress(-1, "Prefetching adjacent photos…")
            for path in paths:
                raise_if_cancelled(cancel_check)
                reader = QImageReader(path)
                reader.setAutoTransform(True)
                _ = reader.size()
            return None

        job = AsyncJob(_run)
        self._start_operation_worker(
            job,
            "Prefetching adjacent photos",
            foreground=False,
            job_attribute="_prefetch_job",
            thread_attribute="_prefetch_thread",
            source_reads=tuple(paths),
        )

    def closeEvent(self, event) -> None:
        if not self._confirm_leave_changes():
            event.ignore()
            return
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
            ("1", self.preview_view.actual_size),
            ("+", self.preview_view.zoom_in),
            ("-", self.preview_view.zoom_out),
            ("F11", self._toggle_full_screen),
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
        self.metadata_summary_label.setVisible(True)
        self.metadata_editor_group.setVisible(True)
        self.inspector_tabs.setTabVisible(self.inspector_tabs.indexOf(self.exif_page), advanced)

    def _find_next_exif_value(self) -> None:
        text = self.exif_search_field.text().strip()
        if not text:
            return
        if not self.info_text.find(text):
            cursor = self.info_text.textCursor()
            cursor.movePosition(cursor.MoveOperation.Start)
            self.info_text.setTextCursor(cursor)
            self.info_text.find(text)

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

    def _track_operation_job(self, job: AsyncJob, label: str, *, foreground: bool) -> None:
        """Expose every inspector worker locally and in the shared Jobs panel."""
        if self.job_manager is not None and self.work_coordinator is None:
            self.job_manager.bind_async_job(
                job,
                label,
                origin="Photo Inspector",
                foreground=foreground,
            )
        self._progress_sequence += 1
        key = id(job)
        self._progress_jobs[key] = {
            "label": str(label),
            "foreground": bool(foreground),
            "progress": None,
            "text": f"{label}…",
            "sequence": self._progress_sequence,
        }
        job.progress.connect(lambda value, text, key=key: self._update_operation_progress(key, value, text))
        job.completed.connect(lambda _result, key=key: self._finish_operation_progress(key))
        job.failed.connect(lambda _message, key=key: self._finish_operation_progress(key))
        job.cancelled.connect(lambda key=key: self._finish_operation_progress(key))
        self._refresh_operation_progress()

    def _cancel_operation_job(self, job: AsyncJob | None) -> None:
        if job is None:
            return
        coordinated_job_id = self._coordinated_job_ids.get(job)
        if coordinated_job_id is not None and self.work_coordinator is not None:
            self.work_coordinator.cancel(coordinated_job_id)
            return
        try:
            job.cancel()
        except Exception:
            pass

    def _start_operation_worker(
        self,
        job: AsyncJob,
        label: str,
        *,
        foreground: bool,
        job_attribute: str,
        thread_attribute: str,
        source_reads: tuple[str, ...] = (),
        source_writes: tuple[str, ...] = (),
        data_home_read: bool = False,
        data_home_write: bool = False,
        model_cache_read: bool = False,
    ) -> None:
        """Start one Inspector worker only after shared resource admission."""

        self._track_operation_job(job, label, foreground=foreground)
        setattr(self, job_attribute, job)

        def _terminal_cleanup(*_args) -> None:
            self._coordinated_job_ids.pop(job, None)
            if getattr(self, thread_attribute, None) is None and getattr(self, job_attribute, None) is job:
                setattr(self, job_attribute, None)

        job.completed.connect(_terminal_cleanup)
        job.failed.connect(_terminal_cleanup)
        job.cancelled.connect(_terminal_cleanup)

        def _launch(_use_cpu_fallback: bool = False) -> None:
            thread = start_job_in_thread(job)
            self._thread_jobs[thread] = job
            thread.finished.connect(
                lambda thread=thread: self._on_async_thread_finished(thread),
                Qt.ConnectionType.QueuedConnection,
            )
            setattr(self, thread_attribute, thread)

        if self.work_coordinator is None:
            _launch()
            return
        coordinated_job_id = self.work_coordinator.submit_async_job(
            JobSpec(
                label,
                origin="Photo Inspector",
                foreground=foreground,
                io_bound=True,
                source_reads=source_reads,
                source_writes=source_writes,
                data_home_read=data_home_read,
                data_home_write=data_home_write,
                model_cache_read=model_cache_read,
            ),
            job,
            _launch,
        )
        manager = self.job_manager or self.work_coordinator.job_manager
        state = manager.get(coordinated_job_id)
        if state is not None and state.status in {"queued", "running", "cancelling"}:
            self._coordinated_job_ids[job] = coordinated_job_id

    def _update_operation_progress(self, key: int, value: int, text: str) -> None:
        state = self._progress_jobs.get(int(key))
        if state is None:
            return
        state["progress"] = None if int(value) < 0 else max(0, min(100, int(value)))
        if text:
            state["text"] = str(text)
        self._refresh_operation_progress()

    def _finish_operation_progress(self, key: int) -> None:
        self._progress_jobs.pop(int(key), None)
        self._refresh_operation_progress()

    def _refresh_operation_progress(self) -> None:
        if not self._progress_jobs:
            self.operation_progress_bar.hide()
            return
        state = max(
            self._progress_jobs.values(),
            key=lambda item: (bool(item["foreground"]), int(item["sequence"])),
        )
        progress = state.get("progress")
        if progress is None:
            self.operation_progress_bar.setRange(0, 0)
        else:
            self.operation_progress_bar.setRange(0, 100)
            self.operation_progress_bar.setValue(int(progress))
        self.operation_progress_bar.show()
        text = str(state.get("text") or "")
        if text:
            self._set_state_text(text)

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
        if not self._confirm_leave_changes():
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
    def _render_metadata_summary(
        cls,
        metadata: PhotoMetadata,
        draft: PhotoEditDraft | None = None,
    ) -> str:
        camera = html.escape(metadata.camera or "Unknown")
        tags = metadata.context.get("tags") if isinstance(metadata.context, dict) else None
        tags_line = ""
        if isinstance(tags, (list, tuple)) and tags:
            tags_line = f"<br><b>Tags</b> {html.escape(', '.join(str(tag) for tag in tags))}"
        date_fields = cls._date_related_exif_fields(metadata.exif)
        date_lines = [
            f"<br><b>{html.escape(key)}</b> {html.escape(value)}"
            for key, value in date_fields
        ]
        if not date_lines:
            date_lines.append("<br><span>No EXIF date/time fields found.</span>")
        sidecar_captured_at = str(getattr(draft, "captured_at", "") or "").strip()
        if sidecar_captured_at:
            date_lines.append(
                "<br><b>Timeline correction (sidecar)</b> "
                f"{html.escape(sidecar_captured_at)}"
            )
        return (
            "<b>Important Details</b><br>"
            f"<b>Dimensions</b> {metadata.width} x {metadata.height}"
            f" &nbsp; | &nbsp; <b>Size</b> {cls._format_file_size(metadata.file_size)}"
            f" &nbsp; | &nbsp; <b>File modified</b> {html.escape(metadata.modified_at)}"
            f"<br><b>Camera</b> {camera}"
            "<br><b>Photo dates</b>"
            f"{''.join(date_lines)}"
            f"{tags_line}"
        )

    @staticmethod
    def _date_related_exif_fields(exif: dict[str, str]) -> list[tuple[str, str]]:
        """Return every scalar EXIF field whose name represents date/time data."""

        priority = {
            "datetimeoriginal": 0,
            "datetimedigitized": 1,
            "datetime": 2,
            "gpsdatestamp": 3,
            "gpstimestamp": 4,
            "offsettimeoriginal": 5,
            "offsettimedigitized": 6,
            "offsettime": 7,
            "subsectimeoriginal": 8,
            "subsectimedigitized": 9,
            "subsectime": 10,
        }
        fields = [
            (str(key), str(value))
            for key, value in dict(exif or {}).items()
            if str(value or "").strip()
            and any(token in str(key).casefold() for token in ("date", "time", "timestamp"))
        ]
        return sorted(
            fields,
            key=lambda item: (priority.get(item[0].casefold(), len(priority)), item[0].casefold()),
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
