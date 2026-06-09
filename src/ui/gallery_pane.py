from __future__ import annotations

from collections import OrderedDict
from queue import Empty, PriorityQueue
from pathlib import Path
from threading import Event, Lock
from time import perf_counter

from PyQt6.QtCore import QUrl, QSize, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QDesktopServices, QGuiApplication, QImage, QPixmap
from PyQt6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QListView, QMenu, QProgressBar, QPushButton, QVBoxLayout, QWidget

from app.selection import SelectionTarget
from app.services.gallery_actions import GalleryActionService
from app.services.image_tags import ImageTagService
from app.services.thumbnails import ThumbnailService
from infra.logging_config import get_logger
from infra.qt_diagnostics import append_qt_diagnostic
from infra.settings import get_settings

from .error_mbox import ExifMetadataDialog, ImageTagsDialog, confirmBox, errorBox, infoBox
from .gallery_model import GalleryImageModel, GalleryItemDelegate
from .photo_inspector_dialog import PhotoInspectorDialog
from .async_job import AsyncJob, start_job_in_thread
from .common import build_help_inline

LOGGER = get_logger(__name__)


GALLERY_HELP = {
    "current_group": (
        "Show which image set gallery actions will use when nothing is checked.\n"
        "In clustering mode this is usually the selected cluster from the right pane.\n"
        "Checked images override the current group for tag, EXIF, copy, move, and delete actions."
    ),
    "select_current_group": (
        "Check or uncheck every image in the current group.\n"
        "The current group is usually the selected cluster in clustering mode.\n"
        "Use this when you want gallery actions to target the whole cluster quickly."
    ),
    "write_exif": (
        "Write EXIF metadata to the checked images, or to the current group if nothing is checked.\n"
        "Checked images win over the current group when both exist.\n"
        "This does not change clustering directly, but the files themselves are updated."
    ),
    "tags": (
        "Edit image tags for the checked images, or for the current group if nothing is checked.\n"
        "Checked images win over the current group when both exist.\n"
        "In clustering mode the current group is usually the selected cluster.\n"
        "Use Tags Selected Only when you do not want this cluster fallback.\n"
        "The local tag database is authoritative, and EXIF mirroring is optional."
    ),
    "tags_selected_only": (
        "Edit image tags only for explicitly selected gallery tiles.\n"
        "Highlighted tiles are used first; checked tiles are used if nothing is highlighted.\n"
        "This action never falls back to the current cluster, so it is safer for small manual edits."
    ),
    "metadata_menu": (
        "Metadata tools for the current gallery target.\n"
        "Use Tags Selected Only for small manual tag edits; this menu keeps broader EXIF and checked/group tag actions out of the main row."
    ),
    "file_ops_menu": (
        "Copy, move, or delete the checked images, or the current group if nothing is checked.\n"
        "Delete is grouped here to reduce accidental clicks on the main gallery row."
    ),
    "more_menu": (
        "Less frequent gallery utilities: copy paths, export paths, and retry failed thumbnails.\n"
        "These actions use the same checked-images-first target rules unless the action says otherwise."
    ),
    "open_folder": (
        "Open the containing folder for the checked images, or for the current group if nothing is checked.\n"
        "Checked images override the current group.\n"
        "If the selection spans multiple folders, the app opens a limited number and reports the rest."
    ),
    "copy_paths": (
        "Copy the file paths for the checked images, or for the current group if nothing is checked.\n"
        "Checked images override the current group.\n"
        "Paths are copied as newline-separated text so they can be pasted into editors or scripts."
    ),
    "export_paths": (
        "Export the file paths for the checked images, or for the current group if nothing is checked.\n"
        "Checked images override the current group.\n"
        "The export writes one absolute path per line to a text file."
    ),
    "copy_selection": (
        "Copy the checked images, or the current group if nothing is checked.\n"
        "Checked images override the current group.\n"
        "Copied files inherit image tags inside the app's tag database."
    ),
    "move_selection": (
        "Move the checked images, or the current group if nothing is checked.\n"
        "Checked images override the current group.\n"
        "Moved files keep their image tags by updating the tag database paths."
    ),
    "delete_selection": (
        "Delete the checked images, or the current group if nothing is checked.\n"
        "Checked images override the current group.\n"
        "Use this carefully because the whole selected cluster can become the current group."
    ),
    "retry_failed": (
        "Retry thumbnail loading for visible images that previously failed.\n"
        "This only affects gallery preview loading, not tags or clustering data.\n"
        "Use it after temporary IO errors or when files become available again."
    ),
}


class ThumbnailRequestQueue:
    def __init__(self) -> None:
        self._stop = Event()
        self._queue: PriorityQueue[tuple[int, int, int]] = PriorityQueue()
        self._lock = Lock()
        self._desired_priority: dict[int, int] = {}
        self._seq = 0
        self._generation = 0
        self._images: list[str] = []
        self._image_size = 0

    def cancel(self) -> None:
        self._stop.set()

    def configure(self, generation: int, images: list[str], image_size: int) -> None:
        with self._lock:
            self._generation = int(generation)
            self._images = list(images)
            self._image_size = int(image_size)
            self._desired_priority.clear()
            self._seq = 0
        self.clear_queue()

    def clear_queue(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break

    def enqueue(self, indexes: list[int], *, priority: int) -> None:
        if not indexes:
            return
        priority = int(priority)
        with self._lock:
            for index in indexes:
                index = int(index)
                if index < 0:
                    continue
                existing = self._desired_priority.get(index)
                if existing is not None and existing <= priority:
                    continue
                self._desired_priority[index] = priority
                self._seq += 1
                self._queue.put((priority, self._seq, index))

    def get_next(self, timeout_s: float = 0.2) -> tuple[int, int, str, int] | None:
        if self._stop.is_set():
            return None
        try:
            priority, _seq, index = self._queue.get(timeout=timeout_s)
        except Empty:
            return None
        with self._lock:
            if self._stop.is_set():
                return None
            desired = self._desired_priority.get(index)
            generation = self._generation
            images = self._images
            image_size = self._image_size
            if desired != priority:
                # Stale request; a newer priority was queued.
                return None
            self._desired_priority.pop(index, None)
        if index >= len(images):
            return None
        return generation, index, images[index], image_size


class ImageLoaderThread(QThread):
    image_loaded = pyqtSignal(int, int, str, object)
    image_failed = pyqtSignal(int, int, str, str)

    def __init__(self, request_queue: ThumbnailRequestQueue, thumbnail_service: ThumbnailService, parent=None):
        super().__init__(parent)
        self.request_queue = request_queue
        self.thumbnail_service = thumbnail_service
        self.setObjectName(f"ImageLoaderThread:{id(self):x}")

    def run(self):
        append_qt_diagnostic(f"[ThreadStart] {self.objectName()}")
        while True:
            item = self.request_queue.get_next(timeout_s=0.2)
            if item is None:
                if self.request_queue._stop.is_set():  # noqa: SLF001
                    append_qt_diagnostic(f"[ThreadFinish] {self.objectName()} stopped=True")
                    return
                continue
            generation, index, image_path, image_size = item
            try:
                qimage = self.thumbnail_service.load_qimage(image_path, image_size)
                if qimage.isNull():
                    raise ValueError("QImage decode returned null")
                self.image_loaded.emit(generation, index, image_path, qimage)
            except Exception as exc:
                self.image_failed.emit(generation, index, image_path, str(exc))


class GalleryPane(QWidget):
    first_paint_ready = pyqtSignal(int)
    image_selected = pyqtSignal(str)
    add_to_review_requested = pyqtSignal(list)
    remove_from_review_requested = pyqtSignal(list)
    paths_removed = pyqtSignal(list)
    metadata_changed = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = get_settings()
        self.job_manager = None
        self.image_size = self.settings.thumbnail_size
        self.max_thumbnail_workers = int(self.settings.max_thumbnail_workers)
        self.thumbnail_prefetch_rows = int(self.settings.thumbnail_prefetch_rows)
        self.images: list[str] = []
        self.request_generation = 0
        self.loader_queue = ThumbnailRequestQueue()
        self.loader_threads: list[ImageLoaderThread] = []
        self._retained_loader_threads: list[ImageLoaderThread] = []
        self._active_action_thread = None
        self._active_action_job = None
        self._retained_action_refs: list[tuple[object | None, object | None]] = []
        self._action_thread_jobs: dict[object, object | None] = {}
        self._active_action_job_id: int | None = None
        self.loaded_indexes: set[int] = set()
        self.failed_indexes: set[int] = set()
        self.pending_indexes: set[int] = set()
        self.pending_ui_items: list[tuple[int, str, object]] = []
        self.pixmap_cache: OrderedDict[tuple[str, int], QPixmap] = OrderedDict()
        self._pixmap_cache_size = int(self.settings.thumbnail_cache_size)
        self.thumbnail_service = ThumbnailService(qimage_cache_size=self._pixmap_cache_size)
        self.action_service = GalleryActionService()
        self.image_tag_service: ImageTagService | None = None
        self.first_paint_start = 0.0
        self.first_paint_emitted = False
        self.membership_context: dict[str, dict[str, dict[str, object]]] = {}
        self.metrics_context: dict[str, dict[str, object]] = {}
        self.inspector_context_provider = None  # optional callable(path) -> dict
        self.inspector_display_mode = "advanced"
        self.action_target_provider = None
        self.metadata_service = None
        self.review_action_mode = "add"
        self.main_layout = QVBoxLayout(self)
        self.action_bar = QHBoxLayout()
        self.status_label = QLabel("")
        self.progress_bar = QProgressBar()
        self.model = GalleryImageModel(self)
        self.delegate = GalleryItemDelegate(self.image_size, self)
        self._item_spacing = 12
        self.refresh_timer = QTimer(self)
        self.refresh_timer.setSingleShot(True)
        self.refresh_timer.timeout.connect(self.load_visible_images)
        self.flush_timer = QTimer(self)
        self.flush_timer.setSingleShot(True)
        self.flush_timer.timeout.connect(self.flush_pending_ui_items)
        self._shutting_down = False
        self._build_ui()

    def _build_ui(self):
        self.target_hint_label = QLabel("Current Group: Visible Images")
        self.select_group_button = QPushButton("Select Current Group")
        self.selected_tags_button = QPushButton("Tags Selected Only")
        self.open_folder_button = QPushButton("Open Folder")
        self.metadata_menu_button = QPushButton("Metadata")
        self.file_ops_menu_button = QPushButton("File Ops")
        self.more_menu_button = QPushButton("More")

        # Keep concrete buttons for stable tests and external callers, but expose
        # the less frequent commands through compact menus to avoid row overflow.
        self.exif_button = QPushButton("Write EXIF", self)
        self.tags_button = QPushButton("Tags: Checked/Group", self)
        self.copy_paths_button = QPushButton("Copy Paths", self)
        self.export_paths_button = QPushButton("Export Paths", self)
        self.copy_button = QPushButton("Copy Selection", self)
        self.move_button = QPushButton("Move Selection", self)
        self.delete_button = QPushButton("Delete Selection", self)
        self.retry_failed_button = QPushButton("Retry Failed", self)
        self.exif_button.setToolTip(GALLERY_HELP["write_exif"])
        self.tags_button.setToolTip(GALLERY_HELP["tags"])
        self.copy_paths_button.setToolTip(GALLERY_HELP["copy_paths"])
        self.export_paths_button.setToolTip(GALLERY_HELP["export_paths"])
        self.copy_button.setToolTip(GALLERY_HELP["copy_selection"])
        self.move_button.setToolTip(GALLERY_HELP["move_selection"])
        self.delete_button.setToolTip(GALLERY_HELP["delete_selection"])
        self.retry_failed_button.setToolTip(GALLERY_HELP["retry_failed"])
        self._legacy_action_buttons = [
            self.exif_button,
            self.tags_button,
            self.copy_paths_button,
            self.export_paths_button,
            self.copy_button,
            self.move_button,
            self.delete_button,
            self.retry_failed_button,
        ]
        for button in self._legacy_action_buttons:
            button.hide()

        self.metadata_menu = QMenu(self.metadata_menu_button)
        self.file_ops_menu = QMenu(self.file_ops_menu_button)
        self.more_menu = QMenu(self.more_menu_button)
        self.metadata_menu_button.setMenu(self.metadata_menu)
        self.file_ops_menu_button.setMenu(self.file_ops_menu)
        self.more_menu_button.setMenu(self.more_menu)
        self.action_bar.addWidget(
            build_help_inline(
                self.target_hint_label,
                GALLERY_HELP["current_group"],
                help_key="current_group",
                primary_stretch=1,
            ),
            stretch=1,
        )
        self.action_bar.addWidget(build_help_inline(self.select_group_button, GALLERY_HELP["select_current_group"], help_key="select_current_group"))
        self.action_bar.addWidget(build_help_inline(self.selected_tags_button, GALLERY_HELP["tags_selected_only"], help_key="tags_selected_only"))
        self.action_bar.addWidget(build_help_inline(self.open_folder_button, GALLERY_HELP["open_folder"], help_key="open_folder"))
        self.action_bar.addWidget(build_help_inline(self.metadata_menu_button, GALLERY_HELP["metadata_menu"], help_key="metadata_menu"))
        self.action_bar.addWidget(build_help_inline(self.file_ops_menu_button, GALLERY_HELP["file_ops_menu"], help_key="file_ops_menu"))
        self.action_bar.addWidget(build_help_inline(self.more_menu_button, GALLERY_HELP["more_menu"], help_key="more_menu"))
        self.main_layout.addLayout(self.action_bar)

        self.list_view = QListView()
        self.list_view.setModel(self.model)
        self.list_view.setItemDelegate(self.delegate)
        self.list_view.setViewMode(QListView.ViewMode.IconMode)
        self.list_view.setResizeMode(QListView.ResizeMode.Adjust)
        self.list_view.setMovement(QListView.Movement.Static)
        self.list_view.setUniformItemSizes(True)
        self.list_view.setSpacing(self._item_spacing)
        self.list_view.setGridSize(QSize(self.delegate.card_width + self._item_spacing, self.delegate.card_height + self._item_spacing))
        self.list_view.setSelectionMode(QListView.SelectionMode.ExtendedSelection)
        self.list_view.clicked.connect(self.on_item_clicked)
        self.list_view.doubleClicked.connect(self.on_item_double_clicked)
        self.list_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.list_view.customContextMenuRequested.connect(self.on_context_menu)
        self.list_view.verticalScrollBar().valueChanged.connect(self.schedule_visible_refresh)
        self.list_view.horizontalScrollBar().valueChanged.connect(self.schedule_visible_refresh)
        self.main_layout.addWidget(self.list_view)

        self.status_label.setWordWrap(True)
        self.main_layout.addWidget(self.status_label)
        self.progress_bar.hide()
        self.main_layout.addWidget(self.progress_bar)

        self.select_group_button.clicked.connect(self.select_current_group)
        self.delete_button.clicked.connect(self.slotDeleteSelect)
        self.exif_button.clicked.connect(self.slotAddExif)
        self.tags_button.clicked.connect(self.slotEditTags)
        self.selected_tags_button.clicked.connect(self.slotEditSelectedOnlyTags)
        self.open_folder_button.clicked.connect(self.slotOpenSelectedFolders)
        self.copy_paths_button.clicked.connect(self.slotCopySelectedPaths)
        self.export_paths_button.clicked.connect(self.slotExportSelectedPaths)
        self.copy_button.clicked.connect(self.slotCopySelected)
        self.move_button.clicked.connect(self.slotMoveSelected)
        self.retry_failed_button.clicked.connect(self.retry_failed_visible)
        self._add_menu_action(self.metadata_menu, "Write EXIF", GALLERY_HELP["write_exif"], self.slotAddExif)
        self._add_menu_action(self.metadata_menu, "Tags: Checked/Group", GALLERY_HELP["tags"], self.slotEditTags)
        self._add_menu_action(self.file_ops_menu, "Copy Selection", GALLERY_HELP["copy_selection"], self.slotCopySelected)
        self._add_menu_action(self.file_ops_menu, "Move Selection", GALLERY_HELP["move_selection"], self.slotMoveSelected)
        self._add_menu_action(self.file_ops_menu, "Delete Selection", GALLERY_HELP["delete_selection"], self.slotDeleteSelect)
        self._add_menu_action(self.more_menu, "Copy Paths", GALLERY_HELP["copy_paths"], self.slotCopySelectedPaths)
        self._add_menu_action(self.more_menu, "Export Paths", GALLERY_HELP["export_paths"], self.slotExportSelectedPaths)
        self._add_menu_action(self.more_menu, "Retry Failed Thumbnails", GALLERY_HELP["retry_failed"], self.retry_failed_visible)
        self._group_action_widgets = [
            self.target_hint_label,
            self.select_group_button,
        ]
        self._metadata_action_widgets = [
            self.selected_tags_button,
            self.metadata_menu_button,
        ]
        self._file_action_widgets = [
            self.open_folder_button,
            self.file_ops_menu_button,
            self.more_menu_button,
        ]
        self._busy_action_widgets = [
            self.select_group_button,
            self.selected_tags_button,
            self.open_folder_button,
            self.metadata_menu_button,
            self.file_ops_menu_button,
            self.more_menu_button,
            *self._legacy_action_buttons,
        ]

    @staticmethod
    def _add_menu_action(menu: QMenu, text: str, tooltip: str, callback) -> None:
        action = menu.addAction(text)
        action.setToolTip(str(tooltip or ""))
        action.setStatusTip(str(tooltip or "").splitlines()[0] if tooltip else "")
        action.triggered.connect(lambda _checked=False, cb=callback: cb())

    def on_context_menu(self, pos) -> None:
        index = self.list_view.indexAt(pos)
        if not index.isValid():
            return
        image_path = index.data(self.model.PathRole)
        if not image_path:
            return
        menu = QMenu(self)
        open_inspector = menu.addAction("Open Inspector")
        open_folder = menu.addAction("Open Containing Folder")
        copy_path = menu.addAction("Copy Path")
        copy_paths_action = menu.addAction("Copy Paths (Selection)")
        export_paths_action = menu.addAction("Export Paths (Selection)")
        copy_image = menu.addAction("Copy Image")
        review_action = None
        if self.review_action_mode == "remove":
            review_action = menu.addAction("Remove From Review")
        elif self.review_action_mode == "add":
            review_action = menu.addAction("Add To Review")
        menu.addSeparator()
        move_to_trash = menu.addAction("Move To Trash (Selection)")
        move_to_dir = menu.addAction("Move To... (Selection)")
        action = menu.exec(self.list_view.viewport().mapToGlobal(pos))
        if action == open_inspector:
            self.on_item_double_clicked(index)
        elif action == open_folder:
            try:
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(str(image_path)).parent)))
            except Exception:
                pass
        elif action == copy_path:
            try:
                QGuiApplication.clipboard().setText(str(image_path))
            except Exception:
                pass
        elif action == copy_paths_action:
            self.slotCopySelectedPaths()
        elif action == export_paths_action:
            self.slotExportSelectedPaths()
        elif action == copy_image:
            try:
                img = QImage(str(image_path))
                if not img.isNull():
                    QGuiApplication.clipboard().setImage(img)
            except Exception:
                pass
        elif review_action is not None and action == review_action:
            selected = self.model.checked_paths() or [str(image_path)]
            if self.review_action_mode == "remove":
                self.remove_from_review_requested.emit(selected)
            else:
                self.add_to_review_requested.emit(selected)
        elif action == move_to_trash:
            self.slotDeleteSelect()
        elif action == move_to_dir:
            self.slotMoveSelected()

    def select_current_group(self) -> None:
        target = self.current_group_target()
        if target is None or not target.paths:
            self.status_label.setText("No group is available for selection.")
            return
        checked = set(self.model.checked_paths())
        group_paths = set(target.paths)
        if group_paths.issubset(checked):
            self.model.set_checked_paths(checked - group_paths)
            self.status_label.setText(f"Unselected {len(target.paths)} images from {target.label}.")
            return
        self.model.set_checked_paths(checked | group_paths)
        self.status_label.setText(f"Selected {len(target.paths)} images from {target.label}.")

    def retry_failed_visible(self) -> None:
        visible, _prefetch = self._visible_and_prefetch_indexes()
        retried = 0
        for index in visible:
            if index in self.failed_indexes:
                self.failed_indexes.discard(index)
                self.model.set_loading(index)
                retried += 1
        if retried:
            self.status_label.setText(f"Retrying {retried} failed thumbnails...")
            self.schedule_visible_refresh()

    def _set_action_busy(self, busy: bool, status: str = "") -> None:
        for button in getattr(self, "_busy_action_widgets", []):
            try:
                button.setEnabled(not busy)
            except Exception:
                pass
        if busy:
            self.progress_bar.setRange(0, 0)
            self.progress_bar.show()
        else:
            self.progress_bar.hide()
        if status:
            self.status_label.setText(status)

    def _start_action_job(self, label: str, fn, on_completed) -> None:
        if self._active_action_thread is not None:
            try:
                if self._active_action_thread.isRunning():
                    errorBox("Busy", "Another gallery operation is already running.")
                    return
            except RuntimeError:
                self._active_action_thread = None
                self._active_action_job = None
                self._active_action_job_id = None

        job = AsyncJob(fn)
        job.progress.connect(self._on_action_progress)
        if self.job_manager is not None:
            self._active_action_job_id = self.job_manager.register_job(label, cancel_fn=job.cancel)
            job.progress.connect(lambda value, text: self.job_manager.update(self._active_action_job_id or -1, progress=value, text=text))

        def _finish(status: str, error: str = "") -> None:
            if self.job_manager is not None and self._active_action_job_id is not None:
                self.job_manager.finish(self._active_action_job_id, status=status, error=error)
            self._active_action_job_id = None

        def _on_failed(message: str) -> None:
            self._set_action_busy(False)
            _finish("failed", message)
            errorBox(f"{label} failed", message)

        def _on_cancelled() -> None:
            self._set_action_busy(False, "Cancelled.")
            _finish("cancelled")

        def _on_completed(result) -> None:
            self._set_action_busy(False)
            # Allow cooperative-cancel jobs to report partial completion.
            if getattr(result, "cancelled", False):
                _finish("cancelled")
            else:
                _finish("finished")
            on_completed(result)

        job.failed.connect(_on_failed)
        job.cancelled.connect(_on_cancelled)
        job.completed.connect(_on_completed)
        self._active_action_job = job
        thread = start_job_in_thread(job)
        self._action_thread_jobs[thread] = job
        thread.finished.connect(
            self._on_action_thread_finished,
            Qt.ConnectionType.QueuedConnection,
        )
        self._active_action_thread = thread
        self._set_action_busy(True, f"{label} running...")
        if self.job_manager is not None and self._active_action_job_id is not None:
            self.job_manager.update(self._active_action_job_id, progress=None, text="Running")

    def _on_action_progress(self, value: int, text: str) -> None:
        try:
            if int(value) < 0:
                self.progress_bar.setRange(0, 0)
            else:
                self.progress_bar.setRange(0, 100)
                self.progress_bar.setValue(max(0, min(100, int(value))))
        except Exception:
            pass
        if text:
            try:
                self.status_label.setText(str(text))
            except Exception:
                pass

    def _summarize_action_result(self, result, *, verb: str, target_label: str, changed_label: str = "changed") -> str:
        changed = list(getattr(result, "changed_paths", []))
        affected = list(getattr(result, "affected_paths", []))
        failures = list(getattr(result, "failures", []))
        success_count = len(changed) if changed else len(affected)
        audit_log_path = str(getattr(result, "audit_log_path", "") or "")
        lines = [
            f"{verb} {success_count} item(s) from {target_label}.",
            f"Successful {changed_label}: {success_count}",
            f"Failures: {len(failures)}",
        ]
        if audit_log_path:
            lines.append(f"Audit log: {audit_log_path}")
        if failures:
            lines.append("")
            lines.append("First failures:")
            lines.extend(str(failure) for failure in failures[:8])
        return "\n".join(lines)

    def _show_action_result(self, title: str, result, *, verb: str, target_label: str, changed_label: str = "changed") -> None:
        failures = list(getattr(result, "failures", []))
        summary = self._summarize_action_result(result, verb=verb, target_label=target_label, changed_label=changed_label)
        self.status_label.setText(summary.splitlines()[0] if summary else title)
        if failures:
            errorBox(f"{title} completed with errors", summary)
        else:
            infoBox(title, summary)

    def slotDeleteSelect(self):
        target = self._selection_for_actions()
        if target is None:
            return
        source_paths = target.as_list()
        if not confirmBox(
            "Delete Selection?",
            f"{target.label}\n\nImages: {len(source_paths)}\nDestination: local TrashImages folder",
            parent=self,
        ):
            self.status_label.setText("Delete cancelled.")
            return

        def _run(progress, cancel_check):
            return self.action_service.move_to_trash(source_paths, progress_callback=progress, cancel_check=cancel_check)

        def _done(result) -> None:
            if self.image_tag_service is not None:
                try:
                    self.image_tag_service.sync_moved_paths(list(getattr(result, "changed_paths", [])))
                except Exception:
                    pass
            moved = {src for src, _dst in getattr(result, "changed_paths", [])}
            if moved:
                self.remove_images(list(moved))
                self.paths_removed.emit(sorted(moved))
            self._show_action_result("Delete complete", result, verb="Deleted", target_label=target.label, changed_label="trash moves")

        self._start_action_job("Deleting images", _run, _done)

    def slotMoveSelected(self):
        target = self._selection_for_actions()
        if target is None:
            return
        destination = QFileDialog.getExistingDirectory(self, "Move Selected Images")
        if not destination:
            return
        source_paths = target.as_list()

        def _run(progress, cancel_check):
            return self.action_service.move_to_directory(
                source_paths,
                destination,
                progress_callback=progress,
                cancel_check=cancel_check,
            )

        def _done(result) -> None:
            if self.image_tag_service is not None:
                try:
                    self.image_tag_service.sync_moved_paths(list(getattr(result, "changed_paths", [])))
                except Exception:
                    pass
            moved = {src for src, _dst in getattr(result, "changed_paths", [])}
            if moved:
                self.remove_images(list(moved))
                self.paths_removed.emit(sorted(moved))
            self._show_action_result("Move complete", result, verb="Moved", target_label=target.label, changed_label="moves")

        self._start_action_job("Moving images", _run, _done)

    def slotAddExif(self):
        dialog = ExifMetadataDialog(self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        key, value = dialog.get_pair()
        if not key or not value:
            self.status_label.setText("EXIF write cancelled.")
            return
        target = self._selection_for_actions()
        if target is None:
            return
        source_paths = target.as_list()

        def _run(progress, cancel_check):
            return self.action_service.write_exif_metadata_pairs(
                source_paths,
                key,
                value,
                progress_callback=progress,
                cancel_check=cancel_check,
            )

        def _done(result) -> None:
            failures = getattr(result, "failures", [])
            affected_paths = list(getattr(result, "affected_paths", []))
            if affected_paths:
                self.metadata_changed.emit(affected_paths)
            if failures:
                self._show_action_result("EXIF write", result, verb="Updated EXIF for", target_label=target.label, changed_label="metadata writes")
            else:
                self._show_action_result("EXIF updated", result, verb="Updated EXIF for", target_label=target.label, changed_label="metadata writes")

        self._start_action_job("Writing EXIF", _run, _done)

    def slotEditTags(self) -> None:
        target = self._selection_for_actions()
        if target is None:
            return
        self._edit_tags_for_target(target)

    def slotEditSelectedOnlyTags(self) -> None:
        target = self._selection_for_explicit_gallery_actions()
        if target is None:
            return
        self._edit_tags_for_target(target)

    def _edit_tags_for_target(self, target: SelectionTarget) -> None:
        if self.image_tag_service is None:
            errorBox("Tags unavailable", "Image tag service is not configured.")
            return
        dialog = ImageTagsDialog(self)
        if dialog.exec() != dialog.DialogCode.Accepted:
            return
        mode, raw_tags, mirror_to_exif = dialog.values()
        tags = self.image_tag_service.parse_tag_text(raw_tags)
        if not tags:
            self.status_label.setText("Tag edit cancelled.")
            return
        source_paths = target.as_list()
        add_tags = tags if mode.casefold() == "add" else []
        remove_tags = tags if mode.casefold() == "remove" else []

        def _run(progress, cancel_check):
            return self.image_tag_service.apply_tag_edit(
                source_paths,
                add_tags=add_tags,
                remove_tags=remove_tags,
                mirror_to_exif=mirror_to_exif,
                progress_callback=progress,
                cancel_check=cancel_check,
            )

        def _done(result) -> None:
            failures = getattr(result, "failures", [])
            affected_paths = list(getattr(result, "affected_paths", []))
            if affected_paths:
                self.metadata_changed.emit(affected_paths)
            if failures:
                errorBox("Tag update failed (some files)", "\n".join(failures[:8]))
            else:
                verb = "Added" if mode.casefold() == "add" else "Removed"
                infoBox("Tags updated", f"{verb} {', '.join(tags)} for {len(affected_paths)} images from {target.label}.")

        self._start_action_job("Updating image tags", _run, _done)

    def slotOpenSelectedFolders(self) -> None:
        target = self._selection_for_actions()
        if target is None:
            return
        folders: list[str] = []
        seen: set[str] = set()
        for image_path in target.as_list():
            normalized = str(Path(str(image_path)).parent)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            folders.append(normalized)
        if not folders:
            self.status_label.setText("No containing folders are available for the current selection.")
            return
        max_open = 8
        failures: list[str] = []
        opened = 0
        for folder in folders[:max_open]:
            try:
                if QDesktopServices.openUrl(QUrl.fromLocalFile(folder)):
                    opened += 1
                else:
                    failures.append(folder)
            except Exception:
                failures.append(folder)
        skipped = max(0, len(folders) - max_open)
        if opened:
            message = f"Opened {opened} folder(s) for {target.label}."
            if skipped:
                message += f" Skipped {skipped} additional folder(s) to avoid opening too many windows."
            self.status_label.setText(message)
        if failures:
            errorBox("Open folder failed", "\n".join(failures[:8]))

    def slotCopySelectedPaths(self) -> None:
        target = self._selection_for_actions()
        if target is None:
            return
        source_paths = target.as_list()
        try:
            QGuiApplication.clipboard().setText("\n".join(source_paths))
            self.status_label.setText(f"Copied {len(source_paths)} path(s) from {target.label} to the clipboard.")
        except Exception as exc:
            errorBox("Copy paths failed", str(exc))

    def slotExportSelectedPaths(self) -> None:
        target = self._selection_for_actions()
        if target is None:
            return
        export_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Image Paths",
            "",
            "Text Files (*.txt);;All Files (*)",
        )
        if not export_path:
            self.status_label.setText("Path export cancelled.")
            return
        source_paths = target.as_list()
        try:
            Path(export_path).write_text("\n".join(source_paths) + "\n", encoding="utf-8")
            self.status_label.setText(f"Exported {len(source_paths)} path(s) from {target.label}.")
            infoBox("Path export complete", f"Exported {len(source_paths)} path(s) to:\n{export_path}")
        except Exception as exc:
            errorBox("Path export failed", str(exc))

    def slotCopySelected(self) -> None:
        target = self._selection_for_actions()
        if target is None:
            return
        destination = QFileDialog.getExistingDirectory(self, "Copy Selected Images")
        if not destination:
            return
        source_paths = target.as_list()

        def _run(progress, cancel_check):
            return self.action_service.copy_to_directory(
                source_paths,
                destination,
                progress_callback=progress,
                cancel_check=cancel_check,
            )

        def _done(result) -> None:
            if self.image_tag_service is not None:
                try:
                    self.image_tag_service.clone_tags_for_copies(list(getattr(result, "changed_paths", [])))
                except Exception:
                    pass
            failures = getattr(result, "failures", [])
            copied = getattr(result, "changed_paths", [])
            _ = copied
            self._show_action_result("Copy complete", result, verb="Copied", target_label=target.label, changed_label="copies")

        self._start_action_job("Copying images", _run, _done)

    def update_gallery(self, images=None):
        return self.update_gallery_with_options(images=images, clear_pixmaps=True, reset_scroll=True)

    def update_gallery_with_options(self, images=None, *, clear_pixmaps: bool = True, reset_scroll: bool = True) -> None:
        if self._shutting_down:
            return
        if images is not None:
            self.images = list(images)
        self.model.set_images(self.images)
        self.reset_gallery_state(clear_pixmaps=clear_pixmaps, reset_scroll=reset_scroll)
        self.refresh_selection_target_hint()
        if not self.images:
            self.status_label.setText("No images in the current selection.")
            return
        self.first_paint_start = perf_counter()
        self.first_paint_emitted = False
        self.status_label.setText(f"Loading {len(self.images)} images...")
        self._ensure_loader()
        self.loader_queue.configure(self.request_generation, self.images, self.image_size)
        self.schedule_visible_refresh()

    def apply_view_preferences(
        self,
        *,
        thumbnail_size: int | None = None,
        worker_count: int | None = None,
        prefetch_rows: int | None = None,
        pixmap_cache_size: int | None = None,
        qimage_cache_size: int | None = None,
    ) -> None:
        if thumbnail_size is not None and int(thumbnail_size) != self.image_size:
            self.image_size = int(thumbnail_size)
            self.delegate = GalleryItemDelegate(self.image_size, self)
            self.list_view.setItemDelegate(self.delegate)
            self.list_view.setGridSize(QSize(self.delegate.card_width + self._item_spacing, self.delegate.card_height + self._item_spacing))
        if worker_count is not None:
            self.max_thumbnail_workers = max(1, int(worker_count))
        if prefetch_rows is not None:
            self.thumbnail_prefetch_rows = max(0, int(prefetch_rows))
        if pixmap_cache_size is not None:
            self._pixmap_cache_size = max(64, int(pixmap_cache_size))
            while len(self.pixmap_cache) > self._pixmap_cache_size:
                self.pixmap_cache.popitem(last=False)
        if qimage_cache_size is not None:
            self.thumbnail_service.set_qimage_cache_size(int(qimage_cache_size))
        if self.images:
            self.update_gallery_with_options(images=self.images, clear_pixmaps=True, reset_scroll=False)

    def set_action_visibility(
        self,
        *,
        show_actions: bool,
        show_metadata_actions: bool,
        show_file_actions: bool,
    ) -> None:
        visible = bool(show_actions)
        for widget in self._group_action_widgets:
            widget.setVisible(visible)
        for widget in self._metadata_action_widgets:
            widget.setVisible(visible and bool(show_metadata_actions))
        for widget in self._file_action_widgets:
            widget.setVisible(visible and bool(show_file_actions))
        for widget in getattr(self, "_legacy_action_buttons", []):
            widget.setVisible(False)

    def set_action_bar_visible(self, visible: bool) -> None:
        self.set_action_visibility(
            show_actions=bool(visible),
            show_metadata_actions=bool(visible),
            show_file_actions=bool(visible),
        )

    def clear_memory_caches(self, *, reload_visible: bool = True) -> None:
        if self._shutting_down:
            return
        self.cancel_loader()
        self.pixmap_cache.clear()
        self.thumbnail_service.clear_memory_cache()
        self.model.clear_pixmaps()
        self.loaded_indexes.clear()
        self.pending_indexes.clear()
        self.pending_ui_items.clear()
        if self.images:
            if reload_visible:
                self.status_label.setText("Gallery caches cleared. Reloading visible thumbnails...")
                self.schedule_visible_refresh()
            else:
                self.status_label.setText("Gallery caches cleared.")

    def remove_images(self, image_paths: list[str]) -> None:
        removed = {str(path) for path in image_paths if path}
        if not removed:
            return
        self.images = [path for path in self.images if path not in removed]
        self.model.remove_paths(removed)
        self.reset_gallery_state(clear_pixmaps=False, reset_scroll=False)
        self.schedule_visible_refresh()
        self.refresh_selection_target_hint()

    def append_images(self, image_paths: list[str]) -> None:
        if not image_paths:
            return
        existing = set(self.images)
        appended = [path for path in image_paths if path and path not in existing]
        if not appended:
            return
        self.images.extend(appended)
        # Preserve scroll position when appending.
        self.update_gallery_with_options(images=self.images, clear_pixmaps=False, reset_scroll=False)

    def reset_gallery_state(self, clear_pixmaps: bool, reset_scroll: bool):
        self.request_generation += 1
        if self.loader_queue:
            self.loader_queue.clear_queue()
        self.loaded_indexes.clear()
        self.failed_indexes.clear()
        self.pending_indexes.clear()
        self.pending_ui_items.clear()
        if clear_pixmaps:
            self.pixmap_cache.clear()
        if reset_scroll:
            self.list_view.scrollToTop()
            self.list_view.verticalScrollBar().setValue(0)
            self.list_view.horizontalScrollBar().setValue(0)

    def set_membership_context(self, membership_by_image, metrics_by_backend) -> None:
        self.membership_context = membership_by_image or {}
        self.metrics_context = metrics_by_backend or {}

    def set_action_target_provider(self, provider) -> None:
        self.action_target_provider = provider
        self.refresh_selection_target_hint()

    def set_inspector_display_mode(self, mode: str) -> None:
        self.inspector_display_mode = "advanced" if str(mode).strip().lower() == "advanced" else "basic"

    def current_group_target(self) -> SelectionTarget | None:
        if callable(self.action_target_provider):
            try:
                target = self.action_target_provider()
            except Exception:
                target = None
            if isinstance(target, SelectionTarget) and target.paths:
                return target
        if not self.images:
            return None
        return SelectionTarget(paths=tuple(self.images), kind="gallery", label="Visible Images")

    def refresh_selection_target_hint(self) -> None:
        target = self.current_group_target()
        if target is None:
            self.target_hint_label.setText("Current Group: None")
            return
        self.target_hint_label.setText(f"Current Group: {target.label} ({len(target.paths)})")

    def _selection_for_actions(self) -> SelectionTarget | None:
        checked = tuple(self.model.checked_paths())
        if checked:
            return SelectionTarget(paths=checked, kind="checked_images", label=f"Checked Photos ({len(checked)})")
        target = self.current_group_target()
        if target is None or not target.paths:
            self.status_label.setText("Select photos or a cluster first.")
            return None
        return target

    def _selection_for_explicit_gallery_actions(self) -> SelectionTarget | None:
        selected = tuple(self._selected_gallery_paths())
        if selected:
            return SelectionTarget(paths=selected, kind="selected_gallery_images", label=f"Selected Gallery Photos ({len(selected)})")
        checked = tuple(self.model.checked_paths())
        if checked:
            return SelectionTarget(paths=checked, kind="checked_images", label=f"Checked Photos ({len(checked)})")
        self.status_label.setText("Select or check one or more photos first. This tag action will not use the current cluster.")
        return None

    def _selected_gallery_paths(self) -> list[str]:
        selection_model = self.list_view.selectionModel()
        if selection_model is None:
            return []
        selected_indexes = sorted(
            [index for index in selection_model.selectedIndexes() if index.isValid()],
            key=lambda index: index.row(),
        )
        paths: list[str] = []
        seen: set[str] = set()
        for index in selected_indexes:
            image_path = str(index.data(self.model.PathRole) or "")
            if not image_path or image_path in seen:
                continue
            seen.add(image_path)
            paths.append(image_path)
        return paths

    def on_item_clicked(self, index) -> None:
        image_path = index.data(self.model.PathRole)
        if image_path:
            self.image_selected.emit(image_path)

    def on_item_double_clicked(self, index) -> None:
        image_path = index.data(self.model.PathRole)
        if not image_path:
            return
        row = int(index.row())

        def _ctx(path: str) -> dict[str, object]:
            ctx: dict[str, object] = {"membership": self.membership_context.get(path, {})}
            extra = self.inspector_context_provider(path) if callable(self.inspector_context_provider) else None
            if isinstance(extra, dict):
                ctx.update(extra)
            return ctx

        dialog = PhotoInspectorDialog(
            image_path=None,
            image_paths=list(self.images),
            start_index=row,
            context={"metrics_by_backend": self.metrics_context},
            context_provider=_ctx,
            metadata_service=self.metadata_service,
            display_mode=self.inspector_display_mode,
            parent=self,
        )
        dialog.exec()

    def cancel_loader(self):
        self.refresh_timer.stop()
        self.flush_timer.stop()
        append_qt_diagnostic(f"[Gallery] cancel_loader thread_count={len(self.loader_threads)}")
        try:
            self.loader_queue.cancel()
        except Exception:
            pass
        for thread in list(self.loader_threads):
            try:
                if thread.isRunning():
                    thread.wait(2500)
                if thread.isRunning():
                    self._retain_loader_thread(thread)
                    append_qt_diagnostic(f"[Gallery] loader_still_running {thread.objectName()}")
                else:
                    append_qt_diagnostic(f"[Gallery] loader_stopped {thread.objectName()}")
            except Exception:
                self._retain_loader_thread(thread)
        self.loader_threads = []

    def shutdown_jobs(self, *, timeout_ms: int = 2500) -> bool:
        self._shutting_down = True
        self.refresh_timer.stop()
        self.flush_timer.stop()
        self.pending_indexes.clear()
        self.pending_ui_items.clear()
        ready_to_close = True
        action_pairs: list[tuple[object | None, object | None]] = []
        seen_action_threads: set[int] = set()
        for job, thread in [(self._active_action_job, self._active_action_thread), *self._retained_action_refs]:
            if thread is None:
                continue
            thread_id = id(thread)
            if thread_id in seen_action_threads:
                continue
            seen_action_threads.add(thread_id)
            action_pairs.append((job, thread))
        for job, thread in action_pairs:
            if job is not None:
                try:
                    job.cancel()
                except Exception:
                    pass
            try:
                if thread.isRunning():
                    thread.quit()
                    ready_to_close = bool(thread.wait(timeout_ms)) and ready_to_close
            except Exception:
                ready_to_close = False
        if ready_to_close:
            self._active_action_thread = None
            self._active_action_job = None
            self._retained_action_refs = []
            self._action_thread_jobs = {}
        self.cancel_loader()
        for thread in list(self._retained_loader_threads):
            try:
                if thread.isRunning():
                    ready_to_close = bool(thread.wait(timeout_ms)) and ready_to_close
            except Exception:
                ready_to_close = False
        self._retained_loader_threads = [thread for thread in self._retained_loader_threads if thread.isRunning()]
        if self.loader_threads or self._retained_loader_threads:
            ready_to_close = False
        return ready_to_close

    @staticmethod
    def _thread_is_running(thread) -> bool:
        if thread is None:
            return False
        try:
            return bool(thread.isRunning())
        except Exception:
            return False

    def _retain_action_refs(self, job: object | None, thread: object | None) -> None:
        if not self._thread_is_running(thread):
            return
        if any(existing_thread is thread for _existing_job, existing_thread in self._retained_action_refs):
            return
        self._retained_action_refs.append((job, thread))

    def _release_action_refs(self, job: object | None, thread: object | None) -> None:
        self._action_thread_jobs.pop(thread, None)
        self._retained_action_refs = [
            (existing_job, existing_thread)
            for existing_job, existing_thread in self._retained_action_refs
            if existing_thread is not thread
        ]
        if self._active_action_thread is thread:
            self._active_action_thread = None
            if self._active_action_job is job:
                self._active_action_job = None

    @pyqtSlot()
    def _on_action_thread_finished(self) -> None:
        thread = self.sender()
        if thread is None:
            return
        self._release_action_refs(self._action_thread_jobs.get(thread), thread)

    def _retain_loader_thread(self, thread: ImageLoaderThread) -> None:
        if not self._thread_is_running(thread):
            return
        if any(existing is thread for existing in self._retained_loader_threads):
            return
        self._retained_loader_threads.append(thread)

    def _release_loader_thread(self, thread: ImageLoaderThread) -> None:
        self.loader_threads = [existing for existing in self.loader_threads if existing is not thread]
        self._retained_loader_threads = [existing for existing in self._retained_loader_threads if existing is not thread]

    @pyqtSlot()
    def _on_loader_thread_finished(self) -> None:
        thread = self.sender()
        if thread is None:
            return
        self._release_loader_thread(thread)

    def _ensure_loader(self) -> None:
        if self._shutting_down:
            return
        desired = max(1, int(self.max_thumbnail_workers))
        running = [thread for thread in self.loader_threads if thread.isRunning()]
        if len(running) == desired and len(running) == len(self.loader_threads):
            return
        # Restart pool.
        self.cancel_loader()
        self.loader_queue = ThumbnailRequestQueue()
        self.loader_threads = []
        for _ in range(desired):
            thread = ImageLoaderThread(self.loader_queue, self.thumbnail_service, self)
            thread.image_loaded.connect(
                self.queueImageForGallery,
                Qt.ConnectionType.QueuedConnection,
            )
            thread.image_failed.connect(
                self.onImageLoadFailed,
                Qt.ConnectionType.QueuedConnection,
            )
            thread.finished.connect(
                self._on_loader_thread_finished,
                Qt.ConnectionType.QueuedConnection,
            )
            thread.start()
            self.loader_threads.append(thread)

    def schedule_visible_refresh(self):
        if self._shutting_down:
            return
        self.refresh_timer.start(self.settings.gallery_flush_interval_ms)

    def _visible_and_prefetch_indexes(self) -> tuple[list[int], list[int]]:
        total = len(self.images)
        if total == 0:
            return [], []
        grid = self.list_view.gridSize()
        cell_w = max(1, int(grid.width()))
        cell_h = max(1, int(grid.height()))
        viewport = self.list_view.viewport().rect()
        cols = max(1, int(viewport.width() // cell_w))
        total_rows = (total + cols - 1) // cols
        first_row = max(0, int(self.list_view.verticalScrollBar().value() // cell_h))
        visible_row_count = max(1, int(viewport.height() // cell_h) + 2)
        prefetch_rows = max(0, int(self.thumbnail_prefetch_rows))

        visible_start = min(total_rows - 1, first_row)
        visible_end = min(total_rows - 1, first_row + visible_row_count)
        prefetch_start = max(0, visible_start - prefetch_rows)
        prefetch_end = min(total_rows - 1, visible_end + prefetch_rows)

        def row_to_indexes(row: int) -> list[int]:
            start_index = row * cols
            end_index = min(total - 1, (row + 1) * cols - 1)
            if end_index < start_index:
                return []
            return list(range(start_index, end_index + 1))

        visible: list[int] = []
        for row in range(visible_start, visible_end + 1):
            visible.extend(row_to_indexes(row))

        prefetch: list[int] = []
        for row in range(prefetch_start, visible_start):
            prefetch.extend(row_to_indexes(row))
        for row in range(visible_end + 1, prefetch_end + 1):
            prefetch.extend(row_to_indexes(row))
        return visible, prefetch

    def load_visible_images(self):
        if self._shutting_down:
            return
        visible_indexes, prefetch_indexes = self._visible_and_prefetch_indexes()
        if not visible_indexes and not prefetch_indexes:
            return
        visible_missing = [
            index
            for index in visible_indexes
            if index not in self.loaded_indexes and index not in self.failed_indexes
        ]
        prefetch_missing = [
            index
            for index in prefetch_indexes
            if index not in self.loaded_indexes and index not in self.failed_indexes
        ]
        if not visible_missing and not prefetch_missing:
            self.update_status()
            return
        self._ensure_loader()
        self.pending_indexes.update(visible_missing)
        self.pending_indexes.update(prefetch_missing)
        self.update_status()
        # Visible items first, then prefetch.
        self.loader_queue.enqueue(visible_missing, priority=0)
        self.loader_queue.enqueue(prefetch_missing, priority=1)

    def queueImageForGallery(self, generation, index, image_path, qimage):
        if self._shutting_down:
            return
        if generation != self.request_generation:
            return
        self.pending_indexes.discard(index)
        self.pending_ui_items.append((index, image_path, qimage))
        if not self.flush_timer.isActive():
            self.flush_timer.start(self.settings.gallery_flush_interval_ms)

    def onImageLoadFailed(self, generation, index, image_path, message):
        if self._shutting_down:
            return
        if generation != self.request_generation:
            return
        self.pending_indexes.discard(index)
        self.failed_indexes.add(index)
        self.model.set_failed(index, message)
        LOGGER.warning("Thumbnail load failed for %s: %s", image_path, message)
        self.update_status(last_error=f"{image_path}: {message}")

    def closeEvent(self, event) -> None:
        if not self.shutdown_jobs():
            event.ignore()
            return
        return super().closeEvent(event)

    def flush_pending_ui_items(self):
        if not self.pending_ui_items:
            self.update_status()
            return
        self.pending_ui_items.sort(key=lambda item: item[0])
        for index, image_path, qimage in self.pending_ui_items:
            pixmap = self.get_or_create_pixmap(image_path, qimage)
            self.model.set_pixmap(index, pixmap)
            self.loaded_indexes.add(index)
            if not self.first_paint_emitted:
                self.first_paint_emitted = True
                latency_ms = int((perf_counter() - self.first_paint_start) * 1000)
                self.first_paint_ready.emit(latency_ms)
        self.pending_ui_items.clear()
        self.update_status()
        self.schedule_visible_refresh()

    def update_status(self, last_error: str = ""):
        if not self.images:
            self.status_label.setText("No images in selected cluster.")
            return
        loaded = len(self.loaded_indexes)
        failed = len(self.failed_indexes)
        if loaded + failed >= len(self.images):
            message = f"Loaded {loaded} / {len(self.images)} images."
            if failed:
                message += f" Failed: {failed}."
            if last_error:
                message += f" Last error: {last_error}"
            self.status_label.setText(message)
            return
        message = f"Loading {loaded} / {len(self.images)} images... Pending: {len(self.pending_indexes)}"
        if failed:
            message += f" Failed: {failed}."
        if last_error:
            message += f" Last error: {last_error}"
        self.status_label.setText(message)

    def get_or_create_pixmap(self, image_path, qimage):
        cache_key = (image_path, self.image_size)
        pixmap = self.pixmap_cache.get(cache_key)
        if pixmap is not None:
            self.pixmap_cache.move_to_end(cache_key)
            return pixmap
        pixmap = QPixmap.fromImage(qimage)
        self.pixmap_cache[cache_key] = pixmap
        self.pixmap_cache.move_to_end(cache_key)
        while len(self.pixmap_cache) > self._pixmap_cache_size:
            self.pixmap_cache.popitem(last=False)
        return pixmap

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.schedule_visible_refresh()

    def closeEvent(self, event) -> None:
        if not self.shutdown_jobs():
            event.ignore()
            return
        return super().closeEvent(event)


