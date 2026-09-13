from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING

from PyQt6.QtCore import QEvent, QItemSelectionModel, QModelIndex, QRect, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QAbstractItemView, QFrame, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListView, QProgressBar, QPushButton, QSplitter, QVBoxLayout, QWidget

from app.services.thumbnails import ThumbnailService
from ui.async_job import AsyncJob, start_job_in_thread, wait_for_thread_shutdown
from ui.common import HelpIconButton
from ui.error_mbox import confirmBox
from ui.gallery_pane import GalleryPane
from ui.job_manager import JobManager
from ui.list_models import ListEntry, ListEntryModel, PagedListEntryModel, SidebarListEntryDelegate
from ui.theme import COLORS

if TYPE_CHECKING:
    from app.services.face_search import FaceIndexService, NamedPhotoSummary


NAMES_HELP = (
    "Names are durable face-to-person assignments stored in the global face database. "
    "Select a name to see every unique photo containing a face saved with that name. "
    "Every photo in this view already has the active saved name. Right-click one or more selected photos "
    "to rename or unlabel only those matching face rows. "
    "Similarity-only matches remain in Faces > Find by Name."
)

SELECTED_IMAGES_HELP = (
    "Select photos with Ctrl-click, Shift-click, or their checkboxes. Rename and Unlabel apply only to "
    "faces that currently have the active saved name, so other people in the same photo are never changed."
)

NAME_HOVER_PREVIEW_IMAGE_SIZE = QSize(216, 216)
NAME_HOVER_PREVIEW_MAX_ITEMS = 9
NAME_HOVER_PREVIEW_CACHE_SIZE = 24


class NameHoverPreviewPopup(QFrame):
    """Non-interactive bounded preview for a durable saved-name sidebar row."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("nameHoverPreviewPopup")
        self.setStyleSheet(
            f"""
            QFrame#nameHoverPreviewPopup {{
                background: {COLORS["surface_raised"]};
                border: 1px solid {COLORS["border_strong"]};
                border-radius: 8px;
            }}
            QLabel {{ color: {COLORS["text"]}; }}
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.title_label = QLabel("")
        title_font = self.title_label.font()
        title_font.setBold(True)
        title_font.setPointSize(max(title_font.pointSize(), 10))
        self.title_label.setFont(title_font)
        self.title_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.title_label)

        self.image_label = QLabel("Loading preview...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(NAME_HOVER_PREVIEW_IMAGE_SIZE)
        self.image_label.setStyleSheet(
            f'background: {COLORS["surface_sunken"]}; border: 1px solid {COLORS["border"]};'
        )
        layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.summary_label)

        self.note_label = QLabel("")
        self.note_label.setStyleSheet(f'color: {COLORS["text_muted"]};')
        layout.addWidget(self.note_label)
        self.setFixedWidth(300)

    def set_name_details(self, name: str, *, face_count: int, photo_count: int, loading: bool) -> None:
        self.title_label.setText(str(name))
        self.summary_label.setText(f"{int(photo_count)} photo(s) · {int(face_count)} saved face(s)")
        self.note_label.setText(f"Showing up to {NAME_HOVER_PREVIEW_MAX_ITEMS} photos")
        if loading:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Loading preview...")

    def set_preview_image(self, image: QImage | None) -> None:
        if image is None or image.isNull():
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Preview unavailable")
            return
        self.image_label.setText("")
        self.image_label.setPixmap(QPixmap.fromImage(image))


class NamesPane(QWidget):
    """Global browser and bounded editor for durable saved face labels."""

    face_labels_changed = pyqtSignal()

    def __init__(
        self,
        face_service_provider: Callable[[], "FaceIndexService"],
        parent=None,
        *,
        job_manager: JobManager | None = None,
    ) -> None:
        super().__init__(parent)
        self._face_service_provider = face_service_provider
        self.job_manager = job_manager
        self._entries: list[ListEntry] = []
        self._selected_name = ""
        self._refresh_token = 0
        self._photos_token = 0
        self._refresh_job = None
        self._refresh_thread = None
        self._photos_job = None
        self._photos_thread = None
        self._mutation_job = None
        self._mutation_thread = None
        self._read_only_mode = False
        self._recovered_database_paths: set[str] = set()
        self._name_summaries: dict[str, "NamedPhotoSummary"] = {}
        self._name_preview_epoch = 0
        self._hover_popup = NameHoverPreviewPopup(self)
        self._hover_name = ""
        self._hover_index = QModelIndex()
        self._hover_preview_generation = 0
        self._hover_preview_job = None
        self._hover_preview_thread = None
        self._retained_hover_preview_refs: list[tuple[object | None, object | None]] = []
        self._hover_preview_cache: OrderedDict[tuple[object, ...], QImage] = OrderedDict()
        self._operation_job_ids: dict[str, int] = {}
        self._visible_operation_slot = ""
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title = QLabel("Names")
        title.setProperty("role", "section")
        title.setToolTip(NAMES_HELP)
        self.help_button = HelpIconButton(NAMES_HELP, self, help_key="names_workspace")
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setToolTip("Reload durable saved names and their photo counts from the global face database.")
        self.refresh_button.clicked.connect(self.refresh_names)
        title_row.addWidget(title)
        title_row.addWidget(self.help_button)
        title_row.addStretch(1)
        title_row.addWidget(self.refresh_button)
        layout.addLayout(title_row)

        self.status_label = QLabel("Open Names to load saved face labels.")
        self.status_label.setToolTip(NAMES_HELP)
        layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setAccessibleName("Names workspace operation progress")
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        sidebar = QWidget(splitter)
        sidebar.setMinimumWidth(220)
        sidebar.setMaximumWidth(360)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(6)
        sidebar_title_row = QHBoxLayout()
        sidebar_title_row.setContentsMargins(0, 0, 0, 0)
        sidebar_title = QLabel("Saved names")
        sidebar_title.setToolTip(NAMES_HELP)
        sidebar_title_row.addWidget(sidebar_title)
        sidebar_title_row.addWidget(HelpIconButton(NAMES_HELP, sidebar, help_key="saved_names"))
        sidebar_title_row.addStretch(1)
        sidebar_layout.addLayout(sidebar_title_row)
        self.search_field = QLineEdit(sidebar)
        self.search_field.setPlaceholderText("Filter names")
        self.search_field.setToolTip("Filter saved names. This does not search visually similar people.")
        self.search_field.setAccessibleName("Filter saved names")
        self.search_field.textChanged.connect(self._apply_filter)
        sidebar_layout.addWidget(self.search_field)
        self.count_label = QLabel("0 names")
        self.count_label.setToolTip("Number of saved names matching the current filter.")
        sidebar_layout.addWidget(self.count_label)
        self.names_model = PagedListEntryModel(self, page_size=50)
        self.names_list = QListView(sidebar)
        self.names_list.setObjectName("namesWorkspaceList")
        self.names_list.setProperty("sidebarList", True)
        self.names_list.setModel(self.names_model)
        self.names_list.setItemDelegate(SidebarListEntryDelegate(self.names_list))
        self.names_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.names_list.setUniformItemSizes(True)
        self.names_list.setToolTip(NAMES_HELP)
        self.names_list.setMouseTracking(True)
        self.names_list.viewport().setMouseTracking(True)
        self.names_list.viewport().setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        self.names_list.viewport().installEventFilter(self)
        self.names_list.verticalScrollBar().valueChanged.connect(lambda _value: self._hide_name_hover_preview(cancel_preview=True))
        self.names_list.selectionModel().selectionChanged.connect(lambda *_args: self._on_name_selected())
        self.names_model.rowsInserted.connect(lambda *_args: self._update_count_label())
        self.names_model.modelReset.connect(lambda: self._hide_name_hover_preview(cancel_preview=True))
        sidebar_layout.addWidget(self.names_list, stretch=1)
        splitter.addWidget(sidebar)

        self.gallery = GalleryPane(splitter)
        self.gallery.setObjectName("namesWorkspaceGallery")
        self.gallery.set_action_bar_visible(False)
        self.gallery.set_empty_state(
            "Choose a saved name",
            "Select a name in the sidebar to show photos with a durable face label.",
            show_select_folder=False,
            show_run=False,
        )
        self.gallery.set_context_menu_action_provider(self._names_context_menu_actions)
        splitter.addWidget(self.gallery)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 9)
        splitter.setSizes([280, 1060])
        layout.addWidget(splitter, stretch=1)

    def set_job_manager(self, job_manager: JobManager | None) -> None:
        """Attach the shared Jobs registry after lazy workspace construction."""

        self.job_manager = job_manager

    def _set_operation_progress(self, slot: str, value: int, text: str) -> None:
        self._visible_operation_slot = str(slot)
        if int(value) < 0:
            self.progress_bar.setRange(0, 0)
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(max(0, min(100, int(value))))
        self.progress_bar.show()
        if text:
            self.status_label.setText(str(text))

    def _finish_operation_progress(self, slot: str) -> None:
        if self._visible_operation_slot != str(slot):
            return
        self.progress_bar.hide()
        self._visible_operation_slot = ""

    def set_read_only_mode(self, enabled: bool) -> None:
        self._read_only_mode = bool(enabled)
        self.gallery.set_read_only_mode(self._read_only_mode)

    def refresh_names(self, *, preserve_name: str | None = None) -> None:
        """Load names asynchronously and recover legacy explicit manual labels once."""

        self._refresh_token += 1
        token = self._refresh_token
        selected_name = str(preserve_name or "").strip() or self._current_name() or self._selected_name
        service = self._face_service_provider()
        database_key = str(getattr(service, "db_path", "") or id(service))
        recover = database_key not in self._recovered_database_paths
        self.refresh_button.setEnabled(False)
        self.status_label.setText("Loading saved names...")

        def _run(progress, cancel_check):
            progress(-1, "Loading saved names...")
            recovery = None
            if recover:
                recovery = service.recover_legacy_manual_face_labels()
            if cancel_check():
                return None
            return recovery, service.list_named_photo_summaries()

        def _done(result) -> None:
            if token != self._refresh_token or result is None:
                return
            recovery, summaries = result
            if recover:
                self._recovered_database_paths.add(database_key)
            self._set_summaries(list(summaries or []), preserve_name=selected_name)
            recovery_text = ""
            if recovery is not None and int(getattr(recovery, "promoted_count", 0) or 0):
                recovery_text = f" Recovered {int(recovery.promoted_count)} earlier manual name(s)."
            conflicts = int(getattr(recovery, "conflict_count", 0) or 0) if recovery is not None else 0
            if conflicts:
                recovery_text += f" {conflicts} conflicting older proposal(s) remain in review."
            self.status_label.setText(f"{len(list(summaries or []))} saved name(s).{recovery_text}")

        self._start_job("refresh", _run, _done, lambda _message: self.status_label.setText("Could not load saved names."))

    def _set_summaries(self, summaries: list["NamedPhotoSummary"], *, preserve_name: str) -> None:
        self._name_summaries = {
            str(summary.person_name): summary
            for summary in summaries
            if str(summary.person_name or "").strip()
        }
        self._name_preview_epoch += 1
        self._hover_preview_cache.clear()
        self._hide_name_hover_preview(cancel_preview=True)
        self._entries = [
            ListEntry(
                title=str(summary.person_name),
                subtitle=f"{summary.photo_count} photo(s) · {summary.face_count} face(s)",
                tooltip=(
                    f"{summary.person_name}: {summary.face_count} saved face(s) across "
                    f"{summary.photo_count} unique photo(s)."
                ),
                payload=summary.person_name,
            )
            for summary in summaries
            if str(summary.person_name or "").strip()
        ]
        self.names_model.set_source_items(self._entries)
        self._update_count_label()
        self._select_name(preserve_name)

    def _apply_filter(self) -> None:
        preserve_name = self._current_name() or self._selected_name
        self.names_model.set_filter_text(self.search_field.text())
        self._update_count_label()
        self._select_name(preserve_name)

    def _update_count_label(self) -> None:
        count = self.names_model.total_count
        self.count_label.setText(f"{count} name{'s' if count != 1 else ''}")

    def _select_name(self, name: str) -> None:
        row = self.names_model.row_for_payload(str(name or "")) if name else -1
        if row < 0 and self.names_model.rowCount():
            row = 0
        if row < 0:
            self._selected_name = ""
            self.gallery.update_gallery_with_options(images=[], clear_pixmaps=False, reset_scroll=True)
            self.gallery.set_empty_state(
                "No saved names",
                "Name faces in Faces to add them here.",
                show_select_folder=False,
                show_run=False,
            )
            return
        index = self.names_model.index(row, 0)
        self.names_list.setCurrentIndex(index)
        self.names_list.selectionModel().select(
            index,
            QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows,
        )

    def _current_name(self) -> str:
        index = self.names_list.currentIndex()
        if not index.isValid():
            return ""
        return str(index.data(ListEntryModel.PayloadRole) or "").strip()

    def eventFilter(self, watched, event):  # type: ignore[override]
        if watched is self.names_list.viewport():
            if event.type() == QEvent.Type.ToolTip:
                return True
            if event.type() == QEvent.Type.MouseMove:
                self._update_name_hover_preview(self.names_list.indexAt(event.pos()))
            elif event.type() in {
                QEvent.Type.Leave,
                QEvent.Type.Hide,
                QEvent.Type.Wheel,
                QEvent.Type.MouseButtonPress,
            }:
                self._hide_name_hover_preview(cancel_preview=True)
        return super().eventFilter(watched, event)

    def _update_name_hover_preview(self, index: QModelIndex) -> None:
        if not index.isValid():
            self._hide_name_hover_preview(cancel_preview=True)
            return
        name = str(index.data(ListEntryModel.PayloadRole) or "").strip()
        summary = self._name_summaries.get(name)
        if not name or summary is None:
            self._hide_name_hover_preview(cancel_preview=True)
            return
        if self._hover_name == name and self._hover_popup.isVisible():
            self._hover_index = index
            self._position_name_hover_preview()
            return
        self._cancel_name_hover_preview_job()
        self._hover_name = name
        self._hover_index = index
        self._hover_preview_generation += 1
        generation = self._hover_preview_generation
        face_count = int(getattr(summary, "face_count", 0) or 0)
        photo_count = int(getattr(summary, "photo_count", 0) or 0)
        self._hover_popup.set_name_details(
            name,
            face_count=face_count,
            photo_count=photo_count,
            loading=True,
        )
        self._position_name_hover_preview()
        self._hover_popup.show()
        cache_key = (self._name_preview_epoch, name, face_count, photo_count)
        cached = self._hover_preview_cache.get(cache_key)
        if cached is not None:
            self._hover_preview_cache.move_to_end(cache_key)
            self._apply_name_hover_preview(cached, generation)
            return
        self._start_name_hover_preview_job(name, cache_key, generation)

    def _start_name_hover_preview_job(
        self,
        name: str,
        cache_key: tuple[object, ...],
        generation: int,
    ) -> None:
        service = self._face_service_provider()

        def _run(_progress, cancel_check):
            if cancel_check():
                return None
            paths = list(service.list_named_photo_paths(name) or [])
            if cancel_check():
                return None
            image = ThumbnailService(qimage_cache_size=32).build_contact_sheet_qimage(
                paths[:NAME_HOVER_PREVIEW_MAX_ITEMS],
                NAME_HOVER_PREVIEW_IMAGE_SIZE,
                max_items=NAME_HOVER_PREVIEW_MAX_ITEMS,
                columns=3,
                cancel_check=cancel_check,
            )
            if cancel_check():
                return None
            return {"generation": generation, "cache_key": cache_key, "image": image}

        job = AsyncJob(_run)

        def _completed(payload) -> None:
            if not isinstance(payload, dict) or int(payload.get("generation", -1)) != self._hover_preview_generation:
                return
            image = payload.get("image")
            if not isinstance(image, QImage):
                image = QImage()
            self._remember_name_hover_preview(cache_key, image)
            self._apply_name_hover_preview(image, generation)

        def _failed(_message: str) -> None:
            if generation == self._hover_preview_generation:
                self._hover_popup.set_preview_image(None)

        def _cancelled() -> None:
            if generation == self._hover_preview_generation:
                self._hover_popup.set_preview_image(None)

        job.completed.connect(_completed)
        job.failed.connect(_failed)
        job.cancelled.connect(_cancelled)
        self._hover_preview_job = job
        thread = start_job_in_thread(job)
        self._hover_preview_thread = thread
        thread.finished.connect(lambda thread=thread, job=job: self._on_name_hover_preview_thread_finished(job, thread))

    def _apply_name_hover_preview(self, image: QImage, generation: int) -> None:
        if generation != self._hover_preview_generation or not self._hover_name:
            return
        self._hover_popup.set_preview_image(image if not image.isNull() else None)
        self._position_name_hover_preview()
        self._hover_popup.show()

    def _remember_name_hover_preview(self, cache_key: tuple[object, ...], image: QImage) -> None:
        self._hover_preview_cache[cache_key] = image
        self._hover_preview_cache.move_to_end(cache_key)
        while len(self._hover_preview_cache) > NAME_HOVER_PREVIEW_CACHE_SIZE:
            self._hover_preview_cache.popitem(last=False)

    def _position_name_hover_preview(self) -> None:
        if not self._hover_index.isValid():
            return
        rect = self.names_list.visualRect(self._hover_index)
        if not rect.isValid():
            return
        anchor = self.names_list.viewport().mapToGlobal(rect.topRight())
        self._hover_popup.adjustSize()
        popup_rect = self._hover_popup.frameGeometry()
        screen = self.names_list.screen() or QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else QRect(anchor, popup_rect.size())
        margin = 12
        target_x = anchor.x() + margin
        target_y = anchor.y() + margin
        if target_x + popup_rect.width() > available.right() - margin:
            left_anchor = self.names_list.viewport().mapToGlobal(rect.topLeft())
            target_x = left_anchor.x() - popup_rect.width() - margin
        if target_y + popup_rect.height() > available.bottom() - margin:
            target_y = max(available.top() + margin, available.bottom() - popup_rect.height() - margin)
        self._hover_popup.move(max(available.left() + margin, target_x), target_y)

    def _hide_name_hover_preview(self, *, cancel_preview: bool) -> None:
        if cancel_preview:
            self._cancel_name_hover_preview_job()
            self._hover_preview_generation += 1
        self._hover_popup.hide()
        self._hover_name = ""
        self._hover_index = QModelIndex()

    def _cancel_name_hover_preview_job(self) -> None:
        job = self._hover_preview_job
        thread = self._hover_preview_thread
        if job is not None:
            if thread is not None:
                try:
                    if thread.isRunning() and not any(existing_thread is thread for _existing_job, existing_thread in self._retained_hover_preview_refs):
                        self._retained_hover_preview_refs.append((job, thread))
                except Exception:
                    pass
            try:
                job.cancel()
            except Exception:
                pass
        self._hover_preview_job = None
        self._hover_preview_thread = None

    def _on_name_hover_preview_thread_finished(self, job, thread) -> None:
        self._retained_hover_preview_refs = [
            (existing_job, existing_thread)
            for existing_job, existing_thread in self._retained_hover_preview_refs
            if existing_thread is not thread
        ]
        if self._hover_preview_thread is thread:
            self._hover_preview_thread = None
            if self._hover_preview_job is job:
                self._hover_preview_job = None

    def _on_name_selected(self) -> None:
        name = self._current_name()
        if not name or name == self._selected_name and self._photos_thread is not None:
            return
        self._selected_name = name
        self._photos_token += 1
        token = self._photos_token
        service = self._face_service_provider()
        self.status_label.setText(f"Loading photos for {name}...")

        def _run(progress, cancel_check):
            progress(-1, f"Loading photos for {name}...")
            paths = service.list_named_photo_paths(name)
            if cancel_check():
                return None
            return paths

        def _done(paths) -> None:
            if token != self._photos_token or name != self._selected_name or paths is None:
                return
            self.gallery.update_gallery_with_options(images=list(paths), clear_pixmaps=True, reset_scroll=True)
            self.gallery.set_empty_state(
                "No labeled photos",
                f"No visible saved face labels remain for {name}.",
                show_select_folder=False,
                show_run=False,
            )
            self.status_label.setText(f"{name} · {len(paths)} unique photo(s)")

        self._start_job("photos", _run, _done, lambda _message: self.status_label.setText(f"Could not load photos for {name}."))

    def _selected_image_paths(self) -> list[str]:
        selected = self.gallery._selected_gallery_paths()
        if selected:
            return selected
        return list(self.gallery.model.checked_paths())

    def _names_context_menu_actions(self, _image_path: str) -> list[tuple[str, str, Callable[[], None], bool]]:
        """Provide face-safe label actions after Gallery prepares right-click selection."""

        selected_count = len(self._selected_image_paths())
        active_name = self._current_name() or self._selected_name
        busy = self._mutation_job is not None
        editable = selected_count > 0 and not busy and not self._read_only_mode
        return [
            (
                "Rename Selected…",
                "Move only the active name's face labels in the selected photos to a different saved name.",
                self._rename_selected_images,
                editable and bool(active_name),
            ),
            (
                "Unlabel Selected",
                "Remove only the active name's face labels in the selected photos. Other people are unchanged.",
                self._unlabel_selected_images,
                editable and bool(active_name),
            ),
        ]

    def _rename_selected_images(self) -> None:
        paths = self._selected_image_paths()
        if not paths:
            return
        source = self._choose_metadata_source_name(paths, preferred=self._current_name() or self._selected_name)
        if not source:
            return
        name, accepted = QInputDialog.getText(
            self,
            "Rename selected face labels",
            f"New name for {source} face label(s) in the selected photo(s):",
        )
        target = str(name or "").strip()
        if not accepted or not target or target == source:
            return
        self._start_selected_image_mutation(
            operation="rename",
            source_name=source,
            target_name=target,
            image_paths=paths,
        )

    def _unlabel_selected_images(self) -> None:
        paths = self._selected_image_paths()
        if not paths:
            return
        source = self._choose_metadata_source_name(paths, preferred=self._current_name() or self._selected_name)
        if not source:
            return
        if not confirmBox(
            "Unlabel selected faces",
            (
                f"Remove {source} only from matching face(s) in {len(paths)} selected photo(s)? "
                "Other face labels in these photos will not change."
            ),
            parent=self,
        ):
            return
        self._start_selected_image_mutation(
            operation="unlabel",
            source_name=source,
            image_paths=paths,
        )

    def _choose_metadata_source_name(self, paths: list[str], *, preferred: str = "") -> str:
        """Choose a name physically present in the selected photos' regions."""

        service = self._face_service_provider()
        names_by_path: dict[str, tuple[str, ...]] = {}
        list_names = getattr(service, "face_region_names_for_paths", None)
        if not callable(list_names):
            return str(preferred or "").strip()
        try:
            names_by_path = dict(list_names(paths) or {})
        except Exception:
            names_by_path = {}
        counts: dict[str, tuple[int, int]] = {}
        for path, names in names_by_path.items():
            region_names = [str(value or "").strip() for value in names if str(value or "").strip()]
            for name in region_names:
                face_count, photo_count = counts.get(name, (0, 0))
                counts[name] = (face_count + 1, photo_count)
            for name in set(region_names):
                face_count, photo_count = counts.get(name, (0, 0))
                counts[name] = (face_count, photo_count + 1)
        preferred = str(preferred or "").strip()
        if preferred and preferred not in counts:
            # A legacy database label has no region yet. Keeping it available
            # lets the next mutation migrate it into face metadata.
            counts[preferred] = (0, 0)
        if not counts:
            return ""
        names = sorted(counts, key=lambda value: (value.casefold(), value))
        labels = [
            f"{name} — {counts[name][0]} region(s) in {counts[name][1]} photo(s)" if counts[name][0] else f"{name} — saved legacy label"
            for name in names
        ]
        selected_index = names.index(preferred) if preferred in names else 0
        selected, accepted = QInputDialog.getItem(
            self,
            "Choose face-region name",
            "Name currently stored in selected photo regions:",
            labels,
            selected_index,
            False,
        )
        if not accepted:
            return ""
        try:
            return names[labels.index(str(selected))]
        except ValueError:
            return ""

    def _start_selected_image_mutation(
        self,
        *,
        operation: str,
        image_paths: list[str],
        source_name: str = "",
        target_name: str = "",
    ) -> None:
        if self._read_only_mode:
            return
        service = self._face_service_provider()
        paths = list(dict.fromkeys(str(path) for path in image_paths if str(path or "").strip()))
        if not paths:
            return
        self.status_label.setText("Saving selected face labels...")

        def _run(progress, cancel_check):
            progress(-1, "Saving selected face labels...")
            if operation == "name":
                return service.label_unlabeled_faces_in_images(target_name, paths)
            if operation == "rename":
                rename_regions = getattr(service, "rename_face_regions_in_images", None)
                return rename_regions(source_name, target_name, paths) if callable(rename_regions) else service.rename_labeled_faces_in_images(source_name, target_name, paths)
            unlabel_regions = getattr(service, "unlabel_face_regions_in_images", None)
            return unlabel_regions(source_name, paths) if callable(unlabel_regions) else service.unlabel_labeled_faces_in_images(source_name, paths)

        def _done(result) -> None:
            refs = list(getattr(result, "affected_refs", ()) or ())
            changed = len(refs) if refs else int(result or 0) if isinstance(result, int) else 0
            failures = len(getattr(result, "failures", ()) or ())
            label = {"name": "Named", "rename": "Renamed", "unlabel": "Unlabeled"}.get(operation, "Updated")
            suffix = f" ({failures} metadata failure(s))" if failures else ""
            self.status_label.setText(f"{label} {changed} face region(s){suffix}. Refreshing saved names...")
            self._selected_name = target_name if target_name else source_name
            self.face_labels_changed.emit()
            self.refresh_names(preserve_name=self._selected_name)

        self._start_job(
            "mutation",
            _run,
            _done,
            lambda _message: self.status_label.setText("Could not save the selected face labels."),
        )

    def _start_job(self, slot: str, fn, on_completed, on_failed) -> None:
        old_job = getattr(self, f"_{slot}_job", None)
        if old_job is not None:
            old_job_id = self._operation_job_ids.pop(str(slot), None)
            if old_job_id is not None and self.job_manager is not None:
                self.job_manager.finish(old_job_id, status="cancelled")
            try:
                old_job.cancel()
            except Exception:
                pass
        job = AsyncJob(fn)
        setattr(self, f"_{slot}_job", job)
        job_id: int | None = None
        if self.job_manager is not None:
            label = {
                "refresh": "Loading saved names",
                "photos": "Loading named photos",
                "mutation": "Saving face-region names",
            }.get(str(slot), "Names workspace work")
            job_id = self.job_manager.register_job(
                label,
                cancel_fn=job.cancel,
                origin="Names",
                foreground=str(slot) in {"refresh", "mutation"},
            )
            self._operation_job_ids[str(slot)] = job_id
            job.progress.connect(
                lambda value, text, job_id=job_id: self.job_manager.update(
                    job_id,
                    progress=value,
                    text=str(text),
                )
            )
        job.progress.connect(
            lambda value, text, slot=slot: self._set_operation_progress(str(slot), int(value), str(text))
        )

        def _finish_registered_job(status: str, error: str = "") -> None:
            current_job_id = self._operation_job_ids.pop(str(slot), None)
            if current_job_id is not None and self.job_manager is not None:
                self.job_manager.finish(current_job_id, status=status, error=error)
            self._finish_operation_progress(str(slot))

        if slot == "refresh":
            self.refresh_button.setEnabled(False)

        def _complete(result) -> None:
            if getattr(self, f"_{slot}_job", None) is not job:
                return
            setattr(self, f"_{slot}_job", None)
            if slot == "refresh":
                self.refresh_button.setEnabled(True)
            _finish_registered_job("finished")
            on_completed(result)

        def _fail(message: str) -> None:
            if getattr(self, f"_{slot}_job", None) is not job:
                return
            setattr(self, f"_{slot}_job", None)
            if slot == "refresh":
                self.refresh_button.setEnabled(True)
            _finish_registered_job("failed", str(message))
            on_failed(message)

        def _cancel() -> None:
            if getattr(self, f"_{slot}_job", None) is job:
                setattr(self, f"_{slot}_job", None)
                if slot == "refresh":
                    self.refresh_button.setEnabled(True)
                _finish_registered_job("cancelled")

        job.completed.connect(_complete)
        job.failed.connect(_fail)
        job.cancelled.connect(_cancel)
        thread = start_job_in_thread(job)
        setattr(self, f"_{slot}_thread", thread)
        thread.finished.connect(lambda thread=thread, slot=slot: self._on_thread_finished(slot, thread))

    def _on_thread_finished(self, slot: str, thread) -> None:
        if getattr(self, f"_{slot}_thread", None) is thread:
            setattr(self, f"_{slot}_thread", None)

    def shutdown_jobs(self, *, timeout_ms: int = 2500) -> bool:
        ready = True
        self._hide_name_hover_preview(cancel_preview=True)
        for slot in ("refresh", "photos", "mutation"):
            job = getattr(self, f"_{slot}_job", None)
            thread = getattr(self, f"_{slot}_thread", None)
            if job is not None:
                try:
                    job.cancel()
                except Exception:
                    ready = False
            if thread is not None:
                ready = wait_for_thread_shutdown(thread, timeout_ms=timeout_ms) and ready
        self._refresh_job = None
        self._refresh_thread = None
        self._photos_job = None
        self._photos_thread = None
        self._mutation_job = None
        self._mutation_thread = None
        for _job, thread in list(self._retained_hover_preview_refs):
            if thread is not None:
                ready = wait_for_thread_shutdown(thread, timeout_ms=timeout_ms) and ready
        self._retained_hover_preview_refs.clear()
        self._hover_preview_cache.clear()
        return self.gallery.shutdown_jobs(timeout_ms=timeout_ms) and ready
