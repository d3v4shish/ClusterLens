from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from PyQt6.QtCore import QItemSelectionModel, Qt, pyqtSignal
from PyQt6.QtWidgets import QAbstractItemView, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QListView, QPushButton, QSplitter, QVBoxLayout, QWidget

from ui.async_job import AsyncJob, start_job_in_thread, wait_for_thread_shutdown
from ui.common import HelpIconButton
from ui.error_mbox import confirmBox
from ui.gallery_pane import GalleryPane
from ui.list_models import ListEntry, ListEntryModel, PagedListEntryModel

if TYPE_CHECKING:
    from app.services.face_search import FaceIndexService, NamedPhotoSummary


NAMES_HELP = (
    "Names are durable face-to-person assignments stored in the global face database. "
    "Select a name to see every unique photo containing a face saved with that name. "
    "Right-click one or more selected photos to change only their relevant face labels: Name affects "
    "unlabeled faces, while Rename and Unlabel affect only the active saved name. "
    "Similarity-only matches remain in Faces > Find by Name."
)

SELECTED_IMAGES_HELP = (
    "Select photos with Ctrl-click, Shift-click, or their checkboxes. Name applies only to "
    "unlabeled visible faces in those photos. Rename and Unlabel apply only to faces that "
    "currently have the active saved name, so other people in the same photo are never changed."
)


class NamesPane(QWidget):
    """Global browser and bounded editor for durable saved face labels."""

    face_labels_changed = pyqtSignal()

    def __init__(self, face_service_provider: Callable[[], "FaceIndexService"], parent=None) -> None:
        super().__init__(parent)
        self._face_service_provider = face_service_provider
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
        self.names_list.setModel(self.names_model)
        self.names_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.names_list.setUniformItemSizes(True)
        self.names_list.setToolTip(NAMES_HELP)
        self.names_list.selectionModel().selectionChanged.connect(lambda *_args: self._on_name_selected())
        self.names_model.rowsInserted.connect(lambda *_args: self._update_count_label())
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
        self._entries = [
            ListEntry(
                title=f"{summary.person_name}  ·  {summary.photo_count} photo(s)",
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
                "Name Selected…",
                "Assign a name to unlabeled visible faces in the selected photos. Existing labels are preserved.",
                self._name_selected_images,
                editable,
            ),
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

    def _name_selected_images(self) -> None:
        paths = self._selected_image_paths()
        if not paths:
            return
        name, accepted = QInputDialog.getText(
            self,
            "Name selected faces",
            "Name for unlabeled visible face(s) in the selected photo(s):",
        )
        target = str(name or "").strip()
        if not accepted or not target:
            return
        self._start_selected_image_mutation(
            operation="name",
            target_name=target,
            image_paths=paths,
        )

    def _rename_selected_images(self) -> None:
        source = self._current_name() or self._selected_name
        paths = self._selected_image_paths()
        if not source or not paths:
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
        source = self._current_name() or self._selected_name
        paths = self._selected_image_paths()
        if not source or not paths:
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
                return service.rename_labeled_faces_in_images(source_name, target_name, paths)
            return service.unlabel_labeled_faces_in_images(source_name, paths)

        def _done(count) -> None:
            changed = int(count or 0)
            label = {"name": "Named", "rename": "Renamed", "unlabel": "Unlabeled"}.get(operation, "Updated")
            self.status_label.setText(f"{label} {changed} face(s). Refreshing saved names...")
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
            try:
                old_job.cancel()
            except Exception:
                pass
        job = AsyncJob(fn)
        setattr(self, f"_{slot}_job", job)
        if slot == "refresh":
            self.refresh_button.setEnabled(False)

        def _complete(result) -> None:
            if getattr(self, f"_{slot}_job", None) is not job:
                return
            setattr(self, f"_{slot}_job", None)
            if slot == "refresh":
                self.refresh_button.setEnabled(True)
            on_completed(result)

        def _fail(message: str) -> None:
            if getattr(self, f"_{slot}_job", None) is not job:
                return
            setattr(self, f"_{slot}_job", None)
            if slot == "refresh":
                self.refresh_button.setEnabled(True)
            on_failed(message)

        def _cancel() -> None:
            if getattr(self, f"_{slot}_job", None) is job:
                setattr(self, f"_{slot}_job", None)
                if slot == "refresh":
                    self.refresh_button.setEnabled(True)

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
        return self.gallery.shutdown_jobs(timeout_ms=timeout_ms) and ready
