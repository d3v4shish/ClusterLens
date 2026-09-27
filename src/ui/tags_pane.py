from __future__ import annotations

from collections.abc import Callable

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListView,
    QMessageBox,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from app.path_scope import PathScope
from app.services.image_tags import ImageTagService, TagInventoryItem
from ui.async_job import AsyncJob, defer_async_job_dispose, start_job_in_thread, wait_for_thread_shutdown
from ui.gallery_pane import GalleryPane
from ui.job_manager import JobManager
from ui.list_models import ListEntry, ListEntryModel, SidebarListEntryDelegate
from ui.work_coordinator import JobSpec, WorkCoordinator


class TagsPane(QWidget):
    """Background-backed browser and editor for the durable image-tag database."""

    run_tag_filter_requested = pyqtSignal(list, str)
    generate_suggestions_requested = pyqtSignal()
    apply_suggestions_requested = pyqtSignal()
    metadata_changed = pyqtSignal(list)
    open_in_gallery_requested = pyqtSignal(list, str)

    INVENTORY_PAGE_SIZE = 100
    PHOTO_PAGE_SIZE = 200

    def __init__(
        self,
        tag_service: ImageTagService,
        current_scope_provider: Callable[[], object],
        parent=None,
        *,
        job_manager: JobManager | None = None,
        work_coordinator: WorkCoordinator | None = None,
    ) -> None:
        super().__init__(parent)
        self.tag_service = tag_service
        self.current_scope_provider = current_scope_provider
        self.job_manager = job_manager
        self.work_coordinator = work_coordinator
        self._read_only_mode = False
        self._inventory_offset = 0
        self._inventory_total = 0
        self._photo_offset = 0
        self._photo_total = 0
        self._selected_tag = ""
        self._filter_tags: list[str] = []
        self._generation = 0
        self._inventory_loading = False
        self._inventory_refresh_pending = False
        self._photo_loading = False
        self._photo_reset_pending = False
        self._mutation_loading = False
        self._jobs: list[tuple[object, object]] = []
        self._coordinated_job_ids: dict[AsyncJob, int] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        title_row = QHBoxLayout()
        title = QLabel("Tags")
        title.setProperty("role", "section")
        title_row.addWidget(title)
        title_row.addStretch(1)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.setToolTip("Reload the durable tag database without scanning or modifying photos.")
        self.refresh_button.clicked.connect(self.refresh)
        title_row.addWidget(self.refresh_button)
        layout.addLayout(title_row)

        self.status_label = QLabel("Open Tags to load the durable tag library.")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        sidebar = QWidget(splitter)
        sidebar.setMinimumWidth(260)
        sidebar.setMaximumWidth(390)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(0, 0, 0, 0)
        self.scope_combo = QComboBox(sidebar)
        self.scope_combo.addItem("All tagged photos", "")
        self.scope_combo.addItem("Active roots", "active")
        self.scope_combo.currentIndexChanged.connect(self.refresh)
        side.addWidget(QLabel("Scope"))
        side.addWidget(self.scope_combo)
        self.search_field = QLineEdit(sidebar)
        self.search_field.setPlaceholderText("Filter tags")
        self.search_field.setToolTip("Search durable tag names in the active scope.")
        self.search_field.returnPressed.connect(self.refresh)
        side.addWidget(self.search_field)
        self.inventory_count_label = QLabel("0 tags")
        side.addWidget(self.inventory_count_label)
        self.inventory_model = ListEntryModel(sidebar)
        self.inventory_list = QListView(sidebar)
        self.inventory_list.setAccessibleName("Tags")
        self.inventory_list.setModel(self.inventory_model)
        self.inventory_list.setItemDelegate(SidebarListEntryDelegate(self.inventory_list))
        self.inventory_list.setUniformItemSizes(True)
        self.inventory_list.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self.inventory_list.selectionModel().selectionChanged.connect(lambda *_args: self._on_tag_selected())
        side.addWidget(self.inventory_list, stretch=1)
        self.more_tags_button = QPushButton("Load more tags")
        self.more_tags_button.clicked.connect(self._load_more_inventory)
        side.addWidget(self.more_tags_button)

        rename_form = QFormLayout()
        self.rename_field = QLineEdit(sidebar)
        self.rename_field.setPlaceholderText("New tag name")
        rename_form.addRow("Rename / merge", self.rename_field)
        side.addLayout(rename_form)
        actions = QHBoxLayout()
        self.rename_button = QPushButton("Rename")
        self.delete_button = QPushButton("Delete")
        self.rename_button.setAccessibleName("Rename tag globally")
        self.delete_button.setAccessibleName("Delete tag globally")
        self.rename_button.clicked.connect(self._rename_selected)
        self.delete_button.clicked.connect(self._delete_selected)
        actions.addWidget(self.rename_button)
        actions.addWidget(self.delete_button)
        side.addLayout(actions)
        splitter.addWidget(sidebar)

        body = QWidget(splitter)
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)
        self.photo_heading = QLabel("Choose a tag")
        photo_heading_row = QHBoxLayout()
        photo_heading_row.addWidget(self.photo_heading, stretch=1)
        self.open_in_gallery_button = QPushButton("Open photos")
        self.open_in_gallery_button.setToolTip("Open the selected tag's currently loaded photos as the current photo set.")
        self.open_in_gallery_button.setAccessibleName("Open tagged photos")
        self.open_in_gallery_button.setEnabled(False)
        self.open_in_gallery_button.clicked.connect(self._open_tagged_photos_in_gallery)
        photo_heading_row.addWidget(self.open_in_gallery_button)
        body_layout.addLayout(photo_heading_row)
        self.gallery = GalleryPane(body)
        self.gallery.configure_jobs(self.job_manager, self.work_coordinator, origin="Organize")
        self.gallery.image_tag_service = self.tag_service
        self.gallery.set_action_visibility(show_actions=True, show_metadata_actions=True, show_file_actions=False)
        self.gallery.metadata_changed.connect(self._on_gallery_metadata_changed)
        self.gallery.set_empty_state("Choose a tag", "Select a tag to browse its tagged photos.")
        body_layout.addWidget(self.gallery, stretch=1)
        self.more_photos_button = QPushButton("Load more photos")
        self.more_photos_button.clicked.connect(self._load_more_photos)
        body_layout.addWidget(self.more_photos_button)

        filter_row = QHBoxLayout()
        self.filter_label = QLabel("Filter: none")
        self.add_filter_button = QPushButton("Add selected tag")
        self.remove_filter_button = QPushButton("Clear filter")
        self.match_combo = QComboBox(body)
        self.match_combo.addItems(["Any", "All"])
        self.run_filter_button = QPushButton("Cluster matches")
        self.run_filter_button.setAccessibleName("Cluster matching photos")
        filter_row.addWidget(self.filter_label, stretch=1)
        filter_row.addWidget(self.add_filter_button)
        filter_row.addWidget(self.remove_filter_button)
        filter_row.addWidget(self.match_combo)
        filter_row.addWidget(self.run_filter_button)
        self.add_filter_button.clicked.connect(self._add_selected_to_filter)
        self.remove_filter_button.clicked.connect(self._clear_filter)
        self.run_filter_button.clicked.connect(self._run_filter)
        body_layout.addLayout(filter_row)

        suggestion_row = QHBoxLayout()
        self.suggestion_label = QLabel("Suggestions: select a group.")
        self.generate_suggestions_button = QPushButton("Generate suggestions")
        self.apply_suggestions_button = QPushButton("Apply suggestions")
        self.generate_suggestions_button.clicked.connect(self.generate_suggestions_requested.emit)
        self.apply_suggestions_button.clicked.connect(self.apply_suggestions_requested.emit)
        suggestion_row.addWidget(self.suggestion_label, stretch=1)
        suggestion_row.addWidget(self.generate_suggestions_button)
        suggestion_row.addWidget(self.apply_suggestions_button)
        body_layout.addLayout(suggestion_row)
        splitter.addWidget(body)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 8)
        layout.addWidget(splitter, stretch=1)
        self._update_controls()

    def set_read_only_mode(self, enabled: bool) -> None:
        self._read_only_mode = bool(enabled)
        self.gallery.set_read_only_mode(self._read_only_mode)
        self._update_controls()

    def set_cluster_suggestion_state(self, text: str, *, can_generate: bool, can_apply: bool) -> None:
        self.suggestion_label.setText(str(text))
        self.generate_suggestions_button.setEnabled(bool(can_generate))
        self.apply_suggestions_button.setEnabled(bool(can_apply and not self._read_only_mode))

    def selected_filter_tags(self) -> list[str]:
        return list(self._filter_tags)

    def selected_match_mode(self) -> str:
        return self.match_combo.currentText().strip() or "Any"

    def set_filter(self, tags: list[str], match: str = "Any") -> None:
        self._filter_tags = self.tag_service.parse_tag_text(",".join(str(tag) for tag in tags))
        self.match_combo.setCurrentText("All" if str(match).casefold() == "all" else "Any")
        self._update_filter_label()

    def refresh(self) -> None:
        self._generation += 1
        self._inventory_offset = 0
        self._inventory_total = 0
        self._photo_offset = 0
        self._photo_total = 0
        self._selected_tag = ""
        self.inventory_model.set_items([])
        self.inventory_count_label.setText("0 tags")
        self.gallery.update_gallery([])
        self.open_in_gallery_button.setEnabled(False)
        self.photo_heading.setText("Choose a tag")
        if self._inventory_loading:
            self._inventory_refresh_pending = True
            self.status_label.setText("Refreshing tag inventory…")
            self._update_controls()
            return
        self._load_inventory(reset=True)

    def _scope_paths(self) -> tuple[str, ...] | None:
        if self.scope_combo.currentData() != "active":
            return None
        try:
            value = self.current_scope_provider()
        except Exception:
            return ()
        if isinstance(value, PathScope):
            return value.roots
        if isinstance(value, (list, tuple, set)):
            return PathScope.from_paths(value).roots
        return PathScope.from_paths([str(value or "")]).roots

    def _active_scope_is_unavailable(self) -> bool:
        return self.scope_combo.currentData() == "active" and not self._scope_paths()

    def _load_inventory(self, *, reset: bool = False) -> None:
        if self._active_scope_is_unavailable():
            self.inventory_model.set_items([])
            self._inventory_total = 0
            self._inventory_offset = 0
            self.inventory_count_label.setText("0 tags")
            self.status_label.setText("Choose a folder or one or more active roots to view their tags.")
            self._update_controls()
            return
        if self._inventory_loading:
            if reset:
                self._inventory_refresh_pending = True
            self._update_controls()
            return
        self._inventory_loading = True
        generation = self._generation
        offset = self._inventory_offset
        query = self.search_field.text().strip()
        scope_paths = self._scope_paths()
        self.status_label.setText("Loading tag inventory…")

        def _run(progress, cancel_check):
            progress(-1, "Loading tag inventory…")
            if cancel_check():
                return None
            return self.tag_service.query_tag_inventory(
                query=query,
                scope_paths=scope_paths,
                limit=self.INVENTORY_PAGE_SIZE,
                offset=offset,
                include_total=reset,
            )

        def _done(page) -> None:
            if generation != self._generation or page is None:
                return
            entries = [
                ListEntry(
                    title=item.display_tag,
                    subtitle=f"{item.image_count} photo(s) · " + ", ".join(source for source, _count in item.sources),
                    tooltip=f"{item.image_count} photo(s) | normalized: {item.normalized_tag}",
                    payload=item,
                )
                for item in page.items
            ]
            if reset:
                self.inventory_model.set_items(entries)
            else:
                self.inventory_model.append_items(entries)
            if page.total_count is not None:
                self._inventory_total = int(page.total_count)
            self._inventory_offset = offset + len(entries)
            self.inventory_count_label.setText(f"{self._inventory_total} tag(s)")
            self.status_label.setText("Tag inventory ready.")
            self._update_controls()

        job = self._start_job("Loading tag inventory", _run, _done)

        def _finished() -> None:
            self._inventory_loading = False
            if self._inventory_refresh_pending:
                self._inventory_refresh_pending = False
                self._load_inventory(reset=True)
            self._update_controls()

        job.completed.connect(lambda _page: _finished())
        job.failed.connect(lambda _message: _finished())
        job.cancelled.connect(_finished)

    def _load_more_inventory(self) -> None:
        if not self._inventory_loading and self._inventory_offset < self._inventory_total:
            self._load_inventory()

    def _on_tag_selected(self) -> None:
        selected = self.inventory_list.selectedIndexes()
        if not selected:
            return
        item = self.inventory_model.item_at(selected[0].row())
        payload = item.payload if item is not None else None
        if not isinstance(payload, TagInventoryItem):
            return
        self._selected_tag = payload.display_tag
        self.rename_field.setText(payload.display_tag)
        self._photo_offset = 0
        self._photo_total = 0
        self.gallery.update_gallery([])
        self._load_photos(reset=True)
        self._update_controls()

    def _load_photos(self, *, reset: bool = False) -> None:
        if not self._selected_tag:
            return
        if self._active_scope_is_unavailable():
            self._photo_total = 0
            self._photo_offset = 0
            self.gallery.update_gallery([])
            self.photo_heading.setText("Choose a folder or active roots to view tagged photos.")
            self._update_controls()
            return
        if self._photo_loading:
            if reset:
                self._photo_reset_pending = True
            self._update_controls()
            return
        self._photo_loading = True
        generation = self._generation
        tag = self._selected_tag
        offset = self._photo_offset
        scope_paths = self._scope_paths()
        self.photo_heading.setText(f"Photos tagged {tag}")

        def _run(progress, cancel_check):
            progress(-1, f"Loading photos tagged {tag}…")
            if cancel_check():
                return None
            return self.tag_service.query_tagged_paths(
                tag,
                scope_paths=scope_paths,
                limit=self.PHOTO_PAGE_SIZE,
                offset=offset,
                include_total=reset,
            )

        def _done(page) -> None:
            if generation != self._generation or tag != self._selected_tag or page is None:
                return
            if reset:
                self.gallery.update_gallery(list(page.paths))
            else:
                self.gallery.append_images(list(page.paths))
            if page.total_count is not None:
                self._photo_total = int(page.total_count)
            self._photo_offset = offset + len(page.paths)
            self.photo_heading.setText(f"Photos tagged {tag} · {self._photo_total} photo(s)")
            self.open_in_gallery_button.setEnabled(bool(self.gallery.images))
            self._update_controls()

        job = self._start_job("Loading tagged photos", _run, _done)

        def _finished() -> None:
            self._photo_loading = False
            if self._photo_reset_pending:
                self._photo_reset_pending = False
                self._load_photos(reset=True)
            self._update_controls()

        job.completed.connect(lambda _page: _finished())
        job.failed.connect(lambda _message: _finished())
        job.cancelled.connect(_finished)

    def _load_more_photos(self) -> None:
        if not self._photo_loading and self._photo_offset < self._photo_total:
            self._load_photos()

    def _rename_selected(self) -> None:
        if self._read_only_mode:
            self.status_label.setText("Read-only safety mode is enabled. Tag writes are disabled.")
            return
        old = self._selected_tag
        new = self.rename_field.text().strip()
        if not old or not new or old.casefold() == new.casefold():
            self.status_label.setText("Select a tag and enter a different name to rename or merge it.")
            return
        if QMessageBox.question(self, "Rename tag globally", f"Rename or merge '{old}' into '{new}' across the whole tag library?") != QMessageBox.StandardButton.Yes:
            return
        self._mutate_tag("Renaming tag", lambda: self.tag_service.rename_tag(old, new), f"Renamed '{old}' to '{new}'")

    def _delete_selected(self) -> None:
        if self._read_only_mode:
            self.status_label.setText("Read-only safety mode is enabled. Tag writes are disabled.")
            return
        tag = self._selected_tag
        if not tag:
            self.status_label.setText("Select a tag first.")
            return
        if QMessageBox.question(self, "Delete tag globally", f"Remove '{tag}' from every photo in the durable tag library?") != QMessageBox.StandardButton.Yes:
            return
        self._mutate_tag("Deleting tag", lambda: self.tag_service.delete_tag(tag), f"Deleted '{tag}'")

    def _mutate_tag(self, title: str, operation, message: str) -> None:
        if self._mutation_loading:
            return
        self._mutation_loading = True
        self._update_controls()
        def _run(progress, cancel_check):
            progress(-1, title + "…")
            if cancel_check():
                return None
            return operation()

        def _done(affected) -> None:
            if affected is None:
                return
            self.status_label.setText(f"{message} for {int(affected)} photo(s).")
            self._selected_tag = ""
            self.gallery.update_gallery([])
            self.refresh()
            self.metadata_changed.emit([])

        job = self._start_job(title, _run, _done)
        job.completed.connect(lambda _affected: self._finish_mutation())
        job.failed.connect(lambda _message: self._finish_mutation())
        job.cancelled.connect(self._finish_mutation)

    def _finish_mutation(self) -> None:
        self._mutation_loading = False
        self._update_controls()

    def _add_selected_to_filter(self) -> None:
        if self._selected_tag and self._selected_tag not in self._filter_tags:
            self._filter_tags.append(self._selected_tag)
            self._update_filter_label()

    def _clear_filter(self) -> None:
        self._filter_tags.clear()
        self._update_filter_label()

    def _run_filter(self) -> None:
        if self._filter_tags:
            self.run_tag_filter_requested.emit(list(self._filter_tags), self.selected_match_mode())

    def _update_filter_label(self) -> None:
        self.filter_label.setText("Filter: " + (", ".join(self._filter_tags) if self._filter_tags else "none"))
        self._update_controls()

    def _update_controls(self) -> None:
        has_tag = bool(self._selected_tag)
        self.more_tags_button.setVisible(self._inventory_offset < self._inventory_total)
        self.more_photos_button.setVisible(self._photo_offset < self._photo_total)
        self.more_tags_button.setEnabled(not self._inventory_loading)
        self.more_photos_button.setEnabled(not self._photo_loading)
        self.rename_button.setEnabled(has_tag and not self._read_only_mode and not self._mutation_loading)
        self.delete_button.setEnabled(has_tag and not self._read_only_mode and not self._mutation_loading)
        self.add_filter_button.setEnabled(has_tag)
        self.run_filter_button.setEnabled(bool(self._filter_tags))
        if self._read_only_mode:
            self.rename_button.setToolTip("Disabled in read-only safety mode.")
            self.delete_button.setToolTip("Disabled in read-only safety mode.")

    def _on_gallery_metadata_changed(self, paths: list[str]) -> None:
        self.metadata_changed.emit(list(paths))
        self.refresh()

    def _open_tagged_photos_in_gallery(self) -> None:
        paths = list(self.gallery.images)
        if paths and self._selected_tag:
            self.open_in_gallery_requested.emit(paths, self._selected_tag)

    def _start_job(self, title: str, run, done) -> AsyncJob:
        job = AsyncJob(run)
        coordinated = self.work_coordinator is not None
        job_id = (
            self.job_manager.register_job(title, cancel_fn=job.cancel, origin="Organize")
            if self.job_manager and not coordinated
            else None
        )
        holder: dict[str, object] = {}

        def _cleanup(status: str, error: str = "") -> None:
            thread = holder.get("thread")
            self._jobs[:] = [pair for pair in self._jobs if pair[0] is not job]
            self._coordinated_job_ids.pop(job, None)
            if job_id is not None and self.job_manager:
                self.job_manager.finish(job_id, status=status, error=error)
            _ = thread

        job.progress.connect(lambda value, text: self.job_manager.update(job_id, progress=value, text=text) if self.job_manager and job_id is not None else None)
        job.completed.connect(done)
        job.completed.connect(lambda _result: _cleanup("finished"))
        job.failed.connect(lambda message: self.status_label.setText(f"{title} failed: {message}"))
        job.failed.connect(lambda message: _cleanup("failed", str(message)))
        job.cancelled.connect(lambda: self.status_label.setText(f"{title} cancelled."))
        job.cancelled.connect(lambda: _cleanup("cancelled"))
        def _launch(_use_cpu_fallback: bool = False) -> None:
            thread = start_job_in_thread(job)
            holder["thread"] = thread
            self._jobs.append((job, thread))

        if self.work_coordinator is not None:
            coordinated_job_id = self.work_coordinator.submit_async_job(
                JobSpec(
                    title,
                    origin="Organize",
                    io_bound=True,
                    data_home_write=title in {"Renaming tag", "Deleting tag"},
                    data_home_read=title not in {"Renaming tag", "Deleting tag"},
                ),
                job,
                _launch,
            )
            state = self.job_manager.get(coordinated_job_id) if self.job_manager is not None else None
            if state is not None and state.status in {"queued", "running", "cancelling"}:
                self._coordinated_job_ids[job] = coordinated_job_id
        else:
            _launch()
        return job

    def shutdown_jobs(self, *, timeout_ms: int = 2500) -> bool:
        """Cancel and join tag-hub jobs before Qt destroys this view."""

        jobs = list(self._jobs)
        if self.work_coordinator is not None:
            for coordinated_job_id in tuple(self._coordinated_job_ids.values()):
                self.work_coordinator.cancel(coordinated_job_id)
        for job, _thread in jobs:
            if job in self._coordinated_job_ids:
                continue
            try:
                job.cancel()
            except Exception:
                pass
        ready = True
        for job, thread in jobs:
            ready = wait_for_thread_shutdown(thread, timeout_ms=timeout_ms) and ready
            if ready:
                try:
                    defer_async_job_dispose(job)
                except Exception:
                    pass
        if ready:
            if self.work_coordinator is not None:
                for coordinated_job_id in tuple(self._coordinated_job_ids.values()):
                    self.work_coordinator.finish(coordinated_job_id, status="cancelled")
            for job, _thread in jobs:
                for signal_name in ("started", "progress", "completed", "failed", "cancelled"):
                    try:
                        getattr(job, signal_name).disconnect()
                    except (TypeError, RuntimeError):
                        pass
                defer_async_job_dispose(job)
            self._coordinated_job_ids.clear()
            self._jobs.clear()
        try:
            ready = self.gallery.shutdown_jobs(timeout_ms=timeout_ms) and ready
        except Exception:
            ready = False
        return ready
