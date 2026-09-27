from __future__ import annotations

import os

from PyQt6.QtCore import QDir, QUrl, Qt, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QTabWidget,
    QToolButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from app.path_scope import PathScope
from infra.settings import get_settings
from ui.common import ResponsiveFlowLayout
from ui.footer_bar import ElidedLabel
from ui.icons import apply_icon

from .mode_panes import ActiveRootFileSystemModel


class SourcePane(QWidget):
    """An on-demand, staged editor for the global photo-root scope.

    The Sources tab intentionally changes only a draft scope.  Applying that
    draft is the single point that can notify the rest of the application and
    start scope-dependent background work.  Catalog management stays separate
    so browsing a filesystem never becomes an accidental Library operation.
    """

    directory_changed = pyqtSignal(str)
    scope_changed = pyqtSignal(object)
    recent_folder_remove_requested = pyqtSignal(str)
    recent_folders_clear_requested = pyqtSignal()
    state_changed = pyqtSignal()
    hide_requested = pyqtSignal()
    close_edit_requested = pyqtSignal()
    draft_applied = pyqtSignal()
    draft_discarded = pyqtSignal()
    run_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    data_home_requested = pyqtSignal()
    register_library_root_requested = pyqtSignal(str)
    refresh_library_root_requested = pyqtSignal(str)
    pause_library_root_requested = pyqtSignal(str)
    remove_library_root_requested = pyqtSignal(str)
    set_library_root_enabled_requested = pyqtSignal(str, bool)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.settings = get_settings()
        self.selected_directory = ""
        self.active_scope = PathScope()
        self.draft_scope: PathScope | None = None
        self._recent_directories: list[str] = []
        self._library_root_states: dict[str, dict[str, object]] = {}
        self._library_root_jobs: dict[str, str] = {}
        self._read_only_mode = False
        self._browse_root = "C:/" if os.name == "nt" else "/"
        self._build_ui()

    @property
    def active_roots(self) -> tuple[str, ...]:
        return self.active_scope.roots

    @property
    def is_editing(self) -> bool:
        return self.draft_scope is not None

    @property
    def has_dirty_draft(self) -> bool:
        return self.draft_scope is not None and self.draft_scope != self.active_scope

    def _edit_scope(self) -> PathScope:
        return self.draft_scope if self.draft_scope is not None else self.active_scope

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        header_widget = QWidget(self)
        header = ResponsiveFlowLayout(header_widget, spacing=6)
        title = QLabel("Roots", self)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        self.recent_folders_button = QToolButton(self)
        self.recent_folders_button.setText("Recent")
        self.recent_folders_button.setToolTip("Browse a recently used folder. It does not add the folder as a source.")
        self.recent_folders_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.recent_folders_menu = QMenu(self.recent_folders_button)
        self.recent_folders_button.setMenu(self.recent_folders_menu)
        self.hide_button = QToolButton(self)
        self.hide_button.setText("Close")
        self.hide_button.setToolTip("Close the Roots editor. Unapplied source changes are reviewed first.")
        self.hide_button.setAccessibleName("Close Roots editor")
        self.hide_button.clicked.connect(self.request_close_edit)
        header.addWidget(title)
        header.addWidget(self.recent_folders_button)
        header.addWidget(self.hide_button)
        layout.addWidget(header_widget)

        self.tabs = QTabWidget(self)
        self.tabs.setObjectName("rootsDrawerTabs")
        layout.addWidget(self.tabs, stretch=1)

        sources_page = QWidget(self.tabs)
        sources_layout = QVBoxLayout(sources_page)
        sources_layout.setContentsMargins(0, 0, 0, 0)
        sources_layout.setSpacing(6)
        self.directory_combobox = QComboBox(sources_page)
        self.directory_combobox.setToolTip("Choose the filesystem location shown by the folder tree.")
        self.populate_drives()
        self.directory_combobox.currentIndexChanged.connect(self.on_directory_changed)
        sources_layout.addWidget(self.directory_combobox)

        self.selected_folder_label = ElidedLabel("Browse: no folder", sources_page)
        self.selected_folder_label.setObjectName("selectedFolderRoot")
        self.selected_folder_label.setWordWrap(False)
        self.selected_folder_label.setToolTip("Browse a row, then check it to stage that folder as a source.")
        sources_layout.addWidget(self.selected_folder_label)

        self.file_model = ActiveRootFileSystemModel(self)
        self.file_model.setFilter(QDir.Filter.Dirs | QDir.Filter.NoDotAndDotDot)
        self.file_model.setRootPath(self._browse_root)
        self.file_tree = QTreeView(sources_page)
        self.file_tree.setAccessibleName("Source folder browser")
        self.file_tree.setModel(self.file_model)
        self.file_tree.setRootIndex(self.file_model.index(self._browse_root))
        self.file_tree.setHeaderHidden(True)
        self.file_tree.setColumnHidden(1, True)
        self.file_tree.setColumnHidden(2, True)
        self.file_tree.setColumnHidden(3, True)
        self.file_tree.setUniformRowHeights(True)
        self.file_tree.setStyleSheet("QTreeView { font-size: 12px; } QTreeView::item { height: 24px; padding: 3px; }")
        self.file_tree.clicked.connect(self.on_directory_selected)
        self.file_model.root_toggle_requested.connect(self._on_tree_root_toggled)
        sources_layout.addWidget(self.file_tree, stretch=1)

        draft_footer = QWidget(sources_page)
        draft_layout = QVBoxLayout(draft_footer)
        draft_layout.setContentsMargins(0, 0, 0, 0)
        draft_layout.setSpacing(4)
        self.draft_summary_label = QLabel(draft_footer)
        self.draft_summary_label.setProperty("role", "help")
        self.draft_summary_label.setWordWrap(True)
        self.draft_summary_label.setToolTip("The staged sources take effect in every workspace only after Apply.")
        draft_layout.addWidget(self.draft_summary_label)
        self.draft_actions = QWidget(draft_footer)
        draft_actions = ResponsiveFlowLayout(self.draft_actions, spacing=6)
        self.apply_draft_button = QPushButton("Apply sources", draft_footer)
        self.apply_draft_button.setProperty("kind", "primary")
        self.apply_draft_button.setToolTip("Apply the staged source scope to Library, Organize, People, and Tools.")
        self.apply_draft_button.clicked.connect(self.apply_draft)
        self.discard_draft_button = QPushButton("Discard", draft_footer)
        self.discard_draft_button.setToolTip("Discard staged source changes and retain the current shared scope.")
        self.discard_draft_button.clicked.connect(self.discard_draft)
        draft_actions.addWidget(self.apply_draft_button)
        draft_actions.addWidget(self.discard_draft_button)
        draft_layout.addWidget(self.draft_actions)
        sources_layout.addWidget(draft_footer)
        self.tabs.addTab(sources_page, "Sources")

        catalog_page = QWidget(self.tabs)
        catalog_layout = QVBoxLayout(catalog_page)
        catalog_layout.setContentsMargins(0, 0, 0, 0)
        catalog_layout.setSpacing(6)
        self.active_roots_list = QListWidget(catalog_page)
        self.active_roots_list.setObjectName("activeRootsList")
        self.active_roots_list.setAccessibleName("Registered photo roots")
        self.active_roots_list.setMinimumHeight(96)
        self.active_roots_list.setToolTip("Active and registered roots. Select one to manage its local Library catalog.")
        self.active_roots_list.itemSelectionChanged.connect(self._refresh_active_root_action_state)
        catalog_layout.addWidget(self.active_roots_list, stretch=1)

        catalog_actions = QWidget(catalog_page)
        catalog_action_layout = ResponsiveFlowLayout(catalog_actions, spacing=6)
        self.library_root_action_button = QPushButton("Register", catalog_actions)
        self.library_root_action_button.setToolTip("Register this active root in Library, or enable or disable its existing catalog.")
        self.library_root_action_button.clicked.connect(self._request_library_root_action)
        self.refresh_library_root_button = QPushButton("Refresh", catalog_actions)
        self.refresh_library_root_button.setToolTip("Refresh the selected root's derived Library catalog in a cancellable Job.")
        self.refresh_library_root_button.clicked.connect(self._request_library_root_refresh)
        self.catalog_more_button = QToolButton(catalog_actions)
        self.catalog_more_button.setText("More")
        self.catalog_more_button.setToolTip("More actions for the selected root.")
        self.catalog_more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.catalog_more_menu = QMenu(self.catalog_more_button)
        self.pause_library_root_button = self.catalog_more_menu.addAction("Pause refresh")
        self.pause_library_root_button.setToolTip("Cancel the selected root's catalog refresh after the current file.")
        self.pause_library_root_button.triggered.connect(self._request_library_root_pause)
        self.reveal_root_button = self.catalog_more_menu.addAction("Reveal folder")
        self.reveal_root_button.setToolTip("Reveal the selected source folder in the system file manager.")
        self.reveal_root_button.triggered.connect(self._reveal_selected_root)
        self.remove_library_root_button = self.catalog_more_menu.addAction("Unregister Library catalog")
        self.remove_library_root_button.setToolTip("Remove only the selected root's derived Library records. Source photos are never deleted.")
        self.remove_library_root_button.triggered.connect(self._request_library_root_removal)
        self.catalog_more_button.setMenu(self.catalog_more_menu)
        self.data_home_button = QPushButton("Storage", catalog_actions)
        self.data_home_button.setToolTip("Open Storage for managed indexes, previews, models, recovery, backups, and reports.")
        self.data_home_button.clicked.connect(self.data_home_requested.emit)
        self.data_home_label = QLabel("Managed data location is available in Storage.", self)
        self.data_home_label.setVisible(False)
        for button, icon in (
            (self.library_root_action_button, "metadata"),
            (self.refresh_library_root_button, "retry"),
            (self.data_home_button, "settings"),
        ):
            apply_icon(button, icon)
        for widget in (
            self.library_root_action_button,
            self.refresh_library_root_button,
            self.catalog_more_button,
            self.data_home_button,
        ):
            catalog_action_layout.addWidget(widget)
        catalog_layout.addWidget(catalog_actions)
        self.tabs.addTab(catalog_page, "Catalog")

        # Kept as inert compatibility attributes for older callers.  The Basic
        # organize action now lives beside the photo grid, not in Roots.
        self.basic_action_section = QWidget(self)
        self.basic_run_button = QPushButton("Organize photos", self.basic_action_section)
        self.basic_cancel_button = QPushButton("Cancel", self.basic_action_section)
        self.basic_action_section.hide()
        self.set_recent_directories([])
        self._refresh_draft_ui()
        self._refresh_active_roots_ui()

    @staticmethod
    def _recent_folder_label(path: str) -> str:
        normalized = os.path.normpath(str(path or ""))
        name = os.path.basename(normalized) or normalized
        parent = os.path.dirname(normalized)
        return f"{name} — {parent}" if parent and parent != normalized else name

    def set_recent_directories(self, directories: list[str]) -> None:
        self._recent_directories = [str(path) for path in directories if str(path or "").strip()]
        self.recent_folders_menu.clear()
        if not self._recent_directories:
            action = self.recent_folders_menu.addAction("No recent folders")
            action.setEnabled(False)
            return
        for path in self._recent_directories:
            action = self.recent_folders_menu.addAction(self._recent_folder_label(path))
            action.setToolTip(path)
            action.triggered.connect(
                lambda _checked=False, value=path: self.set_selected_directory(value, activate_scope=False)
            )
        self.recent_folders_menu.addSeparator()
        remove_menu = self.recent_folders_menu.addMenu("Remove from history")
        for path in self._recent_directories:
            action = remove_menu.addAction(self._recent_folder_label(path))
            action.setToolTip(path)
            action.triggered.connect(
                lambda _checked=False, value=path: self.recent_folder_remove_requested.emit(value)
            )
        self.recent_folders_menu.addAction("Clear history").triggered.connect(self.recent_folders_clear_requested.emit)

    def populate_drives(self) -> None:
        self.directory_combobox.clear()
        if os.name == "nt":
            from string import ascii_uppercase

            drives = [f"{letter}:/" for letter in ascii_uppercase if os.path.exists(f"{letter}:/")]
            self._browse_root = drives[0] if drives else "C:/"
        else:
            drives = ["/", "/home"]
            self._browse_root = "/"
        self.directory_combobox.addItems(drives)

    def begin_edit(self) -> None:
        if self.draft_scope is None:
            self.draft_scope = self.active_scope
        self.file_model.set_active_scope(self._edit_scope())
        self._refresh_draft_ui()
        self._refresh_active_root_action_state()

    def request_close_edit(self) -> None:
        self.close_edit_requested.emit()

    def apply_draft(self) -> None:
        scope = self._edit_scope()
        changed = scope != self.active_scope
        self.active_scope = scope
        self.draft_scope = None
        self.file_model.set_active_scope(self.active_scope)
        self._refresh_draft_ui()
        self._refresh_active_roots_ui()
        if changed:
            self.scope_changed.emit(scope)
            self.state_changed.emit()
        self.draft_applied.emit()

    def discard_draft(self) -> None:
        self.draft_scope = None
        self.file_model.set_active_scope(self.active_scope)
        self._refresh_draft_ui()
        self._refresh_active_root_action_state()
        self.draft_discarded.emit()

    def on_directory_selected(self, index) -> None:
        self.set_selected_directory(self.file_model.filePath(index), emit_state=True, activate_scope=False)

    def on_directory_changed(self, _index) -> None:
        root_directory = self.directory_combobox.currentText()
        self._browse_root = root_directory
        self.file_model.setRootPath(root_directory)
        self.file_tree.setRootIndex(self.file_model.index(root_directory))
        self.set_selected_directory(root_directory, emit_state=True, activate_scope=False)

    @staticmethod
    def _path_is_within(directory: str, root: str) -> bool:
        try:
            normalized_directory = os.path.normcase(os.path.abspath(directory))
            normalized_root = os.path.normcase(os.path.abspath(root))
            return os.path.commonpath((normalized_directory, normalized_root)) == normalized_root
        except (TypeError, ValueError):
            return False

    def _ensure_browse_root_contains(self, directory: str) -> None:
        if self._path_is_within(directory, self._browse_root):
            return
        candidates = [self.directory_combobox.itemText(index) for index in range(self.directory_combobox.count())]
        matching_roots = [root for root in candidates if self._path_is_within(directory, root)]
        if matching_roots:
            browse_root = max(matching_roots, key=len)
        else:
            drive, _ = os.path.splitdrive(directory)
            browse_root = f"{drive}{os.path.sep}" if drive else os.path.abspath(os.path.sep)
        self._browse_root = browse_root
        combo_index = self.directory_combobox.findText(browse_root)
        if combo_index >= 0:
            self.directory_combobox.blockSignals(True)
            self.directory_combobox.setCurrentIndex(combo_index)
            self.directory_combobox.blockSignals(False)
        self.file_model.setRootPath(browse_root)
        self.file_tree.setRootIndex(self.file_model.index(browse_root))

    def _reveal_selected_directory(self, directory: str) -> None:
        selected_index = self.file_model.index(directory)
        if not selected_index.isValid():
            return
        root_index = self.file_model.index(self._browse_root)
        if root_index.isValid():
            self.file_tree.setRootIndex(root_index)
        ancestors = []
        ancestor = selected_index.parent()
        while ancestor.isValid():
            ancestor_path = self.file_model.filePath(ancestor)
            if not self._path_is_within(ancestor_path, self._browse_root):
                break
            ancestors.append(ancestor)
            if os.path.normcase(os.path.abspath(ancestor_path)) == os.path.normcase(os.path.abspath(self._browse_root)):
                break
            ancestor = ancestor.parent()
        for ancestor in reversed(ancestors):
            self.file_tree.expand(ancestor)
        self.file_tree.expand(selected_index)
        self.file_tree.setCurrentIndex(selected_index)
        self.file_tree.scrollTo(selected_index)

    def set_selected_directory(self, directory: str, *, emit_state: bool = False, activate_scope: bool = True) -> None:
        directory = str(directory or "").strip()
        if not directory:
            return
        drive = os.path.splitdrive(directory)[0]
        if drive:
            combo_value = f"{drive}/".replace("\\", "/")
            index = self.directory_combobox.findText(combo_value)
            if index >= 0:
                self.directory_combobox.blockSignals(True)
                self.directory_combobox.setCurrentIndex(index)
                self.directory_combobox.blockSignals(False)
        self.selected_directory = directory
        self._ensure_browse_root_contains(directory)
        self._reveal_selected_directory(directory)
        folder_name = os.path.basename(os.path.normpath(directory)) or directory
        self.selected_folder_label.setText(f"Browse: {folder_name}")
        self.selected_folder_label.setToolTip(
            f"{directory}\n\nBrowse a row, then check it to stage that folder as a source."
        )
        self.directory_changed.emit(directory)
        if activate_scope:
            self._replace_edit_scope([directory], emit_state=emit_state)
        elif emit_state:
            self.state_changed.emit()
        self._refresh_active_root_action_state()

    def set_active_roots(self, roots, *, emit_state: bool = False) -> None:
        """Accept an authoritative shared scope without scanning source media."""

        scope = roots if isinstance(roots, PathScope) else PathScope.from_paths(roots, resolve_symlinks=False)
        changed = scope != self.active_scope
        self.active_scope = scope
        if self.is_editing:
            # This method is reserved for restored/external scope state; it
            # deliberately replaces a stale local draft rather than merging it.
            self.draft_scope = scope
        self.file_model.set_active_scope(self._edit_scope())
        self._refresh_draft_ui()
        self._refresh_active_roots_ui()
        if changed:
            self.scope_changed.emit(scope)
            if emit_state:
                self.state_changed.emit()

    def _replace_edit_scope(self, roots, *, emit_state: bool = False) -> None:
        scope = roots if isinstance(roots, PathScope) else PathScope.from_paths(roots, resolve_symlinks=False)
        if self.is_editing:
            self.draft_scope = scope
            self.file_model.set_active_scope(scope)
            self._refresh_draft_ui()
            self._refresh_active_root_action_state()
            if emit_state:
                self.state_changed.emit()
            return
        self.set_active_roots(scope, emit_state=emit_state)

    def set_data_home_summary(self, path: str, *, detail: str = "") -> None:
        location = str(path or "").strip()
        if not location:
            self.data_home_label.setText("Managed data location is unavailable.")
            self.data_home_label.setToolTip("")
            return
        suffix = f"\n{str(detail).strip()}" if str(detail).strip() else ""
        self.data_home_label.setText(f"{location}{suffix}")
        self.data_home_label.setToolTip(location)
        self.data_home_button.setToolTip(f"Open Storage for managed app data at {location}. Source photos are unaffected.")

    def add_active_root(self, directory: str) -> None:
        path = str(directory or "").strip()
        if path:
            self._replace_edit_scope([*self._edit_scope().roots, path], emit_state=True)

    def remove_active_root(self, directory: str) -> None:
        target = str(directory or "").strip()
        if target:
            self._replace_edit_scope([root for root in self._edit_scope().roots if root != target], emit_state=True)

    def remove_selected_active_root(self) -> None:
        self.remove_active_root(self._selected_root_path())

    def clear_active_roots(self) -> None:
        self._replace_edit_scope((), emit_state=True)

    def _on_tree_root_toggled(self, directory: str, checked: bool) -> None:
        self.begin_edit()
        if checked:
            self.add_active_root(directory)
        else:
            self.remove_active_root(directory)

    def _refresh_draft_ui(self) -> None:
        scope = self._edit_scope()
        count = len(scope.roots)
        label = "No sources selected" if not count else f"{count} selected"
        if self.has_dirty_draft:
            label += " · unapplied"
        self.draft_summary_label.setText(label)
        self.draft_summary_label.setToolTip("\n".join(scope.roots) or "No sources selected")
        self.apply_draft_button.setText(f"Apply {count} source{'s' if count != 1 else ''}")
        self.discard_draft_button.setEnabled(self.is_editing)

    def _refresh_active_roots_ui(self) -> None:
        current = self.active_roots_list.currentItem()
        selected_path = str(current.data(Qt.ItemDataRole.UserRole) or "") if current is not None else ""
        self.active_roots_list.blockSignals(True)
        self.active_roots_list.clear()
        roots = list(self.active_scope.roots)
        roots.extend(path for path in self._library_root_states if path not in roots)
        for root in roots:
            state = self._library_root_states.get(root, {})
            online = "online" if os.path.isdir(root) else "missing"
            scope_state = "active" if root in self.active_scope.roots else "catalog only"
            catalog_state = "not registered"
            if state:
                count = int(state.get("photo_count", 0) or 0)
                catalog_state = f"{count:,} photos" if bool(state.get("enabled", True)) else f"disabled · {count:,} photos"
            job = self._library_root_jobs.get(root, "")
            if job:
                catalog_state = f"{catalog_state} · {job}"
            name = os.path.basename(os.path.normpath(root)) or root
            item = QListWidgetItem(f"{name}\n{scope_state} · {online} · {catalog_state}")
            item.setData(Qt.ItemDataRole.UserRole, root)
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsUserCheckable)
            detail = f"{root}\nScope: {scope_state}\nLibrary: {catalog_state}"
            last_scan = str(state.get("last_scan_at") or "Never refreshed")
            item.setToolTip(f"{detail}\nLast refresh: {last_scan}")
            self.active_roots_list.addItem(item)
            if root == selected_path:
                self.active_roots_list.setCurrentItem(item)
        self.active_roots_list.blockSignals(False)
        self._refresh_active_root_action_state()

    def set_library_root_states(self, states: list[dict[str, object]]) -> None:
        self._library_root_states = {
            str(state.get("path") or ""): dict(state)
            for state in states
            if str(state.get("path") or "").strip()
        }
        self._refresh_active_roots_ui()

    def set_library_root_job_state(self, path: str, state: str) -> None:
        normalized = os.path.normpath(str(path or "").strip())
        if not normalized:
            return
        label = str(state or "").strip()
        if label:
            self._library_root_jobs[normalized] = label
        else:
            self._library_root_jobs.pop(normalized, None)
        self._refresh_active_roots_ui()

    def set_read_only_mode(self, enabled: bool) -> None:
        self._read_only_mode = bool(enabled)
        self._refresh_active_root_action_state()

    def _selected_root_path(self) -> str:
        item = self.active_roots_list.currentItem()
        return str(item.data(Qt.ItemDataRole.UserRole) or "") if item is not None else ""

    def _catalog_mutable(self) -> bool:
        return not self._read_only_mode and not self.has_dirty_draft

    def _request_library_root_action(self) -> None:
        path = self._selected_root_path()
        if not path or not self._catalog_mutable():
            return
        state = self._library_root_states.get(path)
        if state is None:
            self.register_library_root_requested.emit(path)
        else:
            self.set_library_root_enabled_requested.emit(path, not bool(state.get("enabled", True)))

    def _request_library_root_refresh(self) -> None:
        path = self._selected_root_path()
        if path and self._catalog_mutable():
            self.refresh_library_root_requested.emit(path)

    def _request_library_root_pause(self) -> None:
        path = self._selected_root_path()
        if path and self._catalog_mutable() and path in self._library_root_jobs:
            self.pause_library_root_requested.emit(path)

    def _reveal_selected_root(self) -> None:
        path = self._selected_root_path()
        if path and os.path.isdir(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))

    def _request_library_root_removal(self) -> None:
        path = self._selected_root_path()
        if path and self._catalog_mutable():
            self.remove_library_root_requested.emit(path)

    def _refresh_active_root_action_state(self) -> None:
        path = self._selected_root_path()
        state = self._library_root_states.get(path)
        mutable = self._catalog_mutable()
        if self.has_dirty_draft:
            blocked_reason = "Apply or discard staged source changes before changing the Library catalog."
        elif self._read_only_mode:
            blocked_reason = "Read-only safety mode prevents changes to the Library catalog."
        else:
            blocked_reason = ""
        label = "Register" if state is None else ("Disable" if bool(state.get("enabled", True)) else "Enable")
        self.library_root_action_button.setText(label)
        self.library_root_action_button.setEnabled(bool(path) and mutable)
        self.refresh_library_root_button.setEnabled(
            bool(state and bool(state.get("enabled", True)) and os.path.isdir(path) and mutable)
        )
        self.pause_library_root_button.setEnabled(bool(path and path in self._library_root_jobs and mutable))
        self.remove_library_root_button.setEnabled(bool(state and mutable))
        self.reveal_root_button.setEnabled(bool(path and os.path.isdir(path)))
        self.catalog_more_button.setEnabled(
            self.pause_library_root_button.isEnabled()
            or self.remove_library_root_button.isEnabled()
            or self.reveal_root_button.isEnabled()
        )
        if blocked_reason:
            self.library_root_action_button.setToolTip(blocked_reason)
            self.refresh_library_root_button.setToolTip(blocked_reason)
            self.pause_library_root_button.setToolTip(blocked_reason)
            self.remove_library_root_button.setToolTip(blocked_reason)
        else:
            self.library_root_action_button.setToolTip(
                "Register this active root in Library, or enable or disable its existing catalog."
            )
            self.refresh_library_root_button.setToolTip(
                "Refresh the selected root's derived Library catalog in a cancellable Job."
            )
            self.pause_library_root_button.setToolTip(
                "Cancel the selected root's catalog refresh after the current file."
            )
            self.remove_library_root_button.setToolTip(
                "Remove only the selected root's derived Library records. Source photos are never deleted."
            )

    def set_running(self, running: bool) -> None:
        self.basic_run_button.setEnabled(not running)
        self.basic_cancel_button.setEnabled(running)

    def update_progress(self, value: int, status: str) -> None:
        _ = value
        _ = status

    def set_basic_mode(self, enabled: bool, *, running: bool = False) -> None:
        _ = enabled
        self.basic_action_section.hide()
        self.set_running(running)
