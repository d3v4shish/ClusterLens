from __future__ import annotations

import html
import json

from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QSize, Qt, pyqtSlot
from PyQt6.QtGui import QImageReader, QKeyEvent, QKeySequence, QPixmap, QShortcut
from PyQt6.QtWidgets import QDialog, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSplitter, QTextBrowser, QVBoxLayout, QWidget

from app.services.photo_metadata import PhotoMetadata, PhotoMetadataService
from ui.async_job import AsyncJob, raise_if_cancelled, start_job_in_thread
from ui.zoomable_image import ZoomableImageView


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
        parent=None,
    ):
        super().__init__(parent)
        self.metadata_service = metadata_service or PhotoMetadataService()
        self._active_thread = None
        self._active_job = None
        self._preview_thread = None
        self._preview_job = None
        self._retained_async_refs: list[tuple[object | None, object | None]] = []
        self._thread_jobs: dict[object, object | None] = {}
        self._request_id = 0
        self._full_res_loaded = False
        self._context_provider = context_provider
        self._base_context: dict[str, object] = dict(context or {})
        self._display_mode = "advanced" if str(display_mode).strip().lower() == "advanced" else "basic"
        self._shortcuts: list[QShortcut] = []
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

        self.metadata_summary_label = QLabel("")
        self.metadata_summary_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.metadata_summary_label.setWordWrap(True)
        self.metadata_summary_label.setTextFormat(Qt.TextFormat.RichText)
        self.metadata_summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        details_layout.addWidget(self.metadata_summary_label)

        self.info_text = QTextBrowser()
        self.info_text.setReadOnly(True)
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
        self._install_shortcuts()
        self._apply_display_mode()

        self._update_nav_state()
        if self._image_paths:
            self.load_metadata(self._image_paths[self._index], dict(self._base_context))

    def load_metadata(self, image_path: str, context: dict[str, object]) -> None:
        self._request_id += 1
        request_id = self._request_id
        self._full_res_loaded = False
        self.preview_view.set_pixmap(None)
        self._update_identity(image_path)
        self._set_loading_state()
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
            self._set_state_text("Advanced view. Left/Right arrows browse the current gallery group.")

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
            self._on_async_thread_finished,
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
            self._on_async_thread_finished,
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

    def _handle_finished_thread(self, thread) -> None:
        if thread is None:
            return
        self._release_async_refs(self._thread_jobs.get(thread), thread)

    @pyqtSlot()
    def _on_async_thread_finished(self) -> None:
        self._handle_finished_thread(self.sender())

    def shutdown_jobs(self, *, timeout_ms: int = 2500) -> bool:
        ready_to_close = True
        active_pairs = [
            (self._active_job, self._active_thread),
            (self._preview_job, self._preview_thread),
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
                    thread.quit()
                    thread_finished = bool(thread.wait(timeout_ms))
                except RuntimeError:
                    thread_finished = True
            ready_to_close = bool(thread_finished) and ready_to_close
        self._active_job = None
        self._active_thread = None
        self._preview_job = None
        self._preview_thread = None
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
            ("Left", lambda: self._step(-1)),
            ("Right", lambda: self._step(1)),
            ("A", lambda: self._step(-1)),
            ("D", lambda: self._step(1)),
            ("F", self.preview_view.fit_to_window),
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

    def _set_loading_state(self) -> None:
        if self._display_mode == "basic":
            self.metadata_summary_label.clear()
            self.info_text.clear()
            self._set_state_text(
                "Basic view. Left/Right arrows browse the current gallery group. Metadata is hidden in basic mode."
            )
            return
        self.metadata_summary_label.setText("<b>Important Details</b><br>Loading metadata...")
        self.info_text.setHtml("<p>Loading metadata...</p>")
        self._set_state_text("Advanced view. Left/Right arrows browse the current gallery group.")

    def keyPressEvent(self, event: QKeyEvent) -> None:
        return super().keyPressEvent(event)

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
