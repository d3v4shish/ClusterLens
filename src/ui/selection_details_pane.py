from __future__ import annotations

import json
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QDesktopServices, QGuiApplication
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QTextEdit, QVBoxLayout, QWidget

from app.services.photo_metadata import PhotoMetadata, PhotoMetadataService
from ui.async_job import AsyncJob, raise_if_cancelled, start_job_in_thread, wait_for_thread_shutdown


class SelectionDetailsPane(QWidget):
    add_to_review_requested = pyqtSignal(list)
    open_inspector_requested = pyqtSignal(list, int, object)

    def __init__(self, metadata_service: PhotoMetadataService | None = None, parent=None):
        super().__init__(parent)
        self.metadata_service = metadata_service or PhotoMetadataService()
        self._active_thread = None
        self._active_job = None
        self._retained_async_refs: list[tuple[object | None, object | None]] = []
        self._thread_jobs: dict[object, object | None] = {}
        self._request_id = 0
        self._image_paths: list[str] = []
        self._index = -1
        self._context_provider = None
        self._base_context: dict[str, object] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        self.title_label = QLabel("Selection Details")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(self.title_label)

        actions = QHBoxLayout()
        self.open_button = QPushButton("Open")
        self.copy_path_button = QPushButton("Copy Path")
        self.open_folder_button = QPushButton("Open Folder")
        self.review_button = QPushButton("Add To Review")
        actions.addWidget(self.open_button)
        actions.addWidget(self.copy_path_button)
        actions.addWidget(self.open_folder_button)
        actions.addWidget(self.review_button)
        layout.addLayout(actions)

        self.summary_label = QLabel("No image selected.")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        layout.addWidget(self.info_text, stretch=1)

        self.open_button.clicked.connect(self._open_inspector)
        self.copy_path_button.clicked.connect(self._copy_path)
        self.open_folder_button.clicked.connect(self._open_folder)
        self.review_button.clicked.connect(self._add_to_review)

    def set_review_enabled(self, enabled: bool) -> None:
        self.review_button.setVisible(bool(enabled))

    def set_selection(
        self,
        image_paths: list[str],
        current_index: int,
        *,
        base_context: dict[str, object] | None = None,
        context_provider=None,
    ) -> None:
        self._image_paths = list(image_paths or [])
        self._index = int(current_index) if self._image_paths else -1
        self._context_provider = context_provider
        self._base_context = dict(base_context or {})
        if self._index < 0 or self._index >= len(self._image_paths):
            self.summary_label.setText("No image selected.")
            self.info_text.setPlainText("")
            return
        image_path = self._image_paths[self._index]
        self.summary_label.setText(str(image_path))
        self._load_metadata(image_path)

    def clear(self) -> None:
        self._image_paths = []
        self._index = -1
        self.summary_label.setText("No image selected.")
        self.info_text.setPlainText("")

    def refresh_current(self) -> None:
        if self._index < 0 or self._index >= len(self._image_paths):
            return
        self._load_metadata(self._image_paths[self._index])

    def current_path(self) -> str | None:
        if self._index < 0 or self._index >= len(self._image_paths):
            return None
        return self._image_paths[self._index]

    def _load_metadata(self, image_path: str) -> None:
        self._request_id += 1
        request_id = self._request_id
        self.info_text.setPlainText("Loading details...")
        if self._active_job is not None:
            self._retain_async_refs(self._active_job, self._active_thread)
            self._active_job.cancel()
            self._active_job = None
            self._active_thread = None

        context = dict(self._base_context)
        if callable(self._context_provider):
            try:
                context.update(self._context_provider(image_path) or {})
            except Exception:
                pass

        def _run(_progress, cancel_check):
            raise_if_cancelled(cancel_check)
            metadata = self.metadata_service.get_metadata(
                image_path,
                context=context,
                include_hashes=False,
                include_face_boxes=False,
            )
            raise_if_cancelled(cancel_check)
            return metadata

        job = AsyncJob(_run)

        def _on_completed(metadata: PhotoMetadata) -> None:
            if request_id != self._request_id:
                return
            self.info_text.setPlainText(self._render_metadata(metadata))

        def _on_failed(message: str) -> None:
            if request_id != self._request_id:
                return
            self.info_text.setPlainText(f"Failed to load details: {message}")

        def _on_cancelled() -> None:
            if request_id != self._request_id:
                return
            self.info_text.setPlainText("Cancelled.")

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

    def _handle_finished_thread(self, thread) -> None:
        if thread is None:
            return
        self._release_async_refs(self._thread_jobs.get(thread), thread)

    def _on_async_thread_finished(self, thread=None) -> None:
        self._handle_finished_thread(thread)

    def shutdown_jobs(self, *, timeout_ms: int = 2500) -> bool:
        ready_to_close = True
        active_pairs = [(self._active_job, self._active_thread), *self._retained_async_refs]
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
        if ready_to_close:
            self._retained_async_refs = []
            self._thread_jobs = {}
        return ready_to_close

    def _open_inspector(self) -> None:
        if not self._image_paths or self._index < 0:
            return
        self.open_inspector_requested.emit(list(self._image_paths), int(self._index), self._context_provider)

    def _copy_path(self) -> None:
        if not self._image_paths or self._index < 0:
            return
        QGuiApplication.clipboard().setText(self._image_paths[self._index])

    def _open_folder(self) -> None:
        if not self._image_paths or self._index < 0:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(self._image_paths[self._index]).parent)))

    def _add_to_review(self) -> None:
        if not self._image_paths or self._index < 0:
            return
        self.add_to_review_requested.emit([self._image_paths[self._index]])

    @staticmethod
    def _render_metadata(metadata: PhotoMetadata) -> str:
        context_json = json.dumps(metadata.context, indent=2, default=str)
        lines = [
            f"Path: {metadata.image_path}",
            f"Dimensions: {metadata.width}x{metadata.height}",
            f"File size: {metadata.file_size} bytes",
            f"Modified: {metadata.modified_at}",
            f"Camera: {metadata.camera}",
            "",
            f"Hashes: {json.dumps(metadata.hashes, indent=2)}",
            "",
            f"Faces: {json.dumps(metadata.face_boxes, indent=2)}",
            "",
            f"Context: {context_json}",
        ]
        return "\n".join(lines)
