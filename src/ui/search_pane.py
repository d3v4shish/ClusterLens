from __future__ import annotations

import json
import inspect
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from queue import Empty, PriorityQueue
from threading import Event, Lock
from time import perf_counter
from uuid import uuid4

from PyQt6.QtCore import QAbstractListModel, QEvent, QItemSelectionModel, QModelIndex, QRect, QSize, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QFont, QFontMetrics, QGuiApplication, QIcon, QImage, QImageReader, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QFrame,
    QFileDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QLayout,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QSizePolicy,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTabBar,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QStyledItemDelegate,
)
from PIL.ImageQt import ImageQt

from app.path_scope import folder_scope_sql, path_is_within_scope
from app.services.face_search import (
    BUILTIN_HUMAN_DETECTOR_ID,
    BUILTIN_HUMAN_EMBEDDER_ID,
    DEFAULT_FACE_MAX_DETECTIONS,
    DEFAULT_FACE_SCORE_THRESHOLD,
    FaceAlbumGroupPage,
    FaceAlbumGroupSummary,
    FaceAlbumMemberPage,
    FaceAlbumRecord,
    FaceClusterMember,
    FaceClusteringComparisonResult,
    FaceClusterIdentitySuggestion,
    LEGACY_DEFAULT_BUNDLE_ID,
    FaceFolderReviewImage,
    FaceIndexService,
    FaceLabelAssignment,
    FaceLabelRequest,
    FaceSearchRequest,
    IndexedFaceRecord,
    PersonPrototypeFace,
    default_face_detector_id,
    default_face_embedder_id,
    face_cluster_backend_choices,
    face_cluster_backend_tooltip,
    face_cluster_outlier_policy_choices,
    face_detector_choices,
    face_detector_policy_choices,
    face_embedder_choices,
    face_quality_profile_choices,
    face_quality_profile_config,
    face_quality_gate_choices,
    face_model_profile_choices,
    face_model_profile_config,
    face_mode_label,
    face_rerank_policy_choices,
    face_verifier_mode_choices,
    normalize_face_component_id,
    normalize_face_mode,
    resolve_face_detector_bundle,
    resolve_face_embedder_bundle,
    resolve_ready_face_pipeline_ids,
    selected_face_model_profile,
)
from app.services.cluster_explanations import ClusterExplanation
from app.services.gallery_actions import GalleryActionService
from app.services.saved_searches import SavedSearch, SavedSearchService
from app.services.similarity_search import DuplicateReviewGroup, SearchResult, SimilaritySearchRequest, SimilaritySearchService
from infra.logging_config import get_logger
from infra.qt_diagnostics import append_qt_diagnostic
from infra.settings import get_settings
from ui.common import HelpIconButton, build_help_inline
from ui.error_mbox import confirmBox, errorBox, infoBox
from ui.async_job import AsyncJob, Cancelled, detach_running_async_job, start_job_in_thread, wait_for_thread_shutdown
from ui.gallery_pane import GalleryPane
from ui.list_models import ListEntry, ListEntryModel, PagedListEntryModel
from ui.photo_inspector_dialog import EditableFaceDraft
from ui.theme import COLORS

try:
    from ui.job_manager import JobManager
except Exception:  # pragma: no cover
    JobManager = None  # type: ignore[assignment]


LOGGER = get_logger(__name__)
FACE_EXIF_PERSON_KEY = "ic_person_name"
FACE_RESULT_HOVER_PREVIEW_MAX_ITEMS = 6
FACE_RESULT_HOVER_PREVIEW_COLUMNS = 3
FACE_RESULT_HOVER_PREVIEW_TILE_SIZE = QSize(88, 88)
FACE_RESULT_HOVER_PREVIEW_IMAGE_SIZE = QSize(276, 182)
FACE_REVIEW_LAZY_PUBLISH_THRESHOLD = 256
FACE_REVIEW_HYDRATION_DEBOUNCE_MS = 48
FACE_DETECTED_FACE_PUBLISH_BATCH_SIZE = 96
FACE_DETECTED_FACE_PUBLISH_INTERVAL_MS = 1
FACE_DETECTED_FACE_PUBLISH_SLICE_BUDGET_MS = 8.0
FACE_TILE_VISIBLE_REQUEST_LIMIT = 48
FACE_TILE_PREFETCH_REQUEST_LIMIT = 24
# The normal Faces task pane is intentionally narrow.  Do not split the two
# primary "Find a Person" cards until there is enough room for both workflows.
FACE_SEARCH_TWO_COLUMN_MIN_WIDTH = 760


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


@dataclass(frozen=True)
class FaceLibraryUnlabeledGroup:
    image_path: str
    face_refs: tuple[tuple[str, int], ...]
    face_count: int
    records: tuple[IndexedFaceRecord, ...]


@dataclass(frozen=True)
class FaceTileItem:
    image_path: str
    face_index: int
    bbox: tuple[int, int, int, int]
    title: str
    tooltip: str
    status: str = "saved"
    draft_slot: int = -1
    saved_face_index: int = -1
    payload: object | None = None


@dataclass(frozen=True)
class FaceResultGroup:
    group_id: str
    title: str
    summary: str
    items: tuple[FaceTileItem, ...]
    comparison_key: str = ""
    cluster_id: int = -1
    explanation: ClusterExplanation | None = None
    suggestion: FaceClusterIdentitySuggestion | None = None
    cluster_status: str = ""
    review_kind: str = "raw"
    source_group_ids: tuple[str, ...] = ()
    resolved_name: str = ""
    exif_name: str = ""
    merge_notice: str = ""


@dataclass(frozen=True)
class FaceTileLoadRequest:
    cache_key: tuple[object, ...]
    image_path: str
    bbox: tuple[int, int, int, int]
    requested_size: tuple[int, int]
    padding_policy: str = "default_v1"


@dataclass(frozen=True)
class FaceResultFilterSnapshot:
    request_id: int
    groups: tuple[FaceResultGroup, ...]
    group_entries: tuple[ListEntry, ...]
    selected_group_id: str
    summary_text: str
    photo_paths_by_group_id: dict[str, list[str]]
    merged_groups: tuple[FaceResultGroup, ...] = ()
    merged_group_entries: tuple[ListEntry, ...] = ()
    merged_photo_paths_by_group_id: dict[str, list[str]] = field(default_factory=dict)
    exif_groups: tuple[FaceResultGroup, ...] = ()
    exif_group_entries: tuple[ListEntry, ...] = ()
    exif_photo_paths_by_group_id: dict[str, list[str]] = field(default_factory=dict)


@dataclass(frozen=True)
class FaceAlbumPublishSnapshot:
    request_id: int
    summary: str
    groups: tuple[FaceResultGroup, ...]
    photo_paths: tuple[str, ...]
    overlay_by_path: dict[str, str]
    context_by_path: dict[str, dict[str, object]]


@dataclass(frozen=True)
class FaceReviewPublishSnapshot:
    request_id: int
    folder: str
    all_review_images: tuple[FaceFolderReviewImage, ...]
    filtered_review_images: tuple[FaceFolderReviewImage, ...]
    paths: tuple[str, ...]
    overlay_by_path: dict[str, str]
    subtitle_by_path: dict[str, str]
    face_boxes_by_path: dict[str, list[tuple[float, float, float, float]]]
    face_box_states_by_path: dict[str, str]
    context_by_path: dict[str, dict[str, object]]
    review_by_path: dict[str, FaceFolderReviewImage]
    summary_text: str
    sort_mode: str = "name"
    lazy_publish: bool = False
    migrated_image_paths: tuple[str, ...] = ()
    status_text: str = ""


@dataclass(frozen=True)
class FaceReviewHydrationSnapshot:
    request_id: int
    publish_request_id: int
    paths: tuple[str, ...]
    overlay_by_path: dict[str, str]
    subtitle_by_path: dict[str, str]
    face_boxes_by_path: dict[str, list[tuple[float, float, float, float]]]
    face_box_states_by_path: dict[str, str]
    context_by_path: dict[str, dict[str, object]]


@dataclass(frozen=True)
class FacePhotoViewSnapshot:
    title: str
    paths: tuple[str, ...]
    overlay_by_path: dict[str, str]
    subtitle_by_path: dict[str, str]
    face_boxes_by_path: dict[str, list[tuple[float, float, float, float]]]
    face_box_states_by_path: dict[str, str]
    context_by_path: dict[str, dict[str, object]]
    selected_path: str = ""
    scroll_value: int = 0


@dataclass(frozen=True)
class FaceReviewSource:
    service: FaceIndexService
    folder: str
    scope_key: str
    mode: str
    db_path: str
    runtime_root: str
    indexed_image_count: int
    face_image_count: int
    face_count: int
    selection_reason: str


class FaceWorkspaceSplitter(QSplitter):
    def __init__(self, orientation: Qt.Orientation, parent=None) -> None:
        super().__init__(orientation, parent)
        self.requested_sizes: list[int] = []

    def setSizes(self, sizes) -> None:  # noqa: N802 - Qt API override
        try:
            self.requested_sizes = [max(0, int(value)) for value in list(sizes or [])]
        except Exception:
            self.requested_sizes = []
        super().setSizes(sizes)


class FlowLayout(QLayout):
    """A compact wrapping layout for action and filter rows.

    Faces can be hosted in a fairly narrow workspace. A normal horizontal
    layout contributes the sum of every control to its parent's minimum width,
    which makes the pane wider than the screen and clips both columns.
    """

    def __init__(self, parent=None, *, spacing: int = 6) -> None:
        super().__init__(parent)
        self._items: list[object] = []
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(max(0, int(spacing)))

    def __del__(self) -> None:
        while self.takeAt(0) is not None:
            pass

    def addItem(self, item) -> None:  # noqa: N802 - Qt API override
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802 - Qt API override
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int):  # noqa: N802 - Qt API override
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):  # noqa: N802 - Qt API override
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt API override
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt API override
        return self._do_layout(QRect(0, 0, max(0, int(width)), 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 - Qt API override
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API override
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt API override
        size = QSize()
        for item in self._items:
            if item is not None and not item.isEmpty():
                size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(), margins.top() + margins.bottom())

    def _do_layout(self, rect: QRect, *, test_only: bool) -> int:
        margins = self.contentsMargins()
        available = rect.adjusted(margins.left(), margins.top(), -margins.right(), -margins.bottom())
        x = available.x()
        y = available.y()
        line_height = 0
        spacing = max(0, self.spacing())

        for item in self._items:
            if item is None or item.isEmpty():
                continue
            hint = item.sizeHint().expandedTo(item.minimumSize())
            item_width = min(max(0, hint.width()), max(0, available.width()))
            next_x = x + item_width
            if line_height > 0 and next_x > available.right() + 1:
                x = available.x()
                y += line_height + spacing
                next_x = x + item_width
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(x, y, item_width, hint.height()))
            x = next_x + spacing
            line_height = max(line_height, hint.height())

        return max(0, y + line_height - rect.y() + margins.bottom())


class CurrentPageTabWidget(QTabWidget):
    """Size a hidden-tab task stack from its active page, not its tallest page."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.currentChanged.connect(lambda _index: self.updateGeometry())

    def _current_page_size_hint(self, *, minimum: bool) -> QSize:
        page = self.currentWidget()
        if page is None:
            return super().minimumSizeHint() if minimum else super().sizeHint()
        hint = page.minimumSizeHint() if minimum else page.sizeHint()
        frame = max(0, int(self.style().pixelMetric(QStyle.PixelMetric.PM_DefaultFrameWidth)))
        tab_height = self.tabBar().sizeHint().height() if self.tabBar().isVisible() else 0
        return QSize(max(0, hint.width()) + frame * 2, max(0, hint.height()) + frame * 2 + tab_height)

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API override
        return self._current_page_size_hint(minimum=True)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API override
        return self._current_page_size_hint(minimum=False)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt API override
        page = self.currentWidget()
        return bool(page is not None and page.hasHeightForWidth())

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt API override
        page = self.currentWidget()
        if page is None:
            return super().heightForWidth(width)
        frame = max(0, int(self.style().pixelMetric(QStyle.PixelMetric.PM_DefaultFrameWidth)))
        tab_height = self.tabBar().sizeHint().height() if self.tabBar().isVisible() else 0
        page_width = max(0, int(width) - frame * 2)
        page_height = page.heightForWidth(page_width) if page.hasHeightForWidth() else page.sizeHint().height()
        return max(0, int(page_height)) + frame * 2 + tab_height


class FaceResultHoverPreviewPopup(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("faceResultHoverPreviewPopup")
        self.setStyleSheet(
            f"""
            QFrame#faceResultHoverPreviewPopup {{
                background: {COLORS["surface_raised"]};
                border: 1px solid {COLORS["border_strong"]};
                border-radius: 8px;
            }}
            QLabel {{
                color: {COLORS["text"]};
            }}
            """
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        self.title_label = QLabel("")
        self.title_label.setWordWrap(True)
        title_font = self.title_label.font()
        title_font.setBold(True)
        self.title_label.setFont(title_font)
        layout.addWidget(self.title_label)

        self.image_label = QLabel("Loading preview...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(FACE_RESULT_HOVER_PREVIEW_IMAGE_SIZE)
        self.image_label.setStyleSheet(
            f'background: {COLORS["surface_sunken"]}; border: 1px solid {COLORS["border"]};'
        )
        layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.summary_label)

        self.note_label = QLabel("")
        self.note_label.setWordWrap(True)
        self.note_label.setStyleSheet(f'color: {COLORS["text_muted"]};')
        layout.addWidget(self.note_label)

        self.setFixedWidth(320)

    def set_group_details(self, group: FaceResultGroup, *, more_count: int, loading: bool) -> None:
        self.title_label.setText(str(group.title or "Face cluster"))
        self.summary_label.setText(str(group.summary or ""))
        if group.review_kind == "raw":
            note = "Right-click this raw cluster to name it now."
        else:
            note = "Named clusters update live as raw clusters are labeled."
        if more_count > 0:
            note = f"{note} +{more_count} more face(s) not shown in the preview."
        self.note_label.setText(note)
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


class FaceTileRequestQueue:
    def __init__(self) -> None:
        self._stop = Event()
        self._queue: PriorityQueue[tuple[int, int, FaceTileLoadRequest | None]] = PriorityQueue()
        self._lock = Lock()
        self._queued_keys: set[tuple[object, ...]] = set()
        self._seq = 0

    def cancel(self, sentinel_count: int | None = None) -> None:
        self._stop.set()
        count = max(1, int(sentinel_count or 4))
        for index in range(count):
            self._queue.put((-1, -index - 1, None))

    def clear(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                break
        with self._lock:
            self._queued_keys.clear()

    def enqueue(self, request: FaceTileLoadRequest, *, priority: int = 100) -> bool:
        if self._stop.is_set():
            return False
        with self._lock:
            if request.cache_key in self._queued_keys:
                return False
            self._queued_keys.add(request.cache_key)
            self._seq += 1
            self._queue.put((int(priority), int(self._seq), request))
        return True

    def get_next(self, timeout_s: float = 0.2) -> FaceTileLoadRequest | None:
        if self._stop.is_set():
            return None
        try:
            _priority, _seq, request = self._queue.get(timeout=timeout_s)
        except Empty:
            return None
        if request is None:
            return None
        with self._lock:
            self._queued_keys.discard(request.cache_key)
        return request


def _load_face_tile_qimage(request: FaceTileLoadRequest) -> QImage:
    requested_width = max(24, int(request.requested_size[0]))
    requested_height = max(24, int(request.requested_size[1]))
    x1, y1, x2, y2 = [int(value) for value in request.bbox]
    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1, x2)
    bottom = max(y1, y2)
    width = max(2, right - left)
    height = max(2, bottom - top)
    pad_x = max(8, width // 6)
    pad_y = max(8, height // 6)

    scaled: QImage | None = None
    reader = QImageReader(str(request.image_path))
    reader.setAutoTransform(True)
    image_size = reader.size()
    if image_size.isValid() and image_size.width() > 0 and image_size.height() > 0:
        crop_left = max(0, left - pad_x)
        crop_top = max(0, top - pad_y)
        crop_right = min(int(image_size.width()), right + pad_x)
        crop_bottom = min(int(image_size.height()), bottom + pad_y)
        crop_width = max(2, crop_right - crop_left)
        crop_height = max(2, crop_bottom - crop_top)
        try:
            reader.setClipRect(QRect(crop_left, crop_top, crop_width, crop_height))
            scale = min(requested_width / max(1, crop_width), requested_height / max(1, crop_height))
            reader.setScaledSize(
                QSize(
                    max(1, int(round(crop_width * scale))),
                    max(1, int(round(crop_height * scale))),
                )
            )
            clipped = reader.read()
            if not clipped.isNull():
                scaled = clipped
        except Exception:
            scaled = None

    if scaled is None or scaled.isNull():
        reader = QImageReader(str(request.image_path))
        reader.setAutoTransform(True)
        image = reader.read()
        if image.isNull():
            raise ValueError(f"failed to decode image: {request.image_path}")
        crop_left = max(0, left - pad_x)
        crop_top = max(0, top - pad_y)
        crop_right = min(image.width(), right + pad_x)
        crop_bottom = min(image.height(), bottom + pad_y)
        crop_width = max(2, crop_right - crop_left)
        crop_height = max(2, crop_bottom - crop_top)
        crop = image.copy(crop_left, crop_top, crop_width, crop_height)
        scaled = crop.scaled(
            QSize(requested_width, requested_height),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    canvas = QImage(requested_width, requested_height, QImage.Format.Format_ARGB32_Premultiplied)
    canvas.fill(QColor(18, 18, 18))
    painter = QPainter(canvas)
    offset_x = max(0, (requested_width - scaled.width()) // 2)
    offset_y = max(0, (requested_height - scaled.height()) // 2)
    painter.drawImage(offset_x, offset_y, scaled)
    painter.end()
    return canvas


def _call_with_optional_cancel(fn, *args, cancel_check=None, **kwargs):
    """Pass cancellation to current services without breaking older adapters."""
    try:
        parameters = inspect.signature(fn).parameters.values()
        accepts_cancel = any(
            parameter.name == "cancel_check" or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        )
    except (TypeError, ValueError):
        accepts_cancel = False
    if accepts_cancel:
        kwargs["cancel_check"] = cancel_check
    return fn(*args, **kwargs)


class FaceTileLoaderThread(QThread):
    tile_started = pyqtSignal(object)
    tile_loaded = pyqtSignal(object, object, float)
    tile_failed = pyqtSignal(object, str)

    def __init__(self, request_queue: FaceTileRequestQueue, parent=None) -> None:
        super().__init__(parent)
        self._request_queue = request_queue
        self.setObjectName(f"FaceTileLoaderThread:{id(self):x}")

    def run(self) -> None:
        append_qt_diagnostic(f"[ThreadStart] {self.objectName()}")
        while True:
            request = self._request_queue.get_next(timeout_s=0.2)
            if request is None:
                if self._request_queue._stop.is_set():  # noqa: SLF001
                    append_qt_diagnostic(f"[ThreadFinish] {self.objectName()} stopped=True")
                    return
                continue
            started_at = perf_counter()
            try:
                self.tile_started.emit(request.cache_key)
                image = _load_face_tile_qimage(request)
                self.tile_loaded.emit(request.cache_key, image, perf_counter() - started_at)
            except Exception as exc:
                self.tile_failed.emit(request.cache_key, str(exc))


class FaceTileListModel(QAbstractListModel):
    RefRole = Qt.ItemDataRole.UserRole + 1
    StatusRole = Qt.ItemDataRole.UserRole + 2
    PayloadRole = Qt.ItemDataRole.UserRole + 3
    CacheKeyRole = Qt.ItemDataRole.UserRole + 4
    ImageRole = Qt.ItemDataRole.UserRole + 5

    def __init__(self, image_provider, cache_key_provider, requested_icon_size: QSize, parent=None) -> None:
        super().__init__(parent)
        self._image_provider = image_provider
        self._cache_key_provider = cache_key_provider
        self._requested_icon_size = QSize(requested_icon_size)
        self._items: list[FaceTileItem] = []
        self._rows_by_identity: dict[tuple[object, ...], list[int]] = {}

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._items)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self._items)):
            return None
        item = self._items[index.row()]
        if role == Qt.ItemDataRole.DisplayRole:
            return item.title
        if role == Qt.ItemDataRole.AccessibleTextRole:
            state = str(item.status or "face")
            return f"{item.title}, {state}, face {index.row() + 1} of {len(self._items)}"
        if role == Qt.ItemDataRole.ToolTipRole:
            return item.tooltip
        if role == self.RefRole:
            return (item.image_path, int(item.draft_slot if item.draft_slot >= 0 else item.face_index))
        if role == self.StatusRole:
            return item.status
        if role == self.PayloadRole:
            return item.payload
        if role == self.CacheKeyRole:
            return self.cache_key_for_item(item)
        if role == self.ImageRole:
            return self._image_provider(item, self._requested_icon_size)
        return None

    def set_items(self, items: list[FaceTileItem]) -> None:
        self.beginResetModel()
        self._items = list(items)
        self._rows_by_identity = {}
        for row, item in enumerate(self._items):
            identity = self.row_identity_for_item(item)
            if identity is None:
                continue
            self._rows_by_identity.setdefault(identity, []).append(int(row))
        self.endResetModel()

    def append_items(self, items: list[FaceTileItem]) -> None:
        if not items:
            return
        start_row = len(self._items)
        end_row = start_row + len(items) - 1
        self.beginInsertRows(QModelIndex(), start_row, end_row)
        for offset, item in enumerate(list(items)):
            self._items.append(item)
            identity = self.row_identity_for_item(item)
            if identity is None:
                continue
            self._rows_by_identity.setdefault(identity, []).append(int(start_row + offset))
        self.endInsertRows()

    def item_at(self, row: int) -> FaceTileItem | None:
        if 0 <= int(row) < len(self._items):
            return self._items[int(row)]
        return None

    def requested_icon_size(self) -> QSize:
        return QSize(self._requested_icon_size)

    def cache_key_for_item(self, item: FaceTileItem) -> tuple[object, ...] | None:
        try:
            key = self._cache_key_provider(item, self._requested_icon_size)
            if isinstance(key, tuple):
                return key
        except Exception:
            return None
        return None

    def row_identity_for_item(self, item: FaceTileItem) -> tuple[object, ...] | None:
        image_path = str(getattr(item, "image_path", "") or "").strip()
        if not image_path:
            return None
        try:
            bbox = tuple(int(value) for value in getattr(item, "bbox", ()))
        except Exception:
            return None
        if len(bbox) != 4:
            return None
        return (
            image_path,
            bbox,
            int(self._requested_icon_size.width()),
            int(self._requested_icon_size.height()),
            "default_v1",
        )

    @staticmethod
    def _row_identity_from_cache_key(key: tuple[object, ...]) -> tuple[object, ...] | None:
        if not isinstance(key, tuple) or len(key) < 7:
            return None
        try:
            bbox = tuple(int(value) for value in key[3])
        except Exception:
            return None
        if len(bbox) != 4:
            return None
        return (
            str(key[0] or ""),
            bbox,
            int(key[4]),
            int(key[5]),
            str(key[6] or ""),
        )

    def rows_for_cache_key(self, key: tuple[object, ...]) -> list[int]:
        identity = self._row_identity_from_cache_key(key)
        if identity is None:
            return []
        return list(self._rows_by_identity.get(identity, ()))

    def notify_rows_changed(self, rows: list[int]) -> None:
        for row in sorted({int(value) for value in rows if int(value) >= 0}):
            index = self.index(row, 0)
            if index.isValid():
                self.dataChanged.emit(index, index, [self.ImageRole])


class FaceTileItemDelegate(QStyledItemDelegate):
    _background = QColor(COLORS["surface"])
    _placeholder_background = QColor(COLORS["surface_sunken"])
    _border = QColor(COLORS["border"])
    _selected_background = QColor(COLORS["surface_selected"])
    _selected_border = QColor(COLORS["info"])
    _review_border = QColor(COLORS["warning"])
    _draft_border = QColor(COLORS["danger_hover"])
    _pending_border = QColor(COLORS["identity_pending"])
    _text = QColor(COLORS["text"])
    _muted = QColor(COLORS["text_muted"])

    def sizeHint(self, option, index):  # type: ignore[override]
        model = index.model()
        requested_size = model.requested_icon_size() if hasattr(model, "requested_icon_size") else QSize(72, 72)
        return QSize(max(96, int(requested_size.width()) + 30), int(requested_size.height()) + 46)

    def paint(self, painter: QPainter, option, index) -> None:  # type: ignore[override]
        model = index.model()
        requested_size = model.requested_icon_size() if hasattr(model, "requested_icon_size") else QSize(72, 72)
        content_rect = option.rect.adjusted(4, 4, -4, -4)
        image_rect = QRect(
            content_rect.x() + max(0, (content_rect.width() - requested_size.width()) // 2),
            content_rect.y() + 2,
            int(requested_size.width()),
            int(requested_size.height()),
        )
        status = str(index.data(FaceTileListModel.StatusRole) or "").strip().lower()
        is_selected = bool(option.state & QStyle.StateFlag.State_Selected)
        border_color = self._border
        if status == "review":
            border_color = self._review_border
        elif status == "draft":
            border_color = self._draft_border
        elif status == "pending":
            border_color = self._pending_border
        if is_selected:
            border_color = self._selected_border

        painter.save()
        painter.fillRect(content_rect, self._selected_background if is_selected else self._background)
        image = index.data(FaceTileListModel.ImageRole)
        if isinstance(image, QImage) and not image.isNull():
            painter.drawImage(image_rect, image)
        else:
            painter.fillRect(image_rect, self._placeholder_background)
        painter.setPen(border_color)
        painter.drawRect(image_rect.adjusted(0, 0, -1, -1))

        title = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        text_rect = QRect(
            content_rect.x() + 2,
            image_rect.bottom() + 6,
            max(1, content_rect.width() - 4),
            max(1, content_rect.bottom() - image_rect.bottom() - 6),
        )
        text_font = QFont(option.font)
        text_font.setPointSize(max(7, text_font.pointSize() - 1))
        painter.setFont(text_font)
        metrics = QFontMetrics(text_font)
        line_height = metrics.lineSpacing()
        lines = [line for line in title.splitlines() if line] or [title]
        if len(lines) == 1:
            lines.append("")
        painter.setPen(self._text)
        first_line = metrics.elidedText(lines[0], Qt.TextElideMode.ElideMiddle, text_rect.width())
        painter.drawText(text_rect.x(), text_rect.y() + metrics.ascent(), first_line)
        painter.setPen(self._muted)
        second_line = metrics.elidedText(lines[1], Qt.TextElideMode.ElideMiddle, text_rect.width())
        painter.drawText(text_rect.x(), text_rect.y() + line_height + metrics.ascent(), second_line)
        painter.restore()


FACE_HELP = {
    "current_folder_toggle": (
        "When enabled, face searches and clustering prefer the current folder instead of the full face database "
        "when that workflow supports folder scoping."
    ),
    "face_mode": (
        "Choose the active face identity workspace.\n"
        "Each mode uses its own face index and saved identities."
    ),
    "face_detector": (
        "Choose the detector used for the current face mode.\n"
        "YOLO-style ONNX models are supported here as detector bundles."
    ),
    "face_embedder": (
        "Choose the embedder used for matching, clustering, and saved identities for the current face mode."
    ),
    "face_model_profile": (
        "Choose a curated detector/embedder pairing.\n"
        "Accuracy favors quality, Balanced is a practical default, Edge favors smaller models.\n"
        "Custom means the current detector, embedder, or thresholds do not match a curated profile."
    ),
    "face_detector_score_threshold": (
        "Minimum detector confidence for newly scanned faces in the current mode."
    ),
    "face_max_detections": (
        "Maximum number of faces to keep per image for the current detector."
    ),
    "face_quality_profile": (
        "Choose the quality gate preset used to classify faces as clean, review, or reject.\n"
        "High Recall keeps more borderline faces, Balanced is the practical default, and High Precision hides more junk."
    ),
    "face_detector_policy": (
        "Choose whether Faces should use only the primary detector, fall back when it misses faces, or combine two detectors."
    ),
    "face_fallback_detector": (
        "Optional secondary detector used by the active detector policy for the current face mode."
    ),
    "face_verifier_mode": (
        "Optional face-vs-junk verifier gate applied after detection. This is a lightweight local verifier, not an identity model."
    ),
    "face_search_quality_min": (
        "Minimum saved face quality allowed in search results. Clean only is the safest default."
    ),
    "face_cluster_quality_min": (
        "Minimum saved face quality allowed for clustering. Clean only is the safest default."
    ),
    "face_prototype_quality_min": (
        "Minimum face quality required when building identity prototypes from selected faces or example photos."
    ),
    "face_recognition_min_score": (
        "Lower bound for face-search similarity, even when a search UI value is smaller."
    ),
    "face_auto_label_min_score": (
        "Lower bound used when auto-applying saved names to indexed faces."
    ),
    "face_rerank_policy": (
        "Optional rerank pass for top search hits. Quality score promotes cleaner faces ahead of borderline matches."
    ),
    "face_rerank_top_n": (
        "How many top raw search hits are eligible for reranking."
    ),
    "show_tiny_detections": (
        "Show very small face detections that are hidden by default.\n"
        "These are often false positives from timestamps, text overlays, or background details."
    ),
    "face_review_sort": (
        "Sort the Face Library gallery by filename, number of detected faces, or strongest visible face.\n"
        "Detected photos always stay ahead of no-face and unscanned photos."
    ),
    "library_quick_start": (
        "1. Choose a folder from the left sidebar or the Face Library folder box.\n"
        "2. Click Scan Folder for Faces.\n"
        "3. Review the folder gallery below.\n"
        "4. Green boxes mark indexed faces on each photo.\n"
        "5. Click a photo to work with the detected faces for that one image."
    ),
    "scan_source": (
        "Pick where face records are stored and which folder should be scanned.\n"
        "Session only keeps a temporary face database for the current app session."
    ),
    "face_database_path": (
        "The exact SQLite database used by the saved face library.\n"
        "ClusterLens stores face indexes in its cache and never writes them into the source-photo folder."
    ),
    "scan_folder_button": (
        "Detect and index faces for the chosen folder.\n"
        "Run this before reviewing unlabeled faces in a new folder.\n"
        "Re-scanning also replaces old tiny detections with the stricter detector output."
    ),
    "reload_people_button": (
        "Reload the saved identities and unlabeled groups list for the active folder.\n"
        "This list combines saved identities and unlabeled scanned photos."
    ),
    "reload_faces_button": (
        "Reload the folder review gallery for the active folder.\n"
        "Use this after scanning, changing DB scope, or toggling tiny detections."
    ),
    "auto_clean_review": (
        "Apply draft-only cleanup to the current review scope.\n"
        "This removes obvious junk detections and leaves borderline faces for manual review.\n"
        "Save Face Edits later to commit any image-level cleanup."
    ),
    "people_groups": (
        "Saved identities and unlabeled scanned photos appear here.\n"
        "Clicking a saved identity fills the profile fields below.\n"
        "Clicking an unlabeled group selects its face thumbnails."
    ),
    "scanned_faces": (
        "These thumbnails are the detected faces for the currently selected photo.\n"
        "Very small detections are hidden by default because they are often not real faces.\n"
        "Select one or more to name them.\n"
        "Select exactly one to search for matching photos."
    ),
    "tiny_detection_note": (
        "If a face tile looks like stripes, digits, or a background fragment, it is usually a tiny false-positive detection.\n"
        "Use Show Tiny Detections only when you need to inspect those records."
    ),
    "saved_person_name": (
        "Type or load a saved identity name here, then use Find Photos by Saved Name."
    ),
    "selected_face_name": (
        "Enter the name you want to assign to the selected face thumbnails."
    ),
    "profile_notes": "Optional notes stored with the saved identity profile.",
    "profile_tags": "Optional comma-separated tags stored with the saved identity profile.",
    "label_threshold": (
        "Similarity threshold used when saving or applying an identity name.\n"
        "Higher values are stricter."
    ),
    "name_search_top_k": "Maximum number of images to return for a saved-name search.",
    "name_search_min_score": (
        "Minimum similarity score for saved-name search results.\n"
        "Use 0 to fall back to the saved threshold."
    ),
    "result_folder_filter": (
        "Optional folder filter for saved-name searches and selected-face searches."
    ),
    "selected_face_top_k": "Maximum number of images to return when searching from one selected face.",
    "selected_face_min_score": "Minimum similarity score for selected-face search results.",
    "shown_faces_limit": "Maximum number of face records loaded into the face lists at once.",
    "name_selected_faces": (
        "Save the typed name onto the selected face thumbnails and refresh the current folder."
    ),
    "save_profile": (
        "Save or update notes, tags, and cover face for the current identity name."
    ),
    "find_photos_selected_face": (
        "Search the face database for photos that match the one selected face thumbnail."
    ),
    "find_photos_saved_name": (
        "Search for photos that match the saved identity name in the Saved identity field."
    ),
    "search_quick_start": (
        "Use Face Library to review the folder and pick exact faces. Use this tab to name selected faces, work with saved identities, run query-photo searches, cluster faces, and manage labels."
    ),
    "query_face_image": (
        "Choose one photo containing the identity you want to find.\n"
        "The strongest detected face in the photo is used as the query."
    ),
    "index_current_folder": (
        "Index faces for the current folder before using the query-photo tools on a new folder."
    ),
    "find_same_person": (
        "Find photos that match the chosen query face image."
    ),
    "few_shot_examples": (
        "Optional extra example photos for saving a named identity.\n"
        "Use semicolon-separated paths when adding more than one."
    ),
    "save_named_examples": (
        "Save the query photo and any example photos as a reusable named identity."
    ),
    "auto_apply_saved_names": (
        "Apply saved identities to unlabeled indexed faces across the current face database."
    ),
    "show_named_faces": (
        "Show all currently saved named-face assignments in the results gallery."
    ),
    "cluster_faces": (
        "Group similar indexed faces together for review.\n"
        "Use the current folder filter when enabled."
    ),
    "face_cluster_count": "Requested number of face clusters to generate.",
    "face_cluster_backend": "Choose how unknown faces are grouped. HDBSCAN is the default review-first option.",
    "pending_face_review": (
        "Review proposed face-name assignments here before they become permanent labels.\n"
        "Accept writes face labels. Reject discards the proposal."
    ),
    "merge_names": (
        "Move all labels from the source identity into the target identity.\n"
        "Use this when two saved names represent the same identity."
    ),
    "remove_name": (
        "Remove all saved face labels for the typed identity name.\n"
        "This does not delete the original images."
    ),
    "merge_source": "Person name to merge from.",
    "merge_target": "Person name to merge into.",
}

FACE_QUALITY_NUMERIC_FIELDS: tuple[tuple[str, str, float, float, float], ...] = (
    ("reject_confidence", "Reject confidence", 0.0, 1.0, 0.01),
    ("review_confidence", "Review confidence", 0.0, 1.0, 0.01),
    ("min_face_side_px", "Min face side", 1.0, 1000.0, 1.0),
    ("min_face_area_px", "Min face area", 1.0, 1000000.0, 10.0),
    ("max_edge_clip_fraction", "Review edge clip", 0.0, 1.0, 0.01),
    ("max_reject_edge_clip_fraction", "Reject edge clip", 0.0, 1.0, 0.01),
    ("min_bbox_aspect_ratio", "Min aspect", 0.1, 4.0, 0.01),
    ("max_bbox_aspect_ratio", "Max aspect", 0.1, 4.0, 0.01),
    ("min_sharpness_review", "Review sharpness", 0.0, 5000.0, 1.0),
    ("min_sharpness_clean", "Clean sharpness", 0.0, 5000.0, 1.0),
    ("min_luma", "Min luma", 0.0, 255.0, 1.0),
    ("max_luma", "Max luma", 0.0, 255.0, 1.0),
    ("min_contrast", "Min contrast", 0.0, 255.0, 1.0),
    ("duplicate_iou_threshold", "Duplicate IoU", 0.0, 1.0, 0.01),
)

FACE_QUALITY_POLICY_FIELDS: tuple[tuple[str, str], ...] = (
    ("landmarks_policy", "Landmarks"),
    ("alignment_policy", "Alignment"),
)


class _LazyServiceProxy:
    def __init__(self, factory):
        self._factory = factory
        self._instance = None
        self._lock = Lock()

    def _get(self):
        if self._instance is not None:
            return self._instance
        with self._lock:
            if self._instance is None:
                self._instance = self._factory()
        return self._instance

    def __getattr__(self, name: str):
        return getattr(self._get(), name)


class SearchPane(QWidget):
    open_in_gallery_requested = pyqtSignal(list)
    append_to_gallery_requested = pyqtSignal(list)
    face_clusters_ready = pyqtSignal(dict)
    saved_clustering_filter_requested = pyqtSignal(dict)
    open_face_model_settings_requested = pyqtSignal()
    face_pipeline_controls_requested = pyqtSignal()
    face_pipeline_applied = pyqtSignal(dict)
    install_face_model_requested = pyqtSignal(str)
    result_selected = pyqtSignal(str, list, int, object)
    active_tab_changed = pyqtSignal(int)
    results_ready = pyqtSignal(list, object, object, object, str)
    source_folder_changed = pyqtSignal(str)
    recent_folder_remove_requested = pyqtSignal(str)
    recent_folders_clear_requested = pyqtSignal()

    def __init__(
        self,
        parent=None,
        *,
        search_service_global=None,
        search_service_session=None,
        face_service_global=None,
        face_service_session=None,
        face_services_global: dict[str, FaceIndexService] | None = None,
        face_services_session: dict[str, FaceIndexService] | None = None,
        enabled_tabs: list[str] | None = None,
        external_results: bool = False,
        saved_search_service: SavedSearchService | None = None,
        supported_face_modes: list[str] | tuple[str, ...] | None = None,
    ):
        super().__init__(parent)
        self.settings = get_settings()
        self.current_directory_provider = None
        self.current_scope_paths_provider = None
        self.use_onnx_provider = None  # injected callable() -> bool
        self.job_manager: JobManager | None = None  # injected by main window
        self.saved_search_service = saved_search_service or SavedSearchService()
        self.clustering_filter_state_provider = None
        self.search_service_global = search_service_global or _LazyServiceProxy(SimilaritySearchService)
        self.search_service_session = search_service_session or _LazyServiceProxy(SimilaritySearchService)
        self.search_service = self.search_service_global
        self.face_service_global = face_service_global or FaceIndexService()
        self.face_service_session = face_service_session or FaceIndexService(db_path=(self.settings.cache_dir / "face_search_session.db"), reset_db=True)
        self.face_service = self.face_service_global
        self.face_services_global_by_mode = {"human": self.face_service_global}
        self.face_services_session_by_mode = {"human": self.face_service_session}
        self.face_service_provider = None
        self.face_model_root = ""
        self.supported_face_modes = self._normalize_supported_face_modes(supported_face_modes)
        self._last_face_mode = "human"
        self._face_pipeline_prefs_by_mode: dict[str, dict[str, object]] = {
            "human": {
                "detector_id": BUILTIN_HUMAN_DETECTOR_ID,
                "embedder_id": BUILTIN_HUMAN_EMBEDDER_ID,
                "fallback_detector_id": BUILTIN_HUMAN_DETECTOR_ID,
                "detector_policy": "single",
                "verifier_mode": "off",
                "score_threshold": DEFAULT_FACE_SCORE_THRESHOLD,
                "max_detections": DEFAULT_FACE_MAX_DETECTIONS,
                "quality_profile_id": "balanced",
                "quality_thresholds": dict(face_quality_profile_config("human", "balanced")),
                "search_quality_min": "clean",
                "cluster_quality_min": "clean",
                "prototype_quality_min": "clean",
                "recognition_min_score": 0.35,
                "auto_label_min_score": 0.72,
                "rerank_policy": "off",
                "rerank_top_n": 25,
            },
            "dog": {
                "detector_id": LEGACY_DEFAULT_BUNDLE_ID,
                "embedder_id": LEGACY_DEFAULT_BUNDLE_ID,
                "fallback_detector_id": LEGACY_DEFAULT_BUNDLE_ID,
                "detector_policy": "single",
                "verifier_mode": "off",
                "score_threshold": DEFAULT_FACE_SCORE_THRESHOLD,
                "max_detections": DEFAULT_FACE_MAX_DETECTIONS,
                "quality_profile_id": "balanced",
                "quality_thresholds": dict(face_quality_profile_config("dog", "balanced")),
                "search_quality_min": "clean",
                "cluster_quality_min": "clean",
                "prototype_quality_min": "clean",
                "recognition_min_score": 0.35,
                "auto_label_min_score": 0.72,
                "rerank_policy": "off",
                "rerank_top_n": 25,
            },
            "cat": {
                "detector_id": LEGACY_DEFAULT_BUNDLE_ID,
                "embedder_id": LEGACY_DEFAULT_BUNDLE_ID,
                "fallback_detector_id": LEGACY_DEFAULT_BUNDLE_ID,
                "detector_policy": "single",
                "verifier_mode": "off",
                "score_threshold": DEFAULT_FACE_SCORE_THRESHOLD,
                "max_detections": DEFAULT_FACE_MAX_DETECTIONS,
                "quality_profile_id": "balanced",
                "quality_thresholds": dict(face_quality_profile_config("cat", "balanced")),
                "search_quality_min": "clean",
                "cluster_quality_min": "clean",
                "prototype_quality_min": "clean",
                "recognition_min_score": 0.35,
                "auto_label_min_score": 0.72,
                "rerank_policy": "off",
                "rerank_top_n": 25,
            },
        }
        self._face_pipeline_applied_by_mode: dict[str, dict[str, object]] = {
            mode: self._clone_face_pipeline_prefs(prefs)
            for mode, prefs in self._face_pipeline_prefs_by_mode.items()
        }
        self._face_pipeline_dirty_modes: set[str] = set()
        self._face_pipeline_dialog: QDialog | None = None
        if face_services_global:
            self.face_services_global_by_mode.update(
                {
                    normalize_face_mode(mode): service
                    for mode, service in dict(face_services_global).items()
                    if service is not None
                }
            )
        if face_services_session:
            self.face_services_session_by_mode.update(
                {
                    normalize_face_mode(mode): service
                    for mode, service in dict(face_services_session).items()
                    if service is not None
                }
            )
        self._active_thread = None
        self._active_job = None
        self._active_job_id: int | None = None
        self._thread_jobs: dict[object, object | None] = {}
        self._action_buttons: list[QPushButton] = []
        self._mode_required_buttons: list[QPushButton] = []
        self._temp_query_files: dict[int, str] = {}
        self._auto_reindexed: set[tuple[str, str]] = set()
        self._last_results = []
        self._result_by_path: dict[str, SearchResult] = {}
        self._duplicate_review_groups: list[DuplicateReviewGroup] = []
        self._duplicate_decisions: list[dict[str, object]] = []
        self._face_thumb_cache: OrderedDict[tuple[object, ...], QImage] = OrderedDict()
        self._face_thumb_cache_size = max(64, int(self.settings.thumbnail_cache_size))
        self._face_thumb_placeholder_cache: dict[tuple[int, int], QImage] = {}
        self._face_thumb_failed_placeholder_cache: dict[tuple[int, int], QImage] = {}
        self._face_tile_cache_generation = 0
        self._face_tile_path_generations: dict[str, int] = {}
        self._face_tile_request_queue = FaceTileRequestQueue()
        self._face_tile_loader_threads: list[FaceTileLoaderThread] = []
        self._face_tile_loader_target_count = max(1, int(self.settings.max_thumbnail_workers))
        self._face_tile_pending_keys: set[tuple[object, ...]] = set()
        self._face_tile_inflight_keys: set[tuple[object, ...]] = set()
        self._face_tile_failed_keys: set[tuple[object, ...]] = set()
        self._face_tile_failure_messages: dict[tuple[object, ...], str] = {}
        self._detected_faces_context_menu: QMenu | None = None
        self._face_results_context_menu: QMenu | None = None
        self._results_kind = "none"  # similarity|faces|faces_review|face_clusters|labels|none
        self._face_review_by_path: dict[str, FaceFolderReviewImage] = {}
        self._face_review_selected_path = ""
        self._face_review_folder = ""
        self._face_review_all_images: list[FaceFolderReviewImage] = []
        self._face_review_images: list[FaceFolderReviewImage] = []
        self._face_review_sorted_mode = ""
        self._face_review_lazy_publish_enabled = False
        self._face_review_hydrated_paths: set[str] = set()
        self._face_review_pending_hydration_paths: set[str] = set()
        self._face_selection_sync_in_progress = False
        self._active_face_selection_source = ""
        self._active_face_selection_items: list[FaceTileItem] = []
        self._face_profiles_by_name: dict[str, PersonProfile] = {}
        self._face_identity_duplicates: dict[str, tuple[tuple[str, float], ...]] = {}
        self._face_identity_prototypes_by_name: dict[str, list[PersonPrototypeFace]] = {}
        self._face_identity_selected_name = ""
        self._face_review_drafts_by_path: dict[str, list[EditableFaceDraft]] = {}
        self._face_review_draft_dirty_paths: set[str] = set()
        self._face_review_context_by_path: dict[str, dict[str, object]] = {}
        self._face_detected_face_items_by_image: dict[str, list[int]] = {}
        self._face_detected_visible_scope_ref_count = 0
        self._face_detected_publish_request_id = 0
        self._face_detected_publish_active_request_id = 0
        self._face_detected_publish_pending_review_images: list[FaceFolderReviewImage] = []
        self._face_detected_publish_total_faces = 0
        self._face_detected_publish_processed_images = 0
        self._face_detected_publish_target_image_count = 0
        self._face_detected_publish_in_progress = False
        self._face_result_groups: list[FaceResultGroup] = []
        self._face_result_source_groups: list[FaceResultGroup] = []
        self._face_result_group_by_id: dict[str, FaceResultGroup] = {}
        self._face_result_merged_groups: list[FaceResultGroup] = []
        self._face_result_merged_group_by_id: dict[str, FaceResultGroup] = {}
        self._face_result_exif_groups: list[FaceResultGroup] = []
        self._face_result_exif_group_by_id: dict[str, FaceResultGroup] = {}
        self._face_result_selected_group_id = ""
        self._face_result_selected_group_id_by_kind: dict[str, str] = {"raw": "", "merged_name": "", "exif_name": ""}
        self._face_result_active_group_kind = "raw"
        self._face_result_source_summary = ""
        self._face_result_source_kind = ""
        self._face_result_source_context_by_path: dict[str, dict[str, object]] = {}
        self._face_result_source_overlay_by_path: dict[str, str] = {}
        self._face_result_source_photo_paths: list[str] = []
        self._face_result_photo_paths_by_group_id: dict[str, list[str]] = {}
        self._face_result_merged_photo_paths_by_group_id: dict[str, list[str]] = {}
        self._face_result_exif_photo_paths_by_group_id: dict[str, list[str]] = {}
        self._face_result_selection_syncing = False
        self._face_result_hover_popup: FaceResultHoverPreviewPopup | None = None
        self._face_result_hover_group_kind = ""
        self._face_result_hover_group_id = ""
        self._face_result_hover_view: QListView | None = None
        self._face_cluster_compare_membership: dict[tuple[str, int], dict[str, dict[str, object]]] = {}
        self._face_cluster_compare_metrics: dict[str, dict[str, object]] = {}
        self._face_cluster_compare_last_refs: list[tuple[str, int]] = []
        self._face_album_groups: list[FaceAlbumGroupSummary] = []
        self._face_album_records: list[FaceAlbumRecord] = []
        self._face_album_total_groups = 0
        self._face_album_next_group_offset: int | None = None
        self._face_album_loaded_group_ids: set[str] = set()
        self._face_album_member_next_offsets: dict[str, int | None] = {}
        self._face_album_member_totals: dict[str, int] = {}
        self._face_album_loaded = False
        self._face_album_refresh_job = None
        self._face_album_refresh_thread = None
        self._face_album_refresh_job_id: int | None = None
        self._retained_face_album_refresh_refs: list[tuple[object | None, object | None]] = []
        self._face_album_request_id = 0
        self._face_album_refresh_in_progress = False
        self._face_album_publish_job = None
        self._face_album_publish_thread = None
        self._retained_face_album_publish_refs: list[tuple[object | None, object | None]] = []
        self._pending_face_assignments: list[FaceLabelAssignment] = []
        self._last_pending_acceptance_count = 0
        self._face_hidden_rejected_refs: set[tuple[str, int]] = set()
        self._face_query_detected_faces: list[object] = []
        self._enabled_tabs = {str(name).strip().lower() for name in (enabled_tabs or []) if str(name).strip()}
        self._all_faces_tab_label = "All Faces"
        self._face_folder_tab_label = (
            "Folder Review"
            if "folder review" in self._enabled_tabs
            else ("Folder Review" if (not self._enabled_tabs or "faces in folder" in self._enabled_tabs) else "Face Library")
        )
        self._external_results = bool(external_results)
        self._face_ui_mode = "basic"
        self._read_only_mode = False
        self._recent_directories: list[str] = []
        self._face_library_layout_mode = ""
        self._face_search_layout_mode = ""
        self._face_review_publish_in_progress = False
        self._face_review_publish_request_id = 0
        self._face_review_publish_job = None
        self._face_review_publish_thread = None
        self._retained_face_review_publish_refs: list[tuple[object | None, object | None]] = []
        self._face_review_hydration_request_id = 0
        self._face_review_hydration_job = None
        self._face_review_hydration_thread = None
        self._retained_face_review_hydration_refs: list[tuple[object | None, object | None]] = []
        self._face_refresh_in_progress = False
        self._face_refresh_pending = False
        self._face_refresh_pending_people = False
        self._face_refresh_pending_reason = ""
        self._face_refresh_job = None
        self._face_refresh_thread = None
        self._face_refresh_job_id: int | None = None
        self._retained_face_refresh_refs: list[tuple[object | None, object | None]] = []
        self._face_refresh_request_id = 0
        self._face_refresh_force = False
        self._face_refresh_people_requested = True
        self._face_refresh_reason = ""
        self._face_result_filter_request_id = 0
        self._face_result_filter_job = None
        self._face_result_filter_thread = None
        self._retained_face_result_filter_refs: list[tuple[object | None, object | None]] = []
        self._face_discovery_cache: dict[tuple[str, bool], dict[str, object]] = {}
        self._face_review_source: FaceReviewSource | None = None
        self._face_review_source_key: tuple[str, ...] | None = None
        self._face_review_service_cache: dict[tuple[str, str, str, str, str], FaceIndexService] = {}
        self._face_review_service_cache_lock = Lock()
        self._results_gallery_publish_token = 0
        self._folder_photo_snapshot: FacePhotoViewSnapshot | None = None
        self._face_photo_context_kind = "folder"
        self._face_photo_context_title = "Folder photos"
        self._folder_photo_restore_path = ""
        self._folder_photo_restore_scroll = 0
        self._face_splitter_default_sizes = [360, 1240]
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(8)
        self._face_result_filter_timer = QTimer(self)
        self._face_result_filter_timer.setSingleShot(True)
        self._face_result_filter_timer.timeout.connect(self._start_face_result_filter_job)
        self._face_tile_refresh_timer = QTimer(self)
        self._face_tile_refresh_timer.setSingleShot(True)
        self._face_tile_refresh_timer.timeout.connect(self._flush_face_tile_refresh)
        self._face_detected_publish_timer = QTimer(self)
        self._face_detected_publish_timer.setSingleShot(True)
        self._face_detected_publish_timer.timeout.connect(self._flush_detected_faces_publish_batch)
        self._face_review_hydration_timer = QTimer(self)
        self._face_review_hydration_timer.setSingleShot(True)
        self._face_review_hydration_timer.timeout.connect(self._flush_face_review_hydration_paths)
        self.workspace_splitter = FaceWorkspaceSplitter(Qt.Orientation.Horizontal, self)
        self.workspace_splitter.setChildrenCollapsible(False)
        self.workspace_splitter.setHandleWidth(6)
        self.sidebar_panel = QWidget(self.workspace_splitter)
        self.sidebar_panel.setMinimumWidth(280)
        self.sidebar_panel.setMaximumWidth(520)
        self.sidebar_panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.sidebar_layout = QVBoxLayout(self.sidebar_panel)
        self.sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self.sidebar_layout.setSpacing(8)
        self.gallery_panel = QWidget(self.workspace_splitter)
        self.gallery_panel.setMinimumWidth(400)
        self.gallery_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.gallery_layout = QVBoxLayout(self.gallery_panel)
        self.gallery_layout.setContentsMargins(0, 0, 0, 0)
        self.gallery_layout.setSpacing(8)
        self._build_global_controls()
        self.tabs = CurrentPageTabWidget()
        self.results_gallery = GalleryPane(self)
        self.results_gallery.set_empty_state(
            "No face results yet",
            "Choose a Faces task, select a folder, and start a scan or search.",
        )
        self.results_gallery.review_action_mode = "add"
        self.face_review_results_tabs = QTabWidget(self)
        self.face_review_results_tabs.setDocumentMode(True)
        self.face_detected_faces_panel = QWidget(self.face_review_results_tabs)
        detected_layout = QVBoxLayout(self.face_detected_faces_panel)
        detected_layout.setContentsMargins(0, 0, 0, 0)
        detected_layout.setSpacing(8)
        self.face_detected_faces_summary = QLabel(
            "Faces shows every visible face in the active folder review."
        )
        self.face_detected_faces_summary.setWordWrap(True)
        detected_layout.addWidget(self.face_detected_faces_summary)
        detected_actions = FlowLayout()
        detected_actions.setContentsMargins(0, 0, 0, 0)
        self.face_detected_find_button = QPushButton("Find Similar")
        self.face_detected_cluster_selected_button = QPushButton("Cluster Selected")
        self.face_detected_cluster_visible_button = QPushButton("Cluster Visible")
        self.face_detected_name_button = QPushButton("Name Selected")
        self.face_detected_name_button.setProperty("kind", "primary")
        self.face_detected_remove_button = QPushButton("Remove Selected Faces")
        self.face_detected_jump_button = QPushButton("Show Photo")
        self.face_detected_edit_button = QPushButton("Open Inspector")
        for button in (
            self.face_detected_find_button,
            self.face_detected_cluster_selected_button,
            self.face_detected_cluster_visible_button,
            self.face_detected_name_button,
            self.face_detected_remove_button,
            self.face_detected_jump_button,
            self.face_detected_edit_button,
        ):
            button.setEnabled(False)
        self.face_detected_find_button.clicked.connect(self._search_selected_detected_face)
        self.face_detected_cluster_selected_button.clicked.connect(self._cluster_selected_detected_faces)
        self.face_detected_cluster_visible_button.clicked.connect(self._cluster_visible_detected_faces)
        self.face_detected_name_button.clicked.connect(self._prepare_name_selected_detected_faces)
        self.face_detected_remove_button.clicked.connect(self._remove_selected_detected_face_tiles)
        self.face_detected_jump_button.clicked.connect(self._jump_to_selected_detected_face)
        self.face_detected_edit_button.clicked.connect(self._open_selected_detected_face_in_inspector)
        self._action_buttons.extend(
            [
                self.face_detected_find_button,
                self.face_detected_cluster_selected_button,
                self.face_detected_cluster_visible_button,
                self.face_detected_name_button,
                self.face_detected_remove_button,
                self.face_detected_jump_button,
                self.face_detected_edit_button,
            ]
        )
        self._mode_required_buttons.extend(
            [
                self.face_detected_find_button,
                self.face_detected_cluster_selected_button,
                self.face_detected_cluster_visible_button,
                self.face_detected_name_button,
            ]
        )
        detected_actions.addWidget(self.face_detected_find_button)
        detected_actions.addWidget(self.face_detected_cluster_selected_button)
        detected_actions.addWidget(self.face_detected_cluster_visible_button)
        detected_actions.addWidget(self.face_detected_name_button)
        detected_actions.addWidget(self.face_detected_remove_button)
        detected_actions.addWidget(self.face_detected_jump_button)
        detected_actions.addWidget(self.face_detected_edit_button)
        detected_layout.addLayout(detected_actions)
        self.face_detected_faces_model = FaceTileListModel(
            self._image_for_face_tile,
            self._face_tile_cache_key_for_item,
            QSize(88, 88),
            self,
        )
        self.face_detected_faces_list = QListView()
        self.face_detected_faces_list.setViewMode(QListView.ViewMode.IconMode)
        self.face_detected_faces_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.face_detected_faces_list.setMovement(QListView.Movement.Static)
        self.face_detected_faces_list.setLayoutMode(QListView.LayoutMode.Batched)
        self.face_detected_faces_list.setBatchSize(48)
        self.face_detected_faces_list.setUniformItemSizes(True)
        self.face_detected_faces_list.setWrapping(True)
        self.face_detected_faces_list.setWordWrap(True)
        self.face_detected_faces_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.face_detected_faces_list.setToolTip(
            "Select faces to work with them together. Shift-click selects a range; "
            "Ctrl-click adds or removes an individual face. Right-click a selected "
            "face and choose Name Selected Faces to give every selected face one name."
        )
        self.face_detected_faces_list.setStatusTip(
            "Shift-click selects a range; Ctrl-click adds/removes faces; right-click selected faces to name them."
        )
        self.face_detected_faces_list.setIconSize(QSize(88, 88))
        self.face_detected_faces_list.setGridSize(QSize(118, 138))
        self.face_detected_faces_list.setSpacing(10)
        self.face_detected_faces_list.setModel(self.face_detected_faces_model)
        self.face_detected_faces_list.setItemDelegate(FaceTileItemDelegate(self.face_detected_faces_list))
        self.face_detected_faces_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.face_detected_faces_list.doubleClicked.connect(lambda _index: self._open_selected_detected_face_in_inspector())
        self.face_detected_faces_list.verticalScrollBar().valueChanged.connect(lambda _value: self._schedule_detected_face_tile_loads())
        self.face_detected_faces_list.customContextMenuRequested.connect(self._show_detected_faces_context_menu)
        detected_selection_model = self.face_detected_faces_list.selectionModel()
        if detected_selection_model is not None:
            detected_selection_model.selectionChanged.connect(lambda *_args: self._on_detected_face_selection_changed())
        detected_layout.addWidget(self.face_detected_faces_list, stretch=1)
        self.face_results_panel = QWidget(self.face_review_results_tabs)
        face_results_layout = QVBoxLayout(self.face_results_panel)
        face_results_layout.setContentsMargins(0, 0, 0, 0)
        face_results_layout.setSpacing(8)
        self.face_results_summary = QLabel("Face Groups shows named photos, clustering, and search results.")
        self.face_results_summary.setWordWrap(True)
        face_results_layout.addWidget(self.face_results_summary)
        face_results_filters = FlowLayout()
        face_results_filters.setContentsMargins(0, 0, 0, 0)
        face_results_filters.setSpacing(6)
        self.face_results_sort_combo = QComboBox()
        self.face_results_sort_combo.addItem("Score", "score")
        self.face_results_sort_combo.addItem("Newest", "newest")
        self.face_results_sort_combo.addItem("Oldest", "oldest")
        self.face_results_sort_combo.addItem("Same Folder First", "same_folder")
        self.face_results_label_filter_combo = QComboBox()
        self.face_results_label_filter_combo.addItem("All Labels", "all")
        self.face_results_label_filter_combo.addItem("Named Only", "named")
        self.face_results_label_filter_combo.addItem("Unlabeled Only", "unlabeled")
        self.face_results_quality_filter_combo = QComboBox()
        self.face_results_quality_filter_combo.addItem("All Quality", "all")
        self.face_results_quality_filter_combo.addItem("Clean", "clean")
        self.face_results_quality_filter_combo.addItem("Review", "review")
        self.face_results_quality_filter_combo.addItem("Reject", "reject")
        self.face_results_folder_filter = QLineEdit()
        self.face_results_folder_filter.setPlaceholderText("Folder contains")
        self.face_results_date_from = QLineEdit()
        self.face_results_date_from.setPlaceholderText("Date from YYYY-MM-DD")
        self.face_results_date_to = QLineEdit()
        self.face_results_date_to.setPlaceholderText("Date to YYYY-MM-DD")
        for widget in (
            self.face_results_sort_combo,
            self.face_results_label_filter_combo,
            self.face_results_quality_filter_combo,
            self.face_results_folder_filter,
            self.face_results_date_from,
            self.face_results_date_to,
        ):
            face_results_filters.addWidget(widget)
        face_results_layout.addLayout(face_results_filters)
        self.face_results_sort_combo.currentIndexChanged.connect(lambda _index: self._request_face_result_filter_refresh())
        self.face_results_label_filter_combo.currentIndexChanged.connect(lambda _index: self._request_face_result_filter_refresh())
        self.face_results_quality_filter_combo.currentIndexChanged.connect(lambda _index: self._request_face_result_filter_refresh())
        self.face_results_folder_filter.textChanged.connect(lambda _text: self._request_face_result_filter_refresh())
        self.face_results_date_from.textChanged.connect(lambda _text: self._request_face_result_filter_refresh())
        self.face_results_date_to.textChanged.connect(lambda _text: self._request_face_result_filter_refresh())
        face_results_actions = FlowLayout()
        face_results_actions.setContentsMargins(0, 0, 0, 0)
        self.face_results_name_button = QPushButton("Name Selected Faces")
        self.face_results_name_button.setProperty("kind", "primary")
        self.face_results_name_clusters_button = QPushButton("Name Selected Clusters")
        self.face_results_name_clusters_button.setProperty("kind", "secondary")
        self.face_results_queue_suggestion_button = QPushButton("Queue Suggested Identity")
        self.face_results_reject_suggestion_button = QPushButton("Reject Suggestion")
        self.face_results_keep_unknown_button = QPushButton("Keep Cluster Unlabeled")
        self.face_results_jump_button = QPushButton("Show Photo")
        self.face_results_show_group_photos_button = QPushButton("Show Group Photos")
        self.face_results_edit_button = QPushButton("Open Inspector")
        self.face_results_split_button = QPushButton("Create New Cluster From Selected Faces")
        self.face_results_merge_button = QPushButton("Merge Selected Clusters")
        self.face_results_recluster_button = QPushButton("Recluster Selected Cluster")
        self.face_results_compare_again_button = QPushButton("Run Current Backend Again")
        self.face_results_review_pending_button = QPushButton("Review Pending Labels")
        self.face_results_export_button = QPushButton("Export Cluster Results")
        for button in (
            self.face_results_name_button,
            self.face_results_name_clusters_button,
            self.face_results_queue_suggestion_button,
            self.face_results_reject_suggestion_button,
            self.face_results_keep_unknown_button,
            self.face_results_jump_button,
            self.face_results_show_group_photos_button,
            self.face_results_edit_button,
            self.face_results_split_button,
            self.face_results_merge_button,
            self.face_results_recluster_button,
            self.face_results_compare_again_button,
            self.face_results_review_pending_button,
            self.face_results_export_button,
        ):
            button.setEnabled(False)
        self.face_results_name_button.clicked.connect(self._name_face_results_from_prompt)
        self.face_results_name_clusters_button.clicked.connect(self._name_selected_face_result_groups_immediately)
        self.face_results_queue_suggestion_button.clicked.connect(self._queue_selected_face_result_group_suggestion)
        self.face_results_reject_suggestion_button.clicked.connect(self._reject_selected_face_result_group_suggestion)
        self.face_results_keep_unknown_button.clicked.connect(self._mark_selected_face_result_group_unknown)
        self.face_results_jump_button.clicked.connect(self._jump_to_selected_face_result)
        self.face_results_show_group_photos_button.clicked.connect(self._show_current_face_result_group_photos)
        self.face_results_edit_button.clicked.connect(self._open_selected_face_result_in_inspector)
        self.face_results_split_button.clicked.connect(self._split_selected_face_result_group)
        self.face_results_merge_button.clicked.connect(self._merge_selected_face_result_groups)
        self.face_results_recluster_button.clicked.connect(self._recluster_selected_face_result_group)
        self.face_results_compare_again_button.clicked.connect(self._rerun_face_cluster_compare)
        self.face_results_review_pending_button.clicked.connect(self._open_face_result_pending_review)
        self.face_results_export_button.clicked.connect(self._export_face_cluster_results)
        self._action_buttons.extend(
            [
                self.face_results_name_button,
                self.face_results_name_clusters_button,
                self.face_results_queue_suggestion_button,
                self.face_results_reject_suggestion_button,
                self.face_results_keep_unknown_button,
                self.face_results_jump_button,
                self.face_results_show_group_photos_button,
                self.face_results_edit_button,
                self.face_results_split_button,
                self.face_results_merge_button,
                self.face_results_recluster_button,
                self.face_results_compare_again_button,
                self.face_results_review_pending_button,
                self.face_results_export_button,
            ]
        )
        self._mode_required_buttons.append(self.face_results_name_button)
        self._mode_required_buttons.append(self.face_results_name_clusters_button)
        face_results_actions.addWidget(self.face_results_name_button)
        face_results_actions.addWidget(self.face_results_name_clusters_button)
        face_results_actions.addWidget(self.face_results_queue_suggestion_button)
        face_results_actions.addWidget(self.face_results_reject_suggestion_button)
        face_results_actions.addWidget(self.face_results_keep_unknown_button)
        face_results_actions.addWidget(self.face_results_jump_button)
        face_results_actions.addWidget(self.face_results_show_group_photos_button)
        face_results_actions.addWidget(self.face_results_edit_button)
        face_results_actions.addWidget(self.face_results_split_button)
        face_results_actions.addWidget(self.face_results_merge_button)
        face_results_actions.addWidget(self.face_results_recluster_button)
        face_results_actions.addWidget(self.face_results_compare_again_button)
        face_results_actions.addWidget(self.face_results_review_pending_button)
        face_results_actions.addWidget(self.face_results_export_button)
        face_results_layout.addLayout(face_results_actions)
        self.face_result_view_tabs = QTabBar(self.face_results_panel)
        self.face_result_view_tabs.setObjectName("faceResultViewTabs")
        self.face_result_view_tabs.setDocumentMode(True)
        self.face_result_view_tabs.setDrawBase(False)
        self.face_result_view_tabs.addTab("Named Photos")
        self.face_result_view_tabs.addTab("Grouped Photos")
        self.face_result_view_tabs.setTabToolTip(
            0,
            "Browse saved faces grouped by person name. This view updates whenever faces are named anywhere in ClusterLens.",
        )
        self.face_result_view_tabs.setTabToolTip(
            1,
            "Browse the normal face-clustering groups. Select multiple groups, then use Name Selected Clusters to name every face in them.",
        )
        face_results_layout.addWidget(self.face_result_view_tabs)
        face_results_body = QWidget(self.face_results_panel)
        face_results_body_layout = QHBoxLayout(face_results_body)
        face_results_body_layout.setContentsMargins(0, 0, 0, 0)
        face_results_body_layout.setSpacing(8)
        self.face_results_groups_model = ListEntryModel(self)
        self.face_results_groups_list = QListView()
        self.face_results_groups_list.setMaximumWidth(260)
        self.face_results_groups_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.face_results_groups_list.setUniformItemSizes(True)
        self.face_results_groups_list.setModel(self.face_results_groups_model)
        self.face_results_groups_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.face_results_groups_list.customContextMenuRequested.connect(self._show_face_result_group_context_menu)
        face_results_group_selection_model = self.face_results_groups_list.selectionModel()
        if face_results_group_selection_model is not None:
            face_results_group_selection_model.selectionChanged.connect(lambda *_args: self._on_face_result_group_selection_changed("raw"))
        raw_groups_column = QWidget(self.face_results_panel)
        raw_groups_layout = QVBoxLayout(raw_groups_column)
        raw_groups_layout.setContentsMargins(0, 0, 0, 0)
        raw_groups_layout.setSpacing(6)
        self.face_results_groups_label = QLabel("Grouped Photos")
        raw_groups_layout.addWidget(self.face_results_groups_label)
        raw_groups_layout.addWidget(self.face_results_groups_list, stretch=1)
        self.face_results_merged_groups_model = ListEntryModel(self)
        self.face_results_merged_groups_list = QListView()
        self.face_results_merged_groups_list.setMaximumWidth(260)
        self.face_results_merged_groups_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.face_results_merged_groups_list.setUniformItemSizes(True)
        self.face_results_merged_groups_list.setModel(self.face_results_merged_groups_model)
        merged_group_selection_model = self.face_results_merged_groups_list.selectionModel()
        if merged_group_selection_model is not None:
            merged_group_selection_model.selectionChanged.connect(lambda *_args: self._on_face_result_group_selection_changed("merged_name"))
        merged_groups_column = QWidget(self.face_results_panel)
        merged_groups_layout = QVBoxLayout(merged_groups_column)
        merged_groups_layout.setContentsMargins(0, 0, 0, 0)
        merged_groups_layout.setSpacing(6)
        self.face_results_merged_groups_label = QLabel("Named Photos")
        merged_groups_layout.addWidget(self.face_results_merged_groups_label)
        merged_groups_layout.addWidget(self.face_results_merged_groups_list, stretch=1)
        self.face_result_group_stack = QStackedWidget(face_results_body)
        self.face_result_group_stack.setMaximumWidth(260)
        self.face_result_group_stack.addWidget(raw_groups_column)
        self.face_result_group_stack.addWidget(merged_groups_column)
        face_results_body_layout.addWidget(self.face_result_group_stack)
        self.face_results_exif_groups_model = ListEntryModel(self)
        self.face_results_model = FaceTileListModel(
            self._image_for_face_tile,
            self._face_tile_cache_key_for_item,
            QSize(88, 88),
            self,
        )
        self.face_results_list = QListView()
        self.face_results_list.setViewMode(QListView.ViewMode.IconMode)
        self.face_results_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.face_results_list.setMovement(QListView.Movement.Static)
        self.face_results_list.setLayoutMode(QListView.LayoutMode.Batched)
        self.face_results_list.setBatchSize(48)
        self.face_results_list.setUniformItemSizes(True)
        self.face_results_list.setWrapping(True)
        self.face_results_list.setWordWrap(True)
        self.face_results_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.face_results_list.setIconSize(QSize(88, 88))
        self.face_results_list.setGridSize(QSize(118, 138))
        self.face_results_list.setSpacing(10)
        self.face_results_list.setModel(self.face_results_model)
        self.face_results_list.setItemDelegate(FaceTileItemDelegate(self.face_results_list))
        self.face_results_list.setToolTip(
            "Select one or more face tiles. Shift-click selects a range, Ctrl-click changes individual selection, "
            "and right-click offers actions for the selected faces."
        )
        self.face_results_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.face_results_list.doubleClicked.connect(lambda _index: self._open_selected_face_result_in_inspector())
        self.face_results_list.verticalScrollBar().valueChanged.connect(lambda _value: self._schedule_face_result_tile_loads())
        self.face_results_list.customContextMenuRequested.connect(self._show_face_results_context_menu)
        self._install_face_result_group_hover_preview(self.face_results_groups_list, "raw")
        self._install_face_result_group_hover_preview(self.face_results_merged_groups_list, "merged_name")
        face_results_selection_model = self.face_results_list.selectionModel()
        if face_results_selection_model is not None:
            face_results_selection_model.selectionChanged.connect(lambda *_args: self._on_face_result_selection_changed())
        face_results_body_layout.addWidget(self.face_results_list, stretch=1)
        face_results_layout.addWidget(face_results_body, stretch=1)
        self.face_result_view_tabs.currentChanged.connect(self._on_face_result_view_tab_changed)
        self.face_results_cluster_summary = self._helper_label("Select a face cluster group to inspect members and quality.", tooltip=FACE_HELP["cluster_faces"])
        self.face_results_cluster_explanation = self._helper_label("", tooltip=FACE_HELP["cluster_faces"])
        self.face_results_cluster_suggestion = self._helper_label("", tooltip=FACE_HELP["save_named_examples"])
        face_results_layout.addWidget(self.face_results_cluster_summary)
        face_results_layout.addWidget(self.face_results_cluster_explanation)
        face_results_layout.addWidget(self.face_results_cluster_suggestion)
        self.face_results_membership_table = QTableWidget(0, 6, self.face_results_panel)
        self.face_results_membership_table.setHorizontalHeaderLabels(
            ["Backend", "Cluster", "Size", "Rank", "Outlier", "Quality"]
        )
        self.face_results_membership_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.face_results_membership_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.face_results_membership_table.setMaximumHeight(180)
        face_results_layout.addWidget(self.face_results_membership_table)
        self.results_list = QListWidget()
        self.status_label = QLabel("Search index idle.")
        if self._tab_enabled("image search"):
            self._build_image_tab()
        if self._tab_enabled("text search"):
            self._build_text_tab()
        if self._tab_enabled("duplicate search"):
            self._build_duplicate_tab()
        if self._tab_enabled("all faces"):
            self._build_face_album_tab()
        if self._tab_enabled("faces in folder", "face library", "folder review"):
            self._build_face_library_tab()
        if self._tab_enabled("face search"):
            self._build_face_tab()
        if self._tab_enabled("identities"):
            self._build_face_identities_tab()
        self.task_navigation = QTabBar(self.sidebar_content_panel)
        self.task_navigation.setObjectName("facesTaskNavigation")
        self.task_navigation.setAccessibleName("Faces task selector")
        self.task_navigation.setToolTip("Choose what you want to do with faces.")
        self.task_navigation.setDocumentMode(True)
        self.task_navigation.setDrawBase(False)
        self.task_navigation.setExpanding(True)
        self.task_navigation.setUsesScrollButtons(True)
        self.task_navigation.setElideMode(Qt.TextElideMode.ElideRight)
        self.task_navigation.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.task_navigation.setMinimumWidth(0)
        task_labels = {
            "all faces": ("All Faces", "Browse faces already saved in the global face library."),
            "folder review": ("Detect", "Detect or review faces for the selected folder."),
            "face library": ("Detect", "Detect or review faces for the selected folder."),
            "face search": ("Find", "Find a person from a photo or a saved name."),
        }
        self._task_navigation_tab_indexes: list[int] = []
        for tab_index in range(self.tabs.count()):
            internal_label = str(self.tabs.tabText(tab_index) or "")
            task = task_labels.get(internal_label.strip().lower())
            if task is None:
                # Identity management remains available to retained workflows and
                # automation, but it is intentionally not a primary Faces task.
                continue
            display_label, tooltip = task
            navigation_index = self.task_navigation.addTab(display_label)
            self.task_navigation.setTabToolTip(navigation_index, tooltip)
            self._task_navigation_tab_indexes.append(tab_index)
        self.task_navigation.setMinimumHeight(36)
        self.task_navigation.setMaximumHeight(40)
        self.task_navigation.currentChanged.connect(self._select_task_from_navigation)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.tabs.tabBar().hide()
        initial_task_index = next(
            (
                navigation_index
                for navigation_index, tab_index in enumerate(self._task_navigation_tab_indexes)
                if tab_index == self.tabs.currentIndex()
            ),
            0,
        )
        self.task_navigation.setCurrentIndex(initial_task_index)
        self.sidebar_content_layout.addWidget(self.task_navigation)
        self.sidebar_content_layout.addWidget(self.tabs)
        self.sidebar_content_layout.addStretch(1)
        self.status_label.setWordWrap(True)
        self.task_panel_toggle = QToolButton(self.gallery_panel)
        self.task_panel_toggle.setText("Hide task panel")
        self.task_panel_toggle.setCheckable(True)
        self.task_panel_toggle.setChecked(True)
        self.task_panel_toggle.setAccessibleName("Show or hide the Faces task panel")
        self.task_panel_toggle.setToolTip("Collapse the task panel to give results more room.")
        self.task_panel_toggle.toggled.connect(self._set_task_panel_visible)
        self.gallery_layout.addWidget(self.task_panel_toggle, alignment=Qt.AlignmentFlag.AlignLeft)
        self.gallery_layout.addWidget(self.status_label)
        self.results_toolbar = self._build_results_toolbar()
        self.gallery_layout.addWidget(self.results_toolbar)
        self.face_results_context_bar = QWidget(self.gallery_panel)
        context_layout = QHBoxLayout(self.face_results_context_bar)
        context_layout.setContentsMargins(0, 0, 0, 0)
        context_layout.setSpacing(8)
        self.face_results_context_label = QLabel("Choose a folder to begin.")
        self.face_results_context_label.setWordWrap(True)
        self.face_results_context_label.setProperty("role", "section")
        self.face_photo_filter = QComboBox(self.face_results_context_bar)
        self.face_photo_filter.setAccessibleName("Filter folder photos")
        self.face_photo_filter.setToolTip("Choose which photos from the current folder appear in Photos.")
        self.face_photo_filter.addItem("With Faces", "with_faces")
        self.face_photo_filter.addItem("Needs Review", "needs_review")
        self.face_photo_filter.addItem("No Faces", "no_faces")
        self.face_photo_filter.addItem("Not Scanned", "not_scanned")
        self.face_photo_filter.addItem("All", "all")
        self.face_photo_filter.currentIndexChanged.connect(lambda _index: self._on_face_photo_filter_changed())
        self.face_photos_back_button = QPushButton("Back to Folder Photos", self.face_results_context_bar)
        self.face_photos_back_button.setToolTip("Restore the folder photo filter and position used before opening these results.")
        self.face_photos_back_button.clicked.connect(self._restore_folder_photos)
        self.face_photos_back_button.hide()
        context_layout.addWidget(self.face_results_context_label, stretch=1)
        context_layout.addWidget(self.face_photo_filter)
        context_layout.addWidget(self.face_photos_back_button)
        self.gallery_layout.addWidget(self.face_results_context_bar)
        self._set_results_kind("none")
        self._setup_results_gallery()
        self.face_review_results_tabs.addTab(self.results_gallery, "Photos")
        self.face_review_results_tabs.addTab(self.face_detected_faces_panel, "Faces")
        self.face_review_results_tabs.addTab(self.face_results_panel, "Face Groups")
        self.face_review_results_tabs.setTabToolTip(0, "Photo results and folder review")
        self.face_review_results_tabs.setTabToolTip(1, "Every visible detected face in the current folder review")
        self.face_review_results_tabs.setTabToolTip(2, "Browse normal grouped photos or named photos, then select faces or clusters to work with them.")
        self.face_review_results_tabs.currentChanged.connect(self._on_face_review_results_tab_changed)
        results_tab_bar = self.face_review_results_tabs.tabBar()
        results_tab_bar.setExpanding(False)
        results_tab_bar.setUsesScrollButtons(True)
        results_tab_bar.setElideMode(Qt.TextElideMode.ElideRight)
        results_tab_bar.show()
        self.gallery_layout.addWidget(self.face_review_results_tabs, stretch=1)
        # Keep the text list for debugging, but default to the gallery UX.
        self.results_list.hide()
        self.gallery_layout.addWidget(self.results_list)
        self.workspace_splitter.addWidget(self.sidebar_panel)
        self.workspace_splitter.addWidget(self.gallery_panel)
        self.workspace_splitter.setStretchFactor(0, 1)
        self.workspace_splitter.setStretchFactor(1, 5)
        self.workspace_splitter.setSizes(list(self._face_splitter_default_sizes))
        self.workspace_splitter.splitterMoved.connect(self._on_face_workspace_splitter_moved)
        self.main_layout.addWidget(self.workspace_splitter, stretch=1)
        self._start_face_tile_loader_threads()
        self.set_external_results_mode(self._external_results)
        self.set_ui_mode(self._face_ui_mode)
        self._refresh_face_grid_layouts(force=True)

    def _set_task_panel_visible(self, visible: bool) -> None:
        self.sidebar_panel.setVisible(bool(visible))
        self.task_panel_toggle.setText("Hide task panel" if visible else "Show task panel")
        if visible:
            sizes = list(getattr(self.workspace_splitter, "requested_sizes", []) or self._face_splitter_default_sizes)
            self.workspace_splitter.setSizes([max(280, int(sizes[0])), max(1, int(sizes[1]))])

    @staticmethod
    def _normalize_supported_face_modes(modes: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
        raw_modes = modes if modes is not None else ("human", "dog", "cat")
        normalized: list[str] = []
        for mode in raw_modes:
            mode_id = normalize_face_mode(str(mode or "human"))
            if mode_id not in normalized:
                normalized.append(mode_id)
        return tuple(normalized or ["human"])

    def _active_search_service(self, scope_combo) -> SimilaritySearchService:
        selected = "global"
        try:
            if scope_combo is not None:
                selected = str(scope_combo.currentText()).lower()
        except Exception:
            selected = "global"
        return self.search_service_session if "session" in selected else self.search_service_global

    def _tab_enabled(self, *labels: str) -> bool:
        if not self._enabled_tabs:
            return True
        return any(str(label or "").strip().lower() in self._enabled_tabs for label in labels)

    def _current_faces_tab_label(self) -> str:
        try:
            return str(self.tabs.tabText(self.tabs.currentIndex()) or "").strip().lower()
        except Exception:
            return ""

    def _is_all_faces_tab_label(self, label: str) -> bool:
        return str(label or "").strip().lower() == self._all_faces_tab_label.lower()

    def _is_face_folder_tab_label(self, label: str) -> bool:
        normalized = str(label or "").strip().lower()
        return normalized in {"face library", "faces in folder", "folder review"}

    def is_face_folder_tab_active(self) -> bool:
        return self._is_face_folder_tab_label(self._current_faces_tab_label())

    def is_all_faces_tab_active(self) -> bool:
        return self._is_all_faces_tab_label(self._current_faces_tab_label())

    def _current_face_scope_key(self) -> str:
        scope = getattr(self, "face_db_scope", None)
        selected = "global"
        try:
            if scope is not None:
                selected = str(scope.currentText()).lower()
        except Exception:
            selected = "global"
        return "session" if "session" in selected else "global"

    def _face_service_for_scope(self, scope_key: str) -> FaceIndexService:
        selected = "session" if "session" in str(scope_key or "").strip().lower() else "global"
        mode = self.current_face_mode()
        applied_prefs = self._applied_face_pipeline_prefs(mode)
        applied_detector_id = str(applied_prefs.get("detector_id") or default_face_detector_id(self.face_model_root, mode))
        applied_embedder_id = str(applied_prefs.get("embedder_id") or default_face_embedder_id(self.face_model_root, mode))
        if callable(getattr(self, "face_service_provider", None)):
            try:
                service = self.face_service_provider(
                    "session" if "session" in selected else "global",
                    mode,
                    applied_detector_id,
                    applied_embedder_id,
                )
                self._apply_face_pipeline_to_service(service)
                return service
            except Exception:
                LOGGER.exception("FacePane service provider failed mode=%s scope=%s", mode, selected)
                if mode != "human":
                    mapping = self.face_services_session_by_mode if "session" in selected else self.face_services_global_by_mode
                    service = mapping.get(mode)
                    if service is None:
                        suffix = "session" if "session" in selected else "global"
                        service = FaceIndexService(
                            mode=mode,
                            model_root=self.face_model_root,
                            detector_id=applied_detector_id,
                            embedder_id=applied_embedder_id,
                            db_path=self.settings.cache_dir / f"face_search_{mode}_{suffix}.db",
                        )
                        mapping[mode] = service
                    self._apply_face_pipeline_to_service(service)
                    return service
        if mode == "human":
            service = self.face_service_session if "session" in selected else self.face_service_global
            self._apply_face_pipeline_to_service(service)
            return service
        mapping = self.face_services_session_by_mode if "session" in selected else self.face_services_global_by_mode
        service = mapping.get(mode)
        if service is not None:
            self._apply_face_pipeline_to_service(service)
            return service
        suffix = "session" if "session" in selected else "global"
        service = FaceIndexService(
            mode=mode,
            model_root=self.face_model_root,
            detector_id=applied_detector_id,
            embedder_id=applied_embedder_id,
            db_path=self.settings.cache_dir / f"face_search_{mode}_{suffix}.db",
        )
        mapping[mode] = service
        self._apply_face_pipeline_to_service(service)
        return service

    def _active_face_service(self) -> FaceIndexService:
        if self._should_use_resolved_face_review_source():
            source = self._resolve_face_review_source()
            if source is not None:
                return source.service
        return self._face_service_for_scope(self._current_face_scope_key())

    def _global_face_service(self) -> FaceIndexService:
        return self._face_service_for_scope("global")

    def _should_use_resolved_face_review_source(self) -> bool:
        if self.current_face_mode() != "human":
            return False
        if self.is_face_folder_tab_active():
            return True
        if self.is_all_faces_tab_active():
            return False
        if self._results_kind == "faces_review":
            return True
        return str(self._face_result_source_kind or "").strip().lower() == "faces_review"

    def _invalidate_face_review_source(self) -> None:
        self._face_review_source = None
        self._face_review_source_key = None

    def _face_review_source_cache_key_for_folder(self, folder: str | None = None) -> tuple[str, ...] | None:
        folder_value = str(folder if folder is not None else self._effective_face_folder()).strip()
        mode = self.current_face_mode()
        if not folder_value or mode != "human":
            return None
        scope_key = self._current_face_scope_key()
        active_service = self._face_service_for_scope(scope_key)
        return (
            folder_value,
            scope_key,
            mode,
            str(getattr(active_service, "db_path", "") or ""),
            str(getattr(active_service, "detector_id", "") or ""),
            str(getattr(active_service, "embedder_id", "") or ""),
            str(self.face_model_root or "").strip(),
        )

    @staticmethod
    def _face_review_runtime_root_for_db_path(db_path: Path) -> Path:
        path = Path(str(db_path))
        if str(path.parent.name).strip().lower() == "cache":
            return path.parent.parent
        return path.parent

    @staticmethod
    def _face_review_folder_sql_args(folder: str) -> list[str]:
        _clause, args = folder_scope_sql("image_path", folder)
        return args

    def _face_review_db_metrics(self, db_path: Path, folder: str) -> dict[str, int]:
        path = Path(str(db_path or ""))
        if not str(path):
            return {
                "indexed_image_count": 0,
                "face_image_count": 0,
                "face_count": 0,
                "db_mtime_ns": 0,
            }
        try:
            db_mtime_ns = int(path.stat().st_mtime_ns)
        except Exception:
            db_mtime_ns = 0
        if not path.exists() or not path.is_file():
            return {
                "indexed_image_count": 0,
                "face_image_count": 0,
                "face_count": 0,
                "db_mtime_ns": db_mtime_ns,
            }
        args = self._face_review_folder_sql_args(folder)
        where_clause, args = folder_scope_sql("image_path", folder)
        try:
            with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
                indexed_image_count = int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM face_scan_images WHERE {where_clause}",
                        args,
                    ).fetchone()[0]
                    or 0
                )
                face_image_count = int(
                    connection.execute(
                        f"SELECT COUNT(DISTINCT image_path) FROM face_index WHERE {where_clause}",
                        args,
                    ).fetchone()[0]
                    or 0
                )
                face_count = int(
                    connection.execute(
                        f"SELECT COUNT(*) FROM face_index WHERE {where_clause}",
                        args,
                    ).fetchone()[0]
                    or 0
                )
        except sqlite3.OperationalError as exc:
            if "unable to open database file" not in str(exc).lower():
                LOGGER.warning("FacePane review-source metrics skipped db=%s folder=%s error=%s", path, folder, exc)
            indexed_image_count = 0
            face_image_count = 0
            face_count = 0
        except Exception:
            LOGGER.exception("FacePane review-source metrics failed db=%s folder=%s", path, folder)
            indexed_image_count = 0
            face_image_count = 0
            face_count = 0
        return {
            "indexed_image_count": indexed_image_count,
            "face_image_count": face_image_count,
            "face_count": face_count,
            "db_mtime_ns": db_mtime_ns,
        }

    def _iter_face_review_db_candidates(
        self,
        *,
        scope_key: str,
        active_db_path: Path,
    ) -> list[dict[str, object]]:
        active_root = self._face_review_runtime_root_for_db_path(active_db_path)
        preferred_name = str(active_db_path.name or "")
        pattern = "face_search_human_*_session.db" if scope_key == "session" else "face_search_human_*.db"
        seen: set[str] = set()
        candidates: list[dict[str, object]] = []

        def _append(path_like, *, runtime_root: Path | None = None) -> None:
            path = Path(str(path_like or ""))
            key = str(path)
            if not key or key in seen:
                return
            seen.add(key)
            candidate_root = runtime_root or self._face_review_runtime_root_for_db_path(path)
            same_runtime = str(candidate_root) == str(active_root)
            same_pipeline = bool(preferred_name and path.name == preferred_name)
            if str(path) == str(active_db_path):
                rank = 0
            elif same_runtime:
                rank = 1
            elif same_pipeline:
                rank = 2
            else:
                rank = 3
            candidates.append(
                {
                    "db_path": path,
                    "runtime_root": candidate_root,
                    "rank": int(rank),
                    "same_runtime": bool(same_runtime),
                    "same_pipeline": bool(same_pipeline),
                }
            )

        _append(active_db_path, runtime_root=active_root)
        for root in self.settings.preferred_base_dirs():
            runtime_root = Path(str(root))
            cache_dir = runtime_root / "cache"
            preferred_path = cache_dir / preferred_name if preferred_name else None
            if preferred_path is not None and preferred_path.exists():
                _append(preferred_path, runtime_root=runtime_root)
            if not cache_dir.exists():
                continue
            for candidate in sorted(cache_dir.glob(pattern), key=lambda item: str(item).lower()):
                _append(candidate, runtime_root=runtime_root)
        return candidates

    def _face_review_service_for_db(
        self,
        db_path: Path,
        *,
        active_service: FaceIndexService,
    ) -> FaceIndexService:
        key = (
            str(db_path),
            self.current_face_mode(),
            str(self.face_model_root or "").strip(),
            str(getattr(active_service, "detector_id", "") or ""),
            str(getattr(active_service, "embedder_id", "") or ""),
        )
        service = self._face_review_service_cache.get(key)
        if service is None:
            service = FaceIndexService(
                mode=self.current_face_mode(),
                execution_policy=getattr(active_service, "execution_policy", None),
                runtime_service=getattr(active_service, "runtime_service", None),
                model_root=self.face_model_root,
                detector_id=str(getattr(active_service, "detector_id", "") or ""),
                embedder_id=str(getattr(active_service, "embedder_id", "") or ""),
                db_path=str(db_path),
            )
            self._face_review_service_cache[key] = service
        self._apply_face_pipeline_to_service(service)
        return service

    def _resolve_face_review_source_background(
        self,
        *,
        folder: str,
        scope_key: str,
        mode: str,
        active_service: FaceIndexService,
        pipeline_prefs: dict[str, object],
    ) -> FaceReviewSource:
        folder_value = str(folder or "").strip()
        active_db_path = Path(str(getattr(active_service, "db_path", "") or ""))
        default_candidate_resolver = getattr(self._iter_face_review_db_candidates, "__func__", None) is SearchPane._iter_face_review_db_candidates
        if not isinstance(active_service, FaceIndexService) and default_candidate_resolver:
            return FaceReviewSource(
                service=active_service,
                folder=folder_value,
                scope_key=scope_key,
                mode=mode,
                db_path=str(getattr(active_service, "db_path", "") or ""),
                runtime_root="",
                indexed_image_count=0,
                face_image_count=0,
                face_count=0,
                selection_reason="Using the injected review service.",
            )

        selected_entry: dict[str, object] | None = None
        best_score: tuple[int, int, int, int, int, int] | None = None
        for candidate in self._iter_face_review_db_candidates(scope_key=scope_key, active_db_path=active_db_path):
            metrics = self._face_review_db_metrics(Path(candidate["db_path"]), folder_value)
            entry = dict(candidate)
            entry.update(metrics)
            has_data = int(
                max(
                    int(entry.get("indexed_image_count", 0) or 0),
                    int(entry.get("face_image_count", 0) or 0),
                    int(entry.get("face_count", 0) or 0),
                )
                > 0
            )
            rank = int(entry.get("rank", 99) if entry.get("rank", 99) is not None else 99)
            score = (
                has_data,
                -rank,
                int(entry.get("indexed_image_count", 0) or 0),
                int(entry.get("face_image_count", 0) or 0),
                int(entry.get("face_count", 0) or 0),
                int(entry.get("db_mtime_ns", 0) or 0),
            )
            if best_score is None or score > best_score:
                best_score = score
                selected_entry = entry

        if selected_entry is None:
            selected_entry = {
                "db_path": active_db_path,
                "runtime_root": self._face_review_runtime_root_for_db_path(active_db_path),
                "rank": 0,
                "indexed_image_count": 0,
                "face_image_count": 0,
                "face_count": 0,
            }

        selected_db_path = Path(str(selected_entry.get("db_path", "") or ""))
        if str(selected_db_path) == str(active_db_path):
            service = active_service
            self._configure_face_service_from_prefs(service, pipeline_prefs)
        else:
            default_service_resolver = (
                getattr(self._face_review_service_for_db, "__func__", None)
                is SearchPane._face_review_service_for_db
            )
            if not default_service_resolver:
                # Preserve injected service adapters used by integrations and tests.
                service = self._face_review_service_for_db(selected_db_path, active_service=active_service)
            else:
                key = (
                    str(selected_db_path),
                    str(mode),
                    str(self.face_model_root or "").strip(),
                    str(getattr(active_service, "detector_id", "") or ""),
                    str(getattr(active_service, "embedder_id", "") or ""),
                )
                with self._face_review_service_cache_lock:
                    service = self._face_review_service_cache.get(key)
                    if service is None:
                        service = FaceIndexService(
                            mode=mode,
                            execution_policy=getattr(active_service, "execution_policy", None),
                            runtime_service=getattr(active_service, "runtime_service", None),
                            model_root=self.face_model_root,
                            detector_id=str(getattr(active_service, "detector_id", "") or ""),
                            embedder_id=str(getattr(active_service, "embedder_id", "") or ""),
                            db_path=str(selected_db_path),
                        )
                        self._face_review_service_cache[key] = service
                self._configure_face_service_from_prefs(service, pipeline_prefs)

        rank = int(selected_entry.get("rank", 99) if selected_entry.get("rank", 99) is not None else 99)
        indexed_image_count = int(selected_entry.get("indexed_image_count", 0) or 0)
        face_image_count = int(selected_entry.get("face_image_count", 0) or 0)
        face_count = int(selected_entry.get("face_count", 0) or 0)
        if max(indexed_image_count, face_image_count, face_count) <= 0:
            selection_reason = "No indexed faces found yet; using the active review DB."
        elif rank == 0:
            selection_reason = "Using the active review DB."
        elif rank == 1:
            selection_reason = "Reusing a populated review DB from the current runtime."
        elif rank == 2:
            selection_reason = "Reusing the current pipeline DB from another runtime root."
        else:
            selection_reason = "Reusing a populated review DB from another runtime root."
        return FaceReviewSource(
            service=service,
            folder=folder_value,
            scope_key=scope_key,
            mode=mode,
            db_path=str(selected_db_path),
            runtime_root=str(selected_entry.get("runtime_root", "") or ""),
            indexed_image_count=indexed_image_count,
            face_image_count=face_image_count,
            face_count=face_count,
            selection_reason=selection_reason,
        )

    def _adopt_face_review_source(self, source: FaceReviewSource, cache_key: tuple[str, ...] | None) -> None:
        previous_db_path = str(self._face_review_source.db_path) if self._face_review_source is not None else ""
        self._face_review_source = source
        self._face_review_source_key = cache_key
        if previous_db_path != str(source.db_path):
            self._log_face_event(
                "review_source_resolved",
                db_path=source.db_path,
                runtime_root=source.runtime_root,
                indexed_images=source.indexed_image_count,
                face_images=source.face_image_count,
                faces=source.face_count,
                reason=source.selection_reason,
            )

    def _resolve_face_review_source(self, *, folder: str | None = None) -> FaceReviewSource | None:
        cache_key = self._face_review_source_cache_key_for_folder(folder)
        if cache_key is None:
            if self._face_review_source is not None or self._face_review_source_key is not None:
                self._invalidate_face_review_source()
            return None
        if self._face_review_source is not None and self._face_review_source_key == cache_key:
            return self._face_review_source

        folder_value = str(folder if folder is not None else self._effective_face_folder()).strip()
        scope_key = str(cache_key[1])
        active_service = self._face_service_for_scope(scope_key)
        default_candidate_resolver = getattr(self._iter_face_review_db_candidates, "__func__", None) is SearchPane._iter_face_review_db_candidates
        if not isinstance(active_service, FaceIndexService) and default_candidate_resolver:
            source = FaceReviewSource(
                service=active_service,
                folder=folder_value,
                scope_key=scope_key,
                mode=self.current_face_mode(),
                db_path=str(getattr(active_service, "db_path", "") or ""),
                runtime_root="",
                indexed_image_count=0,
                face_image_count=0,
                face_count=0,
                selection_reason="Using the injected review service.",
            )
            previous_db_path = str(self._face_review_source.db_path) if self._face_review_source is not None else ""
            self._face_review_source = source
            self._face_review_source_key = cache_key
            if previous_db_path != source.db_path:
                self._log_face_event(
                    "review_source_resolved",
                    db_path=source.db_path,
                    runtime_root=source.runtime_root,
                    indexed_images=0,
                    face_images=0,
                    faces=0,
                    reason=source.selection_reason,
                )
            return source
        active_db_path = Path(str(getattr(active_service, "db_path", "") or ""))
        selected_entry: dict[str, object] | None = None
        best_score: tuple[int, int, int, int, int, int] | None = None
        for candidate in self._iter_face_review_db_candidates(scope_key=scope_key, active_db_path=active_db_path):
            metrics = self._face_review_db_metrics(Path(candidate["db_path"]), folder_value)
            entry = dict(candidate)
            entry.update(metrics)
            has_data = int(
                max(
                    int(entry.get("indexed_image_count", 0) or 0),
                    int(entry.get("face_image_count", 0) or 0),
                    int(entry.get("face_count", 0) or 0),
                )
                > 0
            )
            rank_value = entry.get("rank", 99)
            rank = 99 if rank_value is None else int(rank_value)
            score = (
                has_data,
                -rank,
                int(entry.get("indexed_image_count", 0) or 0),
                int(entry.get("face_image_count", 0) or 0),
                int(entry.get("face_count", 0) or 0),
                int(entry.get("db_mtime_ns", 0) or 0),
            )
            if best_score is None or score > best_score:
                best_score = score
                selected_entry = entry

        if selected_entry is None:
            return None

        selected_db_path = Path(str(selected_entry.get("db_path", "") or ""))
        service = (
            active_service
            if str(selected_db_path) == str(active_db_path)
            else self._face_review_service_for_db(selected_db_path, active_service=active_service)
        )
        selected_rank_value = selected_entry.get("rank", 99)
        rank = 99 if selected_rank_value is None else int(selected_rank_value)
        indexed_image_count = int(selected_entry.get("indexed_image_count", 0) or 0)
        face_image_count = int(selected_entry.get("face_image_count", 0) or 0)
        face_count = int(selected_entry.get("face_count", 0) or 0)
        if max(indexed_image_count, face_image_count, face_count) <= 0:
            selection_reason = "No indexed faces found yet; using the active review DB."
        elif rank == 0:
            selection_reason = "Using the active review DB."
        elif rank == 1:
            selection_reason = "Reusing a populated review DB from the current runtime."
        elif rank == 2:
            selection_reason = "Reusing the current pipeline DB from another runtime root."
        else:
            selection_reason = "Reusing a populated review DB from another runtime root."

        source = FaceReviewSource(
            service=service,
            folder=folder_value,
            scope_key=scope_key,
            mode=self.current_face_mode(),
            db_path=str(selected_db_path),
            runtime_root=str(selected_entry.get("runtime_root", "") or ""),
            indexed_image_count=indexed_image_count,
            face_image_count=face_image_count,
            face_count=face_count,
            selection_reason=selection_reason,
        )
        previous_db_path = str(self._face_review_source.db_path) if self._face_review_source is not None else ""
        self._face_review_source = source
        self._face_review_source_key = cache_key
        if previous_db_path != source.db_path:
            self._log_face_event(
                "review_source_resolved",
                db_path=source.db_path,
                runtime_root=source.runtime_root,
                indexed_images=source.indexed_image_count,
                face_images=source.face_image_count,
                faces=source.face_count,
                reason=source.selection_reason,
            )
        return source

    def current_face_mode(self) -> str:
        combo = getattr(self, "face_mode_combo", None)
        if combo is None:
            return "human"
        try:
            data = combo.currentData()
            if data:
                return normalize_face_mode(str(data))
            return normalize_face_mode(combo.currentText())
        except Exception:
            return "human"

    def current_ui_mode(self) -> str:
        return "advanced" if str(getattr(self, "_face_ui_mode", "basic")).strip().lower() == "advanced" else "basic"

    def set_ui_mode(self, mode: str) -> None:
        normalized = "advanced" if str(mode or "").strip().lower() == "advanced" else "basic"
        self._face_ui_mode = normalized
        self._apply_face_ui_mode()

    def reveal_face_pipeline_controls(self) -> None:
        """Open the full detector/embedder editor from any Faces view."""
        self.open_face_pipeline_dialog()

    def open_face_pipeline_dialog(self) -> None:
        """Edit all face pipeline settings in a compact modal dialog.

        The editor widgets are moved temporarily instead of duplicated, so all
        validation and pending-change handling remains identical to the former
        inline advanced panel.
        """
        existing = self._face_pipeline_dialog
        if existing is not None and existing.isVisible():
            existing.raise_()
            existing.activateWindow()
            return
        tabs = getattr(self, "face_advanced_tabs", None)
        original_panel = getattr(self, "face_advanced_panel", None)
        if tabs is None or original_panel is None:
            return
        self._reset_pending_face_settings()
        dialog = QDialog(self)
        dialog.setObjectName("facePipelineDialog")
        dialog.setWindowTitle("Advanced Face Pipeline")
        dialog.setModal(True)
        dialog.resize(700, 560)
        dialog_layout = QVBoxLayout(dialog)
        original_layout = original_panel.layout()
        if original_layout is not None:
            original_layout.removeWidget(tabs)
        dialog_layout.addWidget(tabs, stretch=1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, parent=dialog)
        apply_button = buttons.addButton("Apply", QDialogButtonBox.ButtonRole.AcceptRole)
        dialog_layout.addWidget(buttons)

        def _apply() -> None:
            if self._apply_pending_face_settings():
                dialog.accept()

        def _restore(result: int) -> None:
            dialog_layout.removeWidget(tabs)
            if original_layout is not None:
                original_layout.addWidget(tabs)
            if result != QDialog.DialogCode.Accepted.value:
                self._reset_pending_face_settings()
            self._face_pipeline_dialog = None
            dialog.deleteLater()

        apply_button.clicked.connect(_apply)
        buttons.rejected.connect(dialog.reject)
        dialog.finished.connect(_restore)
        self._face_pipeline_dialog = dialog
        dialog.open()
        self.face_detector_combo.setFocus(Qt.FocusReason.ShortcutFocusReason)

    def set_read_only_mode(self, enabled: bool) -> None:
        self._read_only_mode = bool(enabled)
        if hasattr(self, "results_gallery"):
            self.results_gallery.set_read_only_mode(self._read_only_mode)
        # Read-only mode protects source files and curated user data. Face scans
        # only build replaceable ClusterLens indexes, so they remain available.
        index_action_names = (
            "face_scan_button", "face_rescan_selected_button", "face_rescan_suspicious_button",
            "face_rescan_fallback_button", "face_index_button", "face_rebuild_db_button",
            "face_apply_settings_button",
        )
        for name in index_action_names:
            widget = getattr(self, name, None)
            if widget is not None:
                widget.setProperty("mutationAction", False)
        mutation_names = (
            "face_auto_clean_button",
            "face_hide_rejected_button", "face_restore_rejected_button",
            "face_save_name_button",
            "face_save_profile_button", "face_hide_selected_faces_button", "face_detected_name_button",
            "face_detected_remove_button", "face_search_name_selected_card_button", "face_label_button",
            "face_propagate_button", "face_merge_button", "face_clear_button", "face_pending_accept_button",
            "face_pending_accept_above_threshold_button", "face_pending_accept_cluster_button",
            "face_pending_reject_button", "face_pending_reject_cluster_button", "face_pending_undo_button",
            "face_results_name_button", "face_results_name_clusters_button",
            "face_results_queue_suggestion_button", "face_results_reject_suggestion_button",
            "face_results_keep_unknown_button", "face_results_split_button", "face_results_merge_button",
            "face_identity_remove_button", "face_identity_pin_button", "face_identity_merge_button",
            "face_identity_clear_button", "face_import_identities_button",
            "face_disable_recognition_button", "face_enable_recognition_button", "face_delete_db_button",
            "face_purge_data_button", "saved_search_save_button", "saved_search_rename_button",
            "saved_search_delete_button", "duplicate_review_tag_button", "duplicate_review_trash_button",
        )
        message = "Disabled while read-only mode is on. Change it in Settings > Safety & Recovery."
        for name in mutation_names:
            widget = getattr(self, name, None)
            if widget is None:
                continue
            widget.setProperty("mutationAction", True)
            base_tooltip = widget.property("readOnlyBaseToolTip")
            if base_tooltip is None:
                base_tooltip = widget.toolTip() or ""
                widget.setProperty("readOnlyBaseToolTip", base_tooltip)
            if self._read_only_mode:
                widget.setEnabled(False)
                widget.setToolTip(f"{base_tooltip}\n\n{message}" if base_tooltip else message)
            else:
                widget.setEnabled(True)
                widget.setToolTip(str(base_tooltip or ""))
        self._update_face_mode_status()
        self._update_face_settings_action_state()
        if hasattr(self, "face_selected_faces_context_label"):
            self._update_face_selected_context_label()
        if hasattr(self, "face_detected_remove_button"):
            self._update_detected_face_actions()
        if hasattr(self, "face_results_name_button"):
            self._update_face_result_actions()
        if hasattr(self, "face_identity_remove_button"):
            self._update_face_identity_prototype_actions()

    def _set_guarded_action_enabled(self, action, enabled: bool) -> None:
        if action is None:
            return
        mutation_blocked = self._read_only_mode and bool(action.property("mutationAction"))
        action.setEnabled(bool(enabled) and not mutation_blocked)

    def _ensure_writable_face_action(self, action: str) -> bool:
        if not self._read_only_mode:
            return True
        message = f"Read-only mode is on. Turn it off in Settings > Safety & Recovery to {action}."
        self.status_label.setText(message)
        infoBox("Read-only mode", message)
        return False

    def set_active_face_mode(self, mode: str, *, refresh: bool = True) -> None:
        combo = getattr(self, "face_mode_combo", None)
        if combo is None:
            return
        target = normalize_face_mode(mode)
        if target not in self.supported_face_modes:
            target = "human"
        for index in range(combo.count()):
            data = combo.itemData(index)
            if normalize_face_mode(str(data or combo.itemText(index))) == target:
                if combo.currentIndex() != index:
                    combo.setCurrentIndex(index)
                elif refresh:
                    self._on_face_mode_changed()
                return

    def set_face_services(
        self,
        *,
        face_services_global: dict[str, FaceIndexService] | None = None,
        face_services_session: dict[str, FaceIndexService] | None = None,
        refresh: bool = True,
    ) -> None:
        if face_services_global:
            self.face_services_global_by_mode = {
                normalize_face_mode(mode): service
                for mode, service in dict(face_services_global).items()
                if service is not None
            }
            self.face_service_global = self.face_services_global_by_mode.get("human", self.face_service_global)
        if face_services_session:
            self.face_services_session_by_mode = {
                normalize_face_mode(mode): service
                for mode, service in dict(face_services_session).items()
                if service is not None
            }
            self.face_service_session = self.face_services_session_by_mode.get("human", self.face_service_session)
        if refresh:
            self._on_face_mode_changed()

    def set_face_service_provider(self, provider) -> None:
        self.face_service_provider = provider

    def configure_face_pipeline_options(
        self,
        model_root: str,
        defaults_by_mode: dict[str, dict[str, object]] | None = None,
        *,
        refresh: bool = True,
    ) -> None:
        self.face_model_root = str(model_root or "").strip()
        for mode in self.supported_face_modes:
            prefs = self._face_pipeline_prefs_by_mode.setdefault(mode, {})
            configured = dict((defaults_by_mode or {}).get(mode, {}))
            detector_id = str(configured.get("detector_id") or prefs.get("detector_id") or default_face_detector_id(self.face_model_root, mode))
            embedder_id = str(configured.get("embedder_id") or prefs.get("embedder_id") or default_face_embedder_id(self.face_model_root, mode))
            prefs["preferred_detector_id"] = normalize_face_component_id(
                str(configured.get("preferred_detector_id") or prefs.get("preferred_detector_id") or detector_id),
                detector_id,
            )
            prefs["preferred_embedder_id"] = normalize_face_component_id(
                str(configured.get("preferred_embedder_id") or prefs.get("preferred_embedder_id") or embedder_id),
                embedder_id,
            )
            if mode == "human":
                effective_detector_id, effective_embedder_id = resolve_ready_face_pipeline_ids(
                    self.face_model_root,
                    mode,
                    detector_id,
                    embedder_id,
                )
            else:
                effective_detector_id = self._resolve_available_face_detector_id(mode, detector_id)
                effective_embedder_id = self._resolve_available_face_embedder_id(mode, embedder_id)
            prefs["detector_id"] = effective_detector_id
            prefs["embedder_id"] = effective_embedder_id
            prefs["fallback_detector_id"] = self._resolve_available_face_detector_id(
                mode,
                str(configured.get("fallback_detector_id") or prefs.get("fallback_detector_id") or prefs["detector_id"]),
            )
            prefs["detector_policy"] = str(configured.get("detector_policy", prefs.get("detector_policy", "single")) or "single").strip().lower()
            prefs["verifier_mode"] = str(configured.get("verifier_mode", prefs.get("verifier_mode", "off")) or "off").strip().lower()
            prefs["score_threshold"] = float(configured.get("score_threshold", prefs.get("score_threshold", DEFAULT_FACE_SCORE_THRESHOLD)) or 0.0)
            prefs["max_detections"] = max(
                1,
                int(configured.get("max_detections", prefs.get("max_detections", DEFAULT_FACE_MAX_DETECTIONS)) or DEFAULT_FACE_MAX_DETECTIONS),
            )
            profile_id = str(configured.get("quality_profile_id", prefs.get("quality_profile_id", "balanced")) or "balanced").strip().lower()
            prefs["quality_profile_id"] = profile_id
            thresholds = dict(face_quality_profile_config(mode, profile_id))
            thresholds.update(
                {
                    str(key): value
                    for key, value in dict(configured.get("quality_thresholds", prefs.get("quality_thresholds", {})) or {}).items()
                }
            )
            prefs["quality_thresholds"] = thresholds
            prefs["search_quality_min"] = str(configured.get("search_quality_min", prefs.get("search_quality_min", "clean")) or "clean").strip().lower()
            prefs["cluster_quality_min"] = str(configured.get("cluster_quality_min", prefs.get("cluster_quality_min", "clean")) or "clean").strip().lower()
            prefs["prototype_quality_min"] = str(configured.get("prototype_quality_min", prefs.get("prototype_quality_min", "clean")) or "clean").strip().lower()
            prefs["recognition_min_score"] = float(configured.get("recognition_min_score", prefs.get("recognition_min_score", 0.35)) or 0.35)
            prefs["auto_label_min_score"] = float(configured.get("auto_label_min_score", prefs.get("auto_label_min_score", 0.72)) or 0.72)
            prefs["rerank_policy"] = str(configured.get("rerank_policy", prefs.get("rerank_policy", "off")) or "off").strip().lower()
            prefs["rerank_top_n"] = max(1, int(configured.get("rerank_top_n", prefs.get("rerank_top_n", 25)) or 25))
            self._face_pipeline_applied_by_mode[mode] = self._clone_face_pipeline_prefs(prefs)
            self._face_pipeline_dirty_modes.discard(mode)
        self._refresh_face_pipeline_controls()
        if refresh:
            self._on_face_mode_changed()

    def _mode_face_pipeline_prefs(self, mode: str | None = None) -> dict[str, object]:
        mode_id = normalize_face_mode(mode or self.current_face_mode())
        return self._face_pipeline_prefs_by_mode.setdefault(mode_id, {})

    @staticmethod
    def _clone_face_pipeline_prefs(prefs: dict[str, object] | None) -> dict[str, object]:
        data = dict(prefs or {})
        data["quality_thresholds"] = dict(data.get("quality_thresholds", {}) or {})
        return data

    def _applied_face_pipeline_prefs(self, mode: str | None = None) -> dict[str, object]:
        mode_id = normalize_face_mode(mode or self.current_face_mode())
        prefs = self._face_pipeline_applied_by_mode.get(mode_id)
        if prefs is None:
            prefs = self._clone_face_pipeline_prefs(self._mode_face_pipeline_prefs(mode_id))
            self._face_pipeline_applied_by_mode[mode_id] = prefs
        return prefs

    def _update_face_pipeline_dirty_state(self, mode: str | None = None) -> None:
        mode_id = normalize_face_mode(mode or self.current_face_mode())
        pending = self._clone_face_pipeline_prefs(self._mode_face_pipeline_prefs(mode_id))
        applied = self._clone_face_pipeline_prefs(self._applied_face_pipeline_prefs(mode_id))
        if pending != applied:
            self._face_pipeline_dirty_modes.add(mode_id)
        else:
            self._face_pipeline_dirty_modes.discard(mode_id)
        self._update_face_settings_action_state()

    def _resolve_available_face_detector_id(self, mode: str, detector_id: str | None) -> str:
        mode_id = normalize_face_mode(mode)
        requested = normalize_face_component_id(str(detector_id or ""), default_face_detector_id(self.face_model_root, mode_id))
        try:
            bundle = resolve_face_detector_bundle(self.face_model_root, mode_id, requested)
            if bundle.available:
                return requested
        except Exception:
            pass
        preferred_non_builtin: list[str] = []
        available_ids: list[str] = []
        for candidate_id, _label in face_detector_choices(self.face_model_root, mode_id):
            try:
                bundle = resolve_face_detector_bundle(self.face_model_root, mode_id, candidate_id)
            except Exception:
                continue
            if bundle.available:
                normalized = normalize_face_component_id(candidate_id, requested)
                available_ids.append(normalized)
                if normalized != BUILTIN_HUMAN_DETECTOR_ID:
                    preferred_non_builtin.append(normalized)
        if mode_id == "human" and requested != BUILTIN_HUMAN_DETECTOR_ID and preferred_non_builtin:
            return preferred_non_builtin[0]
        if available_ids:
            return available_ids[0]
        return requested

    def _resolve_available_face_embedder_id(self, mode: str, embedder_id: str | None) -> str:
        mode_id = normalize_face_mode(mode)
        requested = normalize_face_component_id(str(embedder_id or ""), default_face_embedder_id(self.face_model_root, mode_id))
        try:
            bundle = resolve_face_embedder_bundle(self.face_model_root, mode_id, requested)
            if bundle.available:
                return requested
        except Exception:
            pass
        preferred_non_builtin: list[str] = []
        available_ids: list[str] = []
        for candidate_id, _label in face_embedder_choices(self.face_model_root, mode_id):
            try:
                bundle = resolve_face_embedder_bundle(self.face_model_root, mode_id, candidate_id)
            except Exception:
                continue
            if bundle.available:
                normalized = normalize_face_component_id(candidate_id, requested)
                available_ids.append(normalized)
                if normalized != BUILTIN_HUMAN_EMBEDDER_ID:
                    preferred_non_builtin.append(normalized)
        if mode_id == "human" and requested != BUILTIN_HUMAN_EMBEDDER_ID and preferred_non_builtin:
            return preferred_non_builtin[0]
        if available_ids:
            return available_ids[0]
        return requested

    def current_face_detector_id(self) -> str:
        combo = getattr(self, "face_detector_combo", None)
        if combo is not None:
            try:
                data = combo.currentData()
                if data:
                    return normalize_face_component_id(str(data), default_face_detector_id(self.face_model_root, self.current_face_mode()))
            except Exception:
                pass
        prefs = self._mode_face_pipeline_prefs()
        return normalize_face_component_id(
            str(prefs.get("detector_id") or default_face_detector_id(self.face_model_root, self.current_face_mode())),
            default_face_detector_id(self.face_model_root, self.current_face_mode()),
        )

    def current_face_embedder_id(self) -> str:
        combo = getattr(self, "face_embedder_combo", None)
        if combo is not None:
            try:
                data = combo.currentData()
                if data:
                    return normalize_face_component_id(str(data), default_face_embedder_id(self.face_model_root, self.current_face_mode()))
            except Exception:
                pass
        prefs = self._mode_face_pipeline_prefs()
        return normalize_face_component_id(
            str(prefs.get("embedder_id") or default_face_embedder_id(self.face_model_root, self.current_face_mode())),
            default_face_embedder_id(self.face_model_root, self.current_face_mode()),
        )

    def current_face_detector_score_threshold(self) -> float:
        spin = getattr(self, "face_detector_score_spin", None)
        if spin is not None:
            try:
                return float(spin.value())
            except Exception:
                pass
        prefs = self._mode_face_pipeline_prefs()
        return float(prefs.get("score_threshold", DEFAULT_FACE_SCORE_THRESHOLD) or 0.0)

    def current_face_max_detections(self) -> int:
        spin = getattr(self, "face_max_detections_spin", None)
        if spin is not None:
            try:
                return max(1, int(spin.value()))
            except Exception:
                pass
        prefs = self._mode_face_pipeline_prefs()
        return max(1, int(prefs.get("max_detections", DEFAULT_FACE_MAX_DETECTIONS) or DEFAULT_FACE_MAX_DETECTIONS))

    def current_face_fallback_detector_id(self) -> str:
        combo = getattr(self, "face_fallback_detector_combo", None)
        if combo is not None:
            try:
                data = combo.currentData()
                if data:
                    return normalize_face_component_id(str(data), self.current_face_detector_id())
            except Exception:
                pass
        prefs = self._mode_face_pipeline_prefs()
        return normalize_face_component_id(str(prefs.get("fallback_detector_id") or self.current_face_detector_id()), self.current_face_detector_id())

    def current_face_detector_policy(self) -> str:
        combo = getattr(self, "face_detector_policy_combo", None)
        if combo is not None:
            try:
                return str(combo.currentData() or combo.currentText() or "single").strip().lower()
            except Exception:
                pass
        return str(self._mode_face_pipeline_prefs().get("detector_policy", "single") or "single").strip().lower()

    def current_face_verifier_mode(self) -> str:
        combo = getattr(self, "face_verifier_mode_combo", None)
        if combo is not None:
            try:
                return str(combo.currentData() or combo.currentText() or "off").strip().lower()
            except Exception:
                pass
        return str(self._mode_face_pipeline_prefs().get("verifier_mode", "off") or "off").strip().lower()

    def current_face_quality_profile_id(self) -> str:
        combo = getattr(self, "face_quality_profile_combo", None)
        if combo is not None:
            try:
                data = combo.currentData()
                if data:
                    return str(data).strip().lower()
            except Exception:
                pass
        prefs = self._mode_face_pipeline_prefs()
        return str(prefs.get("quality_profile_id", "balanced") or "balanced").strip().lower()

    def current_face_quality_thresholds(self) -> dict[str, object]:
        prefs = self._mode_face_pipeline_prefs()
        thresholds = dict(prefs.get("quality_thresholds", {}) or {})
        for key, _label, _minimum, _maximum, _step in FACE_QUALITY_NUMERIC_FIELDS:
            widget = getattr(self, f"face_quality_{key}_spin", None)
            if widget is not None:
                try:
                    value = float(widget.value())
                    thresholds[key] = int(value) if key.endswith("_px") else float(value)
                except Exception:
                    pass
        for key, _label in FACE_QUALITY_POLICY_FIELDS:
            combo = getattr(self, f"face_quality_{key}_combo", None)
            if combo is not None:
                try:
                    thresholds[key] = str(combo.currentData() or combo.currentText() or "off").strip().lower()
                except Exception:
                    pass
        return thresholds

    def current_face_search_quality_min(self) -> str:
        combo = getattr(self, "face_search_quality_min_combo", None)
        if combo is not None:
            try:
                return str(combo.currentData() or combo.currentText() or "clean").strip().lower()
            except Exception:
                pass
        return str(self._mode_face_pipeline_prefs().get("search_quality_min", "clean") or "clean").strip().lower()

    def current_face_cluster_quality_min(self) -> str:
        combo = getattr(self, "face_cluster_quality_min_combo", None)
        if combo is not None:
            try:
                return str(combo.currentData() or combo.currentText() or "clean").strip().lower()
            except Exception:
                pass
        return str(self._mode_face_pipeline_prefs().get("cluster_quality_min", "clean") or "clean").strip().lower()

    def current_face_prototype_quality_min(self) -> str:
        combo = getattr(self, "face_prototype_quality_min_combo", None)
        if combo is not None:
            try:
                return str(combo.currentData() or combo.currentText() or "clean").strip().lower()
            except Exception:
                pass
        return str(self._mode_face_pipeline_prefs().get("prototype_quality_min", "clean") or "clean").strip().lower()

    def current_face_recognition_min_score(self) -> float:
        spin = getattr(self, "face_recognition_min_score_spin", None)
        if spin is not None:
            try:
                return float(spin.value())
            except Exception:
                pass
        return float(self._mode_face_pipeline_prefs().get("recognition_min_score", 0.35) or 0.35)

    def current_face_recognition_mode(self) -> str:
        combo = getattr(self, "face_recognition_mode_combo", None)
        if combo is not None:
            try:
                return str(combo.currentData() or combo.currentText() or "balanced").strip().lower()
            except Exception:
                pass
        return "balanced"

    def _face_recognition_mode_floor(self) -> float:
        mode = self.current_face_recognition_mode()
        if mode == "strict":
            return 0.6
        if mode == "loose":
            return 0.2
        return 0.35

    def _effective_face_search_min_score(self, requested: float) -> float:
        return max(float(requested), float(self._face_recognition_mode_floor()), float(self.current_face_recognition_min_score()))

    def current_face_auto_label_min_score(self) -> float:
        spin = getattr(self, "face_auto_label_min_score_spin", None)
        if spin is not None:
            try:
                return float(spin.value())
            except Exception:
                pass
        return float(self._mode_face_pipeline_prefs().get("auto_label_min_score", 0.72) or 0.72)

    def current_face_rerank_policy(self) -> str:
        combo = getattr(self, "face_rerank_policy_combo", None)
        if combo is not None:
            try:
                return str(combo.currentData() or combo.currentText() or "off").strip().lower()
            except Exception:
                pass
        return str(self._mode_face_pipeline_prefs().get("rerank_policy", "off") or "off").strip().lower()

    def current_face_rerank_top_n(self) -> int:
        spin = getattr(self, "face_rerank_top_n_spin", None)
        if spin is not None:
            try:
                return max(1, int(spin.value()))
            except Exception:
                pass
        return max(1, int(self._mode_face_pipeline_prefs().get("rerank_top_n", 25) or 25))

    def _apply_face_pipeline_to_service(self, service: FaceIndexService) -> None:
        if service is None:
            return
        prefs = self._clone_face_pipeline_prefs(self._applied_face_pipeline_prefs(self.current_face_mode()))
        service_detector_id = normalize_face_component_id(
            str(getattr(service, "detector_id", "") or prefs.get("detector_id") or ""),
            str(prefs.get("detector_id") or ""),
        )
        fallback_detector_id = str(prefs.get("fallback_detector_id") or service_detector_id)
        try:
            fallback_bundle = resolve_face_detector_bundle(
                self.face_model_root,
                self.current_face_mode(),
                fallback_detector_id,
            )
            if not fallback_bundle.available:
                fallback_detector_id = service_detector_id
        except Exception:
            fallback_detector_id = service_detector_id
        prefs["fallback_detector_id"] = fallback_detector_id
        self._configure_face_service_from_prefs(service, prefs)

    @staticmethod
    def _configure_face_service_from_prefs(service: FaceIndexService, prefs: dict[str, object]) -> None:
        if service is None:
            return
        configure_fn = getattr(service, "configure_detector_options", None)
        if callable(configure_fn):
            configure_fn(
                score_threshold=float(prefs.get("score_threshold", DEFAULT_FACE_SCORE_THRESHOLD) or 0.0),
                max_detections=max(1, int(prefs.get("max_detections", DEFAULT_FACE_MAX_DETECTIONS) or DEFAULT_FACE_MAX_DETECTIONS)),
            )
        configure_quality_fn = getattr(service, "configure_quality_options", None)
        if callable(configure_quality_fn):
            configure_quality_fn(
                profile_id=str(prefs.get("quality_profile_id", "balanced") or "balanced"),
                thresholds=dict(prefs.get("quality_thresholds", {}) or {}),
            )
        configure_cascade_fn = getattr(service, "configure_cascade_options", None)
        if callable(configure_cascade_fn):
            configure_cascade_fn(
                detector_policy=str(prefs.get("detector_policy", "single") or "single"),
                fallback_detector_id=str(prefs.get("fallback_detector_id", prefs.get("detector_id", "")) or ""),
                verifier_mode=str(prefs.get("verifier_mode", "off") or "off"),
            )
        configure_recognition_fn = getattr(service, "configure_recognition_options", None)
        if callable(configure_recognition_fn):
            configure_recognition_fn(
                search_quality_min=str(prefs.get("search_quality_min", "clean") or "clean"),
                cluster_quality_min=str(prefs.get("cluster_quality_min", "clean") or "clean"),
                prototype_quality_min=str(prefs.get("prototype_quality_min", "clean") or "clean"),
                recognition_min_score=float(prefs.get("recognition_min_score", 0.35) or 0.35),
                auto_label_min_score=float(prefs.get("auto_label_min_score", 0.72) or 0.72),
                rerank_policy=str(prefs.get("rerank_policy", "off") or "off"),
                rerank_top_n=max(1, int(prefs.get("rerank_top_n", 25) or 25)),
            )

    def _save_face_pipeline_editor_for_mode(self, mode: str | None = None) -> None:
        prefs = self._mode_face_pipeline_prefs(mode)
        prefs["detector_id"] = self.current_face_detector_id()
        prefs["embedder_id"] = self.current_face_embedder_id()
        prefs["score_threshold"] = self.current_face_detector_score_threshold()
        prefs["max_detections"] = self.current_face_max_detections()
        prefs["quality_profile_id"] = self.current_face_quality_profile_id()
        prefs["quality_thresholds"] = self.current_face_quality_thresholds()
        prefs["fallback_detector_id"] = self.current_face_fallback_detector_id()
        prefs["detector_policy"] = self.current_face_detector_policy()
        prefs["verifier_mode"] = self.current_face_verifier_mode()
        prefs["search_quality_min"] = self.current_face_search_quality_min()
        prefs["cluster_quality_min"] = self.current_face_cluster_quality_min()
        prefs["prototype_quality_min"] = self.current_face_prototype_quality_min()
        prefs["recognition_min_score"] = self.current_face_recognition_min_score()
        prefs["auto_label_min_score"] = self.current_face_auto_label_min_score()
        prefs["rerank_policy"] = self.current_face_rerank_policy()
        prefs["rerank_top_n"] = self.current_face_rerank_top_n()

    def _current_face_profile_id(self, mode: str | None = None) -> str:
        mode_id = normalize_face_mode(mode or self.current_face_mode())
        prefs = self._mode_face_pipeline_prefs(mode_id)
        return selected_face_model_profile(
            mode_id,
            str(prefs.get("detector_id") or self.current_face_detector_id()),
            str(prefs.get("embedder_id") or self.current_face_embedder_id()),
            score_threshold=float(prefs.get("score_threshold", self.current_face_detector_score_threshold()) or 0.0),
            max_detections=int(prefs.get("max_detections", self.current_face_max_detections()) or DEFAULT_FACE_MAX_DETECTIONS),
        )

    def _populate_profile_combo(self, mode: str) -> None:
        if not hasattr(self, "face_model_profile_combo"):
            return
        choices = face_model_profile_choices(mode)
        target_id = self._current_face_profile_id(mode)
        combo = self.face_model_profile_combo
        combo.blockSignals(True)
        combo.clear()
        for item_id, label in choices:
            combo.addItem(label, item_id)
        self._set_combo_to_id(combo, target_id if target_id else "custom")
        combo.blockSignals(False)

    def _populate_quality_profile_combo(self, mode: str) -> None:
        if not hasattr(self, "face_quality_profile_combo"):
            return
        combo = self.face_quality_profile_combo
        combo.blockSignals(True)
        combo.clear()
        for item_id, label in face_quality_profile_choices(mode):
            combo.addItem(label, item_id)
        self._set_combo_to_id(combo, self._mode_face_pipeline_prefs(mode).get("quality_profile_id", "balanced"))
        combo.blockSignals(False)

    def _update_face_model_details(self) -> None:
        if not hasattr(self, "face_model_summary_label"):
            return
        if hasattr(self, "face_quality_summary_label"):
            thresholds = self.current_face_quality_thresholds()
            self.face_quality_summary_label.setText(
                "Profile summary: "
                f"reject<{float(thresholds.get('reject_confidence', 0.0) or 0.0):.2f}, "
                f"min_side={int(float(thresholds.get('min_face_side_px', 0.0) or 0.0))} px, "
                f"landmarks={str(thresholds.get('landmarks_policy', 'off') or 'off')}, "
                f"alignment={str(thresholds.get('alignment_policy', 'off') or 'off')}"
            )
        self._update_face_status_strip()

    def _update_face_status_strip(self) -> None:
        label = getattr(self, "face_model_summary_label", None)
        if label is None:
            return
        try:
            tab_label = str(self.tabs.tabText(self.tabs.currentIndex()) or "").strip().lower()
        except Exception:
            tab_label = ""
        folder = ""
        try:
            folder = str(self._effective_face_folder() or "").strip()
        except Exception:
            folder = str(self._current_directory() or "").strip()
        if self._is_all_faces_tab_label(tab_label):
            scope_text = "Global library"
        elif self._is_face_folder_tab_label(tab_label):
            scope_text = f"Folder: {Path(folder).name or folder}" if folder else "No folder selected"
        elif tab_label == "identities":
            scope_text = "Global people library"
        elif bool(getattr(self, "search_only_current_folder", None) and self.search_only_current_folder.isChecked()) and folder:
            scope_text = f"Folder: {Path(folder).name or folder}"
        else:
            scope_text = "Global library"

        service = None
        try:
            service = self._active_face_service()
        except Exception:
            service = None
        policy = getattr(service, "execution_policy", None)
        effective_mode = str(getattr(policy, "effective_mode", "") or "").strip().lower()
        runtime_text = "GPU (CUDA)" if effective_mode == "cuda" else ("CPU" if effective_mode == "cpu" else "Runtime pending")
        torch_device = str(getattr(policy, "torch_device", "") or "unknown")
        onnx_provider = str(getattr(policy, "onnx_provider", "") or "unknown")

        applied_prefs = self._applied_face_pipeline_prefs(self.current_face_mode())
        detector_id = str(applied_prefs.get("detector_id") or self.current_face_detector_id())
        embedder_id = str(applied_prefs.get("embedder_id") or self.current_face_embedder_id())
        detector_name = detector_id
        embedder_name = embedder_id
        try:
            detector_name = resolve_face_detector_bundle(
                self.face_model_root,
                self.current_face_mode(),
                detector_id,
            ).display_name or detector_id
        except Exception:
            pass
        try:
            embedder_name = resolve_face_embedder_bundle(
                self.face_model_root,
                self.current_face_mode(),
                embedder_id,
            ).display_name or embedder_id
        except Exception:
            pass
        def _compact_name(value: str) -> str:
            text = str(value or "").replace(" (built-in)", "").strip()
            text = text.replace("VGGFace2 FaceNet", "VGGFace2")
            text = text.replace(" Glint360K", "")
            return text

        label.setText(f"{_compact_name(detector_name)} — {_compact_name(embedder_name)} — {'GPU' if effective_mode == 'cuda' else 'CPU'}")
        folder_detail = f"\nFolder: {folder}" if folder else ""
        label.setToolTip(
            f"Scope: {scope_text}{folder_detail}\nRuntime: {runtime_text}\nTorch device: {torch_device}\n"
            f"ONNX provider: {onnx_provider}\nDetector: {detector_name} ({detector_id})\n"
            f"Embedder: {embedder_name} ({embedder_id})"
        )

    def _populate_component_combo(self, combo: QComboBox, choices: list[tuple[str, str]], target_id: str) -> None:
        combo.blockSignals(True)
        combo.clear()
        for item_id, label in choices:
            combo.addItem(str(label), str(item_id))
        target = normalize_face_component_id(target_id, "")
        if target and not any(str(item_id) == target for item_id, _label in choices):
            combo.addItem(f"{target} (missing)", target)
        self._set_combo_to_id(combo, target)
        combo.blockSignals(False)

    def _refresh_face_pipeline_controls(self) -> None:
        if not hasattr(self, "face_detector_combo"):
            return
        mode = self.current_face_mode()
        prefs = self._mode_face_pipeline_prefs(mode)
        detector_choices = face_detector_choices(self.face_model_root, mode)
        embedder_choices = face_embedder_choices(self.face_model_root, mode)
        self._populate_component_combo(
            self.face_detector_combo,
            detector_choices,
            str(prefs.get("detector_id") or default_face_detector_id(self.face_model_root, mode)),
        )
        self._populate_component_combo(
            self.face_embedder_combo,
            embedder_choices,
            str(prefs.get("embedder_id") or default_face_embedder_id(self.face_model_root, mode)),
        )
        self._populate_component_combo(
            self.face_fallback_detector_combo,
            detector_choices,
            str(prefs.get("fallback_detector_id") or prefs.get("detector_id") or default_face_detector_id(self.face_model_root, mode)),
        )
        self.face_detector_score_spin.blockSignals(True)
        self.face_detector_score_spin.setValue(float(prefs.get("score_threshold", DEFAULT_FACE_SCORE_THRESHOLD) or 0.0))
        self.face_detector_score_spin.blockSignals(False)
        self.face_max_detections_spin.blockSignals(True)
        self.face_max_detections_spin.setValue(max(1, int(prefs.get("max_detections", DEFAULT_FACE_MAX_DETECTIONS) or DEFAULT_FACE_MAX_DETECTIONS)))
        self.face_max_detections_spin.blockSignals(False)
        self.face_detector_policy_combo.blockSignals(True)
        self._set_combo_to_id(self.face_detector_policy_combo, prefs.get("detector_policy", "single"))
        self.face_detector_policy_combo.blockSignals(False)
        self.face_verifier_mode_combo.blockSignals(True)
        self._set_combo_to_id(self.face_verifier_mode_combo, prefs.get("verifier_mode", "off"))
        self.face_verifier_mode_combo.blockSignals(False)
        self._populate_profile_combo(mode)
        self._populate_quality_profile_combo(mode)
        thresholds = dict(prefs.get("quality_thresholds", {}) or {})
        for key, _label, _minimum, _maximum, _step in FACE_QUALITY_NUMERIC_FIELDS:
            widget = getattr(self, f"face_quality_{key}_spin", None)
            if widget is None:
                continue
            widget.blockSignals(True)
            widget.setValue(float(thresholds.get(key, widget.value()) or 0.0))
            widget.blockSignals(False)
        for key, _label in FACE_QUALITY_POLICY_FIELDS:
            combo = getattr(self, f"face_quality_{key}_combo", None)
            if combo is None:
                continue
            combo.blockSignals(True)
            self._set_combo_to_id(combo, thresholds.get(key, "off"))
            combo.blockSignals(False)
        self.face_search_quality_min_combo.blockSignals(True)
        self._set_combo_to_id(self.face_search_quality_min_combo, prefs.get("search_quality_min", "clean"))
        self.face_search_quality_min_combo.blockSignals(False)
        self.face_cluster_quality_min_combo.blockSignals(True)
        self._set_combo_to_id(self.face_cluster_quality_min_combo, prefs.get("cluster_quality_min", "clean"))
        self.face_cluster_quality_min_combo.blockSignals(False)
        self.face_prototype_quality_min_combo.blockSignals(True)
        self._set_combo_to_id(self.face_prototype_quality_min_combo, prefs.get("prototype_quality_min", "clean"))
        self.face_prototype_quality_min_combo.blockSignals(False)
        self.face_recognition_min_score_spin.blockSignals(True)
        self.face_recognition_min_score_spin.setValue(float(prefs.get("recognition_min_score", 0.35) or 0.35))
        self.face_recognition_min_score_spin.blockSignals(False)
        self.face_auto_label_min_score_spin.blockSignals(True)
        self.face_auto_label_min_score_spin.setValue(float(prefs.get("auto_label_min_score", 0.72) or 0.72))
        self.face_auto_label_min_score_spin.blockSignals(False)
        self.face_rerank_policy_combo.blockSignals(True)
        self._set_combo_to_id(self.face_rerank_policy_combo, prefs.get("rerank_policy", "off"))
        self.face_rerank_policy_combo.blockSignals(False)
        self.face_rerank_top_n_spin.blockSignals(True)
        self.face_rerank_top_n_spin.setValue(max(1, int(prefs.get("rerank_top_n", 25) or 25)))
        self.face_rerank_top_n_spin.blockSignals(False)
        self._update_face_model_details()
        self._update_face_pipeline_dirty_state(mode)

    def _on_face_model_profile_changed(self) -> None:
        profile_combo = getattr(self, "face_model_profile_combo", None)
        if profile_combo is None:
            return
        profile_id = str(profile_combo.currentData() or "custom")
        if profile_id == "custom":
            self._update_face_model_details()
            self._update_face_pipeline_dirty_state()
            return
        config = face_model_profile_config(self.current_face_mode(), profile_id)
        if not config:
            self._update_face_model_details()
            self._update_face_pipeline_dirty_state()
            return
        prefs = self._mode_face_pipeline_prefs()
        prefs["detector_id"] = str(config.get("detector_id") or prefs.get("detector_id") or "")
        prefs["embedder_id"] = str(config.get("embedder_id") or prefs.get("embedder_id") or "")
        prefs["score_threshold"] = float(config.get("score_threshold", DEFAULT_FACE_SCORE_THRESHOLD) or 0.0)
        prefs["max_detections"] = max(1, int(config.get("max_detections", DEFAULT_FACE_MAX_DETECTIONS) or DEFAULT_FACE_MAX_DETECTIONS))
        self._refresh_face_pipeline_controls()
        self._on_face_pipeline_changed()

    def _on_face_quality_profile_changed(self) -> None:
        profile_combo = getattr(self, "face_quality_profile_combo", None)
        if profile_combo is None:
            return
        profile_id = str(profile_combo.currentData() or "balanced").strip().lower()
        prefs = self._mode_face_pipeline_prefs()
        prefs["quality_profile_id"] = profile_id
        prefs["quality_thresholds"] = dict(face_quality_profile_config(self.current_face_mode(), profile_id))
        self._refresh_face_pipeline_controls()
        self._on_face_quality_changed()

    def _on_face_quality_changed(self) -> None:
        self._save_face_pipeline_editor_for_mode(self.current_face_mode())
        self._update_face_model_details()
        self._log_face_event(
            "quality_pending_changed",
            profile=self.current_face_quality_profile_id(),
            thresholds=self.current_face_quality_thresholds(),
        )
        self._update_face_pipeline_dirty_state()

    def _update_face_settings_action_state(self) -> None:
        mode = self.current_face_mode()
        dirty = mode in self._face_pipeline_dirty_modes
        if hasattr(self, "face_settings_dirty_label"):
            self.face_settings_dirty_label.setText(
                "Pending face setting changes. Click Apply Face Settings to refresh the review."
                if dirty
                else "Face settings are applied."
            )
        if hasattr(self, "face_apply_settings_button"):
            self.face_apply_settings_button.setEnabled(dirty)
        if hasattr(self, "face_reset_settings_button"):
            self.face_reset_settings_button.setEnabled(dirty)
        if hasattr(self, "face_settings_group"):
            self.face_settings_group.setVisible(False)

    def _apply_pending_face_settings(self) -> bool:
        mode = self.current_face_mode()
        self._save_face_pipeline_editor_for_mode(mode)
        prefs = self._mode_face_pipeline_prefs(mode)
        requested_detector_id = str(prefs.get("detector_id") or default_face_detector_id(self.face_model_root, mode))
        requested_embedder_id = str(prefs.get("embedder_id") or default_face_embedder_id(self.face_model_root, mode))
        try:
            detector_bundle = resolve_face_detector_bundle(self.face_model_root, mode, requested_detector_id)
            embedder_bundle = resolve_face_embedder_bundle(self.face_model_root, mode, requested_embedder_id)
        except Exception as exc:
            errorBox("Face pipeline is unavailable", str(exc))
            return False
        if not detector_bundle.available or not embedder_bundle.available:
            unavailable = []
            if not detector_bundle.available:
                unavailable.append(detector_bundle.display_name)
            if not embedder_bundle.available:
                unavailable.append(embedder_bundle.display_name)
            errorBox(
                "Selected face models are not downloaded",
                "Download the selected component(s) or choose a downloaded pair in Settings > Face Models:\n\n"
                + "\n".join(unavailable),
            )
            return False
        self._face_pipeline_applied_by_mode[mode] = self._clone_face_pipeline_prefs(prefs)
        self._face_pipeline_dirty_modes.discard(mode)
        self._clear_face_tile_caches()
        self._invalidate_face_review_source()
        self._face_profiles_by_name = {}
        self._update_face_model_details()
        self._update_face_mode_status()
        self._update_face_settings_action_state()
        self._log_face_event(
            "pipeline_applied",
            mode=mode,
            detector=self.current_face_detector_id(),
            embedder=self.current_face_embedder_id(),
            score_threshold=self.current_face_detector_score_threshold(),
            max_detections=self.current_face_max_detections(),
        )
        self.refresh_face_library(reason="face settings applied")
        self.refresh_face_album(reason="face settings applied", force_refresh=True)
        self.face_pipeline_applied.emit({mode: self._clone_face_pipeline_prefs(prefs)})
        return True

    def _reset_pending_face_settings(self) -> None:
        mode = self.current_face_mode()
        self._face_pipeline_prefs_by_mode[mode] = self._clone_face_pipeline_prefs(self._applied_face_pipeline_prefs(mode))
        self._face_pipeline_dirty_modes.discard(mode)
        self._refresh_face_pipeline_controls()
        self._update_face_selected_context_label()

    def _active_face_mode_ready_state(self) -> tuple[bool, str]:
        service = self._active_face_service()
        mode_label = face_mode_label(self.current_face_mode())
        ready = True
        message = f"{mode_label} mode ready."
        try:
            ready_fn = getattr(service, "is_ready", None)
            if callable(ready_fn):
                ready = bool(ready_fn())
        except Exception:
            ready = False
        try:
            message_fn = getattr(service, "readiness_message", None)
            if callable(message_fn):
                message = str(message_fn() or "").strip() or message
        except Exception:
            pass
        return bool(ready), message

    def _ensure_face_models_ready(self, action_label: str) -> bool:
        ready, readiness_message = self._active_face_mode_ready_state()
        if ready:
            return True
        mode_label = face_mode_label(self.current_face_mode())
        applied_prefs = self._applied_face_pipeline_prefs(self.current_face_mode())
        embedder_id = normalize_face_component_id(
            str(applied_prefs.get("embedder_id") or ""),
            BUILTIN_HUMAN_EMBEDDER_ID,
        )
        downloadable_model = (
            "facenet"
            if self.current_face_mode() == "human" and embedder_id == BUILTIN_HUMAN_EMBEDDER_ID
            else ""
        )
        message = (
            f"{mode_label} face models are not ready, so ClusterLens cannot {action_label} yet.\n\n"
            f"{readiness_message}\n\n"
            "The model install is shown in Jobs, can be cancelled, and resumes from the shared cache."
        )
        self.status_label.setText(f"{mode_label} face model setup required. {readiness_message}")
        if downloadable_model:
            self.install_face_model_requested.emit(downloadable_model)
        else:
            infoBox("Face model setup required", f"{message}\n\nSettings > Face Models will open next.")
            self.open_face_model_settings_requested.emit()
        return False

    def _update_face_mode_status(self) -> None:
        if not hasattr(self, "face_mode_status_label"):
            return
        ready, message = self._active_face_mode_ready_state()
        prefix = face_mode_label(self.current_face_mode())
        if ready:
            self.face_mode_status_label.setText(f"{prefix}: {message}")
        else:
            self.face_mode_status_label.setText(f"{prefix}: unavailable. {message}")
        active_thread = getattr(self, "_active_thread", None)
        busy = False
        try:
            busy = bool(active_thread is not None and active_thread.isRunning())
        except Exception:
            busy = False
        setup_actions = tuple(
            button
            for button in (
                getattr(self, "face_scan_button", None),
                getattr(self, "face_index_button", None),
            )
            if button is not None
        )
        for button in self._mode_required_buttons:
            can_open_setup = button in setup_actions
            self._set_guarded_action_enabled(button, (not busy) and (ready or can_open_setup))
            if not can_open_setup:
                continue
            base_tooltip = button.property("faceReadinessBaseToolTip")
            if base_tooltip is None:
                base_tooltip = button.toolTip() or ""
                button.setProperty("faceReadinessBaseToolTip", base_tooltip)
            if ready:
                button.setToolTip(str(base_tooltip or ""))
            else:
                setup_note = f"Model setup required: {message}\nClick to open Settings > Face Models."
                button.setToolTip(f"{base_tooltip}\n\n{setup_note}" if base_tooltip else setup_note)
    def _build_global_controls(self) -> None:
        self.sidebar_scroll = QScrollArea(self.sidebar_panel)
        self.sidebar_scroll.setWidgetResizable(True)
        self.sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.sidebar_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.global_controls_scroll = self.sidebar_scroll
        self.sidebar_content_panel = QWidget(self.sidebar_scroll)
        self.sidebar_content_layout = QVBoxLayout(self.sidebar_content_panel)
        self.sidebar_content_layout.setContentsMargins(0, 0, 0, 0)
        self.sidebar_content_layout.setSpacing(8)
        self.global_controls_panel = QWidget(self.sidebar_content_panel)
        self.global_controls_layout = QVBoxLayout(self.global_controls_panel)
        self.global_controls_layout.setContentsMargins(0, 0, 0, 0)
        self.global_controls_layout.setSpacing(8)
        self.sidebar_content_layout.addWidget(self.global_controls_panel)
        self.sidebar_scroll.setWidget(self.sidebar_content_panel)
        self.sidebar_layout.addWidget(self.sidebar_scroll, stretch=1)

        workspace_group, workspace_layout = self._group_box("Status", tooltip=FACE_HELP["face_mode"])
        self.face_workspace_scope_group = workspace_group
        row = QVBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self.face_mode_label = QLabel("Face mode")
        row.addWidget(self.face_mode_label)
        self.face_mode_combo = QComboBox()
        for mode_id in self.supported_face_modes:
            self.face_mode_combo.addItem(face_mode_label(mode_id), mode_id)
        self.face_mode_combo.setToolTip(FACE_HELP["face_mode"])
        self.face_mode_combo.currentIndexChanged.connect(lambda _index: self._on_face_mode_changed())
        if len(self.supported_face_modes) == 1:
            self.face_mode_combo.setEnabled(False)
            self.face_mode_combo.setVisible(False)
            self.face_mode_label.setVisible(False)
        row.addWidget(self.face_mode_combo)
        workspace_layout.addLayout(row)

        self.search_only_current_folder = QCheckBox("Limit search to current folder")
        self.search_only_current_folder.setChecked(True)
        self.search_only_current_folder.setToolTip(FACE_HELP["current_folder_toggle"])
        self.search_only_current_folder.toggled.connect(lambda _checked: self._update_face_status_strip())
        workspace_layout.addWidget(self.search_only_current_folder)
        self.show_tiny_detections_checkbox = QCheckBox("Show Tiny Detections")
        self.show_tiny_detections_checkbox.setChecked(False)
        self.show_tiny_detections_checkbox.setToolTip(FACE_HELP["show_tiny_detections"])
        self.show_tiny_detections_checkbox.stateChanged.connect(lambda _state: self._on_show_tiny_detections_changed())
        workspace_layout.addWidget(self.show_tiny_detections_checkbox)
        self.face_mode_status_label = QLabel()
        self.face_mode_status_label.setWordWrap(True)
        self.face_mode_status_label.setToolTip(FACE_HELP["face_mode"])
        workspace_layout.addWidget(self.face_mode_status_label)
        self.face_model_summary_label = QPushButton("Resolving active face pipeline…")
        self.face_model_summary_label.setObjectName("facesStatusStrip")
        self.face_model_summary_label.setAccessibleName("Active detector, embedder, and runtime")
        self.face_model_summary_label.setToolTip("Open Advanced Face Pipeline")
        self.face_model_summary_label.setFlat(True)
        self.face_model_summary_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.face_model_summary_label.setProperty("role", "section")
        self.face_model_summary_label.clicked.connect(self.face_pipeline_controls_requested.emit)
        workspace_layout.addWidget(self.face_model_summary_label)
        self.global_controls_layout.addWidget(workspace_group)

        summary_group, summary_layout = self._group_box("Active Pipeline", tooltip=FACE_HELP["face_detector"])
        self.face_pipeline_summary_group = summary_group
        pipeline_actions = QHBoxLayout()
        pipeline_actions.setContentsMargins(0, 0, 0, 0)
        pipeline_actions.setSpacing(6)
        self.face_choose_pipeline_button = QPushButton("Advanced Pipeline…")
        self.face_choose_pipeline_button.setProperty("kind", "secondary")
        self.face_choose_pipeline_button.setToolTip(
            "Choose detector, embedder, quality, cascade, and recognition settings."
        )
        self.face_choose_pipeline_button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.face_choose_pipeline_button.setMinimumWidth(0)
        self.face_choose_pipeline_button.clicked.connect(self.face_pipeline_controls_requested.emit)
        pipeline_actions.addWidget(self.face_choose_pipeline_button, stretch=1)
        self._action_buttons.append(self.face_choose_pipeline_button)
        self.face_model_settings_button = QPushButton("Install / Manage Face Models")
        self.face_model_settings_button.setProperty("kind", "secondary")
        self.face_model_settings_button.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.face_model_settings_button.setMinimumWidth(0)
        self.face_model_settings_button.clicked.connect(self.open_face_model_settings_requested.emit)
        pipeline_actions.addWidget(self.face_model_settings_button, stretch=1)
        self._action_buttons.append(self.face_model_settings_button)
        summary_layout.addLayout(pipeline_actions)
        self.global_controls_layout.addWidget(summary_group)

        actions_group, actions_layout = self._group_box("Face Settings", tooltip=FACE_HELP["face_detector"])
        self.face_settings_group = actions_group
        self.face_settings_dirty_label = self._helper_label("Face settings are applied.", tooltip=FACE_HELP["face_detector"])
        actions_layout.addWidget(self.face_settings_dirty_label)
        actions_row = QVBoxLayout()
        actions_row.setContentsMargins(0, 0, 0, 0)
        actions_row.setSpacing(6)
        self.face_apply_settings_button = QPushButton("Apply Face Settings")
        self.face_reset_settings_button = QPushButton("Reset Pending Changes")
        self.face_apply_settings_button.clicked.connect(self._apply_pending_face_settings)
        self.face_reset_settings_button.clicked.connect(self._reset_pending_face_settings)
        actions_row.addWidget(self.face_apply_settings_button)
        actions_row.addWidget(self.face_reset_settings_button)
        actions_layout.addLayout(actions_row)
        self.global_controls_layout.addWidget(actions_group)

        advanced_group, advanced_layout = self._group_box("Advanced Pipeline", tooltip=FACE_HELP["face_detector"])
        self.face_advanced_group = advanced_group
        self.face_advanced_toggle = self._expander_button("Show advanced tuning", checked=False, tooltip=FACE_HELP["face_detector"])
        advanced_layout.addWidget(self.face_advanced_toggle)
        self.face_advanced_panel = QWidget(advanced_group)
        advanced_panel_layout = QVBoxLayout(self.face_advanced_panel)
        advanced_panel_layout.setContentsMargins(0, 0, 0, 0)
        advanced_panel_layout.setSpacing(8)
        self.face_advanced_tabs = QTabWidget(self.face_advanced_panel)
        self.face_advanced_tabs.setDocumentMode(True)
        advanced_panel_layout.addWidget(self.face_advanced_tabs)
        self.face_advanced_toggle.toggled.connect(
            lambda checked: self._set_expander_state(self.face_advanced_toggle, self.face_advanced_panel, checked)
        )
        advanced_layout.addWidget(self.face_advanced_panel)
        self._set_expander_state(self.face_advanced_toggle, self.face_advanced_panel, False)
        self.global_controls_layout.addWidget(advanced_group)

        model_page = QWidget()
        model_layout = QVBoxLayout(model_page)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(8)
        self.face_model_profile_combo = QComboBox()
        self.face_model_profile_combo.setToolTip(FACE_HELP["face_model_profile"])
        self.face_detector_combo = QComboBox()
        self.face_detector_combo.setToolTip(FACE_HELP["face_detector"])
        self.face_embedder_combo = QComboBox()
        self.face_embedder_combo.setToolTip(FACE_HELP["face_embedder"])
        self.face_fallback_detector_combo = QComboBox()
        self.face_fallback_detector_combo.setToolTip(FACE_HELP["face_fallback_detector"])
        self.face_detector_score_spin = QDoubleSpinBox()
        self.face_detector_score_spin.setRange(0.0, 1.0)
        self.face_detector_score_spin.setSingleStep(0.01)
        self.face_detector_score_spin.setToolTip(FACE_HELP["face_detector_score_threshold"])
        self.face_max_detections_spin = QSpinBox()
        self.face_max_detections_spin.setRange(1, 500)
        self.face_max_detections_spin.setToolTip(FACE_HELP["face_max_detections"])
        for title, widget, tooltip in (
            ("Model profile", self.face_model_profile_combo, FACE_HELP["face_model_profile"]),
            ("Detector", self.face_detector_combo, FACE_HELP["face_detector"]),
            ("Embedder", self.face_embedder_combo, FACE_HELP["face_embedder"]),
            ("Fallback detector", self.face_fallback_detector_combo, FACE_HELP["face_fallback_detector"]),
            ("Score threshold", self.face_detector_score_spin, FACE_HELP["face_detector_score_threshold"]),
            ("Max detections", self.face_max_detections_spin, FACE_HELP["face_max_detections"]),
        ):
            model_layout.addWidget(self._field_widget(title, widget, tooltip=tooltip))
        self.face_model_profile_combo.currentIndexChanged.connect(lambda _index: self._on_face_model_profile_changed())
        self.face_detector_combo.currentIndexChanged.connect(lambda _index: self._on_face_pipeline_changed())
        self.face_embedder_combo.currentIndexChanged.connect(lambda _index: self._on_face_pipeline_changed())
        self.face_fallback_detector_combo.currentIndexChanged.connect(lambda _index: self._on_face_pipeline_changed())
        self.face_detector_score_spin.valueChanged.connect(lambda _value: self._on_face_pipeline_changed())
        self.face_max_detections_spin.valueChanged.connect(lambda _value: self._on_face_pipeline_changed())
        model_layout.addStretch(1)
        self.face_advanced_tabs.addTab(model_page, "Models")

        quality_page = QWidget()
        quality_layout = QVBoxLayout(quality_page)
        quality_layout.setContentsMargins(0, 0, 0, 0)
        quality_layout.setSpacing(8)
        self.face_quality_page_layout = quality_layout
        self.face_quality_profile_combo = QComboBox()
        self.face_quality_profile_combo.setToolTip(FACE_HELP["face_quality_profile"])
        quality_layout.addWidget(self._field_widget("Quality profile", self.face_quality_profile_combo, tooltip=FACE_HELP["face_quality_profile"]))
        self.face_quality_summary_label = self._helper_label("", tooltip=FACE_HELP["face_quality_profile"])
        self.face_quality_manual_toggle = self._expander_button("Manual Thresholds", checked=False, tooltip=FACE_HELP["face_quality_profile"])
        quality_layout.addWidget(self.face_quality_manual_toggle)
        self.face_quality_manual_panel = QWidget(quality_page)
        manual_quality_layout = QVBoxLayout(self.face_quality_manual_panel)
        manual_quality_layout.setContentsMargins(0, 0, 0, 0)
        manual_quality_layout.setSpacing(8)
        for key, label, minimum, maximum, step in FACE_QUALITY_NUMERIC_FIELDS:
            widget = QDoubleSpinBox()
            widget.setRange(float(minimum), float(maximum))
            widget.setSingleStep(float(step))
            widget.setDecimals(0 if step >= 1.0 else 2)
            widget.valueChanged.connect(lambda _value, _key=key: self._on_face_quality_changed())
            setattr(self, f"face_quality_{key}_spin", widget)
            manual_quality_layout.addWidget(self._field_widget(label, widget, tooltip=FACE_HELP["face_quality_profile"]))
        for key, label in FACE_QUALITY_POLICY_FIELDS:
            combo = QComboBox()
            combo.addItem("Off", "off")
            combo.addItem("Prefer", "prefer")
            combo.addItem("Require", "require")
            combo.currentIndexChanged.connect(lambda _index, _key=key: self._on_face_quality_changed())
            setattr(self, f"face_quality_{key}_combo", combo)
            manual_quality_layout.addWidget(self._field_widget(label, combo, tooltip=FACE_HELP["face_quality_profile"]))
        self.face_quality_manual_toggle.toggled.connect(
            lambda checked: self._set_expander_state(self.face_quality_manual_toggle, self.face_quality_manual_panel, checked)
        )
        quality_layout.addWidget(self.face_quality_manual_panel)
        self._set_expander_state(self.face_quality_manual_toggle, self.face_quality_manual_panel, False)
        self.face_quality_profile_combo.currentIndexChanged.connect(lambda _index: self._on_face_quality_profile_changed())
        quality_layout.addStretch(1)
        self.face_advanced_tabs.addTab(quality_page, "Quality")

        cascade_page = QWidget()
        cascade_layout = QVBoxLayout(cascade_page)
        cascade_layout.setContentsMargins(0, 0, 0, 0)
        cascade_layout.setSpacing(8)
        self.face_detector_policy_combo = QComboBox()
        for item_id, label in face_detector_policy_choices():
            self.face_detector_policy_combo.addItem(label, item_id)
        self.face_verifier_mode_combo = QComboBox()
        for item_id, label in face_verifier_mode_choices():
            self.face_verifier_mode_combo.addItem(label, item_id)
        self.face_detector_policy_combo.currentIndexChanged.connect(lambda _index: self._on_face_pipeline_changed())
        self.face_verifier_mode_combo.currentIndexChanged.connect(lambda _index: self._on_face_pipeline_changed())
        cascade_layout.addWidget(self._field_widget("Detector policy", self.face_detector_policy_combo, tooltip=FACE_HELP["face_detector_policy"]))
        cascade_layout.addWidget(self._field_widget("Verifier", self.face_verifier_mode_combo, tooltip=FACE_HELP["face_verifier_mode"]))
        cascade_layout.addStretch(1)
        self.face_advanced_tabs.addTab(cascade_page, "Cascade")

        recognition_page = QWidget()
        recognition_layout = QVBoxLayout(recognition_page)
        recognition_layout.setContentsMargins(0, 0, 0, 0)
        recognition_layout.setSpacing(8)
        self.face_search_quality_min_combo = QComboBox()
        self.face_cluster_quality_min_combo = QComboBox()
        self.face_prototype_quality_min_combo = QComboBox()
        for item_id, label in face_quality_gate_choices():
            self.face_search_quality_min_combo.addItem(label, item_id)
            self.face_cluster_quality_min_combo.addItem(label, item_id)
            self.face_prototype_quality_min_combo.addItem(label, item_id)
        self.face_recognition_min_score_spin = QDoubleSpinBox()
        self.face_recognition_min_score_spin.setRange(0.0, 1.0)
        self.face_recognition_min_score_spin.setSingleStep(0.01)
        self.face_auto_label_min_score_spin = QDoubleSpinBox()
        self.face_auto_label_min_score_spin.setRange(0.0, 1.0)
        self.face_auto_label_min_score_spin.setSingleStep(0.01)
        self.face_rerank_policy_combo = QComboBox()
        for item_id, label in face_rerank_policy_choices():
            self.face_rerank_policy_combo.addItem(label, item_id)
        self.face_rerank_top_n_spin = QSpinBox()
        self.face_rerank_top_n_spin.setRange(1, 500)
        for widget in (
            self.face_search_quality_min_combo,
            self.face_cluster_quality_min_combo,
            self.face_prototype_quality_min_combo,
            self.face_rerank_policy_combo,
        ):
            widget.currentIndexChanged.connect(lambda _index: self._on_face_pipeline_changed())
        for widget in (
            self.face_recognition_min_score_spin,
            self.face_auto_label_min_score_spin,
            self.face_rerank_top_n_spin,
        ):
            widget.valueChanged.connect(lambda _value: self._on_face_pipeline_changed())
        for title, widget, tooltip in (
            ("Search quality", self.face_search_quality_min_combo, FACE_HELP["face_search_quality_min"]),
            ("Cluster quality", self.face_cluster_quality_min_combo, FACE_HELP["face_cluster_quality_min"]),
            ("Prototype quality", self.face_prototype_quality_min_combo, FACE_HELP["face_prototype_quality_min"]),
            ("Recognition min score", self.face_recognition_min_score_spin, FACE_HELP["face_recognition_min_score"]),
            ("Auto-label min score", self.face_auto_label_min_score_spin, FACE_HELP["face_auto_label_min_score"]),
            ("Rerank policy", self.face_rerank_policy_combo, FACE_HELP["face_rerank_policy"]),
            ("Rerank top-N", self.face_rerank_top_n_spin, FACE_HELP["face_rerank_top_n"]),
        ):
            recognition_layout.addWidget(self._field_widget(title, widget, tooltip=tooltip))
        recognition_layout.addStretch(1)
        self.face_advanced_tabs.addTab(recognition_page, "Recognition")

        self._update_face_mode_status()
        self._refresh_face_pipeline_controls()

    def _show_tiny_detections_enabled(self) -> bool:
        checkbox = getattr(self, "show_tiny_detections_checkbox", None)
        return bool(checkbox is not None and checkbox.isChecked())

    def _on_show_tiny_detections_changed(self) -> None:
        if self.is_face_folder_tab_active():
            self.refresh_face_library(reason="tiny detections toggled")
        self.refresh_face_album(reason="tiny detections toggled", force_refresh=True)

    def _on_face_mode_changed(self) -> None:
        self._save_face_pipeline_editor_for_mode(self._last_face_mode)
        self._clear_face_tile_caches()
        self._invalidate_face_review_source()
        self._face_review_by_path = {}
        self._face_review_selected_path = ""
        self._face_review_sorted_mode = ""
        self._face_profiles_by_name = {}
        self._refresh_face_pipeline_controls()
        self._last_face_mode = self.current_face_mode()
        self._log_face_event("mode_changed", mode=self.current_face_mode())
        self._update_face_mode_status()
        if self.is_face_folder_tab_active():
            self.refresh_face_library(reason="face mode changed")
        self.refresh_face_album(reason="face mode changed", force_refresh=True)
        self._update_face_settings_action_state()
        self._update_face_selected_context_label()
        self._refresh_face_db_usage_label()

    def _on_face_pipeline_changed(self) -> None:
        self._save_face_pipeline_editor_for_mode(self.current_face_mode())
        self._populate_profile_combo(self.current_face_mode())
        self._update_face_model_details()
        self._log_face_event(
            "pipeline_pending_changed",
            detector=self.current_face_detector_id(),
            embedder=self.current_face_embedder_id(),
            score_threshold=self.current_face_detector_score_threshold(),
            max_detections=self.current_face_max_detections(),
        )
        self._update_face_mode_status()
        self._update_face_pipeline_dirty_state()
        self._update_face_selected_context_label()

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
        widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        bar = FlowLayout(widget)
        self.results_open_btn = QPushButton("Open In Main Gallery")
        self.results_append_btn = QPushButton("Append To Main Gallery")
        self.results_export_btn = QPushButton("Export Paths")
        bar.addWidget(self.results_open_btn)
        bar.addWidget(self.results_append_btn)
        bar.addWidget(self.results_export_btn)

        score_controls = QWidget(widget)
        score_layout = QHBoxLayout(score_controls)
        score_layout.setContentsMargins(0, 0, 0, 0)
        score_layout.setSpacing(6)
        score_layout.addWidget(QLabel("Min score"))
        self.results_min_score = QDoubleSpinBox()
        self.results_min_score.setRange(0.0, 1.0)
        self.results_min_score.setSingleStep(0.01)
        self.results_min_score.setValue(0.0)
        score_layout.addWidget(self.results_min_score)
        bar.addWidget(score_controls)

        sort_controls = QWidget(widget)
        sort_layout = QHBoxLayout(sort_controls)
        sort_layout.setContentsMargins(0, 0, 0, 0)
        sort_layout.setSpacing(6)
        sort_layout.addWidget(QLabel("Sort"))
        self.results_sort = QComboBox()
        self.results_sort.addItems(["Score (desc)", "Name", "Folder", "Hash dist"])
        sort_layout.addWidget(self.results_sort)
        bar.addWidget(sort_controls)

        self.results_open_btn.clicked.connect(self._open_results_in_main_gallery)
        self.results_append_btn.clicked.connect(self._append_results_to_main_gallery)
        self.results_export_btn.clicked.connect(self._export_results_paths)
        self.results_min_score.valueChanged.connect(lambda _v: self._apply_results_view())
        self.results_sort.currentIndexChanged.connect(lambda _i: self._apply_results_view())
        return widget

    def _set_results_kind(self, kind: str) -> None:
        self._results_kind = str(kind)
        empty_states = {
            "none": ("No face results yet", "Choose a Faces task, then scan or search."),
            "faces_review": ("No indexed faces in this folder", "Scan the selected folder to detect faces, then review them here."),
            "faces": ("The saved face library is empty", "Scan a folder to add detected faces."),
            "face_clusters": ("No face groups found", "Try another selection or clustering option."),
            "labels": ("No saved identities yet", "Select detected faces and assign a name."),
            "similarity": ("No matching photos", "Try another query photo or broaden the search."),
        }
        title, detail = empty_states.get(self._results_kind, empty_states["none"])
        self.results_gallery.set_empty_state(title, detail)
        is_similarity = self._results_kind == "similarity"
        show_toolbar = self._results_kind == "similarity"
        try:
            self.results_min_score.setEnabled(is_similarity)
            self.results_sort.setEnabled(is_similarity)
            self.results_toolbar.setVisible(show_toolbar and not self._external_results)
        except Exception:
            pass
        try:
            is_face_review = self._results_kind == "faces_review"
            is_face_result = self._results_kind in {"faces", "face_clusters", "labels"}
            self._set_face_results_tabs_visibility(is_face_review=is_face_review, is_face_result=is_face_result)
            if is_face_result:
                self.face_review_results_tabs.setCurrentIndex(2)
            elif not is_face_review:
                self.face_review_results_tabs.setCurrentIndex(0)
        except Exception:
            pass
        self._update_face_results_context()

    def _set_face_results_tabs_visibility(self, *, is_face_review: bool, is_face_result: bool) -> None:
        tabs = getattr(self, "face_review_results_tabs", None)
        if tabs is None:
            return
        tab_bar = tabs.tabBar()
        try:
            tabs.setTabVisible(0, True)
            tabs.setTabVisible(1, True)
            tabs.setTabVisible(2, True)
        except Exception:
            pass
        tab_bar.setVisible(True)

    def _current_face_photo_filter(self) -> str:
        combo = getattr(self, "face_photo_filter", None)
        if combo is None:
            return "with_faces"
        return str(combo.currentData() or "with_faces").strip().lower()

    @staticmethod
    def _face_photo_filter_matches(
        item: FaceFolderReviewImage,
        photo_filter: str,
        *,
        visible_face_count: int | None = None,
        needs_review: bool | None = None,
        dirty: bool = False,
    ) -> bool:
        mode = str(photo_filter or "with_faces").strip().lower()
        status = str(getattr(item, "review_status", "") or "").strip().lower()
        records = tuple(getattr(item, "visible_faces", ()) or ())
        visible_count = len(records) if visible_face_count is None else max(0, int(visible_face_count))
        if needs_review is None:
            needs_review = any(
                str(getattr(record, "quality_status", "clean") or "clean").strip().lower() in {"review", "reject"}
                for record in records
            )
        if mode == "all":
            return True
        if mode == "needs_review":
            return bool(dirty or needs_review)
        if mode == "no_faces":
            return status in {"no_faces", "tiny_hidden"} and visible_count == 0
        if mode == "not_scanned":
            return status == "not_scanned"
        return visible_count > 0

    def _on_face_photo_filter_changed(self) -> None:
        if self._face_photo_context_kind == "derived":
            self._face_photo_context_kind = "folder"
            self.face_photos_back_button.hide()
        if self._face_review_all_images or self._face_review_folder:
            self._reapply_face_folder_review()
        else:
            self._update_face_results_context()

    def _update_face_results_context(self) -> None:
        tabs = getattr(self, "face_review_results_tabs", None)
        label = getattr(self, "face_results_context_label", None)
        if tabs is None or label is None:
            return
        photo_count = len(list(getattr(self.results_gallery, "images", []) or []))
        review_images = list(getattr(self, "_face_review_images", []) or [])
        face_count = sum(len(self._review_faces_for_item(item)) for item in review_images)
        group_count = len(list(getattr(self, "_face_result_groups", []) or []))
        tabs.setTabText(0, f"Photos ({photo_count})")
        tabs.setTabText(1, f"Faces ({face_count})")
        tabs.setTabText(2, f"Face Groups ({group_count})")
        folder = str(getattr(self, "_face_review_folder", "") or "").strip()
        if not folder and hasattr(self, "face_folder_path"):
            folder = str(self._effective_face_folder() or "").strip()
        folder_label = Path(folder).name or folder or "No folder selected"
        if self._face_photo_context_kind == "derived":
            label.setText(f"{self._face_photo_context_title} | {photo_count} photo(s) | Source: {folder_label}")
            self.face_photo_filter.hide()
            self.face_photos_back_button.show()
            return
        scanned_count = sum(
            1
            for item in list(getattr(self, "_face_review_all_images", []) or [])
            if str(getattr(item, "review_status", "") or "").strip().lower() != "not_scanned"
        )
        total_count = len(list(getattr(self, "_face_review_all_images", []) or []))
        filter_label = str(self.face_photo_filter.currentText() or "With Faces")
        label.setText(
            f"{folder_label} | {photo_count} {filter_label.lower()} photo(s) | "
            f"{face_count} visible face(s) | {scanned_count}/{total_count} scanned"
        )
        self.face_photo_filter.show()
        self.face_photos_back_button.hide()

    def _show_derived_face_photos(self, paths: list[str], *, title: str, focus_path: str = "") -> None:
        image_paths = [str(path) for path in dict.fromkeys(paths or []) if str(path or "").strip()]
        if self._face_photo_context_kind != "derived":
            self._folder_photo_restore_path = str(getattr(self, "_face_review_selected_path", "") or "")
            self._folder_photo_restore_scroll = int(self.results_gallery.list_view.verticalScrollBar().value())
        source_snapshot = self._folder_photo_snapshot
        overlay_source = dict(self._face_result_source_overlay_by_path or {})
        context_source = dict(self._face_result_source_context_by_path or {})
        subtitle_source: dict[str, str] = {}
        boxes_source: dict[str, list[tuple[float, float, float, float]]] = {}
        states_source: dict[str, str] = {}
        if source_snapshot is not None:
            overlay_source = {**source_snapshot.overlay_by_path, **overlay_source}
            subtitle_source = dict(source_snapshot.subtitle_by_path)
            boxes_source = dict(source_snapshot.face_boxes_by_path)
            states_source = dict(source_snapshot.face_box_states_by_path)
            context_source = {**source_snapshot.context_by_path, **context_source}
        self._face_photo_context_kind = "derived"
        self._face_photo_context_title = str(title or "Result photos")
        self._publish_results_gallery_paths(
            image_paths,
            overlay_by_path={path: overlay_source[path] for path in image_paths if path in overlay_source},
            subtitle_by_path={path: subtitle_source[path] for path in image_paths if path in subtitle_source},
            face_boxes_by_path={path: boxes_source[path] for path in image_paths if path in boxes_source},
            face_box_states_by_path={path: states_source[path] for path in image_paths if path in states_source},
            context_provider=lambda path: context_source.get(path, {}),
            clear_pixmaps=True,
            reset_scroll=True,
        )
        if not image_paths:
            self.results_gallery.set_empty_state("No matching photos", "Return to folder photos or choose another result.")
        self.face_review_results_tabs.setCurrentIndex(0)
        self._update_face_results_context()
        if focus_path:
            QTimer.singleShot(0, lambda path=str(focus_path): self._focus_gallery_image(path))

    def _restore_folder_photos(self) -> None:
        snapshot = self._folder_photo_snapshot
        self._face_photo_context_kind = "folder"
        if snapshot is None:
            self._reapply_face_folder_review()
            return
        self._results_kind = "faces_review"
        self._publish_results_gallery_paths(
            list(snapshot.paths),
            overlay_by_path=dict(snapshot.overlay_by_path),
            subtitle_by_path=dict(snapshot.subtitle_by_path),
            face_boxes_by_path=dict(snapshot.face_boxes_by_path),
            face_box_states_by_path=dict(snapshot.face_box_states_by_path),
            context_provider=lambda path: snapshot.context_by_path.get(path, {}),
            clear_pixmaps=True,
            reset_scroll=True,
        )
        self.face_review_results_tabs.setCurrentIndex(0)
        self._update_face_results_context()
        restore_path = str(self._folder_photo_restore_path or snapshot.selected_path or "")
        restore_scroll = int(self._folder_photo_restore_scroll or snapshot.scroll_value or 0)
        if restore_path:
            QTimer.singleShot(0, lambda path=restore_path: self._focus_gallery_image(path))
        QTimer.singleShot(0, lambda value=restore_scroll: self.results_gallery.list_view.verticalScrollBar().setValue(value))

    def _on_tab_changed(self, index: int) -> None:
        navigation = getattr(self, "task_navigation", None)
        navigation_indexes = list(getattr(self, "_task_navigation_tab_indexes", []) or [])
        navigation_index = next(
            (item_index for item_index, tab_index in enumerate(navigation_indexes) if tab_index == int(index)),
            -1,
        )
        if navigation is not None and navigation_index >= 0 and navigation.currentIndex() != navigation_index:
            navigation.blockSignals(True)
            navigation.setCurrentIndex(navigation_index)
            navigation.blockSignals(False)
        self.active_tab_changed.emit(int(index))
        try:
            label = str(self.tabs.tabText(index) or "").strip().lower()
        except Exception:
            label = ""
        self._log_face_event("tab_changed", tab=label or index)
        if self._is_all_faces_tab_label(label):
            self.ensure_face_album_loaded()
        elif self._is_face_folder_tab_label(label) and hasattr(self, "face_scanned_list"):
            self.ensure_face_library_loaded()
        if label == "face search":
            self._update_face_selected_context_label()
        self._update_search_scope_control_visibility(label)
        self._update_face_status_strip()

    def _select_task_from_navigation(self, index: int) -> None:
        navigation_indexes = list(getattr(self, "_task_navigation_tab_indexes", []) or [])
        if not 0 <= int(index) < len(navigation_indexes):
            return
        tab_index = int(navigation_indexes[int(index)])
        if self.tabs.currentIndex() != tab_index:
            self.tabs.setCurrentIndex(tab_index)

    def _on_face_result_view_tab_changed(self, index: int) -> None:
        group_kind = "merged_name" if int(index) == 0 else "raw"
        group_stack = getattr(self, "face_result_group_stack", None)
        if group_stack is not None:
            group_stack.setCurrentIndex(1 if group_kind == "merged_name" else 0)
        if not self._face_result_group_map_for_kind(group_kind):
            return
        selected_group_id = str(self._face_result_selected_group_id_by_kind.get(group_kind, "") or "")
        if not selected_group_id:
            groups = self._face_result_groups_for_kind(group_kind)
            selected_group_id = str(groups[0].group_id) if groups else ""
            self._face_result_selected_group_id_by_kind[group_kind] = selected_group_id
        self._face_result_active_group_kind = group_kind
        self._face_result_selected_group_id = selected_group_id
        self._clear_other_face_result_group_selections(group_kind)
        self._sync_face_result_group_view_selection(group_kind, selected_group_id)
        self._refresh_current_face_result_group()

    def _open_all_faces_task(self) -> None:
        for index in range(self.tabs.count()):
            if self._is_all_faces_tab_label(self.tabs.tabText(index)):
                self.tabs.setCurrentIndex(index)
                self.ensure_face_album_loaded()
                return

    def _update_search_scope_control_visibility(self, tab_label: str | None = None) -> None:
        control = getattr(self, "search_only_current_folder", None)
        if control is None:
            return
        if tab_label is None:
            try:
                tab_label = str(self.tabs.tabText(self.tabs.currentIndex()) or "").strip().lower()
            except Exception:
                tab_label = ""
        has_folder = bool(self._effective_face_folder()) if hasattr(self, "face_folder_path") else bool(self._current_directory())
        show_folder_scope = bool(has_folder and str(tab_label or "").strip().lower() == "face search")
        control.setVisible(show_folder_scope)
        group = getattr(self, "face_workspace_scope_group", None)
        if group is not None:
            group.setVisible(True)
        self._update_face_status_strip()

    def _on_face_review_results_tab_changed(self, index: int) -> None:
        index_value = int(index)
        if index_value == 1:
            tab_name = "detected_faces"
            self._refresh_detected_faces_review(force=True)
            self._schedule_detected_face_tile_loads()
        elif index_value == 2:
            self._cancel_detected_faces_publish()
            tab_name = "face_results"
            self._sync_face_review_result_groups(force=True)
            self._schedule_face_result_tile_loads()
        else:
            self._cancel_detected_faces_publish()
            tab_name = "photos"
        self._log_face_event("review_results_tab_changed", tab=tab_name)

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
            getattr(self.results_gallery, "selected_tags_button", None),
            getattr(self.results_gallery, "status_label", None),
            getattr(self.results_gallery, "progress_bar", None),
        ]:
            if widget is not None:
                widget.hide()
        from PyQt6.QtWidgets import QListView

        self.results_gallery.list_view.setSelectionMode(QListView.SelectionMode.SingleSelection)
        self.results_gallery.image_selected.connect(self._on_results_image_selected)
        self.results_gallery.visible_paths_changed.connect(self._on_results_gallery_visible_paths_changed)
        self.results_gallery.face_edit_service_provider = lambda: self._active_face_service() if self._results_kind == "faces_review" else None
        self.results_gallery.face_draft_provider = self._face_review_draft_state
        self.results_gallery.face_draft_updated_callback = self._on_face_review_drafts_updated
        self.results_gallery.face_box_drag_enabled_provider = lambda: self._results_kind == "faces_review"
        self.results_gallery.face_box_move_callback = self._on_face_review_box_moved
        self.results_gallery.face_edit_saved_callback = self._on_face_image_edits_saved
        self.results_gallery.face_remove_all_callback = self._remove_all_face_review_boxes_for_image
        self.results_gallery.face_reset_draft_callback = self._reset_face_review_draft_for_image
        # The inspector owns the editable draft and save/undo lifecycle.  Give it
        # the pure draft cleaner it expects: (image_path, drafts) -> (drafts, metrics).
        self.results_gallery.face_auto_clean_callback = self._auto_clean_face_review_drafts

    def _set_busy(self, busy: bool, status: str = "") -> None:
        for button in self._action_buttons:
            self._set_guarded_action_enabled(button, not busy)
        self._update_face_mode_status()
        if not busy:
            self._update_face_selected_context_label()
            self._update_detected_face_actions()
            self._update_face_result_actions()
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

        job_id: int | None = None
        if self.job_manager is not None:
            job_id = self.job_manager.register_job(label, cancel_fn=job.cancel)
            self._active_job_id = job_id
            job.progress.connect(
                lambda value, text, job_id=job_id: self.job_manager.update(
                    job_id,
                    progress=value,
                    text=text,
                )
            )
        # Always reflect progress text in the pane status for better UX.
        job.progress.connect(lambda _value, text: self.status_label.setText(str(text)))

        def _on_failed(message: str) -> None:
            self._set_busy(False, "Search index idle.")
            if self.job_manager is not None and job_id is not None:
                self.job_manager.finish(job_id, status="failed", error=message)
            if self._active_job_id == job_id:
                self._active_job_id = None
            errorBox(f"{label} failed", message)

        def _on_cancelled() -> None:
            self._set_busy(False, "Cancelled.")
            if self.job_manager is not None and job_id is not None:
                self.job_manager.finish(job_id, status="cancelled")
            if self._active_job_id == job_id:
                self._active_job_id = None

        def _on_completed(result) -> None:
            self._set_busy(False)
            if self.job_manager is not None and job_id is not None:
                self.job_manager.finish(job_id, status="finished")
            if self._active_job_id == job_id:
                self._active_job_id = None
            on_completed(result)

        job.failed.connect(_on_failed)
        job.cancelled.connect(_on_cancelled)
        job.completed.connect(_on_completed)
        self._active_job = job
        thread = start_job_in_thread(job)
        self._thread_jobs[thread] = job
        thread.finished.connect(
            lambda thread=thread: self._on_async_thread_finished(thread),
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

    def _on_async_thread_finished(self, thread=None) -> None:
        self._release_finished_thread(thread)

    def shutdown_jobs(self, *, timeout_ms: int = 2500) -> bool:
        ready_to_close = True
        deadline_s = perf_counter() + (max(0, int(timeout_ms)) / 1000.0)
        self._results_gallery_publish_token += 1

        def _remaining_timeout_ms() -> int:
            return max(0, int((deadline_s - perf_counter()) * 1000.0))

        def _thread_timeout_ms(thread) -> int:
            if isinstance(thread, QThread):
                return _remaining_timeout_ms()
            return max(0, int(timeout_ms))

        try:
            self._face_tile_request_queue.cancel(len(list(getattr(self, "_face_tile_loader_threads", []) or [])))
        except Exception:
            ready_to_close = False
        try:
            self.results_gallery.cancel_loader(timeout_ms=0)
        except Exception:
            ready_to_close = False

        def _cancel_and_wait(
            job_thread_pairs: list[tuple[object | None, object | None]],
            *,
            detach_on_timeout: bool = False,
        ) -> bool:
            local_ready = True
            cancelled_jobs: set[int] = set()
            waited_threads: set[int] = set()
            for job, _thread in job_thread_pairs:
                if job is None or id(job) in cancelled_jobs:
                    continue
                try:
                    job.cancel()
                    cancelled_jobs.add(id(job))
                except Exception:
                    pass
            for _job, thread in job_thread_pairs:
                if thread is None or id(thread) in waited_threads:
                    continue
                waited_threads.add(id(thread))
                try:
                    if thread.isRunning():
                        wait_timeout_ms = _thread_timeout_ms(thread)
                        thread_ready = wait_for_thread_shutdown(thread, timeout_ms=wait_timeout_ms)
                        if not thread_ready and detach_on_timeout:
                            thread_ready = detach_running_async_job(_job, thread)
                        local_ready = thread_ready and local_ready
                except Exception:
                    local_ready = False
            return local_ready

        if self._active_job is not None:
            try:
                self._active_job.cancel()
            except Exception:
                pass
        if self._active_thread is not None:
            try:
                if self._active_thread.isRunning():
                    wait_timeout_ms = _thread_timeout_ms(self._active_thread)
                    ready_to_close = wait_for_thread_shutdown(self._active_thread, timeout_ms=wait_timeout_ms) and ready_to_close
            except Exception:
                ready_to_close = False
        try:
            self._face_refresh_request_id += 1
            ready_to_close = _cancel_and_wait([
                (getattr(self, "_face_refresh_job", None), getattr(self, "_face_refresh_thread", None)),
                *list(getattr(self, "_retained_face_refresh_refs", []) or []),
            ], detach_on_timeout=True) and ready_to_close
        except Exception:
            ready_to_close = False
        try:
            self._face_album_request_id += 1
            ready_to_close = _cancel_and_wait([
                (getattr(self, "_face_album_refresh_job", None), getattr(self, "_face_album_refresh_thread", None)),
                *list(getattr(self, "_retained_face_album_refresh_refs", []) or []),
                (getattr(self, "_face_album_publish_job", None), getattr(self, "_face_album_publish_thread", None)),
                *list(getattr(self, "_retained_face_album_publish_refs", []) or []),
                (getattr(self, "_face_review_publish_job", None), getattr(self, "_face_review_publish_thread", None)),
                *list(getattr(self, "_retained_face_review_publish_refs", []) or []),
                (getattr(self, "_face_review_hydration_job", None), getattr(self, "_face_review_hydration_thread", None)),
                *list(getattr(self, "_retained_face_review_hydration_refs", []) or []),
            ]) and ready_to_close
        except Exception:
            ready_to_close = False
        try:
            self._face_tile_request_queue.cancel(len(list(getattr(self, "_face_tile_loader_threads", []) or [])))
            for thread in list(getattr(self, "_face_tile_loader_threads", []) or []):
                try:
                    if thread.isRunning():
                        wait_timeout_ms = _thread_timeout_ms(thread)
                        ready_to_close = wait_for_thread_shutdown(thread, timeout_ms=wait_timeout_ms) and ready_to_close
                except Exception:
                    ready_to_close = False
        except Exception:
            ready_to_close = False
        try:
            ready_to_close = self.results_gallery.shutdown_jobs(timeout_ms=_remaining_timeout_ms()) and ready_to_close
        except Exception:
            ready_to_close = False
        if ready_to_close:
            self._active_job = None
            self._active_thread = None
            self._thread_jobs = {}
            self._face_refresh_job = None
            self._face_refresh_thread = None
            self._retained_face_refresh_refs = []
            self._face_album_refresh_job = None
            self._face_album_refresh_thread = None
            self._retained_face_album_refresh_refs = []
            self._face_album_publish_job = None
            self._face_album_publish_thread = None
            self._retained_face_album_publish_refs = []
            self._face_review_publish_job = None
            self._face_review_publish_thread = None
            self._retained_face_review_publish_refs = []
            self._face_review_hydration_job = None
            self._face_review_hydration_thread = None
            self._retained_face_review_hydration_refs = []
            self._face_refresh_in_progress = False
            self._face_tile_loader_threads = []
        return ready_to_close

    def closeEvent(self, event) -> None:
        if not self.shutdown_jobs():
            event.ignore()
            return
        return super().closeEvent(event)

    def resizeEvent(self, event) -> None:
        result = super().resizeEvent(event)
        self._refresh_face_grid_layouts()
        self._schedule_face_tile_refresh()
        return result

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
        self._log_face_event("gallery_image_selected", image_path=image_path, results_kind=self._results_kind)
        if self._results_kind == "faces_review":
            if self._face_review_publish_in_progress:
                self._log_face_event("review_selection_skipped", image_path=image_path, reason="publish in progress")
                return
            self._load_face_review_selection(image_path)
        self.result_selected.emit(image_path, paths, paths.index(image_path), self.results_gallery.inspector_context_provider)

    def _on_results_gallery_visible_paths_changed(self, paths: list[str]) -> None:
        if self._results_kind != "faces_review" or not self._face_review_lazy_publish_enabled:
            return
        if self.face_review_results_tabs.currentWidget() is not self.results_gallery:
            return
        self._queue_face_review_path_hydration(paths, replace_existing=True)

    def _on_face_image_edits_saved(self, image_path: str) -> None:
        image_path = str(image_path or "").strip()
        if not image_path:
            return
        self._log_face_event("image_face_edits_saved", image_path=image_path)
        self._face_review_drafts_by_path.pop(image_path, None)
        self._face_review_draft_dirty_paths.discard(image_path)
        self._clear_face_tile_caches(image_path)
        self._face_review_selected_path = image_path
        self.refresh_face_library(reason="image faces edited")
        self._maybe_refresh_global_face_album(reason="image faces edited")

    def _face_review_draft_state(self, image_path: str) -> tuple[list[EditableFaceDraft], bool] | None:
        path = str(image_path or "").strip()
        if not path:
            return None
        drafts = self._face_review_drafts_by_path.get(path)
        if drafts is None and path not in self._face_review_draft_dirty_paths:
            return None
        return list(drafts or []), path in self._face_review_draft_dirty_paths

    def _drafts_from_review_item(self, item: FaceFolderReviewImage | None) -> list[EditableFaceDraft]:
        if item is None:
            return []
        return [
            EditableFaceDraft(
                bbox=tuple(int(value) for value in record.face_bbox),
                confidence=float(record.face_confidence or 1.0),
                source="indexed",
                person_name=str(record.person_name or ""),
                face_index=int(record.face_index),
            )
            for record in item.visible_faces
        ]

    def _ensure_face_review_drafts(self, image_path: str) -> list[EditableFaceDraft]:
        path = str(image_path or "").strip()
        if not path:
            return []
        drafts = self._face_review_drafts_by_path.get(path)
        if drafts is not None:
            return list(drafts)
        item = self._face_review_by_path.get(path)
        drafts = self._drafts_from_review_item(item)
        self._face_review_drafts_by_path[path] = list(drafts)
        return list(drafts)

    def _on_face_review_drafts_updated(self, image_path: str, drafts: list[EditableFaceDraft], dirty: bool) -> None:
        path = str(image_path or "").strip()
        if not path:
            return
        self._face_review_drafts_by_path[path] = list(drafts or [])
        if dirty:
            self._face_review_draft_dirty_paths.add(path)
        else:
            self._face_review_draft_dirty_paths.discard(path)
        self._clear_face_tile_caches(path)
        self._refresh_face_review_gallery_state()
        self._refresh_detected_faces_review()
        if path == self._face_review_selected_path:
            self._load_face_review_selection(path)

    @staticmethod
    def _face_bbox_metrics(
        bbox: tuple[int, int, int, int],
        *,
        image_width: int,
        image_height: int,
    ) -> dict[str, float]:
        width = max(1, int(image_width))
        height = max(1, int(image_height))
        x1, y1, x2, y2 = [int(value) for value in bbox]
        box_width = max(0, x2 - x1)
        box_height = max(0, y2 - y1)
        area = box_width * box_height
        aspect = float(box_width) / max(1.0, float(box_height))
        min_side = min(box_width, box_height)
        left_clip = max(0, -x1)
        top_clip = max(0, -y1)
        right_clip = max(0, x2 - width)
        bottom_clip = max(0, y2 - height)
        clipped_px = left_clip + top_clip + right_clip + bottom_clip
        perimeter = max(1, (box_width * 2) + (box_height * 2))
        return {
            "width": float(box_width),
            "height": float(box_height),
            "area": float(area),
            "aspect": float(aspect),
            "min_side": float(min_side),
            "edge_clip_fraction": float(clipped_px) / float(perimeter),
        }

    def _face_quality_status(
        self,
        image_path: str,
        bbox: tuple[int, int, int, int],
        *,
        confidence: float,
        saved_record: IndexedFaceRecord | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        if saved_record is not None:
            status = str(saved_record.quality_status or "clean")
            reasons = tuple(str(value) for value in saved_record.quality_reasons or ())
            return status, reasons
        try:
            status, _score, reasons = self._active_face_service().assess_face_bbox(
                str(image_path),
                tuple(int(value) for value in bbox),
                confidence=float(confidence or 0.0),
            )
            return str(status or "clean"), tuple(str(value) for value in reasons or ())
        except Exception:
            return "clean", ()

    @staticmethod
    def _face_quality_status_for_service(
        service,
        image_path: str,
        bbox: tuple[int, int, int, int],
        *,
        confidence: float,
        saved_record: IndexedFaceRecord | None = None,
    ) -> tuple[str, tuple[str, ...]]:
        if saved_record is not None:
            status = str(saved_record.quality_status or "clean")
            reasons = tuple(str(value) for value in saved_record.quality_reasons or ())
            return status, reasons
        try:
            status, _score, reasons = service.assess_face_bbox(
                str(image_path),
                tuple(int(value) for value in bbox),
                confidence=float(confidence or 0.0),
            )
            return str(status or "clean"), tuple(str(value) for value in reasons or ())
        except Exception:
            return "clean", ()

    @staticmethod
    def _summarize_saved_face_quality(records: list[IndexedFaceRecord] | tuple[IndexedFaceRecord, ...]) -> dict[str, int]:
        summary = {"clean": 0, "review": 0, "reject": 0}
        for record in list(records or []):
            status = str(getattr(record, "quality_status", "clean") or "clean")
            summary[status if status in summary else "clean"] += 1
        return summary

    def _summarize_face_quality(
        self,
        image_path: str,
        drafts: list[EditableFaceDraft],
    ) -> dict[str, int]:
        summary = {"clean": 0, "review": 0, "reject": 0}
        for draft in list(drafts or []):
            status, _reasons = self._face_quality_status(
                image_path,
                tuple(int(value) for value in draft.bbox),
                confidence=float(draft.confidence or 1.0),
            )
            summary[status if status in summary else "clean"] += 1
        return summary

    def _auto_clean_face_review_drafts(
        self,
        image_path: str,
        drafts: list[EditableFaceDraft],
    ) -> tuple[list[EditableFaceDraft], dict[str, int]]:
        kept: list[EditableFaceDraft] = []
        removed = 0
        review = 0
        for draft in list(drafts or []):
            status, _reasons = self._face_quality_status(
                str(image_path or ""),
                tuple(int(value) for value in draft.bbox),
                confidence=float(draft.confidence or 1.0),
            )
            if status == "reject":
                removed += 1
                continue
            if status == "review":
                review += 1
            kept.append(draft)
        return kept, {"kept": len(kept), "removed": removed, "review": review}

    def _on_face_review_box_moved(
        self,
        image_path: str,
        box_index: int,
        normalized_box: tuple[float, float, float, float],
    ) -> None:
        path = str(image_path or "").strip()
        if not path:
            return
        item = self._face_review_by_path.get(path)
        if item is None or int(item.image_width) <= 0 or int(item.image_height) <= 0:
            return
        drafts = self._ensure_face_review_drafts(path)
        if box_index < 0 or box_index >= len(drafts):
            return
        bbox = self._absolute_face_bbox(
            normalized_box,
            image_width=item.image_width,
            image_height=item.image_height,
        )
        if bbox is None:
            return
        draft = drafts[box_index]
        drafts[box_index] = EditableFaceDraft(
            bbox=bbox,
            confidence=float(draft.confidence or 1.0),
            source=str(draft.source or "manual"),
            person_name=str(draft.person_name or ""),
            face_index=int(draft.face_index),
        )
        self._on_face_review_drafts_updated(path, drafts, True)

    def _remove_all_face_review_boxes_for_image(self, image_path: str) -> None:
        path = str(image_path or "").strip()
        if not path:
            return
        drafts = self._ensure_face_review_drafts(path)
        if not drafts:
            return
        if not confirmBox(
            "Remove all face boxes?",
            f"Clear all face boxes from {Path(path).name}? The change stays as an unsaved draft until you save it.",
            parent=self,
        ):
            return
        self._on_face_review_drafts_updated(path, [], True)
        if path == self._face_review_selected_path:
            self.face_scanned_summary.setText(
                f"{Path(path).name} now has no draft faces. Save Face Edits in the inspector to commit that change."
            )

    def _reset_face_review_draft_for_image(self, image_path: str) -> None:
        path = str(image_path or "").strip()
        if not path:
            return
        item = self._face_review_by_path.get(path)
        if item is None:
            return
        if path not in self._face_review_draft_dirty_paths:
            return
        if not confirmBox(
            "Reset face draft?",
            f"Reset the unsaved face draft for {Path(path).name} back to the saved indexed faces?",
            parent=self,
        ):
            return
        self._on_face_review_drafts_updated(path, self._drafts_from_review_item(item), False)

    def _auto_clean_face_review_image(self, image_path: str) -> None:
        path = str(image_path or "").strip()
        if not path:
            return
        item = self._face_review_by_path.get(path)
        if item is None:
            return
        drafts = self._ensure_face_review_drafts(path)
        cleaned, metrics = self._auto_clean_face_review_drafts(path, drafts)
        if metrics["removed"] <= 0:
            if metrics["review"] > 0:
                self.status_label.setText(
                    f"Auto-clean left {metrics['review']} face(s) in review for {Path(path).name}; remove or adjust them manually."
                )
            else:
                self.status_label.setText(f"Auto-clean found no obvious junk in {Path(path).name}.")
            if path == self._face_review_selected_path:
                self.face_scanned_summary.setText(
                    f"No obvious junk was removed from {Path(path).name}. "
                    f"{metrics['review']} face(s) still look borderline and need manual review."
                    if metrics["review"] > 0
                    else f"No obvious junk was removed from {Path(path).name}."
                )
            return
        self._on_face_review_drafts_updated(path, cleaned, True)
        self.status_label.setText(
            f"Auto-clean removed {metrics['removed']} obvious junk detection(s) from {Path(path).name}. "
            f"{metrics['review']} face(s) still need manual review."
        )
        if path == self._face_review_selected_path:
            self.face_scanned_summary.setText(
                f"Auto-clean removed {metrics['removed']} obvious junk detection(s) from {Path(path).name}. "
                "Save Face Edits in the inspector to commit the cleanup."
            )

    def _auto_clean_face_review_folder(self) -> None:
        review_images = list(self._face_review_images or [])
        if not review_images:
            self.status_label.setText("No folder review is loaded yet.")
            return
        changed_images = 0
        removed_faces = 0
        review_faces = 0
        for item in review_images:
            drafts = self._ensure_face_review_drafts(item.image_path)
            cleaned, metrics = self._auto_clean_face_review_drafts(item.image_path, drafts)
            review_faces += int(metrics["review"])
            if int(metrics["removed"]) <= 0:
                continue
            changed_images += 1
            removed_faces += int(metrics["removed"])
            self._face_review_drafts_by_path[str(item.image_path)] = list(cleaned)
            self._face_review_draft_dirty_paths.add(str(item.image_path))
        if changed_images <= 0:
            self._refresh_face_review_gallery_state()
            self.status_label.setText(
                "Auto-clean found no obvious junk detections in the current folder review."
                if review_faces <= 0
                else f"Auto-clean found no obvious junk detections, but {review_faces} face(s) still need manual review."
            )
            return
        self._refresh_face_review_gallery_state()
        if self._face_review_selected_path:
            self._load_face_review_selection(self._face_review_selected_path)
        self.status_label.setText(
            f"Auto-clean updated {changed_images} image(s), removed {removed_faces} obvious junk detection(s), "
            f"and left {review_faces} face(s) for manual review."
        )

    def _review_faces_for_item(self, item: FaceFolderReviewImage | None) -> list[EditableFaceDraft]:
        if item is None:
            return []
        path = str(item.image_path or "")
        filtered: list[EditableFaceDraft] = []
        if path in self._face_review_draft_dirty_paths:
            drafts = list(self._face_review_drafts_by_path.get(path, []))
            for index, draft in enumerate(drafts):
                face_index = int(draft.face_index if int(draft.face_index) >= 0 else index)
                status, reasons = self._face_quality_status(
                    path,
                    tuple(int(value) for value in draft.bbox),
                    confidence=float(draft.confidence or 1.0),
                )
                if not self._review_face_matches_filters(
                    image_path=path,
                    face_index=face_index,
                    status=status,
                    reasons=reasons,
                ):
                    continue
                filtered.append(draft)
            return filtered
        for record in tuple(item.visible_faces or ()):
            status = str(record.quality_status or "clean")
            reasons = tuple(str(value) for value in record.quality_reasons or ())
            if not self._review_face_matches_filters(
                image_path=path,
                face_index=int(record.face_index),
                status=status,
                reasons=reasons,
            ):
                continue
            filtered.append(
                EditableFaceDraft(
                    bbox=tuple(int(value) for value in record.face_bbox),
                    confidence=float(record.face_confidence or 1.0),
                    source="indexed",
                    person_name=str(record.person_name or ""),
                    face_index=int(record.face_index),
                )
            )
        return filtered

    def _face_tile_cache_key_for_item(
        self,
        item: FaceTileItem,
        requested_size: QSize,
    ) -> tuple[object, ...] | None:
        image_path = str(item.image_path or "").strip()
        if not image_path:
            return None
        bbox = tuple(int(value) for value in item.bbox)
        return (
            image_path,
            int(self._face_tile_cache_generation),
            int(self._face_tile_path_generations.get(image_path, 0)),
            bbox,
            int(requested_size.width()),
            int(requested_size.height()),
            "default_v1",
        )

    def _is_face_tile_cache_key_current(self, key: tuple[object, ...]) -> bool:
        if not isinstance(key, tuple) or len(key) < 7:
            return False
        image_path = str(key[0] or "").strip()
        if not image_path:
            return False
        return (
            int(key[1]) == int(self._face_tile_cache_generation)
            and int(key[2]) == int(self._face_tile_path_generations.get(image_path, 0))
        )

    def _face_tile_placeholder_image(self, requested_size: QSize) -> QImage:
        key = (max(24, int(requested_size.width())), max(24, int(requested_size.height())))
        cached = self._face_thumb_placeholder_cache.get(key)
        if cached is not None:
            return cached
        image = QImage(key[0], key[1], QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor(18, 18, 18))
        painter = QPainter(image)
        painter.setPen(QColor(76, 76, 76))
        painter.drawRect(0, 0, max(1, key[0] - 1), max(1, key[1] - 1))
        painter.end()
        self._face_thumb_placeholder_cache[key] = image
        return image

    def _face_tile_failed_placeholder_image(self, requested_size: QSize) -> QImage:
        key = (max(24, int(requested_size.width())), max(24, int(requested_size.height())))
        cached = self._face_thumb_failed_placeholder_cache.get(key)
        if cached is not None:
            return cached
        image = QImage(key[0], key[1], QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(QColor(28, 18, 18))
        painter = QPainter(image)
        painter.setPen(QColor(160, 82, 45))
        painter.drawRect(0, 0, max(1, key[0] - 1), max(1, key[1] - 1))
        painter.setPen(QColor(214, 90, 90))
        inset = max(6, min(key[0], key[1]) // 6)
        painter.drawLine(inset, inset, max(inset, key[0] - inset - 1), max(inset, key[1] - inset - 1))
        painter.drawLine(max(inset, key[0] - inset - 1), inset, inset, max(inset, key[1] - inset - 1))
        painter.end()
        self._face_thumb_failed_placeholder_cache[key] = image
        return image

    def _image_for_face_tile(self, item: FaceTileItem, requested_size: QSize) -> QImage:
        key = self._face_tile_cache_key_for_item(item, requested_size)
        if key is None:
            return QImage()
        cached = self._face_thumb_cache.get(key)
        if cached is not None:
            self._face_thumb_cache.move_to_end(key)
            return cached
        if key in self._face_tile_failed_keys:
            return self._face_tile_failed_placeholder_image(requested_size)
        return self._face_tile_placeholder_image(requested_size)

    def _trim_face_thumb_cache(self) -> None:
        while len(self._face_thumb_cache) > max(64, int(self._face_thumb_cache_size)):
            self._face_thumb_cache.popitem(last=False)

    def _remember_face_tile_image(self, cache_key: tuple[object, ...], image: QImage) -> None:
        self._face_tile_failed_keys.discard(cache_key)
        self._face_tile_failure_messages.pop(cache_key, None)
        self._face_thumb_cache[cache_key] = image
        self._face_thumb_cache.move_to_end(cache_key)
        self._trim_face_thumb_cache()

    def _clear_face_tile_failures(self, image_path: str | None = None) -> None:
        target = str(image_path or "").strip()
        if not target:
            self._face_tile_failed_keys.clear()
            self._face_tile_failure_messages = {}
            return
        self._face_tile_failed_keys = {
            key
            for key in self._face_tile_failed_keys
            if str(key[0]) != target
        }
        self._face_tile_failure_messages = {
            key: value
            for key, value in self._face_tile_failure_messages.items()
            if str(key[0]) != target
        }

    def apply_face_tile_preferences(self, *, worker_count: int, cache_size: int) -> None:
        self._face_tile_loader_target_count = max(1, int(worker_count))
        self._face_thumb_cache_size = max(64, int(cache_size))
        self._trim_face_thumb_cache()
        self._restart_face_tile_loader_threads()

    def _restart_face_tile_loader_threads(self) -> None:
        threads = list(self._face_tile_loader_threads or [])
        if threads:
            try:
                self._face_tile_request_queue.cancel(len(threads))
            except Exception:
                pass
            for thread in threads:
                try:
                    if thread.isRunning():
                        wait_for_thread_shutdown(thread, timeout_ms=1000)
                except Exception:
                    pass
        self._face_tile_loader_threads = []
        self._face_tile_pending_keys.clear()
        self._face_tile_inflight_keys.clear()
        self._face_tile_request_queue = FaceTileRequestQueue()
        self._start_face_tile_loader_threads()

    def _queue_face_tile_load(self, item: FaceTileItem, requested_size: QSize, *, priority: int = 100) -> bool:
        key = self._face_tile_cache_key_for_item(item, requested_size)
        if (
            key is None
            or key in self._face_thumb_cache
            or key in self._face_tile_pending_keys
            or key in self._face_tile_failed_keys
        ):
            return False
        request = FaceTileLoadRequest(
            cache_key=key,
            image_path=str(item.image_path),
            bbox=tuple(int(value) for value in item.bbox),
            requested_size=(max(24, int(requested_size.width())), max(24, int(requested_size.height()))),
        )
        if self._face_tile_request_queue.enqueue(request, priority=int(priority)):
            self._face_tile_pending_keys.add(key)
            return True
        return False

    def _visible_and_prefetch_face_tile_rows(
        self,
        view: QListView,
        model: FaceTileListModel,
    ) -> tuple[list[int], list[int]]:
        if view is None or model is None or model.rowCount() <= 0 or not view.isVisible():
            return [], []
        grid_size = view.gridSize()
        grid_width = max(1, int(grid_size.width() or 0))
        grid_height = max(1, int(grid_size.height() or 0))
        viewport_rect = view.viewport().rect()
        if grid_width <= 0 or grid_height <= 0 or viewport_rect.width() <= 0 or viewport_rect.height() <= 0:
            return list(range(min(model.rowCount(), 12))), []
        columns = max(1, int((viewport_rect.width() + grid_width - 1) // grid_width))
        total_rows = max(1, (model.rowCount() + columns - 1) // columns)
        first_row = max(0, int((view.verticalScrollBar().value() if view.verticalScrollBar() is not None else 0) // grid_height))
        visible_row_count = max(1, int(viewport_rect.height() // grid_height) + 2)
        prefetch_rows = max(0, int(self.settings.thumbnail_prefetch_rows))
        visible_start = min(total_rows - 1, first_row)
        visible_end = min(total_rows - 1, first_row + visible_row_count)
        prefetch_start = max(0, visible_start - prefetch_rows)
        prefetch_end = min(total_rows - 1, visible_end + prefetch_rows)

        def _rows_to_indexes(start_row: int, end_row: int) -> list[int]:
            indexes: list[int] = []
            for row in range(start_row, end_row + 1):
                start_index = row * columns
                end_index = min(model.rowCount() - 1, ((row + 1) * columns) - 1)
                if end_index < start_index:
                    continue
                indexes.extend(range(start_index, end_index + 1))
            return indexes

        visible_rows = _rows_to_indexes(visible_start, visible_end)
        prefetch_before = _rows_to_indexes(prefetch_start, max(prefetch_start - 1, visible_start - 1))
        prefetch_after = _rows_to_indexes(visible_end + 1, prefetch_end)
        return visible_rows, (prefetch_before + prefetch_after)

    def _iter_visible_face_tile_views(self) -> list[tuple[str, QListView, FaceTileListModel]]:
        entries: list[tuple[str, QListView, FaceTileListModel]] = []
        for source, view, model in (
            ("selected_strip", getattr(self, "face_scanned_list", None), getattr(self, "face_scanned_model", None)),
            ("detected_faces", getattr(self, "face_detected_faces_list", None), getattr(self, "face_detected_faces_model", None)),
            ("face_results", getattr(self, "face_results_list", None), getattr(self, "face_results_model", None)),
            ("pending_preview", getattr(self, "face_pending_preview_list", None), getattr(self, "face_pending_preview_model", None)),
            ("identity_prototype", getattr(self, "face_identity_prototype_list", None), getattr(self, "face_identity_prototype_model", None)),
        ):
            if view is None or model is None:
                continue
            if not view.isVisible() or model.rowCount() <= 0:
                continue
            entries.append((source, view, model))
        return entries

    def _queue_visible_face_tiles(self, view: QListView | None, model: FaceTileListModel | None, *, source: str) -> None:
        if view is None or model is None:
            return
        rows, prefetch_rows = self._visible_and_prefetch_face_tile_rows(view, model)
        visible_budget = FACE_TILE_VISIBLE_REQUEST_LIMIT
        prefetch_budget = FACE_TILE_PREFETCH_REQUEST_LIMIT
        visible_priority = 0
        for row in rows:
            if visible_budget <= 0:
                break
            item = model.item_at(row)
            if item is None:
                continue
            if self._queue_face_tile_load(item, model.requested_icon_size(), priority=visible_priority):
                visible_priority += 1
                visible_budget -= 1
        prefetch_priority = 0
        for row in prefetch_rows:
            if prefetch_budget <= 0:
                break
            item = model.item_at(row)
            if item is None:
                continue
            if self._queue_face_tile_load(
                item,
                model.requested_icon_size(),
                priority=FACE_TILE_VISIBLE_REQUEST_LIMIT + prefetch_priority,
            ):
                prefetch_priority += 1
                prefetch_budget -= 1

    def _schedule_face_tile_refresh(self) -> None:
        delay_ms = max(1, int(self.settings.gallery_flush_interval_ms))
        self._face_tile_refresh_timer.start(delay_ms)

    def _flush_face_tile_refresh(self) -> None:
        request_specs: list[tuple[FaceTileItem, QSize, int]] = []
        desired_keys: set[tuple[object, ...]] = set()
        self._face_tile_request_queue.clear()
        self._face_tile_pending_keys = set(self._face_tile_inflight_keys)
        visible_budget = FACE_TILE_VISIBLE_REQUEST_LIMIT
        prefetch_budget = FACE_TILE_PREFETCH_REQUEST_LIMIT
        for _source, view, model in self._iter_visible_face_tile_views():
            visible_rows, prefetch_rows = self._visible_and_prefetch_face_tile_rows(view, model)
            if visible_budget > 0:
                for row in visible_rows:
                    if visible_budget <= 0:
                        break
                    item = model.item_at(row)
                    if item is None:
                        continue
                    key = model.cache_key_for_item(item)
                    if (
                        key is None
                        or key in desired_keys
                        or key in self._face_thumb_cache
                        or key in self._face_tile_pending_keys
                        or key in self._face_tile_failed_keys
                    ):
                        continue
                    desired_keys.add(key)
                    request_specs.append((item, model.requested_icon_size(), len(request_specs)))
                    visible_budget -= 1
            if prefetch_budget > 0:
                for row in prefetch_rows:
                    if prefetch_budget <= 0:
                        break
                    item = model.item_at(row)
                    if item is None:
                        continue
                    key = model.cache_key_for_item(item)
                    if (
                        key is None
                        or key in desired_keys
                        or key in self._face_thumb_cache
                        or key in self._face_tile_pending_keys
                        or key in self._face_tile_failed_keys
                    ):
                        continue
                    desired_keys.add(key)
                    request_specs.append((item, model.requested_icon_size(), FACE_TILE_VISIBLE_REQUEST_LIMIT + len(request_specs)))
                    prefetch_budget -= 1
            if visible_budget <= 0 and prefetch_budget <= 0:
                break
        for item, requested_size, priority in request_specs:
            self._queue_face_tile_load(item, requested_size, priority=priority)

    def _schedule_selected_face_tile_loads(self) -> None:
        self._schedule_face_tile_refresh()

    def _schedule_detected_face_tile_loads(self) -> None:
        if self.face_review_results_tabs.currentWidget() is not self.face_detected_faces_panel:
            return
        self._schedule_face_tile_refresh()

    def _schedule_face_result_tile_loads(self) -> None:
        if self.face_review_results_tabs.currentWidget() is not self.face_results_panel:
            return
        self._schedule_face_tile_refresh()

    def _notify_face_tile_models_for_key(self, cache_key: tuple[object, ...]) -> None:
        for model in (
            getattr(self, "face_detected_faces_model", None),
            getattr(self, "face_scanned_model", None),
            getattr(self, "face_results_model", None),
            getattr(self, "face_pending_preview_model", None),
            getattr(self, "face_identity_prototype_model", None),
        ):
            if model is None:
                continue
            model.notify_rows_changed(model.rows_for_cache_key(cache_key))

    @pyqtSlot(object)
    def _on_face_tile_load_started(self, cache_key) -> None:
        key = tuple(cache_key) if isinstance(cache_key, tuple) else cache_key
        if isinstance(key, tuple):
            self._face_tile_inflight_keys.add(key)

    @pyqtSlot(object, object, float)
    def _on_face_tile_loaded(self, cache_key, image, duration_s: float) -> None:
        key = tuple(cache_key) if isinstance(cache_key, tuple) else cache_key
        self._face_tile_pending_keys.discard(key)
        self._face_tile_inflight_keys.discard(key)
        if not isinstance(key, tuple):
            return
        if not self._is_face_tile_cache_key_current(key):
            return
        if not isinstance(image, QImage) or image.isNull():
            return
        self._remember_face_tile_image(key, image)
        self._notify_face_tile_models_for_key(key)
        self._refresh_detected_faces_summary_for_key(key)
        self._refresh_face_result_hover_popup_if_needed()
        if LOGGER.isEnabledFor(10):
            LOGGER.debug("FacePane face_tile_loaded duration_ms=%s", int(round(float(duration_s) * 1000.0)))

    @pyqtSlot(object, str)
    def _on_face_tile_load_failed(self, cache_key, error: str) -> None:
        key = tuple(cache_key) if isinstance(cache_key, tuple) else cache_key
        self._face_tile_pending_keys.discard(key)
        self._face_tile_inflight_keys.discard(key)
        if isinstance(key, tuple) and not self._is_face_tile_cache_key_current(key):
            return
        if isinstance(key, tuple):
            self._face_tile_failed_keys.add(key)
            self._face_tile_failure_messages[key] = str(error or "unavailable")
            self._notify_face_tile_models_for_key(key)
            self._refresh_detected_faces_summary_for_key(key)
        if LOGGER.isEnabledFor(10):
            LOGGER.debug("FacePane face_tile_failed error=%s", str(error or "unknown"))

    def _start_face_tile_loader_threads(self) -> None:
        desired = max(1, int(self._face_tile_loader_target_count))
        if len(self._face_tile_loader_threads) >= desired:
            return
        for _index in range(len(self._face_tile_loader_threads), desired):
            thread = FaceTileLoaderThread(self._face_tile_request_queue, self)
            thread.tile_started.connect(self._on_face_tile_load_started)
            thread.tile_loaded.connect(self._on_face_tile_loaded)
            thread.tile_failed.connect(self._on_face_tile_load_failed)
            thread.start()
            self._face_tile_loader_threads.append(thread)

    def _clear_face_tile_caches(self, image_path: str | None = None) -> None:
        target = str(image_path or "").strip()
        if not target:
            self._face_tile_cache_generation += 1
            self._face_tile_path_generations = {}
            self._face_thumb_cache = OrderedDict()
            self._face_tile_pending_keys.clear()
            self._face_tile_inflight_keys.clear()
            self._clear_face_tile_failures()
            self._face_thumb_placeholder_cache = {}
            self._face_thumb_failed_placeholder_cache = {}
            self._face_tile_request_queue.clear()
            return
        self._face_tile_path_generations[target] = int(self._face_tile_path_generations.get(target, 0)) + 1
        self._face_thumb_cache = OrderedDict(
            (key, value)
            for key, value in self._face_thumb_cache.items()
            if str(key[0]) != target
        )
        self._face_tile_pending_keys = {
            key
            for key in self._face_tile_pending_keys
            if str(key[0]) != target
        }
        self._face_tile_inflight_keys = {
            key
            for key in self._face_tile_inflight_keys
            if str(key[0]) != target
        }
        self._clear_face_tile_failures(target)

    @staticmethod
    def _selected_tile_indexes(view: QListView | None) -> list[QModelIndex]:
        if view is None:
            return []
        try:
            selection_model = view.selectionModel()
            if selection_model is None:
                return []
            return list(selection_model.selectedIndexes())
        except Exception:
            return []

    @staticmethod
    def _selected_tile_items(view: QListView | None, model: FaceTileListModel | None) -> list[FaceTileItem]:
        if view is None or model is None:
            return []
        items: list[FaceTileItem] = []
        for index in SearchPane._selected_tile_indexes(view):
            item = model.item_at(index.row())
            if item is not None:
                items.append(item)
        return items

    @staticmethod
    def _selected_list_entries(view: QListView | None, model: ListEntryModel | None) -> list[ListEntry]:
        if view is None or model is None:
            return []
        items: list[ListEntry] = []
        for index in SearchPane._selected_tile_indexes(view):
            item = model.item_at(index.row())
            if item is not None:
                items.append(item)
        return items

    @staticmethod
    def _face_ref_from_tile_item(item: FaceTileItem | None) -> tuple[str, int] | None:
        if item is None:
            return None
        image_path = str(item.image_path or "").strip()
        face_index = int(item.saved_face_index if int(item.saved_face_index) >= 0 else item.face_index)
        if not image_path or face_index < 0:
            return None
        return (image_path, face_index)

    def _selected_detected_face_tiles(self) -> list[tuple[str, int]]:
        refs: list[tuple[str, int]] = []
        for index in self._selected_tile_indexes(getattr(self, "face_detected_faces_list", None)):
            data = self.face_detected_faces_model.data(index, FaceTileListModel.RefRole)
            if isinstance(data, tuple) and len(data) == 2:
                refs.append((str(data[0]), int(data[1])))
        return refs

    def _selected_detected_face_indexed_refs(self) -> list[tuple[str, int]]:
        refs: list[tuple[str, int]] = []
        for item in self._selected_tile_items(getattr(self, "face_detected_faces_list", None), getattr(self, "face_detected_faces_model", None)):
            ref = self._face_ref_from_tile_item(item)
            if ref is not None:
                refs.append(ref)
        return refs

    def _selected_face_result_tiles(self) -> list[tuple[str, int]]:
        refs: list[tuple[str, int]] = []
        for item in self._selected_tile_items(getattr(self, "face_results_list", None), getattr(self, "face_results_model", None)):
            ref = self._face_ref_from_tile_item(item)
            if ref is not None:
                refs.append(ref)
        return refs

    def _face_result_group_refs(
        self,
        group: FaceResultGroup | None,
        *,
        prefer_selection: bool = True,
    ) -> list[tuple[str, int]]:
        if group is None:
            return []
        group_refs = [
            ref
            for ref in (self._face_ref_from_tile_item(item) for item in group.items)
            if ref is not None
        ]
        group_ref_set = set(group_refs)
        if prefer_selection:
            selected_refs = [ref for ref in self._selected_face_result_tiles() if ref in group_ref_set]
            if len(selected_refs) >= 2:
                return list(dict.fromkeys(selected_refs))
        return list(dict.fromkeys(group_refs))

    def _update_detected_face_actions(self) -> None:
        any_selected = self._selected_detected_face_tiles()
        searchable_selected = self._selected_detected_face_indexed_refs()
        has_selection = bool(any_selected)
        self._set_guarded_action_enabled(self.face_detected_remove_button, has_selection)
        self._set_guarded_action_enabled(self.face_detected_jump_button, has_selection)
        self._set_guarded_action_enabled(self.face_detected_edit_button, has_selection)
        self._set_guarded_action_enabled(self.face_detected_name_button, bool(searchable_selected))
        self._set_guarded_action_enabled(self.face_detected_find_button, len(searchable_selected) == 1)
        self._set_guarded_action_enabled(self.face_detected_cluster_selected_button, len(searchable_selected) >= 2)
        has_visible = bool(
            int(getattr(self, "_face_detected_visible_scope_ref_count", 0) or 0) > 1
            or getattr(self.face_detected_faces_model, "rowCount", lambda: 0)() > 1
        )
        self._set_guarded_action_enabled(self.face_detected_cluster_visible_button, has_visible)

    def _on_detected_face_selection_changed(self) -> None:
        self._update_detected_face_actions()
        if self._face_selection_sync_in_progress:
            return
        items = self._selected_tile_items(self.face_detected_faces_list, self.face_detected_faces_model)
        if not items:
            self._clear_active_face_selection(source="detected_faces")
            self._update_face_selected_context_label()
            return
        self._set_active_face_selection(items, source="detected_faces")
        image_paths = self._active_face_selection_paths()
        records = self._face_tile_item_records(items)
        if len(image_paths) == 1:
            self._face_selection_sync_in_progress = True
            try:
                self._focus_face_review_image(image_paths[0])
                if records:
                    self._select_scanned_face_records(records)
            finally:
                self._face_selection_sync_in_progress = False
        else:
            scanned_selection_model = self.face_scanned_list.selectionModel()
            if scanned_selection_model is not None:
                scanned_selection_model.blockSignals(True)
                scanned_selection_model.clearSelection()
                scanned_selection_model.blockSignals(False)
        self._update_face_selected_context_label()

    def _set_face_identity_name_fields(self, person_name: str) -> None:
        name = str(person_name or "").strip()
        if not name:
            return
        for widget_name in ("face_name_query", "face_label_name", "person_name"):
            widget = getattr(self, widget_name, None)
            if widget is None:
                continue
            try:
                widget.setText(name)
            except Exception:
                pass

    def _resolve_detected_face_context_name(
        self,
        *,
        prompt: bool = False,
        prompt_title: str = "Name Face",
        prompt_label: str = "Saved identity name",
    ) -> str:
        suggested_names: list[str] = []
        service = self._active_face_service()
        metadata_cache: dict[tuple[str, int], tuple[str, str]] = {}
        for item in self._selected_tile_items(
            getattr(self, "face_detected_faces_list", None),
            getattr(self, "face_detected_faces_model", None),
        ):
            person_name, _quality_status = self._face_result_item_metadata_worker(
                item,
                service=service,
                metadata_cache=metadata_cache,
            )
            if person_name:
                suggested_names.append(person_name)
        preferred = suggested_names[0] if suggested_names else self._current_face_identity_name()
        if prompt or not preferred:
            value, accepted = QInputDialog.getText(
                self,
                str(prompt_title),
                str(prompt_label),
                text=str(preferred or ""),
            )
            if not accepted:
                return ""
            preferred = str(value or "").strip()
        if preferred:
            self._set_face_identity_name_fields(preferred)
        return str(preferred or "").strip()

    def _prepare_detected_faces_context_selection(self, point) -> QModelIndex | None:
        view = getattr(self, "face_detected_faces_list", None)
        if view is None:
            return None
        index = view.indexAt(point)
        if not index.isValid():
            return None
        selection_model = view.selectionModel()
        if selection_model is not None and not selection_model.isSelected(index):
            selection_model.clearSelection()
            selection_model.select(index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
            selection_model.setCurrentIndex(index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        elif selection_model is not None:
            # QListView.setCurrentIndex() updates the selection in icon mode.
            # Keep an existing Shift/Ctrl selection intact when its member is
            # right-clicked to open a batch-action menu.
            selection_model.setCurrentIndex(index, QItemSelectionModel.SelectionFlag.NoUpdate)
        else:
            view.setCurrentIndex(index)
        return index

    def _name_detected_faces_from_context(self) -> None:
        refs = self._selected_detected_face_indexed_refs()
        if not refs:
            errorBox("No faces selected", "Select one or more saved face tiles in Detected Faces first.")
            return
        person_name = self._resolve_detected_face_context_name(prompt=True)
        if not person_name:
            return
        self._save_face_refs_name(
            refs,
            source_label="Detected Faces",
            person_name_override=person_name,
        )

    def _search_detected_faces_name_from_context(self) -> None:
        if not self._selected_detected_face_tiles():
            errorBox("No faces selected", "Select a face tile in Detected Faces first.")
            return
        person_name = self._resolve_detected_face_context_name(
            prompt=False,
            prompt_title="Find Photos By Name",
            prompt_label="Saved identity name",
        )
        if not person_name:
            person_name = self._resolve_detected_face_context_name(
                prompt=True,
                prompt_title="Find Photos By Name",
                prompt_label="Saved identity name",
            )
        if not person_name:
            return
        self._search_by_name()

    def _build_detected_faces_context_menu(self) -> QMenu:
        menu = QMenu(self)
        any_selected = self._selected_detected_face_tiles()
        indexed_refs = self._selected_detected_face_indexed_refs()
        if len(any_selected) > 1:
            name_action = menu.addAction("Name Selected Faces...")
            cluster_action = menu.addAction("Cluster Selected Faces")
            remove_action = menu.addAction("Remove Selected Faces")
            name_action.setToolTip("Apply one saved identity name to every selected face tile.")
            cluster_action.setToolTip("Create face groups using only the selected indexed face tiles.")
            remove_action.setToolTip("Remove every selected face draft from this folder review.")
            name_action.triggered.connect(self._name_detected_faces_from_context)
            cluster_action.triggered.connect(self._cluster_selected_detected_faces)
            remove_action.triggered.connect(self._remove_selected_detected_face_tiles)
            name_action.setEnabled(bool(indexed_refs) and not self._read_only_mode)
            cluster_action.setEnabled(len(indexed_refs) >= 2)
            remove_action.setEnabled(not self._read_only_mode)
            self._detected_faces_context_actions = {
                "name": name_action,
                "cluster_selected": cluster_action,
                "remove": remove_action,
            }
            return menu

        find_similar_action = menu.addAction("Find Similar")
        name_action = menu.addAction("Name Face...")
        find_by_name_action = menu.addAction("Find Photos by This Name")
        menu.addSeparator()
        jump_action = menu.addAction("Jump To Photo")
        inspector_action = menu.addAction("Open Inspector")
        remove_action = menu.addAction("Remove Face")
        name_action.setToolTip("Give this face tile a saved identity name.")
        find_similar_action.triggered.connect(self._search_selected_detected_face)
        name_action.triggered.connect(self._name_detected_faces_from_context)
        find_by_name_action.triggered.connect(self._search_detected_faces_name_from_context)
        jump_action.triggered.connect(self._jump_to_selected_detected_face)
        inspector_action.triggered.connect(self._open_selected_detected_face_in_inspector)
        remove_action.triggered.connect(self._remove_selected_detected_face_tiles)
        find_similar_action.setEnabled(len(indexed_refs) == 1)
        name_action.setEnabled(bool(indexed_refs) and not self._read_only_mode)
        find_by_name_action.setEnabled(bool(any_selected))
        jump_action.setEnabled(bool(any_selected))
        inspector_action.setEnabled(bool(any_selected))
        remove_action.setEnabled(bool(any_selected) and not self._read_only_mode)
        self._detected_faces_context_actions = {
            "find_similar": find_similar_action,
            "name": name_action,
            "find_by_name": find_by_name_action,
            "jump": jump_action,
            "inspect": inspector_action,
            "remove": remove_action,
        }
        return menu

    def _show_detected_faces_context_menu(self, point) -> None:
        view = getattr(self, "face_detected_faces_list", None)
        if view is None:
            return
        self._prepare_detected_faces_context_selection(point)
        menu = self._build_detected_faces_context_menu()
        if not menu.actions():
            return
        previous_menu = self._detected_faces_context_menu
        if previous_menu is not None:
            previous_menu.close()
            previous_menu.deleteLater()
        self._detected_faces_context_menu = menu

        def _discard_menu() -> None:
            if self._detected_faces_context_menu is menu:
                self._detected_faces_context_menu = None
            menu.deleteLater()

        menu.aboutToHide.connect(_discard_menu)
        menu.popup(view.viewport().mapToGlobal(point))

    def _prepare_face_result_context_selection(self, point) -> QModelIndex | None:
        view = getattr(self, "face_results_list", None)
        if view is None:
            return None
        index = view.indexAt(point)
        if not index.isValid():
            return None
        selection_model = view.selectionModel()
        if selection_model is not None and not selection_model.isSelected(index):
            selection_model.clearSelection()
            selection_model.select(index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
            selection_model.setCurrentIndex(index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        elif selection_model is not None:
            # Retain an existing Shift/Ctrl multi-selection when one of its
            # tiles is right-clicked for a batch naming action.
            selection_model.setCurrentIndex(index, QItemSelectionModel.SelectionFlag.NoUpdate)
        else:
            view.setCurrentIndex(index)
        return index

    def _name_face_results_from_prompt(self) -> None:
        refs = self._selected_face_result_tiles()
        if not refs:
            errorBox("No faces selected", "Select one or more saved face tiles in Grouped Photos first.")
            return
        selected_count = len(refs)
        title = "Name Selected Faces" if selected_count > 1 else "Name Face"
        person_name, accepted = QInputDialog.getText(
            self,
            title,
            "Saved identity name",
            text=str(self._current_face_identity_name() or ""),
        )
        if not accepted:
            return
        resolved_name = str(person_name or "").strip()
        if not resolved_name:
            return
        self._set_face_identity_name_fields(resolved_name)
        self._save_face_refs_name(
            refs,
            source_label="Grouped Photos",
            person_name_override=resolved_name,
        )

    def _build_face_results_context_menu(self) -> QMenu:
        menu = QMenu(self)
        selected_items = self._selected_tile_items(
            getattr(self, "face_results_list", None),
            getattr(self, "face_results_model", None),
        )
        refs = self._selected_face_result_tiles()
        if len(selected_items) > 1:
            name_action = menu.addAction("Name Selected Faces…")
            name_action.setToolTip("Apply one saved identity name to every selected face tile.")
            name_action.triggered.connect(self._name_face_results_from_prompt)
            name_action.setEnabled(bool(refs) and not self._read_only_mode)
            self._face_results_context_actions = {"name": name_action}
            return menu

        name_action = menu.addAction("Name Face…")
        name_action.setToolTip("Give this saved face tile an identity name.")
        name_action.triggered.connect(self._name_face_results_from_prompt)
        show_photo_action = menu.addAction("Show Photo")
        show_photo_action.setToolTip("Open the source photo for this face tile.")
        show_photo_action.triggered.connect(self._jump_to_selected_face_result)
        inspector_action = menu.addAction("Open Inspector")
        inspector_action.setToolTip("Open the source photo in the inspector.")
        inspector_action.triggered.connect(self._open_selected_face_result_in_inspector)
        name_action.setEnabled(bool(refs) and not self._read_only_mode)
        show_photo_action.setEnabled(bool(refs))
        inspector_action.setEnabled(bool(refs))
        self._face_results_context_actions = {
            "name": name_action,
            "show_photo": show_photo_action,
            "inspect": inspector_action,
        }
        return menu

    def _show_face_results_context_menu(self, point) -> None:
        view = getattr(self, "face_results_list", None)
        if view is None or self._prepare_face_result_context_selection(point) is None:
            return
        menu = self._build_face_results_context_menu()
        if not menu.actions():
            return
        previous_menu = self._face_results_context_menu
        if previous_menu is not None:
            previous_menu.close()
            previous_menu.deleteLater()
        self._face_results_context_menu = menu

        def _discard_menu() -> None:
            if self._face_results_context_menu is menu:
                self._face_results_context_menu = None
            menu.deleteLater()

        menu.aboutToHide.connect(_discard_menu)
        menu.popup(view.viewport().mapToGlobal(point))

    def _install_face_result_group_hover_preview(self, view: QListView | None, group_kind: str) -> None:
        if view is None:
            return
        if self._face_result_hover_popup is None:
            self._face_result_hover_popup = FaceResultHoverPreviewPopup(self)
            self._face_result_hover_popup.hide()
        viewport = view.viewport()
        viewport.setMouseTracking(True)
        viewport.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        viewport.setProperty("faceResultGroupKind", str(group_kind))
        viewport.installEventFilter(self)
        view.setMouseTracking(True)
        view.verticalScrollBar().valueChanged.connect(lambda _value: self._hide_face_result_hover_popup())
        view.horizontalScrollBar().valueChanged.connect(lambda _value: self._hide_face_result_hover_popup())

    def eventFilter(self, watched, event):  # noqa: N802 - Qt API override
        group_kind = str(getattr(watched, "property", lambda _name: "")("faceResultGroupKind") or "")
        if group_kind in {"raw", "merged_name"}:
            view, _model = self._face_result_list_and_model_for_kind(group_kind)
            if view is not None and watched is view.viewport():
                event_type = event.type()
                if event_type == QEvent.Type.ToolTip:
                    return True
                if event_type == QEvent.Type.MouseMove:
                    self._update_face_result_hover_popup_for_index(group_kind, view, view.indexAt(event.pos()))
                elif event_type in {
                    QEvent.Type.Hide,
                    QEvent.Type.Leave,
                    QEvent.Type.MouseButtonPress,
                    QEvent.Type.Wheel,
                }:
                    self._hide_face_result_hover_popup()
        return super().eventFilter(watched, event)

    def _update_face_result_hover_popup_for_index(self, group_kind: str, view: QListView, index: QModelIndex) -> None:
        popup = self._face_result_hover_popup
        if popup is None or not index.isValid():
            self._hide_face_result_hover_popup()
            return
        group_id = str(index.data(ListEntryModel.PayloadRole) or "").strip()
        if not group_id:
            self._hide_face_result_hover_popup()
            return
        group = self._face_result_group_map_for_kind(group_kind).get(group_id)
        if group is None:
            self._hide_face_result_hover_popup()
            return
        if (
            popup.isVisible()
            and self._face_result_hover_group_kind == group_kind
            and self._face_result_hover_group_id == group_id
            and self._face_result_hover_view is view
        ):
            self._position_face_result_hover_popup(view, index)
            return
        self._face_result_hover_group_kind = str(group_kind)
        self._face_result_hover_group_id = group_id
        self._face_result_hover_view = view
        popup.set_group_details(group, more_count=max(0, len(group.items) - FACE_RESULT_HOVER_PREVIEW_MAX_ITEMS), loading=False)
        popup.set_preview_image(self._render_face_result_hover_preview(group))
        self._position_face_result_hover_popup(view, index)
        popup.show()

    def _render_face_result_hover_preview(self, group: FaceResultGroup | None) -> QImage | None:
        if group is None or not group.items:
            return None
        items = list(group.items[:FACE_RESULT_HOVER_PREVIEW_MAX_ITEMS])
        columns = FACE_RESULT_HOVER_PREVIEW_COLUMNS
        rows = max(1, min(2, (len(items) + columns - 1) // columns))
        tile_size = QSize(FACE_RESULT_HOVER_PREVIEW_TILE_SIZE)
        width = (columns * tile_size.width()) + ((columns - 1) * 6)
        height = (rows * tile_size.height()) + ((rows - 1) * 6)
        canvas = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
        canvas.fill(QColor("#E2E8F0"))
        painter = QPainter(canvas)
        for slot, item in enumerate(items):
            row, column = divmod(slot, columns)
            x = column * (tile_size.width() + 6)
            y = row * (tile_size.height() + 6)
            key = self._face_tile_cache_key_for_item(item, tile_size)
            image = self._face_tile_placeholder_image(tile_size)
            if key is not None:
                cached = self._face_thumb_cache.get(key)
                if cached is None:
                    self._queue_face_tile_load(item, tile_size, priority=slot)
                else:
                    image = cached
            painter.drawImage(QRect(x, y, tile_size.width(), tile_size.height()), image)
            painter.setPen(QColor("#CBD5E1"))
            painter.drawRect(x, y, max(1, tile_size.width() - 1), max(1, tile_size.height() - 1))
        painter.end()
        return canvas

    def _refresh_face_result_hover_popup_if_needed(self) -> None:
        popup = self._face_result_hover_popup
        if popup is None or not popup.isVisible():
            return
        group = self._face_result_group_map_for_kind(self._face_result_hover_group_kind).get(self._face_result_hover_group_id)
        if group is None:
            self._hide_face_result_hover_popup()
            return
        popup.set_group_details(group, more_count=max(0, len(group.items) - FACE_RESULT_HOVER_PREVIEW_MAX_ITEMS), loading=False)
        popup.set_preview_image(self._render_face_result_hover_preview(group))

    def _position_face_result_hover_popup(self, view: QListView, index: QModelIndex) -> None:
        popup = self._face_result_hover_popup
        if popup is None or not index.isValid():
            return
        rect = view.visualRect(index)
        if not rect.isValid():
            return
        anchor = view.viewport().mapToGlobal(rect.topRight())
        popup.adjustSize()
        popup_rect = popup.frameGeometry()
        screen = view.screen() or QGuiApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else popup_rect
        margin = 12
        target_x = anchor.x() + margin
        target_y = anchor.y() + margin
        if target_x + popup_rect.width() > available.right() - margin:
            target_x = view.viewport().mapToGlobal(rect.topLeft()).x() - popup_rect.width() - margin
        if target_y + popup_rect.height() > available.bottom() - margin:
            target_y = max(available.top() + margin, available.bottom() - popup_rect.height() - margin)
        target_x = max(available.left() + margin, target_x)
        target_y = max(available.top() + margin, target_y)
        popup.move(target_x, target_y)

    def _hide_face_result_hover_popup(self) -> None:
        popup = self._face_result_hover_popup
        if popup is not None:
            popup.hide()
        self._face_result_hover_group_kind = ""
        self._face_result_hover_group_id = ""
        self._face_result_hover_view = None

    def _prepare_face_result_group_context_selection(self, point) -> QModelIndex | None:
        view = getattr(self, "face_results_groups_list", None)
        if view is None:
            return None
        index = view.indexAt(point)
        if not index.isValid():
            return None
        selection_model = view.selectionModel()
        selected_rows = set()
        if selection_model is not None:
            selected_rows = {selected_index.row() for selected_index in selection_model.selectedIndexes()}
        if selection_model is not None:
            if index.row() not in selected_rows:
                selection_model.clearSelection()
                selection_model.select(index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
                selection_model.setCurrentIndex(index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
            else:
                selection_model.setCurrentIndex(index, QItemSelectionModel.SelectionFlag.NoUpdate)
        else:
            view.setCurrentIndex(index)
        return index

    def _resolve_face_result_group_context_name(self, group: FaceResultGroup | None, *, cluster_count: int = 1) -> str:
        suggested = ""
        if group is not None and group.suggestion is not None:
            suggested = str(group.suggestion.person_name or "").strip()
        preferred = suggested or self._current_face_identity_name()
        value, accepted = QInputDialog.getText(
            self,
            "Name Selected Clusters" if int(cluster_count) > 1 else "Name Cluster",
            "Saved identity name",
            text=str(preferred or ""),
        )
        if not accepted:
            return ""
        person_name = str(value or "").strip()
        if person_name:
            self._set_face_identity_name_fields(person_name)
        return person_name

    def _show_face_result_group_context_menu(self, point) -> None:
        view = getattr(self, "face_results_groups_list", None)
        if view is None or self._prepare_face_result_group_context_selection(point) is None:
            return
        group = self._current_face_result_raw_group()
        if group is None or not group.items:
            return
        selected_groups = self._selected_face_result_raw_groups()
        menu = QMenu(self)
        name_action = menu.addAction("Name Selected Clusters..." if len(selected_groups) > 1 else "Name Cluster...")
        name_action.triggered.connect(self._name_selected_face_result_groups_immediately)
        name_action.setEnabled(not self._read_only_mode)
        menu.exec(view.viewport().mapToGlobal(point))

    def _name_selected_face_result_group_immediately(self) -> None:
        self._name_selected_face_result_groups_immediately()

    def _name_selected_face_result_groups_immediately(self) -> None:
        groups = self._selected_face_result_raw_groups()
        if not groups:
            group = self._current_face_result_raw_group()
            if group is not None:
                groups = [group]
        if not groups:
            errorBox("Select raw clusters", "Select one or more raw clusters first.")
            return
        prompt_group = self._current_face_result_raw_group()
        if prompt_group is None or prompt_group not in groups:
            prompt_group = groups[0]
        person_name = self._resolve_face_result_group_context_name(prompt_group, cluster_count=len(groups))
        if not person_name:
            return
        self._name_face_result_groups_immediately(groups, person_name_override=person_name)

    def _name_face_result_group_immediately(
        self,
        group: FaceResultGroup | None,
        *,
        person_name_override: str,
    ) -> None:
        raw_group = group if group is not None else self._current_face_result_raw_group()
        if raw_group is None:
            errorBox("Select a raw cluster", "Choose a raw cluster first.")
            return
        self._name_face_result_groups_immediately([raw_group], person_name_override=person_name_override)

    def _name_face_result_groups_immediately(
        self,
        groups: list[FaceResultGroup] | tuple[FaceResultGroup, ...],
        *,
        person_name_override: str,
    ) -> None:
        raw_groups = [group for group in list(groups or []) if group is not None and group.review_kind == "raw"]
        if not raw_groups:
            errorBox("Select raw clusters", "Choose one or more raw clusters first.")
            return
        refs = list(
            dict.fromkeys(
                ref
                for group in raw_groups
                for ref in self._face_result_group_refs(group, prefer_selection=False)
            )
        )
        if not refs:
            errorBox(
                "No saved faces",
                "The selected clusters do not contain any saved indexed faces."
                if len(raw_groups) > 1
                else "The selected cluster does not contain any saved indexed faces.",
            )
            return
        person_name = str(person_name_override or "").strip()
        if not person_name:
            errorBox("Missing name", "Enter a saved identity name first.")
            return
        threshold = float(getattr(self, "face_library_label_threshold", self.face_label_threshold).value())
        cluster_count = len(raw_groups)
        photo_paths = list(dict.fromkeys(str(image_path) for image_path, _face_index in refs if str(image_path or "").strip()))
        action_service = getattr(self.results_gallery, "action_service", None) or GalleryActionService()

        def _run(progress, cancel_check):
            progress(-1, "Naming clusters and writing EXIF..." if cluster_count > 1 else "Naming cluster and writing EXIF...")
            person = self._active_face_service().label_indexed_faces_immediately(
                person_name,
                refs,
                similarity_threshold=threshold,
            )
            exif_result = action_service.write_exif_metadata_pairs(
                photo_paths,
                FACE_EXIF_PERSON_KEY,
                str(person.person_name),
                progress_callback=progress,
                cancel_check=cancel_check,
            )
            return person, exif_result

        def _done(result) -> None:
            person, exif_result = result
            self._set_face_identity_name_fields(str(person.person_name))
            tags = [tag.strip() for tag in self.face_profile_tags.text().split(",") if tag.strip()] if hasattr(self, "face_profile_tags") else []
            cover_ref = refs[0] if refs else None
            self._active_face_service().save_person_profile(
                str(person.person_name),
                notes=self.face_profile_notes.text().strip() if hasattr(self, "face_profile_notes") else "",
                tags=tags,
                cover_face_ref=cover_ref,
            )
            affected = len(getattr(exif_result, "affected_paths", []) or [])
            failures = len(getattr(exif_result, "failures", []) or [])
            failure_suffix = f" ({failures} EXIF failure(s))" if failures else ""
            if cluster_count == 1:
                status_text = (
                    f"Named cluster '{person.person_name}' on {len(refs)} face(s) and wrote EXIF to {affected} photo(s){failure_suffix}."
                )
            else:
                status_text = (
                    f"Named {cluster_count} clusters as '{person.person_name}' on {len(refs)} face(s) "
                    f"and wrote EXIF to {affected} photo(s){failure_suffix}."
                )
            self.status_label.setText(status_text)
            self._refresh_pending_face_labels()
            self.refresh_face_library(reason="face cluster named")
            self._maybe_refresh_global_face_album(reason="face cluster named")
            self._refresh_face_identities(force_reload=True)
            self._apply_face_result_filters()

        self._start_job("Naming cluster", _run, _done)

    def _update_face_result_actions(self) -> None:
        selected = self._selected_face_result_tiles()
        has_selection = bool(selected)
        self._set_guarded_action_enabled(self.face_results_name_button, has_selection)
        self._set_guarded_action_enabled(self.face_results_jump_button, has_selection)
        self._set_guarded_action_enabled(self.face_results_edit_button, has_selection)
        raw_group = self._current_face_result_raw_group()
        current_group = self._current_face_result_group()
        current_group_paths = list(
            self._face_result_photo_paths_for_kind(self._face_result_active_group_kind).get(
                self._face_result_selected_group_id,
                [],
            )
        )
        self._set_guarded_action_enabled(
            self.face_results_show_group_photos_button,
            bool(current_group is not None and current_group_paths),
        )
        can_recluster = bool(raw_group is not None and raw_group.cluster_id >= -1 and raw_group.items)
        can_queue_suggestion = bool(raw_group is not None and raw_group.suggestion is not None and raw_group.items)
        can_split = bool(raw_group is not None and len(self._face_result_group_refs(raw_group, prefer_selection=True)) >= 1 and len(raw_group.items) >= 2)
        selected_group_ids = self._selected_face_result_group_ids()
        self._set_guarded_action_enabled(self.face_results_name_clusters_button, bool(selected_group_ids))
        merge_groups = [self._face_result_group_by_id.get(group_id) for group_id in selected_group_ids]
        merge_groups = [group for group in merge_groups if group is not None]
        can_merge = (
            len(merge_groups) == 2
            and all(group.comparison_key for group in merge_groups)
            and len({group.comparison_key for group in merge_groups}) == 1
        )
        self._set_guarded_action_enabled(self.face_results_queue_suggestion_button, can_queue_suggestion)
        self._set_guarded_action_enabled(self.face_results_reject_suggestion_button, can_queue_suggestion)
        self._set_guarded_action_enabled(
            self.face_results_keep_unknown_button,
            bool(raw_group is not None and raw_group.cluster_id >= -1 and raw_group.items),
        )
        self._set_guarded_action_enabled(self.face_results_split_button, can_split)
        self._set_guarded_action_enabled(self.face_results_merge_button, bool(can_merge))
        self._set_guarded_action_enabled(self.face_results_recluster_button, can_recluster)
        self._set_guarded_action_enabled(
            self.face_results_compare_again_button,
            bool(self._face_cluster_compare_last_refs) and self._face_result_active_group_kind == "raw",
        )
        self._set_guarded_action_enabled(self.face_results_review_pending_button, True)

    def _clear_face_cluster_result_details(self) -> None:
        if hasattr(self, "face_results_cluster_summary"):
            self.face_results_cluster_summary.setText("Select a face cluster group to inspect members and quality.")
        if hasattr(self, "face_results_cluster_explanation"):
            self.face_results_cluster_explanation.setText("")
        if hasattr(self, "face_results_cluster_suggestion"):
            self.face_results_cluster_suggestion.setText("")
        if hasattr(self, "face_results_membership_table"):
            self.face_results_membership_table.setRowCount(0)

    def _set_face_cluster_result_details(self, group: FaceResultGroup | None) -> None:
        if group is None:
            self._clear_face_cluster_result_details()
            return
        if group.review_kind == "merged_name":
            self.face_results_cluster_summary.setText(
                f"Named Cluster | {group.resolved_name or 'Unnamed'} | {len(group.items)} face(s)"
            )
            self.face_results_cluster_explanation.setText(str(group.merge_notice or "Grouped by current saved or pending face labels."))
            self.face_results_cluster_suggestion.setText("Named clusters update automatically when raw clusters are labeled.")
            return
        if group.review_kind == "exif_name":
            self.face_results_cluster_summary.setText(
                f"EXIF:NAME | {group.exif_name or 'No EXIF name'} | {len(group.items)} face(s)"
            )
            self.face_results_cluster_explanation.setText(
                f"Photo-level EXIF review bucket using {FACE_EXIF_PERSON_KEY}."
            )
            self.face_results_cluster_suggestion.setText("EXIF names are informational until you explicitly write or rename them.")
            return
        if group.cluster_id < -1 or not group.comparison_key:
            self._clear_face_cluster_result_details()
            return
        outlier = "outlier/noise" if int(group.cluster_id) == -1 else f"cluster {int(group.cluster_id)}"
        status_text = f" | {group.cluster_status}" if str(group.cluster_status or "").strip() else ""
        self.face_results_cluster_summary.setText(
            f"{group.title} | {outlier} | {len(group.items)} face(s){status_text}"
        )
        explanation = group.explanation
        if explanation is None:
            self.face_results_cluster_explanation.setText("Cluster explanation unavailable for this result.")
        else:
            parts = [
                f"Quality {explanation.cluster_quality_score:.3f}" if explanation.cluster_quality_score is not None else "Quality unavailable",
                f"mean cohesion {explanation.cohesion_mean:.3f}" if explanation.cohesion_mean is not None else "mean cohesion unavailable",
                f"weakest member {explanation.cohesion_min:.3f}" if explanation.cohesion_min is not None else "weakest member unavailable",
            ]
            if explanation.nearest_cluster_id is not None:
                parts.append(f"nearest cluster {int(explanation.nearest_cluster_id)}")
            if explanation.separation_margin is not None:
                parts.append(f"margin {float(explanation.separation_margin):.3f}")
            self.face_results_cluster_explanation.setText(" | ".join(parts))
        suggestion = group.suggestion
        if suggestion is None:
            if str(group.cluster_status or "").strip() == "suggestion dismissed":
                self.face_results_cluster_suggestion.setText("Saved-identity suggestion dismissed for this cluster.")
            elif str(group.cluster_status or "").strip() == "kept unlabeled":
                self.face_results_cluster_suggestion.setText("This cluster is intentionally kept unlabeled in the current review session.")
            else:
                self.face_results_cluster_suggestion.setText("No saved-identity suggestion for this cluster yet.")
        else:
            self.face_results_cluster_suggestion.setText(
                f"Likely identity: {suggestion.person_name} | support {suggestion.support_count}/{suggestion.member_count} | "
                f"mean {suggestion.mean_score:.3f} | range {suggestion.min_score:.3f}-{suggestion.max_score:.3f}"
            )

    def _update_face_result_membership_table(self) -> None:
        table = getattr(self, "face_results_membership_table", None)
        if table is None:
            return
        rows: list[dict[str, object]] = []
        current_group = self._current_face_result_group()
        if self._face_result_active_group_kind == "raw":
            for image_path, face_index in self._selected_face_result_tiles():
                rows.extend(
                    dict(info)
                    for _key, info in sorted(
                        (self._face_cluster_compare_membership.get((str(image_path), int(face_index))) or {}).items(),
                        key=lambda item: str(item[0]),
                    )
                )
                if rows:
                    break
        elif current_group is not None:
            for source_group_id in list(current_group.source_group_ids or ()):
                source_group = self._face_result_group_by_id.get(str(source_group_id))
                if source_group is None:
                    continue
                rows.append(
                    {
                        "backend": str(source_group.comparison_key or ""),
                        "cluster_id": int(source_group.cluster_id),
                        "cluster_size": len(source_group.items),
                        "rank": "",
                        "outlier": int(source_group.cluster_id) == -1,
                        "cluster_quality_score": (
                            None
                            if source_group.explanation is None
                            else source_group.explanation.cluster_quality_score
                        ),
                    }
                )
        table.setRowCount(len(rows))
        for row, info in enumerate(rows):
            values = [
                str(info.get("backend", "")),
                str(info.get("cluster_id", "")),
                str(info.get("cluster_size", "")),
                str(info.get("rank", "")),
                "yes" if info.get("outlier") else "no",
                "" if info.get("cluster_quality_score") in {None, ""} else f"{float(info.get('cluster_quality_score')):.3f}",
            ]
            for column, value in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(value))
        table.resizeColumnsToContents()

    def _on_face_result_selection_changed(self) -> None:
        self._update_face_result_actions()
        self._update_face_result_membership_table()

    def _count_detected_visible_scope_refs(self, review_images: list[FaceFolderReviewImage]) -> int:
        total = 0
        dirty_paths = set(self._face_review_draft_dirty_paths or ())
        for item in list(review_images or []):
            image_path = str(getattr(item, "image_path", "") or "")
            if not image_path or image_path in dirty_paths:
                continue
            total += len(tuple(getattr(item, "visible_faces", ()) or ()))
        return total

    def _detected_faces_review_summary_text(
        self,
        *,
        face_count: int,
        reviewed_photo_count: int,
        loading: bool = False,
        processed_images: int = 0,
        target_image_count: int = 0,
        unavailable_tile_count: int = 0,
    ) -> str:
        if int(face_count) <= 0 and not loading:
            return "Detected Faces is empty for the current folder review."
        suffix = "Delete tiles here to remove the corresponding image boxes from the current draft."
        unavailable_text = ""
        if int(unavailable_tile_count) > 0:
            unavailable_text = f" Unavailable: {max(0, int(unavailable_tile_count))} tile(s)."
        if loading:
            return (
                f"Detected Faces is publishing {max(0, int(face_count))} face tile(s) "
                f"from {max(0, int(processed_images))} of {max(0, int(target_image_count))} reviewed photo(s). "
                f"{suffix}{unavailable_text}"
            )
        return (
            f"Detected Faces is showing {max(0, int(face_count))} face tile(s) "
            f"from {max(0, int(reviewed_photo_count))} reviewed photo(s). {suffix}{unavailable_text}"
        )

    def _detected_face_unavailable_tile_count(self) -> int:
        model = getattr(self, "face_detected_faces_model", None)
        if model is None:
            return 0
        rows: set[int] = set()
        for key in list(self._face_tile_failed_keys):
            if not self._is_face_tile_cache_key_current(key):
                continue
            rows.update(int(row) for row in model.rows_for_cache_key(key) if int(row) >= 0)
        return len(rows)

    def _refresh_detected_faces_summary(self) -> None:
        if not hasattr(self, "face_detected_faces_summary") or not hasattr(self, "face_detected_faces_model"):
            return
        unavailable_tile_count = self._detected_face_unavailable_tile_count()
        if self._face_detected_publish_in_progress:
            self.face_detected_faces_summary.setText(
                self._detected_faces_review_summary_text(
                    face_count=int(self._face_detected_publish_total_faces or 0),
                    reviewed_photo_count=len(list(self._face_review_images or [])),
                    loading=True,
                    processed_images=int(self._face_detected_publish_processed_images or 0),
                    target_image_count=int(self._face_detected_publish_target_image_count or 0),
                    unavailable_tile_count=unavailable_tile_count,
                )
            )
            return
        self.face_detected_faces_summary.setText(
            self._detected_faces_review_summary_text(
                face_count=int(self.face_detected_faces_model.rowCount()),
                reviewed_photo_count=len(list(self._face_review_images or [])),
                unavailable_tile_count=unavailable_tile_count,
            )
        )

    def _refresh_detected_faces_summary_for_key(self, cache_key: tuple[object, ...]) -> None:
        model = getattr(self, "face_detected_faces_model", None)
        if model is None or not model.rows_for_cache_key(cache_key):
            return
        self._refresh_detected_faces_summary()

    def _build_detected_face_review_items_for_image(self, item: FaceFolderReviewImage) -> list[FaceTileItem]:
        faces = self._review_faces_for_item(item)
        if not faces:
            return []
        image_path = str(item.image_path or "")
        is_dirty = image_path in self._face_review_draft_dirty_paths
        visible_faces = list(tuple(item.visible_faces or ()))
        face_items: list[FaceTileItem] = []
        for index, draft in enumerate(faces):
            saved_record = None
            if not is_dirty and index < len(visible_faces):
                saved_record = visible_faces[index]
            quality_status, quality_reasons = self._face_quality_status(
                image_path,
                tuple(int(value) for value in draft.bbox),
                confidence=float(draft.confidence or 1.0),
                saved_record=saved_record,
            )
            label = draft.person_name or "Unlabeled"
            source_text = "draft" if is_dirty else str(draft.source or "indexed")
            face_items.append(
                FaceTileItem(
                    image_path=image_path,
                    face_index=int(index),
                    bbox=tuple(int(value) for value in draft.bbox),
                    title=f"{label}{' [Review]' if quality_status == 'review' else ''}\n{Path(image_path).name}",
                    tooltip=(
                        f"{image_path}\nface #{index + 1}\n"
                        f"bbox={tuple(int(value) for value in draft.bbox)}\n"
                        f"face_conf={float(draft.confidence):.3f}\nsource={source_text}\n"
                        f"quality={quality_status}"
                        + (f"\nreasons={', '.join(quality_reasons)}" if quality_reasons else "")
                    ),
                    status="draft" if is_dirty else str(quality_status or "saved"),
                    draft_slot=int(index),
                    saved_face_index=int(saved_record.face_index) if saved_record is not None else int(draft.face_index),
                    payload=saved_record if saved_record is not None else draft,
                )
            )
        return face_items

    def _cancel_detected_faces_publish(self) -> None:
        was_in_progress = bool(self._face_detected_publish_in_progress)
        published_faces = int(getattr(self.face_detected_faces_model, "rowCount", lambda: 0)())
        processed_images = int(self._face_detected_publish_processed_images or 0)
        target_image_count = int(self._face_detected_publish_target_image_count or 0)
        self._face_detected_publish_timer.stop()
        self._face_detected_publish_active_request_id = 0
        self._face_detected_publish_pending_review_images = []
        self._face_detected_publish_total_faces = 0
        self._face_detected_publish_processed_images = 0
        self._face_detected_publish_target_image_count = 0
        self._face_detected_publish_in_progress = False
        if was_in_progress:
            self._log_face_event(
                "detected_faces_publish_cancelled",
                published_faces=published_faces,
                processed_images=processed_images,
                target_images=target_image_count,
            )

    def _finalize_detected_faces_publish(self) -> None:
        if not hasattr(self, "face_detected_faces_model"):
            return
        face_count = int(self.face_detected_faces_model.rowCount())
        reviewed_photo_count = len(list(self._face_review_images or []))
        self._face_detected_publish_timer.stop()
        self._face_detected_publish_active_request_id = 0
        self._face_detected_publish_pending_review_images = []
        self._face_detected_publish_total_faces = face_count
        self._face_detected_publish_processed_images = int(self._face_detected_publish_target_image_count or 0)
        self._face_detected_publish_target_image_count = 0
        self._face_detected_publish_in_progress = False
        self._refresh_detected_faces_summary()
        self._update_detected_face_actions()
        self._schedule_detected_face_tile_loads()
        self._log_face_event(
            "detected_faces_publish_complete",
            face_count=face_count,
            image_count=reviewed_photo_count,
        )

    def _flush_detected_faces_publish_batch(self) -> None:
        if not self._face_detected_publish_in_progress:
            return
        if self.face_review_results_tabs.currentWidget() is not self.face_detected_faces_panel:
            self._cancel_detected_faces_publish()
            return
        pending_review_images = list(self._face_detected_publish_pending_review_images or [])
        target_image_count = int(self._face_detected_publish_target_image_count or len(pending_review_images))
        processed_images = int(self._face_detected_publish_processed_images or 0)
        if not pending_review_images or processed_images >= target_image_count:
            self._finalize_detected_faces_publish()
            return

        batch_items: list[FaceTileItem] = []
        batch_image_count = 0
        batch_units = 0
        start_row = int(self.face_detected_faces_model.rowCount())
        slice_deadline = perf_counter() + (FACE_DETECTED_FACE_PUBLISH_SLICE_BUDGET_MS / 1000.0)
        while processed_images < target_image_count:
            item = pending_review_images[processed_images]
            image_path = str(getattr(item, "image_path", "") or "")
            image_items = self._build_detected_face_review_items_for_image(item)
            if image_items and image_path:
                item_start_row = start_row + len(batch_items)
                self._face_detected_face_items_by_image[image_path] = list(
                    range(item_start_row, item_start_row + len(image_items))
                )
                batch_items.extend(image_items)
            batch_image_count += 1
            batch_units += max(1, len(image_items))
            processed_images += 1
            if batch_units >= FACE_DETECTED_FACE_PUBLISH_BATCH_SIZE:
                break
            if batch_image_count > 0 and perf_counter() >= slice_deadline:
                break

        self._face_detected_publish_processed_images = processed_images
        if batch_items:
            self.face_detected_faces_model.append_items(batch_items)
        self._face_detected_publish_total_faces = int(self.face_detected_faces_model.rowCount())
        self._refresh_detected_faces_summary()
        self._update_detected_face_actions()
        if batch_items:
            self._schedule_detected_face_tile_loads()
        self._log_face_event(
            "detected_faces_publish_batch",
            batch_faces=len(batch_items),
            batch_images=batch_image_count,
            published_faces=int(self._face_detected_publish_total_faces or 0),
            processed_images=processed_images,
            target_images=target_image_count,
        )
        if processed_images >= target_image_count:
            self._finalize_detected_faces_publish()
            return
        self._face_detected_publish_timer.start(FACE_DETECTED_FACE_PUBLISH_INTERVAL_MS)

    def _refresh_detected_faces_review(self, *, force: bool = False) -> None:
        if not hasattr(self, "face_detected_faces_list"):
            return
        self._clear_face_tile_failures()
        self._face_detected_visible_scope_ref_count = self._count_detected_visible_scope_refs(
            list(self._face_review_images or [])
        )
        if not self._face_review_images:
            self._cancel_detected_faces_publish()
            self._face_detected_face_items_by_image = {}
            self.face_detected_faces_model.set_items([])
            self.face_detected_faces_summary.setText("Detected Faces is empty for the current folder review.")
            self._update_detected_face_actions()
            return
        if (
            not force
            and len(self._face_review_images) > 256
            and self.face_review_results_tabs.currentWidget() is not self.face_detected_faces_panel
        ):
            self._cancel_detected_faces_publish()
            self._face_detected_face_items_by_image = {}
            self.face_detected_faces_model.set_items([])
            self.face_detected_faces_summary.setText(
                f"Detected Faces is deferred for large reviews. Open this tab to load faces from {len(self._face_review_images)} photo(s)."
            )
            self._update_detected_face_actions()
            return
        sorted_review_images = self._ordered_face_review_images()
        self._cancel_detected_faces_publish()
        if len(sorted_review_images) > FACE_REVIEW_LAZY_PUBLISH_THRESHOLD:
            self._face_detected_publish_request_id += 1
            self._face_detected_publish_active_request_id = int(self._face_detected_publish_request_id)
            self._face_detected_publish_pending_review_images = list(sorted_review_images)
            self._face_detected_publish_total_faces = 0
            self._face_detected_publish_processed_images = 0
            self._face_detected_publish_target_image_count = len(sorted_review_images)
            self._face_detected_publish_in_progress = True
            self._face_detected_face_items_by_image = {}
            self.face_detected_faces_model.set_items([])
            self._refresh_detected_faces_summary()
            self._update_detected_face_actions()
            self._log_face_event(
                "detected_faces_publish_start",
                image_count=len(sorted_review_images),
                request_id=int(self._face_detected_publish_active_request_id),
            )
            self._flush_detected_faces_publish_batch()
            return
        self._face_detected_face_items_by_image = {}
        face_items: list[FaceTileItem] = []
        total_faces = 0
        for item in sorted_review_images:
            image_path = str(item.image_path or "")
            image_face_items = self._build_detected_face_review_items_for_image(item)
            if not image_face_items:
                continue
            start_row = len(face_items)
            face_items.extend(image_face_items)
            total_faces += len(image_face_items)
            if image_path:
                self._face_detected_face_items_by_image[image_path] = list(
                    range(start_row, start_row + len(image_face_items))
                )
        self.face_detected_faces_model.set_items(face_items)
        self._refresh_detected_faces_summary()
        self._update_detected_face_actions()
        self._schedule_detected_face_tile_loads()

    def _sync_face_review_result_groups(self, *, force: bool = False) -> None:
        source_kind = str(self._face_result_source_kind or "").strip()
        if source_kind not in {"", "faces_review"}:
            return
        # Folder review already has a dedicated Faces view.  Mirroring every
        # face into a synthetic "Current Review" group duplicates navigation
        # and, for large folders, needlessly builds the same tile set twice.
        self._face_result_source_kind = "faces_review"
        self._face_result_source_groups = []
        self._face_result_groups = []
        self._face_result_group_by_id = {}
        self._face_result_photo_paths_by_group_id = {}
        self._face_result_merged_groups = []
        self._face_result_merged_group_by_id = {}
        self._face_result_merged_photo_paths_by_group_id = {}
        self._face_result_exif_groups = []
        self._face_result_exif_group_by_id = {}
        self._face_result_exif_photo_paths_by_group_id = {}
        self.face_results_groups_model.set_items([])
        self.face_results_merged_groups_model.set_items([])
        exif_model = getattr(self, "face_results_exif_groups_model", None)
        if exif_model is not None:
            exif_model.set_items([])
        self.face_results_model.set_items([])
        self.face_results_groups_label.setText("Result Groups")
        self.face_results_summary.setText(
            "No groups or matches yet. Run face search or clustering; folder faces stay in the Faces view."
        )
        self._update_face_result_actions()
        self._update_face_results_context()
        return
        if (
            not force
            and len(self._face_review_images) > 256
            and self.face_review_results_tabs.currentWidget() is not self.face_results_panel
        ):
            self._face_result_source_kind = "faces_review"
            self._face_result_source_groups = []
            self._face_result_groups = []
            self._face_result_group_by_id = {}
            self._face_result_photo_paths_by_group_id = {}
            self.face_results_groups_model.set_items([])
            self.face_results_summary.setText(
                f"Face Results is deferred for large reviews. Open this tab to load faces from {len(self._face_review_images)} photo(s)."
            )
            return
        items: list[FaceTileItem] = []
        photo_paths: list[str] = []
        overlay_by_path: dict[str, str] = {}
        context_by_path = dict(self._face_review_context_by_path or {})
        for item in self._ordered_face_review_images():
            faces = self._review_faces_for_item(item)
            if not faces:
                continue
            photo_paths.append(str(item.image_path))
            overlay_by_path[str(item.image_path)] = f"{len(faces)} face{'s' if len(faces) != 1 else ''}"
            for index, draft in enumerate(faces):
                saved_record = None
                if item.image_path not in self._face_review_draft_dirty_paths and index < len(item.visible_faces):
                    saved_record = item.visible_faces[index]
                quality_status, quality_reasons = self._face_quality_status(
                    item.image_path,
                    tuple(int(value) for value in draft.bbox),
                    confidence=float(draft.confidence or 1.0),
                    saved_record=saved_record,
                )
                label = draft.person_name or (str(saved_record.person_name) if saved_record is not None else "") or "Unlabeled"
                items.append(
                    FaceTileItem(
                        image_path=str(item.image_path),
                        face_index=int(index),
                        bbox=tuple(int(value) for value in draft.bbox),
                        title=f"{label}{' [Review]' if quality_status == 'review' else ''}\n{Path(item.image_path).name}",
                        tooltip=(
                            f"{item.image_path}\nface #{index + 1}\n"
                            f"bbox={tuple(int(value) for value in draft.bbox)}\n"
                            f"face_conf={float(draft.confidence):.3f}\n"
                            f"quality={quality_status}"
                            + (f"\nreasons={', '.join(quality_reasons)}" if quality_reasons else "")
                        ),
                        status="draft" if item.image_path in self._face_review_draft_dirty_paths else str(quality_status or "saved"),
                        draft_slot=int(index),
                        saved_face_index=int(saved_record.face_index) if saved_record is not None else int(draft.face_index),
                        payload=saved_record if saved_record is not None else draft,
                    )
                )
        photo_paths = list(dict.fromkeys(photo_paths))
        self._face_result_source_kind = "faces_review"
        self._face_result_source_overlay_by_path = overlay_by_path
        self._face_result_source_context_by_path = context_by_path
        self._face_result_source_photo_paths = list(photo_paths)
        self._face_result_merged_groups = []
        self._face_result_merged_group_by_id = {}
        self._face_result_merged_photo_paths_by_group_id = {}
        self._face_result_exif_groups = []
        self._face_result_exif_group_by_id = {}
        self._face_result_exif_photo_paths_by_group_id = {}
        self.face_results_merged_groups_model.set_items([])
        exif_model = getattr(self, "face_results_exif_groups_model", None)
        if exif_model is not None:
            exif_model.set_items([])
        self.face_results_groups_label.setText("Review Faces")
        if items:
            group_id = f"review:{self._face_review_folder or 'current'}"
            summary = (
                f"Current review: {len(items)} face tile(s) across {len(photo_paths)} photo(s) "
                f"from {self._face_review_folder or 'the current folder'}."
            )
            group = FaceResultGroup(
                group_id=group_id,
                title=f"Current Review\n{len(items)} face(s) | {len(photo_paths)} photo(s)",
                summary=summary,
                items=tuple(items),
                review_kind="review",
            )
            self._face_result_source_groups = [group]
            self._face_result_source_summary = summary
            self._face_result_groups = [group]
            self._face_result_group_by_id = {group_id: group}
            self._face_result_photo_paths_by_group_id = {group_id: list(photo_paths)}
            self.face_results_groups_model.set_items([ListEntry(title=group.title, tooltip=group.summary, payload=group_id)])
            self._face_result_selected_group_id_by_kind["raw"] = group_id
            self._face_result_selected_group_id = group_id
            if self.face_review_results_tabs.currentIndex() == 2:
                self._face_result_active_group_kind = "raw"
                self._sync_face_result_group_view_selection("raw", group_id)
                self._refresh_current_face_result_group()
            else:
                self.face_results_summary.setText(summary)
        else:
            self._face_result_source_groups = []
            self._face_result_source_summary = "Current review has no visible faces."
            self._face_result_groups = []
            self._face_result_group_by_id = {}
            self._face_result_photo_paths_by_group_id = {}
            self.face_results_groups_model.set_items([])
            self._face_result_selected_group_id_by_kind["raw"] = ""
            self._face_result_selected_group_id = ""
            self.face_results_summary.setText(self._face_result_source_summary)

    def _remove_selected_detected_face_tiles(self) -> None:
        refs = self._selected_detected_face_tiles()
        if not refs:
            return
        grouped: dict[str, list[int]] = {}
        for image_path, slot_index in refs:
            grouped.setdefault(str(image_path), []).append(int(slot_index))
        for image_path, slot_indexes in grouped.items():
            drafts = self._ensure_face_review_drafts(image_path)
            for slot_index in sorted(set(slot_indexes), reverse=True):
                if 0 <= int(slot_index) < len(drafts):
                    drafts.pop(int(slot_index))
            self._on_face_review_drafts_updated(image_path, drafts, True)

    def _jump_to_selected_detected_face(self) -> None:
        refs = self._selected_detected_face_tiles()
        if not refs:
            return
        image_path = str(refs[0][0] or "")
        self._show_derived_face_photos(
            [image_path],
            title="Selected face photo",
            focus_path=image_path,
        )

    def _open_face_review_inspector(self, image_path: str) -> bool:
        target = str(image_path or "")
        if not target:
            return False
        try:
            row = list(getattr(self.results_gallery, "images", []) or []).index(target)
        except ValueError:
            return False
        index = self.results_gallery.model.index(row, 0)
        if not index.isValid():
            return False
        self.results_gallery._open_inspector(index, allow_face_edit=True)
        return True

    def _open_selected_detected_face_in_inspector(self) -> None:
        refs = self._selected_detected_face_tiles()
        if not refs:
            return
        image_path = str(refs[0][0] or "")
        self._show_derived_face_photos([image_path], title="Selected face photo", focus_path=image_path)
        QTimer.singleShot(0, lambda path=image_path: self._open_face_review_inspector(path))

    def _focus_gallery_image(self, image_path: str) -> bool:
        target = str(image_path or "")
        if not target:
            return False
        try:
            row = list(getattr(self.results_gallery, "images", []) or []).index(target)
        except ValueError:
            return False
        index = self.results_gallery.model.index(row, 0)
        if not index.isValid():
            return False
        selection_model = self.results_gallery.list_view.selectionModel()
        if selection_model is not None:
            selection_model.select(index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
            selection_model.setCurrentIndex(index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        self.results_gallery.list_view.setCurrentIndex(index)
        self.results_gallery.list_view.scrollTo(index)
        return True

    def _open_results_gallery_inspector(self, image_path: str, *, allow_face_edit: bool = False) -> bool:
        target = str(image_path or "")
        if not target:
            return False
        try:
            row = list(getattr(self.results_gallery, "images", []) or []).index(target)
        except ValueError:
            return False
        index = self.results_gallery.model.index(row, 0)
        if not index.isValid():
            return False
        self.results_gallery._open_inspector(index, allow_face_edit=allow_face_edit)
        return True

    def _show_face_search_tab(self) -> None:
        for index in range(self.tabs.count()):
            try:
                label = str(self.tabs.tabText(index) or "").strip().lower()
            except Exception:
                label = ""
            if label == "face search":
                self.tabs.setCurrentIndex(index)
                return

    def _show_all_faces_tab(self) -> None:
        for index in range(self.tabs.count()):
            try:
                label = str(self.tabs.tabText(index) or "").strip().lower()
            except Exception:
                label = ""
            if self._is_all_faces_tab_label(label):
                self.tabs.setCurrentIndex(index)
                return

    def _show_face_library_tab(self) -> None:
        for index in range(self.tabs.count()):
            try:
                label = str(self.tabs.tabText(index) or "").strip().lower()
            except Exception:
                label = ""
            if self._is_face_folder_tab_label(label):
                self.tabs.setCurrentIndex(index)
                return

    def _show_face_identities_tab(self) -> None:
        for index in range(self.tabs.count()):
            try:
                label = str(self.tabs.tabText(index) or "").strip().lower()
            except Exception:
                label = ""
            if label == "identities":
                self.tabs.setCurrentIndex(index)
                return

    def _selected_identity_prototype_refs(self) -> list[tuple[str, int]]:
        refs: list[tuple[str, int]] = []
        for item in self._selected_tile_items(
            getattr(self, "face_identity_prototype_list", None),
            getattr(self, "face_identity_prototype_model", None),
        ):
            ref = self._face_ref_from_tile_item(item)
            if ref is not None:
                refs.append(ref)
        return refs

    def _refresh_face_identities(
        self,
        *,
        profiles: list[PersonProfile] | None = None,
        force_reload: bool = False,
    ) -> None:
        if not hasattr(self, "face_identity_list"):
            return
        service = self._active_face_service()
        if force_reload:
            self._face_identity_prototypes_by_name = {}
        if profiles is None or force_reload:
            try:
                profiles = service.load_person_profiles(limit=max(1, int(self.face_browser_limit.value() or 1)))
            except Exception:
                profiles = []
        self._face_profiles_by_name.update({str(profile.person_name): profile for profile in list(profiles or [])})
        duplicate_warnings = getattr(service, "identity_duplicate_warnings", None)
        try:
            self._face_identity_duplicates = dict(duplicate_warnings() if callable(duplicate_warnings) else {})
        except Exception:
            self._face_identity_duplicates = {}
        current_name = str(self._face_identity_selected_name or "").strip()
        entries: list[ListEntry] = []
        for profile in list(profiles or []):
            duplicates = self._face_identity_duplicates.get(str(profile.person_name), ())
            suffix = f" | possible duplicate x{len(duplicates)}" if duplicates else ""
            prefix = "[Favorite] " if bool(getattr(profile, "favorite", False)) else ""
            entries.append(
                ListEntry(
                    title=(
                        f"{prefix}{profile.person_name} | examples={profile.example_count} | "
                        f"labeled={profile.labeled_count} | visible={profile.visible_face_count}{suffix}"
                    ),
                    tooltip=f"Saved identity {profile.person_name}",
                    payload=str(profile.person_name),
                )
            )
        self._face_identity_entries = entries
        self._apply_face_identity_filters(preserve_name=current_name)
        self._refresh_selected_face_identity()

    def _request_face_identity_refresh(self) -> None:
        self._face_identity_prototypes_by_name = {}
        service = self._active_face_service()
        limit = max(1, int(self.face_browser_limit.value() or 1))

        def _run(progress, cancel_check):
            progress(-1, "Loading saved identities...")
            if cancel_check():
                raise Cancelled()
            return service.load_person_profiles(limit=limit)

        self._start_job(
            "Loading saved identities",
            _run,
            lambda profiles: self._refresh_face_identities(profiles=list(profiles or [])),
        )

    def _apply_face_identity_filters(self, *, preserve_name: str = "") -> None:
        if not hasattr(self, "face_identity_model"):
            return
        entries = list(getattr(self, "_face_identity_entries", []))
        filter_mode = str(self.face_identity_filter.currentData() or "all") if hasattr(self, "face_identity_filter") else "all"
        if filter_mode == "duplicates":
            entries = [entry for entry in entries if self._face_identity_duplicates.get(str(entry.payload), ())]
        self.face_identity_model.set_source_items(entries)
        self.face_identity_model.set_filter_text(self.face_identity_search.text() if hasattr(self, "face_identity_search") else "")
        self.face_identity_model.set_sort_descending(
            bool(hasattr(self, "face_identity_sort") and self.face_identity_sort.currentData() == "descending")
        )
        total = self.face_identity_model.total_count
        loaded = self.face_identity_model.rowCount()
        if hasattr(self, "face_identity_count_label"):
            self.face_identity_count_label.setText(f"Showing {loaded} of {total} saved identities")
        name = str(preserve_name or self._face_identity_selected_name or "").strip()
        row = self.face_identity_model.row_for_payload(name) if name else (0 if loaded else -1)
        if row >= 0:
            index = self.face_identity_model.index(row, 0)
            self.face_identity_list.setCurrentIndex(index)
            self.face_identity_list.selectionModel().select(
                index,
                QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows,
            )

    def _refresh_selected_face_identity(self) -> None:
        if not hasattr(self, "face_identity_summary"):
            return
        person_name = ""
        if hasattr(self, "face_identity_list") and hasattr(self, "face_identity_model"):
            current = self.face_identity_list.currentIndex()
            if current.isValid():
                person_name = str(current.data(ListEntryModel.PayloadRole) or "").strip()
        self._face_identity_selected_name = person_name
        if not person_name:
            self.face_identity_summary.setText("Select a saved identity to inspect its prototype examples.")
            self.face_identity_duplicate_warning.setVisible(False)
            self.face_identity_prototype_model.set_items([])
            return
        profile = self._face_profiles_by_name.get(person_name)
        if person_name in self._face_identity_prototypes_by_name:
            faces = list(self._face_identity_prototypes_by_name.get(person_name, []))
        else:
            try:
                faces = list(self._active_face_service().load_person_prototype_faces(person_name, limit=200, include_tiny_faces=True))
            except Exception:
                faces = []
            self._face_identity_prototypes_by_name[person_name] = list(faces)
        duplicate_entries = self._face_identity_duplicates.get(person_name, ())
        duplicate_text = ", ".join(f"{other} ({score:.2f})" for other, score in duplicate_entries[:3])
        self.face_identity_duplicate_warning.setVisible(bool(duplicate_text))
        self.face_identity_duplicate_warning.setText(
            f"Possible duplicate identities: {duplicate_text}" if duplicate_text else ""
        )
        profile_bits: list[str] = []
        if bool(getattr(profile, "favorite", False)):
            profile_bits.append("favorite")
        if str(getattr(profile, "birth_date", "") or "").strip():
            profile_bits.append(f"DOB {getattr(profile, 'birth_date', '')}")
            cover_path = str(getattr(profile, "cover_image_path", "") or "")
            age_fn = getattr(self._active_face_service(), "age_at_photo", None)
            if cover_path and callable(age_fn):
                try:
                    age = age_fn(person_name, cover_path)
                except Exception:
                    age = None
                if age is not None:
                    profile_bits.append(f"age at cover {int(age)}")
        profile_text = f" {'; '.join(profile_bits)}." if profile_bits else ""
        self.face_identity_summary.setText(
            f"{person_name}: {len(faces)} prototype face(s), labeled={int(getattr(profile, 'labeled_count', 0) or 0)}, visible={int(getattr(profile, 'visible_face_count', 0) or 0)}.{profile_text}"
        )
        self.face_identity_prototype_model.set_items(
            [
                FaceTileItem(
                    image_path=str(face.image_path),
                    face_index=int(face.face_index),
                    bbox=tuple(face.face_bbox),
                    title=f"{'[Pinned] ' if face.pinned else ''}{Path(str(face.image_path)).name}\nFace #{int(face.face_index) + 1}",
                    tooltip=f"{face.image_path}\nquality={face.quality_status} score={face.quality_score:.2f}",
                    status=str(face.quality_status or "clean"),
                    saved_face_index=int(face.face_index),
                    payload=face,
                )
                for face in faces
            ]
        )
        self._schedule_face_tile_refresh()
        self._update_face_identity_prototype_actions()

    def _update_face_identity_prototype_actions(self) -> None:
        person_name = str(self._face_identity_selected_name or "").strip()
        faces = list(self._face_identity_prototypes_by_name.get(person_name, []))
        can_open = bool(faces)
        has_selected_faces = bool(self._selected_identity_prototype_refs())
        self.face_identity_open_photo_button.setEnabled(has_selected_faces or can_open)
        self.face_identity_open_inspector_button.setEnabled(has_selected_faces or can_open)
        writable = not bool(self._read_only_mode)
        self.face_identity_remove_button.setEnabled(writable and has_selected_faces and len(faces) > 1)
        self.face_identity_pin_button.setEnabled(writable and has_selected_faces)
        self.face_identity_merge_button.setEnabled(writable and bool(person_name))
        self.face_identity_clear_button.setEnabled(writable and bool(person_name))

    def _focus_selected_identity_prototype_photo(self) -> None:
        refs = self._selected_identity_prototype_refs()
        if not refs and self._face_identity_selected_name:
            faces = list(self._face_identity_prototypes_by_name.get(self._face_identity_selected_name, []))
            if faces:
                refs = [(faces[0].image_path, int(faces[0].face_index))]
        if not refs:
            return
        image_path = str(refs[0][0] or "")
        self._show_face_library_tab()
        self.face_review_results_tabs.setCurrentIndex(0)
        self._focus_face_review_image(image_path)

    def _open_selected_identity_prototype_in_inspector(self) -> None:
        refs = self._selected_identity_prototype_refs()
        if not refs and self._face_identity_selected_name:
            faces = list(self._face_identity_prototypes_by_name.get(self._face_identity_selected_name, []))
            if faces:
                refs = [(faces[0].image_path, int(faces[0].face_index))]
        if not refs:
            return
        image_path = str(refs[0][0] or "")
        self._show_face_library_tab()
        self.face_review_results_tabs.setCurrentIndex(0)
        self._focus_face_review_image(image_path)
        self._open_results_gallery_inspector(image_path, allow_face_edit=False)

    def _remove_selected_identity_prototype_faces(self) -> None:
        if not self._ensure_writable_face_action("remove identity prototype examples"):
            return
        person_name = str(self._face_identity_selected_name or "").strip()
        refs = self._selected_identity_prototype_refs()
        if not person_name or not refs:
            errorBox("No prototype selected", "Select one or more prototype faces first.")
            return
        if len(list(self._face_identity_prototypes_by_name.get(person_name, []))) <= len(refs):
            errorBox("Cannot remove all", "Keep at least one prototype example for a saved identity.")
            return
        if not confirmBox(
            "Remove identity examples?",
            (
                f"Target: {person_name}\n"
                f"Prototype examples affected: {len(refs)}\n"
                "Result: the selected faces stop serving as identity examples; their photos and labels remain.\n"
                "Recovery: this change is not automatically recoverable."
            ),
            parent=self,
        ):
            self.status_label.setText("Example removal cancelled. No face data was changed.")
            return
        service = self._active_face_service()
        try:
            for image_path, face_index in refs:
                service.remove_person_prototype_face(person_name, image_path, face_index)
        except Exception as exc:
            errorBox("Remove failed", str(exc))
            return
        self.refresh_face_library(refresh_people=True, reason="prototype face removed")
        self._maybe_refresh_global_face_album(reason="prototype face removed")
        self._refresh_face_identities(force_reload=True)
        self._show_face_identities_tab()

    def _pin_selected_identity_prototype_face(self) -> None:
        if not self._ensure_writable_face_action("change the pinned identity example"):
            return
        person_name = str(self._face_identity_selected_name or "").strip()
        refs = self._selected_identity_prototype_refs()
        if not person_name or not refs:
            errorBox("No prototype selected", "Select one prototype face first.")
            return
        image_path, face_index = refs[0]
        try:
            self._active_face_service().pin_person_prototype_face(person_name, image_path, face_index)
        except Exception as exc:
            errorBox("Pin failed", str(exc))
            return
        self.refresh_face_library(refresh_people=True, reason="prototype face pinned")
        self._maybe_refresh_global_face_album(reason="prototype face pinned")
        self._refresh_face_identities(force_reload=True)
        self._show_face_identities_tab()

    def _merge_selected_identity_into_target(self) -> None:
        if not self._ensure_writable_face_action("merge saved identities"):
            return
        source = str(self._face_identity_selected_name or "").strip()
        target = str(self.face_identity_merge_target.text().strip() if hasattr(self, "face_identity_merge_target") else "")
        if not source or not target:
            errorBox("Missing names", "Select an identity and enter the target identity name.")
            return
        if source == target:
            errorBox("Invalid merge", "The source identity and target identity must be different.")
            return
        label_counts = getattr(self._active_face_service(), "label_counts", None)
        affected = int((label_counts() if callable(label_counts) else {}).get(source, 0))
        if not confirmBox(
            "Merge saved identities?",
            (
                f"Target: {source} → {target}\n"
                f"Affected face labels: {affected}\n"
                "Result: the source identity is removed and its labels move to the target.\n"
                "Recovery: this merge is not automatically recoverable."
            ),
            parent=self,
        ):
            self.status_label.setText("Identity merge cancelled. No face data was changed.")
            return
        try:
            self._active_face_service().merge_person_identities(source, target)
        except Exception as exc:
            errorBox("Merge failed", str(exc))
            return
        self.face_identity_merge_target.clear()
        self.refresh_face_library(refresh_people=True, reason="identities merged from identities tab")
        self._maybe_refresh_global_face_album(reason="identities merged from identities tab")
        self._refresh_face_identities(force_reload=True)
        self._show_face_identities_tab()

    def _clear_selected_identity_labels(self) -> None:
        if not self._ensure_writable_face_action("remove a saved identity name"):
            return
        person_name = str(self._face_identity_selected_name or "").strip()
        if not person_name:
            errorBox("No identity selected", "Select an identity first.")
            return
        label_counts = getattr(self._active_face_service(), "label_counts", None)
        affected = int((label_counts() if callable(label_counts) else {}).get(person_name, 0))
        if not confirmBox(
            "Remove identity name?",
            (
                f"Target: {person_name}\n"
                f"Affected face labels: {affected}\n"
                "Result: the faces become unnamed; source photos remain unchanged.\n"
                "Recovery: this action is not automatically recoverable."
            ),
            parent=self,
        ):
            self.status_label.setText("Remove name cancelled. No face data was changed.")
            return
        try:
            self._active_face_service().clear_person_labels(person_name)
        except Exception as exc:
            errorBox("Clear failed", str(exc))
            return
        self.refresh_face_library(refresh_people=True, reason="identity labels cleared from identities tab")
        self._maybe_refresh_global_face_album(reason="identity labels cleared from identities tab")
        self._refresh_face_identities(force_reload=True)
        self._show_face_identities_tab()

    def _current_face_identity_name(self) -> str:
        for widget_name in ("face_label_name", "face_find_name_query", "person_name", "face_name_query"):
            widget = getattr(self, widget_name, None)
            if widget is None:
                continue
            try:
                value = str(widget.text() or "").strip()
            except Exception:
                value = ""
            if value:
                return value
        return ""

    def current_face_cluster_backends(self) -> list[str]:
        override_checkbox = getattr(self, "face_cluster_backend_override_checkbox", None)
        if override_checkbox is None or not bool(override_checkbox.isChecked()):
            return ["hdbscan"]
        combo = getattr(self, "face_cluster_backend", None)
        if combo is not None:
            try:
                return [str(combo.currentData() or "hdbscan")]
            except Exception:
                pass
        return ["hdbscan"]

    def current_face_cluster_backend(self) -> str:
        return str((self.current_face_cluster_backends() or ["hdbscan"])[0] or "hdbscan").strip().lower()

    def current_face_cluster_outlier_policy(self) -> str:
        combo = getattr(self, "face_cluster_outlier_policy_combo", None)
        if combo is not None:
            try:
                return str(combo.currentData() or combo.currentText() or "isolate").strip().lower()
            except Exception:
                pass
        return "isolate"

    def _on_face_cluster_backends_changed(self, changed_backend: str) -> None:
        _ = changed_backend
        self._refresh_face_cluster_backend_controls()

    def current_face_cluster_backend_options(self) -> dict[str, object]:
        if self.current_face_cluster_backend() != "hdbscan":
            return {}
        min_cluster_size = max(
            2,
            int(getattr(self, "face_hdbscan_min_cluster_size_spin", self.face_cluster_count).value())
            if hasattr(self, "face_hdbscan_min_cluster_size_spin")
            else 2,
        )
        min_samples = 0
        if hasattr(self, "face_hdbscan_min_samples_spin"):
            try:
                min_samples = max(0, int(self.face_hdbscan_min_samples_spin.value()))
            except Exception:
                min_samples = 0
        cluster_selection_epsilon = 0.0
        if hasattr(self, "face_hdbscan_cluster_selection_epsilon_spin"):
            try:
                cluster_selection_epsilon = max(0.0, float(self.face_hdbscan_cluster_selection_epsilon_spin.value()))
            except Exception:
                cluster_selection_epsilon = 0.0
        allow_single_cluster = False
        if hasattr(self, "face_hdbscan_allow_single_cluster_checkbox"):
            allow_single_cluster = bool(self.face_hdbscan_allow_single_cluster_checkbox.isChecked())
        return {
            "min_cluster_size": int(min_cluster_size),
            "min_samples": int(min_samples),
            "cluster_selection_epsilon": float(cluster_selection_epsilon),
            "allow_single_cluster": bool(allow_single_cluster),
        }

    def _refresh_face_cluster_backend_controls(self) -> None:
        backend = self.current_face_cluster_backend()
        override_enabled = bool(getattr(self, "face_cluster_backend_override_checkbox", None) and self.face_cluster_backend_override_checkbox.isChecked())
        backend_field = getattr(self, "face_cluster_backend_field", None)
        if backend_field is not None:
            backend_field.setVisible(override_enabled)
        combo = getattr(self, "face_cluster_backend", None)
        if combo is not None:
            combo.setEnabled(override_enabled)
            combo.setToolTip(
                f"{FACE_HELP['face_cluster_backend']}\n\n{face_cluster_backend_tooltip(backend)}".strip()
            )
        hdbscan_only = backend == "hdbscan"
        widget = getattr(self, "face_hdbscan_options_widget", None)
        if widget is not None:
            widget.setVisible(hdbscan_only)

    def _save_face_refs_name(
        self,
        face_refs: list[tuple[str, int]],
        *,
        source_label: str,
        person_name_override: str | None = None,
    ) -> None:
        refs = list(dict.fromkeys((str(image_path), int(face_index)) for image_path, face_index in face_refs if str(image_path or "").strip()))
        if not refs:
            errorBox("No indexed faces selected", "Select one or more saved face tiles first.")
            return
        person_name = str(person_name_override or "").strip() or self._current_face_identity_name()
        if not person_name:
            self._show_face_search_tab()
            errorBox("Missing name", "Enter the identity name first, then run Name Selected again.")
            return
        threshold = float(getattr(self, "face_library_label_threshold", self.face_label_threshold).value())

        def _run(progress, cancel_check):
            progress(-1, "Saving selected face label...")
            _ = cancel_check
            return self._active_face_service().label_indexed_faces(
                person_name,
                refs,
                similarity_threshold=threshold,
            )

        def _done(person) -> None:
            for widget_name in ("face_find_name_query", "face_name_query", "face_label_name", "person_name"):
                widget = getattr(self, widget_name, None)
                if widget is not None:
                    try:
                        widget.setText(person.person_name)
                    except Exception:
                        pass
            tags = [tag.strip() for tag in self.face_profile_tags.text().split(",") if tag.strip()] if hasattr(self, "face_profile_tags") else []
            cover_ref = refs[0] if refs else None
            self._active_face_service().save_person_profile(
                person.person_name,
                notes=self.face_profile_notes.text().strip() if hasattr(self, "face_profile_notes") else "",
                tags=tags,
                cover_face_ref=cover_ref,
            )
            self.status_label.setText(
                f"Queued review for '{person.person_name}' on {len(refs)} face(s) from {source_label}."
            )
            self._refresh_pending_face_labels()
            self.refresh_face_library(reason="face refs named")
            self._maybe_refresh_global_face_album(reason="face refs named")
            self._refresh_face_identities(force_reload=True)
            self._apply_face_result_filters()

        self._start_job("Saving selected face label", _run, _done)

    def _prepare_name_selected_detected_faces(self) -> None:
        self._save_face_refs_name(self._selected_detected_face_indexed_refs(), source_label="Detected Faces")

    def _prepare_name_selected_face_results(self) -> None:
        self._name_face_results_from_prompt()

    def _show_face_result_groups(
        self,
        groups: list[FaceResultGroup],
        *,
        summary: str,
        photo_paths: list[str],
        overlay_by_path: dict[str, str],
        context_by_path: dict[str, dict[str, object]],
        kind: str,
    ) -> None:
        self._face_result_source_groups = list(groups)
        self._face_result_source_summary = str(summary or "")
        self._face_result_source_kind = str(kind or "")
        self._face_result_source_context_by_path = dict(context_by_path or {})
        self._face_result_source_overlay_by_path = dict(overlay_by_path or {})
        self._face_result_source_photo_paths = list(photo_paths or [])
        self._face_result_active_group_kind = "raw"
        self._face_result_selected_group_id = ""
        self._face_result_selected_group_id_by_kind = {"raw": "", "merged_name": "", "exif_name": ""}
        self._request_face_result_filter_refresh(immediate=True)

    def _face_result_filter_state(self) -> dict[str, str]:
        return {
            "label_filter": str(self.face_results_label_filter_combo.currentData() or "all").strip().lower(),
            "quality_filter": str(self.face_results_quality_filter_combo.currentData() or "all").strip().lower(),
            "folder_filter": self.face_results_folder_filter.text().strip().lower(),
            "date_from": self.face_results_date_from.text().strip(),
            "date_to": self.face_results_date_to.text().strip(),
            "sort_mode": str(self.face_results_sort_combo.currentData() or "score").strip().lower(),
        }

    def _request_face_result_filter_refresh(self, *, immediate: bool = False) -> None:
        if self._face_result_filter_timer.isActive():
            self._face_result_filter_timer.stop()
        if immediate:
            self._start_face_result_filter_job()
            return
        self.face_results_summary.setText("Refreshing face results...")
        self._face_result_filter_timer.start(150)

    def _cancel_face_result_filter_job(self) -> None:
        job = self._face_result_filter_job
        thread = self._face_result_filter_thread
        if job is not None:
            try:
                job.cancel()
            except Exception:
                pass
        self._face_result_filter_job = None
        self._face_result_filter_thread = None
        _ = thread

    def _on_face_result_filter_thread_finished(self, thread=None) -> None:
        retained: list[tuple[object | None, object | None]] = []
        for job, retained_thread in self._retained_face_result_filter_refs:
            if retained_thread is thread:
                continue
            retained.append((job, retained_thread))
        self._retained_face_result_filter_refs = retained
        if thread is self._face_result_filter_thread:
            self._face_result_filter_thread = None
            self._face_result_filter_job = None

    def _start_face_result_filter_job(self) -> None:
        if self._face_result_filter_timer.isActive():
            self._face_result_filter_timer.stop()
        groups = list(self._face_result_source_groups or [])
        self._cancel_face_result_filter_job()
        self._face_result_filter_request_id += 1
        request_id = int(self._face_result_filter_request_id)
        if not groups:
            self._apply_face_result_filter_snapshot(
                FaceResultFilterSnapshot(
                    request_id=request_id,
                    groups=(),
                    group_entries=(),
                    selected_group_id="",
                    summary_text=str(self._face_result_source_summary or ""),
                    photo_paths_by_group_id={},
                )
            )
            return
        filter_state = self._face_result_filter_state()
        selected_group_id = str(self._face_result_selected_group_id or "").strip()
        current_folder = str(self._current_directory() or "")
        source_photo_paths = [str(path) for path in self._face_result_source_photo_paths if str(path or "").strip()]
        source_summary = str(self._face_result_source_summary or "")
        kind = str(self._face_result_source_kind or "")
        service = self._active_face_service()
        pending_assignments = list(self._pending_face_assignments or [])

        def _run(progress, cancel_check):
            progress(-1, "Filtering face results...")
            return self._build_face_result_filter_snapshot(
                request_id=request_id,
                groups=groups,
                filter_state=filter_state,
                selected_group_id=selected_group_id,
                source_photo_paths=source_photo_paths,
                source_summary=source_summary,
                current_folder=current_folder,
                service=service,
                pending_assignments=pending_assignments,
                cancel_check=cancel_check,
            )

        job = AsyncJob(_run)
        job.progress.connect(lambda _value, text: self.face_results_summary.setText(str(text)))

        def _failed(message: str) -> None:
            text = str(message or "").strip()
            if text.casefold() == "cancelled":
                return
            self.face_results_summary.setText(f"Face result filter failed: {text}")

        def _completed(result) -> None:
            if not isinstance(result, FaceResultFilterSnapshot):
                return
            if int(result.request_id) != int(self._face_result_filter_request_id):
                return
            self._apply_face_result_filter_snapshot(result, kind=kind)

        job.failed.connect(_failed)
        job.completed.connect(_completed)
        self._face_result_filter_job = job
        thread = start_job_in_thread(job)
        self._face_result_filter_thread = thread
        self._retained_face_result_filter_refs.append((job, thread))
        thread.finished.connect(
            lambda thread=thread: self._on_face_result_filter_thread_finished(thread),
            Qt.ConnectionType.QueuedConnection,
        )

    @staticmethod
    def _face_result_saved_ref_worker(item: FaceTileItem) -> tuple[str, int] | None:
        image_path = str(item.image_path or "").strip()
        if not image_path:
            return None
        face_index = int(item.saved_face_index) if int(item.saved_face_index) >= 0 else int(item.face_index)
        return (image_path, face_index)

    @staticmethod
    def _face_result_exif_name_worker(image_path: str, *, exif_name_cache: dict[str, str]) -> str:
        path = str(image_path or "").strip()
        if not path:
            return ""
        cached = exif_name_cache.get(path)
        if cached is not None:
            return cached
        try:
            cached = str(GalleryActionService.read_exif_metadata_value(path, FACE_EXIF_PERSON_KEY) or "").strip()
        except Exception:
            cached = ""
        exif_name_cache[path] = cached
        return cached

    def _face_result_resolved_name_worker(
        self,
        item: FaceTileItem,
        *,
        service,
        metadata_cache: dict[tuple[str, int], tuple[str, str]],
        pending_name_by_ref: dict[tuple[str, int], str],
    ) -> tuple[str, str]:
        ref = self._face_result_saved_ref_worker(item)
        if ref is not None:
            pending_name = str(pending_name_by_ref.get(ref, "") or "").strip()
            if pending_name:
                return pending_name, "pending"
        person_name, _quality_status = self._face_result_item_metadata_worker(
            item,
            service=service,
            metadata_cache=metadata_cache,
        )
        if person_name:
            return str(person_name).strip(), "saved"
        return "", ""

    def _face_result_raw_group_display_worker(
        self,
        group: FaceResultGroup,
        *,
        service,
        metadata_cache: dict[tuple[str, int], tuple[str, str]],
        pending_name_by_ref: dict[tuple[str, int], str],
    ) -> tuple[str, str, str]:
        title = str(group.title or "")
        summary = str(group.summary or "")
        resolved_name = str(group.resolved_name or "").strip()
        comparison_key = str(group.comparison_key or "").strip()
        cluster_id = int(group.cluster_id)
        if group.review_kind != "raw" or not comparison_key:
            return title, summary, resolved_name
        if str(group.group_id or "") != f"{comparison_key}:{cluster_id}":
            return title, summary, resolved_name
        matched_names: set[str] = set()
        all_named = True
        for item in list(group.items):
            item_name, _name_source = self._face_result_resolved_name_worker(
                item,
                service=service,
                metadata_cache=metadata_cache,
                pending_name_by_ref=pending_name_by_ref,
            )
            item_name = str(item_name or "").strip()
            if not item_name:
                all_named = False
                continue
            matched_names.add(item_name)
        resolved_name = next(iter(matched_names)) if all_named and len(matched_names) == 1 else ""
        title = f"{self._face_cluster_backend_label(comparison_key)} | {'Outlier' if cluster_id == -1 else f'Cluster {cluster_id}'}"
        if matched_names:
            title = f"{title} | {', '.join(sorted(matched_names, key=str.casefold))}"
        if summary.startswith(str(group.title or "")) and str(group.title or ""):
            summary = f"{title}{summary[len(str(group.title or '')):]}"
        return title, summary, resolved_name

    def _build_face_result_review_groups_worker(
        self,
        groups: list[FaceResultGroup],
        *,
        service,
        metadata_cache: dict[tuple[str, int], tuple[str, str]],
        pending_name_by_ref: dict[tuple[str, int], str],
    ) -> tuple[
        tuple[FaceResultGroup, ...],
        tuple[ListEntry, ...],
        dict[str, list[str]],
        tuple[FaceResultGroup, ...],
        tuple[ListEntry, ...],
        dict[str, list[str]],
    ]:
        merged_items_by_key: dict[str, list[FaceTileItem]] = {}
        merged_seen_refs_by_key: dict[str, set[tuple[str, int]]] = {}
        merged_source_groups_by_key: dict[str, list[str]] = {}
        merged_source_titles_by_key: dict[str, list[str]] = {}
        merged_name_source_by_key: dict[str, set[str]] = {}
        merged_order: list[str] = []
        for group in list(groups or []):
            raw_group_id = str(group.group_id)
            raw_group_title = str(group.title)
            for item in list(group.items):
                ref = self._face_result_saved_ref_worker(item)
                resolved_name, name_source = self._face_result_resolved_name_worker(
                    item,
                    service=service,
                    metadata_cache=metadata_cache,
                    pending_name_by_ref=pending_name_by_ref,
                )
                if not resolved_name:
                    continue
                merged_key = resolved_name
                if merged_key not in merged_items_by_key:
                    merged_items_by_key[merged_key] = []
                    merged_seen_refs_by_key[merged_key] = set()
                    merged_source_groups_by_key[merged_key] = []
                    merged_source_titles_by_key[merged_key] = []
                    merged_name_source_by_key[merged_key] = set()
                    merged_order.append(merged_key)
                if ref is None or ref not in merged_seen_refs_by_key[merged_key]:
                    merged_items_by_key[merged_key].append(item)
                    if ref is not None:
                        merged_seen_refs_by_key[merged_key].add(ref)
                if raw_group_id not in merged_source_groups_by_key[merged_key]:
                    merged_source_groups_by_key[merged_key].append(raw_group_id)
                    merged_source_titles_by_key[merged_key].append(raw_group_title)
                if name_source:
                    merged_name_source_by_key[merged_key].add(name_source)

        merged_groups: list[FaceResultGroup] = []
        merged_entries: list[ListEntry] = []
        merged_photo_paths_by_group_id: dict[str, list[str]] = {}
        for group_key in sorted(merged_order, key=lambda value: str(value).casefold()):
            items = list(merged_items_by_key.get(group_key, []))
            source_ids = tuple(merged_source_groups_by_key.get(group_key, []))
            source_titles = list(merged_source_titles_by_key.get(group_key, []))
            photo_paths = list(dict.fromkeys(str(item.image_path) for item in items if str(item.image_path or "").strip()))
            source_count = len(source_ids)
            photo_count = len(photo_paths)
            resolved_name = str(group_key)
            source_labels = merged_name_source_by_key.get(group_key, set())
            source_label = "pending labels" if "pending" in source_labels else "saved labels"
            title = f"{resolved_name}\n{len(items)} face(s) | {source_count} raw group(s)"
            merge_notice = f"Built from {source_label}"
            if source_count > 1:
                merge_notice += f" across {source_count} raw clusters: {', '.join(source_titles[:4])}"
            summary = f"{resolved_name}: {len(items)} face(s) across {photo_count} photo(s). {merge_notice}."
            group_id = f"merged_name:{resolved_name}"
            group = FaceResultGroup(
                group_id=group_id,
                title=title,
                summary=summary,
                items=tuple(items),
                review_kind="merged_name",
                source_group_ids=source_ids,
                resolved_name=resolved_name,
                merge_notice=merge_notice,
            )
            merged_groups.append(group)
            merged_entries.append(ListEntry(title=title, tooltip=summary, payload=group_id))
            merged_photo_paths_by_group_id[group_id] = photo_paths
        return (
            tuple(merged_groups),
            tuple(merged_entries),
            merged_photo_paths_by_group_id,
            (),
            (),
            {},
        )

    def _build_face_result_filter_snapshot(
        self,
        *,
        request_id: int,
        groups: list[FaceResultGroup],
        filter_state: dict[str, str],
        selected_group_id: str,
        source_photo_paths: list[str],
        source_summary: str,
        current_folder: str,
        service,
        pending_assignments: list[FaceLabelAssignment],
        cancel_check,
    ) -> FaceResultFilterSnapshot:
        metadata_cache: dict[tuple[str, int], tuple[str, str]] = {}
        mtime_cache: dict[str, float] = {}
        date_cache: dict[str, str] = {}
        source_refs = list(
            dict.fromkeys(
                ref
                for group in list(groups or [])
                for item in list(group.items)
                for ref in [self._face_result_saved_ref_worker(item)]
                if ref is not None
            )
        )
        load_metadata_many = getattr(service, "load_face_metadata_many", None)
        if callable(load_metadata_many) and source_refs:
            try:
                metadata_cache.update(dict(load_metadata_many(source_refs) or {}))
            except Exception:
                metadata_cache = {}
        filtered_groups: list[FaceResultGroup] = []
        photo_paths_by_group_id: dict[str, list[str]] = {}
        for group in list(groups or []):
            if cancel_check():
                raise Cancelled()
            filtered_items = [
                item
                for item in list(group.items)
                if self._face_result_item_passes_filters_worker(
                    item,
                    filter_state=filter_state,
                    service=service,
                    metadata_cache=metadata_cache,
                    date_cache=date_cache,
                )
            ]
            if not filtered_items:
                continue
            sorted_items = self._sorted_face_result_items_worker(
                filtered_items,
                sort_mode=filter_state.get("sort_mode", "score"),
                current_folder=current_folder,
                mtime_cache=mtime_cache,
            )
            filtered_group = FaceResultGroup(
                group_id=group.group_id,
                title=group.title,
                summary=group.summary,
                items=tuple(sorted_items),
                comparison_key=group.comparison_key,
                cluster_id=group.cluster_id,
                explanation=group.explanation,
                suggestion=group.suggestion,
                cluster_status=group.cluster_status,
                review_kind=group.review_kind,
                source_group_ids=group.source_group_ids,
                resolved_name=group.resolved_name,
                exif_name=group.exif_name,
                merge_notice=group.merge_notice,
            )
            filtered_groups.append(filtered_group)
            item_paths = list(dict.fromkeys(str(item.image_path) for item in sorted_items if str(item.image_path or "").strip()))
            if source_photo_paths:
                item_path_set = set(item_paths)
                group_photo_paths = [path for path in source_photo_paths if path in item_path_set]
                photo_paths_by_group_id[group.group_id] = group_photo_paths or item_paths
            else:
                photo_paths_by_group_id[group.group_id] = item_paths
        pending_name_by_ref: dict[tuple[str, int], str] = {}
        visible_refs = {
            ref
            for group in filtered_groups
            for item in list(group.items)
            for ref in [self._face_result_saved_ref_worker(item)]
            if ref is not None
        }
        for assignment in list(pending_assignments or []):
            if cancel_check():
                raise Cancelled()
            key = (str(assignment.image_path), int(assignment.face_index))
            if key not in visible_refs:
                continue
            pending_name = str(getattr(assignment, "person_name", "") or "").strip()
            if pending_name and key not in pending_name_by_ref:
                pending_name_by_ref[key] = pending_name
        display_groups: list[FaceResultGroup] = []
        for group in filtered_groups:
            display_title, display_summary, resolved_name = self._face_result_raw_group_display_worker(
                group,
                service=service,
                metadata_cache=metadata_cache,
                pending_name_by_ref=pending_name_by_ref,
            )
            display_groups.append(
                FaceResultGroup(
                    group_id=group.group_id,
                    title=display_title,
                    summary=display_summary,
                    items=group.items,
                    comparison_key=group.comparison_key,
                    cluster_id=group.cluster_id,
                    explanation=group.explanation,
                    suggestion=group.suggestion,
                    cluster_status=group.cluster_status,
                    review_kind=group.review_kind,
                    source_group_ids=group.source_group_ids,
                    resolved_name=resolved_name,
                    exif_name=group.exif_name,
                    merge_notice=group.merge_notice,
                )
            )
        filtered_groups = display_groups
        if selected_group_id not in {group.group_id for group in filtered_groups}:
            selected_group_id = filtered_groups[0].group_id if filtered_groups else ""
        summary_text = (
            f"{source_summary} Showing "
            f"{sum(len(group.items) for group in filtered_groups)} face tile(s) across "
            f"{len(filtered_groups)} group(s)."
        ).strip()
        group_entries = tuple(
            ListEntry(
                title=str(group.title),
                tooltip=str(group.summary),
                payload=str(group.group_id),
            )
            for group in filtered_groups
        )
        (
            merged_groups,
            merged_group_entries,
            merged_photo_paths_by_group_id,
            exif_groups,
            exif_group_entries,
            exif_photo_paths_by_group_id,
        ) = self._build_face_result_review_groups_worker(
            filtered_groups,
            service=service,
            metadata_cache=metadata_cache,
            pending_name_by_ref=pending_name_by_ref,
        )
        return FaceResultFilterSnapshot(
            request_id=int(request_id),
            groups=tuple(filtered_groups),
            group_entries=group_entries,
            selected_group_id=selected_group_id,
            summary_text=summary_text,
            photo_paths_by_group_id=photo_paths_by_group_id,
            merged_groups=merged_groups,
            merged_group_entries=merged_group_entries,
            merged_photo_paths_by_group_id=merged_photo_paths_by_group_id,
            exif_groups=exif_groups,
            exif_group_entries=exif_group_entries,
            exif_photo_paths_by_group_id=exif_photo_paths_by_group_id,
        )

    @staticmethod
    def _face_result_item_metadata_worker(
        item: FaceTileItem,
        *,
        service,
        metadata_cache: dict[tuple[str, int], tuple[str, str]],
    ) -> tuple[str, str]:
        person_name = ""
        quality_status = str(item.status or "saved").strip().lower()
        payload = item.payload
        if payload is not None:
            person_name = str(getattr(payload, "person_name", "") or person_name).strip()
            payload_quality = str(getattr(payload, "quality_status", "") or "").strip().lower()
            if payload_quality:
                quality_status = payload_quality
        saved_face_index = int(item.saved_face_index)
        if saved_face_index < 0:
            return person_name, quality_status
        cache_key = (str(item.image_path), saved_face_index)
        cached = metadata_cache.get(cache_key)
        if cached is not None:
            return cached
        load_metadata_many = getattr(service, "load_face_metadata_many", None)
        if callable(load_metadata_many):
            try:
                bulk_metadata = dict(load_metadata_many([cache_key]) or {})
            except Exception:
                bulk_metadata = {}
            if cache_key in bulk_metadata:
                metadata_cache[cache_key] = bulk_metadata[cache_key]
                return metadata_cache[cache_key]
        try:
            record = service.load_face_record(str(item.image_path), saved_face_index)
        except Exception:
            record = None
        if record is not None:
            person_name = str(getattr(record, "person_name", "") or person_name).strip()
            quality_status = str(getattr(record, "quality_status", quality_status) or quality_status).strip().lower()
        metadata_cache[cache_key] = (person_name, quality_status)
        return metadata_cache[cache_key]

    @staticmethod
    def _face_result_item_date_text_worker(item: FaceTileItem, *, date_cache: dict[str, str]) -> str:
        image_path = str(item.image_path or "")
        cached = date_cache.get(image_path)
        if cached is not None:
            return cached
        try:
            mtime = Path(image_path).stat().st_mtime
            cached = datetime.fromtimestamp(mtime).date().isoformat()
        except Exception:
            cached = ""
        date_cache[image_path] = cached
        return cached

    def _face_result_item_passes_filters_worker(
        self,
        item: FaceTileItem,
        *,
        filter_state: dict[str, str],
        service,
        metadata_cache: dict[tuple[str, int], tuple[str, str]],
        date_cache: dict[str, str],
    ) -> bool:
        label_filter = str(filter_state.get("label_filter", "all") or "all").strip().lower()
        quality_filter = str(filter_state.get("quality_filter", "all") or "all").strip().lower()
        folder_filter = str(filter_state.get("folder_filter", "") or "").strip().lower()
        date_from = str(filter_state.get("date_from", "") or "").strip()
        date_to = str(filter_state.get("date_to", "") or "").strip()
        person_name, quality_status = self._face_result_item_metadata_worker(
            item,
            service=service,
            metadata_cache=metadata_cache,
        )
        if label_filter == "named" and not person_name:
            return False
        if label_filter == "unlabeled" and person_name:
            return False
        if quality_filter != "all" and str(quality_status or "").strip().lower() != quality_filter:
            return False
        if folder_filter and folder_filter not in str(Path(str(item.image_path)).parent).lower():
            return False
        if date_from or date_to:
            date_text = self._face_result_item_date_text_worker(item, date_cache=date_cache)
            if not date_text:
                return False
            if date_from and date_text < date_from:
                return False
            if date_to and date_text > date_to:
                return False
        return True

    @staticmethod
    def _sorted_face_result_items_worker(
        items: list[FaceTileItem],
        *,
        sort_mode: str,
        current_folder: str,
        mtime_cache: dict[str, float],
    ) -> list[FaceTileItem]:
        sort_mode = str(sort_mode or "score").strip().lower()

        def _score_for_item(item: FaceTileItem) -> float:
            return float(getattr(item.payload, "score", 0.0) or 0.0)

        def _mtime_for_item(item: FaceTileItem) -> float:
            image_path = str(item.image_path or "")
            if image_path in mtime_cache:
                return float(mtime_cache[image_path])
            try:
                mtime_cache[image_path] = float(Path(image_path).stat().st_mtime)
            except Exception:
                mtime_cache[image_path] = 0.0
            return float(mtime_cache[image_path])

        def _same_folder_key(item: FaceTileItem) -> tuple[int, float, str]:
            parent = str(Path(str(item.image_path)).parent)
            same_folder = 0 if current_folder and path_is_within_scope(parent, current_folder) else 1
            return (same_folder, -_score_for_item(item), str(item.image_path))

        if sort_mode == "newest":
            return sorted(items, key=lambda item: (-_mtime_for_item(item), -_score_for_item(item), str(item.image_path)))
        if sort_mode == "oldest":
            return sorted(items, key=lambda item: (_mtime_for_item(item), -_score_for_item(item), str(item.image_path)))
        if sort_mode == "same_folder":
            return sorted(items, key=_same_folder_key)
        return sorted(items, key=lambda item: (-_score_for_item(item), str(item.image_path), int(item.face_index)))

    def _apply_face_result_filter_snapshot(self, snapshot: FaceResultFilterSnapshot, *, kind: str | None = None) -> None:
        self._hide_face_result_hover_popup()
        active_kind = str(kind or self._face_result_source_kind or "")
        self._set_results_kind(active_kind)
        if active_kind != "face_clusters":
            self._face_cluster_compare_membership = {}
            self._face_cluster_compare_metrics = {}
            self._face_cluster_compare_last_refs = []
        self._face_result_groups = list(snapshot.groups)
        self._face_result_group_by_id = {group.group_id: group for group in self._face_result_groups}
        self._face_result_merged_groups = list(snapshot.merged_groups or ())
        self._face_result_merged_group_by_id = {group.group_id: group for group in self._face_result_merged_groups}
        self._face_result_exif_groups = list(snapshot.exif_groups or ())
        self._face_result_exif_group_by_id = {group.group_id: group for group in self._face_result_exif_groups}
        self._face_result_photo_paths_by_group_id = {
            str(group_id): list(paths or [])
            for group_id, paths in dict(snapshot.photo_paths_by_group_id or {}).items()
        }
        self._face_result_merged_photo_paths_by_group_id = {
            str(group_id): list(paths or [])
            for group_id, paths in dict(snapshot.merged_photo_paths_by_group_id or {}).items()
        }
        self._face_result_exif_photo_paths_by_group_id = {
            str(group_id): list(paths or [])
            for group_id, paths in dict(snapshot.exif_photo_paths_by_group_id or {}).items()
        }
        self.face_results_groups_model.set_items(list(snapshot.group_entries))
        self.face_results_merged_groups_model.set_items(list(snapshot.merged_group_entries))
        exif_model = getattr(self, "face_results_exif_groups_model", None)
        if exif_model is not None:
            exif_model.set_items(list(snapshot.exif_group_entries))
        unique_backends = {str(group.comparison_key or "").strip() for group in self._face_result_groups if str(group.comparison_key or "").strip()}
        if len(unique_backends) == 1:
            self.face_results_groups_label.setText(f"{self._face_cluster_backend_label(next(iter(unique_backends)))} Clusters")
        else:
            self.face_results_groups_label.setText("Grouped Photos")
        if not self._face_result_groups:
            self._face_result_photo_paths_by_group_id = {}
            self._face_result_merged_photo_paths_by_group_id = {}
            self._face_result_exif_photo_paths_by_group_id = {}
            self.face_results_summary.setText(str(snapshot.summary_text or self._face_result_source_summary or "No face results."))
            self.face_results_model.set_items([])
            self.results_gallery.update_gallery([])
            try:
                self.results_gallery.model.set_overlays_by_path({})
            except Exception:
                pass
            self.results_gallery.inspector_context_provider = lambda _p: {}
            self._publish_results([], lambda _p: {}, {}, {}, self._results_kind)
            self._update_face_result_actions()
            self._clear_face_cluster_result_details()
            return
        previous_raw = str(self._face_result_selected_group_id_by_kind.get("raw", "") or snapshot.selected_group_id or "")
        previous_merged = str(self._face_result_selected_group_id_by_kind.get("merged_name", "") or "")
        previous_exif = str(self._face_result_selected_group_id_by_kind.get("exif_name", "") or "")
        raw_selected = previous_raw if previous_raw in self._face_result_group_by_id else str(snapshot.selected_group_id or self._face_result_groups[0].group_id)
        merged_selected = previous_merged if previous_merged in self._face_result_merged_group_by_id else str(self._face_result_merged_groups[0].group_id if self._face_result_merged_groups else "")
        exif_selected = previous_exif if previous_exif in self._face_result_exif_group_by_id else str(self._face_result_exif_groups[0].group_id if self._face_result_exif_groups else "")
        self._face_result_selected_group_id_by_kind["raw"] = raw_selected
        self._face_result_selected_group_id_by_kind["merged_name"] = merged_selected
        self._face_result_selected_group_id_by_kind["exif_name"] = exif_selected
        named_view_selected = bool(
            active_kind == "face_album"
            and hasattr(self, "face_result_view_tabs")
            and self.face_result_view_tabs.currentIndex() == 0
        )
        preferred_group_kind = "merged_name" if named_view_selected else "raw"
        self._face_result_active_group_kind = preferred_group_kind
        if active_kind != "face_album" and hasattr(self, "face_result_view_tabs"):
            self.face_result_view_tabs.blockSignals(True)
            self.face_result_view_tabs.setCurrentIndex(1)
            self.face_result_view_tabs.blockSignals(False)
        if not self._face_result_group_map_for_kind(self._face_result_active_group_kind):
            self._face_result_active_group_kind = "raw"
            if named_view_selected and hasattr(self, "face_result_view_tabs"):
                self.face_result_view_tabs.blockSignals(True)
                self.face_result_view_tabs.setCurrentIndex(1)
                self.face_result_view_tabs.blockSignals(False)
        group_stack = getattr(self, "face_result_group_stack", None)
        if group_stack is not None:
            group_stack.setCurrentIndex(1 if self._face_result_active_group_kind == "merged_name" else 0)
        for group_kind, group_id in (
            ("raw", raw_selected),
            ("merged_name", merged_selected),
            ("exif_name", exif_selected),
        ):
            if group_kind == self._face_result_active_group_kind:
                self._sync_face_result_group_view_selection(group_kind, group_id)
            else:
                self._clear_face_result_group_view_selection(group_kind)
        self._face_result_selected_group_id = str(self._face_result_selected_group_id_by_kind.get(self._face_result_active_group_kind, "") or "")
        self._refresh_current_face_result_group()
        self.face_review_results_tabs.setCurrentIndex(2)

    def _apply_face_result_filters(self) -> None:
        self._request_face_result_filter_refresh(immediate=True)

    def _selected_face_result_group_ids(self) -> list[str]:
        if self._face_result_active_group_kind != "raw":
            return []
        ids = [
            str(entry.payload or "")
            for entry in self._selected_list_entries(self.face_results_groups_list, self.face_results_groups_model)
        ]
        return [group_id for group_id in ids if group_id]

    def _selected_face_result_raw_groups(self) -> list[FaceResultGroup]:
        if self._face_result_active_group_kind != "raw":
            return []
        selected_group_ids = set(self._selected_face_result_group_ids())
        if not selected_group_ids:
            return []
        return [group for group in self._face_result_groups if group.group_id in selected_group_ids]

    def _reload_face_result_group_list(self, *, select_group_id: str = "", summary: str | None = None) -> None:
        self._hide_face_result_hover_popup()
        selected_group_id = str(select_group_id or self._face_result_selected_group_id_by_kind.get("raw", "") or "").strip()
        self._face_result_group_by_id = {group.group_id: group for group in self._face_result_groups}
        self.face_results_groups_model.set_items(
            [
                ListEntry(
                    title=str(group.title),
                    tooltip=str(group.summary),
                    payload=str(group.group_id),
                )
                for group in self._face_result_groups
            ]
        )
        if self._face_result_groups:
            if selected_group_id not in self._face_result_group_by_id:
                selected_group_id = str(self._face_result_groups[0].group_id)
            self._face_result_selected_group_id_by_kind["raw"] = selected_group_id
            if self._face_result_active_group_kind == "raw":
                self._sync_face_result_group_view_selection("raw", selected_group_id)
                self._refresh_current_face_result_group()
        else:
            self.face_results_model.set_items([])
            self._update_face_result_actions()
            self._clear_face_cluster_result_details()
        if summary is not None:
            self.face_results_summary.setText(str(summary))

    def _face_result_groups_for_kind(self, group_kind: str) -> list[FaceResultGroup]:
        if group_kind == "merged_name":
            return list(self._face_result_merged_groups)
        if group_kind == "exif_name":
            return list(self._face_result_exif_groups)
        return list(self._face_result_groups)

    def _face_result_group_map_for_kind(self, group_kind: str) -> dict[str, FaceResultGroup]:
        if group_kind == "merged_name":
            return dict(self._face_result_merged_group_by_id)
        if group_kind == "exif_name":
            return dict(self._face_result_exif_group_by_id)
        return dict(self._face_result_group_by_id)

    def _face_result_photo_paths_for_kind(self, group_kind: str) -> dict[str, list[str]]:
        if group_kind == "merged_name":
            return dict(self._face_result_merged_photo_paths_by_group_id)
        if group_kind == "exif_name":
            return dict(self._face_result_exif_photo_paths_by_group_id)
        return dict(self._face_result_photo_paths_by_group_id)

    def _face_result_list_and_model_for_kind(self, group_kind: str) -> tuple[QListView | None, ListEntryModel | None]:
        if group_kind == "merged_name":
            return getattr(self, "face_results_merged_groups_list", None), getattr(self, "face_results_merged_groups_model", None)
        if group_kind == "exif_name":
            return getattr(self, "face_results_exif_groups_list", None), getattr(self, "face_results_exif_groups_model", None)
        return getattr(self, "face_results_groups_list", None), getattr(self, "face_results_groups_model", None)

    def _clear_face_result_group_view_selection(self, group_kind: str) -> None:
        view, _model = self._face_result_list_and_model_for_kind(group_kind)
        if view is None:
            return
        selection_model = view.selectionModel()
        self._face_result_selection_syncing = True
        try:
            if selection_model is not None:
                selection_model.blockSignals(True)
                selection_model.clearSelection()
                selection_model.blockSignals(False)
            view.setCurrentIndex(QModelIndex())
        finally:
            self._face_result_selection_syncing = False

    def _sync_face_result_group_view_selection(self, group_kind: str, group_id: str) -> None:
        view, model = self._face_result_list_and_model_for_kind(group_kind)
        groups = self._face_result_groups_for_kind(group_kind)
        if view is None or model is None:
            return
        target_row = next((row for row, group in enumerate(groups) if group.group_id == group_id), -1)
        selection_model = view.selectionModel()
        self._face_result_selection_syncing = True
        try:
            if selection_model is not None:
                selection_model.blockSignals(True)
                selection_model.clearSelection()
                if target_row >= 0:
                    index = model.index(target_row, 0)
                    selection_model.select(index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
                    selection_model.setCurrentIndex(index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
                selection_model.blockSignals(False)
            if target_row >= 0:
                index = model.index(target_row, 0)
                view.setCurrentIndex(index)
                view.scrollTo(index)
            else:
                view.setCurrentIndex(QModelIndex())
        finally:
            self._face_result_selection_syncing = False

    def _clear_other_face_result_group_selections(self, active_group_kind: str) -> None:
        for group_kind in ("raw", "merged_name", "exif_name"):
            if group_kind == active_group_kind:
                continue
            view, _model = self._face_result_list_and_model_for_kind(group_kind)
            if view is None or view.selectionModel() is None:
                continue
            view.selectionModel().blockSignals(True)
            view.selectionModel().clearSelection()
            view.setCurrentIndex(QModelIndex())
            view.selectionModel().blockSignals(False)

    def _current_face_result_group(self) -> FaceResultGroup | None:
        return self._face_result_group_map_for_kind(self._face_result_active_group_kind).get(self._face_result_selected_group_id)

    def _current_face_result_raw_group(self) -> FaceResultGroup | None:
        if self._face_result_active_group_kind != "raw":
            return None
        return self._face_result_group_by_id.get(self._face_result_selected_group_id)

    def _on_face_result_group_selection_changed(self, group_kind: str) -> None:
        if self._face_result_selection_syncing:
            return
        self._hide_face_result_hover_popup()
        view, _model = self._face_result_list_and_model_for_kind(group_kind)
        if view is None:
            return
        current_index = view.currentIndex()
        group_id = str(current_index.data(ListEntryModel.PayloadRole) or "") if current_index.isValid() else ""
        self._face_result_active_group_kind = str(group_kind or "raw")
        self._face_result_selected_group_id_by_kind[self._face_result_active_group_kind] = group_id
        self._face_result_selected_group_id = group_id
        self._clear_other_face_result_group_selections(self._face_result_active_group_kind)
        self._refresh_current_face_result_group()
        if (
            self._face_result_active_group_kind == "raw"
            and self.is_all_faces_tab_active()
            and any(str(group.group_id) == group_id for group in self._face_album_groups)
        ):
            self._load_face_album_group_members(group_id, append=False)
        self._update_face_album_paging_actions()

    def _on_face_result_group_changed(self) -> None:
        self._on_face_result_group_selection_changed("raw")

    def _refresh_current_face_result_group(self) -> None:
        group = self._current_face_result_group()
        self.face_results_model.set_items(list(group.items) if group is not None else [])
        self._update_face_result_actions()
        self._schedule_face_result_tile_loads()
        if group is not None:
            self.face_results_summary.setText(group.summary)
        self._set_face_cluster_result_details(group)
        self._update_face_result_membership_table()
        self._update_face_results_context()

    def _show_current_face_result_group_photos(self) -> None:
        group = self._current_face_result_group()
        if group is None:
            return
        photo_paths = list(
            self._face_result_photo_paths_for_kind(self._face_result_active_group_kind).get(
                self._face_result_selected_group_id,
                [],
            )
        )
        self._show_derived_face_photos(
            photo_paths,
            title=f"{group.title.splitlines()[0]} photos",
        )

    def _show_face_match_results(
        self,
        results: list[FaceSearchResult],
        *,
        title: str,
        summary: str,
        match_label: str,
    ) -> None:
        items: list[FaceTileItem] = []
        overlay_by_path: dict[str, str] = {}
        context_by_path: dict[str, dict[str, object]] = {}
        photo_paths: list[str] = []
        for result in results:
            photo_paths.append(result.image_path)
            overlay_by_path.setdefault(result.image_path, f"{match_label} {float(result.score):.3f}")
            context_by_path.setdefault(
                result.image_path,
                {
                    "face_search": {
                        "score": float(result.score),
                        "bbox": tuple(result.face_bbox),
                        "confidence": float(result.face_confidence),
                    }
                },
            )
            items.append(
                FaceTileItem(
                    image_path=str(result.image_path),
                    face_index=int(result.face_index),
                    bbox=tuple(int(value) for value in result.face_bbox),
                    title=f"{Path(result.image_path).name}\n{float(result.score):.3f}",
                    tooltip=(
                        f"{result.image_path}\nface #{int(result.face_index) + 1}\n"
                        f"bbox={tuple(int(value) for value in result.face_bbox)}\n"
                        f"score={float(result.score):.4f}\nconf={float(result.face_confidence):.3f}\n"
                        f"{result.match_reason}"
                    ),
                    status="saved",
                    saved_face_index=int(result.face_index),
                    payload=result,
                )
            )
        self._show_face_result_groups(
            [
                FaceResultGroup(
                    group_id=f"match:{uuid4()}",
                    title=title,
                    summary=summary,
                    items=tuple(items),
                )
            ],
            summary=summary,
            photo_paths=photo_paths,
            overlay_by_path=overlay_by_path,
            context_by_path=context_by_path,
            kind="faces",
        )

    def _face_cluster_backend_label(self, backend_id: str) -> str:
        for item_id, label in face_cluster_backend_choices():
            if str(item_id) == str(backend_id):
                return str(label)
        return str(backend_id or "").replace("-", " ").title()

    def _show_face_cluster_results(
        self,
        cluster_result: FaceClusteringComparisonResult | dict[int, list[FaceClusterMember]],
        *,
        summary: str,
        compare_refs: list[tuple[str, int]] | None = None,
    ) -> None:
        groups: list[FaceResultGroup] = []
        overlay_by_path: dict[str, str] = {}
        context_by_path: dict[str, dict[str, object]] = {}
        photo_paths: list[str] = []
        clusters_by_key: dict[str, dict[int, list[FaceClusterMember]]]
        explanations_by_key: dict[str, dict[int, ClusterExplanation]]
        suggestions_by_key: dict[str, dict[int, FaceClusterIdentitySuggestion]]
        membership_by_face_ref: dict[tuple[str, int], dict[str, dict[str, object]]]
        metrics_by_key: dict[str, dict[str, object]]
        if isinstance(cluster_result, FaceClusteringComparisonResult):
            clusters_by_key = dict(cluster_result.clusters_by_key)
            explanations_by_key = dict(cluster_result.explanations_by_key)
            suggestions_by_key = dict(cluster_result.identity_suggestions_by_key)
            membership_by_face_ref = dict(cluster_result.membership_by_face_ref)
            metrics_by_key = dict(cluster_result.metrics_by_key)
            self._face_cluster_compare_last_refs = list(compare_refs or [])
        else:
            clusters_by_key = {"hdbscan": dict(cluster_result or {})}
            explanations_by_key = {}
            suggestions_by_key = {}
            membership_by_face_ref = {}
            metrics_by_key = {}
            self._face_cluster_compare_last_refs = []
        self._face_cluster_compare_membership = membership_by_face_ref
        self._face_cluster_compare_metrics = metrics_by_key
        for comparison_key, clusters in sorted(clusters_by_key.items(), key=lambda item: str(item[0])):
            backend_label = self._face_cluster_backend_label(comparison_key)
            for cluster_id, members in sorted(clusters.items()):
                ordered_members = list(members)
                ordered_members.sort(
                    key=lambda member: (
                        int(
                            (
                                membership_by_face_ref.get((str(member.image_path), int(member.face_index)), {})
                                .get(str(comparison_key), {})
                                .get("rank", 1_000_000)
                            )
                        ),
                        {"clean": 0, "review": 1, "reject": 2}.get(str(member.quality_status or "clean").strip().lower(), 3),
                        -float(member.face_confidence),
                        str(member.image_path),
                        int(member.face_index),
                    )
                )
                items: list[FaceTileItem] = []
                outlier = int(cluster_id) == -1
                suggestion = (suggestions_by_key.get(comparison_key, {}) or {}).get(int(cluster_id))
                explanation = (explanations_by_key.get(comparison_key, {}) or {}).get(int(cluster_id))
                for member in ordered_members:
                    photo_paths.append(member.image_path)
                    overlay_by_path.setdefault(member.image_path, f"{backend_label} | cluster {int(cluster_id)}")
                    context_by_path.setdefault(
                        member.image_path,
                        {"face_cluster": {"cluster_id": int(cluster_id), "backend": comparison_key}},
                    )
                    label = member.person_name or "Unlabeled"
                    items.append(
                        FaceTileItem(
                            image_path=str(member.image_path),
                            face_index=int(member.face_index),
                            bbox=tuple(int(value) for value in member.face_bbox),
                            title=f"{label}\n{Path(member.image_path).name}",
                            tooltip=(
                                f"{member.image_path}\nface #{int(member.face_index) + 1}\n"
                                f"bbox={tuple(int(value) for value in member.face_bbox)}\n"
                                f"conf={float(member.face_confidence):.3f}\nquality={member.quality_status}\n"
                                f"backend={comparison_key}\ncluster={int(cluster_id)}"
                            ),
                            status=str(member.quality_status or "saved"),
                            saved_face_index=int(member.face_index),
                            payload=member,
                        )
                    )
                suggestion_text = ""
                if suggestion is not None:
                    suggestion_text = (
                        f" | likely {suggestion.person_name} ({suggestion.support_count}/{suggestion.member_count}, "
                        f"{suggestion.mean_score:.3f})"
                    )
                cluster_names = sorted(
                    {
                        str(member.person_name or "").strip()
                        for member in ordered_members
                        if str(member.person_name or "").strip()
                    },
                    key=str.casefold,
                )
                cluster_title = f"{backend_label} | {'Outlier' if outlier else f'Cluster {int(cluster_id)}'}"
                if cluster_names:
                    cluster_title = f"{cluster_title} | {', '.join(cluster_names)}"
                group_summary = (
                    f"{backend_label} | {'Outlier' if outlier else f'Cluster {int(cluster_id)}'} | "
                    f"{len(items)} face tile(s){suggestion_text}"
                )
                if cluster_names:
                    group_summary += f" | names: {', '.join(cluster_names)}"
                groups.append(
                    FaceResultGroup(
                        group_id=f"{comparison_key}:{int(cluster_id)}",
                        title=cluster_title,
                        summary=group_summary,
                        items=tuple(items),
                        comparison_key=str(comparison_key),
                        cluster_id=int(cluster_id),
                        explanation=explanation,
                        suggestion=suggestion,
                    )
                )
        self._show_face_result_groups(
            groups,
            summary=summary,
            photo_paths=photo_paths,
            overlay_by_path=overlay_by_path,
            context_by_path=context_by_path,
            kind="face_clusters",
        )

    def _search_face_ref(self, image_path: str, face_index: int, *, source_label: str) -> None:
        folder = self.face_name_folder_filter.text().strip() if hasattr(self, "face_name_folder_filter") else ""
        scope_paths = self._current_scope_paths()
        if not folder and self.search_only_current_folder.isChecked():
            folder = self._current_directory()
        effective_min_score = self._effective_face_search_min_score(float(self.face_browser_min_score.value()))

        def _run(progress, cancel_check):
            progress(-1, "Searching selected face...")
            return _call_with_optional_cancel(
                self._active_face_service().search_similar_face,
                str(image_path),
                int(face_index),
                top_k=int(self.face_browser_top_k.value()),
                min_score=effective_min_score,
                folder_prefix=folder,
                candidate_paths=scope_paths,
                include_tiny_faces=self._show_tiny_detections_enabled(),
                cancel_check=cancel_check,
            )

        def _done(results) -> None:
            self.status_label.setText(f"Found {len(results)} matches for the selected face.")
            self._show_face_match_results(
                list(results or []),
                title=f"{source_label} matches",
                summary=f"{source_label} returned {len(results)} matching face tile(s).",
                match_label="match",
            )

        self._start_job("Searching selected face", _run, _done)

    def _search_selected_detected_face(self) -> None:
        refs = self._selected_detected_face_indexed_refs()
        if not refs:
            errorBox("No faces selected", "Select one or more saved face tiles in Detected Faces first.")
            return
        self._search_face_refs(refs, source_label="Detected Faces")

    def _cluster_face_refs(self, face_refs: list[tuple[str, int]], *, source_label: str) -> None:
        refs = list(dict.fromkeys((str(image_path), int(face_index)) for image_path, face_index in face_refs if str(image_path or "").strip()))
        if len(refs) < 2:
            errorBox("Not enough faces", "Select at least two saved face tiles first.")
            return
        num_clusters = min(self.face_cluster_count.value(), len(refs))
        min_score = self.face_min_score.value()
        backends = list(self.current_face_cluster_backends())
        backend = str(backends[0] if backends else "hdbscan")
        backend_options = self.current_face_cluster_backend_options()
        outlier_policy = self.current_face_cluster_outlier_policy()

        def _run(progress, cancel_check):
            progress(-1, "Clustering faces...")
            return _call_with_optional_cancel(
                self._active_face_service().cluster_faces_compare,
                num_clusters=num_clusters,
                min_face_score=min_score,
                backends=backends,
                outlier_policy=outlier_policy,
                backend_options_by_backend={str(item): dict(backend_options) for item in backends if str(item) == "hdbscan"},
                face_refs=refs,
                include_tiny_faces=self._show_tiny_detections_enabled(),
                cancel_check=cancel_check,
            )

        def _done(result) -> None:
            cluster_map = dict(getattr(result, "clusters_by_key", {}) or {})
            cluster_count = sum(len(value) for value in cluster_map.values())
            backend_label = self._face_cluster_backend_label(backend)
            self.status_label.setText(
                f"Generated {cluster_count} {backend_label} face cluster group(s) from {source_label.lower()}."
            )
            self._show_face_cluster_results(
                result,
                summary=f"{source_label} produced {cluster_count} {backend_label} face cluster group(s).",
                compare_refs=refs,
            )
            self.face_clusters_ready.emit(cluster_map)

        self._start_job("Clustering faces", _run, _done)

    def _cluster_selected_detected_faces(self) -> None:
        self._cluster_face_refs(self._selected_detected_face_indexed_refs(), source_label="Detected Faces selection")

    def _cluster_visible_detected_faces(self) -> None:
        refs: list[tuple[str, int]] = []
        for item in list(self._face_review_images or []):
            if str(item.image_path or "") in self._face_review_draft_dirty_paths:
                continue
            for record in tuple(item.visible_faces or ()):
                refs.append((str(record.image_path), int(record.face_index)))
        self._cluster_face_refs(refs, source_label="Detected Faces visible scope")

    def _jump_to_selected_face_result(self) -> None:
        refs = self._selected_face_result_tiles()
        if not refs:
            return
        image_path = str(refs[0][0] or "")
        group = self._current_face_result_group()
        title = f"{group.title.splitlines()[0]} photo" if group is not None else "Selected result photo"
        self._show_derived_face_photos([image_path], title=title, focus_path=image_path)

    def _open_selected_face_result_in_inspector(self) -> None:
        refs = self._selected_face_result_tiles()
        if not refs:
            return
        image_path = str(refs[0][0] or "")
        self._show_derived_face_photos([image_path], title="Selected result photo", focus_path=image_path)
        QTimer.singleShot(
            0,
            lambda path=image_path: self._open_results_gallery_inspector(path, allow_face_edit=False),
        )

    def _recluster_selected_face_result_group(self) -> None:
        group = self._current_face_result_raw_group()
        if group is None or not group.items:
            return
        refs = self._face_result_group_refs(group, prefer_selection=True)
        if len(refs) < 2:
            errorBox("Not enough faces", "Select a cluster with at least two saved faces to recluster it.")
            return
        use_subset_label = len(refs) < len(group.items)
        self._cluster_face_refs(
            refs,
            source_label=f"{group.title} selected subset" if use_subset_label else group.title,
        )

    def _rerun_face_cluster_compare(self) -> None:
        if not self._face_cluster_compare_last_refs:
            return
        self._cluster_face_refs(list(self._face_cluster_compare_last_refs), source_label="Compared face selection")

    def _write_exif_name_from_face_result_group(self) -> None:
        group = self._current_face_result_group()
        if group is None or self._face_result_active_group_kind != "merged_name":
            errorBox("Select a merged group", "Choose a merged-by-name group first.")
            return
        resolved_name = str(group.resolved_name or "").strip()
        if not resolved_name:
            errorBox("Missing name", "Only merged groups with a resolved name can be written to EXIF.")
            return
        image_paths = list(dict.fromkeys(str(item.image_path) for item in list(group.items) if str(item.image_path or "").strip()))
        if not image_paths:
            errorBox("No photos selected", "The current merged group does not include any photos.")
            return
        action_service = getattr(self.results_gallery, "action_service", None) or GalleryActionService()

        def _run(progress, cancel_check):
            return action_service.write_exif_metadata_pairs(
                image_paths,
                FACE_EXIF_PERSON_KEY,
                resolved_name,
                progress_callback=progress,
                cancel_check=cancel_check,
            )

        def _done(result) -> None:
            affected = len(getattr(result, "affected_paths", []) or [])
            self.status_label.setText(f"Wrote EXIF name '{resolved_name}' to {affected} photo(s).")
            self._apply_face_result_filters()

        self._start_job("Writing EXIF names", _run, _done)

    def _queue_selected_face_result_group_suggestion(self) -> None:
        group = self._current_face_result_raw_group()
        if group is None or group.suggestion is None:
            errorBox("No suggestion available", "Select a cluster with a likely identity suggestion first.")
            return
        refs = self._face_result_group_refs(group, prefer_selection=False)
        if not refs:
            errorBox("No saved faces", "The selected cluster does not contain any saved indexed faces.")
            return
        self._save_face_refs_name(
            refs,
            source_label=f"{group.title} suggestion",
            person_name_override=str(group.suggestion.person_name or "").strip(),
        )

    def _open_face_result_pending_review(self) -> None:
        self._show_face_search_tab()
        if hasattr(self, "face_pending_items_toggle") and hasattr(self, "face_pending_items_panel"):
            self._set_expander_state(self.face_pending_items_toggle, self.face_pending_items_panel, True)
        self._refresh_pending_face_labels()

    def _reject_selected_face_result_group_suggestion(self) -> None:
        group = self._current_face_result_raw_group()
        if group is None or group.suggestion is None:
            errorBox("No suggestion available", "Select a cluster with a likely identity suggestion first.")
            return
        replacement = FaceResultGroup(
            group_id=group.group_id,
            title=group.title,
            summary=f"{group.title} suggestion dismissed for this review session.",
            items=tuple(group.items),
            comparison_key=group.comparison_key,
            cluster_id=group.cluster_id,
            explanation=group.explanation,
            suggestion=None,
            cluster_status="suggestion dismissed",
        )
        self._face_result_groups = [replacement if item.group_id == group.group_id else item for item in self._face_result_groups]
        self._face_result_source_groups = list(self._face_result_groups)
        self._face_result_source_summary = f"Dismissed the suggested identity for {group.title}."
        self._apply_face_result_filters()

    def _mark_selected_face_result_group_unknown(self) -> None:
        group = self._current_face_result_raw_group()
        if group is None or not group.items:
            return
        replacement = FaceResultGroup(
            group_id=group.group_id,
            title=group.title,
            summary=f"{group.title} is intentionally kept unlabeled for this review session.",
            items=tuple(group.items),
            comparison_key=group.comparison_key,
            cluster_id=group.cluster_id,
            explanation=group.explanation,
            suggestion=None,
            cluster_status="kept unlabeled",
        )
        self._face_result_groups = [replacement if item.group_id == group.group_id else item for item in self._face_result_groups]
        self._face_result_source_groups = list(self._face_result_groups)
        self._face_result_source_summary = f"Marked {group.title} as intentionally unlabeled."
        self._apply_face_result_filters()

    def _split_selected_face_result_group(self) -> None:
        group = self._current_face_result_raw_group()
        if group is None or len(group.items) < 2:
            return
        selected_refs = set(self._selected_face_result_tiles())
        selected_items = [
            item for item in group.items
            if self._face_ref_from_tile_item(item) in selected_refs
        ]
        if not selected_items or len(selected_items) >= len(group.items):
            errorBox("Select a subset", "Select one or more faces inside the current cluster, but not the entire cluster.")
            return
        remaining_items = tuple(item for item in group.items if item not in selected_items)
        split_group = FaceResultGroup(
            group_id=f"{group.group_id}:split:{uuid4()}",
            title=f"{group.title} | Split",
            summary=f"Manual split from {group.title} with {len(selected_items)} face(s).",
            items=tuple(selected_items),
            comparison_key=group.comparison_key,
            cluster_id=group.cluster_id,
            cluster_status="manual split",
        )
        retained_group = FaceResultGroup(
            group_id=group.group_id,
            title=group.title,
            summary=f"{group.title} after manual split with {len(remaining_items)} remaining face(s).",
            items=remaining_items,
            comparison_key=group.comparison_key,
            cluster_id=group.cluster_id,
            explanation=group.explanation,
            suggestion=group.suggestion,
            cluster_status="manual split",
        )
        next_groups: list[FaceResultGroup] = []
        for item in self._face_result_groups:
            if item.group_id == group.group_id:
                next_groups.append(retained_group)
                next_groups.append(split_group)
            else:
                next_groups.append(item)
        self._face_result_groups = next_groups
        self._face_result_source_groups = list(self._face_result_groups)
        self._face_result_source_summary = f"Split {group.title} into two session-local groups."
        self._face_result_active_group_kind = "raw"
        self._face_result_selected_group_id = split_group.group_id
        self._face_result_selected_group_id_by_kind["raw"] = split_group.group_id
        self._apply_face_result_filters()

    def _merge_selected_face_result_groups(self) -> None:
        selected_group_ids = self._selected_face_result_group_ids()
        if len(selected_group_ids) != 2:
            errorBox("Select two groups", "Select exactly two face result groups from the same backend first.")
            return
        groups = [self._face_result_group_by_id.get(group_id) for group_id in selected_group_ids]
        if any(group is None for group in groups):
            return
        left, right = groups  # type: ignore[misc]
        if not left.comparison_key or str(left.comparison_key) != str(right.comparison_key):
            errorBox("Mismatched groups", "Only groups from the same backend can be merged.")
            return
        merged_items: list[FaceTileItem] = []
        seen_refs: set[tuple[str, int]] = set()
        for item in list(left.items) + list(right.items):
            ref = self._face_ref_from_tile_item(item)
            if ref is not None and ref in seen_refs:
                continue
            if ref is not None:
                seen_refs.add(ref)
            merged_items.append(item)
        merged_group = FaceResultGroup(
            group_id=f"{left.comparison_key}:merged:{uuid4()}",
            title=f"{self._face_cluster_backend_label(left.comparison_key)} | Manual Merge",
            summary=f"Merged {left.title} and {right.title} into one session-local group with {len(merged_items)} face(s).",
            items=tuple(merged_items),
            comparison_key=left.comparison_key,
            cluster_id=-1,
            cluster_status="manual merge",
        )
        remove_ids = {left.group_id, right.group_id}
        self._face_result_groups = [item for item in self._face_result_groups if item.group_id not in remove_ids]
        self._face_result_groups.append(merged_group)
        self._face_result_source_groups = list(self._face_result_groups)
        self._face_result_source_summary = f"Merged {left.title} and {right.title}."
        self._face_result_active_group_kind = "raw"
        self._face_result_selected_group_id = merged_group.group_id
        self._face_result_selected_group_id_by_kind["raw"] = merged_group.group_id
        self._apply_face_result_filters()

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

    def _export_face_cluster_results(self) -> None:
        groups = [group for group in list(self._face_result_source_groups or []) if str(group.comparison_key or "").strip() or int(group.cluster_id) >= 0]
        if not groups:
            errorBox("No cluster results", "Run clustering first, then export the current cluster result set.")
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Export Face Cluster Results", "", "JSON Files (*.json)")
        if not out_path:
            return
        payload = {
            "summary": str(self._face_result_source_summary or ""),
            "groups": [
                {
                    "group_id": group.group_id,
                    "title": group.title,
                    "comparison_key": group.comparison_key,
                    "cluster_id": int(group.cluster_id),
                    "summary": group.summary,
                    "items": [
                        {
                            "image_path": str(item.image_path),
                            "face_index": int(item.face_index),
                            "bbox": [int(value) for value in item.bbox],
                        }
                        for item in group.items
                    ],
                }
                for group in groups
            ],
        }
        try:
            Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.status_label.setText(f"Exported {len(groups)} face cluster group(s) to {out_path}.")
        except Exception as exc:
            errorBox("Export failed", str(exc))

    def _export_face_identities(self) -> None:
        service = self._active_face_service()
        out_path, _ = QFileDialog.getSaveFileName(self, "Export Face Identities", "", "JSON Files (*.json)")
        if not out_path:
            return
        export_fn = getattr(service, "export_identity_data", None)
        if callable(export_fn):
            try:
                payload = dict(export_fn())
                Path(out_path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
                self.status_label.setText(f"Exported {len(payload.get('identities', []) or [])} identity record(s) to {out_path}.")
                return
            except Exception as exc:
                errorBox("Export failed", str(exc))
                return
        try:
            profiles = list(service.load_person_profiles(limit=max(1, int(getattr(self, "face_browser_limit", self.face_name_top_k).value()))))
        except Exception:
            profiles = []
        payload = {
            "identities": [
                {
                    "person_name": str(profile.person_name),
                    "similarity_threshold": float(profile.similarity_threshold),
                    "example_count": int(profile.example_count),
                    "labeled_count": int(profile.labeled_count),
                    "visible_face_count": int(profile.visible_face_count),
                    "notes": str(profile.notes or ""),
                    "tags": list(profile.tags or ()),
                    "favorite": bool(getattr(profile, "favorite", False)),
                    "birth_date": str(getattr(profile, "birth_date", "") or ""),
                    "hidden": bool(getattr(profile, "hidden", False)),
                    "prototype_refs": [
                        {
                            "image_path": str(face.image_path),
                            "face_index": int(face.face_index),
                            "pinned": bool(face.pinned),
                        }
                        for face in list(getattr(service, "load_person_prototype_faces", lambda *_args, **_kwargs: [])(str(profile.person_name), limit=200, include_tiny_faces=True) or [])
                    ],
                }
                for profile in profiles
            ]
        }
        try:
            Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.status_label.setText(f"Exported {len(payload['identities'])} identity record(s) to {out_path}.")
        except Exception as exc:
            errorBox("Export failed", str(exc))

    def _import_face_identities(self) -> None:
        service = self._active_face_service()
        in_path, _ = QFileDialog.getOpenFileName(self, "Import Face Identities", "", "JSON Files (*.json)")
        if not in_path:
            return
        try:
            payload = json.loads(Path(in_path).read_text(encoding="utf-8"))
            preview = self._preview_identity_import(service, payload)
            mode = self._choose_identity_import_mode(preview)
            if not mode:
                self.status_label.setText("Identity import cancelled. No face database changes were made.")
                return
            import_fn = getattr(service, "import_identity_data", None)
            if not callable(import_fn):
                raise RuntimeError("The active face service does not support identity import.")
            import_fn(payload, mode=mode)
            self._refresh_face_identities(force_reload=True)
            self._request_face_library_refresh(refresh_people=True, reason="identities imported", force_refresh=True)
            self._maybe_refresh_global_face_album(reason="identities imported")
            self.status_label.setText(f"Imported face identities from {in_path} using {mode} mode.")
            infoBox("Import complete", "Face identities were imported into the active face database.")
        except Exception as exc:
            errorBox("Import failed", str(exc))

    def _preview_identity_import(self, service, payload: dict[str, object]) -> dict[str, object]:
        preview_fn = getattr(service, "preview_identity_import", None)
        if callable(preview_fn):
            return dict(preview_fn(payload) or {})
        identities = list(payload.get("identities", []) or [])
        labels = list(payload.get("labels", []) or [])
        face_rows = list(payload.get("face_index", []) or [])
        corrections = list(payload.get("corrections", []) or [])
        hidden_items = sum(1 for item in identities if isinstance(item, dict) and bool(item.get("hidden", False)))
        hidden_items += sum(1 for item in face_rows if isinstance(item, dict) and bool(item.get("hidden", False)))
        return {
            "counts": {
                "identities": len(identities),
                "labels": len(labels),
                "prototypes": sum(len(list(item.get("prototype_refs", []) or [])) for item in identities if isinstance(item, dict)),
                "hidden_items": hidden_items,
                "corrections": len(corrections),
                "pending_labels": len(list(payload.get("pending_labels", []) or [])),
                "face_index": len(face_rows),
            },
            "conflicts": [],
            "conflict_count": 0,
        }

    def _choose_identity_import_mode(self, preview: dict[str, object]) -> str:
        counts = dict(preview.get("counts", {}) or {})
        conflict_count = int(preview.get("conflict_count", len(list(preview.get("conflicts", []) or []))) or 0)
        message = (
            "Preview identity import:\n"
            f"Identities: {int(counts.get('identities', 0) or 0)}\n"
            f"Labels: {int(counts.get('labels', 0) or 0)}\n"
            f"Prototypes: {int(counts.get('prototypes', 0) or 0)}\n"
            f"Hidden items: {int(counts.get('hidden_items', 0) or 0)}\n"
            f"Corrections: {int(counts.get('corrections', 0) or 0)}\n"
            f"Conflicts: {conflict_count}\n\n"
            "Choose Yes to merge, No to replace the active face identity database, or Cancel to leave it unchanged."
        )
        choice = QMessageBox.question(
            self,
            "Import Face Identities",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        if choice == QMessageBox.StandardButton.Yes:
            return "merge"
        if choice == QMessageBox.StandardButton.No:
            return "replace"
        return ""

    def _set_face_recognition_enabled(self, enabled: bool) -> None:
        if not self._ensure_writable_face_action("change face recognition state"):
            return
        service = self._active_face_service()
        set_enabled_fn = getattr(service, "set_face_recognition_enabled", None)
        if not callable(set_enabled_fn):
            errorBox("Face recognition setting unavailable", "The active face service does not support this setting.")
            return
        if not enabled and not confirmBox(
            "Disable face recognition?",
            (
                "Target: the active saved face library\n"
                "Affected state: automatic identity recognition\n"
                "Result: saved identities remain, but automatic recognition stops.\n"
                "Recovery: recognition can be enabled again."
            ),
            parent=self,
        ):
            self.status_label.setText("Recognition change cancelled. No settings were changed.")
            return
        try:
            set_enabled_fn(bool(enabled))
            state = "enabled" if bool(enabled) else "disabled"
            self.status_label.setText(f"Face recognition {state} for the saved face library.")
        except Exception as exc:
            errorBox("Face recognition setting failed", str(exc))

    def _purge_face_data(self) -> None:
        if not self._ensure_writable_face_action("purge face data"):
            return
        service = self._active_face_service()
        purge_fn = getattr(service, "purge_face_data", None)
        if not callable(purge_fn):
            errorBox("Purge unavailable", "The active face service does not support full face-data purge.")
            return
        report = {}
        prepare_fn = getattr(service, "prepare_face_data_purge", None)
        if callable(prepare_fn):
            try:
                report = dict(prepare_fn() or {})
            except Exception:
                report = {}
        table_counts = dict(report.get("table_counts", {}) or {})
        row_count = sum(int(value or 0) for value in table_counts.values())
        ann_files = list(report.get("ann_files", []) or [])
        model_targets = list(report.get("model_cache_targets", []) or [])
        if not confirmBox(
            "Purge face data?",
            (
                "Delete stored face data for the active Faces mode?\n\n"
                f"Rows: {row_count}\n"
                f"ANN files: {len(ann_files)}\n"
                f"Model cache targets: {len(model_targets)}\n\n"
                "Source image files are not deleted."
            ),
            parent=self,
        ):
            self.status_label.setText("Face-data purge cancelled. No face data was changed.")
            return
        typed, accepted = QInputDialog.getText(
            self,
            "Confirm complete face-data purge",
            "Type PURGE to permanently remove the listed face data:",
        )
        if not accepted or str(typed).strip() != "PURGE":
            self.status_label.setText("Face-data purge cancelled. Type PURGE exactly to continue.")
            return
        try:
            result = dict(purge_fn(confirm=True) or {})
            removed_ann = len(list(result.get("removed_ann_files", []) or []))
            removed_models = len(list(result.get("removed_model_cache_targets", []) or []))
            self._face_review_by_path.clear()
            self._face_review_all_images = []
            self._set_pending_face_assignments([])
            self._refresh_face_identities(force_reload=True)
            self._request_face_library_refresh(refresh_people=True, reason="face data purged", force_refresh=True)
            self._maybe_refresh_global_face_album(reason="face data purged")
            self._refresh_face_db_usage_label()
            self._refresh_face_action_audit()
            self.status_label.setText(
                f"Purged face data. Removed {removed_ann} ANN file(s) and {removed_models} model cache target(s)."
            )
        except Exception as exc:
            errorBox("Purge failed", str(exc))

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
        self.face_review_results_tabs.setVisible(not self._external_results)
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

    def _log_face_event(self, event: str, **fields: object) -> None:
        payload = {
            "event": str(event or "").strip() or "unknown",
            "mode": self.current_face_mode(),
            "scope": self._face_db_scope_label() if hasattr(self, "face_db_scope") else "Saved face library",
            "folder": self._effective_face_folder() if hasattr(self, "face_folder_path") else "",
        }
        for key, value in fields.items():
            if value is None:
                continue
            payload[str(key)] = value
        message = " ".join(f"{key}={value}" for key, value in payload.items())
        LOGGER.info("FacePane %s", message)
        append_qt_diagnostic(f"[FacePane] {message}")

    def refresh_face_library(self, *, refresh_people: bool = True, reason: str = "") -> None:
        self._request_face_library_refresh(
            refresh_people=bool(refresh_people),
            reason=str(reason or "unspecified"),
            force_refresh=False,
        )

    def refresh_face_album(self, *, reason: str = "", force_refresh: bool = False) -> None:
        self._request_face_album_refresh(
            reason=str(reason or "unspecified"),
            force_refresh=bool(force_refresh),
        )

    def _maybe_refresh_global_face_album(self, *, reason: str, force_refresh: bool = True) -> None:
        if not hasattr(self, "_face_album_summary_text"):
            return
        if self._current_face_scope_key() != "global":
            return
        self.refresh_face_album(reason=reason, force_refresh=force_refresh)

    def _invalidate_face_discovery_cache(self, folder: str | None = None) -> None:
        target = str(folder or "").strip()
        if not target:
            self._face_discovery_cache.clear()
            return
        for key in list(self._face_discovery_cache):
            if str(key[0]).strip() == target:
                self._face_discovery_cache.pop(key, None)

    @staticmethod
    def _merge_face_review_paths(*path_groups: list[str] | tuple[str, ...] | None) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for group in path_groups:
            for raw_path in list(group or []):
                path = str(raw_path or "").strip()
                if not path or path in seen:
                    continue
                seen.add(path)
                merged.append(path)
        return merged

    def _indexed_face_review_paths(self, service: FaceIndexService, folder: str) -> list[str]:
        load_scan_records = getattr(service, "load_scan_image_records", None)
        if not callable(load_scan_records):
            return []
        try:
            scan_records = list(load_scan_records(folder_prefix=folder))
        except Exception:
            LOGGER.exception("FacePane indexed review path load failed folder=%s", folder)
            return []
        return [
            str(record.image_path)
            for record in scan_records
            if str(getattr(record, "image_path", "") or "").strip()
        ]

    def _load_face_refresh_result(
        self,
        *,
        request_id: int,
        reason: str,
        folder: str,
        include_tiny_faces: bool,
        refresh_people: bool,
        limit: int,
        candidate_paths: list[str] | None,
        discovery_complete: bool,
        service: FaceIndexService | None = None,
        summary_notice: str = "",
        status_text: str = "",
        cancel_check=None,
    ) -> dict[str, object]:
        review_service = service or self._active_face_service()
        review_kwargs = {
            "recursive": True,
            "include_tiny_faces": include_tiny_faces,
        }
        merged_paths = self._merge_face_review_paths(candidate_paths)
        if candidate_paths is not None:
            review_kwargs["candidate_paths"] = merged_paths
        review_images = _call_with_optional_cancel(
            review_service.load_folder_review_images,
            folder,
            cancel_check=cancel_check,
            **review_kwargs,
        )
        if callable(cancel_check) and cancel_check():
            raise Cancelled()
        migrated_image_paths = []
        consume_migrated = getattr(review_service, "consume_recent_migrated_face_paths", None)
        if callable(consume_migrated):
            migrated_image_paths = list(consume_migrated() or [])
        profiles = []
        if refresh_people:
            profiles = _call_with_optional_cancel(
                review_service.load_person_profiles,
                folder_prefix=folder,
                limit=max(1, int(limit or 1)),
                include_tiny_faces=include_tiny_faces,
                cancel_check=cancel_check,
            )
        return {
            "request_id": request_id,
            "reason": str(reason or "background refresh"),
            "folder": str(folder or ""),
            "include_tiny_faces": bool(include_tiny_faces),
            "refresh_people": bool(refresh_people),
            "discovered_paths": list(merged_paths),
            "review_images": review_images,
            "migrated_image_paths": list(migrated_image_paths or []),
            "profiles": profiles,
            "discovery_complete": bool(discovery_complete),
            "summary_notice": str(summary_notice or ""),
            "status_text": str(status_text or ""),
        }

    def _request_face_album_refresh(self, *, reason: str, force_refresh: bool) -> None:
        if not hasattr(self, "_face_album_summary_text"):
            return
        pending_reason = str(reason or "unspecified")
        include_tiny_faces = self._show_tiny_detections_enabled()
        self._face_album_request_id += 1
        request_id = int(self._face_album_request_id)
        self._face_album_refresh_in_progress = True
        self._cancel_face_album_refresh_job(replaced=True)
        self._log_face_event("album_refresh_requested", reason=pending_reason, request_id=request_id)
        service = self._global_face_service()

        def _run(progress, cancel_check):
            progress(-1, "Loading global detected-face album...")
            group_page_loader = getattr(service, "load_face_album_group_page", None)
            member_page_loader = getattr(service, "load_face_album_member_page", None)
            if callable(group_page_loader) and callable(member_page_loader):
                group_page = _call_with_optional_cancel(
                    group_page_loader,
                    offset=0,
                    limit=100,
                    include_tiny_faces=include_tiny_faces,
                    cancel_check=cancel_check,
                )
                member_page = None
                if group_page.items:
                    member_page = _call_with_optional_cancel(
                        member_page_loader,
                        str(group_page.items[0].group_id),
                        offset=0,
                        limit=200,
                        include_tiny_faces=include_tiny_faces,
                        cancel_check=cancel_check,
                    )
                if cancel_check():
                    raise Cancelled()
                return {
                    "request_id": request_id,
                    "reason": pending_reason,
                    "include_tiny_faces": bool(include_tiny_faces),
                    "groups": list(group_page.items),
                    "members": list(member_page.items if member_page is not None else ()),
                    "group_page": group_page,
                    "member_page": member_page,
                }
            snapshot_loader = getattr(service, "load_face_album_snapshot", None)
            if callable(snapshot_loader):
                groups, members = _call_with_optional_cancel(
                    snapshot_loader,
                    include_tiny_faces=include_tiny_faces,
                    cancel_check=cancel_check,
                )
            else:  # Compatibility for injected service adapters.
                groups = _call_with_optional_cancel(
                    service.load_face_album_groups,
                    include_tiny_faces=include_tiny_faces,
                    cancel_check=cancel_check,
                )
                members = _call_with_optional_cancel(
                    service.load_face_album_members,
                    include_tiny_faces=include_tiny_faces,
                    cancel_check=cancel_check,
                )
            if cancel_check():
                raise Cancelled()
            return {
                "request_id": request_id,
                "reason": pending_reason,
                "include_tiny_faces": bool(include_tiny_faces),
                "groups": groups,
                "members": members,
            }

        job = AsyncJob(_run)
        album_job_id: int | None = None
        if self.job_manager is not None:
            album_job_id = self.job_manager.register_job("All Faces refresh", cancel_fn=job.cancel)
            self._face_album_refresh_job_id = album_job_id
            job.progress.connect(
                lambda value, text, job_id=album_job_id: self.job_manager.update(
                    job_id,
                    progress=value,
                    text=text,
                )
            )
        job.progress.connect(lambda _value, text: self.status_label.setText(str(text)))

        def _finish_job(status: str, error: str = "") -> None:
            if self.job_manager is not None and album_job_id is not None:
                self.job_manager.finish(album_job_id, status=status, error=error)
            if self._face_album_refresh_job_id == album_job_id:
                self._face_album_refresh_job_id = None

        def _failed(message: str) -> None:
            text = str(message or "").strip()
            if text.casefold() == "cancelled":
                self._face_album_refresh_in_progress = False
                _finish_job("cancelled")
                return
            self._face_album_refresh_in_progress = False
            self.status_label.setText(f"All Faces refresh failed: {text}")
            self._log_face_event("album_refresh_failed", reason=pending_reason, request_id=request_id, error=text)
            _finish_job("failed", text)
            errorBox("All Faces refresh failed", text)

        def _cancelled() -> None:
            self._face_album_refresh_in_progress = False
            self._log_face_event("album_refresh_cancelled", reason=pending_reason, request_id=request_id)
            _finish_job("cancelled")

        def _completed(result) -> None:
            self._face_album_refresh_in_progress = False
            if not isinstance(result, dict):
                _finish_job("failed", "invalid face album result")
                return
            result_request_id = int(result.get("request_id", 0) or 0)
            if result_request_id != int(self._face_album_request_id):
                self._log_face_event("album_refresh_discarded", reason=pending_reason, request_id=result_request_id, latest_request_id=self._face_album_request_id)
                _finish_job("cancelled")
                return
            self._apply_face_album_snapshot(result)
            _finish_job("finished")

        job.completed.connect(_completed)
        job.failed.connect(_failed)
        job.cancelled.connect(_cancelled)
        self._face_album_refresh_job = job
        thread = start_job_in_thread(job)
        self._face_album_refresh_thread = thread
        self._retained_face_album_refresh_refs.append((job, thread))
        thread.finished.connect(
            lambda thread=thread: self._on_face_album_refresh_thread_finished(thread),
            Qt.ConnectionType.QueuedConnection,
        )

    def _cancel_face_album_refresh_job(self, *, replaced: bool = False) -> None:
        job = self._face_album_refresh_job
        thread = self._face_album_refresh_thread
        if job is None and thread is None:
            return
        if job is not None:
            try:
                job.cancel()
            except Exception:
                pass
        if replaced:
            self._log_face_event("album_refresh_replaced", request_id=self._face_album_request_id)
        self._face_album_refresh_job = None
        self._face_album_refresh_thread = None

    def _on_face_album_refresh_thread_finished(self, thread=None) -> None:
        retained: list[tuple[object | None, object | None]] = []
        for job, retained_thread in self._retained_face_album_refresh_refs:
            if retained_thread is thread:
                continue
            retained.append((job, retained_thread))
        self._retained_face_album_refresh_refs = retained
        if thread is self._face_album_refresh_thread:
            self._face_album_refresh_thread = None
            self._face_album_refresh_job = None

    def _apply_face_album_snapshot(self, snapshot: dict[str, object]) -> None:
        self._face_album_groups = list(snapshot.get("groups", []) or [])
        self._face_album_records = list(snapshot.get("members", []) or [])
        group_page = snapshot.get("group_page")
        member_page = snapshot.get("member_page")
        if isinstance(group_page, FaceAlbumGroupPage):
            self._face_album_total_groups = int(group_page.total_count)
            self._face_album_next_group_offset = group_page.next_offset
        else:
            self._face_album_total_groups = len(self._face_album_groups)
            self._face_album_next_group_offset = None
        self._face_album_loaded_group_ids = set()
        self._face_album_member_next_offsets = {}
        self._face_album_member_totals = {}
        if isinstance(member_page, FaceAlbumMemberPage):
            group_id = str(member_page.group_id)
            self._face_album_loaded_group_ids.add(group_id)
            self._face_album_member_next_offsets[group_id] = member_page.next_offset
            self._face_album_member_totals[group_id] = int(member_page.total_count)
        elif self._face_album_records:
            for group in self._face_album_groups:
                group_id = str(group.group_id)
                self._face_album_loaded_group_ids.add(group_id)
                self._face_album_member_next_offsets[group_id] = None
                self._face_album_member_totals[group_id] = int(group.face_count)
        self._face_album_loaded = True
        named_groups = [group for group in self._face_album_groups if str(getattr(group, "group_kind", "")) == "named"]
        total_faces = sum(int(getattr(group, "face_count", 0) or 0) for group in self._face_album_groups)
        total_photos = len({str(record.image_path) for record in self._face_album_records})
        bucket_bits = [
            group.title.split("|", 1)[0].strip()
            for group in self._face_album_groups
            if str(getattr(group, "group_kind", "")) != "named"
        ]
        summary = (
            f"Showing {len(self._face_album_groups)} of {self._face_album_total_groups} group(s) "
            f"from the saved face library ({total_faces} face(s) in the loaded group summaries)."
        )
        if named_groups:
            summary += f" {len(named_groups)} named identity group(s)."
        if total_photos:
            summary += f" {total_photos} photo(s) represented."
        if bucket_bits:
            summary += f" Buckets: {', '.join(bucket_bits)}."
        self._face_album_summary_text = summary
        self._update_face_album_paging_actions()
        self._log_face_event("album_refresh_complete", reason=str(snapshot.get("reason", "") or ""), faces=total_faces, groups=len(self._face_album_groups))
        if self._is_all_faces_tab_label(self.tabs.tabText(self.tabs.currentIndex())):
            self._request_face_album_publish(summary)

    def _show_face_album_results(self, summary_text: str | None = None) -> None:
        self._request_face_album_publish(str(summary_text or self._face_album_summary_text or "All Faces album"))

    def _update_face_album_paging_actions(self) -> None:
        load_groups = getattr(self, "face_album_load_more_groups_button", None)
        if load_groups is not None:
            load_groups.setEnabled(self._face_album_next_group_offset is not None and not self._face_album_refresh_in_progress)
        load_faces = getattr(self, "face_album_load_more_faces_button", None)
        if load_faces is not None:
            selected_group_id = str(self._face_result_selected_group_id or "")
            next_offset = self._face_album_member_next_offsets.get(selected_group_id)
            load_faces.setEnabled(next_offset is not None and not self._face_album_refresh_in_progress)

    def _start_face_album_page_job(self, label: str, run, apply_result) -> None:
        request_id = int(self._face_album_request_id)
        self._face_album_refresh_in_progress = True
        self._cancel_face_album_refresh_job(replaced=True)
        job = AsyncJob(run)
        album_job_id: int | None = None
        if self.job_manager is not None:
            album_job_id = self.job_manager.register_job(label, cancel_fn=job.cancel)
            self._face_album_refresh_job_id = album_job_id
            job.progress.connect(
                lambda value, text, job_id=album_job_id: self.job_manager.update(
                    job_id,
                    progress=value,
                    text=text,
                )
            )
        job.progress.connect(lambda _value, text: self.status_label.setText(str(text)))

        def _finish(status: str, error: str = "") -> None:
            self._face_album_refresh_in_progress = False
            if self.job_manager is not None and album_job_id is not None:
                self.job_manager.finish(album_job_id, status=status, error=error)
            if self._face_album_refresh_job_id == album_job_id:
                self._face_album_refresh_job_id = None
            self._update_face_album_paging_actions()

        def _failed(message: str) -> None:
            text = str(message or "").strip()
            _finish("cancelled" if text.casefold() == "cancelled" else "failed", text)
            if text.casefold() != "cancelled":
                self.status_label.setText(f"{label} failed: {text}")

        def _completed(result) -> None:
            if request_id != int(self._face_album_request_id):
                _finish("cancelled")
                return
            apply_result(result)
            _finish("finished")

        job.failed.connect(_failed)
        job.cancelled.connect(lambda: _finish("cancelled"))
        job.completed.connect(_completed)
        self._face_album_refresh_job = job
        thread = start_job_in_thread(job)
        self._face_album_refresh_thread = thread
        self._retained_face_album_refresh_refs.append((job, thread))
        thread.finished.connect(
            lambda thread=thread: self._on_face_album_refresh_thread_finished(thread),
            Qt.ConnectionType.QueuedConnection,
        )
        self._update_face_album_paging_actions()

    def _load_more_face_album_groups(self) -> None:
        offset = self._face_album_next_group_offset
        if offset is None:
            return
        service = self._global_face_service()
        loader = getattr(service, "load_face_album_group_page", None)
        if not callable(loader):
            return
        include_tiny_faces = self._show_tiny_detections_enabled()

        def _run(progress, cancel_check):
            progress(-1, "Loading more face groups...")
            page = _call_with_optional_cancel(
                loader,
                offset=int(offset),
                limit=100,
                include_tiny_faces=include_tiny_faces,
                cancel_check=cancel_check,
            )
            if cancel_check():
                raise Cancelled()
            return page

        def _apply(page) -> None:
            if not isinstance(page, FaceAlbumGroupPage):
                return
            existing_ids = {str(group.group_id) for group in self._face_album_groups}
            self._face_album_groups.extend(group for group in page.items if str(group.group_id) not in existing_ids)
            self._face_album_total_groups = int(page.total_count)
            self._face_album_next_group_offset = page.next_offset
            summary = (
                f"Showing {len(self._face_album_groups)} of {self._face_album_total_groups} face groups. "
                "Select a group to load its faces."
            )
            self._face_album_summary_text = summary
            self._request_face_album_publish(summary)

        self._start_face_album_page_job("Loading more face groups", _run, _apply)

    def _load_face_album_group_members(self, group_id: str, *, append: bool) -> None:
        target_group_id = str(group_id or "").strip()
        if not target_group_id:
            return
        if not append and target_group_id in self._face_album_loaded_group_ids:
            self._update_face_album_paging_actions()
            return
        offset = self._face_album_member_next_offsets.get(target_group_id, 0) if append else 0
        if offset is None:
            return
        service = self._global_face_service()
        loader = getattr(service, "load_face_album_member_page", None)
        if not callable(loader):
            return
        include_tiny_faces = self._show_tiny_detections_enabled()

        def _run(progress, cancel_check):
            progress(-1, "Loading faces for the selected group...")
            page = _call_with_optional_cancel(
                loader,
                target_group_id,
                offset=int(offset),
                limit=200,
                include_tiny_faces=include_tiny_faces,
                cancel_check=cancel_check,
            )
            if cancel_check():
                raise Cancelled()
            return page

        def _apply(page) -> None:
            if not isinstance(page, FaceAlbumMemberPage):
                return
            if append:
                existing_refs = {
                    (str(record.group_id), str(record.image_path), int(record.face_index))
                    for record in self._face_album_records
                }
                self._face_album_records.extend(
                    record
                    for record in page.items
                    if (str(record.group_id), str(record.image_path), int(record.face_index)) not in existing_refs
                )
            else:
                self._face_album_records = [
                    record for record in self._face_album_records if str(record.group_id) != target_group_id
                ]
                self._face_album_records.extend(page.items)
            self._face_album_loaded_group_ids.add(target_group_id)
            self._face_album_member_next_offsets[target_group_id] = page.next_offset
            self._face_album_member_totals[target_group_id] = int(page.total_count)
            loaded_count = sum(1 for record in self._face_album_records if str(record.group_id) == target_group_id)
            summary = f"Loaded {loaded_count} of {int(page.total_count)} face(s) for the selected group."
            self._face_album_summary_text = summary
            self._request_face_album_publish(summary)

        self._start_face_album_page_job("Loading face group", _run, _apply)

    def _load_more_selected_face_album_members(self) -> None:
        self._load_face_album_group_members(str(self._face_result_selected_group_id or ""), append=True)

    def _cancel_face_album_publish_job(self) -> None:
        job = self._face_album_publish_job
        if job is not None:
            try:
                job.cancel()
            except Exception:
                pass
        self._face_album_publish_job = None
        self._face_album_publish_thread = None

    def _on_face_album_publish_thread_finished(self, thread=None) -> None:
        retained: list[tuple[object | None, object | None]] = []
        for job, retained_thread in self._retained_face_album_publish_refs:
            if retained_thread is thread:
                continue
            retained.append((job, retained_thread))
        self._retained_face_album_publish_refs = retained
        if thread is self._face_album_publish_thread:
            self._face_album_publish_thread = None
            self._face_album_publish_job = None

    def _request_face_album_publish(self, summary_text: str) -> None:
        groups = list(self._face_album_groups or [])
        records = list(self._face_album_records or [])
        if not groups and not records:
            return
        self._cancel_face_album_publish_job()
        request_id = int(self._face_album_request_id)

        def _run(progress, cancel_check):
            progress(-1, "Preparing global detected-face album...")
            return self._build_face_album_publish_snapshot(
                request_id=request_id,
                groups=groups,
                records=records,
                summary_text=summary_text,
                cancel_check=cancel_check,
            )

        job = AsyncJob(_run)
        job.progress.connect(lambda _value, text: self.status_label.setText(str(text)))

        def _failed(message: str) -> None:
            text = str(message or "").strip()
            if text.casefold() == "cancelled":
                return
            self.status_label.setText(f"All Faces publish failed: {text}")

        def _completed(result) -> None:
            if not isinstance(result, FaceAlbumPublishSnapshot):
                return
            if int(result.request_id) != int(self._face_album_request_id):
                return
            self._apply_face_album_publish_snapshot(result)

        job.failed.connect(_failed)
        job.completed.connect(_completed)
        self._face_album_publish_job = job
        thread = start_job_in_thread(job)
        self._face_album_publish_thread = thread
        self._retained_face_album_publish_refs.append((job, thread))
        thread.finished.connect(
            lambda thread=thread: self._on_face_album_publish_thread_finished(thread),
            Qt.ConnectionType.QueuedConnection,
        )

    def _build_face_album_publish_snapshot(
        self,
        *,
        request_id: int,
        groups: list[FaceAlbumGroupSummary],
        records: list[FaceAlbumRecord],
        summary_text: str,
        cancel_check,
    ) -> FaceAlbumPublishSnapshot:
        records_by_group: dict[str, list[FaceAlbumRecord]] = {}
        records_by_path: dict[str, list[FaceAlbumRecord]] = {}
        for record in list(records or []):
            if cancel_check():
                raise Cancelled()
            records_by_group.setdefault(str(record.group_id), []).append(record)
            records_by_path.setdefault(str(record.image_path), []).append(record)
        result_groups: list[FaceResultGroup] = []
        for group in list(groups or []):
            if cancel_check():
                raise Cancelled()
            group_id = str(group.group_id)
            members = list(records_by_group.get(group_id, []))
            items: list[FaceTileItem] = []
            for member in members:
                primary_line = str(member.display_name or member.person_name or "Unlabeled").strip()
                if str(member.group_kind) == "named":
                    primary_line = Path(str(member.image_path)).name
                secondary_line = f"Face #{int(member.face_index) + 1}"
                if str(member.group_kind) == "pending" and primary_line:
                    secondary_line = f"{Path(str(member.image_path)).name} | pending"
                items.append(
                    FaceTileItem(
                        image_path=str(member.image_path),
                        face_index=int(member.face_index),
                        bbox=tuple(int(value) for value in member.face_bbox),
                        title=f"{primary_line}\n{secondary_line}",
                        tooltip=(
                            f"{member.image_path}\nface #{int(member.face_index) + 1}\n"
                            f"name={member.person_name or 'unlabeled'}\n"
                            f"quality={member.quality_status} score={float(member.quality_score):.2f}"
                        ),
                        status=str(member.quality_status or "clean"),
                        saved_face_index=int(member.face_index),
                        payload=member,
                    )
                )
            result_groups.append(
                FaceResultGroup(
                    group_id=group_id,
                    title=str(group.title),
                    summary=str(group.summary),
                    items=tuple(items),
                )
            )
        overlay_by_path: dict[str, str] = {}
        context_by_path: dict[str, dict[str, object]] = {}
        photo_paths: list[str] = []
        for image_path, image_records in records_by_path.items():
            photo_paths.append(str(image_path))
            photo_count = len(image_records)
            overlay_by_path[str(image_path)] = f"{photo_count} face{'s' if photo_count != 1 else ''}"
            context_by_path[str(image_path)] = {
                "indexed_faces": {
                    "count": photo_count,
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
        return FaceAlbumPublishSnapshot(
            request_id=int(request_id),
            summary=str(summary_text or "All Faces album"),
            groups=tuple(result_groups),
            photo_paths=tuple(photo_paths),
            overlay_by_path=overlay_by_path,
            context_by_path=context_by_path,
        )

    def _apply_face_album_publish_snapshot(self, snapshot: FaceAlbumPublishSnapshot) -> None:
        self._show_face_result_groups(
            list(snapshot.groups),
            summary=str(snapshot.summary or "All Faces album"),
            photo_paths=list(snapshot.photo_paths),
            overlay_by_path=dict(snapshot.overlay_by_path or {}),
            context_by_path=dict(snapshot.context_by_path or {}),
            kind="face_album",
        )
        self.status_label.setText("Loaded the saved detected-face album.")
        if not snapshot.groups:
            self.face_results_summary.setText("The saved detected-face album is empty. Scan a folder to add faces.")


    def _request_face_library_refresh(self, *, refresh_people: bool, reason: str, force_refresh: bool) -> None:
        pending_reason = str(reason or "unspecified")
        folder = self._effective_face_folder()
        self._update_face_scope_summary(folder=folder)
        self._face_refresh_request_id += 1
        request_id = int(self._face_refresh_request_id)
        self._face_refresh_people_requested = bool(refresh_people)
        self._face_refresh_reason = pending_reason
        self._face_refresh_force = bool(force_refresh)
        self._log_face_event(
            "refresh_requested",
            reason=pending_reason,
            refresh_people=bool(refresh_people),
            force=bool(force_refresh),
            request_id=request_id,
        )
        self._cancel_face_refresh_job(replaced=True)
        if not folder:
            if hasattr(self, "face_scanned_list"):
                self._refresh_scanned_faces()
            if refresh_people and hasattr(self, "face_named_people_list"):
                self._apply_face_people_data([], [], folder=folder, include_tiny_faces=self._show_tiny_detections_enabled())
            self._log_face_event("refresh_complete", reason=pending_reason, request_id=request_id, empty_folder=True)
            return
        if force_refresh:
            self._invalidate_face_review_source()
        scope_key = self._current_face_scope_key()
        mode = self.current_face_mode()
        active_service = self._face_service_for_scope(scope_key)
        pipeline_prefs = dict(self._applied_face_pipeline_prefs(mode))
        source_cache_key = self._face_review_source_cache_key_for_folder(folder)
        cached_source = (
            self._face_review_source
            if self._face_review_source is not None and self._face_review_source_key == source_cache_key
            else None
        )
        include_tiny_faces = self._show_tiny_detections_enabled()
        limit = int(self.face_browser_limit.value()) if refresh_people and hasattr(self, "face_browser_limit") else 0
        self._face_refresh_in_progress = True

        result_holder: dict[int, dict[str, object]] = {}

        def _run(progress, cancel_check):
            if cancel_check():
                raise Cancelled()
            progress(-1, f"Resolving saved face data for {Path(folder).name or folder}...")
            source = cached_source or self._resolve_face_review_source_background(
                folder=folder,
                scope_key=scope_key,
                mode=mode,
                active_service=active_service,
                pipeline_prefs=pipeline_prefs,
            )
            if cancel_check():
                raise Cancelled()
            indexed_paths = self._indexed_face_review_paths(source.service, folder)
            self._log_face_event(
                "review_source_selected",
                db_path=str(getattr(source.service, "db_path", "") or ""),
                indexed_images=len(indexed_paths),
            )
            if indexed_paths:
                summary_notice = (
                    "Showing indexed images from the saved face library. "
                    "Run Scan Folder for Faces to discover newly added photos."
                )
                status_text = f"Loaded indexed face review for {folder}."
            else:
                self._log_face_event(
                    "review_source_empty",
                    db_path=str(getattr(source.service, "db_path", "") or ""),
                )
                summary_notice = (
                    "No indexed face review data is available yet. "
                    "Run Scan Folder for Faces to populate this folder review."
                )
                status_text = f"No indexed face review is available for {folder} yet."
            progress(-1, f"Loading indexed face review for {Path(folder).name or folder}...")
            payload = self._load_face_refresh_result(
                request_id=request_id,
                reason=pending_reason,
                folder=folder,
                include_tiny_faces=include_tiny_faces,
                refresh_people=bool(refresh_people),
                limit=limit,
                candidate_paths=indexed_paths,
                discovery_complete=True,
                service=source.service,
                summary_notice=summary_notice,
                status_text=status_text,
                cancel_check=cancel_check,
            )
            payload["review_source"] = source
            payload["review_source_cache_key"] = source_cache_key
            result_holder[request_id] = payload
            return {"request_id": request_id}

        job = AsyncJob(_run)
        refresh_job_id: int | None = None
        if self.job_manager is not None:
            refresh_job_id = self.job_manager.register_job("Face review refresh", cancel_fn=job.cancel)
            self._face_refresh_job_id = refresh_job_id
            job.progress.connect(
                lambda value, text, job_id=refresh_job_id: self.job_manager.update(
                    job_id,
                    progress=value,
                    text=text,
                )
            )
        job.progress.connect(lambda _value, text: self.status_label.setText(str(text)))

        def _finish_job(status: str, error: str = "") -> None:
            if self.job_manager is not None and refresh_job_id is not None:
                self.job_manager.finish(refresh_job_id, status=status, error=error)
            if self._face_refresh_job_id == refresh_job_id:
                self._face_refresh_job_id = None

        def _failed(message: str) -> None:
            text = str(message or "").strip()
            result_holder.pop(request_id, None)
            if text.casefold() == "cancelled":
                self._face_refresh_in_progress = False
                _finish_job("cancelled")
                self._settle_face_refresh_threads()
                return
            self._face_refresh_in_progress = False
            self.status_label.setText(f"Face review refresh failed: {text}")
            self._log_face_event("refresh_failed", reason=pending_reason, request_id=request_id, error=text)
            _finish_job("failed", text)
            self._settle_face_refresh_threads()
            errorBox("Face review refresh failed", text)

        def _cancelled() -> None:
            self._face_refresh_in_progress = False
            result_holder.pop(request_id, None)
            self._log_face_event("refresh_cancelled", reason=pending_reason, request_id=request_id)
            _finish_job("cancelled")
            self._settle_face_refresh_threads()

        def _completed(result) -> None:
            self._face_refresh_in_progress = False
            if not isinstance(result, dict):
                _finish_job("failed", "invalid face review result")
                self._settle_face_refresh_threads()
                return
            result_request_id = int(result.get("request_id", 0) or 0)
            payload = result_holder.pop(result_request_id, None)
            if not isinstance(payload, dict):
                _finish_job("failed", "missing face review payload")
                self._settle_face_refresh_threads()
                return
            if result_request_id != int(self._face_refresh_request_id):
                self._log_face_event("refresh_discarded", reason=pending_reason, request_id=result_request_id, latest_request_id=self._face_refresh_request_id)
                _finish_job("cancelled")
                self._settle_face_refresh_threads()
                return
            source = payload.get("review_source")
            if isinstance(source, FaceReviewSource):
                cache_key_value = payload.get("review_source_cache_key")
                cache_key = cache_key_value if isinstance(cache_key_value, tuple) else source_cache_key
                self._adopt_face_review_source(source, cache_key)
            self._apply_face_review_snapshot(payload)
            _finish_job("finished")
            self._settle_face_refresh_threads()

        job.completed.connect(_completed)
        job.failed.connect(_failed)
        job.cancelled.connect(_cancelled)
        self._face_refresh_job = job
        thread = start_job_in_thread(job)
        self._face_refresh_thread = thread
        self._retained_face_refresh_refs.append((job, thread))
        thread.finished.connect(
            lambda thread=thread: self._on_face_refresh_thread_finished(thread),
            Qt.ConnectionType.QueuedConnection,
        )

    def _cancel_face_refresh_job(self, *, replaced: bool = False) -> None:
        job = self._face_refresh_job
        thread = self._face_refresh_thread
        if job is None and thread is None:
            return
        if job is not None:
            try:
                job.cancel()
            except Exception:
                pass
        if replaced:
            self._log_face_event("refresh_replaced", request_id=self._face_refresh_request_id)
        self._face_refresh_job = None
        self._face_refresh_thread = None

    def _on_face_refresh_thread_finished(self, thread=None) -> None:
        retained: list[tuple[object | None, object | None]] = []
        for job, retained_thread in self._retained_face_refresh_refs:
            if retained_thread is thread:
                continue
            retained.append((job, retained_thread))
        self._retained_face_refresh_refs = retained
        if thread is self._face_refresh_thread:
            self._face_refresh_thread = None
            self._face_refresh_job = None

    def _settle_face_refresh_threads(self, *, timeout_ms: int = 100) -> None:
        retained: list[tuple[object | None, object | None]] = []
        active_thread = self._face_refresh_thread
        for job, thread in list(self._retained_face_refresh_refs or []):
            if thread is None:
                continue
            try:
                if thread.isRunning():
                    wait_for_thread_shutdown(thread, timeout_ms=int(timeout_ms))
                if thread.isRunning():
                    retained.append((job, thread))
            except Exception:
                retained.append((job, thread))
        self._retained_face_refresh_refs = retained
        try:
            if active_thread is not None and not active_thread.isRunning():
                self._face_refresh_thread = None
                self._face_refresh_job = None
        except Exception:
            pass

    def _apply_face_review_snapshot(self, snapshot: dict[str, object]) -> None:
        folder = str(snapshot.get("folder", "") or "")
        review_images = list(snapshot.get("review_images", []) or [])
        profiles = list(snapshot.get("profiles", []) or [])
        include_tiny_faces = bool(snapshot.get("include_tiny_faces", False))
        refresh_people = bool(snapshot.get("refresh_people", False))
        reason = str(snapshot.get("reason", "background refresh") or "background refresh")
        summary_notice = str(snapshot.get("summary_notice", "") or "")
        status_text = str(snapshot.get("status_text", "") or "")
        self._log_face_event("review_loaded_background", folder=folder, image_count=len(review_images), refresh_people=refresh_people)
        self._request_face_review_publish(
            review_images,
            folder=folder,
            summary_notice=summary_notice,
            status_text=status_text,
        )
        if refresh_people and hasattr(self, "face_named_people_list"):
            self._apply_face_people_data(profiles, review_images, folder=folder, include_tiny_faces=include_tiny_faces)
        self._log_face_event("refresh_complete", reason=reason, request_id=int(snapshot.get("request_id", 0) or 0))

    def ensure_face_library_loaded(self, *, allow_inactive: bool = False) -> None:
        try:
            label = str(self.tabs.tabText(self.tabs.currentIndex()) or "").strip().lower()
        except Exception:
            label = ""
        if self._is_all_faces_tab_label(label) and not allow_inactive:
            self.ensure_face_album_loaded(allow_inactive=allow_inactive)
            return
        if not hasattr(self, "face_scanned_list") or (not allow_inactive and not self._is_face_folder_tab_label(label)):
            return
        effective_folder = self._effective_face_folder()
        if not effective_folder:
            return
        if (
            (not self._face_review_by_path or self._face_review_folder != effective_folder)
            and not self._face_refresh_in_progress
        ):
            self._request_face_library_refresh(refresh_people=False, reason="ensure library loaded", force_refresh=False)
        elif self._results_kind == "faces_review" and self._is_face_folder_tab_label(label):
            self._log_face_event("review_reapplied", image_count=len(self._face_review_images))
            self._reapply_face_folder_review()

    def ensure_current_faces_tab_loaded(self) -> None:
        if self.is_all_faces_tab_active():
            self.ensure_face_album_loaded()
            return
        if self.is_face_folder_tab_active():
            self.ensure_face_library_loaded()

    def ensure_faces_workspace_loaded(self) -> None:
        """Load saved face views on entry without indexing or rescanning photos."""
        self.ensure_face_album_loaded(allow_inactive=True)
        self.ensure_face_library_loaded(allow_inactive=True)

    def ensure_face_album_loaded(self, *, allow_inactive: bool = False) -> None:
        if not hasattr(self, "_face_album_summary_text"):
            return
        try:
            label = str(self.tabs.tabText(self.tabs.currentIndex()) or "").strip().lower()
        except Exception:
            label = ""
        if not allow_inactive and not self._is_all_faces_tab_label(label):
            return
        if not self._face_album_loaded:
            if not self._face_album_refresh_in_progress:
                self._request_face_album_refresh(reason="ensure all faces loaded", force_refresh=False)
            return
        if self._is_all_faces_tab_label(label):
            self._show_face_album_results()

    def _build_image_tab(self) -> None:
        tab = QWidget()
        form = QFormLayout(tab)
        self.image_query_path = PasteAwareLineEdit(lambda mime: self._handle_paste_mime(self.image_query_path, mime, multi_append=False), self)
        self.image_db_scope = QComboBox()
        self.image_db_scope.addItem("Saved face library", "global")
        self.image_db_scope.addItem("Current review session", "session")
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
        self.duplicate_db_scope.addItem("Saved face library", "global")
        self.duplicate_db_scope.addItem("Current review session", "session")
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
        self.duplicate_review_list = QListWidget()
        self.duplicate_review_list.setMaximumHeight(180)
        self.duplicate_review_list.itemSelectionChanged.connect(self._update_duplicate_review_selection)
        self.duplicate_review_summary = QLabel("Duplicate review: run a duplicate search to compare candidates.")
        self.duplicate_review_summary.setWordWrap(True)
        self.duplicate_review_tag = QLineEdit()
        self.duplicate_review_tag.setPlaceholderText("Decision tag")
        self.duplicate_review_tag_button = QPushButton("Tag Duplicate Decision")
        self.duplicate_review_trash_button = QPushButton("Trash Duplicate Decision")
        self.duplicate_review_export_button = QPushButton("Export Duplicate Decisions")
        self.duplicate_review_tag_button.clicked.connect(lambda: self._record_duplicate_decision("tag"))
        self.duplicate_review_trash_button.clicked.connect(lambda: self._record_duplicate_decision("trash"))
        self.duplicate_review_export_button.clicked.connect(self._export_duplicate_decisions)
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
        form.addRow("Review pairs", self.duplicate_review_list)
        form.addRow("Selected pair", self.duplicate_review_summary)
        form.addRow("Decision tag", self.duplicate_review_tag)
        decision_buttons = QHBoxLayout()
        decision_buttons.addWidget(self.duplicate_review_tag_button)
        decision_buttons.addWidget(self.duplicate_review_trash_button)
        decision_buttons.addWidget(self.duplicate_review_export_button)
        form.addRow(decision_buttons)
        self._action_buttons.extend(
            [
                self.duplicate_review_tag_button,
                self.duplicate_review_trash_button,
                self.duplicate_review_export_button,
            ]
        )
        self.tabs.addTab(tab, "Duplicate Search")

    @staticmethod
    def _helper_label(text: str, *, tooltip: str = "") -> QLabel:
        label = QLabel(str(text or ""))
        label.setWordWrap(True)
        if tooltip:
            label.setToolTip(str(tooltip))
        return label

    @staticmethod
    def _group_box(title: str, *, tooltip: str = "") -> tuple[QGroupBox, QVBoxLayout]:
        group = QGroupBox(title)
        group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        if tooltip:
            group.setToolTip(str(tooltip))
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        return group, layout

    @staticmethod
    def _field_widget(title: str, widget: QWidget, *, tooltip: str = "") -> QWidget:
        cell = QWidget()
        cell.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        layout = QVBoxLayout(cell)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        label = QLabel(title)
        label.setWordWrap(True)
        label_row = QHBoxLayout()
        label_row.setContentsMargins(0, 0, 0, 0)
        label_row.setSpacing(4)
        label_row.addWidget(label, stretch=1)
        if tooltip:
            label.setToolTip(str(tooltip))
            widget.setToolTip(str(tooltip))
            label_row.addWidget(HelpIconButton(str(tooltip), cell, help_key=str(title).casefold().replace(" ", "_")))
        layout.addLayout(label_row)
        layout.addWidget(widget)
        return cell

    @staticmethod
    def _expander_button(title: str, *, checked: bool = False, tooltip: str = "") -> QToolButton:
        button = QToolButton()
        button.setText(str(title))
        button.setCheckable(True)
        button.setChecked(bool(checked))
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        button.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
        if tooltip:
            button.setToolTip(str(tooltip))
        return button

    @staticmethod
    def _set_expander_state(button: QToolButton | None, panel: QWidget | None, expanded: bool) -> None:
        if button is not None:
            button.blockSignals(True)
            button.setChecked(bool(expanded))
            button.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
            button.blockSignals(False)
        if panel is not None:
            panel.setVisible(bool(expanded))

    @staticmethod
    def _clear_layout(layout) -> None:
        while layout.count():
            layout.takeAt(0)

    def _face_layout_mode(self) -> str:
        sidebar_width = 0
        try:
            sizes = self.workspace_splitter.sizes() if hasattr(self, "workspace_splitter") else []
            if sizes:
                sidebar_width = int(sizes[0])
            requested_sizes = getattr(self.workspace_splitter, "requested_sizes", [])
            if requested_sizes:
                sidebar_width = max(sidebar_width, int(requested_sizes[0]))
            if sidebar_width <= 0:
                sidebar_width = int(getattr(self, "sidebar_panel", self).width())
        except Exception:
            sidebar_width = int(getattr(self, "sidebar_panel", self).width())
        return "wide" if sidebar_width >= 480 else "compact"

    def _face_search_layout_mode_for_viewport(self) -> str:
        """Return the layout mode for the Find-a-Person workflow cards.

        This must use the live page width, rather than the splitter's requested
        size.  Requested splitter sizes can remain at the old maximum after a
        resize, which previously left two cards squeezed into the narrow task
        pane.
        """
        available_width = 0
        try:
            page = self.face_search_grid.parentWidget()
            if page is not None:
                available_width = int(page.contentsRect().width())
            if available_width <= 0:
                viewport = getattr(self, "sidebar_scroll", None)
                if viewport is not None:
                    available_width = int(viewport.viewport().contentsRect().width())
        except Exception:
            available_width = 0

        margins = self.face_search_grid.contentsMargins()
        usable_width = max(0, available_width - margins.left() - margins.right())
        return "wide" if usable_width >= FACE_SEARCH_TWO_COLUMN_MIN_WIDTH else "compact"

    def _on_face_workspace_splitter_moved(self, *_args) -> None:
        try:
            self.workspace_splitter.requested_sizes = [int(value) for value in self.workspace_splitter.sizes()]
        except Exception:
            pass
        self._refresh_face_grid_layouts()

    def _refresh_face_grid_layouts(self, *, force: bool = False) -> None:
        self._refresh_face_library_layout(force=force)
        self._refresh_face_search_layout(force=force)

    def _apply_face_ui_mode(self) -> None:
        advanced = self.current_ui_mode() == "advanced"
        for group_name in (
            "face_manage_group",
            "face_people_query_group",
            "face_save_group",
            "face_pending_group",
        ):
            group = getattr(self, group_name, None)
            if group is not None:
                group.setVisible(bool(advanced))
        for group_name in ("face_advanced_group", "face_settings_group"):
            group = getattr(self, group_name, None)
            if group is not None:
                group.setVisible(False)
        for widget_name in (
            "show_tiny_detections_checkbox",
            "face_mode_status_label",
            "face_identity_management_group",
            "face_identity_danger_group",
            "face_save_profile_button",
            "face_hide_selected_faces_button",
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setVisible(bool(advanced))
        pipeline_summary = getattr(self, "face_pipeline_summary_group", None)
        if pipeline_summary is not None:
            pipeline_summary.setVisible(True)
        library_tabs = getattr(self, "face_library_tabs", None)
        if library_tabs is not None:
            library_tabs.tabBar().setVisible(True)
            library_tabs.tabBar().setTabVisible(1, False)
            if library_tabs.currentIndex() == 1:
                library_tabs.setCurrentIndex(0)
        if hasattr(self, "face_search_quick_group"):
            self.face_search_quick_group.setVisible(bool(advanced))
        for widget in getattr(self, "face_profile_technical_fields", []):
            widget.setVisible(bool(advanced))
        for widget_name in (
            "face_results_folder_filter",
            "face_results_date_from",
            "face_results_date_to",
            "face_results_split_button",
            "face_results_merge_button",
            "face_results_recluster_button",
            "face_results_compare_again_button",
            "face_results_export_button",
            "face_results_cluster_explanation",
            "face_results_cluster_suggestion",
            "face_results_membership_table",
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setVisible(bool(advanced))
        self._update_face_scope_summary()
        for widget_name in (
            "saved_search_group",
            "face_hide_rejected_button",
            "face_restore_rejected_button",
            "face_rescan_selected_button",
            "face_rescan_suspicious_button",
            "face_rescan_fallback_button",
            "face_pending_accept_above_threshold_button",
            "face_pending_accept_cluster_button",
            "face_pending_reject_cluster_button",
            "face_pending_undo_button",
        ):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                widget.setVisible(bool(advanced))
        for button_name, panel_name in (
            ("face_search_selected_options_toggle", "face_search_selected_options_panel"),
            ("face_find_options_toggle", "face_find_options_panel"),
            ("face_save_options_toggle", "face_save_options_panel"),
            ("face_manage_options_toggle", "face_manage_options_panel"),
            ("face_pending_items_toggle", "face_pending_items_panel"),
            ("face_people_query_toggle", "face_people_query_panel"),
        ):
            button = getattr(self, button_name, None)
            panel = getattr(self, panel_name, None)
            if button is None:
                continue
            button.setVisible(bool(advanced))
            if not advanced:
                self._set_expander_state(button, panel, False)
        self._update_search_scope_control_visibility()
        if hasattr(self, "face_scan_button"):
            self.face_scan_button.setText("Scan Folder for Faces" if advanced else "Detect Faces")
        self._refresh_face_grid_layouts(force=True)

    def _refresh_face_library_layout(self, *, force: bool = False) -> None:
        if not hasattr(self, "face_library_grid") or not hasattr(self, "face_library_selection_grid"):
            return
        mode = self._face_layout_mode()
        if not force and mode == self._face_library_layout_mode:
            return
        detection_grid = self.face_library_grid
        selection_grid = self.face_library_selection_grid
        self._clear_layout(detection_grid)
        self._clear_layout(selection_grid)
        for grid in (detection_grid, selection_grid):
            grid.setColumnStretch(0, 1)
            grid.setColumnStretch(1, 1 if mode == "wide" else 0)
            for row in range(4):
                grid.setRowStretch(row, 0)
        detection_grid.addWidget(self.face_source_group, 0, 0, 1, 2)
        detection_grid.setRowStretch(1, 1)
        selection_grid.addWidget(self.face_scanned_group, 0, 0, 1, 2)
        selection_grid.addWidget(self.face_people_group, 1, 0, 1, 2)
        selection_grid.addWidget(self.face_profile_group, 2, 0, 1, 2)
        selection_grid.setRowStretch(3, 1)
        self._face_library_layout_mode = mode

    def _refresh_face_search_layout(self, *, force: bool = False) -> None:
        if not hasattr(self, "face_search_grid"):
            return
        mode = self._face_search_layout_mode_for_viewport()
        if not force and mode == self._face_search_layout_mode:
            return
        self._clear_layout(self.face_search_grid)
        self.face_search_grid.setColumnStretch(0, 1)
        self.face_search_grid.setColumnStretch(1, 1 if mode == "wide" else 0)
        for row in range(9):
            self.face_search_grid.setRowStretch(row, 0)
        self.face_search_grid.addWidget(self.face_search_quick_group, 0, 0, 1, 2)
        if mode == "wide":
            self.face_search_grid.addWidget(self.face_search_selected_group, 1, 0)
            self.face_search_grid.addWidget(self.face_find_group, 1, 1)
            self.face_search_grid.addWidget(self.face_find_name_group, 2, 0, 1, 2)
            self.face_search_grid.addWidget(self.face_manage_group, 3, 0, 1, 2)
            self.face_search_grid.addWidget(self.face_save_group, 4, 0, 1, 2)
            self.face_search_grid.addWidget(self.face_pending_group, 5, 0, 1, 2)
            self.face_search_grid.addWidget(self.face_people_query_group, 6, 0, 1, 2)
            self.face_search_grid.setRowStretch(7, 1)
        else:
            self.face_search_grid.addWidget(self.face_search_selected_group, 1, 0, 1, 2)
            self.face_search_grid.addWidget(self.face_find_group, 2, 0, 1, 2)
            self.face_search_grid.addWidget(self.face_find_name_group, 3, 0, 1, 2)
            self.face_search_grid.addWidget(self.face_manage_group, 4, 0, 1, 2)
            self.face_search_grid.addWidget(self.face_save_group, 5, 0, 1, 2)
            self.face_search_grid.addWidget(self.face_pending_group, 6, 0, 1, 2)
            self.face_search_grid.addWidget(self.face_people_query_group, 7, 0, 1, 2)
            self.face_search_grid.setRowStretch(8, 1)
        self._face_search_layout_mode = mode

    def _build_face_album_tab(self) -> None:
        tab_body = QWidget()
        tab_layout = QVBoxLayout(tab_body)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.setSpacing(8)

        album_group, album_layout = self._group_box("All Faces", tooltip=FACE_HELP["people_groups"])
        # The summary remains state for result publishing, rather than a
        # hidden QLabel that could still affect the sidebar's size hint.
        self._face_album_summary_text = "Refresh All Faces to load the global detected-face album."
        album_buttons = QGridLayout()
        album_buttons.setContentsMargins(0, 0, 0, 0)
        album_buttons.setHorizontalSpacing(6)
        album_buttons.setVerticalSpacing(6)
        self.face_album_refresh_button = QPushButton("Refresh All Faces")
        self.face_album_refresh_button.setToolTip("Reload the saved global face library. This does not rescan folders.")
        self.face_album_refresh_button.clicked.connect(lambda: self.refresh_face_album(reason="manual album refresh", force_refresh=True))
        album_buttons.addWidget(self.face_album_refresh_button, 0, 0)
        self.face_album_load_more_groups_button = QPushButton("Load more groups")
        self.face_album_load_more_groups_button.setToolTip("Load the next page of saved face groups from the global library.")
        self.face_album_load_more_groups_button.setEnabled(False)
        self.face_album_load_more_groups_button.clicked.connect(self._load_more_face_album_groups)
        album_buttons.addWidget(self.face_album_load_more_groups_button, 0, 1)
        self.face_album_load_more_faces_button = QPushButton("Load more selected faces")
        self.face_album_load_more_faces_button.setToolTip("Load the next page of faces for the currently selected group.")
        self.face_album_load_more_faces_button.setEnabled(False)
        self.face_album_load_more_faces_button.clicked.connect(self._load_more_selected_face_album_members)
        album_buttons.addWidget(self.face_album_load_more_faces_button, 1, 0)
        self.face_album_help_button = HelpIconButton(
            FACE_HELP["people_groups"],
            album_group,
            help_key="all_faces",
        )
        album_buttons.addWidget(self.face_album_help_button, 1, 1, alignment=Qt.AlignmentFlag.AlignRight)
        album_layout.addLayout(album_buttons)
        tab_layout.addWidget(album_group)
        tab_layout.addStretch(1)

        self._action_buttons.append(self.face_album_refresh_button)
        self._action_buttons.extend(
            [self.face_album_load_more_groups_button, self.face_album_load_more_faces_button]
        )
        self.tabs.addTab(tab_body, self._all_faces_tab_label)

    def _build_face_library_tab(self) -> None:
        tab_body = QWidget()
        tab_layout = QVBoxLayout(tab_body)
        tab_layout.setContentsMargins(0, 0, 0, 0)
        tab_layout.setSpacing(8)
        self.face_library_tabs = QTabWidget(tab_body)
        self.face_library_tabs.setDocumentMode(True)
        detection_page = QWidget(self.face_library_tabs)
        detection_layout = QGridLayout(detection_page)
        detection_layout.setContentsMargins(0, 0, 0, 0)
        detection_layout.setSpacing(10)
        self.face_library_grid = detection_layout
        selection_page = QWidget(self.face_library_tabs)
        selection_layout = QGridLayout(selection_page)
        selection_layout.setContentsMargins(0, 0, 0, 0)
        selection_layout.setSpacing(10)
        self.face_library_selection_grid = selection_layout
        self.face_library_tabs.addTab(detection_page, "Detect Faces")
        self.face_library_tabs.addTab(selection_page, "Review & Name")
        # The retained review workflow stays implemented, but is hidden from
        # the primary Faces navigation until its redesign is ready.
        self.face_library_tabs.tabBar().setTabVisible(1, False)
        tab_layout.addWidget(self.face_library_tabs)

        self.face_db_scope = QComboBox(detection_page)
        self.face_db_scope.addItem("Saved face library", "global")
        self.face_db_scope.addItem("This session only", "session")
        self.face_db_scope.setToolTip(FACE_HELP["scan_source"])
        self.face_db_scope.hide()

        self.face_folder_path = QLineEdit()
        self.face_folder_path.setPlaceholderText("Folder to scan for faces (optional)")
        self.face_folder_path.setToolTip(FACE_HELP["scan_source"])

        self.face_named_people_list = QListWidget()
        self.face_named_people_list.setMinimumHeight(220)
        self.face_named_people_list.setMaximumHeight(280)
        self.face_named_people_list.setToolTip(
            "Saved identities for this folder. Click one to load its name and profile fields."
        )
        self.face_unlabeled_groups_list = QListWidget()
        self.face_unlabeled_groups_list.setMinimumHeight(220)
        self.face_unlabeled_groups_list.setMaximumHeight(280)
        self.face_unlabeled_groups_list.setToolTip(
            "Photos with unlabeled faces in this folder. Click one to show and select its faces."
        )

        self.face_name_query = QLineEdit()
        self.face_name_query.setPlaceholderText("Saved identity name")
        self.face_name_query.setToolTip(FACE_HELP["saved_person_name"])

        self.face_name_top_k = QSpinBox()
        self.face_name_top_k.setRange(1, 500)
        self.face_name_top_k.setValue(60)
        self.face_name_top_k.setToolTip(FACE_HELP["name_search_top_k"])

        self.face_name_min_score = QDoubleSpinBox()
        self.face_name_min_score.setRange(0.0, 1.0)
        self.face_name_min_score.setSingleStep(0.01)
        self.face_name_min_score.setValue(0.0)
        self.face_name_min_score.setToolTip(FACE_HELP["name_search_min_score"])

        self.face_name_folder_filter = QLineEdit()
        self.face_name_folder_filter.setPlaceholderText("Optional folder filter for search results")
        self.face_name_folder_filter.setToolTip(FACE_HELP["result_folder_filter"])

        self.face_label_name = QLineEdit()
        self.face_label_name.setPlaceholderText("Name for selected face thumbnails")
        self.face_label_name.setToolTip(FACE_HELP["selected_face_name"])
        self.face_profile_notes = QLineEdit()
        self.face_profile_notes.setPlaceholderText("Profile notes")
        self.face_profile_notes.setToolTip(FACE_HELP["profile_notes"])
        self.face_profile_tags = QLineEdit()
        self.face_profile_tags.setPlaceholderText("Comma-separated tags")
        self.face_profile_tags.setToolTip(FACE_HELP["profile_tags"])
        self.face_profile_birth_date = QLineEdit()
        self.face_profile_birth_date.setPlaceholderText("YYYY-MM-DD")
        self.face_profile_birth_date.setToolTip("Optional birth date stored with the saved identity profile.")
        self.face_profile_favorite = QCheckBox("Favorite")
        self.face_profile_favorite.setToolTip("Pin this identity ahead of non-favorites in People and Identities.")
        self.face_profile_hidden = QCheckBox("Hidden")
        self.face_profile_hidden.setToolTip("Hide this identity from default People and face result lists without deleting labels.")

        self.face_library_label_threshold = QDoubleSpinBox()
        self.face_library_label_threshold.setRange(0.0, 1.0)
        self.face_library_label_threshold.setSingleStep(0.01)
        self.face_library_label_threshold.setValue(0.72)
        self.face_library_label_threshold.setToolTip(FACE_HELP["label_threshold"])

        self.face_browser_top_k = QSpinBox()
        self.face_browser_top_k.setRange(1, 500)
        self.face_browser_top_k.setValue(40)
        self.face_browser_top_k.setToolTip(FACE_HELP["selected_face_top_k"])

        self.face_browser_min_score = QDoubleSpinBox()
        self.face_browser_min_score.setRange(0.0, 1.0)
        self.face_browser_min_score.setSingleStep(0.01)
        self.face_browser_min_score.setValue(0.45)
        self.face_browser_min_score.setToolTip(FACE_HELP["selected_face_min_score"])

        self.face_browser_limit = QSpinBox()
        self.face_browser_limit.setRange(25, 5000)
        self.face_browser_limit.setSingleStep(25)
        self.face_browser_limit.setValue(400)
        self.face_browser_limit.setToolTip(FACE_HELP["shown_faces_limit"])

        self.face_review_sort = QComboBox()
        self.face_review_sort.addItem("Name", "name")
        self.face_review_sort.addItem("Detected Faces", "detected_faces")
        self.face_review_sort.addItem("Best Detected Face", "best_detected_face")
        self.face_review_sort.setToolTip(FACE_HELP["face_review_sort"])
        self.face_review_sort.currentIndexChanged.connect(lambda _index: self._on_face_review_sort_changed())

        self.face_library_review_summary = QLabel("Choose a folder and scan it to review the folder gallery.")
        self.face_library_review_summary.setWordWrap(True)
        self.face_library_workflow_label = self._helper_label(
            "1. Clean detections in Photos or Faces.\n"
            "2. Save image edits in the inspector.\n"
            "3. Switch to Face Search for clustering, search, and naming.",
            tooltip=FACE_HELP["library_quick_start"],
        )

        self.face_selected_photo_label = QLabel("None")
        self.face_selected_photo_label.setWordWrap(True)
        self.face_selected_photo_label.setToolTip(FACE_HELP["scanned_faces"])
        self.face_selected_photo_label.setAccessibleName("Selected photo")

        self.face_scanned_summary = QLabel("Choose a photo in the folder gallery to load its detected faces.")
        self.face_scanned_summary.setWordWrap(True)

        self.face_scanned_model = FaceTileListModel(
            self._image_for_face_tile,
            self._face_tile_cache_key_for_item,
            QSize(72, 72),
            self,
        )
        self.face_scanned_list = QListView()
        self.face_scanned_list.setViewMode(QListView.ViewMode.IconMode)
        self.face_scanned_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.face_scanned_list.setMovement(QListView.Movement.Static)
        self.face_scanned_list.setLayoutMode(QListView.LayoutMode.Batched)
        self.face_scanned_list.setBatchSize(32)
        self.face_scanned_list.setUniformItemSizes(True)
        self.face_scanned_list.setWrapping(True)
        self.face_scanned_list.setWordWrap(True)
        self.face_scanned_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.face_scanned_list.setIconSize(QSize(72, 72))
        self.face_scanned_list.setGridSize(QSize(96, 118))
        self.face_scanned_list.setSpacing(8)
        self.face_scanned_list.setMinimumHeight(180)
        self.face_scanned_list.setMaximumHeight(260)
        self.face_scanned_list.setToolTip(FACE_HELP["scanned_faces"])
        self.face_scanned_list.setModel(self.face_scanned_model)
        self.face_scanned_list.setItemDelegate(FaceTileItemDelegate(self.face_scanned_list))
        self.face_scanned_list.verticalScrollBar().valueChanged.connect(lambda _value: self._schedule_selected_face_tile_loads())
        scanned_selection_model = self.face_scanned_list.selectionModel()
        if scanned_selection_model is not None:
            scanned_selection_model.selectionChanged.connect(lambda *_args: self._on_scanned_face_selection_changed())

        folder_row = QWidget(detection_page)
        folder_row.hide()
        self.face_folder_override_row = folder_row
        folder_layout = QHBoxLayout(folder_row)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self._browse_face_folder)
        self.face_recent_folders_button = QToolButton(folder_row)
        self.face_recent_folders_button.setText("Recent")
        self.face_recent_folders_button.setToolTip("Open or manage recently selected folders.")
        self.face_recent_folders_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.face_recent_folders_menu = QMenu(self.face_recent_folders_button)
        self.face_recent_folders_button.setMenu(self.face_recent_folders_menu)
        folder_layout.addWidget(self.face_folder_path)
        folder_layout.addWidget(self.face_recent_folders_button)
        folder_layout.addWidget(browse_btn)

        source_group, source_layout = self._group_box("Scan", tooltip=FACE_HELP["scan_source"])
        self.face_library_quick_start_label = self._helper_label(
            "Choose a folder and detect faces. Results open in Photos and Faces for review.",
            tooltip=FACE_HELP["library_quick_start"],
        )
        source_fields = QVBoxLayout()
        source_fields.setContentsMargins(0, 0, 0, 0)
        source_fields.setSpacing(8)
        self.face_source_fields_grid = source_fields
        self.face_review_quality_filter = QComboBox()
        self.face_review_quality_filter.addItem("All Faces", "all")
        self.face_review_quality_filter.addItem("Clean Only", "clean")
        self.face_review_quality_filter.addItem("Needs Review", "review")
        self.face_review_quality_filter.addItem("Rejected", "reject")
        self.face_review_quality_filter.addItem("Suspicious Only", "suspicious")
        self.face_review_reason_filter = QComboBox()
        self.face_review_reason_filter.addItem("All Reasons", "")
        for reason in (
            "low_detector_score",
            "review_confidence",
            "too_small",
            "low_area",
            "edge_clipped",
            "touches_edge",
            "bad_aspect_ratio",
            "unusual_aspect_ratio",
            "missing_landmarks",
            "invalid_landmarks",
            "alignment_unavailable",
            "blurred",
            "soft_focus",
            "too_dark",
            "too_bright",
            "low_contrast",
            "not_face_verifier_failed",
        ):
            self.face_review_reason_filter.addItem(reason.replace("_", " ").title(), reason)
        self.face_review_quality_filter.currentIndexChanged.connect(lambda _index: self._reapply_face_folder_review())
        self.face_review_reason_filter.currentIndexChanged.connect(lambda _index: self._reapply_face_folder_review())
        self.face_review_quality_field = self._field_widget(
            "Face quality filter", self.face_review_quality_filter, tooltip=FACE_HELP["pending_face_review"]
        )
        self.face_review_reason_field = self._field_widget(
            "Reason filter", self.face_review_reason_filter, tooltip=FACE_HELP["pending_face_review"]
        )
        quality_layout = self.face_quality_page_layout
        quality_layout.insertWidget(1, self.face_review_quality_field)
        quality_layout.insertWidget(2, self.face_review_reason_field)

        source_layout.addWidget(self._field_widget("Sort", self.face_review_sort, tooltip=FACE_HELP["face_review_sort"]))

        buttons = QGridLayout()
        buttons.setContentsMargins(0, 0, 0, 0)
        buttons.setSpacing(6)
        self.face_scan_actions_grid = buttons
        self.face_scan_button = QPushButton("Detect Faces")
        self.face_scan_button.setProperty("kind", "primary")
        self.face_scan_button.setToolTip(FACE_HELP["scan_folder_button"])
        self.face_refresh_people_button = QPushButton("Reload People List")
        self.face_refresh_people_button.setToolTip(FACE_HELP["reload_people_button"])
        self.face_refresh_faces_button = QPushButton("Refresh Folder Review")
        self.face_refresh_faces_button.setToolTip(FACE_HELP["reload_faces_button"])
        self.face_auto_clean_button = QPushButton("Auto-Clean Folder Review")
        self.face_auto_clean_button.setToolTip(FACE_HELP["auto_clean_review"])
        self.face_hide_rejected_button = QPushButton("Hide Rejected Faces")
        self.face_restore_rejected_button = QPushButton("Restore Rejected Faces")
        self.face_rescan_selected_button = QPushButton("Rescan Selected Photos")
        self.face_rescan_suspicious_button = QPushButton("Rescan Review/Rejected Photos")
        self.face_rescan_fallback_button = QPushButton("Rescan With Fallback Detector")
        self.face_scan_button.clicked.connect(self._scan_face_folder)
        self.face_refresh_people_button.clicked.connect(lambda: self._request_face_library_refresh(refresh_people=True, reason="reload people", force_refresh=True))
        self.face_refresh_faces_button.clicked.connect(lambda: self._request_face_library_refresh(refresh_people=False, reason="reload review", force_refresh=True))
        self.face_auto_clean_button.clicked.connect(self._auto_clean_face_review_folder)
        self.face_hide_rejected_button.clicked.connect(self._hide_rejected_face_review_faces)
        self.face_restore_rejected_button.clicked.connect(self._restore_rejected_face_review_faces)
        self.face_rescan_selected_button.clicked.connect(self._rescan_selected_face_review_images)
        self.face_rescan_suspicious_button.clicked.connect(self._rescan_suspicious_face_review_images)
        self.face_rescan_fallback_button.clicked.connect(self._rescan_selected_face_review_images_with_fallback)
        self._action_buttons.extend([
            self.face_scan_button,
            self.face_refresh_people_button,
            self.face_refresh_faces_button,
            self.face_auto_clean_button,
            self.face_hide_rejected_button,
            self.face_restore_rejected_button,
            self.face_rescan_selected_button,
            self.face_rescan_suspicious_button,
            self.face_rescan_fallback_button,
        ])
        self._mode_required_buttons.append(self.face_scan_button)
        buttons.addWidget(self.face_scan_button, 0, 0)
        buttons.addWidget(self.face_refresh_faces_button, 0, 1)
        buttons.addWidget(self.face_auto_clean_button, 1, 0)
        buttons.addWidget(self.face_rescan_selected_button, 1, 1)
        buttons.addWidget(self.face_hide_rejected_button, 2, 0)
        buttons.addWidget(self.face_restore_rejected_button, 2, 1)
        buttons.addWidget(self.face_rescan_suspicious_button, 3, 0)
        buttons.addWidget(self.face_rescan_fallback_button, 3, 1)
        source_layout.addLayout(buttons)
        self.face_source_group = source_group

        people_group, people_layout = self._group_box("Saved identities + unlabeled groups", tooltip=FACE_HELP["people_groups"])
        people_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        self.face_people_summary = self._helper_label(
            "Choose a folder, then click Scan Folder for Faces or Reload People List.",
            tooltip=FACE_HELP["people_groups"],
        )
        people_buttons = QGridLayout()
        people_buttons.setContentsMargins(0, 0, 0, 0)
        people_buttons.setSpacing(6)
        people_buttons.addWidget(self.face_refresh_people_button, 0, 0)
        people_layout.addLayout(people_buttons)
        people_lists = QGridLayout()
        people_lists.setContentsMargins(0, 0, 0, 0)
        people_lists.setSpacing(8)
        named_panel = QWidget(people_group)
        named_layout = QVBoxLayout(named_panel)
        named_layout.setContentsMargins(0, 0, 0, 0)
        named_layout.setSpacing(4)
        named_header = QHBoxLayout()
        named_header.setContentsMargins(0, 0, 0, 0)
        named_header.addWidget(QLabel("Named"), stretch=1)
        named_header.addWidget(
            HelpIconButton(
                "Saved identities for this folder. Click one to load its name and profile fields.",
                named_panel,
                help_key="named_identities",
            )
        )
        named_layout.addLayout(named_header)
        named_layout.addWidget(self.face_named_people_list)
        unlabeled_panel = QWidget(people_group)
        unlabeled_layout = QVBoxLayout(unlabeled_panel)
        unlabeled_layout.setContentsMargins(0, 0, 0, 0)
        unlabeled_layout.setSpacing(4)
        unlabeled_header = QHBoxLayout()
        unlabeled_header.setContentsMargins(0, 0, 0, 0)
        unlabeled_header.addWidget(QLabel("Unlabeled"), stretch=1)
        unlabeled_header.addWidget(
            HelpIconButton(
                "Photos with unlabeled faces in this folder. Click one to show and select its faces.",
                unlabeled_panel,
                help_key="unlabeled_groups",
            )
        )
        unlabeled_layout.addLayout(unlabeled_header)
        unlabeled_layout.addWidget(self.face_unlabeled_groups_list)
        people_lists.addWidget(named_panel, 0, 0)
        people_lists.addWidget(unlabeled_panel, 0, 1)
        people_lists.setColumnStretch(0, 1)
        people_lists.setColumnStretch(1, 1)
        people_layout.addLayout(people_lists)
        self.face_people_group = people_group

        scanned_group, scanned_layout = self._group_box("Selected photo", tooltip=FACE_HELP["scanned_faces"])
        selected_photo_row = QHBoxLayout()
        selected_photo_row.setContentsMargins(0, 0, 0, 0)
        selected_photo_row.setSpacing(4)
        selected_photo_row.addWidget(self.face_selected_photo_label, stretch=1)
        selected_photo_row.addWidget(
            HelpIconButton(FACE_HELP["scanned_faces"], scanned_group, help_key="selected_photo_faces")
        )
        scanned_layout.addLayout(selected_photo_row)
        scanned_layout.addWidget(self.face_scanned_list)
        self.face_scanned_group = scanned_group

        profile_group, profile_layout = self._group_box("Selected faces and profile", tooltip=FACE_HELP["selected_face_name"])
        self.face_selected_faces_context_label = self._helper_label(
            "No photo selected in Face Library yet.",
            tooltip=FACE_HELP["selected_face_name"],
        )
        profile_layout.addWidget(self.face_selected_faces_context_label)

        name_row = QWidget()
        name_layout = QVBoxLayout(name_row)
        name_layout.setContentsMargins(0, 0, 0, 0)
        name_layout.setSpacing(6)
        self.face_search_by_name_button = QPushButton("Find Photos by Saved Name")
        self.face_search_by_name_button.setToolTip(FACE_HELP["find_photos_saved_name"])
        self.face_search_by_name_button.clicked.connect(self._search_by_name)
        self._action_buttons.append(self.face_search_by_name_button)
        name_layout.addWidget(self.face_name_query)
        name_layout.addWidget(self.face_search_by_name_button)

        profile_fields = QVBoxLayout()
        profile_fields.setContentsMargins(0, 0, 0, 0)
        profile_fields.setSpacing(8)
        self.face_profile_fields_grid = profile_fields
        profile_fields.addWidget(self._field_widget("Saved identity", name_row, tooltip=FACE_HELP["saved_person_name"]))
        profile_fields.addWidget(self._field_widget("Name for selected face(s)", self.face_label_name, tooltip=FACE_HELP["selected_face_name"]))
        self.face_profile_technical_fields = []
        for title, widget, tooltip in (
            ("Saved-name maximum results", self.face_name_top_k, FACE_HELP["name_search_top_k"]),
            ("Saved-name minimum score", self.face_name_min_score, FACE_HELP["name_search_min_score"]),
            ("Selected-face maximum results", self.face_browser_top_k, FACE_HELP["selected_face_top_k"]),
            ("Selected-face minimum score", self.face_browser_min_score, FACE_HELP["selected_face_min_score"]),
            ("Label confidence", self.face_library_label_threshold, FACE_HELP["label_threshold"]),
            ("Faces to load", self.face_browser_limit, FACE_HELP["shown_faces_limit"]),
        ):
            field = self._field_widget(title, widget, tooltip=tooltip)
            self.face_profile_technical_fields.append(field)
            profile_fields.addWidget(field)
        profile_fields.addWidget(self._field_widget("Profile notes", self.face_profile_notes, tooltip=FACE_HELP["profile_notes"]))
        profile_fields.addWidget(self._field_widget("Profile tags", self.face_profile_tags, tooltip=FACE_HELP["profile_tags"]))
        profile_fields.addWidget(self._field_widget("Birth date", self.face_profile_birth_date, tooltip=self.face_profile_birth_date.toolTip()))
        profile_fields.addWidget(self.face_profile_favorite)
        profile_fields.addWidget(self.face_profile_hidden)
        profile_fields.addWidget(self._field_widget("Result folder filter", self.face_name_folder_filter, tooltip=FACE_HELP["result_folder_filter"]))
        profile_layout.addLayout(profile_fields)

        face_actions = QVBoxLayout()
        face_actions.setContentsMargins(0, 0, 0, 0)
        face_actions.setSpacing(6)
        self.face_save_name_button = QPushButton("Name Selected Face(s)")
        self.face_save_name_button.setToolTip(FACE_HELP["name_selected_faces"])
        self.face_save_profile_button = QPushButton("Save/Update Profile")
        self.face_save_profile_button.setToolTip(FACE_HELP["save_profile"])
        self.face_hide_selected_faces_button = QPushButton("Hide Selected Face(s)")
        self.face_hide_selected_faces_button.setToolTip("Hide selected face tiles without deleting labels or source images.")
        self.face_search_selected_button = QPushButton("Find Photos of Selected Face")
        self.face_search_selected_button.setToolTip(FACE_HELP["find_photos_selected_face"])
        self.face_save_name_button.clicked.connect(self._save_selected_face_name)
        self.face_save_profile_button.clicked.connect(self._save_person_profile)
        self.face_hide_selected_faces_button.clicked.connect(self._hide_selected_faces)
        self.face_search_selected_button.clicked.connect(self._search_selected_face)
        self._action_buttons.extend([self.face_save_name_button, self.face_save_profile_button, self.face_hide_selected_faces_button, self.face_search_selected_button])
        face_actions.addWidget(self.face_save_name_button)
        face_actions.addWidget(self.face_save_profile_button)
        face_actions.addWidget(self.face_hide_selected_faces_button)
        face_actions.addWidget(self.face_search_selected_button)
        profile_layout.addLayout(face_actions)
        self.face_profile_group = profile_group

        self.face_named_people_list.itemClicked.connect(self._on_person_clicked)
        self.face_unlabeled_groups_list.itemClicked.connect(self._on_person_clicked)
        self.face_db_scope.currentTextChanged.connect(lambda _text: self._update_face_scope_summary())
        self.face_db_scope.currentTextChanged.connect(lambda _text: self._invalidate_face_review_source())
        self.face_db_scope.currentTextChanged.connect(self._on_face_db_scope_changed)
        self.face_folder_path.textChanged.connect(lambda _text: self._update_face_scope_summary())
        self.face_folder_path.textChanged.connect(lambda _text: self._invalidate_face_review_source())
        self.face_folder_path.editingFinished.connect(
            lambda: self.source_folder_changed.emit(self.face_folder_path.text().strip())
        )
        self._update_face_scope_summary()
        self._update_face_selected_context_label()
        self._update_face_mode_status()
        self._refresh_face_library_layout(force=True)
        self.set_recent_directories([])
        self.tabs.addTab(tab_body, self._face_folder_tab_label)

    @staticmethod
    def _recent_folder_label(path: str) -> str:
        normalized = str(Path(str(path or "")))
        name = Path(normalized).name or normalized
        parent = str(Path(normalized).parent)
        return f"{name} — {parent}" if parent and parent != normalized else name

    def set_recent_directories(self, directories: list[str]) -> None:
        self._recent_directories = [str(path) for path in directories if str(path or "").strip()]
        menu = getattr(self, "face_recent_folders_menu", None)
        if menu is None:
            return
        menu.clear()
        if not self._recent_directories:
            action = menu.addAction("No recent folders")
            action.setEnabled(False)
            self.face_recent_folders_button.setToolTip("No recent folders yet.")
            return
        self.face_recent_folders_button.setToolTip("Open or manage recently selected folders.")
        for path in self._recent_directories:
            action = menu.addAction(self._recent_folder_label(path))
            action.setToolTip(path)
            action.triggered.connect(lambda _checked=False, value=path: self._select_recent_face_folder(value))
        menu.addSeparator()
        remove_menu = menu.addMenu("Remove from history")
        for path in self._recent_directories:
            action = remove_menu.addAction(self._recent_folder_label(path))
            action.setToolTip(path)
            action.triggered.connect(
                lambda _checked=False, value=path: self.recent_folder_remove_requested.emit(value)
            )
        clear_action = menu.addAction("Clear history")
        clear_action.triggered.connect(self.recent_folders_clear_requested.emit)

    def _select_recent_face_folder(self, directory: str) -> None:
        path = str(directory or "").strip()
        if not path or not Path(path).is_dir():
            return
        self.face_folder_path.setText(path)
        self.source_folder_changed.emit(path)
        self._log_face_event("recent_folder_selected", directory=path)
        if self.is_face_folder_tab_active():
            self.refresh_face_library(reason="recent folder selected")

    def _browse_face_folder(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "Select Folder")
        if directory:
            self.face_folder_path.setText(str(directory))
            self.source_folder_changed.emit(str(directory))
            self._log_face_event("folder_override_browsed", directory=directory)
            if self.is_face_folder_tab_active():
                self.refresh_face_library(reason="folder override browsed")

    def _on_face_db_scope_changed(self, _text: str) -> None:
        if self.is_face_folder_tab_active():
            self.refresh_face_library(reason="db scope changed")

    def _effective_face_folder(self) -> str:
        current_folder = str(self._current_directory() or "").strip()
        if current_folder:
            return current_folder
        # Preserve a legacy/session value only when the shared source pane has
        # not selected a folder yet. The Faces UI itself no longer offers an
        # independent scan target.
        return self.face_folder_path.text().strip()

    def _face_db_scope_label(self) -> str:
        try:
            return str(self.face_db_scope.currentText() or "").strip() or "Saved face library"
        except Exception:
            return "Saved face library"

    def _update_face_scope_summary(self, *, folder: str | None = None) -> None:
        _ = folder
        self._update_search_scope_control_visibility()
        self._update_face_status_strip()

    def _face_review_sort_mode(self) -> str:
        combo = getattr(self, "face_review_sort", None)
        if combo is None:
            return "name"
        try:
            data = combo.currentData()
            if data:
                return str(data)
        except Exception:
            pass
        try:
            return str(combo.currentText() or "name").strip().lower().replace(" ", "_")
        except Exception:
            return "name"

    def _current_face_review_quality_filter(self) -> str:
        combo = getattr(self, "face_review_quality_filter", None)
        if combo is None:
            return "all"
        try:
            return str(combo.currentData() or "all").strip().lower()
        except Exception:
            return "all"

    def _current_face_review_reason_filter(self) -> str:
        combo = getattr(self, "face_review_reason_filter", None)
        if combo is None:
            return ""
        try:
            return str(combo.currentData() or "").strip().lower()
        except Exception:
            return ""

    def _face_review_filter_state(self) -> dict[str, str]:
        return {
            "quality_filter": self._current_face_review_quality_filter(),
            "reason_filter": self._current_face_review_reason_filter(),
            "sort_mode": self._face_review_sort_mode(),
            "photo_filter": self._current_face_photo_filter(),
        }

    @staticmethod
    def _review_face_matches_filters_state(
        *,
        quality_filter: str,
        reason_filter: str,
        hidden_rejected_refs: set[tuple[str, int]],
        image_path: str,
        face_index: int,
        status: str,
        reasons: tuple[str, ...],
    ) -> bool:
        normalized_status = str(status or "clean").strip().lower()
        normalized_reasons = tuple(str(value or "").strip().lower() for value in reasons or ())
        ref = (str(image_path or ""), int(face_index))
        if ref in hidden_rejected_refs and normalized_status == "reject":
            return False
        if quality_filter == "clean" and normalized_status != "clean":
            return False
        if quality_filter == "review" and normalized_status != "review":
            return False
        if quality_filter == "reject" and normalized_status != "reject":
            return False
        if quality_filter == "suspicious" and normalized_status not in {"review", "reject"}:
            return False
        if reason_filter and reason_filter not in normalized_reasons:
            return False
        return True

    def _review_face_matches_filters(
        self,
        *,
        image_path: str,
        face_index: int,
        status: str,
        reasons: tuple[str, ...],
    ) -> bool:
        return self._review_face_matches_filters_state(
            quality_filter=self._current_face_review_quality_filter(),
            reason_filter=self._current_face_review_reason_filter(),
            hidden_rejected_refs=set(self._face_hidden_rejected_refs),
            image_path=image_path,
            face_index=face_index,
            status=status,
            reasons=reasons,
        )

    def _on_face_review_sort_changed(self) -> None:
        if getattr(self, "_results_kind", "none") == "faces_review":
            self._reapply_face_folder_review()

    @staticmethod
    def _face_review_status_bucket(item: FaceFolderReviewImage) -> int:
        if str(getattr(item, "review_status", "")) in {"detected", "tiny_hidden"} and int(getattr(item, "total_face_count", 0) or 0) > 0:
            return 0
        if str(getattr(item, "review_status", "")) == "no_faces":
            return 1
        return 2

    @staticmethod
    def _face_bbox_area(bbox: tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = [int(value) for value in bbox]
        return max(0, x2 - x1) * max(0, y2 - y1)

    @classmethod
    def _review_best_face_metrics(cls, item: FaceFolderReviewImage) -> tuple[float, int]:
        best_score = 0.0
        best_area = 0
        for record in tuple(getattr(item, "visible_faces", ()) or ()):
            score = float(getattr(record, "face_confidence", 0.0) or 0.0)
            area = cls._face_bbox_area(tuple(getattr(record, "face_bbox", (0, 0, 0, 0))))
            if score > best_score or (score == best_score and area > best_area):
                best_score = score
                best_area = area
        return best_score, best_area

    @staticmethod
    def _face_review_name_key(item: FaceFolderReviewImage) -> tuple[str, str]:
        image_path = str(getattr(item, "image_path", "") or "")
        lowered_path = image_path.lower()
        file_name = lowered_path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
        return file_name, lowered_path

    @classmethod
    def _sort_face_review_images_for_mode(cls, review_images: list[FaceFolderReviewImage], *, mode: str) -> list[FaceFolderReviewImage]:
        items = list(review_images or [])
        if mode == "name":
            return sorted(
                items,
                key=lambda item: (
                    cls._face_review_status_bucket(item),
                    *cls._face_review_name_key(item),
                ),
            )

        def _detection_key(item: FaceFolderReviewImage):
            bucket = cls._face_review_status_bucket(item)
            visible_faces = getattr(item, "visible_faces", ()) or ()
            visible_count = len(visible_faces)
            best_score = 0.0
            best_area = 0
            for record in visible_faces:
                score = float(getattr(record, "face_confidence", 0.0) or 0.0)
                area = cls._face_bbox_area(tuple(getattr(record, "face_bbox", (0, 0, 0, 0))))
                if score > best_score or (score == best_score and area > best_area):
                    best_score = score
                    best_area = area
            name_key = cls._face_review_name_key(item)
            if mode == "detected_faces":
                return (bucket, -visible_count, -best_score, -best_area, name_key[0], name_key[1])
            return (bucket, -best_score, -best_area, -visible_count, name_key[0], name_key[1])

        return sorted(items, key=_detection_key)

    def _sort_face_review_images(self, review_images: list[FaceFolderReviewImage]) -> list[FaceFolderReviewImage]:
        return self._sort_face_review_images_for_mode(review_images, mode=self._face_review_sort_mode())

    def _ordered_face_review_images(self) -> list[FaceFolderReviewImage]:
        review_images = list(self._face_review_images or [])
        if not review_images:
            return []
        sort_mode = self._face_review_sort_mode()
        if str(self._face_review_sorted_mode or "") == sort_mode:
            return review_images
        ordered = self._sort_face_review_images_for_mode(review_images, mode=sort_mode)
        self._face_review_images = list(ordered)
        self._face_review_sorted_mode = sort_mode
        return list(self._face_review_images)

    def _reapply_face_folder_review(self) -> None:
        if not self._face_review_all_images and not self._face_review_folder:
            return
        self._apply_face_folder_review(list(self._face_review_all_images), folder=self._face_review_folder)

    def _current_face_review_item(self) -> FaceFolderReviewImage | None:
        selected_path = str(getattr(self, "_face_review_selected_path", "") or "")
        if not selected_path:
            return None
        return self._face_review_by_path.get(selected_path)

    def _clear_face_identity_fields(self) -> None:
        for widget_name in (
            "face_find_name_query",
            "face_name_query",
            "face_label_name",
            "person_name",
            "face_profile_notes",
            "face_profile_tags",
            "face_profile_birth_date",
        ):
            widget = getattr(self, widget_name, None)
            if widget is None:
                continue
            try:
                widget.clear()
            except Exception:
                pass
        for widget_name in ("face_profile_favorite", "face_profile_hidden"):
            widget = getattr(self, widget_name, None)
            if widget is not None:
                try:
                    widget.setChecked(False)
                except Exception:
                    pass

    @staticmethod
    def _face_tile_item_records(items: list[FaceTileItem]) -> list[IndexedFaceRecord]:
        return [item.payload for item in items if isinstance(item.payload, IndexedFaceRecord)]

    def _set_active_face_selection(self, items: list[FaceTileItem], *, source: str) -> None:
        self._active_face_selection_source = str(source or "")
        self._active_face_selection_items = list(items or [])
        records = self._face_tile_item_records(self._active_face_selection_items)
        names = {
            str(record.person_name or "").strip()
            for record in records
            if str(record.person_name or "").strip()
        }
        if records and len(names) == 1 and len(records) == len(self._active_face_selection_items):
            person_name = next(iter(names))
            self._set_face_identity_name_fields(person_name)
            self._populate_profile_fields(person_name)
        else:
            self._clear_face_identity_fields()

    def _clear_active_face_selection(self, *, source: str = "") -> None:
        if source and source != self._active_face_selection_source:
            return
        self._active_face_selection_source = ""
        self._active_face_selection_items = []
        self._clear_face_identity_fields()

    def _active_face_selection_paths(self) -> list[str]:
        return list(
            dict.fromkeys(
                str(item.image_path or "").strip()
                for item in self._active_face_selection_items
                if str(item.image_path or "").strip()
            )
        )

    def _update_face_selected_context_label(self) -> None:
        if not hasattr(self, "face_selected_faces_context_label"):
            return
        active_items = list(getattr(self, "_active_face_selection_items", []) or [])
        active_paths = self._active_face_selection_paths()
        if active_items:
            if len(active_paths) == 1:
                text = f"{Path(active_paths[0]).name} · selected {len(active_items)}"
            else:
                text = f"{len(active_items)} selected faces · {len(active_paths)} photos"
            self.face_selected_faces_context_label.setText(text)
            if hasattr(self, "face_search_selected_context_label"):
                self.face_search_selected_context_label.setText(text)
            selected_count = len(self._selected_scanned_faces()) if hasattr(self, "face_scanned_list") else 0
            if hasattr(self, "face_save_name_button"):
                self._set_guarded_action_enabled(self.face_save_name_button, selected_count > 0)
            if hasattr(self, "face_search_selected_button"):
                self._set_guarded_action_enabled(self.face_search_selected_button, selected_count == 1)
            if hasattr(self, "face_search_name_selected_card_button"):
                self._set_guarded_action_enabled(self.face_search_name_selected_card_button, selected_count > 0)
            if hasattr(self, "face_search_selected_card_button"):
                self._set_guarded_action_enabled(self.face_search_selected_card_button, selected_count == 1)
            return
        item = self._current_face_review_item()
        selected_count = len(self._selected_scanned_faces()) if hasattr(self, "face_scanned_list") else 0
        if item is None:
            text = "No photo selected."
        else:
            photo_name = Path(item.image_path).name
            quality_review_count = int(
                (self._face_review_context_by_path.get(item.image_path, {}).get("face_review", {}) or {}).get("quality_review_count", 0)
            )
            if item.review_status == "detected":
                text = f"{photo_name} · faces {len(item.visible_faces)} · selected {selected_count}"
                if quality_review_count > 0:
                    text += f" · review {quality_review_count}"
            elif item.review_status == "no_faces":
                text = f"{photo_name} · no face detected"
            elif item.review_status == "tiny_hidden":
                text = f"{photo_name} · tiny detections hidden"
            else:
                text = f"{photo_name} · not scanned"
        self.face_selected_faces_context_label.setText(text)
        if hasattr(self, "face_search_selected_context_label"):
            self.face_search_selected_context_label.setText(text)
        if hasattr(self, "face_save_name_button"):
            self._set_guarded_action_enabled(self.face_save_name_button, selected_count > 0)
        if hasattr(self, "face_search_selected_button"):
            self._set_guarded_action_enabled(self.face_search_selected_button, selected_count == 1)
        if hasattr(self, "face_search_name_selected_card_button"):
            self._set_guarded_action_enabled(self.face_search_name_selected_card_button, selected_count > 0)
        if hasattr(self, "face_search_selected_card_button"):
            self._set_guarded_action_enabled(self.face_search_selected_card_button, selected_count == 1)

    def _focus_face_review_image(self, image_path: str) -> bool:
        target = str(image_path or "")
        if not target or target not in self._face_review_by_path:
            return False
        try:
            row = list(getattr(self.results_gallery, "images", []) or []).index(target)
        except ValueError:
            self._load_face_review_selection(target)
            return True
        index = self.results_gallery.model.index(row, 0)
        if not index.isValid():
            self._load_face_review_selection(target)
            return True
        selection_model = self.results_gallery.list_view.selectionModel()
        if selection_model is not None:
            selection_model.select(index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
            selection_model.setCurrentIndex(index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        self.results_gallery.list_view.setCurrentIndex(index)
        self.results_gallery.list_view.scrollTo(index)
        self._load_face_review_selection(target)
        return True

    @staticmethod
    def _clamped_face_bbox(
        bbox: tuple[int | float, int | float, int | float, int | float],
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int] | None:
        if int(image_width) <= 0 or int(image_height) <= 0:
            return None
        width = max(1, int(image_width))
        height = max(1, int(image_height))
        x1, y1, x2, y2 = [int(round(float(value))) for value in bbox]
        left = max(0, min(width, min(x1, x2)))
        top = max(0, min(height, min(y1, y2)))
        right = max(0, min(width, max(x1, x2)))
        bottom = max(0, min(height, max(y1, y2)))
        if right <= left or bottom <= top:
            return None
        return (left, top, right, bottom)

    @classmethod
    def _normalized_face_bbox(
        cls,
        bbox: tuple[int, int, int, int],
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[float, float, float, float] | None:
        clamped = cls._clamped_face_bbox(bbox, image_width=image_width, image_height=image_height)
        if clamped is None:
            return None
        width = max(1, int(image_width))
        height = max(1, int(image_height))
        x1, y1, x2, y2 = clamped
        return (
            max(0.0, min(1.0, float(x1) / width)),
            max(0.0, min(1.0, float(y1) / height)),
            max(0.0, min(1.0, float(x2) / width)),
            max(0.0, min(1.0, float(y2) / height)),
        )

    @classmethod
    def _absolute_face_bbox(
        cls,
        normalized_bbox: tuple[float, float, float, float],
        *,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int, int, int] | None:
        if int(image_width) <= 0 or int(image_height) <= 0:
            return None
        width = max(1, int(image_width))
        height = max(1, int(image_height))
        x1 = float(normalized_bbox[0]) * width
        y1 = float(normalized_bbox[1]) * height
        x2 = float(normalized_bbox[2]) * width
        y2 = float(normalized_bbox[3]) * height
        return cls._clamped_face_bbox((x1, y1, x2, y2), image_width=width, image_height=height)

    def _scan_face_folder(self) -> None:
        directory = self._effective_face_folder()
        if not directory:
            errorBox("No folder selected", "Choose a folder path or select a folder in the sidebar first.")
            return
        if not self._ensure_face_models_ready("scan this folder"):
            return
        service = self._active_face_service()
        self._update_face_scope_summary(folder=directory)
        self._log_face_event("scan_requested", directory=directory)

        def _run(progress, cancel_check):
            progress(0, "0/0 images, faces=0")
            return service.index_directory(directory, recursive=True, progress_callback=progress, cancel_check=cancel_check)

        def _done(metrics: dict) -> None:
            faces = int((metrics or {}).get("faces_indexed", 0))
            done = int((metrics or {}).get("images_done", 0))
            total = int((metrics or {}).get("images_total", 0))
            self._log_face_event("scan_completed", directory=directory, images_done=done, images_total=total, faces_indexed=faces)
            self.status_label.setText(f"Indexed {faces} faces from {done}/{total} images in {directory}.")
            self._invalidate_face_discovery_cache(directory)
            self._request_face_library_refresh(refresh_people=True, reason="scan complete", force_refresh=True)
            self._maybe_refresh_global_face_album(reason="scan complete", force_refresh=True)

        self._start_job("Indexing faces", _run, _done)

    @staticmethod
    def _payload_face_ref(payload_item: dict[str, object]) -> tuple[str, int]:
        return (
            str(payload_item.get("image_path", "") or ""),
            int(payload_item.get("face_index", 0) or 0),
        )

    def _filter_face_identity_payload_for_folder(self, payload: dict[str, object], folder: str) -> dict[str, object]:
        folder_prefix = str(folder or "").strip()
        face_rows = [
            dict(item)
            for item in list(payload.get("face_index", []) or [])
            if isinstance(item, dict) and path_is_within_scope(str(item.get("image_path", "") or ""), folder_prefix)
        ]
        face_scan_rows = [
            dict(item)
            for item in list(payload.get("face_scan_images", []) or [])
            if isinstance(item, dict) and path_is_within_scope(str(item.get("image_path", "") or ""), folder_prefix)
        ]
        face_refs = {self._payload_face_ref(item) for item in face_rows}
        labels = [
            dict(item)
            for item in list(payload.get("labels", []) or [])
            if isinstance(item, dict) and self._payload_face_ref(item) in face_refs
        ]
        pending_labels = [
            dict(item)
            for item in list(payload.get("pending_labels", []) or [])
            if isinstance(item, dict) and self._payload_face_ref(item) in face_refs
        ]
        corrections = [
            dict(item)
            for item in list(payload.get("corrections", []) or [])
            if isinstance(item, dict) and self._payload_face_ref(item) in face_refs
        ]
        people_names = {
            str(item.get("person_name", "") or "").strip()
            for item in labels + pending_labels + corrections
            if str(item.get("person_name", "") or "").strip()
        }
        identities: list[dict[str, object]] = []
        for identity in list(payload.get("identities", []) or []):
            if not isinstance(identity, dict):
                continue
            person_name = str(identity.get("person_name", "") or "").strip()
            prototype_refs = [
                dict(ref)
                for ref in list(identity.get("prototype_refs", []) or [])
                if isinstance(ref, dict) and self._payload_face_ref(ref) in face_refs
            ]
            if person_name not in people_names and not prototype_refs:
                continue
            filtered_identity = dict(identity)
            filtered_identity["prototype_refs"] = prototype_refs
            cover_image_path = str(filtered_identity.get("cover_image_path", "") or "")
            if cover_image_path and not path_is_within_scope(cover_image_path, folder_prefix):
                filtered_identity["cover_image_path"] = prototype_refs[0]["image_path"] if prototype_refs else ""
                filtered_identity["cover_face_index"] = int(prototype_refs[0].get("face_index", 0) or 0) if prototype_refs else 0
            identities.append(filtered_identity)
        return {
            "version": int(payload.get("version", 1) or 1),
            "mode": str(payload.get("mode", self.current_face_mode()) or self.current_face_mode()),
            "identities": identities,
            "face_index": face_rows,
            "face_scan_images": face_scan_rows,
            "labels": labels,
            "pending_labels": pending_labels,
            "corrections": corrections,
        }

    def _upload_face_folder_to_global_db(self) -> None:
        if self._current_face_scope_key() != "session":
            errorBox("Session review required", "Store faces in the current review session before adding a folder review to the saved face library.")
            return
        folder = self._effective_face_folder()
        if not folder:
            errorBox("No folder selected", "Choose a folder path or select a folder in the sidebar first.")
            return
        source_service = self._active_face_service()
        target_service = self._global_face_service()

        def _run(progress, cancel_check):
            progress(-1, "Preparing folder faces for the saved face library...")
            _ = cancel_check
            payload = source_service.export_identity_data()
            filtered_payload = self._filter_face_identity_payload_for_folder(payload, folder)
            if not list(filtered_payload.get("face_index", []) or []):
                raise ValueError("No scanned faces from this folder are available in the session DB yet.")
            progress(-1, "Adding folder faces to the saved face library...")
            target_service.import_identity_data(filtered_payload, mode="merge")
            return {
                "faces": len(list(filtered_payload.get("face_index", []) or [])),
                "labels": len(list(filtered_payload.get("labels", []) or [])),
                "identities": len(list(filtered_payload.get("identities", []) or [])),
            }

        def _done(result: dict[str, object]) -> None:
            faces = int(result.get("faces", 0) or 0)
            labels = int(result.get("labels", 0) or 0)
            identities = int(result.get("identities", 0) or 0)
            self.status_label.setText(
                f"Added {faces} face(s), {labels} label(s), and {identities} identity record(s) from {folder} to the saved face library."
            )
            self._log_face_event("folder_uploaded_to_global", folder=folder, faces=faces, labels=labels, identities=identities)
            self.refresh_face_album(reason="folder uploaded to global", force_refresh=True)

        self._start_job("Adding folder faces to saved face library", _run, _done)

    def _selected_face_review_image_paths(self) -> list[str]:
        selection_model = getattr(getattr(self.results_gallery, "list_view", None), "selectionModel", lambda: None)()
        paths: list[str] = []
        if selection_model is not None:
            for index in selection_model.selectedIndexes():
                if index.isValid():
                    path = index.data(Qt.ItemDataRole.UserRole)
                    if str(path or "").strip():
                        paths.append(str(path))
        return list(dict.fromkeys(paths))

    def _suspicious_face_review_image_paths(self) -> list[str]:
        paths: list[str] = []
        for item in list(self._face_review_all_images or self._face_review_images or []):
            for record in tuple(item.visible_faces or ()):
                status = str(record.quality_status or "clean").strip().lower()
                if status in {"review", "reject"}:
                    paths.append(str(item.image_path))
                    break
        return list(dict.fromkeys(paths))

    def _hide_rejected_face_review_faces(self) -> None:
        hidden = 0
        for item in list(self._face_review_all_images or self._face_review_images or []):
            for record in tuple(item.visible_faces or ()):
                if str(record.quality_status or "clean").strip().lower() == "reject":
                    ref = (str(record.image_path), int(record.face_index))
                    if ref not in self._face_hidden_rejected_refs:
                        self._face_hidden_rejected_refs.add(ref)
                        hidden += 1
        self._reapply_face_folder_review()
        self.status_label.setText(f"Hidden {hidden} rejected face(s) from the current review session.")

    def _restore_rejected_face_review_faces(self) -> None:
        restored = len(self._face_hidden_rejected_refs)
        self._face_hidden_rejected_refs.clear()
        self._reapply_face_folder_review()
        self.status_label.setText(f"Restored {restored} rejected face(s) to the current review session.")

    def _rescan_face_review_image_paths(
        self,
        image_paths: list[str],
        *,
        reason: str,
        use_fallback_policy: bool = False,
    ) -> None:
        paths = [str(path) for path in list(dict.fromkeys(image_paths or [])) if str(path or "").strip()]
        if not paths:
            errorBox("No images selected", "Select one or more review photos first.")
            return
        service = self._active_face_service()
        original_policy = str(getattr(service, "detector_policy", "single") or "single")
        original_fallback = str(getattr(service, "fallback_detector_id", getattr(service, "detector_id", "")) or "")
        target_fallback = self.current_face_fallback_detector_id()
        if use_fallback_policy and (not target_fallback or target_fallback == self.current_face_detector_id()):
            errorBox("Missing fallback detector", "Choose a different fallback detector in Faces settings first.")
            return

        def _run(progress, cancel_check):
            progress(0, "0/0 images, faces=0")
            job_service = service
            restore_shared_service = False
            if use_fallback_policy:
                copy_for_job = getattr(service, "copy_for_job", None)
                if callable(copy_for_job):
                    job_service = copy_for_job(
                        detector_policy="union_then_verify",
                        fallback_detector_id=target_fallback,
                    )
                else:
                    service.configure_cascade_options(
                        detector_policy="union_then_verify",
                        fallback_detector_id=target_fallback,
                    )
                    restore_shared_service = True
            try:
                return job_service.index_paths(paths, progress_callback=progress, cancel_check=cancel_check, force=True)
            finally:
                if restore_shared_service:
                    service.configure_cascade_options(
                        detector_policy=original_policy,
                        fallback_detector_id=original_fallback,
                    )

        def _done(metrics: dict) -> None:
            faces = int((metrics or {}).get("faces_indexed", 0))
            done = int((metrics or {}).get("images_done", 0))
            total = int((metrics or {}).get("images_total", 0))
            self.status_label.setText(f"Rescanned {done}/{total} image(s) and indexed {faces} face(s) for {reason}.")
            self._request_face_library_refresh(refresh_people=True, reason=f"{reason} rescan complete", force_refresh=True)

        self._start_job("Rescanning face review images", _run, _done)

    def _rescan_selected_face_review_images(self) -> None:
        self._rescan_face_review_image_paths(
            self._selected_face_review_image_paths(),
            reason="selected photos",
        )

    def _rescan_suspicious_face_review_images(self) -> None:
        self._rescan_face_review_image_paths(
            self._suspicious_face_review_image_paths(),
            reason="review/rejected photos",
        )

    def _rescan_selected_face_review_images_with_fallback(self) -> None:
        self._rescan_face_review_image_paths(
            self._selected_face_review_image_paths(),
            reason="selected photos with fallback detector",
            use_fallback_policy=True,
        )

    def _apply_face_people_data(
        self,
        profiles,
        review_images: list[FaceFolderReviewImage] | None,
        *,
        folder: str,
        include_tiny_faces: bool,
    ) -> None:
        self.face_named_people_list.clear()
        self.face_unlabeled_groups_list.clear()
        if not folder:
            self._face_profiles_by_name = {}
            if hasattr(self, "face_people_summary"):
                self.face_people_summary.setText(
                    "No folder selected. Choose a folder in the sidebar or enter one in the Face Library folder box."
                )
            return
        hidden_face_count = sum(int(getattr(item, "hidden_face_count", 0) or 0) for item in review_images or [])
        visible_profiles = 0
        for profile in profiles:
            if folder and int(getattr(profile, "visible_face_count", 0)) <= 0:
                continue
            tags_text = ", ".join(profile.tags[:3]) if profile.tags else "no tags"
            notes_text = str(profile.notes or "").strip()
            prefix = "[Favorite] " if bool(getattr(profile, "favorite", False)) else ""
            summary = f"{prefix}{profile.person_name} | examples={profile.example_count} | labeled={profile.labeled_count} | visible={profile.visible_face_count} | thr={profile.similarity_threshold:.2f}"
            if str(getattr(profile, "birth_date", "") or "").strip():
                summary += f" | DOB {getattr(profile, 'birth_date', '')}"
            if notes_text:
                summary += f" | {notes_text[:36]}"
            summary += f" | tags={tags_text}"
            item = QListWidgetItem(summary)
            item.setData(Qt.ItemDataRole.UserRole, profile)
            self.face_named_people_list.addItem(item)
            visible_profiles += 1
        grouped_records: dict[str, list[IndexedFaceRecord]] = {}
        for item in review_images or []:
            for record in tuple(getattr(item, "visible_faces", ()) or ()):
                if str(record.person_name or "").strip():
                    continue
                grouped_records.setdefault(record.image_path, []).append(record)
        self._face_profiles_by_name = {str(profile.person_name): profile for profile in profiles}
        if hasattr(self, "face_identity_list"):
            self._refresh_face_identities(profiles=list(profiles or []))
        unlabeled_group_count = 0
        for image_path in sorted(grouped_records):
            image_records = grouped_records[image_path]
            face_refs = tuple((record.image_path, int(record.face_index)) for record in image_records)
            average_confidence = sum(float(record.face_confidence) for record in image_records) / max(1, len(image_records))
            payload = FaceLibraryUnlabeledGroup(
                image_path=image_path,
                face_refs=face_refs,
                face_count=len(image_records),
                records=tuple(image_records),
            )
            item = QListWidgetItem(
                f"Unlabeled | {Path(image_path).name} | faces={len(image_records)} | conf={average_confidence:.2f}"
            )
            item.setData(Qt.ItemDataRole.UserRole, payload)
            item.setToolTip(f"{image_path}\n{len(image_records)} unlabeled face(s)")
            self.face_unlabeled_groups_list.addItem(item)
            unlabeled_group_count += 1
        if hasattr(self, "face_people_summary"):
            if self.face_named_people_list.count() + self.face_unlabeled_groups_list.count() <= 0:
                if hidden_face_count and not include_tiny_faces:
                    self.face_people_summary.setText(
                        "Only tiny detections were found for this folder. Turn on Show Tiny Detections or rescan a better folder."
                    )
                else:
                    self.face_people_summary.setText(
                        "No saved identities or unlabeled groups found for this folder. Scan the folder or name selected faces first."
                    )
            else:
                summary = f"Loaded {visible_profiles} saved identities and {unlabeled_group_count} unlabeled group(s)."
                if hidden_face_count and not include_tiny_faces:
                    summary += f" {hidden_face_count} tiny detection(s) hidden."
                summary += " Click a row to review it."
                self.face_people_summary.setText(summary)
        self._log_face_event(
            "people_load_complete",
            folder=folder,
            visible_profiles=visible_profiles,
            unlabeled_groups=unlabeled_group_count,
            hidden_faces=hidden_face_count,
        )

    def _refresh_face_people(self, *, review_images: list[FaceFolderReviewImage] | None = None) -> None:
        folder = self._effective_face_folder()
        limit = int(self.face_browser_limit.value())
        include_tiny_faces = self._show_tiny_detections_enabled()
        self._log_face_event("people_load_requested", limit=limit, include_tiny=include_tiny_faces)
        self._update_face_scope_summary(folder=folder)
        self.face_named_people_list.clear()
        self.face_unlabeled_groups_list.clear()
        if not folder:
            self._apply_face_people_data([], [], folder=folder, include_tiny_faces=include_tiny_faces)
            return
        service = self._active_face_service()
        try:
            if review_images is None and self._face_review_folder == folder and self._face_review_images:
                review_images = list(self._face_review_images)
            if review_images is None:
                indexed_paths = self._indexed_face_review_paths(service, folder)
                review_images = service.load_folder_review_images(
                    folder,
                    recursive=True,
                    candidate_paths=indexed_paths,
                    include_tiny_faces=include_tiny_faces,
                )
            profiles = service.load_person_profiles(
                folder_prefix=folder,
                limit=limit,
                include_tiny_faces=include_tiny_faces,
            )
        except Exception as exc:
            LOGGER.exception("FacePane people load failed folder=%s", folder)
            errorBox("Load failed", str(exc))
            return
        self._apply_face_people_data(profiles, review_images, folder=folder, include_tiny_faces=include_tiny_faces)

    def _refresh_scanned_faces(self) -> list[FaceFolderReviewImage] | None:
        folder = self._effective_face_folder()
        include_tiny_faces = self._show_tiny_detections_enabled()
        self._log_face_event("review_load_requested", include_tiny=include_tiny_faces)
        self._update_face_scope_summary(folder=folder)
        self.face_scanned_model.set_items([])
        self._face_review_by_path = {}
        if not folder:
            self._face_review_selected_path = ""
            self._face_review_folder = ""
            self._face_review_images = []
            self._face_review_sorted_mode = ""
            self.face_library_review_summary.setText(
                "No folder selected. Choose a folder in the sidebar or enter one in the Face Library folder box."
            )
            self.face_selected_photo_label.setText("None")
            self.face_scanned_summary.setText(
                "No folder selected. Choose a folder in the sidebar or enter one in the Face Library folder box."
            )
            self._set_results_kind("faces_review")
            self.results_gallery.update_gallery([])
            try:
                self.results_gallery.model.set_overlays_by_path({})
                self.results_gallery.model.set_face_boxes_by_path({})
                self.results_gallery.model.set_face_box_states_by_path({})
            except Exception:
                pass
            self.results_gallery.inspector_context_provider = lambda _p: {}
            self._publish_results([], lambda _p: {}, {}, {}, self._results_kind)
            self._update_face_selected_context_label()
            return []
        service = self._active_face_service()
        try:
            indexed_paths = self._indexed_face_review_paths(service, folder)
            review_images = service.load_folder_review_images(
                folder,
                recursive=True,
                candidate_paths=indexed_paths,
                include_tiny_faces=include_tiny_faces,
            )
            migrated_image_paths = []
            consume_migrated = getattr(service, "consume_recent_migrated_face_paths", None)
            if callable(consume_migrated):
                migrated_image_paths = list(consume_migrated() or [])
        except Exception as exc:
            LOGGER.exception("FacePane review load failed folder=%s", folder)
            errorBox("Load failed", str(exc))
            return None
        try:
            self._apply_face_folder_review(review_images, folder=folder, migrated_image_paths=migrated_image_paths)
        except Exception as exc:
            LOGGER.exception("FacePane review publish failed folder=%s image_count=%s", folder, len(review_images or []))
            errorBox("Review failed", str(exc))
            return None
        return review_images

    def _build_face_review_gallery_state(
        self,
        image_paths: list[str],
    ) -> tuple[dict[str, str], dict[str, str], dict[str, list[tuple[float, float, float, float]]], dict[str, str], dict[str, dict[str, object]], dict[str, int]]:
        overlay_by_path: dict[str, str] = {}
        subtitle_by_path: dict[str, str] = {}
        face_boxes_by_path: dict[str, list[tuple[float, float, float, float]]] = {}
        face_box_states_by_path: dict[str, str] = {}
        ctx_by_path: dict[str, dict[str, object]] = {}
        stats = {
            "detected_images": 0,
            "no_face_images": 0,
            "not_scanned_images": 0,
            "hidden_face_count": 0,
            "visible_face_count": 0,
            "review_face_count": 0,
        }
        for image_path in image_paths:
            item = self._face_review_by_path.get(str(image_path))
            if item is None:
                continue
            is_dirty = item.image_path in self._face_review_draft_dirty_paths
            drafts = self._review_faces_for_item(item)
            faces_payload = []
            normalized_boxes: list[tuple[float, float, float, float]] = []
            quality_summary = {"clean": 0, "review": 0, "reject": 0}
            if is_dirty:
                quality_summary = self._summarize_face_quality(
                    item.image_path,
                    drafts,
                )
                face_box_states_by_path[item.image_path] = "draft"
                stats["visible_face_count"] += len(drafts)
                stats["review_face_count"] += int(quality_summary["review"])
                for index, draft in enumerate(drafts):
                    faces_payload.append(
                        {
                            "face_index": int(draft.face_index if int(draft.face_index) >= 0 else index),
                            "bbox": tuple(int(value) for value in draft.bbox),
                            "confidence": float(draft.confidence or 1.0),
                            "person_name": str(draft.person_name or ""),
                            "source": str(draft.source or ""),
                        }
                    )
                    normalized = self._normalized_face_bbox(
                        tuple(int(value) for value in draft.bbox),
                        image_width=item.image_width,
                        image_height=item.image_height,
                    )
                    if normalized is not None:
                        normalized_boxes.append(normalized)
                if drafts:
                    stats["detected_images"] += 1
                    overlay_by_path[item.image_path] = f"{len(drafts)} face{'s' if len(drafts) != 1 else ''}"
                    subtitle_by_path[item.image_path] = (
                        f"Unsaved edits | {int(quality_summary['review'])} need review"
                        if int(quality_summary["review"]) > 0
                        else "Unsaved face edits"
                    )
                else:
                    stats["no_face_images"] += 1
                    overlay_by_path[item.image_path] = "No faces detected"
                    subtitle_by_path[item.image_path] = "Unsaved face edits"
            else:
                saved_records = [
                    record
                    for record in tuple(item.visible_faces or ())
                    if self._review_face_matches_filters(
                        image_path=item.image_path,
                        face_index=int(record.face_index),
                        status=str(record.quality_status or "clean"),
                        reasons=tuple(str(value) for value in record.quality_reasons or ()),
                    )
                ]
                quality_summary = self._summarize_saved_face_quality(saved_records)
                stats["visible_face_count"] += len(saved_records)
                stats["hidden_face_count"] += int(item.hidden_face_count)
                stats["review_face_count"] += int(quality_summary["review"])
                labels = [record.person_name for record in saved_records if record.person_name]
                unique_labels = list(dict.fromkeys(labels))
                if item.review_status == "detected":
                    if saved_records:
                        stats["detected_images"] += 1
                        overlay_by_path[item.image_path] = f"{len(saved_records)} face{'s' if len(saved_records) != 1 else ''}"
                    else:
                        stats["no_face_images"] += 1
                        overlay_by_path[item.image_path] = "No matching faces"
                    base_subtitle = ", ".join(unique_labels[:2]) if unique_labels else "Detected faces"
                    if int(quality_summary["review"]) > 0:
                        base_subtitle += f" | {int(quality_summary['review'])} need review"
                    subtitle_by_path[item.image_path] = base_subtitle
                elif item.review_status == "tiny_hidden":
                    overlay_by_path[item.image_path] = "Tiny detections hidden"
                    subtitle_by_path[item.image_path] = "Turn on Show Tiny Detections"
                elif item.review_status == "no_faces":
                    stats["no_face_images"] += 1
                    overlay_by_path[item.image_path] = "No faces detected"
                    subtitle_by_path[item.image_path] = "Scanned"
                else:
                    stats["not_scanned_images"] += 1
                    overlay_by_path[item.image_path] = "Not scanned"
                    subtitle_by_path[item.image_path] = self._face_db_scope_label()
                for record in saved_records:
                    faces_payload.append(
                        {
                            "face_index": int(record.face_index),
                            "bbox": tuple(record.face_bbox),
                            "confidence": float(record.face_confidence),
                            "person_name": str(record.person_name or ""),
                        }
                    )
                    normalized = self._normalized_face_bbox(
                        record.face_bbox,
                        image_width=item.image_width,
                        image_height=item.image_height,
                    )
                    if normalized is not None:
                        normalized_boxes.append(normalized)
            if normalized_boxes:
                face_boxes_by_path[item.image_path] = normalized_boxes
            ctx_by_path[item.image_path] = {
                "face_review": {
                    "status": "draft" if is_dirty else item.review_status,
                    "total_face_count": len(drafts) if is_dirty else int(item.total_face_count),
                    "hidden_face_count": 0 if is_dirty else int(item.hidden_face_count),
                    "dirty": bool(is_dirty),
                    "quality_review_count": int(quality_summary["review"]),
                },
                "indexed_faces": {
                    "count": len(drafts) if is_dirty else len(item.visible_faces),
                    "faces": faces_payload,
                },
            }
        return overlay_by_path, subtitle_by_path, face_boxes_by_path, face_box_states_by_path, ctx_by_path, stats

    def _gallery_paths_match(self, image_paths: list[str]) -> bool:
        return list(getattr(self.results_gallery, "images", []) or []) == list(image_paths or [])

    def _publish_results_gallery_paths(
        self,
        paths: list[str],
        *,
        overlay_by_path: dict[str, str] | None = None,
        subtitle_by_path: dict[str, str] | None = None,
        face_boxes_by_path: dict[str, list[tuple[float, float, float, float]]] | None = None,
        face_box_states_by_path: dict[str, str] | None = None,
        context_provider=None,
        clear_pixmaps: bool = True,
        reset_scroll: bool = True,
    ) -> None:
        image_paths = [str(path) for path in list(dict.fromkeys(paths or [])) if str(path or "").strip()]
        overlay_by_path = dict(overlay_by_path or {})
        subtitle_by_path = dict(subtitle_by_path or {})
        face_boxes_by_path = dict(face_boxes_by_path or {})
        face_box_states_by_path = dict(face_box_states_by_path or {})
        provider = context_provider or (lambda _p: {})
        self._results_gallery_publish_token += 1
        publish_token = int(self._results_gallery_publish_token)
        if not image_paths:
            self.results_gallery.update_gallery([])
            try:
                self.results_gallery.model.set_overlays_by_path({})
                self.results_gallery.model.set_face_boxes_by_path({})
                self.results_gallery.model.set_face_box_states_by_path({})
            except Exception:
                pass
            self.results_gallery.inspector_context_provider = provider
            self._publish_results([], provider, {}, {}, self._results_kind)
            return
        if len(image_paths) <= 96:
            self.results_gallery.update_gallery_with_options(
                image_paths,
                clear_pixmaps=clear_pixmaps,
                reset_scroll=reset_scroll,
            )
            limit_paths = set(image_paths)
            try:
                self.results_gallery.model.set_overlays_by_path(
                    overlay_by_path,
                    subtitle_by_path,
                    limit_paths=limit_paths,
                )
                self.results_gallery.model.set_face_boxes_by_path(
                    face_boxes_by_path,
                    limit_paths=limit_paths,
                )
                self.results_gallery.model.set_face_box_states_by_path(
                    face_box_states_by_path,
                    limit_paths=limit_paths,
                )
            except Exception:
                pass
            self.results_gallery.inspector_context_provider = provider
            self._publish_results(image_paths, provider, overlay_by_path, subtitle_by_path, self._results_kind)
            return

        self.results_gallery.update_gallery_with_options([], clear_pixmaps=clear_pixmaps, reset_scroll=reset_scroll)
        self.results_gallery.inspector_context_provider = provider
        total = len(image_paths)
        chunk_size = 96
        chunk_delay_ms = 1

        def _step(offset: int) -> None:
            if publish_token != int(self._results_gallery_publish_token):
                return
            chunk = image_paths[offset : offset + chunk_size]
            if not chunk:
                self._publish_results(image_paths, provider, overlay_by_path, subtitle_by_path, self._results_kind)
                self.status_label.setText(f"Loaded {total} photo(s).")
                self._update_face_results_context()
                return
            self.results_gallery.append_images(chunk)
            limit_paths = set(chunk)
            try:
                self.results_gallery.model.set_overlays_by_path(
                    overlay_by_path,
                    subtitle_by_path,
                    limit_paths=limit_paths,
                )
                self.results_gallery.model.set_face_boxes_by_path(
                    face_boxes_by_path,
                    limit_paths=limit_paths,
                )
                self.results_gallery.model.set_face_box_states_by_path(
                    face_box_states_by_path,
                    limit_paths=limit_paths,
                )
            except Exception:
                pass
            loaded = min(total, offset + len(chunk))
            self.status_label.setText(f"Loading photos {loaded}/{total}...")
            self._update_face_results_context()
            QTimer.singleShot(chunk_delay_ms, lambda next_offset=offset + chunk_size: _step(next_offset))

        _step(0)

    def _can_use_lazy_face_review_publish(
        self,
        review_images: list[FaceFolderReviewImage],
        *,
        filter_state: dict[str, str],
        hidden_rejected_refs: set[tuple[str, int]],
        dirty_paths: set[str],
    ) -> bool:
        if len(list(review_images or [])) <= FACE_REVIEW_LAZY_PUBLISH_THRESHOLD:
            return False
        if str(filter_state.get("quality_filter", "all") or "all").strip().lower() != "all":
            return False
        if str(filter_state.get("reason_filter", "") or "").strip().lower():
            return False
        if str(filter_state.get("photo_filter", "with_faces") or "with_faces").strip().lower() == "needs_review":
            return False
        if hidden_rejected_refs or dirty_paths:
            return False
        return True

    @staticmethod
    def _face_review_visible_count(item: FaceFolderReviewImage) -> int:
        total_faces = max(0, int(getattr(item, "total_face_count", 0) or 0))
        hidden_faces = max(0, int(getattr(item, "hidden_face_count", 0) or 0))
        return max(0, total_faces - hidden_faces)

    def _face_review_summary_text(
        self,
        review_images: list[FaceFolderReviewImage],
        *,
        folder: str,
        include_tiny_faces: bool,
        summary_notice: str = "",
    ) -> str:
        if not review_images:
            summary_text = f"No images found in {folder}."
            if str(summary_notice or "").strip():
                summary_text = f"{summary_text} {str(summary_notice).strip()}".strip()
            return summary_text
        detected_images = 0
        no_face_images = 0
        not_scanned_images = 0
        hidden_face_count = 0
        visible_face_count = 0
        review_face_count = 0
        for item in list(review_images or []):
            status = str(getattr(item, "review_status", "") or "").strip().lower()
            visible_count = self._face_review_visible_count(item)
            hidden_count = max(0, int(getattr(item, "hidden_face_count", 0) or 0))
            if status == "detected":
                if visible_count > 0:
                    detected_images += 1
                else:
                    no_face_images += 1
            elif status == "no_faces":
                no_face_images += 1
            elif status == "not_scanned":
                not_scanned_images += 1
            hidden_face_count += hidden_count
            visible_face_count += visible_count
            review_face_count += sum(
                1
                for record in tuple(getattr(item, "visible_faces", ()) or ())
                if str(getattr(record, "quality_status", "clean") or "clean").strip().lower() in {"review", "reject"}
            )
        summary_text = (
            f"Loaded {len(review_images)} image(s) from {folder}. "
            f"{int(detected_images)} photo(s) with visible faces, {int(no_face_images)} with no faces, "
            f"{int(not_scanned_images)} not scanned. {int(visible_face_count)} visible face(s)."
        )
        if int(hidden_face_count) and not include_tiny_faces:
            summary_text += f" {int(hidden_face_count)} tiny detection(s) hidden."
        if int(review_face_count):
            summary_text += f" {int(review_face_count)} face(s) still need manual review."
        if str(summary_notice or "").strip():
            summary_text = f"{summary_text} {str(summary_notice).strip()}".strip()
        return summary_text

    def _build_lazy_face_review_publish_snapshot(
        self,
        *,
        request_id: int,
        folder: str,
        review_images: list[FaceFolderReviewImage],
        scope_label: str,
        include_tiny_faces: bool,
        summary_notice: str,
        status_text: str,
    ) -> FaceReviewPublishSnapshot:
        sorted_review_images = self._sort_face_review_images(list(review_images or []))
        overlay_by_path: dict[str, str] = {}
        subtitle_by_path: dict[str, str] = {}
        context_by_path: dict[str, dict[str, object]] = {}
        review_by_path: dict[str, FaceFolderReviewImage] = {}
        paths: list[str] = []
        photo_filter = self._current_face_photo_filter()
        for item in sorted_review_images:
            image_path = str(getattr(item, "image_path", "") or "")
            if not image_path:
                continue
            total_faces = max(0, int(getattr(item, "total_face_count", 0) or 0))
            hidden_faces = max(0, int(getattr(item, "hidden_face_count", 0) or 0))
            visible_faces = max(0, total_faces - hidden_faces)
            status = str(getattr(item, "review_status", "") or "").strip().lower()
            if status == "detected":
                if visible_faces > 0:
                    overlay_by_path[image_path] = f"{visible_faces} face{'s' if visible_faces != 1 else ''}"
                    subtitle_by_path[image_path] = "Detected faces"
                else:
                    overlay_by_path[image_path] = "No matching faces"
                    subtitle_by_path[image_path] = "Open photo to load face details"
            elif status == "tiny_hidden":
                overlay_by_path[image_path] = "Tiny detections hidden"
                subtitle_by_path[image_path] = "Turn on Show Tiny Detections"
            elif status == "no_faces":
                overlay_by_path[image_path] = "No faces detected"
                subtitle_by_path[image_path] = "Scanned"
            else:
                overlay_by_path[image_path] = "Not scanned"
                subtitle_by_path[image_path] = scope_label
            context_by_path[image_path] = {
                "face_review": {
                    "status": status,
                    "total_face_count": total_faces,
                    "hidden_face_count": hidden_faces,
                    "dirty": False,
                    "quality_review_count": 0,
                },
                "indexed_faces": {
                    "count": visible_faces,
                    "faces": [],
                },
            }
            review_by_path[image_path] = item
            if self._face_photo_filter_matches(
                item,
                photo_filter,
                visible_face_count=visible_faces,
            ):
                paths.append(image_path)
        return FaceReviewPublishSnapshot(
            request_id=int(request_id),
            folder=str(folder or ""),
            all_review_images=tuple(review_images or []),
            filtered_review_images=tuple(sorted_review_images),
            paths=tuple(paths),
            overlay_by_path=overlay_by_path,
            subtitle_by_path=subtitle_by_path,
            face_boxes_by_path={},
            face_box_states_by_path={},
            context_by_path=context_by_path,
            review_by_path=review_by_path,
            summary_text=self._face_review_summary_text(
                sorted_review_images,
                folder=str(folder or ""),
                include_tiny_faces=bool(include_tiny_faces),
                summary_notice=str(summary_notice or ""),
            ),
            sort_mode=self._face_review_sort_mode(),
            lazy_publish=True,
            status_text=str(status_text or ""),
        )

    @classmethod
    def _build_lazy_face_review_hydration_snapshot(
        cls,
        *,
        request_id: int,
        publish_request_id: int,
        image_paths: list[str],
        review_by_path: dict[str, FaceFolderReviewImage],
        scope_label: str,
    ) -> FaceReviewHydrationSnapshot:
        overlay_by_path: dict[str, str] = {}
        subtitle_by_path: dict[str, str] = {}
        face_boxes_by_path: dict[str, list[tuple[float, float, float, float]]] = {}
        context_by_path: dict[str, dict[str, object]] = {}
        normalized_paths: list[str] = []
        for image_path in list(image_paths or []):
            normalized_path = str(image_path or "").strip()
            if not normalized_path:
                continue
            item = review_by_path.get(normalized_path)
            if item is None:
                continue
            saved_records = list(tuple(item.visible_faces or ()))
            quality_summary = cls._summarize_saved_face_quality(saved_records)
            labels = [record.person_name for record in saved_records if record.person_name]
            unique_labels = list(dict.fromkeys(labels))
            if item.review_status == "detected":
                if saved_records:
                    overlay_by_path[item.image_path] = f"{len(saved_records)} face{'s' if len(saved_records) != 1 else ''}"
                else:
                    overlay_by_path[item.image_path] = "No matching faces"
                base_subtitle = ", ".join(unique_labels[:2]) if unique_labels else "Detected faces"
                if int(quality_summary["review"]) > 0:
                    base_subtitle += f" | {int(quality_summary['review'])} need review"
                subtitle_by_path[item.image_path] = base_subtitle
            elif item.review_status == "tiny_hidden":
                overlay_by_path[item.image_path] = "Tiny detections hidden"
                subtitle_by_path[item.image_path] = "Turn on Show Tiny Detections"
            elif item.review_status == "no_faces":
                overlay_by_path[item.image_path] = "No faces detected"
                subtitle_by_path[item.image_path] = "Scanned"
            else:
                overlay_by_path[item.image_path] = "Not scanned"
                subtitle_by_path[item.image_path] = str(scope_label or "")
            faces_payload: list[dict[str, object]] = []
            normalized_boxes: list[tuple[float, float, float, float]] = []
            for record in saved_records:
                faces_payload.append(
                    {
                        "face_index": int(record.face_index),
                        "bbox": tuple(record.face_bbox),
                        "confidence": float(record.face_confidence),
                        "person_name": str(record.person_name or ""),
                    }
                )
                normalized_box = cls._normalized_face_bbox(
                    tuple(record.face_bbox),
                    image_width=item.image_width,
                    image_height=item.image_height,
                )
                if normalized_box is not None:
                    normalized_boxes.append(normalized_box)
            if normalized_boxes:
                face_boxes_by_path[item.image_path] = normalized_boxes
            context_by_path[item.image_path] = {
                "face_review": {
                    "status": item.review_status,
                    "total_face_count": int(item.total_face_count),
                    "hidden_face_count": int(item.hidden_face_count),
                    "dirty": False,
                    "quality_review_count": int(quality_summary["review"]),
                },
                "indexed_faces": {
                    "count": len(saved_records),
                    "faces": faces_payload,
                },
            }
            normalized_paths.append(str(item.image_path))
        return FaceReviewHydrationSnapshot(
            request_id=int(request_id),
            publish_request_id=int(publish_request_id),
            paths=tuple(normalized_paths),
            overlay_by_path=overlay_by_path,
            subtitle_by_path=subtitle_by_path,
            face_boxes_by_path=face_boxes_by_path,
            face_box_states_by_path={},
            context_by_path=context_by_path,
        )

    def _apply_face_review_hydration_snapshot(self, snapshot: FaceReviewHydrationSnapshot) -> None:
        limit_paths = {str(path) for path in list(snapshot.paths or ()) if str(path or "").strip()}
        if not limit_paths:
            return
        current_gallery_paths = {
            str(path)
            for path in list(getattr(self.results_gallery, "images", []) or [])
            if str(path or "").strip()
        }
        self._face_review_context_by_path.update(dict(snapshot.context_by_path or {}))
        try:
            self.results_gallery.model.set_overlays_by_path(
                dict(snapshot.overlay_by_path or {}),
                dict(snapshot.subtitle_by_path or {}),
                limit_paths=limit_paths,
            )
            self.results_gallery.model.set_face_boxes_by_path(
                dict(snapshot.face_boxes_by_path or {}),
                limit_paths=limit_paths,
            )
            self.results_gallery.model.set_face_box_states_by_path(
                dict(snapshot.face_box_states_by_path or {}),
                limit_paths=limit_paths,
            )
        except Exception:
            pass
        self.results_gallery.inspector_context_provider = lambda p: self._face_review_context_by_path.get(p, {})
        hydrated_paths = {path for path in limit_paths if path in current_gallery_paths}
        self._face_review_hydrated_paths.update(hydrated_paths)
        self._log_face_event("review_paths_hydrated", path_count=len(hydrated_paths), requested_paths=len(limit_paths))

    def _cancel_face_review_hydration_job(self) -> None:
        job = self._face_review_hydration_job
        if job is not None:
            try:
                job.cancel()
            except Exception:
                pass
        self._face_review_hydration_job = None
        self._face_review_hydration_thread = None

    def _on_face_review_hydration_thread_finished(self, thread=None) -> None:
        retained: list[tuple[object | None, object | None]] = []
        for job, retained_thread in self._retained_face_review_hydration_refs:
            if retained_thread is thread:
                continue
            retained.append((job, retained_thread))
        self._retained_face_review_hydration_refs = retained
        if thread is self._face_review_hydration_thread:
            self._face_review_hydration_thread = None
            self._face_review_hydration_job = None

    def _request_face_review_hydration(self, paths: list[str]) -> None:
        target_paths = [
            str(path)
            for path in list(paths or [])
            if str(path or "").strip()
            and str(path or "").strip() in self._face_review_by_path
            and str(path or "").strip() not in self._face_review_hydrated_paths
        ]
        if not target_paths:
            return
        self._cancel_face_review_hydration_job()
        self._face_review_hydration_request_id += 1
        request_id = int(self._face_review_hydration_request_id)
        publish_request_id = int(self._face_review_publish_request_id)
        review_by_path = dict(self._face_review_by_path or {})
        scope_label = self._face_db_scope_label()

        def _run(_progress, cancel_check):
            if cancel_check():
                raise Cancelled()
            snapshot = self._build_lazy_face_review_hydration_snapshot(
                request_id=request_id,
                publish_request_id=publish_request_id,
                image_paths=list(target_paths),
                review_by_path=review_by_path,
                scope_label=scope_label,
            )
            if cancel_check():
                raise Cancelled()
            return snapshot

        job = AsyncJob(_run)

        def _completed(result) -> None:
            if not isinstance(result, FaceReviewHydrationSnapshot):
                return
            if int(result.request_id) != int(self._face_review_hydration_request_id):
                return
            if int(result.publish_request_id) != int(self._face_review_publish_request_id):
                return
            if not self._face_review_lazy_publish_enabled:
                return
            self._apply_face_review_hydration_snapshot(result)

        job.completed.connect(_completed)
        self._face_review_hydration_job = job
        thread = start_job_in_thread(job)
        self._face_review_hydration_thread = thread
        self._retained_face_review_hydration_refs.append((job, thread))
        thread.finished.connect(
            lambda thread=thread: self._on_face_review_hydration_thread_finished(thread),
            Qt.ConnectionType.QueuedConnection,
        )

    def _queue_face_review_path_hydration(
        self,
        paths: list[str],
        *,
        immediate: bool = False,
        replace_existing: bool = False,
    ) -> None:
        if not self._face_review_lazy_publish_enabled:
            return
        queued_paths: list[str] = []
        for path in list(paths or []):
            normalized = str(path or "").strip()
            if not normalized or normalized not in self._face_review_by_path:
                continue
            if normalized in self._face_review_hydrated_paths:
                continue
            queued_paths.append(normalized)
        queued_paths = list(dict.fromkeys(queued_paths))
        if replace_existing:
            self._cancel_face_review_hydration_job()
            self._face_review_pending_hydration_paths = set(queued_paths)
        else:
            self._face_review_pending_hydration_paths.update(queued_paths)
        if not queued_paths and not self._face_review_pending_hydration_paths:
            self._face_review_hydration_timer.stop()
            return
        if immediate:
            self._face_review_hydration_timer.stop()
            self._hydrate_face_review_paths(queued_paths)
            return
        self._face_review_hydration_timer.start(FACE_REVIEW_HYDRATION_DEBOUNCE_MS)

    def _hydrate_face_review_paths(self, paths: list[str]) -> None:
        target_paths = [
            str(path)
            for path in list(paths or [])
            if str(path or "").strip()
            and str(path or "").strip() in self._face_review_by_path
            and str(path or "").strip() not in self._face_review_hydrated_paths
        ]
        if not target_paths:
            return
        snapshot = self._build_lazy_face_review_hydration_snapshot(
            request_id=0,
            publish_request_id=int(self._face_review_publish_request_id),
            image_paths=target_paths,
            review_by_path=dict(self._face_review_by_path or {}),
            scope_label=self._face_db_scope_label(),
        )
        self._apply_face_review_hydration_snapshot(snapshot)

    def _flush_face_review_hydration_paths(self) -> None:
        if not self._face_review_pending_hydration_paths:
            return
        current_paths = list(getattr(self.results_gallery, "images", []) or [])
        path_order = {str(path): index for index, path in enumerate(current_paths)}
        pending = sorted(
            list(self._face_review_pending_hydration_paths),
            key=lambda path: path_order.get(str(path), len(path_order)),
        )
        self._face_review_pending_hydration_paths.clear()
        self._request_face_review_hydration(pending)

    def _refresh_face_review_gallery_state(self) -> None:
        if self._results_kind != "faces_review":
            return
        paths = list(getattr(self.results_gallery, "images", []) or [])
        if not paths:
            self._refresh_detected_faces_review()
            return
        (
            overlay_by_path,
            subtitle_by_path,
            face_boxes_by_path,
            face_box_states_by_path,
            ctx_by_path,
            stats,
        ) = self._build_face_review_gallery_state(paths)
        self._face_review_context_by_path = dict(ctx_by_path)
        summary = (
            f"Loaded {len(paths)} image(s) from {self._face_review_folder}. "
            f"{int(stats['detected_images'])} photo(s) with visible faces, "
            f"{int(stats['no_face_images'])} with no faces, {int(stats['not_scanned_images'])} not scanned. "
            f"{int(stats['visible_face_count'])} visible face(s)."
        )
        if int(stats["review_face_count"]):
            summary += f" {int(stats['review_face_count'])} face(s) still need manual review."
        if int(stats["hidden_face_count"]) and not self._show_tiny_detections_enabled():
            summary += f" {int(stats['hidden_face_count'])} tiny detection(s) hidden."
        self.face_library_review_summary.setText(summary)
        self._log_face_event("review_publish_mode", folder=self._face_review_folder, mode="overlay_only", image_count=len(paths))
        self.results_gallery.model.set_overlays_by_path(overlay_by_path, subtitle_by_path)
        self.results_gallery.model.set_face_boxes_by_path(face_boxes_by_path)
        self.results_gallery.model.set_face_box_states_by_path(face_box_states_by_path)
        self.results_gallery.inspector_context_provider = lambda p: self._face_review_context_by_path.get(p, {})
        self._publish_results(paths, self.results_gallery.inspector_context_provider, overlay_by_path, subtitle_by_path, self._results_kind)
        self._refresh_detected_faces_review()

    def _filtered_face_review_images(self, review_images: list[FaceFolderReviewImage]) -> list[FaceFolderReviewImage]:
        quality_filter = self._current_face_review_quality_filter()
        reason_filter = self._current_face_review_reason_filter()
        if quality_filter == "all" and not reason_filter and not self._face_hidden_rejected_refs:
            return list(review_images or [])
        filtered: list[FaceFolderReviewImage] = []
        for item in list(review_images or []):
            if self._review_faces_for_item(item):
                filtered.append(item)
                continue
            if quality_filter == "all" and not reason_filter and str(getattr(item, "review_status", "")) in {"no_faces", "not_scanned", "tiny_hidden"}:
                filtered.append(item)
        return filtered

    def _cancel_face_review_publish_job(self) -> None:
        job = self._face_review_publish_job
        if job is not None:
            try:
                job.cancel()
            except Exception:
                pass
        self._face_review_publish_job = None
        self._face_review_publish_thread = None

    def _on_face_review_publish_thread_finished(self, thread=None) -> None:
        retained: list[tuple[object | None, object | None]] = []
        for job, retained_thread in self._retained_face_review_publish_refs:
            if retained_thread is thread:
                continue
            retained.append((job, retained_thread))
        self._retained_face_review_publish_refs = retained
        if thread is self._face_review_publish_thread:
            self._face_review_publish_thread = None
            self._face_review_publish_job = None

    def _request_face_review_publish(
        self,
        review_images: list[FaceFolderReviewImage],
        *,
        folder: str,
        migrated_image_paths: list[str] | None = None,
        summary_notice: str = "",
        status_text: str = "",
    ) -> None:
        self._cancel_face_review_publish_job()
        self._face_review_publish_request_id += 1
        request_id = int(self._face_review_publish_request_id)
        filter_state = self._face_review_filter_state()
        hidden_rejected_refs = set(self._face_hidden_rejected_refs)
        dirty_paths = set(self._face_review_draft_dirty_paths)
        drafts_by_path = {
            str(path): list(drafts)
            for path, drafts in self._face_review_drafts_by_path.items()
        }
        scope_label = self._face_db_scope_label()
        include_tiny_faces = self._show_tiny_detections_enabled()
        service = self._active_face_service()
        if self._can_use_lazy_face_review_publish(
            list(review_images or []),
            filter_state=filter_state,
            hidden_rejected_refs=hidden_rejected_refs,
            dirty_paths=dirty_paths,
        ):
            snapshot = self._build_lazy_face_review_publish_snapshot(
                request_id=request_id,
                folder=folder,
                review_images=list(review_images or []),
                scope_label=scope_label,
                include_tiny_faces=include_tiny_faces,
                summary_notice=str(summary_notice or ""),
                status_text=str(status_text or ""),
            )
            if int(snapshot.request_id) == int(self._face_review_publish_request_id):
                self._apply_face_review_publish_snapshot(snapshot)
            return
        if len(review_images or []) <= 24:
            try:
                snapshot = self._build_face_review_publish_snapshot(
                    request_id=request_id,
                    folder=folder,
                    review_images=list(review_images or []),
                    filter_state=filter_state,
                    hidden_rejected_refs=hidden_rejected_refs,
                    dirty_paths=dirty_paths,
                drafts_by_path=drafts_by_path,
                scope_label=scope_label,
                include_tiny_faces=include_tiny_faces,
                service=service,
                migrated_image_paths=list(migrated_image_paths or []),
                summary_notice=str(summary_notice or ""),
                status_text=str(status_text or ""),
                cancel_check=lambda: False,
            )
            except Exception as exc:
                LOGGER.exception("FacePane review publish failed folder=%s image_count=%s", folder, len(review_images or []))
                self.status_label.setText(f"Face review publish failed: {exc}")
                return
            if int(snapshot.request_id) == int(self._face_review_publish_request_id):
                self._apply_face_review_publish_snapshot(snapshot)
            return

        def _run(progress, cancel_check):
            progress(-1, "Preparing folder face review...")
            return self._build_face_review_publish_snapshot(
                request_id=request_id,
                folder=folder,
                review_images=list(review_images or []),
                filter_state=filter_state,
                hidden_rejected_refs=hidden_rejected_refs,
                dirty_paths=dirty_paths,
                drafts_by_path=drafts_by_path,
                scope_label=scope_label,
                include_tiny_faces=include_tiny_faces,
                service=service,
                migrated_image_paths=list(migrated_image_paths or []),
                summary_notice=str(summary_notice or ""),
                status_text=str(status_text or ""),
                cancel_check=cancel_check,
            )

        job = AsyncJob(_run)
        job.progress.connect(lambda _value, text: self.status_label.setText(str(text)))

        def _failed(message: str) -> None:
            text = str(message or "").strip()
            if text.casefold() == "cancelled":
                return
            self.status_label.setText(f"Face review publish failed: {text}")

        def _completed(result) -> None:
            if not isinstance(result, FaceReviewPublishSnapshot):
                return
            if int(result.request_id) != int(self._face_review_publish_request_id):
                return
            self._apply_face_review_publish_snapshot(result)

        job.failed.connect(_failed)
        job.completed.connect(_completed)
        self._face_review_publish_job = job
        thread = start_job_in_thread(job)
        self._face_review_publish_thread = thread
        self._retained_face_review_publish_refs.append((job, thread))
        thread.finished.connect(
            lambda thread=thread: self._on_face_review_publish_thread_finished(thread),
            Qt.ConnectionType.QueuedConnection,
        )

    def _build_face_review_publish_snapshot(
        self,
        *,
        request_id: int,
        folder: str,
        review_images: list[FaceFolderReviewImage],
        filter_state: dict[str, str],
        hidden_rejected_refs: set[tuple[str, int]],
        dirty_paths: set[str],
        drafts_by_path: dict[str, list[EditableFaceDraft]],
        scope_label: str,
        include_tiny_faces: bool,
        service,
        migrated_image_paths: list[str],
        summary_notice: str,
        status_text: str,
        cancel_check,
    ) -> FaceReviewPublishSnapshot:
        quality_filter = str(filter_state.get("quality_filter", "all") or "all").strip().lower()
        reason_filter = str(filter_state.get("reason_filter", "") or "").strip().lower()
        sort_mode = str(filter_state.get("sort_mode", "name") or "name").strip().lower()
        photo_filter = str(filter_state.get("photo_filter", "with_faces") or "with_faces").strip().lower()
        include_unfiltered_extras = quality_filter == "all" and not reason_filter and not hidden_rejected_refs
        filtered_item_state_by_path: dict[str, dict[str, object]] = {}
        face_review_images: list[FaceFolderReviewImage] = []
        for item in list(review_images or []):
            if cancel_check():
                raise Cancelled()
            image_path = str(getattr(item, "image_path", "") or "")
            is_dirty = image_path in dirty_paths
            drafts = list(drafts_by_path.get(image_path, []))
            filtered_drafts: list[EditableFaceDraft] = []
            saved_records: list[IndexedFaceRecord] = []
            quality_summary = {"clean": 0, "review": 0, "reject": 0}
            faces_payload: list[dict[str, object]] = []
            normalized_boxes: list[tuple[float, float, float, float]] = []
            if is_dirty:
                for index, draft in enumerate(drafts):
                    face_index = int(draft.face_index if int(draft.face_index) >= 0 else index)
                    status, reasons = self._face_quality_status_for_service(
                        service,
                        image_path,
                        tuple(int(value) for value in draft.bbox),
                        confidence=float(draft.confidence or 1.0),
                    )
                    if not self._review_face_matches_filters_state(
                        quality_filter=quality_filter,
                        reason_filter=reason_filter,
                        hidden_rejected_refs=hidden_rejected_refs,
                        image_path=image_path,
                        face_index=face_index,
                        status=status,
                        reasons=reasons,
                    ):
                        continue
                    filtered_drafts.append(draft)
                    quality_summary[status if status in quality_summary else "clean"] += 1
                    faces_payload.append(
                        {
                            "face_index": face_index,
                            "bbox": tuple(int(value) for value in draft.bbox),
                            "confidence": float(draft.confidence or 1.0),
                            "person_name": str(draft.person_name or ""),
                            "source": str(draft.source or ""),
                        }
                    )
                    normalized = self._normalized_face_bbox(
                        tuple(int(value) for value in draft.bbox),
                        image_width=int(getattr(item, "image_width", 0) or 0),
                        image_height=int(getattr(item, "image_height", 0) or 0),
                    )
                    if normalized is not None:
                        normalized_boxes.append(normalized)
            else:
                for record in tuple(getattr(item, "visible_faces", ()) or ()):
                    status = str(getattr(record, "quality_status", "clean") or "clean")
                    reasons = tuple(str(value) for value in getattr(record, "quality_reasons", ()) or ())
                    if not self._review_face_matches_filters_state(
                        quality_filter=quality_filter,
                        reason_filter=reason_filter,
                        hidden_rejected_refs=hidden_rejected_refs,
                        image_path=image_path,
                        face_index=int(getattr(record, "face_index", 0) or 0),
                        status=status,
                        reasons=reasons,
                    ):
                        continue
                    saved_records.append(record)
                    quality_summary[status if status in quality_summary else "clean"] += 1
                    faces_payload.append(
                        {
                            "face_index": int(record.face_index),
                            "bbox": tuple(record.face_bbox),
                            "confidence": float(record.face_confidence),
                            "person_name": str(record.person_name or ""),
                        }
                    )
                    normalized = self._normalized_face_bbox(
                        tuple(record.face_bbox),
                        image_width=int(getattr(item, "image_width", 0) or 0),
                        image_height=int(getattr(item, "image_height", 0) or 0),
                    )
                    if normalized is not None:
                        normalized_boxes.append(normalized)
            include_item = bool(filtered_drafts or saved_records)
            review_status = str(getattr(item, "review_status", "") or "")
            if not include_item and include_unfiltered_extras and review_status in {"no_faces", "not_scanned", "tiny_hidden"}:
                include_item = True
            if not include_item:
                continue
            face_review_images.append(item)
            filtered_item_state_by_path[image_path] = {
                "is_dirty": bool(is_dirty),
                "drafts": filtered_drafts,
                "saved_records": saved_records,
                "quality_summary": quality_summary,
                "faces_payload": faces_payload,
                "normalized_boxes": normalized_boxes,
            }
        sorted_face_review_images = self._sort_face_review_images_for_mode(face_review_images, mode=sort_mode)
        filtered_review_images = [
            item
            for item in sorted_face_review_images
            if self._face_photo_filter_matches(
                item,
                photo_filter,
                visible_face_count=len(list(filtered_item_state_by_path.get(str(item.image_path), {}).get("drafts", []) or []))
                + len(list(filtered_item_state_by_path.get(str(item.image_path), {}).get("saved_records", []) or [])),
                needs_review=bool(
                    int(dict(filtered_item_state_by_path.get(str(item.image_path), {}).get("quality_summary", {}) or {}).get("review", 0) or 0)
                    or int(dict(filtered_item_state_by_path.get(str(item.image_path), {}).get("quality_summary", {}) or {}).get("reject", 0) or 0)
                ),
                dirty=bool(filtered_item_state_by_path.get(str(item.image_path), {}).get("is_dirty", False)),
            )
        ]
        sorted_review_images = list(filtered_review_images)
        overlay_by_path: dict[str, str] = {}
        subtitle_by_path: dict[str, str] = {}
        face_boxes_by_path: dict[str, list[tuple[float, float, float, float]]] = {}
        face_box_states_by_path: dict[str, str] = {}
        context_by_path: dict[str, dict[str, object]] = {}
        review_by_path: dict[str, FaceFolderReviewImage] = {
            str(item.image_path): item for item in list(review_images or []) if str(getattr(item, "image_path", "") or "")
        }
        stats = {
            "detected_images": 0,
            "no_face_images": 0,
            "not_scanned_images": 0,
            "hidden_face_count": 0,
            "visible_face_count": 0,
            "review_face_count": 0,
        }
        paths: list[str] = []
        for item in sorted_review_images:
            if cancel_check():
                raise Cancelled()
            image_path = str(item.image_path)
            state = filtered_item_state_by_path.get(image_path, {})
            is_dirty = bool(state.get("is_dirty", False))
            drafts = list(state.get("drafts", []) or [])
            saved_records = list(state.get("saved_records", []) or [])
            quality_summary = dict(state.get("quality_summary", {}) or {"clean": 0, "review": 0, "reject": 0})
            normalized_boxes = list(state.get("normalized_boxes", []) or [])
            faces_payload = list(state.get("faces_payload", []) or [])
            paths.append(image_path)
            if is_dirty:
                face_box_states_by_path[image_path] = "draft"
                stats["visible_face_count"] += len(drafts)
                stats["review_face_count"] += int(quality_summary.get("review", 0) or 0)
                if drafts:
                    stats["detected_images"] += 1
                    overlay_by_path[image_path] = f"{len(drafts)} face{'s' if len(drafts) != 1 else ''}"
                    subtitle_by_path[image_path] = (
                        f"Unsaved edits | {int(quality_summary.get('review', 0) or 0)} need review"
                        if int(quality_summary.get("review", 0) or 0) > 0
                        else "Unsaved face edits"
                    )
                else:
                    stats["no_face_images"] += 1
                    overlay_by_path[image_path] = "No faces detected"
                    subtitle_by_path[image_path] = "Unsaved face edits"
            else:
                stats["visible_face_count"] += len(saved_records)
                stats["hidden_face_count"] += int(getattr(item, "hidden_face_count", 0) or 0)
                stats["review_face_count"] += int(quality_summary.get("review", 0) or 0)
                labels = [record.person_name for record in saved_records if record.person_name]
                unique_labels = list(dict.fromkeys(labels))
                if str(getattr(item, "review_status", "")) == "detected":
                    if saved_records:
                        stats["detected_images"] += 1
                        overlay_by_path[image_path] = f"{len(saved_records)} face{'s' if len(saved_records) != 1 else ''}"
                    else:
                        stats["no_face_images"] += 1
                        overlay_by_path[image_path] = "No matching faces"
                    base_subtitle = ", ".join(unique_labels[:2]) if unique_labels else "Detected faces"
                    if int(quality_summary.get("review", 0) or 0) > 0:
                        base_subtitle += f" | {int(quality_summary.get('review', 0) or 0)} need review"
                    subtitle_by_path[image_path] = base_subtitle
                elif str(getattr(item, "review_status", "")) == "tiny_hidden":
                    overlay_by_path[image_path] = "Tiny detections hidden"
                    subtitle_by_path[image_path] = "Turn on Show Tiny Detections"
                elif str(getattr(item, "review_status", "")) == "no_faces":
                    stats["no_face_images"] += 1
                    overlay_by_path[image_path] = "No faces detected"
                    subtitle_by_path[image_path] = "Scanned"
                else:
                    stats["not_scanned_images"] += 1
                    overlay_by_path[image_path] = "Not scanned"
                    subtitle_by_path[image_path] = scope_label
            if normalized_boxes:
                face_boxes_by_path[image_path] = normalized_boxes
            context_by_path[image_path] = {
                "face_review": {
                    "status": "draft" if is_dirty else getattr(item, "review_status", ""),
                    "total_face_count": len(drafts) if is_dirty else int(getattr(item, "total_face_count", 0) or 0),
                    "hidden_face_count": 0 if is_dirty else int(getattr(item, "hidden_face_count", 0) or 0),
                    "dirty": bool(is_dirty),
                    "quality_review_count": int(quality_summary.get("review", 0) or 0),
                },
                "indexed_faces": {
                    "count": len(drafts) if is_dirty else len(saved_records),
                    "faces": faces_payload,
                },
            }
        if sorted_review_images:
            summary_text = (
                f"Loaded {len(sorted_review_images)} image(s) from {folder}. "
                f"{int(stats['detected_images'])} photo(s) with visible faces, {int(stats['no_face_images'])} with no faces, "
                f"{int(stats['not_scanned_images'])} not scanned. {int(stats['visible_face_count'])} visible face(s)."
            )
            if int(stats["review_face_count"]):
                summary_text += f" {int(stats['review_face_count'])} face(s) still need manual review."
            if int(stats["hidden_face_count"]) and not include_tiny_faces:
                summary_text += f" {int(stats['hidden_face_count'])} tiny detection(s) hidden."
        else:
            summary_text = f"No images found in {folder}."
        if str(summary_notice or "").strip():
            summary_text = f"{summary_text} {str(summary_notice).strip()}".strip()
        if quality_filter == "all" and not reason_filter and not hidden_rejected_refs:
            summary_text = self._face_review_summary_text(
                sorted_face_review_images,
                folder=str(folder or ""),
                include_tiny_faces=bool(include_tiny_faces),
                summary_notice=str(summary_notice or ""),
            )
        return FaceReviewPublishSnapshot(
            request_id=int(request_id),
            folder=str(folder or ""),
            all_review_images=tuple(review_images or []),
            filtered_review_images=tuple(sorted_face_review_images),
            paths=tuple(paths),
            overlay_by_path=overlay_by_path,
            subtitle_by_path=subtitle_by_path,
            face_boxes_by_path=face_boxes_by_path,
            face_box_states_by_path=face_box_states_by_path,
            context_by_path=context_by_path,
            review_by_path=review_by_path,
            summary_text=summary_text,
            sort_mode=sort_mode,
            migrated_image_paths=tuple(str(path) for path in list(migrated_image_paths or []) if str(path or "").strip()),
            status_text=str(status_text or ""),
        )

    def _apply_face_review_publish_snapshot(self, snapshot: FaceReviewPublishSnapshot) -> None:
        folder = str(snapshot.folder or "")
        paths = list(snapshot.paths or [])
        migrated_image_paths = [str(path) for path in list(snapshot.migrated_image_paths or ()) if str(path or "").strip()]
        lazy_publish = bool(getattr(snapshot, "lazy_publish", False))
        self._log_face_event("review_publish_start", folder=folder, image_count=len(paths))
        self._set_results_kind("faces_review")
        self.results_list.clear()
        self._face_review_all_images = list(snapshot.all_review_images or [])
        self._face_review_folder = folder
        self._face_review_images = list(snapshot.filtered_review_images or [])
        self._face_review_sorted_mode = str(getattr(snapshot, "sort_mode", "") or "")
        self._face_review_by_path = dict(snapshot.review_by_path or {})
        self._face_review_context_by_path = dict(snapshot.context_by_path or {})
        self._face_photo_context_kind = "folder"
        self._face_photo_context_title = "Folder photos"
        self._folder_photo_snapshot = FacePhotoViewSnapshot(
            title="Folder photos",
            paths=tuple(paths),
            overlay_by_path=dict(snapshot.overlay_by_path or {}),
            subtitle_by_path=dict(snapshot.subtitle_by_path or {}),
            face_boxes_by_path=dict(snapshot.face_boxes_by_path or {}),
            face_box_states_by_path=dict(snapshot.face_box_states_by_path or {}),
            context_by_path=dict(snapshot.context_by_path or {}),
            selected_path=str(self._face_review_selected_path or ""),
            scroll_value=int(self.results_gallery.list_view.verticalScrollBar().value()),
        )
        self._face_review_lazy_publish_enabled = bool(lazy_publish)
        self._face_review_hydration_timer.stop()
        self._cancel_face_review_hydration_job()
        self._face_review_pending_hydration_paths.clear()
        self._face_review_hydrated_paths = set() if lazy_publish else set(paths)
        if not paths:
            self._face_review_selected_path = ""
            self.face_library_review_summary.setText(str(snapshot.summary_text or f"No images found in {folder}."))
            self.face_selected_photo_label.setText("None")
            self.face_scanned_summary.setText(str(snapshot.summary_text or "No images found for folder review."))
            self._publish_results_gallery_paths([])
            filter_label = str(self.face_photo_filter.currentText() or "With Faces")
            self.results_gallery.set_empty_state(
                f"No photos match {filter_label}",
                "Choose another Photos filter. Faces and Groups remain available in their own views.",
            )
            self._refresh_detected_faces_review()
            self._sync_face_review_result_groups()
            self._update_face_selected_context_label()
            self._update_face_results_context()
            self._log_face_event("review_publish_complete", folder=folder, image_count=0)
            return
        self.face_library_review_summary.setText(str(snapshot.summary_text or ""))
        selected_path = self._face_review_selected_path if self._face_review_selected_path in self._face_review_by_path else ""
        if not selected_path:
            selected_path = paths[0]
        self._face_review_publish_in_progress = True
        try:
            provider = lambda p: self._face_review_context_by_path.get(p, {})
            if migrated_image_paths and self._gallery_paths_match(paths):
                limit_paths = set(paths)
                try:
                    self.results_gallery.model.set_overlays_by_path(
                        dict(snapshot.overlay_by_path or {}),
                        dict(snapshot.subtitle_by_path or {}),
                        limit_paths=limit_paths,
                    )
                    self.results_gallery.model.set_face_boxes_by_path(
                        dict(snapshot.face_boxes_by_path or {}),
                        limit_paths=limit_paths,
                    )
                    self.results_gallery.model.set_face_box_states_by_path(
                        dict(snapshot.face_box_states_by_path or {}),
                        limit_paths=limit_paths,
                    )
                except Exception:
                    pass
                self.results_gallery.inspector_context_provider = provider
                self._publish_results(paths, provider, dict(snapshot.overlay_by_path or {}), dict(snapshot.subtitle_by_path or {}), self._results_kind)
                self.results_gallery.invalidate_image_paths(migrated_image_paths, reload_visible=True)
            else:
                self._publish_results_gallery_paths(
                    paths,
                    overlay_by_path=dict(snapshot.overlay_by_path or {}),
                    subtitle_by_path=dict(snapshot.subtitle_by_path or {}),
                    face_boxes_by_path=dict(snapshot.face_boxes_by_path or {}),
                    face_box_states_by_path=dict(snapshot.face_box_states_by_path or {}),
                    context_provider=provider,
                    clear_pixmaps=True,
                    reset_scroll=True,
                )
            self._refresh_detected_faces_review()
            self._sync_face_review_result_groups()
            self._load_face_review_selection(selected_path)
            if lazy_publish:
                self._queue_face_review_path_hydration([selected_path], immediate=False)
        finally:
            self._face_review_publish_in_progress = False
        self.status_label.setText(str(snapshot.status_text or f"Loaded face review for {folder}."))
        self._update_face_results_context()
        self._log_face_event("review_publish_complete", folder=folder, image_count=len(paths))

    def _apply_face_folder_review(
        self,
        review_images: list[FaceFolderReviewImage],
        *,
        folder: str,
        migrated_image_paths: list[str] | None = None,
        summary_notice: str = "",
        status_text: str = "",
    ) -> None:
        self._request_face_review_publish(
            review_images,
            folder=folder,
            migrated_image_paths=list(migrated_image_paths or []),
            summary_notice=str(summary_notice or ""),
            status_text=str(status_text or ""),
        )

    def _load_face_review_selection(self, image_path: str) -> None:
        try:
            if not self._face_selection_sync_in_progress:
                detected_selection_model = getattr(self, "face_detected_faces_list", None)
                detected_selection_model = detected_selection_model.selectionModel() if detected_selection_model is not None else None
                if detected_selection_model is not None:
                    detected_selection_model.blockSignals(True)
                    detected_selection_model.clearSelection()
                    detected_selection_model.blockSignals(False)
                self._clear_active_face_selection()
            item = self._face_review_by_path.get(str(image_path))
            self.face_scanned_model.set_items([])
            if item is None:
                self._face_review_selected_path = ""
                self.face_selected_photo_label.setText("None")
                self.face_scanned_summary.setText("Choose a photo in the folder gallery to load its detected faces.")
                self._update_face_selected_context_label()
                return
            if self._face_review_lazy_publish_enabled:
                self._queue_face_review_path_hydration([str(item.image_path)], immediate=True)
            is_dirty = item.image_path in self._face_review_draft_dirty_paths
            draft_faces = list(self._face_review_drafts_by_path.get(item.image_path, []))
            self._log_face_event(
                "review_selection_load",
                image_path=item.image_path,
                status="draft" if is_dirty else item.review_status,
                face_count=len(draft_faces) if is_dirty else len(item.visible_faces),
            )
            self._face_review_selected_path = item.image_path
            self.face_selected_photo_label.setText(Path(item.image_path).name)
            face_items: list[FaceTileItem] = []
            if is_dirty:
                for index, draft in enumerate(draft_faces):
                    label = draft.person_name or "Unlabeled"
                    face_items.append(
                        FaceTileItem(
                            image_path=str(item.image_path),
                            face_index=int(index),
                            bbox=tuple(int(value) for value in draft.bbox),
                            title=f"{label}\nFace #{index + 1}",
                            tooltip=(
                                f"{item.image_path}\nface #{index + 1}\n"
                                f"bbox={tuple(int(value) for value in draft.bbox)}\n"
                                f"face_conf={float(draft.confidence):.3f}\nunsaved draft"
                            ),
                            status="draft",
                            draft_slot=int(index),
                            saved_face_index=int(draft.face_index),
                            payload=draft,
                        )
                    )
            else:
                for record in item.visible_faces:
                    label = record.person_name or "Unlabeled"
                    face_items.append(
                        FaceTileItem(
                            image_path=str(record.image_path),
                            face_index=int(record.face_index),
                            bbox=tuple(int(value) for value in record.face_bbox),
                            title=f"{label}\nFace #{int(record.face_index) + 1}",
                            tooltip=(
                                f"{record.image_path}\nface #{int(record.face_index) + 1}\n"
                                f"bbox={tuple(record.face_bbox)}\n"
                                f"face_conf={float(record.face_confidence):.3f}"
                            ),
                            status=str(record.quality_status or "saved"),
                            draft_slot=int(record.face_index),
                            saved_face_index=int(record.face_index),
                            payload=record,
                        )
                    )
            self.face_scanned_model.set_items(face_items)
            self._schedule_selected_face_tile_loads()
            if is_dirty:
                self.face_scanned_summary.setText(
                    f"{Path(item.image_path).name} has unsaved face edits. "
                    "Red boxes in the gallery and inspector show the draft positions. Save Face Edits in the inspector to commit them."
                )
            elif item.review_status == "detected":
                quality_review_count = int(
                    (self._face_review_context_by_path.get(item.image_path, {}).get("face_review", {}) or {}).get("quality_review_count", 0)
                )
                self.face_scanned_summary.setText(
                    f"{len(item.visible_faces)} detected face(s) in {Path(item.image_path).name}. "
                    "Select one or more thumbnails to name them. Select exactly one to search for matching photos."
                    + (
                        f" {quality_review_count} face(s) still look suspicious and should be reviewed first."
                        if quality_review_count > 0
                        else ""
                    )
                )
            elif item.review_status == "tiny_hidden":
                self.face_scanned_summary.setText(
                    f"{Path(item.image_path).name} only has tiny detections hidden by the current filter."
                )
            elif item.review_status == "no_faces":
                self.face_scanned_summary.setText(
                    f"{Path(item.image_path).name} was scanned and no faces were detected."
                )
            else:
                self.face_scanned_summary.setText(
                    f"{Path(item.image_path).name} has not been scanned in {self._face_db_scope_label()}."
                )
            self._update_face_selected_context_label()
        except Exception:
            LOGGER.exception("FacePane selection load failed image_path=%s", image_path)
            self.face_scanned_summary.setText("Failed to load the selected photo face strip. Check logs for details.")

    def _show_indexed_face_images(self, records: list[IndexedFaceRecord]) -> None:
        grouped: dict[str, list[IndexedFaceRecord]] = {}
        for record in records:
            grouped.setdefault(record.image_path, []).append(record)
        self._set_results_kind("faces")
        self.results_list.clear()
        if not grouped:
            self._publish_results_gallery_paths([])
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
        provider = lambda p: ctx_by_path.get(p, {})
        self._publish_results_gallery_paths(
            paths,
            overlay_by_path=overlay_by_path,
            subtitle_by_path=subtitle_by_path,
            context_provider=provider,
            clear_pixmaps=True,
            reset_scroll=True,
        )

    def _selected_scanned_faces(self) -> list[IndexedFaceRecord]:
        records: list[IndexedFaceRecord] = []
        for item in self._selected_tile_items(getattr(self, "face_scanned_list", None), getattr(self, "face_scanned_model", None)):
            record = item.payload
            if isinstance(record, IndexedFaceRecord):
                records.append(record)
        return records

    def _select_scanned_face_records(self, records: list[IndexedFaceRecord]) -> int:
        refs = {(record.image_path, int(record.face_index)) for record in records}
        selection_model = self.face_scanned_list.selectionModel()
        if selection_model is None:
            return 0
        selection_model.blockSignals(True)
        selection_model.clearSelection()
        selected = 0
        first_index = QModelIndex()
        for row in range(self.face_scanned_model.rowCount()):
            index = self.face_scanned_model.index(row, 0)
            record = self.face_scanned_model.data(index, FaceTileListModel.PayloadRole)
            if not isinstance(record, IndexedFaceRecord):
                continue
            if (record.image_path, int(record.face_index)) not in refs:
                continue
            selection_model.select(index, QItemSelectionModel.SelectionFlag.Select)
            if not first_index.isValid():
                first_index = index
            selected += 1
        selection_model.blockSignals(False)
        if first_index.isValid():
            self.face_scanned_list.scrollTo(first_index)
        self._on_scanned_face_selection_changed()
        return selected

    def _make_face_thumbnail_icon(self, record: IndexedFaceRecord) -> QIcon:
        return self._make_face_thumbnail_icon_for_bbox(record.image_path, tuple(record.face_bbox), face_index=int(record.face_index))

    def _make_face_thumbnail_icon_for_bbox(
        self,
        image_path: str,
        bbox: tuple[int, int, int, int],
        *,
        face_index: int = -1,
        requested_size: QSize | None = None,
    ) -> QIcon:
        item = FaceTileItem(
            image_path=str(image_path),
            face_index=int(face_index),
            bbox=tuple(int(value) for value in bbox),
            title="",
            tooltip="",
            saved_face_index=int(face_index),
        )
        image = self._image_for_face_tile(item, QSize(requested_size or QSize(72, 72)))
        if image.isNull():
            return QIcon()
        return QIcon(QPixmap.fromImage(image))

    def _on_scanned_face_selection_changed(self) -> None:
        if self._face_selection_sync_in_progress:
            return
        items = self._selected_tile_items(self.face_scanned_list, self.face_scanned_model)
        records = self._selected_scanned_faces()
        if not records:
            if items:
                self._set_active_face_selection(items, source="selected_photo")
            else:
                self._clear_active_face_selection(source="selected_photo")
            if self._face_review_selected_path in self._face_review_draft_dirty_paths and self._selected_tile_indexes(getattr(self, "face_scanned_list", None)):
                self.face_scanned_summary.setText(
                    "These face thumbnails are unsaved draft edits. Save Face Edits in the inspector before using them for naming or search."
                )
                self._update_face_selected_context_label()
                return
            if self.face_scanned_model.rowCount() > 0:
                self.face_scanned_summary.setText(
                    "Select one or more face thumbnails to name them. Select exactly one to find matching photos."
                )
            self._update_face_selected_context_label()
            return
        first = records[0]
        self._set_active_face_selection(items, source="selected_photo")
        self.face_scanned_summary.setText(
            f"Selected {len(records)} face(s). First: {Path(first.image_path).name} #{int(first.face_index) + 1} | "
            f"bbox={tuple(first.face_bbox)} | conf={float(first.face_confidence):.3f}"
        )
        self._update_face_selected_context_label()

    def _save_selected_face_name(self) -> None:
        records = self._selected_scanned_faces()
        if not records:
            errorBox("No face selected", "Select one or more scanned faces first.")
            return
        refs = [(record.image_path, int(record.face_index)) for record in records]
        self._save_face_refs_name(refs, source_label="Selected photo")

    def _search_selected_face(self) -> None:
        records = self._selected_scanned_faces()
        if not records:
            errorBox("No face selected", "Select one or more scanned faces to search across photos.")
            return
        self._search_face_refs(
            [(record.image_path, int(record.face_index)) for record in records],
            source_label="Selected photo",
        )

    def _save_person_profile(self) -> None:
        person_name = self.face_label_name.text().strip() or self.face_name_query.text().strip()
        if not person_name:
            errorBox("Missing name", "Enter or select an identity name first.")
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
                favorite=bool(self.face_profile_favorite.isChecked()) if hasattr(self, "face_profile_favorite") else None,
                birth_date=self.face_profile_birth_date.text().strip() if hasattr(self, "face_profile_birth_date") else None,
                hidden=bool(self.face_profile_hidden.isChecked()) if hasattr(self, "face_profile_hidden") else None,
            )
            return True

        def _done(_ok: bool) -> None:
            self.status_label.setText(f"Saved profile for {person_name}.")
            self.refresh_face_library(refresh_people=True, reason="profile saved")
            self._maybe_refresh_global_face_album(reason="profile saved")

        self._start_job("Saving identity profile", _run, _done)

    def _hide_selected_faces(self) -> None:
        records = self._selected_scanned_faces()
        if not records:
            errorBox("No faces selected", "Select one or more saved face tiles in Face Library first.")
            return
        refs = [(str(record.image_path), int(record.face_index)) for record in records]

        def _run(progress, cancel_check):
            progress(-1, "Hiding selected faces...")
            _ = cancel_check
            service = self._active_face_service()
            for image_path, face_index in refs:
                service.hide_face(image_path, face_index)
            return len(refs)

        def _done(count: int) -> None:
            self.status_label.setText(f"Hidden {int(count)} selected face(s).")
            self._request_face_library_refresh(refresh_people=True, reason="faces hidden", force_refresh=True)
            self._maybe_refresh_global_face_album(reason="faces hidden")

        self._start_job("Hiding selected faces", _run, _done)

    def _populate_profile_fields(self, person_name: str) -> None:
        profiles = list(getattr(self, "_face_profiles_by_name", {}).values())
        if not profiles:
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
            if hasattr(self, "face_profile_birth_date"):
                self.face_profile_birth_date.setText(str(getattr(profile, "birth_date", "") or ""))
            if hasattr(self, "face_profile_favorite"):
                self.face_profile_favorite.setChecked(bool(getattr(profile, "favorite", False)))
            if hasattr(self, "face_profile_hidden"):
                self.face_profile_hidden.setChecked(bool(getattr(profile, "hidden", False)))
            break

    def _on_person_clicked(self, item) -> None:
        profile = item.data(Qt.ItemDataRole.UserRole)
        if hasattr(self, "face_library_tabs"):
            self.face_library_tabs.setCurrentIndex(0)
        if isinstance(profile, FaceLibraryUnlabeledGroup):
            if hasattr(self, "face_people_summary"):
                self.face_people_summary.setText(
                    f"Loaded unlabeled group from {Path(profile.image_path).name}. The folder review now focuses that photo."
                )
            self.face_name_query.clear()
            self.face_label_name.clear()
            self.face_profile_notes.clear()
            self.face_profile_tags.clear()
            if hasattr(self, "face_profile_birth_date"):
                self.face_profile_birth_date.clear()
            if hasattr(self, "face_profile_favorite"):
                self.face_profile_favorite.setChecked(False)
            if hasattr(self, "face_profile_hidden"):
                self.face_profile_hidden.setChecked(False)
            if hasattr(self, "person_name"):
                try:
                    self.person_name.clear()
                except Exception:
                    pass
            self._focus_face_review_image(profile.image_path)
            selected = self._select_scanned_face_records(list(profile.records))
            if selected:
                self.face_scanned_summary.setText(
                    f"Selected {selected} unlabeled face(s) from {Path(profile.image_path).name}."
                )
            else:
                self.face_scanned_summary.setText(
                    f"Loaded {profile.face_count} unlabeled face(s) from {Path(profile.image_path).name}."
                )
            return
        if profile is not None and hasattr(profile, "person_name"):
            name = str(profile.person_name or "").strip()
        else:
            text = str(item.text() or "")
            name = text.split("|", 1)[0].strip()
        if name:
            self.face_name_query.setText(name)
            self.face_label_name.setText(name)
            if hasattr(self, "face_find_name_query"):
                self.face_find_name_query.setText(name)
            if hasattr(self, "person_name"):
                try:
                    self.person_name.setText(name)
                except Exception:
                    pass
            if profile is not None:
                try:
                    self.face_profile_notes.setText(str(getattr(profile, "notes", "") or ""))
                    self.face_profile_tags.setText(", ".join(getattr(profile, "tags", ()) or ()))
                    if hasattr(self, "face_profile_birth_date"):
                        self.face_profile_birth_date.setText(str(getattr(profile, "birth_date", "") or ""))
                    if hasattr(self, "face_profile_favorite"):
                        self.face_profile_favorite.setChecked(bool(getattr(profile, "favorite", False)))
                    if hasattr(self, "face_profile_hidden"):
                        self.face_profile_hidden.setChecked(bool(getattr(profile, "hidden", False)))
                except Exception:
                    pass
            if hasattr(self, "face_people_summary"):
                self.face_people_summary.setText(
                    f"Loaded saved identity '{name}'. Use Find Photos by Saved Name or update the profile below."
                )
            if hasattr(self, "face_identity_list"):
                row = self.face_identity_model.row_for_payload(name)
                if row >= 0:
                    identity_index = self.face_identity_model.index(row, 0)
                    self.face_identity_list.setCurrentIndex(identity_index)
                    self.face_identity_list.selectionModel().select(
                        identity_index,
                        QItemSelectionModel.SelectionFlag.ClearAndSelect | QItemSelectionModel.SelectionFlag.Rows,
                    )
                self._refresh_selected_face_identity()

    def _search_by_name(self) -> None:
        service = self._active_face_service()
        name = self.face_find_name_query.text().strip() if hasattr(self, "face_find_name_query") else ""
        if not name:
            name = self.face_name_query.text().strip()
        if not name and hasattr(self, "person_name"):
            name = self.person_name.text().strip()
        if not name:
            errorBox("Missing name", "Enter or select a saved identity name first.")
            return
        folder = self.face_name_folder_filter.text().strip()
        scope_paths = self._current_scope_paths()
        if not folder and self.search_only_current_folder.isChecked():
            folder = self._current_directory()
        min_score = float(self.face_name_min_score.value())
        min_score_arg = self._effective_face_search_min_score(min_score if min_score > 0.0 else 0.0)

        def _run(progress, cancel_check):
            progress(-1, "Searching by name...")
            _ = cancel_check
            return service.search_by_person_name(
                name,
                top_k=int(self.face_name_top_k.value()),
                min_score=min_score_arg,
                folder_prefix=folder,
                candidate_paths=scope_paths,
                include_tiny_faces=self._show_tiny_detections_enabled(),
            )

        def _done(results) -> None:
            for widget_name in ("face_find_name_query", "face_name_query", "face_label_name", "person_name"):
                widget = getattr(self, widget_name, None)
                if widget is not None:
                    widget.setText(name)
            self.status_label.setText(f"Found {len(results)} saved-name and similar-face matches for '{name}'.")
            self._show_face_match_results(
                list(results or []),
                title=f"Name + similar: {name}",
                summary=(
                    f"Saved identity '{name}' returned {len(results)} matching face tile(s), including visually similar faces "
                    "that may not have a name yet."
                ),
                match_label=name,
            )

        self._start_job("Searching faces by name", _run, _done)

    def _show_face_record_query_results(self, records: list[IndexedFaceRecord], *, title: str, summary: str) -> None:
        items: list[FaceTileItem] = []
        overlay_by_path: dict[str, str] = {}
        context_by_path: dict[str, dict[str, object]] = {}
        photo_paths: list[str] = []
        for record in list(records or []):
            photo_paths.append(str(record.image_path))
            label = str(record.person_name or "").strip() or "Unlabeled"
            overlay_by_path.setdefault(str(record.image_path), label)
            context_by_path.setdefault(
                str(record.image_path),
                {
                    "face_search": {
                        "bbox": tuple(record.face_bbox),
                        "confidence": float(record.face_confidence),
                        "person_name": label,
                    }
                },
            )
            items.append(
                FaceTileItem(
                    image_path=str(record.image_path),
                    face_index=int(record.face_index),
                    bbox=tuple(int(value) for value in record.face_bbox),
                    title=f"{label}\n{Path(record.image_path).name}",
                    tooltip=(
                        f"{record.image_path}\nface #{int(record.face_index) + 1}\n"
                        f"bbox={tuple(int(value) for value in record.face_bbox)}\n"
                        f"conf={float(record.face_confidence):.3f}\nquality={str(record.quality_status or 'clean')}"
                    ),
                    status=str(record.quality_status or "clean"),
                    saved_face_index=int(record.face_index),
                    payload=record,
                )
            )
        self._show_face_result_groups(
            [
                FaceResultGroup(
                    group_id=f"people-query:{uuid4()}",
                    title=title,
                    summary=summary,
                    items=tuple(items),
                )
            ],
            summary=summary,
            photo_paths=photo_paths,
            overlay_by_path=overlay_by_path,
            context_by_path=context_by_path,
            kind="faces",
        )

    def _show_photos_with_people_query(self) -> None:
        names = [value.strip() for value in self.face_people_query_names.text().split(",") if value.strip()]
        if not names:
            errorBox("Missing names", "Enter one or more people names first.")
            return
        scope_paths = self._current_scope_paths()

        def _run(progress, cancel_check):
            progress(-1, "Finding people together...")
            _ = cancel_check
            return self._active_face_service().find_face_records_by_people(
                names,
                require_all=True,
                candidate_paths=scope_paths,
                include_tiny_faces=self._show_tiny_detections_enabled(),
            )

        def _done(records) -> None:
            self.status_label.setText(f"Found {len(records)} matching face record(s) for the requested people.")
            self._show_face_record_query_results(
                list(records or []),
                title="People together",
                summary=f"Found {len(records)} matching face tile(s) for {' + '.join(names)}.",
            )

        self._start_job("Finding people together", _run, _done)

    def _show_photos_with_any_named_people(self) -> None:
        scope_paths = self._current_scope_paths()

        def _run(progress, cancel_check):
            progress(-1, "Finding named people...")
            _ = cancel_check
            return self._active_face_service().find_face_records_with_named_people(
                candidate_paths=scope_paths,
                include_tiny_faces=self._show_tiny_detections_enabled(),
            )

        def _done(records) -> None:
            self.status_label.setText(f"Found {len(records)} named face record(s).")
            self._show_face_record_query_results(
                list(records or []),
                title="Any named people",
                summary=f"Found {len(records)} face tile(s) that already have saved identities.",
            )

        self._start_job("Finding named people", _run, _done)

    def _show_photos_with_unknown_people(self) -> None:
        scope_paths = self._current_scope_paths()

        def _run(progress, cancel_check):
            progress(-1, "Finding unknown people...")
            _ = cancel_check
            return self._active_face_service().find_face_records_with_unknown_people(
                candidate_paths=scope_paths,
                include_tiny_faces=self._show_tiny_detections_enabled(),
            )

        def _done(records) -> None:
            self.status_label.setText(f"Found {len(records)} unknown face record(s).")
            self._show_face_record_query_results(
                list(records or []),
                title="Unknown people",
                summary=f"Found {len(records)} face tile(s) that still need identity assignment.",
            )

        self._start_job("Finding unknown people", _run, _done)

    def _show_photos_with_primary_person(self) -> None:
        name = self.face_primary_person_query.text().strip()
        if not name:
            errorBox("Missing name", "Enter the primary person name first.")
            return
        scope_paths = self._current_scope_paths()

        def _run(progress, cancel_check):
            progress(-1, "Finding primary person photos...")
            _ = cancel_check
            return self._active_face_service().find_face_records_with_primary_person(
                name,
                candidate_paths=scope_paths,
                include_tiny_faces=self._show_tiny_detections_enabled(),
            )

        def _done(records) -> None:
            self.status_label.setText(f"Found {len(records)} primary-person face record(s) for {name}.")
            self._show_face_record_query_results(
                list(records or []),
                title=f"Primary person: {name}",
                summary=f"Found {len(records)} face tile(s) where {name} is the largest face.",
            )

        self._start_job("Finding primary person", _run, _done)

    @staticmethod
    def _face_query_quote(value: str) -> str:
        return str(value or "").replace('"', "").strip()

    def _build_face_query_from_controls(self) -> str:
        tokens: list[str] = []
        face_count = int(self.face_query_builder_face_count.value()) if hasattr(self, "face_query_builder_face_count") else 0
        faces_state = self.face_query_builder_faces_state.currentText() if hasattr(self, "face_query_builder_faces_state") else "Any"
        if face_count > 0:
            tokens.append(f"faces:{face_count}")
        elif faces_state == "Has faces":
            tokens.append("faces:true")
        elif faces_state == "No faces":
            tokens.append("faces:false")
        person_name = self._face_query_quote(self.face_query_builder_person.text()) if hasattr(self, "face_query_builder_person") else ""
        if person_name:
            tokens.append(f'person:"{person_name}"')
        partial_name = self._face_query_quote(self.face_query_builder_partial.text()) if hasattr(self, "face_query_builder_partial") else ""
        if partial_name:
            tokens.append(f'people:"{partial_name}"')
        people_text = self.face_query_builder_people.text() if hasattr(self, "face_query_builder_people") else ""
        people = [self._face_query_quote(value) for value in str(people_text).split(",") if self._face_query_quote(value)]
        if people:
            joiner = "&" if self.face_query_builder_people_mode.currentText() == "All" else "|"
            tokens.append(f'person:"{joiner.join(people)}"')
        if hasattr(self, "face_query_builder_unknown") and self.face_query_builder_unknown.isChecked():
            tokens.append("unknown:true")
        if hasattr(self, "face_query_builder_hidden") and self.face_query_builder_hidden.isChecked():
            tokens.append("hidden:true")
        return " ".join(tokens)

    def _run_face_query_string(
        self,
        query: str,
        *,
        include_hidden: bool = False,
        include_tiny_faces: bool | None = None,
        title: str = "Face query",
    ) -> None:
        query = str(query or "").strip()
        if not query:
            errorBox("Missing face query", "Choose at least one face query condition first.")
            return
        scope_paths = self._current_scope_paths()
        include_tiny = self._show_tiny_detections_enabled() if include_tiny_faces is None else bool(include_tiny_faces)

        def _run(progress, cancel_check):
            progress(-1, "Running face query...")
            _ = cancel_check
            return self._active_face_service().search_face_query(
                query,
                candidate_paths=scope_paths,
                include_tiny_faces=include_tiny,
                include_hidden=bool(include_hidden),
            )

        def _done(records) -> None:
            self.status_label.setText(f"Face query matched {len(records or [])} face record(s).")
            self._show_face_record_query_results(
                list(records or []),
                title=title,
                summary=f"{query}\nMatched {len(records or [])} face tile(s).",
            )

        self._start_job("Running face query", _run, _done)

    def _run_face_query_builder(self) -> None:
        query = self._build_face_query_from_controls()
        include_hidden = bool(hasattr(self, "face_query_builder_hidden") and self.face_query_builder_hidden.isChecked())
        self._run_face_query_string(query, include_hidden=include_hidden)

    def _saved_search_kind(self) -> str:
        combo = getattr(self, "saved_search_kind", None)
        if combo is None:
            return ""
        return str(combo.currentData() or combo.currentText() or "").strip()

    @staticmethod
    def _saved_search_type_label(search_type: str) -> str:
        labels = {
            "clustering_filter": "Clustering tag filter",
            "face_query": "Face query",
            "people_together": "People together",
            "any_named_people": "Any named people",
            "unknown_people": "Unknown people",
            "hidden_faces": "Hidden faces",
            "primary_person": "Primary person",
        }
        return labels.get(str(search_type or "").strip(), str(search_type or "").strip() or "Saved search")

    def _saved_search_payload_from_controls(self) -> dict[str, object]:
        kind = self._saved_search_kind()
        if kind == "clustering_filter":
            payload = {}
            provider = getattr(self, "clustering_filter_state_provider", None)
            if callable(provider):
                payload = dict(provider() or {})
            tags = payload.get("tags", payload.get("tag_filter", []))
            if isinstance(tags, str):
                tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
            else:
                tags = [str(tag).strip() for tag in list(tags or []) if str(tag).strip()]
            if not tags:
                raise ValueError("Enter a clustering tag filter before saving this search.")
            return {"tags": tags, "tag_match": str(payload.get("tag_match") or payload.get("match") or "Any")}
        if kind == "face_query":
            query = self._build_face_query_from_controls()
            if not query:
                raise ValueError("Choose at least one face query condition before saving this search.")
            return {
                "query": query,
                "include_hidden": bool(self.face_query_builder_hidden.isChecked()) if hasattr(self, "face_query_builder_hidden") else False,
                "include_tiny_faces": bool(self._show_tiny_detections_enabled()),
            }
        if kind == "people_together":
            names = [value.strip() for value in self.face_people_query_names.text().split(",") if value.strip()]
            if not names:
                raise ValueError("Enter people names before saving this search.")
            return {"names": names, "require_all": True}
        if kind == "any_named_people":
            return {"mode": "any_named"}
        if kind == "unknown_people":
            return {"mode": "unknown"}
        if kind == "hidden_faces":
            return {"query": "hidden:true", "include_hidden": True, "include_tiny_faces": bool(self._show_tiny_detections_enabled())}
        if kind == "primary_person":
            name = self.face_primary_person_query.text().strip()
            if not name:
                raise ValueError("Enter a primary person name before saving this search.")
            return {"person_name": name}
        raise ValueError("Choose a saved search type first.")

    def _refresh_saved_searches(self, selected_id: str = "") -> None:
        if not hasattr(self, "saved_searches_list"):
            return
        self.saved_searches_list.clear()
        for record in self.saved_search_service.list_searches():
            item = QListWidgetItem(f"{record.name} | {self._saved_search_type_label(record.search_type)}")
            item.setData(Qt.ItemDataRole.UserRole, record.search_id)
            item.setToolTip(json.dumps(record.payload, indent=2, sort_keys=True))
            self.saved_searches_list.addItem(item)
            if selected_id and record.search_id == selected_id:
                self.saved_searches_list.setCurrentItem(item)

    def _selected_saved_search(self) -> SavedSearch | None:
        if not hasattr(self, "saved_searches_list"):
            return None
        item = self.saved_searches_list.currentItem()
        if item is None:
            return None
        search_id = str(item.data(Qt.ItemDataRole.UserRole) or "").strip()
        if not search_id:
            return None
        return self.saved_search_service.get_search(search_id)

    def _save_current_saved_search(self) -> None:
        name = self.saved_search_name.text().strip() if hasattr(self, "saved_search_name") else ""
        try:
            payload = self._saved_search_payload_from_controls()
            record = self.saved_search_service.save_search(name, self._saved_search_kind(), payload)
            self._refresh_saved_searches(record.search_id)
            self.status_label.setText(f"Saved search '{record.name}'.")
        except Exception as exc:
            errorBox("Save search failed", str(exc))

    def _rename_selected_saved_search(self) -> None:
        record = self._selected_saved_search()
        if record is None:
            errorBox("No saved search selected", "Select a saved search first.")
            return
        new_name = self.saved_search_name.text().strip() if hasattr(self, "saved_search_name") else ""
        try:
            updated = self.saved_search_service.rename_search(record.search_id, new_name)
            self._refresh_saved_searches(updated.search_id)
            self.status_label.setText(f"Renamed saved search to '{updated.name}'.")
        except Exception as exc:
            errorBox("Rename search failed", str(exc))

    def _delete_selected_saved_search(self) -> None:
        record = self._selected_saved_search()
        if record is None:
            errorBox("No saved search selected", "Select a saved search first.")
            return
        deleted = self.saved_search_service.delete_search(record.search_id)
        self._refresh_saved_searches()
        if deleted:
            self.status_label.setText(f"Deleted saved search '{record.name}'.")

    def _run_selected_saved_search(self) -> None:
        record = self._selected_saved_search()
        if record is None:
            errorBox("No saved search selected", "Select a saved search first.")
            return
        self._run_saved_search(record)

    def _run_saved_search(self, record: SavedSearch) -> None:
        payload = dict(record.payload or {})
        search_type = str(record.search_type or "").strip()
        if search_type == "clustering_filter":
            self.saved_clustering_filter_requested.emit(payload)
            self.status_label.setText(f"Ran saved clustering filter '{record.name}'.")
            return
        if search_type == "face_query":
            self._run_face_query_string(
                str(payload.get("query") or ""),
                include_hidden=bool(payload.get("include_hidden", False)),
                include_tiny_faces=bool(payload.get("include_tiny_faces", self._show_tiny_detections_enabled())),
                title=record.name,
            )
            return
        if search_type == "people_together":
            names = [str(name).strip() for name in list(payload.get("names", []) or []) if str(name).strip()]
            self.face_people_query_names.setText(", ".join(names))
            self._show_photos_with_people_query()
            return
        if search_type == "any_named_people":
            self._show_photos_with_any_named_people()
            return
        if search_type == "unknown_people":
            self._show_photos_with_unknown_people()
            return
        if search_type == "hidden_faces":
            self._run_face_query_string(
                str(payload.get("query") or "hidden:true"),
                include_hidden=True,
                include_tiny_faces=bool(payload.get("include_tiny_faces", self._show_tiny_detections_enabled())),
                title=record.name,
            )
            return
        if search_type == "primary_person":
            self.face_primary_person_query.setText(str(payload.get("person_name") or ""))
            self._show_photos_with_primary_person()
            return
        errorBox("Unsupported saved search", f"Cannot run saved search type: {search_type}")

    def _set_face_walkthrough_visible(self, visible: bool) -> None:
        if hasattr(self, "face_walkthrough_panel"):
            self.face_walkthrough_panel.setVisible(bool(visible))
        if hasattr(self, "face_walkthrough_reopen_button"):
            self.face_walkthrough_reopen_button.setVisible(not bool(visible))

    def _refresh_face_db_usage_label(self) -> None:
        if not hasattr(self, "face_db_usage_label"):
            return
        service = self._active_face_service()
        db_path = Path(str(getattr(service, "db_path", "") or ""))
        total_bytes = 0
        existing_paths: list[Path] = []
        for candidate in [db_path, Path(f"{db_path}.faiss"), Path(f"{db_path}.faiss.json")]:
            if candidate.exists():
                existing_paths.append(candidate)
                try:
                    total_bytes += int(candidate.stat().st_size)
                except Exception:
                    pass
        if not existing_paths:
            self.face_db_usage_label.setText("Face data usage: no saved data has been created for this library yet.")
            return
        self.face_db_usage_label.setText(
            f"Face data usage: {self._human_size(total_bytes)} across {len(existing_paths)} file(s).\n{db_path}"
        )

    @staticmethod
    def _human_size(size_bytes: int) -> str:
        value = float(max(0, int(size_bytes)))
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if value < 1024.0 or unit == "TB":
                return f"{value:.1f} {unit}"
            value /= 1024.0
        return f"{value:.1f} TB"

    def _rebuild_current_face_index(self) -> None:
        if hasattr(self, "face_index_button"):
            self._index_faces()
        self._refresh_face_db_usage_label()

    def _delete_active_face_db(self) -> None:
        if not self._ensure_writable_face_action("delete the active face library"):
            return
        service = self._active_face_service()
        db_path = Path(str(getattr(service, "db_path", "") or ""))
        targets = [db_path, Path(f"{db_path}.faiss"), Path(f"{db_path}.faiss.json")]
        existing = [path for path in targets if path.exists()]
        if not existing:
            self._refresh_face_db_usage_label()
            return
        if not confirmBox(
            "Delete active face library?",
            (
                f"Target: {db_path}\n"
                f"Affected files: {len(existing)}\n"
                "Result: indexed faces, saved labels, and identity data in this library are removed.\n"
                "Recovery: this deletion is irreversible unless you have a separate backup."
            ),
            parent=self,
        ):
            self.status_label.setText("Face-library deletion cancelled. No files were changed.")
            return
        self._record_face_ui_action_audit(
            "delete_active_face_db",
            target=str(db_path),
            details={"deleted_files": [str(path) for path in existing]},
            reversible=False,
        )
        for path in existing:
            try:
                path.unlink()
            except FileNotFoundError:
                continue
        self._refresh_face_db_usage_label()
        self.status_label.setText("Deleted the selected face-library data.")

    def _refresh_face_label_audit(self, assignments: list[FaceLabelAssignment]) -> None:
        if not hasattr(self, "face_label_audit_label"):
            return
        if not assignments:
            self.face_label_audit_label.setText("Label audit: select one or more proposals to inspect the source and confidence.")
            return
        first = assignments[0]
        self.face_label_audit_label.setText(
            f"Label audit: {len(assignments)} proposal(s) | target={first.person_name} | "
            f"source={first.source or 'pending'} | confidence={float(first.confidence):.4f} | "
            f"face={Path(first.image_path).name} #{int(first.face_index) + 1}"
        )

    def _face_ui_action_audit_path(self) -> Path:
        return self.settings.log_dir / "face_action_audit.jsonl"

    def _record_face_ui_action_audit(
        self,
        action: str,
        *,
        target: str = "",
        details: dict[str, object] | None = None,
        reversible: bool = False,
    ) -> None:
        payload = {
            "action": str(action),
            "target": str(target or ""),
            "details": dict(details or {}),
            "reversible": bool(reversible),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        try:
            path = self._face_ui_action_audit_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")
        except Exception:
            return

    def _load_face_ui_action_audit(self, *, limit: int = 30) -> list[dict[str, object]]:
        path = self._face_ui_action_audit_path()
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()[-max(1, int(limit)) :]
        except Exception:
            return []
        events: list[dict[str, object]] = []
        for line in reversed(lines):
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                events.append(payload)
        return events

    @staticmethod
    def _audit_event_value(event, key: str, default=""):
        if isinstance(event, dict):
            return event.get(key, default)
        return getattr(event, key, default)

    def _refresh_face_action_audit(self) -> None:
        if not hasattr(self, "face_action_audit_list"):
            return
        self.face_action_audit_list.clear()
        events: list[object] = []
        load_fn = getattr(self._active_face_service(), "load_face_action_audit", None)
        if callable(load_fn):
            try:
                events.extend(list(load_fn(limit=30) or []))
            except Exception as exc:
                self.face_action_audit_list.addItem(f"Action audit failed: {exc}")
        events.extend(self._load_face_ui_action_audit(limit=30))
        if not events:
            self.face_action_audit_list.addItem("No recent face action audit events.")
            return
        for event in events:
            action = str(self._audit_event_value(event, "action", "") or "")
            target = str(self._audit_event_value(event, "target", "") or "")
            created_at = str(self._audit_event_value(event, "created_at", "") or "")
            undo_hint = ""
            if bool(self._audit_event_value(event, "reversible", False)) and action == "accept_pending_face_labels":
                undo_hint = " | Undo: use Review Pending Labels > Undo Last Accept"
            self.face_action_audit_list.addItem(f"{created_at} | {action} | {target}{undo_hint}")

    def _build_face_tab(self) -> None:
        tab_body = QWidget()
        layout = QGridLayout(tab_body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)
        self.face_search_grid = layout
        self.face_query_path = PasteAwareLineEdit(lambda mime: self._handle_paste_mime(self.face_query_path, mime, multi_append=False), self)
        self.face_examples = QLineEdit()
        self.face_examples.setPlaceholderText("Optional extra example photos")
        self.person_name = QLineEdit()
        self.person_name.setPlaceholderText("Saved identity name")
        self.face_top_k = QSpinBox()
        self.face_top_k.setRange(1, 500)
        self.face_top_k.setValue(30)
        self.face_min_score = QDoubleSpinBox()
        self.face_min_score.setRange(-1.0, 1.0)
        self.face_min_score.setSingleStep(0.05)
        self.face_min_score.setValue(0.35)
        self.face_recognition_mode_combo = QComboBox()
        self.face_recognition_mode_combo.addItem("Balanced", "balanced")
        self.face_recognition_mode_combo.addItem("Strict", "strict")
        self.face_recognition_mode_combo.addItem("Loose", "loose")
        self.face_label_threshold = QDoubleSpinBox()
        self.face_label_threshold.setRange(0.0, 1.0)
        self.face_label_threshold.setSingleStep(0.01)
        self.face_label_threshold.setValue(0.72)
        self.face_cluster_count = QSpinBox()
        self.face_cluster_count.setRange(2, 200)
        self.face_cluster_count.setValue(12)
        self.face_cluster_backend = QComboBox()
        for backend_id, label in face_cluster_backend_choices():
            self.face_cluster_backend.addItem(label, backend_id)
        self.face_cluster_backend.setCurrentIndex(max(0, self.face_cluster_backend.findData("hdbscan")))
        self.face_cluster_backend.currentIndexChanged.connect(lambda _index: self._refresh_face_cluster_backend_controls())
        self.face_cluster_backend_override_checkbox = QCheckBox("Advanced backend override")
        self.face_cluster_backend_override_checkbox.setToolTip(
            "Standard Faces clustering always uses HDBSCAN. Enable this to run a different backend for manual review."
        )
        self.face_cluster_backend_override_checkbox.toggled.connect(lambda _checked: self._refresh_face_cluster_backend_controls())
        self.face_hdbscan_min_cluster_size_spin = QSpinBox()
        self.face_hdbscan_min_cluster_size_spin.setRange(2, 200)
        self.face_hdbscan_min_cluster_size_spin.setValue(4)
        self.face_hdbscan_min_samples_spin = QSpinBox()
        self.face_hdbscan_min_samples_spin.setRange(0, 200)
        self.face_hdbscan_min_samples_spin.setSpecialValueText("Auto")
        self.face_hdbscan_min_samples_spin.setValue(0)
        self.face_hdbscan_cluster_selection_epsilon_spin = QDoubleSpinBox()
        self.face_hdbscan_cluster_selection_epsilon_spin.setRange(0.0, 1.0)
        self.face_hdbscan_cluster_selection_epsilon_spin.setSingleStep(0.01)
        self.face_hdbscan_cluster_selection_epsilon_spin.setDecimals(2)
        self.face_hdbscan_cluster_selection_epsilon_spin.setValue(0.0)
        self.face_hdbscan_allow_single_cluster_checkbox = QCheckBox("Allow single cluster")
        self.face_hdbscan_options_widget = QWidget()
        face_hdbscan_layout = QVBoxLayout(self.face_hdbscan_options_widget)
        face_hdbscan_layout.setContentsMargins(0, 0, 0, 0)
        face_hdbscan_layout.setSpacing(8)
        face_hdbscan_layout.addWidget(self._field_widget("HDBSCAN min cluster size", self.face_hdbscan_min_cluster_size_spin, tooltip=FACE_HELP["face_cluster_backend"]))
        face_hdbscan_layout.addWidget(self._field_widget("HDBSCAN min samples", self.face_hdbscan_min_samples_spin, tooltip=FACE_HELP["face_cluster_backend"]))
        face_hdbscan_layout.addWidget(
            self._field_widget(
                "HDBSCAN merge epsilon",
                self.face_hdbscan_cluster_selection_epsilon_spin,
                tooltip=FACE_HELP["face_cluster_backend"],
            )
        )
        face_hdbscan_layout.addWidget(build_help_inline(self.face_hdbscan_allow_single_cluster_checkbox, FACE_HELP["face_cluster_backend"], help_key="face_cluster_backend"))
        self.face_cluster_outlier_policy_combo = QComboBox()
        for item_id, label in face_cluster_outlier_policy_choices():
            self.face_cluster_outlier_policy_combo.addItem(label, item_id)
        self.face_cluster_outlier_policy_combo.setToolTip(FACE_HELP["face_cluster_backend"])
        self.face_cluster_outlier_policy_combo.setCurrentIndex(self.face_cluster_outlier_policy_combo.findData("isolate"))
        self.merge_source_person = QLineEdit()
        self.merge_source_person.setPlaceholderText("Existing saved name")
        self.merge_target_person = QLineEdit()
        self.merge_target_person.setPlaceholderText("Saved name to keep")
        self.face_people_query_names = QLineEdit()
        self.face_people_query_names.setPlaceholderText("Alice, Bob")
        self.face_primary_person_query = QLineEdit()
        self.face_primary_person_query.setPlaceholderText("Primary person name")
        self.face_query_builder_faces_state = QComboBox()
        self.face_query_builder_faces_state.addItems(["Any", "Has faces", "No faces"])
        self.face_query_builder_face_count = QSpinBox()
        self.face_query_builder_face_count.setRange(0, 99)
        self.face_query_builder_face_count.setSpecialValueText("Any")
        self.face_query_builder_person = QLineEdit()
        self.face_query_builder_person.setPlaceholderText("Exact person")
        self.face_query_builder_partial = QLineEdit()
        self.face_query_builder_partial.setPlaceholderText("Partial person text")
        self.face_query_builder_people = QLineEdit()
        self.face_query_builder_people.setPlaceholderText("Alice, Bob")
        self.face_query_builder_people_mode = QComboBox()
        self.face_query_builder_people_mode.addItems(["All", "Any"])
        self.face_query_builder_unknown = QCheckBox("Unknown")
        self.face_query_builder_hidden = QCheckBox("Hidden")
        self.saved_search_name = QLineEdit()
        self.saved_search_name.setPlaceholderText("Saved search name")
        self.saved_search_kind = QComboBox()
        for label, value in (
            ("Clustering tag filter", "clustering_filter"),
            ("Face query", "face_query"),
            ("People together", "people_together"),
            ("Any named people", "any_named_people"),
            ("Unknown people", "unknown_people"),
            ("Hidden faces", "hidden_faces"),
            ("Primary person", "primary_person"),
        ):
            self.saved_search_kind.addItem(label, value)
        self.saved_searches_list = QListWidget()
        self.saved_searches_list.setMaximumHeight(150)
        self.saved_searches_list.setToolTip("Saved searches persist across app restarts.")
        self.saved_search_save_button = QPushButton("Save Search")
        self.saved_search_run_button = QPushButton("Run Saved")
        self.saved_search_rename_button = QPushButton("Rename")
        self.saved_search_delete_button = QPushButton("Delete")
        self.face_query_path.setToolTip(FACE_HELP["query_face_image"])
        self.face_examples.setToolTip(FACE_HELP["few_shot_examples"])
        self.person_name.setToolTip(FACE_HELP["saved_person_name"])
        self.face_top_k.setToolTip(FACE_HELP["name_search_top_k"])
        self.face_min_score.setToolTip(FACE_HELP["selected_face_min_score"])
        self.face_recognition_mode_combo.setToolTip(FACE_HELP["face_recognition_min_score"])
        self.face_label_threshold.setToolTip(FACE_HELP["label_threshold"])
        self.face_cluster_count.setToolTip(FACE_HELP["face_cluster_count"])
        self.face_cluster_backend.setToolTip(FACE_HELP["face_cluster_backend"])
        self.face_cluster_backend_override_checkbox.setToolTip(
            "Standard Faces clustering uses HDBSCAN by default. Enable this only when you want to override the backend."
        )
        self.face_hdbscan_options_widget.setToolTip(FACE_HELP["face_cluster_backend"])
        self.merge_source_person.setToolTip(FACE_HELP["merge_source"])
        self.merge_target_person.setToolTip(FACE_HELP["merge_target"])
        self.face_people_query_names.setToolTip("Comma-separated people names for co-occurrence queries.")
        self.face_primary_person_query.setToolTip("Find photos where this identity is the largest face.")
        self.face_query_detect_button = QPushButton("Detect Query Faces")
        self.face_query_detect_button.setToolTip(FACE_HELP["query_face_image"])
        self.face_query_detect_button.clicked.connect(self._detect_query_photo_faces)
        self.face_query_faces_model = ListEntryModel(self)
        self.face_query_faces_list = QListView()
        self.face_query_faces_list.setViewMode(QListView.ViewMode.IconMode)
        self.face_query_faces_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.face_query_faces_list.setMovement(QListView.Movement.Static)
        self.face_query_faces_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.face_query_faces_list.setIconSize(QSize(72, 72))
        self.face_query_faces_list.setGridSize(QSize(102, 120))
        self.face_query_faces_list.setSpacing(8)
        self.face_query_faces_list.setUniformItemSizes(True)
        self.face_query_faces_list.setMaximumHeight(210)
        self.face_query_faces_list.setModel(self.face_query_faces_model)
        self.face_query_faces_summary = QLabel("Detect faces in the query photo, then pick one or more faces to search.")
        self.face_query_faces_summary.setWordWrap(True)
        self.face_query_path.textChanged.connect(lambda _text: self._populate_query_face_candidates(self.face_query_path.text().strip(), []))
        self.face_people_together_button = QPushButton("Show Photos With These People")
        self.face_people_any_named_button = QPushButton("Show Photos With Any Named People")
        self.face_people_unknown_button = QPushButton("Show Photos With Unknown People")
        self.face_people_primary_button = QPushButton("Show Photos Where This Person Is Primary")
        self.face_query_builder_run_button = QPushButton("Run Face Query")
        self.face_people_together_button.clicked.connect(self._show_photos_with_people_query)
        self.face_people_any_named_button.clicked.connect(self._show_photos_with_any_named_people)
        self.face_people_unknown_button.clicked.connect(self._show_photos_with_unknown_people)
        self.face_people_primary_button.clicked.connect(self._show_photos_with_primary_person)
        self.face_query_builder_run_button.clicked.connect(self._run_face_query_builder)
        self.saved_search_save_button.clicked.connect(self._save_current_saved_search)
        self.saved_search_run_button.clicked.connect(self._run_selected_saved_search)
        self.saved_search_rename_button.clicked.connect(self._rename_selected_saved_search)
        self.saved_search_delete_button.clicked.connect(self._delete_selected_saved_search)
        self._action_buttons.extend(
            [
                self.face_people_together_button,
                self.face_people_any_named_button,
                self.face_people_unknown_button,
                self.face_people_primary_button,
                self.face_query_builder_run_button,
                self.saved_search_save_button,
                self.saved_search_run_button,
                self.saved_search_rename_button,
                self.saved_search_delete_button,
            ]
        )

        quick_group, quick_layout = self._group_box("Workflow", tooltip=FACE_HELP["search_quick_start"])
        self.face_search_quick_start_label = self._helper_label(
            "Choose a query photo or use a selected face, then start the search.",
            tooltip=FACE_HELP["search_quick_start"],
        )
        quick_layout.addWidget(self.face_search_quick_start_label)
        self.face_walkthrough_panel = QWidget(quick_group)
        face_walkthrough_layout = QVBoxLayout(self.face_walkthrough_panel)
        face_walkthrough_layout.setContentsMargins(0, 0, 0, 0)
        face_walkthrough_layout.setSpacing(6)
        self.face_walkthrough_label = self._helper_label(
            "Scan or refresh a folder, select a detected face, then search or name it.",
            tooltip=FACE_HELP["search_quick_start"],
        )
        self.face_walkthrough_dismiss_button = QPushButton("Dismiss Walkthrough")
        self.face_walkthrough_reopen_button = QPushButton("Show Faces Walkthrough")
        self.face_walkthrough_dismiss_button.clicked.connect(lambda: self._set_face_walkthrough_visible(False))
        self.face_walkthrough_reopen_button.clicked.connect(lambda: self._set_face_walkthrough_visible(True))
        face_walkthrough_layout.addWidget(self.face_walkthrough_label)
        face_walkthrough_layout.addWidget(self.face_walkthrough_dismiss_button)
        quick_layout.addWidget(self.face_walkthrough_panel)
        quick_layout.addWidget(self.face_walkthrough_reopen_button)
        saved_group = QWidget(quick_group)
        saved_layout = QVBoxLayout(saved_group)
        saved_layout.setContentsMargins(0, 0, 0, 0)
        saved_layout.setSpacing(8)
        saved_layout.addWidget(
            self._helper_label(
                "Save the current clustering tag filter, face query, people query, or review queue.",
                tooltip="Saved searches persist across app restarts.",
            )
        )
        saved_layout.addWidget(self._field_widget("Name", self.saved_search_name, tooltip="Saved searches persist across app restarts."))
        saved_layout.addWidget(self._field_widget("Type", self.saved_search_kind, tooltip="Choose which current search inputs to save."))
        saved_button_row = QWidget(saved_group)
        saved_button_layout = FlowLayout(saved_button_row)
        saved_button_layout.setContentsMargins(0, 0, 0, 0)
        saved_button_layout.setSpacing(6)
        saved_button_layout.addWidget(self.saved_search_save_button)
        saved_button_layout.addWidget(self.saved_search_run_button)
        saved_button_layout.addWidget(self.saved_search_rename_button)
        saved_button_layout.addWidget(self.saved_search_delete_button)
        saved_layout.addWidget(saved_button_row)
        saved_layout.addWidget(self.saved_searches_list)
        quick_layout.addWidget(saved_group)
        self.saved_search_group = saved_group
        self._refresh_saved_searches()
        self._set_face_walkthrough_visible(True)
        self.face_search_quick_group = quick_group

        selected_group, selected_layout = self._group_box("1. Find From Selected Face", tooltip=FACE_HELP["find_photos_selected_face"])
        self.face_search_selected_context_label = self._helper_label(
            "Select one face in Face Library or Detected Faces first.",
            tooltip=FACE_HELP["find_photos_selected_face"],
        )
        selected_layout.addWidget(self.face_search_selected_context_label)
        selected_layout.addWidget(
            self._helper_label(
                "Use this when you already picked the exact face tile you want to search or name.",
                tooltip=FACE_HELP["find_photos_selected_face"],
            )
        )
        selected_buttons = QVBoxLayout()
        selected_buttons.setContentsMargins(0, 0, 0, 0)
        selected_buttons.setSpacing(6)
        self.face_search_selected_card_button = QPushButton("Find Similar From Selection")
        self.face_search_selected_card_button.setProperty("kind", "primary")
        self.face_search_name_selected_card_button = QPushButton("Name Selected Faces")
        self.face_search_name_selected_card_button.setProperty("kind", "secondary")
        self.face_search_selected_card_button.clicked.connect(self._search_selected_face)
        self.face_search_name_selected_card_button.clicked.connect(self._save_selected_face_name)
        self._action_buttons.extend([self.face_search_selected_card_button, self.face_search_name_selected_card_button])
        self._mode_required_buttons.extend([self.face_search_selected_card_button, self.face_search_name_selected_card_button])
        selected_buttons.addWidget(self.face_search_selected_card_button)
        selected_buttons.addWidget(self.face_search_name_selected_card_button)
        selected_layout.addLayout(selected_buttons)
        self.face_search_selected_options_toggle = self._expander_button("Options", checked=False, tooltip=FACE_HELP["find_photos_selected_face"])
        selected_layout.addWidget(self.face_search_selected_options_toggle)
        self.face_search_selected_options_panel = QWidget(selected_group)
        selected_options_layout = QVBoxLayout(self.face_search_selected_options_panel)
        selected_options_layout.setContentsMargins(0, 0, 0, 0)
        selected_options_layout.setSpacing(8)
        selected_options_layout.addWidget(self._field_widget("Name for selected face(s)", self.face_label_name, tooltip=FACE_HELP["selected_face_name"]))
        selected_options_layout.addWidget(self._field_widget("Selected-face Top-K", self.face_browser_top_k, tooltip=FACE_HELP["selected_face_top_k"]))
        selected_options_layout.addWidget(self._field_widget("Selected-face Min score", self.face_browser_min_score, tooltip=FACE_HELP["selected_face_min_score"]))
        selected_options_layout.addWidget(self._field_widget("Label threshold", self.face_library_label_threshold, tooltip=FACE_HELP["label_threshold"]))
        selected_options_layout.addWidget(self._field_widget("Result folder filter", self.face_name_folder_filter, tooltip=FACE_HELP["result_folder_filter"]))
        self.face_search_selected_options_toggle.toggled.connect(
            lambda checked: self._set_expander_state(
                self.face_search_selected_options_toggle,
                self.face_search_selected_options_panel,
                checked,
            )
        )
        selected_layout.addWidget(self.face_search_selected_options_panel)
        self._set_expander_state(self.face_search_selected_options_toggle, self.face_search_selected_options_panel, False)
        self.face_search_selected_group = selected_group

        find_group, find_layout = self._group_box("Find by Face", tooltip=FACE_HELP["find_same_person"])
        find_layout.addWidget(
            self._helper_label(
                "Choose one clear face photo and search the indexed folder for similar people.",
                tooltip=FACE_HELP["query_face_image"],
            )
        )
        find_fields = QVBoxLayout()
        find_fields.setContentsMargins(0, 0, 0, 0)
        find_fields.setSpacing(8)
        self.face_find_fields_grid = find_fields
        find_fields.addWidget(self._field_widget("Query face image", self._path_row(self.face_query_path), tooltip=FACE_HELP["query_face_image"]))
        find_fields.addWidget(self.face_query_detect_button)
        find_fields.addWidget(self.face_query_faces_summary)
        find_fields.addWidget(self.face_query_faces_list)
        find_layout.addLayout(find_fields)
        find_buttons = QVBoxLayout()
        find_buttons.setContentsMargins(0, 0, 0, 0)
        find_buttons.setSpacing(6)
        self.face_index_button = QPushButton("Index Current Folder")
        self.face_index_button.setToolTip(FACE_HELP["index_current_folder"])
        self.face_search_button = QPushButton("Find by Face")
        self.face_search_button.setProperty("kind", "primary")
        self.face_search_button.setToolTip(FACE_HELP["find_same_person"])
        self.face_index_button.clicked.connect(self._index_faces)
        self.face_search_button.clicked.connect(self._search_faces)
        self._mode_required_buttons.extend([self.face_index_button, self.face_search_button])
        find_buttons.addWidget(self.face_search_button)
        find_layout.addLayout(find_buttons)
        self.face_find_options_toggle = self._expander_button("Options", checked=False, tooltip=FACE_HELP["find_same_person"])
        find_layout.addWidget(self.face_find_options_toggle)
        self.face_find_options_panel = QWidget(find_group)
        find_options_layout = QVBoxLayout(self.face_find_options_panel)
        find_options_layout.setContentsMargins(0, 0, 0, 0)
        find_options_layout.setSpacing(8)
        find_options_layout.addWidget(self._field_widget("Recognition mode", self.face_recognition_mode_combo, tooltip=FACE_HELP["face_recognition_min_score"]))
        find_options_layout.addWidget(self._field_widget("Top-K", self.face_top_k, tooltip=FACE_HELP["name_search_top_k"]))
        find_options_layout.addWidget(self._field_widget("Min face score", self.face_min_score, tooltip=FACE_HELP["selected_face_min_score"]))
        find_options_layout.addWidget(self.face_index_button)
        self.face_find_options_toggle.toggled.connect(
            lambda checked: self._set_expander_state(self.face_find_options_toggle, self.face_find_options_panel, checked)
        )
        find_layout.addWidget(self.face_find_options_panel)
        self._set_expander_state(self.face_find_options_toggle, self.face_find_options_panel, False)
        self.face_find_group = find_group

        find_name_group, find_name_layout = self._group_box("Find by Name", tooltip=FACE_HELP["find_photos_saved_name"])
        find_name_layout.addWidget(
            self._helper_label(
                "Enter a saved name to show its stored faces and visually similar faces that are still unlabeled.",
                tooltip=FACE_HELP["find_photos_saved_name"],
            )
        )
        self.face_find_name_query = QLineEdit(find_name_group)
        self.face_find_name_query.setPlaceholderText("Saved person name")
        self.face_find_name_query.setToolTip(FACE_HELP["find_photos_saved_name"])
        self.face_find_name_button = QPushButton("Find by Name + Similar")
        self.face_find_name_button.setProperty("kind", "primary")
        self.face_find_name_button.setToolTip(
            "Search from the saved identity embedding. Results include matching faces even when they have not been named yet."
        )
        self.face_find_name_button.clicked.connect(self._search_by_name)
        self.face_find_name_query.returnPressed.connect(self._search_by_name)
        find_name_layout.addWidget(
            self._field_widget("Name", self.face_find_name_query, tooltip=FACE_HELP["find_photos_saved_name"])
        )
        find_name_layout.addWidget(self.face_find_name_button)
        self._action_buttons.append(self.face_find_name_button)
        self._mode_required_buttons.append(self.face_find_name_button)
        self.face_find_name_group = find_name_group

        save_group, save_layout = self._group_box("4. Name Or Update A Saved Identity", tooltip=FACE_HELP["save_named_examples"])
        save_layout.addWidget(
            self._helper_label(
                "Create or update a reusable identity name, then search or auto-apply it to the folder review.",
                tooltip=FACE_HELP["save_named_examples"],
            )
        )
        save_fields = QVBoxLayout()
        save_fields.setContentsMargins(0, 0, 0, 0)
        save_fields.setSpacing(8)
        self.face_save_fields_grid = save_fields
        save_fields.addWidget(self._field_widget("Saved identity name", self.person_name, tooltip=FACE_HELP["saved_person_name"]))
        save_layout.addLayout(save_fields)
        save_buttons = QVBoxLayout()
        save_buttons.setContentsMargins(0, 0, 0, 0)
        save_buttons.setSpacing(6)
        self.face_search_by_name_card_button = QPushButton("Find Photos by Saved Name")
        self.face_label_button = QPushButton("Save Named Examples")
        self.face_propagate_button = QPushButton("Auto-Apply Saved Names")
        self.face_search_by_name_card_button.clicked.connect(self._search_by_name)
        self.face_label_button.setToolTip(FACE_HELP["save_named_examples"])
        self.face_label_button.clicked.connect(self._label_person)
        self.face_propagate_button.setToolTip(FACE_HELP["auto_apply_saved_names"])
        self.face_propagate_button.clicked.connect(self._auto_propagate_labels)
        self._mode_required_buttons.append(self.face_label_button)
        self._mode_required_buttons.append(self.face_search_by_name_card_button)
        self._mode_required_buttons.append(self.face_propagate_button)
        self._action_buttons.append(self.face_search_by_name_card_button)
        self._action_buttons.append(self.face_propagate_button)
        save_buttons.addWidget(self.face_search_by_name_card_button)
        save_buttons.addWidget(self.face_label_button)
        save_buttons.addWidget(self.face_propagate_button)
        save_layout.addLayout(save_buttons)
        self.face_save_options_toggle = self._expander_button("Options", checked=False, tooltip=FACE_HELP["save_named_examples"])
        save_layout.addWidget(self.face_save_options_toggle)
        self.face_save_options_panel = QWidget(save_group)
        save_options_layout = QVBoxLayout(self.face_save_options_panel)
        save_options_layout.setContentsMargins(0, 0, 0, 0)
        save_options_layout.setSpacing(8)
        save_options_layout.addWidget(self._field_widget("Example photos", self._path_row(self.face_examples, multi_append=True, enable_paste=False), tooltip=FACE_HELP["few_shot_examples"]))
        save_options_layout.addWidget(self._field_widget("Label threshold", self.face_label_threshold, tooltip=FACE_HELP["label_threshold"]))
        save_options_layout.addWidget(self._field_widget("Saved-name Top-K", self.face_name_top_k, tooltip=FACE_HELP["name_search_top_k"]))
        save_options_layout.addWidget(self._field_widget("Saved-name Min score", self.face_name_min_score, tooltip=FACE_HELP["name_search_min_score"]))
        self.face_save_options_toggle.toggled.connect(
            lambda checked: self._set_expander_state(self.face_save_options_toggle, self.face_save_options_panel, checked)
        )
        save_layout.addWidget(self.face_save_options_panel)
        self._set_expander_state(self.face_save_options_toggle, self.face_save_options_panel, False)
        self.face_save_group = save_group

        manage_group, manage_layout = self._group_box("3. Cluster Unknown Faces", tooltip=FACE_HELP["cluster_faces"])
        manage_layout.addWidget(
            self._helper_label(
                "Group similar indexed faces in the current scope. Use Detected Faces for visual preselection first.",
                tooltip=FACE_HELP["cluster_faces"],
            )
        )
        manage_fields = QVBoxLayout()
        manage_fields.setContentsMargins(0, 0, 0, 0)
        manage_fields.setSpacing(8)
        self.face_manage_fields_grid = manage_fields
        manage_buttons = QVBoxLayout()
        manage_buttons.setContentsMargins(0, 0, 0, 0)
        manage_buttons.setSpacing(6)
        self.face_cluster_button = QPushButton("Cluster Faces")
        self.face_cluster_button.setToolTip(FACE_HELP["cluster_faces"])
        self.face_list_button = QPushButton("Show Named Faces")
        self.face_list_button.setToolTip(FACE_HELP["show_named_faces"])
        self.face_merge_button = QPushButton("Merge Two Names")
        self.face_merge_button.setToolTip(FACE_HELP["merge_names"])
        self.face_clear_button = QPushButton("Remove Name From Faces")
        self.face_clear_button.setToolTip(FACE_HELP["remove_name"])
        self.face_cluster_button.clicked.connect(self._cluster_faces)
        self.face_list_button.clicked.connect(self._list_face_labels)
        self.face_merge_button.clicked.connect(self._merge_people)
        self.face_clear_button.clicked.connect(self._clear_person)
        manage_buttons.addWidget(self.face_cluster_button)
        manage_layout.addLayout(manage_buttons)
        self.face_manage_options_toggle = self._expander_button("Options", checked=False, tooltip=FACE_HELP["cluster_faces"])
        manage_layout.addWidget(self.face_manage_options_toggle)
        self.face_manage_options_panel = QWidget(manage_group)
        manage_options_layout = QVBoxLayout(self.face_manage_options_panel)
        manage_options_layout.setContentsMargins(0, 0, 0, 0)
        manage_options_layout.setSpacing(8)
        manage_options_layout.addWidget(self._field_widget("Face clusters", self.face_cluster_count, tooltip=FACE_HELP["face_cluster_count"]))
        manage_options_layout.addWidget(build_help_inline(self.face_cluster_backend_override_checkbox, FACE_HELP["face_cluster_backend"], help_key="face_cluster_backend"))
        self.face_cluster_backend_field = self._field_widget("Cluster backend", self.face_cluster_backend, tooltip=FACE_HELP["face_cluster_backend"])
        manage_options_layout.addWidget(self.face_cluster_backend_field)
        manage_options_layout.addWidget(self._field_widget("Outlier policy", self.face_cluster_outlier_policy_combo, tooltip=FACE_HELP["face_cluster_backend"]))
        manage_options_layout.addWidget(self.face_hdbscan_options_widget)
        self.face_manage_options_toggle.toggled.connect(
            lambda checked: self._set_expander_state(self.face_manage_options_toggle, self.face_manage_options_panel, checked)
        )
        manage_layout.addWidget(self.face_manage_options_panel)
        self._set_expander_state(self.face_manage_options_toggle, self.face_manage_options_panel, False)
        self.face_manage_group = manage_group
        self._refresh_face_cluster_backend_controls()

        review_group, review_layout = self._group_box("5. Review Pending Labels", tooltip=FACE_HELP["pending_face_review"])
        review_layout.addWidget(self._helper_label("Review queued naming proposals before they become permanent labels.", tooltip=FACE_HELP["pending_face_review"]))
        self.face_pending_summary_label = self._helper_label("Pending review: 0 proposal(s).", tooltip=FACE_HELP["pending_face_review"])
        review_layout.addWidget(self.face_pending_summary_label)
        review_buttons = QVBoxLayout()
        review_buttons.setContentsMargins(0, 0, 0, 0)
        review_buttons.setSpacing(6)
        self.face_pending_refresh_button = QPushButton("Refresh Pending")
        self.face_pending_accept_button = QPushButton("Accept Selected")
        self.face_pending_accept_above_threshold_button = QPushButton("Accept Above Threshold")
        self.face_pending_accept_cluster_button = QPushButton("Accept Current Cluster")
        self.face_pending_reject_button = QPushButton("Reject Selected")
        self.face_pending_reject_cluster_button = QPushButton("Reject Current Cluster")
        self.face_pending_undo_button = QPushButton("Undo Last Accept")
        self.face_pending_undo_button.setEnabled(False)
        self.face_pending_accept_threshold = QDoubleSpinBox()
        self.face_pending_accept_threshold.setRange(0.0, 1.0)
        self.face_pending_accept_threshold.setSingleStep(0.01)
        self.face_pending_accept_threshold.setValue(0.85)
        self.face_pending_refresh_button.setToolTip(FACE_HELP["pending_face_review"])
        self.face_pending_accept_button.setToolTip(FACE_HELP["pending_face_review"])
        self.face_pending_accept_above_threshold_button.setToolTip(FACE_HELP["pending_face_review"])
        self.face_pending_accept_cluster_button.setToolTip(FACE_HELP["pending_face_review"])
        self.face_pending_reject_button.setToolTip(FACE_HELP["pending_face_review"])
        self.face_pending_reject_cluster_button.setToolTip(FACE_HELP["pending_face_review"])
        self.face_pending_undo_button.setToolTip(FACE_HELP["pending_face_review"])
        self.face_pending_refresh_button.clicked.connect(self._refresh_pending_face_labels)
        self.face_pending_accept_button.clicked.connect(self._accept_pending_face_labels)
        self.face_pending_accept_above_threshold_button.clicked.connect(self._accept_pending_face_labels_above_threshold)
        self.face_pending_accept_cluster_button.clicked.connect(self._accept_current_cluster_pending_face_labels)
        self.face_pending_reject_button.clicked.connect(self._reject_pending_face_labels)
        self.face_pending_reject_cluster_button.clicked.connect(self._reject_current_cluster_pending_face_labels)
        self.face_pending_undo_button.clicked.connect(self._undo_last_pending_face_acceptance)
        review_buttons.addWidget(self.face_pending_refresh_button)
        review_buttons.addWidget(self._field_widget("Accept threshold", self.face_pending_accept_threshold, tooltip=FACE_HELP["pending_face_review"]))
        review_layout.addLayout(review_buttons)
        self.face_pending_items_toggle = self._expander_button("Show pending proposals", checked=False, tooltip=FACE_HELP["pending_face_review"])
        review_layout.addWidget(self.face_pending_items_toggle)
        self.face_pending_items_panel = QWidget(review_group)
        pending_items_layout = QVBoxLayout(self.face_pending_items_panel)
        pending_items_layout.setContentsMargins(0, 0, 0, 0)
        pending_items_layout.setSpacing(8)
        self.face_pending_model = ListEntryModel(self)
        self.face_pending_list = QListView()
        self.face_pending_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.face_pending_list.setMaximumHeight(180)
        self.face_pending_list.setToolTip(FACE_HELP["pending_face_review"])
        self.face_pending_list.setUniformItemSizes(True)
        self.face_pending_list.setModel(self.face_pending_model)
        pending_selection_model = self.face_pending_list.selectionModel()
        if pending_selection_model is not None:
            pending_selection_model.selectionChanged.connect(lambda *_args: self._refresh_pending_face_preview())
        pending_items_layout.addWidget(self.face_pending_list)
        pending_items_layout.addWidget(self.face_pending_accept_button)
        pending_items_layout.addWidget(self.face_pending_accept_above_threshold_button)
        pending_items_layout.addWidget(self.face_pending_accept_cluster_button)
        pending_items_layout.addWidget(self.face_pending_reject_button)
        pending_items_layout.addWidget(self.face_pending_reject_cluster_button)
        pending_items_layout.addWidget(self.face_pending_undo_button)
        self.face_pending_preview_summary = self._helper_label("Pending preview: select one or more proposals.", tooltip=FACE_HELP["pending_face_review"])
        pending_items_layout.addWidget(self.face_pending_preview_summary)
        self.face_label_audit_label = self._helper_label("Label audit: select one or more proposals to inspect the source and confidence.", tooltip=FACE_HELP["pending_face_review"])
        pending_items_layout.addWidget(self.face_label_audit_label)
        self.face_pending_preview_model = FaceTileListModel(
            self._image_for_face_tile,
            self._face_tile_cache_key_for_item,
            QSize(72, 72),
            self,
        )
        self.face_pending_preview_list = QListView()
        self.face_pending_preview_list.setViewMode(QListView.ViewMode.IconMode)
        self.face_pending_preview_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.face_pending_preview_list.setMovement(QListView.Movement.Static)
        self.face_pending_preview_list.setLayoutMode(QListView.LayoutMode.Batched)
        self.face_pending_preview_list.setBatchSize(24)
        self.face_pending_preview_list.setUniformItemSizes(True)
        self.face_pending_preview_list.setWrapping(True)
        self.face_pending_preview_list.setWordWrap(True)
        self.face_pending_preview_list.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.face_pending_preview_list.setIconSize(QSize(72, 72))
        self.face_pending_preview_list.setGridSize(QSize(96, 118))
        self.face_pending_preview_list.setSpacing(8)
        self.face_pending_preview_list.setMaximumHeight(180)
        self.face_pending_preview_list.setModel(self.face_pending_preview_model)
        self.face_pending_preview_list.setItemDelegate(FaceTileItemDelegate(self.face_pending_preview_list))
        self.face_pending_preview_list.verticalScrollBar().valueChanged.connect(lambda _value: self._schedule_face_tile_refresh())
        pending_items_layout.addWidget(self.face_pending_preview_list)
        self.face_pending_items_toggle.toggled.connect(
            lambda checked: self._set_expander_state(self.face_pending_items_toggle, self.face_pending_items_panel, checked)
        )
        review_layout.addWidget(self.face_pending_items_panel)
        self._set_expander_state(self.face_pending_items_toggle, self.face_pending_items_panel, False)
        self.face_pending_group = review_group

        self._action_buttons.extend(
            [
                self.face_index_button,
                self.face_search_button,
                self.face_search_by_name_card_button,
                self.face_label_button,
                self.face_cluster_button,
                self.face_list_button,
                self.face_merge_button,
                self.face_clear_button,
                self.face_pending_refresh_button,
                self.face_pending_accept_button,
                self.face_pending_accept_above_threshold_button,
                self.face_pending_accept_cluster_button,
                self.face_pending_reject_button,
                self.face_pending_reject_cluster_button,
                self.face_pending_undo_button,
            ]
        )
        self._action_buttons.append(self.face_propagate_button)

        people_query_group, people_query_layout = self._group_box("People Queries", tooltip=FACE_HELP["find_photos_saved_name"])
        people_query_layout.addWidget(
            self._helper_label(
                "Use these to find co-occurrences, named-person photos, unknown people, or photos where one person is dominant.",
                tooltip=FACE_HELP["find_photos_saved_name"],
            )
        )
        self.face_people_query_toggle = self._expander_button("Show people queries", checked=False, tooltip=FACE_HELP["find_photos_saved_name"])
        people_query_layout.addWidget(self.face_people_query_toggle)
        self.face_people_query_panel = QWidget(people_query_group)
        people_query_panel_layout = QVBoxLayout(self.face_people_query_panel)
        people_query_panel_layout.setContentsMargins(0, 0, 0, 0)
        people_query_panel_layout.setSpacing(8)
        people_query_panel_layout.addWidget(self._field_widget("People together", self.face_people_query_names, tooltip=FACE_HELP["find_photos_saved_name"]))
        people_query_panel_layout.addWidget(self.face_people_together_button)
        people_query_panel_layout.addWidget(self.face_people_any_named_button)
        people_query_panel_layout.addWidget(self.face_people_unknown_button)
        people_query_panel_layout.addWidget(self._field_widget("Primary person", self.face_primary_person_query, tooltip=FACE_HELP["find_photos_saved_name"]))
        people_query_panel_layout.addWidget(self.face_people_primary_button)
        people_query_panel_layout.addWidget(self._helper_label("Face query builder", tooltip=FACE_HELP["find_photos_saved_name"]))
        people_query_panel_layout.addWidget(self._field_widget("Faces", self.face_query_builder_faces_state, tooltip=FACE_HELP["find_photos_saved_name"]))
        people_query_panel_layout.addWidget(self._field_widget("Face count", self.face_query_builder_face_count, tooltip=FACE_HELP["find_photos_saved_name"]))
        people_query_panel_layout.addWidget(self._field_widget("Exact person", self.face_query_builder_person, tooltip=FACE_HELP["find_photos_saved_name"]))
        people_query_panel_layout.addWidget(self._field_widget("Partial person", self.face_query_builder_partial, tooltip=FACE_HELP["find_photos_saved_name"]))
        people_query_panel_layout.addWidget(self._field_widget("All/any people", self.face_query_builder_people, tooltip=FACE_HELP["find_photos_saved_name"]))
        people_query_panel_layout.addWidget(self._field_widget("People mode", self.face_query_builder_people_mode, tooltip=FACE_HELP["find_photos_saved_name"]))
        query_flags = QWidget(self.face_people_query_panel)
        query_flags_layout = FlowLayout(query_flags)
        query_flags_layout.setContentsMargins(0, 0, 0, 0)
        query_flags_layout.addWidget(self.face_query_builder_unknown)
        query_flags_layout.addWidget(self.face_query_builder_hidden)
        people_query_panel_layout.addWidget(query_flags)
        people_query_panel_layout.addWidget(self.face_query_builder_run_button)
        self.face_people_query_toggle.toggled.connect(
            lambda checked: self._set_expander_state(self.face_people_query_toggle, self.face_people_query_panel, checked)
        )
        people_query_layout.addWidget(self.face_people_query_panel)
        self._set_expander_state(self.face_people_query_toggle, self.face_people_query_panel, False)
        self.face_people_query_group = people_query_group

        self._update_face_selected_context_label()
        self._update_face_mode_status()
        self._refresh_face_db_usage_label()
        self._refresh_face_search_layout(force=True)
        self.tabs.addTab(tab_body, "Face Search")

    def _build_face_identities_tab(self) -> None:
        tab_body = QWidget()
        layout = QVBoxLayout(tab_body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(
            self._helper_label(
                "Browse known people, group unlabeled faces, and give groups a name.",
                tooltip=FACE_HELP["people_groups"],
            )
        )
        people_actions = QHBoxLayout()
        people_actions.setContentsMargins(0, 0, 0, 0)
        people_actions.setSpacing(6)
        self.face_people_browse_groups_button = QPushButton("Browse Face Groups")
        self.face_people_browse_groups_button.setToolTip("Open All Faces and browse named and unlabeled groups.")
        self.face_people_browse_groups_button.clicked.connect(self._open_all_faces_task)
        self.face_people_cluster_button = QPushButton("Cluster Unlabeled Faces")
        self.face_people_cluster_button.setToolTip("Group similar unlabeled faces so you can review and name them.")
        self.face_people_cluster_button.clicked.connect(lambda: self._cluster_faces(force_global=True))
        people_actions.addWidget(self.face_people_browse_groups_button, stretch=1)
        people_actions.addWidget(self.face_people_cluster_button, stretch=1)
        layout.addLayout(people_actions)
        self._action_buttons.extend([self.face_people_browse_groups_button, self.face_people_cluster_button])
        self._mode_required_buttons.append(self.face_people_cluster_button)
        self.face_identity_refresh_button = QPushButton("Reload Identities")
        self.face_identity_refresh_button.clicked.connect(self._request_face_identity_refresh)
        layout.addWidget(self.face_identity_refresh_button)

        identity_filter_row = QWidget(tab_body)
        identity_filter_layout = FlowLayout(identity_filter_row)
        identity_filter_layout.setContentsMargins(0, 0, 0, 0)
        self.face_identity_search = QLineEdit(identity_filter_row)
        self.face_identity_search.setPlaceholderText("Search saved identities")
        self.face_identity_search.setAccessibleName("Search saved identities")
        self.face_identity_filter = QComboBox(identity_filter_row)
        self.face_identity_filter.addItem("All identities", "all")
        self.face_identity_filter.addItem("Possible duplicates", "duplicates")
        self.face_identity_sort = QComboBox(identity_filter_row)
        self.face_identity_sort.addItem("Name A–Z", "ascending")
        self.face_identity_sort.addItem("Name Z–A", "descending")
        identity_filter_layout.addWidget(self.face_identity_search)
        identity_filter_layout.addWidget(self.face_identity_filter)
        identity_filter_layout.addWidget(self.face_identity_sort)
        layout.addWidget(identity_filter_row)
        self.face_identity_count_label = self._helper_label("No saved identities", tooltip=FACE_HELP["people_groups"])
        layout.addWidget(self.face_identity_count_label)

        self.face_identity_model = PagedListEntryModel(self, page_size=50)
        self.face_identity_list = QListView()
        self.face_identity_list.setModel(self.face_identity_model)
        self.face_identity_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.face_identity_list.setUniformItemSizes(True)
        self.face_identity_list.setToolTip(FACE_HELP["people_groups"])
        self.face_identity_list.selectionModel().selectionChanged.connect(lambda *_args: self._refresh_selected_face_identity())
        self.face_identity_search.textChanged.connect(lambda _text: self._apply_face_identity_filters())
        self.face_identity_filter.currentIndexChanged.connect(lambda _index: self._apply_face_identity_filters())
        self.face_identity_sort.currentIndexChanged.connect(lambda _index: self._apply_face_identity_filters())
        self.face_identity_model.rowsInserted.connect(
            lambda *_args: self.face_identity_count_label.setText(
                f"Showing {self.face_identity_model.rowCount()} of {self.face_identity_model.total_count} saved identities"
            )
        )
        layout.addWidget(self.face_identity_list)

        self.face_identity_summary = self._helper_label(
            "Select a saved identity to inspect its prototype examples.",
            tooltip=FACE_HELP["people_groups"],
        )
        layout.addWidget(self.face_identity_summary)
        self.face_identity_duplicate_warning = self._helper_label("", tooltip=FACE_HELP["people_groups"])
        self.face_identity_duplicate_warning.setVisible(False)
        layout.addWidget(self.face_identity_duplicate_warning)

        self.face_identity_prototype_model = FaceTileListModel(
            self._image_for_face_tile,
            self._face_tile_cache_key_for_item,
            QSize(72, 72),
            self,
        )
        self.face_identity_prototype_list = QListView()
        self.face_identity_prototype_list.setViewMode(QListView.ViewMode.IconMode)
        self.face_identity_prototype_list.setResizeMode(QListView.ResizeMode.Adjust)
        self.face_identity_prototype_list.setMovement(QListView.Movement.Static)
        self.face_identity_prototype_list.setLayoutMode(QListView.LayoutMode.Batched)
        self.face_identity_prototype_list.setBatchSize(24)
        self.face_identity_prototype_list.setUniformItemSizes(True)
        self.face_identity_prototype_list.setWrapping(True)
        self.face_identity_prototype_list.setWordWrap(True)
        self.face_identity_prototype_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.face_identity_prototype_list.setIconSize(QSize(72, 72))
        self.face_identity_prototype_list.setGridSize(QSize(96, 118))
        self.face_identity_prototype_list.setSpacing(8)
        self.face_identity_prototype_list.setMaximumHeight(220)
        self.face_identity_prototype_list.setModel(self.face_identity_prototype_model)
        self.face_identity_prototype_list.setItemDelegate(FaceTileItemDelegate(self.face_identity_prototype_list))
        if self.face_identity_prototype_list.selectionModel() is not None:
            self.face_identity_prototype_list.selectionModel().selectionChanged.connect(
                lambda *_args: self._update_face_identity_prototype_actions()
            )
        self.face_identity_prototype_list.verticalScrollBar().valueChanged.connect(
            lambda _value: self._schedule_face_tile_refresh()
        )
        layout.addWidget(self.face_identity_prototype_list)

        self.face_identity_open_photo_button = QPushButton("Show Parent Photo")
        self.face_identity_open_photo_button.clicked.connect(self._focus_selected_identity_prototype_photo)
        layout.addWidget(self.face_identity_open_photo_button)
        self.face_identity_open_inspector_button = QPushButton("Open Parent Photo In Inspector")
        self.face_identity_open_inspector_button.clicked.connect(self._open_selected_identity_prototype_in_inspector)
        layout.addWidget(self.face_identity_open_inspector_button)

        self.face_identity_management_group = QGroupBox("Management", tab_body)
        self.face_identity_management_group.setCheckable(True)
        self.face_identity_management_group.setChecked(False)
        identity_management_layout = QVBoxLayout(self.face_identity_management_group)
        self.face_identity_remove_button = QPushButton("Remove From Prototype")
        self.face_identity_remove_button.clicked.connect(self._remove_selected_identity_prototype_faces)
        identity_management_layout.addWidget(self.face_identity_remove_button)
        self.face_identity_pin_button = QPushButton("Pin Canonical Example")
        self.face_identity_pin_button.clicked.connect(self._pin_selected_identity_prototype_face)
        identity_management_layout.addWidget(self.face_identity_pin_button)
        self.face_identity_merge_target = QLineEdit()
        self.face_identity_merge_target.setPlaceholderText("Saved identity to keep")
        identity_management_layout.addWidget(self._field_widget("Merge into", self.face_identity_merge_target, tooltip=FACE_HELP["merge_target"]))
        self.face_identity_merge_button = QPushButton("Merge Selected Identity Into Target")
        self.face_identity_merge_button.clicked.connect(self._merge_selected_identity_into_target)
        self.face_identity_merge_button.setProperty("kind", "danger")
        identity_management_layout.addWidget(self.face_identity_merge_button)
        self.face_identity_clear_button = QPushButton("Remove Name From Selected Identity")
        self.face_identity_clear_button.clicked.connect(self._clear_selected_identity_labels)
        self.face_identity_clear_button.setProperty("kind", "danger")
        identity_management_layout.addWidget(self.face_identity_clear_button)
        self.face_export_identities_button = QPushButton("Export Identities")
        self.face_export_identities_button.clicked.connect(self._export_face_identities)
        identity_management_layout.addWidget(self.face_export_identities_button)
        self.face_import_identities_button = QPushButton("Import Identities")
        self.face_import_identities_button.clicked.connect(self._import_face_identities)
        identity_management_layout.addWidget(self.face_import_identities_button)
        self.face_rebuild_db_button = QPushButton("Rebuild Current Face Index")
        self.face_rebuild_db_button.clicked.connect(self._rebuild_current_face_index)
        identity_management_layout.addWidget(self.face_rebuild_db_button)
        self.face_action_audit_refresh_button = QPushButton("Refresh Action Audit")
        self.face_action_audit_refresh_button.clicked.connect(self._refresh_face_action_audit)
        identity_management_layout.addWidget(self.face_action_audit_refresh_button)
        self.face_action_audit_list = QListWidget()
        self.face_action_audit_list.setMaximumHeight(180)
        identity_management_layout.addWidget(self.face_action_audit_list)
        layout.addWidget(self.face_identity_management_group)

        self.face_identity_danger_group = QGroupBox("Danger zone", tab_body)
        self.face_identity_danger_group.setCheckable(True)
        self.face_identity_danger_group.setChecked(False)
        self.face_identity_danger_group.setProperty("role", "danger")
        identity_danger_layout = QVBoxLayout(self.face_identity_danger_group)
        self.face_disable_recognition_button = QPushButton("Disable Face Recognition")
        self.face_enable_recognition_button = QPushButton("Enable Face Recognition")
        self.face_disable_recognition_button.clicked.connect(lambda: self._set_face_recognition_enabled(False))
        self.face_enable_recognition_button.clicked.connect(lambda: self._set_face_recognition_enabled(True))
        self.face_disable_recognition_button.setProperty("kind", "danger")
        identity_danger_layout.addWidget(self.face_disable_recognition_button)
        identity_danger_layout.addWidget(self.face_enable_recognition_button)
        self.face_db_usage_label = self._helper_label("Face data usage: unavailable.", tooltip=FACE_HELP["scan_source"])
        identity_danger_layout.addWidget(self.face_db_usage_label)
        self.face_delete_db_button = QPushButton("Delete Active Face DB")
        self.face_purge_data_button = QPushButton("Purge Face Data")
        self.face_delete_db_button.clicked.connect(self._delete_active_face_db)
        self.face_purge_data_button.clicked.connect(self._purge_face_data)
        self.face_delete_db_button.setProperty("kind", "danger")
        self.face_purge_data_button.setProperty("kind", "danger")
        identity_danger_layout.addWidget(self.face_delete_db_button)
        identity_danger_layout.addWidget(self.face_purge_data_button)
        layout.addWidget(self.face_identity_danger_group)
        self._action_buttons.extend(
            [
                self.face_identity_refresh_button,
                self.face_identity_open_photo_button,
                self.face_identity_open_inspector_button,
                self.face_identity_remove_button,
                self.face_identity_pin_button,
                self.face_identity_merge_button,
                self.face_identity_clear_button,
                self.face_export_identities_button,
                self.face_import_identities_button,
                self.face_disable_recognition_button,
                self.face_enable_recognition_button,
                self.face_rebuild_db_button,
                self.face_delete_db_button,
                self.face_purge_data_button,
                self.face_action_audit_refresh_button,
            ]
        )
        self._refresh_selected_face_identity()
        self._refresh_face_db_usage_label()
        self._refresh_face_action_audit()
        self.tabs.addTab(tab_body, "Identities")

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

        face_library_tab = 0
        if hasattr(self, "face_library_tabs"):
            try:
                if self.face_library_tabs.tabBar().isTabVisible(1):
                    face_library_tab = int(self.face_library_tabs.currentIndex())
            except Exception:
                pass

        return {
            "active_tab": int(self.tabs.currentIndex()),
            "face_mode": str(self.current_face_mode()),
            "face_ui_mode": str(self.current_ui_mode()),
            "face_pipeline_by_mode": {
                mode: {
                    "detector_id": str(dict(prefs).get("detector_id", "")),
                    "embedder_id": str(dict(prefs).get("embedder_id", "")),
                    "preferred_detector_id": str(dict(prefs).get("preferred_detector_id", "")),
                    "preferred_embedder_id": str(dict(prefs).get("preferred_embedder_id", "")),
                    "fallback_detector_id": str(dict(prefs).get("fallback_detector_id", "")),
                    "detector_policy": str(dict(prefs).get("detector_policy", "single") or "single"),
                    "verifier_mode": str(dict(prefs).get("verifier_mode", "off") or "off"),
                    "score_threshold": float(dict(prefs).get("score_threshold", DEFAULT_FACE_SCORE_THRESHOLD) or 0.0),
                    "max_detections": int(dict(prefs).get("max_detections", DEFAULT_FACE_MAX_DETECTIONS) or DEFAULT_FACE_MAX_DETECTIONS),
                    "quality_profile_id": str(dict(prefs).get("quality_profile_id", "balanced") or "balanced"),
                    "quality_thresholds": dict(dict(prefs).get("quality_thresholds", {}) or {}),
                    "search_quality_min": str(dict(prefs).get("search_quality_min", "clean") or "clean"),
                    "cluster_quality_min": str(dict(prefs).get("cluster_quality_min", "clean") or "clean"),
                    "prototype_quality_min": str(dict(prefs).get("prototype_quality_min", "clean") or "clean"),
                    "recognition_min_score": float(dict(prefs).get("recognition_min_score", 0.35) or 0.35),
                    "auto_label_min_score": float(dict(prefs).get("auto_label_min_score", 0.72) or 0.72),
                    "rerank_policy": str(dict(prefs).get("rerank_policy", "off") or "off"),
                    "rerank_top_n": int(dict(prefs).get("rerank_top_n", 25) or 25),
                }
                for mode, prefs in dict(self._face_pipeline_prefs_by_mode).items()
            },
            "splitter_sizes": [int(value) for value in self.workspace_splitter.sizes()],
            "face_library_tab": face_library_tab,
            "face_review_sort": _combo_id(getattr(self, "face_review_sort", None)),
            "face_photo_filter": _combo_id(getattr(self, "face_photo_filter", None)),
            "face_cluster_backend": _combo_id(getattr(self, "face_cluster_backend", None)),
            "face_cluster_backends": list(self.current_face_cluster_backends()),
            "face_cluster_backend_override_enabled": bool(getattr(self, "face_cluster_backend_override_checkbox", None).isChecked()) if hasattr(self, "face_cluster_backend_override_checkbox") else False,
            "face_cluster_outlier_policy": _combo_id(getattr(self, "face_cluster_outlier_policy_combo", None)),
            "face_hdbscan_min_cluster_size": int(self.face_hdbscan_min_cluster_size_spin.value()) if hasattr(self, "face_hdbscan_min_cluster_size_spin") else 4,
            "face_hdbscan_min_samples": int(self.face_hdbscan_min_samples_spin.value()) if hasattr(self, "face_hdbscan_min_samples_spin") else 0,
            "face_hdbscan_cluster_selection_epsilon": float(self.face_hdbscan_cluster_selection_epsilon_spin.value()) if hasattr(self, "face_hdbscan_cluster_selection_epsilon_spin") else 0.0,
            "face_hdbscan_allow_single_cluster": bool(self.face_hdbscan_allow_single_cluster_checkbox.isChecked()) if hasattr(self, "face_hdbscan_allow_single_cluster_checkbox") else False,
            "search_only_current_folder": bool(self.search_only_current_folder.isChecked()),
            "show_tiny_detections": bool(self._show_tiny_detections_enabled()),
            "image_model": _combo_id(getattr(self, "image_model", None)),
            "text_model": _combo_id(getattr(self, "text_model", None)),
            "duplicate_model": _combo_id(getattr(self, "duplicate_model", None)),
            "face_advanced_expanded": bool(getattr(self, "face_advanced_toggle", None).isChecked()) if hasattr(self, "face_advanced_toggle") else False,
            "face_advanced_tab": int(getattr(self, "face_advanced_tabs", None).currentIndex()) if hasattr(self, "face_advanced_tabs") else 0,
            "face_quality_manual_expanded": bool(getattr(self, "face_quality_manual_toggle", None).isChecked()) if hasattr(self, "face_quality_manual_toggle") else False,
            "face_selected_options_expanded": bool(getattr(self, "face_search_selected_options_toggle", None).isChecked()) if hasattr(self, "face_search_selected_options_toggle") else False,
            "face_find_options_expanded": bool(getattr(self, "face_find_options_toggle", None).isChecked()) if hasattr(self, "face_find_options_toggle") else False,
            "face_save_options_expanded": bool(getattr(self, "face_save_options_toggle", None).isChecked()) if hasattr(self, "face_save_options_toggle") else False,
            "face_manage_options_expanded": bool(getattr(self, "face_manage_options_toggle", None).isChecked()) if hasattr(self, "face_manage_options_toggle") else False,
            "face_pending_items_expanded": bool(getattr(self, "face_pending_items_toggle", None).isChecked()) if hasattr(self, "face_pending_items_toggle") else False,
            "face_people_expanded": bool(getattr(self, "face_people_toggle", None).isChecked()) if hasattr(self, "face_people_toggle") else False,
            "face_profile_expanded": bool(getattr(self, "face_profile_toggle", None).isChecked()) if hasattr(self, "face_profile_toggle") else False,
            "face_identity_maintenance_expanded": bool(getattr(self, "face_identity_maintenance_toggle", None).isChecked()) if hasattr(self, "face_identity_maintenance_toggle") else False,
            "face_people_query_expanded": bool(getattr(self, "face_people_query_toggle", None).isChecked()) if hasattr(self, "face_people_query_toggle") else False,
            "face_walkthrough_dismissed": bool(hasattr(self, "face_walkthrough_panel") and not self.face_walkthrough_panel.isVisible()),
        }

    def apply_state(self, state: dict[str, object] | None) -> None:
        if not state:
            return
        try:
            self.tabs.setCurrentIndex(int(state.get("active_tab", 0)))
        except Exception:
            pass
        if "face_mode" in state:
            self.set_active_face_mode(str(state.get("face_mode", "human")), refresh=False)
        self.set_ui_mode(str(state.get("face_ui_mode", self.current_ui_mode())))
        pipeline_by_mode = state.get("face_pipeline_by_mode")
        if isinstance(pipeline_by_mode, dict):
            for raw_mode, raw_prefs in pipeline_by_mode.items():
                if not isinstance(raw_prefs, dict):
                    continue
                mode = normalize_face_mode(str(raw_mode))
                prefs = self._mode_face_pipeline_prefs(mode)
                requested_detector_id = str(
                    raw_prefs.get("detector_id")
                    or prefs.get("detector_id")
                    or default_face_detector_id(self.face_model_root, mode)
                )
                requested_embedder_id = str(
                    raw_prefs.get("embedder_id")
                    or prefs.get("embedder_id")
                    or default_face_embedder_id(self.face_model_root, mode)
                )
                prefs["preferred_detector_id"] = normalize_face_component_id(
                    str(raw_prefs.get("preferred_detector_id") or prefs.get("preferred_detector_id") or requested_detector_id),
                    requested_detector_id,
                )
                prefs["preferred_embedder_id"] = normalize_face_component_id(
                    str(raw_prefs.get("preferred_embedder_id") or prefs.get("preferred_embedder_id") or requested_embedder_id),
                    requested_embedder_id,
                )
                if mode == "human":
                    prefs["detector_id"], prefs["embedder_id"] = resolve_ready_face_pipeline_ids(
                        self.face_model_root,
                        mode,
                        requested_detector_id,
                        requested_embedder_id,
                    )
                else:
                    prefs["detector_id"] = self._resolve_available_face_detector_id(mode, requested_detector_id)
                    prefs["embedder_id"] = self._resolve_available_face_embedder_id(mode, requested_embedder_id)
                prefs["fallback_detector_id"] = self._resolve_available_face_detector_id(
                    mode,
                    str(raw_prefs.get("fallback_detector_id") or prefs.get("fallback_detector_id") or prefs["detector_id"]),
                )
                prefs["detector_policy"] = str(raw_prefs.get("detector_policy", prefs.get("detector_policy", "single")) or "single").strip().lower()
                prefs["verifier_mode"] = str(raw_prefs.get("verifier_mode", prefs.get("verifier_mode", "off")) or "off").strip().lower()
                prefs["score_threshold"] = float(raw_prefs.get("score_threshold", prefs.get("score_threshold", DEFAULT_FACE_SCORE_THRESHOLD)) or 0.0)
                prefs["max_detections"] = max(
                    1,
                    int(raw_prefs.get("max_detections", prefs.get("max_detections", DEFAULT_FACE_MAX_DETECTIONS)) or DEFAULT_FACE_MAX_DETECTIONS),
                )
                profile_id = str(raw_prefs.get("quality_profile_id", prefs.get("quality_profile_id", "balanced")) or "balanced").strip().lower()
                prefs["quality_profile_id"] = profile_id
                thresholds = dict(face_quality_profile_config(mode, profile_id))
                thresholds.update({str(key): value for key, value in dict(raw_prefs.get("quality_thresholds", {}) or {}).items()})
                prefs["quality_thresholds"] = thresholds
                prefs["search_quality_min"] = str(raw_prefs.get("search_quality_min", prefs.get("search_quality_min", "clean")) or "clean").strip().lower()
                prefs["cluster_quality_min"] = str(raw_prefs.get("cluster_quality_min", prefs.get("cluster_quality_min", "clean")) or "clean").strip().lower()
                prefs["prototype_quality_min"] = str(raw_prefs.get("prototype_quality_min", prefs.get("prototype_quality_min", "clean")) or "clean").strip().lower()
                prefs["recognition_min_score"] = float(raw_prefs.get("recognition_min_score", prefs.get("recognition_min_score", 0.35)) or 0.35)
                prefs["auto_label_min_score"] = float(raw_prefs.get("auto_label_min_score", prefs.get("auto_label_min_score", 0.72)) or 0.72)
                prefs["rerank_policy"] = str(raw_prefs.get("rerank_policy", prefs.get("rerank_policy", "off")) or "off").strip().lower()
                prefs["rerank_top_n"] = max(1, int(raw_prefs.get("rerank_top_n", prefs.get("rerank_top_n", 25)) or 25))
        self._face_pipeline_applied_by_mode = {
            mode: self._clone_face_pipeline_prefs(prefs)
            for mode, prefs in self._face_pipeline_prefs_by_mode.items()
        }
        self._face_pipeline_dirty_modes.clear()
        splitter_sizes = state.get("splitter_sizes")
        if isinstance(splitter_sizes, (list, tuple)) and len(splitter_sizes) == 2:
            try:
                self.workspace_splitter.setSizes([max(1, int(splitter_sizes[0])), max(1, int(splitter_sizes[1]))])
            except Exception:
                pass
        if hasattr(self, "face_library_tabs"):
            try:
                index = max(0, min(int(state.get("face_library_tab", 0)), self.face_library_tabs.count() - 1))
                # Review & Name is retained but intentionally hidden in this
                # pass, so a previous workspace state must not reopen it.
                if index == 1 and not self.face_library_tabs.tabBar().isTabVisible(1):
                    index = 0
                self.face_library_tabs.setCurrentIndex(index)
            except Exception:
                pass
        self._set_combo_to_id(getattr(self, "face_review_sort", None), state.get("face_review_sort"))
        self._set_combo_to_id(getattr(self, "face_photo_filter", None), state.get("face_photo_filter", "with_faces"))
        self._set_combo_to_id(getattr(self, "face_cluster_backend", None), state.get("face_cluster_backend"))
        if hasattr(self, "face_cluster_backend_override_checkbox"):
            self.face_cluster_backend_override_checkbox.setChecked(bool(state.get("face_cluster_backend_override_enabled", False)))
        if hasattr(self, "face_hdbscan_min_cluster_size_spin"):
            self.face_hdbscan_min_cluster_size_spin.setValue(max(2, int(state.get("face_hdbscan_min_cluster_size", 4) or 4)))
        if hasattr(self, "face_hdbscan_min_samples_spin"):
            self.face_hdbscan_min_samples_spin.setValue(max(0, int(state.get("face_hdbscan_min_samples", 0) or 0)))
        if hasattr(self, "face_hdbscan_cluster_selection_epsilon_spin"):
            self.face_hdbscan_cluster_selection_epsilon_spin.setValue(max(0.0, float(state.get("face_hdbscan_cluster_selection_epsilon", 0.0) or 0.0)))
        if hasattr(self, "face_hdbscan_allow_single_cluster_checkbox"):
            self.face_hdbscan_allow_single_cluster_checkbox.setChecked(bool(state.get("face_hdbscan_allow_single_cluster", False)))
        self._refresh_face_cluster_backend_controls()
        self._set_combo_to_id(getattr(self, "face_cluster_outlier_policy_combo", None), state.get("face_cluster_outlier_policy", "isolate"))
        self.search_only_current_folder.setChecked(bool(state.get("search_only_current_folder", True)))
        self.show_tiny_detections_checkbox.setChecked(bool(state.get("show_tiny_detections", False)))
        self._set_combo_to_id(getattr(self, "image_model", None), state.get("image_model"))
        self._set_combo_to_id(getattr(self, "text_model", None), state.get("text_model"))
        self._set_combo_to_id(getattr(self, "duplicate_model", None), state.get("duplicate_model"))
        self._refresh_face_pipeline_controls()
        if hasattr(self, "face_advanced_toggle"):
            self._set_expander_state(
                self.face_advanced_toggle,
                getattr(self, "face_advanced_panel", None),
                bool(state.get("face_advanced_expanded", False)),
            )
        if hasattr(self, "face_advanced_tabs"):
            try:
                index = max(0, min(int(state.get("face_advanced_tab", 0)), self.face_advanced_tabs.count() - 1))
                self.face_advanced_tabs.setCurrentIndex(index)
            except Exception:
                pass
        if hasattr(self, "face_quality_manual_toggle"):
            self._set_expander_state(
                self.face_quality_manual_toggle,
                getattr(self, "face_quality_manual_panel", None),
                bool(state.get("face_quality_manual_expanded", False)),
            )
        for button_name, panel_name, state_key in (
            ("face_search_selected_options_toggle", "face_search_selected_options_panel", "face_selected_options_expanded"),
            ("face_find_options_toggle", "face_find_options_panel", "face_find_options_expanded"),
            ("face_save_options_toggle", "face_save_options_panel", "face_save_options_expanded"),
            ("face_manage_options_toggle", "face_manage_options_panel", "face_manage_options_expanded"),
            ("face_pending_items_toggle", "face_pending_items_panel", "face_pending_items_expanded"),
            ("face_people_toggle", "face_people_panel", "face_people_expanded"),
            ("face_profile_toggle", "face_profile_panel", "face_profile_expanded"),
            ("face_identity_maintenance_toggle", "face_identity_maintenance_panel", "face_identity_maintenance_expanded"),
            ("face_people_query_toggle", "face_people_query_panel", "face_people_query_expanded"),
        ):
            if hasattr(self, button_name):
                self._set_expander_state(
                    getattr(self, button_name, None),
                    getattr(self, panel_name, None),
                    bool(state.get(state_key, False)),
                )
        self._set_face_walkthrough_visible(not bool(state.get("face_walkthrough_dismissed", False)))
        self._apply_face_ui_mode()
        self._on_face_mode_changed()

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

    def _populate_duplicate_review(self, results: list[SearchResult], query_image_path: str) -> None:
        if not hasattr(self, "duplicate_review_list"):
            return
        self._duplicate_review_groups = SimilaritySearchService.build_duplicate_review_groups(query_image_path, list(results or []))
        self._duplicate_decisions = []
        self.duplicate_review_list.clear()
        for index, group in enumerate(self._duplicate_review_groups):
            kind = "Exact" if group.exact else "Near"
            item = QListWidgetItem(
                f"{kind} | hash={group.hash_backend}:{group.hash_distance} | score={group.score:.3f} | "
                f"{Path(group.candidate_path).name}"
            )
            item.setData(Qt.ItemDataRole.UserRole, int(index))
            item.setToolTip(f"Keeper: {group.query_image_path}\nDuplicate candidate: {group.candidate_path}\n{group.match_reason}")
            self.duplicate_review_list.addItem(item)
        if self.duplicate_review_list.count() > 0:
            self.duplicate_review_list.setCurrentRow(0)
        else:
            self.duplicate_review_summary.setText("Duplicate review: no duplicate candidates matched this query.")

    def _selected_duplicate_review_group(self) -> DuplicateReviewGroup | None:
        if not hasattr(self, "duplicate_review_list"):
            return None
        item = self.duplicate_review_list.currentItem()
        if item is None:
            return None
        try:
            index = int(item.data(Qt.ItemDataRole.UserRole))
        except Exception:
            return None
        if 0 <= index < len(self._duplicate_review_groups):
            return self._duplicate_review_groups[index]
        return None

    def _update_duplicate_review_selection(self) -> None:
        if not hasattr(self, "duplicate_review_summary"):
            return
        group = self._selected_duplicate_review_group()
        if group is None:
            self.duplicate_review_summary.setText("Duplicate review: select a pair.")
            return
        self.duplicate_review_summary.setText(
            f"Keeper: {group.query_image_path}\n"
            f"Duplicate candidate: {group.candidate_path}\n"
            f"{'Exact' if group.exact else 'Near'} duplicate, score={group.score:.3f}, "
            f"hash={group.hash_backend}:{group.hash_distance}."
        )

    def _record_duplicate_decision(self, action: str) -> None:
        group = self._selected_duplicate_review_group()
        if group is None:
            errorBox("No duplicate pair selected", "Run duplicate search and select a review pair first.")
            return
        action = str(action or "review").strip().lower()
        if action == "trash" and not confirmBox(
            "Record trash decision?",
            "Record this duplicate as a trash candidate? Source image files will not be moved.",
            parent=self,
        ):
            return
        decision = {
            "keeper_path": group.query_image_path,
            "duplicate_path": group.candidate_path,
            "action": action,
            "tag": self.duplicate_review_tag.text().strip() if hasattr(self, "duplicate_review_tag") else "",
            "score": float(group.score),
            "hash_distance": int(group.hash_distance),
        }
        self._duplicate_decisions.append(decision)
        self.status_label.setText(
            f"Recorded {action} duplicate decision for {Path(group.candidate_path).name}. Source files were not moved."
        )

    def _export_duplicate_decisions(self) -> None:
        if not self._duplicate_decisions:
            errorBox("No duplicate decisions", "Tag or trash at least one duplicate candidate before export.")
            return
        out_path, _ = QFileDialog.getSaveFileName(self, "Export Duplicate Decisions", "", "JSON Files (*.json)")
        if not out_path:
            return
        try:
            payload = SimilaritySearchService.export_duplicate_decisions(self._duplicate_decisions, out_path)
            self.status_label.setText(f"Exported {len(payload.get('decisions', []) or [])} duplicate decision(s) to {out_path}.")
        except Exception as exc:
            errorBox("Export failed", str(exc))

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
            if request.search_mode == "duplicate":
                self._populate_duplicate_review(list(results or []), request.query_image_path)

        self._start_job("Searching images", _run, _done)

    def _index_faces(self) -> None:
        directory = self._current_directory()
        scope_paths = self._current_scope_paths()
        if not directory and not scope_paths:
            errorBox("No folder selected", "Choose a folder in the file pane first.")
            return
        if not self._ensure_face_models_ready("index faces"):
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

    def _populate_query_face_candidates(self, image_path: str, faces: list[object]) -> None:
        self._face_query_detected_faces = list(faces or [])
        self.face_query_faces_model.clear()
        if not str(image_path or "").strip():
            self.face_query_faces_summary.setText("Detect faces in the query photo, then pick one or more faces to search.")
            return
        if not faces:
            self.face_query_faces_summary.setText(f"No detectable faces were found in {Path(image_path).name}.")
            return
        entries: list[ListEntry] = []
        for index, face in enumerate(list(faces or []), start=1):
            try:
                crop_qimage = ImageQt(face.crop.convert("RGBA"))
                icon = QIcon(QPixmap.fromImage(crop_qimage))
            except Exception:
                icon = QIcon()
            entries.append(
                ListEntry(
                    title=f"Face #{index}",
                    tooltip=f"{image_path}\nface #{index}\nbbox={tuple(int(value) for value in face.bbox)}\nconf={float(face.confidence):.3f}",
                    payload=tuple(int(value) for value in face.bbox),
                    icon=icon,
                )
            )
        self.face_query_faces_model.set_items(entries)
        if self.face_query_faces_model.rowCount() == 1:
            index = self.face_query_faces_model.index(0, 0)
            selection_model = self.face_query_faces_list.selectionModel()
            if selection_model is not None and index.isValid():
                selection_model.select(index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
                selection_model.setCurrentIndex(index, QItemSelectionModel.SelectionFlag.ClearAndSelect)
        self.face_query_faces_summary.setText(
            f"Detected {self.face_query_faces_model.rowCount()} face(s) in {Path(image_path).name}. Select one or more to search."
        )

    def _selected_query_face_bboxes(self) -> list[tuple[int, int, int, int]]:
        boxes: list[tuple[int, int, int, int]] = []
        for entry in self._selected_list_entries(self.face_query_faces_list, self.face_query_faces_model):
            bbox = entry.payload
            if isinstance(bbox, tuple) and len(bbox) == 4:
                boxes.append(tuple(int(value) for value in bbox))
        return boxes

    def _detect_query_photo_faces(self) -> None:
        query_face_image = self.face_query_path.text().strip()
        if not query_face_image:
            errorBox("Missing query image", "Choose a query face image first.")
            return

        def _run(progress, cancel_check):
            progress(-1, "Detecting query faces...")
            if cancel_check():
                raise Cancelled()
            result = self._active_face_service().detect_faces_for_image(
                query_face_image,
                include_tiny_faces=self._show_tiny_detections_enabled(),
            )
            if cancel_check():
                raise Cancelled()
            return result

        def _done(faces) -> None:
            self._populate_query_face_candidates(query_face_image, list(faces or []))
            self.status_label.setText(f"Detected {len(faces or [])} query face(s).")

        self._start_job("Detecting query faces", _run, _done)

    def _search_face_refs(self, face_refs: list[tuple[str, int]], *, source_label: str) -> None:
        refs = list(dict.fromkeys((str(image_path), int(face_index)) for image_path, face_index in (face_refs or []) if str(image_path or "").strip()))
        if not refs:
            errorBox("No faces selected", "Select one or more saved faces first.")
            return
        if len(refs) == 1:
            return self._search_face_ref(refs[0][0], refs[0][1], source_label=source_label)
        folder = self.face_name_folder_filter.text().strip() if hasattr(self, "face_name_folder_filter") else ""
        scope_paths = self._current_scope_paths()
        if not folder and self.search_only_current_folder.isChecked():
            folder = self._current_directory()
        effective_min_score = self._effective_face_search_min_score(float(self.face_browser_min_score.value()))

        def _run(progress, cancel_check):
            progress(-1, "Searching selected faces...")
            return _call_with_optional_cancel(
                self._active_face_service().search_similar_faces,
                refs,
                top_k=int(self.face_browser_top_k.value()),
                min_score=effective_min_score,
                folder_prefix=folder,
                candidate_paths=scope_paths,
                include_tiny_faces=self._show_tiny_detections_enabled(),
                cancel_check=cancel_check,
            )

        def _done(results) -> None:
            self.status_label.setText(f"Found {len(results)} matches from {len(refs)} example faces.")
            self._show_face_match_results(
                list(results or []),
                title=f"{source_label} matches",
                summary=f"{source_label} returned {len(results)} matching face tile(s) from {len(refs)} selected example face(s).",
                match_label="match",
            )

        self._start_job("Searching selected faces", _run, _done)

    def _search_faces(self) -> None:
        query_face_image = self.face_query_path.text().strip()
        if not query_face_image:
            errorBox("Missing query image", "Choose a query face image first.")
            return
        scope_paths = self._current_scope_paths()
        selected_bboxes = self._selected_query_face_bboxes()
        effective_min_score = self._effective_face_search_min_score(float(self.face_min_score.value()))

        def _run(progress, cancel_check):
            progress(-1, "Searching faces...")
            if selected_bboxes:
                return _call_with_optional_cancel(
                    self._active_face_service().search_query_faces,
                    query_face_image,
                    selected_bboxes,
                    top_k=self.face_top_k.value(),
                    min_face_score=effective_min_score,
                    candidate_paths=scope_paths,
                    include_tiny_faces=self._show_tiny_detections_enabled(),
                    cancel_check=cancel_check,
                )
            request = FaceSearchRequest(
                query_face_image=query_face_image,
                top_k=self.face_top_k.value(),
                min_face_score=effective_min_score,
                candidate_paths=scope_paths,
                include_tiny_faces=self._show_tiny_detections_enabled(),
            )
            return _call_with_optional_cancel(
                self._active_face_service().search_faces,
                request,
                cancel_check=cancel_check,
            )

        def _done(results) -> None:
            self.status_label.setText(f"Found {len(results)} face matches.")
            self._show_face_match_results(
                list(results or []),
                title="Query photo matches",
                summary=(
                    f"Query photo search returned {len(results)} matching face tile(s) using "
                    f"{max(1, len(selected_bboxes))} query face example(s)."
                ),
                match_label="query",
            )

        self._start_job("Searching faces", _run, _done)

    def _cluster_faces(self, *, force_global: bool = False) -> None:

        num_clusters = self.face_cluster_count.value()
        min_score = self.face_min_score.value()
        backends = list(self.current_face_cluster_backends())
        backend = str(backends[0] if backends else "hdbscan")
        backend_options = self.current_face_cluster_backend_options()
        outlier_policy = self.current_face_cluster_outlier_policy()
        scope_paths = None if force_global else self._current_scope_paths()
        folder = "" if force_global else (self._current_directory() if self.search_only_current_folder.isChecked() else "")

        def _run(progress, cancel_check):
            progress(-1, "Clustering faces...")
            return _call_with_optional_cancel(
                self._active_face_service().cluster_faces_compare,
                num_clusters=num_clusters,
                min_face_score=min_score,
                backends=backends,
                outlier_policy=outlier_policy,
                backend_options_by_backend={str(item): dict(backend_options) for item in backends if str(item) == "hdbscan"},
                folder_prefix=folder,
                candidate_paths=scope_paths,
                include_tiny_faces=self._show_tiny_detections_enabled(),
                cancel_check=cancel_check,
            )

        def _done(result) -> None:
            clusters_by_key = dict(getattr(result, "clusters_by_key", {}) or {})
            cluster_count = sum(len(value) for value in clusters_by_key.values())
            backend_label = self._face_cluster_backend_label(backend)
            self.status_label.setText(f"Generated {cluster_count} {backend_label} face cluster group(s).")
            self._show_face_cluster_results(
                result,
                summary=(
                    f"Global clustering produced {cluster_count} {backend_label} face cluster group(s)."
                    if force_global
                    else f"Folder clustering produced {cluster_count} {backend_label} face cluster group(s)."
                ),
            )
            self.face_clusters_ready.emit(clusters_by_key)

        self._start_job("Clustering faces", _run, _done)

    def _label_person(self) -> None:
        examples = [path.strip() for path in self.face_examples.text().split(";") if path.strip()]
        if self.face_query_path.text().strip():
            examples.append(self.face_query_path.text().strip())
        examples = list(dict.fromkeys(examples))
        person_name = self.person_name.text().strip()
        if not person_name:
            errorBox("Missing name", "Enter the saved identity name first.")
            return
        if not examples:
            errorBox("Missing examples", "Choose a query photo or example photos first.")
            return

        request = FaceLabelRequest(
            person_name=person_name,
            example_image_paths=examples,
            similarity_threshold=self.face_label_threshold.value(),
        )

        def _run(progress, cancel_check):
            progress(-1, "Saving identity prototype...")
            _ = cancel_check
            return self._active_face_service().label_face_examples(request)

        def _done(person) -> None:
            profile_tags_widget = getattr(self, "face_profile_tags", None)
            profile_notes_widget = getattr(self, "face_profile_notes", None)
            tags = [tag.strip() for tag in profile_tags_widget.text().split(",") if tag.strip()] if profile_tags_widget is not None else []
            self._active_face_service().save_person_profile(
                person.person_name,
                notes=profile_notes_widget.text().strip() if profile_notes_widget is not None else "",
                tags=tags,
                favorite=bool(self.face_profile_favorite.isChecked()) if hasattr(self, "face_profile_favorite") else None,
                birth_date=self.face_profile_birth_date.text().strip() if hasattr(self, "face_profile_birth_date") else None,
                hidden=bool(self.face_profile_hidden.isChecked()) if hasattr(self, "face_profile_hidden") else None,
            )
            self.status_label.setText(f"Saved identity prototype for {person.person_name} from {person.example_count} examples.")
            infoBox("Face label saved", self.status_label.text())
            self.refresh_face_library(refresh_people=True, reason="identity prototype saved")
            self._maybe_refresh_global_face_album(reason="identity prototype saved")

        self._start_job("Saving face label", _run, _done)

    def _auto_propagate_labels(self) -> None:

        def _run(progress, cancel_check):
            progress(-1, "Auto-propagating...")
            _ = cancel_check
            return self._active_face_service().auto_propagate_labels(
                include_tiny_faces=self._show_tiny_detections_enabled(),
            )

        def _done(assignments) -> None:
            self._set_results_kind("labels")
            self._set_pending_face_assignments(list(assignments or []))
            self.status_label.setText(f"Queued {len(assignments)} proposed face labels for review.")
            self._show_face_result_groups(
                [
                    FaceResultGroup(
                        group_id=f"pending:{uuid4()}",
                        title="Pending label proposals",
                        summary=f"Auto-apply queued {len(assignments)} proposal(s). Review them before permanent labels.",
                        items=tuple(
                            FaceTileItem(
                                image_path=str(assignment.image_path),
                                face_index=int(assignment.face_index),
                                bbox=tuple(int(value) for value in assignment.face_bbox),
                                title=f"{assignment.person_name}\n{Path(assignment.image_path).name}",
                                tooltip=(
                                    f"{assignment.image_path}\nface #{int(assignment.face_index) + 1}\n"
                                    f"bbox={tuple(int(value) for value in assignment.face_bbox)}\n"
                                    f"confidence={float(assignment.confidence):.4f}\nsource={assignment.source or 'pending'}"
                                ),
                                status="pending",
                                saved_face_index=int(assignment.face_index),
                                payload=assignment,
                            )
                            for assignment in assignments
                        ),
                    )
                ],
                summary=f"Auto-apply queued {len(assignments)} proposal(s).",
                photo_paths=[assignment.image_path for assignment in assignments],
                overlay_by_path={assignment.image_path: f"pending {assignment.person_name}" for assignment in assignments},
                context_by_path={
                    assignment.image_path: {
                        "face_label": {
                            "person_name": str(assignment.person_name),
                            "confidence": float(assignment.confidence),
                            "bbox": tuple(assignment.face_bbox),
                        }
                    }
                    for assignment in assignments
                },
                kind="labels",
            )

        self._start_job("Auto-propagating labels", _run, _done)

    def _list_face_labels(self) -> None:

        def _run(progress, cancel_check):
            progress(-1, "Loading labels...")
            _ = cancel_check
            return self._active_face_service().list_face_labels(
                include_tiny_faces=self._show_tiny_detections_enabled(),
            )

        def _done(labels) -> None:
            self.status_label.setText(f"Loaded {len(labels)} saved labels.")
            groups: dict[str, list[FaceTileItem]] = {}
            photo_paths: list[str] = []
            overlay_by_path: dict[str, str] = {}
            context_by_path: dict[str, dict[str, object]] = {}
            for assignment in labels:
                photo_paths.append(assignment.image_path)
                overlay_by_path.setdefault(assignment.image_path, str(assignment.person_name))
                context_by_path.setdefault(
                    assignment.image_path,
                    {
                        "face_label": {
                            "person_name": str(assignment.person_name),
                            "confidence": float(assignment.confidence),
                            "bbox": tuple(assignment.face_bbox),
                        }
                    },
                )
                groups.setdefault(str(assignment.person_name), []).append(
                    FaceTileItem(
                        image_path=str(assignment.image_path),
                        face_index=int(assignment.face_index),
                        bbox=tuple(int(value) for value in assignment.face_bbox),
                        title=f"{assignment.person_name}\n{Path(assignment.image_path).name}",
                        tooltip=(
                            f"{assignment.image_path}\nface #{int(assignment.face_index) + 1}\n"
                            f"bbox={tuple(int(value) for value in assignment.face_bbox)}\n"
                            f"confidence={float(assignment.confidence):.4f}"
                        ),
                        status="saved",
                        saved_face_index=int(assignment.face_index),
                        payload=assignment,
                    )
                )
            self._show_face_result_groups(
                [
                    FaceResultGroup(
                        group_id=f"label:{name}",
                        title=name,
                        summary=f"{name} has {len(items)} labeled face(s).",
                        items=tuple(items),
                    )
                    for name, items in sorted(groups.items(), key=lambda item: item[0].lower())
                ],
                summary=f"Loaded {len(labels)} labeled face(s).",
                photo_paths=photo_paths,
                overlay_by_path=overlay_by_path,
                context_by_path=context_by_path,
                kind="labels",
            )

        self._start_job("Loading labels", _run, _done)

    def _set_pending_face_assignments(self, assignments: list[FaceLabelAssignment]) -> None:
        self._pending_face_assignments = list(assignments or [])
        self.face_pending_model.clear()
        if hasattr(self, "face_pending_summary_label"):
            self.face_pending_summary_label.setText(f"Pending review: {len(assignments)} proposal(s).")
        entries: list[ListEntry] = []
        for assignment in assignments:
            label = (
                f"{assignment.person_name} | {Path(assignment.image_path).name} | face={int(assignment.face_index) + 1} | "
                f"conf={float(assignment.confidence):.3f} | source={assignment.source or 'pending'}"
            )
            entries.append(ListEntry(title=label, payload=assignment))
        self.face_pending_model.set_items(entries)
        self._refresh_pending_face_preview()

    def _selected_pending_face_assignments(self) -> list[FaceLabelAssignment]:
        assignments: list[FaceLabelAssignment] = []
        for entry in self._selected_list_entries(self.face_pending_list, self.face_pending_model):
            assignment = entry.payload
            if isinstance(assignment, FaceLabelAssignment) and int(getattr(assignment, "proposal_id", 0) or 0) > 0:
                assignments.append(assignment)
        return assignments

    def _pending_assignments_for_current_cluster(self) -> list[FaceLabelAssignment]:
        group = self._current_face_result_group()
        if group is None:
            fallback_groups = self._face_result_groups_for_kind(self._face_result_active_group_kind)
            if not fallback_groups and self._face_result_active_group_kind == "raw":
                fallback_groups = list(self._face_result_source_groups or [])
            if fallback_groups:
                group = fallback_groups[0]
                group_id = str(group.group_id)
                self._face_result_selected_group_id_by_kind[self._face_result_active_group_kind] = group_id
                self._face_result_selected_group_id = group_id
        if group is None or not group.items:
            return []
        refs = set(self._face_result_group_refs(group, prefer_selection=False))
        if not refs:
            return []
        return [
            assignment
            for assignment in list(self._pending_face_assignments or [])
            if (str(assignment.image_path), int(assignment.face_index)) in refs
        ]

    @staticmethod
    def _pending_assignment_to_tile_item(assignment: FaceLabelAssignment) -> FaceTileItem:
        return FaceTileItem(
            image_path=str(assignment.image_path),
            face_index=int(assignment.face_index),
            bbox=tuple(int(value) for value in assignment.face_bbox),
            title=f"{assignment.person_name}\n{Path(assignment.image_path).name}",
            tooltip=(
                f"{assignment.image_path}\nface #{int(assignment.face_index) + 1}\n"
                f"bbox={tuple(int(value) for value in assignment.face_bbox)}\n"
                f"confidence={float(assignment.confidence):.4f}\nsource={assignment.source or 'pending'}"
            ),
            status="pending",
            saved_face_index=int(assignment.face_index),
            payload=assignment,
        )

    def _refresh_pending_face_preview(self) -> None:
        assignments = self._selected_pending_face_assignments()
        if not hasattr(self, "face_pending_preview_model"):
            return
        if not assignments:
            self.face_pending_preview_model.set_items([])
            if hasattr(self, "face_pending_preview_summary"):
                self.face_pending_preview_summary.setText("Pending preview: select one or more proposals.")
            self._refresh_face_label_audit([])
            return
        items = [self._pending_assignment_to_tile_item(assignment) for assignment in assignments]
        self.face_pending_preview_model.set_items(items)
        names = sorted({str(assignment.person_name) for assignment in assignments if str(assignment.person_name or "").strip()})
        if hasattr(self, "face_pending_preview_summary"):
            self.face_pending_preview_summary.setText(
                f"Pending preview: {len(assignments)} proposal(s) for {', '.join(names) if names else 'selected targets'}."
            )
        self._refresh_face_label_audit(assignments)
        self._schedule_face_tile_refresh()

    def _accept_pending_face_assignments(self, assignments: list[FaceLabelAssignment], *, source_label: str) -> None:
        proposal_ids = [int(assignment.proposal_id) for assignment in assignments if int(getattr(assignment, "proposal_id", 0) or 0) > 0]
        if not proposal_ids:
            errorBox("No pending labels selected", "Select one or more pending face-label proposals first.")
            return

        def _run(progress, cancel_check):
            progress(-1, "Accepting face-label proposals...")
            _ = cancel_check
            return self._active_face_service().accept_pending_face_labels_batch(proposal_ids)

        def _done(batch) -> None:
            accepted = list(getattr(batch, "accepted", ()) or ())
            self._last_pending_acceptance_count = len(accepted)
            self._set_guarded_action_enabled(self.face_pending_undo_button, bool(accepted))
            self.status_label.setText(f"Accepted {len(accepted)} face-label proposal(s) from {source_label}.")
            self.refresh_face_library(reason="pending face labels accepted")
            self._maybe_refresh_global_face_album(reason="pending face labels accepted")
            self._refresh_pending_face_labels()

        self._start_job("Accepting face-label proposals", _run, _done)

    def _reject_pending_face_assignments(self, assignments: list[FaceLabelAssignment], *, source_label: str) -> None:
        proposal_ids = [int(assignment.proposal_id) for assignment in assignments if int(getattr(assignment, "proposal_id", 0) or 0) > 0]
        if not proposal_ids:
            errorBox("No pending labels selected", "Select one or more pending face-label proposals first.")
            return

        def _run(progress, cancel_check):
            progress(-1, "Rejecting face-label proposals...")
            _ = cancel_check
            return self._active_face_service().reject_pending_face_labels(proposal_ids)

        def _done(count) -> None:
            self.status_label.setText(f"Rejected {int(count or 0)} face-label proposal(s) from {source_label}.")
            self._refresh_pending_face_labels()

        self._start_job("Rejecting face-label proposals", _run, _done)

    def _refresh_pending_face_labels(self) -> None:
        folder = self._current_directory() if self.search_only_current_folder.isChecked() else ""
        scope_paths = self._current_scope_paths()

        def _run(progress, cancel_check):
            progress(-1, "Loading pending face labels...")
            _ = cancel_check
            return self._active_face_service().load_pending_face_labels(
                folder_prefix=folder,
                candidate_paths=scope_paths,
                include_tiny_faces=self._show_tiny_detections_enabled(),
            )

        def _done(assignments) -> None:
            self._set_pending_face_assignments(list(assignments or []))
            self.status_label.setText(f"Loaded {len(assignments or [])} pending face-label proposals.")

        self._start_job("Loading pending face labels", _run, _done)

    def _accept_pending_face_labels(self) -> None:
        self._accept_pending_face_assignments(self._selected_pending_face_assignments(), source_label="selection")

    def _reject_pending_face_labels(self) -> None:
        self._reject_pending_face_assignments(self._selected_pending_face_assignments(), source_label="selection")

    def _accept_pending_face_labels_above_threshold(self) -> None:
        threshold = float(self.face_pending_accept_threshold.value()) if hasattr(self, "face_pending_accept_threshold") else 0.85
        assignments = [
            assignment
            for assignment in list(self._pending_face_assignments or [])
            if float(assignment.confidence) >= threshold
        ]
        if not assignments:
            errorBox("No proposals above threshold", "No pending proposals meet the current acceptance threshold.")
            return
        self._accept_pending_face_assignments(assignments, source_label=f"threshold {threshold:.2f}")

    def _accept_current_cluster_pending_face_labels(self) -> None:
        assignments = self._pending_assignments_for_current_cluster()
        if not assignments:
            errorBox("No cluster proposals", "The current face cluster has no pending label proposals in this scope.")
            return
        self._accept_pending_face_assignments(assignments, source_label="current cluster")

    def _reject_current_cluster_pending_face_labels(self) -> None:
        assignments = self._pending_assignments_for_current_cluster()
        if not assignments:
            errorBox("No cluster proposals", "The current face cluster has no pending label proposals in this scope.")
            return
        self._reject_pending_face_assignments(assignments, source_label="current cluster")

    def _undo_last_pending_face_acceptance(self) -> None:
        def _run(progress, cancel_check):
            progress(-1, "Undoing face-label acceptance...")
            _ = cancel_check
            return self._active_face_service().undo_last_pending_face_acceptance()

        def _done(restored) -> None:
            count = len(list(restored or []))
            self._last_pending_acceptance_count = 0
            self.face_pending_undo_button.setEnabled(False)
            self.status_label.setText(f"Restored {count} proposal(s) from the last accepted batch.")
            self.refresh_face_library(reason="pending face labels undone")
            self._maybe_refresh_global_face_album(reason="pending face labels undone")
            self._refresh_pending_face_labels()

        self._start_job("Undoing face-label acceptance", _run, _done)

    def _merge_people(self) -> None:

        source = self.merge_source_person.text().strip()
        target = self.merge_target_person.text().strip()
        if not source or not target:
            errorBox("Missing names", "Enter both the merge source and merge target names first.")
            return
        if source == target:
            errorBox("Invalid merge", "Merge source and target must be different names.")
            return
        if not confirmBox(
            "Merge two names",
            f"Move all labels from '{source}' into '{target}'?\n\nThis updates saved face labels immediately.",
            parent=self,
        ):
            return

        def _run(progress, cancel_check):
            progress(-1, "Merging labels...")
            _ = cancel_check
            self._active_face_service().merge_person_labels(source, target)
            return True

        def _done(_ok: bool) -> None:
            self.refresh_face_library(reason="identities merged")
            self._maybe_refresh_global_face_album(reason="identities merged")
            self._list_face_labels()

        self._start_job("Merging labels", _run, _done)

    def _clear_person(self) -> None:

        person_name = self.person_name.text().strip()
        if not person_name:
            errorBox("Missing name", "Enter the saved identity name you want to remove first.")
            return
        if not confirmBox(
            "Remove name from faces",
            f"Remove saved face labels for '{person_name}'?\n\nThe original images stay in place.",
            parent=self,
        ):
            return

        def _run(progress, cancel_check):
            progress(-1, "Clearing person...")
            _ = cancel_check
            self._active_face_service().clear_person_labels(person_name)
            return True

        def _done(_ok: bool) -> None:
            self.refresh_face_library(reason="identity labels cleared")
            self._maybe_refresh_global_face_album(reason="identity labels cleared")
            self._list_face_labels()

        self._start_job("Clearing person", _run, _done)






