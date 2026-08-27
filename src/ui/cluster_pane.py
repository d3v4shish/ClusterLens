from __future__ import annotations

import html
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QAbstractTableModel, QEvent, QModelIndex, QPoint, QRect, QSize, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.selection import SelectionTarget
from app.services.cluster_explanations import ClusterExplanation
from app.services.cluster_meanings import ClusterMeaning
from app.services.image_tags import ClusterTagSummary
from app.services.thumbnails import ThumbnailService
from .async_job import AsyncJob, raise_if_cancelled, start_job_in_thread, wait_for_thread_shutdown
from .common import build_help_inline
from .theme import COLORS


CLUSTER_PANE_HELP = {
    "cluster_comparisons": (
        "Compare the current clustering result across enabled models and backends.\n"
        "Each column is one comparison, and each cell is a cluster from that result.\n"
        "Selecting a cluster updates the preview, current gallery group, and membership details.\n"
        "Hover a cluster cell to see a small visual mosaic plus the derived tag summary."
    ),
    "preview": (
        "Preview the selected cluster with a short filename list.\n"
        "Selecting a cluster also makes it the current group for gallery actions.\n"
        "If no images are checked in the gallery, actions usually use this current group."
    ),
    "selected_cluster_summary": (
        "Show derived tag totals for the selected cluster.\n"
        "These counts come from image-level tags, not from a cluster annotation.\n"
        "Tag edits refresh this summary against the cluster's current image membership."
    ),
    "cluster_basis": (
        "Explain why the selected cluster grouped together.\n"
        "These values come from the prepared clustering matrix, centroid similarity, backend quality, and tag evidence.\n"
        "Use this in advanced mode when you want to inspect cluster behavior beyond the preview list."
    ),
    "cluster_shape": (
        "Visualize why the selected cluster is tight or mixed.\n"
        "Each bar is one centroid-ranked representative image, where taller bars are closer to the cluster center.\n"
        "This is a compact view of embedding-space cohesion, not the raw embedding vector."
    ),
    "cluster_meaning": (
        "Translate the selected cluster into human-readable model-derived labels.\n"
        "This uses CLIP/SigLIP-style text-image similarity plus observed tags or filename terms when available.\n"
        "Treat this as useful evidence, not ground truth."
    ),
    "membership": (
        "Show how the selected image maps across clustering comparisons.\n"
        "Each row reports cluster id, cluster size, rank, outlier status, and quality.\n"
        "Use this to compare where one image lands across different runs."
    ),
    "recluster": (
        "Run clustering again using only the selected cluster's images.\n"
        "It reuses the main clustering settings from the left controls.\n"
        "The source folder is not rediscovered for this run.\n"
        "Small subsets can still fail if Clusters is higher than the selected image count."
    ),
}

HOVER_PREVIEW_IMAGE_SIZE = QSize(216, 216)
HOVER_PREVIEW_MAX_ITEMS = 9
HOVER_PREVIEW_CACHE_SIZE = 24
CLUSTER_BASIS_EMPTY_TEXT = "Select a cluster to inspect why it grouped this way."
CLUSTER_MEANING_EMPTY_TEXT = "Select a cluster to inspect what this group may be about."
CLUSTER_SHAPE_EMPTY_TEXT = "Select a cluster to see representative closeness to the cluster center."
TEXT_LABEL_MODELS = {"clip", "openclip", "siglip"}
MODEL_DISPLAY_NAMES = {
    "model": "model",
    "fast_preview": "Fast Preview",
    "mobileclip": "MobileCLIP S0",
    "clip": "CLIP",
    "openclip": "OpenCLIP",
    "siglip": "SigLIP",
    "dino": "DINO",
    "dinov2_base": "DINOv2 Base",
    "dino_large": "DINO Large",
    "resnet": "ResNet",
    "convnext": "ConvNeXt",
    "vit": "ViT",
    "phash_embedding": "pHash Embedding",
}


@dataclass(frozen=True)
class ClusterCell:
    comparison_key: str
    cluster_id: int
    images: tuple[str, ...]


def _cluster_summary_lines(cell: ClusterCell, summary: ClusterTagSummary | None) -> list[str]:
    lines = [
        str(cell.comparison_key),
        f"Cluster {cell.cluster_id}",
        f"{len(cell.images)} images",
    ]
    if summary is not None and summary.top_tags:
        top_tags = ", ".join(f"{tag} ({count})" for tag, count in summary.top_tags)
        lines.append(f"Top tags: {top_tags}")
        lines.append(f"Tagged images: {summary.tagged_image_count}/{len(cell.images)}")
    elif summary is not None:
        lines.append("No tags in this cluster.")
    return lines


def _selected_cluster_summary_text(images: tuple[str, ...] | list[str], summary: ClusterTagSummary | None) -> str:
    if summary is None or not summary.top_tags:
        return "No tags in this cluster."
    top_tags = ", ".join(f"{tag} ({count})" for tag, count in summary.top_tags)
    return f"Top tags: {top_tags} | Tagged images: {summary.tagged_image_count}/{len(images)} | Unique tags: {summary.unique_tag_count}"


def _format_score(value: float | None) -> str:
    if value is None:
        return "Unavailable"
    return f"{value:.3f}"


def _cluster_meaning_text(
    meaning: ClusterMeaning | None,
    summary: ClusterTagSummary | None,
    explanation: ClusterExplanation | None,
    comparison_key: str = "",
) -> str:
    if meaning is None:
        return "\n".join(
            [
                f"Cluster Verdict: {_cluster_verdict_text(None, explanation)}",
                "Likely About: Unavailable",
                f"Why Clustered: {_why_clustered_text(comparison_key, explanation)}",
                "Why Named: Text naming unavailable; the app cannot name this cluster from text-model evidence.",
                f"Review: {_review_text(explanation)}",
                f"Suggested Action: {_suggested_action_text(explanation, None)}",
                f"Source: {_meaning_source_text(comparison_key, None)}",
                f"Observed Evidence: {_observed_evidence_text(None, summary)}",
                "Caution: Model-derived label, not ground truth. Review before file operations.",
            ]
        )
    if meaning.status != "ok" or not meaning.labels:
        message = meaning.message or "The app grouped these visually, but cannot name this cluster from text-model evidence."
        return "\n".join(
            [
                f"Cluster Verdict: {_cluster_verdict_text(meaning, explanation)}",
                "Likely About: Unavailable",
                "Confidence: Unavailable",
                f"Why Clustered: {_why_clustered_text(comparison_key, explanation)}",
                "Why Named: Text naming unavailable; the app cannot name this cluster from text-model evidence.",
                f"Review: {_review_text(explanation)}",
                f"Suggested Action: {_suggested_action_text(explanation, meaning)}",
                f"Source: {_meaning_source_text(comparison_key, meaning)}",
                f"Model Evidence: {message}",
                f"Observed Evidence: {_observed_evidence_text(meaning, summary)}",
                f"Reliability: {_reliability_text(explanation)}",
                "Caution: Model-derived label, not ground truth. Review before file operations.",
            ]
        )

    likely_theme = " / ".join(label.label for label in meaning.labels[:3])
    top_scores = ", ".join(f"{label.label} {_format_score(label.score)}" for label in meaning.labels[:3])
    model_name = _display_model_name(meaning.explanation_model)
    return "\n".join(
        [
            f"Cluster Verdict: {_cluster_verdict_text(meaning, explanation)}",
            f"Likely About: {likely_theme}",
            f"Confidence: {meaning.confidence}",
            f"Why Clustered: {_why_clustered_text(comparison_key, explanation)}",
            f"Why Named: {_why_named_text(meaning, summary)}",
            f"Review: {_review_text(explanation)}",
            f"Suggested Action: {_suggested_action_text(explanation, meaning)}",
            f"Source: {_meaning_source_text(comparison_key, meaning)}",
            f"Model Evidence: {model_name} | {meaning.image_count_used} representative image(s) | {top_scores}",
            f"Observed Evidence: {_observed_evidence_text(meaning, summary)}",
            f"Reliability: {_reliability_text(explanation)}",
            "Caution: Model-derived label, not ground truth. Review before file operations.",
        ]
    )


def _cluster_verdict_text(meaning: ClusterMeaning | None, explanation: ClusterExplanation | None) -> str:
    if explanation is not None and explanation.is_outlier:
        return "Outlier/noise group; inspect manually."
    reliability = _reliability_level(explanation)
    confidence = str(meaning.confidence if meaning is not None else "").lower()
    has_labels = bool(meaning is not None and meaning.status == "ok" and meaning.labels)
    if reliability == "high" and has_labels and confidence in {"high", "medium"}:
        return "Strong visual theme."
    if reliability == "high":
        return "Strong visual grouping; label is uncertain."
    if reliability == "medium":
        return "Usable cluster; review nearby images."
    if reliability == "unavailable":
        return "Cluster explanation is still loading or unavailable."
    return "Weak or mixed cluster; review manually."


def _why_clustered_text(comparison_key: str, explanation: ClusterExplanation | None) -> str:
    model_name = _display_model_name(_comparison_model_key(comparison_key))
    if explanation is None:
        return f"{model_name} produced high-dimensional image feature vectors; numeric basis is unavailable."
    if explanation.is_outlier:
        return f"{model_name} did not find a stable centroid group here, so this is treated as outlier/noise."

    parts = [f"{model_name} image-feature vectors are close to this cluster center"]
    if explanation.cohesion_mean is not None:
        parts.append(f"mean cohesion {_format_score(explanation.cohesion_mean)}")
    if explanation.cohesion_min is not None:
        parts.append(f"weakest member {_format_score(explanation.cohesion_min)}")
    if explanation.separation_margin is not None:
        parts.append(f"separation margin {_format_score(explanation.separation_margin)}")
    if explanation.nearest_cluster_id is not None:
        parts.append(f"nearest competing cluster {explanation.nearest_cluster_id}")
    parts.append("preview images are ordered by closeness to the cluster center")
    return "; ".join(parts) + "."


def _why_named_text(
    meaning: ClusterMeaning,
    summary: ClusterTagSummary | None,
) -> str:
    parts = [f"{_display_model_name(meaning.explanation_model)} matched representative images to text prompts"]
    if summary is not None and summary.top_tags:
        parts.append("tags include " + ", ".join(f"{tag} ({count})" for tag, count in summary.top_tags[:3]))
    if meaning.filename_terms:
        parts.append("filenames include " + ", ".join(f"{term} ({count})" for term, count in meaning.filename_terms[:3]))
    return "; ".join(parts) + "."


def _review_text(explanation: ClusterExplanation | None) -> str:
    if explanation is None:
        return "Review the preview images; numeric cluster basis is unavailable."
    if explanation.is_outlier:
        return "This is an outlier/noise group, so inspect images one by one."
    if explanation.nearest_cluster_id is not None:
        margin = explanation.separation_margin
        if margin is not None and margin < 0.05:
            return f"Check against cluster {explanation.nearest_cluster_id}; separation is weak (margin {_format_score(margin)})."
        return f"Closest alternative is cluster {explanation.nearest_cluster_id}; margin {_format_score(margin)}."
    if _reliability_level(explanation) == "low":
        return "Review lowest-ranked preview images before using this cluster."
    return "Quickly review the preview list, especially before destructive operations."


def _suggested_action_text(explanation: ClusterExplanation | None, meaning: ClusterMeaning | None) -> str:
    reliability = _reliability_level(explanation)
    confidence = str(meaning.confidence if meaning is not None else "").lower()
    if explanation is not None and explanation.is_outlier:
        return "Avoid bulk operations; inspect images individually."
    if reliability == "high" and confidence in {"high", "medium"}:
        return "Safe to move/copy as a group; review before delete."
    if reliability in {"high", "medium"}:
        return "Move/copy after a quick visual review; review carefully before delete."
    return "Use as a starting point only; avoid bulk delete or move without manual checks."


def _meaning_source_text(comparison_key: str, meaning: ClusterMeaning | None) -> str:
    cluster_model_key = _comparison_model_key(comparison_key)
    cluster_model = _display_model_name(cluster_model_key)
    if meaning is None or meaning.status != "ok" or not meaning.labels:
        model_name = _display_model_name(meaning.explanation_model) if meaning is not None and meaning.explanation_model else ""
        if model_name:
            return f"Clustered by {cluster_model}; text naming unavailable ({model_name} sidecar)."
        return f"Clustered by {cluster_model}; text naming unavailable."

    naming_model_key = str(meaning.explanation_model or "").strip().lower()
    naming_model = _display_model_name(naming_model_key)
    if cluster_model_key in TEXT_LABEL_MODELS and naming_model_key == cluster_model_key:
        return f"Clustered by {cluster_model}; Named by {naming_model} native text-image model."
    return f"Clustered by {cluster_model}; Named by {naming_model} sidecar."


def _comparison_model_key(comparison_key: str) -> str:
    parts = [part for part in str(comparison_key or "").split("::") if part]
    if not parts:
        return "model"
    return parts[0].strip().lower()


def _display_model_name(model_name: str) -> str:
    normalized = str(model_name or "").strip().lower()
    return MODEL_DISPLAY_NAMES.get(normalized, normalized.upper() if normalized else "model")


def _reliability_level(explanation: ClusterExplanation | None) -> str:
    if explanation is None:
        return "unavailable"
    if explanation.is_outlier:
        return "low"
    margin = explanation.separation_margin
    cohesion = explanation.cohesion_mean
    if margin is not None and margin >= 0.20 and cohesion is not None and cohesion >= 0.80:
        return "high"
    if margin is not None and margin >= 0.05:
        return "medium"
    return "low"


def _observed_evidence_text(meaning: ClusterMeaning | None, summary: ClusterTagSummary | None) -> str:
    parts: list[str] = []
    if summary is not None and summary.top_tags:
        parts.append("Tags: " + ", ".join(f"{tag} ({count})" for tag, count in summary.top_tags))
    if meaning is not None and meaning.filename_terms:
        parts.append("Filenames: " + ", ".join(f"{term} ({count})" for term, count in meaning.filename_terms))
    if not parts:
        return "No tags or useful filename terms available."
    return " | ".join(parts)


def _reliability_text(explanation: ClusterExplanation | None) -> str:
    level = _reliability_level(explanation)
    if level == "unavailable":
        return "Unavailable until cluster basis is loaded."
    if explanation is not None and explanation.is_outlier:
        return "Low: outlier/noise cluster."
    if level == "high":
        return "High: tight cluster with clear separation."
    if level == "medium":
        return "Medium: usable grouping, but review nearby clusters."
    return "Low: weak separation or mixed membership; review manually."


def _cluster_basis_text(
    explanation: ClusterExplanation | None,
    summary: ClusterTagSummary | None,
) -> str:
    if explanation is None:
        return "Cluster basis unavailable for this cluster."

    if summary is not None and summary.top_tags:
        tag_evidence = (
            f"Top tags: {', '.join(f'{tag} ({count})' for tag, count in summary.top_tags)} | "
            f"Tagged {summary.tagged_image_count}/{explanation.cluster_size} | "
            f"Unique tags: {summary.unique_tag_count}"
        )
    elif summary is not None:
        tag_evidence = "No tags in this cluster."
    else:
        tag_evidence = "Tag summary not loaded yet."

    lines = [
        f"Size: {explanation.cluster_size} image(s)",
        f"Quality: {_format_score(explanation.cluster_quality_score)}",
    ]
    if explanation.similarity_space:
        lines.append(f"Similarity Space: {explanation.similarity_space}")
    if explanation.is_outlier:
        lines.append("Cohesion: Outlier/noise cluster; centroid cohesion is not reported.")
        lines.append("Nearest Competing Cluster: Unavailable")
        lines.append("Margin: Unavailable")
    else:
        lines.append(
            "Cohesion: "
            f"mean {_format_score(explanation.cohesion_mean)} | "
            f"median {_format_score(explanation.cohesion_median)} | "
            f"min {_format_score(explanation.cohesion_min)} | "
            f"max {_format_score(explanation.cohesion_max)}"
        )
        if explanation.nearest_cluster_id is None:
            lines.append("Nearest Competing Cluster: Unavailable")
        else:
            lines.append(
                "Nearest Competing Cluster: "
                f"{explanation.nearest_cluster_id} @ {_format_score(explanation.nearest_cluster_similarity)}"
            )
        lines.append(f"Margin: {_format_score(explanation.separation_margin)}")
    lines.append(f"Representative Images: {explanation.representative_images_note}")
    lines.append(f"Tag Evidence: {tag_evidence}")
    return "\n".join(lines)


def _similarity_difference_text(
    comparison_key: str,
    cluster_id: int,
    clusters_by_backend: dict[str, dict[int, list[str]]],
) -> str:
    key_parts = str(comparison_key).split("::")
    if len(key_parts) != 3 or key_parts[1] not in {"semantic", "cosine"}:
        return ""

    model_name, similarity_mode, backend = key_parts
    sibling_mode = "cosine" if similarity_mode == "semantic" else "semantic"
    sibling_key = f"{model_name}::{sibling_mode}::{backend}"
    sibling_clusters = clusters_by_backend.get(sibling_key)
    if not sibling_clusters:
        return f"Similarity Difference: No sibling {sibling_mode} column in this run."

    selected_images = set(str(path) for path in clusters_by_backend.get(comparison_key, {}).get(cluster_id, []))
    if not selected_images:
        return "Similarity Difference: Selected cluster has no images to compare."

    best_cluster_id = None
    best_cluster_images: set[str] = set()
    best_score = -1.0
    for candidate_cluster_id, candidate_images in sibling_clusters.items():
        candidate_set = set(str(path) for path in candidate_images)
        union_count = len(selected_images | candidate_set)
        overlap_score = (len(selected_images & candidate_set) / union_count) if union_count else 0.0
        if overlap_score > best_score:
            best_score = overlap_score
            best_cluster_id = int(candidate_cluster_id)
            best_cluster_images = candidate_set

    if best_cluster_id is None:
        return f"Similarity Difference: Sibling {sibling_key} has no comparable clusters."

    shared_count = len(selected_images & best_cluster_images)
    moved_out_count = len(selected_images - best_cluster_images)
    moved_in_count = len(best_cluster_images - selected_images)
    return "\n".join(
        [
            f"Similarity Difference: compared with {sibling_key}",
            (
                "Best Match: "
                f"Cluster {best_cluster_id} | overlap {best_score:.3f} "
                f"({shared_count} shared / {len(selected_images)} selected / {len(best_cluster_images)} sibling)"
            ),
            f"Moved Out: {moved_out_count} image(s)",
            f"Moved In: {moved_in_count} image(s)",
        ]
    )


class ClusterHoverPreviewPopup(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("clusterHoverPreviewPopup")
        self.setStyleSheet(
            f"""
            QFrame#clusterHoverPreviewPopup {{
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
        title_font.setPointSize(max(title_font.pointSize(), 10))
        self.title_label.setFont(title_font)
        layout.addWidget(self.title_label)

        self.image_label = QLabel("Loading preview...")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setFixedSize(HOVER_PREVIEW_IMAGE_SIZE)
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

    def set_cluster_details(
        self,
        cell: ClusterCell,
        summary_lines: list[str],
        *,
        more_count: int,
        loading: bool,
    ) -> None:
        title = f"{html.escape(str(cell.comparison_key))} | Cluster {cell.cluster_id}"
        body_lines = [html.escape(line) for line in summary_lines[2:]]
        self.title_label.setText(title)
        self.summary_label.setText("<br>".join(body_lines))
        self.note_label.setText(f"+{more_count} more" if more_count > 0 else "")
        if loading:
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Loading preview...")

    def set_preview_image(self, image: QImage | None) -> None:
        if image is None or image.isNull():
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText("Preview unavailable")
            return
        pixmap = QPixmap.fromImage(image)
        self.image_label.setText("")
        self.image_label.setPixmap(pixmap)


class ClusterShapeWidget(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._title = CLUSTER_SHAPE_EMPTY_TEXT
        self._details = ""
        self._scores: tuple[float, ...] = ()
        self.setMinimumHeight(104)
        self.setToolTip(CLUSTER_PANE_HELP["cluster_shape"])

    def set_explanation(self, explanation: ClusterExplanation | None, comparison_key: str) -> None:
        model_name = _display_model_name(_comparison_model_key(comparison_key))
        if explanation is None:
            self._title = CLUSTER_SHAPE_EMPTY_TEXT
            self._details = "No numeric cluster basis has been loaded yet."
            self._scores = ()
            self.update()
            return
        if explanation.is_outlier:
            self._title = f"{model_name} cluster shape: outlier/noise"
            self._details = "No centroid-closeness bars are shown for outlier groups."
            self._scores = ()
            self.update()
            return

        self._scores = tuple(float(score) for score in explanation.member_cohesion_scores[:24])
        if not self._scores:
            self._title = f"{model_name} cluster shape unavailable"
            self._details = "Rerun clustering to capture representative member closeness scores."
            self.update()
            return

        self._title = f"{model_name} cluster shape: representative closeness to center"
        self._details = (
            f"mean {_format_score(explanation.cohesion_mean)} | "
            f"weakest {_format_score(explanation.cohesion_min)} | "
            f"margin {_format_score(explanation.separation_margin)}"
        )
        self.update()

    def clear(self) -> None:
        self._title = CLUSTER_SHAPE_EMPTY_TEXT
        self._details = ""
        self._scores = ()
        self.update()

    def paintEvent(self, event) -> None:
        _ = event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = self.rect().adjusted(1, 1, -2, -2)
        painter.fillRect(rect, QColor(COLORS["surface_raised"]))
        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        painter.drawRoundedRect(rect, 6, 6)

        text_rect = QRect(rect.left() + 10, rect.top() + 8, rect.width() - 20, 18)
        painter.setPen(QColor(COLORS["text"]))
        title_font = QFont(self.font())
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._title)

        detail_rect = QRect(rect.left() + 10, rect.top() + 28, rect.width() - 20, 18)
        painter.setFont(self.font())
        painter.setPen(QColor(COLORS["text_muted"]))
        painter.drawText(detail_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._details)

        chart_rect = QRect(rect.left() + 10, rect.top() + 52, rect.width() - 20, max(34, rect.height() - 64))
        if not self._scores:
            painter.setPen(QColor(COLORS["text_muted"]))
            painter.drawText(chart_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, self._details or CLUSTER_SHAPE_EMPTY_TEXT)
            painter.end()
            return

        painter.setPen(QPen(QColor(COLORS["border"]), 1))
        baseline_y = chart_rect.bottom() - 2
        painter.drawLine(chart_rect.left(), baseline_y, chart_rect.right(), baseline_y)

        count = len(self._scores)
        gap = 2
        bar_width = max(3, int((chart_rect.width() - (count - 1) * gap) / max(1, count)))
        for index, score in enumerate(self._scores):
            height_ratio = max(0.03, min(1.0, (float(score) + 1.0) / 2.0))
            bar_height = max(2, int((chart_rect.height() - 6) * height_ratio))
            x = chart_rect.left() + index * (bar_width + gap)
            y = baseline_y - bar_height
            color = (
                QColor(COLORS["success"])
                if score >= 0.80
                else QColor(COLORS["warning"])
                if score >= 0.60
                else QColor(COLORS["danger"])
            )
            painter.fillRect(QRect(x, y, bar_width, bar_height), color)

        painter.end()


class ClusterGridModel(QAbstractTableModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._columns: list[str] = []
        self._rows: list[list[ClusterCell | None]] = []
        self._cluster_summaries: dict[tuple[str, int], ClusterTagSummary] = {}
        self._highlighted: set[tuple[str, int]] = set()
        self._selected: tuple[str, int] | None = None

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._columns)

    def summary_for_cell(self, cell: ClusterCell) -> ClusterTagSummary | None:
        return self._cluster_summaries.get((str(cell.comparison_key), int(cell.cluster_id)))

    def summary_lines_for_cell(self, cell: ClusterCell) -> list[str]:
        return _cluster_summary_lines(cell, self.summary_for_cell(cell))

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self._rows) or index.column() >= len(self._columns):
            return None
        cell = self._rows[index.row()][index.column()]
        if cell is None:
            return None
        key = (cell.comparison_key, int(cell.cluster_id))
        if role == Qt.ItemDataRole.DisplayRole:
            return f"Cluster {cell.cluster_id}\n{len(cell.images)} images"
        if role == Qt.ItemDataRole.UserRole:
            return key
        if role == Qt.ItemDataRole.ToolTipRole:
            return "\n".join(self.summary_lines_for_cell(cell))
        if role == Qt.ItemDataRole.BackgroundRole and key in self._highlighted:
            return QColor("#FFF3B0")
        if role == Qt.ItemDataRole.FontRole and key in self._highlighted:
            font = QFont()
            font.setBold(True)
            return font
        if role == Qt.ItemDataRole.BackgroundRole and self._selected == key:
            return QColor("#D9ECFF")
        return None

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal and 0 <= section < len(self._columns):
            return self._columns[section]
        return str(section + 1)

    def update_data(
        self,
        clusters_by_backend: dict[str, dict[int, list[str]]],
        cluster_summaries: dict[str, dict[int, ClusterTagSummary]] | None = None,
    ) -> None:
        self.beginResetModel()
        self._columns = sorted(clusters_by_backend.keys())
        self._cluster_summaries = {
            (str(comparison_key), int(cluster_id)): summary
            for comparison_key, summaries in (cluster_summaries or {}).items()
            for cluster_id, summary in summaries.items()
        }
        max_rows = max((len(clusters) for clusters in clusters_by_backend.values()), default=0)
        self._rows = []
        sorted_clusters = {
            key: [ClusterCell(key, int(cluster_id), tuple(images)) for cluster_id, images in sorted(clusters.items(), key=lambda item: item[0])]
            for key, clusters in clusters_by_backend.items()
        }
        for row in range(max_rows):
            row_cells: list[ClusterCell | None] = []
            for key in self._columns:
                cells = sorted_clusters.get(key, [])
                row_cells.append(cells[row] if row < len(cells) else None)
            self._rows.append(row_cells)
        self._highlighted.clear()
        self._selected = None
        self.endResetModel()

    def set_highlighted(self, highlighted: set[tuple[str, int]]) -> None:
        self._highlighted = set(highlighted)
        if self.rowCount() and self.columnCount():
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1))

    def set_selected(self, selection: tuple[str, int] | None) -> None:
        self._selected = selection
        if self.rowCount() and self.columnCount():
            self.dataChanged.emit(self.index(0, 0), self.index(self.rowCount() - 1, self.columnCount() - 1))

    def cell_at(self, index: QModelIndex) -> ClusterCell | None:
        if not index.isValid() or index.row() >= len(self._rows) or index.column() >= len(self._columns):
            return None
        return self._rows[index.row()][index.column()]


class ClusterPane(QWidget):
    cluster_selected = pyqtSignal(str, int)
    recluster_requested = pyqtSignal()
    selection_target_changed = pyqtSignal(object)
    hide_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.prev_selection: tuple[str, int] | None = None
        self._basic_mode = False
        self._clusters_by_backend: dict[str, dict[int, list[str]]] = {}
        self._membership_by_image: dict[str, dict[str, dict[str, object]]] = {}
        self._metrics_by_backend: dict[str, dict[str, object]] = {}
        self._cluster_summaries: dict[str, dict[int, ClusterTagSummary]] = {}
        self._cluster_explanations: dict[str, dict[int, ClusterExplanation]] = {}
        self._cluster_meanings: dict[str, dict[int, ClusterMeaning]] = {}
        self._grid_model = ClusterGridModel(self)
        self._hover_popup = ClusterHoverPreviewPopup(self)
        self._hover_popup.hide()
        self._hover_popup_key: tuple[str, int] | None = None
        self._hover_popup_index: QModelIndex = QModelIndex()
        self._hover_preview_generation = 0
        self._hover_preview_job = None
        self._hover_preview_thread = None
        self._thread_jobs: dict[object, object | None] = {}
        self._retained_async_refs: list[tuple[object | None, object | None]] = []
        self._hover_preview_cache: OrderedDict[tuple[object, ...], QImage] = OrderedDict()
        self.init_ui()
        self._install_hover_preview()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        def add_centered_help_row(text: str, tooltip_text: str, help_key: str, target_layout: QVBoxLayout | None = None) -> QLabel:
            _ = help_key
            target = target_layout or layout
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(0)
            label = QLabel(text)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setStyleSheet("font-size: 13px; font-weight: bold;")
            label.setToolTip(tooltip_text)
            row.addWidget(label)
            target.addLayout(row)
            return label

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(6)

        title_label = QLabel("Cluster Comparisons")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        title_label.setToolTip(CLUSTER_PANE_HELP["cluster_comparisons"])
        self.title_label = title_label
        self.recluster_button = QPushButton("Recluster")
        self.recluster_button.setEnabled(False)
        self.recluster_button.clicked.connect(self.recluster_requested.emit)
        self.hide_button = QPushButton("Hide")
        self.hide_button.setProperty("paneToggle", True)
        self.hide_button.setFixedHeight(24)
        self.hide_button.clicked.connect(self.hide_requested.emit)
        header_row.addWidget(title_label, stretch=1)
        header_row.addWidget(build_help_inline(self.recluster_button, CLUSTER_PANE_HELP["recluster"], help_key="recluster"))
        header_row.addWidget(self.hide_button)
        layout.addLayout(header_row)

        self.cluster_table = QTableView()
        self.cluster_table.setModel(self._grid_model)
        self.cluster_table.horizontalHeader().setStretchLastSection(True)
        self.cluster_table.verticalHeader().hide()
        self.cluster_table.clicked.connect(self.on_cluster_selected)
        self.cluster_table.setToolTip(CLUSTER_PANE_HELP["cluster_comparisons"])
        self.cluster_table.setMinimumHeight(360)
        self.cluster_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.cluster_table, stretch=6)

        self.details_scroll = QScrollArea()
        self.details_scroll.setWidgetResizable(True)
        self.details_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.details_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.details_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.details_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.details_widget = QWidget()
        self.details_layout = QVBoxLayout(self.details_widget)
        self.details_layout.setContentsMargins(0, 0, 0, 0)
        self.details_layout.setSpacing(8)
        self.details_scroll.setWidget(self.details_widget)
        layout.addWidget(self.details_scroll, stretch=4)

        self.preview_label = add_centered_help_row("Preview", CLUSTER_PANE_HELP["preview"], "preview", self.details_layout)

        self.summary_label = QLabel("No cluster selected.")
        self.summary_label.setWordWrap(True)
        self.summary_label.setToolTip(CLUSTER_PANE_HELP["selected_cluster_summary"])
        self.summary_section_label = add_centered_help_row(
            "Selected Cluster Summary",
            CLUSTER_PANE_HELP["selected_cluster_summary"],
            "selected_cluster_summary",
            self.details_layout,
        )
        self.details_layout.addWidget(self.summary_label)

        self.meaning_section_label = add_centered_help_row(
            "Cluster Meaning",
            CLUSTER_PANE_HELP["cluster_meaning"],
            "cluster_meaning",
            self.details_layout,
        )
        self.meaning_label = QLabel(CLUSTER_MEANING_EMPTY_TEXT)
        self.meaning_label.setWordWrap(True)
        self.meaning_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.meaning_label.setToolTip(CLUSTER_PANE_HELP["cluster_meaning"])
        self.details_layout.addWidget(self.meaning_label)

        self.shape_section_label = add_centered_help_row(
            "Cluster Shape",
            CLUSTER_PANE_HELP["cluster_shape"],
            "cluster_shape",
            self.details_layout,
        )
        self.shape_widget = ClusterShapeWidget()
        self.details_layout.addWidget(self.shape_widget)

        self.basis_section_label = add_centered_help_row(
            "Cluster Basis",
            CLUSTER_PANE_HELP["cluster_basis"],
            "cluster_basis",
            self.details_layout,
        )
        self.basis_label = QLabel(CLUSTER_BASIS_EMPTY_TEXT)
        self.basis_label.setWordWrap(True)
        self.basis_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.basis_label.setToolTip(CLUSTER_PANE_HELP["cluster_basis"])
        self.details_layout.addWidget(self.basis_label)

        self.preview_list = QListWidget()
        self.preview_list.setToolTip(CLUSTER_PANE_HELP["preview"])
        self.preview_list.setMaximumHeight(140)
        self.details_layout.addWidget(self.preview_list)

        self.membership_section_label = add_centered_help_row(
            "Selected Image Membership",
            CLUSTER_PANE_HELP["membership"],
            "membership",
            self.details_layout,
        )

        self.membership_table = QTableWidget()
        self.membership_table.setColumnCount(7)
        self.membership_table.setHorizontalHeaderLabels([
            "Embedding",
            "Backend",
            "Cluster ID",
            "Cluster Size",
            "Rank",
            "Outlier",
            "Quality",
        ])
        self.membership_table.setToolTip(CLUSTER_PANE_HELP["membership"])
        self.membership_table.setMinimumHeight(160)
        self.details_layout.addWidget(self.membership_table)
        self.details_layout.addStretch(1)

    def _install_hover_preview(self) -> None:
        viewport = self.cluster_table.viewport()
        viewport.setMouseTracking(True)
        viewport.setAttribute(Qt.WidgetAttribute.WA_Hover, True)
        viewport.installEventFilter(self)
        self.cluster_table.setMouseTracking(True)
        self.cluster_table.verticalScrollBar().valueChanged.connect(lambda _value: self._hide_hover_popup(cancel_preview=True))
        self.cluster_table.horizontalScrollBar().valueChanged.connect(lambda _value: self._hide_hover_popup(cancel_preview=True))
        self._grid_model.modelReset.connect(lambda: self._hide_hover_popup(cancel_preview=True))

    def eventFilter(self, watched, event):
        if watched is self.cluster_table.viewport():
            if event.type() == QEvent.Type.ToolTip:
                return True
            if event.type() == QEvent.Type.MouseMove:
                self._update_hover_popup_for_index(self.cluster_table.indexAt(event.pos()))
            elif event.type() in {
                QEvent.Type.Leave,
                QEvent.Type.Hide,
                QEvent.Type.Wheel,
                QEvent.Type.MouseButtonPress,
            }:
                self._hide_hover_popup(cancel_preview=True)
        return super().eventFilter(watched, event)

    def set_basic_mode(self, enabled: bool) -> None:
        self._basic_mode = bool(enabled)
        detail_visible = not self._basic_mode
        for widget in [
            self.recluster_button,
            self.hide_button,
            self.preview_label,
            self.details_scroll,
            self.summary_section_label,
            self.summary_label,
            self.meaning_section_label,
            self.meaning_label,
            self.shape_section_label,
            self.shape_widget,
            self.basis_section_label,
            self.basis_label,
            self.preview_list,
            self.membership_section_label,
            self.membership_table,
        ]:
            widget.setVisible(detail_visible)

    def update_clusters(
        self,
        clusters_by_backend: dict[str, dict[int, list[str]]],
        membership_by_image: dict[str, dict[str, dict[str, object]]] | None = None,
        metrics_by_backend: dict[str, dict[str, object]] | None = None,
        cluster_summaries: dict[str, dict[int, ClusterTagSummary]] | None = None,
        cluster_explanations: dict[str, dict[int, ClusterExplanation]] | None = None,
        cluster_meanings: dict[str, dict[int, ClusterMeaning]] | None = None,
        *,
        preserve_selection: bool = False,
    ):
        selection = self.prev_selection if preserve_selection else None
        self._hide_hover_popup(cancel_preview=True)
        self._hover_preview_cache.clear()
        self._clusters_by_backend = clusters_by_backend
        self._membership_by_image = membership_by_image or {}
        self._metrics_by_backend = metrics_by_backend or {}
        self._cluster_summaries = cluster_summaries or {}
        self._cluster_explanations = cluster_explanations or {}
        self._cluster_meanings = cluster_meanings or {}
        self._grid_model.update_data(clusters_by_backend, self._cluster_summaries)
        self.cluster_table.resizeColumnsToContents()
        self.membership_table.setRowCount(0)
        self.preview_list.clear()
        self.preview_label.setText("Preview")
        self.summary_label.setText("No cluster selected.")
        self.meaning_label.setText(CLUSTER_MEANING_EMPTY_TEXT)
        self.shape_widget.clear()
        self.basis_label.setText(CLUSTER_BASIS_EMPTY_TEXT)
        self.recluster_button.setEnabled(False)
        self.prev_selection = None
        if selection is not None and self._clusters_by_backend.get(selection[0], {}).get(selection[1]):
            self.prev_selection = selection
            self._grid_model.set_selected(selection)
            self._update_preview(selection[0], selection[1])
            self.recluster_button.setEnabled(True)
            self.selection_target_changed.emit(self.current_selection_target())
            return
        self.selection_target_changed.emit(None)

    def select_default_cluster(self) -> tuple[str, int] | None:
        for comparison_key in sorted(self._clusters_by_backend.keys()):
            clusters = self._clusters_by_backend.get(comparison_key, {})
            if not clusters:
                continue
            cluster_id = min(int(value) for value in clusters.keys())
            selection = (comparison_key, cluster_id)
            self.prev_selection = selection
            self._grid_model.set_selected(selection)
            self._update_preview(selection[0], selection[1])
            self.recluster_button.setEnabled(True)
            self.selection_target_changed.emit(self.current_selection_target())
            return selection
        self.selection_target_changed.emit(None)
        return None

    def on_cluster_selected(self, index: QModelIndex):
        cell = self._grid_model.cell_at(index)
        if cell is None:
            return
        selection = (str(cell.comparison_key), int(cell.cluster_id))
        self._grid_model.set_selected(selection)
        self._update_preview(selection[0], selection[1])
        self.recluster_button.setEnabled(True)
        if self.prev_selection == selection:
            return
        self.prev_selection = selection
        self.cluster_selected.emit(selection[0], selection[1])
        self.selection_target_changed.emit(self.current_selection_target())

    def update_membership(
        self,
        image_path: str,
        membership_by_image: dict[str, dict[str, dict[str, object]]] | None = None,
        metrics_by_backend: dict[str, dict[str, object]] | None = None,
    ) -> None:
        if membership_by_image is not None:
            self._membership_by_image = membership_by_image
        if metrics_by_backend is not None:
            self._metrics_by_backend = metrics_by_backend

        backend_map = self._membership_by_image.get(image_path, {})
        self.membership_table.setRowCount(len(backend_map))
        for row, backend in enumerate(sorted(backend_map.keys())):
            info = backend_map[backend]
            quality = self._metrics_by_backend.get(backend, {}).get("cluster_quality_score", "")
            embedding_model = str(info.get("embedding_model", ""))
            backend_name = str(info.get("backend", backend))
            values = [
                embedding_model,
                backend_name,
                str(info.get("cluster_id", "")),
                str(info.get("cluster_size", "")),
                str(info.get("rank", "")),
                "yes" if info.get("outlier") else "no",
                "" if quality is None else str(quality),
            ]
            for column, value in enumerate(values):
                self.membership_table.setItem(row, column, QTableWidgetItem(value))
        self.membership_table.resizeColumnsToContents()

    def highlight_membership(self, image_path: str) -> None:
        backend_map = self._membership_by_image.get(image_path, {})
        highlighted: set[tuple[str, int]] = set()
        for key, info in backend_map.items():
            try:
                highlighted.add((str(key), int(info.get("cluster_id", -1))))
            except Exception:
                continue
        self._grid_model.set_highlighted(highlighted)

    def _update_preview(self, backend: str, cluster_id: int) -> None:
        images = self._clusters_by_backend.get(backend, {}).get(cluster_id, [])
        self.preview_label.setText(f"Preview | {backend} | Cluster {cluster_id} | {len(images)} images")
        self.preview_list.clear()
        for image_path in images[:12]:
            self.preview_list.addItem(Path(image_path).name)
        summary = self._cluster_summaries.get(backend, {}).get(cluster_id)
        self.summary_label.setText(_selected_cluster_summary_text(images, summary))
        explanation = self._cluster_explanations.get(backend, {}).get(cluster_id)
        meaning = self._cluster_meanings.get(backend, {}).get(cluster_id)
        self.meaning_label.setText(_cluster_meaning_text(meaning, summary, explanation, backend))
        self.shape_widget.set_explanation(explanation, backend)
        basis_text = _cluster_basis_text(explanation, summary)
        difference_text = _similarity_difference_text(backend, cluster_id, self._clusters_by_backend)
        if difference_text:
            basis_text = f"{basis_text}\n{difference_text}"
        self.basis_label.setText(basis_text)

    def current_selection_target(self) -> SelectionTarget | None:
        if self.prev_selection is None:
            return None
        backend, cluster_id = self.prev_selection
        images = tuple(self._clusters_by_backend.get(backend, {}).get(cluster_id, []))
        if not images:
            return None
        summary = self._cluster_summaries.get(backend, {}).get(cluster_id)
        return SelectionTarget(
            paths=images,
            kind="cluster",
            label=f"{backend} | Cluster {cluster_id}",
            source_context={
                "comparison_key": backend,
                "backend": backend,
                "cluster_id": int(cluster_id),
                "cluster_size": len(images),
                "cluster_tag_summary": summary.as_context() if summary is not None else {},
            },
        )

    def _update_hover_popup_for_index(self, index: QModelIndex) -> None:
        cell = self._grid_model.cell_at(index)
        if cell is None:
            self._hide_hover_popup(cancel_preview=True)
            return
        hover_key = (str(cell.comparison_key), int(cell.cluster_id))
        if self._hover_popup_key == hover_key and self._hover_popup.isVisible():
            self._hover_popup_index = index
            self._position_hover_popup(index)
            return
        self._hover_popup_key = hover_key
        self._hover_popup_index = index
        self._hover_preview_generation += 1
        generation = self._hover_preview_generation
        summary_lines = self._grid_model.summary_lines_for_cell(cell)
        more_count = max(0, len(cell.images) - HOVER_PREVIEW_MAX_ITEMS)
        self._hover_popup.set_cluster_details(cell, summary_lines, more_count=more_count, loading=True)
        self._position_hover_popup(index)
        self._hover_popup.show()
        cache_key = self._build_hover_preview_cache_key(cell)
        cached = self._hover_preview_cache.get(cache_key)
        if cached is not None:
            self._hover_preview_cache.move_to_end(cache_key)
            self._apply_hover_preview_image(cached, generation)
            return
        self._start_hover_preview_job(cell, cache_key, generation)

    def _build_hover_preview_cache_key(self, cell: ClusterCell) -> tuple[object, ...]:
        preview_paths = tuple(str(path) for path in cell.images[:HOVER_PREVIEW_MAX_ITEMS])
        return (
            str(cell.comparison_key),
            int(cell.cluster_id),
            len(cell.images),
            HOVER_PREVIEW_IMAGE_SIZE.width(),
            HOVER_PREVIEW_IMAGE_SIZE.height(),
            preview_paths,
        )

    def _start_hover_preview_job(self, cell: ClusterCell, cache_key: tuple[object, ...], generation: int) -> None:
        self._cancel_hover_preview_job()
        preview_paths = tuple(str(path) for path in cell.images[:HOVER_PREVIEW_MAX_ITEMS])

        def _run(_progress, cancel_check):
            raise_if_cancelled(cancel_check)
            service = ThumbnailService(qimage_cache_size=32)
            image = service.build_contact_sheet_qimage(
                preview_paths,
                HOVER_PREVIEW_IMAGE_SIZE,
                max_items=HOVER_PREVIEW_MAX_ITEMS,
                columns=3,
                cancel_check=cancel_check,
            )
            raise_if_cancelled(cancel_check)
            return {"cache_key": cache_key, "image": image, "generation": generation}

        job = AsyncJob(_run)

        def _completed(payload: dict[str, object]) -> None:
            image = payload.get("image")
            if not isinstance(image, QImage):
                image = QImage()
            if int(payload.get("generation", -1)) != self._hover_preview_generation:
                return
            self._remember_hover_preview(cache_key, image)
            self._apply_hover_preview_image(image, generation)

        def _failed(_message: str) -> None:
            if generation != self._hover_preview_generation:
                return
            self._hover_popup.set_preview_image(None)

        def _cancelled() -> None:
            if generation != self._hover_preview_generation:
                return
            self._hover_popup.set_preview_image(None)

        job.completed.connect(_completed)
        job.failed.connect(_failed)
        job.cancelled.connect(_cancelled)
        self._hover_preview_job = job
        thread = start_job_in_thread(job)
        self._thread_jobs[thread] = job
        thread.finished.connect(
            lambda thread=thread: self._on_async_thread_finished(thread),
            Qt.ConnectionType.QueuedConnection,
        )
        self._hover_preview_thread = thread

    def _apply_hover_preview_image(self, image: QImage, generation: int) -> None:
        if generation != self._hover_preview_generation or self._hover_popup_key is None:
            return
        self._hover_popup.set_preview_image(image if not image.isNull() else None)
        if self._hover_popup_index.isValid():
            self._position_hover_popup(self._hover_popup_index)
        self._hover_popup.show()

    def _remember_hover_preview(self, cache_key: tuple[object, ...], image: QImage) -> None:
        self._hover_preview_cache[cache_key] = image
        self._hover_preview_cache.move_to_end(cache_key)
        while len(self._hover_preview_cache) > HOVER_PREVIEW_CACHE_SIZE:
            self._hover_preview_cache.popitem(last=False)

    def _position_hover_popup(self, index: QModelIndex) -> None:
        if not index.isValid():
            return
        rect = self.cluster_table.visualRect(index)
        if not rect.isValid():
            return
        anchor = self.cluster_table.viewport().mapToGlobal(rect.topRight())
        self._hover_popup.adjustSize()
        popup_rect = self._hover_popup.frameGeometry()
        screen = self.cluster_table.screen() or QApplication.primaryScreen()
        available = screen.availableGeometry() if screen is not None else QRect(anchor, popup_rect.size())
        margin = 12
        target_x = anchor.x() + margin
        target_y = anchor.y() + margin
        if target_x + popup_rect.width() > available.right() - margin:
            target_x = rect.topLeft().x()
            target_x = self.cluster_table.viewport().mapToGlobal(QPoint(target_x, rect.top())).x() - popup_rect.width() - margin
        if target_y + popup_rect.height() > available.bottom() - margin:
            target_y = max(available.top() + margin, available.bottom() - popup_rect.height() - margin)
        target_x = max(available.left() + margin, target_x)
        target_y = max(available.top() + margin, target_y)
        self._hover_popup.move(target_x, target_y)

    def _hide_hover_popup(self, *, cancel_preview: bool) -> None:
        if cancel_preview:
            self._cancel_hover_preview_job()
            self._hover_preview_generation += 1
        self._hover_popup.hide()
        self._hover_popup_key = None
        self._hover_popup_index = QModelIndex()

    def _cancel_hover_preview_job(self) -> None:
        job = self._hover_preview_job
        thread = self._hover_preview_thread
        if job is not None:
            self._retain_async_refs(job, thread)
            try:
                job.cancel()
            except Exception:
                pass
        self._hover_preview_job = None
        self._hover_preview_thread = None

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
        if self._hover_preview_thread is thread:
            self._hover_preview_thread = None
            if self._hover_preview_job is job:
                self._hover_preview_job = None

    def _on_async_thread_finished(self, thread=None) -> None:
        if thread is None:
            return
        self._release_async_refs(self._thread_jobs.get(thread), thread)

    def shutdown_jobs(self, *, timeout_ms: int = 2500) -> bool:
        ready_to_close = True
        active_pairs = [(self._hover_preview_job, self._hover_preview_thread), *self._retained_async_refs]
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
        self._hover_preview_job = None
        self._hover_preview_thread = None
        if ready_to_close:
            self._retained_async_refs = []
            self._thread_jobs = {}
        self._hover_popup.hide()
        return ready_to_close

    def hideEvent(self, event) -> None:
        self._hide_hover_popup(cancel_preview=True)
        return super().hideEvent(event)

    def closeEvent(self, event) -> None:
        if not self.shutdown_jobs():
            event.ignore()
            return
        return super().closeEvent(event)
