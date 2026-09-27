from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QColor, QImage, QPainter, QPalette
from PyQt6.QtWidgets import QApplication, QLineEdit, QPushButton, QStyle, QStyleOptionViewItem

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from infra.settings import get_production_settings_registry
from ui.icons import apply_icon
from ui import theme as theme_module
from ui.cluster_pane import ClusterCellDelegate, ClusterGridModel, ClusterHoverPreviewPopup
from ui.footer_bar import WorkspaceFooter
from ui.gallery_model import GalleryItemDelegate
from ui.runtime_widgets import RuntimeBadge
from ui.sectioned_gallery import _GroupHoverPreviewPopup
from ui.theme import COLORS, DARK_COLORS, ThemeManager, apply_app_density, apply_app_text_scale, apply_app_theme, current_theme, install_theme_manager, stylesheet_for_theme
from ui.zoomable_image import ZoomableImageView, adaptive_neutral_backdrop


APP = QApplication.instance() or QApplication([])


class _Store:
    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = dict(values or {})

    def value(self, key, default=None, value_type=None):
        value = self.values.get(key, default)
        return value_type(value) if value_type is not None and value is not None else value

    def setValue(self, key, value) -> None:
        self.values[str(key)] = value

    def sync(self) -> None:
        return None


def _solid_image(color: QColor) -> QImage:
    image = QImage(96, 64, QImage.Format.Format_RGBA8888)
    image.fill(color)
    return image


def _contrast_ratio(foreground: QColor, background: QColor) -> float:
    def luminance(color: QColor) -> float:
        channels = [channel / 255.0 for channel in (color.red(), color.green(), color.blue())]
        linear = [
            channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return sum(
            channel * weight
            for channel, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True)
        )

    first = luminance(foreground)
    second = luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def test_appearance_settings_validate_to_safe_defaults() -> None:
    registry = get_production_settings_registry()
    store = _Store({"appearance/theme": "sepia", "appearance/viewer_backdrop": "photo-accent", "appearance/text_scale": 175})

    assert registry.get(store, "appearance/theme") == "system"
    assert registry.get(store, "appearance/viewer_backdrop") == "adaptive_neutral"
    assert registry.get(store, "appearance/text_scale") == 100


def test_adaptive_neutral_backdrop_uses_bounded_neutral_classes() -> None:
    assert adaptive_neutral_backdrop(_solid_image(QColor("#050607"))) == "#050607"
    assert adaptive_neutral_backdrop(_solid_image(QColor("#B0B0B0"))) == "#30343A"
    assert adaptive_neutral_backdrop(_solid_image(QColor("#FFFFFF"))) == "#E6E8EB"
    assert adaptive_neutral_backdrop(_solid_image(QColor(255, 255, 255, 0))) is None


def test_theme_manager_applies_fixed_modes_and_persists_viewer_choice() -> None:
    store = _Store()
    manager = ThemeManager(APP)
    try:
        manager.configure(store)
        manager.set_preferences("light", "middle_gray", persist=True)
        assert current_theme() == "light"
        assert COLORS["text"] == "#252A2D"
        assert store.values["appearance/theme"] == "light"
        assert store.values["appearance/viewer_backdrop"] == "middle_gray"

        manager.set_preferences("dark", "theme", persist=False)
        assert current_theme() == "dark"
        assert COLORS["text"] == "#E8EAED"
    finally:
        apply_app_theme(APP, "dark")


def test_system_preference_re_resolves_without_changing_saved_mode() -> None:
    manager = ThemeManager(APP)
    try:
        with patch.object(theme_module, "_resolved_system_theme", return_value="light"):
            manager.set_preferences("system", "theme", persist=False)
        assert manager.theme_preference == "system"
        assert manager.resolved_theme == "light"
        with patch.object(theme_module, "_resolved_system_theme", return_value="dark"):
            manager._system_color_scheme_changed()
        assert manager.theme_preference == "system"
        assert manager.resolved_theme == "dark"
    finally:
        apply_app_theme(APP, "dark")


def test_light_stylesheet_replaces_dark_canvas_and_text_tokens() -> None:
    stylesheet = stylesheet_for_theme("light")
    assert "ClusterLens light design system" in stylesheet
    assert "background: #0B0D10" not in stylesheet
    assert "color: #F2F5F8" not in stylesheet
    assert "background: palette(window)" in stylesheet
    assert "color: palette(window-text)" in stylesheet


def test_changed_theme_updates_palette_without_rebuilding_global_stylesheet() -> None:
    apply_app_theme(APP, "dark")
    dark_stylesheet = APP.styleSheet()
    dark_palette = APP.palette()

    apply_app_theme(APP, "light")

    assert APP.styleSheet() == dark_stylesheet
    assert APP.palette().color(QPalette.ColorRole.Window) == QColor(COLORS["canvas"])
    assert APP.palette().color(QPalette.ColorRole.Window) != dark_palette.color(QPalette.ColorRole.Window)
    assert APP.palette().color(QPalette.ColorRole.Link) == QColor(COLORS["accent"])
    apply_app_theme(APP, "dark")


def test_live_theme_refreshes_auto_colored_icons_and_viewer_backgrounds() -> None:
    button = QPushButton()
    view = ZoomableImageView()
    try:
        apply_app_theme(APP, "dark")
        apply_icon(button, "settings")
        dark_icon_key = button.icon().cacheKey()

        apply_app_theme(APP, "light")
        assert button.icon().cacheKey() != dark_icon_key
        view.set_backdrop_mode("theme")
        assert view.current_backdrop_color() == "#F2F3F2"
        view.set_backdrop_mode("black")
        assert view.current_backdrop_color() == "#050607"
    finally:
        button.close()
        view.close()
        apply_app_theme(APP, "dark")


def test_theme_reapply_is_a_noop_and_density_changes_the_shared_stylesheet() -> None:
    apply_app_theme(APP, "dark")
    revision = theme_module.theme_revision()
    apply_app_theme(APP, "dark")
    assert theme_module.theme_revision() == revision
    assert apply_app_density(APP, False)
    assert "padding: 7px 11px" in APP.styleSheet()
    assert apply_app_density(APP, True)
    assert "padding: 5px 9px" in APP.styleSheet()


def test_text_scale_is_persisted_and_survives_theme_and_density_changes() -> None:
    store = _Store({"appearance/theme": "dark", "appearance/text_scale": 150})
    manager = ThemeManager(APP)
    try:
        manager.configure(store)
        assert manager.text_scale == 150
        assert APP.property("clusterlens_text_scale") == 150
        assert "font-size: 21px" in APP.styleSheet()

        apply_app_density(APP, False)
        apply_app_theme(APP, "light")
        assert "font-size: 21px" in APP.styleSheet()

        manager.set_text_scale(200, persist=True)
        assert store.values["appearance/text_scale"] == 200
        assert "font-size: 28px" in APP.styleSheet()
        assert "font-size: 40px" in APP.styleSheet()
        assert "QWidget#applicationHeader QPushButton, QWidget#applicationHeader QToolButton { min-height: 44px; }" in APP.styleSheet()
        assert "QLineEdit { min-height: 44px; }" in APP.styleSheet()
        assert apply_app_text_scale(APP, "invalid")
        assert APP.property("clusterlens_text_scale") == 100
    finally:
        apply_app_text_scale(APP, 100)
        apply_app_density(APP, True)
        apply_app_theme(APP, "dark")


def test_large_text_single_line_fields_have_readable_vertical_geometry() -> None:
    field = QLineEdit()
    field.setPlaceholderText("Search settings")
    try:
        apply_app_text_scale(APP, 200)
        field.show()
        APP.processEvents()
        assert field.height() >= field.fontMetrics().height() + 8
    finally:
        field.close()
        apply_app_text_scale(APP, 100)


def test_dark_action_hover_tokens_keep_white_text_readable() -> None:
    def luminance(value: str) -> float:
        channels = [int(value[index : index + 2], 16) / 255.0 for index in (1, 3, 5)]
        linear = [channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4 for channel in channels]
        return sum(channel * weight for channel, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True))

    white = luminance("#FFFFFF")
    for color in (DARK_COLORS["accent_hover"], DARK_COLORS["danger"], DARK_COLORS["danger_hover"]):
        contrast = (white + 0.05) / (luminance(color) + 0.05)
        assert contrast >= 4.5


def test_cluster_delegate_and_footer_use_readable_resolved_theme_states() -> None:
    model = ClusterGridModel()
    model.update_data({"fixture": {0: ["photo.jpg"]}})
    index = model.index(0, 0)
    delegate = ClusterCellDelegate()
    option = QStyleOptionViewItem()
    option.rect = QRect(0, 0, 196, 54)
    option.state = QStyle.StateFlag.State_Enabled | QStyle.StateFlag.State_MouseOver

    try:
        for theme in ("dark", "light"):
            apply_app_theme(APP, theme)
            image = QImage(196, 54, QImage.Format.Format_RGB32)
            image.fill(QColor("#FF00FF"))
            painter = QPainter(image)
            delegate.paint(painter, option, index)
            painter.end()
            hover_background = image.pixelColor(190, 26)
            assert hover_background == QColor(COLORS["surface_raised"])
            assert _contrast_ratio(QColor(COLORS["text"]), hover_background) >= 4.5

            model.set_highlighted({("fixture", 0)})
            highlighted = index.data(Qt.ItemDataRole.BackgroundRole)
            assert highlighted == QColor(COLORS["surface_checked"])
            assert _contrast_ratio(QColor(COLORS["text"]), highlighted) >= 4.5

            footer = WorkspaceFooter()
            footer.resize(900, 38)
            footer.show()
            APP.processEvents()
            footer_image = footer.folder_label.grab().toImage()
            footer_background = footer_image.pixelColor(
                max(0, footer_image.width() - 4),
                max(0, footer_image.height() // 2),
            )
            footer_foreground = footer.folder_label.palette().color(
                footer.folder_label.foregroundRole()
            )
            assert _contrast_ratio(footer_foreground, footer_background) >= 4.5
            footer.close()
            model.set_highlighted(set())
    finally:
        apply_app_theme(APP, "dark")


def test_semantic_readable_and_focus_states_meet_contrast_targets_in_both_themes() -> None:
    readable_pairs = (
        ("text", "surface"),
        ("text", "surface_selected"),
        ("text_muted", "surface_sunken"),
        ("danger_text", "danger_surface"),
        ("warning", "surface"),
        ("success_bright", "surface"),
    )
    try:
        for theme in ("dark", "light"):
            apply_app_theme(APP, theme)
            disabled_button = QPushButton("Disabled action")
            disabled_button.setEnabled(False)
            disabled_foreground = disabled_button.palette().color(
                QPalette.ColorGroup.Disabled,
                QPalette.ColorRole.ButtonText,
            )
            assert _contrast_ratio(disabled_foreground, QColor(COLORS["surface_sunken"])) >= 4.5
            disabled_button.close()
            for foreground, background in readable_pairs:
                assert _contrast_ratio(QColor(COLORS[foreground]), QColor(COLORS[background])) >= 4.5, (
                    theme,
                    foreground,
                    background,
                )
            for background in ("canvas", "surface", "surface_raised", "surface_selected"):
                assert _contrast_ratio(QColor(COLORS["focus"]), QColor(COLORS[background])) >= 3.0, (
                    theme,
                    background,
                )
    finally:
        apply_app_theme(APP, "dark")


def test_cached_delegate_popup_icon_and_inspector_colors_re_resolve_on_runtime_switch() -> None:
    manager = install_theme_manager(APP, _Store({"appearance/theme": "dark", "appearance/viewer_backdrop": "theme"}))
    delegate = GalleryItemDelegate(96)
    cluster_popup = ClusterHoverPreviewPopup()
    group_popup = _GroupHoverPreviewPopup()
    badge = RuntimeBadge()
    view = ZoomableImageView()
    button = QPushButton()
    try:
        apply_icon(button, "settings")
        dark_icon_key = button.icon().cacheKey()
        dark_card = QColor(delegate._card)
        assert str(COLORS["surface_raised"]) in cluster_popup.styleSheet()
        assert str(COLORS["surface_raised"]) in group_popup.styleSheet()
        assert view.current_backdrop_color() == COLORS["viewer_canvas"]
        assert badge.property("state") == "warning"
        assert "Checking" in badge.text()

        manager.set_preferences("light", "theme", persist=False)
        APP.processEvents()
        delegate._sync_theme_colors()

        assert QColor(delegate._card) == QColor(COLORS["surface"])
        assert QColor(delegate._card) != dark_card
        assert str(COLORS["surface_raised"]) in cluster_popup.styleSheet()
        assert str(COLORS["surface_raised"]) in group_popup.styleSheet()
        assert view.current_backdrop_color() == COLORS["viewer_canvas"]
        assert button.icon().cacheKey() != dark_icon_key

        badge.set_runtime_failure("fixture failure")
        assert badge.property("state") == "error"
        assert "Failed" in badge.text()
        assert "fixture failure" in badge.accessibleDescription()
    finally:
        cluster_popup.close()
        group_popup.close()
        badge.close()
        view.close()
        button.close()
        manager.set_preferences("dark", "theme", persist=False)
