from __future__ import annotations

from PyQt6.QtGui import QColor, QImage

from ui.gallery_model import GalleryImageModel


def _image(red: int) -> QImage:
    image = QImage(2, 2, QImage.Format.Format_RGB32)
    image.fill(QColor(red, 0, 0))
    return image


def test_gallery_model_keeps_a_bounded_lru_image_cache() -> None:
    model = GalleryImageModel(image_cache_capacity=32)
    model.set_images([f"/{row}.jpg" for row in range(40)])

    for row in range(40):
        model.set_image(row, _image(row))

    assert model.cached_rows() == set(range(8, 40))
    model.data(model.index(8, 0), GalleryImageModel.PixmapRole)
    model.set_image(0, _image(100))
    assert len(model.cached_rows()) == 32
    assert 8 in model.cached_rows()
    assert 9 not in model.cached_rows()


def test_gallery_model_remaps_lru_rows_after_discontiguous_removal() -> None:
    model = GalleryImageModel(image_cache_capacity=32)
    paths = [f"/{row}.jpg" for row in range(5)]
    model.set_images(paths)
    for row in (4, 0, 3, 1):
        model.set_image(row, _image(row * 20))

    cached_rows = model.remove_paths({paths[1], paths[3]})

    assert model.image_paths() == [paths[0], paths[2], paths[4]]
    assert cached_rows == {0, 2}
    assert model.data(model.index(0, 0), GalleryImageModel.PixmapRole).pixelColor(0, 0).red() == 0
    assert model.data(model.index(2, 0), GalleryImageModel.PixmapRole).pixelColor(0, 0).red() == 80
