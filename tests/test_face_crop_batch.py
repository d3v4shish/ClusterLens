from __future__ import annotations

import gc
import weakref
from unittest import mock

import numpy as np
from PIL import Image

from app.services import face_search


def _fixture_image() -> Image.Image:
    y, x = np.indices((360, 640), dtype=np.uint16)
    pixels = np.empty((360, 640, 3), dtype=np.uint8)
    pixels[:, :, 0] = (x + y) % 256
    pixels[:, :, 1] = (x // 2 + 31) % 256
    pixels[:, :, 2] = (y // 3 + 79) % 256
    return Image.fromarray(pixels, mode="RGB")


def _landmarks(box: tuple[int, int, int, int]) -> tuple[tuple[float, float], ...]:
    x1, y1, x2, y2 = box
    width = float(x2 - x1)
    height = float(y2 - y1)
    return (
        (x1 + width * 0.30, y1 + height * 0.35),
        (x1 + width * 0.70, y1 + height * 0.35),
        (x1 + width * 0.50, y1 + height * 0.55),
        (x1 + width * 0.35, y1 + height * 0.75),
        (x1 + width * 0.65, y1 + height * 0.75),
    )


def test_detector_cropper_reuses_one_full_frame_without_changing_pixels():
    image = _fixture_image()
    boxes = ((40, 30, 180, 210), (220, 50, 390, 250), (410, 90, 590, 320))
    expected = [
        face_search._crop_detected_face(image, box, _landmarks(box)).tobytes()
        for box in boxes
    ]

    with mock.patch.object(face_search, "_full_rgb_array", wraps=face_search._full_rgb_array) as convert:
        cropper = face_search._DetectedFaceCropper(image)
        actual = [cropper.crop(box, _landmarks(box)).tobytes() for box in boxes]

    assert actual == expected
    assert convert.call_count == 1


def test_detector_cropper_does_not_retain_full_frame_after_batch():
    image = _fixture_image()
    box = (40, 30, 180, 210)
    cropper = face_search._DetectedFaceCropper(image)
    crop = cropper.crop(box, _landmarks(box))
    source_ref = weakref.ref(cropper._source_rgb)

    del cropper
    gc.collect()

    assert source_ref() is None
    assert crop.size == (112, 112)


def test_unaligned_crop_does_not_materialize_full_frame_array():
    image = _fixture_image()
    box = (40, 30, 180, 210)

    with mock.patch.object(face_search, "_full_rgb_array", wraps=face_search._full_rgb_array) as convert:
        crop = face_search._DetectedFaceCropper(image).crop(box)

    assert convert.call_count == 0
    assert crop.size == (140, 180)


def test_builtin_detector_reuses_one_array_for_all_landmarked_faces():
    image = _fixture_image()
    boxes = np.asarray(((40, 30, 180, 210), (220, 50, 390, 250), (410, 90, 590, 320)), dtype=np.float32)

    class Detector:
        def detect(self, _image, *, landmarks):
            assert landmarks is True
            return boxes, np.asarray((0.99, 0.98, 0.97)), np.asarray([_landmarks(tuple(box)) for box in boxes])

    service = face_search.FaceDetectionService.__new__(face_search.FaceDetectionService)
    service._detector = Detector()
    service.score_threshold = 0.0
    service.max_detections = 50

    with mock.patch.object(face_search, "_full_rgb_array", wraps=face_search._full_rgb_array) as convert:
        faces = service.detect_faces_in_image("fixture.jpg", image)

    assert len(faces) == 3
    assert convert.call_count == 1
