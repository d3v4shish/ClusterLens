from __future__ import annotations

import pytest

from ui.count_copy import counted, loaded_and_showing, loading, showing, thumbnail_progress


@pytest.mark.parametrize(
    ("value", "unit", "expected"),
    [
        (0, "photo", "0 photos"),
        (1, "photo", "1 photo"),
        (2, "photo", "2 photos"),
        (1_001, "saved identity", "1,001 saved identities"),
    ],
)
def test_counted_uses_natural_plural_and_grouped_numbers(value: int, unit: str, expected: str) -> None:
    assert counted(value, unit) == expected


def test_loaded_showing_and_partial_loading_have_distinct_contracts() -> None:
    assert loaded_and_showing(36, 12, "photo") == "Loaded 36 photos · Showing 12 photos"
    assert showing(0, "photo", total=36, matching=True) == "Showing 0 of 36 matching photos"
    assert showing(1, "face", total=2, matching=True) == "Showing 1 of 2 matching faces"
    assert loading(12, 36, "photo") == "Loading metadata 12/36 photos"


def test_thumbnail_progress_never_calls_tasks_photos_or_loaded() -> None:
    assert thumbnail_progress(0, 3, qualifier="visible") == "Loading thumbnails 0/3 visible"
    assert thumbnail_progress(3, 3, done=True) == "Thumbnails ready 3/3"
