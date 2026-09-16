from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from app.path_scope import (
    PathScope,
    folder_scope_sql,
    normalize_scoped_path,
    normalized_path_is_within_scope,
    path_is_within_scope,
    roots_scope_sql,
)
from app.services.discovery import ImageDiscoveryService
from app.services.face_search import FaceIndexRecord, FaceIndexService
from app.services.image_tags import ImageTagService


def test_path_scope_rejects_sibling_prefixes(tmp_path: Path) -> None:
    selected = tmp_path / "a"
    assert path_is_within_scope(str(selected), str(selected))
    assert path_is_within_scope(str(selected / "photo.jpg"), str(selected))
    assert not path_is_within_scope(str(tmp_path / "archive" / "photo.jpg"), str(selected))


def test_path_scope_handles_windows_paths_without_prefix_leaks() -> None:
    assert path_is_within_scope(r"C:\Photos\A\one.jpg", r"c:\photos\a")
    assert not path_is_within_scope(r"C:\Photos\Archive\one.jpg", r"c:\photos\a")
    assert not path_is_within_scope(r"D:\Photos\A\one.jpg", r"c:\photos\a")


def test_normalized_path_scope_comparison_preserves_platform_and_boundary_rules(tmp_path: Path) -> None:
    root = normalize_scoped_path(str(tmp_path / "photos"))
    inside = normalize_scoped_path(str(tmp_path / "photos" / "one.jpg"))
    sibling = normalize_scoped_path(str(tmp_path / "photos-old" / "one.jpg"))

    assert normalized_path_is_within_scope(inside, root)
    assert not normalized_path_is_within_scope(sibling, root)
    assert normalized_path_is_within_scope(r"c:\photos\a\one.jpg", r"c:\photos\a")
    assert not normalized_path_is_within_scope(r"D:\photos\a\one.jpg", r"c:\photos\a")


def test_path_scope_resolves_symlink_escape(tmp_path: Path) -> None:
    selected = tmp_path / "selected"
    outside = tmp_path / "outside"
    selected.mkdir()
    outside.mkdir()
    link = selected / "escape"
    link.symlink_to(outside, target_is_directory=True)
    assert not path_is_within_scope(str(link / "photo.jpg"), str(selected))


def test_folder_scope_sql_treats_like_wildcards_as_literals(tmp_path: Path) -> None:
    selected = tmp_path / "set_100%"
    sibling = tmp_path / "setX1000"
    rows = [
        (str(selected),),
        (str(selected / "inside.jpg"),),
        (str(sibling / "outside.jpg"),),
    ]
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE images (image_path TEXT NOT NULL)")
    connection.executemany("INSERT INTO images (image_path) VALUES (?)", rows)
    clause, args = folder_scope_sql("image_path", str(selected))
    matches = [row[0] for row in connection.execute(f"SELECT image_path FROM images WHERE {clause}", args)]
    assert matches == [str(selected), str(selected / "inside.jpg")]


def test_normalize_scoped_path_is_absolute(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    assert normalize_scoped_path("folder/photo.jpg") == str(tmp_path / "folder" / "photo.jpg")


def test_active_root_scope_collapses_nested_paths_and_has_stable_signature(tmp_path: Path) -> None:
    parent = tmp_path / "photos"
    child = parent / "2026"
    sibling = tmp_path / "archive"
    scope = PathScope.from_paths([str(child), str(sibling), str(parent)])
    reversed_scope = PathScope.from_paths([str(sibling), str(parent), str(child)])
    assert set(scope.roots) == {str(parent), str(sibling)}
    assert scope.signature == reversed_scope.signature
    assert scope.contains(str(child / "one.jpg"))
    assert not scope.contains(str(tmp_path / "photos-old" / "outside.jpg"))


def test_roots_scope_sql_matches_union_but_not_siblings(tmp_path: Path) -> None:
    left = tmp_path / "left_100%"
    right = tmp_path / "right"
    other = tmp_path / "leftX1000"
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE images (image_path TEXT NOT NULL)")
    connection.executemany(
        "INSERT INTO images (image_path) VALUES (?)",
        [(str(left / "one.jpg"),), (str(right / "two.jpg"),), (str(other / "outside.jpg"),)],
    )
    clause, args = roots_scope_sql("image_path", (str(left), str(right)))
    matches = [row[0] for row in connection.execute(f"SELECT image_path FROM images WHERE {clause}", args)]
    assert matches == [str(left / "one.jpg"), str(right / "two.jpg")]


def test_multi_root_discovery_deduplicates_nested_active_roots(tmp_path: Path) -> None:
    parent = tmp_path / "photos"
    nested = parent / "trip"
    second = tmp_path / "archive"
    nested.mkdir(parents=True)
    second.mkdir()
    (nested / "one.jpg").write_bytes(b"fixture")
    (second / "two.jpg").write_bytes(b"fixture")

    result = ImageDiscoveryService().discover_roots_result([str(parent), str(nested), str(second)], recursive=True)

    assert result.image_count == 2
    assert result.paths == tuple(sorted((str(nested / "one.jpg"), str(second / "two.jpg")), key=str.casefold))


def test_face_album_and_review_are_limited_to_multiple_active_roots(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    excluded_root = tmp_path / "excluded"
    for directory in (first_root, second_root, excluded_root):
        directory.mkdir()
    first_image = first_root / "one.jpg"
    second_image = second_root / "two.jpg"
    excluded_image = excluded_root / "outside.jpg"
    for image_path in (first_image, second_image, excluded_image):
        image_path.write_bytes(b"fixture")

    service = FaceIndexService(db_path=tmp_path / "faces.sqlite3")
    for image_path in (first_image, second_image, excluded_image):
        service.save_face_records(
            [
                FaceIndexRecord(
                    image_path=str(image_path),
                    face_index=0,
                    face_bbox=(0, 0, 50, 50),
                    face_confidence=0.99,
                    embedding=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
                )
            ],
            mtime_ns=image_path.stat().st_mtime_ns,
            file_size=image_path.stat().st_size,
            assess_quality=False,
        )

    roots = [str(first_root), str(second_root)]
    group_page = service.load_face_album_group_page(scope_roots=roots)
    members = service.load_face_album_member_page("unlabeled", scope_roots=roots)
    review = service.load_folder_review_images(
        str(first_root),
        scope_roots=roots,
        candidate_paths=[str(first_image), str(second_image)],
    )

    assert group_page.total_count == 1
    assert {record.image_path for record in members.items} == {str(first_image), str(second_image)}
    assert {record.image_path for record in review} == {str(first_image), str(second_image)}

    service.label_indexed_faces_immediately("Alice", [(str(first_image), 0), (str(excluded_image), 0)])
    named = service.list_named_photo_summaries(scope_roots=roots)
    named_paths = service.list_named_photo_paths("Alice", scope_roots=roots)
    assert [(item.person_name, item.photo_count) for item in named] == [("Alice", 1)]
    assert named_paths == [str(first_image)]


def test_tag_inventory_and_paths_are_limited_to_multiple_active_roots(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    excluded_root = tmp_path / "excluded"
    for directory in (first_root, second_root, excluded_root):
        directory.mkdir()
    first_image = first_root / "one.jpg"
    second_image = second_root / "two.jpg"
    excluded_image = excluded_root / "outside.jpg"
    for image_path in (first_image, second_image, excluded_image):
        image_path.write_bytes(b"fixture")

    service = ImageTagService(db_path=tmp_path / "tags.sqlite3")
    service.apply_tag_edit(
        [str(first_image), str(second_image), str(excluded_image)],
        add_tags=["Family"],
        mirror_to_exif=False,
    )

    roots = [str(first_root), str(second_root)]
    inventory = service.query_tag_inventory(scope_paths=roots)
    paths = service.query_tagged_paths("Family", scope_paths=roots)

    assert [(item.display_tag, item.image_count) for item in inventory.items] == [("Family", 2)]
    assert set(paths.paths) == {str(first_image), str(second_image)}
