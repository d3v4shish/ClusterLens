from __future__ import annotations

import sqlite3
from pathlib import Path

from app.path_scope import folder_scope_sql, normalize_scoped_path, path_is_within_scope


def test_path_scope_rejects_sibling_prefixes(tmp_path: Path) -> None:
    selected = tmp_path / "a"
    assert path_is_within_scope(str(selected), str(selected))
    assert path_is_within_scope(str(selected / "photo.jpg"), str(selected))
    assert not path_is_within_scope(str(tmp_path / "archive" / "photo.jpg"), str(selected))


def test_path_scope_handles_windows_paths_without_prefix_leaks() -> None:
    assert path_is_within_scope(r"C:\Photos\A\one.jpg", r"c:\photos\a")
    assert not path_is_within_scope(r"C:\Photos\Archive\one.jpg", r"c:\photos\a")
    assert not path_is_within_scope(r"D:\Photos\A\one.jpg", r"c:\photos\a")


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
