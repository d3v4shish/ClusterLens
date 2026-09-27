from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np

from app.services.face_search import FaceIndexRecord, FaceIndexService


class _TracingFaceIndexService(FaceIndexService):
    def __init__(self, *args, **kwargs) -> None:
        self.sql_statements: list[str] = []
        super().__init__(*args, **kwargs)

    def _connect(self) -> sqlite3.Connection:
        connection = super()._connect()
        connection.set_trace_callback(self.sql_statements.append)
        return connection


def _save_face(service: FaceIndexService, path: Path, *, face_index: int = 0) -> None:
    service.save_face_records(
        [
            FaceIndexRecord(
                image_path=str(path),
                face_index=face_index,
                face_bbox=(0, 0, 36, 36),
                face_confidence=0.99,
                embedding=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
            )
        ],
        mtime_ns=1,
        file_size=1,
        assess_quality=False,
    )


def test_face_album_pages_are_bounded_scoped_and_embedding_free(tmp_path: Path) -> None:
    db_path = tmp_path / "faces.sqlite3"
    service = _TracingFaceIndexService(db_path=db_path)
    scope = tmp_path / "scope%_photos"
    sibling = tmp_path / "scope%_photos-old"
    paths = [scope / f"photo_{index}.jpg" for index in range(6)]
    outside = sibling / "outside.jpg"
    for path in [*paths, outside]:
        _save_face(service, path)

    with sqlite3.connect(db_path) as connection:
        connection.executemany(
            "INSERT INTO face_labels(image_path, face_index, person_name, confidence) VALUES (?, 0, ?, 1.0)",
            [
                (str(paths[0]), "Alice"),
                (str(paths[1]), "Alice"),
                (str(paths[2]), "Bob"),
                (str(outside), "Outside"),
            ],
        )
        connection.execute("UPDATE face_index SET hidden=1 WHERE image_path=?", (str(paths[4]),))
        connection.execute(
            """
            INSERT INTO pending_face_labels(image_path, face_index, person_name, confidence, source)
            VALUES (?, 0, 'Carol', 0.8, 'test')
            """,
            (str(paths[5]),),
        )

    service.sql_statements.clear()
    first_page = service.load_face_album_group_page(folder_prefix=str(scope), limit=2)
    second_page = service.load_face_album_group_page(folder_prefix=str(scope), offset=2, limit=2)

    # Pending assignments stay in Review and must not become a duplicate
    # normal-All-Faces group. The explicit pending request below covers that
    # separate surface.
    assert first_page.total_count == 4
    assert first_page.next_offset == 2
    assert [item.group_id for item in first_page.items] == ["person:Alice", "person:Bob"]
    assert len(second_page.items) == 2
    assert all("Outside" not in item.title for item in [*first_page.items, *second_page.items])

    labelled_page = service.load_face_album_group_page(
        folder_prefix=str(scope),
        limit=10,
        group_kinds=("named",),
    )
    unlabelled_page = service.load_face_album_group_page(
        folder_prefix=str(scope),
        limit=10,
        group_kinds=("unlabeled", "pending"),
    )
    assert labelled_page.total_count == 2
    assert [item.group_id for item in labelled_page.items] == ["person:Alice", "person:Bob"]
    assert unlabelled_page.total_count == 2
    assert [item.group_id for item in unlabelled_page.items] == ["unlabeled", "pending"]

    alice_first = service.load_face_album_member_page("person:Alice", folder_prefix=str(scope), limit=1)
    alice_second = service.load_face_album_member_page(
        "person:Alice",
        folder_prefix=str(scope),
        offset=alice_first.next_offset or 0,
        limit=1,
    )
    assert alice_first.total_count == 2
    assert alice_first.next_offset == 1
    assert {alice_first.items[0].image_path, alice_second.items[0].image_path} == {
        str(paths[0]),
        str(paths[1]),
    }

    select_statements = [statement.lower() for statement in service.sql_statements if statement.lstrip().upper().startswith("SELECT")]
    assert select_statements
    assert all("embedding" not in statement for statement in select_statements)


def test_face_album_member_loading_does_not_hydrate_full_records(tmp_path: Path) -> None:
    service = FaceIndexService(db_path=tmp_path / "faces.sqlite3")
    image_path = tmp_path / "photo.jpg"
    _save_face(service, image_path)

    def _unexpected_full_load(*_args, **_kwargs):
        raise AssertionError("album loading must not hydrate embedding records")

    service.load_all_records = _unexpected_full_load  # type: ignore[method-assign]
    members = service.load_face_album_members(candidate_paths=[str(image_path)])

    assert [(item.image_path, item.face_index) for item in members] == [(str(image_path), 0)]
