from __future__ import annotations

import os
from pathlib import Path

import pytest

from scripts import verify_clean_source


def test_snapshot_digest_tracks_content_mode_and_symlink_target(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        root.mkdir()
        script = root / "script.sh"
        script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        script.chmod(0o755)
        (root / "link").symlink_to("script.sh")
    paths = (Path("link"), Path("script.sh"))
    assert verify_clean_source._snapshot_digest(
        first, paths
    ) == verify_clean_source._snapshot_digest(second, paths)

    (second / "script.sh").chmod(0o644)
    assert verify_clean_source._snapshot_digest(
        first, paths
    ) != verify_clean_source._snapshot_digest(second, paths)


def test_materialize_source_preserves_candidate_files_and_symlinks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "nested").mkdir()
    executable = source / "nested" / "tool.sh"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)
    (source / "tool-link").symlink_to("nested/tool.sh")
    monkeypatch.setattr(verify_clean_source, "REPO_ROOT", source)
    paths = (Path("nested/tool.sh"), Path("tool-link"))
    destination = tmp_path / "destination"

    verify_clean_source._materialize_source(destination, paths)

    assert os.access(destination / "nested" / "tool.sh", os.X_OK)
    assert (destination / "tool-link").is_symlink()
    assert os.readlink(destination / "tool-link") == "nested/tool.sh"
    assert verify_clean_source._snapshot_digest(
        source, paths
    ) == verify_clean_source._snapshot_digest(destination, paths)


def test_verifier_refuses_existing_report_directory(tmp_path: Path) -> None:
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        verify_clean_source.main(["--report-dir", str(existing)])
