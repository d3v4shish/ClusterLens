from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from apps.shared.release_fixture import ReleaseFixtureValidationError, verified_photo_fixture_metadata


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"clusterlens_clean_{name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_synthetic_release_fixture_is_verified_and_tampering_is_rejected(tmp_path: Path) -> None:
    creator = _load_script("create_release_fixture")
    fixture_root = tmp_path / "release_fixture"
    manifest = creator.build_fixture(fixture_root, photo_count=9, seed=42)

    metadata = verified_photo_fixture_metadata(fixture_root / "photos")
    assert metadata is not None
    assert metadata["fixture_photo_count"] == 9
    assert manifest["manifest_sha256"] == metadata["fixture_manifest_sha256"]
    from apps.pyqt_production.release_gates import _fixture_manifest_gate

    assert _fixture_manifest_gate(str(fixture_root / "photos")).status == "PASS"

    (fixture_root / "photos" / "group_00" / "synthetic_000000.png").write_bytes(b"tampered")
    with pytest.raises(ReleaseFixtureValidationError, match="aggregate checksum"):
        verified_photo_fixture_metadata(fixture_root / "photos")
    assert _fixture_manifest_gate(str(fixture_root / "photos")).status == "FAIL"


def test_synthetic_release_fixture_refuses_unrecognized_overwrite(tmp_path: Path) -> None:
    creator = _load_script("create_release_fixture")
    unsafe_root = tmp_path / "ordinary_photos"
    unsafe_root.mkdir()
    (unsafe_root / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="only accepts"):
        creator.build_fixture(unsafe_root, photo_count=1, overwrite=True)
    assert (unsafe_root / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_trash_recovery_verifier_uses_only_disposable_paths(tmp_path: Path, monkeypatch) -> None:
    verifier = _load_script("verify_trash_recovery")
    fixture_dir = tmp_path / "fixture"
    report_dir = tmp_path / "report"
    from app.services import gallery_actions

    monkeypatch.setattr(
        gallery_actions,
        "get_settings",
        lambda: (_ for _ in ()).throw(AssertionError("explicit verifier paths must not resolve global settings")),
    )

    assert verifier.main(["--fixture-dir", str(fixture_dir), "--report-dir", str(report_dir)]) == 0
    payload = json.loads((report_dir / "trash_recovery.json").read_text(encoding="utf-8"))
    assert payload["validation"] == "PASS"
    assert payload["network"] == "disabled by workload"

    with pytest.raises(ValueError, match="must be empty"):
        verifier.main(["--fixture-dir", str(fixture_dir), "--report-dir", str(tmp_path / "second-report")])


def test_packaged_launch_report_validation_requires_every_check(tmp_path: Path) -> None:
    verifier = _load_script("verify_packaged_launch")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps(
            {
                "scenario": "packaged-launch-settings-storage",
                "validation": "PASS",
                "checks": {name: True for name in verifier.REQUIRED_CHECKS},
            }
        ),
        encoding="utf-8",
    )
    assert verifier._read_and_validate_report(report)["validation"] == "PASS"

    report.write_text(
        json.dumps(
            {
                "scenario": "packaged-launch-settings-storage",
                "validation": "PASS",
                "checks": {"frozen_executable": True},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="checks failed"):
        verifier._read_and_validate_report(report)


def test_packaged_artifact_tree_digest_covers_files_links_modes_and_content(tmp_path: Path) -> None:
    verifier = _load_script("verify_packaged_launch")
    artifact = tmp_path / "ClusterLens"
    internal = artifact / "_internal"
    internal.mkdir(parents=True)
    executable = artifact / "ClusterLens"
    executable.write_bytes(b"binary")
    executable.chmod(0o755)
    payload = internal / "payload.bin"
    payload.write_bytes(b"payload")
    (internal / "payload-link").symlink_to("payload.bin")

    first = verifier._artifact_tree_evidence(artifact)
    second = verifier._artifact_tree_evidence(artifact)

    assert first == second
    assert first["artifact_file_count"] == 2
    assert first["artifact_symlink_count"] == 1
    assert first["artifact_total_bytes"] == len(b"binarypayloadpayload.bin")

    payload.write_bytes(b"changed")
    assert verifier._artifact_tree_evidence(artifact)["artifact_tree_sha256"] != first["artifact_tree_sha256"]
