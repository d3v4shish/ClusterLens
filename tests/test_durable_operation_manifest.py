from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "docs" / "durable_operation_fault_matrix.json"
REQUIRED_OPERATION_IDS = {
    "embedding_index_generation",
    "face_index_batch",
    "library_catalog_batch",
    "photo_sidecar_save",
    "gallery_file_mutations",
    "embedded_metadata_rewrite",
    "face_model_promotion",
    "rebuildable_cache_clear",
    "sqlite_cache_clear",
    "data_home_backup",
    "data_home_relocation",
    "face_storage_removal",
    "saved_search_mutation",
    "image_tag_mutation",
    "clustering_result_cache",
}


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_durable_operation_manifest_covers_every_production_operation() -> None:
    payload = _manifest()
    operations = list(payload["operations"])

    assert payload["schema_version"] == 1
    assert payload["coverage_granularity"] == "operation_fault"
    assert {operation["id"] for operation in operations} == REQUIRED_OPERATION_IDS
    assert len({operation["id"] for operation in operations}) == len(operations)
    assert set(payload["platforms"]) == {"linux", "windows", "macos"}
    assert payload["platforms"]["windows"]["status"] == "NOT_RUN"
    assert payload["platforms"]["macos"]["status"] == "NOT_RUN"


def test_durable_operation_manifest_classifies_every_applicable_fault_and_boundary() -> None:
    payload = _manifest()
    declared_faults = set(payload["fault_classes"])
    phases = {"before", "commit", "after"}
    statuses = set(payload["status_values"])
    all_boundary_ids: set[str] = set()

    for operation in payload["operations"]:
        operation_id = operation["id"]
        applicable = set(operation["applicable_faults"])
        covered = set(operation["covered_faults"])
        gaps = set(operation["known_gaps"])
        assert applicable
        assert applicable <= declared_faults
        assert not covered & gaps
        assert covered | gaps == applicable
        assert operation["linux_status"] in statuses
        assert operation["linux_status"] == ("PASS" if not gaps else "PARTIAL")
        assert operation["idempotent_recovery"] is True
        for field in (
            "implementation",
            "commit_model",
            "precommit_cancel",
            "postcommit_cancel",
            "committed_subset",
            "recovery_action",
        ):
            value = str(operation[field]).strip()
            assert value and "TBD" not in value.upper()
        boundaries = list(operation["boundaries"])
        assert boundaries
        assert any(boundary["phase"] == "commit" for boundary in boundaries)
        for boundary in boundaries:
            assert boundary["phase"] in phases
            assert str(boundary["outcome"]).strip()
            qualified = f"{operation_id}:{boundary['id']}"
            assert qualified not in all_boundary_ids
            all_boundary_ids.add(qualified)

    actual_exit_cases = list(payload["actual_process_exit_cases"])
    assert len(actual_exit_cases) == 40
    assert len({case["id"] for case in actual_exit_cases}) == len(actual_exit_cases)
    operation_boundaries = {
        (operation["id"], boundary["id"])
        for operation in payload["operations"]
        for boundary in operation["boundaries"]
    }
    assert {(case["operation"], case["boundary"]) for case in actual_exit_cases} == operation_boundaries
    for case in actual_exit_cases:
        assert (case["operation"], case["boundary"]) in operation_boundaries
        assert str(case["checkpoint"]).strip()
        assert str(case["expected_state"]).strip()
        assert case["id"].startswith(f"{case['operation']}:")


def test_durable_operation_manifest_references_real_implementations_and_tests() -> None:
    for operation in _manifest()["operations"]:
        implementation_path, symbol = operation["implementation"].split("#", 1)
        implementation = REPO_ROOT / implementation_path
        assert implementation.is_file(), implementation
        implementation_text = implementation.read_text(encoding="utf-8")
        assert symbol.rsplit(".", 1)[-1] in implementation_text
        tests = list(operation["tests"])
        assert tests
        for node_id in tests:
            test_path, *node_parts = node_id.split("::")
            source = REPO_ROOT / test_path
            assert source.is_file(), source
            assert node_parts
            assert f"def {node_parts[-1]}" in source.read_text(encoding="utf-8"), node_id
