#!/usr/bin/env python3
"""Qualify explicitly supplied face models against an approved local fixture.

This opt-in gate never downloads models and never scans user media. Its
manifest pins every input/model file by SHA-256 and declares the exact model
pairs and clustering backends that must run. Missing assets are NOT_RUN (exit
3), while an executed contract failure exits 1.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

try:
    import resource
except ImportError:  # pragma: no cover - Windows qualification host
    resource = None


REPO_ROOT = Path(__file__).resolve().parents[1]
for import_path in (REPO_ROOT / "src", REPO_ROOT):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

REPORT_VERSION = 1
MANIFEST_VERSION = 1
EXIT_NOT_RUN = 3
SUPPORTED_BACKENDS = frozenset({"cosine-kmeans", "sklearn", "faiss", "hdbscan", "graph"})


def _is_sha256(value: object) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(character in "0123456789abcdef" for character in text)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(root: Path, value: object, *, label: str) -> Path:
    text = str(value or "").strip()
    relative = Path(text)
    if not text or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"{label} must be a non-empty relative path")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} escapes its declared root") from exc
    return candidate


def _load_manifest(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be an object")
    if int(payload.get("manifest_version", 0) or 0) != MANIFEST_VERSION:
        raise ValueError(f"manifest_version must be {MANIFEST_VERSION}")
    if not str(payload.get("fixture_id") or "").strip():
        raise ValueError("fixture_id is required")
    if not str(payload.get("fixture_license") or "").strip():
        raise ValueError("fixture_license is required")
    if not str(payload.get("fixture_source") or "").strip():
        raise ValueError("fixture_source is required")
    if not isinstance(payload.get("images"), list) or not payload["images"]:
        raise ValueError("images must be a non-empty list")
    if not isinstance(payload.get("model_cases"), list) or not payload["model_cases"]:
        raise ValueError("model_cases must be a non-empty list")
    coverage = payload.get("required_coverage")
    if not isinstance(coverage, dict):
        raise ValueError("required_coverage must declare detectors, embedders, and clustering_backends")
    for field in ("detectors", "embedders", "clustering_backends"):
        values = coverage.get(field)
        if not isinstance(values, list) or not values or any(not str(value).strip() for value in values):
            raise ValueError(f"required_coverage.{field} must be a non-empty list")
    return payload


def _validate_case_coverage(manifest: dict[str, object]) -> list[str]:
    coverage = dict(manifest["required_coverage"])
    cases = [case for case in manifest["model_cases"] if isinstance(case, dict)]
    declared = {
        "detectors": {str(case.get("detector_id") or "").strip() for case in cases},
        "embedders": {str(case.get("embedder_id") or "").strip() for case in cases},
        "clustering_backends": {
            str(backend).strip().lower()
            for case in cases
            for backend in case.get("clustering_backends", ())
        },
    }
    failures: list[str] = []
    for field in ("detectors", "embedders", "clustering_backends"):
        required = {str(value).strip().lower() for value in coverage[field]}
        present = {value.lower() for value in declared[field] if value}
        missing = sorted(required - present)
        unexpected = sorted(present - required)
        if missing:
            failures.append(f"required_coverage.{field} is missing cases for: {', '.join(missing)}")
        if unexpected:
            failures.append(f"required_coverage.{field} omits declared cases for: {', '.join(unexpected)}")
    return failures


def _verify_inputs(
    manifest: dict[str, object],
    *,
    fixture_root: Path,
    model_root: Path,
) -> tuple[list[dict[str, object]], list[str], list[str]]:
    images: list[dict[str, object]] = []
    missing: list[str] = []
    failures: list[str] = []
    for index, item in enumerate(manifest["images"]):
        if not isinstance(item, dict):
            failures.append(f"images[{index}] must be an object")
            continue
        try:
            path = _safe_relative(fixture_root, item.get("path"), label=f"images[{index}].path")
        except ValueError as exc:
            failures.append(str(exc))
            continue
        expected_sha = str(item.get("sha256") or "").strip().lower()
        if not _is_sha256(expected_sha):
            failures.append(f"images[{index}].sha256 must be a complete SHA-256")
            continue
        if not path.is_file():
            missing.append(f"fixture image missing: {path}")
            continue
        actual_sha = _sha256_file(path)
        if actual_sha != expected_sha:
            failures.append(f"fixture checksum mismatch: {path}")
            continue
        images.append(
            {
                "path": str(path),
                "relative_path": str(item["path"]),
                "sha256": actual_sha,
                "expected_min_faces": {
                    str(key): max(0, int(value))
                    for key, value in dict(item.get("expected_min_faces") or {}).items()
                },
            }
        )

    for case_index, case in enumerate(manifest["model_cases"]):
        if not isinstance(case, dict):
            failures.append(f"model_cases[{case_index}] must be an object")
            continue
        case_id = str(case.get("case_id") or "").strip()
        if not case_id:
            failures.append(f"model_cases[{case_index}].case_id is required")
        for field in ("detector_id", "embedder_id", "revision"):
            if not str(case.get(field) or "").strip():
                failures.append(f"model case {case_id or case_index} requires {field}")
        artifacts = case.get("artifacts")
        if not isinstance(artifacts, list) or not artifacts:
            failures.append(f"model case {case_id or case_index} requires pinned artifacts")
            continue
        for artifact_index, artifact in enumerate(artifacts):
            if not isinstance(artifact, dict):
                failures.append(f"model case {case_id} artifact {artifact_index} must be an object")
                continue
            try:
                artifact_path = _safe_relative(
                    model_root,
                    artifact.get("path"),
                    label=f"model case {case_id} artifact path",
                )
            except ValueError as exc:
                failures.append(str(exc))
                continue
            expected_sha = str(artifact.get("sha256") or "").strip().lower()
            if not _is_sha256(expected_sha):
                failures.append(f"model case {case_id} artifact SHA-256 is incomplete")
                continue
            if not str(artifact.get("source") or "").strip():
                failures.append(f"model case {case_id} artifact source is required")
                continue
            if not str(artifact.get("license") or "").strip():
                failures.append(f"model case {case_id} artifact license is required")
                continue
            if not artifact_path.is_file():
                missing.append(f"model artifact missing: {artifact_path}")
                continue
            if _sha256_file(artifact_path) != expected_sha:
                failures.append(f"model artifact checksum mismatch: {artifact_path}")
        backends = tuple(str(value).strip().lower() for value in case.get("clustering_backends", ()))
        unknown = sorted(set(backends) - SUPPORTED_BACKENDS)
        if unknown:
            failures.append(f"model case {case_id} has unsupported clustering backends: {', '.join(unknown)}")
    return images, missing, failures


def _peak_rss_bytes() -> int | None:
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform.startswith("linux"):
        return int(usage.ru_maxrss) * 1024
    if sys.platform == "darwin":
        return int(usage.ru_maxrss)
    return None


def _normalized_clusters(clusters: dict[int, list[int]]) -> list[list[int]]:
    return sorted(sorted(int(value) for value in members) for members in clusters.values() if members)


def _provider_evidence(service) -> dict[str, object]:
    detector = service.detection_service
    embedder = service.embedding_service
    evidence: dict[str, object] = {
        "policy": asdict(service.execution_policy),
        "detector_service": type(detector).__name__,
        "embedder_service": type(embedder).__name__,
    }
    for label, component in (("detector", detector), ("embedder", embedder)):
        session = getattr(component, "_session", None)
        if session is not None and callable(getattr(session, "get_providers", None)):
            evidence[f"{label}_providers"] = list(session.get_providers())
        model = getattr(component, "_model", None)
        if model is not None:
            try:
                evidence[f"{label}_torch_device"] = str(next(model.parameters()).device)
            except (StopIteration, AttributeError):
                pass
    return evidence


def _run_case(
    case: dict[str, object],
    *,
    images: list[dict[str, object]],
    model_root: Path,
    policy,
    runtime_service,
    runtime_root: Path,
) -> dict[str, object]:
    from app.services.face_search import FaceIndexService

    case_id = str(case["case_id"])
    detector_id = str(case["detector_id"])
    embedder_id = str(case["embedder_id"])
    expected_dimension = int(case.get("embedding_dimension", 512) or 512)
    tolerance = float(case.get("embedding_tolerance", 1e-5) or 1e-5)
    detection_tolerance = float(case.get("detection_tolerance", 1e-3) or 1e-3)

    def build_service(database_name: str):
        return FaceIndexService(
            mode=str(case.get("mode") or "human"),
            execution_policy=policy,
            runtime_service=runtime_service,
            model_root=model_root,
            detector_id=detector_id,
            embedder_id=embedder_id,
            fallback_detector_id=detector_id,
            db_path=runtime_root / database_name,
        )

    service = build_service(f"{case_id}-initial.sqlite3")
    if not service.detection_service.is_ready() or not service.embedding_service.is_ready():
        return {
            "case_id": case_id,
            "status": "NOT_RUN",
            "reason": (
                f"detector={service.detection_service.readiness_message()} "
                f"embedder={service.embedding_service.readiness_message()}"
            ),
        }

    started = time.perf_counter()
    crops = []
    detections: list[dict[str, object]] = []
    for image in images:
        faces = service.detection_service.detect_faces(str(image["path"]))
        expected_min = int(dict(image["expected_min_faces"]).get(detector_id, 1))
        if len(faces) < expected_min:
            raise RuntimeError(
                f"{case_id}: {image['relative_path']} produced {len(faces)} faces; expected at least {expected_min}"
            )
        detections.append(
            {
                "image": image["relative_path"],
                "faces": len(faces),
                "boxes": [list(face.bbox) for face in faces],
            }
        )
        crops.extend(face.crop for face in faces)
    cold_load_and_detect_ms = (time.perf_counter() - started) * 1000.0
    if not crops:
        raise RuntimeError(f"{case_id}: no face crops were produced")

    embed_started = time.perf_counter()
    embeddings = np.asarray(service.embedding_service.embed_faces(crops), dtype=np.float32)
    cold_embed_ms = (time.perf_counter() - embed_started) * 1000.0
    warm_started = time.perf_counter()
    warm_embeddings = np.asarray(service.embedding_service.embed_faces(crops), dtype=np.float32)
    warm_embed_ms = (time.perf_counter() - warm_started) * 1000.0
    if embeddings.shape != (len(crops), expected_dimension):
        raise RuntimeError(f"{case_id}: embedding shape {embeddings.shape} != {(len(crops), expected_dimension)}")
    if not np.isfinite(embeddings).all() or not np.isfinite(warm_embeddings).all():
        raise RuntimeError(f"{case_id}: embeddings contain non-finite values")
    max_warm_difference = float(np.max(np.abs(embeddings - warm_embeddings)))
    if max_warm_difference > tolerance:
        raise RuntimeError(f"{case_id}: warm embedding difference {max_warm_difference} > {tolerance}")

    clustering: dict[str, object] = {}
    for backend in tuple(str(value).strip().lower() for value in case.get("clustering_backends", ())):
        if len(embeddings) < 4:
            raise RuntimeError(f"{case_id}: at least four embeddings are required for clustering qualification")
        first, first_metrics = service.clustering_service.cluster(
            list(embeddings), 2, backend=backend, similarity_mode="cosine", outlier_policy="keep"
        )
        second, _second_metrics = service.clustering_service.cluster(
            list(embeddings), 2, backend=backend, similarity_mode="cosine", outlier_policy="keep"
        )
        first_membership = _normalized_clusters(first)
        if first_membership != _normalized_clusters(second):
            raise RuntimeError(f"{case_id}: {backend} membership was not deterministic")
        clustering[backend] = {"membership": first_membership, "metrics": first_metrics}

    output_digest = hashlib.sha256(
        embeddings.tobytes() + json.dumps(detections, sort_keys=True).encode("utf-8")
    ).hexdigest()
    initial_provider = _provider_evidence(service)
    reference_embeddings = embeddings.copy()
    for crop in crops:
        crop.close()
    del service, crops, embeddings, warm_embeddings
    gc.collect()
    rss_after_initial_release_bytes = _peak_rss_bytes()

    reload_started = time.perf_counter()
    reloaded_service = build_service(f"{case_id}-reloaded.sqlite3")
    if not reloaded_service.detection_service.is_ready() or not reloaded_service.embedding_service.is_ready():
        raise RuntimeError(f"{case_id}: model services were not ready after recreation")
    reloaded_crops = []
    reloaded_detections: list[dict[str, object]] = []
    for image in images:
        faces = reloaded_service.detection_service.detect_faces(str(image["path"]))
        reloaded_detections.append(
            {
                "image": image["relative_path"],
                "faces": len(faces),
                "boxes": [list(face.bbox) for face in faces],
            }
        )
        reloaded_crops.extend(face.crop for face in faces)
    reload_load_and_detect_ms = (time.perf_counter() - reload_started) * 1000.0
    if len(reloaded_detections) != len(detections):
        raise RuntimeError(f"{case_id}: recreated detector changed image result count")
    for initial, reloaded in zip(detections, reloaded_detections):
        if initial["image"] != reloaded["image"] or initial["faces"] != reloaded["faces"]:
            raise RuntimeError(f"{case_id}: recreated detector changed face ordering or count")
        if not np.allclose(
            np.asarray(initial["boxes"], dtype=np.float32),
            np.asarray(reloaded["boxes"], dtype=np.float32),
            rtol=0.0,
            atol=detection_tolerance,
        ):
            raise RuntimeError(f"{case_id}: recreated detector boxes exceeded tolerance {detection_tolerance}")

    reload_embed_started = time.perf_counter()
    reloaded_embeddings = np.asarray(
        reloaded_service.embedding_service.embed_faces(reloaded_crops),
        dtype=np.float32,
    )
    reload_embed_ms = (time.perf_counter() - reload_embed_started) * 1000.0
    if reloaded_embeddings.shape != reference_embeddings.shape:
        raise RuntimeError(
            f"{case_id}: recreated embedding shape {reloaded_embeddings.shape} != {reference_embeddings.shape}"
        )
    max_reload_difference = float(np.max(np.abs(reference_embeddings - reloaded_embeddings)))
    if max_reload_difference > tolerance:
        raise RuntimeError(f"{case_id}: recreated embedding difference {max_reload_difference} > {tolerance}")
    reloaded_provider = _provider_evidence(reloaded_service)
    for crop in reloaded_crops:
        crop.close()
    del reloaded_service, reloaded_crops, reloaded_embeddings, reference_embeddings
    gc.collect()
    return {
        "case_id": case_id,
        "status": "PASS",
        "detector_id": detector_id,
        "embedder_id": embedder_id,
        "revision": str(case["revision"]),
        "cold_load_and_detect_ms": round(cold_load_and_detect_ms, 3),
        "cold_embed_ms": round(cold_embed_ms, 3),
        "warm_embed_ms": round(warm_embed_ms, 3),
        "reload_load_and_detect_ms": round(reload_load_and_detect_ms, 3),
        "reload_embed_ms": round(reload_embed_ms, 3),
        "embedding_dimension": expected_dimension,
        "max_warm_embedding_difference": max_warm_difference,
        "max_reload_embedding_difference": max_reload_difference,
        "detection_tolerance": detection_tolerance,
        "detections": detections,
        "clustering": clustering,
        "provider": {"initial": initial_provider, "reloaded": reloaded_provider},
        "output_digest": output_digest,
        "rss_after_initial_release_bytes": rss_after_initial_release_bytes,
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def qualify(args: argparse.Namespace) -> tuple[dict[str, object], int]:
    manifest_path = args.manifest.expanduser().resolve()
    fixture_root = args.fixture_root.expanduser().resolve()
    model_root = args.model_root.expanduser().resolve()
    report: dict[str, object] = {
        "report_version": REPORT_VERSION,
        "operation": "real_model_qualification",
        "status": "FAIL",
        "manifest": str(manifest_path),
        "fixture_root": str(fixture_root),
        "model_root": str(model_root),
        "execution_mode": args.execution_mode,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "network_policy": "no downloads; explicit local assets only",
        },
        "cases": [],
        "failures": [],
        "not_run_reasons": [],
    }
    try:
        manifest = _load_manifest(manifest_path)
        report["manifest_sha256"] = _sha256_file(manifest_path)
        report["fixture_id"] = manifest["fixture_id"]
        report["fixture_license"] = manifest["fixture_license"]
        report["fixture_source"] = manifest["fixture_source"]
        report["required_coverage"] = manifest["required_coverage"]
        images, missing, failures = _verify_inputs(
            manifest,
            fixture_root=fixture_root,
            model_root=model_root,
        )
        failures.extend(_validate_case_coverage(manifest))
        if failures:
            report["failures"] = failures
            return report, 1
        if missing:
            report["status"] = "NOT_RUN"
            report["not_run_reasons"] = missing
            return report, EXIT_NOT_RUN

        from infra.runtime import RuntimeCapabilityService

        runtime_service = RuntimeCapabilityService()
        policy = runtime_service.select_policy(args.execution_mode)
        report["runtime_policy"] = asdict(policy)
        if args.execution_mode == "cuda" and not policy.uses_cuda:
            report["status"] = "NOT_RUN"
            report["not_run_reasons"] = [policy.error or policy.fallback_reason or "CUDA policy is unavailable"]
            return report, EXIT_NOT_RUN

        with tempfile.TemporaryDirectory(prefix="clusterlens-real-model-qualification-") as temporary:
            runtime_root = Path(temporary)
            cases: list[dict[str, object]] = []
            for case in manifest["model_cases"]:
                try:
                    cases.append(
                        _run_case(
                            case,
                            images=images,
                            model_root=model_root,
                            policy=policy,
                            runtime_service=runtime_service,
                            runtime_root=runtime_root,
                        )
                    )
                except Exception as exc:
                    cases.append(
                        {
                            "case_id": str(case.get("case_id") or "unknown"),
                            "status": "FAIL",
                            "reason": f"{type(exc).__name__}: {exc}",
                        }
                    )
            report["cases"] = cases
        if any(case["status"] == "FAIL" for case in report["cases"]):
            report["failures"] = [
                f"{case['case_id']}: {case.get('reason', 'qualification failed')}"
                for case in report["cases"]
                if case["status"] == "FAIL"
            ]
            return report, 1
        if any(case["status"] == "NOT_RUN" for case in report["cases"]):
            report["status"] = "NOT_RUN"
            report["not_run_reasons"] = [
                f"{case['case_id']}: {case.get('reason', 'not ready')}"
                for case in report["cases"]
                if case["status"] == "NOT_RUN"
            ]
            return report, EXIT_NOT_RUN
        report["status"] = "PASS"
        return report, 0
    except Exception as exc:
        report["failures"] = [f"{type(exc).__name__}: {exc}"]
        return report, 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fixture-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--execution-mode", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    report_path = args.report.expanduser().resolve()
    if report_path.exists():
        parser.error(f"refusing to overwrite report: {report_path}")
    payload, exit_code = qualify(args)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    report_path.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
