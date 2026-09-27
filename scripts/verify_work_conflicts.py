"""Verify cross-feature scheduling, cancellation, and recovery contracts.

This verifier intentionally drives the production ``WorkCoordinator`` rather
than a scheduler double.  Every transition is synchronous and signal-driven;
there are no sleeps or timing-dependent cancellation points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ui.job_manager import JobManager  # noqa: E402
from ui.work_coordinator import JobSpec, WorkCoordinator  # noqa: E402


REPORT_VERSION = 1
DEFAULT_SEED = 20260925
TerminalStatus = Literal["finished", "failed", "cancelled"]


@dataclass(frozen=True)
class ConflictCase:
    case_id: str
    first: JobSpec
    second: JobSpec
    expected_conflict: str | None


def _fixture_path(*parts: str) -> str:
    return os.path.join(os.sep, "clusterlens-conflict-fixture", *parts)


def _cases() -> tuple[ConflictCase, ...]:
    photos = _fixture_path("photos")
    return (
        ConflictCase(
            "gpu_embedding_and_clustering",
            JobSpec("Build embedding index", origin="Library", uses_gpu=True),
            JobSpec(
                "Cluster faces",
                origin="People",
                uses_gpu=True,
                cpu_fallback=True,
            ),
            "GPU",
        ),
        ConflictCase(
            "face_detection_and_identity_edit",
            JobSpec("Detect and index faces", origin="People", data_home_write=True),
            JobSpec("Merge identities", origin="People", data_home_write=True),
            "Data Home access",
        ),
        ConflictCase(
            "inference_and_model_replacement",
            JobSpec("Face inference", origin="People", model_cache_read=True),
            JobSpec("Replace face model", origin="Models", model_cache_write=True),
            "model cache access",
        ),
        ConflictCase(
            "source_read_and_overlapping_mutation",
            JobSpec(
                "Hash duplicate candidates",
                origin="Tools",
                source_reads=(photos,),
            ),
            JobSpec(
                "Rename selected photo",
                origin="Photos",
                source_writes=(os.path.join(photos, "trip", "a.jpg"),),
            ),
            "the same source files",
        ),
        ConflictCase(
            "backup_and_managed_store_writer",
            JobSpec("Back up Data Home", origin="Settings", data_home_read=True),
            JobSpec("Refresh catalog", origin="Library", data_home_write=True),
            "Data Home access",
        ),
        ConflictCase(
            "disjoint_read_only_workflows",
            JobSpec(
                "Browse library",
                origin="Library",
                source_reads=(os.path.join(photos, "library"),),
            ),
            JobSpec(
                "Inspect people",
                origin="People",
                source_reads=(os.path.join(photos, "people"),),
            ),
            None,
        ),
    )


class _Scenario:
    def __init__(self, case: ConflictCase, *, reverse: bool) -> None:
        self.case = case
        self.reverse = bool(reverse)
        self.manager = JobManager()
        self.coordinator = WorkCoordinator(
            self.manager,
            gpu_policy="queue",
            cpu_capacity=4,
            io_capacity=4,
        )
        self.timeline: list[dict[str, object]] = []
        self.started: list[int] = []
        self.cancel_calls: list[int] = []
        self.coordinator.job_queued.connect(
            lambda job_id, reason: self._record("queued", job_id, reason=str(reason))
        )
        self.coordinator.job_started.connect(
            lambda job_id, cpu: self._record("started", job_id, cpu_fallback=bool(cpu))
        )
        self.coordinator.job_finished.connect(
            lambda job_id, status: self._record("finished", job_id, status=str(status))
        )

    def _record(self, event: str, job_id: int, **details: object) -> None:
        self.timeline.append({"event": event, "job_id": int(job_id), **details})

    def _starter(self, job_id: int, _use_cpu_fallback: bool) -> None:
        self.started.append(int(job_id))

    def _cancel_running(self, job_id: int) -> None:
        self.cancel_calls.append(int(job_id))
        self.coordinator.finish(job_id, status="cancelled")

    def submit(self, spec: JobSpec) -> int:
        holder: dict[str, int] = {}

        def _cancel() -> None:
            self._cancel_running(holder["job_id"])

        job_id = self.coordinator.submit(spec, self._starter, cancel=_cancel)
        holder["job_id"] = int(job_id)
        return int(job_id)

    def ordered_specs(self) -> tuple[JobSpec, JobSpec]:
        if self.reverse:
            return self.case.second, self.case.first
        return self.case.first, self.case.second

    def assert_released(self) -> None:
        if self.coordinator._queued or self.coordinator._running:  # noqa: SLF001 - verifier invariant
            raise AssertionError(
                f"scheduler retained work: queued={tuple(self.coordinator._queued)}, "  # noqa: SLF001
                f"running={tuple(self.coordinator._running)}"  # noqa: SLF001
            )


def _state(manager: JobManager, job_id: int) -> str:
    state = manager.get(job_id)
    if state is None:
        raise AssertionError(f"job {job_id} has no visible state")
    return str(state.status)


def _normal_and_retry(case: ConflictCase, *, reverse: bool) -> dict[str, object]:
    scenario = _Scenario(case, reverse=reverse)
    first_spec, second_spec = scenario.ordered_specs()
    first = scenario.submit(first_spec)
    second = scenario.submit(second_spec)

    if case.expected_conflict is None:
        if scenario.started != [first, second]:
            raise AssertionError(f"compatible work did not overlap: {scenario.started}")
    else:
        if scenario.started != [first] or _state(scenario.manager, second) != "queued":
            raise AssertionError(f"conflicting work did not queue: {scenario.started}")
        queued_text = str(scenario.manager.get(second).text)
        if case.expected_conflict not in queued_text:
            raise AssertionError(
                f"wrong queue reason for {case.case_id}: {queued_text!r}"
            )

    scenario.coordinator.finish(first, status="finished")
    if second not in scenario.started:
        raise AssertionError(f"second job never started after release: {scenario.started}")
    scenario.coordinator.cancel(second)
    if _state(scenario.manager, second) != "cancelled":
        raise AssertionError("running cancellation did not reach cancelled")
    if scenario.cancel_calls != [second]:
        raise AssertionError(f"running cancel callback mismatch: {scenario.cancel_calls}")

    retry = scenario.submit(second_spec)
    if _state(scenario.manager, retry) != "running":
        raise AssertionError("cancelled work could not restart")
    scenario.coordinator.finish(retry, status="failed", error="injected verifier failure")
    if _state(scenario.manager, retry) != "failed":
        raise AssertionError("injected failure was not visible")

    recovered = scenario.submit(second_spec)
    if _state(scenario.manager, recovered) != "running":
        raise AssertionError("failed work could not restart")
    scenario.coordinator.finish(recovered, status="finished")
    scenario.assert_released()
    return {
        "case_id": case.case_id,
        "launch_order": "second_first" if reverse else "first_first",
        "mode": "normal_running_cancel_failure_retry",
        "started_job_ids": scenario.started,
        "terminal_states": {
            str(job_id): _state(scenario.manager, job_id)
            for job_id in (first, second, retry, recovered)
        },
        "timeline": scenario.timeline,
        "resource_counts_after": {"queued": 0, "running": 0},
        "status": "PASS",
    }


def _queued_cancel(case: ConflictCase, *, reverse: bool) -> dict[str, object] | None:
    if case.expected_conflict is None:
        return None
    scenario = _Scenario(case, reverse=reverse)
    first_spec, second_spec = scenario.ordered_specs()
    first = scenario.submit(first_spec)
    queued = scenario.submit(second_spec)
    if _state(scenario.manager, queued) != "queued":
        raise AssertionError("queued-cancel scenario did not queue")

    scenario.coordinator.cancel(queued)
    scenario.coordinator.finish(first, status="finished")
    if queued in scenario.started:
        raise AssertionError("cancelled queued work started later")
    if scenario.cancel_calls:
        raise AssertionError("pre-start cancellation called the worker cancel callback")

    retry = scenario.submit(second_spec)
    if _state(scenario.manager, retry) != "running":
        raise AssertionError("pre-start cancelled work could not restart")
    scenario.coordinator.finish(retry, status="finished")
    scenario.assert_released()
    return {
        "case_id": case.case_id,
        "launch_order": "second_first" if reverse else "first_first",
        "mode": "queued_cancel_retry",
        "started_job_ids": scenario.started,
        "terminal_states": {
            str(job_id): _state(scenario.manager, job_id)
            for job_id in (first, queued, retry)
        },
        "timeline": scenario.timeline,
        "resource_counts_after": {"queued": 0, "running": 0},
        "status": "PASS",
    }


def run_matrix(*, seed: int = DEFAULT_SEED) -> dict[str, object]:
    scheduled = [(case, reverse) for case in _cases() for reverse in (False, True)]
    random.Random(int(seed)).shuffle(scheduled)
    results: list[dict[str, object]] = []
    for case, reverse in scheduled:
        results.append(_normal_and_retry(case, reverse=reverse))
        queued_result = _queued_cancel(case, reverse=reverse)
        if queued_result is not None:
            results.append(queued_result)

    stable_payload = {
        "seed": int(seed),
        "case_order": [
            {"case_id": case.case_id, "reverse": reverse}
            for case, reverse in scheduled
        ],
        "results": results,
    }
    result_digest = hashlib.sha256(
        json.dumps(stable_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "report_version": REPORT_VERSION,
        "operation": "ux31_cross_feature_conflict_matrix",
        "status": "PASS",
        "seed": int(seed),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "qt_platform": os.environ.get("QT_QPA_PLATFORM", ""),
        },
        "coverage": {
            "conflict_pairs": len(_cases()),
            "launch_orders": 2,
            "normal_running_cancel_failure_retry_scenarios": len(scheduled),
            "queued_cancel_retry_scenarios": sum(
                1 for case, _reverse in scheduled if case.expected_conflict is not None
            ),
            "timing_based_synchronization": False,
        },
        "result_digest": result_digest,
        "results": results,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    report_path = args.report.expanduser().resolve()
    if report_path.exists():
        print(f"Refusing to overwrite existing report: {report_path}", file=sys.stderr)
        return 2
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = run_matrix(seed=args.seed)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    report_path.write_text(f"{rendered}\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
