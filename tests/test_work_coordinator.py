from __future__ import annotations

from ui.job_manager import JobManager
from ui.work_coordinator import JobSpec, WorkCoordinator


def test_independent_jobs_start_together_and_data_home_writes_queue() -> None:
    manager = JobManager()
    coordinator = WorkCoordinator(manager)
    started: list[tuple[int, bool]] = []

    first = coordinator.submit(JobSpec("Read A", source_locks=("/photos/a",)), lambda job_id, cpu: started.append((job_id, cpu)))
    second = coordinator.submit(JobSpec("Read B", source_locks=("/photos/b",)), lambda job_id, cpu: started.append((job_id, cpu)))
    writer = coordinator.submit(JobSpec("Move data", data_home_write=True), lambda job_id, cpu: started.append((job_id, cpu)))
    other_writer = coordinator.submit(JobSpec("Backup data", data_home_write=True), lambda job_id, cpu: started.append((job_id, cpu)))

    assert {job_id for job_id, _cpu in started} == {first, second, writer}
    assert manager.get(other_writer).text == "Queued: waiting for Data Home write."
    coordinator.finish(writer)
    assert {job_id for job_id, _cpu in started} == {first, second, writer, other_writer}


def test_gpu_conflict_requires_a_choice_then_can_use_cpu_fallback() -> None:
    manager = JobManager()
    coordinator = WorkCoordinator(manager, gpu_policy="ask")
    requested: list[int] = []
    started: list[tuple[int, bool]] = []
    coordinator.gpu_conflict_requested.connect(requested.append)

    first = coordinator.submit(JobSpec("GPU index", uses_gpu=True), lambda job_id, cpu: started.append((job_id, cpu)))
    second = coordinator.submit(
        JobSpec("GPU search", uses_gpu=True, cpu_fallback=True),
        lambda job_id, cpu: started.append((job_id, cpu)),
    )

    assert started == [(first, False)]
    assert requested == [second]
    assert "Choose Queue, CPU fallback, or Cancel" in manager.get(second).text
    coordinator.resolve_gpu_conflict(second, "cpu")
    assert started == [(first, False), (second, True)]
    assert manager.get(second).cache_status == "cpu-fallback"


def test_failed_dependency_cancels_waiting_work() -> None:
    manager = JobManager()
    coordinator = WorkCoordinator(manager)
    first = coordinator.submit(JobSpec("Index"), lambda _job_id, _cpu: None)
    dependent = coordinator.submit(JobSpec("Search", depends_on=(first,)), lambda _job_id, _cpu: None)

    coordinator.finish(first, status="failed", error="index failed")

    assert manager.get(dependent).status == "cancelled"
    assert "Dependency" in manager.get(dependent).error
