from __future__ import annotations

from threading import Event

from ui.job_manager import JobManager
from ui.work_coordinator import JobSpec, WorkCoordinator, prepare_source_scope


class _Signal:
    def __init__(self) -> None:
        self._callbacks = []

    def connect(self, callback) -> None:
        self._callbacks.append(callback)

    def emit(self, *args) -> None:
        for callback in tuple(self._callbacks):
            callback(*args)


class _AsyncJobFixture:
    def __init__(self) -> None:
        self.progress = _Signal()
        self.completed = _Signal()
        self.failed = _Signal()
        self.cancelled = _Signal()
        self.cancel_calls = 0

    def cancel(self) -> None:
        self.cancel_calls += 1


def test_independent_jobs_start_together_and_data_home_writes_queue() -> None:
    manager = JobManager()
    coordinator = WorkCoordinator(manager)
    started: list[tuple[int, bool]] = []

    first = coordinator.submit(JobSpec("Read A", source_locks=("/photos/a",)), lambda job_id, cpu: started.append((job_id, cpu)))
    second = coordinator.submit(JobSpec("Read B", source_locks=("/photos/b",)), lambda job_id, cpu: started.append((job_id, cpu)))
    writer = coordinator.submit(JobSpec("Move data", data_home_write=True), lambda job_id, cpu: started.append((job_id, cpu)))
    other_writer = coordinator.submit(JobSpec("Backup data", data_home_write=True), lambda job_id, cpu: started.append((job_id, cpu)))

    assert {job_id for job_id, _cpu in started} == {first, second, writer}
    assert manager.get(other_writer).text == "Queued: waiting for Data Home access."
    assert manager.get(other_writer).status == "queued"
    coordinator.finish(writer)
    assert {job_id for job_id, _cpu in started} == {first, second, writer, other_writer}
    assert manager.get(other_writer).status == "running"


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


def test_queue_choice_under_ask_is_durable_for_that_job() -> None:
    manager = JobManager()
    coordinator = WorkCoordinator(manager, gpu_policy="ask")
    requested: list[int] = []
    started: list[tuple[int, bool]] = []
    coordinator.gpu_conflict_requested.connect(requested.append)
    first = coordinator.submit(
        JobSpec("GPU index", uses_gpu=True),
        lambda job_id, cpu: started.append((job_id, cpu)),
    )
    second = coordinator.submit(
        JobSpec("GPU search", uses_gpu=True, cpu_fallback=True),
        lambda job_id, cpu: started.append((job_id, cpu)),
    )

    coordinator.resolve_gpu_conflict(second, "queue")
    coordinator._schedule()

    assert requested == [second]
    assert manager.get(second).status == "queued"
    assert "GPU" in manager.get(second).text
    coordinator.finish(first)
    assert started == [(first, False), (second, False)]


def test_cpu_fallback_rechecks_later_source_conflicts() -> None:
    manager = JobManager()
    coordinator = WorkCoordinator(manager, gpu_policy="cpu")
    started: list[tuple[int, bool]] = []
    gpu = coordinator.submit(
        JobSpec("GPU index", uses_gpu=True, source_writes=("/photos/gpu",)),
        lambda job_id, cpu: started.append((job_id, cpu)),
    )
    writer = coordinator.submit(
        JobSpec("Metadata write", source_writes=("/photos/shared",)),
        lambda job_id, cpu: started.append((job_id, cpu)),
    )
    candidate = coordinator.submit(
        JobSpec(
            "GPU search",
            uses_gpu=True,
            cpu_fallback=True,
            source_reads=("/photos/shared/child.jpg",),
        ),
        lambda job_id, cpu: started.append((job_id, cpu)),
    )

    assert started == [(gpu, False), (writer, False)]
    assert manager.get(candidate).status == "queued"
    assert "same source files" in manager.get(candidate).text
    coordinator.finish(writer)
    assert started[-1] == (candidate, True)


def test_capacity_queue_is_bounded_and_cancel_holds_slots_until_finish() -> None:
    manager = JobManager()
    coordinator = WorkCoordinator(manager, cpu_capacity=2, max_pending=2)
    cancelled: list[int] = []
    started: list[int] = []
    first = coordinator.submit(
        JobSpec("Large", cpu_slots=2),
        lambda job_id, _cpu: started.append(job_id),
        cancel=lambda: cancelled.append(first),
    )
    second = coordinator.submit(JobSpec("Second"), lambda job_id, _cpu: started.append(job_id))
    third = coordinator.submit(JobSpec("Third"), lambda job_id, _cpu: started.append(job_id))
    rejected = coordinator.submit(JobSpec("Rejected"), lambda job_id, _cpu: started.append(job_id))

    assert started == [first]
    assert manager.get(rejected).status == "failed"
    assert "queue capacity" in manager.get(rejected).error

    manager.cancel(first)
    assert cancelled == [first]
    assert manager.get(first).status == "cancelling"
    assert started == [first]

    coordinator.finish(first, status="cancelled")
    assert started == [first, second, third]
    assert manager.get(second).status == "running"
    assert manager.get(third).status == "running"


def test_io_capacity_does_not_block_independent_cpu_work() -> None:
    manager = JobManager()
    coordinator = WorkCoordinator(manager, cpu_capacity=4, io_capacity=1)
    started: list[int] = []
    first_io = coordinator.submit(
        JobSpec("Read catalog", io_bound=True),
        lambda job_id, _cpu: started.append(job_id),
    )
    second_io = coordinator.submit(
        JobSpec("Read metadata", io_bound=True),
        lambda job_id, _cpu: started.append(job_id),
    )
    cpu = coordinator.submit(
        JobSpec("Prepare rows"),
        lambda job_id, _cpu: started.append(job_id),
    )

    assert started == [first_io, cpu]
    assert manager.get(second_io).status == "queued"
    coordinator.finish(first_io)
    assert started == [first_io, cpu, second_io]


def test_source_reads_overlap_but_writes_wait_for_nested_scope() -> None:
    manager = JobManager()
    coordinator = WorkCoordinator(manager)
    started: list[int] = []
    first_read = coordinator.submit(
        JobSpec("Read root", source_reads=("/photos",)),
        lambda job_id, _cpu: started.append(job_id),
    )
    second_read = coordinator.submit(
        JobSpec("Read nested", source_reads=("/photos/trip",)),
        lambda job_id, _cpu: started.append(job_id),
    )
    writer = coordinator.submit(
        JobSpec("Rename nested", source_writes=("/photos/trip/image.jpg",)),
        lambda job_id, _cpu: started.append(job_id),
    )

    assert started == [first_read, second_read]
    assert manager.get(writer).status == "queued"
    coordinator.finish(first_read)
    assert started == [first_read, second_read]
    coordinator.finish(second_read)
    assert started == [first_read, second_read, writer]


def test_prepared_source_scopes_detect_symlink_and_hardlink_aliases(tmp_path) -> None:
    real_dir = tmp_path / "real"
    real_dir.mkdir()
    alias_dir = tmp_path / "alias"
    alias_dir.symlink_to(real_dir, target_is_directory=True)
    original = real_dir / "original.jpg"
    original.write_bytes(b"fixture")
    hardlink = real_dir / "hardlink.jpg"
    hardlink.hardlink_to(original)

    manager = JobManager()
    coordinator = WorkCoordinator(manager)
    started: list[int] = []
    directory_job = coordinator.submit(
        JobSpec("Write directory", source_writes=(prepare_source_scope(real_dir),)),
        lambda job_id, _cpu: started.append(job_id),
    )
    symlink_job = coordinator.submit(
        JobSpec("Read alias", source_reads=(prepare_source_scope(alias_dir / "child.jpg"),)),
        lambda job_id, _cpu: started.append(job_id),
    )
    hardlink_job = coordinator.submit(
        JobSpec("Write hardlink", source_writes=(prepare_source_scope(hardlink),)),
        lambda job_id, _cpu: started.append(job_id),
    )
    original_job = coordinator.submit(
        JobSpec("Read original", source_reads=(prepare_source_scope(original),)),
        lambda job_id, _cpu: started.append(job_id),
    )

    assert started == [directory_job]
    assert manager.get(symlink_job).status == "queued"
    coordinator.finish(directory_job)
    assert symlink_job in started
    assert hardlink_job in started
    assert manager.get(original_job).status == "queued"
    coordinator.finish(hardlink_job)
    assert original_job in started


def test_failed_dependency_cancels_waiting_work() -> None:
    manager = JobManager()
    coordinator = WorkCoordinator(manager)
    first = coordinator.submit(JobSpec("Index"), lambda _job_id, _cpu: None)
    dependent = coordinator.submit(JobSpec("Search", depends_on=(first,)), lambda _job_id, _cpu: None)

    coordinator.finish(first, status="failed", error="index failed")

    assert manager.get(dependent).status == "cancelled"
    assert "Dependency" in manager.get(dependent).error


def test_terminal_paths_release_once_and_queued_finish_is_terminal() -> None:
    manager = JobManager()
    coordinator = WorkCoordinator(manager, cpu_capacity=1)
    finished: list[tuple[int, str]] = []
    coordinator.job_finished.connect(lambda job_id, status: finished.append((job_id, status)))
    first = coordinator.submit(JobSpec("First"), lambda _job_id, _cpu: None)
    queued = coordinator.submit(JobSpec("Queued"), lambda _job_id, _cpu: None)

    coordinator.finish(queued, status="failed", error="fixture")
    coordinator.finish(queued, status="failed", error="duplicate")
    coordinator.finish(first)
    coordinator.finish(first)

    assert finished == [(queued, "failed"), (first, "finished")]
    assert manager.get(queued).status == "failed"
    assert manager.get(queued).error == "fixture"
    assert manager.get(first).status == "finished"


def test_model_cache_readers_overlap_and_replacement_waits_for_all_readers() -> None:
    manager = JobManager()
    coordinator = WorkCoordinator(manager)
    started: list[int] = []
    first_reader = coordinator.submit(
        JobSpec("Face inference", model_cache_read=True),
        lambda job_id, _cpu: started.append(job_id),
    )
    second_reader = coordinator.submit(
        JobSpec("Embedding inference", model_cache_read=True),
        lambda job_id, _cpu: started.append(job_id),
    )
    replacement = coordinator.submit(
        JobSpec("Replace model", model_cache_write=True),
        lambda job_id, _cpu: started.append(job_id),
    )

    assert started == [first_reader, second_reader]
    assert manager.get(replacement).text == "Queued: waiting for model cache access."
    coordinator.finish(first_reader)
    assert replacement not in started
    coordinator.finish(second_reader)
    assert started == [first_reader, second_reader, replacement]


def test_queued_cancel_never_starts_and_terminal_signal_is_normalized() -> None:
    manager = JobManager()
    coordinator = WorkCoordinator(manager, cpu_capacity=1)
    started: list[int] = []
    finished: list[tuple[int, str]] = []
    coordinator.job_finished.connect(lambda job_id, status: finished.append((job_id, status)))
    blocker = coordinator.submit(JobSpec("Blocker"), lambda job_id, _cpu: started.append(job_id))
    queued = coordinator.submit(JobSpec("Queued"), lambda job_id, _cpu: started.append(job_id))

    manager.cancel(queued)
    coordinator.finish(blocker, status="not-a-terminal-state")

    assert started == [blocker]
    assert manager.get(queued).status == "cancelled"
    assert manager.get(blocker).status == "failed"
    assert finished == [(queued, "cancelled"), (blocker, "failed")]


def test_cpu_capacity_preserves_fifo_for_an_older_large_waiter() -> None:
    manager = JobManager()
    coordinator = WorkCoordinator(manager, cpu_capacity=3)
    started: list[int] = []
    blocker = coordinator.submit(
        JobSpec("Blocker", cpu_slots=2),
        lambda job_id, _cpu: started.append(job_id),
    )
    older_large = coordinator.submit(
        JobSpec("Older large", cpu_slots=2),
        lambda job_id, _cpu: started.append(job_id),
    )
    newer_small = coordinator.submit(
        JobSpec("Newer small", cpu_slots=1),
        lambda job_id, _cpu: started.append(job_id),
    )

    assert started == [blocker]
    assert manager.get(older_large).text == "Queued: waiting for CPU capacity."
    assert manager.get(newer_small).status == "queued"
    coordinator.finish(blocker)
    assert started == [blocker, older_large, newer_small]


def test_dependency_success_and_failure_have_deterministic_start_order() -> None:
    manager = JobManager()
    coordinator = WorkCoordinator(manager, cpu_capacity=3)
    started: list[int] = []
    first_started = Event()
    dependent_started = Event()

    first = coordinator.submit(
        JobSpec("Index"),
        lambda job_id, _cpu: (started.append(job_id), first_started.set()),
    )
    dependent = coordinator.submit(
        JobSpec("Search", depends_on=(first,)),
        lambda job_id, _cpu: (started.append(job_id), dependent_started.set()),
    )
    failed_dependency = coordinator.submit(
        JobSpec("Broken prerequisite"),
        lambda job_id, _cpu: started.append(job_id),
    )
    cancelled_dependent = coordinator.submit(
        JobSpec("Dependent on failure", depends_on=(failed_dependency,)),
        lambda job_id, _cpu: started.append(job_id),
    )

    assert first_started.is_set()
    assert not dependent_started.is_set()
    coordinator.finish(failed_dependency, status="failed", error="fixture failure")
    assert manager.get(cancelled_dependent).status == "cancelled"
    assert cancelled_dependent not in started
    coordinator.finish(first)
    assert dependent_started.is_set()
    assert started == [first, failed_dependency, dependent]


def test_running_cancel_can_finish_reentrantly_and_releases_once() -> None:
    manager = JobManager()
    coordinator = WorkCoordinator(manager, cpu_capacity=1)
    started: list[int] = []
    finished: list[tuple[int, str]] = []
    cancel_calls: list[int] = []
    coordinator.job_finished.connect(lambda job_id, status: finished.append((job_id, status)))
    holder: dict[str, int] = {}

    def _cancel_running() -> None:
        cancel_calls.append(holder["job_id"])
        coordinator.finish(holder["job_id"], status="cancelled")

    running = coordinator.submit(
        JobSpec("Running"),
        lambda job_id, _cpu: started.append(job_id),
        cancel=_cancel_running,
    )
    holder["job_id"] = running
    queued = coordinator.submit(
        JobSpec("Queued"),
        lambda job_id, _cpu: started.append(job_id),
    )

    manager.cancel(running)
    manager.cancel(running)
    coordinator.finish(running, status="cancelled")

    assert cancel_calls == [running]
    assert started == [running, queued]
    assert finished == [(running, "cancelled")]
    assert manager.get(queued).status == "running"


def test_starter_failure_releases_capacity_and_starts_next_job() -> None:
    manager = JobManager()
    coordinator = WorkCoordinator(manager, cpu_capacity=1)
    started: list[int] = []
    finished: list[tuple[int, str]] = []
    coordinator.job_finished.connect(lambda job_id, status: finished.append((job_id, status)))

    def _fail_to_start(job_id: int, _cpu: bool) -> None:
        started.append(job_id)
        raise RuntimeError("could not create worker")

    failed = coordinator.submit(JobSpec("Fails"), _fail_to_start)
    following = coordinator.submit(
        JobSpec("Following"),
        lambda job_id, _cpu: started.append(job_id),
    )

    assert started == [failed, following]
    assert manager.get(failed).status == "failed"
    assert manager.get(failed).error == "could not create worker"
    assert manager.get(following).status == "running"
    assert finished == [(failed, "failed")]


def test_cpu_fallback_rechecks_data_home_and_model_cache_conflicts() -> None:
    manager = JobManager()
    coordinator = WorkCoordinator(manager, gpu_policy="cpu", cpu_capacity=4)
    started: list[tuple[int, bool]] = []
    gpu = coordinator.submit(
        JobSpec("GPU owner", uses_gpu=True),
        lambda job_id, cpu: started.append((job_id, cpu)),
    )
    data_writer = coordinator.submit(
        JobSpec("Data writer", data_home_write=True),
        lambda job_id, cpu: started.append((job_id, cpu)),
    )
    model_writer = coordinator.submit(
        JobSpec("Model writer", model_cache_write=True),
        lambda job_id, cpu: started.append((job_id, cpu)),
    )
    candidate = coordinator.submit(
        JobSpec(
            "Fallback reader",
            uses_gpu=True,
            cpu_fallback=True,
            data_home_read=True,
            model_cache_read=True,
        ),
        lambda job_id, cpu: started.append((job_id, cpu)),
    )

    assert started == [(gpu, False), (data_writer, False), (model_writer, False)]
    assert manager.get(candidate).text == "Queued: waiting for Data Home access."
    coordinator.finish(data_writer)
    assert manager.get(candidate).text == "Queued: waiting for model cache access."
    coordinator.finish(model_writer)
    assert started[-1] == (candidate, True)


def test_oversized_submission_and_missing_dependency_never_start() -> None:
    manager = JobManager()
    coordinator = WorkCoordinator(manager, cpu_capacity=2)
    started: list[int] = []

    oversized = coordinator.submit(
        JobSpec("Too large", cpu_slots=3),
        lambda job_id, _cpu: started.append(job_id),
    )
    missing_dependency = coordinator.submit(
        JobSpec("Missing dependency", depends_on=(999,)),
        lambda job_id, _cpu: started.append(job_id),
    )

    assert started == []
    assert manager.get(oversized).status == "failed"
    assert "3 CPU slots" in manager.get(oversized).error
    assert manager.get(missing_dependency).status == "cancelled"
    assert "Dependency 999 is unavailable" in manager.get(missing_dependency).error


def test_cancelled_dependency_cancels_waiter_without_consuming_capacity() -> None:
    manager = JobManager()
    coordinator = WorkCoordinator(manager, cpu_capacity=1)
    started: list[int] = []
    dependency = coordinator.submit(
        JobSpec("Dependency"),
        lambda job_id, _cpu: started.append(job_id),
    )
    dependent = coordinator.submit(
        JobSpec("Dependent", depends_on=(dependency,)),
        lambda job_id, _cpu: started.append(job_id),
    )
    independent = coordinator.submit(
        JobSpec("Independent"),
        lambda job_id, _cpu: started.append(job_id),
    )

    coordinator.finish(dependency, status="cancelled")

    assert manager.get(dependent).status == "cancelled"
    assert dependent not in started
    assert started == [dependency, independent]
    assert manager.get(independent).status == "running"


def test_async_job_adapter_starts_after_ownership_and_uses_one_lifecycle() -> None:
    manager = JobManager()
    coordinator = WorkCoordinator(manager, cpu_capacity=1)
    blocker = coordinator.submit(JobSpec("Blocker"), lambda _job_id, _cpu: None)
    job = _AsyncJobFixture()
    starts: list[bool] = []

    job_id = coordinator.submit_async_job(
        JobSpec("Catalog refresh", origin="Library", io_bound=True, data_home_write=True),
        job,
        starts.append,
    )

    assert starts == []
    assert manager.get(job_id).status == "queued"
    assert len([state for state in manager.history() if state.job_id == job_id]) == 1

    coordinator.finish(blocker)
    assert starts == [False]
    assert manager.get(job_id).status == "running"
    job.progress.emit(0, "Opening catalog")
    assert manager.get(job_id).progress == 0
    assert manager.get(job_id).text == "Opening catalog"
    job.completed.emit({"rows": 10})
    job.completed.emit({"rows": 10})
    assert manager.get(job_id).status == "finished"
    assert job_id not in coordinator._running


def test_async_job_adapter_reports_prestart_cancel_and_rejection_without_starting() -> None:
    manager = JobManager()
    coordinator = WorkCoordinator(manager, cpu_capacity=1, max_pending=1)
    blocker = coordinator.submit(JobSpec("Blocker"), lambda _job_id, _cpu: None)
    queued_job = _AsyncJobFixture()
    queued_starts: list[bool] = []
    queued_cancellations: list[bool] = []
    queued_job.cancelled.connect(lambda: queued_cancellations.append(True))
    queued_id = coordinator.submit_async_job(
        JobSpec("Queued"),
        queued_job,
        queued_starts.append,
    )

    coordinator.cancel(queued_id)
    coordinator.finish(blocker)

    assert queued_starts == []
    assert queued_cancellations == [True]
    assert manager.get(queued_id).status == "cancelled"

    rejected_job = _AsyncJobFixture()
    rejection_errors: list[str] = []
    rejected_job.failed.connect(rejection_errors.append)
    rejected_id = coordinator.submit_async_job(
        JobSpec("Oversized", cpu_slots=2),
        rejected_job,
        lambda _fallback: None,
    )

    assert manager.get(rejected_id).status == "failed"
    assert rejection_errors == ["Requires 2 CPU slots, but scheduler capacity is 1."]


def test_async_job_adapter_reports_thread_start_failure_once() -> None:
    manager = JobManager()
    coordinator = WorkCoordinator(manager)
    job = _AsyncJobFixture()
    errors: list[str] = []
    job.failed.connect(errors.append)

    job_id = coordinator.submit_async_job(
        JobSpec("Start failure"),
        job,
        lambda _fallback: (_ for _ in ()).throw(RuntimeError("thread start failed")),
    )

    assert errors == ["thread start failed"]
    assert manager.get(job_id).status == "failed"
    assert manager.get(job_id).error == "thread start failed"


def test_cross_surface_resource_conflict_matrix_matches_production_contract() -> None:
    cases = (
        (
            JobSpec("Clear caches", origin="Settings", io_bound=True, data_home_write=True),
            JobSpec("Refresh timeline", origin="Library", io_bound=True, data_home_read=True),
            False,
        ),
        (
            JobSpec("Install model", origin="Models", io_bound=True, model_cache_write=True),
            JobSpec("Find similar faces", origin="People", uses_gpu=True, model_cache_read=True),
            False,
        ),
        (
            JobSpec("Rename photo", origin="Photos", source_writes=("/photos/a.jpg",)),
            JobSpec("Inspect photo", origin="People", source_reads=("/photos/a.jpg",)),
            False,
        ),
        (
            JobSpec("Rename photo", origin="Photos", source_writes=("/photos/a.jpg",)),
            JobSpec("Inspect other root", origin="People", source_reads=("/archive/b.jpg",)),
            True,
        ),
    )

    for blocker_spec, candidate_spec, can_overlap in cases:
        manager = JobManager()
        coordinator = WorkCoordinator(manager, cpu_capacity=4, io_capacity=4, gpu_policy="queue")
        starts: list[str] = []
        blocker_id = coordinator.submit(
            blocker_spec,
            lambda _job_id, _cpu, label=blocker_spec.label: starts.append(label),
        )
        candidate_id = coordinator.submit(
            candidate_spec,
            lambda _job_id, _cpu, label=candidate_spec.label: starts.append(label),
        )

        assert manager.get(blocker_id).origin == blocker_spec.origin
        assert manager.get(candidate_id).origin == candidate_spec.origin
        assert (candidate_spec.label in starts) is can_overlap
        assert manager.get(candidate_id).status == ("running" if can_overlap else "queued")

        coordinator.finish(blocker_id)
        assert candidate_spec.label in starts
        assert manager.get(candidate_id).status == "running"
