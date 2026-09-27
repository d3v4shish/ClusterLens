# Benchmarks

## 2026-09-26 durable-operation fault inventory (UX-17, partial)

`scripts/verify_durable_fault_matrix.py` validated the machine-readable
`docs/durable_operation_fault_matrix.json` inventory in
`/tmp/clusterlens-ux17-durable-fault-matrix-20260926-g/durable_fault_matrix.json`.
The report covers **15 operations**, **40 commit-boundary records** and **65
unique referenced test nodes**. The isolated run passed **92 tests, 3 warnings
and 103 subtests** in 8.03 s, exited 0 without timeout and left zero PIDs in its
owned process session. Fourteen independent children were terminated by real
`SIGKILL` at pre/post commit checkpoints for seven operations. The report
retains exact pre-crash, crash and twice-recovered state digests for every case;
the production UI regression also reconciles a killed file move and refreshes
Safety & Recovery with its committed restorable state. All verifier checks are
PASS, Linux operation/fault coverage is PASS and no declared operation-level
fault gap remains.

Release qualification is deliberately **PARTIAL**, not PASS: the manifest's
coverage granularity is `operation_fault`, not every boundary/fault Cartesian
cell, actual-kill evidence does not yet cover all 15 operations, and Windows/
macOS are NOT_RUN. This is
deterministic correctness and coverage evidence, not a performance improvement
or cross-platform filesystem claim.

## 2026-09-26 complete serial benchmark rerun

`bash scripts/benchmark.sh` completed every canonical stage serially on the
reference host after the UX-15/31 implementation slices: acceleration,
metadata, thumbnail cache, tags, deep face search, Faces arrangement/index/
review, People Detect, the 100,000-path section gallery, Library catalog/
Timeline, entity picker, UI workflows, theme, Jobs and production workloads.
The final current-candidate production-workload sample recorded 338.818 ms
median cold aligned crops, 260.544 ms warm crops, 331.305 ms mixed completion
and 11.932 ms first content
with deterministic digest `f213b8f4…9435e6`. Five traced cycles retained no
RSS growth and 5,902 bytes of Python allocations. The People sample recorded
571.189 ms first Photos, 469.082 ms first Faces, 43.084 ms cached tail and a
73.930 ms maximum observed offscreen pump gap. The 100,000-path Sectioned
Gallery sample recorded 54.935 ms worker p50, 60.526 ms p95 and a 5.699 ms Qt
commit.

These are current one-host regression samples, not improvement claims and not
native/model/GPU qualification. The canonical script prints each stage report
independently rather than publishing one combined JSON; final sign-off must
retain the per-stage logs with the release-candidate digest.

## 2026-09-26 native-descendant network isolation (UX-33, partial)

`scripts/verify_process_network.py` ran the complete canonical suite beneath
`strace -f -e trace=%network`, while also inheriting the Python socket guard
and Hugging Face/Transformers offline flags. The current report
`/tmp/clusterlens-ux33-process-network-20260926-b/process_network.json` is
**PASS**: **906 tests and 4 warnings** passed in 279.39 s; the audited command
exited 0 after 280.710 s, did not time out, produced trace SHA-256
`27b18aa3…2b09`, recorded zero non-loopback or unparseable IP destinations and
left zero PIDs in its owned process session. Unix sockets, netlink and IPv4/
IPv6 loopback remain allowed for Qt and deterministic local fault servers.

This is a correctness/security audit rather than a benchmark improvement.
Syscall tracing raised the suite wall time materially and is therefore a
release verifier, not the normal test command. It proves the observed Linux
candidate process tree made no external network attempt; it does not impose a
production sandbox or qualify another operating system's tracing boundary.

## 2026-09-25 cross-feature coordinator recovery matrix (UX-31, partial)

`scripts/verify_work_conflicts.py` drives the production `WorkCoordinator`
without sleeps or elapsed-time cancellation. With seed `20260925`, it covered
all six documented conflict pairs in both launch orders: GPU embedding versus
clustering, face-index writes versus identity edits, inference versus model
replacement, overlapping source reads/writes, Data Home backup versus a
managed-store writer, and disjoint read-only browsing. Twelve normal journeys
cancelled running work, injected a failure, and successfully retried; ten
conflicting journeys cancelled queued work and successfully retried. Every
scenario ended with zero queued/running resources. The focused scheduler and
verifier set passed **25 tests**; standalone report
`/tmp/clusterlens-ux31-conflicts-20260925-a.json` passed with result digest
`d302c41083a0f595b88922a28c35115bdbb1b5bec9c839d1c6ce23845e38fea0`.

This is deterministic scheduler/recovery evidence, not a performance result.
The next offscreen production-shell slice used four generated 3840×2160 JPEGs
(4,687,872 bytes, manifest `96de2d3b…1488e4`) and three coordinated workers:
full-resolution aligned crops, thumbnail decode and catalog scan/query. While
they were active it traversed all four workspace groups three times, alternated
1280×720/1920×1080, switched dark/light, changed scroll positions, opened Jobs
with Ctrl+J and cancelled the foreground thumbnail job through the visible
footer control. Crop/catalog digests matched serial execution; cancellation
produced no thumbnail result; final Jobs/coordinator ownership was empty.
`tests/test_mixed_ui_workload.py` passed in 7.05 s with profiled report
`/tmp/clusterlens-ux31-mixed-ui-20260925-c.json`.

The first run exposed a qualification failure rather than hiding it: full
production-shell theme changes took 241.031 and 277.237 ms, the heartbeat gap
reached 271.633 ms, and `QApplication.setStyleSheet()` consumed 208.322 ms of a
237.386 ms profiled switch. The theme stylesheet now resolves semantic colors
through `QPalette` roles and remains unchanged across dark/light switches;
density and text scale still rebuild it because they alter geometry. The
focused theme matrix added a contract that a changed theme updates the palette
without replacing the global stylesheet.

The final workload activates the generated root and preloads production crop
dependencies outside the measured interval just like the UX-28 driver. A
four-process check first exposed that byte-identical JPEGs inherited wall-clock
mtimes and could produce two catalog order digests. The fixture now assigns a
fixed epoch plus image index and tests both byte manifest and mtimes. Final
fresh-process reports `/tmp/clusterlens-ux31-mixed-ui-20260925-{o,p}.json` have
identical crop digest `16df2456…c6f5c` and catalog digest `34640284…081c6`.
Cold-route maxima were **8.558/8.319 ms**, active UI-action maxima
**21.172/20.528 ms**, and heartbeat maxima **21.094/20.462 ms**; both ended
with zero active jobs and zero queued/running resources. The earlier isolated
final-theme report recorded changed themes at **0.854–6.855 ms**, Jobs opening
at **6.695 ms**, and visible cancellation at **2.022 ms**. The
profiled post-work theme call fell from 237.386 ms to about 6 ms; `setPalette`
used about 0.14 ms and no global stylesheet rebuild occurred. The deterministic
250-iteration microbenchmark measured **0.0789 ms median / 0.0904 ms p95**
theme changes and **0.0246/0.0399 ms** no-op reapplication. The populated
dark/light 200% verifier passed at both logical sizes under
`/tmp/clusterlens-ux31-palette-ui-20260925-b`.

The production-shell case was then run by
`scripts/verify_mixed_ui_lifecycle.py` in its own Linux process session. The
current wrapper invokes the locked interpreter directly with isolated
runtime/settings/bytecode paths and offline guards rather than requiring a
writable global uv cache.
`/tmp/clusterlens-ux33-mixed-lifecycle-20260925-b/lifecycle.json` is **PASS**:
the inner mixed workload passed, the test process exited 0 after 6.976 s, and
`/proc` contained zero surviving PIDs in its owned session after close.
Process-session cleanup proves the test process, its threads and inherited
descendants are gone; it is not native compositor or cross-platform evidence.

This closes the measured offscreen CPU/theme responsiveness defect, not all of
UX-31. The qualifier excludes native input/compositor behavior, real model/GPU
work and module/model-load responsiveness, and an approved photo corpus. The complete
offline-isolated post-change suite passed **843 tests with 4 warnings in 143.92
s**.

## 2026-09-26 source-only offline reproducibility (UX-33, partial)

`scripts/verify_clean_source.py` exported all current tracked and non-ignored
untracked candidate files into two independently created trees, without
`.git`, ignored cache state or an existing virtual environment. From different
outside working directories, each tree ran frozen `uv` resolution with
`UV_OFFLINE=1`, the canonical build, `run.sh --check-launch`, and the complete
suite. Both source digests must remain identical before/after; every command
must exit 0 and every owned Linux process session must end with zero PIDs. The
current report and retained logs are under
`/tmp/clusterlens-ux33-clean-source-20260926-final5`. It is **PASS**: the exact
275-file candidate digest is
`24cbd3ffbc6a4b5a18937e34a1d2c9e5d9b04ba8f17d5e5d79fa27f4c8d01057`;
both independently exported trees built and launch-checked from outside their
tree, then passed **906 tests and 4 warnings** in 143.33 s and 137.36 s. Every
command exited 0, both source digests were unchanged and every owned process
session drained. UX-33 still requires the separate fresh committed-clone gate
before it can close.

This proves the recorded candidate does not need repository metadata, an
existing `.venv`, the caller's working directory, mutable settings/runtime
state or online package resolution. The offline run intentionally consumes a
pre-populated host uv cache. Because the candidate is still an uncommitted
worktree export, a fresh committed checkout and installer lifecycle remain
release gates; this evidence does not claim installer reproducibility.

The subsequent model-provenance slice pins managed MobileCLIP, DINO, DINOv2,
CLIP, OpenCLIP and SigLIP snapshot downloads plus their online runtime loaders
to the same reviewed 40-character commits. Cache-only loading continues to
consume its recovered local snapshot, preserving offline recovery. The focused
online-pin/offline-recovery set passed **10 tests, 209 deselected and 3
warnings**. The current complete canonical suite passed **906 tests with 4
warnings in 155.14 s**, and `scripts/build.sh` exited 0. The current isolation
report `/tmp/clusterlens-ux33-test-isolation-20260926-final4/test_isolation.json`
is **PASS**: repository-order and reverse `/tmp` runs each passed **421 tests,
4 warnings and 52 subtests** in 106.53 s and 109.59 s; source
aggregate/status/diff digests were
unchanged and both Linux process sessions had zero survivors. No network
download or real inference ran in those tests, so model-output qualification
remains UX-30 work.

## 2026-09-25 native Linux UI matrix (UX-32, partial)

The current-code native verifier passed all 16 visual/keyboard cells across
Wayland and X11/XWayland, logical sizes 1280×720 and 1920×1080, and text scales
100/125/150/200%. Each report covers 18 populated routes, Roots
Sources/Catalog, all 10 Settings sections, requested client geometry, clipping,
keyboard traversal, screenshots and clean shutdown. Wayland recorded DPR 2.0;
the X11/XWayland cells recorded fractional DPR 1.4479166667 and passed
`--require-fractional-scale`. Reports are under
`/tmp/clusterlens-ux32-native-1280x720-200-20260925-b`,
`/tmp/clusterlens-ux32-wayland-*-20260925-c` and
`/tmp/clusterlens-ux32-x11-*-20260925-a`.

Visual inspection of the first 1280×720/200% Wayland run found a vertically
clipped Settings search field that the old control probe did not reject. The
shared theme now scales single-line editor minimum height, and the verifier has
a `QLineEdit` negative control. The corrected hardest Wayland and X11 cells
were visually inspected. A current normal-path regression at
`/tmp/clusterlens-ux32-native-transition-regression-20260925-b/native_display.json`
also passed all 18 routes, all 10 Settings sections, visual/keyboard checks and
clean shutdown after the verifier gained an opt-in multi-screen transition
gate. Qt exposes this host's screens as 2648×1490 and 1080×1920 logical, so
only one can contain the documented 1280×720 minimum; multi-monitor movement
is NOT_RUN rather than PASS here. An Orca 50.2 trial observed AT-SPI events but
did not reliably speak ClusterLens controls, so it is also not screen-reader
qualification. This matrix is correctness evidence, not a latency or
throughput result; screen reader, two-fitting-screen movement, active
inference/decode and non-Linux qualification remain open.

## 2026-09-25 local Linux CPU package smoke (UX-34, partial)

The current source built as a Linux CPU onedir artifact and passed the frozen
shell/Settings/Storage/icon smoke at
`/tmp/clusterlens-ux34-package-20260925-b/smoke-v2/packaged_launch_verifier.json`.
The executable SHA-256 is
`507011e566b42145147d939401d08986a64d829573f7c4889a379ed3c0a7bbd2`.
The version 2 whole-tree digest is
`247d1f840dbabd012842d5084dd5aae9bd5f0f4d89d2c3c9df5873a2d7355cd6`
over 5,129 files, 80 symlinks and 1,743,520,839 bytes. The digest includes
sorted relative paths, entry kinds, user-executable mode, sizes, file content
and symlink targets. No packaged process survived the smoke.

This is local artifact identity and launch evidence, not a performance
improvement or reproducible-installer claim. The build environment still uses
range-based package requirements; clean-VM install/upgrade/uninstall, signing,
GPU-package provider checks and locked-build reproducibility remain open.

## 2026-09-25 real-model qualification preflight (UX-30, incomplete)

The opt-in `scripts/qualify_real_models.py` gate now verifies manifest/fixture/model SHA-256 pins and provenance fields before importing inference services; rejects incomplete detector/embedder/backend coverage; runs cold, warm and recreated-service detection/embedding; checks dimensions, finite values, ordering and declared numeric tolerances; executes every declared clustering backend twice; and records actual provider evidence plus peak RSS. It never downloads models. Missing assets or an unavailable explicit CUDA provider are `NOT_RUN` (exit 3), never a generated substitute for inference. `tests/test_face_model_download_http.py` uses only a localhost server to verify interrupted Range resume, verified-cache reuse, corrupt checksum rejection and cancellation-safe partial retention. The production downloader now treats a truncated HTTP body/`IncompleteRead` as resumable instead of deleting the partial after a misleading final checksum failure.

The focused UX-30 preflight passed **20 tests, 203 deselected, 3 warnings and 2 subtests** in 7.68 s. It covers the local HTTP matrix, managed promotion/cache-only recovery, interrupted promotion boundaries, incomplete-bundle rejection, stale embedding revision invalidation, explicit CUDA provider/fallback policy and no implicit FaceNet download. The complete canonical suite then passed **831 tests with 4 warnings in 148.72 s**, and the canonical build exited 0. `/tmp/clusterlens-ux30-not-run-20260925-a.json` exited **3 / NOT_RUN** because the host has no approved fixture/model directory; no detector, embedder or clustering case executed. CUDA/model throughput, provider memory, OOM behavior on real assets and complete advertised component coverage therefore remain release blockers, not passes.

## 2026-09-25 real CPU/CUDA warm-worker resource qualification (UX-15)

`scripts/qualify_warm_worker.py` copied the checksum-verified 3,717,015-byte Fast Preview ONNX asset into an isolated runtime, generated four fixed 320×240 PNGs, and ran the production persistent clustering worker twice with cache reads/result reuse disabled. Each cycle performed real ONNX embedding and cosine K-means, remained idle after completion, then exited only after warm mode was disabled. The next cycle required a different PID and identical root-relative membership. Missing or checksum-invalid assets return `NOT_RUN`; the verifier never downloads a replacement.

`/tmp/clusterlens-ux15-warm-worker-cpu-20260925-b.json` passed. Both cycles used `CPUExecutionProvider`; model load was 0.712/0.666 s, inference 0.013/0.009 s and complete workload 2,173.473/1,953.567 ms. Idle/peak RSS was 610,033,664 and 610,029,568 bytes. Shutdown took 471.449/344.840 ms; both PIDs were absent before restart, and membership digest `7e5d6dbc…d02a34f0` matched. This establishes the CPU half of the lifecycle measurement.

The subsequent CUDA run closes that lifecycle half. The same checksum-verified
asset and fixture ran under `.venv-gpu-cu121` twice; both cycles selected
`CUDAExecutionProvider`, used distinct worker PIDs, and produced the same
membership digest as CPU. Model load was 0.886/0.683 s, inference was
0.550/0.383 s, and per-process VRAM observed after the workload while the
worker intentionally remained warm was 710,934,528 bytes in both cycles.
Disabling warm mode stopped each worker in 430.696/451.177 ms; its exact PID
was then absent from both `/proc` and the NVIDIA compute-process table. Report:
`/tmp/clusterlens-ux15-warm-worker-cuda-20260925-a.json`. This proves lifecycle
release for the measured process; it does not assert that a live CUDA process
must report zero allocator reservation.

## 2026-09-25 aligned full-resolution crop optimization (UX-29)

Method: before changing product code, the UX-28 driver was tightened from unaligned box crops to the five-landmark alignment path used by production detectors. `bash scripts/benchmark.sh --production-workloads-only --report /tmp/clusterlens-ux29-aligned-baseline-20260925-a.json` is the five-process untraced baseline plus separate trace/profile runs. The identical optimized command wrote `/tmp/clusterlens-ux29-aligned-final-20260925-a.json`. Both use the same eight generated 3840×2160 JPEGs, 32 aligned crops, fixed seed/configuration, empty per-process application caches, offscreen Qt contention workload and reference host documented below. The pre/post crop and thumbnail hashes are byte-identical (`015388b4…989b6d` and `d4063208…c93523`); both catalog runs contain the same eight fixture members. The final driver hashes catalog membership relative to the generated root, making future reports comparable across independent temporary directories.

| Measurement | Before samples (ms) | After samples (ms) | Median change |
| --- | --- | --- | --- |
| Cold 32 aligned crops | 1124.832, 1098.347, 1119.084, 1099.525, 1082.736 | 327.133, 336.849, 339.721, 339.003, 340.140 | 1099.525 → 339.003 ms (**69.2% lower**) |
| Warm 32 aligned crops | 971.621, 966.098, 987.895, 967.797, 944.285 | 263.393, 260.928, 268.321, 268.569, 273.104 | 967.797 → 268.321 ms (**72.3% lower**) |
| Mixed three-worker completion | 1127.236, 1125.298, 1094.755, 1094.350, 1095.233 | 327.067, 336.117, 334.598, 328.171, 337.199 | 1095.233 → 334.598 ms (**69.4% lower**) |
| Mixed first thumbnail content | 11.386, 11.730, 11.424, 11.244, 12.895 | 11.830, 12.127, 11.276, 11.335, 13.323 | 11.424 → 11.830 ms (**3.6% higher; retained as run variance/tradeoff**) |

The change lazily materializes one full-frame RGB NumPy buffer for each detector/image pass and reuses it for that image’s aligned crops. It does not retain a full-photo cache or change worker/cache limits. Median whole-process CPU fell from 3.444 to 1.154 seconds (**66.5% lower**). Absolute peak RSS changed from 780,693,504–781,881,344 bytes to 756,248,576–757,440,512 bytes; because native allocator/import state dominates this value, it is reported as a range rather than attributed entirely to the code change. The final traced run peaked at 51,933,946 Python bytes and retained 1,638,469 bytes. Five additional traced crop cycles retained 1,630,971, 1,633,329, 1,635,377, 1,636,625 and 1,637,159 bytes (6,188-byte first-to-last growth), while current RSS remained exactly 676,040,704 bytes after every cycle. The full-frame buffer is therefore released after each batch instead of retaining one generation per cycle.

The final mixed runs kept three workers, a 32-item thumbnail memory cache and one full-frame crop buffer in the crop worker. Per-run callback p95 remained 0.026–0.031 ms, posted-event input p95 0.018–0.022 ms and pump p95 1.610–1.792 ms. Maxima were 6.492, 1.379 and 12.816 ms respectively; cancellation acknowledgement/safe-drain medians were 0.026/0.029 ms. Physical I/O remained page-cache-dependent; the final report additionally records 49.9–50.1 MB of logical reads per full process. The profile moved from 2.193 s cumulative in repeated aligned-crop work and 1.390 s of repeated conversion to 0.933 s for the complete crop passes, with JPEG decode/color conversion now the residual cost.

The production-capacity People fixture was then extended with a third, separately profiled workflow. Before the UI change, its untraced 14,477-face publication was **1,551.424 ms** and the profile recorded 151 batch flushes plus 154 action/readiness refreshes; repeated face-service pipeline application and detector-bundle resolution contributed about 0.92 profiled seconds. Action eligibility depends on the captured complete review scope, selection and publication state, so recomputing it after every append did not change the contract. Refreshing actions at publication start/finalization only reduced the identical untraced publication to **400.848 ms (74.2% lower)**. The final profile records three action refreshes, 0.522 s cumulative in the 151 progressive batch flushes, 0.379 s building 14,477 tile records and 0.055 s appending them. First Photos (779.843 ms), metadata completion (581.854 ms), cached tail (43.274 ms), cancellation (5.533 ms), 76.168 ms maximum pump gap, 30,024,723 retained Python bytes and exact 500→1,000 paging remain inside the existing watch budgets; no rows, page limits or memberships changed.

The 100,000-path Sectioned Gallery fixture now also traces six alternating full/filtered model replacements with five/six-column reflow. Full-model retained Python samples were 20,318,348, 19,751,200 and 19,751,328 bytes (last minus first **−567,020**); filtered samples were 9,872,712, 10,156,736 and 9,873,096 bytes (**+384**), with a 36,345,908-byte traced peak. The final real-widget publication preserved selection, committed in 3.238 ms and completed worker-to-UI in 62.129 ms. This proves the measured model generations plateau instead of retaining every replacement; native/QImage cache memory remains outside `tracemalloc` and retains its separate bounded-cache contract.

The initial box-only UX-28 watch budgets were revised because they were not representative of landmark alignment: aligned-crop reference budgets are now cold ≤425 ms, warm ≤340 ms, mixed completion ≤425 ms, callback p95/max ≤1/10 ms and pump p95/max ≤5/20 ms. Mixed first content remains ≤25 ms, cancellation acknowledgement/drain ≤10 ms, absolute process peak RSS ≤800 MiB and five-cycle RSS growth must remain zero on this reference fixture. These are regression-watch values, not correctness synchronization. A follow-up attempt to omit Pillow’s explicit RGB copy did not improve the repeated timings and raised one five-run RSS range to 798.0–798.7 MB; it was discarded rather than hidden.

## 2026-09-25 full-resolution and mixed-workload baseline (UX-28)

Method: `bash scripts/benchmark.sh --production-workloads-only --report /tmp/clusterlens-ux28-production-workloads-20260925-a.json` generated eight fixed 3840×2160 JPEGs (66,355,200 pixels and 9,642,426 bytes total, seed `20260925`, corpus SHA-256 `3c716688…b9fb36e`) and ran five fresh untraced application processes followed by separate allocation-traced and CPU-profiled processes. Every child used an empty runtime/settings/cache. This first pass measured production `ThumbnailService` decode/cache, 32 unaligned box crops plus quality metrics, warm repeats and cooperative cancellation. The contention phase ran thumbnail decode/cache, full-resolution crop/quality, and catalog scan/query in three bounded workers while an offscreen Qt loop measured timer callbacks, synthetic posted-event input round trips and pump gaps. “Cold” means empty application caches, not a dropped OS page cache. The fixture performs no detector/embedder inference and uses no GPU; it must not be labelled model throughput. Before product optimization, the corrected aligned-crop baseline above superseded the box-crop timings for detector-path decisions.

Reference environment: AMD Ryzen 7 7800X3D (16 logical CPUs), Linux 7.0.0-31 x86-64, Python 3.12.13, PyQt 6.11.0, Pillow 10.2.0, NumPy 1.26.4, Torch 2.2.2 with a CUDA 12.1 build but `torch.cuda.is_available() == false`, ONNX Runtime 1.24.4, Qt offscreen at DPR 1.0 / 96 DPI. GPU execution, memory and driver are `NOT_USED` / `NOT_AVAILABLE`, not PASS.

| Untraced phase | Five fresh-process samples (ms) | Median / range (ms) |
| --- | --- | --- |
| Cold thumbnail completion | 78.908, 77.270, 78.001, 77.650, 77.218 | 77.650 / 77.218–78.908 |
| Warm thumbnail-cache completion | 1.633, 1.627, 1.620, 1.631, 1.696 | 1.631 / 1.620–1.696 |
| Cold 32 box-crop completion | 270.608, 283.070, 275.034, 283.645, 283.404 | 283.070 / 270.608–283.645 |
| Warm 32 box-crop completion | 259.450, 256.903, 260.154, 257.423, 254.523 | 257.423 / 254.523–260.154 |
| Mixed first thumbnail content | 12.398, 11.416, 11.164, 12.204, 14.521 | 12.204 / 11.164–14.521 |
| Mixed three-worker completion | 285.745, 279.496, 279.773, 267.595, 276.893 | 279.496 / 267.595–285.745 |
| Cancellation acknowledgement | 0.034, 0.029, 2.553, 0.033, 0.034 | 0.034 / 0.029–2.553 |
| Cancellation safe drain | 0.037, 0.032, 2.556, 0.035, 0.038 | 0.037 / 0.032–2.556 |

Each mixed run collected 226–237 UI samples. Per-run timer-callback p95 was 0.027–0.032 ms (maximum 0.148–1.828 ms), posted-event input p95 was 0.019–0.020 ms (maximum 0.100–0.248 ms), and event-pump p95 was 1.226–1.273 ms (maximum 2.473–2.969 ms). These are offscreen synthetic event-loop values, not native input/compositor latency. The three-worker phase consumed 0.365–0.387 CPU-seconds and submitted exactly three concurrent workloads. Whole-process CPU was 1.040–1.060 seconds; absolute peak RSS was 712,228,864–717,119,488 bytes and final-minus-initial RSS was 17,702,912–83,267,584 bytes. Physical read/write counters were zero because the already-generated corpus remained in the host page cache; the report retains that result rather than calling it disk-cold I/O. The separate traced process retained 1,685,022 Python bytes and peaked at 5,603,525 bytes; its 713,240,576-byte process RSS is reported separately because native libraries and image buffers are not Python allocations.

No aggregate p95 is published from five process samples. The report includes every sample, deterministic component/result hashes, dependency/display/cache state, queue depths, CPU/RSS/I/O, traced allocations and profiled call sites. The box-crop profile identified the crop pass (0.871 s cumulative across three invocations), Pillow color conversion (0.761 s) and JPEG decoder work (0.439 s) as material costs; the aligned baseline above then established the production detector path and its revised watch budgets. Any budget change requires an identical-fixture rerun and a recorded reason.

The retained companion fixtures were rerun on the same checkout. `--people-detect-workflow-only` published first Photos in 771.625 ms, Faces in 1,561.034 ms, completed background metadata in 598.849 ms, revealed the cached tail in 43.112 ms, acknowledged cancellation in 5.559 ms, observed a 79.455 ms maximum pump gap, retained 30,169,525 Python bytes and peaked at 31,400,479 bytes. It remains a generated 1px/SQLite metadata and Qt-publication fixture, not full-media or inference evidence. `--sectioned-gallery-only` produced five worker samples of 61.893, 56.443, 60.501, 53.353 and 53.421 ms (56.443 ms p50 / 61.893 ms p95), a 6.229 ms Qt commit, 64.457 ms worker-to-UI completion and 73.155 ms maximum observed pump gap for 100,000 in-memory paths. It remains source-I/O-free.

## 2026-09-25 locked test-runner and outside-workdir validation

Method: after adding the exact `pytest==9.1.1` development dependency to `pyproject.toml`/`uv.lock` and removing mutable `uv run --with pytest` resolution from `scripts/test.sh`, the canonical scripts were invoked by absolute path with `/tmp` as the working directory. `bash /home/d3v/Workspace/Desktop/ClusterLens-clean/scripts/build.sh` completed with the frozen lockfile. `bash /home/d3v/Workspace/Desktop/ClusterLens-clean/scripts/test.sh` ran the complete isolated suite, and `bash /home/d3v/Workspace/Desktop/ClusterLens-clean/scripts/benchmark.sh --jobs-history-only --rows 500 --updates 50` ran a bounded generated Qt fixture. No command read user photos or an existing ClusterLens runtime.

The outside-workdir build exited 0 and the complete test process passed **788 tests with 4 warnings in 121.11 s**. The Jobs fixture measured **0.189 ms** for its initial refresh and **0.159 ms p50 / 0.197 ms p95** for 50 changed-row refreshes, with a **260 px** Details column. This is reproducibility and component-regression evidence, not a performance improvement, clean-checkout/package certification, or worker/storage/paint measurement. The canonical launcher was exercised separately through the native application verifier; its source-runtime selection path was not certified from a fresh checkout by this benchmark.

The canonical runner now additionally forces the Hugging Face/Transformers offline variables and prepends `tests/network_guard` to `PYTHONPATH`. Its test-only `sitecustomize.py` is therefore inherited by Python workers and subprocesses, blocks external DNS plus direct non-loopback connections, and permits loopback. The complete isolated suite passed **836 tests with 4 warnings in 269.48 s** under that guard. A subsequent three-probe child-process contract rejected `example.com` DNS and `203.0.113.1:443`, while the three localhost download interruption/checksum/cancellation tests remained green (**6 passed** combined). The longer suite duration is reported as observed and is not attributed to the guard without repeated profiling.

## 2026-09-24 local release-gate rerun

Method: `bash scripts/build.sh`, `bash scripts/benchmark.sh`, `.venv/bin/python -B scripts/verify_ui_redesign.py --report-dir /tmp/clusterlens-production-ui-20260924-303`, `.venv/bin/python -B scripts/verify_native_display.py --report-dir /tmp/clusterlens-native-display-20260924-303 --logical-size 1280x720`, and the rebuilt CPU packaged-launch verifier each used the documented source tree or a fresh isolated runtime/profile. The generated benchmark does not read user media or mutable user data; the UI/native/package checks use empty runtime roots.

The build, generated benchmark suite, offscreen UI report, native 1280x720 report, and rebuilt CPU package smoke all passed. The benchmark's fixed 20,000 × 128 clustering fixture measured **423.784 ms** median for explicit CPU MiniBatch K-means and **113.040 ms** for the selected CUDA path. Its fixed 10,000 × 32 HDBSCAN fixture measured **377.706 ms** explicit CPU and **361.994 ms** selected-runtime fallback: the host has CUDA for the selected K-means path but lacks `cuml`, so the report correctly records native HDBSCAN plus the fallback reason. The generated 48-image CPU face-index scheduler measured **78.447 ms** median after warm-up with its bounded four decode workers, two quality workers, and 16-image write batches. These are one-host synthetic baselines, not a clean-VM, GPU-package, source-media, or inference-throughput claim.

## 2026-09-24 Jobs history incremental-refresh baseline

Method: `bash scripts/benchmark.sh --jobs-history-only` creates 500 generated in-memory job records, retaining 125 active rows and 375 terminal rows. It opens the real offscreen `JobsDialog` at 1280×720, asserts the 500-row history and minimum readable Detail column, then drives 100 `JobManager.update()` signals against changed active rows. The fixture opens no source media, database, models, cache, settings, or network connection.

The initial no-op refresh was **0.190 ms**. Changed-row refresh measured **0.157 ms p50** and **0.170 ms p95**, with a **654 px** Detail column. This is a Qt table/component baseline only; it does not measure worker execution, storage, compositor paint, or a screen reader.

## 2026-09-24 Sectioned Gallery worker-publication baseline

Method: `bash scripts/benchmark.sh --sectioned-gallery-only` generated 100 sections containing 1,000 unique in-memory path strings each. It has no files, thumbnails, databases, models, cache, network access, user runtime, or randomized input. Five pure `SectionedGalleryModel.prepare_sections()` samples measure normalized hierarchy/row construction at five columns. A separate offscreen `SectionedGallery` run measures the owned worker through the single Qt model-reset commit, asserts stable selection over a hierarchy replacement, and records the largest interval between deliberate Qt event-pump calls. The adjacent deterministic 12-test gallery module also exercises stale-result discard, cancellation/shutdown, header spans, viewport bounds, and a 2,200-path resize reflow that keeps selection.

The five worker samples were **58.138, 56.731, 59.654, 52.779, and 54.818 ms** (**56.731 ms p50**, **59.654 ms p95**). The real-widget run reached the initial published model in **77.239 ms**, with a **6.104 ms** Qt-side commit, **58.440 ms** maximum observed offscreen event-pump gap, 16,800 rows at six columns, and selection preserved through replacement. The profiled single preparation spent 120.445 ms under profiler instrumentation, principally 67.806 ms normalizing paths and 28.555 ms forming rows; profiler overhead makes that number non-comparable with the unprofiled samples. This is a component responsiveness baseline, not a source-media, thumbnail, paint, or native-compositor claim.

## 2026-09-24 People Detect Qt workflow capacity baseline

Method: `bash scripts/benchmark.sh --people-detect-workflow-only` ran the real offscreen `SearchPane` workflow twice from fresh temporary directories. Each deterministic fixture writes 11,284 identical 1px PNG source files and seeds 14,477 visible face rows in a private SQLite DB. The untraced workflow measures the initial Photos publication, controlled background-cache completion, explicit Faces-tab virtual-model publication, one cached Photos-tail reveal, and cancellation acknowledgement. A separate traced workflow measures retained/peak Python allocation through metadata-cache completion, so tracing cannot distort the published timing values. It never reads user runtime data or network resources and deliberately excludes model inference, embedding, full-resolution thumbnail/crop decoding, and native-display paint.

The untraced run published the bounded first **500 Photos in 575.856 ms**, read/prepared the remaining review metadata after release in **566.021 ms**, published all **14,477 Faces tiles in 1,189.821 ms** after the user opened Faces, and revealed one cached tail page in **42.908 ms** without an additional SQLite page query. Cancellation of a deliberately gated page was acknowledged in **5.524 ms** and left exactly the initial 500 Photos with zero hidden Faces rows. The largest observed offscreen event-pump interval was **75.044 ms**. The separate traced run retained **30,031,853 bytes** at cache completion and reached a **31,262,770-byte** Python peak. The self-validating fixture checks every source-order page offset `0` through `11000`, 11,284 prepared review images, 14,477 model rows, tail/cache separation, and stale-generation cancellation. These are one-host offscreen component baselines, not a claim about inference, full-media decode, or native compositor paint performance.

## 2026-09-24 People Detect candidate-snapshot paging

Method: `bash scripts/benchmark.sh --face-review-paging-only --photos 11284 --faces 14477 --repeats 5` created five fresh, temporary SQLite fixtures. Each fixture has 11,284 source-ordered photo paths and 14,477 deterministic face rows; it reads no source image, thumbnail, crop, model, network resource, user runtime data, or Qt widget. Every repeat measures the former page-by-page canonicalization path and the candidate-snapshot path on separate fresh databases, then verifies all 23 offsets from `0` through `11000` and all 11,284 returned review items. A separate `tracemalloc` pass measures the live Python allocation for one snapshot and is deliberately outside the timed repetitions.

The baseline's first page was **223.292 ms** median and its complete cache read was **5,134.527 ms** median. With one scoped, symlink-safe snapshot prepared per review generation, the first page was **225.710 ms** median (the required one-time validation remains on the first request) and the complete read was **253.105 ms** median—**95.1% lower** for the complete metadata-cache path. Snapshot preparation was **210.028 ms** median; its retained Python allocation was **1,764,104 bytes** with a **2,648,135-byte** traced peak. The optimized first-page profile is now dominated by the bounded page's review/scan-state work and normalized scope comparisons, not repeated whole-review `Path.resolve()` calls. The scoped/outside-path and symlink-escape regressions pass. This is a component measurement, not an end-to-end Detect, paint, thumbnail, crop, model, or event-loop responsiveness claim.

### Automated release evidence

The documented generated-fixture benchmark suite completed successfully on the same checkout. The CPU production package was built from the existing CPU build environment, and `verify_packaged_launch.py` passed against its fresh isolated runtime. `verify_native_display.py` passed with report `/tmp/clusterlens-native-display-20260924/native_display.json`; the offscreen 1280×720 and 1920×1080 shell report passed at `/tmp/clusterlens-production-ui-20260924-final/ui_redesign.json`. Two independent full test processes passed all 649 tests in 159.88 s and 162.55 s. These are automated local evidence only; clean-VM upgrade/uninstall and GPU-package validation remain separate release checklist items.

## 2026-09-21 repeat UX and performance audit baseline

Method: `QT_QPA_PLATFORM=offscreen .venv/bin/python scripts/verify_ui_redesign.py --report-dir /tmp/clusterlens-ux-pass-20260921` created a fresh runtime and captured the Library, Organize, People, Tools, Settings, Jobs, and Inspector at 1280×720 and 1920×1080. It passed with no reported clipped controls; empty-shell construction was 290.881 ms and the 1280×720 Library first-visible sample was 2.607 ms. The short native verifier also passed at 1920×1080 from a separate temporary runtime. These layout-only checks intentionally contain no media, catalog rows, models, or user data.

The complete isolated benchmark suite was rerun. Relevant current baselines are: 10,000 generated virtual-list first content 0.499 ms p50 / 0.731 ms p95; All Faces' two-group initial snapshot plus first 500 members 11.784 ms median on 5,000 seeded face rows, where SQLite execution was 9.740 ms in the profile; 4,096-face sectioned-model publication 21.902 ms median; and the 1,001-image Folder Review service read 28.498 ms first-page / 77.007 ms complete-page median. These are host-specific component baselines only. They do not measure a live 11,284-photo / 14,477-face Detect cache, Qt paint/event-loop stalls, retained cache memory, thumbnails/crops, source discovery, or inference; that missing fixture is tracked in `TODO.md`.

## 2026-09-20 readable-adaptive implementation slice

### People Detect viewport paging follow-up

Method: before this change, `bash scripts/benchmark.sh --face-review-paging-only` measured the existing bounded SQLite page reader. The same deterministic command was rerun after replacing the UI's timer-driven cumulative gallery republishes with tail-triggered page appends. It creates five fresh temporary SQLite fixtures with 1,001 seeded image/face rows, no media files, models, network, or user runtime data; it verifies source-order pages `0`, `500`, and `1000`. Focused offscreen coverage verifies initial Detect routing, no timer-driven second request, tail-only page requests, in-place gallery growth, and post-commit index observer batches.

The post-change samples were 30.778, 29.814, 29.865, 35.196, and 28.975 ms for first-page retrieval (29.865 ms median), and 77.735, 80.103, 77.751, 79.515, and 78.374 ms for all three service pages (78.374 ms median). The earlier service baseline was 32.065 ms first page / 87.156 ms all pages under the same temporary seeded workload. This run does not measure Qt publication, thumbnails, scrolling, source discovery, or inference, so it makes no end-to-end UI throughput claim.

### People Detect background-cache capacity check (2026-09-21)

Method: `bash scripts/benchmark.sh --face-review-paging-only --photos 10001 --repeats 3` used three fresh temporary SQLite fixtures with one deterministic indexed face per image. It read the initial and subsequent 500-image pages used by Detect's metadata-cache path, with no source media, thumbnails, crop decoding, models, network, user runtime, or Qt publication. The observed first-page median was **200.904 ms** and the complete 10,001-image cache-read median was **4,147.547 ms**. This is a host-specific capacity baseline, not an end-to-end or before/after performance claim. The focused offscreen regression `test_folder_review_caches_pages_in_background_and_appends_only_at_viewport_tail` separately proves that cached pages leave Photos unchanged until a tail request and that no hidden Faces rows are built while Photos is active.

Method: all commands used an isolated temporary runtime with `QT_QPA_PLATFORM=offscreen`, generated paths/rows only, no user media, models, network, or persistent cache. `scripts/benchmark_theme.py --iterations 250` alternated the complete stylesheet and then reapplied an already-resolved dark theme. A 100,000-path `SectionedGallery` fixture measured only model publication and visible-viewport queueing. A 500-row `JobsDialog` fixture measured unchanged refreshes and one progress update. `scripts/verify_ui_redesign.py --report-dir /tmp/clusterlens-clean-ui-evidence` captured the empty shell at 1280×720 and 1920×1080.

The theme fixture measured 0.2607 ms median / 0.2856 ms p95 for an actual light/dark change and 0.0014 / 0.0015 ms for the no-op reapplication; the latter avoids global stylesheet/icon work when Settings reapplies the current mode. The generated 100,000-path gallery published its full model in 466.3865 ms, while the visible-only queue was 0.0893 ms median / 0.1347 ms p95. The Jobs fixture measured 0.2434 / 0.3395 ms for an unchanged 500-row refresh and 0.3617 ms for one changed row. These are host-specific regression figures, not end-to-end claims. The final verifier passed with zero clipped visible controls, including the previously squeezed 1280×720 advanced Organize action flow; its empty-shell construction sample was 293.854 ms. The consolidated UI/theme/scheduler/gallery/single-instance regression suite and focused catalog regressions passed. People’s isolated basic-mode pane construction again met its five-second acceptance condition after runtime probing moved to the already-gated readiness Job.

The remaining audit fixtures ran on 2026-09-20 with isolated generated data. `bash scripts/benchmark.sh --library-catalog-only --fts-only --fts-assets 1000 --repeats 2` indexed 1,000 deterministic catalog rows in a 30.172 ms median and made a warm FTS lookup for 143 expected matches in 1.599 ms median. `bash scripts/benchmark.sh --duplicate-review-only --paths 128 --repeats 2` produced the expected 8,128 conservative near-pairs from all-identical 64-bit hashes in an 8.798 ms median; its profile is intentionally collision-heavy, not an estimate of a photo library. `scripts/verify_ui_redesign.py` passed again at 1280×720 and 1920×1080. `scripts/verify_native_display.py --logical-size 1280x720` passed on the available Wayland display, including focus, labels, contrast, visible controls, screenshots, and compositor-assigned client-surface bounds. These measurements do not claim a before/after throughput improvement.

## 2026-09-19 People → Faces correctness and first-content pass

Method: `bash scripts/benchmark.sh --people-faces-only --faces 5000 --repeats 3` created a fresh temporary SQLite face index for every repeat: 5,000 visible rows, one durable named identity, durable unnamed rows, and pending proposals. It reads no source media, crops, models, user runtime data, network, or GPU. The public `load_face_album_initial_page` API reads the first group page and first selected 500-member page from one SQLite snapshot. The fixture asserts that normal All Faces contains exactly durable named/unlabelled groups and that pending proposals remain Review-only.

The three samples were 12.352, 12.605, and 12.333 ms (12.352 ms median) for two groups and the first 500 members. The profile's material residual cost was SQLite execution (9.557 ms cumulative); JSON decoding of the 500 returned face boxes/reason arrays was 1.823 ms. This establishes a new workload baseline rather than a comparison with a prior whole-album or folder-review benchmark. The request runs on the existing background job path, so the UI thread only publishes the completed bounded snapshot.

The adjacent regression check, `bash scripts/benchmark.sh --face-review-paging-only --photos 1001 --repeats 3`, measured 36.128 ms median to first 500-image Folder Review page and 94.137 ms median for all `0, 500, 1000` pages. It is recorded separately because Folder Review does canonical path validation and review/scan-state joins that All Faces intentionally does not perform.

## 2026-09-19 measured Timeline filter-layout regression record

Method: this is a UI-geometry correctness change, not a catalog/query throughput change. The earlier same-day `--theme-only --iterations 250` sample (0.1335 ms theme median / 0.1479 ms p95; 9.6698 ms four-image classifier median / 9.8992 ms p95) is the visual-fixture baseline. After replacing fixed Timeline breakpoints with content/viewport measurements, the identical deterministic offscreen command ran on Python 3.12.13 / PyQt 6.11.0. It alternates palettes and classifies four fixed generated RGBA images; it reads no media, settings, runtime cache, model, GPU, network, or mutable user data.

The rerun measured 0.1326 ms theme median / 0.1501 ms p95 and 9.5078 ms classifier batch median / 10.3459 ms p95 (2.3769 ms per image; 35,256 traced peak bytes). This small host-time movement is a regression record only, not a claim about Timeline speed or real-widget repaint cost. `QT_QPA_PLATFORM=offscreen .venv/bin/python -B scripts/verify_ui_redesign.py --report-dir /tmp/clusterlens-timeline-layout-20260919d` separately passed the geometry contract: the generated empty 1280×720 shell used the semantic two-row Timeline at 914 usable px / 610 px required, while 1920×1080 used one row at 1,500 usable px / 1,194 px required; neither created an unnecessary vertical scroll range. Focused wide/medium/narrow grid tests also passed.

## 2026-09-19 staged Roots drawer and responsive Cleanup regression record

Method: the earlier same-day matte-rail `--theme-only --iterations 250` fixture is the baseline for this visual-only slice. After adding the staged Roots Sources/Catalog drawer and the Cleanup action flow, the exact command ran again on Python 3.12.13 / PyQt 6.11.0. It alternates the palette and classifies four fixed generated RGBA images; it reads no user media, settings, cache, model, GPU, network, or mutable runtime data. `QT_QPA_PLATFORM=offscreen .venv/bin/python -B scripts/verify_ui_redesign.py --report-dir /tmp/clusterlens-roots-cleanup-ui-20260919b` separately produced generated-empty-directory screenshots for Sources and Catalog at 1280×720 and 1920×1080.

The rerun measured 0.1335 ms median / 0.1479 ms p95 for theme application and 9.6698 ms batch median / 9.8992 ms p95 for the four-image viewer classifier (2.4175 ms per image; 35,256 traced peak bytes). The previous visual-pass sample was 0.1321/0.1419 ms and 9.5160/10.1065 ms respectively; this small host-time variation is a regression record, not a performance claim. The verifier passed with no clipped Sources/Catalog controls at compact or wide evidence sizes, and focused staged-Roots/Cleanup UI coverage passed 5/5. Layout does not scan, catalog, decode, or schedule work; applying a changed root scope remains the sole work-triggering boundary.

## 2026-09-19 matte typography and responsive-rail validation

Method: after the matte palette, tabbed Organize controls, and responsive Source/People rails, `bash scripts/benchmark.sh --theme-only --iterations 250` ran the fixed offscreen Qt fixture on Python 3.12.13 / PyQt 6.11. It used only generated RGBA images and no source media, persistent settings, cache, model, GPU, network, or randomness.

The post-change theme-switch sample was 0.1321 ms median and 0.1419 ms p95. The four-image 32×32 border-classifier sample was 9.5160 ms batch median / 10.1065 ms p95, 2.3790 ms per image median, and 35,256 Python-traced peak bytes. This is a regression record for the visual pass, not a throughput-improvement claim or a measure of real-widget repaint cost. The offscreen UI verifier passed its compact and wide dark/light evidence separately.

## 2026-09-19 theme and adaptive viewer-canvas pass

Method: `bash scripts/benchmark.sh --theme-only --iterations 250` ran Qt 6.11 offscreen on Python 3.12.13. Each classifier iteration used four generated 320×240 RGBA images (dark, middle gray, white, and transparent), downsampled each to the production 32×32 border sample, and verified the expected three neutral results plus transparent fallback. Theme measurements alternated the complete light/dark palette and stylesheet application. The fixture used no source photos, settings, runtime cache, model, GPU, network, or randomness.

The initial per-pixel `QColor` implementation measured 3.1209 ms per generated preview (12.4835 ms four-image batch median; 12.9066 ms p95). Profiling identified Qt color-object construction/access as the only avoidable new work. Reusing a 256-entry sRGB-linear lookup and reading the fixed RGBA byte buffer measured 2.4128 ms per preview (9.6514 ms batch median; 9.8773 ms p95), a 22.7% reduction in this classifier-only fixture. Python-traced peak allocation was 35,256 bytes. Final theme application measured 0.0853 ms median and 0.1306 ms p95 with no open production widgets; real repaint cost remains widget-count/platform-dependent and is not represented by that number. Classification remains in the existing preview worker, not the UI thread.

## 2026-09-17 Timeline hierarchy buckets

Method: `bash scripts/benchmark.sh --library-timeline-only --timeline-photos 1200 --repeats 5` built fresh SQLite-only catalog rows with no source images, then read the full filtered Timeline result. It now retains descending day and ISO-week scalar buckets so Year, Year → Month, Year → Month → Week, and Year → Month → Day can regroup in the UI without another query, source-media read, or thumbnail decode.

The final five samples were 1.401, 1.426, 1.344, 2.134, and 2.338 ms, for a 1.426 ms median. This is a new workload shape, not a comparison with the earlier Year → Month-only baselines. The profile attributed 1.165 ms cumulative to the 1,200 bounded ISO day parses and 0.403 ms to six SQLite fetch batches. Changing the drop-down republishes the loaded in-memory result and is outside this query benchmark.

## 2026-09-17 global source-admission filters

Method: `bash scripts/benchmark.sh --library-catalog-only --source-filters --photos 120 --timeline-photos 1200 --repeats 5` created five isolated generated roots. Each had 120 candidate names: 12 conventional `_thumb` names, 12 files below 512 bytes, 12 32×24 JPEGs, and 84 admitted 64×48 JPEGs. The enabled policy was thumbnail-name exclusion, 48 px minimum width, 36 px minimum height, and 512-byte minimum file size. The fixture exercised all filename-time parser paths for admitted files, read no user media/runtime data/models/network, and used only temporary databases.

The scan samples were 25.682, 20.184, 20.860, 20.102, and 19.976 ms (20.184 ms median); the warm FTS query median was 0.348 ms. The catalog reported exactly 84 discovered and 36 filtered paths. The cProfile pass attributed 10.190 ms cumulative to policy admission, including 96 Pillow header opens; 84 later metadata reads and SQLite upserts remain expected. The adjacent all-admitted 120-photo run measured 27.862 ms scan and 0.407 ms query medians, but it is a different workload and is retained only as a disabled-policy sanity baseline, not as a speed comparison.

## 2026-09-17 Timeline year bounds and manual capture corrections

Method: `bash scripts/benchmark.sh --library-catalog-only --photos 120 --timeline-photos 1200 --repeats 5` used fresh generated 120-JPEG/15-XMP roots plus source-free 1,200-row Timeline catalogs. It reads no user media, runtime data, models, GPU, or network. This follow-up validates that the catalog's 1991-through-current-year guard and reversible photo-edit-sidecar fingerprint do not move media work onto the UI thread. The fixture's normal filename parser paths remain 30 date counters, 60 declared epochs, and 30 raw-ID epochs.

The guarded baseline measured 27.623 ms catalog scan, 0.424 ms warm FTS query, and 1.359 ms Timeline query/grouping medians. Profiling showed that calling the clock once per Timeline row was avoidable. Snapshotting the current UTC year once at query start preserved the same bounds; the final rerun, including explicit sorting that keeps stale out-of-range rows in Unparsed last, measured 23.468 ms scan, 0.390 ms query, and 1.204 ms Timeline medians. The 0.155 ms / 11.4% fixture improvement is limited to this source-free 1,200-row scalar Timeline query; it is not a media-decode or whole-library throughput claim. The final profile retained 1.108 ms in ISO bucketing and 0.317 ms in SQLite fetches, with no EXIF/XMP decoding or filesystem reads in Timeline publication.

## 2026-09-17 Timeline named filename-time rules

Method: `bash scripts/benchmark.sh --library-catalog-only --photos 120 --timeline-photos 1200 --repeats 5` created five independent temporary 120-JPEG/15-XMP roots and five source-free 1,200-row Timeline catalogs. The scan fixture deterministically cycles through `{date:DDMMYYYY}{sequence:6}`, `IMG_Epoch_{epoch:s}`, `IMG_Epoch_{epoch:ms}`, and one raw hash-style seconds epoch with the opt-in fallback enabled. Each run verifies that all 120 filenames parse (30 date counters, 60 declared epochs, 30 raw-ID epochs); it reads no user photos, runtime data, models, GPU, or network.

The final rerun measured medians of 21.903 ms for the catalog scan, 0.398 ms for the warm FTS query, and 1.071 ms for the full 1,200-row Timeline query/grouping. This is a new, broader parser-workload baseline, not a speedup comparison with older plain-filename fixtures. The profiled scan spent 19.122 ms in media/catalog asset reads, 15.862 ms resolving paths, and 11.982 ms in three bounded SQLite upsert batches; named-rule matching was not a material top-level hotspot. Timeline remained a scalar SQLite read plus ISO bucketing (0.749 ms in bucketing and 0.349 ms fetching in the one profiled sample).

## 2026-09-16 filename-date catalog and batch-rename safety

Method: `bash scripts/benchmark.sh --library-catalog-only --photos 120 --timeline-photos 1200 --repeats 5` generated fresh JPEG/XMP and SQLite-only timeline fixtures. Before any retained timeline hot-path change, it measured a 23.796 ms catalog scan, 0.406 ms warm FTS query, and 1.072 ms full 1,200-row timeline query/grouping median. The profile attributes catalog time to canonical filesystem paths, Pillow metadata reads, and bounded SQLite writes; timeline time is indexed SQLite fetching plus timestamp bucketing. No user media, runtime cache, model, GPU, or network data was used.

An attempted Python fast path for ISO month extraction regressed a subsequent noisy run and profiled worse than CPython's `datetime.fromisoformat`, so it was removed. There is intentionally no speedup claim for the filename-date policy. Batch rename is a safety workflow rather than a throughput claim: deterministic recovery tests cover template rendering, collision blocking, journalling, cancellation boundaries, and restore.

The configurable filename-pattern follow-up repeated the exact generated command after adding validated ordered formats and a compiled per-service matcher. It measured a 19.898 ms catalog-scan median, 0.398 ms warm FTS-query median, and 1.051 ms full 1,200-row Timeline median. The earlier 23.796/0.406/1.072 ms sample is a different host-time sample, so these values are regression evidence rather than a speedup claim. The final profile attributed 44.155 ms of one profiled scan to the scan itself, 18.095 ms to metadata reads, 15.681 ms to path resolution, and 11.621 ms to bounded SQLite upserts; filename-date parsing did not appear among its material hotspots. The parser first applies a cheap four-digit-year gate and reuses compiled user formats, so ordinary filenames do not pay per-pattern validation or compilation.

## 2026-09-16 shared autocomplete control

Method: `bash scripts/benchmark.sh --entity-picker-only --names 10000 --repeats 7` creates 10,000 generated saved-person names in an offscreen Qt process, then measures one `EntityPicker` population plus a contains query. It opens no database, source photo, model, network connection, or persistent runtime data.

The initial implementation measured a 3.253 ms median. Its cProfile result identified repeated `casefold()` scans while both filtering and checking whether the typed name already existed. The picker now retains normalized search keys when choices arrive; the same fixture measured a 2.879 ms median. This is a host-specific UI-control regression benchmark, not an end-to-end Faces latency claim. The remaining population cost is intentional bounded normalization/sorting performed when a background name query publishes a new saved-name list; keystroke filtering itself remains below 1 ms in the profile.

## 2026-09-16 Clean performance pass

Method: `bash scripts/benchmark.sh` ran the complete shipped deterministic suite in a fresh temporary runtime, then the affected fixtures were profiled and rerun after each code change. No command read user photos, runtime data, caches, or model files. The broad baseline reported 360.755 ms CPU and 108.548 ms selected-CUDA median cosine K-means, 377.587 ms native CPU and 365.642 ms selected-runtime HDBSCAN (native fallback because cuML was unavailable), 11.009 ms JPEG/XMP merge/readback, 33.364 ms global Tags inventory, 10.193 ms selected-runtime deep face search, 21.468 ms sectioned Faces model publication, 121.775 ms bounded face-index scheduling, 39.295 ms Library scan, and 0.988 ms Timeline grouping. These are host-specific regression baselines, not end-to-end photo-library claims.

The generated 1,024-entry/four-service thumbnail-index fixture initially measured 34.536 ms for the first service and 41.152 ms across all services. cProfile attributed 67.856 ms of an 82.714 ms profiled run to per-entry canonical path walks during the first reconciliation. A normal direct cache-file fast path now avoids `resolve()` while symlink and database-originated paths retain canonical validation. The same fixture measured 12.437 ms first-service and 19.107 ms four-service medians after the change; the profiled reconciliation was 18.218 ms. The isolated cache recovery, concurrent-reader, and external-symlink rejection regressions passed.

`bash scripts/benchmark.sh --face-review-paging-only --photos 1001 --repeats 5` seeds one temporary face-index/scan row per synthetic path, makes no media files, and verifies three bounded source-order pages at offsets `0`, `500`, and `1000`. Before candidate-key reuse, its first-page and complete-stream medians were 124.156 ms and 282.138 ms; cProfile recorded 7,502 `Path.resolve()` calls while repeatedly checking the same paths. With canonical candidates computed once per page and passed to the nested review/scan reads, final medians were 28.911 ms for the first 500-item page and 76.459 ms for all 1,001 items. The final first-page profile resolved 1,004 paths, all at the outer candidate boundary; it measures SQLite queries and review-object construction only, not real-media discovery, migration, crop decoding, thumbnail work, or model inference. Scoped path, symlink-escape, folder-review, and progressive page regressions passed.

## 2026-09-15 progressive Faces publication validation

This change is a responsiveness contract, not a measured throughput claim. Its deterministic service/UI fixture uses 1,005 source paths and holds later worker pages behind an event: the first 500-item folder-review page and the first 500-member All Faces group page must publish before the gate opens, then offsets `0, 500, 1000` must produce all 1,001 fixture items. The fixture uses no user photos, models, network, or persistent runtime data. It verifies page size, cancellation-safe publication, and bounded viewport thumbnail work; record a cold/warm end-to-end photo-folder latency benchmark before making any numerical performance claim.

## 2026-09-14 bounded face-index scheduler baseline

Method: `bash scripts/benchmark.sh --face-indexing-only --images 48 --repeats 3 --decode-workers 4 --quality-workers 2 --embedding-batch-size 16 --write-batch-images 16` creates a fresh temporary set of 48 deterministic 96×96 JPEGs and SQLite database for every repeat. It uses a deterministic one-face detector and embedder, therefore it measures decode, bounded scheduling, crop-quality work, and durable writes without downloading models, using a GPU, reading user media, or sharing cache state.

Samples were 177.240, 100.786, and 96.165 ms, for a 100.786 ms median. Each run decoded and persisted 48 images/faces, maintained a queue peak of four, and used three 16-face embedding calls. Worker-stage accounting attributed the larger aggregate costs to decode (193.495 ms) and SQLite persistence (86.813 ms) in the median run; these are aggregate concurrent-stage timings and can exceed wall time. The coordinator profile mostly waits for worker shutdown, so stage accounting is the useful hotspot signal. This is a regression baseline for scheduler behavior, not an end-to-end model or GPU throughput claim.

## 2026-09-14 active-root union discovery baseline

Method: `CLUSTERLENS_BENCHMARK_PYTHON="$PWD/.venv/bin/python" bash scripts/benchmark.sh --multi-root-discovery-only --photos 40 --repeats 5` creates two effective generated JPEG roots with 40 photos each, then adds the nested root as a third active-root input. It measures only canonical root normalization and recursive discovery; it does not decode photos, create a catalog, load a model, or read user data.

The five samples were 1.250, 1.128, 1.078, 1.176, and 1.082 ms, for a 1.128 ms median while returning exactly 80 de-duplicated paths. The profile attributed 1.381 ms of its 2.213 ms discovery sample to the two filesystem walks and 0.372 ms to canonical active-root normalization. This is a regression baseline, not a before/after performance claim. Larger-root work remains filesystem-stat bound and stays in a cancellable worker.

## 2026-09-14 Faces similarity-arrangement publication baseline

Method: `CLUSTERLENS_BENCHMARK_PYTHON="$PWD/.venv/bin/python" bash scripts/benchmark.sh --faces-arrangement-only --faces 4096 --columns 6 --repeats 5` constructs one deterministic in-memory group of 4,096 `FaceTileItem` values, then measures the transient `SectionedFaceTileModel` publication and six-column layout. It starts an offscreen Qt application but creates no photos, thumbnails, databases, models, GPU work, or mutable application state.

The samples were 24.043, 21.888, 24.562, 86.257, and 21.597 ms, with a 24.043 ms median for 684 model rows (one heading plus 683 face-grid rows). The high sample is retained rather than discarded. This is a presentation-model baseline, not a clustering or end-to-end UI-throughput claim; face embeddings/clustering remain background work and crop decoding remains demand-driven.

## 2026-09-13 Gallery routing and cuML-provider validation

Method: `bash scripts/benchmark.sh` after the Gallery-route, non-blocking completion, and cuML-provider changes. The command uses only its generated seeded fixtures and a temporary runtime; it does not read user photos, durable labels, or application caches. This is a reproducibility check, not a before/after throughput claim.

On the current source runtime, the 20,000 × 128 cosine K-means fixture measured 585.865 ms median for CPU and 117.610 ms for the selected CUDA path. The 10,000 × 32 HDBSCAN fixture measured 394.895 ms with explicit CPU; the selected CUDA policy recorded `cuML HDBSCAN is unavailable: No module named 'cuml'`, used native CPU HDBSCAN, and measured 379.063 ms. The exact fallback is now shown in metrics, the footer, and Jobs; an explicitly selected CUDA HDBSCAN operation instead fails with remediation. The generated eight-region 1600 × 1200 JPEG XMP merge/readback measured 10.702 ms median. The remaining generated fixtures measured 42.322 ms for four thumbnail-index service opens, 34.633 ms for a global Tags inventory page, and 0.099 ms for a global tagged-photo follow-up page. Timings are host- and cache-state-specific and are not compared with earlier runs.

The same checkout passed `bash scripts/build.sh` and the complete `bash scripts/test.sh` suite (517 passed, 4 environment/provider warnings) after this measurement.

## 2026-09-13 operational-safety validation

This pass changed storage/release behavior, not an algorithmic hot path, so it makes no throughput claim. The release fixture is seeded synthetic PNG data with a checksum manifest; category-clear, manifest-tamper, trash collision/restart/restore, and packaged-report validation run in temporary directories only. The offscreen source smoke wrote its expected non-frozen report and exited 0.702 seconds after the window-show log; this validates smoke teardown only, not packaged startup performance. A real frozen executable and native-display/clean-VM evidence remain required for release qualification.

## 2026-09-13 Tags paging audit

Method: the same isolated `bash scripts/benchmark.sh --tag-workspace-only` fixture was extended with matched global follow-up pages at offset 100. Each pair runs seven warm repetitions against the identical 25,000-path / 75,000-row SQLite database; the only difference is whether the already-known total is counted again. A cProfile run spent 0.552 of 0.842 seconds in SQLite `Connection.execute`, confirming SQL rather than Python conversion or connection setup as the target.

The normal first global inventory page measured 35.554 ms median. On a later inventory page, retaining the known total avoided the redundant `COUNT(DISTINCT tag_norm)` query: 33.333 ms median versus 36.741 ms with the count (9.3% lower on this fixture). A 50-row global tagged-photo follow-up page measured 0.117 ms without a repeated count versus 0.131 ms with one (10.7% lower). First pages still compute totals for correct paging controls; these results apply only to follow-up pages and do not claim end-to-end gallery performance.

## 2026-09-13 Tags workspace SQLite baseline

Method: `bash scripts/benchmark.sh --tag-workspace-only` creates a temporary SQLite tag database with a fixed 25,000 generated path rows, 500 normalized tags, and exactly three tag rows per path (75,000 rows total). It creates no photo files and performs no EXIF, thumbnail, model, cache, GPU, or network work. After one warm-up, it measures seven pages of 100 inventory rows and one 200-row tagged-photo page.

On the current local `.venv`, global inventory pagination measured 35.388 ms median (34.485–36.325 ms), current-folder inventory pagination measured 24.429 ms median (23.788–26.466 ms), and a current-folder tagged-photo page measured 0.164 ms median (0.153–0.199 ms). The fixture's queried folder had 150 tags and the selected tag had 50 photos. This is a baseline rather than an improvement claim; it validates that opening/browsing Tags is bounded SQLite work and does not depend on media scans.

## 2026-09-13 thumbnail-index reconciliation

Method: `bash scripts/benchmark.sh --thumbnail-index-only` creates five temporary directories, each with 1,024 valid generated 8 × 8 WebP files, then opens four separately constructed `ThumbnailService` instances against each directory. It measures the first index open and the sum of all four opens; an additional generated directory supplies the profile. The fixture has no source photos, model files, persistent runtime cache, GPU computation, or network input.

Before this pass, the four-service median was 200.411 ms (200.224–207.382 ms); the first service alone was 30.299 ms median. Its profile spent 0.475 s in four `_reconcile_disk_index` calls, including 0.431 s validating cache paths and 0.379 s in `Path.resolve`. This identified duplicate interruption-recovery scans from independently owned gallery/preview thumbnail services as the bottleneck.

After sharing a bounded, process-local completed-recovery state for a cache directory while its WebP/index fingerprint matches, the four-service median was 43.321 ms (42.401–43.607 ms), a 4.63× throughput improvement for this multiple-view workload. The post-change profile performed one reconciliation rather than four (0.071 s), with 1,024 rather than 7,168 managed-path validations. The first-service median was 36.399 ms on this separate run, so this is specifically a redundant-reconciliation improvement rather than a claim that initial recovery is faster. Missing/corrupt-index, orphan, safe-prune, and concurrent-reader tests continue to cover recovery correctness.

## 2026-09-13 recovery/readiness validation rerun

Method: `bash scripts/benchmark.sh` used its generated seeded vector and JPEG fixtures in a fresh temporary runtime after the thumbnail-recovery and startup-readiness changes. This pass changes recovery and UI scheduling rather than a vector or metadata algorithm, so these numbers are a reproducibility check and not a performance-improvement claim.

On the current local `.venv`, the generated 20,000 × 128 cosine K-means fixture measured 1203.728 ms median for the explicit CPU control and 138.809 ms for the selected CUDA path; both produced deterministic memberships. The generated 10,000 × 32 HDBSCAN fixture measured 403.828 ms median for explicit CPU. The selected CUDA policy reported a visible CPU fallback because that environment lacked `cuml`, with a 380.252 ms median and deterministic membership. The generated eight-region 1600 × 1200 JPEG metadata merge/readback fixture measured 11.189 ms median. No user photos, downloaded models, or persistent caches were read or changed.

## 2026-09-13 per-face metadata baseline

Method: `bash scripts/benchmark.sh` created a temporary 1600 × 1200 JPEG with eight deterministic normalized face rectangles, performed one warm-up, then measured five embedded MWG/XMP region merges followed by readback. The command also creates a temporary runtime and reads no user photos, global face labels, model files, or persistent application caches.

The five metadata samples were 7.280, 11.753, 9.925, 11.044, and 11.037 ms, for an 11.037 ms median. This is a baseline for generated local JPEG metadata I/O on the validation host, not an end-to-end face-indexing or GPU performance claim. The operation intentionally remains CPU/filesystem work; its purpose is to check that merging a multi-person photo remains bounded and deterministic in structure.

## 2026-09-13 GPU HDBSCAN

Method: generated a scikit-learn `make_blobs` fixture with seed 42, 10,000 rows, 32 float32 dimensions, 24 centers, and standard deviation 0.65. HDBSCAN used minimum cluster size 20, minimum samples 5, merge epsilon 0, single-cluster disabled, and outlier policy `keep`. One warm-up preceded five complete `cluster_prepared` measurements per device, including post-processing and sampled cluster-quality scoring. The command was the CUDA benchmark command below; it read no photos or mutable application caches.

The native CPU samples were 390.741, 398.561, 389.019, 390.809, and 390.689 ms (390.741 ms median). RAPIDS cuML 25.10 on the RTX 4090 produced 32.135, 32.734, 34.737, 31.908, and 32.496 ms (32.496 ms median), or 12.02× the CPU-control throughput. Both paths found 24 clusters and zero outliers and were deterministic across their five repetitions. Their membership digests differ because cuML and native HDBSCAN can resolve equal-distance graph/MST choices differently; cache schema 4 separates their implementations and backend parameters.

The final same-process verifier reported Torch 2.2.2+cu121, ONNX Runtime GPU 1.18.0 with `CUDAExecutionProvider`, cuML 25.10.0, CuPy 13.6.0, and no failures across all vector smokes including HDBSCAN. The initial integration probe exposed an incompatible scikit-learn 1.9 private API; the runtime now pins the validated scikit-learn 1.7.2 line.

## 2026-09-13 application vector acceleration

Method: generated `20,000 × 128` float32 vectors with NumPy seed 42, normalized them once, requested 64 cosine clusters with the `max_speed` profile, performed one warm-up, then measured five complete clustering calls including assignment post-processing and the 2,000-row sampled silhouette score. The fixture uses an isolated temporary runtime and reads no photos, model files, or application caches. Hardware/runtime matched the RTX 4090 CUDA environment described below.

Before the change, clustering ignored the embedding execution policy and used CPU MiniBatchKMeans. Five samples were 678.118, 386.802, 405.594, 371.638, and 428.127 ms (405.594 ms median). A cProfile sample attributed 0.756 of 0.820 seconds to `MiniBatchKMeans.fit_predict`, chiefly K-means++ distance calculation and label/inertia work; that identified the target rather than assuming GPU transfer would help.

Final command:

```bash
CLUSTERLENS_BENCHMARK_PYTHON="$PWD/.venv-gpu-cu121/bin/python" bash scripts/benchmark.sh
```

The final CPU-control samples were 386.211, 505.271, 424.828, 448.779, and 384.176 ms (424.828 ms median). CUDA samples were 99.286, 112.681, 111.628, 110.839, and 111.439 ms (111.439 ms median), 3.81× the CPU-control throughput for this generated workload. All five repetitions in each mode produced identical membership digests within that mode. CPU and CUDA digests differ because the CPU path retains MiniBatchKMeans while CUDA uses deterministic full-batch Torch K-means; result-cache compute signatures keep those outputs separate. This is evidence for the synthetic vector workload only, not an end-to-end photo indexing claim.

The expanded `scripts/verify_gpu_runtime.py` check passed with no failures. It selected CUDA independently for Torch inference, ONNX inference, semantic PCA, cosine K-means, silhouette quality, graph-neighbor top-k, and dense similarity scoring. CPU fallback inspection reported OpenBLAS with 16 threads, libjpeg-turbo, and NumPy dispatch through AVX2/FMA3 and available AVX-512 feature paths. The GPU runtime's optional FAISS wheel advertised optimized compile options but loaded its generic extension rather than the absent AVX2/AVX-512 extension modules, so production vector fallback relies primarily on NumPy/OpenBLAS/scikit-learn; no FAISS speed claim is made.

## 2026-09-12 GPU resource-rescan validation

The dedicated CUDA 12.1 source runtime was checked on the existing Ubuntu host with an NVIDIA GeForce RTX 4090 (24,328 MiB reported), Torch 2.2.2+cu121, and ONNX Runtime GPU 1.18.0. `scripts/verify_gpu_runtime.py` passed both smoke tests: Torch CUDA matrix multiplication took 28.64 ms and ONNX selected `CUDAExecutionProvider` in 808.43 ms. A separate temporary-database check opened the installed SCRFD 10G detector and ArcFace R100 embedder; both sessions reported `CUDAExecutionProvider` first. No user photos or face-database rows were read or changed.

## 2026-09-14 launcher and resource-check validation

No clustering algorithm or performance setting changed in this pass. The canonical launcher now selects an already-installed dedicated CUDA source runtime without setup/download work, while the UI continues to perform its provider checks on a background worker. On the same RTX 4090 host, `scripts/verify_gpu_runtime.py` reported Torch CUDA 3.11 ms, CUDA ONNX 396.94 ms, cuML 25.10, and no failures. The verifier selected CUDA for semantic PCA, cosine K-means, sampled silhouette, graph neighbors, dense scores, and HDBSCAN. This is a local capability/smoke result, not a new end-to-end performance claim.

Before the UI change, five direct capability-refresh samples took 1316.577, 0.015, 0.008, 0.006, and 0.006 ms; that path did not verify an ONNX provider. After the change, three complete `diagnostics("auto", refresh=True)` rescans took 705.585, 4.597, and 6.299 ms (6.299 ms median), and every sample selected `CUDAExecutionProvider`. The different workloads are recorded as validation evidence, not as a performance-improvement claim. The first probe includes lazy CUDA initialization; the UI therefore runs it on an `AsyncJob` worker.

## 2026-09-10 Names workspace assessment

This is a baseline for the committed Names workspace (`7fa09be`), not a comparison with an earlier revision. It used an isolated temporary SQLite database and did not read or alter user photos, caches, downloaded models, or the global face database.

- Environment: Ubuntu 26.04.1, Python 3.12.13, AMD Ryzen 7 7800X3D (8 cores / 16 threads), 60 GiB RAM, NVIDIA GeForce RTX 4090 (24 GiB; driver 595.91.07).
- Startup: five offscreen release-gate runs with `PYTHONHASHSEED=0`. Import median 123.265 ms (120.599–130.851 ms); window-construction median 404.640 ms (392.415–414.372 ms); RSS median 94.430 MiB (94.348–94.469 MiB). All five runs met the 1.0 s import, 1.5 s window, and 256 MiB RSS budgets. None eagerly imported Torch, ONNX Runtime, the face search service, or the face UI.
- 2026-09-14 audit rerun: one isolated offscreen release-gate sample measured 333 ms import, 406 ms window construction, and 137.3 MiB RSS. It met the same budgets with no eager Torch, ONNX Runtime, face-search, or face-UI modules. This is a validation sample, not a claim of improvement over the five-run baseline above.
- Names SQLite fixture: 50,000 indexed and durably labeled face rows, 25,000 distinct photo paths, and 500 names; seven warm query repetitions after one warm-up. The global name aggregate had a 36.122 ms median (35.572–37.279 ms); the selected-name unique-photo query for 100 photos had a 1.119 ms median (0.663–1.195 ms).
- Profile: a single aggregate query took 40 ms, of which 39 ms was SQLite `Connection.execute`; Python result conversion was not material. The aggregate and photo queries already run in a worker thread, the name list is paged, and the gallery uses its bounded thumbnail loader. No speculative caching or indexing change was made from this baseline.
- Runtime smoke: `timeout 15 bash scripts/run_app_gpu.sh` constructed the CUDA-source window and entered the event loop. It did not perform face inference, so it is a startup/stability smoke test rather than GPU throughput evidence.

Command used for the isolated startup/query assessment:

```bash
PYTHONPATH="$PWD:$PWD/src" QT_QPA_PLATFORM=offscreen PYTHONHASHSEED=0 \
  .venv/bin/python /tmp/clusterlens_names_assessment.py
```

The temporary assessment script and database were intentionally not retained. Recreate the same seeded fixture from the documented row counts above when comparing a later revision.

For a release benchmark, use a fixed local photo fixture and record the fixture manifest, hardware, runtime mode, cold/warm cache state, command, wall time, peak RSS/VRAM, first visible gallery paint, and cache growth:

```bash
.venv/bin/python -m apps.pyqt_production.release_gates \
  --folder /absolute/path/to/fixed/photo-fixture \
  --report-dir benchmarks/release_gates
```

Run the same fixture at least twice for cold and warm-cache measurements. Never compare results from different fixtures or unseeded/randomized workloads.

## 2026-09-14 deep saved-name expansion baseline

Method: `CLUSTERLENS_BENCHMARK_PYTHON="$PWD/.venv/bin/python" bash scripts/benchmark.sh --deep-face-search-only --samples 512 --dimensions 64 --repeats 2` built 512 deterministic, normalized float32 indexed-face records from NumPy seed 42. All rows were unlabelled, so the fixture exercised a complete first frontier plus its fixed-point confirmation round. It uses a temporary face database, reads no photos, models, user labels, or persistent caches.

The explicit CPU control used contiguous NumPy/BLAS matrices and measured 3.541 and 3.708 ms (3.625 ms median), returning all 512 regions in two rounds. The selected source runtime used CUDA Torch matrix multiplication and measured 2.430 and 48.822 ms (25.626 ms median); the second sample includes CUDA worker/runtime variation, so this two-sample result is validation evidence rather than a throughput claim. The benchmark is now part of `scripts/benchmark.sh`; larger comparisons must use the same fixture, repeat count, and warm/cold policy.

## 2026-09-14 registered-library catalog baseline and batching pass

Method: `bash scripts/benchmark.sh --library-catalog-only --photos 240 --repeats 5` created five independent temporary fixtures, each with 240 deterministic 64×48 JPEGs and 30 XMP sidecars. It timed an initial explicit-root catalog scan and a warm FTS text query. No user media, runtime catalog, models, or network endpoints were read. The profiler first identified per-photo SQLite connection setup as avoidable scan overhead. Catalog writes now commit in bounded batches of 48 assets, preserving cancellation granularity and source-read-only behavior.

Before batching, the initial scan samples were 121.527, 118.741, 121.507, 121.480, and 121.486 ms (121.486 ms median); warm search median was 0.484 ms. After batching and XMP-sidecar revision detection, scans were 47.388, 47.413, 46.624, 46.323, and 47.632 ms (47.388 ms median), a 2.56× improvement on this fixture. The 2026-09-14 correctness/performance audit reran the same command: scans were 48.703, 49.503, 49.430, 48.927, and 47.419 ms (48.927 ms median), and warm queries were 0.620, 0.564, 0.584, 0.567, and 0.551 ms (0.567 ms median). The small difference is normal timing variance, not a further performance claim. The post-change profile attributes the expected work to image metadata decode, canonical-path resolution, sidecar stat checks, and the five bounded SQLite batches.

## 2026-09-14 full Library Timeline baseline

Method: `bash scripts/benchmark.sh --library-timeline-only --timeline-photos 1200 --repeats 3` created three independent SQLite catalogs with 1,200 deterministic rows spanning capture months. It deliberately created no source images, so the measurement covers only the full filtered scalar query and Year → Month grouping used by Timeline; the `CatalogQuery` page limit was set to 240 and intentionally ignored. The UI consumes the result through its virtual section model, so thumbnail decode is outside this measurement.

Samples were 1.198, 1.020, and 1.087 ms (1.087 ms median). This is an initial regression baseline rather than a before/after performance claim. The profile attributes 0.453 ms to 1,200 ISO timestamp buckets and 0.344 ms to six SQLite cursor fetches; it confirms there is no EXIF/XMP deserialization or source-file I/O in the Timeline query path. Larger-library measurements must use the same source-free catalog fixture and report the row count, warm/cold policy, and environment.

## 2026-09-16 virtual workspace publication baseline

Method: `bash scripts/benchmark.sh --ux-workflow-only` created generated
in-memory `ListEntry` rows only and ran seven repetitions per size in an
offscreen Qt process. It reads no files, user media, cache, database, model,
or network data. Queue delay is therefore 0 ms and cache/I/O/GPU fallback are
explicitly not applicable; this is a UI publication baseline, not an end-to-end
photo throughput claim.

For 500 rows, first content was 0.027 ms p50 / 0.040 ms p95 and full virtual
publication was 0.000 ms p50 / 0.002 ms p95. For 10,000 rows, first content
was 0.748 ms p50 / 0.879 ms p95 and full publication was 0.090 ms p50 / 0.114
ms p95. The profile attributed the 10,000-row path to `PagedListEntryModel`
source/filter rebuilding (3.194 ms profiled once), chiefly the one-time sort
and 10,000 normalized-title key calls; 19 page insertions totalled 0.111 ms.
The host's RSS provider was unavailable, so no memory claim is made. Re-run the
same command and repeat count before comparing a change.

## 2026-09-16 viewport-first loading baseline

Method: `bash scripts/benchmark.sh --ux-workflow-only` was rerun before the
viewport-queue change. It uses the same seven-repeat, generated 500/10,000-row
offscreen fixture described above, so it establishes publication overhead only;
it intentionally does not decode photo files. The new queue behavior is
validated deterministically by fake request ordering rather than timing a
machine-dependent decoder.

The baseline measured 500-row first content at 0.027 ms p50 / 0.038 ms p95 and
full publication at 0.000 ms p50 / 0.001 ms p95. At 10,000 rows it measured
0.506 ms p50 / 0.750 ms p95 first content and 0.058 ms p50 / 0.075 ms p95 full
publication. No performance-improvement claim is made: this implementation
changes thumbnail ordering and stale-work suppression, not this source-free
list-publication fixture. A fixed photo-decoder fixture is required before
making decode-latency or I/O claims.

The post-change rerun measured 500-row first content at 0.027 ms p50 / 0.038
ms p95 and full publication at 0.000 ms p50 / 0.002 ms p95. At 10,000 rows it
measured 0.511 ms p50 / 0.744 ms p95 first content and 0.058 ms p50 / 0.074 ms
p95 full publication. These small differences are normal timing variation in a
fixture that does not exercise image decoding or the thumbnail queue.

## 2026-09-18 Library-first review pass

Method: `bash scripts/benchmark.sh --library-timeline-only --timeline-photos 1200 --repeats 5` before and after the Library-first navigation, DateTimeOriginal policy, and progressive Search changes. The fixture creates only temporary deterministic catalog rows; it reads no source photos, caches, models, or network data.

The baseline median was 1.342 ms and the post-change median was 1.336 ms (samples 1.344, 1.379, 1.336, 1.324, 1.334 ms). This is normal run-to-run variation, not a performance-improvement claim. The post-change profile remained in `query_timeline` (2.943 ms profiled once), 1,200 ISO day buckets (1.075 ms), SQLite fetches (0.342 ms), and SQLite execution (0.343 ms). The policy selection and Library Search pager do not add source I/O to Timeline.

Cleanup and hover changes are intentionally measured by deterministic contract tests rather than a timing claim: exact review verifies that it does not load perceptual hashes/vectors; similar review verifies three hash-backend loads and two-backend agreement; the 1,001-row offscreen Search fixture verifies pages `0, 500, 1000` publish to completion. A fixed image/hash corpus is still required before making a SHA-256, perceptual-hash, or thumbnail-throughput comparison.

## Four-workspace UI redesign (2026-09-19)

The deterministic shell evidence command is:

```bash
.venv/bin/python scripts/verify_ui_redesign.py --report-dir /tmp/ui_redesign
```

It uses a fresh empty runtime, no user media/models/cache/network, and captures only ClusterLens widgets at 1280×720 and 1920×1080. The final run passed all primary-control bounds and screenshot checks, reported zero clipped visible controls across Library, Organize, People, Tools, Settings, Jobs, and Photo Inspector, and measured isolated shell construction at 106.433 ms. Empty-state first-visible samples were 2.550–25.594 ms at 1280×720 and 3.326–10.216 ms at 1920×1080. These figures measure generated empty-state UI publication, not photo/catalog/model throughput.

A same-process `cProfile` comparison identified repeated bundled-model checksum verification in routine header health refresh as the dominant avoidable construction cost. Before the change, the profiled constructor recorded 91,096 calls in 0.342 s; `_refresh_health_badge`/`validate_bundle`/SHA-256 accounted for 0.258 s and 0.231 s was hash updates. After routine health refresh was changed to enumerate bundles without hashing—and explicit Settings/model-selection verification was retained—the same isolated profile recorded 88,254 calls in 0.081 s. That is a 76.3% reduction in profiler-accounted construction time for this fixture. It is not an end-to-end launch claim; imports, platform window creation, startup readiness, and real runtime/model state remain outside this comparison.

The unchanged virtual-publication benchmark was also rerun with `bash scripts/benchmark.sh --ux-workflow-only --repeats 7`: 500 rows measured 0.026 ms median / 0.037 ms p95 to first content and 0.000 / 0.001 ms to publish all pages; 10,000 rows measured 0.503 / 0.742 ms to first content and 0.057 / 0.075 ms to publish all pages. Differences from the earlier 0.511/0.744 and 0.058/0.074 ms samples are ordinary variation, not a speed claim.

## 2026-09-25 deterministic Qt test cleanup

Method: `bash scripts/test.sh -q tests/test_production_support.py tests/test_theme_system.py --durations=12` ran the same 78-test ordered offscreen fixture before and after explicitly closing, scheduling deletion for, and flushing deferred deletion of production test windows between cases. No application runtime path was changed.

Before cleanup, the subset took 163.28 s; six global theme/text-scale tests each took 15.86–65.68 s because Qt repolished widget trees retained from earlier production-window tests. After cleanup, the identical subset took 18.94 s and none of the theme tests appeared in the slowest-12 list (the cutoff was 0.32 s), an 88.4% reduction in this test-process wall time. This is a deterministic test-harness improvement, not an application theme-switch performance claim.
