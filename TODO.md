# Current implementation plan

## 2026-09-25 complete UI/UX, concurrency and stability remediation plan

Target: **ClusterLens**, branch **main**. This is the authoritative implementation and remaining-work plan for the September 25 audit and the earlier performance/stability requests. Checked work requires the evidence recorded below; unchecked work is not a release claim. Preserve the existing implementation and user changes; root-only pre-promotion work remains in a separate, unmerged local backup branch.

Seven of the 35 tasks below remain open; UX-01 through UX-16 and UX-18–29 have complete reproducible evidence except for the explicitly open UX-17. Earlier completed items are historical implementation evidence, not current release approval. This plan supersedes overlapping acceptance criteria below; unrelated unfinished backlog remains required and must not be silently removed. The five reopened September 24 items identify partial implementations whose wider promises failed the latest audit.

### Baseline and rules for completion

- Audit starting baseline: **333 tests, 329 passed, 4 failed**, exit 1. Three failures reproduced independently; the Jobs-width failure was theme/order-sensitive but also reproduced in a production-themed probe. Current verified baseline after the completed slices below: complete UI smoke/acceptance **300 passed, 3 warnings, 25 subtests**, exit 0; complete suite **949 passed, 4 warnings**, exit 0. These passes do not replace the model/package gates below.
- The existing offscreen verifier passed, but 74 additional captures/probes exposed missing coverage. The current native Linux visual/keyboard matrix passes all 16 Wayland/X11 size/scale cells, but real screen-reader, multi-monitor, active-model/GPU contention, non-POSIX fault, and clean-VM installer qualification remain open.
- Retain working foundations: empty-scope guidance, lazy People construction, scoped/readiness guards, virtual views, bounded paging, immutable gallery publication, generation checks, explicit confirmations and asynchronous close.
- **P0**: data integrity, thread/lifecycle safety, scheduler correctness and trustworthy tests. **P1**: required user-visible correctness, responsiveness and release qualification. Neither priority may remain open at sign-off.
- Every task must supply implementation, contract tests, actual command/exit result and reproducible evidence before checking its box. If hardware/display/VM/model assets are unavailable, record **blocked/not run**, never PASS.
- Correctness tests use fixed fixtures and controlled events/barriers, not sleeps, random delays or elapsed-time guesses. Bounded watchdogs may detect deadlocks. Performance tests use a separately specified reference environment and repeated measurements.
- Use temporary generated media/databases/settings, fixed seeds, local fake download servers and pinned model manifests. Never read developer photos/model caches implicitly, mutate user media, or download models during default tests.
- Keep changes small and explicit; adapt existing job/controllers/services and model/view widgets. No wholesale UI rewrite, extra framework or dependency unless a measured/verified requirement needs it.
- Diagnostics/recovery logs must identify job, phase, scope and recovery outcome without exporting private photo/name data. Reports remain local/ignored; source fixtures and harnesses are the reproducible evidence.
- Product defaults for this pass: all four workspaces remain; production People remains human-focused; no new CLI mutations. Guarantee 1280×720 logical client size independent of monitor size; 1920×1080 and 100/125/150/200% display/text scaling are qualification targets. Smaller viewports remain an explicit support limitation unless subsequently implemented and tested.

### Execution ledger — 2026-09-25 (work in progress)

Checked tasks have satisfied their complete contract and validation scope. Unchecked entries remain completed slices and reproducible evidence, not release sign-off.

| Tasks | Implemented/verified slice | Evidence | Still required before completion |
| --- | --- | --- | --- |
| UX-01 | Isolated first-run QSettings, kept face-tile loader threads disabled through deterministic queue inspection, and flushes deferred production/UI widget deletion between cases. Added `verify_test_isolation.py`, a concrete named/unnamed/ignored/pending face fixture contract, offline empty external paths, source integrity and owned process-session checks. | `/tmp/clusterlens-test-isolation-20260925-v2/test_isolation.json`: **PASS**. Repository and reverse `/tmp` orders each passed **367 tests, 4 warnings, 16 subtests**, exit 0; source aggregate/status/diff digests match before/after; zero surviving session PIDs. Report records revision `05a8edb…`, seed, Python/Qt/platform and fixture/model/dependency SHA-256 values. Focused verifier/fixture tests: **5 passed**. | — |
| UX-02 | Fixed the four audited expectations/product failures: explicit root/readiness fixtures, stable review-route restoration, and production Jobs Details width. Prevented a late Qt diagnostic from recreating a removed runtime during teardown. | Ordered and reverse five-module runs in the UX-01 report each pass **367 tests, 4 warnings, 16 subtests**. Current complete suite before the four new isolation-contract tests: **691 passed, 4 warnings, 18 subtests**, exit 0. Original focused failures and their explicit negative cases pass without xfail or relaxed safety checks. | — |
| UX-03 | Added durable per-job GPU decisions, CPU/I/O capacities, bounded FIFO queueing, successful/failed/cancelled dependency handling, read/write/Data Home/model-cache conflicts, nested/symlink/hardlink scopes, CPU fallback re-checks, queued and reentrant running-cancel protection, starter-failure recovery and normalized exactly-once terminal release. | `tests/test_work_coordinator.py`: **19 passed**, exit 0. Event-driven assertions cover start order, held capacity through cancellation acknowledgement, dependent dispatch and zero duplicate terminal signals. | —; production entry-point integration is UX-04. |
| UX-04 | Added a single-lifecycle AsyncJob adapter: coordinated work creates one Jobs row and creates its thread only after ownership. Production startup maintenance/readiness, lazy People initialization, Library, Tags, Names, SearchPane, shared Gallery actions, Sectioned layout preparation, Photo Inspector, Settings, shell discovery/preflight/tag/storage work, model-download processes and clustering processes now declare workspace ownership plus Data Home/source/model-cache/GPU resources. Queued work counts as busy, has no thread/process, and cancellation cannot start it later. Gallery, Sectioned Gallery and face-tile decoders remain explicit bounded, generation-checked internal pools because per-tile Jobs rows would flood the monitor and a gallery-lifetime lease would block file actions. | Complete UI smoke suite after SearchPane migration: **260 passed, 3 warnings, 3 subtests**, exit 0. Complete production-support suite before the final process regression: **70 passed, 4 warnings, 13 subtests**, exit 0; the added real-shell queued process cancellation regression passes (**1 passed, 70 deselected**). `tests/test_work_coordinator.py` plus `tests/test_producer_inventory.py`: **26 passed**, exit 0, including the owner/resource conflict matrix and direct-start guard. | — |
| UX-05 | Every coordinated production worker closure now consumes values captured on Qt instead of reading/writing live widgets or resolving mutable UI context. The shared People job boundary records a route/root/model generation, suppresses late progress/success/failure publication while preserving durable completion and Jobs history, and selection-specific searches retain their narrower request tokens. Settings runtime verification also captures its requested execution mode before dispatch. | Static producer audit plus focused immutable/late-event tests: **6 passed**, exit 0. Complete UI smoke/acceptance/inventory gate: **282 passed, 3 warnings**, exit 0. The deterministic matrix covers changed controls/scope and late progress, success, failure and cancellation; lifecycle suites cover close-time relays. | — |
| UX-06 | Clustering launch consumes the immutable Startup readiness policy and performs no synchronous capability refresh, ONNX provider probe or model construction. Settings invalidates the snapshot and shows Checking before the coordinated readiness worker refreshes it. Face-service constructors and readiness probes are asserted off Qt; explicit CUDA unavailability and OOM fail terminally without silent CPU retry. Model/session preparation remains inside the owned cancellable worker process. | Readiness/production/inventory gate: **81 passed, 4 warnings**, exit 0; focused launch/settings/CUDA tests: **4 passed**, exit 0. Static launch-path guard rejects runtime probes and provider constructors. | — |
| UX-07 | Identity profile/duplicate/prototype reads, pin/remove/merge/clear mutations, and asynchronous pre-confirmation label counts now run in owned workers. Saved-search initial load and atomic CRUD use an independent background lifecycle, immutable UI snapshots, request-aware publication and an in-process per-path lock, so they do not occupy the primary People worker or lose concurrent edits. Malformed JSON is reported without being overwritten. | Broad UI/service/inventory gate: **485 passed, 3 warnings**, exit 0. Deterministic blocked-read/cancel tests prove Qt responsiveness and no mutation after cancellation; service tests cover competing edits, failed atomic writes and malformed JSON preservation. | — |
| UX-08 | Clustering and model-download requests are serialized off Qt and streamed after QProcess starts. Child stdout has explicit line/buffer caps, stderr/log snapshots retain a 256 KiB marked tail, progress is coalesced, and log writes never recreate a removed runtime. Workers publish large results through managed atomic files; parsing, validation, cleanup and clustering-result normalization run in cancellable background jobs with generation checks. QProcess remains Qt-owned, terminal progress is flushed, and cancellation/close requests `terminate()` before an exact-process bounded timer escalates to `kill()` without `waitForFinished`. | `tests/test_process_transport.py`: **13 passed**, exit 0, covering partial/malformed/oversized records, rapid progress, long diagnostics, managed paths, off-Qt serialization/parse/log I/O, stale publication, large-result event-loop responsiveness, cooperative termination, hard escalation, stale-process identity and real-child restart. Transport/production/UI smoke gate: **345 passed, 4 warnings, 19 subtests**, exit 0. | — |
| UX-09 | Completed the production main-thread inventory. Library SQLite initialization and navigation snapshots, settings-triggered face-service rebuilding, People maintenance/paste preparation, startup path validation, source canonicalization, gallery root partitioning and face-provider selection now stay outside proportional Qt callbacks. UI commits consume immutable snapshots; saved recent paths restore lexically without probing unavailable mounts. `HOTSPOTS.md` records each proportional boundary and the bounded Qt commit. | Thread-affinity/static inventory and event-gated Library DB, provider, purge, transport and 5,000-path gallery barriers pass. Broad production/UI/archive/transport/inventory gate: **422 passed, 4 warnings, 19 subtests**, exit 0, 108.75 s. Focused path/history inventory: **25 passed, 3 warnings**, exit 0. | — |
| UX-10 | One strict parser now serves clustering and model-download protocols: 0 remains determinate, only -1 means unknown, other out-of-range values clamp, malformed/non-finite/count/unit fields are rejected, phase-local totals derive percentages, and processed/skipped/failed outcomes remain visible. Queued/cancelling widgets are indeterminate and explicit, completed empty work renders Done, failures never render 100%, controller progress is flushed before a terminal event, and both controllers suppress every late/competing terminal or progress signal. | Table-driven focused matrix: **4 passed, 3 warnings, 27 subtests**, exit 0. Broad production/transport/coordinator/UI gate: **392 passed, 4 warnings, 33 subtests**, exit 0, 107.93 s. | — |
| UX-11 | The newest foreground job has deterministic footer/cancel ownership based on registration, not progress frequency. Older generations and background work cannot steal it; when the owner terminates, the next foreground job resumes. The footer defers unrelated direct status copy while progress is owned, queued/cancelling work is indeterminate, coalesced terminal events retain the newest foreground outcome, and stale outcomes remain in Jobs. | Deterministic owner/terminal matrix: **3 passed, 3 warnings, 3 subtests**, exit 0. Broad production/transport/coordinator/UI gate: **394 passed, 4 warnings, 33 subtests**, exit 0, 108.13 s. | — |
| UX-12 | Centralized natural count copy across People Photos/Faces/groups, derived views, Library Search/Timeline and gallery publication. `Loaded` now denotes the unfiltered metadata snapshot, `Showing` denotes the filtered/paged publication, progressive metadata uses `Loading metadata X/Y photos`, and thumbnail decode totals exclude label-cache work. SearchPane internal jobs retain the initiating People owner; Timeline catalog counts remain separate from Library thumbnail tasks. | Complete affected count/paging/UI gate: **366 passed, 4 warnings, 33 subtests**, exit 0, 81.28 s. Focused 36-photo/3-thumbnail Timeline test, formatter table, two-faces-in-one-photo filters, cached paging, pending/ignored, derived result and owner assertions pass. | — |
| UX-13 | Jobs is an owned reusable modeless monitor; Details retains at least 260 logical px; compact mode names the active owner/cancel target. Its 500-row incremental refresh preserves selection, completion wins a cancel race, and production shutdown disconnects and closes the monitor before any asynchronous shell drain/retry. Failed-start tests now deterministically drain the observable off-Qt diagnostic write before deleting disposable runtimes. | Ordered theme/production/UI gate: **357 passed, 4 warnings, 33 subtests**, exit 0, 81.45 s. Focused monitor/close set: **7 passed, 3 warnings**, exit 0; failed-start/transport set: **13 passed, 3 warnings**, exit 0. | — |
| UX-14 | Query-photo Detect, saved-search Run/Rename/Delete, Detected Faces and Grouped Photos menus/toolbars now use the shared predicate, visible/accessibility reason, and direct-handler guard. Queued work is busy before its thread starts; stale selections are rejected at trigger time. Independent saved-search edits and clustering-filter handoffs remain usable during primary People work, while real GPU conflicts queue and unrelated Library work proceeds. Query-photo existence/full decode validation remains inside the worker before inference. | Broad coordinator/inventory/acceptance/UI gate: **319 passed, 3 warnings, 6 subtests**, exit 0, 93.16 s. Focused action/scheduler matrix: **26 passed, 3 warnings**, exit 0. | — |
| UX-15 | Separated desired warm policy from actual idle process state; disabling warm mode stops idle workers or exits after an active job; restarting remains possible. The opt-in qualifier runs the checksum-verified Fast Preview ONNX model against four generated images in two separately started warm workers, records actual CPU/CUDA provider, load/inference time, RSS and per-process VRAM, requires identical membership, and verifies each PID and its CUDA allocation disappear before restart. CUDA-unavailable environments report NOT_RUN. | Focused policy/qualifier set: **12 passed, 76 deselected, 3 warnings**, exit 0. CPU report `/tmp/clusterlens-ux15-warm-worker-cpu-20260925-b.json`: **PASS**; load 0.712/0.666 s, inference 0.013/0.009 s and distinct removed PIDs. CUDA report `/tmp/clusterlens-ux15-warm-worker-cuda-20260925-a.json`: **PASS**; `CUDAExecutionProvider`, load 0.886/0.683 s, inference 0.550/0.383 s, 710,934,528 bytes VRAM while intentionally warm, shutdown 430.696/451.177 ms, distinct removed PIDs, allocation absent after each shutdown and identical membership digest. | — |
| UX-16 | Face DB removal now checkpoints SQLite, releases service caches, and journals an exact DB/WAL/SHM/ANN staged move into managed recovery storage. Pre-commit cancel/failure/restart restores the old manifest; committed work survives late cancellation, remains explicitly restorable/discardable, and invalidates every published People snapshot. Startup and Settings clear use the same scoped recovery service. | Broad affected gate: **559 passed, 4 warnings**, exit 0, 88.60 s. The deterministic matrix injects cancel/process exit/callback I/O plus atomic journal-write failure at every journal/rename/commit checkpoint, runs recovery twice, holds a real WAL reader, compares exact manifests, rejects unscoped paths, and covers read-only/confirmation/late-cancel UI behavior. | — |
| UX-17 | Partial: all earlier journalled file/DB/model/catalog/metadata boundaries remain. Large rebuildable cache trees atomically detach before cancellable deletion and Data Home backup/relocation journals recover idempotently. The machine-readable inventory covers 15 production operations and 40 before/commit/after boundary records. Every declared boundary now has a separate child process genuinely terminated with `SIGKILL`, exact pre/crash/recovered state digests, and two recovery attempts. This exposed and fixed relocation recovery rejecting a durable `switched` journal after target publication. A production-shell restart plus Safety & Recovery refresh test reconciles a killed file move and displays its committed restorable result. | `/tmp/clusterlens-ux17-durable-fault-matrix-20260926-j/durable_fault_matrix.json`: validation **PASS**, qualification **PARTIAL**; Linux **PASS**, 118 tests, 3 warnings and 103 subtests passed, exit 0, all 40 boundary-specific killed-process cases, valid state digests, zero survivors and no operation-level fault gaps. Complete canonical suite: **949 passed, 4 warnings in 230.36 s**, exit 0 (outside the restricted sandbox because localhost/Qt local-socket tests require socket creation). | Report granularity is still operation/fault rather than the complete boundary-by-fault cross-product. Native Windows/macOS locking, ACL, rename and permission behavior also remains required. UX-17 stays open. |
| UX-24 | Completed the semantic theme-state pass. Light muted/success tokens now meet readable-text contrast; focused controls retain a 3:1 boundary; disabled controls resolve through the active palette; Jobs progress widgets expose exact queued/running/cancelling/done/cancelled/failed semantic states while retaining textual status and accessible descriptions. Gallery delegates, cluster/group previews, cached icons, photo-viewer chrome and runtime badges re-resolve after live theme changes instead of retaining stale colors. | Theme/state matrix: **13 passed, 3 warnings**, exit 0. Complete UI smoke/acceptance: **298 passed, 3 warnings, 16 subtests**, exit 0, 89.93 s. Offscreen dark/light verifier: **PASS**, zero clipped controls, `/tmp/clusterlens-ux24-after-20260925-a/ui_redesign.json`; key shell, People, Jobs, Inspector and Settings captures visually inspected. Complete canonical suite: **812 passed, 4 warnings**, exit 0, 117.52 s; canonical build and `git diff --check` exit 0. | — |
| UX-18 | SearchPane shutdown invalidates, cancels and drains all generic and face-specific tracked workers, disconnects UI observers before widget deletion and defers finished-thread reference disposal until queued terminal relays drain. Photo Inspector and the coordinated Library/Tags paths apply the same contract. The shell retains every still-running job/thread pair across repeated event-loop close attempts. People can no longer detach a live QThread and claim close is ready; Gallery treats a deleted finished wrapper as stopped so retries converge. Clustering/model children terminate cooperatively and escalate only the exact still-owned process; a real SIGTERM-resistant child exits and the same controller restarts. Deleted-window retry timers and late JobManager/WorkCoordinator signals are safe no-ops. Missing face-DB paths no longer raise from a deferred usage refresh. | Complete suite: **805 passed, 4 warnings**, exit 0, 147.55 s. The deterministic seven-owner barrier closes during probe, gallery decode, People indexing, inference, model download, DB maintenance and journal commit simultaneously; Qt remains responsive, pre-commit DB work stays absent, the post-commit journal marker survives cancellation, a partial download is not promoted, repeated Close waits for every owner, and a replacement shell opens. Isolation report `/tmp/clusterlens-ux18-isolation-20260925-c/test_isolation.json`: **PASS**; repository and reverse outside-workdir orders each passed **410 tests, 4 warnings, 33 subtests**, source digests were unchanged and both process sessions had zero survivors. | — |
| UX-19 | Find now exposes an explicit compact Photo/Saved person task selector and renders only the chosen primary workflow before selection, Workflow and administration. The photo path row wraps, the status copy is concise, a sole detected face is selected without reserving an unnecessary chooser, and an owned timer keeps the selector/query/current primary action together in the scroll viewport. Return launches Detect/Find or saved-person search; switching task, route and Basic/Advanced mode preserves both queries. The pipeline status strip no longer widens the task rail at large text. | Deterministic dark/light × 100/125/150/200% viewport matrix: **1 passed, 8 subtests** across photo/name empty, populated, busy and error states, plus optional saved-search reachability. Related keyboard/layout/saved-search/state set: **8 passed, 8 subtests**. Ordered UI smoke then UI/UX acceptance boundary: **294 passed, 3 warnings, 14 subtests**, exit 0, 88.88 s. Complete suite: **806 passed, 4 warnings**, exit 0, 115.81 s. | — |
| UX-20 | Replaced checkable group boxes with explicit Management and Danger-zone expanders whose bodies consume no layout or keyboard-focus space while collapsed. Identity browsing remains first. Opening a disclosure reapplies no-selection, busy and read-only predicates instead of re-enabling children; destructive actions are visibly marked, non-default, and retain confirmation/recovery copy. Existing lightweight persistence keys remain compatible. | Focused disclosure, identity action/cancel and persistence set: **4 passed, 2 dark/light 200% subtests**, exit 0. Ordered UI smoke then UI/UX acceptance boundary: **295 passed, 3 warnings, 16 subtests**, exit 0, 95.19 s. Complete suite: **807 passed, 4 warnings**, exit 0, 198.96 s. | — |
| UX-22 | People now normalizes supported legacy route aliases to stable IDs and rejects unknown routes without silently opening All Faces. Shell tabs, keyboard navigation, inner Review & Name changes, task selection, person selection and photo-review handoffs remain bidirectionally synchronized. Review and Review & Name have distinct pending-proposal versus durable-name help. Unnamed constrains the durable album to `unlabeled`; UI publication also excludes stale named, ignored and pending rows. Photo-set handoff explicitly selects Detect but never invokes detection. | Expanded route/membership/navigation matrix: **8 passed, 3 warnings**, exit 0; shell/task/person/context focused set: **4 passed, 3 warnings**, exit 0. Complete UI smoke/acceptance: **298 passed, 3 warnings, 16 subtests**, exit 0, 87.71 s. Complete canonical suite: **810 passed, 4 warnings**, exit 0, 115.32 s; canonical build exit 0. | — |
| UX-23 | Added versioned lightweight People state, explicit walkthrough dismissal and lazy QSettings restore. Unknown/malformed versions fail safely. The supported unversioned payload migrates old route/disclosure keys once, drops result paths, and persists v1. Missing or replaced roots do not reset the task or trigger a scan. An independent writer process exits and a reader process restores Find, walkthrough and disclosure state. | Five focused production persistence regressions pass; complete `tests/test_ui_ux_acceptance.py`: **22 passed, 3 warnings**, exit 0, 47.97 s. The subprocess check passes independently in 9.22 s. | — |
| UX-25 | Completed responsive text scaling and logical-window constraints. Text-scale changes now reflow shell contents once without replacing the active People route, focus, query input or user splitter baseline. Compact headers/runtime copy, metric-sized footer, wrapping Roots/People/cluster actions, vertical Settings/Advanced scroll boundaries, compact high-scale technical tabs and full tooltips keep primary tasks reachable at 1280×720. Settings previews theme/backdrop/scale live; Cancel restores the exact snapshot without persistence. The offscreen and native gates now accept an explicit scale, and the native gate rejects the wrong monitor, undersized clients, partial off-screen placement and rendered text clipping. | Repeated live-scale/focus route test: **1 passed, 5 subtests**; focused responsive/settings set: **5 passed, 3 warnings, 5 subtests**. Complete UI smoke/acceptance rerun: **299 passed, 3 warnings, 21 subtests**, exit 0, 93.84 s; one preceding isolated-process attempt aborted in an AsyncJob teardown relay, while the exact 11-test ordered slice and both subsequent complete gates passed, so the event remains recorded for UX-29 stress rather than hidden. Four dark/light offscreen reports at 100/125/150/200%, 1280×720 and 1920×1080: **PASS**, zero clipped controls (`/tmp/clusterlens-ux25-final-{100,125,200}-20260925-a` and `/tmp/clusterlens-ux25-final-150-20260925-c`). Native X11 fractional-DPI + 200% text: **PASS**, exact 1280×720 client, DPR 1.8099, zero clipped copy, `/tmp/clusterlens-ux25-native-xcb-125dpi-text200-20260925-a/native_display.json`; screenshots visually inspected. Complete canonical suite: **813 passed, 4 warnings**, exit 0, 124.09 s. | — |
| UX-26 | Completed logical accessibility and keyboard behavior for the audited shell/People/Jobs paths. Composite fields retain label buddies; route bars use arrows internally and release Tab/Shift+Tab; the header targets the focusable Jobs action instead of its container; collapsed disclosures leave traversal; Settings/runtime/source dialogs and the modeless Jobs monitor restore focus. People and Jobs expose named status/progress/cancel state without relying on color. AsyncJob disposal also makes queued relays inert before public signal emission, eliminating the reproduced teardown abort. | Expected-versus-observed named People control traversal passes on All Faces, Detect, Find and Manage; focused Tab/Shift+Tab, arrows, Enter, Space and Escape/focus-return checks pass. Complete UI smoke/acceptance: **300 passed, 3 warnings, 25 subtests**, exit 0, 95.96 s. Complete canonical suite: **814 passed, 4 warnings**, exit 0, 130.82 s. The gate first reproduced the prior late-relay abort twice, then passed after the disposal guard and regression. Native screen-reader behavior remains explicitly unclaimed and scheduled at UX-32. | — |
| UX-21 | Detect, Library batch/recovery/People review, Roots draft, Organize Advanced, Inspector and both Settings surfaces now use the shared measured `ResponsiveFlowLayout` for finite actions. Roots paths elide with the complete path and guidance in the tooltip. Inspector People/Metadata plus production Storage/Safety use vertical scroll containers with horizontal panning disabled. Detect's disabled Refresh/Auto-Clean controls remain in the measured flow and every long scan action has explanatory tooltip copy. | Focused cross-surface contract set: **7 passed, 3 warnings**, then disabled Detect/Settings/Inspector set: **3 passed, 3 warnings**. Complete UI smoke/acceptance: **296 passed, 3 warnings, 16 subtests**, exit 0, 86.80 s. Complete canonical suite: **808 passed, 4 warnings**, exit 0, 115.36 s; canonical build exit 0; `git diff --check` exit 0. Offscreen verifier: **PASS**, zero reported clipped controls at 1280×720 and 1920×1080, report/screenshots `/tmp/clusterlens-ux21-after-20260925-a/ui_redesign.json`; corrected Roots, Advanced, Inspector and Settings captures were visually inspected. | — |
| UX-27 | Both verifiers now use an isolated generated active root, so the global first-run panel cannot mask route evidence. The offscreen gate captures exactly 18 contextual routes at 1280×720 and 1920×1080, all 10 Settings sections, all 4 Inspector panels, Roots Sources/Catalog, both painted themes, six Jobs states, complete logical focus coverage and clean asynchronous shutdown. Missing routes, incomplete traversal, unnamed controls, disabled clipping, offscreen primary actions and poor contrast are explicit failures. The native gate applies the same 18-route/10-section coverage and separates visual, keyboard and screen-reader qualification. The populated matrix exposed and fixed clipped large-text actions, unnamed list/table views and the nested Review & Name tab order. | Populated two-size 200% offscreen matrix: **PASS**, `/tmp/clusterlens-ux27-populated-matrix-20260925-a/ui_redesign.json`. Populated native X11 1280×720/200% visual and keyboard gate: **PASS**, `/tmp/clusterlens-ux27-populated-native-xcb-20260925-c/native_display.json`; representative screenshots visually inspected. Screen reader is explicitly **NOT_RUN** and remains UX-32. Focused negative-control/route/Settings/Inspector regressions: **5 passed, 3 warnings, 4 subtests**. Complete UI smoke/acceptance: **300 passed, 3 warnings, 25 subtests**, exit 0, 89.08 s. Complete canonical suite: **814 passed, 4 warnings, 136 subtests**, exit 0, 125.28 s; canonical build and `git diff --check` exit 0. | — |
| UX-28 | Added a fixed seeded eight-photo 3840×2160 JPEG corpus and production workload driver. Five fresh untraced processes measure cold/warm thumbnail and full-resolution crop work; separate traced/profiled processes avoid instrumentation pollution. A three-worker thumbnail/crop/catalog contention phase records first content, completion, Qt callback, posted-event input, event-pump, CPU/RSS/I/O, queue and deterministic result evidence. Cancellation records acknowledgement and safe drain. Environment, dependencies, display scale, cache state and explicit CPU-only/GPU-not-used status are embedded in the report. The existing People Detect and 100,000-path Sectioned Gallery fixtures were retained and rerun. The initial box-crop probe was corrected to the representative five-landmark detector path before optimization. | `/tmp/clusterlens-ux29-aligned-baseline-20260925-a.json`: five deterministic fresh-process aligned-crop runs plus traced/profiled runs, exit 0. Cold/warm 32-crop medians **1,099.525/967.797 ms**; mixed first/complete **11.424/1,095.233 ms**; process peak **780.7–781.9 MB**; traced Python peak/retained **51,929,019/1,716,420 bytes**. Aggregate p95 deliberately omitted at five samples. People Detect and Sectioned Gallery reruns exited 0. Benchmark regressions pass. Methods, every timing sample, watch budgets, the earlier box-only boundary and non-claims are in `BENCHMARKS.md`; confirmed decode/RSS/I/O hotspots are in `HOTSPOTS.md`. | — |
| UX-29 | Reused one lazy full-frame RGB array for every aligned crop from the same detector/image pass, without introducing a persistent full-photo cache. All five detector paths use the scoped cropper and release it at batch end. Profiled People publication also stopped recomputing unchanged action/readiness/model-bundle state after every one of 151 row batches; actions update at the existing start/final boundaries while complete-scope eligibility remains available. Benchmark membership hashes are now root-relative and stable across temporary directories. The production and Sectioned Gallery drivers record fixed queue/cache bounds plus crop/model replacement plateaus. A proposed no-copy RGB variant was discarded after it failed timing and raised RSS. | Aligned baseline `/tmp/clusterlens-ux29-aligned-baseline-20260925-a.json`; optimized `/tmp/clusterlens-ux29-aligned-final-20260925-a.json`. Identical crop/thumbnail hashes; cold/warm aligned crops **69.2%/72.3% lower**, mixed completion **69.4% lower**, CPU **66.5% lower**. Five crop cycles: **0 RSS growth**, 6,188-byte Python growth. Same-fixture 14,477-face publication **1,551.424 → 400.848 ms (74.2% lower)** with exact membership/page limits; action refreshes **154 → 3** in the profile. Six full/filtered 100,000-path replacements plateau (full **−567,020 bytes**, filtered **+384 bytes**) and preserve selection. Focused crop/benchmark and progressive-action tests pass. Complete canonical suite: **823 passed, 4 warnings**, exit 0, 138.44 s; canonical build and `git diff --check` exit 0. | — |
| UX-30 | Partial: added an opt-in, no-download real-model gate with pinned fixture/model checksums, provenance, explicit detector/embedder/backend coverage, CPU/CUDA policy, cold/warm/recreated-service inference, finite/dimension/order/tolerance checks, deterministic clustering, provider/output/RSS evidence and truthful PASS/FAIL/NOT_RUN exits. The deterministic recovery matrix now covers loopback interrupted/resumed/cancelled/checksum/retry downloads, verified cache reuse, cache-only restoration, stale revisions, deleted bundles, incompatible embedding signatures/dimensions, provider failure, CPU OOM propagation, recursive CUDA OOM batch backoff and irreducible single-item CUDA OOM. Managed MobileCLIP, DINO, DINOv2, CLIP, OpenCLIP and SigLIP acquisition and online runtime loading share reviewed immutable commits; cache-only mode retains recovered local-snapshot compatibility. The production and 14-entry face catalogs expose explicit source/license policy instead of generic “verify” placeholders. | Focused local model-recovery/OOM matrix: **12 passed, 209 deselected, 3 warnings**, exit 0. Earlier immutable-revision/runtime-loader set: **10 passed, 209 deselected, 3 warnings**; acquisition/inventory/provenance set: **25 passed, 277 deselected, 3 warnings**. Missing-asset preflight `/tmp/clusterlens-ux30-not-run-20260925-a.json`: **exit 3 / NOT_RUN**, no cases executed. `docs/MODEL_PROVENANCE.md` records pinned/reviewed/unresolved status; missing models are never replaced by synthetic inference. | Approved licensed fixture and pinned files for every advertised detector/embedder, resolved weight-redistribution approval for every shipped asset, real CPU and CUDA runs, actual provider/resource evidence, and complete output-tolerance review remain required. UX-30 stays open. |
| UX-31 | Partial: added a deterministic, report-producing verifier around the production `WorkCoordinator`. It covers all six required conflict pairs in both launch orders, signal-journals queue/start/finish transitions, cancels queued and running work, injects failure, retries without restarting, verifies compatible read overlap, and requires an empty queued/running ownership set after every scenario. Added a populated-root offscreen production-shell journey with generated 4K thumbnail/crop/catalog workers, all four workspace groups, resize/theme/visible-scroll, Ctrl+J, visible-control cancellation, serial-result equivalence and final in-process drain. Replaced changed-theme global stylesheet rebuilding with semantic palette roles while retaining geometry rebuilds for density/text scale. Fixed the generated fixture's wall-clock mtimes after repeated processes exposed nondeterministic catalog order. Added a Linux process-session wrapper that retains the child report/log and rejects surviving PIDs after production-window/test-process close. The wrapper now launches the locked interpreter directly with isolated runtime/settings/bytecode paths and offline guards, so it no longer depends on a writable global uv cache. | Conflict report `/tmp/clusterlens-ux33-work-conflicts-20260925-a.json`: **PASS**, 6 pairs, 2 orders, 22 cancel/recovery journeys. Final fresh-process reports `/tmp/clusterlens-ux31-mixed-ui-20260925-{o,p}.json`: identical crop/catalog digests, cold route max **8.319–8.558 ms**, active action max **20.528–21.172 ms**, heartbeat max **20.462–21.094 ms**, and zero final jobs/resources. Current lifecycle report `/tmp/clusterlens-ux33-mixed-lifecycle-20260925-b/lifecycle.json`: **PASS**, exit 0, zero surviving session PIDs. The pre-change theme profile was 237.386 ms; the post-change profile is about 6 ms. Theme matrix: **13 passed**; populated dark/light 200% two-size verifier: **PASS**, `/tmp/clusterlens-ux31-palette-ui-20260925-b`. | Add real model/module-load/GPU, approved media, native input/compositor, broader persisted-operation equivalence and peak VRAM evidence. UX-31 stays open. |
| UX-32 | Completed a current-code native Linux visual/keyboard matrix for Wayland and X11/XWayland at 1280×720 and 1920×1080 with 100/125/150/200% text scales. Each of 16 cells covers 18 populated routes, Roots Sources/Catalog, all 10 Settings sections, exact requested geometry/scale, clipping, keyboard traversal, screenshots and clean shutdown. The first 200% Wayland inspection exposed a vertically clipped Settings search field; the shared theme now gives single-line editors a scale-aware minimum height and the verifier rejects undersized `QLineEdit` controls. The native verifier can now require observed, contained transitions across multiple fitting Qt screens without moving the normal one-screen path across every output. | **16/16 visual/keyboard PASS**. Wayland reports: `/tmp/clusterlens-ux32-native-1280x720-200-20260925-b` and `/tmp/clusterlens-ux32-wayland-*-20260925-c`; X11/XWayland reports: `/tmp/clusterlens-ux32-x11-*-20260925-a`. Wayland recorded DPR 2.0; X11/XWayland recorded fractional DPR 1.4479166667 and passed `--require-fractional-scale`. The latest normal-path regression `/tmp/clusterlens-ux32-native-transition-regression-20260925-b/native_display.json` is **PASS** with 18 routes, 10 Settings sections, visual/keyboard PASS and clean shutdown. Focused clipping/theme regressions: **5 passed, 34 deselected, 3 warnings**. A bounded Orca 50.2 trial at `/tmp/clusterlens-orca-trial-20260925.log` observed AT-SPI window events but did not reliably speak ClusterLens controls, so screen reader remains **NOT_RUN**. Current Qt screens are 2648×1490 and 1080×1920 logical; only one fits the documented 1280×720 minimum, so multi-monitor certification is also **NOT_RUN** on this topology. | Real screen-reader qualification, movement across two screens that both fit 1280×720, active inference/decode with Jobs monitoring and every advertised non-Linux platform remain required. Native Linux visual/keyboard PASS does not close accessibility or workload qualification. |
| UX-33 | Added exact `pytest==9.1.1` development locking and changed the canonical test runner to frozen resolution without an ad-hoc `--with` environment. Every canonical script resolves its repository. The CPU launcher changes to that root, and a no-GUI `--check-launch` validates both launch paths from an outside directory. The canonical test runner forces model libraries offline and injects a Python network guard; the native-descendant verifier additionally follows every Linux child with `strace`, rejects non-loopback or unparseable IP destinations and audits process-session cleanup. The clean-source verifier exports the exact tracked/non-ignored candidate twice without `.git` or `.venv`, forces frozen offline resolution, then builds, launch-checks and fully tests from independent outside directories while auditing source digests and process sessions. Added a source-linked model provenance ledger and tests enforcing full immutable revisions plus explicit license policy for every production/face catalog entry. | Current canonical suite: **949 passed, 4 warnings in 230.36 s**, exit 0. Current-tree isolation target `/tmp/clusterlens-ux33-test-isolation-20260926-final7/test_isolation.json`: 422 tests plus 52 subtests in both orders, unchanged 277-file source digest and zero survivors. Current-tree clean-source target `/tmp/clusterlens-ux33-clean-source-20260926-final9/clean_source.json`: two 277-file trees, offline builds/launch checks and two 949-test passes, unchanged digests and zero survivors. Current-tree native-process target `/tmp/clusterlens-ux33-process-network-20260926-e/process_network.json`: all 949 tests under tracing, zero external attempts and zero survivors. The complete deterministic benchmark rerun exited 0 with unchanged production digest `f213b8f4…9435e6` before the durability-only checkpoint slice. | A fresh committed clone, unresolved weight-redistribution approvals, benchmark reconciliation and final evidence reconciliation after every other open gate remain required. |
| UX-34 | Partial: rebuilt the Linux CPU onedir package from current source, verified bundled FaceNet stage assets and ran the frozen executable smoke. The first fresh artifact correctly failed because the smoke coupled Storage selection to obsolete display copy; the selector now uses the stable Storage section identity and verifies the actual current/nested state. Version 2 package evidence records both the launcher digest and a deterministic complete-tree digest. | Corrected smoke `/tmp/clusterlens-ux34-package-20260925-b/smoke-v2/packaged_launch_verifier.json`: **PASS**. Executable SHA-256 `507011e566b42145147d939401d08986a64d829573f7c4889a379ed3c0a7bbd2`; complete-tree SHA-256 `247d1f840dbabd012842d5084dd5aae9bd5f0f4d89d2c3c9df5873a2d7355cd6`; 5,129 files, 80 symlinks and 1,743,520,839 bytes. All shell/Settings/Storage/icon checks passed and no process survived. | Clean-VM native install, upgrade/interrupted migration, uninstall retain/delete choices, GPU package/provider, signing and a locked reproducible package environment remain required. A local offscreen onedir smoke is not installer qualification. |

Latest ordered audit/isolation target: `tests/test_theme_system.py tests/test_work_coordinator.py tests/test_production_support.py tests/test_ui_ux_acceptance.py tests/test_ui_smoke.py` — repository and reverse outside-workdir orders each pass **422 tests, 4 warnings, 52 subtests**, with unchanged source digests and zero surviving session PIDs; report `/tmp/clusterlens-ux33-test-isolation-20260926-final7/test_isolation.json`. Latest UX-17 operation/fault report: `/tmp/clusterlens-ux17-durable-fault-matrix-20260926-j/durable_fault_matrix.json`, validation PASS / qualification PARTIAL, Linux PASS, 118 tests plus 103 subtests, one real `SIGKILL` case for each of all 40 declared boundaries, exact state digests and zero survivors. Current complete untraced suite: **949 tests, 4 warnings** in 230.36 s, exit 0; the restricted sandbox run separately reached 924 passes but could not create the localhost/Qt sockets required by 25 tests, so it is not treated as an application failure. Current-tree clean-source target `/tmp/clusterlens-ux33-clean-source-20260926-final9/clean_source.json` covers two independent 277-file exports, builds, launch checks and 949-test suites with unchanged digests and zero survivors. Current-tree process/network target `/tmp/clusterlens-ux33-process-network-20260926-e/process_network.json` covers all 949 tests with zero external attempts or survivors. The full serial benchmark suite predates the durability-only checkpoint slice; four offscreen scale matrices, runtime badge, work-conflict, Trash recovery and isolated mixed-lifecycle checks also completed successfully, while the native Orca and two-fitting-screen gates remain open.

### Delivery order and checkpoints

1. **Phase 0:** reproduce failures and freeze contracts (UX-01–02).
2. **Phase 1:** fix scheduler before wiring production jobs; remove unsafe/off-thread work (UX-03–09).
3. **Phase 2:** unify progress, counts, Jobs, eligibility and resource lifecycle (UX-10–15).
4. **Phase 3:** prove cancellation, durable recovery and shutdown (UX-16–18).
5. **Phase 4:** repair People workflows, routes, persistence, themes, responsive type and accessibility (UX-19–26).
6. **Phase 5:** expand verifiers and measure representative mixed workloads (UX-27–31). Capture UX-28 baselines before performance edits in any phase.
7. **Phase 6:** native/package/clean-checkout qualification and sign-off (UX-32–35).

Dependencies below determine exact order. Independent visual fixes may land once their failing tests exist; this does not waive scheduler or recovery gates. A phase is complete only when all its tasks pass; no release based solely on screenshots or unit tests.

### Phase 0 — trustworthy contracts and reproduction

- [x] **UX-01 — Bring audit reproductions into isolated, maintained fixtures.** (P0; no dependencies)
  Implementation: extend the existing UI/production test helpers and verification scripts with fixed generated photos, seeded saved faces/identities/proposals, temporary Data Home/QSettings, explicit runtime/provider fixtures and controllable workers. Capture the existing failing routes, palette states, job transitions and scheduler reproducers before fixes. Store environment, revision plus dirty-diff digest, seed, fixture/model hashes and completed process exit codes with reports.
  Contract: fixtures reproduce the current defects without source-media access, inference, network, developer settings or abandoned Qt workers; synthetic coverage is labelled as such.
  Validation: run the five audit modules from two working directories with poisoned external runtime/settings variables; verify isolation, unchanged source hashes and no surviving child processes. Assert fixtures contain real seeded named/unnamed/ignored/pending face rows. Each targeted regression fails on its known bad behavior and passes only after its owning task lands.

- [x] **UX-02 — Reconcile all four failing tests without weakening safety.** (P0; depends UX-01)
  Implementation: inject ready models and active roots into the full-visible-scope clustering test; split missing-animal-model setup availability from model-dependent action eligibility; update review-state expectations to preserve the selected stable route; retain the Jobs minimum-width failure until UX-13 fixes product geometry. Isolate theme/QSettings/timers between tests and execute the keyboard assertions previously hidden behind the width failure.
  Contract: no xfail/deletion, empty-root bypass, automatic model download or forced review-tab reset conceals a bug. Generic-pane animal tests do not assert an unsupported production route.
  Validation: add explicit negative no-root/not-ready cases; run each original failure alone, the ordered five-module selection and deterministic alternate orders. All pass after their product dependencies land; repeat full-suite validation at UX-35 and report actual discovered totals rather than assuming the old 649 count.

### Phase 1 — safe resource scheduling and a responsive Qt thread

- [x] **UX-03 — Correct scheduler decisions, resource conflicts and capacity.** (P0; depends UX-01)
  Implementation: in `src/ui/work_coordinator.py`, retain a per-job Queue/CPU decision under global Ask without prompting twice; re-evaluate every non-GPU conflict after CPU fallback. Implement CPU/I/O queue limits, deterministic fairness, failed/cancelled dependencies and bounded pending submissions. Define canonical read/write source scopes, ancestor/descendant overlap, model-cache and Data Home locks; perform filesystem normalization outside Qt callbacks and cover aliases. Hold resources through actual worker drain/commit, not merely cancellation acknowledgement.
  Contract: conflicting writers cannot overlap; compatible reads and disjoint jobs can proceed within configured limits; CPU fallback cannot bypass file/DB locks. Queued cancellation never starts a worker, and every terminal path releases ownership exactly once.
  Validation: extend `tests/test_work_coordinator.py` with the duplicate-Ask and three-job CPU-bypass reproducers, nested/symlink paths, hard-linked file mutations, queue saturation, slot caps, FIFO/fairness, failed dependencies and cancellation races. Use event-gated starters; assert start order, running-resource sets, one prompt/terminal event and zero leaked slots.

- [x] **UX-04 — Route every production operation through real coordination.** (P0; depends UX-03)
  Implementation: inventory starts in `app.py`, session/download controllers and Library/People/Tools/Settings/Inspector services. Adapt existing starters to submit resource-bearing jobs for startup probes, catalog/discovery, embedding/indexing, face detection/search/clustering, duplicate hashing, model install/load/unload, metadata/rename/trash and managed storage. Keep viewport thumbnail/crop work bounded under its owning job/resource group instead of flooding Jobs with per-image rows. Carry visible workspace owner, immutable scope and actual backend into each request.
  Contract: production cannot bypass scheduler policy through an alternative button, context menu, shortcut or subprocess; one operation has one authoritative lifecycle. UI browsing remains available while independent work runs. Scheduler ownership includes child-process lifetime and excludes incompatible maintenance.
  Validation: production-shell integration tests spy on submissions and run the conflict matrix below through real UI entry points. Verify independent features overlap, conflicting jobs stay queued with a reason, cancellation before dispatch creates no work and CPU fallback records its reason. Guard against direct-start regressions with an operation inventory test covering every listed producer.

- [x] **UX-05 — Capture immutable worker requests and reject stale results.** (P0; depends UX-01)
  Implementation: audit worker closures in `search_pane.py` and related panes. Capture mode, canonical scope identity, selected face refs, query path, tiny-face option, thresholds and model configuration on Qt before dispatch. Resolve heavy services on the worker from that snapshot, never by reading the currently selected widget. Attach request/scope/model generations to progress, results and mutations.
  Contract: no worker reads or writes a QWidget. Changing tabs, roots, model settings or selections cannot change an already-started operation or publish its results into a newer view.
  Validation: replace relevant widget getters with thread-asserting spies; gate Detect and selected-face search, alter all initiating controls, release work and verify captured arguments. Exercise late success/error/progress after cancel, scope change and close; only current-generation UI publication is accepted.

- [x] **UX-06 — Move provider probing and model readiness off the start path.** (P0; depends UX-03–05)
  Implementation: remove synchronous `select_policy(refresh=True)`/ONNX construction from clustering launch in `app.py`; run capability imports/probes and model/session preparation as cancellable coordinated work. Reuse validated capability snapshots with explicit invalidation for runtime/settings changes; expose checking, fallback and failure transitions with retry/setup actions.
  Contract: clicking Run, opening People or changing Settings never synchronously imports a heavy runtime, constructs a model or waits for a driver. No inference starts until the requested backend/model is ready; explicit GPU requests do not silently become CPU.
  Validation: injected blocked imports/provider construction leave posted Qt input, navigation and cancel callbacks operational. Cover cancellation/retry, startup plus launch overlap, missing/corrupt model, CUDA unavailable/OOM and stale readiness. Assert provider construction thread/process, single ownership and accurate backend/fallback display.

- [x] **UX-07 — Move identity and saved-search I/O out of click handlers.** (P0; depends UX-04–05)
  Implementation: dispatch identity count/merge/pin/remove/prototype work and saved-search JSON load/CRUD through owned workers; perform required pre-confirmation counts asynchronously. Use atomic persistence and revision-aware refresh; keep confirmations on Qt and avoid lost updates from concurrent edits.
  Contract: slow storage does not block input; cancelled or failed operations leave the documented old/committed state and an actionable message. Independent browsing stays enabled, and stale results cannot replace current identity/search state.
  Validation: temporary DB/JSON tests inject blocked reads, write failure, malformed files and competing edits; assert Qt input acknowledgements before releasing the service, exact persisted records, handler guards, cancellation outcome and refreshed labels/counts. Run the normal saved-search and identity service/UI regressions.

- [x] **UX-08 — Bound process transport, log handling and result publication.** (P0; depends UX-04–05)
  Implementation: in `session_controller.py` and download transport, move request directory creation/serialization, log-file writes and large result parsing off Qt callbacks. Bound stdout/stderr/progress buffering, coalesce redundant updates, rotate/truncate diagnostic capture with a visible truncation marker, and apply prepared results in bounded commits. Keep QProcess ownership/signals on its owning thread.
  Contract: a noisy, malformed or stalled child cannot exhaust memory or freeze the UI; required terminal/error events are never dropped. Cancelling and closing never synchronously wait for process completion or flush large files on Qt.
  Validation: controllable child fixtures emit partial/malformed records, long stderr, rapid progress, large result JSON and premature exit; assert buffer caps, event ordering, responsive cancel, explicit errors, cleanup and no stale publication. Verify request/result/log filesystem and JSON work occurs off Qt, including failure paths.

- [x] **UX-09 — Close the remaining main-thread I/O and compute inventory.** (P0; depends UX-04–08)
  Implementation: instrument catalog/FTS, EXIF, decode/crops, thumbnail lookup/invalidation, saved-state serialization, backup/restore, inventory, migrations, hashing and proportional model preparation. Record each hot entry point, owner, thread and bounded UI commit in `HOTSPOTS.md`; move only confirmed expensive paths into existing owned workers. Ensure filesystem canonicalization and error/confirmation preparation are included.
  Contract: Qt handles interaction, bounded model commits and paint only; no source I/O, DB scans, model construction/inference or unbounded per-row work on input/paint/timer callbacks.
  Validation: thread-affinity guards plus event-gated slow disk/DB/decode/provider tests cover all inventory entries. Verify UI input and cancellation execute while work is held, resize/selection remain correct, no worker touches widgets and close drains all workers. Use UX-28 profiling for any performance claim.

### Phase 2 — consistent progress, counts and control of work

- [x] **UX-10 — Fix progress protocol and define phase/terminal semantics.** (P1; depends UX-01)
  Implementation: replace truthiness parsing in clustering/download controllers so zero is not missing. Validate numeric values and units; represent unknown totals explicitly; use known phase totals where available instead of fabricated overall percentages. Keep Cancelling distinct until the worker reaches a safe terminal boundary.
  Contract: 0% is determinate, unknown work is indeterminate, completed work reaches Done, and cancellation/failure never fabricates 100% success. Phase resets identify a new phase; processed/skipped/failed counts explain the result.
  Validation: table-driven tests cover 0, 1, 100, -1, missing/None, malformed/non-finite/out-of-range inputs and multi-phase transitions in both protocols and widgets. Verify known/unknown total changes, empty work, queued cancellation, partial failure and exactly one terminal event.

- [x] **UX-11 — Give status text and progress one owner.** (P1; depends UX-04, UX-10)
  Implementation: unify job-to-footer/header/inline presentation in `job_presentation.py`, `footer_bar.py` and pane bindings. Key updates by job plus scope/request generation; clear or replace the last running message on finish/cancel/error; choose a documented foreground owner when multiple jobs run. Preserve background outcomes in Jobs without overwriting an unrelated active view.
  Contract: text, bar, cancel target and terminal state agree. No finished job leaves “Loading 37/100”; an old Library task cannot replace People status or a newer Library request.
  Validation: deterministic interleavings of two workspaces and two generations cover finish-first/finish-last, cancel, failure, navigation and no active jobs. Assert exact labels, bars and cancel IDs at each transition, including coalesced progress followed by terminal completion.

- [x] **UX-12 — Make photo/face counts and workspace ownership uniform.** (P1; depends UX-04, UX-10–11)
  Implementation: apply the count contract below across People Photos/Faces/groups, Library Search/Timeline, Organize, derived galleries, footer and Jobs. Replace technical thumbnail-task “Loaded” copy with “Thumbnails ready”; carry actual initiating workspace labels instead of internal “Faces” or shared-pane “Library”. Centralize formatting without conflating different datasets.
  Contract: Loaded describes unfiltered metadata in the current snapshot, Showing the currently published filtered/paged result set, not tiles physically on screen. Photos, faces, groups and thumbnail work never share a misleading denominator; unknown totals stay unknown.
  Validation: exact-copy/count tests cover zero rows, partial loading, filters, pagination, cached tails, two faces in one photo, pending/ignored faces, scope switches, cancellation and derived results. Timeline with 36 photos and 3 thumbnail tasks must show separate units. Tools/People cleanup jobs retain their initiating workspace.

- [x] **UX-13 — Make Jobs usable as a concurrent, modeless monitor.** (P1; depends UX-10–11)
  Implementation: replace modal `exec()` with one owned reusable modeless Jobs window/dock; preserve selection and keyboard focus through incremental updates. Fix default 860×420 geometry so Details has at least 260 logical pixels or an equally readable dedicated details pane; avoid stretch overriding the contract. In compact mode retain a visible job/phase cue and explicitly labelled cancel target; guard completion-versus-click races.
  Contract: users can monitor Jobs while using another feature; repeat opening raises the same window. Every queued/running/cancelling/done/cancelled/failed state and selected cancel action is understandable at normal launch size; closing the monitor does not cancel jobs.
  Validation: production-themed dark/light/default-size tests, 500-row refresh, selection stability, keyboard-only open/select/cancel/close, job-finish-before-click and app-close cleanup. Test compact header/footer geometry and the original Details-width failure in suite order; do not enlarge the fixture to hide the bug.

- [x] **UX-14 — Complete shared action eligibility and actionable empty/error states.** (P1; depends UX-04–05)
  Implementation: extend the eligibility predicate to query-photo Detect, saved-search Run/Rename/Delete and all menu/shortcut/context equivalents. Include valid scope/query/selection, readiness, read-only restrictions and conflicting busy state; retain handler guards. Provide setup/retry/edit-roots guidance and appropriate tooltips, not modal errors for predictably invalid actions.
  Contract: enabled actions are executable in their current scope; model setup remains available when inference is unavailable. Safe viewing and unrelated operations are not blanket-disabled by any People job.
  Validation: a table-driven matrix covers no roots, stale/empty selection, missing/corrupt query, checking/missing/ready/failed models, queued/running conflicting and nonconflicting work, read-only and inactive routes. Assert affordance, accessible reason, shortcut/menu parity and direct-handler rejection without mutation.

- [x] **UX-15 — Correct warm-worker policy and resource lifecycle.** (P1; depends UX-04, UX-08)
  Implementation: distinguish actual process warm/idle state from desired `keep_worker_warm` policy. Turning it off stops an idle process; a busy process finishes or cancels safely and then exits. Expose accurate loaded/unloading state, clear model/session references and allow a later job to restart normally.
  Contract: switching the preference off cannot leave an idle warm worker running indefinitely or kill a job mid-commit; CPU/GPU resources are released only after actual process shutdown.
  Validation: fake-QProcess tests reproduce the current unreachable stop branch and cover idle/busy/startup/error/repeated toggles. A controlled real subprocess test verifies exit, restart and no leaked child; qualified CPU/GPU runs measure retained resources without treating allocator reservation as a universal zero-VRAM requirement.

### Phase 3 — cancellation, incoherent-state recovery and shutdown

- [x] **UX-16 — Make Face DB removal recoverable and reader-safe.** (P0; depends UX-03–05, UX-10–11)
  Implementation: replace independent cancellable DB/WAL/SHM unlinks with an explicitly journalled staged removal of the exact managed file set. Exclude readers/writers, close/checkpoint owned connections, preserve a recoverable quarantine through commit, recover incomplete stages at restart and invalidate services/models/counts in all People views. Define cancellation before staging, during staging and after committed removal; make permanent cleanup a separate explicit safe operation.
  Contract: Cancel never implies “nothing changed” after a commit. Before commit, cancellation restores the usable prior store; after commit, report completed removal/recovery options accurately. No user photo, external model directory or unrelated Data Home content is a deletion target.
  Validation: temporary SQLite fixtures with WAL and active readers inject cancellation, process exit and I/O failure at every journal/rename/commit boundary. Restart twice to prove idempotence, compare labels/file manifests, assert restored-or-committed coherent states, scoped paths and no stale cached faces. Test read-only mode and confirmation cancellation.

- [ ] **UX-17 — Finish the durable-operation interruption and fault matrix.** (P0; depends UX-04, UX-16)
  Implementation: add controlled crash/fault checkpoints to indexing/embedding cache and face writes, catalog refresh, metadata/sidecar save, rename, Trash/restore, model promotion, cache clearing, backup/restore and Data Home relocation. Exercise missing/corrupt staging files, disk-full/permission failures, vanished roots and retry/restart; reuse existing atomic/journalled services.
  Contract: source files are unchanged until an explicitly confirmed mutation; committed subsets are reported exactly. Derived stores are transactionally usable or quarantined with a safe retry/rollback action; recovery is idempotent and never restores over unrelated files.
  Validation: the operation matrix below must pass before/after each durable boundary. Compare pre/post media hashes, labels/catalog rows, manifests and journals; run resume/rollback twice, verify DB integrity and actual refreshed UI state. Simulated disk-full and terminated child processes use disposable fixtures only; list every exercised boundary and uncovered platform case in the report.

- [x] **UX-18 — Prove scope replacement, cancellation and asynchronous close.** (P0; depends UX-04–09, UX-15–17)
  Implementation: audit ownership and cancellation from root edits, navigation, new requests, read-only changes, settings/model changes and shutdown. Keep the shell interactive while cooperative workers drain; use a bounded asynchronous escalation only for owned stuck subprocesses after their durable safety boundary is defined. Do not terminate QThreads or block Qt with joins/waits.
  Contract: no callbacks touch destroyed widgets, no stale job releases another job's locks, no background result reopens a closed pane, and no process/thread/timer survives completed application close. Cancelling acknowledgement is distinct from safe drain/commit completion.
  Validation: barrier-driven late-success/error/progress tests; rapid repeated open/close and scope replacement; close during probe, decode, indexing, inference, download, DB maintenance and journal commit. Verify close-state transitions, surviving data and a clean process tree, then reopen and run another operation successfully.

### Phase 4 — coherent, readable and accessible UI

- [x] **UX-19 — Put Find's primary query above the fold.** (P1; depends UX-01, UX-14)
  Implementation: reorder the Find rail so photo/name task choice, active query and primary Search/Detect action precede optional Workflow and saved-search administration. Turn long guidance into concise help/disclosures; preserve both query modes and saved-search capabilities.
  Contract: at 1280×720, the chosen Find task has an immediately visible input and primary action without initial scrolling; optional information cannot dominate the page. Large-text layouts keep the task clear and all controls scroll-reachable without clipping.
  Validation: viewport-intersection tests on the actual scroll area, not only `isVisible()`, cover photo/name modes, empty/populated/busy/error states, both themes and supported text scales. Verify keyboard launch, saved-search restore and no query loss on mode/navigation changes.

- [x] **UX-20 — Implement genuine Manage disclosures and safe hierarchy.** (P1; depends UX-14, UX-16)
  Implementation: replace unchecked-but-expanded QGroupBoxes with real body visibility toggles; put normal identity browsing/actions first and collapsed maintenance/danger sections below. Keep read-only/eligibility checks independent from disclosure expansion and retain confirmations/recovery guidance.
  Contract: collapsed controls consume no body space and are absent from keyboard traversal; opening a disclosure never enables an otherwise forbidden action. Destructive operations are explicit, distinguishable and not a default Enter action.
  Validation: assert collapsed height reduction, hidden children, focus exclusion and persistence; expand under read-only/busy/no-selection states and verify actions remain guarded. Test identity browse/name/merge and cancel-confirmation flows at 1280×720, both themes and large text.

- [x] **UX-21 — Reflow finite actions and audit task copy across workspaces.** (P1; depends UX-01, UX-14)
  Implementation: replace Detect's fixed two-column action grid with content-aware wrapping/one-column fallback using existing responsive layout conventions. Inspect Library, Organize Advanced, Tools, Roots, Inspector and Settings for clipped labels, unusable splitters, crowded action rows, misleading help, empty states and error recovery. Keep dense matte desktop styling and meaningful full labels/tooltips.
  Contract: enabled and disabled finite actions fit measured text, icons and padding at supported sizes; essential actions are reachable without horizontal panning. Long roots/model/person labels elide only where full content remains accessible.
  Validation: geometry checks include disabled Refresh/Auto-Clean controls, long synthetic strings, minimum/default widths, collapsed/expanded inspectors and both themes. Exercise staged Roots Apply/Discard/Continue, Basic/Advanced routing and Settings Apply/Cancel; record app-only screenshots for each corrected surface.

- [x] **UX-22 — Make People routes truthful and synchronized.** (P1; depends UX-05, UX-14)
  Implementation: map Unnamed to the durable unlabelled filter while retaining an explicit way to view all faces. Synchronize shell sections and nested Review & Name changes in both directions using stable route IDs; remove person-click resets to child index 0. Clarify pending-proposal review versus saved-face naming labels/help; preserve existing capabilities and legacy route aliases.
  Contract: navigation label, filter and content always agree; selecting a person does not unexpectedly change tasks. Entering a route starts at most its needed read, never a scan/index merely to navigate.
  Validation: table-driven route tests use shell clicks, inner tabs, task selector, keyboard, context handoffs, person selection and legacy aliases. Assert selected IDs, filter/content membership and load counts; named/ignored/pending faces cannot leak into Unnamed.

- [x] **UX-23 — Persist production workspace state, not incidental visibility.** (P1; depends UX-05, UX-22)
  Implementation: integrate pane export/apply with production shell QSettings using a versioned lightweight schema for contextual route, task, filters, splitter/disclosure and explicit walkthrough-dismissed state. Restore lazily when a pane exists; use stable IDs and migrate invalid/legacy values. Do not infer dismissal from parent `isVisible()` or serialize large results on Qt.
  Contract: restart restores the user's last valid workspace/task without scanning or loading hidden panes; hiding People cannot dismiss its walkthrough. Missing roots/models and malformed saved settings produce a safe usable fallback; saved-search records remain separately durable.
  Validation: production close/reconstruct and subprocess restart tests with fresh settings cover every route, walkthrough hide versus explicit dismiss, changed/missing roots, stale version/IDs and lazy People construction. Test Settings Cancel and ensure restore cannot replay a destructive action or resurrect a stale job.

- [x] **UX-24 — Repair actual theme states and semantic contrast.** (P1; depends UX-01)
  Implementation: replace hard-coded cluster hover/highlight backgrounds and footer chip combinations with existing semantic theme tokens; audit custom delegates, selection, focus, disabled/error/queued/progress states, popups, icons and Inspector chrome. Re-resolve cached paint assets on light/dark/system changes without invalidating user content.
  Contract: project acceptance targets are at least 4.5:1 for normal readable text and 3:1 for meaningful focus/indicator boundaries; disabled controls remain legible and recognizable. Do not rely on color alone for selected, failed or cancelled state.
  Validation: inspect resolved production palettes and delegate-rendered states, not token strings or a hard-coded pair. Reproduce the cluster 1.134:1/1.008:1 and footer 1.135:1 light/2.547:1 dark failures; assert corrected ratios in both themes, runtime switches, selection/hover and progress states. Include screenshot review for cases numerical checks cannot interpret.

- [x] **UX-25 — Make text scaling and window constraints genuinely responsive.** (P1; depends UX-19–21, UX-24)
  Implementation: base minimum geometry/reflow on supported logical client size and actual window contents, not a large monitor-derived 1440×900 minimum. Centralize typography roles and introduce persisted 100/125/150/200% text scaling independent of density; respect Qt/OS font and DPI behavior without double-scaling. Recompute content hints, splitters and dialogs, retaining desktop visual hierarchy.
  Contract: a user on a 1920×1080 monitor can resize to the supported 1280×720 client area. Primary tasks remain clear; larger text may scroll secondary content, but no label/action is clipped or permanently unreachable. Settings Cancel restores prior live preview and restart retains only applied preferences.
  Validation: test 1280×720 and 1920×1080 with light/dark themes, supported text scales, long labels and fractional display scaling in native qualification. Resize through breakpoints repeatedly, verify focus/selection/scroll preservation and accessible dialog buttons. Document behavior below the supported minimum rather than claiming untested support.

- [x] **UX-26 — Complete meaningful names, keyboard flows and screen-reader feedback.** (P1; depends UX-13–14, UX-19–25)
  Implementation: associate field labels with buddies/accessibility names, name People lists/query controls, expose help and validation descriptions, and establish intentional focus order. Cover nested tabs, virtual views, splitters, dialogs, popups and disclosure controls; throttle progress announcements and restore focus after dialogs/navigation.
  Contract: every enabled action is operable without a mouse through its documented Tab/arrow/shortcut path; collapsed/disabled controls do not trap focus. Selection, busy/error state and cancel target are perceivable without color. Modal confirmation traps focus only while intentionally open.
  Validation: expected-versus-observed actionable-control coverage, Tab/Shift-Tab, arrows, Enter, Space and Escape tests on all routes; account for composite-widget keyboard conventions instead of demanding every table cell in Tab order. Verify native keyboard/screen-reader names, roles, progress and error announcements at UX-32; record unsupported platform cases honestly.

### Phase 5 — verifiers, performance and representative feature stress

- [x] **UX-27 — Make UI verifiers reject the audit's visible failures.** (P0; depends UX-01; final pass depends UX-13, UX-19–26)
  Implementation: extend `scripts/verify_ui_redesign.py` and `scripts/verify_native_display.py` to cover every People route, populated face/identity/proposal data, all Settings/Inspector sections and Tools/Roots states. Include disabled-control clipping, primary-action viewport position, default-size Jobs, actual painted theme states and complete logical focus coverage. Respect asynchronous close before process exit.
  Contract: verifier PASS means its declared assertions and all required captures ran; missing/empty/skipped routes or failed shutdown cannot produce success. Native reports clearly separate visual, keyboard and screen-reader checks.
  Validation: negative-control tests deliberately insert an unnamed control, clipped disabled button, offscreen primary action, poor delegate contrast, missing route and incomplete focus traversal; each must fail with a useful locator. Run the corrected fixture matrix twice with clean settings and ensure expected report/capture counts and clean exit.

- [x] **UX-28 — Establish repeatable latency, memory and contention baselines.** (P1; depends UX-01; run before performance changes)
  Implementation: retain the existing People/Sectioned Gallery fixtures and add a fixed full-resolution decode/crop corpus plus a representative mixed-workload driver. Record runtime/dependency versions, CPU/GPU, driver, display scale, cache state, fixed seed and workload. Separate first content, complete cache, UI callback duration, input round-trip/event-pump gaps, cancellation acknowledgement, safe drain, CPU/RSS/Python allocations, GPU memory, I/O and queue depths.
  Contract: reported timings identify what was actually measured; synthetic metadata is not labelled inference or full-photo performance. Before/after comparisons use the same hardware/input/configuration and separate serial microbenchmarks from contention runs.
  Validation: at least five fresh-process untraced cold/warm runs plus separate traced/profile runs; report all samples, median/range and p95 only when sample count is sufficient. Publish methods and selected regression budgets in `BENCHMARKS.md`, resource hotspots in `HOTSPOTS.md`; do not use timing thresholds as correctness-test synchronization.

- [x] **UX-29 — Optimize only measured latency and memory hotspots.** (P1; depends UX-09, UX-28)
  Implementation: profile remaining first Faces publication, decode/crop scheduling, model commit/reflow, metadata retention and concurrent queue contention. Apply bounded batches, deduplicated requests or cache invalidation fixes only where measurements justify them; retain virtualization, stable selection/scroll and CPU fallback. Do not solve latency by dropping rows, lowering correctness or raising unlimited cache/worker limits.
  Contract: visible results and memberships are unchanged; active queues/caches remain within documented caps, old generations are released and hidden views do not eagerly build full models.
  Validation: run identical baseline/after benchmarks and compare result hashes/membership sets, page limits, responsiveness, cancellation and peak/retained memory. Repeated open/filter/resize/model-switch cycles reach a documented memory plateau rather than retaining each generation. Record regressions/tradeoffs as well as improvements; meet the qualification targets below or keep this task open with a measured revised proposal.

- [ ] **UX-30 — Qualify real model installation, loading, embeddings and backends.** (P1; depends UX-03–18, UX-28)
  Implementation: add a reproducible opt-in real-model qualification harness separate from offline unit tests. Pin model identifiers/revisions/checksums and use an explicit approved fixture model directory; exercise each advertised detector/embedder/clustering backend and CPU/CUDA policy. Use a local server for deterministic partial/corrupt/interrupted downloads and test managed bundle promotion, cache-only recovery, load/unload and warm reuse.
  Contract: browsing never downloads or infers implicitly; missing/incompatible models have actionable setup/retry states. Cancelled downloads/loads never appear ready; incompatible cached embeddings/results are invalidated by model/backend/config version. Advertised GPU execution is actually verified, with explicit policy-consistent fallback.
  Validation: generated/licensed fixed fixtures cover download/load/cancel/retry, checksum failure, offline mode, deleted bundle, stale revision, provider failure/OOM, embedding dimensions/finite values/ordering, cache reuse/invalidation and clustering deterministic membership or documented numeric tolerances. Record actual providers and resource peaks; lack of a supported GPU/model is NOT RUN, not a substituted synthetic PASS.

- [ ] **UX-31 — Execute cross-feature stress and recovery journeys.** (P1; depends UX-04–18, UX-22–23, UX-27–30)
  Implementation: build an event-driven scenario runner for the operation/conflict matrices below, then exercise the same journeys in the production UI with representative full-resolution media. Start/cancel/retry jobs through buttons, menus and keyboard while scrolling, resizing, switching theme/roots/routes, inspecting photos and changing settings.
  Contract: concurrent features obey locks/capacity; no silent job loss, wrong-scope publication, duplicate commit or unresponsive monitor. A cancelled/failed workflow can be restarted successfully without restarting the app unless a documented safe recovery is required.
  Validation: deterministic seeded action sequences plus a bounded repeated mixed-workload qualification cover all matrix rows and phase boundaries. Record operation journal/job timeline, UI input latency, peak queues/memory and final persisted invariants. Check result equivalence to serial execution for compatible work and verify no thread/process leak after final close.

### Phase 6 — native and packaged release qualification

- [ ] **UX-32 — Complete native responsiveness and accessibility coverage.** (P1; depends UX-19–27, UX-31)
  Implementation: run the complete route/dialog/state matrix on each supported desktop/display backend, starting with native Linux Wayland/X11 where supported. Cover minimum/large logical size, fractional DPI, large fonts, multi-monitor moves, keyboard-only workflows and a real screen reader; validate active inference/decode and Jobs monitoring, not only static empty shells.
  Contract: documented support matches tested platforms; no app clipping, illegible state, inaccessible critical action or input hang at supported sizes/scales. Offscreen PASS is not native certification.
  Validation: retain environment-specific app-only captures, keyboard/screen-reader checklist, measured input latency and clean shutdown. Cover every required matrix cell or mark it blocked with hardware/display requirements; unsupported Windows/macOS claims must be removed or independently qualified.

- [ ] **UX-33 — Make clean-checkout commands and documents match the shipped behavior.** (P1; depends UX-01–31)
  Implementation: retain and verify `scripts/build.sh`, `run.sh`, `test.sh`, `benchmark.sh` from outside the repository as well as its root. Pin the test/qualification dependencies and remove uncontrolled `--with` resolution where it defeats the lockfile. Isolate both runtime-root aliases, settings/cache locations and test network access; use explicit opt-in assets for model qualification. Update README, BUILD, ARCHITECTURE, BENCHMARKS and HOTSPOTS as the corresponding changes actually land.
  Contract: a clean checkout with documented prerequisites builds/runs/tests reproducibly; offline tests do not depend on mutable downloads or developer state. Documentation accurately describes scheduling, thread ownership, storage boundaries, routes, disclosures, progress, support limits and recovery—not future intentions.
  Validation: fresh-checkout/fresh-profile build and test from two working directories; inspect frozen dependency resolution and inherited environment isolation. Execute all documented commands, verify six required documents, model licensing/provenance and links, and run `git diff --check`. Record build environment/artifact digest and reproducibility limits rather than promising bit-identical binaries without evidence.

- [ ] **UX-34 — Validate packaged install, upgrade, recovery and clean uninstall.** (P1; depends UX-17–18, UX-30–33)
  Implementation: rebuild CPU and every advertised GPU package from the qualified source; run the packaged-launch verifier against the actual executable digest. Use fresh VMs/profiles for first run, missing-model guidance, install/cancel/retry, single-instance activation, read-only mode and upgrade from each supported prior profile/schema. Verify executable icons, launchers, logs/storage inventory and uninstaller retain/delete choices for managed data.
  Contract: packages require no source checkout or developer environment; upgrade preserves labels/settings/media and can recover interrupted migration. Uninstall never deletes source photos or unrelated files, and retained Data Home behavior is explicit.
  Validation: clean-VM install → launch → model/workflow smoke → upgrade → interrupted recovery → uninstall/reinstall scripts/checklists with fixture hashes before/after. Confirm paths/permissions, CPU fallback/GPU provider guidance, executable identity and no orphan worker/startup entry. A source-interpreter run or stale binary cannot satisfy this gate.

- [ ] **UX-35 — Close every gate and publish an evidence-backed release decision.** (P0; depends UX-01–34 and any unrelated open release backlog)
  Implementation: reconcile the remaining historical unchecked items below against the new task evidence; do not mark a broad task complete from a narrow passing test. Run final clean build, two independent full test processes, all deterministic benchmarks/verifiers, real-model/mixed-workload qualification, native checks, fault matrix and fresh-VM package lifecycle against the same release candidate.
  Contract: no open P0/P1 issue, unexplained failure/skip, hang, data-integrity gap, resource leak/unbounded queue or unsupported production claim remains. Every audit finding has a passing regression and every advertised feature a verified happy/error/cancel/recovery path.
  Validation: maintain a final evidence ledger with task ID, revision/diff and executable digest, fixture/config hashes, command, actual exit, test/capture counts, measurements, platform/provider and local report location. All boxes require linked evidence; missing GPU/VM/screen-reader results remain explicit blockers. Update release documentation and this checklist only after final results complete.

### Remaining release-gate execution checklist

The seven parent tasks above are the authoritative open gates. The child items below are the concrete work needed to close them; checking a child records evidence but does not close its parent until every child and the parent contract pass on the same release candidate. Reports must be written to new paths, must not inspect a user profile or private media, and must identify NOT_RUN separately from PASS.

#### UX-17 — cross-platform durable-operation faults

- [x] Inventory every durable commit/journal boundary for face/index writes, catalog refresh, metadata/sidecars, rename, Trash/restore, model promotion, cache clear, backup/restore and Data Home relocation. Add the boundary ID, pre-state, possible committed subset and recovery action to one machine-readable fault manifest.
  Expected behavior: every source-changing or managed-store operation has an explicit before-commit and after-commit meaning; no operation can report cancellation while concealing a committed subset.
  Verification: `tests/test_durable_operation_manifest.py` enforces the exact 15-operation inventory, classified fault coverage, unique qualified boundary IDs, recovery semantics and real implementation/test references. Together with verifier contracts it passes **6 tests**; the isolated evidence report validates 40 boundary records and every referenced test node.
- [ ] Run the existing Linux boundary matrix against disposable fixtures with injected permission denial, disk-full, vanished-root, corrupt/missing staging, locked-file and killed-process cases; compare exact media hashes, SQLite integrity, manifests, journals and UI refresh after two recovery attempts.
  Expected behavior: recovery is idempotent and ends in either the exact old state or the accurately reported committed state, never a hybrid or overwrite of an unrelated file.
  Verification: operation/fault coverage passes on Linux in `/tmp/clusterlens-ux17-durable-fault-matrix-20260926-j/durable_fault_matrix.json` with no declared operation-level gap. The manifest test now requires an exact one-to-one mapping between all 40 declared boundaries and 40 real `SIGKILL` cases; each retains pre/crash/recovered state digests and two recovery attempts. The matrix also includes production startup reconciliation and a populated Safety & Recovery UI refresh. Still required before checking this child: represent and enforce the complete applicable boundary-by-fault cross-product rather than operation-level fault coverage; an unexercised cell must fail coverage rather than be omitted.
- [ ] Port the same fixtures to native Windows locking/ACL behavior and to macOS if macOS remains advertised. Where a platform cannot support an operation, document and enforce that support boundary in product copy.
  Expected behavior: Windows sharing violations and platform-specific rename/permission rules produce safe retry/recovery states; Linux fault injection is not used as a substitute.
  Verification: retain native platform reports plus pre/post fixture digests and zero surviving worker processes. UX-17 remains open until every advertised filesystem platform passes or is removed from the support claim.

#### UX-30 — real models, embeddings and backend qualification

- [ ] Resolve the redistribution/license decision for every shipped or auto-installable weight in `docs/MODEL_PROVENANCE.md`; create an approved, non-private manifest from `docs/real_model_manifest.example.json` with immutable revisions and SHA-256 values for every advertised detector/embedder/backend case.
  Expected behavior: the default package and installer expose only assets whose distribution and acquisition policy is explicit; unresolved weights are not bundled or presented as approved.
  Verification: provenance tests reject mutable revisions, missing checksums, unknown licenses, unreviewed redistribution status and catalog/manifest coverage gaps.
- [ ] Run `scripts/qualify_real_models.py` once with `--execution-mode cpu` and once with `--execution-mode cuda` against the same approved images and pinned model files.
  Expected behavior: each advertised pair loads cold, reuses warm state, recreates cleanly, emits finite embeddings of the declared dimensions/order, detects the expected minimum faces and produces deterministic clustering or an approved numeric tolerance. CUDA reports the actual provider; explicit CUDA never silently falls back.
  Verification: retain both JSON reports with input/model hashes, provider, dependency/driver versions, per-case outputs, load/inference timings, RSS/VRAM peaks, membership digest and zero process leak. Missing assets produce exit 3/NOT_RUN and do not close the task.
- [x] Exercise local-server install/resume/cancel/checksum/retry/cache-only recovery plus stale revision, deleted bundle, incompatible embedding cache, provider failure and injected CPU/CUDA OOM.
  Expected behavior: partial/corrupt/cancelled content is never promoted as ready; version/config changes invalidate incompatible results; retry succeeds without an application restart; fallback occurs only where policy explicitly permits it.
  Verification: the focused local recovery/OOM matrix passes **12 tests**. It asserts exact HTTP Range requests, partial retention/removal, checksum promotion, cache-only restoration, stale/deleted state, cache signatures and dimensions, provider fallback policy, CPU OOM propagation, recursive CUDA batch bisection and single-item CUDA failure without a retry loop. Re-running the real qualifier after recovery remains part of the two unchecked real-model children.

#### UX-31 — production mixed-workload and recovery journeys

- [ ] Extend the event-driven scenario runner from synthetic crop/catalog work to approved representative full-resolution media and the qualified UX-30 detector/embedder modules on CPU and CUDA.
  Expected behavior: all six conflict pairs run in both launch orders; compatible reads make progress, conflicts queue with a reason, GPU policy is honored, and serial-versus-concurrent result digests match.
  Verification: report operation/job journals, backend/provider, queue/resource maxima, CPU/RSS/VRAM/I/O peaks and serial/concurrent hashes for every scenario.
- [ ] Drive start, queued cancel, mid-work cancel, failure, retry, scope/model change and close/reopen through the production buttons, context menus and keyboard while scrolling, resizing, switching themes/routes/roots, opening Inspector and monitoring Jobs.
  Expected behavior: input remains responsive; no stale result, duplicate commit or wrong-scope publication occurs; cancellation distinguishes acknowledgement from safe drain; every failed/cancelled operation can restart successfully.
  Verification: use controlled barriers for correctness and native event timestamps for responsiveness. Retain action/event traces, terminal UI state, persisted invariant hashes and screenshots; timing alone must not synchronize a correctness assertion.
- [ ] Repeat the bounded journey in a fresh process session and after reopening the same disposable Data Home.
  Expected behavior: durable labels/catalog/recovery state equals the serial reference, transient work is absent, and no thread, child process, resource lease or partial model promotion survives final close.
  Verification: `scripts/verify_work_conflicts.py` and `scripts/verify_mixed_ui_lifecycle.py` pass on the candidate, followed by the real-model native journey and a zero-survivor process-tree report.

#### UX-32 — native responsiveness and accessibility

- [ ] Run the native 18-route/10-Settings-section matrix with active decode/inference and live queued/running/cancelling/failed/completed Jobs states on each advertised display backend, at 1280×720 and 1920×1080, all supported text scales and a genuine fractional-DPI configuration.
  Expected behavior: no supported cell clips copy or actions, loses keyboard access, freezes input or leaks work at shutdown.
  Verification: `scripts/verify_native_display.py` plus the active-workload driver retain app-only captures, focus coverage, input-latency samples and final process/resource state for every matrix cell.
- [ ] Qualify movement between two screens that each fit the documented 1280×720 logical client minimum using `--require-multiple-screens`.
  Expected behavior: the window remains contained and usable, preserves route/selection/focus, and recomputes scale/layout on each screen.
  Verification: the report must observe at least two distinct target screens and save a contained capture on each. Current-host note: Qt reports HDMI-A-1 as 2648×1490 at DPR 2.0 and DP-3 as 1080×1920 at DPR 1.0, so this host cannot satisfy the two-fitting-screen gate without an explicitly approved display reconfiguration; this is NOT_RUN, not PASS.
- [ ] Run a real screen reader on every advertised desktop accessibility stack (Orca/AT-SPI for Linux; platform equivalents only if those platforms remain advertised) and complete the critical-action checklist.
  Expected behavior: workspace/routes, labels, list selection, disclosure state, progress/terminal state, validation errors and the active cancel target are announced meaningfully; keyboard focus follows the spoken item.
  Verification: retain screen-reader version/config, spoken-output or observer evidence, focus/action trace and human checklist. The September 25 Orca trial only confirmed AT-SPI events; it did not reliably speak ClusterLens controls and therefore does not pass this gate.
- [ ] Publish the exact supported OS/display/accessibility matrix and remove any unqualified platform claim.
  Expected behavior: release notes distinguish offscreen, native visual, native keyboard and native screen-reader evidence.
  Verification: documentation and package metadata match the passed matrix exactly.

#### UX-33 — reproducible clean source and truthful documentation

- [x] Re-run the complete canonical suite, build and benchmark after the final source edit; then run `scripts/verify_test_isolation.py` from the repository and an outside working directory.
  Expected behavior: frozen/offline resolution succeeds, both test orders have the same discovered counts, source digests stay unchanged and no process survives.
  Verification: current code candidate complete suite passed **949 tests, 4 warnings in 230.36 s**; build and `git diff --check` exit 0. `/tmp/clusterlens-ux33-test-isolation-20260926-final7/test_isolation.json` passes both 422-test/52-subtest orders from the repository and reverse order from `/tmp`, with identical 277-file source digests and zero survivors. The complete deterministic benchmark predates the durability-only checkpoint slice and remains to be reconciled.
- [ ] Run `scripts/verify_clean_source.py` on the final candidate and then repeat from a fresh committed clone.
  Expected behavior: two independently materialized source-only trees have identical digests and each builds, launch-checks and passes the full suite from outside its tree without `.git`, a local `.venv`, mutable network data or developer settings.
  Verification: `/tmp/clusterlens-ux33-clean-source-20260926-final9/clean_source.json` passes two independently materialized 277-file trees, offline frozen builds/launch checks, two 949-test suites, unchanged digests and zero survivors. The child remains open because this is a dirty-worktree export and cannot replace the fresh committed-clone gate.
- [x] Audit non-Python child processes for inherited network isolation and either enforce an offline boundary or list the exact opt-in exception used only by model qualification.
  Expected behavior: default tests/builds make no external request; local loopback fault servers remain usable.
  Verification: `/tmp/clusterlens-ux33-process-network-20260926-e/process_network.json` traces the complete 949-test suite and descendants, records no external IP attempt, drains the process session and retains trace/command-log digests; intentional loopback tests pass.
- [ ] Reconcile `README.md`, `TODO.md`, `BUILD.md`, `ARCHITECTURE.md`, `BENCHMARKS.md` and `HOTSPOTS.md` with the final implementation and measured evidence.
  Expected behavior: build/run/test/benchmark commands, routes, concurrency, storage, support limits, provenance, results and remaining blockers describe what actually ships.
  Verification: required-file check, command/link validation, `git diff --check`, and an evidence review that rejects stale report paths or unsupported performance claims.

#### UX-34 — package, install, upgrade and uninstall lifecycle

- [ ] Lock and document the packaging environment, then build fresh CPU and every advertised GPU artifact from the same qualified revision; run `scripts/verify_packaged_launch.py` against each actual executable.
  Expected behavior: packages need no source tree/developer environment, have the correct identity/icons, create only documented data, expose truthful provider guidance and leave no child after close.
  Verification: record dependency lock, build command, executable and complete-tree digests, file/symlink counts, package size, smoke report and actual CPU/CUDA provider.
- [ ] On a clean disposable VM/profile for every supported OS/package format, test install, first launch, missing-model setup, install cancel/retry, representative workflow, single-instance activation and read-only mode.
  Expected behavior: first run is usable and recoverable; cancelled setup is not marked ready; CPU fallback/GPU selection agrees with package claims.
  Verification: VM image identifier, installer digest, app logs, screenshots, provider report, generated-data inventory and zero orphan processes/startup entries.
- [ ] Upgrade from every supported prior schema/profile and inject interruption before/after each migration commit.
  Expected behavior: labels, settings, media and recoverable generated data are preserved; restart resumes or rolls back idempotently without touching source photos.
  Verification: before/after fixture hashes, schema versions, migration journal and two repeated recovery attempts match the declared result.
- [ ] Test uninstall twice: retain Data Home, then delete ClusterLens-managed data after explicit user choice; reinstall after each path.
  Expected behavior: the uninstaller asks the user, never deletes source media/unrelated files, removes binaries/integration/startup entries, and honors the chosen Data Home policy.
  Verification: filesystem/registry or desktop-entry manifests before/after, retained/deleted Data Home hashes and clean reinstall smoke. Signing/notarization verification is required wherever the package format advertises it.

#### UX-35 — final release decision

- [ ] Freeze one release-candidate revision/diff and artifact set; do not edit code or documentation between gate execution and sign-off. Any change invalidates downstream evidence and restarts the affected dependency chain.
- [ ] Run two independent full test processes, all deterministic benchmarks/verifiers, UX-17 fault reports, UX-30 CPU/CUDA qualification, UX-31 mixed journeys, UX-32 native/accessibility matrix and UX-34 VM lifecycle against that exact candidate.
- [ ] Build the final evidence ledger with task ID, revision/diff digest, executable/tree digest, fixture/config/model hashes, exact command, exit, counts, measurements, platform/provider and report location. Every required cell must be PASS; NOT_RUN remains a blocker.
- [ ] Review all historical unchecked backlog below, P0/P1 defects, warnings/skips, resource bounds and support claims. Publish GO only when none remain unexplained; otherwise publish NO-GO with the exact owner, prerequisite and rerun command.

Final verification contract: the release decision is reproducible from a clean committed checkout and its packages. A screenshot, a unit-test subset, a source-interpreter launch, synthetic inference, offscreen UI run or stale binary can support a gate but cannot substitute for a required native/model/VM result.

### Shared count and progress contract

| Surface/state | Required meaning/copy | Verification invariant |
| --- | --- | --- |
| People snapshot loading, finite total known | “Loading metadata X/Y photos” (or faces, explicitly) | X and Y refer to the same unfiltered snapshot and unit; 0 is valid; Y is not invented during discovery. |
| People snapshot and active result | “Loaded X photos · Showing Y photos”; face surfaces use faces | Loaded = metadata acquired for the unfiltered current snapshot; Showing = published filtered/paged membership, including scroll-offscreen rows. For one comparable snapshot, Showing cannot exceed Loaded. |
| Filtered result with a known larger match total | “Showing Y of Z matching photos/faces” alongside Loaded only if comparable | Matching total is not silently substituted for loaded metadata or decoded thumbnails; paging changes Y, filtering can change Z. |
| Derived search/cluster result | “Showing Y photos/faces/groups”, with known match total if useful | Do not borrow Loaded from an unrelated folder review or count groups as photos. |
| Timeline/Library metadata versus previews | “Showing N catalogued photos” and, while useful, “Thumbnails ready A/B” | 3 preview tasks and 36 catalogued photos never appear as conflicting “Loaded 3/3 images” and “Showing 36 images”. |
| Job progress/terminal state | Phase + owner + known completed/total or indeterminate; then Done/Cancelled/Failed with outcome | Bar, text, cancellation target and surviving committed state agree; stale progress cannot follow a terminal event. |

### Required operation and fault coverage

Apply **normal completion, queued cancellation where applicable, start/mid-work cancellation, worker error, stale completion after scope/model change, retry, close/restart and read-only behavior** to every row. For durable operations, additionally inject failure/cancellation/process exit immediately before and after every commit/journal boundary. Correctness uses controlled barriers, not a requirement to cancel “after N milliseconds”.

| Workflow | Required result/data checks | Tasks |
| --- | --- | --- |
| Startup, readiness and model inventory | No inference/download on browse; checking/ready/fallback/error agrees with verified backend; interrupted readiness does not block shell | UX-04–06, UX-18, UX-30 |
| Model download/install/load/unload/warm reuse | Partial archives/bundles never ready; verified promotion/cache recovery; policy and actual child lifetime agree | UX-08, UX-10, UX-15, UX-17, UX-30 |
| Roots, discovery, catalog/FTS, Search/Timeline/gallery load | Active-root isolation, correct counts/pages, no hidden view work, stale generation discarded, no duplicate source admission | UX-04–05, UX-09, UX-12, UX-17–18 |
| Full-resolution thumbnails, previews, face crops and Inspector | Bounded visible work, orientation/crop correctness, no destroyed-widget access, unchanged scroll/selection | UX-05, UX-09, UX-18, UX-28–31 |
| Embeddings, similarity and image/face clustering | Model/config cache signatures; complete initiating scope; deterministic memberships/tolerances; no partial ready index | UX-04–06, UX-08, UX-17, UX-29–31 |
| Detect/index/reindex, Review & Name, Unnamed and pending review | Atomic face rows, label/ignored/proposal separation, correct photo/face/group counts and restored route | UX-05, UX-12, UX-14, UX-17, UX-22–23 |
| Identity merge/pin/remove, query-photo/name/deep search and saved searches | Exact captured options, no unintended label mutation, durable atomic CRUD, no cross-scope publication | UX-05, UX-07, UX-14, UX-19–23 |
| Duplicate exact/similar/burst review and Trash/restore | Serial-equivalent candidates, only explicitly checked files moved, journal/restart/collision policy and media hashes intact | UX-04, UX-17–18, UX-31 |
| Rename, tags, metadata/EXIF/face regions and sidecars | Preview/confirmation scope; atomic per-file commit; accurate saved/skipped/failed totals; no overwrite of unrelated data | UX-04, UX-09, UX-17–18, UX-31 |
| DB/cache cleanup, backup/restore and Data Home relocation | Exclusive ownership, scoped targets, integrity/manifests, recoverable removal, idempotent resume/rollback and cache invalidation | UX-03–04, UX-16–18 |
| Settings, navigation, theme/text scale, Jobs and shutdown | Apply/Cancel contract, modeless monitoring, correct cancel target, persisted stable state and no leaked workers | UX-11, UX-13–15, UX-18, UX-22–27, UX-32 |

Required conflict pairs (test both launch orders and queued/running cancellation):

| Pair | Expected coordination |
| --- | --- |
| Embedding/indexing + image/face clustering on one GPU | Queue/Ask/CPU policy enforced; CPU choice rechecks all remaining source/DB/capacity locks. |
| Face detection/index writes + identity edits or face DB removal | Mutations serialize at the necessary store/source scope; removal excludes readers and invalidates cached state. |
| Inference/model load + replacement/removal of that model | Loaded session/model ownership protects the active version; replacement cannot expose a partial bundle. |
| Duplicate hashing/catalog/decode + rename/metadata/Trash on overlapping source paths | Compatible reads overlap; conflicting mutations queue or use a documented snapshot-safe boundary; nested/aliased paths are covered. |
| Backup/Data Home move/restore/clear + any managed-store writer | Operation-specific shared/exclusive locks protect a consistent snapshot and prevent writes to a relocated/removed store. |
| Disjoint read-only workflows + UI browse/navigation | Both progress within capacity; one feature's busy state does not freeze the other workspace or Jobs. |

UI matrix: all four workspaces and contextual routes; People Detect/Review & Name/Find-photo/Find-name/Unnamed/Manage/pending review; Organize Basic/Advanced; Roots Sources/Catalog clean/dirty; every Inspector and Settings section; Jobs at its real default and compact shell sizes. Include empty, populated, loading, queued, cancelling, failed, completed, read-only, missing-model and stale-selection states where meaningful. Run dark/light, 1280×720/1920×1080, enabled/disabled controls, long strings and 100/125/150/200% text scale offscreen; qualify fractional DPI and keyboard/screen-reader behavior natively. Use explicit state-machine edge coverage and documented pairwise reduction for orthogonal combinations, never silently omit a required route/state.

### Performance baseline and qualification targets

September 25 serial audit samples, **not before/after improvement claims**:

| Fixture | Observed sample | Boundary |
| --- | --- | --- |
| 11,284 photos / 14,477 seeded SQLite faces | First Photos 768.270 ms; first Faces publication 1,624.644 ms; background metadata 504.387 ms; cached tail 44.057 ms | Generated 1px images; excludes inference/full-resolution decode. |
| Same People fixture | Max event-pump gap 84.028 ms; cancellation acknowledgement 0.851 ms; retained Python allocations 30,082,868 bytes | Event-pump gap is not one UI callback; traced allocation is not RSS; acknowledgement is not durable drain. |
| 100,000 paths / 100 Sectioned Gallery sections | Preparation median 57.370 ms; Qt commit 6.860 ms; max event-pump gap 76.942 ms | In-memory paths; no source I/O, model load or real thumbnail decode. |

Initial **qualification targets**, not already achieved claims or hardware-independent unit-test limits: on the recorded reference host, UI callback p95 ≤16 ms/max ≤50 ms; input/heartbeat p95 ≤50 ms/max ≤100 ms; synthetic first Photos ≤1 s, first Faces ≤2 s, cached-tail reveal ≤100 ms and cancellation acknowledgement ≤100 ms. Measure sufficient samples before publishing p95. Separately publish per-operation safe-drain/commit limits, real-model/decode throughput and absolute bounded queue/cache limits from UX-28; these vary by fixture/backend and cannot be guessed from the metadata samples. Any target adjustment needs recorded baseline/reason and explicit review, not a silently raised threshold. Measure warm/cold behavior and mixed contention separately; no universal “UI never hangs” claim from one event-pump sample.

### Reproduction and implementation-time validation commands

These are **existing entry points**, not evidence that the new tests are implemented or passing. Run from the Clean checkout; the default test runner is offscreen and uses temporary runtime/settings. UX-33 closes remaining dependency/environment-isolation gaps. Use fresh output directories and record process exit codes; run benchmarks serially.

```bash
bash scripts/build.sh
bash scripts/run.sh
bash scripts/test.sh tests/test_theme_system.py tests/test_work_coordinator.py tests/test_production_support.py tests/test_ui_ux_acceptance.py tests/test_ui_smoke.py -vv --tb=short
bash scripts/test.sh
bash scripts/benchmark.sh
bash scripts/benchmark.sh --people-detect-workflow-only
bash scripts/benchmark.sh --sectioned-gallery-only
git diff --check
```

`scripts/run.sh` normally opens the user profile: for validation, supply a fresh isolated runtime/settings environment first or use the generated-fixture verifier. Never exercise destructive tests against that normal profile.

Example fresh evidence commands (shell directory variables are task-specific):

```bash
ui_evidence_parent="$(mktemp -d -t clusterlens-ui-XXXXXXXX)"
.venv/bin/python -B scripts/verify_ui_redesign.py --report-dir "$ui_evidence_parent/report"

native_evidence_parent="$(mktemp -d -t clusterlens-native-XXXXXXXX)"
.venv/bin/python -B scripts/verify_native_display.py --logical-size 1280x720 --report-dir "$native_evidence_parent/report"
```

Native verification requires an accessible real display and explicitly rejects headless Qt. New model/stress/fault entry points and exact invocation/fixture setup must be added to BUILD.md when UX-17/30/31 land; they do not exist merely because this plan names them.

### Audit-to-task traceability

| Audit finding | Required remediation/testing |
| --- | --- |
| A01 Production coordinator not used | UX-03–04, UX-31 |
| A02 Ask/CPU bypass, alias locks and capacity gaps | UX-03–04 |
| A03 Remaining UI-thread provider/DB/JSON/process work | UX-06–09 |
| A04 Workers read live controls | UX-05, UX-18 |
| A05 Actual light/dark contrast failures | UX-24, UX-27, UX-32 |
| A06 Find primary actions below viewport | UX-19, UX-27 |
| A07 Manage checkable groups do not collapse | UX-20, UX-27 |
| A08 Disabled Detect labels clipped | UX-21, UX-25, UX-27 |
| A09 Unnamed/review navigation mismatch | UX-22–23 |
| A10 Production state integration/walkthrough bug | UX-23 |
| A11 Incomplete action eligibility | UX-14, UX-02 |
| A12 Zero progress treated as unknown | UX-10 |
| A13 Stale footer and competing inline status | UX-11 |
| A14 Compact/default-size/modal Jobs limitations | UX-13, UX-26 |
| A15 Mixed owners and Loaded/Showing units | UX-12 |
| A16 Destructive maintenance cancellation/recovery | UX-16–18 |
| A17 Warm-worker disable does not stop idle child | UX-15 |
| A18 Text scale and monitor-derived minimum | UX-25, UX-32 |
| A19 Accessible names/focus evidence gaps | UX-26–27, UX-32 |
| A20 Verifier blind spots | UX-01, UX-27 |
| A21 Residual latency and unmeasured contention | UX-28–31 |
| A22 Four failures and incomplete release evidence | UX-01–02, UX-32–35 |

## Previous implementation records and remaining backlog

The dated sections below are retained, including their historical measurements and existing user work. References to “this pass” or “out of scope” there describe their original slice, not an exemption from the current plan. Historical Validation paragraphs record past results; they do not supersede the open UX-01–35 gates above.

## Readable adaptive UX redesign (canonical branch: `main`)

### 2026-09-24 production-readiness plan

Execute these phases in order. A later phase does not waive an earlier gate, and no item is complete until its contract and validation both pass.

#### Phase 0 — restore a trustworthy release gate

- [x] Eliminate the deterministic Qt/native test-process crashes and modal-test hangs.
  Implementation: reduce the `test_production_file_hdbscan_options_persist_and_enter_request` → `test_production_footer_storage_summary_and_clear_action` crash to the smallest shared lifecycle/state boundary; explicitly drain or stop every worker/thread/timer owned by a closed production window; make test teardown wait for Qt deferred deletion; and prevent `errorBox()`/other modal dialogs from entering a nested event loop in automated tests. Add an ordered regression that constructs, closes, and reconstructs the affected shell repeatedly in one process.
  Contract: closing a window leaves no running `QThread`, timer callback, queued signal, global runtime override, or owned widget capable of touching a later test/window; expected validation errors are observable without blocking headless execution; normal interactive dialogs retain their current behavior.
  Validation: the ordered lifecycle pair passed 20 consecutive in-process repetitions with `PYTHONFAULTHANDLER=1`; the missing-model/no-root test exits normally; and two independent full `bash scripts/test.sh` processes passed 649 tests (159.88 s and 162.55 s) with no signal, timeout, or modal hang.

- [x] Reconcile the test suite with the current product contracts, fixing product code where the UI promise is wrong and fixing tests only where the expectation is obsolete.
  Closed 2026-09-26: UX-01–02 fixed the four reproduced failures without weakening the documented contracts; UX-35 retains final release-wide requalification.
  Implementation: classify each current failure before editing it. Update the album paging expectation so Review-only pending proposals remain excluded from normal All Faces; update CPU/CUDA-dependent face-clustering defaults to derive from the deterministic injected capability rather than developer-machine state; supply active-root fixtures to scoped face-search tests; and keep enabled actions aligned with the same scope/readiness predicate used by their handler.
  Contract: no test passes by weakening active-root isolation, pending-proposal separation, read-only safety, cancellation, or provider-selection behavior; an enabled action must be executable in the current state, otherwise it is disabled with an actionable reason.
  Validation: focused album paging, face context-menu, query-photo, scan/refresh, clustering-default, model-readiness, and production lifecycle tests pass independently and in suite order. The current exact-tree clean-source verifier independently passed two complete 949-test runs from fresh temporary runtimes.

#### Phase 1 — make first-run and status UX coherent

- [x] Implement one shared no-active-roots state across Library, Organize, People, and Tools.
  Implementation: route every empty-scope workspace through one reusable state with a concise explanation and one **Edit roots** primary action. Disable or hide catalog filters, album saving, People loading/search, organizing, duplicate scans, and cleanup actions that cannot operate without scope; retain task-specific empty-result states only after roots exist.
  Contract: an empty scope is never presented as a completed empty search/scan or an in-progress load. A user can establish scope from every workspace in one keyboard-accessible action, and applying roots refreshes only the active workspace while invalidating the others for lazy refresh.
  Validation: the production empty-scope regression asserts all four workspace routes, one primary **Edit roots** action, no lazy People construction, hidden subordinate navigation, and no new Jobs; `.venv/bin/python -B scripts/verify_ui_redesign.py --report-dir /tmp/clusterlens-production-ui-20260924-final` passed at 1280×720 and 1920×1080 with no clipped controls.

- [x] Replace contradictory runtime and loading messages with one explicit state machine.
  Implementation: `RuntimeBadge` has mutually exclusive `checking`, `ready`, `fallback`, `unavailable`, and `failed` states. A new request resets the badge to checking, stale readiness completions are generation-discarded, and error/cancellation produces an explicit retry path. Provider/model detail remains in the badge tooltip, Jobs, and Settings; workspace loading copy remains owned by real workspace requests.
  Contract: the header never combines text such as `Runtime checking…` and `Ready`; background readiness is not confused with catalog, People, or storage loading; every degraded/failed state provides a concrete next action. The runtime badge's accessible name now announces the same label/state shown visually.
  Validation: focused runtime-badge tests passed 3/3 and the shell transition regression covers stale replacement plus cancellation. `.venv/bin/python -B scripts/verify_runtime_badge.py --report-dir /tmp/clusterlens-runtime-badge-20260924` passed, capturing all five states in dark and light themes and verifying each visible state, semantic property, tooltip detail, and accessibility name.

- [x] Make People actions state-valid and keep `Loaded`/`Showing` terminology uniform.
  Closed 2026-09-26: UX-12/14 provide the shared count contract, owner preservation, eligibility predicate and direct-handler guards while retaining scope/readiness protections.
  Implementation: use one eligibility function for Find Similar, Ignore Similar, Detect, Index/Reindex, naming, and clustering across buttons, menus, keyboard actions, and handlers. Retain `Loaded X` for the unfiltered folder-review snapshot and `Showing Y` for the current filtered/paged surface; derived result galleries report only `Showing Y`.
  Contract: no action appears enabled if its handler will reject it for missing roots, missing models, read-only mode, empty/invalid selection, or inactive result scope. Count labels never mix photos, faces, groups, loaded rows, and visible rows.
  Validation: the table-driven People eligibility regression and focused scope/readiness/menu suite passed 12/12, covering zero/one/multiple selections, no roots, active roots, all-indexed override, checking/missing models, read-only mode, ignored rows, and guarded handlers. Existing folder-review/derived-gallery count regressions cover loaded versus showing counts across filters and paging.

- [x] Correct Jobs history presentation and completion semantics.
  Closed 2026-09-26: UX-10–13 cover zero/unknown progress, terminal precedence, compact context, readable Details, incremental history and modeless monitoring.
  Implementation: render indeterminate work distinctly while active, but show completed jobs as complete or as `Done` rather than 0%; prevent no-op empty-root navigation from creating misleading completed jobs; make the Detail column readable without mandatory horizontal scrolling; and disable Cancel when the selected row is not cancellable.
  Contract: status, progress, cache state, and detail never contradict one another; every active job has a visible owner and cancellation state; completed/cancelled/failed rows remain distinguishable and accessible.
  Validation: deterministic 0/unknown/partial/complete/cancelled/failed fixtures, keyboard selection/cancel coverage, and the 500-row incremental-refresh benchmark pass. The Jobs fixture is included in the 1280×720 offscreen evidence and uses a readable 654 px Detail column on this host.

- [x] Fix visible text and accessibility-verifier coverage gaps.
  Closed 2026-09-26: UX-21/26–27 enforce literal labels, clipping, meaningful names and complete logical keyboard coverage. Native screen-reader qualification remains separately open under UX-32.
  Implementation: escape literal ampersands in Qt tab labels such as Sources & Library, Compute & Models, Safety & Storage, and Batch Rename & Metadata; extend the verifier beyond top-level focus proxies so Settings, Jobs, and Inspector report their actual keyboard traversal and accessible names.
  Contract: user-facing labels render literally and consistently, every actionable control is reachable with a meaningful accessible name, and verifier success cannot be produced by an empty focus-order result.
  Validation: focused Settings/People label regressions passed. The offscreen verifier passed at 1280×720 and 1920×1080 in dark/light coverage, recording real Tab-key traversal plus every visible focusable Settings, Jobs, and Inspector control; empty traversal or an unnamed actionable control now fails the report.

#### Phase 2 — close measured performance and responsiveness gaps

- [x] Add the missing 11,284-photo / approximately 14,477-face People Detect end-to-end fixture before changing limits.
  Implementation: generate a fixed-seed temporary SQLite/source fixture and measure first Photos content, completion of the background metadata cache, tail reveal, first active-Faces rows, full active model publication, cancellation latency, peak retained metadata, and maximum Qt event-loop slice/queue delay. Separate SQLite/path validation, model publication, crop/thumbnail work, and paint measurements.
  Contract: opening Detect publishes at most the first 500 photos, does not construct hidden Faces rows, never resets visible scroll/selection during background caching, and remains cancellable without stale publication. The benchmark reads no user runtime or network data.
  Validation: `bash scripts/benchmark.sh --people-detect-workflow-only` passed against 11,284 temporary 1px PNG source files and 14,477 seeded SQLite faces. Its untraced run measured 575.856 ms first Photos, 566.021 ms cache completion after the controlled release, 1,189.821 ms visible Faces publication, 42.908 ms cached-tail reveal, 5.524 ms cancellation acknowledgement, and a 75.044 ms maximum observed offscreen event-pump gap; a separate traced workflow retained 30,031,853 bytes with a 31,262,770-byte peak. It asserts the 500-photo initial bound, no hidden Faces rows, source-order cached offsets, no tail re-query, and no stale cancellation publication. The focused offscreen cache/tail regression also passes. Method/results and the baseline-derived watch budgets are recorded in `BENCHMARKS.md` and `HOTSPOTS.md`.

- [x] Remove redundant path canonicalization from People paging without weakening scope or symlink safety.
  Implementation: profile the 10,001-photo fixture, carry already-canonical candidate identities through page queries, and avoid repeating `Path.resolve()`/`stat()` for the same generation while invalidating the cache on scope or source changes. Do not bypass canonicalization at an untrusted boundary.
  Contract: scoped paths, symlink escapes, missing/reconnected roots, and generation cancellation behave exactly as before; memory stays bounded by the active snapshot/page policy.
  Validation: five fresh 11,284-photo / 14,477-face fixtures measured 223.292 ms first-page / 5,134.527 ms complete-cache median without reuse and 225.710 ms / 253.105 ms with the snapshot; its separate traced retention/peak was 1,764,104 / 2,648,135 bytes. Scoped, outside-path, symlink-escape, source-change invalidation, and offscreen tail-publication regressions pass. The profile and residual bounded page work are recorded in `BENCHMARKS.md` and `HOTSPOTS.md`.

- [x] Move proportional full-gallery publication off latency-sensitive UI paths.
  Implementation: `SectionedGallery` builds an immutable normalized section/row snapshot in an owned cancellable worker for hierarchies of 2,000+ declared paths, including explicit collapsed Timeline publication and resize-driven column reflow. The UI atomically applies only that ready snapshot, restores stable selection/scroll, and ignores stale generations; small hierarchies retain the direct low-overhead path.
  Contract: scrolling, resizing, image arrival, filters, and background page completion do no proportional normalization or row construction on the Qt UI thread. A genuine large hierarchy/reflow keeps the current gallery visible until a current generation is ready; cancellation, stale completion, and shutdown cannot publish obsolete rows.
  Validation: `bash scripts/benchmark.sh --sectioned-gallery-only` passed on the deterministic 100,000-path/100-section in-memory fixture: worker preparation p50/p95 was 56.731/59.654 ms, the UI commit was 6.104 ms, worker-to-UI publication 77.239 ms, and the maximum observed event-pump gap 58.440 ms. The focused 12-test Sectioned Gallery module covers viewport bounds/spans, stale worker discard, explicit collapse publication, large resize background reflow with selection restoration, and worker shutdown.

- [ ] Complete the UI-thread and resource-contention audit for every production workflow.
  Implementation: instrument discovery, metadata/EXIF, thumbnails, face crops, inference/model loading, embedding, clustering, duplicate hashing, SQLite maintenance, backup/restore, and storage clearing. Route expensive work through owned cancellable jobs; add explicit same-asset/Data-Home/GPU conflict rules and bounded queues.
  Contract: no source I/O, database migration, image decode, model construction, inference, hashing, or proportional model preparation occurs on the Qt thread; conflicting work is queued or rejected with a visible explanation; shutdown/cancellation leaves storage coherent.
  Validation: deterministic injected-slow-worker tests assert event-loop progress and cancellation bounds; job dependency/priority/GPU-policy tests pass; CPU, memory, I/O, queue depth, contention, and fallback findings are recorded in `HOTSPOTS.md`, with measurements in `BENCHMARKS.md`.

#### Phase 3 — feature completeness and release hardening

- [x] Make an explicit product decision for retained-but-hidden features.
  Closed 2026-09-26: UX-19–23 expose and qualify Find, Manage, Review and Review & Name as synchronized, persisted, keyboard-reachable production routes with genuinely collapsed destructive controls.
  Implementation: expose every retained People capability in the four-workspace shell: **Review & Name** opens Folder Review's saved-face naming sub-surface, **Find** visibly includes the walkthrough and persisted saved searches, and **Manage** opens identities, import/export, maintenance, and the collapsed safeguarded danger zone. Keep the existing **Review** people-cleanup route separate. Update the standalone task selector and state restoration so these surfaces remain keyboard reachable and persist normally.
  Contract: no documented user data or capability is silently unreachable; the visible Manage route retains read-only restrictions, confirmations, and collapsed destructive controls; hidden widgets do not start work merely by being constructed.
  Validation: the People route/layout suite passed 6/6, covering shell Review & Name/Manage selection, Folder Review child-tab activation, standalone Manage keyboard navigation, saved route state, and responsive Find layout. `README.md` and `ARCHITECTURE.md` contain the public capability inventory.

- [ ] Validate recovery under forced interruption and incoherent intermediate states.
  Implementation: inject cancellation/process interruption at every durable boundary in indexing, catalog refresh, face writes, rename, metadata embed/sidecar save, Trash/restore, cache clearing, backup/restore, and Data Home relocation. Exercise restart, resume, rollback, idempotent retry, corrupt/stale temp files, unavailable roots, missing models, and disk-full/write-error paths.
  Contract: source media is never silently lost or partially overwritten; derived stores either recover automatically or present one safe resume/rollback action; repeated recovery is idempotent; the UI remains responsive and accurately reports the surviving state.
  Validation: the isolated real-file Trash/restart verifier passed at `/tmp/clusterlens-trash-recovery-20260924-304/trash_recovery.json`, preserving collision/partial/restart/skip/unique-name cases and the operation journal. The evidence writer records the skip-policy result at its assertion boundary, before the following unique-name restore intentionally moves the retained copy. Existing deterministic service tests cover interrupted Data Home relocation resume/rollback, corrupt/interrupted thumbnail-index recovery, corrupt tag-store quarantine, interrupted face-model promotion, and atomic face-index transactions. A unified forced-interruption matrix for every durable boundary, disk-full/write-error injection, and the manual interrupted CPU/GPU/native smoke remain required.

- [ ] Complete build, package, native-display, clean-profile, upgrade, and uninstall validation.
  Implementation: run the documented build/run/test/benchmark commands from a clean checkout; verify the packaged executable on a clean VM/profile; test single-instance activation, first launch, model-missing/install/cancel/retry, CPU-only and CUDA-capable runtime policy, upgrade from the prior supported schema/profile, read-only mode, backup/restore, and uninstall while preserving user media and documented retained data.
  Contract: the packaged application starts without source-tree state, gives actionable dependency/provider errors, never reads the developer runtime, and install/uninstall/upgrade behavior matches `BUILD.md` and `README.md`.
  Validation: `bash scripts/build.sh`, `bash scripts/benchmark.sh`, offscreen report `/tmp/clusterlens-production-ui-20260924-303/ui_redesign.json`, native report `/tmp/clusterlens-native-display-20260924-303/native_display.json`, and the rebuilt CPU packaged-launch report `/tmp/clusterlens-packaged-report-20260924-303/packaged_launch_verifier.json` pass. Clean-VM install/upgrade/uninstall, clean-profile upgrade, GPU-package, and manual interrupted-job validation remain required before this item can close.

#### Production release gate

- [ ] Declare production-ready only when every Phase 0–3 item above and the authoritative UX-01–35 remediation/release gates are complete.
  Updated 2026-09-25: historical passing evidence below does not close the current failures, recovery, native or package qualification gaps.
  Required evidence: `bash scripts/build.sh`, two consecutive clean `bash scripts/test.sh` runs, `bash scripts/benchmark.sh`, offscreen UI evidence, native-display evidence, packaged-launch evidence, fault-injection/recovery results, clean-VM install/upgrade/uninstall results, and `git diff --check` all pass. `BENCHMARKS.md`, `HOTSPOTS.md`, `ARCHITECTURE.md`, `BUILD.md`, and `README.md` must describe the shipped behavior and known bounded limitations. No open P0/P1 defect, native crash, modal test hang, unbounded queue/cache, UI-thread I/O/compute, or undocumented feature gap may remain.

### 2026-09-23 People photo-count clarity

- [x] Make People folder-photo counts unambiguous.
  Contract: `Loaded` is the unfiltered folder-review snapshot; `Showing` is the active filtered gallery. Derived result galleries state only their showing count.
  Validation: focused offscreen folder-review and derived-gallery tests cover visible, empty, and all-photo filters.

### 2026-09-20 implementation slice

- [x] Cache People → Detect review metadata in the background while keeping the Photos viewport stable.
  Contract: opening Detect never starts All Faces; the first 500 saved-review photos appear immediately; a generation-guarded worker reads and prepares all remaining metadata without republishing Photos, resetting scroll/selection/tile caches, or constructing the hidden Faces model. A tail request reveals one already-cached page. Faces and Face Groups wait for cache completion and are built only when their result tab is active. Whole-review actions remain explicit until every page is visible.
  Validation: deterministic post-commit batch callback, initial Detect route, and offscreen cache/tail test pass; the focused Detect/Faces run is 3 passed. The seeded 10,001-photo service capacity fixture recorded a 200.904 ms first-page median and 4,147.547 ms complete-cache median on this host in `BENCHMARKS.md`; residual canonical-path and metadata-cache costs are in `HOTSPOTS.md`.

- [x] Remove the measured visible-tile and Jobs-refresh UI work, complete the compact-theme preference, and tighten review/search flows.
  Contract: Sectioned Gallery queues only visible rows and reapplies spans after a changed hierarchy; Jobs visibly distinguishes queued work and updates only changed history rows; repeated current-theme application is a no-op; Search exposes local capture/camera/folder/type filters; duplicate groups open in the full-size main gallery; thumbnail letterboxing remains transparent; Esc targets the current foreground Job.
  Validation: focused offscreen UI/theme/scheduler/gallery suites, thumbnail orientation/transparency tests, duplicate-review tests, generated 100,000-path/500-job benchmarks, and `verify_ui_redesign.py` PASS at 1280×720 and 1920×1080 are recorded in `BENCHMARKS.md`.
- [x] Finish the remaining audit work: single-instance activation, a virtual thumbnail/contact-sheet review inside Cleanup, fixed-corpus FTS and high-collision duplicate profiling, and native-display/accessibility validation.
  Contract: a normal second GUI launch hands off activation without constructing a second window; Cleanup compares the selected group through one virtual table; catalog writes preserve FTS replacement semantics; duplicate recall remains unchanged; and native evidence checks focus, labels, contrast, visibility, and client bounds.
  Validation: deterministic primary/secondary IPC ownership tests, a virtual-preview UI acceptance test, 1,000-row fixed-FTS and 128-path collision profiles, a restored five-second People acceptance test, offscreen 1280×720/1920×1080 evidence, and a native Wayland evidence PASS are recorded in `BENCHMARKS.md`.

### Implemented foundation slices (the broader contracts below remain open)

- [x] Expose shared Sources/Data Home inventory, wider readable typography, visible Jobs history, and a zoom/pan/full-resolution inspector foundation.
  Acceptance: source roots remain separate from managed data; viewer controls and job history are keyboard-accessible; covered by focused offscreen smoke tests.
- [x] Add reusable autocomplete inputs to face inspection, Names mutations, and Library People Cleanup.
  Acceptance: a saved choice commits by click, Enter, or Tab; explicit creation remains distinct; focused picker/UI tests pass.
- [x] Add derived filename-date timeline policy and a preview-first journalled batch file rename.
  Acceptance: filename dates never rewrite metadata, an unsafe rename makes no file changes, completed renames restore through Recovery; deterministic catalog/recovery tests pass.
- [x] Make Timeline filename-date parsing configurable and its recatalog action discoverable.
  Contract: users can maintain a persisted, ordered list of safe year-first `strptime` filename patterns, see the accepted syntax beside the Timeline source selector, and force a cancellable recatalog from Timeline after a policy or pattern change. Filename parsing remains derived catalog data only and never edits EXIF/XMP or source files.
  Validation: `tests/test_local_archive_curation.py` passed 15/15, including custom parsing, malformed/ambiguous pattern rejection, and forced recomputation; focused offscreen Timeline controls passed 2/2. `bash scripts/build.sh`, `git diff --check`, and the generated 120-photo/1,200-row catalog regression benchmark passed; results are recorded in `BENCHMARKS.md`.
- [x] Extend Timeline filename-time rules for date counters, explicit Unix epochs, and opt-in numeric-ID fallback.
  Contract: readable rules parse `DDMMYYYY` plus a same-day sequence and explicit Unix seconds/milliseconds; an opt-in fallback accepts exactly one valid raw 10/13-digit Unix epoch from an ID/hash-looking name. The catalog retains a separate sequence sort key, stays UTC/deterministic, rejects ambiguous candidates, and never changes source metadata.
  Validation: `tests/test_local_archive_curation.py` passed 18/18, covering date-counter order, seconds/milliseconds epochs, opt-in raw hash values, counter/non-date rejection, ambiguity/range rejection, and a database migration. The offscreen persisted-preferences/refresh test passed. `bash scripts/benchmark.sh --library-catalog-only --photos 120 --timeline-photos 1200 --repeats 5` passed with every generated parser path exercised; results are recorded in `BENCHMARKS.md`.

- [x] Bound Timeline capture dates and make manual capture-time corrections visible.
  Contract: Timeline shows only 1991 through the current UTC year. Invalid filename/EXIF/filesystem values become the explicit **Unparsed filename time** row; filename-only mode never substitutes filesystem time. A valid capture time saved from the photo viewer’s reversible sidecar is explicit user intent and takes precedence after a Timeline date refresh.
  Validation: `tests/test_local_archive_curation.py` passed 23/23, including out-of-range years, an unparseable filename with valid EXIF, filename-only Unparsed grouping, stale-row ordering, and sidecar revision recataloging. Three focused Timeline UI tests plus the photo-viewer control test passed offscreen. `bash scripts/benchmark.sh --library-catalog-only --photos 120 --timeline-photos 1200 --repeats 5`, `bash scripts/build.sh`, and `git diff --check` passed; measured results are in `BENCHMARKS.md` and residual costs in `HOTSPOTS.md`.
- [x] Use EXIF DateTimeOriginal for unparsed filename-first Timeline entries.
  Contract: a valid filename date remains authoritative in the filename-first policy; when no filename rule yields a time, a valid camera DateTimeOriginal is used before the Unparsed row. The fallback never invents a modified-time date and missing/invalid DateTimeOriginal values remain unparsed.
  Validation: deterministic catalog coverage creates a UUID-like filename with DateTimeOriginal in the nested camera EXIF IFD, verifies its EXIF provenance and 2024 Timeline placement, while the no-metadata UUID regression remains in the Unparsed bucket. The shared popup metadata reader, focused catalog tests, and `bash scripts/build.sh` pass.
- [x] Add persisted Timeline hierarchy controls.
  Contract: the Timeline **Group photos** drop-down offers Year, Year → Month, Year → Month → Week, and Year → Month → Day. It uses the existing cancellable scalar date query, never rereads source media when regrouping, retains virtual/viewport-bounded thumbnail behavior, and leaves rows without a valid time in an explicit Unparsed branch.
  Validation: deterministic catalog coverage retains descending day and ISO-week buckets; offscreen UI coverage verifies each hierarchy's nested section IDs/titles and persistence. Focused Timeline tests, the source-free Timeline benchmark, `bash scripts/build.sh`, and `git diff --check` pass.

- [x] Show every photo date/time field in the standard inspector.
  Contract: the ordinary photo popup always distinguishes filesystem modification time, every EXIF date/time field, and any editable sidecar capture-time correction; no switch to an advanced route is needed to inspect those values.
  Validation: focused offscreen rendering covers EXIF capture/digitized/modified/GPS/offset/subsecond fields, an absent-EXIF state, and the sidecar correction label; the deterministic metadata-reader test covers standard EXIF capture/digitized/modified/offset fields. `bash scripts/build.sh` and `git diff --check` passed.
- [x] Add global source-admission controls for thumbnail-like names and minimum image/file sizes.
  Contract: Gallery, clustering, Faces, similarity search, and registered-Library scans share the persisted exclusion policy before accepting a source path. The policy never changes a source file; disabled rules retain existing behavior, dimension headers are read only in cancellable workers, and a Library rescan removes catalog entries that no longer qualify.
  Validation: 219 deterministic discovery/service/Library tests passed, including thumbnail/byte/dimension exclusions and stale catalog removal; focused offscreen production Settings coverage verified the persisted controls. `bash scripts/build.sh`, `git diff --check`, and both five-repeat all-admitted and source-filter catalog benchmarks passed. The filter fixture admitted 84 of 120 generated candidates and excluded 36; its measured profile is recorded in `BENCHMARKS.md` and `HOTSPOTS.md`.
- [x] Make applied source filters immediately observable and reconcile Library catalog results.
  Contract: Settings explicitly states whether thumbnail-name matching is on before save; Gallery reports the number excluded (including an all-filtered state), and a policy change starts a visible, cancellable registered-Library refresh so stale derived entries cannot remain visible.
  Validation: deterministic source-name coverage includes compact `thumbnail320`/`thumb300x200` forms while retaining `thumbprint`/`thumbnailist` non-matches; focused production Settings coverage asserts the live summary. Focused service/UI tests and `bash scripts/build.sh` pass.

- [x] Establish deterministic UX and performance baselines before each redesign slice.
  Contract: generated fixtures cover 500 and 10,000 model-backed assets without user media or mutable runtime state; compute-backed CPU/GPU fallback remains covered by the existing vector fixtures.
  Validation: `bash scripts/benchmark.sh --ux-workflow-only` reports p50/p95 first-content and full publication, queue delay, cache/I/O/backend applicability, profiler evidence, and the unavailable RSS provider in `BENCHMARKS.md`.
- [x] Make Sources the single global multi-root scope and expose a relocatable Data Home inventory.
  Contract: Photos, Faces, Organize, Timeline, search, and maintenance share the exact same selected roots; the user can inspect, back up, verify, rebuild, or relocate every managed-data category without touching source photos.
  Validation: existing deterministic root-scope/unavailable-root coverage plus `tests/test_data_home.py` verify category inventory, checksummed backup, relocation, resume/rollback, and source safety.
- [x] Ship the common visual system, accessible typography, contextual help, and Power-user mode.
  Contract: controls, icons, spacing, focus, loading/error states, tooltips, help affordances, and advanced details are consistent; Power-user mode exposes complete diagnostics and advanced settings without removing safeguards.
  Validation: semantic light/dark token tests, the 11-test redesign acceptance suite, focused inspector/help/mode coverage, and isolated 1280×720/1920×1080 evidence cover focus order, clipping, tooltips/help, icon labels, breakpoint bounds, and Power-user visibility.
- [x] Replace ad-hoc name entry with reusable autocomplete entity pickers.
  Contract: person assignments/renames always offer cancellable saved-name suggestions and consistently commit a highlighted result on click, Enter, or Tab; legal new values use an explicit create choice.
  Validation: picker tests cover casing, stale requests, keyboard/mouse commit, empty values, and new-value creation.
- [x] Upgrade the photo inspector into the shared zoomable viewer and staged editor.
  Contract: every standard and sectioned-gallery route opens the same fit/1:1/deep-zoom viewer with pan, a face-region toggle, face draft tools, staged sidecar metadata, explicit JPEG/TIFF embedding, and safe filename rename. Source writes are explicit, atomic, journalled, and cancellable between files.
  Validation: focused inspector/face-editor/metadata/service tests cover overlays, drafts, sidecars, supported EXIF fields, and the existing collision-safe rename path.
- [x] Make thumbnail loading viewport-first across normal and sectioned photo galleries.
  Contract: exact visible tiles are ordered from the viewport centre, every visible request enters before prefetch, stale queued work is invalidated on viewport changes, and section-header previews never delay visible photos. Active decodes finish safely but cannot repaint a newer viewport.
  Validation: deterministic queue promotion/stale-drop tests, visible batch tests, and sectioned-gallery smoke coverage pass offscreen.
- [x] Add shared inspector visual guidance and staged photo-metadata controls.
  Contract: every inspector exposes the face overlay toggle, contextual metadata help, consistent semantic typography/icons, reversible sidecar saves, and a separately confirmed embedded-metadata action.
  Validation: focused offscreen inspector checks plus deterministic sidecar/EXIF service tests pass.
- [x] Add timeline filename-date provenance, previewed safe batch rename, and exact/near/visual duplicate review.
  Contract: no source filename or metadata changes without preview/confirmation; all completed mutation steps are journalled for Undo/Recovery.
  Validation: seeded parser, collision, rename-undo, duplicate-classification, and cancellation fixtures pass.
- [x] Make every job visible, cancellable, dependency-aware, and resource-aware.
  Contract: independent work runs concurrently; same-asset/data-home/GPU conflicts are explained, queued, and cancellable; the user can choose automatic, queue-for-GPU, CPU fallback, or ask-on-conflict behavior.
  Validation: UX-03–13 deterministic job lifecycle, priority, conflict, cancellation, GPU-policy, progress/terminal, compact Jobs and recovery tests pass; the current order-isolation gate passes both 422-test/52-subtest orders with zero survivors.
- [x] Add Recovery Center, checksummed backups, restore previews, and resumable derived-data migration.
  Contract: interrupted moves, migrations, scans, renames, metadata edits, backups, restores, and cache maintenance leave data valid and provide a visible resume/rollback path.
  Validation: existing journal/recovery coverage plus deterministic Data Home backup-tamper, cancellation, resume, rollback, and source-preservation tests pass; Storage exposes the incomplete-move chooser in visible Jobs.
- [x] Profile and eliminate every measured avoidable hotspot in each migrated workflow.
  Contract: no UI-thread I/O/compute, unbounded queue/cache, duplicate query/decode, or unnecessary serialization remains in completed workflows; irreducible costs are bounded and documented.
  Validation: UX-28–29 retain baseline/profile/change/rerun evidence in `BENCHMARKS.md`, exact output/membership hashes and bounded memory/queue checks; remaining real-model and mixed-contention measurements are explicitly separate UX-30–31 qualification gates in `HOTSPOTS.md`.
- [x] Prevent duplicate desktop launches from multiplying full GUI, CUDA, and cache-maintenance processes.
  Contract: a normal launch uses a local endpoint keyed by canonical Data Home; a secondary launch requests activation and exits before `ProductionClusterApp` construction. `--new-instance` deliberately bypasses that guard for isolated profiles or debugging.
  Validation: deterministic primary/secondary handoff and stale-owner takeover tests pass; full system RSS/GPU telemetry remains a release-environment observation rather than an unmeasured performance claim.

## UI redesign

Completion audit (2026-09-19): the production shell now exposes only Library, Organize, People, and Tools; old route IDs migrate without dropping photo-set context. The shared Roots manager, contextual disclosure, four-panel inspector, six-category searchable Settings, semantic light/dark themes, compact/medium/wide breakpoints, global Jobs presentation, and task-specific empty/loading/partial/error states are implemented. `scripts/verify_ui_redesign.py` produces isolated app-only screenshots plus focus, clipping, bounds, and first-visible evidence at 1280×720 and 1920×1080 and refuses overwrite. The 104-test production/Library/theme/redesign bundle, focused 24-test inspector/organize bundle, build, UX benchmark, startup test, and whitespace check are the release gate for this section; exact measurements and residual constraints are in `BENCHMARKS.md` and `HOTSPOTS.md`.

### Matte typography, density, and task-rail clarity pass

- [x] Apply a shared matte light/dark palette, compact field-width contract, and concise semantic hierarchy.
  Implementation: use neutral graphite/slate and paper/charcoal tokens, 6 px control radii, 12/14/16/20 px text roles, numeric/short/medium field sizing, palette-aware icons, and contextual tooltips instead of repeated visible prose.
  Contract: theme switching updates text, focus, selection, icons, disabled controls, and primary/danger state together; text-heavy fields may grow while finite controls retain a stable readable width.
  Validation: `tests/test_theme_system.py` and the focused theme/production/UI suite pass; the deterministic offscreen evidence includes both dark and light Organize advanced controls.

- [x] Rebuild dense Organize and People task rails for narrow and wide layouts.
  Implementation: Organize advanced options use Models, Grouping, and Runtime tabs; Faces uses compact icon-backed Similar/Name/Detect/Index/Find actions; narrow source and query controls use shorter labels or an extra row before clipping.
  Contract: advanced controls remain reachable without changing saved options, every shortened action retains a full tooltip, and primary actions remain visible at 1280×720.
  Validation: `QT_QPA_PLATFORM=offscreen .venv/bin/python -B scripts/verify_ui_redesign.py --report-dir /tmp/clusterlens-typography-responsive-20260919` passed; focused source-rail tests passed 4/4 and focused theme/People/Organize tests passed 47/47.

### Roots drawer and adaptive Cleanup actions

- [x] Replace the persistent Roots action rail with an on-demand staged Sources/Catalog drawer.
  Implementation: Sources holds a checkable filesystem-tree draft and sticky Apply/Discard footer; Catalog holds compact registered-root status and catalog actions. Close of a dirty draft offers Apply, Discard, or Continue editing. Apply is the sole path that emits the shared scope change; Catalog mutations remain disabled for an unapplied draft or read-only safety mode.
  Contract: browsing never changes the active scope; discarded drafts cannot start discovery, indexing, or catalog work; committed roots remain canonical/non-overlapping; Data Home remains a Storage link rather than a source row.
  Validation: focused offscreen staged-root and Catalog tests pass, including browse-versus-draft, Apply/Discard, dirty Catalog safeguards, first-row actions, and active/registered status. `verify_ui_redesign.py` records Sources and Catalog screenshots at each supported evidence size using generated empty directories only.

- [x] Make duplicate actions consume available horizontal space and show review controls only when useful.
  Implementation: Cleanup scan actions use the shared `ResponsiveFlowLayout`; the selection-only not-duplicate and Trash actions live in a second flow that is hidden until a valid duplicate group is selected.
  Contract: wide displays do not reserve empty grid rows, narrow displays wrap safely, all existing Job/confirmation/read-only behavior is retained, and the first duplicate result is selectable.
  Validation: focused offscreen Cleanup coverage checks wide/narrow flow heights, selection-only visibility, and index-zero group selection; the existing duplicate service and Trash-preview contracts remain covered by deterministic tests.

### Measured Timeline filter layout

- [x] Reflow Timeline filters from their measured content requirements rather than fixed screen-width breakpoints.
  Implementation: calculate one-row and two-row requirements from each primary cell's live `minimumSizeHint`, layout spacing, and the Timeline scroll viewport. Use a single row for From, To, Camera, Timeline date, Group photos, and Apply filters only when it fits; otherwise retain the semantic From/To/Camera plus date-policy/action rows, then stack only below the measured two-row requirement. Apply the same measured contract to Advanced date parsing and derive tab height from the active form, with a bounded scroll fallback for genuinely tall compact forms.
  Contract: date fields retain their natural short width while descriptive controls consume spare width; a wide Timeline has no artificial vertical scrollbar or blank panel; resize/font/style changes preserve entered values and never query the catalog, media, EXIF, or thumbnails.
  Validation: `tests/test_production_support.py -k 'library_compact_layout or library_timeline_controls_reflow'` passes with wide/medium/narrow grid-position, size-allocation, and scrollbar assertions. `verify_ui_redesign.py` records measured widths, positions, mode, and contract status at every evidence size.

### Phase 1 — baseline, terminology, and shell consistency

- [x] Capture a deterministic UI baseline before rearranging the shell.
  Implementation: extend the isolated native/offscreen display fixtures to capture Library, Organize/Clustering, Faces, Names, Tags, the photo inspector, Jobs, and Settings at representative wide and compact logical sizes. Record widget visibility, focus order, clipped text, content bounds, and time to first visible placeholder/content without reading user media or persistent runtime data.
  Contract: every later layout change has a reproducible before/after comparison; the verifier captures ClusterLens widgets only and never captures the surrounding desktop.
  Validation: the isolated UI-evidence command exits successfully from a clean checkout, produces deterministic reports/screenshots under an explicit output directory, and refuses to overwrite existing evidence.

- [x] Remove obsolete user-facing Gallery terminology and repair navigation/help contracts.
  Implementation: replace destination-specific labels such as **Open in Gallery**, **Main Gallery**, and **Gallery set** with **Open photos**, **View selected photos**, **Review matches**, or **Current photo set** as appropriate. Update tooltips, accessible names, status messages, help text, onboarding, and documentation while retaining internal class names where renaming would add risk. Fix the shortcut map and its help so `Ctrl+1` through the final workspace are unique and match the visible navigation.
  Contract: no visible control promises a removed top-level Gallery workspace; legacy persisted Gallery routes still migrate safely to the supported photo surface, and every advertised shortcut opens exactly one matching destination.
  Validation: repository string checks plus offscreen shortcut, tooltip, accessible-name, legacy-route, and help-dialog tests pass.

- [x] Replace the global Basic/Advanced selector with contextual disclosure and a persistent Power-user preference.
  Implementation: remove the mode selector from Library, Names/People, Tags, and other workspaces where it has no effect. Put **Advanced options** drawers/tabs in the workspace that owns each option; keep the global Power-user preference as the explicit way to reveal complete diagnostics, pipeline controls, and destructive administration surfaces.
  Contract: basic mode never removes data or safety controls; Power-user mode reveals advanced controls without changing algorithms, confirmation requirements, read-only mode, or saved values. Existing Basic/Advanced settings migrate predictably.
  Validation: offscreen visibility/state-migration tests cover every workspace in both modes, including keyboard access and persistence after restart.

- [x] Simplify the application header and eliminate duplicate status surfaces.
  Implementation: retain product identity, primary workspace navigation, compact scope summary, Jobs, and Settings in the header. Move runtime details into a warning badge/Jobs/Settings, remove the repurposed folder/activity label, and show active-job progress/cancellation in one global presentation rather than both header and footer. Keep the footer only for concise transient status when useful.
  Contract: root scope, active work, cancellation, runtime problems, and Settings remain reachable within one click; idle startup/runtime prose does not consume persistent horizontal or vertical space; no status is lost when the footer is hidden.
  Validation: offscreen wide/compact layout tests verify unique status ownership, no duplicate progress indicators, job cancellation, runtime-warning disclosure, focus order, and stable header height.

### Phase 2 — information architecture and workspace rearrangement

- [x] Introduce the four-workspace information architecture: Library, Organize, People, and Tools.
  Implementation: add the new navigation model and route current capabilities as follows: Timeline/Photos/Search/Albums to Library; clustering/tags/context to Organize; Faces/Names to People; duplicate cleanup/metadata and rename utilities/recovery entry points to Tools. Preserve temporary photo-set routes and back navigation. Add versioned migration for saved workspace IDs and shortcuts.
  Contract: existing capabilities remain reachable, saved legacy workspaces open the nearest new destination, route context is retained, and changing workspaces never starts indexing, clustering, scanning, or source mutation implicitly.
  Validation: a route matrix test covers every old/new workspace ID, startup migration, return-to-source behavior, shortcut activation, stale-work cancellation, and persistence across restart.

- [x] Build one Roots manager as the source-of-truth UI for all workspaces.
  Implementation: consolidate active and registered-root presentation into reusable root rows showing enabled/disabled, online/missing, catalogued/not catalogued, photo count, last refresh, current job, and contextual Refresh/Pause/Reveal/Remove actions. Keep Data Home visually and semantically separate from photo sources. Allow the manager to collapse into the compact scope chip.
  Contract: all workspaces use the same canonical root set; catalog registration remains explicit but is presented as a state of a root rather than a separate source system; nested roots remain de-duplicated; remove/disable/refresh never deletes source photos.
  Validation: deterministic multi-root tests cover add, nested collapse, missing/reconnected roots, enable/disable, registration, refresh, removal, cross-workspace propagation, Data Home separation, cancellation, and read-only behavior.

- [x] Rebuild Library around Timeline, All Photos, Search, and Albums.
  Implementation: make these the Library browsing tabs; default Timeline to DateTimeOriginal-first; keep date source, grouping, date range, and camera in the primary filter grid; move filename rules, epoch recognition, derived-date rebuilding, and other specialist controls under **Advanced date parsing**. Present active filters as removable chips and show Save Album only when a query differs from the current saved state. Remove Cleanup, People Cleanup, and Cluster Context from Library after their new destinations are available.
  Contract: opening Library shows photos or a useful add/refresh-root empty state; changing grouping never rescans source media; advanced parser values and current date provenance remain inspectable; no saved albums or timeline preferences are lost during migration.
  Validation: focused catalog/UI tests cover default policy, filter chips, grouping without source reads, advanced disclosure, album dirty state, empty/loading/error states, preference migration, and registered-root refresh cancellation.

- [x] Replace Clustering with the photo-first Organize workspace.
  Implementation: show active-root or routed photos immediately; expose one primary **Organize photos** action; add clear presets such as Similar scenes, Events, Documents, People-heavy, and Custom; replace the same grid with collapsible result groups. Put backend/model/tuning controls in an Advanced clustering drawer, comparison in an optional details drawer, Tags in its own Organize tab, and Cluster Context in a Power-user tab or drawer.
  Contract: Organize does not scan or mutate photos on entry; preset choice fully determines the visible request summary before execution; all long work is visible/cancellable; switching presets or routes cannot publish stale results; tag and context data remain durable.
  Validation: offscreen workflow tests cover immediate photos, each preset request, advanced option persistence/cache-key separation, Tags routing, context routing, result replacement, return navigation, cancellation, stale publication, and read-only mode.

- [x] Merge Faces and Names into the People workspace.
  Implementation: provide **People**, **Unnamed**, **Review**, and **Find** tabs. Move the durable Names list/photo view into People; grouped unlabeled faces into Unnamed; detected/ignored/proposed decisions into Review; and query-photo/name similarity into Find. Present indexing as a status card/action rather than the default content. Restrict pipeline tuning, identity prototypes, import/export, database rebuild/delete, and other administration to Power-user disclosure or Settings.
  Contract: durable names, face refs, ignored state, proposals, active-root scope, multi-selection, autocomplete, similarity results, and exact photo context survive the rearrangement. Opening People never indexes or rescans implicitly. Every face mutation remains face-region-specific, journalled where applicable, metadata-aware, cancellable, and read-only safe.
  Validation: a migration/routing matrix plus deterministic service/offscreen tests cover all four tabs, existing named/unnamed/ignored/proposed records, multi-face operations, deep similarity confirmation, hover previews, lazy initialization, stale-result cancellation, Power-user visibility, and shutdown.

- [x] Add the Tools workspace for cleanup, metadata utilities, and recovery entry points.
  Implementation: provide task-oriented tabs for **Duplicates**, **Batch Rename & Metadata**, and **Recovery**. Keep hash duplicates, similar duplicates, combined review, and bursts as explicit scan choices. Show keeper/candidate groups with unchecked destructive selections, recoverable Trash language, preview/confirmation, and direct links to the global Jobs and full Recovery settings when needed.
  Contract: no scan moves files; no candidate is selected for Trash by default; exact and similar classifications remain distinguishable; batch rename/metadata writes remain previewed, collision-safe, journalled, cancellable between files, and recoverable under their existing contracts.
  Validation: deterministic duplicate classification, unchecked selection, keeper exclusion, rename collision, sidecar/embed distinction, partial cancellation, Trash/restore, operation-journal, read-only, and cross-workspace refresh tests pass.

### Phase 3 — photo viewer, settings, and responsive presentation

- [x] Add system, light, and dark appearance modes with a photo-aware viewer canvas.
  Implementation: introduce semantic light/dark tokens and a live theme manager; add General settings for **Follow system**, **Light**, and **Dark**, plus viewer backgrounds for **Adaptive neutral**, **Follow app theme**, **Black**, **Middle gray**, and **Light gray**. Classify only the already-decoded preview's outer pixels at a bounded 32×32 sample, keep the result neutral, and refresh custom painters/icons without rereading the source photo. Preserve the historical dark appearance for migrated profiles while new profiles default to system theme and adaptive viewer canvas.
  Contract: the application chrome never changes per photo; system-theme changes apply live only in Follow system mode; Settings Cancel changes nothing; the adaptive classifier performs no source I/O or GPU work and falls back to the active theme when a useful border sample is unavailable. Text, focus, selection, face regions, danger state, and disabled state remain distinguishable in both app themes.
  Validation: deterministic unit/offscreen tests cover preference validation and migration, system-mode resolution, live switching, Settings Save/Cancel, icon/custom-painter refresh, all viewer background choices, dark/mid/light/transparent image samples, preview navigation, and no additional decode. Record bounded classifier/theme-switch measurements in `BENCHMARKS.md` and remaining repaint/platform constraints in `HOTSPOTS.md`.

- [x] Reorganize the shared photo inspector into Info, People, Metadata, and EXIF panels.
  Implementation: keep the image as the dominant surface with a compact navigation/zoom/face-overlay/fullscreen toolbar. Put filename/path/dimensions/dates/camera/location in Info; face boxes/naming/rescan/draw tools in People; curated staged fields and rename in Metadata; and a searchable, copyable, complete read-only metadata table in EXIF. Show Save/Discard only for dirty staged changes and move infrequent/destructive face actions to a labelled overflow menu.
  Contract: every photo route opens the same viewer; keyboard navigation, pointer-centred zoom, pan, 1:1, full resolution, face-region editing, autocomplete, manual capture time, sidecar save, confirmed embed, and journalled rename remain available. Closing or navigating with dirty state requires an explicit choice and never silently discards or writes changes.
  Validation: focused inspector tests cover every panel, focus/shortcut behavior, dirty-state navigation/close, face overlays and edits, complete date/EXIF display, copy/search, sidecar versus embedded writes, rename preview/recovery, cancellation, and compact layout.

- [x] Reorganize Settings into General, Sources & Library, Compute & Models, Safety & Storage, Advanced, and About.
  Implementation: merge the current model tabs under Compute & Models; place source filters/catalog behavior with Sources & Library; keep read-only, recovery, Data Home, backups, and category clearing under Safety & Storage; move raw runtime diagnostics, logs, support export, pipeline details, and update-channel/developer options under Advanced. Add in-tab search and preserve direct deep links from errors/help buttons.
  Contract: every existing setting/action remains reachable; advanced and dangerous controls are clearly separated; accepting/cancelling retains current transactional behavior; deep links focus and reveal the requested setting; no cache/storage action changes scope.
  Validation: a settings-route inventory test proves every former control has one destination; offscreen tests cover search, deep links, saved/cancelled values, Power-user visibility, category-specific clearing, backup/verify/relocation, recovery, keyboard order, and compact sizing.

- [x] Implement responsive workspace breakpoints down to 1280×720.
  Implementation: support wide layouts with persistent rails/drawers, medium layouts with collapsible rails and slide-over details, and compact layouts with one primary content pane plus explicit drawers. Replace fixed minimum/maximum widths that cause clipping; preserve user-resized splitter state per breakpoint without applying incompatible sizes after a display change.
  Contract: 1280×720, 1366×768, 1440×900, 1920×1080, ultrawide, high-DPI, and portrait-supported layouts expose all primary actions without mandatory horizontal scrolling or off-screen dialogs. Compact presentation does not hide safety, Jobs, cancellation, or dirty-state actions.
  Validation: parameterized offscreen/native geometry tests assert widget bounds, drawer reachability, dialog containment, no clipped primary labels, splitter migration, DPI/text scaling, keyboard traversal, and screenshot evidence at every supported breakpoint.

- [x] Complete the shared typography, iconography, spacing, selection, and accessibility pass.
  Implementation: enforce the documented 12/14/16/20 type scale, 8 px spacing grid, consistent section hierarchy, minimum target heights, one primary action per region, semantic success/warning/danger styling, and distinct icons for Library/Organize/People/Tools/Jobs/Settings. Use icon-only controls only for universal actions with tooltips and accessible names. Standardize verbs and plural/count formatting.
  Contract: state is never communicated by color or icon alone; focus is always visible; disabled controls explain why; text remains readable at supported scaling; selection and destructive scope are explicit; help is contextual without persistent instructional clutter.
  Validation: automated style-token/string/accessibility checks plus offscreen focus, contrast proxy, tooltip, accessible-name, target-size, text-scaling, selection, and destructive-action tests pass; native screenshots receive a manual release review checklist.

### Phase 4 — feedback, performance, and release validation

- [x] Standardize empty, loading, partial, error, success, selection, and job states across every workspace.
  Implementation: provide task-specific empty states with one primary next step; viewport-first skeleton tiles; `Loaded N of M` partial progress; retryable inline errors; a contextual selection action bar; and concise success notifications with Undo/Recovery where supported. Keep independent jobs in the global tray with origin, CPU/GPU resource, queue/running state, progress, fallback, pause/cancel capability, and outcome.
  Contract: no large content surface is blank while work is pending; passive loading does not steal foreground progress; stale work cannot repaint a newer view; cancellation is always safe and accurately reports completed versus pending work; GPU conflicts expose Queue/CPU fallback/Cancel according to policy.
  Validation: deterministic state-machine/offscreen tests cover first placeholder, viewport-first publication, progressive pages, empty/error/retry, stale cancellation, selection transitions, concurrent independent jobs, GPU contention/fallback, recovery notification, and shutdown with active work.

- [x] Validate the redesigned end-to-end task flows and keyboard model.
  Implementation: add deterministic fixtures for first run/add roots, browse Timeline, search/save album, organize photos, tag photos, name/review/find a person, inspect/edit a photo, review duplicates, batch rename, cancel work, and restore a file. Document matching keyboard shortcuts and contextual help.
  Contract: a new user can add a root and reach photos in no more than three explicit actions; each workspace has one obvious primary task; every mutation has preview/confirmation or staged Save/Discard; every background action is visible and cancellable; all flows are completable without a mouse.
  Validation: an automated workflow matrix plus a manual release checklist records action counts, focus order, screen-reader labels, cancellation, read-only behavior, recovery, and results for wide and compact layouts.

- [x] Profile and benchmark every redesigned high-volume surface before claiming responsiveness improvements.
  Implementation: run baseline → profile → change → benchmark for Library Timeline/Search, Organize publication, People lists/review/similarity, Tags, Tools duplicate review, inspector metadata/EXIF, Settings inventory, Roots manager, and Jobs presentation using generated seeded fixtures only. Measure first placeholder, first useful content, full publication, UI-thread stalls, CPU, memory, I/O, queue contention, and cancellation latency where deterministically available.
  Contract: the redesign introduces no UI-thread source I/O/model work, unbounded publication/cache, repeated decode/query, or hidden machine-state dependency; visual rearrangement does not claim speedups without measurements.
  Validation: the relevant deterministic test/build/run/benchmark commands pass from a clean checkout; before/after methods and results are appended to `BENCHMARKS.md`, remaining constraints to `HOTSPOTS.md`, architecture changes to `ARCHITECTURE.md`, and reproducible commands to `BUILD.md`/`README.md` before this section is marked complete.

## Complete Clean-app performance pass

- [x] Establish fresh deterministic baselines for every shipped benchmark path, including the progressive Faces review request path.
  Contract: all timing/profiling fixtures use generated temporary media or in-memory/database fixture data only; they never inspect user photos, models, caches, or runtime data.
  Validation: `bash scripts/benchmark.sh` and the focused Faces paging benchmark each exit successfully and their fixture/configuration/result are recorded in `BENCHMARKS.md`.
- [x] Profile and remove any measured avoidable UI-thread, repeated-work, or unbounded-work bottleneck without changing visible behavior.
  Contract: expensive source, SQLite, crop-decoding, and model work stays cancellable and worker-bound; visible lists retain bounded/virtualized publication and stale requests cannot publish.
  Validation: focused deterministic service/offscreen UI tests cover changed behavior; the affected benchmark is rerun after the change and `HOTSPOTS.md` explains both residual constraints and cache/concurrency bounds.
- [x] Verify reproducible release health after the performance pass.
  Contract: a clean checked-out source tree can compile, run its deterministic validations, and execute all benchmark scripts through documented commands.
  Validation: `bash scripts/build.sh`, relevant isolated `pytest` slices, `git diff --check`, and the relevant `scripts/benchmark.sh` commands pass.

## People → Faces correctness and performance pass

- [x] Restore Detect as a visible People route.
  Contract: **People → Detect** opens Folder Review, exposing the existing explicit Detect Faces and Index Active Roots actions; it does not start a scan merely by opening the route.
  Validation: `test_people_detect_route_opens_folder_review` verifies the shell-level route selects Folder Review and invokes only its lazy-load boundary.

- [x] Make normal All Faces memberships mutually exclusive and keep proposals in Review.
  Contract: a durable named face appears only under its saved person, a durable unnamed face appears only under Unlabelled, and an unaccepted proposal appears only in Review; compatibility adapters cannot reintroduce pending rows into the visible album.
  Validation: `test_face_album_keeps_pending_proposals_out_of_normal_membership` passes and the generated People benchmark asserts no pending group in normal album output.
- [x] Remove duplicate People routing/reloads and reject stale Find work.
  Contract: a shell route delegates one lazy load to the active tab; inactive People data remains stale until opened; late query detection/similarity results cannot overwrite a changed photo, scope, or tab.
  Validation: focused All Faces UI regression plus request-token checks in `SearchPane`; `git diff --check` and source compilation pass.
- [x] Bound first All Faces content and batch similarity-record hydration.
  Contract: first publication reads a single consistent group/member SQLite snapshot, only 500 selected-group members, and ANN/multi-face matching hydrates records in SQL batches rather than opening one connection per candidate.
  Validation: `bash scripts/benchmark.sh --people-faces-only --faces 5000 --repeats 3` records a 12.352 ms median temporary-SQLite baseline; `test_search_similar_faces_combines_multiple_example_embeddings` passes.

## ClusterLens source-parity reconciliation

- [x] Port source-only behavior without replacing Clean-native Library, Names, or parallel indexing implementations.
  Contract: Clean remains a superset at the user-facing feature level; shared files are reconciled feature-by-feature rather than copied over dirty local work.
  Validation: reconciled the source worktree against Clean-native equivalents; focused service and offscreen UI coverage exercises the newly ported behavior, with documented deterministic checks below.
- [x] Add durable per-face ignore/restore and catalog-name completion from the current ClusterLens workspace.
  Contract: ignored regions leave normal Face search/clustering and are recoverable in an explicit Faces view; all face naming fields and prompts autocomplete saved names while accepting new names.
  Validation: `uv run --with pytest python -m pytest tests/test_services.py -k 'face_folder_review_distinguishes_detected_no_faces_tiny_hidden_and_not_scanned or hidden_people_and_faces_persist_and_are_excluded_by_default or nested_external_scrfd_and_arcface_files_are_discovered_without_copying'` and `QT_QPA_PLATFORM=offscreen uv run --with pytest python -m pytest tests/test_ui_smoke.py -k 'faces_filter_ignored_and_unlabelled_photo_views_and_name_picker'` pass.
- [x] Publish Faces progressively while folder-review and selected All Faces group data is still loading.
  Contract: a folder review resolves bounded, source-ordered 500-image pages in workers and shows its first page before later pages finish; detected-face tiles receive later pages incrementally and display visible loading feedback. All Faces requests 500 members per selected group page and automatically continues until that group is complete, cancelling a stale folder/filter/sort/mode/tab/group request. The final folder view applies the user's selected sort only after all page metadata is available; face crop decoding remains viewport-bounded.
  Validation: deterministic service paging covers 1,005 candidates and page offsets; offscreen UI coverage holds later pages behind an event to prove the initial 500 folder-review items and All Faces members publish before completion, then verifies all 1,001 items and automatic offsets `0, 500, 1000`. Focused service/UI suite, compilation, and whitespace checks pass.

## Adaptive GPU face-indexing pipeline

- [x] Feed the verified CUDA face pipeline with bounded parallel decode/quality work, cross-photo embedding batches, and batched durable writes.
  Contract: normal indexing remains incremental, cancellable, source-read-only, and label-safe; the actual detector/embedder provider and CPU fallback are visible throughout the Job.
  Validation: deterministic scheduler/persistence coverage, focused offscreen UI coverage, generated fixture benchmark/profile, CUDA SCRFD/ArcFace smoke, and documentation updates. Full-suite/build validation remains pending.
- [x] Add an explicit, cancellable **Reindex active roots** action.
  Contract: normal indexing keeps unchanged rows; the visible force action reports its active-root scope, preserves compatible durable labels, and has no hidden source mutation.
  Validation: focused offscreen action/force-request coverage and service-level replacement/label preservation coverage; full-suite/build validation remains pending.
- [x] Complete the full-suite run in a persistent terminal session.
  Contract: no test process is forcibly truncated before pytest prints its terminal summary.
  Validation: two persistent `bash scripts/test.sh` processes each exited 0 with a complete 649-test terminal summary (159.88 s and 162.55 s). `bash scripts/build.sh`, `git diff --check`, the generated benchmark suite, and focused service/path-scope/offscreen Faces slices also pass.

## Shared multi-root workspace scope

- [x] Replace the single selected source folder with a persisted, canonical active-root set shared by Gallery, Clustering, Faces, Names, Tags, and Library.
  Contract: each active root includes descendants; overlapping roots are collapsed; root changes cancel obsolete visible work and no workspace silently widens to the filesystem.
  Validation: deterministic path-scope/discovery/service coverage verifies persistence-compatible root normalization, overlap de-duplication, source-safe union discovery, and explicit empty-scope behavior; focused offscreen UI coverage verifies the active-root lifecycle.
- [x] Add a checkbox-based active-root picker and an always-visible workspace scope summary.
  Contract: clicking a tree row only browses; checking it changes the active scope; every workspace can inspect/edit the same roots even when the folder pane is hidden.
  Validation: focused offscreen coverage verifies browse-versus-scope behavior, nested-root collapse, root-list state, removal/clear, and scope-strip visibility.
- [x] Apply root-union discovery and multi-prefix database filtering to every workspace.
  Contract: Gallery and Clustering consume one deterministic union snapshot; Faces/Names/Tags/Library query only selected roots unless their visible global override is chosen; Library registration remains explicit.
  Validation: 21 temporary-root path/discovery/Tags/Faces/Library tests and five focused offscreen UI tests pass; the generated 80-photo root-union discovery benchmark records the current baseline and profile.

## Virtual nested Library timeline

- [x] Replace Timeline's 240-photo page with a cancellable full-filtered lightweight catalog read and Gallery-style nested Year → Month sections.
  Contract: Timeline lists every matching registered photo by catalogued capture month without source reads or UI-thread database work; newest month is initially open; header expansion/collapse is virtualized; Search remains paged and source-changing Gallery actions retain their current safety behavior.
  Validation: deterministic service/UI coverage includes 1,001 matches despite `limit=240`, chronological Year/Month grouping, cancellation, default/latest expansion, and Search paging preservation; the source-free 1,200-row `--library-timeline-only` baseline recorded 1.087 ms median query/grouping cost (2026-09-14).

## Library workspace usability and compact-layout pass

## Library-first navigation, richer previews, and explicit duplicate cleanup

- [x] Make Library the default workspace, and make Basic Clustering a photo-first workspace rather than a blank results surface.
  Contract: a fresh or legacy-Gallery session opens Library; the Gallery navigation item is removed; Basic Clustering shows the active-root photo grid immediately and `Organize photos` replaces that same grid with cluster sections. Existing cross-workspace photo routes remain usable and return to their initiating workspace.
  Validation: deterministic offscreen production coverage verifies legacy-Gallery → Library startup migration, explicit Gallery-route → Basic-Clustering translation, Basic/Advanced central-surface selection, route context/return controls, and the Tags-filter preflight exception. `bash scripts/test.sh tests/test_local_archive_curation.py tests/test_production_support.py` passed 88 tests.
- [x] Make saved-name hover previews adaptive and informative.
  Contract: the asynchronous preview uses a screen-size-bounded contact sheet, reports shown versus total photos/faces, and names only people co-occurring in the displayed sample photos; stale hover work is cancelled, cache keys include layout, and selection remains unchanged.
  Validation: focused offscreen Names coverage verifies contact-sheet publication, shown/total text, co-occurring-name counts, cache reuse, unchanged selection, and clean worker shutdown. The layout chooser bounds samples to 9/12/20 by available screen geometry; `bash scripts/test.sh tests/test_ui_smoke.py -k names_pane_hover_preview_is_cached_and_does_not_change_selection` passed.
- [x] Add a DateTimeOriginal-first Timeline date policy.
  Contract: manual dates remain first; the new policy resolves DateTimeOriginal, then other capture metadata, filename rules, and file modified time, retaining the selected provenance for display and tests without rewriting a source file.
  Validation: deterministic catalog coverage verifies `exif_original` provenance and the new priority path; offscreen production coverage retains persisted Timeline controls. The complete local-archive/production-support run passed 88 tests.
- [x] Split Cleanup review and recovery actions by exact versus similar duplicate kind, and progressively load all Library search photos.
  Contract: exact SHA-256 and similar perceptual-hash review can run separately or together; only checked non-keeper candidates in the previewed kind can move to recoverable Trash; burst review remains advanced. Search publishes every matching catalog photo in cancellable bounded pages rather than stopping at the first 240.
  Validation: deterministic duplicate-service tests prove exact scans do not request perceptual/vector work, similar scans use all three hash backends and records agreement, and feedback keys retain `near` compatibility. Offscreen UI tests verify unchecked preview behavior and 1,001 search rows publish at offsets `0, 500, 1000`. `bash scripts/test.sh tests/test_local_archive_curation.py tests/test_production_support.py` passed 88 tests; `bash scripts/build.sh` and `git diff --check` passed.

- [x] Hide the generated cluster-context panel unless a Library search returns context matches.
  Contract: normal filename/metadata searches use only a compact search row; the generated-context list and its open action appear only for actual saved-context matches, and that action remains disabled until a result is selected.
  Validation: deterministic offscreen coverage verifies compact/expanded Search tab heights, result-panel visibility, selection enablement, and return to the empty state. `bash scripts/test.sh tests/test_production_support.py` passed 57/57; `bash scripts/build.sh` and `git diff --check` passed.
- [x] Make the Library Timeline controls responsive and remove idle header copy.
  Contract: Library guidance is available from an accessible help icon; idle completion text does not consume header height; all Timeline filters, date-rule controls, and actions remain visible in a three-column wide grid, a two-column medium grid, or a vertically scrollable one-column compact grid without changing saved settings or source-safe Job behavior.
  Validation: deterministic offscreen coverage exercises wide/medium/narrow control placement, visibility, protected field widths, narrow scrolling, help/status behavior, preference preservation, and the existing Timeline refresh semantics. `bash scripts/test.sh tests/test_production_support.py tests/test_local_archive_curation.py` passed 81/81; `bash scripts/build.sh` and `git diff --check` passed.
- [x] Rework Library's first-run guidance, protected control widths, action rows, and tab heights for a readable production layout.
  Contract: every Library action remains visible with an intelligible label at the supported compact workspace width; the initial empty state explains the next action; Timeline, Search, Cleanup, People Cleanup, and Cluster Context retain their existing source-safe/background-job behavior.
  Validation: deterministic offscreen UI coverage checks protected sidebar/filter/action geometry and initial guidance at the compact 720 px workspace width; the focused production Library workspace and local-archive suite passed 11/11, and desktop/compact offscreen renders were visually reviewed.

## Library correctness and performance audit

- [x] Make Library request publication, cancellation, source revision, and large-catalog database access robust under repeated user actions.
  Contract: an obsolete Library request cannot publish over a newer view; cancellation remains a cancellation; photo/XMP revisions invalidate generated context; and catalog/hash reads remain bounded by SQLite parameter limits.
  Validation: the deterministic local-archive suite passed 9/9 (including source-revision, cancellation, and >999-path hash coverage); focused production Library/vector/startup checks passed, and the full deterministic suite was rerun with an isolated runtime.

## Local archive curation suite

- [x] Add registered library roots, a cancellable incremental local catalog, and virtualized Library timeline/search/smart-album views.
  Contract: only explicit enabled roots participate in global work; all catalog scans and metadata/index refreshes run in visible background Jobs and never modify source photos.
  Validation: six deterministic temporary-root catalog/context/feedback tests (including XMP-sidecar refresh and cache-clear preservation), offscreen Library shell coverage, and the isolated generated catalog benchmark.
- [x] Add a global People Cleanup Inbox and recoverable duplicate/burst review workflow.
  Contract: face suggestions are constrained to registered roots; confirm/name/reject/split/hide decisions persist safely; duplicate cleanup is always manual and moves only selected files through journaled ClusterLens Trash.
  Validation: deterministic split and source-revision feedback tests, existing journaled trash/recovery verifier, and Library background-job lifecycle coverage.
- [x] Add opt-in local-first vision-LLM cluster context and cluster-first textual search.
  Contract: manual descriptions are always available and automatic descriptions require an explicit setting; a deterministic representative image and scalar EXIF/XMP context are sent only to the chosen provider; remote runs require consent and no source metadata is written.
  Validation: fake local/remote-provider cache and consent tests, deterministic representative selection, FTS search, and Library Jobs integration coverage.

## Faces inline unlabelled-similarity arrangement

- [x] Keep similar unlabelled face regions together directly in the folder-review Faces tab without replacing the existing Face Groups workflow.
  Contract: **Arrange Unlabelled by Similarity** explicitly submits only durable, unlabelled indexed refs from the current loaded folder review to the configured face-clustering backend in a cancellable foreground Job. The result is a session-only virtual grid with inline headings: a folder-ordered Labelled faces section when All is shown, stable Similar groups, and Other / not grouped for outliers, drafts, and excluded faces. It never writes labels; resetting, filter/review/tab changes, successful naming, cancellation, or stale completion restores/retains raw folder order.
  Validation: focused deterministic offscreen coverage verifies named-ref exclusion, inline grouping/outlier presentation, filter invalidation, reset behavior, raw streamed review compatibility, and shared face-tile queue compatibility. Build/whitespace checks and the generated vector baseline are recorded with this pass.

## All Faces labelled/unlabelled view

- [x] Put an explicit labelled/unlabelled selector in the All Faces task and apply it before face-album pagination.
  Contract: All Faces visibly offers All, Labelled, and Unlabelled views; the selected view is backed by the global face-album group query, loads only the matching category's first group page, keeps pending proposals in Review because they do not yet have a durable name, and publishes no stale result after a rapid switch.
  Validation: `test_face_album_pages_are_bounded_scoped_and_embedding_free` and `test_all_faces_visible_label_filter_reloads_labelled_and_unlabelled_faces` passed in the focused deterministic service/offscreen suite; compilation, whitespace, and `bash scripts/build.sh` also passed.

## Inspector face naming recovery and layout

- [x] Treat an explicit user name for an already indexed face region as an override of the automatic prototype-quality gate.
  Contract: Photos, Faces, and Names can persist a user-selected face-region name and its XMP/EXIF metadata even when that region is excluded from automatic-quality workflows; automatic example selection, search, and clustering keep their configured quality filters.
  Validation: `test_explicit_face_region_name_overrides_automatic_quality_gate` and the strict `test_prototype_quality_gate_and_auto_label_threshold_apply` passed in the focused deterministic suite.
- [x] Reflow the Photo Inspector face-name controls into a labeled, keyboard-accessible section.
  Contract: the name input and Name, Rename, and Unlabel actions remain fully legible in the supported right-hand inspector width; face-edit jobs disable all mutating name controls until their background work completes.
  Validation: `test_photo_inspector_face_name_controls_are_spacious_and_busy_safe` passed offscreen at the supported 1200×800 layout.

## Names similarity results

- [x] Expose Find by Name + Similar inside Names as a bounded face-tile result view.
  Contract: selecting a durable name enables an explicit background search using that name's saved prototype; the result displays face crops, score, identity state, and the exact source-face details without replacing the normal durable named-photo view.
  Validation: deterministic offscreen coverage verifies selected-name gating, the `search_by_person_name` request, face-tile publication, stale-result suppression, and worker shutdown.

## Folder-tree visibility and GPU runtime readiness

- [x] Keep the Folder pane anchored at its browse root while revealing the selected directory inside the complete on-demand tree.
  Contract: choosing a nested folder must not hide its parents or sibling folders; changing the browse root remains an explicit action and no folder is recursively enumerated on the Qt thread.
  Validation: focused offscreen UI coverage preserves the browse-root index, selects the nested target, and expands its ancestor path.
- [x] Make the canonical source launcher prefer the installed dedicated CUDA runtime and make its CPU fallback explicit.
  Contract: launching does not install packages, move model files, or trigger model downloads; the CUDA and CPU launchers share the normal per-user runtime-data location.
  Validation: shell syntax checks passed; the physical RTX 4090 verifier selected CUDA for Torch, ONNX, cuML HDBSCAN, and every vector-compute smoke operation with no failures.
- [x] Make Support rescan progress and diagnostics explain each acceleration dependency and model-cache behavior.
  Contract: the background-only resource check visibly distinguishes Torch CUDA, CUDA ONNX for SCRFD/ArcFace, and cuML HDBSCAN; it never opens user photos or downloads models.
  Validation: focused dialog/runtime/UI tests pass for the checklist, staged background refresh, remediation, and complete folder tree.
- [x] Prevent late startup and footer-progress callbacks from touching deleted Qt widgets during close.
  Contract: shutdown cancels visible startup work, stops the footer presentation timer, and makes queued completions no-ops once the window is closing.
  Validation: the deterministic footer shutdown test passed; a forced 15-second offscreen GPU-launcher close completed without deleted-widget tracebacks.

## Gallery-first cross-workspace routing and lifecycle completion

- [x] Make the top-level Gallery the explicit, session-only viewer for face, name, tag, and cluster photo sets.
  Contract: routes retain their source and return workspace, never trigger face detection implicitly, and preserve exact per-photo face context while active; metadata and tag mutations invalidate affected views safely.
  Validation: production route coverage verifies Faces routes into the top Gallery with preserved context; the complete 517-test suite covers the shared Gallery, Names, Tags, cluster, and Faces UI contracts.
- [x] Bind folder discovery and Gallery-owned background work to the shared Jobs lifecycle and close it without UI-thread waits.
  Contract: folder discovery is visible/cancellable; obsolete worker results cannot publish; application shutdown cancels and drains all Gallery work without freezing the Qt event loop.
  Validation: Gallery discovery is bound to `JobManager`; close-pending cancellation/drain behavior and Gallery worker ownership pass in the complete deterministic suite.
- [x] Make CUDA HDBSCAN readiness explicit and preserve an actionable CPU fallback policy.
  Contract: Auto reports cached cuML unavailability and uses vectorized CPU HDBSCAN; explicit CUDA blocks unavailable HDBSCAN instead of silently changing policy.
  Validation: runtime/provider cache, explicit-policy, and Jobs fallback tests pass; the benchmark recorded `No module named 'cuml'` as an exact Auto-mode fallback.
- [x] Resolve the order-sensitive full-suite Qt teardown failure.
  Contract: test teardown leaves no active QThreads, queued late publications, mutable runtime overrides, or service singletons for later tests.
  Validation: `bash scripts/test.sh` completed normally with 517 passed and 4 provider/dependency warnings; no tests were skipped.

## Clean-native operational safety and release evidence

- [x] Expose generated-storage categories in the production Settings dialog without importing the legacy folder cache or catalog.
  Contract: Storage lists each generated category with its managed paths and size; each clear action is an explicit, cancellable background operation that preserves source photos, tags, durable face labels, recovery history, settings, and non-selected categories.
  Validation: six deterministic cache-service checks cover category boundaries and durable-data preservation; an offscreen production-dialog check covers the controls and worker callback contract; the isolated source smoke confirms clean teardown.
- [x] Add deterministic release-evidence commands built around isolated synthetic inputs.
  Contract: fixture generation, fixture-manifest verification, trash recovery, and packaged-launch verification refuse unsafe paths or report overwrites; they never read user photos or runtime data. Existing release gates remain the source of startup/UI readiness checks.
  Validation: four temporary-directory tests cover manifest tampering, empty-directory guards, trash collision/restore behavior, and report validation; build and whitespace validation are recorded below.
- [x] Design a Clean-native CLI after the user chooses its supported workflows.
  Contract: the agreed scope is read-only inspection plus explicit managed-data maintenance. It composes Clean services only and documents GUI-only indexing, model install, metadata, face-label, and photo-file mutation workflows.
  Validation: `tests/test_cli.py` exercises JSON Data Home inventory plus explicit backup/verify; `scripts/cli.sh` provides the documented deterministic entry point.

## Tags correctness and performance audit

- [x] Prevent concurrent or stale Tags inventory/photo page loads from duplicating rows, publishing an old selection, or consuming unbounded worker threads.
  Contract: rapid Refresh, scope changes, tag selection, and Load more requests coalesce to the newest view state; each page appears at most once and all work remains cancellable/visible.
  Validation: focused offscreen production checks simulate rapid page/selection requests, verify exact rows and final selection, and retain clean gallery-worker shutdown coverage.
- [x] Profile and reduce redundant SQLite total-count work on the Tags read path without weakening concurrent write safety.
  Contract: inventory/photo pages retain correct scope, escaping, source counts, and database recovery while later pages avoid recounting a total already established by the first page.
  Validation: the generated 75,000-row benchmark identified SQLite execute as the hot path and showed 33.333 ms later pages without the already-known count versus 36.741 ms with it; focused query tests preserve scope, escaping, and pagination behavior.

## Tags workspace migration

- [x] Add a top-level Tags workspace with paged global/current-folder tag inventory and tagged-photo gallery.
  Contract: browsing tags reads only the durable tag database, never scans media or blocks Qt; selection, paging, cancellation, and errors remain visible through Jobs and inline status.
  Validation: deterministic SQLite service tests cover query, folder scope, escaping, ordering, and pagination; focused offscreen production coverage confirms current-folder paging, selected-tag gallery publication, and clean embedded-gallery shutdown.
- [x] Move tag-filter clustering controls and tag management out of Advanced Clustering into Tags.
  Contract: Gallery quick-edit remains; Tags owns Any/All filters, global database-only rename/delete, and launches the existing tag-filtered clustering preflight.
  Validation: focused production preflight coverage confirms Tags values reach the existing off-thread discovery flow; deterministic tag-service tests cover durable inventory rename/merge/delete behavior.
- [x] Make selected-cluster tag suggestions an explicit on-demand Tags action.
  Contract: suggestion generation and application run as visible cancellable background work, require a selected cluster, and cannot publish stale results or mutate metadata in read-only mode.
  Validation: focused offscreen production tests cover selection gating, successful mocked on-demand generation, and database-only application. The generation guard, cancellation, and failure paths are implemented; add explicit stale-publication/cancellation regression cases before treating the broad suite as a complete release gate.
- [x] Measure and document tag-hub query performance.
  Contract: benchmark fixture is generated locally with deterministic content and does not read user media or caches.
  Validation: `bash scripts/benchmark.sh --tag-workspace-only`, `bash scripts/build.sh`, the focused production/service/UX suites, compilation, and `git diff --check` pass; results and unresolved hotspots are recorded.

## Source-parity responsiveness and recovery pass

- [x] Make Photos lazily prepare face tools before opening the face-region editor.
  Contract: **Edit Face Regions** is available from Photos even before Faces or Names has been opened; preparation is cancellable, visible in Jobs, and resumes or reports the requested edit without blocking Qt.
  Validation: focused offscreen tests cover Photos and main-gallery lazy requests, callback success/failure, and read-only safety mode.
- [x] Guard Faces initialization against obsolete UI state.
  Contract: changing away from Faces, switching to an unsupported-resolution screen, or closing the window cancels pending initialization; a late worker result cannot create or update deleted/replaced widgets.
  Validation: focused lifecycle tests simulate a workspace change while a face service is blocked and verify cancellation plus generation-guarded non-publication.
- [x] Recover thumbnail-cache indexes after interrupted or corrupt state.
  Contract: a missing/corrupt SQLite index is rebuilt from valid WebP thumbnails, stale rows and orphan files reconcile deterministically, and pruning remains safe under concurrent readers.
  Validation: temporary-cache tests cover missing/corrupt indexes, orphan reconciliation, interrupted cache writes, and concurrent readers.
- [x] Add a no-download startup readiness gate for runtime and local models.
  Contract: a visible background job checks the selected CPU/CUDA policy and locally installed clustering/face models; dependent actions stay disabled until the result is known, errors remain actionable, and **Rescan Available Resources** refreshes the gate without changing settings.
  Validation: deterministic service and offscreen production tests cover missing assets, unavailable explicit CUDA, successful local CPU readiness, action gating, and the rescan completion signal.

## Names workspace visibility, recovery, and non-blocking pass

- [x] Port the shared job presentation and recovery contract into the Names checkout.
  Contract: explicit work appears in the global footer/header and Jobs with its origin, progress, outcome, and cancellation; passive work remains visible in its owning view without stealing foreground progress.
  Validation: deterministic offscreen tests cover concurrent foreground/background work, Names refresh/mutation progress, stale results, cancellation, and recovery after a failed worker; the 209-test UI smoke suite and 41-test production-support suite pass independently.
- [ ] Keep every Names, Photos, Faces, Inspector, Settings, startup-maintenance, metadata, clustering, and storage action off the Qt UI thread.
  Contract: affected controls and inline feedback remain responsive while work executes; source-specific results cannot overwrite a newer selection or refresh.
  Validation: focused worker lifecycle tests, compilation, whitespace checks, and two full test-suite runs pass. A complete UI-thread audit and injected-slow-worker matrix remain outstanding; this item is intentionally still open.

## Per-face XMP metadata and naming

- [x] Add a face-region metadata service that reads and merges standard MWG/XMP face regions, writes embedded JPEG XMP or an XMP sidecar as appropriate, and retains a ClusterLens metadata mirror.
  Contract: one image can hold independently named face boxes; updating one region preserves all other regions and unrelated metadata.
  Validation: deterministic temporary-image tests cover two regions, embedded/sidecar round trips, legacy metadata fallback, and write failures.
- [x] Make durable face-label mutations metadata-aware.
  Contract: naming, renaming, and unlabeling update exactly the requested face regions and durable `(image_path, face_index)` rows; a metadata write failure does not commit that face's database change; existing database labels win over conflicting external metadata.
  Validation: isolated SQLite tests cover conflict precedence, re-index preservation, and mixed-person photos.
- [x] Wire per-region naming through Faces clusters, Names, and the Photos inspector.
  Contract: cluster naming updates every displayed cluster member's face region; Names lets users select a source metadata name; Photos can draw/scan/select a face region and name it.
  Validation: offscreen UI tests cover the source-name chooser, selected photo regions, and multiple cluster faces from the same photo.
- [x] Benchmark and document metadata operations.
  Contract: the benchmark uses generated local fixtures only and makes no hardware-independent timing claim.
  Validation: `bash scripts/benchmark.sh` measured the generated eight-region JPEG fixture; `bash scripts/build.sh`, focused service/UI tests, compilation, and whitespace checks passed.
- [x] Resolve the existing order-sensitive full-suite fixture failure before claiming a green full run.
  Contract: `bash scripts/test.sh` must complete without sharing mutable model-asset state between production-support tests.
  Validation: fixture teardown now drains queued Qt deletion work; asynchronous Faces completions are non-modal; model-availability tests explicitly provide ready fixtures. `bash scripts/test.sh` completed with 517 passed and 4 warnings.

## GPU HDBSCAN and clustering tuning

- [x] Add cuML HDBSCAN to the pinned Linux CUDA runtime without replacing the verified Torch/ONNX CUDA 12.1 libraries.
  Contract: the existing `hdbscan` backend uses cuML on a verified CUDA policy, uses native HDBSCAN on CPU, and reports any GPU-to-CPU fallback without caching it as a GPU result.
  Validation: the corrected dependency dry-run resolved 122 packages; the physical RTX 4090 same-process verifier selected CUDA for Torch, ONNX, and cuML HDBSCAN with no failures; the seeded benchmark was deterministic on CPU and GPU; the GPU source app entered its event loop after implementation.
- [x] Carry validated backend options through the production request, worker, clustering service, metrics, and result-cache key.
  Contract: HDBSCAN and K-means option changes cannot reuse incompatible cached memberships; older request payloads retain existing defaults.
  Validation: all 176 service tests passed, covering option forwarding, cache-key separation, cuML selection, native fallback, and no cache write under a failed GPU signature; production protocol/request tests passed.
- [x] Add HDBSCAN controls to file clustering and K-means controls to face clustering.
  Contract: file HDBSCAN exposes minimum cluster size, automatic/explicit minimum samples, merge epsilon, and single-cluster allowance; Faces cosine K-means exposes restarts, maximum iterations, and deterministic seed in every face-clustering entry point.
  Validation: six focused production/offscreen UI tests passed for visibility, request values, state persistence, and both face-clustering actions. The broad non-blocking run passed 473 tests with four pre-existing state-sensitive tests excluded and retained two documented unrelated UI failures.
- [x] Benchmark and document the completed implementation.
  Contract: results use a seeded synthetic fixture and distinguish CPU from GPU execution; no user photos or mutable application caches are read.
  Validation: the seeded 10,000 × 32 fixture measured 32.496 ms cuML versus 390.741 ms native CPU median (12.02× throughput); `BENCHMARKS.md`, `HOTSPOTS.md`, `ARCHITECTURE.md`, `BUILD.md`, and `README.md` record the method, results, runtime pins, cache contract, and remaining CPU boundaries.

## Application-wide compute acceleration

- [x] Route GPU-suitable vector workloads through the verified runtime policy.
  Contract: Torch/ONNX inference, cosine K-means, graph-neighbor search, and dense similarity scoring use CUDA when a compatible Torch CUDA device is verified; an operation-level CUDA failure falls back safely and is reported in metrics/logs.
  Validation: deterministic policy/mocked-failure service tests passed; the physical-GPU verifier selected CUDA for inference, PCA, K-means, silhouette, graph top-k, and dense scoring with no failures.
- [x] Make the CPU fallback explicitly vectorized and cache-conscious.
  Contract: CPU similarity scoring uses contiguous batched NumPy/BLAS operations; clustering retains native scikit-learn/NumPy execution; existing embedding, thumbnail, and result caches remain bounded and reusable.
  Validation: all 173 service tests passed; runtime inspection recorded OpenBLAS with 16 threads, libjpeg-turbo, AVX2/FMA3, and available AVX-512 dispatch paths. Result-cache schema 3 separates compute implementations.
- [x] Propagate one execution policy through production, benchmark, face, and legacy service stacks.
  Contract: a run does not silently create a CPU-only clustering service beside a CUDA embedding service; standard presets and Faces grouping choose CUDA-capable backends, while CPU-only algorithms and I/O/UI boundaries remain identified rather than being mislabeled GPU.
  Validation: focused construction, CUDA-default, resource-rescan, and production diagnostics tests passed; the updated GPU source app entered its event loop and loaded the saved Names/Faces data.
- [x] Measure the fixed synthetic vector workload before and after the change and update architecture/performance documentation.
  Contract: measurements use a seeded, generated matrix and distinguish transfer/initialization costs from warm computation; no user files or mutable caches are involved.
  Validation: the final five-repeat fixture measured 111.439 ms CUDA versus 424.828 ms CPU median with deterministic memberships; exact method/results are in `BENCHMARKS.md`, and remaining CPU-only work is in `HOTSPOTS.md`.

## GPU face indexing and runtime rescan

- [x] Report the device used by the active face detector/embedder accurately.
  Contract: ONNX face pipelines show GPU only when the verified `CUDAExecutionProvider` is active; Torch-backed face pipelines continue to follow the selected Torch device.
  Validation: the focused status-strip test passed for CUDA Torch with CPU ONNX and for verified CUDA ONNX; the existing CPU status checks also passed.
- [x] Add a user-triggered available-resource rescan to Support settings.
  Contract: rescanning re-probes CUDA/ONNX resources off the Qt UI thread, reports the resulting face-indexing device, and does not install packages or silently change the saved compute preference.
  Validation: focused service and production Settings tests verify `refresh=True`, provider revalidation, disabled actions while busy, result reporting, and clean thread shutdown; both targeted test runs passed all eight selected checks.
- [x] Validate the dedicated GPU face-indexing runtime and document the measured probe.
  Contract: the existing CPU fallback remains usable, while the CUDA source runtime selects `CUDAExecutionProvider` for SCRFD/ArcFace face indexing when available.
  Validation: 8 focused checks passed, followed by 8 expanded runtime checks; the broad run finished with 400 passed and the same 9 pre-existing failures recorded below. The GPU verifier passed on the RTX 4090, and direct installed-model checks selected `CUDAExecutionProvider` for both SCRFD 10G and ArcFace R100. Python compilation, `git diff --check`, and the resource-probe measurements recorded in `BENCHMARKS.md` passed.

## Cluster comparison typography

- [x] Replace the default one-line cluster-table renderer with compact two-line cluster rows and readable comparison headers.
  Contract: cluster IDs, selection, highlighting, tooltips, hover previews, and the underlying comparison keys remain unchanged; each visible row shows a clear cluster/outlier label plus photo and tag summary without relying on Qt's clipped default text.
  Validation: `QT_QPA_PLATFORM=offscreen uv run --with pytest python -m pytest tests/test_ui_smoke.py -k 'cluster_pane_uses_readable_two_line_cluster_cells or cluster_pane_renders_cluster_tag_summary or cluster_pane_exposes_selected_cluster_target or cluster_pane_hover_popup_opens_on_hover_and_hides_on_empty_target'` passed (4 tests); an offscreen rendered table was visually checked with the application theme.

## Sidebar typography and Names hover preview

- [x] Render sidebar name, identity, and unlabeled-group rows as a dense title plus muted metadata line rather than delimiter-heavy text.
  Contract: the existing selection payloads, filtering, tooltips, and virtual/paged models remain unchanged; dense rows preserve their counts and full detail on hover.
  Validation: `QT_QPA_PLATFORM=offscreen uv run --with pytest python -m pytest tests/test_ui_smoke.py -k 'names_pane_lists_durable_names_and_unique_photo_paths or names_pane_hover_preview_is_cached_and_does_not_change_selection or face_library_scan_and_refresh_ignore_hidden_candidate_scope or profile_controls_save_favorite_birth_date_hidden_and_identity_summary'` passed (4 tests); Python compilation and `git diff --check` passed.
- [x] Add a bounded, asynchronous contact-sheet preview when hovering a saved name.
  Contract: hover never blocks the UI; it shows only photos explicitly labeled with that name, cancels stale work, bounds memory with an LRU cache, and does not alter selection or gallery contents.
  Validation: the focused Names hover test verifies worker completion, exact labeled-photo content, cache reuse, unchanged selection, and clean shutdown. The full offscreen UX acceptance suite passed 6 checks; its one existing header-state failure is unrelated to these sidebar views.

## Compact Find and strict Names editing

- [x] Expose the Find walkthrough/saved-search panel, replace explanatory prose in core Find sections with existing hover/help affordances, and use the available grid layout for selected-face, query-face, and name-query controls.
  Contract: Find retains selected-face, query-photo, and saved-name search behavior; the visible Workflow card provides the optional walkthrough and persisted saved searches; direct controls fit horizontally when their containing pane is sufficiently wide.
  Validation: the focused route/layout suite passed 6/6 after adding the Workflow card to the responsive Find grid. `README.md`, `ARCHITECTURE.md`, and the current production-readiness plan document the supported route.
- [x] Restrict Names context editing to labels that are already named.
  Contract: Names continues to list only exact durable label photos, and its context menu contains Rename Selected and Unlabel Selected only; initial naming remains in Faces detection/group workflows.
  Validation: `QT_QPA_PLATFORM=offscreen uv run --with pytest python -m pytest tests/test_ui_smoke.py::UiSmokeTests::test_names_pane_selected_image_actions_are_scoped_to_the_active_name tests/test_services.py -k 'selected_image or named_photo or label'` passed (15 tests). It covers both operations, preserved two-photo selection, read-only guards, and durable SQLite mappings.

## Names context-menu actions

- [x] Move selected-photo Name, Rename, and Unlabel actions from the Names header into the photo gallery right-click menu.
  Contract: right-clicking an unselected photo selects it; right-clicking an already-selected photo preserves the complete multi-photo selection. The menu retains the ordinary gallery actions and exposes the same face-safe label mutations with read-only guards.
  Validation: focused offscreen Names tests verify the menu items, two-photo selection preservation, disabled read-only behavior, and the existing async durable-label operation. Production Names workspace coverage confirms the optional gallery action hook is present without opening Faces.

## Names selected-image label actions

- [x] Add Name, Rename, and Unlabel actions for selected photos in the Names workspace.
  Contract: selection is by photo for a compact gallery workflow, but mutations remain face-level: Name affects only unlabeled visible faces in the selected photos; Rename and Unlabel affect only faces currently carrying the active saved name. The actions update the durable global face-label mapping and refresh Faces and Names without blocking the UI.
  Validation: focused SQLite service test covers a mixed-person photo, identity cleanup, and reopening persistence; two offscreen Names UI tests cover selection, action enablement, and asynchronous mutation; the production-shell Names test confirms the actions are visible without opening Faces. Python compilation and `git diff --check` passed. The broader suite retains pre-existing environment/model and layout failures; focused potential-regression checks for bulk face naming and lazy startup passed, while the header acceptance test remains order-sensitive to a persisted Gallery workspace mode.

## Production Names workspace alignment

- [x] Expose the existing Names workspace from the production PyQt shell used by `scripts/run_app*.sh`.
  Contract: the visible production toolbar has Clustering, Faces, and Names; Names uses the production global face database and remains usable before the lazy Faces pane is opened.
  Validation: focused production test creates Names, waits for its durable-label refresh, and verifies that Faces remains unopened; startup-performance and all UX-acceptance tests passed. The GPU source launch follows this checklist update.

## Names workspace validation pass

- [x] Measure deterministic startup and Names query latency on an isolated temporary runtime/SQLite fixture.
  Contract: the assessment does not read, mutate, or depend on a user's photos, caches, or face-label database.
  Validation: five release startup gates passed; seeded 50,000-face SQLite fixture measured 36.122 ms median global-name aggregation and 1.119 ms selected-name photo retrieval. See `BENCHMARKS.md`.
- [x] Run correctness and stability validation for the committed Names workspace.
  Contract: durable-label recovery, exact-name filtering, async refresh/shutdown, and the full test suite are assessed without changing user-visible behavior.
  Validation: four focused Names tests passed in five consecutive runs (20 passes); full suite result was 449 passed / 9 known non-Names failures; CUDA source smoke entered the event loop. Python compilation is rerun after these documentation updates.
- [x] Record measured results and unresolved failures in the project documentation.
  Contract: no performance or stability claim is made without command output and retained measurements.
  Validation: `BENCHMARKS.md`, `HOTSPOTS.md`, and this checklist name the fixture, environment, measurements, and test result.

## Durable Names workspace

- [x] Make explicit face naming durable and recover legacy manual pending labels.
  Contract: a user-entered name immediately updates the global face-label mapping; recovery promotes only unconflicted legacy manual labels.
  Validation: isolated SQLite service tests cover persistence, conflicts, and recovery after reopening the database.
- [x] Add the top-level Names workspace and synchronize it with saved face labels.
  Contract: Names shows a paged global name sidebar and exact unique photos for the selected name; pending and similarity-only matches are excluded.
  Validation: offscreen workspace tests cover selection, refresh, and unique-photo results.
- [x] Compact the Face Groups action area without removing operations.
  Contract: primary naming actions remain visible; remaining standard and advanced actions remain reachable through documented menus.
  Validation: offscreen UI tests cover menus, action enablement, tooltips, and help affordances.
- [x] Preserve the existing Gallery behavior while adding Names as a peer workspace.
  Contract: Names does not reparent, duplicate, or change the Gallery/Clustering workflow.
  Validation: focused main-window workspace test verifies the Names switch independently of the existing Gallery state.

## Folder navigation and status

- [x] Make the selected folder the visible tree root and preserve its descendants.
  Contract: selecting a folder emits the canonical folder path and shows its subfolders.
  Validation: focused `SourcePane` test and application smoke launch.
- [x] Keep the selected folder visible in the footer while normal job status changes.
  Contract: the footer path is not overwritten by progress text.
  Validation: focused footer test.

## Face models and pipeline

- [x] Inventory every supported face component and distinguish Built-in, Downloaded, and Install for CPU/GPU.
  Contract: managed, canonical external, and direct external SCRFD/ArcFace ONNX files are recognized without copying.
  Validation: service tests with temporary direct ONNX filenames.
- [x] Split Settings model management into Clustering Models and Face Models, with a chooser/open action for the external face-model folder.
  Contract: the selected folder is saved only after Settings is accepted; external models are read, never moved or deleted.
  Validation: offscreen Settings-dialog test.
- [x] Move detector/embedder tuning into an Advanced Face Pipeline dialog and show the applied detector, embedder, and CPU/GPU runtime in one line.
  Contract: Apply rejects unavailable components, persists the complete applied profile, and updates the status only after Apply.
  Validation: offscreen face-pane dialog and persistence tests.

## Completion checks

- [x] Run focused UI/service tests.
  Evidence: all 163 service tests passed; four focused offscreen Names/Faces UI tests, the isolated multi-cluster naming test, and all seven UX-acceptance tests passed.
- [x] Run the complete deterministic test suite.
  Evidence: `tests/test_services.py` passed (163 tests). The complete UI-smoke and production-support runs exposed nine pre-existing environment/runtime failures, recorded here rather than hidden: UI layout assertions at compact widths, a test assuming no installed persistent GPU model, model-download approval and fallback expectations, home-runtime isolation, startup staged-bundle cleanup, and migration-version expectations. The Names-focused paths passed.
- [x] Launch the app using the persistent Linux runtime.
  Evidence: the application entered its event loop from the clean feature checkout using `/home/d3v/.local/share/ClusterLens`.
- [x] Verify applied face-pipeline settings persist.
  Evidence: isolated QSettings smoke check saved detector, embedder, and advanced JSON preferences.
- [x] Update the historical performance backlog after a benchmark is actually run.
  Evidence: the 2026-09-10 Names workspace assessment is recorded in `BENCHMARKS.md` and `HOTSPOTS.md`.

## Persistent managed face models

- [x] Launch normal source runs without a temporary runtime override and retain managed face models in the Linux application-data cache.
  Contract: `/tmp` validation overrides cannot make normal launches lose installed face models; explicit persistent overrides continue to work.
  Validation: `bash -n scripts/run_app.sh`; two focused runtime tests and the managed-installer persistence test passed; 155 service tests passed.
- [x] Preserve and recover a corrupt image-tag SQLite file so normal persistent startup can complete.
  Contract: an invalid tag database is renamed to a timestamped backup and replaced with an empty valid SQLite database.
  Validation: focused service test passed; the normal application launch preserved the invalid file and entered the event loop.

## Downloaded model persistence and recovery

- [x] Make managed face-model installs atomic and retain enough verified local provenance to restore a missing bundle without a network download.
  Contract: a cancelled or interrupted install never appears ready; a missing managed bundle is restored at startup from a verified retained download when available, otherwise is reported as requiring reinstallation.
  Validation: temporary-cache service tests cover atomic promotion, interrupted-install recovery, cache-only restoration, manual deletion, and unavailable recovery sources.
- [x] Reconcile clustering model caches at startup and repair a broken Hugging Face current-revision reference from a complete local snapshot.
  Contract: application, worker, and model-download processes resolve the same persistent Torch/Hugging Face roots; a complete downloaded clustering model is not made invisible by a stale revision pointer.
  Validation: temporary-cache service test repairs a stale `clip` reference to a complete local snapshot.
- [x] Run one startup reconciliation before face controls and model inventory are created.
  Contract: model selectors and Settings inventory see recovered managed assets on their first render; recovery does not trigger network activity.
  Validation: focused production startup test promoted a complete staged YuNet bundle before Faces UI initialization; the full service suite passed 161 tests and focused production startup checks passed.

## Dedicated CUDA inference runtime

- [x] Add a reproducible Debian/Python 3.12 CUDA 12.1 runtime for face and ONNX inference.
  Contract: the dedicated runtime uses the verified CUDA ONNX provider without replacing the CPU-default source environment or moving model files.
  Validation: setup and verifier reported RTX 4090 CUDA Torch plus `CUDAExecutionProvider`; both SCRFD 10G and ArcFace R100 sessions selected CUDA; the GPU launcher entered its event loop using `/home/d3v/.local/share/ClusterLens`; 155 service tests passed.

## Out of scope for this pass

- The user-facing CLI remains unchanged, as requested.

## Faces workflow refinement

- [x] Strengthen Faces visual hierarchy for tabs, sections, and actions.
  Contract: navigation tabs have a clearly visible selected state, fieldset sections separate related controls, and primary actions are visually distinct from configuration/secondary actions while retaining the dense desktop layout.
  Validation: focused offscreen UI test verifies the Faces tab identifiers, style selectors, and semantic button roles; compile and application launch remain clean.

- [x] Compact All Faces, add direct grouped-photo naming, and make CUDA readiness unambiguous.
  Contract: All Faces keeps only its necessary actions with hover/help affordances; selected face tiles in Grouped Photos can be named directly through the same multi-selection workflow as Detected Faces; and a verified CUDA runtime remains shown as ready even when separate application-health notes need attention.
  Validation: six focused offscreen UI tests passed for compact controls, help, grouped-tile naming/menu selection, and the CUDA-ready badge; `scripts/verify_gpu_runtime.py` reported an RTX 4090 with CUDA Torch and `CUDAExecutionProvider` smoke tests passing.

- [x] Redesign the primary Faces workflow without removing review/identity tools.
  Contract: Faces opens the saved global album and the selected folder's cached review without rescanning; the task selector exposes All Faces, Detect, Find, and Manage; shell routes expose Review & Name and identity administration; named photos and normal grouped photos have explicit views; cluster labels list all saved names present; and Find supports both a query photo and saved-name similarity search.
  Validation: current focused offscreen route/layout coverage passed after Review & Name, Find Workflow, and Manage became supported routes; the existing cached-workspace, naming, query/name Find, and state-restoration tests remain part of the full suite.

- [x] Make Review & Name context face-driven rather than photo- or form-driven.
  Contract: a single-source selected face focuses its source photo; empty, unlabeled, and mixed-name selections clear stale identity state; cross-photo selections report their scope without moving the gallery.
  Validation: focused offscreen UI smoke tests cover stale-name clearing, single-source focus/mirroring, multi-photo stability, and existing naming/context-menu paths.

- [x] Make selected-photo faces visible without a disclosure and split Review & Name identities from unlabeled groups.
  Contract: a folder-gallery photo always exposes its detected face tiles, with compact hover/help affordances; named identities and unlabeled photo groups remain independently selectable in taller lists.
  Validation: focused offscreen UI smoke tests load selected-photo tiles, verify the two list populations and help icon, and check workspace-state restoration no longer hides face tiles.

- [x] Remove redundant explanatory prose from Review & Name.
  Contract: the selected photo, face-tile disclosure, identity fields, and actions remain; status content stays available to the app and hover help without duplicating it in the pane.
  Validation: focused offscreen Faces UI tests retain the selected-photo and naming controls.

- [x] Make multi-face naming discoverable from Detected Faces.
  Contract: Shift-click selects a contiguous range, Ctrl-click adds/removes individual tiles, and right-clicking any selected tile preserves the selection. Single-face and multi-face menus expose only actions that safely apply to their respective selection sizes; opening either menu does not block the UI event loop.
  Validation: focused offscreen context-menu tests simulate Shift/Ctrl clicks, name two selected faces, and verify the single- and multi-face menu action sets. Full UI smoke: 188 passed; see the completion-check blockers above.

- [x] Add face and similar-face ignore actions to the selected-face context menus.
  Contract: detected and grouped-result face tiles offer reversible Ignore Face(s), Find Similar, and Ignore Similar Faces actions; multi-selection searches from the aggregate selected-face embedding and ignores only the returned similar indexed faces, never the selected source faces.
  Validation: isolated offscreen UI tests `test_detected_faces_context_menu_searches_and_ignores_similar_faces_from_multi_selection` and `test_detected_faces_context_menu_ignores_only_similar_faces_from_single_selection` passed (2 passed); they verify the multi-source request and that only matched refs—not source refs—are hidden.

- [x] Make the face-name completion popup commit its choice on click, Enter, and Tab.
  Contract: an activated saved name replaces the input text with its canonical spelling and closes the popup; Enter and Tab commit the currently highlighted completion without accepting the naming dialog or moving focus away from the input.
  Validation: isolated offscreen UI test `test_face_name_completion_commits_on_activation_enter_and_tab_without_closing_dialog` passed; it activates a completion and sends Enter/Tab while the popup is visible, asserting the input receives the selected name, the popup closes, and the dialog remains open.

- [x] Remove obsolete detached Faces controls that become standalone Qt windows in Advanced mode.
  Contract: the shared-folder/global-library workflow remains intact and no unparented legacy widget can be shown as a popup.
  Validation: focused advanced-mode UI test and application launch.

- [x] Remove the Active Pipeline details expander and keep the applied model status as the single entry point to pipeline settings.
  Contract: status and Advanced Pipeline open the same modal; no secondary diagnostic text appears in the sidebar.
  Validation: offscreen face-pane test.
- [x] Use the global selected folder as the scan target, move review filters into the pipeline modal, and arrange the remaining scan actions in a grid.
  Contract: users cannot choose a conflicting folder in Faces; filters remain functional from the modal.
  Validation: offscreen face-pane test.
- [x] Move saved identities/unlabeled groups to Review & Name and attach visible help icons plus hover help to option fields.
  Contract: Scan stays focused on detection; each labeled option exposes the same help text by icon and tooltip.
  Validation: focused UI test.
- [x] Remove the stale model-inventory update for the deleted Settings license panel.
  Contract: model inventory updates the Clustering Models table without accessing removed widgets.
  Validation: clean application startup and production Settings acceptance test.
## Faces-tab labels and Deep Name + Similar

- [x] Put the labelled/unlabelled control in the folder-review Faces tab and keep publication responsive.
  Contract: All, Labelled, and Unlabelled filter only the visible face tiles in the active review; changing it restarts the existing bounded publisher without rescanning or embedding photos.
  Validation: deterministic offscreen `test_faces_tab_label_filter_changes_only_detected_face_tiles` passed.
- [x] Add full-scope, reviewed deep saved-name expansion in Names and Faces.
  Contract: all eligible indexed faces in the initiating scope are compared in GPU/CPU vector batches; newly found durable-unlabelled faces expand the temporary corpus until stable, existing names never expand or overwrite, and no mutation occurs before explicit confirmation.
  Validation: deterministic service chain test and seeded benchmark fixture passed; selected metadata writes report saved, skipped, failed, and cancellation-after-completed-files outcomes.

## Measured performance pass

- [x] Establish a deterministic baseline and profile the current hot path.
  Contract: use generated, seeded inputs only; identify a CPU/GPU, memory, I/O, or contention bottleneck without reading user photos or caches.
  Validation: `bash scripts/benchmark.sh --thumbnail-index-only` used 1,024 generated WebP entries and four independently created services in a temporary directory. Before the change, four opens took a 200.411 ms median; the profile spent 0.474 s reconciling the same cache four times. Evidence and environment constraints are recorded in `BENCHMARKS.md` and `HOTSPOTS.md`.
- [x] Apply the smallest safe optimization to the measured bottleneck.
  Contract: preserve deterministic memberships/results and existing CPU/GPU fallback semantics; keep all work off the Qt UI thread.
  Validation: one process-wide, bounded ready state shares completed recovery only while the WebP/index fingerprint matches. The same fixture measured 43.321 ms median for four opens (4.63× faster); corruption, missing-index, orphan, concurrent-reader, and shared-reconciliation tests, build, syntax, and whitespace checks pass. Results are documented in `BENCHMARKS.md` and `HOTSPOTS.md`.
