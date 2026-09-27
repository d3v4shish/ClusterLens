# Build and validation

From a clean checkout with Python 3.11 or 3.12 and `uv` installed:

```bash
bash scripts/build.sh
bash scripts/run.sh
bash scripts/test.sh
bash scripts/benchmark.sh
bash scripts/cli.sh status
```

The canonical scripts use the locked dependency graph, including the exact pytest version in the development dependency group; test execution performs no ad-hoc `--with` resolution. Tests and the synthetic benchmark use isolated temporary runtime/settings/bytecode directories; the benchmark uses a fixed seed and reads no user photos or mutable application cache. `scripts/test.sh` also prepends a test-only network guard to `PYTHONPATH`, so spawned Python workers inherit the same external-DNS/non-loopback ban while deterministic localhost servers remain usable. Hugging Face/Transformers offline flags are forced for the complete test process. To benchmark the CUDA runtime instead of the default source interpreter:

```bash
CLUSTERLENS_BENCHMARK_PYTHON="$PWD/.venv-gpu-cu121/bin/python" bash scripts/benchmark.sh
```

Each canonical script resolves the repository from its own location, so it may be invoked by absolute path from another working directory. For example, from `/tmp`, run `/path/to/ClusterLens/scripts/test.sh`; relative test arguments are still interpreted from the repository root. This behavior is covered by the outside-workdir evidence in `BENCHMARKS.md`.

To verify that the current candidate does not rely on `.git`, an existing
virtual environment, ignored caches, the repository working directory or
network dependency resolution, run:

```bash
.venv/bin/python -B scripts/verify_clean_source.py \
  --report-dir /new/path/clean-source
```

It exports tracked and non-ignored untracked files into two temporary
source-only trees, creates a fresh `.venv` in each with `UV_OFFLINE=1`, and
runs build, `run.sh --check-launch` and the complete canonical suite from two
outside directories. The host must already have the locked packages in its uv
cache. Until the candidate is committed, this validates the exact dirty
worktree export recorded by the report digest rather than a Git clone.

To validate launcher selection and repository resolution without opening Qt or touching a runtime profile, use `bash scripts/run.sh --check-launch`. The same check works through `scripts/run_app.sh` and by absolute path from another working directory; it exits 0 only after the selected Python can resolve the production module.

Build release packages explicitly:

```bash
uv run python scripts/build_pyqt_binary.py --variant cpu --recreate-venv
uv run python scripts/build_pyqt_binary.py --variant gpu-cu121 --recreate-venv
```

`run_app.sh` launches with the normal per-user application-data directory. On Linux, managed face models are retained in `~/.local/share/ClusterLens/cache/face_model_assets`; reusable face-model download archives are retained in `~/.local/share/ClusterLens/cache/face_model_downloads`. `run.sh` is the canonical developer launcher: it selects the already-installed CUDA source runtime when present and otherwise reports its CPU-compatible fallback. It never runs GPU setup or downloads model files.

For the dedicated CUDA 12.1 source runtime on Linux x86-64 with Python 3.12:

```bash
bash scripts/setup_gpu_runtime.sh
.venv-gpu-cu121/bin/python scripts/verify_gpu_runtime.py
bash scripts/run.sh
```

The GPU setup is deliberately separate from `uv sync --frozen`, so CPU source environments and non-Linux platforms retain their existing dependency set. It pins RAPIDS cuML 25.10 and scikit-learn 1.7.2 while preserving Torch 2.2.2's CUDA 12.1 libraries. `uv` uses best-match resolution only across the explicitly declared PyPI, PyTorch, and NVIDIA indexes because RAPIDS' `cuda-python` wrapper is not present on the first index. The verifier must report `CUDAExecutionProvider` and successful Torch, ONNX, semantic-PCA, cosine-K-means, cuML-HDBSCAN, silhouette, graph-neighbor, and dense-similarity smoke tests before GPU indexing or clustering is used.

The CPU and CUDA packages are separate artifacts. The CUDA variant requires a compatible NVIDIA driver and CUDA-enabled Torch/ONNX Runtime packages. In Settings → Support, **Rescan GPU Resources** is a visible background-only checklist for Torch CUDA, CUDA ONNX, and cuML HDBSCAN. It does not install packages or download/move model files; use it to inspect the active process after hardware changes.

Warm-worker release qualification uses an explicit checksum-verified model
asset and never downloads one:

```bash
.venv/bin/python -B scripts/qualify_warm_worker.py --model-asset-dir /approved/fast_preview --execution-mode cpu --report /tmp/warm-worker-cpu.json
.venv-gpu-cu121/bin/python -B scripts/qualify_warm_worker.py --model-asset-dir /approved/fast_preview --execution-mode cuda --report /tmp/warm-worker-cuda.json
```

CUDA mode requires Torch CUDA, `CUDAExecutionProvider`, and `nvidia-smi` process
visibility. Missing capability is `NOT_RUN`; a PASS records the exact worker
PID's warm VRAM allocation and proves that allocation disappears with the PID.

`scripts/benchmark.sh` runs generated, isolated fixtures for vector work, JPEG/XMP merge/readback, thumbnail-index recovery, SQLite tag queries, deep face search, Faces-tab arrangement, face-index scheduling, progressive folder-review paging, the full offscreen People/Detect publication workflow, first-page People/All-Faces publication, the registered-root catalog, and large name-picker filtering. The catalog fixture also includes a Timeline query/grouping and active-root union discovery baseline. Its catalog names deterministically exercise named date-plus-sequence rules, declared seconds/milliseconds epochs, and the opt-in raw-ID epoch fallback. They are comparable only within their own fixtures. Run just the active-root union fixture with `bash scripts/benchmark.sh --multi-root-discovery-only --photos 40 --repeats 5`; it creates two effective generated roots plus one nested duplicate root and reads no user media. Run the autocomplete benchmark with `bash scripts/benchmark.sh --entity-picker-only --names 10000 --repeats 7`; it uses synthetic names in an offscreen Qt process. `--face-indexing-only` writes deterministic temporary JPEGs, uses a fixture detector/embedder (no model load), and reports bounded decode/quality/batch/write scheduling. `--face-review-paging-only` uses temporary seeded SQLite rows only, verifies source-order 500-image pages, and measures the query/review-object work used by Detect's background metadata cache without source-media decoding. It reports an equal-fixture no-snapshot control, snapshot preparation time, and separately traced retained/peak Python allocation; use `--photos 11284 --faces 14477 --repeats 5` for the production-size capacity fixture. `--people-detect-workflow-only` adds real offscreen Photos/Faces Qt publication at that scale with temporary 1px source files, SQLite rows, cached-tail and cancellation assertions, plus separate memory and Faces-publication profile runs; it excludes inference/full-resolution crop and thumbnail decode. The accompanying offscreen UI test verifies that caching changes neither the visible Photos viewport nor the hidden Faces model before a tail request. `--people-faces-only --faces 5000 --repeats 5` measures one snapshot-consistent All Faces first group plus its first bounded 500-member page, and verifies pending proposals stay out of normal album membership. The other focused modes remain `--thumbnail-index-only`, `--tag-workspace-only`, `--deep-face-search-only`, `--faces-arrangement-only`, `--library-catalog-only`, and `--library-timeline-only`. End-to-end photo-pipeline claims still require the fixed photo fixture described in `BENCHMARKS.md`.

`bash scripts/benchmark.sh --sectioned-gallery-only` generates 100,000 in-memory paths and measures worker hierarchy preparation, the Qt model commit, selection preservation, event-loop gaps, and traced retention across alternating full/filtered five/six-column model replacements. It opens no source media, database, thumbnail cache, model, or network resource.

`bash scripts/benchmark.sh --production-workloads-only --report /tmp/clusterlens-production-workloads.json` generates a fixed seeded 4K JPEG corpus, launches five fresh untraced processes plus separate traced/profiled processes, and records cold/warm thumbnail and full-resolution crop work, bounded thumbnail/crop/catalog contention, Qt callback/input/pump latency, cancellation acknowledgement/drain, CPU/RSS/Python allocation, physical I/O and queue depth. The report path must be new. This CPU-only fixture performs no model inference and does not qualify a GPU or native compositor; its exact method and reference-host budgets are in `BENCHMARKS.md`.

Real-model qualification is deliberately opt-in and asset-local. Start from
`docs/real_model_manifest.example.json`; replace every placeholder, pin the
approved fixture source/license and every model revision/file SHA-256, and
declare complete detector, embedder, and production clustering-backend
coverage. `docs/MODEL_PROVENANCE.md` records the current upstream identities,
managed Hugging Face revisions and unresolved redistribution policies; it must
be reconciled with the approved release manifest. The harness performs no
download:

```bash
.venv/bin/python -B scripts/qualify_real_models.py \
  --manifest /absolute/path/manifest.json \
  --fixture-root /absolute/path/approved-fixture \
  --model-root /absolute/path/pinned-models \
  --execution-mode cpu \
  --report /new/path/real-model-cpu.json
```

Run the same manifest through the qualified CUDA interpreter with
`--execution-mode cuda` and another new report path. Exit 0 means every case
executed and passed; exit 1 is a validation/inference failure; exit 3 is
`NOT_RUN` because an asset, service, or explicitly requested provider is
unavailable. The report includes exact input hashes, provider evidence,
cold/warm/recreated-service timings, output digest, deterministic clustering
memberships, numeric tolerances, and process RSS. Generated placeholder media
may test refusal paths but cannot produce real-model PASS evidence.

The warm-worker resource gate uses the build-produced Fast Preview ONNX asset
and four generated images. It performs real CPU inference twice in separately
started persistent workers, records idle/peak RSS, requires identical
membership, disables warm mode, and verifies that each PID disappears before
the next cycle:

```bash
.venv/bin/python -B scripts/qualify_warm_worker.py \
  --model-asset-dir build/model_assets/fast_preview \
  --cycles 2 \
  --report /new/path/warm-worker-cpu.json
```

The report path must be new. Missing or checksum-invalid build assets return
exit 3 / `NOT_RUN`; they are not replaced or downloaded.

The deterministic cross-feature scheduler gate drives the production
`WorkCoordinator` through both launch orders for every documented conflict
pair, cancellation before and after dispatch, injected failure, retry, and
final resource release:

```bash
.venv/bin/python -B scripts/verify_work_conflicts.py \
  --seed 20260925 \
  --report /new/path/work-conflicts.json
```

It uses no timing synchronization, media, model, network, or developer state.
It is scheduler evidence only; UX-31 still requires the production UI and
representative full-resolution mixed-workload qualification.

The bounded offscreen production-shell slice uses four generated 3840×2160
images, three coordinated thumbnail/crop/catalog workers, every top-level
workspace, both supported logical sizes, both themes, scroll surfaces, the
Jobs shortcut and the visible cancellation control. Set an unused report path:

```bash
CLUSTERLENS_UX31_REPORT=/new/path/mixed-ui.json \
  bash scripts/test.sh tests/test_mixed_ui_workload.py -q
```

The JSON distinguishes callback samples from native input latency and records
its CPU/generated/offscreen limits. Real model/GPU, repeated fresh-process,
native compositor and approved-photo qualification remain separate gates.

For Linux post-close process/thread evidence, run the same journey in a new
operating-system process session:

```bash
.venv/bin/python -B scripts/verify_mixed_ui_lifecycle.py \
  --report-dir /new/path/mixed-ui-lifecycle
```

The verifier invokes the frozen canonical test command, retains its log and
inner workload report, and fails if the test exits unsuccessfully or any PID
survives in the owned session. It returns `NOT_RUN` outside a `/proc` host;
native-platform cleanup remains part of UX-32/UX-34 qualification.

`bash scripts/benchmark.sh --ux-workflow-only` measures deterministic 500 and
10,000-row virtual-list first-content/full-publication latency with no I/O,
cache, model, user media, or GPU backend. `scripts/cli.sh` is intentionally
limited to read-only inspection plus explicit Data Home backup/verification;
source-changing workflows are GUI-only.

The appearance microbenchmark is available independently as `bash scripts/benchmark.sh --theme-only --iterations 250`. It alternates light/dark palette application, measures reapplying the already-resolved theme as a no-op, and classifies fixed generated dark, mid-tone, light, and transparent images through the production 32×32 border sampler. It uses an offscreen Qt application and reads no source media, settings, model, cache, network, or GPU state.

The catalog/duplicate-review performance checks use source-free fixed corpora:

```bash
bash scripts/benchmark.sh --library-catalog-only --fts-only --fts-assets 1000 --repeats 2
bash scripts/benchmark.sh --duplicate-review-only --paths 128 --repeats 2
```

The first checks batched catalog/FTS replacement and a warm lookup; the second deliberately uses identical 64-bit hashes to expose review-pair growth. Neither reads user photos. For native release-display evidence, run `uv run python scripts/verify_native_display.py --logical-size 1280x720 --text-scale 200 --report-dir /tmp/native_display`; it refuses headless Qt and records keyboard focus, accessible labels, token contrast, screenshots, rendered-copy clipping, DPI, and exact client/screen bounds.

The production shell opens **Library** by default and exposes Library, Organize, People, and Tools. **Edit roots** opens the staged Sources/Catalog drawer: checked tree rows are a draft until **Apply**, while Catalog actions remain separate and disable during a dirty draft or read-only safety mode. Organize Basic presents the active-root photo grid and groups it in place; Advanced reveals tuning/comparison. Legacy Gallery/Clustering/Faces/Names/Tags routes migrate to the nearest new surface. Timeline measures its actual control hints and viewport: it uses one row when all filters fit, a semantic two-row form when constrained, and a compact stack only when necessary. Library Search starts with a 500-row catalog page and automatically requests later pages; use **Stop loading** or Jobs cancellation to stop safely. Tools' duplicate scan actions automatically wrap to content width and its previewed Trash actions remain GUI-only and move only manually checked candidates through the journalled recovery path.

To measure global source-admission filtering with generated inputs only:

```bash
bash scripts/benchmark.sh --library-catalog-only --source-filters --photos 120 --timeline-photos 1200 --repeats 5
```

The fixture deterministically mixes thumbnail-like names, files below 512 bytes, and 32×24 images with 64×48 admitted JPEGs. It does not read user media; its result is not comparable to the all-admitted catalog baseline because it intentionally indexes fewer photos.

For the viewport-first gallery and shared inspector slice, run the deterministic
focused validation from an isolated runtime:

```bash
bash scripts/test.sh tests/test_ui_smoke.py -k 'gallery_load_visible_images or thumbnail_queue_discards or photo_inspector_basic_mode or photo_inspector_advanced_mode or photo_inspector_face_editor or sectioned_gallery' -q
bash scripts/test.sh tests/test_services.py -k 'photo_edit_draft or exif_draft' -q
```

## Release evidence

These commands never read user photos or existing runtime data. Every target is a new/empty evidence directory; the fixture generator allows overwrite only for a deliberately named fixture directory.

```bash
uv run python scripts/create_release_fixture.py --output-dir /tmp/release_fixture --photos 128 --seed 20260913
uv run python -m apps.pyqt_production.release_gates --folder /tmp/release_fixture/photos --report-dir /tmp/release_gates
uv run python scripts/verify_trash_recovery.py --fixture-dir /tmp/trash_fixture --report-dir /tmp/trash_evidence
uv run python scripts/verify_packaged_launch.py --executable /path/to/ClusterLens --runtime-root /tmp/packaged_runtime --report-path /tmp/packaged_launch.json
uv run python scripts/verify_native_display.py --logical-size 1280x720 --text-scale 200 --report-dir /tmp/native_display
.venv/bin/python scripts/verify_ui_redesign.py --text-scale 200 --report-dir /tmp/ui_redesign
.venv/bin/python scripts/verify_runtime_badge.py --report-dir /tmp/runtime_badge
.venv/bin/python scripts/verify_test_isolation.py --report-dir /tmp/test_isolation --outside-workdir /tmp
.venv/bin/python -B scripts/verify_process_network.py --report-dir /tmp/process_network -- bash scripts/test.sh
.venv/bin/python -B scripts/verify_durable_fault_matrix.py --report-dir /tmp/durable_fault_matrix
bash scripts/benchmark.sh --jobs-history-only
```

The native-display command requires a real display or clean VM and rejects offscreen/minimal Qt platforms. It records actual client/screen rectangles and DPI, requires the requested client size, full on-screen containment, readable control height, exact 18-route and 10-Settings-section coverage, complete logical keyboard reachability, a populated generated active root, zero rendered-copy clipping, and clean shutdown. Add `--require-fractional-scale` only when the target display is configured with a non-integer Qt device-pixel ratio. Add `--require-multiple-screens` only when at least two Qt screens can each contain the requested logical client size; the gate moves the real window to every fitting screen and requires the compositor to report the requested target plus a contained app-only capture. A physical output narrower than 1280 logical pixels cannot qualify the documented 1280×720 minimum by using a smaller test window. The report separates visual and keyboard results from the real assistive-technology run; screen-reader status remains `NOT_RUN` until that separate release gate executes.

`verify_ui_redesign.py` is the deterministic offscreen shell check. It accepts `--text-scale 100|125|150|200` and captures app-only Library/Organize/People/Tools, all 18 contextual routes, staged Roots Sources/Catalog, all 10 Settings sections, a generated six-state Jobs history, and all 4 generated-image Photo Inspector panels at 1280×720 and 1920×1080. Organize Advanced is painted in dark and light themes. The generated active photo root prevents the first-run panel from masking routes. Version 3 reports verify exact capture sets, real Tab-key traversal for every visible focusable control, accessible name/text fallbacks, disabled and enabled clipping, primary bounds, palette contrast, background-idle state, fixture population, shutdown, and the measured Timeline layout contract. Empty/incomplete traversal, missing or extra routes, unnamed controls, poor contrast, offscreen primary actions, clipping, busy capture, or failed close makes the run fail. Run four fresh report directories to qualify all supported text scales. It uses only generated temporary data and refuses an existing output directory. `verify_runtime_badge.py` is a separate no-I/O offscreen fixture that captures checking, ready, fallback, unavailable, and failed badge states in both themes and verifies matching accessibility names. `verify_test_isolation.py` runs the five audited UI/concurrency modules from the repository and an outside directory in reverse order. It records revision, source-tree/diff/status digests, fixture and model-manifest hashes, isolated environment paths, exact exits/log hashes, and descendant process cleanup; it fails if tests mutate source or leave their process session alive. The packaged-launch verifier requires a PyInstaller-built executable and intentionally fails for a source interpreter because it is not a frozen artifact. Its version 2 report hashes the launcher and the complete onedir tree, including relative paths, entry kinds, user-executable mode, sizes, contents and symlink targets. The current package requirements are range-based, so identical tree digests across clean locked build hosts remain a release qualification task rather than a documented guarantee.

`verify_process_network.py` is the Linux native-descendant network audit. It
inherits the Python socket guard and offline model flags, follows the complete
command process tree with `strace -f -e trace=%network`, permits Unix/netlink
sockets and IPv4/IPv6 loopback, and fails on any non-loopback or unparseable IP
destination. It also requires the command to pass before its watchdog and its
owned process session to drain. Missing `strace` is exit 3/NOT_RUN, never PASS.
Use a fresh report directory; the full canonical suite is the release command,
while focused parser tests alone do not close the native-child audit.

`verify_durable_fault_matrix.py` reads the authoritative
`docs/durable_operation_fault_matrix.json`, runs each unique referenced pytest
node with the locked interpreter and offline/disposable runtime settings, and
requires its owned process session to drain. On POSIX, declared actual-exit
cases must end with `SIGKILL`, provide unique pre/crash/recovered state digests,
and complete at least two recovery attempts; a raised Python exception cannot
satisfy that evidence. The JSON report separates
regression `validation` from release `qualification`: validation PASS proves
the referenced tests passed, but any listed fault gap or NOT_RUN platform keeps
qualification PARTIAL. The verifier refuses report paths inside the repository
and refuses to overwrite an existing evidence directory.
