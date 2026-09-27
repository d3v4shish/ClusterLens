# ClusterLens

ClusterLens is a local-first desktop photo manager. Its four workspaces are **Library** (Timeline, All Photos, Search, Albums), **Organize** (photo grouping, tags, context), **People** (named people, unnamed faces, Detect, review, Find, and identity management), and **Tools** (duplicates, batch rename/metadata, recovery). The shared Roots manager controls source scope; source-changing work is previewed, journalled where applicable, visible in Jobs, and safely cancellable.

## Local archive curation

The **Library** workspace is a local-first curation layer over explicit, user-registered roots. It incrementally catalogs filenames, scalar EXIF, and XMP sidecars without editing source photos, then provides a virtualized timeline, all-photo view, text search, and dynamic smart albums. Duplicate review lives in Tools and durable face work lives in People. Disabled roots are visibly excluded from global work.

Timeline has an explicit **Timeline date** policy: metadata only, metadata then filename, **DateTimeOriginal then all metadata**, filename first, or filename with a camera-DateTimeOriginal fallback. The DateTimeOriginal-first policy orders manual corrections, camera DateTimeOriginal, other capture metadata, filename rules, then file-modified time. Its ordered **Filename time rules** accept one safe legacy `strptime` format or named rule per line. Examples are `%Y%m%d_%H%M%S`, `{date:DDMMYYYY}{sequence:3}` for names such as `06052024001`, `IMG_Epoch_{epoch:s}`, and `IMG_Epoch_{epoch:ms}`. A date counter orders same-day photos but deliberately does not invent a clock time. The optional **Recognize raw Unix epochs in IDs / hashes** control is off by default; it accepts exactly one bounded 10-digit seconds or 13-digit milliseconds value from 1991 through the current UTC year, and skips a filename containing multiple valid candidates. Rules are fixed grammar rather than arbitrary regular expressions, so locale-ambiguous parsing remains rejected. `IMG_Epoch_001` is a counter, not a date. Timeline accepts only 1991 through the current UTC year; unsupported or missing values remain visible under **Unparsed filename time**. When the filename-first option cannot parse a name, a valid EXIF **DateTimeOriginal** is used before leaving it unparsed; file-modified time is never substituted. The **Group photos** drop-down instantly changes the virtual hierarchy between Year, Year → Month, Year → Month → Week, and Year → Month → Day without rescanning photos. Its filter grid measures its actual controls: wide viewports use one row, constrained viewports use the semantic two-row form, and only genuinely narrow space stacks fields. A valid capture time saved from a photo viewer is explicit user intent and takes precedence after **Refresh dates** force-recatalogs the active registered roots. Changing date controls refreshes only derived catalog rows; it never rewrites EXIF/XMP. The current source and same-day sequence are stored with each catalogued timestamp.

Library is the default entry point. **Organize** shows active-root photos immediately; **Organize photos** replaces that grid with collapsible result groups. Its Basic view keeps one primary action, while Advanced reveals backend/model tuning and comparison details. Older Gallery/Clustering routes migrate to this photo surface, so no routed path set is lost.

Duplicate and burst results are review-only. Tools → Duplicates can independently find **Hash duplicates** (same-byte SHA-256), **Similar photos** (at least two of pHash/dHash/wHash agree), or both; burst review is separate. Scan actions share one responsive flow row and wrap only when the available workspace width requires it. Selection-dependent review and Trash actions appear only after selecting a result group. The selected group has an embedded virtual thumbnail preview that loads visible tiles only; **Open selected photos** still hands the complete group to the full-size viewer. Each Trash action opens an unchecked batch preview, excludes proposed keepers, and moves only checked files to recoverable ClusterLens Trash. People keeps name, reject, split, hide, ignore-similar, and merge actions explicit; source-changing actions run in visible Jobs and respect read-only safety mode.

Finite action rows in Detect, Library tools, Roots, Organize Advanced, the photo Inspector, and Settings use the same content-aware wrapping behavior. At supported text scales they add vertical rows instead of horizontal panning; long source paths are visually elided while their complete value remains available in the tooltip. Inspector editor pages and Settings storage/recovery pages scroll vertically when their content exceeds the viewport.

Organize can also store searchable cluster context. Manual description is always explicit. Automatic description remains off until enabled. The default provider is a local Ollama vision endpoint; an OpenAI-compatible endpoint is available only after the user configures it and confirms, for the current app run, that the representative photo plus displayed scalar EXIF/XMP may leave the device. Generated descriptions never write source metadata. Settings → Storage reports the Library catalog’s managed path and size; **Clear Library Cache** removes only derived photo metadata and generated descriptions, preserving registered roots, smart albums, review decisions, face labels, and source media.

## Active roots

**Edit roots** opens an on-demand Roots drawer rather than keeping a permanent management rail beside photos. Its **Sources** tab separates browsing from scope: clicking a tree row only browses, while checking rows creates a staged multi-root draft. **Apply** is the only action that updates the persisted shared scope; **Discard** restores it, and closing a dirty draft offers Apply, Discard, or Continue editing. Every root includes descendants and nested selections fold into their selected parent, so source photos are never discovered twice. The separate **Catalog** tab lists active and registered roots with compact online/catalog/job state plus Register/Enable/Disable, Refresh, Pause, Reveal, and Unregister actions. Catalog mutations are unavailable while a source draft is unapplied or read-only safety is enabled. **Storage** opens the separate Data Home manager; it is never a photo source.

Library, Organize, People, and Tools use the active roots by default. Face review/search/grouping and saved-person searches stay inside that set; Power-user mode retains an explicit **All indexed faces** override. Library registration remains explicit: selecting an active root never scans it until **Register in Library** is chosen.

The Sources pane also points to the separate **ClusterLens Data Home**. It contains indexes, previews, models, recovery journals, backups, logs, and reports; it is never a photo source. Storage exposes its managed categories and safe clear/recovery controls.

## Global source filters

**Settings → Sources** has global, source-read-only admission rules: ignore conventional thumbnail-like names, minimum image width, minimum image height, and minimum file size. All rules are disabled by default. The dialog shows a live **Active source filters** summary, so it is clear whether thumbnail matching is on before choosing **OK**. A thumbnail rule matches deliberate name tokens such as `IMG_123_thumb.jpg`, `thumbnail320.jpg`, and `.thumbnails` folders; it does not broadly exclude unrelated names such as `thumbprint.jpg`. Width or height below an enabled limit excludes the image, and file size uses KiB (1,024 bytes).

Photo views reload the active roots after these settings change and report their excluded count; subsequent Organize, People, and similarity scans use the same policy. Registered Library roots refresh immediately in a visible, cancellable Job; the catalog scan reports its excluded count and removes its own newly ineligible derived rows. No filter deletes, moves, rewrites, or otherwise changes a source file.

## Appearance and photo canvas

**Settings → General** offers **Follow system**, **Light**, and **Dark** application themes plus independent 100%, 125%, 150%, and 200% text scales. Follow system responds to the desktop color-scheme signal while ClusterLens is running; an unknown platform scheme safely falls back to dark. Profiles created before these settings retain the historical dark/100% appearance, while new profiles default to Follow system/100%. Appearance choices preview live; **OK** persists them and **Cancel** restores the exact prior theme, canvas, and text scale.

Both themes use a matte, low-glare palette with a consistent type scale and compact controls. Readable text states meet a 4.5:1 contrast target and meaningful focus boundaries meet 3:1 against their surrounding surfaces. Live dark/light changes update semantic palette roles, delegates, previews, icons, viewer chrome, runtime badges, splitters, and scroll boundaries without rebuilding the global geometry stylesheet. Job progress retains explicit queued/running/cancelling/done/cancelled/failed text and accessibility descriptions in addition to semantic color. Arrow keys move within workspace and People route tabs; Tab and Shift+Tab leave those composite controls, collapsed disclosure bodies leave the focus order, and closing Settings or Jobs returns focus to the invoking control. People status/progress and the active Jobs cancel target have explicit accessible names and state descriptions. At the supported 1280×720 minimum, large text keeps primary actions visible and moves secondary Settings, Roots, People, and Organize content into vertical scrolling or wrapping rows without horizontal panning. Numeric controls stay narrow, selectors use a short field width, and descriptive fields expand in their grid. Icons shorten common actions while the complete action description remains available by hover and keyboard focus. Native screen-reader qualification remains a release gate rather than an offscreen-test claim.

The shared photo inspector has a separate background selector: Adaptive, Theme, Black, Middle gray, or Light gray. Adaptive samples only the outer pixels of the preview that was already decoded, chooses a neutral background, and never changes the rest of the application. It performs no additional photo read, metadata scan, GPU work, or source-file change. Face-region outlines use a contrasting outer stroke so they remain visible against both photos and canvas choices.

## Public Repository Contents

This repository is source-only. It intentionally excludes generated runtime data:

- no image databases
- no embedding caches
- no FAISS or NumPy indexes
- no model weights
- no build artifacts or binaries
- no benchmark runtime output

Runtime data is created under the user's application data directory, such as `~/.local/share/ClusterLens` on Linux.

## Maintenance and release evidence

Settings → Storage lists the managed generated-data categories, their paths, and their sizes. Each clear action is explicit and runs in the background; source photos, tags, durable face labels, recovery history, and settings are never part of a category clear. Large rebuildable cache directories are detached atomically before deletion; if cancellation or a filesystem error leaves physical cleanup pending, Storage says which targets remain and **Clear Rebuildable Caches** retries only managed staging. The safe command-line cleanup preview remains available through `python scripts/cleanup_production_runtime.py`.

Tools → **Batch Rename & Metadata** and photo right-click menus include **Preview batch rename**. Templates support `{stem}`, `{index}`, `{date}`, `{year}`, `{month}`, and `{day}` while keeping each original extension. Every collision blocks the full preview; the confirmed operation runs in Jobs, can stop between files, and each completed rename can be restored from Settings → Safety & Recovery.

Create deterministic, non-private release inputs and evidence with:

```bash
uv run python scripts/create_release_fixture.py --output-dir /tmp/release_fixture --photos 128
uv run python scripts/verify_trash_recovery.py --fixture-dir /tmp/trash_fixture --report-dir /tmp/trash_report
uv run python scripts/verify_packaged_launch.py --executable /path/to/ClusterLens --runtime-root /tmp/launch_runtime --report-path /tmp/launch_report.json
uv run python scripts/verify_native_display.py --logical-size 1280x720 --text-scale 200 --report-dir /tmp/native_display_report
.venv/bin/python scripts/verify_ui_redesign.py --text-scale 200 --report-dir /tmp/ui_redesign_report
.venv/bin/python scripts/verify_runtime_badge.py --report-dir /tmp/runtime_badge_report
.venv/bin/python -B scripts/verify_work_conflicts.py --report /tmp/work_conflicts.json
CLUSTERLENS_UX31_REPORT=/tmp/mixed-ui.json bash scripts/test.sh tests/test_mixed_ui_workload.py -q
.venv/bin/python -B scripts/verify_mixed_ui_lifecycle.py --report-dir /tmp/mixed-ui-lifecycle
.venv/bin/python -B scripts/verify_clean_source.py --report-dir /tmp/clean-source
.venv/bin/python -B scripts/verify_process_network.py --report-dir /tmp/process-network -- bash scripts/test.sh
.venv/bin/python -B scripts/verify_durable_fault_matrix.py --report-dir /tmp/durable-fault-matrix
bash scripts/benchmark.sh --jobs-history-only
```

The launch and native-display checks are release/clean-VM commands: they require a built executable or a real display. Add `--require-fractional-scale` to the native command only on a display where Qt reports a non-integer device-pixel ratio. Add `--require-multiple-screens` only when at least two Qt screens can each contain the requested logical client size; a smaller window on a narrower output does not qualify the documented 1280×720 minimum. The redesign verifier is offscreen and accepts `--text-scale 100|125|150|200`; run all four values for release qualification. Its isolated generated active root prevents first-run guidance from masking the exact 18-route matrix. Version 3 evidence covers both logical sizes, all 10 Settings sections, all 4 Inspector panels, Roots Sources/Catalog, painted light/dark palette contrast, six Jobs states, clipping, primary bounds, complete logical focus reachability, fixture population, idle background work, and clean shutdown. The native verifier applies the same route and Settings coverage on a real Qt platform and reports visual, keyboard, and screen-reader qualification separately; a native visual/keyboard PASS does not imply screen-reader certification. The runtime-badge verifier captures all runtime states in dark and light themes from fixture capabilities. The work-conflict verifier drives the production coordinator through every documented conflict pair in both launch orders, including queued/running cancellation and failure/retry where applicable. The opt-in mixed-UI test runs bounded generated 4K decode/crop/catalog work through the production shell and writes a report when `CLUSTERLENS_UX31_REPORT` names a new file; the lifecycle wrapper runs that case with the locked interpreter in an isolated Linux process session and rejects a surviving descendant after close. Both remain offscreen CPU evidence, not native/model/GPU qualification. The packaged-launch verifier records both the frozen launcher digest and a deterministic complete onedir-tree digest, including symlink targets and executable mode. The clean-source verifier exports tracked plus non-ignored untracked candidate files into two source-only trees, forces offline frozen dependency resolution, and runs build, launch-check and the complete suite from separate outside directories; while the worktree is uncommitted it is not a substitute for the final committed checkout/package gates. The Linux process-network verifier follows Python and native descendants with `strace`, allows only Unix/netlink and loopback traffic, and treats a missing tracer, unparseable IP destination, timeout, failed command or surviving process as non-PASS. The durable-fault verifier runs every test referenced by `docs/durable_operation_fault_matrix.json` in an isolated process session. Declared POSIX killed-process cases must terminate with `SIGKILL`, emit unique state digests and recover twice; an injected Python exception does not count as a killed process. Its `validation` can pass while release `qualification` remains partial when a fault pair or native platform is explicitly untested; gaps are never converted to PASS. Every verifier uses a fresh evidence path and refuses to overwrite prior evidence.

## Run From Source

Normal desktop launches are single-instance per Data Home: a later launch focuses the existing window before app/model/cache construction. Use `--new-instance` only when intentionally running a separate desktop instance.

```bash
bash scripts/run_app.sh
```

## Command-line companion

`scripts/cli.sh` is a deliberately scoped companion for read-only inspection
and explicit managed-data maintenance. It never indexes media, changes photo
metadata, assigns faces, renames/moves/trashes/restores photos, or installs
models; those reviewed source-changing workflows remain GUI-only.

```bash
bash scripts/cli.sh status
bash scripts/cli.sh --json storage inventory
bash scripts/cli.sh storage backup --destination /absolute/backup-parent
bash scripts/cli.sh storage verify --backup /absolute/backup-folder
bash scripts/cli.sh storage recovery list
```

Settings → Storage is the equivalent visual surface. It lists each Data Home
category, creates and verifies checksummed backups, and can relocate only
managed derived data. An interrupted relocation retains the active Data Home
and a safe staging directory; choose **Resume Data Home Move** or **Discard
Staged Move** in Storage. Interrupted backups appear separately: **Discard
Staged Backup** removes only a marker-validated incomplete staging folder, while
**Finalize Published Backup** verifies and retains a backup whose publication
already committed. None of these choices touches source photos.

On Windows, the PowerShell helper uses the local virtual environment:

```powershell
.\scripts\run_app.ps1
```

## CUDA Source Runtime (Linux x86-64, Python 3.12)

The default source runtime remains CPU-compatible. On a CUDA 12.1 NVIDIA system, create the dedicated GPU runtime once and verify it:

```bash
bash scripts/setup_gpu_runtime.sh
.venv-gpu-cu121/bin/python scripts/verify_gpu_runtime.py
```

After setup, the normal `bash scripts/run.sh` launcher automatically uses `.venv-gpu-cu121` when it is installed; `bash scripts/run_app_gpu.sh` remains available to select it explicitly. Neither launcher installs packages, moves models, or downloads duplicate GPU copies. This uses the CUDA 12.1 ONNX Runtime wheel compatible with the bundled Torch 2.2.2/cuDNN 8 stack plus pinned RAPIDS cuML 25.10 for GPU HDBSCAN. The GPU launch shares `~/.local/share/ClusterLens` with the CPU-compatible launcher for managed model files. Image decoding, thumbnail generation, storage, SQLite, and filesystem work remain on CPU.

SCRFD/ArcFace face indexing uses the GPU only when the active runtime reports `CUDAExecutionProvider`; a CUDA Torch device alone does not move those ONNX models off CPU. ONNX model files are provider-neutral, so GPU execution uses the existing SCRFD/ArcFace files rather than a separate GPU model download. The Faces status strip reports the detector and embedder device separately in its tooltip. After local startup recovery, ClusterLens visibly checks the selected CPU/CUDA policy and installed clustering/face models without downloading or constructing inference models; model-dependent actions stay disabled with an actionable status until that check finishes. If a GPU or driver becomes available after startup, click the runtime badge, then **Rescan GPU Resources** in Support. The rescan runs in a background worker and visibly checks Torch CUDA, CUDA ONNX for SCRFD/ArcFace, and cuML HDBSCAN without opening user photos, downloading models, or changing the saved compute preference. Package/runtime changes still require a restart; `bash scripts/run.sh` will then select the installed GPU runtime.

In People → Find, **Index Active Roots** incrementally discovers the complete active-root union, skips unchanged image rows, and keeps decoding, crop-quality checks, GPU detection/embedding, and SQLite persistence in bounded background stages. Progress identifies the active detector provider and worker/batch limits; completion shows the detector and embedder provider separately with throughput. **Reindex** reruns detection and embedding for the same scope after a model or detection-setting change, while retaining compatible durable names. Neither action writes source media.

The verified execution policy is shared by embedding, face, clustering, and similarity services. With CUDA available, semantic PCA, cosine K-means, cuML HDBSCAN, cluster-quality scoring, graph-neighbor search, image/face similarity matrices, identity propagation, and duplicate-identity comparison use the GPU. The file-clustering HDBSCAN panel exposes minimum cluster size, automatic or explicit minimum samples, merge epsilon, and single-cluster allowance. Faces exposes K-means restarts, maximum iterations, and seed. Auto mode reports a missing or failed cuML HDBSCAN provider and uses native CPU HDBSCAN; an explicitly requested CUDA HDBSCAN run stops with install/change-policy guidance instead of silently changing policy. Other CUDA operation failures are recorded visibly before falling back to contiguous NumPy/OpenBLAS or native scikit-learn; fallback results are not cached under a GPU signature. The CPU runtime diagnostics show detected SIMD dispatch, BLAS threads, and libjpeg-turbo support.

## Tests

```bash
bash scripts/test.sh
```

The canonical runner uses frozen dependencies and isolated runtime, settings,
and bytecode directories. A test-only guard is inherited by spawned Python
workers: loopback remains available for deterministic local servers, while
external DNS and non-loopback connections fail immediately. Hugging Face and
Transformers offline modes are also forced. Real downloads belong only to
explicit qualification commands outside the default test suite.

Run the seeded synthetic CPU/GPU vector benchmark without reading user photos or caches:

```bash
bash scripts/benchmark.sh
CLUSTERLENS_BENCHMARK_PYTHON="$PWD/.venv-gpu-cu121/bin/python" bash scripts/benchmark.sh
```

To run only the generated thumbnail-index recovery fixture, which measures four independently created gallery/preview services against one temporary cache directory:

```bash
bash scripts/benchmark.sh --thumbnail-index-only
```

To run only the generated SQLite tag-workspace fixture (no media files are created or read):

```bash
bash scripts/benchmark.sh --tag-workspace-only
```

To measure only publication of the virtual Faces similarity grid (no photos, thumbnails, databases, models, or GPU work):

```bash
bash scripts/benchmark.sh --faces-arrangement-only
```

To benchmark only the bounded face-index scheduler with generated JPEGs, a deterministic detector/embedder, and a temporary SQLite database (no user media or models):

```bash
bash scripts/benchmark.sh --face-indexing-only
```

To benchmark only progressive Faces folder-review paging, using 1,001 temporary seeded index rows and no source-media reads, crop decoding, models, or user runtime data:

```bash
bash scripts/benchmark.sh --face-review-paging-only
```

For the production-size deterministic capacity fixture, which compares page-by-page canonicalization with the safe per-generation candidate snapshot:

```bash
bash scripts/benchmark.sh --face-review-paging-only --photos 11284 --faces 14477 --repeats 5
```

To exercise the complete offscreen **People → Detect** publication workflow at that same scale (temporary source files and SQLite data only), including first Photos, background metadata, Faces-tab rows, cached tail reveal, cancellation, plus separate retained-memory and Faces-publication profile passes:

```bash
bash scripts/benchmark.sh --people-detect-workflow-only
```

It intentionally does not load models or benchmark inference/full-resolution crop and thumbnail decoding; those have dedicated fixtures.

To measure large Library Timeline and grouped-gallery publication without source I/O, run the fixed 100,000-path hierarchy fixture. It reports worker preparation, the Qt model commit, selection preservation, the maximum offscreen event-pump interval, and retained allocation over alternating full/filtered resize-reflow cycles:

```bash
bash scripts/benchmark.sh --sectioned-gallery-only
```

To benchmark fixed full-resolution thumbnail/crop work and a three-worker
thumbnail/crop/catalog contention run in five fresh processes, with separate
allocation-traced and CPU-profiled runs:

```bash
bash scripts/benchmark.sh --production-workloads-only --report /tmp/clusterlens-production-workloads.json
```

The report refuses to overwrite an existing path. It uses generated local
JPEGs and an empty runtime; it does not run detector/embedder inference or use
the GPU, and “cold” means an empty application cache rather than a dropped OS
page cache.

The current production-model identities, upstream evidence, immutable managed
snapshot revisions and unresolved redistribution decisions are maintained in
[`docs/MODEL_PROVENANCE.md`](docs/MODEL_PROVENANCE.md). A repository or code
license is not treated as automatic permission to redistribute pretrained
weights.

Real detector/embedder inference is a separate opt-in release gate. Copy
`docs/real_model_manifest.example.json`, replace every placeholder with an
approved licensed fixture and pinned model revision/checksum, enumerate the
complete detector/embedder/backend coverage, then run CPU and CUDA separately:

```bash
.venv/bin/python -B scripts/qualify_real_models.py \
  --manifest /absolute/path/manifest.json \
  --fixture-root /absolute/path/approved-fixture \
  --model-root /absolute/path/pinned-models \
  --execution-mode cpu \
  --report /new/path/real-model-cpu.json
```

The gate never downloads or substitutes generated inference. Missing assets,
an unavailable requested GPU, or an unavailable model service produce
`NOT_RUN` with exit 3; a contract/checksum/provider failure exits 1. A PASS
records cold/warm and recreated-service inference, embedding dimensions and
ordering, clustering memberships, actual providers, and peak RSS. Use a fresh
report path and repeat with the qualified CUDA interpreter and
`--execution-mode cuda`.

To qualify the persistent clustering worker with the verified Fast Preview
CPU ONNX asset, including real inference, idle RSS, shutdown, process absence,
and restart with a new PID:

```bash
.venv/bin/python -B scripts/qualify_warm_worker.py \
  --model-asset-dir build/model_assets/fast_preview \
  --cycles 2 \
  --report /new/path/warm-worker-cpu.json
```

This command is opt-in because a source-only checkout may not contain the
build-produced ONNX asset. A missing or checksum-invalid asset is `NOT_RUN`,
not a fixture-model PASS.

To benchmark the first useful **People → All Faces** page, using 5,000 temporary seeded SQLite face rows and no source-media reads, crop decoding, models, or user runtime data:

```bash
bash scripts/benchmark.sh --people-faces-only --faces 5000 --repeats 5
```

To benchmark only the generated registered-root catalog and full-text query fixture:

```bash
bash scripts/benchmark.sh --library-catalog-only
```

To benchmark only Timeline's full filtered catalog read and virtual Year → Month grouping (the fixture has 1,200 catalog rows and no source media reads):

```bash
bash scripts/benchmark.sh --library-timeline-only
```

To benchmark the shared saved-name autocomplete control with a generated 10,000-name fixture (no database, photos, models, or network):

```bash
bash scripts/benchmark.sh --entity-picker-only --names 10000 --repeats 7
```

To benchmark only live palette application and the bounded generated-image viewer classifier:

```bash
bash scripts/benchmark.sh --theme-only --iterations 250
```

To measure virtual first-content and complete publication for generated 500
and 10,000-row workspace lists (no filesystem, model, cache, or user data):

```bash
bash scripts/benchmark.sh --ux-workflow-only
```

## Model Downloads

Model weights are not checked into the repository and are not bundled by default. The app asks before downloading missing model files and stores them in the runtime cache. On Linux, normal source launches keep managed face-model files in `~/.local/share/ClusterLens/cache/face_model_assets` and reusable download archives in `~/.local/share/ClusterLens/cache/face_model_downloads`, so they remain available after restarts. The source launcher ignores only stale `/tmp` runtime overrides; explicit persistent runtime locations remain supported.

Face Models in Settings lists every supported detector and embedder. It marks managed or external files as `Downloaded` and missing files as `Install for CPU/GPU`. Use **Choose Face Model Folder** to point ClusterLens at an existing folder containing downloaded ONNX files. Common direct SCRFD and ArcFace filenames, plus recognized SCRFD/ArcFace files in a bounded nested layout, are read in place and are never copied or deleted.

## People and face regions

Entering **People** loads only its active section; inactive face views become stale and load when opened, and no entry action rescans images. Its contextual sections are **People**, **All Faces**, **Unnamed**, **Detect**, **Review**, **Review & Name**, **Find**, and **Manage**. Shell tabs, the inner task selector, Folder Review subtabs, keyboard navigation, and legacy route aliases resolve to the same stable route IDs; unknown routes do nothing. **Review** is the pending automatic-name cleanup surface. **Detect** opens the active-root Folder Review and exposes **Detect Faces** / **Index Active Roots**; **Review & Name** opens that same saved review at its durable face-selection and naming surface. A photo-set **Review People** handoff selects Detect and shows the explicit scope, but waits for the user to start detection. The folder review publishes its first 500-image page, then a background worker prepares every remaining metadata page without changing the Photos viewport or constructing Faces tiles. Reaching the visible photo tail reveals one prepared page in place; an early tail request waits for the cache rather than rebuilding the gallery. Durable scan batches safely restart that unpublished cache, while changes to scope/filter/sort/task discard an obsolete generation. Loading tiles remain visible while viewport-bound crops arrive. Labelled means a durable saved face name; Unlabelled means no durable name; pending automatic proposals appear only in **Review**. Selected groups page 500 face members at a time and automatically continue to completion; selecting another group cancels the prior stream.

In **Detect**, the folder-wide **Faces** result tab has a visible Show selector for All, Labelled, Unlabelled, and Ignored tiles; it filters the loaded review only and never starts a rescan. **Ignore face** is available in its visible action row and face context menus. It removes only that saved region from ordinary Face search and clustering; select **Ignored** and choose **Restore face** to recover it. **Arrange Unlabelled by Similarity** is an explicit, cancellable background action that keeps the current folder review in this tab while placing its unlabelled indexed faces under inline similarity headings. It uses the current face-clustering backend and settings, does not write names, leaves draft/excluded/outlier faces under **Other / not grouped**, and **Reset to Folder Order** removes the session-only arrangement. In All faces, named faces remain in a separate folder-ordered section. Shift-click selects a range, Ctrl-click adds or removes individual faces, and the batch naming action or right-click menu assigns one name to every selected face. Every naming field and name prompt suggests saved names while still accepting a new one. In **Grouped Photos**, either select one or many clusters and use **Name Selected Clusters**, or select face tiles and use **Name Selected Faces** (or its right-click menu). Cluster naming writes every displayed member's own face region, including several different faces in one photo. Names are stored in the global face database, so they update Named Photos, clustered results, and searches everywhere in the application.

In **Find**, choose **Photo** to find from a supplied example or **Saved person** to search with a saved identity embedding; only the chosen primary workflow is shown, and switching tasks preserves both queries. A single detected face is selected automatically, while photos with multiple faces show the bounded chooser. Return runs the meaningful action for the focused query. The visible **Workflow** card contains the optional walkthrough and saved-search controls; saved searches persist across restarts. The active cards are compact: hover a control or use its `i` help icon for guidance. **Manage** keeps identity browsing first, with explicit collapsed **Management** and **Danger zone** disclosures below it. Collapsed bodies take no layout or keyboard-focus space; expanding them preserves selection, busy, and read-only guards. Destructive actions are never the default Enter action and retain their own confirmation and recovery guidance. Naming remains unavailable while **Read-only safety mode** is enabled.

People → **People** is the durable-name browser and editor. Its sidebar lists saved people with compact face/photo counts; hover a name to see an adaptive 9/12/20-photo contact sheet, shown/total counts, and co-occurring saved names without changing selection. Selecting a person shows each unique photo containing that saved face name. **Find Similar Faces** and **Deep Name + Similar** use the saved identity prototype without changing labels; users can review and deselect temporary matches before applying names. Photo selection remains photo-level, but Rename and Unlabel change only matching face regions, so other people in the same photo remain unchanged.

Organize → **Tags** is the durable photo-tag hub. It pages tags from the local tag database globally or within the current roots and shows the selected tag's photos without scanning source media. Use it to rename/merge or delete tags globally in the database; these bulk operations deliberately do not rewrite EXIF. Per-photo tag editing remains available from photo actions. Tag-filtered organization and group suggestions are explicit Jobs and write only the app tag database.

**Open photos** from People, Tags, Search, or an Organize result opens a session-only photo set. Its toolbar identifies the source and provides return, folder, organization, and people-review actions. Analyze uses only the routed paths; Review People transfers the same paths but never starts detection by itself. Per-photo face context is preserved while the route is active and discarded on return, so opening a result set does not create durable state.

Every photo surface exposes **Edit Face Regions** outside read-only mode. Its filter distinguishes **Unlabelled Faces** (saved regions, none named) from **No Face Regions** (a scanned photo with no detected region). The local face service prepares through Jobs and either opens the inspector or explains what is missing; it never blocks the window. The shared inspector keeps the image dominant and divides the right rail into **Info**, **People**, **Metadata**, and searchable **EXIF**. Use Fit, 1:1, deep zoom, cursor-centred wheel zoom, or middle-button pan; full resolution is requested only after zooming past preview scale. Scan or draw a box, save it, then name/rename/unlabel selected regions with autocomplete. Navigating or closing with dirty metadata or face regions requires Save, Discard, or Cancel.

Photo tiles load the visible viewport before any look-ahead or section preview. A fast scroll invalidates queued stale thumbnails; a decode already in progress finishes safely but cannot paint into the new viewport. Every photo inspector route has a **Face regions** overlay control, face-draft tools when the local face service is ready, and editable curated metadata. Its always-visible **Photo dates** block distinguishes file-modified time, every available EXIF date/time field (including capture, digitized, GPS, offset, and subsecond values), and any sidecar Timeline correction. **Set capture time…** focuses the date-and-time field; **Save sidecar** stores title, description, rating, keywords, creator, copyright, capture time, location, and textual custom fields in a reversible `<photo>.clusterlens.json` file without modifying the image. A valid capture time is used as an explicit Timeline correction on the next **Refresh dates**. **Embed into original** is separately confirmed, journalled, and limited to supported JPEG/TIFF textual fields. **Rename file** opens the existing collision-checked preview and recovery workflow.

Downloaded face and clustering models are not part of **Clear Rebuildable Caches**. On startup ClusterLens performs a local-only reconciliation before showing model selectors: it promotes a fully staged face install, restores a missing managed face bundle from its verified retained download archive when possible, and repairs a stale Hugging Face snapshot reference. It never downloads during this recovery. The rebuildable thumbnail SQLite index likewise reconciles only its own cache directory after missing/corrupt/interrupted state; it never touches source photos. **Clear Model Caches** remains the explicit action that removes downloaded model files.

See `BUILD.md` for deterministic build/run/test commands and `ARCHITECTURE.md` for storage and concurrency boundaries.

## Builds

ClusterLens has two package variants:

- `cpu`: default public build, CPU-only Torch runtime.
- `gpu-cu121`: separate CUDA 12.1 build for NVIDIA GPU systems.

```bash
python scripts/build_pyqt_binary.py --variant cpu --recreate-venv
python scripts/build_pyqt_binary.py --variant gpu-cu121 --recreate-venv
```

CPU is the default release target. GPU builds are intentionally separate because CUDA libraries make the package much larger.

## License

No open-source license has been selected yet. Until a license is added, all rights are reserved by default.
