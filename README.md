# ClusterLens

ClusterLens is a desktop image-clustering workspace for local photo folders. It scans image directories, generates cached embeddings, groups visually or semantically similar files, and provides a PyQt interface for reviewing, tagging, comparing, and organizing clusters.

## Local archive curation

The **Library** workspace is a local-first curation layer over explicit, user-registered roots. It incrementally catalogs filenames, scalar EXIF, and XMP sidecars without editing source photos, then provides a virtualized timeline, text search, dynamic smart albums, a manual duplicate/burst review queue, and a registered-root People Cleanup Inbox. Disabled roots are visibly excluded from all global work.

Timeline has an explicit **Timeline date** policy: metadata only, metadata then an unambiguous year-first filename date, filename first, or filename only. Changing it refreshes only derived catalog rows; it never rewrites EXIF/XMP. The current source is stored with each catalogued timestamp.

Duplicate and burst results are review-only: ClusterLens proposes a keeper but moves nothing until the user explicitly sends selected candidates to recoverable ClusterLens Trash. People cleanup keeps name, reject, split, hide, and merge actions explicit; source-changing actions run in visible Jobs and respect read-only safety mode.

Library can also store searchable cluster context. Manual description is always explicit. Automatic description remains off until enabled in Library. The default provider is a local Ollama vision endpoint; an OpenAI-compatible endpoint is available only after the user configures it and confirms, for the current app run, that the representative photo plus displayed scalar EXIF/XMP may leave the device. Generated descriptions are clearable from Library and never write source metadata. Settings → Storage reports the Library catalog’s managed path and size; **Clear Library Cache** removes only derived photo metadata and generated descriptions, preserving registered roots, smart albums, review decisions, face labels, and source media.

## Active roots

The Folder pane separates browsing from scope. Clicking a tree row only browses it; checking a row or using **Add browsed** adds that directory to the persisted shared **Active roots** set. Every root includes descendants, and nested selections fold into their selected parent so source photos are never discovered twice. The scope strip remains visible when the Folder pane is hidden.

Gallery, Clustering, Faces, Names, Tags, and Library use the active roots by default. Faces album/review/search/clustering and Names searches stay inside that set; Names provides an explicit **All indexed faces** override. Library remains explicit: selecting an active root never scans it until **Register active roots** is chosen.

The Sources pane also points to the separate **ClusterLens Data Home**. It contains indexes, previews, models, recovery journals, backups, logs, and reports; it is never a photo source. Storage exposes its managed categories and safe clear/recovery controls.

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

Settings → Storage lists the managed generated-data categories, their paths, and their sizes. Each clear action is explicit and runs in the background; source photos, tags, durable face labels, recovery history, and settings are never part of a category clear. The safe command-line cleanup preview remains available through `python scripts/cleanup_production_runtime.py`.

Gallery **File actions** and its right-click menu include **Preview batch rename**. Templates support `{stem}`, `{index}`, `{date}`, `{year}`, `{month}`, and `{day}` while keeping each original extension. Every collision blocks the full preview; the confirmed operation runs in Jobs, can stop between files, and each completed rename can be restored from Settings → Safety & Recovery.

Create deterministic, non-private release inputs and evidence with:

```bash
uv run python scripts/create_release_fixture.py --output-dir /tmp/release_fixture --photos 128
uv run python scripts/verify_trash_recovery.py --fixture-dir /tmp/trash_fixture --report-dir /tmp/trash_report
uv run python scripts/verify_packaged_launch.py --executable /path/to/ClusterLens --runtime-root /tmp/launch_runtime --report-path /tmp/launch_report.json
uv run python scripts/verify_native_display.py --report-dir /tmp/native_display_report
```

The launch and native-display checks are release/clean-VM commands: they require a built executable or a real display, use a fresh runtime root, and refuse to overwrite prior evidence.

## Run From Source

```bash
bash scripts/run_app.sh
```

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

In Faces → Find, **Index Active Roots** incrementally discovers the complete active-root union, skips unchanged image rows, and keeps decoding, crop-quality checks, GPU detection/embedding, and SQLite persistence in bounded background stages. Progress identifies the active detector provider and worker/batch limits; completion shows the detector and embedder provider separately with throughput. **Reindex** reruns detection and embedding for the same scope after a model or detection-setting change, while retaining compatible durable names. Neither action writes source media.

The verified execution policy is shared by embedding, face, clustering, and similarity services. With CUDA available, semantic PCA, cosine K-means, cuML HDBSCAN, cluster-quality scoring, graph-neighbor search, image/face similarity matrices, identity propagation, and duplicate-identity comparison use the GPU. The file-clustering HDBSCAN panel exposes minimum cluster size, automatic or explicit minimum samples, merge epsilon, and single-cluster allowance. Faces exposes K-means restarts, maximum iterations, and seed. Auto mode reports a missing or failed cuML HDBSCAN provider and uses native CPU HDBSCAN; an explicitly requested CUDA HDBSCAN run stops with install/change-policy guidance instead of silently changing policy. Other CUDA operation failures are recorded visibly before falling back to contiguous NumPy/OpenBLAS or native scikit-learn; fallback results are not cached under a GPU signature. The CPU runtime diagnostics show detected SIMD dispatch, BLAS threads, and libjpeg-turbo support.

## Tests

```bash
bash scripts/test.sh
```

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

## Model Downloads

Model weights are not checked into the repository and are not bundled by default. The app asks before downloading missing model files and stores them in the runtime cache. On Linux, normal source launches keep managed face-model files in `~/.local/share/ClusterLens/cache/face_model_assets` and reusable download archives in `~/.local/share/ClusterLens/cache/face_model_downloads`, so they remain available after restarts. The source launcher ignores only stale `/tmp` runtime overrides; explicit persistent runtime locations remain supported.

Face Models in Settings lists every supported detector and embedder. It marks managed or external files as `Downloaded` and missing files as `Install for CPU/GPU`. Use **Choose Face Model Folder** to point ClusterLens at an existing folder containing downloaded ONNX files. Common direct SCRFD and ArcFace filenames, plus recognized SCRFD/ArcFace files in a bounded nested layout, are read in place and are never copied or deleted.

## Naming faces

Entering **Faces** first loads the saved global album and the cached review for the selected folder; it does not rescan images. Folder review begins publishing as soon as its first 500-image page is ready, then continues loading pages in the background. The Detect tile grid visibly reports that progress and uses visible **Loading** tile placeholders while its viewport-bound crop queue works; source-path order is temporary until the complete review receives the chosen sort. Changing folder, scope, filter, sort, mode, or task safely discards the obsolete stream. The task selector has three primary actions: **All Faces**, **Detect**, and **Find**. **All Faces** has a visible **Show** selector for **All faces**, **Labelled faces**, and **Unlabelled faces**. Labelled means a durable saved face name; Unlabelled includes faces with no durable name plus pending automatic proposals that have not been accepted. The selector reloads only the matching paged album groups in the background. Its selected group loads 500 face members at a time and automatically continues to completion; selecting another group cancels the prior group stream. **All Faces** opens **Named Photos** first, grouping saved face tiles by person name; **Grouped Photos** shows the normal clustering groups. Its compact actions have hover text and an `i` help button. A cluster title lists every saved name currently found in that cluster.

In **Detect**, the folder-wide **Faces** result tab has a visible Show selector for All, Labelled, Unlabelled, and Ignored tiles; it filters the loaded review only and never starts a rescan. **Ignore face** is available in its visible action row and face context menus. It removes only that saved region from ordinary Face search and clustering; select **Ignored** and choose **Restore face** to recover it. **Arrange Unlabelled by Similarity** is an explicit, cancellable background action that keeps the current folder review in this tab while placing its unlabelled indexed faces under inline similarity headings. It uses the current face-clustering backend and settings, does not write names, leaves draft/excluded/outlier faces under **Other / not grouped**, and **Reset to Folder Order** removes the session-only arrangement. In All faces, named faces remain in a separate folder-ordered section. Shift-click selects a range, Ctrl-click adds or removes individual faces, and the batch naming action or right-click menu assigns one name to every selected face. Every naming field and name prompt suggests saved names while still accepting a new one. In **Grouped Photos**, either select one or many clusters and use **Name Selected Clusters**, or select face tiles and use **Name Selected Faces** (or its right-click menu). Cluster naming writes every displayed member's own face region, including several different faces in one photo. Names are stored in the global face database, so they update Named Photos, clustered results, and searches everywhere in the application.

In **Find**, **Find by Face** uses a supplied photo as the example. **Find by Name + Similar** searches with the saved identity embedding, so it returns both already named faces and visually similar faces that have not been named yet. The active cards are compact: hover a control or use its `i` help icon for guidance. The retained walkthrough and saved-search panel is hidden during this redesign pass, alongside the older Review & Name and identity-administration screens; no stored faces, identities, or actions are removed. Naming remains unavailable while **Read-only safety mode** is enabled.

The top-level **Names** workspace is the durable-name browser and editor. Its sidebar lists saved people globally with a name and compact face/photo count; hover a name to see a contact-sheet preview without changing the current selection. Selecting a name shows each unique photo containing a face explicitly saved with that name. **Find Similar Faces** explicitly searches that saved identity's prototype and opens a bounded face-crop result view. **Deep Name + Similar** visits the complete indexed scope, adds each newly found unlabelled face to a temporary search corpus until no new face remains, and presents the combined face result without changing labels. Its unlabeled regions are preselected, but can be deselected before **Apply Selected Names** writes successful XMP/EXIF regions and durable labels. The equivalent Deep Name + Similar action is also available in Faces → Find. Other people's existing names remain visible for review but are never overwritten or used to expand the corpus. Return with **Show Named Photos** to the exact durable-photo view. Open one selected match or all match photos in the top-level Gallery with its face-region context. Select photos with Ctrl-click, Shift-click, or their checkboxes, then right-click any selected photo for **Rename Selected** or **Unlabel Selected**. Before either action, choose the actual face-region name found in the selected files; this handles photos containing more than one named person. Selection is by photo, but every change is face-level: Rename and Unlabel change only matching regions, so other people detected in the same photo remain unchanged. The durable global mapping and open Faces views refresh after each completed action. On the first global label refresh, ClusterLens also recovers legacy `manual_selected_faces` proposals created by older versions, without overwriting a conflicting durable label.

The top-level **Tags** workspace is the durable photo-tag hub. It pages tags from the local tag database globally or within the current folder and shows the selected tag's photos without scanning source media. Use it to rename/merge or delete tags globally in the database; these bulk operations deliberately do not rewrite EXIF. Gallery tag editing remains available for per-photo edits. Add one or more selected tags to the Tags filter, choose **Any** or **All**, and run the existing tag-filtered clustering flow. Tags also contains the selected-cluster suggestion controls: generation is explicit and runs in Jobs, and applying model suggestions writes only the app tag database. Open Tags after selecting a cluster in Clustering to generate or apply its suggestions.

**Open in Gallery** from Faces, Names, Tags, or a selected cluster always opens the top-level Photos workspace as a session-only route. Its toolbar identifies the source and provides return, folder, analysis, and face-review actions. Analyze uses only the routed/selected paths; Review Faces transfers the same paths to Faces but never starts detection by itself. Per-photo face context is preserved while the route is active and discarded when returning to the normal folder gallery, so opening a result set does not create durable state.

The **Photos** workspace and the main gallery always expose **Edit Face Regions** outside read-only mode, even before Faces or Names has been opened. Its face filter distinguishes **Unlabelled Faces** (one or more saved regions, none named) from **No Face Regions** (a scanned photo with no detected region). The requested local face service prepares through the visible Jobs flow and either opens the inspector or explains the missing local requirement; it never blocks the window. In the inspector, use Fit, 1:1, deep zoom (up to 6400%), scroll-wheel cursor zoom, or middle-button pan to inspect a photo; full resolution is requested once only after zooming past preview scale. Scan or draw a box, save it, select that face region, then use **Apply Name**, **Rename**, or **Unlabel** in the **Selected face regions** section. A JPEG receives standard MWG/XMP face regions plus a ClusterLens EXIF mirror; PNG and other formats, or an image with an existing `.xmp` sidecar, use that sidecar without rewriting pixels. An explicit selected-region name is user intent and is saved even when automatic prototype quality would reject that region; automatic matching, suggestions, search, and clustering continue to use their configured quality filters. On re-index, an existing durable database label wins over a conflicting external XMP name.

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
