# ClusterLens Performance and UI Action Backlog

Last updated: 2026-08-25

This is the active backlog. It tracks remaining risk, not every historical UI idea. An item is complete only when the production path and its evidence are both present.

## Current verified baseline

- [x] Full automated suite: 401 tests passed on Python 3.12.
- [x] Production startup gate: import 0.125s, window construction 0.329s, current RSS 94.6 MiB.
- [x] Startup keeps `torch`, `onnxruntime`, `app.services.face_search`, and `ui.search_pane` unloaded.
- [x] Logical minimum display contract remains 1920x1080 in landscape or portrait.
- [x] Folder scope, face-album paging, gallery cache bounds, recovery, async Faces startup, and source-launch desktop behavior have focused regression tests.
- [x] The Linux CUDA package runs Torch and ONNX inference on the RTX 4090; a repeated packaged request reuses both embeddings and clustering results.
- [x] Installed SigLIP image/text requests collapse to one cache key and reuse the existing 815.9 MB cache without another download.

## Implemented in the current performance/UI pass

### Startup and responsiveness

- [x] Lazily import Torch and ONNX Runtime only when runtime detection or execution needs them.
- [x] Construct the production shell without importing or constructing the Faces stack.
- [x] Open Faces through a cancellable background job with a truthful loading/error/retry state.
- [x] Resolve alternate face databases, inspect database metrics, and load folder-review data off the UI thread.
- [x] Keep injected face-library adapters working in the background resolver.

### Folder and data correctness

- [x] Use canonical path containment instead of string-prefix matching.
- [x] Escape SQLite `LIKE` wildcards and match the exact folder or a real descendant only.
- [x] Apply the same scope semantics to face search, albums, review metrics, perceptual hashes, and same-folder sorting.
- [x] Cover sibling-prefix, wildcard, Windows-style, and symlink-escape cases.

### Gallery performance

- [x] Replace per-item retained images with a bounded model-owned LRU image cache.
- [x] Evict thumbnails outside the visible/prefetch window and ignore stale off-screen results.
- [x] Preserve model rows, selections, overlays, and cached images during incremental removals.
- [x] Decode clipboard images in a worker and update the clipboard only on the UI thread.
- [x] Report visible thumbnail activity separately from the total gallery size; do not leave a false loading state.
- [x] Track disk-thumbnail size/access in SQLite and coalesce access updates instead of rescanning the cache directory for every prune.

### Faces album scale

- [x] Load album projections without embedding blobs.
- [x] Query group counts and member pages in SQL rather than hydrating the complete face index.
- [x] Load at most 100 groups and 200 members initially, with explicit incremental actions.
- [x] Retain compatibility with older/injected face-service adapters.

### UI consistency and safety

- [x] Replace clipped horizontal Faces task tabs with an accessible vertical task selector.
- [x] Keep the results canvas dominant and retain the collapsible task panel.
- [x] Hide the folder-only scope control on All Faces and when no folder is selected.
- [x] Make field labels wrap and split inspector face actions into a non-clipping grid.
- [x] Keep the View and Cluster actions labels visible beside their icons.
- [x] Route gallery tiles, face tiles, and hover previews through shared dark-theme tokens.
- [x] Standardize recoverable file moves as “ClusterLens Trash” in the destination folder, confirmations, progress, results, and Recovery UI.
- [x] Do not write a user desktop entry during a source launch; packaged builds or explicit installers own that mutation.
- [x] Keep the Faces workspace usable at compact widths with a bounded task panel, a collapsible splitter, scrollable task controls, and scrollable result tabs.
- [x] Restore visible Advanced clustering/model/backend controls, including HDBSCAN tuning, and persist them across restarts.
- [x] Keep the production Models page scrollable on short displays and expose every optional face inference pack from Settings.

### Models, caching, and cancellation

- [x] Centralize standard model acquisition in a killable worker process with shared cache leases, deduplication, resumable partials, visible Jobs progress, and cancellation.
- [x] Verify cached model readiness before suppressing the download prompt; reject zero-byte or incomplete assets.
- [x] Install face-model packs atomically from checksum-verified, resumable, cross-process-deduplicated archives.
- [x] Expose face-pack installation, installed-component deletion, cache opening, and reusable-download-cache clearing as confirmed Jobs actions.
- [x] Prevent the built-in FaceNet embedder from downloading invisibly; it now routes users to Settings > Models and reloads Faces after installation.
- [x] Version embedding/result cache signatures, validate dimensions and payload structure, and invalidate incompatible cache entries safely.

### GPU execution

- [x] Treat explicit CUDA as a requirement: unavailable CUDA fails visibly instead of silently running the whole job on CPU.
- [x] Prefer CUDA Torch when CUDA ONNX Runtime is unavailable, apply CUDA OOM batch backoff, and keep the worker warm when requested.
- [x] Package the CUDA 12.1 Torch/torchvision stack plus CUDAExecutionProvider and validate actual packaged inference on the host GPU.

### Regression protection

- [x] Add release budgets: import <=1.0s, window construction <=1.5s, RSS <=256 MiB, and no eager heavy modules.
- [x] Add focused tests for path scope, paged albums, bounded gallery images, desktop entry behavior, and startup performance.
- [x] Run the complete deterministic suite after the changes.

## Next actions, in priority order

### P0 — Required release evidence

- [ ] Run a real-folder smoke benchmark with at least 10,000 photos and record cold/warm clustering, first visible gallery paint, peak RSS, and thumbnail-cache growth under `benchmarks/release_gates/`.
- [ ] Run a face-library scale fixture with at least 100,000 face rows; verify first group paint, group paging, member paging, filtering, cancellation, and peak RSS against an agreed baseline.
- [ ] Capture native X11 and Wayland checks at logical 1920x1080 plus 150% and 200% scaling. Verify header, Faces navigation, inspector actions, menus, dialogs, focus rings, and no clipping.
- [ ] Exercise ClusterLens Trash end to end on real files: move, name collision, partial failure, app restart, restore, and restore conflict policies.
- [ ] Qualify CPU-only and CUDA packages offline, including first launch, Faces open, model readiness, fallback messaging, and clean shutdown.

### P1 — Stress and observability

- [ ] Add a rapid-switch stress test: Clustering -> Faces -> Clustering while Faces initialization, folder review, and album paging are cancelled or replaced.
- [ ] Add a long-scroll gallery benchmark that asserts bounded QImage count, bounded request queue size, stable RSS after eviction, and no stale tile repaint.
- [ ] Add thumbnail-index recovery tests for a missing/corrupt index, orphaned WebP files, concurrent readers, and pruning after interrupted shutdown.
- [ ] Record Faces initialization, first album page, first member page, and first visible thumbnail timings in the performance report—not only logs.
- [ ] Make every warning/error state link to the relevant Settings, Recovery, folder, or diagnostic action where a useful action exists.

### P2 — Reduce maintenance risk

- [ ] Split `src/ui/search_pane.py` into task controllers/widgets for All Faces, Folder Review, Face Search, Identities, and shared async coordination. Keep one public `SearchPane` facade during migration.
- [ ] Split `src/app/services/face_search.py` into schema/repository, album queries, indexing, recognition, identities, and model-runtime modules. Keep compatibility exports until callers migrate.
- [ ] Move the remaining custom-painted colors in `zoomable_image.py` and contact-sheet rendering to semantic theme tokens, with contrast tests for normal, selected, warning, and danger states.
- [ ] Remove deprecated compatibility shims once packaged and source entry points import only authoritative `src/ui` modules.

## Release commands

```bash
uv run --with pytest python -m pytest -q
.venv/bin/python -m apps.pyqt_production.release_gates \
  --folder /path/to/representative/photos \
  --report-dir benchmarks/release_gates
```

Do not mark native display, CUDA, package lifecycle, scale, or soak evidence complete based only on offscreen tests.
