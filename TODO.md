# Current implementation plan

## Clean-native operational safety and release evidence

- [x] Expose generated-storage categories in the production Settings dialog without importing the legacy folder cache or catalog.
  Contract: Storage lists each generated category with its managed paths and size; each clear action is an explicit, cancellable background operation that preserves source photos, tags, durable face labels, recovery history, settings, and non-selected categories.
  Validation: six deterministic cache-service checks cover category boundaries and durable-data preservation; an offscreen production-dialog check covers the controls and worker callback contract; the isolated source smoke confirms clean teardown.
- [x] Add deterministic release-evidence commands built around isolated synthetic inputs.
  Contract: fixture generation, fixture-manifest verification, trash recovery, and packaged-launch verification refuse unsafe paths or report overwrites; they never read user photos or runtime data. Existing release gates remain the source of startup/UI readiness checks.
  Validation: four temporary-directory tests cover manifest tampering, empty-directory guards, trash collision/restore behavior, and report validation; build and whitespace validation are recorded below.
- [ ] Design a Clean-native CLI after the user chooses its supported workflows.
  Contract: it will compose Clean services only and explicitly identify GUI-only workflows; it will not import the legacy per-folder cache or face catalog.
  Validation: pending user scope decision; this is deliberately not implemented as an unreviewed port.

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
  Validation: focused worker lifecycle tests, compilation, and whitespace checks pass. The documented full-script run remains outstanding because its fresh offscreen runtime can stall while Qt drains queued worker events across the broad UI fixture; this is retained visibly rather than treated as a green full run.

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
- [ ] Resolve the existing order-sensitive full-suite fixture failure before claiming a green full run.
  Contract: `bash scripts/test.sh` must complete without sharing mutable model-asset state between production-support tests.
  Validation: the 2026-09-13 isolated rerun reached 67% after all production-support, service, startup-readiness, and early UI-smoke checks, then ended without a pytest summary while UI smoke was still running. Focused production (46), service plus UX-acceptance (188), build, compilation, and whitespace checks pass. This remains visible rather than being presented as a green full run.

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

- [x] Hide the verbose advanced Find walkthrough/saved-search panel, replace explanatory prose in core Find sections with existing hover/help affordances, and use the available grid layout for selected-face, query-face, and name-query controls.
  Contract: Find retains selected-face, query-photo, and saved-name search behavior; direct controls fit horizontally when their containing pane is sufficiently wide; advanced walkthrough and saved-search implementation remains retained but hidden.
  Validation: `QT_QPA_PLATFORM=offscreen uv run --with pytest python -m pytest tests/test_ui_smoke.py -k 'face_workspace_guidance_and_action_labels_are_visible or face_workspace_layout_keeps_search_cards_readable_in_task_pane'` passed (2 tests); the focused three-test Find/Names command passed after grid assertions were added. `uv run python -m py_compile src/ui/search_pane.py src/ui/names_pane.py tests/test_ui_smoke.py` and `git diff --check` passed. A broader offscreen UI rerun retains two recorded unrelated failures: the unchanged Scan group's 330 px size hint at a 280 px sidebar viewport, and a test that assumes the currently persistent SCRFD/ArcFace profile is not installed.
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

- [x] Redesign the primary Faces workflow without removing the retained review/identity tools.
  Contract: Faces opens the saved global album and the selected folder's cached review without rescanning; the task selector exposes only All Faces, Detect, and Find; Review & Name and identity administration remain implemented but hidden; named photos and normal grouped photos have explicit views; cluster labels list all saved names present; and Find supports both a query photo and saved-name similarity search.
  Validation: 11 focused offscreen UI tests passed for task-to-internal-tab mapping, cached workspace loading, hidden retained pages, named/grouped result views, multiple cluster naming, query/name Find workflows, and state restoration. The GPU source app also launched and loaded the saved global album without scanning.

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
## Measured performance pass

- [x] Establish a deterministic baseline and profile the current hot path.
  Contract: use generated, seeded inputs only; identify a CPU/GPU, memory, I/O, or contention bottleneck without reading user photos or caches.
  Validation: `bash scripts/benchmark.sh --thumbnail-index-only` used 1,024 generated WebP entries and four independently created services in a temporary directory. Before the change, four opens took a 200.411 ms median; the profile spent 0.474 s reconciling the same cache four times. Evidence and environment constraints are recorded in `BENCHMARKS.md` and `HOTSPOTS.md`.
- [x] Apply the smallest safe optimization to the measured bottleneck.
  Contract: preserve deterministic memberships/results and existing CPU/GPU fallback semantics; keep all work off the Qt UI thread.
  Validation: one process-wide, bounded ready state shares completed recovery only while the WebP/index fingerprint matches. The same fixture measured 43.321 ms median for four opens (4.63× faster); corruption, missing-index, orphan, concurrent-reader, and shared-reconciliation tests, build, syntax, and whitespace checks pass. Results are documented in `BENCHMARKS.md` and `HOTSPOTS.md`.
