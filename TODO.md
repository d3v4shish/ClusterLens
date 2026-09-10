# Current implementation plan

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
- [ ] Update the historical performance backlog only when a benchmark is actually run.

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
