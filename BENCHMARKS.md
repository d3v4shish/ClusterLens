# Benchmarks

## 2026-09-16 filename-date catalog and batch-rename safety

Method: `bash scripts/benchmark.sh --library-catalog-only --photos 120 --timeline-photos 1200 --repeats 5` generated fresh JPEG/XMP and SQLite-only timeline fixtures. Before any retained timeline hot-path change, it measured a 23.796 ms catalog scan, 0.406 ms warm FTS query, and 1.072 ms full 1,200-row timeline query/grouping median. The profile attributes catalog time to canonical filesystem paths, Pillow metadata reads, and bounded SQLite writes; timeline time is indexed SQLite fetching plus timestamp bucketing. No user media, runtime cache, model, GPU, or network data was used.

An attempted Python fast path for ISO month extraction regressed a subsequent noisy run and profiled worse than CPython's `datetime.fromisoformat`, so it was removed. There is intentionally no speedup claim for the filename-date policy. Batch rename is a safety workflow rather than a throughput claim: deterministic recovery tests cover template rendering, collision blocking, journalling, cancellation boundaries, and restore.

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
