# Benchmarks

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

Before the UI change, five direct capability-refresh samples took 1316.577, 0.015, 0.008, 0.006, and 0.006 ms; that path did not verify an ONNX provider. After the change, three complete `diagnostics("auto", refresh=True)` rescans took 705.585, 4.597, and 6.299 ms (6.299 ms median), and every sample selected `CUDAExecutionProvider`. The different workloads are recorded as validation evidence, not as a performance-improvement claim. The first probe includes lazy CUDA initialization; the UI therefore runs it on an `AsyncJob` worker.

## 2026-09-10 Names workspace assessment

This is a baseline for the committed Names workspace (`7fa09be`), not a comparison with an earlier revision. It used an isolated temporary SQLite database and did not read or alter user photos, caches, downloaded models, or the global face database.

- Environment: Ubuntu 26.04.1, Python 3.12.13, AMD Ryzen 7 7800X3D (8 cores / 16 threads), 60 GiB RAM, NVIDIA GeForce RTX 4090 (24 GiB; driver 595.91.07).
- Startup: five offscreen release-gate runs with `PYTHONHASHSEED=0`. Import median 123.265 ms (120.599–130.851 ms); window-construction median 404.640 ms (392.415–414.372 ms); RSS median 94.430 MiB (94.348–94.469 MiB). All five runs met the 1.0 s import, 1.5 s window, and 256 MiB RSS budgets. None eagerly imported Torch, ONNX Runtime, the face search service, or the face UI.
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
