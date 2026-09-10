# Benchmarks

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
