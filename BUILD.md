# Build and validation

From a clean checkout with Python 3.11 or 3.12 and `uv` installed:

```bash
bash scripts/build.sh
bash scripts/run.sh
bash scripts/test.sh
bash scripts/benchmark.sh
```

The canonical scripts use the locked dependency graph. Tests and the synthetic benchmark use isolated temporary runtime/bytecode directories; the benchmark uses a fixed seed and reads no user photos or mutable application cache. To benchmark the CUDA runtime instead of the default source interpreter:

```bash
CLUSTERLENS_BENCHMARK_PYTHON="$PWD/.venv-gpu-cu121/bin/python" bash scripts/benchmark.sh
```

Build release packages explicitly:

```bash
uv run python scripts/build_pyqt_binary.py --variant cpu --recreate-venv
uv run python scripts/build_pyqt_binary.py --variant gpu-cu121 --recreate-venv
```

`run_app.sh` launches with the normal per-user application-data directory. On Linux, managed face models are retained in `~/.local/share/ClusterLens/cache/face_model_assets`; reusable face-model download archives are retained in `~/.local/share/ClusterLens/cache/face_model_downloads`.

For the dedicated CUDA 12.1 source runtime on Linux x86-64 with Python 3.12:

```bash
bash scripts/setup_gpu_runtime.sh
.venv-gpu-cu121/bin/python scripts/verify_gpu_runtime.py
bash scripts/run_app_gpu.sh
```

The GPU setup is deliberately separate from `uv sync --frozen`, so CPU source environments and non-Linux platforms retain their existing dependency set. It pins RAPIDS cuML 25.10 and scikit-learn 1.7.2 while preserving Torch 2.2.2's CUDA 12.1 libraries. `uv` uses best-match resolution only across the explicitly declared PyPI, PyTorch, and NVIDIA indexes because RAPIDS' `cuda-python` wrapper is not present on the first index. The verifier must report `CUDAExecutionProvider` and successful Torch, ONNX, semantic-PCA, cosine-K-means, cuML-HDBSCAN, silhouette, graph-neighbor, and dense-similarity smoke tests before GPU indexing or clustering is used.

The CPU and CUDA packages are separate artifacts. The CUDA variant requires a compatible NVIDIA driver and CUDA-enabled Torch/ONNX Runtime packages. Use the Settings runtime diagnostics to verify the effective device after installation.

`scripts/benchmark.sh` runs four generated, isolated fixtures: the fixed vector workload, an eight-region 1600×1200 JPEG XMP merge/readback workload, a four-service thumbnail-index recovery workload, and a 75,000-row SQLite tag-workspace query workload. They are comparable only within their own fixtures. Run only the deterministic thumbnail recovery fixture with `bash scripts/benchmark.sh --thumbnail-index-only`, or only the SQLite tag fixture with `bash scripts/benchmark.sh --tag-workspace-only`. End-to-end photo-pipeline claims still require the fixed photo fixture described in `BENCHMARKS.md`.

## Release evidence

These commands never read user photos or existing runtime data. Every target is a new/empty evidence directory; the fixture generator allows overwrite only for a deliberately named fixture directory.

```bash
uv run python scripts/create_release_fixture.py --output-dir /tmp/release_fixture --photos 128 --seed 20260913
uv run python -m apps.pyqt_production.release_gates --folder /tmp/release_fixture/photos --report-dir /tmp/release_gates
uv run python scripts/verify_trash_recovery.py --fixture-dir /tmp/trash_fixture --report-dir /tmp/trash_evidence
uv run python scripts/verify_packaged_launch.py --executable /path/to/ClusterLens --runtime-root /tmp/packaged_runtime --report-path /tmp/packaged_launch.json
uv run python scripts/verify_native_display.py --report-dir /tmp/native_display
```

Run the last command only on a real display or clean VM; it rejects offscreen/minimal Qt platforms. The packaged-launch verifier requires a PyInstaller-built executable and intentionally fails for a source interpreter because it is not a frozen artifact.
