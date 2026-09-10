# Build and validation

From a clean checkout with Python 3.11 or 3.12 and `uv` installed:

```bash
uv sync --frozen
bash scripts/run_app.sh
uv run --with pytest python -m pytest tests
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

The GPU setup is deliberately separate from `uv sync --frozen`, so CPU source environments and non-Linux platforms retain their existing dependency set. The verifier must report `CUDAExecutionProvider` and successful Torch and ONNX smoke tests before face indexing is run with Compute device set to CUDA.

The CPU and CUDA packages are separate artifacts. The CUDA variant requires a compatible NVIDIA driver and CUDA-enabled Torch/ONNX Runtime packages. Use the Settings runtime diagnostics to verify the effective device after installation.

There is no benchmark command that can claim a comparable result without a fixed photo fixture. See `BENCHMARKS.md` for the required invocation and reporting fields.
