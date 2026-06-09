# ClusterLens

ClusterLens is a desktop image-clustering workspace for local photo folders. It scans image directories, generates cached embeddings, groups visually or semantically similar files, and provides a PyQt interface for reviewing, tagging, comparing, and organizing clusters.

## Public Repository Contents

This repository is source-only. It intentionally excludes generated runtime data:

- no image databases
- no embedding caches
- no FAISS or NumPy indexes
- no model weights
- no build artifacts or binaries
- no benchmark runtime output

Runtime data is created under the user's application data directory, such as `~/.local/share/ClusterLens` on Linux.

## Run From Source

```bash
uv run python -m apps.pyqt_production
```

On Windows, the PowerShell helper uses the local virtual environment:

```powershell
.\scripts\run_app.ps1
```

## Tests

```bash
uv run --with pytest python -m pytest tests
```

## Model Downloads

Model weights are not checked into the repository and are not bundled by default. The app asks before downloading missing model files and stores them in the runtime cache. First-run setup can also download the default model after installation.

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
