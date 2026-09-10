# Hotspots

- Image decode, embedding inference, ONNX model loading, face indexing, and filesystem discovery are potentially expensive and must remain off the UI thread.
- Thumbnail and face-tile memory are bounded by model-owned caches; gallery/album views are paged to avoid creating widgets for all records.
- Names queries aggregate durable labels and resolve the selected name's distinct photo paths in worker threads. The names sidebar is paged; its gallery retains the standard bounded thumbnail loader rather than constructing one widget per photo.
- Model inventory only checks known managed paths and direct files in a user-selected model folder. It deliberately does not recursively scan arbitrary disks from the UI thread.
- Startup model recovery is local-only and runs before selectors are created. It hashes a retained face archive only when its verification sidecar is missing and a managed bundle needs restoration; normal launches do not rehash installed model payloads.
- CUDA availability depends on the installed Torch/ONNX Runtime provider and NVIDIA driver. The status line and runtime badge report the effective runtime, not the requested preference. A verified CUDA runtime remains ready even when the badge tooltip also contains separate application-health notes. On Debian/Linux source installs, `scripts/setup_gpu_runtime.sh` creates the separate CUDA 12.1 runtime and its verifier must confirm `CUDAExecutionProvider` before Face indexing is run with CUDA selected.
