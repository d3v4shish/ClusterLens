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
bash scripts/run_app.sh
```

On Windows, the PowerShell helper uses the local virtual environment:

```powershell
.\scripts\run_app.ps1
```

## CUDA Source Runtime (Linux x86-64, Python 3.12)

The default source runtime remains CPU-compatible. On a CUDA 12.1 NVIDIA system, create the dedicated GPU runtime once, verify both Torch and ONNX CUDA inference, then start the app from it:

```bash
bash scripts/setup_gpu_runtime.sh
.venv-gpu-cu121/bin/python scripts/verify_gpu_runtime.py
bash scripts/run_app_gpu.sh
```

This uses the CUDA 12.1 ONNX Runtime wheel compatible with the bundled Torch 2.2.2/cuDNN 8 stack plus pinned RAPIDS cuML 25.10 for GPU HDBSCAN. It does not download or move face models; the GPU launch still uses `~/.local/share/ClusterLens` for managed model files. Image decoding, thumbnail generation, storage, SQLite, and filesystem work remain on CPU.

SCRFD/ArcFace face indexing uses the GPU only when the active runtime reports `CUDAExecutionProvider`; a CUDA Torch device alone does not move those ONNX models off CPU. The Faces status strip reports the detector and embedder device separately in its tooltip. If a GPU or driver becomes available after startup, click the runtime badge, then **Rescan Available Resources** in Support. The rescan runs in a background worker and does not install packages or change the saved compute preference. Package/runtime changes still require restarting with `scripts/run_app_gpu.sh`.

The verified execution policy is shared by embedding, face, clustering, and similarity services. With CUDA available, semantic PCA, cosine K-means, cuML HDBSCAN, cluster-quality scoring, graph-neighbor search, image/face similarity matrices, identity propagation, and duplicate-identity comparison use the GPU. The file-clustering HDBSCAN panel exposes minimum cluster size, automatic or explicit minimum samples, merge epsilon, and single-cluster allowance. Faces exposes K-means restarts, maximum iterations, and seed. CUDA failures are reported visibly and fall back to contiguous NumPy/OpenBLAS, native scikit-learn, or native HDBSCAN execution; fallback results are not cached under a GPU signature. The CPU runtime diagnostics show detected SIMD dispatch, BLAS threads, and libjpeg-turbo support.

## Tests

```bash
bash scripts/test.sh
```

Run the seeded synthetic CPU/GPU vector benchmark without reading user photos or caches:

```bash
bash scripts/benchmark.sh
CLUSTERLENS_BENCHMARK_PYTHON="$PWD/.venv-gpu-cu121/bin/python" bash scripts/benchmark.sh
```

## Model Downloads

Model weights are not checked into the repository and are not bundled by default. The app asks before downloading missing model files and stores them in the runtime cache. On Linux, normal source launches keep managed face-model files in `~/.local/share/ClusterLens/cache/face_model_assets` and reusable download archives in `~/.local/share/ClusterLens/cache/face_model_downloads`, so they remain available after restarts. The source launcher ignores only stale `/tmp` runtime overrides; explicit persistent runtime locations remain supported.

Face Models in Settings lists every supported detector and embedder. It marks managed or external files as `Downloaded` and missing files as `Install for CPU/GPU`. Use **Choose Face Model Folder** to point ClusterLens at an existing folder containing downloaded ONNX files (including direct SCRFD and ArcFace filenames); the files are read in place and are never copied or deleted.

## Naming faces

Entering **Faces** first loads the saved global album and the cached review for the selected folder; it does not rescan images. The task selector has three primary actions: **All Faces**, **Detect**, and **Find**. **All Faces** opens **Named Photos** first, grouping saved face tiles by person name; **Grouped Photos** shows the normal clustering groups. Its compact actions have hover text and an `i` help button. A cluster title lists every saved name currently found in that cluster.

In **Detect**, use the folder-wide detected-face tiles to select one or many faces. Shift-click selects a range, Ctrl-click adds or removes individual faces, and the batch naming action or right-click menu assigns one name to every selected face. In **Grouped Photos**, either select one or many clusters and use **Name Selected Clusters**, or select face tiles and use **Name Selected Faces** (or its right-click menu). Cluster naming writes every displayed member's own face region, including several different faces in one photo. Names are stored in the global face database, so they update Named Photos, clustered results, and searches everywhere in the application.

In **Find**, **Find by Face** uses a supplied photo as the example. **Find by Name + Similar** searches with the saved identity embedding, so it returns both already named faces and visually similar faces that have not been named yet. The active cards are compact: hover a control or use its `i` help icon for guidance. The retained walkthrough and saved-search panel is hidden during this redesign pass, alongside the older Review & Name and identity-administration screens; no stored faces, identities, or actions are removed. Naming remains unavailable while **Read-only safety mode** is enabled.

The top-level **Names** workspace is the durable-name browser and editor. Its sidebar lists saved people globally with a name and compact face/photo count; hover a name to see a contact-sheet preview without changing the current selection. Selecting a name shows each unique photo containing a face explicitly saved with that name. Select photos with Ctrl-click, Shift-click, or their checkboxes, then right-click any selected photo for **Rename Selected** or **Unlabel Selected**. Before either action, choose the actual face-region name found in the selected files; this handles photos containing more than one named person. Selection is by photo, but every change is face-level: Rename and Unlabel change only matching regions, so other people detected in the same photo remain unchanged. The durable global mapping and open Faces views refresh after each completed action. Names does not include pending proposals or visually similar, unlabeled faces—use **Faces → Find by Name + Similar** for those. On the first global label refresh, ClusterLens also recovers legacy `manual_selected_faces` proposals created by older versions, without overwriting a conflicting durable label.

The **Photos** workspace and the main gallery offer **Edit Face Regions** when face indexing is available. In the inspector, scan or draw a box, save it, select that face region, then use **Name Selected Face(s)**, **Rename Selected**, or **Unlabel Selected**. A JPEG receives standard MWG/XMP face regions plus a ClusterLens EXIF mirror; PNG and other formats, or an image with an existing `.xmp` sidecar, use that sidecar without rewriting pixels. On re-index, an existing durable database label wins over a conflicting external XMP name.

Downloaded face and clustering models are not part of **Clear Rebuildable Caches**. On startup ClusterLens performs a local-only reconciliation before showing model selectors: it promotes a fully staged face install, restores a missing managed face bundle from its verified retained download archive when possible, and repairs a stale Hugging Face snapshot reference. It never downloads during this recovery. **Clear Model Caches** remains the explicit action that removes downloaded model files.

See `BUILD.md` for deterministic build/run/test commands and `ARCHITECTURE.md` for storage and concurrency boundaries.

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
