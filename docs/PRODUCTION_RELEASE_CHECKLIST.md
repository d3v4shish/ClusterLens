# Production Release Checklist

ClusterLens releases contain the authoritative production PyQt shell, clustering, gallery operations, Recovery, and the lazy-loaded human Faces workspace. Release builds use Python 3.12 and support source environments on Python 3.11 and 3.12.

## Automated gates

Run release gates from a terminal or CI, never from production Settings:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python -m apps.pyqt_production.release_gates \
  --folder /path/to/real/images \
  --report-dir benchmarks/release_gates
```

The release report must show:

- ordered runtime and SQLite migrations passed without advancing a failed version;
- logs, cache, crash, and support targets are writable;
- every packaged model checksum and manifest entry is valid;
- a bundled offline fallback is present;
- CPU, explicit-CUDA failure, automatic fallback, and real-folder clustering checks passed for the artifact variant;
- UI/UX inventory, accessibility, paging, orientation, icon, and shutdown gates passed;
- the real-folder smoke benchmark passed.

Do not publish when the optional folder smoke is skipped. Explicit CUDA silently running on CPU is a release failure.

## UI review and snapshots

Review deterministic snapshots for these states before accepting an intentional visual update:

- Clustering: no folder, preflight, running, completed, failed, cancelled, Basic, and Advanced;
- Faces: empty and populated All Faces, Folder Review, Face Search, and Identities in Basic and Advanced;
- Settings: every category, advanced performance collapsed and open, Safety & Recovery, confirmation, and error states;
- gallery menus, Photo Inspector, runtime fallback, job indicator, and Recovery conflicts.

Capture logical 1920x1080 and 1080x1920 plus 150% and 200% DPI variants. Confirm no clipping, the result canvas remains dominant, focus indicators are visible, and dark surfaces remain consistent. Snapshot updates require reviewer sign-off and a short reason in the release report.

Run native Linux smoke tests under both X11 (`QT_QPA_PLATFORM=xcb`) and Wayland (`QT_QPA_PLATFORM=wayland`). Exercise focus order, menus, dialogs, icon/window identity, Faces lazy loading, and shutdown. Both runs must exit with zero surviving ClusterLens UI threads or worker processes.

## Models and CUDA

- Export approved ONNX assets with `scripts/export_prod_onnx_assets.py`.
- Bundle the checksum-verified `fast_preview` clustering fallback for offline runs.
- Ship the face-pack metadata catalog, but keep YuNet, SFace, SCRFD, ArcFace, AdaFace, YOLO5Face, and FaceNet weights as explicit cached installs until redistribution permission is documented. Their downloads must remain visible, cancellable, resumable, and checksum-verified.
- Every artifact has a `ModelAssetManifestV2` entry containing purpose, format, CPU/CUDA target, minimum compute capability, precision, checksum, size, source, and license.
- Package validated FP32 CPU artifacts separately from validated FP32/FP16 CUDA artifacts. Never convert an unqualified model at runtime.
- Verify the reported provider, NVIDIA device, precision, checksum, free-VRAM decision, and utilization on representative supported hardware.
- Record quality results: cosine similarity >=0.999, top-20 overlap >=98%, ARI >=0.98, face-decision agreement >=99.5%, and detection recall loss <=0.5 percentage points.

## Performance and lifecycle

- Establish a five-run median for CPU and CUDA variants.
- Fail on regressions above 15% CPU throughput, 10% CUDA throughput, 15% memory, or UI heartbeat p99 above 100 ms.
- Verify discovery runs once, remains cancellable, and its snapshot is reused.
- Verify 100,000-identity and operation-journal paging keeps initial loads bounded and UI work under 100 ms.
- Run the two-hour mixed clustering/Faces/gallery/Recovery soak. Require zero surviving threads/processes and less than 10% post-warm-up RSS drift.
- Exercise corrupt inputs, worker crash, cancellation, OOM batch reduction, read-only/full disks, and restart recovery.

## Runtime data and recovery

- `infra.settings.PRODUCTION_SETTING_SPECS` validates runtime, precision, batches, workers, caches, clustering, Faces, gallery safety, storage, and update keys.
- Generic cleanup preserves identities, face labels, tags, settings, operation history, logs, model assets, and source photos.
- The SQLite operation journal is written before a file mutation; unavailable journal storage blocks mutation.
- Test normal restore, skip conflict, unique-name restore, missing target, partial retry, read-only storage, and restart persistence.
- Uninstall defaults to preserving all user data. Package-manager removal preserves user data; explicit cleanup options remove rebuildable data or all data.

## Packages and installers

Build PyInstaller one-folder payloads inside native installers for:

- Windows x64 CPU and CUDA 12.1;
- Linux x64 CPU and CUDA 12.1;
- macOS x64 CPU and macOS ARM64 CPU.

There are no DirectML, MPS, CoreML, ROCm, or GPU-branded macOS variants.

- Windows: signed Inno Setup executable with the shared `.ico`.
- Linux: DEB and RPM with the ClusterLens desktop id and PNG icon.
- macOS: signed/notarized PKG inside a DMG.
- Verify install, upgrade, preserve-data uninstall, rebuildable cleanup, full removal, and reinstall for every package type.
- Verify the root, module, compatibility, and packaged launchers construct `apps.pyqt_production.app.ProductionClusterApp`.

The Linux desktop launcher, application window, and switcher must use the same ClusterLens mark. The checked-in runtime assets are `apps/pyqt_production/assets/app_icon.png` and `app_icon.ico`.

## Updates and signing

- Update checks default to off.
- Sign `UpdateManifestV2` and every artifact. Entries include channel, version, minimum version, OS, architecture, CPU/CUDA variant, CUDA runtime, size, SHA-256, URL, and Ed25519 signature.
- Before handoff, verify the manifest signature, artifact hash, OS signature, architecture, and current variant.
- Never switch CPU/CUDA variants silently.
- Store signing material outside the repository and record the signer identity and verification output in the release report.

## Final public-release gate

Public release requires signed artifacts, clean offline startup, bundled model integrity, real-folder clustering, human-face indexing/search/naming, Recovery, package lifecycle, and native UI smoke tests to pass. Record any intentionally accepted snapshot or benchmark baseline change in the release report.
