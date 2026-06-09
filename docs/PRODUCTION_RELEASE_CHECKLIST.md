# Production Release Checklist

This release is clustering-only. Face, search, review, and experimental flows stay outside the production PyQt app.

## Model Assets

- Export packaged ONNX assets with `scripts/export_prod_onnx_assets.py`.
- Every packaged model bundle should contain `model.onnx`, `metadata.json`, and a checksum in `sha256`.
- The top-level `model_assets/manifest.json` should use `model-assets-v1` and list source URL, license, size, and checksum.
- Production workers reject invalid packaged checksums and will not download missing models unless the UI or CLI explicitly allows it.
- The Settings `Models` tab shows packaged, cached, missing, invalid, and license/source state.

## Download Policy

- Interactive production runs ask before downloading any model file not shipped with the binary or already cached.
- Offline mode disables model downloads and forces a bundled/local fallback.
- Non-interactive benchmarks block downloads by default; use `--allow-downloads` only when measuring first-time model acquisition.

## Runtime Data

- Runtime root contains `logs/`, `cache/`, `crash/`, `benchmarks/`, `model_assets/`, and `support/`.
- Default runtime roots are platform-specific: `%LOCALAPPDATA%` on Windows, `$XDG_DATA_HOME` or `~/.local/share` on Linux, and `~/Library/Application Support` on macOS.
- Footer storage usage shows total runtime footprint.
- Clear Caches / Temp removes rebuildable/generated data but preserves tags, logs, crash records, and model assets.
- Use `scripts/cleanup_production_runtime.ps1 -RemoveAll` on Windows or `python scripts/cleanup_production_runtime.py --remove-all` cross-platform from an installer/uninstaller flow when the user asks to remove generated runtime data.
- First-run setup exposes offline model policy, read-only safety mode, runtime preference, performance profile, and worker warm-state.
- Settings `Safety` tab exposes read-only mode, the file-operation journal, and conservative restore for move/trash operations.
- Settings `Diagnostics` tab can run release gates and view recent `app.log` lines without leaving the app.

## Support and Privacy

- Support bundles redact runtime root, cache paths, log paths, crash paths, and user home paths by default.
- Exported metadata includes runtime diagnostics, offline model policy, selected model inventory, and last-run metrics.
- Users can open logs, cache, support bundles, and crash reports from Settings.

## Release Gates

Run:

```powershell
.venv\Scripts\python.exe -m apps.pyqt_production.release_gates --folder D:\Images --report-dir .\benchmarks\release_gates
```

On Linux/macOS, use the active interpreter or `.venv/bin/python` with the same module arguments.

Expected gates:

- runtime migrations complete
- logs/cache/crash/support directories are writable
- packaged model checksums are valid
- at least one bundled fallback model exists
- optional real-folder smoke benchmark passes

Packaging scaffolding:

- `packaging/production_release_manifest.json` records installer, signing, cleanup, release-gate, and updater requirements.
- `scripts/build_pyqt_binary.py` builds separate `cpu` and `gpu-cu121` PyInstaller variants in isolated build venvs. CPU is the default public artifact; GPU is a separate large package.
- `packaging/pyqt_production.spec` packages the production PyQt executable, supports one-file/one-dir modes, and embeds the checked-in production icon.
- `packaging/windows_installer.iss` builds the Windows installer and reuses the same `.ico` for setup and shortcuts.
- `apps/pyqt_production/assets/app_icon.png` and `apps/pyqt_production/assets/app_icon.ico` are the shared runtime and Windows packaging icon assets.
- `packaging/update_manifest.example.json` documents the signed-update manifest shape.
- `scripts/build_production_package.ps1` and `scripts/build_production_package.py` intentionally fail when release gates or signing inputs are missing unless unsigned local testing is explicitly requested. Linux/macOS still need platform-specific package recipes around the same release gates and runtime cleanup script.
- Model weights are not bundled by default; first-run and on-demand prompts download them into the user runtime cache.

## Remaining Manual Gates

- Run a real 1920x1080 visual pass.
- Run CPU and GPU clustering on a real folder.
- Run corrupt image and corrupt cache fault injection.
- Run a long soak pass with open/cluster/inspect/tag/file-op/cancel/retry cycles.
- Verify code signing and installer uninstall prompts in the actual packaging tool.
