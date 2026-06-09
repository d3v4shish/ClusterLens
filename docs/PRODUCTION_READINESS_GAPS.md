# Production Readiness Gaps

These are the remaining production-grade items that should become release gates before the app is treated as a packaged end-user product.

## Release Packaging

- Build final signed installers/packages with version metadata for Windows, Linux, and macOS.
- Bundle required model assets and optional accelerators such as `hf_xet`, CUDA/DirectML/MPS-relevant runtimes, ONNX Runtime providers, and any chosen Flash Attention stack where the target OS supports them.
- Define uninstall behavior for the runtime root, including whether logs, cache, tags, support bundles, and model assets are deleted or preserved.

## Validation Gates

- Make the `1920x1080` visual pass mandatory for release candidates.
- Run corrupt-image, corrupt-cache, missing-model, worker-crash, CPU fallback, and GPU fallback scenarios as automated fault tests.
- Promote soak runs from ad hoc scripts to a required release matrix across realistic folders and cache states.

## Data Safety

- Implemented: gallery copy/move/delete/EXIF operations write `logs/file_operations.jsonl` with source path, destination/trash path, timestamps, and partial-failure details.
- Implemented: copy/move/trash/EXIF writes use safer temp-then-atomic-finish paths where the platform allows it, and operation temp files are cleaned in `finally` blocks.
- Improve recovery UX for move/delete workflows so users can locate or restore files after an operation.
- Implemented: file-operation result summaries show success/failure counts and the audit log path.

## Runtime Migration

- Implemented: runtime temp files under `cache/tmp/` are reported separately in the footer and cleared automatically on production startup.
- Implemented: `Clear Caches / Temp` removes rebuildable caches, runtime temp files, support bundles, benchmark artifacts, and in-memory gallery caches while preserving tags, logs, crash records, and model assets.
- Version settings, cache, and tag database schemas.
- Add migrations for schema changes instead of relying on manual cache clears.
- Record runtime schema/app versions in support bundles.

## Rust Mirror Parity

- Complete cold native inference parity for production models.
- Remove remaining Python-worker fallback paths only when native Rust can populate equivalent caches, metrics, and failure reports.
- Keep benchmark reports explicit about native-vs-fallback execution until parity is complete.

## Accessibility And Polish

- Verify keyboard focus order and shortcut consistency across the production shell.
- Run high-DPI and common-resolution checks beyond `1920x1080`.
- Keep empty, loading, error, and partial-success states compact and explicit in every pane.
