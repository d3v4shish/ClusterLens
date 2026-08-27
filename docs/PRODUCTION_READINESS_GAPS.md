# Production Readiness Gaps

These are the remaining production-grade items that should become release gates before the app is treated as a packaged end-user product.

## Release Packaging

- Build final signed installers/packages with version metadata for Windows, Linux, and macOS.
- Bundle required model assets and optional accelerators such as `hf_xet`, CUDA-relevant runtimes, ONNX Runtime providers, and any chosen Flash Attention stack where the target OS supports them.
- Define uninstall behavior for the runtime root, including whether logs, cache, tags, support bundles, and model assets are deleted or preserved.

## Validation Gates

- Make the `1920x1080` visual pass mandatory for release candidates.
- Run corrupt-image, corrupt-cache, missing-model, worker-crash, CPU fallback, and GPU fallback scenarios as automated fault tests.
- Promote soak runs from ad hoc scripts to a required release matrix across realistic folders and cache states.

## Data Safety

- Implemented: gallery copy/move/delete/EXIF operations write a durable SQLite operation journal before file mutation, with JSONL retained for compatibility export/import.
- Implemented: copy/move/trash/EXIF writes use safer temp-then-atomic-finish paths where the platform allows it, and operation temp files are cleaned in `finally` blocks.
- Implemented: the operation journal exposes reveal, refresh/retry, skip-conflict restore, and unique-name restore actions for move/delete recovery.
- Implemented: file-operation result summaries show success/failure counts and the audit log path.

## Runtime Migration

- Implemented: runtime temp files under `cache/tmp/` are reported separately in the footer and cleared automatically on production startup.
- Implemented: **Clear rebuildable data** removes rebuildable caches, runtime temp files, support bundles, benchmark artifacts, and in-memory gallery caches while preserving identities, face labels, tags, settings, operation history, logs, crash records, and model assets.
- Implemented: runtime migrations use an ordered state store, transactionally record successful versions, keep a JSON compatibility state file, and back up user-authored tag and face databases before schema work.
- Remaining: add per-database schema migrations as future cache/tag/face schemas change.
- Record runtime schema/app versions in support bundles.

## Accessibility And Polish

- Verify keyboard focus order and shortcut consistency across the production shell.
- Run high-DPI and common-resolution checks beyond `1920x1080`.
- Keep empty, loading, error, and partial-success states compact and explicit in every pane.
