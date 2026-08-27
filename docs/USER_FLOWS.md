# ClusterLens User Flows

ClusterLens has three production workspaces: Gallery, Clustering, and Faces. They use the same selected folder. The header always shows the current folder, active jobs, runtime state, and Settings.

## Keyboard shortcuts

| Shortcut | Action |
| --- | --- |
| `Ctrl+1` | Open Gallery |
| `Ctrl+2` | Open Clustering |
| `Ctrl+3` | Open Faces |
| `Ctrl+O` | Focus folder selection |
| `Ctrl+F` | Focus Faces search |
| `Ctrl+R` | Run clustering |
| `Esc` | Cancel active work |
| `Ctrl+,` | Open Settings |
| `F1` | Show keyboard help |

## Select a folder

1. Choose a folder in the Folders pane or use `Ctrl+O`.
2. Turn recursive discovery on only when nested folders should be included.
3. Confirm the resolved folder in the header. Clustering and Faces now use that same folder; changing it does not start a scan by itself.

A fresh launch with no restored folder displays a genuine no-folder state. It never scans `/home` implicitly.

## Browse and organize photos

1. Open Gallery and select a folder. Every supported photo appears immediately while thumbnails load in the background.
2. Gallery inherits recursive discovery and the primary model, similarity mode, backend, outlier policy, cache, and runtime settings from Clustering. New runs begin with SigLIP and HDBSCAN; if SigLIP is not available locally, ClusterLens asks before downloading it and otherwise uses its available local fallback. Gallery uses only the first configured comparison so the photo view remains simple.
3. Select **Organize** to arrange the whole folder. The largest groups appear first. Group headers show only photo counts and actions; they intentionally do not expose cluster IDs, models, scores, or explanations.
4. Use a group checkbox to act on several groups together. A group header can collapse, inspect, reveal, tag, copy, move, or send its photos to ClusterLens Trash.
5. Face detection, face search, and face review are available only in the Faces workspace. Gallery stays focused on browsing and organizing the folder's photos.

Photo tiles show the ClusterLens EXIF person name when available, followed by the filename. Double-click a photo or press Enter to open the basic Photo Inspector; wheel zoom, fit, previous/next navigation, and Escape-to-close are available there.

## Cluster photos

1. Open Clustering and choose Basic for a guided run or Advanced for presets and technical controls.
2. In Advanced, choose Fast Preview, Balanced, High Quality, or Custom. Presets deterministically select validated model, similarity, clustering, cache, and runtime options. Editing a preset-controlled technical option changes the preset to Custom.
3. Start clustering. Folder discovery runs as a cancellable preflight job and its snapshot is reused by the run.
4. Review the current group in the cluster list and gallery. Completion shows result count, elapsed time, and the actual runtime used.

The View menu controls Folders, clustering Controls, and Compare panes. Advanced cluster tag actions are in View > Cluster actions.

## Select and act on photos

ClusterLens uses these terms consistently:

- **Selected photos** are highlighted gallery tiles, or checked photos when no tile is highlighted.
- **Visible results** are all photos currently shown by the result view.
- **Current group** is the active cluster or result group shown beside gallery actions.
- **Selected folder** is the shared source folder shown in the header.

The gallery shows the resolved target and count. Use Tag selected photos for an explicit selection only. More actions contains Metadata, File actions, path export, and thumbnail retry commands. Copy, move, trash, EXIF, and tag actions are disabled when there is no valid target or read-only mode is enabled.

## Work with human faces

Faces is human-only in the production application. Face models, indexes, and thumbnail workers load when Faces is opened.

- **All Faces** browses the saved face library.
- **Folder Review** scans the selected folder and reviews photos and detected faces.
- **Face Search** finds similar human faces and assigns names.
- **Identities** searches saved people, reviews prototype faces, and edits an identity.

Basic mode keeps model names, thresholds, database paths, maintenance, import/export, recognition switches, and destructive data controls out of the task pages. Advanced mode exposes model and identity-management details without clearing the folder, page, query, or selection.

The saved face library is durable user data. A session face library contains the current folder review until it is intentionally added to the saved library.

## Settings

Settings is organized as General, Performance, Models, Storage, Safety & Recovery, Updates, and Support.

- General contains thumbnail and interface preferences.
- Performance contains Auto/CUDA/CPU and a performance preset. Precision, batches, workers, prefetch, and CUDA memory reserve are under Advanced performance.
- Models shows installed assets, checksum state, source, and license.
- Storage reports rebuildable data and can clear it without deleting user-authored data.
- Safety & Recovery contains read-only mode and human-readable operation recovery.
- Updates defaults to off and never changes CPU/CUDA variants silently.
- Support contains runtime paths, provider details, logs, the raw operation journal, and diagnostic verification.

Choose Cancel to discard pending settings. Choose OK to validate and apply them once.

## Recover a file operation

1. Open Settings > Safety & Recovery.
2. Select a move or ClusterLens Trash operation. Per-file successes and failures appear below the paged operation list.
3. Use Reveal target to inspect the current file, Retry failed files for incomplete work, or Restore (skip conflicts) for the safest restore.
4. If an original path is occupied, Restore with unique names previews and uses a non-overwriting path.

Recovery state is stored in SQLite before a file changes and remains available after restart. If the journal is unavailable or read-only, ClusterLens blocks the mutation. Raw SQLite and JSONL details are support tools, not part of the normal recovery flow.

## Data cleanup and support

Clear rebuildable data removes generated caches, indexes, thumbnails, temporary files, benchmarks, and support bundles. It does not remove identities, face labels, tags, settings, operation history, logs, model assets, or source photos.

Export a support bundle from Settings > Support when reporting an issue. Support bundles redact local runtime and home paths and include relevant diagnostics, logs, crash records, model inventory, and recent operation history.
