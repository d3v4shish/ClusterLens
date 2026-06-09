# ClusterLens: A Desktop Workspace for Making Sense of Large Photo Folders

Most photo folders do not fail because they have too few tools. They fail because they have too many images and no useful structure. A trip dump, a camera card, a product shoot, a backup drive, or a family archive can easily contain hundreds or thousands of files that are technically named and sorted, but not actually understandable.

ClusterLens is a desktop application built for that problem. It scans local image folders, groups visually or semantically similar photos, and gives the user a practical workspace for reviewing, comparing, tagging, and organizing the results.

It is not a gallery viewer with a clustering button bolted on. The app is built around the clustering workflow from the start: choose a source folder, pick the embedding models and clustering backends, run the job, inspect the groups, and act on the files.

## What the App Does

ClusterLens turns a folder of images into navigable clusters.

Instead of making the user manually scroll through every file, the app computes image embeddings, runs clustering algorithms, and presents the results in a PyQt desktop interface. The user can move through folders, run basic or advanced clustering, browse cluster members in a gallery, compare groups, inspect image metadata, and perform file actions such as opening folders, copying paths, moving files, deleting files, and editing tags or EXIF comments where supported.

The production interface is organized around four main areas:

- Source selection for choosing the image folder.
- Clustering controls for selecting models, backends, similarity modes, and runtime behavior.
- A gallery for reviewing the images in the selected cluster.
- A comparison rail for understanding and comparing clusters.

This makes the app useful for messy real-world image collections where chronological order or filenames are not enough.

## Local, Cache-Aware Clustering

The app is designed as a desktop-first tool. It works with local folders and keeps its runtime data in a local application directory. On Linux, the production build uses a runtime root under `~/.local/share/ClusterLens`, with separate folders for logs, cache, crash records, benchmarks, support bundles, and model assets.

That local runtime layout matters because clustering images is expensive. Model loading, preprocessing, embedding generation, thumbnail creation, and backend clustering can take time on large folders. ClusterLens reduces repeated work through persistent caches:

- Embedding results can be reused across runs.
- Thumbnail loading is lazy and cache-backed.
- Cluster result caches can avoid unnecessary recomputation.
- Warm worker mode can keep the production worker process alive between runs to avoid repeated startup and model-load costs.

In practice, this means the first run on a folder may do the heavy lifting, while later runs can become much faster when the same images and model settings are reused.

## Multiple Ways to Understand Similarity

Different image collections need different clustering strategies. A folder of screenshots, portraits, product photos, landscapes, memes, and near-duplicates may not respond well to a single one-size-fits-all model.

The production app exposes a curated set of embedding models and clustering backends. Visible model choices include options such as Fast Preview, MobileCLIP, DINO, DINOv2 Base, CLIP, OpenCLIP, SigLIP, and ResNet. The visible backend choices include cosine k-means, HDBSCAN, and graph-based clustering.

Advanced mode can compare semantic and cosine similarity spaces. Semantic clustering uses a deterministic PCA-compressed projection of the embedding space, while cosine clustering works on the full normalized embedding vectors. The app can show these as separate comparison results so users can see whether a cluster is stable across different similarity assumptions.

This is one of the app's strengths: it does not pretend that clustering is a single button with a single truth. It exposes enough of the model and backend choice to let power users compare outcomes, while still keeping a basic mode for straightforward organization work.

## Cluster Meaning, Shape, and Basis

A cluster is only useful if the user can understand why it exists.

ClusterLens includes advanced explanation panels that help translate raw clustering output into something reviewable. The production shell can show cluster meaning, cluster shape, and cluster basis information for the selected group.

Cluster Meaning is a model-derived labeling layer. It can suggest a likely theme, review guidance, and file-operation safety context. The app treats this as a sidecar interpretation rather than ground truth.

Cluster Shape helps show whether the group is tight or loose. It can expose whether the cluster has a strong representative center or a weaker tail of borderline images.

Cluster Basis gives more deterministic numeric context, such as cohesion, nearest competitor information, backend quality, similarity-space metadata, and overlap against related comparison runs.

Together, these panels make the app more than a black box. The user can inspect both the images and the reasoning signals behind the grouping.

## Built for Real File Work

The goal is not just to produce clusters. The goal is to help the user do something with them.

The gallery supports practical review and organization actions. Users can open folders, copy paths, export paths, move files, delete files, inspect metadata, and work with tags. In basic mode, the core file operations stay visible. In advanced mode, the app adds deeper EXIF, tag editing, tag management, and suggested tags based on cluster meaning.

Production file operations are also audited. Copy, move, trash, and metadata operations write to a JSONL operation log, and result dialogs include success and failure counts. Restore behavior is conservative and avoids overwriting existing original paths.

That makes the app safer for large organizing sessions where a partial failure or accidental move should be inspectable instead of silent.

## A Production Shell with a Safer Failure Boundary

The production PyQt app runs clustering work in a supervised worker process. That is important because ML workloads can fail in ways that should not bring down the entire desktop UI. Torch, ONNX Runtime, FAISS, image decoding, and model loading all live closer to the risky side of the application.

By running clustering through a worker process, the UI host can remain alive if a worker crashes. Packaged builds re-enter the same executable with a `--worker` flag, while source runs use the worker module directly. Progress, results, and errors stream back to the shell through a structured protocol.

The app also writes rotating logs, Qt diagnostics, crash records, support bundles, and benchmark reports into the runtime directory. This gives both users and developers a way to understand what happened after a long clustering run.

## Model Assets and Offline Behavior

The production app distinguishes between the application binary, ML framework dependencies, cached model downloads, and packaged model assets.

The binary can include the Python runtime and libraries such as PyQt, Torch, TorchVision, Transformers, ONNX Runtime, FAISS, and scikit-learn. Model weights are handled separately through local caches or `model_assets`. Before a production worker request runs, the shell checks whether selected models require files that are not already packaged or cached. If a download is needed, the user must explicitly approve it. Offline mode can block hidden downloads and force the run to use local or bundled assets.

This keeps model availability visible instead of surprising the user during a run.

## Who It Is For

ClusterLens is useful for people who regularly deal with large image folders:

- Photographers sorting raw imports or duplicate-heavy sessions.
- Designers reviewing asset dumps, references, screenshots, or generated images.
- Researchers inspecting visual datasets.
- Developers validating clustering models and performance profiles.
- Anyone cleaning up a local photo archive without wanting to manually review every image one by one.

It is especially useful when the problem is not just "find duplicates", but "show me what kinds of images are in this folder and help me act on those groups."

## Why It Matters

Image organization is usually presented as either manual browsing or fully automated magic. ClusterLens sits in the more useful middle ground.

It uses modern embedding models and clustering algorithms, but it keeps the user in control. It shows the groups, exposes the model choices, caches expensive work, logs file operations, and gives practical review tools in the same interface.

The result is a desktop workspace for turning image chaos into workable structure. Not by replacing human judgment, but by making the first pass through a large folder much faster, more explainable, and easier to act on.
