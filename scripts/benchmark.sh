#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
benchmark_python="${CLUSTERLENS_BENCHMARK_PYTHON:-$repo_root/.venv/bin/python}"
benchmark_runtime="$(mktemp -d -t clusterlens-benchmark-XXXXXXXX)"
trap 'rm -rf -- "$benchmark_runtime"' EXIT

if [[ ! -x "$benchmark_python" ]]; then
    printf '%s\n' "Benchmark Python is unavailable: $benchmark_python" >&2
    exit 2
fi

export CLUSTERLENS_RUNTIME_ROOT="$benchmark_runtime"
export PYTHONPYCACHEPREFIX="$benchmark_runtime/pycache"
export PYTHONPATH="$repo_root/src:$repo_root"
cd "$repo_root"
if [[ "${1:-}" == "--thumbnail-index-only" ]]; then
    shift
    "$benchmark_python" scripts/benchmark_thumbnail_index.py "$@"
    exit 0
fi
if [[ "${1:-}" == "--tag-workspace-only" ]]; then
    shift
    "$benchmark_python" scripts/benchmark_tag_workspace.py "$@"
    exit 0
fi
if [[ "${1:-}" == "--deep-face-search-only" ]]; then
    shift
    "$benchmark_python" scripts/benchmark_deep_face_search.py "$@"
    exit 0
fi
if [[ "${1:-}" == "--faces-arrangement-only" ]]; then
    shift
    "$benchmark_python" scripts/benchmark_face_tile_arrangement.py "$@"
    exit 0
fi
if [[ "${1:-}" == "--face-indexing-only" ]]; then
    shift
    "$benchmark_python" scripts/benchmark_face_indexing.py "$@"
    exit 0
fi
if [[ "${1:-}" == "--face-review-paging-only" ]]; then
    shift
    "$benchmark_python" scripts/benchmark_face_review_paging.py "$@"
    exit 0
fi
if [[ "${1:-}" == "--library-catalog-only" ]]; then
    shift
    "$benchmark_python" scripts/benchmark_library_catalog.py "$@"
    exit 0
fi
if [[ "${1:-}" == "--library-timeline-only" ]]; then
    shift
    "$benchmark_python" scripts/benchmark_library_catalog.py --timeline-only "$@"
    exit 0
fi
if [[ "${1:-}" == "--multi-root-discovery-only" ]]; then
    shift
    "$benchmark_python" scripts/benchmark_library_catalog.py --multi-root-discovery-only "$@"
    exit 0
fi
if [[ "${1:-}" == "--entity-picker-only" ]]; then
    shift
    "$benchmark_python" scripts/benchmark_entity_picker.py "$@"
    exit 0
fi
"$benchmark_python" scripts/benchmark_acceleration.py "$@"
"$benchmark_python" scripts/benchmark_face_region_metadata.py
"$benchmark_python" scripts/benchmark_thumbnail_index.py
"$benchmark_python" scripts/benchmark_tag_workspace.py
"$benchmark_python" scripts/benchmark_deep_face_search.py
"$benchmark_python" scripts/benchmark_face_tile_arrangement.py
"$benchmark_python" scripts/benchmark_face_indexing.py
"$benchmark_python" scripts/benchmark_face_review_paging.py
"$benchmark_python" scripts/benchmark_library_catalog.py
"$benchmark_python" scripts/benchmark_entity_picker.py
