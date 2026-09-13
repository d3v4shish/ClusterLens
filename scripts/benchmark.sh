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
export PYTHONPATH="$repo_root/src:$repo_root"
cd "$repo_root"
"$benchmark_python" scripts/benchmark_acceleration.py "$@"
"$benchmark_python" scripts/benchmark_face_region_metadata.py
