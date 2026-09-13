#!/usr/bin/env bash
# Create the dedicated CUDA 12.1 source runtime without changing the CPU-default uv environment.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
runtime_dir="$repo_root/.venv-gpu-cu121"
runtime_python="$runtime_dir/bin/python"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
    printf '%s\n' "The CUDA 12.1 source runtime is supported only on Linux x86-64." >&2
    exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
    printf '%s\n' "uv is required. Install uv, then rerun this script." >&2
    exit 2
fi

if [[ ! -x "$runtime_python" ]]; then
    uv venv --python 3.12 "$runtime_dir"
fi

if [[ "$("$runtime_python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" != "3.12" ]]; then
    printf '%s\n' "Expected Python 3.12 in $runtime_dir; remove that dedicated runtime and rerun this script." >&2
    exit 2
fi

uv pip install \
    --python "$runtime_python" \
    --upgrade \
    --index-strategy unsafe-best-match \
    -r "$repo_root/packaging/requirements-build-gpu-cu121.txt"
uv pip install --python "$runtime_python" --no-deps --editable "$repo_root"
"$runtime_python" "$repo_root/scripts/verify_gpu_runtime.py"
