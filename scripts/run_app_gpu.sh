#!/usr/bin/env bash
# Launch ClusterLens through the dedicated CUDA 12.1 source runtime.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
runtime_python="$repo_root/.venv-gpu-cu121/bin/python"

if [[ ! -x "$runtime_python" ]]; then
    printf '%s\n' "GPU runtime is not installed. Run: bash scripts/setup_gpu_runtime.sh" >&2
    exit 2
fi

for variable in CLUSTERLENS_RUNTIME_ROOT IMAGE_CLUSTERING_APP_DIR; do
    value="${!variable:-}"
    if [[ "$value" == /tmp/* ]]; then
        unset "$variable"
    fi
done

cd "$repo_root"
exec "$runtime_python" -m apps.pyqt_production "$@"
