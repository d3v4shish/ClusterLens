#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
gpu_python="$repo_root/.venv-gpu-cu121/bin/python"

# The dedicated CUDA environment is opt-in at setup time, but once present it
# is the canonical desktop launcher.  Do not create it here: a normal launch
# must never install packages or cause models to be downloaded again.
if [[ -x "$gpu_python" ]]; then
    printf '%s\n' "ClusterLens: using the installed dedicated CUDA runtime."
    printf '%s\n' "ClusterLens: existing model files stay in the shared per-user runtime; this launch does not download models."
    exec bash "$script_dir/run_app_gpu.sh" "$@"
fi

printf '%s\n' "ClusterLens: dedicated CUDA runtime is not installed; using the CPU-compatible source runtime."
printf '%s\n' "ClusterLens: Settings > Support > Rescan GPU Resources explains the resources detected by this process."
exec bash "$script_dir/run_app.sh" "$@"
