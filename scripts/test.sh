#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
test_runtime="$(mktemp -d -t clusterlens-tests-XXXXXXXX)"
pycache_root="$(mktemp -d -t clusterlens-pycache-XXXXXXXX)"
trap 'rm -rf -- "$test_runtime" "$pycache_root"' EXIT

unset CLUSTERLENS_RUNTIME_ROOT
export IMAGE_CLUSTERING_APP_DIR="$test_runtime"
export PYTHONPYCACHEPREFIX="$pycache_root"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
cd "$repo_root"
uv run --frozen --with pytest python -m pytest -p no:cacheprovider tests "$@"
