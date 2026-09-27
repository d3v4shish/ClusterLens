#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
test_runtime="$(mktemp -d -t clusterlens-tests-XXXXXXXX)"
pycache_root="$(mktemp -d -t clusterlens-pycache-XXXXXXXX)"
settings_root="$(mktemp -d -t clusterlens-settings-XXXXXXXX)"
network_guard="$repo_root/tests/network_guard"
trap 'rm -rf -- "$test_runtime" "$pycache_root" "$settings_root"' EXIT

unset CLUSTERLENS_RUNTIME_ROOT
export IMAGE_CLUSTERING_APP_DIR="$test_runtime"
export PYTHONPYCACHEPREFIX="$pycache_root"
export XDG_CONFIG_HOME="$settings_root"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-offscreen}"
export CLUSTERLENS_TEST_NO_EXTERNAL_NETWORK=1
# Test-only sitecustomize is inherited by worker/subprocess Python commands.
# It permits loopback fixtures but rejects external DNS and connections.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="$network_guard${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_root"
if (( "$#" == 0 )); then
    set -- tests
fi
uv run --frozen python -m pytest -p no:cacheprovider "$@"
