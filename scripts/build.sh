#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"
pycache_root="$(mktemp -d -t clusterlens-build-XXXXXXXX)"
trap 'rm -rf -- "$pycache_root"' EXIT

cd "$repo_root"
uv sync --frozen
PYTHONPYCACHEPREFIX="$pycache_root" uv run --frozen python -m compileall -q apps src
