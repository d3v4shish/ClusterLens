#!/usr/bin/env bash
# Launch source builds with the normal per-user application-data directory.
# Temporary overrides are useful for automated tests, but must not make model
# downloads disappear after a regular desktop restart.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/.." && pwd)"

for variable in CLUSTERLENS_RUNTIME_ROOT IMAGE_CLUSTERING_APP_DIR; do
    value="${!variable:-}"
    if [[ "$value" == /tmp/* ]]; then
        unset "$variable"
    fi
done

cd "$repo_root"
exec uv run --locked python -m apps.pyqt_production "$@"
