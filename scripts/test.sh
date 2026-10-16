#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
if ! "$repo_dir/scripts/catk.sh" -c 'import pytest' 2>/dev/null; then
  export PYTHONPATH="$repo_dir/.cache/test-tools${PYTHONPATH:+:$PYTHONPATH}"
fi
cd "$repo_dir"
exec "$repo_dir/scripts/catk.sh" -m pytest "$@"
