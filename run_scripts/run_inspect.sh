#!/usr/bin/env bash
# Stage 2, cache shapes and bytes per token. Saves to logs/<label>/inspect.txt
#   bash scripts/run_inspect.sh m5pro
set -euo pipefail
LABEL=${1:?usage: run_inspect.sh <label>}
cd "$(dirname "$0")/.."
mkdir -p "logs/$LABEL"

uv run python scripts/inspect_cache.py \
  2>&1 | tee "logs/$LABEL/inspect.txt"