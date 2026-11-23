#!/usr/bin/env bash
# Stage 1, cached loop. Saves to logs/<label>/cached_<tokens>.txt
#   bash scripts/run_cached.sh m5pro 32
#   bash scripts/run_cached.sh h100 256
set -euo pipefail
LABEL=${1:?usage: run_cached.sh <label> [tokens]}
TOKENS=${2:-32}
cd "$(dirname "$0")/.."
mkdir -p "logs/$LABEL"

uv run python src/build_your_own_kv_cache/greedy_generation_cached.py --tokens "$TOKENS" \
  2>&1 | tee "logs/$LABEL/cached_$TOKENS.txt"