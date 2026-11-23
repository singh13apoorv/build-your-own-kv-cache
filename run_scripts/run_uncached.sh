#!/usr/bin/env bash
# Stage 1, uncached loop. Saves to logs/<label>/uncached_<tokens>.txt
#   bash scripts/run_uncached.sh m5pro 32
#   bash scripts/run_uncached.sh h100 256
set -euo pipefail
LABEL=${1:?usage: run_uncached.sh <label> [tokens]}
TOKENS=${2:-32}
cd "$(dirname "$0")/.."
mkdir -p "logs/$LABEL"

uv run python src/build_your_own_kv_cache/greedy_generation_uncached.py --tokens "$TOKENS" \
  2>&1 | tee "logs/$LABEL/uncached_$TOKENS.txt"