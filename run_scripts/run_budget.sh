#!/usr/bin/env bash
# Stages 3 + 4, budget and envelope. Saves to logs/<label>/budget.txt
#   bash scripts/run_budget.sh m5pro
#   bash scripts/run_budget.sh h100 2     # second arg = allowance in GiB (default 2)
set -euo pipefail
LABEL=${1:?usage: run_budget.sh <label> [allowance_gib]}
ALLOWANCE_GIB=${2:-2}
cd "$(dirname "$0")/.."
mkdir -p "logs/$LABEL"

uv run python scripts/budget.py --allowance-gib "$ALLOWANCE_GIB" \
  2>&1 | tee "logs/$LABEL/budget.txt"