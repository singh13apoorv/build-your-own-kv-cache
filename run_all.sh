#!/usr/bin/env bash
# Run every measurement on this machine, with the label detected automatically.
#
#   bash run_scripts/run_all.sh              # auto label, tokens 32 256 512
#   bash run_scripts/run_all.sh h100-sxm     # force a label
#   TOKENS="256" bash run_scripts/run_all.sh # override token lengths
#
# Labels: Apple M5 Pro -> m5pro, RTX 4090 -> rtx4090,
#         H100 -> h100-sxm / h100-pcie / h100-nvl, anything else -> slug of the GPU name.
set -euo pipefail
cd "$(dirname "$0")"

detect_label() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    local name
    name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    case "$name" in
      *H100*NVL*)   echo "h100-nvl" ;;
      *H100*PCIe*)  echo "h100-pcie" ;;
      *H100*HBM3*)  echo "h100-sxm" ;;
      *)            # "NVIDIA GeForce RTX 4090" -> "rtx4090"
        echo "$name" | tr '[:upper:]' '[:lower:]' \
          | sed -e 's/nvidia//g' -e 's/geforce//g' -e 's/[^a-z0-9]//g' ;;
    esac
  elif [[ "$(uname)" == "Darwin" ]]; then
    # "Apple M5 Pro" -> "m5pro"
    sysctl -n machdep.cpu.brand_string | tr '[:upper:]' '[:lower:]' \
      | sed -e 's/apple//g' -e 's/[^a-z0-9]//g'
  else
    echo "cpu"
  fi
}

LABEL=${1:-$(detect_label)}
TOKENS=${TOKENS:-"32 256 512"}
mkdir -p "logs/$LABEL"
echo "=== machine label: $LABEL   tokens: $TOKENS"

# Record the hardware once.
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi > "logs/$LABEL/nvidia-smi.txt"
else
  { sysctl -n machdep.cpu.brand_string
    echo "memory: $(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 )) GB"; } > "logs/$LABEL/machine.txt"
fi

# One process per measurement, one after another.
for n in $TOKENS; do
  bash run_scripts/run_uncached.sh "$LABEL" "$n"
  bash run_scripts/run_cached.sh   "$LABEL" "$n"
done
bash run_scripts/run_inspect.sh "$LABEL"
bash run_scripts/run_budget.sh  "$LABEL"

echo "=== done -> logs/$LABEL"
ls -1 "logs/$LABEL"