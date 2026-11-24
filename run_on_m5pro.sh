for n in 32 256 512; do
  bash run_scripts/run_uncached.sh m5pro $n
  bash run_scripts/run_cached.sh   m5pro $n
done
bash run_scripts/run_inspect.sh m5pro
bash run_scripts/run_budget.sh  m5pro