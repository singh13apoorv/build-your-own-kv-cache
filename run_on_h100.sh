for n in 32 256 512; do
  bash run_scripts/run_uncached.sh H100 $n
  bash run_scripts/run_cached.sh   H100 $n
done
bash run_scripts/run_inspect.sh H100
bash run_scripts/run_budget.sh  H100