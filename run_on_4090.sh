for n in 32 256 512; do
  bash run_scripts/run_uncached.sh rtx4090 $n
  bash run_scripts/run_cached.sh   rtx4090 $n
done
bash run_scripts/run_inspect.sh rtx4090
bash run_scripts/run_budget.sh  rtx4090