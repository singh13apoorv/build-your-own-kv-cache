# memory.md
 
Cached vs uncached generation, KV-cache price, and how many requests fit, measured per machine.
 
Llama 3.1 8B (`NousResearch/Meta-Llama-3.1-8B` @ `1f47e50`), BF16, greedy, prompt "The capital of France is".
Each loop runs in its own process with one warm-up generation before the timed one.
 
## Apple M5 Pro (MPS), measured 2026-11-24
 
```
box:          Apple M5 Pro · MPS (unified memory)
uncached 32:  3.26 s   cached 32: 2.00 s   (1.6x)   same text: yes
steps:        uncached grow 1.2x (62.0 -> 75.4 ms) · cached flat after the prefill step
              (prefill 62.4 ms, then 63.5 -> 59.8 ms)
cache/token:  131072 bytes  (2 x 32 x 8 x 128 x 2, from real shapes)
weights+rt:   15,325 MiB (≈15.0 GiB)   allowance: 2 GiB   budget: 20.5 GiB
capacity:     37.4 GiB  (torch.mps.recommended_max_memory, shared with macOS)
envelope:     4K -> 40 req · 32K -> 5 req · 128K -> 1 req
```
 
Notes:
 
- **Small contrast at 32 tokens is expected.** Each step reads all ~16 GB of weights, and that
  dominates while the sequence is short. Re-processing 6 to 37 tokens adds little on top, so uncached
  steps grow only 1.2x. The gap widens with longer runs.
- **Prefill costs about the same as one decode step here.** A 6-token prompt is still bounded by
  reading the weights once.
- **The Mac envelope is optimistic.** Capacity is the share of unified memory Metal allows one process,
  and macOS and other apps compete for it.
- **Residents on MPS** come from `torch.mps.driver_allocated_memory()`;
  `current_allocated_memory()` reports about 0 for weights loaded this way.


## NVIDIA RTX 4090 (CUDA)
 
_pending_
 
## NVIDIA H100 (CUDA)
 
_pending_
 