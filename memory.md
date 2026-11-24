# memory.md

Cached vs uncached generation, KV-cache price, and how many requests fit, measured per machine.

Llama 3.1 8B (`NousResearch/Meta-Llama-3.1-8B` @ `1f47e50`), BF16, greedy, prompt "The capital of France is".
Each loop runs in its own process, with one warm-up generation before the timed one. Times are totals
for 32, 256 and 512 new tokens. Steps are "step 2 → last step" (step 1 is the prefill).
All runs measured 2026-11-24.

## Apple M5 Pro, 48 GB unified memory (MPS)

```
box:          Apple M5 Pro · MPS · 48 GB unified memory
uncached 32:   2.20 s   cached 32:   2.03 s   (1.1x)   same text: yes
uncached 256: 28.56 s   cached 256: 16.59 s   (1.7x)   same text: no (diverge after "a city of science,")
uncached 512: 90.46 s   cached 512: 32.44 s   (2.8x)   same text: no (same point)
steps:        uncached grow · 32: 61.8 -> 76.1 ms (1.2x) · 256: 62.4 -> 177.7 ms (2.8x) · 512: 63.6 -> 322.5 ms (5.1x)
              cached flat after the prefill step (prefill ~62 ms, then ~62-67 ms at every length)
cache/token:  131072 bytes  (2 x 32 x 8 x 128 x 2, from real shapes)
weights+rt:   15,325 MiB (≈15.0 GiB)   allowance: 2 GiB   budget: 20.5 GiB
capacity:     37.4 GiB of 48 GB  (torch.mps.recommended_max_memory, shared with macOS)
envelope:     4K -> 40 req · 32K -> 5 req · 128K -> 1 req
```

- **The textbook result.** Uncached steps grow with length (5.1x slower by step 512), cached steps stay
  flat, and the cache's advantage grows from 1.1x at 32 tokens to 2.8x at 512.
- **Cached decode is about 16 tok/s** (≈63 ms per step), matching Module 01's 17.2 tok/s on this Mac.
- **The loops agree for about 60 tokens, then diverge.** After "a city of science,", the cached loop
  continues "a city of education" and the uncached loop "a city of cinema". It's a late near-tie flip,
  like the one the lab describes; 32-token runs end before it, so they match.
- **The envelope is optimistic.** Capacity is the share of unified memory Metal allows one process,
  and macOS and other apps compete for it.
- **Residents on MPS** come from `torch.mps.driver_allocated_memory()`;
  `current_allocated_memory()` reports about 0 for weights loaded this way.

## NVIDIA RTX 4090, 24 GB (CUDA)

```
box:          NVIDIA GeForce RTX 4090 · 24 GB · Vast.ai (CUDA 13.2)
uncached 32:   0.90 s   cached 32:   0.82 s   (1.1x)   same text: no (diverge after "a city of history,")
uncached 256:  7.73 s   cached 256:  6.71 s   (1.2x)   same text: no (same point)
uncached 512: 21.75 s   cached 512: 13.35 s   (1.6x)   same text: no (same point)
steps:        uncached grow · 32: 27.7 -> 28.0 ms (1.0x) · 256: 27.3 -> 42.8 ms (1.6x) · 512: 27.6 -> 73.0 ms (2.6x)
              cached flat after the prefill step (prefill ~27 ms, then ~25-26 ms at every length)
cache/token:  131072 bytes  (2 x 32 x 8 x 128 x 2, from real shapes)
weights+rt:   15,716 MiB (≈15.3 GiB, nvidia-smi)   weights alone: 14.96 GiB   allowance: 2 GiB   budget: 6.2 GiB
capacity:     23.5 GiB  (torch.cuda.mem_get_info)
envelope:     4K -> 12 req · 32K -> 1 req · 128K -> 0 req
```

- **The two loops diverge early on this GPU.** Right after "a city of history,", the cached loop
  continues "a city of culture…" and the uncached loop continues "culture, and art…". The cached text
  matches the Mac, the H100 and Module 01's greedy output, so the uncached loop flipped a near-tie.
  The 4090's cached text is identical to the H100's for all 512 tokens.
  Re-processing the whole sequence goes through different attention kernels, with slightly different
  rounding, and one close call goes the other way. Everything after that point differs.
- **Steps behave as expected.** Uncached steps grow and cached steps stay flat. Cached decode is about
  38 tok/s (≈26 ms per step), matching Module 01's 38.7 tok/s on a 4090.
- **24 GB can't hold one maximum-length request.** After 15.3 GiB of weights and runtime plus a 2 GiB
  allowance, 6.2 GiB is left, and a 128K request needs 16 GiB.

## NVIDIA H100 SXM, 80 GB HBM3 (CUDA)

```
box:          NVIDIA H100 80GB HBM3 (SXM) · Vast.ai · driver 615.71.09, CUDA 13.4
uncached 32:   1.09 s   cached 32:   2.34 s   (0.5x)   same text: yes
uncached 256: 10.99 s   cached 256: 10.82 s   (1.0x)   same text: yes
uncached 512: 22.61 s   cached 512: 20.39 s   (1.1x)   same text: yes
steps:        uncached grow · 32: 12.3 -> 40.2 ms (3.3x) · 256: 12.2 -> 43.3 ms (3.5x) · 512: 12.4 -> 51.9 ms (4.2x)
              cached NOT flat · prefill ~20 ms · 32: 18.4 -> 60.2 ms · 256: 18.8 -> 36.0 ms · 512: 18.8 -> 35.9 ms
cache/token:  131072 bytes  (2 x 32 x 8 x 128 x 2, from real shapes)
weights+rt:   15,847 MiB (≈15.5 GiB, nvidia-smi)   weights alone: 14.96 GiB   allowance: 2 GiB   budget: 61.7 GiB
capacity:     79.2 GiB  (torch.cuda.mem_get_info)
envelope:     4K -> 123 req · 32K -> 15 req · 128K -> 3 req
```

- **The Stage 1 gate fails on this box.** The cache barely helps (1.0-1.1x at 256-512 tokens), and the
  cached loop is slower at 32 tokens. Cached steps don't stay flat either.
- **The timings are dominated by work outside the GPU, and they're noisy.** Reading the weights takes
  about 5 ms per step on an H100 (16 GB ÷ 3,350 GB/s), yet even the fastest steps take 12-19 ms. The
  totals also don't match the per-step numbers: cached 256 averaged 42 ms per step while its step 2
  took 19 ms. On the 4090 the average and the individual steps agree (26 ms vs 25 ms). So most of each
  H100 step is host-side time: Python, launching hundreds of small kernels one by one, and the cache's
  own bookkeeping on every layer. That cost is higher on the cached path (its prefill takes 20 ms vs
  12 ms uncached for the same 6 tokens), so the saving from not re-processing old tokens doesn't show.
  The host was a 24-vCPU share of a larger server, which likely added the noise.
- **This measures this loop on this host, not the H100's limit.** Cached decode averaged about 25 tok/s,
  slower than the 4090. Removing the host overhead (a static cache with CUDA graphs, `torch.compile`,
  or an inference engine) is what would let the card's bandwidth show.
- **The memory result is unaffected.** 61.7 GiB of budget fits 123 requests at 4K, 15 at 32K and 3 at
  128K, in line with the course's representative H100 envelope (111 / 13 / 3).