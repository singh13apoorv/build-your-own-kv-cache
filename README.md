# Build Your Own KV Cache

Write the generation loop for Llama 3.1 8B twice, once recomputing everything each step and once with a KV cache, and measure the difference. Then price the cache per token from the model's real tensors, measure how much memory a machine has left for it, and work out how many requests fit at different context lengths.

Everything is pinned (Python, every library, and the exact model snapshot), so anyone with this repository and a suitable machine gets the same results.

Built while working through Module 03 ("Prefill, Decode, KV Cache & Memory") of the [Fanout Inference Engineering](https://fanout.sh) course.

## Results

### Cached vs uncached generation

Total time to generate each number of new tokens, uncached / cached, with the speedup from the cache.

| Machine | 32 tokens | 256 tokens | 512 tokens |
|---|---|---|---|
| Apple M5 Pro, 48 GB unified memory (MPS) | 2.20 s / 2.03 s (1.1x) | 28.56 s / 16.59 s (1.7x) | 90.46 s / 32.44 s (**2.8x**) |
| NVIDIA RTX 4090, 24 GB (CUDA) | 0.90 s / 0.82 s (1.1x) | 7.73 s / 6.71 s (1.2x) | 21.75 s / 13.35 s (**1.6x**) |
| NVIDIA H100 SXM, 80 GB (CUDA) | 1.09 s / 2.34 s (0.5x) | 10.99 s / 10.82 s (1.0x) | 22.61 s / 20.39 s (1.1x) |

How much slower the last step is than step 2:

| Machine | Uncached at 512 tokens | Cached at 512 tokens |
|---|---|---|
| Apple M5 Pro | 63.6 → 322.5 ms (5.1x) | 63.4 → 61.7 ms (flat) |
| NVIDIA RTX 4090 | 27.6 → 73.0 ms (2.6x) | 25.3 → 26.0 ms (flat) |
| NVIDIA H100 SXM | 12.4 → 51.9 ms (4.2x) | 18.8 → 35.9 ms (not flat, see below) |

### How many requests fit

| Machine | Capacity | Weights + runtime | KV budget | 4K context | 32K context | 128K context |
|---|---|---|---|---|---|---|
| Apple M5 Pro (MPS) | 37.4 GiB | 15.0 GiB | 20.5 GiB | 40 | 5 | 1 |
| NVIDIA RTX 4090 | 23.5 GiB | 15.3 GiB | 6.2 GiB | 12 | 1 | 0 |
| NVIDIA H100 SXM | 79.2 GiB | 15.5 GiB | 61.7 GiB | 123 | 15 | 3 |

Budget = capacity − weights and runtime − a 2 GiB working-memory allowance. Each column assumes every request uses that full context length.

Llama 3.1 8B in BF16, greedy decoding, prompt "The capital of France is". Full measurements and notes for each machine: [`memory.md`](memory.md).

## What the numbers show

- **The cache turns growing work into flat work.** Without a cache, every step re-processes the whole sequence, so steps get slower as the text grows: 5.1x slower by step 512 on the Mac, 2.6x on the 4090. With a cache, step 1 processes the prompt (the prefill), and every later step processes only the newest token, so step times stay flat.
- **The advantage grows with length.** Each step reads all ~16 GB of weights regardless. At 32 tokens that dominates, and re-processing a few dozen tokens adds little, so the cache saves only about 10%. By 512 tokens it saves 2.8x on the Mac and 1.6x on the 4090.
- **On the H100, this loop is limited by the host, not the GPU.** Reading the weights takes about 5 ms per step on an H100, but steps took 12-60 ms and varied a lot. Most of each step is time outside the GPU: Python, launching hundreds of small kernels one at a time, and the cache's bookkeeping on every layer. That overhead is larger on the cached path, so the cache barely helped (1.0-1.1x) and was slower at 32 tokens. Cached decode averaged about 25 tok/s, slower than the 4090. A loop like this can't use an H100's speed. Static caches with CUDA graphs, or an inference engine, remove the overhead; that's the subject of later modules.
- **Cached and uncached text can disagree.** On the H100 the two loops matched for all 512 tokens. On the Mac they diverged after about 60 tokens ("a city of science," then "education" vs "cinema"), and on the 4090 after about 14 ("a city of history," then "a city of culture" vs "culture, and art"). Re-processing the whole sequence uses different attention kernels, which round slightly differently, and once a close call between two tokens goes the other way, everything after it differs. The cache changes the work, not the math, but floating-point details can still flip a near-tie. Across machines, the Mac's cached text matches the GPUs' for about 80 tokens ("sport" vs "sports"), and the 4090 and H100 match each other for all 512.
- **The cache costs 131,072 bytes (128 KiB) per token:** 2 (K and V) × 32 layers × 8 KV heads × 128 head dimension × 2 bytes (BF16). It's computed from the cache tensors the model actually produces, and it's the same on every machine.
- **Long contexts are expensive.** One 128K-token request needs 16 GiB of cache, about as much as the model's weights. The 4090 can't hold even one; the H100 holds three.

## What's inside

```
build-your-own-kv-cache/
├── src/build_your_own_kv_cache/
│   ├── loader.py                       # pinned model, device choice, sync(), report()
│   ├── greedy_generation_uncached.py   # re-runs the whole sequence every step
│   └── greedy_generation_cached.py     # prefill once, then one token per step
├── scripts/
│   ├── inspect_cache.py                # cache shapes and bytes per token
│   └── budget.py                       # memory budget and requests per context length
├── run_scripts/
│   ├── run_uncached.sh                 # each saves raw output to logs/<machine>/
│   ├── run_cached.sh
│   ├── run_inspect.sh
│   └── run_budget.sh
├── run_all.sh                          # detects the machine and runs everything
├── memory.md                           # measurements per machine
├── pyproject.toml
└── uv.lock                             # every dependency pinned
```

`logs/` holds raw output from each run and isn't committed.

## Requirements

- **Hardware**, one of:
  - An NVIDIA GPU with at least 24 GB of VRAM and compute capability 8.0 or higher (needed for BF16). The driver must support **CUDA 13.2 or newer** (check the top-right of `nvidia-smi`).
  - An Apple Silicon Mac with at least 32 GB of unified memory.
- [uv](https://docs.astral.sh/uv/), which installs Python 3.12 and every pinned dependency.
- About 20 GB of free disk space for the model.

## Setup

```
git clone https://github.com/singh13apoorv/build-your-own-kv-cache.git
cd build-your-own-kv-cache
uv sync
uv run hf download NousResearch/Meta-Llama-3.1-8B \
    --revision 1f47e50cdbe801ad8a5174156ec3a0655108fb9f --exclude "original/*"
```

No Hugging Face login is needed: `NousResearch/Meta-Llama-3.1-8B` is an ungated copy of Meta's base model. `--exclude "original/*"` skips a second copy of the weights that transformers doesn't use.

## Running

One command runs everything on the current machine:

```
bash run_all.sh
```

It names the machine from its hardware (`m5pro`, `rtx4090`, `h100-sxm`, and so on), records it, and runs each measurement in its own process, one after another, saving the output to `logs/<machine>/`. Pass a name to override it (`bash run_all.sh mybox`), or set `TOKENS="256"` to change the lengths (default `32 256 512`).

To run a single measurement:

```
bash run_scripts/run_uncached.sh <machine> <tokens>
bash run_scripts/run_cached.sh   <machine> <tokens>
bash run_scripts/run_inspect.sh  <machine>
bash run_scripts/run_budget.sh   <machine> [allowance_gib]   # default 2
```

Don't run two at once; they would compete for the GPU.

## How the measurements work

- **Timing.** Each step is timed between two GPU synchronizations, because GPU work runs asynchronously and the clock would otherwise stop before the work is done. Each loop runs once untimed as a warm-up before the timed run.
- **Separate processes.** The two loops never run in the same process, so memory left behind by one can't affect the other.
- **Uncached means really uncached.** That loop passes `use_cache=False`. Otherwise transformers would build a cache on every step and throw it away.
- **Budget.** Budget = memory capacity − what's already used after loading (weights plus runtime) − an allowance you choose for working memory.
  - On NVIDIA, "used" comes from `nvidia-smi` and capacity from `torch.cuda.mem_get_info()`.
  - On a Mac, capacity is `torch.mps.recommended_max_memory()`, the share of unified memory Metal allows one process. macOS and other apps also use that memory, so the Mac figures are optimistic.
- **Requests that fit** = budget ÷ (bytes per token × context length), assuming every request uses the full context. Real requests are usually shorter, so this is the worst case.

## License

The code in this repository is MIT licensed. The model weights are **not** included: they're downloaded from Hugging Face and covered by the [Llama 3.1 Community License](https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/LICENSE).