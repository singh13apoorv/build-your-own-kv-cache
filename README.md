# Build Your Own KV Cache

Write the generation loop for Llama 3.1 8B twice, once recomputing everything each step and once with a KV cache, and measure the difference. Then price the cache per token from the model's real tensors, measure how much memory a machine has left for it, and work out how many requests fit at different context lengths.

Everything is pinned (Python, every library, and the exact model snapshot), so anyone with this repository and a suitable machine gets the same results.

Built while working through Module 03 ("Prefill, Decode, KV Cache & Memory") of the [Fanout Inference Engineering](https://fanout.sh) course.

## Results

| Machine | Uncached | Cached | Speedup | KV budget | 4K | 32K | 128K |
|---|---|---|---|---|---|---|---|
| Apple M5 Pro, 48 GB unified memory (MPS) | 3.26 s | 2.00 s | 1.6x | 20.5 GiB | 40 req | 5 req | 1 req |
| NVIDIA RTX 4090 (CUDA) | _pending_ | _pending_ | | | | | |
| NVIDIA H100 (CUDA) | _pending_ | _pending_ | | | | | |

Llama 3.1 8B in BF16, greedy decoding, prompt "The capital of France is". Generation totals are for 32 new tokens on the Mac (256 on GPUs). Both loops produce identical text. The last three columns say how many requests fit at once if every request uses that full context length.

Full measurements and notes for each machine: [`results/memory.md`](results/memory.md).

## What the numbers show

- **The cache turns growing work into flat work.** Without a cache, every step re-processes the whole sequence, so steps get slower as the text gets longer (1.2x slower by step 32 on the Mac). With a cache, step 1 processes the prompt (the prefill), and every later step processes only the newest token, so step times stay flat.
- **The gap is small at 32 tokens and grows with length.** Each step reads all ~16 GB of weights regardless. While the sequence is short, that dominates, and re-processing a few dozen tokens adds little on top. Longer runs make the difference much larger, which is why GPU runs use 256 tokens.
- **The cache costs 131,072 bytes (128 KiB) per token:** 2 (K and V) × 32 layers × 8 KV heads × 128 head dimension × 2 bytes (BF16). It's computed from the cache tensors the model actually produces, not from the config.
- **Long contexts are expensive.** One 128K-token request needs 16 GiB of cache, about as much as the model's weights. That's why the number of requests a GPU can hold drops so fast as context grows.

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
├── results/
│   └── memory.md                       # measurements per machine
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

Each script runs one measurement in its own process and saves its output to `logs/<machine>/`. The first argument is a short name for the machine you're on. Run them one at a time, not in parallel, so they don't compete for the GPU.

```
bash run_scripts/run_uncached.sh <machine> <tokens>   # e.g. m5pro 32, h100 256
bash run_scripts/run_cached.sh   <machine> <tokens>
bash run_scripts/run_inspect.sh  <machine>
bash run_scripts/run_budget.sh   <machine> [allowance_gib]   # default 2
```

Use 32 tokens on a Mac and 256 or more on a fast GPU, where short runs hide the difference.

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