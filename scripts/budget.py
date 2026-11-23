"""
Stages 3 + 4: measure the box's KV budget, then the request envelope.

    uv run python scripts/budget.py --allowance-gib 2

CUDA: capacity from torch.cuda.mem_get_info(), residents from nvidia-smi.
MPS:  capacity = torch.mps.recommended_max_memory() (the share of unified memory
      Metal lets one process use), residents = torch.mps.driver_allocated_memory().
"""

import argparse
import subprocess

import torch

from build_your_own_kv_cache.loader import DEVICE, PROMPT, load_model

GiB = 1024**3
MiB = 1024**2


def nvidia_smi_used_mib():
    line = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    ).decode().splitlines()[0]   # First GPU
    return int(line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allowance-gib", type=float, default=2.0,
                        help="Workspace you reserve for activations etc.")
    args = parser.parse_args()
    assert DEVICE in ("cuda", "mps"), "budget.py needs an NVIDIA GPU or Apple Silicon."

    tokenizer, model = load_model()
    allowance = args.allowance_gib * GiB

    # Stage 3: residents, measured right after load.
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        used = nvidia_smi_used_mib() * MiB         # Weights + CUDA context + allocator
        free, capacity = torch.cuda.mem_get_info() # From inside the processor

        print(f"capacity:        {capacity / GiB:6.1f} GiB")
        print(f"weights+runtime: {used / MiB:,.0f} MiB ({used / GiB:.1f} GiB)  [nvidia-smi]")
        print(f"free (torch):    {free / GiB:6.1f} GiB   "
              f"capacity - used = {(capacity - used) / GiB:.1f} GiB")
        print(f"  (weights alone, torch.cuda.memory_allocated: "
              f"{torch.cuda.memory_allocated() / GiB:.2f} GiB)")
    else:
        torch.mps.synchronize()
        used = torch.mps.driver_allocated_memory()      # Everything Metal holds for us
        capacity = torch.mps.recommended_max_memory()

        print(f"capacity:        {capacity / GiB:6.1f} GiB  "
              f"[recommended_max_memory, shared with macOS]")
        print(f"weights+runtime: {used / MiB:,.0f} MiB ({used / GiB:.1f} GiB)  "
              f"[driver_allocated_memory]")

    budget = capacity - used - allowance
    print(f"allowance:       {args.allowance_gib:6.1f} GiB")
    print(f"KV budget:       {budget / GiB:6.1f} GiB")

    # Per-token price from live tensors (same chain as inspect_cache.py).
    enc = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model(**enc, use_cache=True)
    k = out.past_key_values.layers[0].keys  # [batch, kv_heads, seq, head_dim]
    per_token = 2 * len(out.past_key_values.layers) * k.shape[1] * k.shape[3] * k.element_size()
    print(f"cache/token:     {per_token} bytes")

    # Stage 4: how many full-length requests fit.
    for ctx in (4096, 32768, model.config.max_position_embeddings):
        per_request = per_token * ctx
        print(f"  {ctx // 1024:>4}K context: {per_request / GiB:6.2f} GiB/request "
              f"-> {int(budget // per_request)} requests")

if __name__ == "__main__":
    main()