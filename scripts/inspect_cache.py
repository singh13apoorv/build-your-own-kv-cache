import torch
from build_your_own_kv_cache.loader import load_model, PROMPT

def run_model(tokenizer, model, prompt):
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model(**enc, use_cache=True)
    return out

def main():
    tokenizer, model = load_model()
    out = run_model(tokenizer, model, PROMPT)

    cache = out.past_key_values

    k0 = cache.layers[0].keys
    print(type(cache).__name__, len(cache.layers), " layers")
    print("1st layer key shape and data type: ", end="")
    print(k0.shape, k0.dtype)

    v0 = cache.layers[0].values
    print("1st layer value shape and data type: ", end="")
    print(v0.shape, v0.dtype)

    per_token = 2 * len(cache.layers) * k0.shape[1] * k0.shape[3] * k0.element_size()
    print(f"cache per token: 2 x {len(cache.layers)} layers x {k0.shape[1]} kv_heads "
        f"x {k0.shape[3]} head_dim x {k0.element_size()} bytes = {per_token} bytes")

if __name__ == "__main__":
    main()