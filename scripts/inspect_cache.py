import torch
from common import load_model

def run_model(tokenizer, model, prompt):
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        out = model(**enc, use_cache=True)
    return out

def main():
    PROMPT = "The capital of France is"
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

if __name__ == "__main__":
    main()