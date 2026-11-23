import argparse, time
import torch
from build_your_own_kv_cache.loader import PROMPT, load_model, sync, report

def generate_tiny(model, tokenizer, prompt, max_new_tokens=4):
    seq = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    step_times = []

    for _ in range(max_new_tokens):
        sync(); t0 = time.perf_counter()

        with torch.inference_mode():
            logits = model(input_ids=seq).logits
        next_id = logits[0, -1].argmax(-1)
        sync(); step_times.append(time.perf_counter() - t0)

        if next_id == tokenizer.eos_token_id:
            break
        seq = torch.cat([seq, next_id.view(1, 1)], dim=1)
    return tokenizer.decode(seq[0], skip_special_tokens=True), step_times

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32)
    args = parser.parse_args()

    tokenizer, model = load_model()

    generate_tiny(model, tokenizer, PROMPT, 8)         # warm-up run
    text, times = generate_tiny(model, tokenizer, PROMPT, args.tokens)
    report("uncached", text, times)

if __name__=="__main__":
    main()