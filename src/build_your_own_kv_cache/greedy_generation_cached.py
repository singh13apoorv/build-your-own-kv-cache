import argparse, time
import torch
from build_your_own_kv_cache.loader import PROMPT, load_model, sync, report 

def generate_cached(model, tokenizer, prompt, max_new_token=4, verbose=False):
    seq = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    past, step_times = None, []

    for _ in range(max_new_token):
        sync(); t0 = time.perf_counter()
        with torch.inference_mode():
            if past is None:
                #Prefil
                out = model(input_ids=seq, use_cache=True)
            else:
                #Decode
                out = model(input_ids=seq[:,-1:], past_key_values=past, use_cache=True)

        past = out.past_key_values
        next_id = out.logits[0, -1].argmax()
        sync(); step_times.append(time.perf_counter() - t0)

        if verbose:
            print("Past layer key shape:", past.layers[0].keys.shape)
        if next_id == tokenizer.eos_token_id:
            break
        seq = torch.cat([seq, next_id.view(1, 1)], dim=1)

    return tokenizer.decode(seq[0], skip_special_tokens=True), step_times

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32)
    args = parser.parse_args()

    tokenizer, model = load_model()

    generate_cached(model, tokenizer, PROMPT, 8)                            # warm-up, discarded
    text, times = generate_cached(model, tokenizer, PROMPT, args.tokens)
    report("cached", text, times)

if __name__=="__main__":
    main()