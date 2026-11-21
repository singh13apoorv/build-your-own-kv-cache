import torch
from build_your_own_kv_cache.loader import load_model

def generate_cached(model, tokenizer, prompt, max_new_token=4):
    seq = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    past = None

    for _ in range(max_new_token):
        with torch.inference_mode():
            if past is None:
                #Prefil
                out = model(input_ids=seq, use_cache=True)
            else:
                #Decode
                out = model(input_ids=seq[:,-1:], past_key_values=past, use_cache=True)

        past = out.past_key_values
        next_id = out.logits[0, -1].argmax()
        if next_id == tokenizer.eos_token_id:
            break
        seq = torch.cat([seq, next_id.view(1, 1)], dim=1)

    return tokenizer.decode(seq[0], skip_special_tokens=True)

def main():
    PROMPT = "The capital of France is"
    tokenizer, model = load_model()

    generated_text = generate_cached(model, tokenizer, PROMPT)
    print(generated_text)

if __name__=="__main__":
    main()