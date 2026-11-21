import torch
from build_your_own_kv_cache.loader import load_model

def generate_tiny(model, tokenizer, prompt, max_new_tokens=4):
    seq = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    for _ in range(max_new_tokens):
        with torch.inference_mode():
            logits = model(input_ids=seq).logits
        next_id = logits[0, -1].argmax(-1)
        if next_id == tokenizer.eos_token_id:
            break
        seq = torch.cat([seq, next_id.view(1, 1)], dim=1)
    return tokenizer.decode(seq[0], skip_special_tokens=True)

def main():
    PROMPT = "The capital of France is"
    tokenizer, model = load_model()

    generated_text = generate_tiny(model, tokenizer, PROMPT)
    print(generated_text)

if __name__=="__main__":
    main()