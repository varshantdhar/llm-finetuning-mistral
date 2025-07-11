import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from utils.pissa_utils import apply_pissa_to_model  # if using PiSSA
from config import EvalConfig
from datasets import load_dataset

def load_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.bfloat16,
        load_in_4bit=True,
        device_map="auto"
    )

    if config.use_pissa:
        apply_pissa_to_model(model, config.pissa_r)
    else:
        model = PeftModel.from_pretrained(model, config.adapter_path)

    model.eval()
    return model, tokenizer

def evaluate_sample_prompts(model, tokenizer, config):
    prompts = config.eval_prompts
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            do_sample=True,
            top_p=config.top_p,
            temperature=config.temperature,
            pad_token_id=tokenizer.eos_token_id
        )
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Prompt:\n{prompt}\n\n---\nResponse:\n{decoded}\n\n{'='*60}")

if __name__ == "__main__":
    config = EvalConfig()
    model, tokenizer = load_model(config)
    evaluate_sample_prompts(model, tokenizer, config)
