import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import os

def load_model(model_path, adapter_path=None, use_4bit=True):
    print(f"Loading base model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        load_in_4bit=use_4bit
    )

    if adapter_path:
        print(f"Loading adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model.merge_and_unload()  # Optional: merge for faster inference

    return model

def generate_response(model, tokenizer, prompt, max_new_tokens=128, temperature=0.7, top_p=0.9):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device) # returns a dictionary of tensors with keys: input_ids, attention_mask, token_type_ids
    output = model.generate(
        **inputs, # unpacks the dictionary into keyword arguments
        max_new_tokens=max_new_tokens, # max number of tokens to generate
        do_sample=True, # sample from the model
        temperature=temperature, # temperature for sampling, where 0 is greedy, 1 is random
        top_p=top_p, # top-p sampling, where 0 is greedy, 1 is random
        pad_token_id=tokenizer.eos_token_id # pad token id, which is the token id of the end of sequence token
    )
    return tokenizer.decode(output[0], skip_special_tokens=True) # decode the output tensor into a string, skipping special tokens

def format_instruction_prompt(instruction, input_text=None):
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:\n"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text from a fine-tuned Mistral model.")
    parser.add_argument("--instruction", type=str, required=True, help="Instruction prompt")
    parser.add_argument("--input", type=str, help="Optional input text")
    parser.add_argument("--model_path", type=str, default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--adapter_path", type=str, help="Optional LoRA or PiSSA adapter path")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token # set the pad token to the end of sequence token

    model = load_model(args.model_path, args.adapter_path)
    prompt = format_instruction_prompt(args.instruction, args.input)

    response = generate_response(model, tokenizer, prompt, args.max_new_tokens, args.temperature, args.top_p)

    print("\n=== Generated Response ===\n")
    print(response)