import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
from scripts.pissa_utils import apply_pissa_to_model
from scripts.config import Config

import os

# Instantiate config
config = Config()

def format_prompt(example):
    """
    Format each dataset example into a structured prompt for instruction fine-tuning.
    
    Args:
        example (dict): Dictionary containing 'instruction' and 'output' keys
        
    Returns:
        dict: Formatted example with 'text' key containing the structured prompt
    """
    return {
        "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
    }

# Load the instruction dataset from HuggingFace
# Using Guanaco dataset which contains instruction-response pairs
dataset = load_dataset(config.dataset_name)["train"]
# Toggle to enable PiSSA adapter initialization
use_pissa = config.use_pissa

# Apply prompt formatting to the entire dataset
dataset = dataset.map(format_prompt)

# Load the Mistral-7B-Instruct-v0.2 model and tokenizer
model_id = config.model_id

# Load tokenizer with fast tokenizer for better performance
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
# Set padding token to EOS token for training compatibility
tokenizer.pad_token = tokenizer.eos_token

# Load the model with 4-bit quantization for memory efficiency
# This allows training on consumer GPUs with limited VRAM
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,  # Enable 4-bit quantization
    device_map="auto",  # Automatically handle device placement
    torch_dtype=torch.bfloat16  # Use bfloat16 for better numerical stability
)

# Configure LoRA (Low-Rank Adaptation) for efficient fine-tuning
peft_config = config.lora_config

# Apply LoRA configuration to the model
model = get_peft_model(model, peft_config)

if use_pissa:
    print("Applying PiSSA initialization...")
    apply_pissa_to_model(model, rank=peft_config.r)
else:
    print("Using default random LoRA initialization.")


def tokenize(example):
    """
    Tokenize the formatted text data for training.
    
    Args:
        example (dict): Dictionary containing 'text' key with formatted prompt
        
    Returns:
        dict: Tokenized example with 'input_ids', 'attention_mask', and 'labels'
    """
    result = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=config.max_seq_length
    )
    # For language modeling, labels are the same as input_ids
    result["labels"] = result["input_ids"].copy()
    return result

# Tokenize the entire dataset
tokenized_dataset = dataset.map(tokenize, batched=True)

# Configure training arguments
args = TrainingArguments(
    output_dir=config.output_dir,
    per_device_train_batch_size=config.per_device_train_batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    num_train_epochs=config.num_train_epochs,
    logging_dir=config.logging_dir,
    logging_steps=config.logging_steps,
    save_steps=config.save_steps,
    learning_rate=config.learning_rate,
    fp16=config.use_fp16,
    report_to=config.report_to,
)

# Initialize the trainer with model, data, and configuration
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()

output_dir = config.output_dir

# Save the model and tokenizer
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model + tokenizer saved to: {output_dir}")
