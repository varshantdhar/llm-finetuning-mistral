import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import os

# Load the instruction dataset from HuggingFace
# Using Guanaco dataset which contains instruction-response pairs
print("Loading dataset...")
dataset = load_dataset("mlabonne/guanaco-llama2-1k")["train"]

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

# Apply prompt formatting to the entire dataset
print("Formatting prompts...")
dataset = dataset.map(format_prompt)

# Load the Mistral-7B-Instruct-v0.2 model and tokenizer
print("Loading model and tokenizer...")
model_id = "mistralai/Mistral-7B-Instruct-v0.2"

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
# LoRA adds small trainable matrices to the model instead of training all parameters
print("Setting up LoRA configuration...")
peft_config = LoraConfig(
    r=8,  # Rank of the LoRA matrices (higher = more parameters, better performance)
    lora_alpha=32,  # Scaling factor for LoRA weights
    task_type=TaskType.CAUSAL_LM,  # Task type for causal language modeling
    lora_dropout=0.1,  # Dropout rate for LoRA layers
    bias="none"  # Don't train bias terms
)

# Apply LoRA configuration to the model
model = get_peft_model(model, peft_config)


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
        truncation=True,  # Truncate sequences longer than max_length
        padding="max_length",  # Pad sequences to max_length
        max_length=512  # Maximum sequence length
    )
    # For language modeling, labels are the same as input_ids
    result["labels"] = result["input_ids"].copy()
    return result

# Tokenize the entire dataset
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize, batched=True)

# Configure training arguments
print("Setting up training arguments...")
args = TrainingArguments(
    output_dir="checkpoints",  # Directory to save model checkpoints
    per_device_train_batch_size=1,  # Batch size per device (small for memory constraints)
    gradient_accumulation_steps=8,  # Accumulate gradients over 8 steps (effective batch size = 8)
    num_train_epochs=3,  # Number of training epochs
    logging_dir="logs",  # Directory for training logs
    logging_steps=10,  # Log every 10 steps
    save_steps=100,  # Save checkpoint every 100 steps
    learning_rate=2e-4,  # Learning rate for optimization
    fp16=True,  # Use mixed precision training for speed and memory efficiency
    report_to="wandb",  # Log metrics to Weights & Biases
)

# Initialize the trainer with model, data, and configuration
print("Initializing trainer...")
trainer = Trainer(
    model=model,  # The LoRA-fine-tuned model
    args=args,  # Training arguments
    train_dataset=tokenized_dataset,  # Tokenized training dataset
    tokenizer=tokenizer,  # Tokenizer for text processing
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)  # Data collator for language modeling (not masked LM)
)


trainer.train()