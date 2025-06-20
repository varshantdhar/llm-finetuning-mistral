# LLM Fine-tuning Mistral

Fine-tune a quantized Mistral-7B-Instruct-v0.2 model on instruction dataset using LoRA (Low-Rank Adaptation).

## Features

- LoRA (r=8) PEFT
- 4-bit quantization with bitsandbytes
- WandB logging
- 512-token input truncation
- Checkpoints saved every 100 steps

## Scaling Notes

- This setup fits on a single consumer GPU (e.g. 4090, A6000).
- For multi-GPU, replace `Trainer` with FSDP + `accelerate launch`.
- For large-scale datasets: switch to streaming HF datasets.
