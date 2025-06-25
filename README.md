# 🧠 LLM Fine-Tuning with Mistral + LoRA + PiSSA

Fine-tune a 4-bit quantized [`Mistral-7B-Instruct-v0.2`](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) model on instruction-tuned datasets using [LoRA](https://arxiv.org/abs/2106.09685), with optional [PiSSA](https://arxiv.org/abs/2404.19753) initialization for improved low-rank adapter performance.



## Features

- LoRA (r=8) PEFT adapters  
- PiSSA adapter initialization via SVD  
- 4-bit quantization (via `bitsandbytes`)  
- Instruction-response formatting (`guanaco-llama2-1k`)  
- Inference CLI  
- ROUGE-based evaluation  
- WandB logging  
- Checkpointing every 100 steps  
- Trains on consumer GPUs with 24GB VRAM (4090, A6000)  



## Setup

Install dependencies using Pipenv:

```bash
pipenv install
pipenv shell
```

> Make sure you have Python ≥ 3.10 and CUDA 11.8+ available.

## Training

Train with LoRA (random init):

```bash
python train.py
```

Train with PiSSA initialization:

```python
# Inside train.py
use_pissa = True
```

> Model checkpoints are saved to `checkpoints/mistral-guanaco-lora-pissa-{True|False}`.


## Project Structure

```
.
├── train.py                  # Training loop
├── Pipfile / Pipfile.lock    # Dependency manager (Pipenv)
├── scripts/
│   ├── config.py             # Central config (model, LoRA, training)
│   ├── evaluate.py           # Evaluation pipeline (ROUGE)
│   └── pissa_utils.py        # SVD-based adapter init (PiSSA)
├── inference/
│   └── generate.py           # CLI for inference
├── checkpoints/              # Saved adapter weights
└── logs/                     # Training logs (WandB or local)
```



## About PiSSA

PiSSA (Principal Singular Spectrum Alignment) uses SVD to extract structure from frozen weight matrices in attention projections (`q_proj`, `k_proj`, etc). These structured low-rank adapters:
- Improve generalization over random-initialized LoRA  
- Reduce quantization error by aligning activations  
- Require no additional training time  

> Toggle `use_pissa = True` in `train.py` to enable.


## Scaling Tips

| Use Case         | Tip                                   |
|------------------|----------------------------------------|
| Multi-GPU        | Use `accelerate` + FSDP               |
| Larger datasets  | Use streaming HF datasets             |
| Cost tracking    | Use `wandb` or local JSON logs        |
| Quant stability  | Use `bfloat16` over `float16`         |



## To Do

## Inference

Run from saved adapter:

```bash
python inference/generate.py
```

You'll be prompted to enter an instruction and receive a response.


## Evaluation

Evaluate ROUGE scores using:

```bash
python scripts/evaluate.py
```

- Compares model outputs vs. references  
- Useful for testing PiSSA vs. Random initialization  

## Other items

- [ ] Add PiSSA vs Random cosine similarity comparisons  
- [ ] Visualize adapter norms and projection alignment  
- [ ] Add OpenAI Eval JSON format for benchmark uploads  


