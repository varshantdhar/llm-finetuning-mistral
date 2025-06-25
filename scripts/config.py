from peft import LoraConfig, TaskType

class Config:
    def __init__(self):
        # Model and tokenizer
        self.model_id = "mistralai/Mistral-7B-Instruct-v0.2"
        self.use_pissa = True

        # Dataset
        self.dataset_name = "mlabonne/guanaco-llama2-1k"
        self.max_seq_length = 512

        # LoRA config
        self.lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            task_type=TaskType.CAUSAL_LM,
            lora_dropout=0.1,
            bias="none"
        )

        # Training args
        self.output_dir = f"checkpoints/mistral-guanaco-lora-pissa-{self.use_pissa}"
        self.logging_dir = "logs"
        self.per_device_train_batch_size = 1
        self.gradient_accumulation_steps = 8
        self.num_train_epochs = 3
        self.logging_steps = 10
        self.save_steps = 100
        self.learning_rate = 2e-4
        self.use_fp16 = True

        # WandB / logging
        self.report_to = "wandb"
        self.wandb_project = "mistral-lora-pissa"
