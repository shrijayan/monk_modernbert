# config.py
from dataclasses import dataclass
import datetime

@dataclass
class ModelConfig:
    model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 8192
    batch_size: int = 2
    num_epochs: int = 5
    seed: int = 42
    classification_type: str = "multi_label_classification"
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    learning_rate: float = 2e-5
    fp16: bool = True
    early_stopping_patience: int = 50
    eval_and_save_steps: int = 500
    eval_and_save_strategy: str = "steps"
    logging_steps: int = 10
    dataset_name: str = "shrijayan/medical_mimic"
    report_to: str = "wandb"
    output_dir: str = f"/content/drive/MyDrive/Projects/ModernBertTuning/{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"