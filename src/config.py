# config.py
from dataclasses import dataclass
import datetime

@dataclass
class ModelConfig:
    model_name: str = "answerdotai/ModernBERT-large"
    max_length: int = 512
    batch_size: int = 16
    num_epochs: int = 10
    seed: int = 42
    classification_type: str = "multi_label_classification"
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    learning_rate: float = 5e-5
    max_grad_norm: float = 1.0
    fp16: bool = True
    early_stopping_patience: int = 50
    eval_and_save_steps: int = 100
    eval_and_save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "combined_f1"
    greater_is_better: bool = True
    save_strategy = "best"
    logging_steps: int = 100
    dataset_name: str = "shrijayan/medical_mimic"
    save_total_limit: int = 2
    report_to: str = "wandb"
    output_dir: str = f"/content/drive/MyDrive/Projects/ModernBertTuning/{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}"
    logging_dir: str = f"{output_dir}/logs"