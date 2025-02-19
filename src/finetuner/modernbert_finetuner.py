# src/finetuner/modernbert_finetuner.py
from typing import Dict, List, Optional
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

from src.models.modernbert import ModernBERTForMultilabel
from src.trainers.multilabel_trainer import MultilabelTrainer
from src.config import ModelConfig
config = ModelConfig()

class ModernBERTMultilabelFinetuner:
    def __init__(
        self,
        model_name: str = config.model_name,
        max_length: int = 512,
        test_size: float = 0.2,
        output_dir: str = config.output_dir,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.test_size = test_size
        self.output_dir = output_dir
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, reference_compile=False)
        self.severity_mapping = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}

    def prepare_dataset(self, texts: List[str], labels: List[List]) -> DatasetDict:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=self.test_size, random_state=42
        )

        train_dataset = Dataset.from_dict({
            'text': train_texts,
            'labels': train_labels
        })
        val_dataset = Dataset.from_dict({
            'text': val_texts,
            'labels': val_labels
        })

        def tokenize(batch):
            return self.tokenizer(
                batch['text'],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

        train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=['text'])
        val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=['text'])

        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })

    def compute_metrics(self, eval_pred):
        outputs, labels = eval_pred
        severity_logits = outputs["severity_logits"]
        action_logits = outputs["action_logits"]

        severity_preds = np.argmax(severity_logits, axis=1)
        action_preds = (action_logits > 0).astype(int)

        severity_true = labels[:, 0]
        action_true = labels[:, 1]

        severity_precision, severity_recall, severity_f1, _ = precision_recall_fscore_support(
            severity_true, severity_preds, average='weighted'
        )
        severity_accuracy = accuracy_score(severity_true, severity_preds)

        action_precision, action_recall, action_f1, _ = precision_recall_fscore_support(
            action_true, action_preds, average='binary'
        )
        action_accuracy = accuracy_score(action_true, action_preds)

        return {
            'severity_accuracy': severity_accuracy,
            'severity_f1': severity_f1,
            'severity_precision': severity_precision,
            'severity_recall': severity_recall,
            'action_accuracy': action_accuracy,
            'action_f1': action_f1,
            'action_precision': action_precision,
            'action_recall': action_recall,
            'combined_f1': (severity_f1 + action_f1) / 2
        }

    def train(
        self,
        dataset: DatasetDict,
        training_args: Optional[Dict] = None,
    ) -> MultilabelTrainer:
        model = ModernBERTForMultilabel(self.model_name)

        default_args = TrainingArguments(
            output_dir=self.output_dir,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            learning_rate=5e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=10,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="combined_f1",
            warmup_ratio=0.1,
            max_grad_norm=1.0,
            logging_dir=f"{self.output_dir}/logs",
            logging_strategy="steps",
            logging_steps=100,
            report_to="wandb"
        )

        if training_args:
            for key, value in training_args.items():
                setattr(default_args, key, value)

        trainer = MultilabelTrainer(
            model=model,
            args=default_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        return trainer