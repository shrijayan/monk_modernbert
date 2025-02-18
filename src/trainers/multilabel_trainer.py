# src/trainers/multilabel_trainer.py
from transformers import Trainer
import torch

class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs["loss"]

        if num_items_in_batch is not None:
            loss = loss * (inputs["labels"].size(0) / num_items_in_batch)

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad():
            outputs = model(**inputs)

        return (
            outputs["loss"],
            {
                "severity_logits": outputs["severity_logits"],
                "action_logits": outputs["action_logits"]
            },
            inputs["labels"]
        )