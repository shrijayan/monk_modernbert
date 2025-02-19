# src/models/modernbert.py
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel

class ModernBERTForMultilabel(PreTrainedModel):
    def __init__(self, model_name, num_severity_labels=4):
        super().__init__(AutoModelForSequenceClassification.from_pretrained(model_name).config)
        self.num_severity_labels = num_severity_labels
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_severity_labels + 1  # severity + action
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits

        # Split logits into severity and action
        severity_logits = logits[:, :self.num_severity_labels]
        action_logits = logits[:, self.num_severity_labels]

        loss = None
        if labels is not None:
            severity_labels = labels[:, 0]
            action_labels = labels[:, 1].float()

            severity_loss = nn.CrossEntropyLoss()(severity_logits, severity_labels)
            action_loss = nn.BCEWithLogitsLoss()(action_logits, action_labels)
            loss = severity_loss + action_loss

        return {"loss": loss, "severity_logits": severity_logits, "action_logits": action_logits}