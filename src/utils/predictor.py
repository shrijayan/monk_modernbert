# src/utils/predictor.py
import torch
from transformers import AutoTokenizer
from src.models.modernbert import ModernBERTForMultilabel

def predict(text: str, model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
    model = ModernBERTForMultilabel(model_dir)
    model.load_state_dict(torch.load(f"{model_dir}/pytorch_model.bin"))
    model.eval()

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**inputs)
        severity_logits = outputs["severity_logits"]
        action_logits = outputs["action_logits"]

        severity_pred = severity_logits[0].argmax().item()
        action_pred = (action_logits[0] > 0).item()

    severity_mapping = {0: 'low', 1: 'medium', 2: 'high', 3: 'critical'}
    severity_label = severity_mapping[severity_pred]

    return {
        'severity': severity_label,
        'immediate_action_required': bool(action_pred)
    }