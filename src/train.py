# src/train.py
from src.utils.data_processor import load_and_preprocess_data
from src.finetuner.modernbert_finetuner import ModernBERTMultilabelFinetuner

def train_multilabel_model(file1: str, file2: str):
    texts, labels = load_and_preprocess_data(file1, file2)

    multilabel_finetuner = ModernBERTMultilabelFinetuner(
        model_name="answerdotai/ModernBERT-base",
        output_dir='modernbert-multilabel'
    )

    dataset = multilabel_finetuner.prepare_dataset(texts, labels)
    trainer = multilabel_finetuner.train(dataset=dataset)
    trainer.save_model('multilabel_model')
    return trainer