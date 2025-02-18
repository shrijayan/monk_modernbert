# main.py
from src.train import train_multilabel_model
from src.utils.predictor import predict

if __name__ == "__main__":
    file1 = "dataset/medical_mimic(1).json"
    file2 = "dataset/medical_mimic_new_(1).json"

    print("Training Multilabel Model...")
    trainer = train_multilabel_model(file1, file2)
    print("Multilabel Model Training Completed!")

    test_text = "Your test text here"
    predictions = predict(test_text, 'multilabel_model')
    print(f"\nTest Predictions for: {test_text}")
    print(f"Severity: {predictions['severity']}")
    print(f"Immediate Action Required: {predictions['immediate_action_required']}")