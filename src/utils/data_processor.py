# src/utils/data_processor.py
import json
from typing import List, Tuple

def load_and_preprocess_data(file1: str, file2: str) -> Tuple[List[str], List[List[int]]]:
    with open(file1, 'r', encoding='utf-8') as f:
        data1 = json.load(f)
    with open(file2, 'r', encoding='utf-8') as f:
        data2 = json.load(f)

    data = data1 + data2
    severity_mapping = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}
    texts = []
    labels = []

    for entry in data:
        texts.append(entry['raw_data']['text'])
        severity_label = severity_mapping[entry['parsed_data']['severity']]
        action_label = int(entry['parsed_data']['immediate_action_required'])
        labels.append([severity_label, action_label])

    return texts, labels