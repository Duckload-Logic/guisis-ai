import os
import sys
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.config import settings
from src.utils.text_cleaning import anonymize_text
from scripts.evaluate_model import _load_model

def main():
    model_ref = settings.model_path
    tokenizer, model, device, label_mapping, _ = _load_model(model_ref)

    label2id = label_mapping.get(
        "label2id", {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    )
    id2label = {v: k for k, v in label2id.items()}

    data_path = "ai_models/distilbert/datasets/labeled_dataset.csv"
    df = pd.read_csv(data_path)
    df["label"] = df["urgency"].map(label2id)
    df = df.dropna(subset=['label', 'text'])

    _, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    texts = val_df["text"].fillna("").astype(str).map(anonymize_text).tolist()
    orig_texts = val_df["text"].tolist()
    true_urgencies = val_df["urgency"].tolist()

    print("Analyzing validation errors between MEDIUM and HIGH...\n")
    batch_size = 16

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_orig = orig_texts[i:i+batch_size]
        batch_labels = true_urgencies[i:i+batch_size]

        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            confidences, preds = torch.max(probs, dim=-1)

            confidences = confidences.cpu().numpy()
            preds = preds.cpu().numpy()

            for j in range(len(batch_texts)):
                pred_label = id2label[preds[j]]
                true_label = batch_labels[j]
                conf = confidences[j]

                # Focus on HIGH <-> MEDIUM confusion
                if (true_label == "HIGH" and pred_label == "MEDIUM") or \
                   (true_label == "MEDIUM" and pred_label == "HIGH"):
                    print(f"[{true_label} -> {pred_label}] Conf: {conf:.2%}")
                    print(f"  Text: {batch_orig[j]}")
                    print("-" * 50)

if __name__ == "__main__":
    main()
