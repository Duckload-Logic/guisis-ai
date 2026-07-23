import os
import sys
import pandas as pd
import torch
import torch.nn.functional as F

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.config import settings
from src.utils.text_cleaning import anonymize_text
from scripts.evaluate_model import _load_model

def main():
    model_ref = settings.model_path
    print(f"[NoiseDetector] Loading model from: {model_ref}")
    tokenizer, model, device, label_mapping, _ = _load_model(model_ref)

    label2id = label_mapping.get(
        "label2id", {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    )
    id2label = {v: k for k, v in label2id.items()}

    data_path = "ai_models/distilbert/datasets/labeled_dataset.csv"
    df = pd.read_csv(data_path)

    texts = df["text"].fillna("").astype(str).map(anonymize_text).tolist()
    orig_texts = df["text"].tolist()
    labels = df["urgency"].tolist()

    suspect_rows = []
    batch_size = 16

    print(f"[NoiseDetector] Scanning {len(texts)} samples for label noise...")

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_orig = orig_texts[i:i+batch_size]
        batch_labels = labels[i:i+batch_size]

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

                # Highly confident mismatch
                if pred_label != true_label and conf > 0.80:
                    row_idx = i + j
                    suspect_rows.append({
                        "index": row_idx,
                        "text": batch_orig[j],
                        "true_label": true_label,
                        "pred_label": pred_label,
                        "confidence": f"{conf:.2%}"
                    })

    if suspect_rows:
        suspect_df = pd.DataFrame(suspect_rows)
        out_path = "ai_models/distilbert/datasets/suspect_labels.csv"
        suspect_df.to_csv(out_path, index=False)
        print(f"[NoiseDetector] Found {len(suspect_rows)} suspect rows.")
        print(f"[NoiseDetector] Saved to {out_path}")
    else:
        print("[NoiseDetector] No suspect rows found! Dataset is clean.")

if __name__ == "__main__":
    main()
