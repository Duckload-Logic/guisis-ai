import os
import sys
import json
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.text_cleaning import anonymize_text

MODEL_PATH = "ai_models/distilbert/model/outputs"
DATA_PATH = "ai_models/distilbert/datasets/labeled_dataset.csv"
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE_SEED = 42
INFERENCE_BATCH_SIZE = 16
MAX_SEQUENCE_LENGTH = 256

def evaluate():
    if not os.path.exists(MODEL_PATH):
        print(f"[Error] Model path {MODEL_PATH} not found.")
        sys.exit(1)

    print("[Evaluate] Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # Load label mapping
    mapping_path = os.path.join(MODEL_PATH, "label_mapping.json")
    with open(mapping_path, "r") as f:
        mapping = json.load(f)

    label2id = mapping["label2id"]
    id2label = {v: k for k, v in label2id.items()}

    df["label"] = df["urgency"].map(label2id)

    print(
        "[Evaluate] Splitting data "
        f"({int((1-TEST_SPLIT_SIZE)*100)}/"
        f"{int(TEST_SPLIT_SIZE*100)} stratified split)...",
    )
    _, val_df = train_test_split(
        df,
        test_size=TEST_SPLIT_SIZE,
        random_state=RANDOM_STATE_SEED,
        stratify=df["label"]
    )

    print("[Evaluate] Loading tokenizer and model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH
    ).to(device)

    model.eval()

    texts = val_df["text"].fillna("").astype(str).map(anonymize_text).tolist()
    true_labels = val_df["label"].tolist()
    pred_labels = []

    print(f"[Evaluate] Running inference on {len(texts)} test samples...")

    for i in range(0, len(texts), INFERENCE_BATCH_SIZE):
        batch_texts = texts[i:i+INFERENCE_BATCH_SIZE]

        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=MAX_SEQUENCE_LENGTH,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=-1).cpu().numpy().tolist()
            pred_labels.extend(batch_preds)

    print("\n" + "="*50)
    print("           CLASSIFICATION REPORT")
    print("="*50)
    target_names = [id2label[str(i)] if str(i) in id2label else id2label[i]
                    for i in range(len(id2label))]
    print(classification_report(
        true_labels,
        pred_labels,
        target_names=target_names
    ))

    print("\n" + "="*50)
    print("             CONFUSION MATRIX")
    print("="*50)
    cm = confusion_matrix(true_labels, pred_labels)
    cm_df = pd.DataFrame(
        cm,
        index=[f"True {name}" for name in target_names],
        columns=[f"Pred {name}" for name in target_names]
    )
    print(cm_df)
    print("="*50)

if __name__ == "__main__":
    evaluate()