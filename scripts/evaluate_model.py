import os
import sys
import json
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.config import settings
from src.utils.text_cleaning import anonymize_text

MODEL_REF = settings.model_path
DATA_PATH = "ai_models/distilbert/datasets/labeled_dataset.csv"
TEST_SPLIT_SIZE = 0.2
RANDOM_STATE_SEED = 42
INFERENCE_BATCH_SIZE = 16
MAX_SEQUENCE_LENGTH = 256


def _resolve_local_model_path(model_ref: str) -> str | None:
    """Return a filesystem path for a local model directory if one exists."""
    candidate_paths = [
        model_ref,
        os.path.join(project_root, model_ref),
    ]

    for candidate in candidate_paths:
        if os.path.isdir(candidate):
            return candidate

    return None


def _load_label_mapping(model_ref: str, local_model_path: str | None) -> dict:
    """Load label mapping from disk or the Hugging Face Hub."""
    if local_model_path:
        mapping_path = os.path.join(local_model_path, "label_mapping.json")
        if os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                return json.load(f)

    try:
        from huggingface_hub import hf_hub_download

        mapping_file = hf_hub_download(
            repo_id=model_ref,
            filename="label_mapping.json",
            token=settings.hf_token,
        )
        with open(mapping_file, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_model(model_ref: str):
    """Load a local checkpoint or a Hugging Face Hub model."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    local_model_path = _resolve_local_model_path(model_ref)
    load_source = local_model_path or model_ref
    is_hub_model = local_model_path is None

    if is_hub_model and settings.hf_token:
        import huggingface_hub

        huggingface_hub.login(settings.hf_token)

    label_mapping = _load_label_mapping(model_ref, local_model_path)

    tokenizer = AutoTokenizer.from_pretrained(
        load_source,
        token=settings.hf_token if is_hub_model else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        load_source,
        token=settings.hf_token if is_hub_model else None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return tokenizer, model, device, label_mapping, is_hub_model


def _get_target_names(id2label: dict, num_labels: int) -> list[str]:
    """Return labels in id order, regardless of whether keys are strings or ints."""
    ordered_names = []
    for label_id in range(num_labels):
        ordered_names.append(
            id2label.get(str(label_id), id2label.get(label_id, str(label_id)))
        )
    return ordered_names


def evaluate():
    local_model_path = _resolve_local_model_path(MODEL_REF)
    if local_model_path is None and "/" not in MODEL_REF:
        print(f"[Error] Model path {MODEL_REF} not found.")
        sys.exit(1)

    print("[Evaluate] Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    mapping = _load_label_mapping(MODEL_REF, local_model_path)
    if not mapping:
        print("[Warning] label_mapping.json not found; falling back to model config labels.")

    print("[Evaluate] Loading tokenizer and model...")
    tokenizer, model, device, _, is_hub_model = _load_model(MODEL_REF)

    if mapping and "label2id" in mapping:
        label2id = mapping["label2id"]
        id2label = {v: k for k, v in label2id.items()}
    else:
        model_label2id = getattr(model.config, "label2id", {}) or {}
        model_id2label = getattr(model.config, "id2label", {}) or {}

        if model_label2id:
            label2id = model_label2id
            id2label = {v: k for k, v in label2id.items()}
        elif model_id2label:
            id2label = model_id2label
            label2id = {label_name: int(label_id) for label_id, label_name in id2label.items()}
        else:
            raise ValueError("Could not determine label mapping from the checkpoint or model config.")

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
    target_names = _get_target_names(id2label, len(label2id))
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
    model_type = "huggingface-hub" if is_hub_model else "local"
    print(f"[Evaluate] Completed evaluation using {model_type} model: {MODEL_REF}")
    print("="*50)

if __name__ == "__main__":
    evaluate()