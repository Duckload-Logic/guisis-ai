import os
import sys
import json
import asyncio
from sklearn.metrics import classification_report, confusion_matrix

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.core.config import settings
from src.services.classifier import ClassifierService
from src.schemas.prediction import ClassificationRequest

BENCHMARK_PATH = os.path.join(
    project_root,
    "ai_models/distilbert/datasets/ood_benchmark.json"
)


def _resolve_local_model_path(model_ref: str) -> str | None:
    """Return a filesystem path for a local model directory if it exists."""
    candidate_paths = [
        model_ref,
        os.path.join(project_root, model_ref),
        os.path.join(project_root, "guisis-ai", model_ref),
    ]
    for candidate in candidate_paths:
        if os.path.isdir(candidate):
            return candidate
    return None


def run_benchmark():
    print(f"[Benchmark] Loading benchmark suite from {BENCHMARK_PATH}...")
    if not os.path.exists(BENCHMARK_PATH):
        print(f"[Error] Benchmark file not found at {BENCHMARK_PATH}")
        sys.exit(1)

    with open(BENCHMARK_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Force ClassifierService to find the local model path
    local_path = _resolve_local_model_path(settings.model_path)
    if local_path:
        settings.model_path = local_path
        # Force model source to local so it doesn't fall back to HF api
        settings.model_source = "local"
    else:
        print(
            f"[Warning] Could not resolve local model path, "
            f"using settings: {settings.model_path}"
        )

    print("[Benchmark] Initializing ClassifierService and loading model...")
    classifier = ClassifierService()
    classifier._load_local_model()

    label2id = classifier._label_mapping["label2id"]
    id2label = {v: k for k, v in label2id.items()}

    # Prepare inputs
    texts = [item["text"] for item in data]
    true_labels = [label2id[item["urgency"]] for item in data]
    pred_labels = []

    print(f"[Benchmark] Running inference on {len(texts)} OOD test cases...")

    async def run_classification():
        preds = []
        for text in texts:
            req = ClassificationRequest(text=text)
            resp = await classifier.classify(req)
            preds.append(label2id[resp.level])
        return preds

    pred_labels = asyncio.run(run_classification())

    # Calculate metrics
    target_names = [id2label[i] for i in range(len(label2id))]

    print("\n" + "=" * 60)
    print("         STANDARDIZED OOD BENCHMARK REPORT")
    print("=" * 60)
    print(classification_report(
        true_labels,
        pred_labels,
        target_names=target_names,
        digits=4
    ))

    print("\n" + "=" * 60)
    print("               CONFUSION MATRIX")
    print("=" * 60)
    cm = confusion_matrix(true_labels, pred_labels)
    for i, row in enumerate(cm):
        row_str = " | ".join(f"{val:2d}" for val in row)
        print(f"True {target_names[i]:8s}: {row_str}")

    # Detailed mismatch analysis
    print("\n" + "=" * 60)
    print("              DETAILED MISCLASSIFICATIONS")
    print("=" * 60)
    mismatch_count = 0
    for idx, (true_id, pred_id) in enumerate(zip(true_labels, pred_labels)):
        if true_id != pred_id:
            mismatch_count += 1
            print(f"Sample #{idx + 1}:")
            print(f"  Text: {data[idx]['text']}")
            print(f"  True Urgency: {target_names[true_id]}")
            print(f"  Pred Urgency: {target_names[pred_id]}")
            print("-" * 40)

    print(f"\n[Benchmark] Complete. Total errors: {mismatch_count}/{len(data)}")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
