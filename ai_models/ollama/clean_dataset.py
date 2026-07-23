import os
import re
import csv
import json
import time
import requests
from typing import List, Dict

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:4b"
INPUT_CSV = "ai_models/distilbert/datasets/labeled_dataset_OLD(1).csv"
OUTPUT_CSV = "ai_models/distilbert/datasets/labeled_dataset.csv"
CHECKPOINT_CSV = "ai_models/distilbert/datasets/cleaned_checkpoint.csv"

BATCH_SIZE = 16
TIMEOUT = 300
RETRY_BACKOFF = 5.0
MAX_RETRIES = 2
LABELING_TEMP = 0.1

FALLBACK_URGENCY = "MEDIUM"
FALLBACK_CATEGORY = "PERSONAL"

OLLAMA_NUM_PREDICT = 4096
OLLAMA_NUM_CTX = 2048
OLLAMA_NUM_GPU = 999


def call_ollama(
    prompt: str,
    temperature: float = 0.1,
    timeout: int = 300,
    max_retries: int = MAX_RETRIES,
) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": OLLAMA_NUM_PREDICT,
            "num_ctx": OLLAMA_NUM_CTX,
            "num_gpu": OLLAMA_NUM_GPU,
        },
    }
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                OLLAMA_URL, json=payload, timeout=timeout
            )
            if response.status_code != 200:
                print(
                    "[OllamaClient] call_ollama: API error "
                    f"{response.status_code}: {response.text}"
                )
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.Timeout as e:
            print(f"[OllamaClient] call_ollama: Timeout error: {e}")
            if attempt == max_retries:
                return ""
            time.sleep(RETRY_BACKOFF)
        except Exception as e:
            print(f"[OllamaClient] call_ollama: Connection/API error: {e}")
            if attempt == max_retries:
                return ""
            time.sleep(RETRY_BACKOFF)
    return ""


def auto_label_batch(texts: List[str]) -> List[Dict[str, str]]:
    """Labels a batch of texts in one request with ID tracking."""
    if not texts:
        return []

    items_str = ""
    for i, txt in enumerate(texts):
        safe_txt = txt.replace('"', '\\"')
        items_str += f"[ID: {i}] Text: \"{safe_txt}\"\n"

    prompt = (
        "Classify the following student concerns.\n"
        "For each item, classify Urgency and Category based on these "
        "strict definitions:\n\n"
        "URGENCY:\n"
        "- LOW: Administrative/routine queries, scheduling questions, excuse "
        "slips, office hours, general info requests, without expressing "
        "emotional pain, worry, or stress.\n"
        "- MEDIUM: Academic difficulty, study fatigue, minor time management "
        "issues, or mild worries. Student is stressed or tired but not in "
        "despair or crisis.\n"
        "- HIGH: Overwhelm, severe stress, anxiety, panic attacks, depression, "
        "fear, bullying, harassment, stalkers, or major family conflicts.\n"
        "- CRITICAL: Active self-harm, suicide ideation, wanting to die or "
        "disappear, severe panic/breathing issues, domestic abuse/violence, "
        "or immediate safety danger.\n\n"
        "CATEGORY:\n"
        "Choose exactly one: ACADEMIC, FINANCIAL, PERSONAL, FAMILY, HEALTH, "
        "CAREER\n\n"
        "Output ONLY a JSON list of objects. Each object MUST include the "
        "\"id\".\n"
        "Format: [\n"
        "  {{\"id\": 0, \"urgency\": \"...\", \"category\": \"...\"}},\n"
        "  ...\n"
        "]\n\n"
        f"Items to classify:\n{items_str}\n"
    )

    response = call_ollama(
        prompt, temperature=LABELING_TEMP, timeout=TIMEOUT
    )

    json_match = re.search(r"\[\s*\{.*\}\s*\]", response, re.DOTALL)
    if json_match:
        try:
            results = json.loads(json_match.group())
            if len(results) == len(texts):
                results.sort(key=lambda x: x.get("id", 0))
                return results
        except (json.JSONDecodeError, ValueError):
            pass

    print("   Batch failed. Falling back to individual labeling...")
    fallback_results = []
    for i, text in enumerate(texts):
        labels = auto_label_text_individual(text)
        fallback_results.append(
            {
                "id": i,
                "urgency": labels["urgency"],
                "category": labels["category"],
            }
        )
    return fallback_results


def auto_label_text_individual(text: str) -> Dict[str, str]:
    """Single-item fallback classifier."""
    prompt = (
        "Classify the student concern into:\n"
        "Urgency: LOW, MEDIUM, HIGH, CRITICAL\n"
        "Category: ACADEMIC, FINANCIAL, PERSONAL, FAMILY, HEALTH, CAREER\n\n"
        "URGENCY DEFINITIONS:\n"
        "- LOW: Administrative/routine scheduling/excuse slip queries.\n"
        "- MEDIUM: Academic/minor worry/study fatigue, no panic/despair.\n"
        "- HIGH: Severe stress, anxiety, panic, fear, bullying, harassment.\n"
        "- CRITICAL: Self-harm, suicide ideation, wanting to die/disappear.\n\n"
        "Output ONLY JSON: {\"urgency\": \"...\", \"category\": \"...\"}\n"
        f"Text: \"{text}\"\n"
    )

    response = call_ollama(
        prompt, temperature=LABELING_TEMP, timeout=120
    )
    json_match = re.search(r"\{.*\}", response, re.DOTALL)

    if json_match:
        try:
            return json.loads(json_match.group())
        except Exception:
            pass

    return {
        "urgency": FALLBACK_URGENCY,
        "category": FALLBACK_CATEGORY,
    }


def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file {INPUT_CSV} not found!")
        return

    # Load existing progress if any
    cleaned_rows = []
    processed_texts = set()
    if os.path.exists(CHECKPOINT_CSV):
        with open(CHECKPOINT_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cleaned_rows.append(row)
                processed_texts.add(row["text"])
        print(f"Loaded {len(cleaned_rows)} processed rows from checkpoint.")

    # Read original rows
    original_rows = []
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["text"] not in processed_texts:
                original_rows.append(row)

    print(f"Remaining rows to process: {len(original_rows)}")
    if not original_rows:
        print("Everything has been processed!")
        finalize()
        return

    # Open checkpoint in append mode
    checkpoint_exists = os.path.exists(CHECKPOINT_CSV)
    with open(CHECKPOINT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["text", "urgency", "category"]
        )
        if not checkpoint_exists:
            writer.writeheader()

        # Process in chunks
        for i in range(0, len(original_rows), BATCH_SIZE):
            chunk = original_rows[i : i + BATCH_SIZE]
            chunk_texts = [row["text"] for row in chunk]

            print(
                f"Processing items {i+1}-{min(i+BATCH_SIZE, len(original_rows))}"
                f" / {len(original_rows)}..."
            )

            batch_results = auto_label_batch(chunk_texts)

            for j, res in enumerate(batch_results):
                row_data = {
                    "text": chunk_texts[j],
                    "urgency": res.get("urgency", FALLBACK_URGENCY),
                    "category": res.get("category", FALLBACK_CATEGORY),
                }
                writer.writerow(row_data)
                cleaned_rows.append(row_data)

            f.flush()

    finalize()


def finalize():
    # Copy checkpoint to final output
    if os.path.exists(CHECKPOINT_CSV):
        import shutil

        shutil.copy(CHECKPOINT_CSV, OUTPUT_CSV)
        print(f"Successfully wrote cleaned dataset to: {OUTPUT_CSV}")
    else:
        print("No checkpoint found to finalize.")


if __name__ == "__main__":
    main()
