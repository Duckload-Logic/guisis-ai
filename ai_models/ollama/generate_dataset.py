import argparse
import requests
import json
import csv
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

# Named Constants to replace magic values
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:4b"  # Excellent for Tagalog/Taglish
NUM_VARIATIONS_PER_SEED = 10
OUTPUT_CSV = "synthetic_dataset.csv"

AUTO_LABEL = True
LABELED_OUTPUT_CSV = "labeled_dataset.csv"

MAX_WORKERS = 1  # Keep 1 to avoid swapping with limited VRAM
BATCH_SIZE = 8
GENERATION_TIMEOUT = 600
LABELING_TIMEOUT = 300
INDIVIDUAL_LABELING_TIMEOUT = 120

GENERATION_TEMP = 0.95
LABELING_TEMP = 0.1
RETRY_BACKOFF_SECONDS = 5.0
MAX_RETRIES = 2

FALLBACK_URGENCY = "MEDIUM"
FALLBACK_CATEGORY = "PERSONAL"

OLLAMA_NUM_PREDICT = 2048
OLLAMA_NUM_CTX = 1024
OLLAMA_NUM_GPU = 999


def parse_dataset(path: str) -> list:
    with open(path, "r", encoding="utf-8") as file:
        return [
            line.strip()
            for line in file.readlines()
            if line.strip() and not line.strip().startswith("#")
        ]


def call_ollama(
    prompt: str,
    temperature: float = 0.8,
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
            time.sleep(RETRY_BACKOFF_SECONDS)
        except Exception as e:
            print(f"[OllamaClient] call_ollama: Connection/API error: {e}")
            if attempt == max_retries:
                return ""
            time.sleep(RETRY_BACKOFF_SECONDS)
    return ""


def generate_variations(
    seed_text: str, num_variations: int = 20
) -> List[str]:
    prompt = (
        "You are a Filipino college student at PUP-Taguig writing a short note "
        "to the guidance counselor.\n"
        f"Write {num_variations} DIFFERENT variations of the student concern "
        "below.\n\n"
        "Each variation MUST:\n"
        "- Be 1 to 2 short sentences only.\n"
        "- Sound casual, like a real student texting or speaking.\n"
        "- Mix English and Tagalog naturally (Taglish) or use pure Filipino.\n"
        "- Use a DIFFERENT opening phrase (e.g., \"Hi po\", "
        "\"Tanong ko lang po\", \"Good morning\", \"Ma'am/Sir\", "
        "\"Gusto ko lang po sana...\").\n"
        "- Include at least one Filipino word or phrase in each variation.\n"
        "- NEVER repeat the same sentence structure twice.\n\n"
        "Examples of good variations for a different concern:\n"
        "Concern: \"Nahihirapan po ako sa Math.\"\n"
        "Good variations:\n"
        "\"Di ko na po maintindihan yung Calculus, baka bumagsak ako.\"\n"
        "\"Ang hirap po ng Math, parang susuko na ako.\"\n"
        "\"Ma'am, tulong naman po sa Math, nalilito na po kasi ako.\"\n\n"
        "**EMOTIONAL VARIATION REQUIRED:**\n"
        "Make sure the variations cover DIFFERENT emotional tones. Mix the "
        "following feelings across the outputs:\n"
        "- Calm / routine / neutral (e.g., \"just scheduling po\")\n"
        "- Mild worry / slight anxiety (e.g., \"medyo kinakabahan po ako\")\n"
        "- Significant stress / overwhelm (e.g., \"sobrang stressed po\")\n"
        "- Sadness / hopelessness (e.g., \"wala na po akong gana\")\n"
        "- Fear / panic (e.g., \"natatakot po ako pumasok\")\n"
        "- Desperation / crisis (e.g., \"hindi ko na po kaya\")\n\n"
        f"Student concern: \"{seed_text}\"\n\n"
        f"Output exactly {num_variations} variations, each on a new line. "
        "Do NOT number them. Do NOT add extra commentary.\n"
        "Variations:"
    )

    response_text = call_ollama(
        prompt,
        temperature=GENERATION_TEMP,
        timeout=GENERATION_TIMEOUT,
    )
    if not response_text:
        return []

    lines = [
        line.strip() for line in response_text.split("\n") if line.strip()
    ]
    cleaned = [re.sub(r"^\d+[\.\)]\s*", "", line) for line in lines]

    return [
        line for line in cleaned if len(line) > 10
    ][:num_variations]


def process_seed(seed: str, index: int) -> List[str]:
    """Wrapper for concurrent execution."""
    print(f"[Thread] Seed {index}: {seed[:50]}...")
    variations = generate_variations(seed, NUM_VARIATIONS_PER_SEED)
    print(f"[Thread] Seed {index}: generated {len(variations)} variations")
    return variations


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
        prompt, temperature=LABELING_TEMP, timeout=LABELING_TIMEOUT
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

    print("   Batch failed validation. Falling back to individual labeling...")
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
        prompt, temperature=LABELING_TEMP, timeout=INDIVIDUAL_LABELING_TIMEOUT
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


def main(path: str):
    seeds: list = parse_dataset(path)

    print("Starting threaded dataset generation...")
    print(f"Model: {MODEL}")
    print(f"Seeds: {len(seeds)}")
    print(f"Workers: {MAX_WORKERS} (concurrent requests)")
    print(f"Variations per seed: {NUM_VARIATIONS_PER_SEED}\n")

    all_texts = []

    # Phase 1: Generate variations in parallel
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_seed = {
            executor.submit(process_seed, seed, i): i
            for i, seed in enumerate(seeds, 1)
        }
        for future in as_completed(future_to_seed):
            idx = future_to_seed[future]
            try:
                variations = future.result()
                all_texts.extend(variations)
            except Exception as e:
                print(f"Seed {idx} failed: {e}")

    print(f"\nTotal generated texts: {len(all_texts)}")

    # Save raw texts
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["text"])
        writer.writeheader()
        for text in all_texts:
            writer.writerow({"text": text})

    print(f"Raw dataset saved to: {OUTPUT_CSV}")

    # Phase 2: Auto-labeling (Optimized Batching)
    if AUTO_LABEL and all_texts:
        print(f"\nStarting batch auto-labeling (Size: {BATCH_SIZE})...")
        labeled_data = []

        # Process in chunks
        for i in range(0, len(all_texts), BATCH_SIZE):
            chunk = all_texts[i : i + BATCH_SIZE]
            print(
                f"   Batch {i//BATCH_SIZE + 1}: Processing items "
                f"{i+1}-{min(i+BATCH_SIZE, len(all_texts))}..."
            )

            batch_results = auto_label_batch(chunk)

            for j, res in enumerate(batch_results):
                labeled_data.append(
                    {
                        "text": chunk[j],
                        "urgency": res.get("urgency", FALLBACK_URGENCY),
                        "category": res.get("category", FALLBACK_CATEGORY),
                    }
                )

        with open(LABELED_OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["text", "urgency", "category"],
            )
            writer.writeheader()
            writer.writerows(labeled_data)

        print(f"Labeled dataset saved to: {LABELED_OUTPUT_CSV}")

        # Distribution summary
        urgency_counts = {}
        category_counts = {}
        for item in labeled_data:
            urg = item["urgency"]
            cat = item["category"]
            urgency_counts[urg] = urgency_counts.get(urg, 0) + 1
            category_counts[cat] = category_counts.get(cat, 0) + 1

        print("\nLabel Distribution:")
        print("   Urgency:", urgency_counts)
        print("   Category:", category_counts)

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to seed dataset")
    args, _ = parser.parse_known_args()

    if args.path is None:
        import os

        defaults = [
            "seed_concerns.txt",
            "ai_models/ollama/datasets/seed_concerns.txt",
            "guisis-ai/ai_models/ollama/datasets/seed_concerns.txt",
        ]
        detected_path = None
        for p in defaults:
            if os.path.exists(p):
                detected_path = p
                break

        if detected_path:
            print(f"Auto-detected seed file at: {detected_path}")
            main(path=detected_path)
        else:
            print("No path specified.")
            print("Expects: --path [PATH_NAME]")
    else:
        main(path=args.path)
