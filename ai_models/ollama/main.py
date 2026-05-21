import argparse
import requests
import json
import csv
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional



OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma3:4b"                # Excellent for Tagalog/Taglish
NUM_VARIATIONS_PER_SEED = 20    # Generates 20 variations per seed
OUTPUT_CSV = "synthetic_dataset.csv"

AUTO_LABEL = True               # Set to False if you want to label manually
LABELED_OUTPUT_CSV = "labeled_dataset.csv"

MAX_WORKERS = 1                 # Set to 1 for 4GB VRAM to avoid swapping
BATCH_SIZE = 8                  # Number of texts to label in one prompt
GENERATION_TIMEOUT = 600        # 10 minutes for bulk generation
LABELING_TIMEOUT = 300          # 5 minutes for a batch of labels

def parse_dataset(path: str) -> list:
    with open(path, 'r') as file:
        return file.readlines()

def call_ollama(prompt: str, temperature: float = 0.8, timeout: int = 300, max_retries: int = 2) -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 2048,
            "stop": ["\n\n", "Variations:", "Student concern:"],
            "num_ctx": 1024,
            "num_gpu": 999
        }
    }
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json().get("response", "").strip()
        except requests.exceptions.Timeout:
            if attempt == max_retries:
                return ""
            time.sleep(5)
        except Exception:
            if attempt == max_retries:
                return ""
            time.sleep(5)
    return ""

def generate_variations(seed_text: str, num_variations: int = 20) -> List[str]:
    prompt = f"""You are a Filipino college student at PUP-Taguig writing a short note to the guidance counselor.
Write {num_variations} DIFFERENT variations of the student concern below.

Each variation MUST:
- Be 1 to 2 short sentences only.
- Sound casual, like a real student texting or speaking.
- Mix English and Tagalog naturally (Taglish) or use pure Filipino.
- Use a DIFFERENT opening phrase (e.g., "Hi po", "Tanong ko lang po", "Good morning", "Ma'am/Sir", "Gusto ko lang po sana...").
- Include at least one Filipino word or phrase in each variation.
- NEVER repeat the same sentence structure twice.

Examples of good variations for a different concern:
Concern: "Nahihirapan po ako sa Math."
Good variations:
"Di ko na po maintindihan yung Calculus, baka bumagsak ako."
"Ang hirap po ng Math, parang susuko na ako."
"Ma'am, tulong naman po sa Math, nalilito na po kasi ako."

**EMOTIONAL VARIATION REQUIRED:**
Make sure the variations cover DIFFERENT emotional tones. Mix the following feelings across the outputs:
- Calm / routine / neutral (e.g., "just scheduling po")
- Mild worry / slight anxiety (e.g., "medyo kinakabahan po ako")
- Significant stress / overwhelm (e.g., "sobrang stressed na po ako")
- Sadness / hopelessness (e.g., "wala na po akong gana")
- Fear / panic (e.g., "natatakot po ako pumasok")
- Desperation / crisis (e.g., "hindi ko na po kaya")

Student concern: "{seed_text}"

Output exactly {num_variations} variations, each on a new line. Do NOT number them. Do NOT add extra commentary.
Variations:"""
    response_text = call_ollama(prompt, temperature=0.95, timeout=GENERATION_TIMEOUT)
    if not response_text:
        return []
    lines = [line.strip() for line in response_text.split('\n') if line.strip()]
    cleaned = [re.sub(r'^\d+[\.\)]\s*', '', line) for line in lines]
    return [line for line in cleaned if len(line) > 10][:num_variations]

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

    # Build the batch prompt with unique IDs
    items_str = ""
    for i, txt in enumerate(texts):
        # Escaping double quotes for the prompt to avoid JSON confusion
        safe_txt = txt.replace('"', '\\"')
        items_str += f"[ID: {i}] Text: \"{safe_txt}\"\n"

    prompt = f"""Classify the following student concerns.
For each item, provide:
1. Urgency: LOW (routine), MEDIUM (stress), HIGH (distress), CRITICAL (danger)
2. Category: ACADEMIC, FINANCIAL, PERSONAL, FAMILY, HEALTH, CAREER

Output ONLY a JSON list of objects. Each object MUST include the "id".
Format: [
  {{"id": 0, "urgency": "...", "category": "..."}},
  ...
]

Items to classify:
{items_str}
"""
    response = call_ollama(prompt, temperature=0.1, timeout=LABELING_TIMEOUT)

    # Try to extract and parse JSON
    json_match = re.search(r'\[\s*\{.*\}\s*\]', response, re.DOTALL)
    if json_match:
        try:
            results = json.loads(json_match.group())
            # Basic validation: check if we got the same count
            if len(results) == len(texts):
                # Sort by ID to ensure order parity
                results.sort(key=lambda x: x.get('id', 0))
                return results
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: If batch fails, label items individually (very slow but safe)
    print(f"   ⚠️ Batch failed validation. Falling back to individual labeling...")
    fallback_results = []
    for i, text in enumerate(texts):
        labels = auto_label_text_individual(text)
        fallback_results.append({
            "id": i,
            "urgency": labels["urgency"],
            "category": labels["category"]
        })
    return fallback_results

def auto_label_text_individual(text: str) -> Dict[str, str]:
    """Single-item fallback classifier."""
    prompt = f"""Classify into:
Urgency: LOW, MEDIUM, HIGH, CRITICAL
Category: ACADEMIC, FINANCIAL, PERSONAL, FAMILY, HEALTH, CAREER
Output ONLY JSON: {{"urgency": "...", "category": "..."}}
Text: "{text}" """
    response = call_ollama(prompt, temperature=0.1, timeout=120)
    json_match = re.search(r'\{.*\}', response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except: pass
    return {"urgency": "MEDIUM", "category": "PERSONAL"}




def main(path: str):
    seeds: list = parse_dataset(path)

    print(f"Starting threaded dataset generation...")
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
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['text'])
        writer.writeheader()
        for text in all_texts:
            writer.writerow({'text': text})
    print(f"Raw dataset saved to: {OUTPUT_CSV}")

    # Phase 2: Auto-labeling (Optimized Batching)
    if AUTO_LABEL and all_texts:
        print(f"\n🏷️  Starting batch auto-labeling (Size: {BATCH_SIZE})...")
        labeled_data = []

        # Process in chunks
        for i in range(0, len(all_texts), BATCH_SIZE):
            chunk = all_texts[i : i + BATCH_SIZE]
            print(f"   Batch {i//BATCH_SIZE + 1}: Processing items {i+1}-{min(i+BATCH_SIZE, len(all_texts))}...")

            batch_results = auto_label_batch(chunk)

            for j, res in enumerate(batch_results):
                labeled_data.append({
                    'text': chunk[j],
                    'urgency': res.get('urgency', 'MEDIUM'),
                    'category': res.get('category', 'PERSONAL')
                })

        with open(LABELED_OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['text', 'urgency', 'category'])
            writer.writeheader()
            writer.writerows(labeled_data)
        print(f"Labeled dataset saved to: {LABELED_OUTPUT_CSV}")

        # Distribution summary
        urgency_counts = {}
        category_counts = {}
        for item in labeled_data:
            urgency_counts[item['urgency']] = urgency_counts.get(item['urgency'], 0) + 1
            category_counts[item['category']] = category_counts.get(item['category'], 0) + 1
        print("\nLabel Distribution:")
        print("   Urgency:", urgency_counts)
        print("   Category:", category_counts)

    print("\nDone!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to seed dataset")
    args = parser.parse_args()

    if args.path is None:
        print("No path specified.")
        print("Expects: --path [PATH_NAME]")
        exit()

    main(path=args.path)
