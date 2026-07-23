import os
import re
import random
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

CSV_PATH = os.path.join(
    project_root, "ai_models/distilbert/datasets/labeled_dataset.csv"
)
BACKUP_PATH = os.path.join(
    project_root, "ai_models/distilbert/datasets/labeled_dataset_clean.csv"
)

# Intro prefixes to remove to prevent template overfitting
PREFIXES = [
    "hi po", "hello po", "good morning po", "good morning", "good day po",
    "good day", "ma'am", "sir", "excuse me po", "excuse me",
    "tanong ko lang po", "tanong ko lang", "gusto ko lang po sana ipaalam",
    "gusto ko lang po sana", "gusto ko lang po", "gusto ko lang",
    "pwede po ba", "pwede ba", "pakiusap po", "sana po", "please po",
    "sa totoo lang po", "seriously", "uy", "ma'am/sir", "sir/ma'am",
    "salamat po", "salamat", "good afternoon po", "good afternoon"
]


def clean_and_strip(text: str) -> str:
    """Recursively strip introductory prefixes from the student query."""
    text = text.strip('"\' ')
    changed = True
    while changed:
        changed = False
        lower_text = text.lower()
        for prefix in PREFIXES:
            if lower_text.startswith(prefix):
                slice_idx = len(prefix)
                while slice_idx < len(text) and text[slice_idx] in ",.!?’- \t":
                    slice_idx += 1
                text = text[slice_idx:]
                changed = True
                break
    # Capitalize the first letter of the stripped text for grammatical realism
    if text:
        text = text[0].upper() + text[1:]
    return text.strip()


def apply_taglish_noise(text: str) -> str:
    """Inject realistic Taglish slang, typos, and drop polite particles."""
    # Randomly drop "po" / "opo" (stressed students rarely stay polite)
    words = text.split()
    new_words = []
    for w in words:
        w_lower = w.lower().strip(",.!?")
        if w_lower in ["po", "opo"] and random.random() < 0.7:
            continue
        new_words.append(w)
    text = " ".join(new_words)

    # Conversational slang substitutions
    abbrev_map = {
        r"\bkasi\b": "kc",
        r"\bsiya\b": "sya",
        r"\bniya\b": "nya",
        r"\bna lang\b": "nalang",
        r"\bkapag\b": "pag",
        r"\bako\b": "aq",
        r"\bplease\b": "pls",
        r"\bkaibigan\b": "tropa",
        r"\bschool\b": "iskul",
    }

    for pattern, repl in abbrev_map.items():
        if random.random() < 0.5:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    return text.strip()


def augment_data():
    random.seed(42)
    print(f"[Augmenter] Reading cleaned dataset from {CSV_PATH}...")
    if not os.path.exists(CSV_PATH):
        print(f"[Error] Dataset not found at {CSV_PATH}")
        return

    # Backup the clean dataset before augmenting
    if not os.path.exists(BACKUP_PATH):
        import shutil
        shutil.copyfile(CSV_PATH, BACKUP_PATH)
        print(f"[Augmenter] Backup of clean dataset created at {BACKUP_PATH}")

    df = pd.read_csv(CSV_PATH)
    augmented_rows = []

    for _, row in df.iterrows():
        orig_text = str(row["text"])
        urgency = row["urgency"]
        category = row["category"]

        # Step 1: Prefix-stripped version (OOD simulation)
        stripped = clean_and_strip(orig_text)
        if stripped and stripped.lower() != orig_text.lower():
            augmented_rows.append({
                "text": stripped,
                "urgency": urgency,
                "category": category
            })

            # Step 1b: Noisy version of stripped text
            noisy_stripped = apply_taglish_noise(stripped)
            if noisy_stripped and noisy_stripped.lower() != stripped.lower():
                augmented_rows.append({
                    "text": noisy_stripped,
                    "urgency": urgency,
                    "category": category
                })

        # Step 2: Noisy version of original text
        noisy_orig = apply_taglish_noise(orig_text)
        if noisy_orig and noisy_orig.lower() != orig_text.lower():
            augmented_rows.append({
                "text": noisy_orig,
                "urgency": urgency,
                "category": category
            })

    if augmented_rows:
        aug_df = pd.DataFrame(augmented_rows)
        # Drop duplicates in the augmented set to keep it clean
        aug_df = aug_df.drop_duplicates(subset=["text"])
        
        # Combine original and augmented data
        combined_df = pd.concat([df, aug_df], ignore_index=True)
        # Final safety deduplication
        combined_df = combined_df.drop_duplicates(subset=["text"])

        combined_df.to_csv(CSV_PATH, index=False)
        print(f"[Augmenter] Success! Generated {len(aug_df)} augmented rows.")
        print(f"[Augmenter] New dataset size: {len(combined_df)} rows.")
    else:
        print("[Augmenter] No new rows generated.")


if __name__ == "__main__":
    augment_data()
