import os
import shutil
import pandas as pd

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

def clean_data():
    csv_path = os.path.join(
        project_root, "ai_models/distilbert/datasets/labeled_dataset.csv"
    )
    backup_path = os.path.join(
        project_root, "ai_models/distilbert/datasets/labeled_dataset_original.csv"
    )

    # Backup the original dataset first if backup doesn't exist
    if not os.path.exists(backup_path):
        shutil.copyfile(csv_path, backup_path)
        print(f"[Cleaner] Backup created at: {backup_path}")

    df = pd.read_csv(csv_path)
    original_urgencies = df["urgency"].copy()

    # Define clean rules based on our error analysis
    critical_keywords = [
        "cut ng kamay", "nagko-cut", "gupitin yung braso", "laslas",
        "magpakamatay", "suicide", "suicidal", "mamatay", "stalker",
        "nagbabanta", "banta", "sinusundan", "stalk", "bullying"
    ]

    high_keywords = [
        "hindi ko na kaya", "sasabog", "drowning", "stressed",
        "panic attack", "hirap huminga", "emergency", "ma-dismiss",
        "dismiss", "failing", "depress", "desperate", "gana",
        "overwhelmed", "natatakot", "maintindihan"
    ]

    medium_keywords = [
        "kinakabahan", "nag-aalala", "hirap", "sad", "lungkot",
        "problemado", "kaba"
    ]

    low_keywords = [
        "reunion", "walk-in", "walk in", "excuse letter", "excuse slip",
        "resume", "part-time", "part time", "tutorial", "allowance",
        "registration", "OJT interview", "tutorial", "slots", "salamat",
        "format", "sample"
    ]

    changes_count = 0

    for idx, row in df.iterrows():
        text = str(row["text"]).lower()
        current_urgency = row["urgency"]
        new_urgency = current_urgency

        # Rule 1: Safety overrides for life-threatening issues (CRITICAL)
        if any(kw in text for kw in critical_keywords):
            new_urgency = "CRITICAL"
        
        # Rule 2: Elevate to HIGH for severe stress/crisis
        elif any(kw in text for kw in high_keywords):
            # Only elevate if it's not already CRITICAL
            if current_urgency != "CRITICAL":
                new_urgency = "HIGH"

        # Rule 2.5: Standardize MEDIUM for mild emotional distress
        elif any(kw in text for kw in medium_keywords):
            if current_urgency not in ["CRITICAL", "HIGH"]:
                new_urgency = "MEDIUM"

        # Rule 3: Demote to LOW for simple administrative queries
        elif any(kw in text for kw in low_keywords):
            # Only demote if it doesn't contain HIGH/CRITICAL terms
            if not any(kw in text for kw in high_keywords + critical_keywords):
                new_urgency = "LOW"

        # Rule 4: Fix specific boundary conflicts from our analysis
        if "medyo kinakabahan" in text and "grades" in text:
            # typical MEDIUM academic stress
            new_urgency = "MEDIUM"
        if "medyo kinakabahan" in text and "heartbreak" in text:
            new_urgency = "MEDIUM"

        if new_urgency != current_urgency:
            df.at[idx, "urgency"] = new_urgency
            changes_count += 1

    if changes_count > 0:
        df.to_csv(csv_path, index=False)
        print(f"[Cleaner] Dataset cleaned! Changed {changes_count} labels.")
        
        # Compare distribution
        print("\nOld distribution:")
        print(original_urgencies.value_counts())
        print("\nNew distribution:")
        print(df["urgency"].value_counts())
    else:
        print("[Cleaner] No changes needed. Dataset matches the clean rules.")

if __name__ == "__main__":
    clean_data()
