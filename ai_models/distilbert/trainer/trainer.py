import os
import re
import json
import random
import logging
# pyrefly: ignore [missing-import]
import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
# pyrefly: ignore [missing-import]
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from datasets import Dataset, DatasetDict
from src.utils.text_cleaning import anonymize_text

# We use the existing logger pattern for consistency across the project
logger = logging.getLogger(__name__)

class DeviceManager:
    """
    Handles local hardware detection and configuration.
    Portability is key, so we detect CUDA, MPS (Apple), or CPU fallback.
    """

    @staticmethod
    def get_device() -> torch.device:
        """Determines the best available device for training."""
        if torch.cuda.is_available():
            logger.info("[DeviceManager] NVIDIA GPU detected. Using CUDA.")
            return torch.device("cuda")

        # Check for Apple Silicon GPU support
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("[DeviceManager] Apple Silicon detected. Using MPS.")
            return torch.device("mps")

        logger.info("[DeviceManager] No GPU found. Falling back to CPU.")
        return torch.device("cpu")

class DistilBertTrainer:
    """
    Decoupled trainer for fine-tuning DistilBERT on student concerns.
    Handles the end-to-end pipeline from data loading to model export.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-multilingual-cased",
        label_mapping: dict = None
    ):
        self.model_name = model_name
        self.label2id = label_mapping or {
            "LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3
        }
        self.id2label = {v: k for k, v in self.label2id.items()}

        self.device = DeviceManager.get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

        # We initialize the model with the specific label mapping
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        ).to(self.device)

    def _balance_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs random oversampling to balance class distributions.
        All classes will be upsampled to match the majority class count.
        """
        if 'label' not in df.columns:
            return df

        counts = df['label'].value_counts()
        max_size = counts.max()

        logger.info(f"[Trainer] Balancing dataset. Current distribution: "
                    f"{counts.to_dict()}")

        lst = [df]
        for class_id, count in counts.items():
            if count < max_size:
                # Upsample minority class
                diff = max_size - count
                upsampled = df[df['label'] == class_id].sample(
                    n=diff,
                    replace=True,
                    random_state=42
                )
                lst.append(upsampled)

        # Shuffle the resulting dataframe
        balanced_df = pd.concat(lst).sample(
            frac=1,
            random_state=42
        ).reset_index(drop=True)

        counts_new = balanced_df['label'].value_counts().to_dict()
        logger.info(f"[Trainer] Balancing complete: {counts_new}")
        return balanced_df

    def _augment_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Augments training data by stripping prefixes and injecting noise.
        Only applied to train split to avoid data leakage to validation.
        """
        prefixes = [
            "hi po", "hello po", "good morning po", "good morning",
            "good day po", "good day", "ma'am", "sir", "excuse me po",
            "excuse me", "tanong ko lang po", "tanong ko lang",
            "gusto ko lang po sana ipaalam", "gusto ko lang po sana",
            "gusto ko lang po", "gusto ko lang", "pwede po ba",
            "pwede ba", "pakiusap po", "sana po", "please po",
            "sa totoo lang po", "seriously", "uy", "ma'am/sir",
            "sir/ma'am", "salamat po", "salamat", "good afternoon po",
            "good afternoon"
        ]

        def clean_and_strip(text: str) -> str:
            text = text.strip('"\' ')
            changed = True
            while changed:
                changed = False
                lower_text = text.lower()
                for prefix in prefixes:
                    if lower_text.startswith(prefix):
                        slice_idx = len(prefix)
                        while (
                            slice_idx < len(text)
                            and text[slice_idx] in ",.!?’- \t"
                        ):
                            slice_idx += 1
                        text = text[slice_idx:]
                        changed = True
                        break
            if text:
                text = text[0].upper() + text[1:]
            return text.strip()

        def apply_taglish_noise(text: str) -> str:
            words = text.split()
            new_words = []
            for w in words:
                w_lower = w.lower().strip(",.!?")
                if w_lower in ["po", "opo"] and random.random() < 0.7:
                    continue
                new_words.append(w)
            text = " ".join(new_words)

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

        augmented_rows = []
        random.seed(42)

        for _, row in df.iterrows():
            orig_text = str(row["text"])
            label = row["label"]
            row_dict = row.to_dict()

            # 1. Prefix stripped version
            stripped = clean_and_strip(orig_text)
            if stripped and stripped.lower() != orig_text.lower():
                new_row = row_dict.copy()
                new_row["text"] = stripped
                augmented_rows.append(new_row)

                # 1b. Slang/typo version of stripped
                noisy_stripped = apply_taglish_noise(stripped)
                if (
                    noisy_stripped
                    and noisy_stripped.lower() != stripped.lower()
                ):
                    new_row = row_dict.copy()
                    new_row["text"] = noisy_stripped
                    augmented_rows.append(new_row)

            # 2. Slang/typo version of original
            noisy_orig = apply_taglish_noise(orig_text)
            if noisy_orig and noisy_orig.lower() != orig_text.lower():
                new_row = row_dict.copy()
                new_row["text"] = noisy_orig
                augmented_rows.append(new_row)

        if augmented_rows:
            aug_df = pd.DataFrame(augmented_rows)
            aug_df = aug_df.drop_duplicates(subset=["text"])
            combined_df = pd.concat([df, aug_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=["text"])
            logger.info(
                f"[Trainer] Augmentation added {len(combined_df) - len(df)} "
                f"rows. New training size: {len(combined_df)}"
            )
            return combined_df
        return df

    def prepare_dataset(
        self,
        csv_path: str,
        val_size: float = 0.2,
        max_length: int = 256,
        balance: bool = False,
        augment: bool = False
    ) -> DatasetDict:
        """
        Loads CSV and converts it into a tokenized Hugging Face Dataset.
        Includes optional oversampling to balance class distribution.
        """
        logger.info(f"[Trainer] Loading dataset from {csv_path}")
        df = pd.read_csv(csv_path)

        # Keep preprocessing consistent with the inference service.
        df['text'] = df['text'].fillna('').astype(str).map(anonymize_text)

        # Ensure our target column is mapped correctly to numeric IDs
        df['label'] = df['urgency'].map(self.label2id)

        # Drop any rows with NaN labels or missing text
        df = df.dropna(subset=['label', 'text'])
        df['label'] = df['label'].astype(int)

        # Stratified split ensures even label distribution in small sets
        train_df, val_df = train_test_split(
            df,
            test_size=val_size,
            random_state=42,
            stratify=df['label']
        )

        # Augment training set to prevent OOD leakage
        if augment:
            train_df = self._augment_dataframe(train_df)

        # Optional balancing via oversampling
        if balance:
            train_df = self._balance_dataframe(train_df)

        # Build the HF datasets
        train_ds = Dataset.from_pandas(train_df[['text', 'label']])
        val_ds = Dataset.from_pandas(val_df[['text', 'label']])

        raw_datasets = DatasetDict({'train': train_ds, 'validation': val_ds})

        def tokenize_fn(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,
                padding='max_length',
                max_length=max_length
            )

        tokenized = raw_datasets.map(tokenize_fn, batched=True)
        tokenized.set_format(
            type='torch',
            columns=['input_ids', 'attention_mask', 'label']
        )

        return tokenized

    @staticmethod
    def compute_metrics(eval_pred):
        """Calculates evaluation metrics during training."""
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            labels, predictions, average='macro'
        )

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
            , 'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'f1_macro': macro_f1
        }

    def train(
        self,
        tokenized_datasets: DatasetDict,
        output_dir: str,
        epochs: int = 5,
        batch_size: int = 16,
        learning_rate: float = 3e-5
    ):
        """Executes the fine-tuning process."""
        training_args = TrainingArguments(
            output_dir=os.path.join(output_dir, "results"),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="f1_macro",
            greater_is_better=True,
            weight_decay=0.01,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            # Enable fp16 only if using NVIDIA GPU
            fp16=self.device.type == "cuda",
            report_to="none"
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets['train'],
            eval_dataset=tokenized_datasets['validation'],
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        logger.info("[Trainer] Starting training loop...")
        trainer.train()

        # Final export
        logger.info(f"[Trainer] Exporting model to {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)

        # Save mapping for the inference layer to use later
        mapping_path = os.path.join(output_dir, "label_mapping.json")
        with open(mapping_path, "w") as f:
            json.dump(
                {"label2id": self.label2id, "id2label": self.id2label},
                f,
                indent=2
            )

        logger.info("[Trainer] Training and export complete.")