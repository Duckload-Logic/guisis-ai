import os
import json
import logging
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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

    def prepare_dataset(
        self,
        csv_path: str,
        val_size: float = 0.2,
        max_length: int = 64,
        balance: bool = False
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

        # Stratified split ensures even label distribution in small sets
        train_df, val_df = train_test_split(
            df,
            test_size=val_size,
            random_state=42,
            stratify=df['label']
        )

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
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
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