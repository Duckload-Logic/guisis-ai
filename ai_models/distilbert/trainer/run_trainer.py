import logging
import argparse
import os
import sys

# Ensure the project root is in the path so we can import ai_models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from ai_models.distilbert.trainer.trainer import DistilBertTrainer

# Configure logger for visibility
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """
    Main entry point for local DistilBERT training.
    Provides a CLI interface to trigger the fine-tuning process.
    """
    parser = argparse.ArgumentParser(
        description="Local DistilBERT fine-tuning CLI"
    )

    # Path configuration
    parser.add_argument(
        "--csv",
        type=str,
        default="ai_models/distilbert/datasets/labeled_dataset.csv",
        help="Path to the labeled CSV dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ai_models/distilbert/model/outputs",
        help="Directory to save the trained model"
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Training batch size"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="Learning rate"
    )

    # Debugging / Testing / Data
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run a single step for verification"
    )
    parser.add_argument(
        "--balance",
        action="store_true",
        help="Balance the dataset using oversampling before training"
    )

    args = parser.parse_args()

    # Initialize the trainer
    trainer = DistilBertTrainer()

    # Data preparation
    try:
        tokenized_datasets = trainer.prepare_dataset(
            csv_path=args.csv,
            balance=args.balance
        )
    except Exception as e:
        logger.error(f"[CLI] Data Preparation Error: {e}")
        return

    # If dry run, limit steps to 1 and epochs to 1 for quick validation
    if args.dry_run:
        logger.info("[CLI] Performing dry-run verification...")
        trainer.train(
            tokenized_datasets=tokenized_datasets,
            output_dir=os.path.join(args.output, "dry_run"),
            epochs=1,
            batch_size=args.batch_size,
            learning_rate=args.lr
        )
        return

    # Full training execution
    trainer.train(
        tokenized_datasets=tokenized_datasets,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

if __name__ == "__main__":
    main()
