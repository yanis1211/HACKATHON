"""Helper script to run the full minimal pipeline."""

from src.etl.fetch_sample import main as fetch_sample
from src.features.builder import build_training_dataset
from src.models.trainer import train


def run_pipeline() -> None:
    print("📥 Fetching data...")
    fetch_sample()
    print("🧮 Building features...")
    dataset_path = build_training_dataset()
    print(f"Features saved to {dataset_path}")
    print("🤖 Training model...")
    metrics = train()
    print("Training done. Metrics:\n", metrics)


if __name__ == "__main__":
    run_pipeline()
