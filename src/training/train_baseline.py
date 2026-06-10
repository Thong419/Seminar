"""Command-line entrypoint for baseline model training."""

from __future__ import annotations

from pathlib import Path

from src.config.pipeline import (
    load_dataset_config,
    load_evaluation_config,
    load_split_config_from_dataset,
    load_training_config,
)
from src.data.dataset import prepare_dataset
from src.data.splitter import split_dataset
from src.training.trainer import BaselineTrainer


def main() -> None:
    root = Path.cwd()
    dataset_config = load_dataset_config(root / "configs" / "dataset.yaml")
    split_config = load_split_config_from_dataset(root / "configs" / "dataset.yaml")
    training_config = load_training_config(root / "configs" / "training.yaml")
    evaluation_config = load_evaluation_config(root / "configs" / "evaluation.yaml")

    dataset_frame, _summary = prepare_dataset(dataset_config)
    train_frame, validation_frame, test_frame, _split_summary = split_dataset(
        dataset_frame,
        split_config,
        label_column=dataset_config.label_column,
    )

    trainer = BaselineTrainer(
        dataset_config=dataset_config,
        training_config=training_config,
        evaluation_config=evaluation_config,
        enable_mlflow=True,
    )
    outcome = trainer.run(train_frame, validation_frame, test_frame)
    print(
        "Baseline training complete. "
        f"Model saved to {outcome.model_path}. "
        f"Test F1: {outcome.test_metrics.f1:.4f}"
    )


if __name__ == "__main__":
    main()