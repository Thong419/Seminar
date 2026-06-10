"""Command-line entrypoint for transformer training and comparison reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config.pipeline import (
    TrainingConfig,
    load_dataset_config,
    load_evaluation_config,
    load_model_config,
    load_split_config_from_dataset,
)
from src.data.dataset import prepare_dataset
from src.data.splitter import split_dataset
from src.evaluation.evaluator import save_evaluation_artifacts
from src.evaluation.error_analysis import build_error_analysis_frame, save_error_analysis
from src.evaluation.metrics import compute_classification_metrics
from src.training.trainer import BaselineTrainer
from src.training.transformer_trainer import TransformerTrainer


def _write_model_comparison(
    baseline_metrics: dict[str, float], transformer_metrics: dict[str, float | list[list[int]]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(
        [
            {"model": "baseline", **baseline_metrics},
            {"model": "roberta-base", **{k: v for k, v in transformer_metrics.items() if isinstance(v, (int, float))}},
        ]
    )
    comparison.to_csv(output_path, index=False)
    return output_path


def main() -> None:
    root = Path.cwd()
    dataset_config = load_dataset_config(root / "configs" / "dataset.yaml")
    split_config = load_split_config_from_dataset(root / "configs" / "dataset.yaml")
    evaluation_config = load_evaluation_config(root / "configs" / "evaluation.yaml")
    model_config, training_config, tokenization_config = load_model_config(root / "configs" / "model.yaml")

    dataset_frame, _summary = prepare_dataset(dataset_config)
    train_frame, validation_frame, test_frame, _split_summary = split_dataset(
        dataset_frame,
        split_config,
        label_column=dataset_config.label_column,
    )

    baseline_trainer = BaselineTrainer(
        dataset_config=dataset_config,
        training_config=TrainingConfig(
            experiment_name="fake_news_baseline",
            model_output_dir=root / "models" / "baseline",
            mlflow_tracking_uri="http://localhost:5000",
            dataset_version=dataset_config.version,
            random_state=42,
        ),
        evaluation_config=evaluation_config,
        enable_mlflow=False,
    )
    baseline_trainer.fit(train_frame)
    baseline_test_metrics = baseline_trainer.evaluate(test_frame)
    baseline_metrics = {
        "accuracy": baseline_test_metrics.accuracy,
        "precision": baseline_test_metrics.precision,
        "recall": baseline_test_metrics.recall,
        "f1": baseline_test_metrics.f1,
    }

    transformer_trainer = TransformerTrainer(
        model_config=model_config,
        training_config=training_config,
        evaluation_config=evaluation_config,
        tokenizer_config=tokenization_config,
        enable_mlflow=True,
    )
    transformer_output = transformer_trainer.train(
        train_frame,
        validation_frame,
        dataset_config.text_column,
        dataset_config.label_column,
    )
    transformer_metrics = transformer_trainer.evaluate_predictions(
        test_frame,
        dataset_config.text_column,
        dataset_config.label_column,
    )
    predicted_labels, confidences = transformer_trainer.predict_batch(
        test_frame,
        dataset_config.text_column,
    )

    save_evaluation_artifacts(
        compute_classification_metrics(
            y_true=test_frame[dataset_config.label_column].astype(str).tolist(),
            y_pred=predicted_labels,
        ),
        evaluation_config,
    )
    save_error_analysis(
        build_error_analysis_frame(
            texts=test_frame[dataset_config.text_column].astype(str).tolist(),
            y_true=test_frame[dataset_config.label_column].astype(str).tolist(),
            y_pred=predicted_labels,
            confidences=confidences,
        ),
        evaluation_config.output_dir / "error_analysis.csv",
    )

    comparison_path = _write_model_comparison(
        baseline_metrics=baseline_metrics,
        transformer_metrics=transformer_metrics,
        output_path=evaluation_config.output_dir / "model_comparison.csv",
    )
    print(f"Transformer training complete. Model saved to {transformer_output.model_dir}.")
    print(f"Comparison saved to {comparison_path}.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()