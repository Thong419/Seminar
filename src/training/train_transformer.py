"""Command-line entrypoint for transformer training and comparison reporting."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
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
from src.utils.logging import configure_logging


LOGGER = logging.getLogger(__name__)
CHECKPOINT_PATTERN = re.compile(r"^checkpoint-(\d+)$")


def _write_model_comparison(
    baseline_metrics: dict[str, float],
    transformer_metrics: dict[str, float | list[list[int]]],
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison = pd.DataFrame(
        [
            {"model": "baseline", **baseline_metrics},
            {
                "model": "roberta-base",
                **{k: v for k, v in transformer_metrics.items() if isinstance(v, (int, float))},
            },
        ]
    )
    comparison.to_csv(output_path, index=False)
    return output_path


def _print_diagnostics(
    model_name: str,
    train_rows: int,
    validation_rows: int,
    test_rows: int,
) -> None:
    try:
        import torch
    except Exception as exc:
        print("Transformer training diagnostics")
        print(f"- python_version: {sys.version.split()[0]}")
        print(f"- torch_import_error: {exc}")
        raise

    try:
        import transformers
    except Exception as exc:
        print("Transformer training diagnostics")
        print(f"- python_version: {sys.version.split()[0]}")
        print(f"- torch_version: {torch.__version__}")
        print(f"- transformers_import_error: {exc}")
        raise

    selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Transformer training diagnostics")
    print(f"- python_version: {sys.version.split()[0]}")
    print(f"- torch_version: {torch.__version__}")
    print(f"- transformers_version: {transformers.__version__}")
    print(f"- cuda_available: {torch.cuda.is_available()}")
    print(f"- selected_device: {selected_device}")
    print(f"- model_name: {model_name}")
    print(f"- train_rows: {train_rows}")
    print(f"- validation_rows: {validation_rows}")
    print(f"- test_rows: {test_rows}")


def _read_checkpoint_global_step(checkpoint_dir: Path) -> int | None:
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.exists():
        return None

    try:
        with state_path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)
    except Exception:
        LOGGER.exception("Failed to read checkpoint state file: %s", state_path)
        return None

    value = state.get("global_step")
    return int(value) if isinstance(value, (int, float)) else None


def _find_latest_checkpoint(output_dir: Path) -> tuple[Path | None, int | None]:
    if not output_dir.exists():
        return None, None

    latest_checkpoint: Path | None = None
    latest_step = -1

    for candidate in output_dir.iterdir():
        if not candidate.is_dir():
            continue
        match = CHECKPOINT_PATTERN.match(candidate.name)
        if not match:
            continue
        step = int(match.group(1))
        if step > latest_step:
            latest_step = step
            latest_checkpoint = candidate

    if latest_checkpoint is None:
        return None, None

    state_step = _read_checkpoint_global_step(latest_checkpoint)
    return latest_checkpoint, state_step if state_step is not None else latest_step


def main() -> None:
    parser = argparse.ArgumentParser(description="Train transformer model with comparison reporting.")
    parser.add_argument("--diagnostic", action="store_true", help="Print runtime diagnostics before training.")
    parser.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="Print runtime diagnostics and exit without running training.",
    )
    parser.add_argument("--log-level", default="INFO", help="Log level for detailed training workflow tracing.")
    args = parser.parse_args()

    configure_logging(args.log_level)

    root = Path.cwd()
    LOGGER.info("Loading dataset/training/model/evaluation configuration files.")
    try:
        dataset_config = load_dataset_config(root / "configs" / "dataset.yaml")
        split_config = load_split_config_from_dataset(root / "configs" / "dataset.yaml")
        evaluation_config = load_evaluation_config(root / "configs" / "evaluation.yaml")
        model_config, training_config, tokenization_config = load_model_config(root / "configs" / "model.yaml")
    except Exception:
        LOGGER.exception("Failed during configuration loading.")
        raise

    LOGGER.info("Preparing dataset from source path: %s", dataset_config.source_path)
    try:
        dataset_frame, _summary = prepare_dataset(dataset_config)
    except Exception:
        LOGGER.exception("Failed during dataset preparation.")
        raise

    LOGGER.info("Splitting dataset into train/validation/test sets.")
    try:
        train_frame, validation_frame, test_frame, _split_summary = split_dataset(
            dataset_frame,
            split_config,
            label_column=dataset_config.label_column,
        )
    except Exception:
        LOGGER.exception("Failed during dataset split.")
        raise

    if args.diagnostic or args.diagnostic_only:
        _print_diagnostics(
            model_name=model_config.name,
            train_rows=int(train_frame.shape[0]),
            validation_rows=int(validation_frame.shape[0]),
            test_rows=int(test_frame.shape[0]),
        )
    if args.diagnostic_only:
        LOGGER.info("Diagnostic-only mode enabled. Exiting before model initialization/training.")
        return

    LOGGER.info("Running baseline metrics for comparison table.")
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
    try:
        baseline_trainer.fit(train_frame)
        baseline_test_metrics = baseline_trainer.evaluate(test_frame)
    except Exception:
        LOGGER.exception("Failed during baseline comparison run.")
        raise

    baseline_metrics = {
        "accuracy": baseline_test_metrics.accuracy,
        "precision": baseline_test_metrics.precision,
        "recall": baseline_test_metrics.recall,
        "f1": baseline_test_metrics.f1,
    }

    LOGGER.info("Initializing transformer trainer.")
    try:
        from src.training.transformer_trainer import TransformerTrainer
    except Exception:
        LOGGER.exception("Failed while importing TransformerTrainer dependencies.")
        raise

    checkpoint_dir = model_config.model_output_dir
    LOGGER.info("Checkpoint directory pattern: %s", checkpoint_dir / "checkpoint-*")
    latest_checkpoint, global_step = _find_latest_checkpoint(checkpoint_dir)
    if latest_checkpoint is not None:
        LOGGER.info(
            "Latest checkpoint found: %s | global_step=%s | training mode=resumed",
            latest_checkpoint,
            global_step,
        )
    else:
        LOGGER.info(
            "No checkpoint found under %s | global_step=0 | training mode=fresh",
            checkpoint_dir,
        )

    transformer_trainer = TransformerTrainer(
        model_config=model_config,
        training_config=training_config,
        evaluation_config=evaluation_config,
        tokenizer_config=tokenization_config,
        enable_mlflow=True,
    )
    try:
        transformer_output = transformer_trainer.train(
            train_frame,
            validation_frame,
            dataset_config.text_column,
            dataset_config.label_column,
            resume_from_checkpoint=latest_checkpoint,
        )
    except Exception:
        LOGGER.exception("Failed during transformer training.")
        raise

    LOGGER.info("Evaluating transformer predictions on test split.")
    try:
        transformer_metrics = transformer_trainer.evaluate_predictions(
            test_frame,
            dataset_config.text_column,
            dataset_config.label_column,
        )
        predicted_labels, confidences = transformer_trainer.predict_batch(
            test_frame,
            dataset_config.text_column,
        )
    except Exception:
        LOGGER.exception("Failed during transformer evaluation/prediction.")
        raise

    LOGGER.info("Writing evaluation metrics and error analysis artifacts.")
    try:
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
    except Exception:
        LOGGER.exception("Failed while persisting evaluation artifacts.")
        raise

    LOGGER.info("Writing model comparison artifact.")
    try:
        comparison_path = _write_model_comparison(
            baseline_metrics=baseline_metrics,
            transformer_metrics=transformer_metrics,
            output_path=evaluation_config.output_dir / "model_comparison.csv",
        )
    except Exception:
        LOGGER.exception("Failed while writing model comparison output.")
        raise

    print(f"Transformer training complete. Model saved to {transformer_output.model_dir}.")
    print(f"Comparison saved to {comparison_path}.")


if __name__ == "__main__":
    main()