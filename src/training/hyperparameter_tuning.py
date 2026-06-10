"""Optuna-based hyperparameter tuning for the transformer model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import optuna
import pandas as pd

from src.config.pipeline import EvaluationConfig, ModelConfig, TokenizationConfig, TransformerTrainingConfig
from src.config.pipeline import load_dataset_config, load_evaluation_config, load_model_config, load_split_config_from_dataset
from src.data.dataset import prepare_dataset
from src.data.splitter import split_dataset
from src.training.transformer_trainer import TransformerTrainer


@dataclass(frozen=True, slots=True)
class TuningConfig:
    n_trials: int = 10
    timeout_seconds: int | None = None
    study_name: str = "transformer_tuning"
    direction: str = "maximize"


class HyperparameterTuner:
    def __init__(
        self,
        model_config: ModelConfig,
        base_training_config: TransformerTrainingConfig,
        evaluation_config: EvaluationConfig,
        tokenizer_config: TokenizationConfig,
        text_column: str,
        label_column: str,
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame,
        enable_mlflow: bool = True,
    ) -> None:
        self.model_config = model_config
        self.base_training_config = base_training_config
        self.evaluation_config = evaluation_config
        self.tokenizer_config = tokenizer_config
        self.text_column = text_column
        self.label_column = label_column
        self.train_frame = train_frame
        self.validation_frame = validation_frame
        self.enable_mlflow = enable_mlflow

    def _objective(self, trial: optuna.Trial) -> float:
        training_config = TransformerTrainingConfig(
            batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
            learning_rate=trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
            epochs=self.base_training_config.epochs,
            weight_decay=trial.suggest_float("weight_decay", 0.0, 0.1),
            warmup_ratio=self.base_training_config.warmup_ratio,
            evaluation_strategy=self.base_training_config.evaluation_strategy,
            save_strategy=self.base_training_config.save_strategy,
            load_best_model_at_end=self.base_training_config.load_best_model_at_end,
            metric_for_best_model=self.base_training_config.metric_for_best_model,
            greater_is_better=self.base_training_config.greater_is_better,
            gradient_accumulation_steps=self.base_training_config.gradient_accumulation_steps,
            seed=self.base_training_config.seed,
        )
        trainer = TransformerTrainer(
            model_config=self.model_config,
            training_config=training_config,
            evaluation_config=self.evaluation_config,
            tokenizer_config=self.tokenizer_config,
            enable_mlflow=False,
        )
        trainer = trainer.clone_for_output_dir(self.model_config.model_output_dir / f"trial_{trial.number}")
        outcome = trainer.train(
            self.train_frame,
            self.validation_frame,
            self.text_column,
            self.label_column,
        )
        return float(outcome.metrics["f1"])

    def tune(self, tuning_config: TuningConfig) -> optuna.Study:
        study = optuna.create_study(direction=tuning_config.direction, study_name=tuning_config.study_name)
        study.optimize(self._objective, n_trials=tuning_config.n_trials, timeout=tuning_config.timeout_seconds)
        return study


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Tune transformer hyperparameters with Optuna.")
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--timeout-seconds", type=int, default=None)
    args = parser.parse_args()

    root = Path.cwd()
    dataset_config = load_dataset_config(root / "configs" / "dataset.yaml")
    split_config = load_split_config_from_dataset(root / "configs" / "dataset.yaml")
    evaluation_config = load_evaluation_config(root / "configs" / "evaluation.yaml")
    model_config, training_config, tokenization_config = load_model_config(root / "configs" / "model.yaml")

    dataset_frame, _ = prepare_dataset(dataset_config)
    train_frame, validation_frame, _, _ = split_dataset(
        dataset_frame,
        split_config,
        label_column=dataset_config.label_column,
    )

    tuner = HyperparameterTuner(
        model_config=model_config,
        base_training_config=training_config,
        evaluation_config=evaluation_config,
        tokenizer_config=tokenization_config,
        text_column=dataset_config.text_column,
        label_column=dataset_config.label_column,
        train_frame=train_frame,
        validation_frame=validation_frame,
        enable_mlflow=False,
    )
    study = tuner.tune(TuningConfig(n_trials=args.n_trials, timeout_seconds=args.timeout_seconds))
    print(f"Best trial: {study.best_trial.number}")
    print(f"Best value: {study.best_value}")
    print(f"Best params: {study.best_trial.params}")


if __name__ == "__main__":
    main()
