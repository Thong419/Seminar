"""Trainer API wrapper for fine-tuning transformer classifiers."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from transformers import DataCollatorWithPadding, EvalPrediction, Trainer, TrainingArguments

from src.config.pipeline import EvaluationConfig, ModelConfig, TokenizationConfig, TransformerTrainingConfig
from src.evaluation.error_analysis import build_error_analysis_frame, save_error_analysis
from src.models.transformer_model import build_sequence_classification_model, build_tokenizer


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class TransformerTrainOutput:
    model_dir: Path
    metrics: dict[str, float | list[list[int]]]


class DatasetLabelEncoder:
    def __init__(self, label_names: tuple[str, str]) -> None:
        self.label_to_id = {label: index for index, label in enumerate(label_names)}
        self.id_to_label = {index: label for index, label in enumerate(label_names)}

    def encode(self, label: str) -> int:
        return self.label_to_id[label]

    def decode(self, label_id: int) -> str:
        return self.id_to_label[label_id]


class NewsDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        text_column: str,
        label_column: str,
        encoder: DatasetLabelEncoder,
        tokenizer,
        max_length: int,
        truncation: bool,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.text_column = text_column
        self.label_column = label_column
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation

    def __len__(self) -> int:
        return int(self.frame.shape[0])

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        encoded = self.tokenizer(
            str(row[self.text_column]),
            truncation=self.truncation,
            max_length=self.max_length,
            padding=False,
        )
        encoded["labels"] = self.encoder.encode(str(row[self.label_column]))
        return encoded


class TransformerTrainer:
    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TransformerTrainingConfig,
        evaluation_config: EvaluationConfig,
        tokenizer_config: TokenizationConfig,
        enable_mlflow: bool = True,
    ) -> None:
        LOGGER.info("Initializing TransformerTrainer for model=%s", model_config.name)
        self.model_config = model_config
        self.training_config = training_config
        self.evaluation_config = evaluation_config
        self.tokenizer_config = tokenizer_config
        self.enable_mlflow = enable_mlflow

        try:
            LOGGER.info(
                "Loading tokenizer tokenizer_name=%s use_fast=%s",
                model_config.tokenizer_name,
                model_config.use_fast_tokenizer,
            )
            self.tokenizer = build_tokenizer(model_config)
        except Exception:
            LOGGER.exception("Tokenizer loading failed for tokenizer_name=%s", model_config.tokenizer_name)
            raise

        try:
            LOGGER.info(
                "Loading sequence classification model model_name=%s num_labels=%s",
                model_config.name,
                model_config.num_labels,
            )
            self.model = build_sequence_classification_model(model_config)
        except Exception:
            LOGGER.exception("Model loading failed for model_name=%s", model_config.name)
            raise

        self.device = self._detect_device()
        LOGGER.info("Selected device=%s cuda_available=%s", self.device, self.device.startswith("cuda"))

        self.encoder = DatasetLabelEncoder(model_config.label_names)
        self.model_dir = self.model_config.model_output_dir

    @staticmethod
    def _detect_device() -> str:
        import torch

        if torch.cuda.is_available():
            return f"cuda:{torch.cuda.current_device()}"
        return "cpu"

    def clone_for_output_dir(self, output_dir: Path) -> "TransformerTrainer":
        cloned_model_config = ModelConfig(
            name=self.model_config.name,
            max_length=self.model_config.max_length,
            num_labels=self.model_config.num_labels,
            model_output_dir=output_dir,
            label_names=self.model_config.label_names,
            tokenizer_name=self.model_config.tokenizer_name,
            use_fast_tokenizer=self.model_config.use_fast_tokenizer,
        )
        return TransformerTrainer(
            model_config=cloned_model_config,
            training_config=self.training_config,
            evaluation_config=self.evaluation_config,
            tokenizer_config=self.tokenizer_config,
            enable_mlflow=self.enable_mlflow,
        )

    def _metric_function(self, eval_prediction: EvalPrediction) -> dict[str, float | list[list[int]]]:
        logits = eval_prediction.predictions
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)
        labels = eval_prediction.label_ids
        return {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, zero_division=0)),
            "recall": float(recall_score(labels, predictions, zero_division=0)),
            "f1": float(f1_score(labels, predictions, zero_division=0)),
            "confusion_matrix": confusion_matrix(labels, predictions).astype(int).tolist(),
        }

    def _build_training_arguments(self) -> TrainingArguments:
        LOGGER.info(
            "Creating TrainingArguments output_dir=%s batch_size=%s lr=%s epochs=%s",
            self.model_dir,
            self.training_config.batch_size,
            self.training_config.learning_rate,
            self.training_config.epochs,
        )
        try:
            return TrainingArguments(
                output_dir=str(self.model_dir),
                learning_rate=self.training_config.learning_rate,
                per_device_train_batch_size=self.training_config.batch_size,
                per_device_eval_batch_size=self.training_config.batch_size,
                num_train_epochs=self.training_config.epochs,
                weight_decay=self.training_config.weight_decay,
                warmup_ratio=self.training_config.warmup_ratio,
                evaluation_strategy=self.training_config.evaluation_strategy,
                save_strategy=self.training_config.save_strategy,
                load_best_model_at_end=self.training_config.load_best_model_at_end,
                metric_for_best_model=self.training_config.metric_for_best_model,
                greater_is_better=self.training_config.greater_is_better,
                gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
                seed=self.training_config.seed,
                report_to=["none"],
                remove_unused_columns=False,
            )
        except Exception:
            LOGGER.exception("TrainingArguments creation failed.")
            raise

    def train(
        self,
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame,
        text_column: str,
        label_column: str,
        resume_from_checkpoint: Path | None = None,
    ) -> TransformerTrainOutput:
        LOGGER.info(
            "Preparing transformer datasets train_rows=%s validation_rows=%s text_column=%s label_column=%s",
            int(train_frame.shape[0]),
            int(validation_frame.shape[0]),
            text_column,
            label_column,
        )
        try:
            tokenized_train = NewsDataset(
                train_frame,
                text_column,
                label_column,
                self.encoder,
                self.tokenizer,
                self.model_config.max_length,
                self.tokenizer_config.truncation,
            )
            tokenized_validation = NewsDataset(
                validation_frame,
                text_column,
                label_column,
                self.encoder,
                self.tokenizer,
                self.model_config.max_length,
                self.tokenizer_config.truncation,
            )
        except Exception:
            LOGGER.exception("Dataset preparation for transformer training failed.")
            raise

        LOGGER.info("Creating Hugging Face Trainer instance.")
        try:
            trainer = Trainer(
                model=self.model,
                args=self._build_training_arguments(),
                train_dataset=tokenized_train,
                eval_dataset=tokenized_validation,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer, pad_to_multiple_of=None),
                compute_metrics=self._metric_function,
            )
        except Exception:
            LOGGER.exception("Trainer initialization failed.")
            raise

        training_mode = "resumed" if resume_from_checkpoint is not None else "fresh"
        if resume_from_checkpoint is not None:
            LOGGER.info(
                "Starting trainer.train() in %s mode with checkpoint=%s",
                training_mode,
                resume_from_checkpoint,
            )
        else:
            LOGGER.info("Starting trainer.train() in %s mode", training_mode)
        try:
            trainer.train(
                resume_from_checkpoint=str(resume_from_checkpoint) if resume_from_checkpoint is not None else None
            )
        except Exception:
            LOGGER.exception("trainer.train() failed.")
            raise

        LOGGER.info("Running trainer.evaluate()")
        try:
            metrics = trainer.evaluate()
        except Exception:
            LOGGER.exception("trainer.evaluate() failed.")
            raise

        model_dir = Path(trainer.args.output_dir)
        model_dir.mkdir(parents=True, exist_ok=True)

        LOGGER.info("Saving trained model/tokenizer artifacts to %s", model_dir)
        trainer.save_model(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        self.model.save_pretrained(model_dir)
        self._persist_metadata(model_dir)
        self._log_mlflow(trainer, metrics, model_dir)
        return TransformerTrainOutput(model_dir=model_dir, metrics=metrics)

    def evaluate_predictions(
        self,
        frame: pd.DataFrame,
        text_column: str,
        label_column: str,
    ) -> dict[str, float | list[list[int]]]:
        LOGGER.info("Preparing dataset for predict/eval rows=%s", int(frame.shape[0]))
        dataset = NewsDataset(
            frame,
            text_column,
            label_column,
            self.encoder,
            self.tokenizer,
            self.model_config.max_length,
            self.tokenizer_config.truncation,
        )
        try:
            trainer = Trainer(
                model=self.model,
                args=self._build_training_arguments(),
                eval_dataset=dataset,
                tokenizer=self.tokenizer,
                data_collator=DataCollatorWithPadding(tokenizer=self.tokenizer),
                compute_metrics=self._metric_function,
            )
        except Exception:
            LOGGER.exception("Trainer initialization for evaluate_predictions failed.")
            raise

        try:
            evaluation = trainer.predict(dataset)
        except Exception:
            LOGGER.exception("trainer.predict() failed during evaluate_predictions.")
            raise

        metrics = self._metric_function(EvalPrediction(predictions=evaluation.predictions, label_ids=evaluation.label_ids))
        return metrics

    def predict_batch(self, frame: pd.DataFrame, text_column: str) -> tuple[list[str], list[float]]:
        LOGGER.info("Running predict_batch rows=%s", int(frame.shape[0]))
        texts = frame[text_column].astype(str).tolist()
        if not texts:
            return [], []

        # Use a modest batch size for inference to avoid OOM; fall back to training batch size
        batch_size = max(1, getattr(self.training_config, "batch_size", 16))

        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        LOGGER.info("predict_batch using device=%s cuda_available=%s", device, torch.cuda.is_available())
        self.model.to(device)
        self.model.eval()

        labels: list[str] = []
        confidences: list[float] = []

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                encoded = self.tokenizer(
                    batch_texts,
                    truncation=self.tokenizer_config.truncation,
                    padding=True,
                    max_length=self.model_config.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}

                outputs = self.model(**encoded)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                batch_conf, batch_label_ids = probabilities.max(dim=-1)

                labels.extend([self.model.config.id2label[int(idx)] for idx in batch_label_ids.cpu().tolist()])
                confidences.extend(batch_conf.cpu().tolist())

                # free intermediate tensors
                del encoded, outputs, logits, probabilities, batch_conf, batch_label_ids
                if device.type == "cuda":
                    torch.cuda.empty_cache()

        return labels, confidences

    def save_error_analysis(
        self,
        frame: pd.DataFrame,
        text_column: str,
        label_column: str,
        output_path: Path | None = None,
    ) -> Path:
        labels, confidences = self.predict_batch(frame, text_column)
        analysis_frame = build_error_analysis_frame(
            texts=frame[text_column].astype(str).tolist(),
            y_true=frame[label_column].astype(str).tolist(),
            y_pred=labels,
            confidences=confidences,
        )
        return save_error_analysis(
            analysis_frame,
            output_path or self.evaluation_config.output_dir / "error_analysis.csv",
        )

    def _persist_metadata(self, model_dir: Path) -> None:
        metadata = {
            "model_name": self.model_config.name,
            "tokenizer_name": self.model_config.tokenizer_name,
            "max_length": self.model_config.max_length,
            "label_names": list(self.model_config.label_names),
            "training": {
                "batch_size": self.training_config.batch_size,
                "learning_rate": self.training_config.learning_rate,
                "epochs": self.training_config.epochs,
                "weight_decay": self.training_config.weight_decay,
            },
        }
        with (model_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2)

    def _log_mlflow(self, trainer: Trainer, metrics: dict[str, float | list[list[int]]], model_dir: Path) -> None:
        if not self.enable_mlflow:
            return

        mlflow.set_experiment(f"transformer_{self.model_config.name.replace('/', '_')}")
        with mlflow.start_run():
            mlflow.log_param("model_name", self.model_config.name)
            mlflow.log_param("tokenizer_name", self.model_config.tokenizer_name)
            mlflow.log_param("max_length", self.model_config.max_length)
            mlflow.log_param("batch_size", self.training_config.batch_size)
            mlflow.log_param("learning_rate", self.training_config.learning_rate)
            mlflow.log_param("epochs", self.training_config.epochs)
            mlflow.log_param("weight_decay", self.training_config.weight_decay)
            mlflow.log_metrics({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.log_artifacts(str(model_dir))