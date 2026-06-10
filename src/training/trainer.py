"""Baseline training utilities for the TF-IDF + Logistic Regression model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config.pipeline import DatasetConfig, EvaluationConfig, TrainingConfig
from src.evaluation.evaluator import evaluate_predictions, save_evaluation_artifacts
from src.evaluation.metrics import ClassificationMetrics
from src.features.tfidf_vectorizer import build_tfidf_vectorizer


@dataclass(frozen=True, slots=True)
class TrainingOutcome:
    model_path: Path
    validation_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics


class BaselineTrainer:
    def __init__(
        self,
        dataset_config: DatasetConfig,
        training_config: TrainingConfig,
        evaluation_config: EvaluationConfig,
        enable_mlflow: bool = True,
    ) -> None:
        self.dataset_config = dataset_config
        self.training_config = training_config
        self.evaluation_config = evaluation_config
        self.enable_mlflow = enable_mlflow
        self._pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        tfidf = build_tfidf_vectorizer(self.training_config.tfidf)
        classifier = LogisticRegression(
            C=self.training_config.logistic_regression.C,
            max_iter=self.training_config.logistic_regression.max_iter,
            solver=self.training_config.logistic_regression.solver,
            class_weight=self.training_config.logistic_regression.class_weight,
            random_state=self.training_config.random_state,
        )
        return Pipeline([
            ("tfidf", tfidf),
            ("classifier", classifier),
        ])

    def fit(self, train_frame: pd.DataFrame) -> None:
        self._pipeline.fit(train_frame[self.dataset_config.text_column], train_frame[self.dataset_config.label_column])

    def predict(self, frame: pd.DataFrame) -> list[str]:
        predictions = self._pipeline.predict(frame[self.dataset_config.text_column])
        return [str(label) for label in predictions]

    def evaluate(self, frame: pd.DataFrame) -> ClassificationMetrics:
        predictions = self.predict(frame)
        return evaluate_predictions(
            y_true=frame[self.dataset_config.label_column].astype(str).tolist(),
            y_pred=predictions,
        )

    def save_model(self) -> Path:
        self.training_config.model_output_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.training_config.model_output_dir / "baseline_model.joblib"
        joblib.dump(self._pipeline, model_path)
        return model_path

    def _log_mlflow(
        self,
        train_metrics: ClassificationMetrics,
        validation_metrics: ClassificationMetrics,
        test_metrics: ClassificationMetrics,
        model_path: Path,
    ) -> None:
        if not self.enable_mlflow:
            return

        mlflow.set_tracking_uri(self.training_config.mlflow_tracking_uri)
        mlflow.set_experiment(self.training_config.experiment_name)

        with mlflow.start_run():
            mlflow.log_param("dataset_name", self.dataset_config.name)
            mlflow.log_param("dataset_version", self.training_config.dataset_version)
            mlflow.log_param("text_column", self.dataset_config.text_column)
            mlflow.log_param("label_column", self.dataset_config.label_column)
            mlflow.log_params(
                {
                    "tfidf_max_features": self.training_config.tfidf.max_features,
                    "tfidf_ngram_range": str(self.training_config.tfidf.ngram_range),
                    "tfidf_min_df": self.training_config.tfidf.min_df,
                    "tfidf_max_df": self.training_config.tfidf.max_df,
                    "logreg_C": self.training_config.logistic_regression.C,
                    "logreg_max_iter": self.training_config.logistic_regression.max_iter,
                    "logreg_solver": self.training_config.logistic_regression.solver,
                    "logreg_class_weight": self.training_config.logistic_regression.class_weight,
                }
            )
            mlflow.log_metrics(
                {
                    "validation_accuracy": validation_metrics.accuracy,
                    "validation_precision": validation_metrics.precision,
                    "validation_recall": validation_metrics.recall,
                    "validation_f1": validation_metrics.f1,
                    "test_accuracy": test_metrics.accuracy,
                    "test_precision": test_metrics.precision,
                    "test_recall": test_metrics.recall,
                    "test_f1": test_metrics.f1,
                }
            )
            mlflow.sklearn.log_model(self._pipeline, artifact_path="model")
            mlflow.log_artifact(str(model_path))

    def run(
        self,
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame,
        test_frame: pd.DataFrame,
    ) -> TrainingOutcome:
        self.fit(train_frame)
        validation_metrics = self.evaluate(validation_frame)
        test_metrics = self.evaluate(test_frame)
        model_path = self.save_model()
        self._log_mlflow(
            train_metrics=self.evaluate(train_frame),
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
            model_path=model_path,
        )
        save_evaluation_artifacts(test_metrics, self.evaluation_config)
        return TrainingOutcome(
            model_path=model_path,
            validation_metrics=validation_metrics,
            test_metrics=test_metrics,
        )