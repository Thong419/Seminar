"""Evaluation orchestration and artifact persistence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd

from src.config.pipeline import EvaluationConfig
from src.evaluation.metrics import ClassificationMetrics, compute_classification_metrics, metrics_to_dict


@dataclass(frozen=True, slots=True)
class EvaluationArtifacts:
    metrics_path: Path
    confusion_matrix_path: Path


def evaluate_predictions(y_true: list[str], y_pred: list[str]) -> ClassificationMetrics:
    return compute_classification_metrics(y_true=y_true, y_pred=y_pred)


def save_evaluation_artifacts(
    metrics: ClassificationMetrics,
    config: EvaluationConfig,
) -> EvaluationArtifacts:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_dir / "metrics.json"
    matrix_path = config.output_dir / "confusion_matrix.csv"

    if config.save_metrics_json:
        with metrics_path.open("w", encoding="utf-8") as handle:
            json.dump(metrics_to_dict(metrics), handle, indent=2)

    if config.save_confusion_matrix:
        pd.DataFrame(metrics.confusion_matrix, columns=["pred_fake", "pred_real"]).to_csv(
            matrix_path,
            index=False,
        )

    return EvaluationArtifacts(metrics_path=metrics_path, confusion_matrix_path=matrix_path)