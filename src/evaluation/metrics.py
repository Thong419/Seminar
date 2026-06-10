"""Classification metrics for the fake-news baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]


def compute_classification_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    positive_label: str = "fake",
) -> ClassificationMetrics:
    labels = [positive_label, "real" if positive_label == "fake" else positive_label]
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        confusion_matrix=matrix.astype(int).tolist(),
    )


def metrics_to_dict(metrics: ClassificationMetrics) -> dict[str, float | list[list[int]]]:
    return {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "confusion_matrix": metrics.confusion_matrix,
    }