"""Prediction drift monitoring for label and trust distributions."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import json
from typing import Sequence

from src.monitoring.config import MonitoringConfig
from src.monitoring.visualization import save_bar_chart_png


@dataclass(frozen=True, slots=True)
class PredictionDriftReport:
    reference_fake_ratio: float
    production_fake_ratio: float
    decision_distribution: dict[str, int]
    trust_score_mean: float
    trust_score_std: float
    drift_score: float
    status: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class PredictionDriftMonitor:
    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config

    def detect(
        self,
        reference_predictions: Sequence[str],
        production_predictions: Sequence[str],
        production_trust_scores: Sequence[int],
    ) -> PredictionDriftReport:
        reference_distribution = Counter(label.lower() for label in reference_predictions)
        production_distribution = Counter(label.lower() for label in production_predictions)
        reference_total = sum(reference_distribution.values()) or 1
        production_total = sum(production_distribution.values()) or 1
        reference_fake_ratio = reference_distribution.get("fake", 0) / reference_total
        production_fake_ratio = production_distribution.get("fake", 0) / production_total
        trust_mean = sum(production_trust_scores) / max(len(production_trust_scores), 1)
        trust_variance = sum((score - trust_mean) ** 2 for score in production_trust_scores) / max(
            len(production_trust_scores),
            1,
        )
        decision_distribution = dict(production_distribution)
        decision_shift = _distribution_shift(reference_distribution, production_distribution)
        trust_shift = abs(trust_mean - 50.0) / 100.0
        drift_score = max(
            0.0,
            min(1.0, (decision_shift + abs(production_fake_ratio - reference_fake_ratio) + trust_shift) / 3.0),
        )
        status = _status_from_score(drift_score, self.config)
        return PredictionDriftReport(
            reference_fake_ratio=round(reference_fake_ratio, 4),
            production_fake_ratio=round(production_fake_ratio, 4),
            decision_distribution=decision_distribution,
            trust_score_mean=round(trust_mean, 4),
            trust_score_std=round(trust_variance ** 0.5, 4),
            drift_score=round(drift_score, 4),
            status=status,
        )

    def save_report(self, report: PredictionDriftReport) -> Path:
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.artifact_dir / "prediction_drift_report.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(report.as_dict(), handle, indent=2)
        return path

    def save_distribution_plot(self, report: PredictionDriftReport) -> Path:
        labels = list(report.decision_distribution.keys())
        values = list(report.decision_distribution.values())
        return save_bar_chart_png(labels, values, self.config.prediction_plot_path, "Prediction Distribution")


def _distribution_shift(reference: Counter[str], production: Counter[str]) -> float:
    keys = set(reference) | set(production)
    if not keys:
        return 0.0
    ref_total = sum(reference.values()) or 1
    prod_total = sum(production.values()) or 1
    return 0.5 * sum(
        abs((reference.get(key, 0) / ref_total) - (production.get(key, 0) / prod_total))
        for key in keys
    )


def _status_from_score(score: float, config: MonitoringConfig) -> str:
    if score >= config.red_threshold:
        return "RED"
    if score >= config.yellow_threshold:
        return "YELLOW"
    return "GREEN"
