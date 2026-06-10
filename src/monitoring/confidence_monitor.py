"""Confidence drift monitoring for prediction certainty over time."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
from statistics import mean
from typing import Sequence

from src.monitoring.config import MonitoringConfig
from src.monitoring.visualization import save_line_chart_png


@dataclass(frozen=True, slots=True)
class ConfidenceReport:
    average_confidence: float
    confidence_histogram: dict[str, int]
    confidence_trend: list[float]
    confidence_slope: float
    drift_score: float
    status: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class ConfidenceMonitor:
    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config

    def detect(self, confidences: Sequence[float]) -> ConfidenceReport:
        history = [max(0.0, min(1.0, float(value))) for value in confidences]
        average_confidence = mean(history) if history else 0.0
        confidence_slope = _linear_slope(history)
        histogram = _histogram(history, self.config.confidence_bins)
        drift_score = max(abs(average_confidence - 0.5), abs(confidence_slope))
        status = _status_from_score(drift_score, self.config)
        return ConfidenceReport(
            average_confidence=round(average_confidence, 4),
            confidence_histogram=histogram,
            confidence_trend=history,
            confidence_slope=round(confidence_slope, 4),
            drift_score=round(drift_score, 4),
            status=status,
        )

    def save_report(self, report: ConfidenceReport) -> Path:
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        with self.config.confidence_report_path.open("w", encoding="utf-8") as handle:
            json.dump(report.as_dict(), handle, indent=2)
        return self.config.confidence_report_path

    def save_trend_plot(self, report: ConfidenceReport) -> Path:
        return save_line_chart_png(report.confidence_trend, self.config.confidence_plot_path, "Confidence Trend", "Confidence")


def _histogram(values: Sequence[float], bins: int) -> dict[str, int]:
    if bins <= 0:
        bins = 10
    counts = [0 for _ in range(bins)]
    for value in values:
        index = min(bins - 1, int(value * bins))
        counts[index] += 1
    return {f"{index / bins:.1f}-{(index + 1) / bins:.1f}": count for index, count in enumerate(counts)}


def _linear_slope(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2.0
    y_mean = mean(values)
    numerator = sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values))
    denominator = sum((index - x_mean) ** 2 for index in range(len(values))) or 1.0
    return numerator / denominator


def _status_from_score(score: float, config: MonitoringConfig) -> str:
    if score >= config.red_threshold:
        return "RED"
    if score >= config.yellow_threshold:
        return "YELLOW"
    return "GREEN"
