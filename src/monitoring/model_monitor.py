"""Model health monitoring and dashboard summary generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
import json
from typing import Any

from src.monitoring.config import MonitoringConfig
from src.monitoring.confidence_monitor import ConfidenceReport
from src.monitoring.data_drift import DataDriftReport
from src.monitoring.prediction_drift import PredictionDriftReport


class HealthLevel(str, Enum):
    GREEN = "GREEN"
    YELLOW = "YELLOW"
    RED = "RED"


@dataclass(frozen=True, slots=True)
class ModelHealthReport:
    status: HealthLevel
    indicators: dict[str, float | str]
    alerts: list[str]

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


class ModelMonitor:
    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config

    def evaluate(
        self,
        data_drift_report: DataDriftReport,
        prediction_drift_report: PredictionDriftReport,
        confidence_report: ConfidenceReport,
        current_metrics: dict[str, float] | None = None,
        baseline_metrics: dict[str, float] | None = None,
    ) -> ModelHealthReport:
        indicators: dict[str, float | str] = {
            "data_drift_score": data_drift_report.drift_score,
            "prediction_drift_score": prediction_drift_report.drift_score,
            "confidence_drift_score": confidence_report.drift_score,
        }
        performance_drop = _performance_drop(current_metrics or {}, baseline_metrics or {})
        indicators["performance_drop"] = round(performance_drop, 4)

        alerts: list[str] = []
        score_vector = [
            data_drift_report.drift_score,
            prediction_drift_report.drift_score,
            confidence_report.drift_score,
            performance_drop,
        ]
        if any(score >= self.config.red_threshold for score in score_vector):
            status = HealthLevel.RED
            alerts.append("At least one monitoring signal crossed the red threshold.")
        elif any(score >= self.config.yellow_threshold for score in score_vector):
            status = HealthLevel.YELLOW
            alerts.append("At least one monitoring signal crossed the yellow threshold.")
        else:
            status = HealthLevel.GREEN

        if performance_drop >= self.config.performance_drop_threshold:
            alerts.append("Model performance is degrading relative to baseline metrics.")
            if status == HealthLevel.GREEN:
                status = HealthLevel.YELLOW

        return ModelHealthReport(status=status, indicators=indicators, alerts=alerts)

    def save_report(self, report: ModelHealthReport) -> Path:
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        with self.config.health_report_path.open("w", encoding="utf-8") as handle:
            json.dump(report.as_dict(), handle, indent=2)
        return self.config.health_report_path

    def build_dashboard_summary(
        self,
        data_drift_report: DataDriftReport,
        prediction_drift_report: PredictionDriftReport,
        confidence_report: ConfidenceReport,
        current_metrics: dict[str, float] | None = None,
        baseline_metrics: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        health_report = self.evaluate(
            data_drift_report=data_drift_report,
            prediction_drift_report=prediction_drift_report,
            confidence_report=confidence_report,
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
        )
        return {
            "health": health_report.as_dict(),
            "data_drift": data_drift_report.as_dict(),
            "prediction_drift": prediction_drift_report.as_dict(),
            "confidence": confidence_report.as_dict(),
        }


def _performance_drop(current_metrics: dict[str, float], baseline_metrics: dict[str, float]) -> float:
    if not current_metrics or not baseline_metrics:
        return 0.0
    tracked = [key for key in ("accuracy", "precision", "recall", "f1") if key in current_metrics and key in baseline_metrics]
    if not tracked:
        return 0.0
    drops = [max(0.0, baseline_metrics[key] - current_metrics[key]) for key in tracked]
    return sum(drops) / len(drops)
