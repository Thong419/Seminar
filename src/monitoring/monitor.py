"""Monitoring service façade for logging, drift detection, and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

from src.monitoring.config import MonitoringConfig, load_monitoring_config
from src.monitoring.confidence_monitor import ConfidenceMonitor, ConfidenceReport
from src.monitoring.data_drift import DataDriftMonitor, DataDriftReport
from src.monitoring.model_monitor import ModelHealthReport, ModelMonitor
from src.monitoring.prediction_drift import PredictionDriftMonitor, PredictionDriftReport
from src.monitoring.prediction_logger import PredictionLogger, PredictionLogEntry
from src.monitoring.retraining_manager import RetrainingManager, RetrainingRecommendation


@dataclass(frozen=True, slots=True)
class MonitoringBundle:
    data_drift: DataDriftReport
    prediction_drift: PredictionDriftReport
    confidence: ConfidenceReport
    health: ModelHealthReport
    retraining: RetrainingRecommendation


class MonitoringService:
    def __init__(self, config: MonitoringConfig | None = None) -> None:
        self.config = config or MonitoringConfig()
        self.logger = PredictionLogger(self.config.prediction_log_path)
        self.data_drift_monitor = DataDriftMonitor(self.config)
        self.prediction_drift_monitor = PredictionDriftMonitor(self.config)
        self.confidence_monitor = ConfidenceMonitor(self.config)
        self.model_monitor = ModelMonitor(self.config)
        self.retraining_manager = RetrainingManager(self.config)

    @classmethod
    def from_config_path(cls, config_path: Path) -> "MonitoringService":
        return cls(load_monitoring_config(config_path))

    def log_prediction(
        self,
        prediction: str,
        confidence: float,
        trust_score: int,
        article_text: str,
        request_id: str | None = None,
        endpoint: str = "predict",
    ) -> PredictionLogEntry:
        return self.logger.log_prediction(
            prediction=prediction,
            confidence=confidence,
            trust_score=trust_score,
            article_length=len(article_text.split()),
            request_id=request_id,
            endpoint=endpoint,
        )

    def run_monitoring(
        self,
        reference_texts: Sequence[str],
        production_texts: Sequence[str],
        reference_predictions: Sequence[str],
        production_predictions: Sequence[str],
        production_confidences: Sequence[float],
        production_trust_scores: Sequence[int],
        current_metrics: dict[str, float] | None = None,
        baseline_metrics: dict[str, float] | None = None,
        last_retraining_date: date | None = None,
    ) -> MonitoringBundle:
        data_drift = self.data_drift_monitor.detect(reference_texts, production_texts)
        prediction_drift = self.prediction_drift_monitor.detect(
            reference_predictions=reference_predictions,
            production_predictions=production_predictions,
            production_trust_scores=production_trust_scores,
        )
        confidence = self.confidence_monitor.detect(production_confidences)
        health = self.model_monitor.evaluate(
            data_drift_report=data_drift,
            prediction_drift_report=prediction_drift,
            confidence_report=confidence,
            current_metrics=current_metrics,
            baseline_metrics=baseline_metrics,
        )
        retraining = self.retraining_manager.recommend(
            health_report=health,
            last_retraining_date=last_retraining_date,
        )

        self.data_drift_monitor.save_report(data_drift)
        self.prediction_drift_monitor.save_report(prediction_drift)
        self.confidence_monitor.save_report(confidence)
        self.model_monitor.save_report(health)
        self.retraining_manager.save_recommendation(retraining)
        self.data_drift_monitor.save_trend_plot([data_drift.drift_score])
        self.prediction_drift_monitor.save_distribution_plot(prediction_drift)
        self.confidence_monitor.save_trend_plot(confidence)

        return MonitoringBundle(
            data_drift=data_drift,
            prediction_drift=prediction_drift,
            confidence=confidence,
            health=health,
            retraining=retraining,
        )


def load_monitoring_service(config_path: Path | None = None) -> MonitoringService:
    if config_path is None:
        return MonitoringService()
    return MonitoringService.from_config_path(config_path)
