"""Configuration for monitoring, drift detection, and retraining policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class MonitoringConfig:
    artifact_dir: Path = Path("artifacts/monitoring")
    prediction_log_path: Path = Path("artifacts/monitoring/predictions.csv")
    drift_report_path: Path = Path("artifacts/monitoring/drift_report.json")
    confidence_report_path: Path = Path("artifacts/monitoring/confidence_report.json")
    health_report_path: Path = Path("artifacts/monitoring/health_report.json")
    confidence_plot_path: Path = Path("artifacts/monitoring/confidence_trend.png")
    prediction_plot_path: Path = Path("artifacts/monitoring/prediction_distribution.png")
    drift_plot_path: Path = Path("artifacts/monitoring/drift_trend.png")
    data_drift_threshold: float = 0.20
    prediction_drift_threshold: float = 0.20
    confidence_drift_threshold: float = 0.10
    performance_drop_threshold: float = 0.05
    yellow_threshold: float = 0.40
    red_threshold: float = 0.70
    monthly_retraining_day: int = 1
    retraining_cooldown_days: int = 30
    drift_trigger_threshold: float = 0.60
    confidence_bins: int = 10
    reference_window_size: int = 500
    production_window_size: int = 500


def load_monitoring_config(path: Path) -> MonitoringConfig:
    if not path.exists():
        return MonitoringConfig()

    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("Monitoring configuration must be a mapping.")

    thresholds = data.get("thresholds", {}) if isinstance(data.get("thresholds", {}), dict) else {}
    retraining = data.get("retraining", {}) if isinstance(data.get("retraining", {}), dict) else {}
    outputs = data.get("outputs", {}) if isinstance(data.get("outputs", {}), dict) else {}

    return MonitoringConfig(
        artifact_dir=Path(outputs.get("artifact_dir", MonitoringConfig.artifact_dir)),
        prediction_log_path=Path(outputs.get("prediction_log_path", MonitoringConfig.prediction_log_path)),
        drift_report_path=Path(outputs.get("drift_report_path", MonitoringConfig.drift_report_path)),
        confidence_report_path=Path(outputs.get("confidence_report_path", MonitoringConfig.confidence_report_path)),
        health_report_path=Path(outputs.get("health_report_path", MonitoringConfig.health_report_path)),
        confidence_plot_path=Path(outputs.get("confidence_plot_path", MonitoringConfig.confidence_plot_path)),
        prediction_plot_path=Path(outputs.get("prediction_plot_path", MonitoringConfig.prediction_plot_path)),
        drift_plot_path=Path(outputs.get("drift_plot_path", MonitoringConfig.drift_plot_path)),
        data_drift_threshold=float(thresholds.get("data_drift_threshold", MonitoringConfig.data_drift_threshold)),
        prediction_drift_threshold=float(
            thresholds.get("prediction_drift_threshold", MonitoringConfig.prediction_drift_threshold)
        ),
        confidence_drift_threshold=float(
            thresholds.get("confidence_drift_threshold", MonitoringConfig.confidence_drift_threshold)
        ),
        performance_drop_threshold=float(
            thresholds.get("performance_drop_threshold", MonitoringConfig.performance_drop_threshold)
        ),
        yellow_threshold=float(thresholds.get("yellow_threshold", MonitoringConfig.yellow_threshold)),
        red_threshold=float(thresholds.get("red_threshold", MonitoringConfig.red_threshold)),
        monthly_retraining_day=int(retraining.get("monthly_retraining_day", MonitoringConfig.monthly_retraining_day)),
        retraining_cooldown_days=int(
            retraining.get("retraining_cooldown_days", MonitoringConfig.retraining_cooldown_days)
        ),
        drift_trigger_threshold=float(
            retraining.get("drift_trigger_threshold", MonitoringConfig.drift_trigger_threshold)
        ),
        confidence_bins=int(data.get("confidence_bins", MonitoringConfig.confidence_bins)),
        reference_window_size=int(data.get("reference_window_size", MonitoringConfig.reference_window_size)),
        production_window_size=int(data.get("production_window_size", MonitoringConfig.production_window_size)),
    )
