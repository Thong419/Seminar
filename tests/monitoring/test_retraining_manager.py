from __future__ import annotations

from datetime import date

from src.monitoring.config import MonitoringConfig
from src.monitoring.model_monitor import HealthLevel, ModelHealthReport
from src.monitoring.retraining_manager import RetrainingManager


def test_retraining_manager_triggers_on_red_health(tmp_path) -> None:
    config = MonitoringConfig(artifact_dir=tmp_path, retraining_cooldown_days=30)
    manager = RetrainingManager(config)
    health = ModelHealthReport(
        status=HealthLevel.RED,
        indicators={"data_drift_score": 0.91},
        alerts=["Data drift is extreme."],
    )

    recommendation = manager.recommend(health, current_date=date(2026, 6, 10), last_retraining_date=date(2026, 5, 1))

    assert recommendation.should_retrain is True
    assert recommendation.strategy == "drift_triggered"
    assert recommendation.priority == "high"


def test_retraining_manager_keeps_schedule_when_healthy(tmp_path) -> None:
    config = MonitoringConfig(artifact_dir=tmp_path, retraining_cooldown_days=30)
    manager = RetrainingManager(config)
    health = ModelHealthReport(status=HealthLevel.GREEN, indicators={}, alerts=[])

    recommendation = manager.recommend(health, current_date=date(2026, 6, 10), last_retraining_date=date(2026, 6, 1))

    assert recommendation.should_retrain is False
    assert recommendation.priority == "low"
