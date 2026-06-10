"""Retraining strategy generation for continual learning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
import json

from src.monitoring.config import MonitoringConfig
from src.monitoring.model_monitor import HealthLevel, ModelHealthReport


@dataclass(frozen=True, slots=True)
class RetrainingRecommendation:
    should_retrain: bool
    strategy: str
    priority: str
    reasons: list[str]
    next_review_date: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class RetrainingManager:
    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config

    def recommend(
        self,
        health_report: ModelHealthReport,
        current_date: date | None = None,
        last_retraining_date: date | None = None,
    ) -> RetrainingRecommendation:
        today = current_date or date.today()
        reasons: list[str] = []
        should_retrain = False
        strategy = "scheduled"
        priority = "low"
        drift_triggered = False

        if health_report.status == HealthLevel.RED:
            should_retrain = True
            strategy = "drift_triggered"
            priority = "high"
            drift_triggered = True
            reasons.extend(health_report.alerts or ["Health status is RED."])
        elif health_report.status == HealthLevel.YELLOW:
            should_retrain = True
            strategy = "drift_triggered"
            priority = "medium"
            drift_triggered = True
            reasons.extend(health_report.alerts or ["Health status is YELLOW."])

        if last_retraining_date is not None:
            days_since_last = (today - last_retraining_date).days
            if days_since_last >= self.config.retraining_cooldown_days:
                should_retrain = True
                if not drift_triggered:
                    strategy = "monthly"
                priority = max(priority, "medium", key=_priority_rank)
                reasons.append(f"It has been {days_since_last} days since the last retraining cycle.")

        next_review_date = (today + timedelta(days=self.config.retraining_cooldown_days)).isoformat()
        if not reasons:
            reasons.append("Current monitoring signals are within acceptable limits.")

        return RetrainingRecommendation(
            should_retrain=should_retrain,
            strategy=strategy,
            priority=priority,
            reasons=reasons,
            next_review_date=next_review_date,
        )

    def save_recommendation(self, recommendation: RetrainingRecommendation) -> Path:
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.artifact_dir / "retraining_recommendation.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(recommendation.as_dict(), handle, indent=2)
        return path


def _priority_rank(value: str) -> int:
    order = {"low": 0, "medium": 1, "high": 2}
    return order.get(value, 0)
