"""Bias analysis framework for source, topic, and label imbalance."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
import json
from statistics import mean
from typing import Iterable, Sequence


@dataclass(frozen=True, slots=True)
class BiasAnalysisReport:
    sample_size: int
    source_distribution: dict[str, int]
    topic_distribution: dict[str, int]
    label_distribution: dict[str, int]
    source_imbalance: float
    topic_imbalance: float
    label_imbalance: float
    bias_risk_score: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class BiasAnalyzer:
    def analyze(self, records: Sequence[dict[str, str]]) -> BiasAnalysisReport:
        sample_size = len(records)
        source_distribution = Counter(record.get("source", "unknown") for record in records)
        topic_distribution = Counter(record.get("topic", "unknown") for record in records)
        label_distribution = Counter(record.get("label", "unknown") for record in records)

        source_imbalance = _imbalance_score(source_distribution)
        topic_imbalance = _imbalance_score(topic_distribution)
        label_imbalance = _imbalance_score(label_distribution)
        bias_risk_score = mean([source_imbalance, topic_imbalance, label_imbalance]) if records else 0.0

        return BiasAnalysisReport(
            sample_size=sample_size,
            source_distribution=dict(source_distribution),
            topic_distribution=dict(topic_distribution),
            label_distribution=dict(label_distribution),
            source_imbalance=round(source_imbalance, 4),
            topic_imbalance=round(topic_imbalance, 4),
            label_imbalance=round(label_imbalance, 4),
            bias_risk_score=round(bias_risk_score, 4),
        )

    def save_report(self, report: BiasAnalysisReport, output_path: Path = Path("artifacts/fairness/bias_report.json")) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(report.as_dict(), handle, indent=2)
        return output_path


def _imbalance_score(distribution: Counter[str]) -> float:
    total = sum(distribution.values())
    if total == 0:
        return 0.0
    proportions = [count / total for count in distribution.values()]
    if len(proportions) <= 1:
        return 0.0
    ideal = 1.0 / len(proportions)
    return 0.5 * sum(abs(proportion - ideal) for proportion in proportions)
