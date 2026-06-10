"""Data drift detection for article length, vocabulary, and token distribution."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from math import log2
from pathlib import Path
import json
import re
from typing import Sequence

from src.monitoring.config import MonitoringConfig
from src.monitoring.visualization import save_line_chart_png


TOKEN_PATTERN = re.compile(r"[a-z0-9']+")


@dataclass(frozen=True, slots=True)
class DataDistribution:
    sample_size: int
    article_length_mean: float
    article_length_std: float
    vocabulary_size: int
    token_frequencies: dict[str, float]


@dataclass(frozen=True, slots=True)
class DataDriftReport:
    reference_sample_size: int
    production_sample_size: int
    length_mean_shift: float
    vocabulary_js_divergence: float
    token_frequency_tvd: float
    drift_score: float
    status: str
    top_changed_tokens: list[dict[str, float | str]]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class DataDriftMonitor:
    def __init__(self, config: MonitoringConfig) -> None:
        self.config = config

    def distribution(self, texts: Sequence[str]) -> DataDistribution:
        tokens = Counter()
        lengths: list[int] = []
        for text in texts:
            token_list = self._tokenize(text)
            tokens.update(token_list)
            lengths.append(len(token_list))

        sample_size = len(texts)
        total_tokens = sum(tokens.values()) or 1
        mean_length = sum(lengths) / max(sample_size, 1)
        variance = sum((length - mean_length) ** 2 for length in lengths) / max(sample_size, 1)
        return DataDistribution(
            sample_size=sample_size,
            article_length_mean=mean_length,
            article_length_std=variance ** 0.5,
            vocabulary_size=len(tokens),
            token_frequencies={token: count / total_tokens for token, count in tokens.items()},
        )

    def detect(self, reference_texts: Sequence[str], production_texts: Sequence[str]) -> DataDriftReport:
        reference = self.distribution(reference_texts)
        production = self.distribution(production_texts)

        length_mean_shift = abs(production.article_length_mean - reference.article_length_mean) / max(
            reference.article_length_std or reference.article_length_mean or 1.0,
            1.0,
        )
        vocabulary_js_divergence = _jensen_shannon_divergence(
            reference.token_frequencies,
            production.token_frequencies,
        )
        token_frequency_tvd = _total_variation_distance(reference.token_frequencies, production.token_frequencies)
        drift_score = _bounded_mean([_clip01(length_mean_shift / 3.0), vocabulary_js_divergence, token_frequency_tvd])
        status = _status_from_score(drift_score, self.config)
        top_changed_tokens = _top_changed_tokens(reference.token_frequencies, production.token_frequencies)
        return DataDriftReport(
            reference_sample_size=reference.sample_size,
            production_sample_size=production.sample_size,
            length_mean_shift=round(length_mean_shift, 4),
            vocabulary_js_divergence=round(vocabulary_js_divergence, 4),
            token_frequency_tvd=round(token_frequency_tvd, 4),
            drift_score=round(drift_score, 4),
            status=status,
            top_changed_tokens=top_changed_tokens,
        )

    def save_report(self, report: DataDriftReport) -> Path:
        self.config.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.config.drift_report_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config.drift_report_path.open("w", encoding="utf-8") as handle:
            json.dump(report.as_dict(), handle, indent=2)
        return self.config.drift_report_path

    def save_trend_plot(self, scores: Sequence[float]) -> Path:
        return save_line_chart_png(scores, self.config.drift_plot_path, "Drift Trend", "Drift Score")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _bounded_mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def _jensen_shannon_divergence(reference: dict[str, float], production: dict[str, float]) -> float:
    keys = set(reference) | set(production)
    if not keys:
        return 0.0

    epsilon = 1e-12
    p = [reference.get(key, 0.0) + epsilon for key in keys]
    q = [production.get(key, 0.0) + epsilon for key in keys]
    total_p = sum(p)
    total_q = sum(q)
    p = [value / total_p for value in p]
    q = [value / total_q for value in q]
    m = [(a + b) / 2.0 for a, b in zip(p, q)]
    kl_pm = sum(a * log2(a / m_i) for a, m_i in zip(p, m) if a > 0 and m_i > 0)
    kl_qm = sum(b * log2(b / m_i) for b, m_i in zip(q, m) if b > 0 and m_i > 0)
    return max(0.0, min(1.0, (kl_pm + kl_qm) / 2.0))


def _total_variation_distance(reference: dict[str, float], production: dict[str, float]) -> float:
    keys = set(reference) | set(production)
    return 0.5 * sum(abs(reference.get(key, 0.0) - production.get(key, 0.0)) for key in keys)


def _top_changed_tokens(reference: dict[str, float], production: dict[str, float], limit: int = 10) -> list[dict[str, float | str]]:
    keys = set(reference) | set(production)
    deltas = [
        {"token": key, "delta": abs(reference.get(key, 0.0) - production.get(key, 0.0))}
        for key in keys
    ]
    deltas.sort(key=lambda item: float(item["delta"]), reverse=True)
    return deltas[:limit]


def _status_from_score(score: float, config: MonitoringConfig) -> str:
    if score >= config.red_threshold:
        return "RED"
    if score >= config.yellow_threshold:
        return "YELLOW"
    return "GREEN"
