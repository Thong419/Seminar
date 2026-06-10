from __future__ import annotations

from src.fairness.bias_analysis import BiasAnalyzer


def test_bias_analyzer_reports_imbalances(tmp_path) -> None:
    analyzer = BiasAnalyzer()
    records = [
        {"source": "Reuters", "topic": "health", "label": "real"},
        {"source": "Reuters", "topic": "health", "label": "real"},
        {"source": "Blog", "topic": "politics", "label": "fake"},
        {"source": "Blog", "topic": "politics", "label": "fake"},
    ]

    report = analyzer.analyze(records)

    assert report.sample_size == 4
    assert report.source_imbalance >= 0.0
    assert report.bias_risk_score >= 0.0
