from __future__ import annotations

from src.monitoring.config import MonitoringConfig
from src.monitoring.data_drift import DataDriftMonitor
from src.monitoring.prediction_drift import PredictionDriftMonitor


def test_data_drift_detects_vocab_shift(tmp_path) -> None:
    config = MonitoringConfig(artifact_dir=tmp_path)
    monitor = DataDriftMonitor(config)

    reference = ["the cat sat on the mat", "a small cat stayed home"]
    production = ["breaking miracle cure discovered overnight", "exclusive miracle cure claim"]

    report = monitor.detect(reference, production)

    assert report.reference_sample_size == 2
    assert report.production_sample_size == 2
    assert report.drift_score >= 0.0
    assert report.status in {"GREEN", "YELLOW", "RED"}


def test_prediction_drift_reports_label_shift(tmp_path) -> None:
    config = MonitoringConfig(artifact_dir=tmp_path)
    monitor = PredictionDriftMonitor(config)

    report = monitor.detect(
        reference_predictions=["real", "real", "fake", "real"],
        production_predictions=["fake", "fake", "fake", "real"],
        production_trust_scores=[40, 42, 38, 41],
    )

    assert report.production_fake_ratio > report.reference_fake_ratio
    assert report.drift_score >= 0.0
