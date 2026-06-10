from __future__ import annotations

from src.monitoring.config import MonitoringConfig
from src.monitoring.confidence_monitor import ConfidenceMonitor


def test_confidence_monitor_builds_histogram_and_trend(tmp_path) -> None:
    config = MonitoringConfig(artifact_dir=tmp_path, confidence_bins=5)
    monitor = ConfidenceMonitor(config)

    report = monitor.detect([0.91, 0.88, 0.83, 0.79, 0.76, 0.74])

    assert report.average_confidence > 0.0
    assert len(report.confidence_histogram) == 5
    assert report.confidence_trend[-1] == 0.74
    assert report.status in {"GREEN", "YELLOW", "RED"}
