from __future__ import annotations

from pathlib import Path

from src.monitoring.agent_metrics import AgentMetricsTracker
from src.retrieval.source_credibility import score_source_credibility


def test_agent_metrics_tracker_updates_snapshot(tmp_path: Path) -> None:
    tracker = AgentMetricsTracker(tmp_path / "metrics.json")

    snapshot = tracker.record(
        evidence_found=True,
        conflict_flag=False,
        human_review_state="REAL",
        confidence=0.9,
        trust_score=0.85,
        response_time_ms=1000.0,
        evidence_source_count=3,
    )

    assert snapshot.total_agent_requests == 1
    assert snapshot.evidence_retrieval_success_rate == 1.0
    assert snapshot.conflict_rate == 0.0
    assert snapshot.uncertain_rate == 0.0
    assert snapshot.average_confidence == 0.9
    assert tracker.artifact_path.exists()


def test_source_credibility_policy_prefers_trusted_domains() -> None:
    assert score_source_credibility("NASA", "https://www.nasa.gov/mission") > score_source_credibility("Wikipedia", "https://en.wikipedia.org/wiki/NASA")
    assert score_source_credibility("Wikipedia", "https://en.wikipedia.org/wiki/NASA") > score_source_credibility("Unknown Blog", "https://example.com/post")
    assert score_source_credibility("Reuters", "https://www.reuters.com/world/") >= 0.9
