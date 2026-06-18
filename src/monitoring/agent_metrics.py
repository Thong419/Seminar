"""Lightweight agent workflow metrics storage."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock
from typing import Any
import json


@dataclass
class AgentMetricsSnapshot:
    total_agent_requests: int = 0
    evidence_retrieval_success_rate: float = 0.0
    conflict_rate: float = 0.0
    uncertain_rate: float = 0.0
    average_confidence: float = 0.0
    average_trust_score: float = 0.0
    average_response_time_ms: float = 0.0
    evidence_source_count: float = 0.0


class AgentMetricsTracker:
    """Persist agent metrics to a small JSON artifact."""

    def __init__(self, artifact_path: Path | str = Path("artifacts/agent_metrics/metrics.json")) -> None:
        self.artifact_path = Path(artifact_path)
        self._lock = Lock()
        self._snapshot = AgentMetricsSnapshot()
        self._counters = {
            "evidence_retrieval_success": 0,
            "conflict": 0,
            "uncertain": 0,
        }

    def record(
        self,
        *,
        evidence_found: bool,
        conflict_flag: bool,
        human_review_state: str,
        confidence: float,
        trust_score: float,
        response_time_ms: float,
        evidence_source_count: int,
    ) -> AgentMetricsSnapshot:
        with self._lock:
            previous = self._snapshot.total_agent_requests
            current = previous + 1

            self._counters["evidence_retrieval_success"] += int(evidence_found)
            self._counters["conflict"] += int(conflict_flag)
            self._counters["uncertain"] += int(human_review_state.upper() == "UNCERTAIN")

            self._snapshot.total_agent_requests = current
            self._snapshot.evidence_retrieval_success_rate = self._counters["evidence_retrieval_success"] / current
            self._snapshot.conflict_rate = self._counters["conflict"] / current
            self._snapshot.uncertain_rate = self._counters["uncertain"] / current
            self._snapshot.average_confidence = self._rolling_average(self._snapshot.average_confidence, confidence, previous)
            self._snapshot.average_trust_score = self._rolling_average(self._snapshot.average_trust_score, trust_score, previous)
            self._snapshot.average_response_time_ms = self._rolling_average(self._snapshot.average_response_time_ms, response_time_ms, previous)
            self._snapshot.evidence_source_count = self._rolling_average(self._snapshot.evidence_source_count, float(evidence_source_count), previous)

            self._persist()
            return self._snapshot

    def snapshot(self) -> AgentMetricsSnapshot:
        with self._lock:
            return AgentMetricsSnapshot(**asdict(self._snapshot))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self.snapshot())

    def _rolling_average(self, previous_average: float, new_value: float, previous_count: int) -> float:
        if previous_count <= 0:
            return float(new_value)
        return ((previous_average * previous_count) + new_value) / (previous_count + 1)

    def _persist(self) -> None:
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_path.write_text(json.dumps(asdict(self._snapshot), indent=2), encoding="utf-8")


_TRACKER = AgentMetricsTracker()


def get_agent_metrics_tracker() -> AgentMetricsTracker:
    return _TRACKER
