"""Execution trace logging for the agent workflow."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any
import json
from pathlib import Path


@dataclass
class ToolTrace:
    """Record of a single tool invocation."""
    tool_name: str
    input_data: dict[str, Any]
    output_data: dict[str, Any]
    execution_time_ms: float
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class AgentTrace:
    """Complete execution trace for an agent run."""
    request_id: str
    article_text: str
    tool_traces: list[ToolTrace] = field(default_factory=list)
    final_trust_score: float | None = None
    final_decision: str | None = None
    total_execution_time_ms: float = 0.0
    timestamp_start: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    timestamp_end: str = ""

    def add_tool_trace(self, trace: ToolTrace) -> None:
        """Record a tool invocation."""
        self.tool_traces.append(trace)

    def finalize(self, trust_score: float | None = None, decision: str | None = None, exec_time_ms: float = 0.0) -> None:
        """Finalize the trace with results."""
        self.final_trust_score = trust_score
        self.final_decision = decision
        self.total_execution_time_ms = exec_time_ms
        self.timestamp_end = datetime.utcnow().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to dictionary."""
        return {
            "request_id": self.request_id,
            "article_text": self.article_text,
            "tool_traces": [asdict(t) for t in self.tool_traces],
            "final_trust_score": self.final_trust_score,
            "final_decision": self.final_decision,
            "total_execution_time_ms": self.total_execution_time_ms,
            "timestamp_start": self.timestamp_start,
            "timestamp_end": self.timestamp_end,
        }

    def to_json(self) -> str:
        """Serialize to JSON."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def save(self, artifact_dir: Path | str = "artifacts/agent_traces") -> str:
        """Save trace to file."""
        artifact_path = Path(artifact_dir)
        artifact_path.mkdir(parents=True, exist_ok=True)
        filename = f"trace_{self.request_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = artifact_path / filename
        filepath.write_text(self.to_json())
        return str(filepath)
