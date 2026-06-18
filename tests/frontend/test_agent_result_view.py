from __future__ import annotations

from frontend.agent_result_view import render_agent_result


def test_render_agent_result_groups_agent_fields() -> None:
    result = render_agent_result(
        {
            "prediction": "real",
            "confidence": 0.9231,
            "trust_score": 0.8812,
            "human_review_state": "REAL",
            "conflict_flag": False,
            "evidence_summary": "Retrieved evidence bundle.",
            "decision_reason": "Aligned evidence and classifier.",
            "sources": [{"title": "NASA", "source": "Wikipedia"}],
            "important_tokens": [{"token": "NASA", "importance": 0.91}],
            "trace": {
                "tool_traces": [
                    {"tool_name": "classifier", "execution_time_ms": 12, "output_data": {"label": "real"}},
                    {"tool_name": "evidence", "execution_time_ms": 34, "output_data": {"num_sources": 5}},
                    {"tool_name": "decision", "execution_time_ms": 56, "output_data": {"trust_score": 0.8812}},
                    {"tool_name": "explainability", "execution_time_ms": 78, "output_data": {"num_tokens": 4}},
                ],
                "final_decision": "real",
                "total_execution_time_ms": 1234,
            },
        }
    )

    assert result["Prediction"] == "REAL"
    assert result["Confidence"] == "92.31%"
    assert result["Trust Score"] == "88.12%"
    assert result["Human Review State"] == "REAL"
    assert result["Conflict Flag"] == "No"
    assert result["Source Providers"] == ["Wikipedia"]
    assert result["Evidence Titles"] == ["NASA"]
    assert "4 tool steps" in result["Trace Summary"]
    assert result["Trace Steps"][0]["tool_name"] == "Classifier Tool"
