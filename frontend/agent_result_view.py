"""Streamlit helpers for rendering agent results."""

from __future__ import annotations

from typing import Any


def _format_percent(value: Any) -> str:
    return f"{float(value):.2%}"


def _unique_non_empty(values: list[Any]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        ordered.append(text)
    return ordered


def _normalize_trace(trace: Any) -> dict[str, Any] | None:
    if isinstance(trace, dict):
        return trace
    return None


def _tool_display_name(tool_name: str) -> str:
    mapping = {
        "classifier": "Classifier Tool",
        "evidence": "Evidence Tool",
        "decision": "Decision Tool",
        "explainability": "Explainability Tool",
    }
    return mapping.get(tool_name, tool_name.replace("_", " ").title())


def _trace_steps(trace: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not trace:
        return []

    steps: list[dict[str, Any]] = []
    for index, tool_trace in enumerate(trace.get("tool_traces", []) or [], start=1):
        if not isinstance(tool_trace, dict):
            continue
        tool_name = str(tool_trace.get("tool_name", f"step_{index}"))
        output_data = tool_trace.get("output_data", {}) if isinstance(tool_trace.get("output_data", {}), dict) else {}
        summary = ""
        if tool_name == "classifier":
            summary = f"Output label: {output_data.get('label', 'unknown')}"
        elif tool_name == "evidence":
            summary = f"Evidence count: {output_data.get('num_sources', 0)}"
        elif tool_name == "decision":
            summary = f"Trust score: {_format_percent(output_data.get('trust_score', 0.0))}"
        elif tool_name == "explainability":
            summary = f"Tokens explained: {output_data.get('num_tokens', 0)}"
        steps.append(
            {
                "step": index,
                "tool_name": _tool_display_name(tool_name),
                "duration_ms": float(tool_trace.get("execution_time_ms", 0.0)),
                "summary": summary,
            }
        )
    return steps


def render_agent_result(result: dict[str, Any]) -> dict[str, Any]:
    """Return a presentation-friendly summary for the agent response."""

    sources = list(result.get("sources", []) or result.get("evidence", []) or [])
    trace = _normalize_trace(result.get("trace"))

    summary = {
        "Prediction": str(result.get("prediction", "uncertain")).upper(),
        "Confidence": _format_percent(result.get("confidence", 0.0)),
        "Trust Score": _format_percent(result.get("trust_score", 0.0)),
        "Trust Score Raw": float(result.get("trust_score", 0.0)),
        "Human Review State": str(result.get("human_review_state", "UNCERTAIN")),
        "Conflict Flag": "Yes" if result.get("conflict_flag") else "No",
        "Evidence Sources": sources,
        "Source Providers": _unique_non_empty([item.get("source") or item.get("provider") for item in sources if isinstance(item, dict)]),
        "Evidence Titles": _unique_non_empty([item.get("title") for item in sources if isinstance(item, dict)]),
        "Important Tokens": result.get("important_tokens", []),
        "Evidence Summary": result.get("evidence_summary", ""),
        "Decision Reason": result.get("decision_reason", ""),
        "Trace Summary": _trace_summary(trace),
        "Trace Steps": _trace_steps(trace),
        "Trace Available": trace is not None,
        "Total Execution Time": f"{float(trace.get('total_execution_time_ms', 0.0)):.0f} ms" if trace else "No trace available.",
    }
    return summary


def _trace_summary(trace: Any) -> str:
    if not trace:
        return "No trace available."

    if isinstance(trace, dict):
        tool_traces = trace.get("tool_traces", []) or []
        final_decision = trace.get("final_decision", "unknown")
        total_ms = trace.get("total_execution_time_ms", 0.0)
        if not tool_traces:
            return f"Trace captured, final decision={final_decision}, total time={float(total_ms):.0f} ms"
        return f"{len(tool_traces)} tool steps, final decision={final_decision}, total time={float(total_ms):.0f} ms"

    return "Trace data is available."
