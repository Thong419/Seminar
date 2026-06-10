"""Explanation rendering helpers for the Streamlit frontend."""

from __future__ import annotations


def build_explanation_lines(evidence_summary: str, final_explanation: str) -> list[str]:
    return [evidence_summary, final_explanation]
