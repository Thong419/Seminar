"""Evidence rendering helpers for the Streamlit frontend."""

from __future__ import annotations

from typing import Iterable


def render_evidence_cards(evidence: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    return [dict(item) for item in evidence]
