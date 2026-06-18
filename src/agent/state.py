"""Typed state definitions for the agentic fake-news workflow."""

from __future__ import annotations

from enum import Enum
from typing import TypedDict

from pydantic import BaseModel, Field


class Verdict(str, Enum):
    confirmed_fake = "confirmed_fake"
    likely_fake = "likely_fake"
    uncertain = "uncertain"
    likely_real = "likely_real"
    confirmed_real = "confirmed_real"


class ReviewState(str, Enum):
    real = "REAL"
    fake = "FAKE"
    uncertain = "UNCERTAIN"


class EvidenceItem(BaseModel):
    title: str
    source: str
    content: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    url: str | None = None
    query: str | None = None
    provider: str | None = None
    source_credibility: float = Field(default=0.5, ge=0.0, le=1.0)
    stance: str = Field(default="neutral")
    matched_terms: list[str] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    article_text: str
    predicted_label: str
    confidence: float
    retrieved_evidence: list[EvidenceItem]
    evidence_score: float
    final_decision: Verdict
    explanation: str
    trust_score: int
    important_tokens: list[dict[str, float | str]]
    evidence_summary: str
    explanation_details: dict[str, object]
    human_review_state: ReviewState
    conflict_flag: bool


class AgentConfig(BaseModel):
    confidence_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    retrieval_limit: int = Field(default=3, ge=1)
    evidence_agreement_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    contradiction_weight: float = Field(default=0.4, ge=0.0, le=1.0)
