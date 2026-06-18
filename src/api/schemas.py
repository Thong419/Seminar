"""Pydantic schemas for the FastAPI backend."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{"text": "article text"}]})

    text: str = Field(min_length=1, description="News article text to classify.")


class PredictResponse(BaseModel):
    model_config = ConfigDict(json_schema_extra={"examples": [{"prediction": "fake", "confidence": 0.91}]})

    prediction: str = Field(description="Predicted label returned by the classifier.")
    confidence: float = Field(ge=0.0, le=1.0, description="Classifier confidence score.")


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{"text": "article text"}]})

    text: str = Field(min_length=1, description="News article text to analyze end to end.")


class EvidenceItemSchema(BaseModel):
    title: str
    source: str
    content: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    url: str | None = None


class TokenImportanceSchema(BaseModel):
    token: str
    importance: float


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "prediction": "fake",
                    "confidence": 0.91,
                    "trust_score": 88,
                    "important_tokens": [{"token": "fake", "importance": 0.81}],
                    "evidence": [
                        {
                            "title": "Reuters verifies claim",
                            "source": "Reuters",
                            "content": "Reuters verifies the claim.",
                            "relevance_score": 0.95,
                            "url": "https://example.com/reuters",
                        }
                    ],
                    "evidence_summary": "Retrieved 1 evidence item from 1 sources.",
                    "final_explanation": "The article was classified as fake with confidence 0.91.",
                }
            ]
        }
    )

    prediction: str
    confidence: float = Field(ge=0.0, le=1.0)
    trust_score: int = Field(ge=0, le=100)
    important_tokens: list[TokenImportanceSchema]
    evidence: list[EvidenceItemSchema]
    evidence_summary: str
    final_explanation: str
    explanation_details: dict[str, Any] | None = None


class AgentResponse(BaseModel):
    """Response from the Agent Controller endpoint."""
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "prediction": "fake",
                    "confidence": 0.91,
                    "trust_score": 0.88,
                    "risk_level": "low",
                    "human_review_state": "FAKE",
                    "conflict_flag": False,
                    "decision_reason": "Evidence found (3 sources). confidence=0.91, support_score=0.82, contradiction_score=0.14, source_credibility=0.90.",
                    "explanation": "Based on classifier and evidence, the article is likely fake.",
                    "important_tokens": [{"token": "fake", "importance": 0.81}],
                    "evidence_found": True,
                    "sources": [{"title": "Example", "url": "https://example.com", "snippet": "...", "relevance": 0.9}],
                }
            ]
        }
    )

    prediction: str = Field(description="Predicted label (fake/real/uncertain)")
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence score")
    trust_score: float = Field(ge=0.0, le=1.0, description="Trust score combining confidence and evidence")
    risk_level: str = Field(description="Risk level assessment (low/medium/high)")
    human_review_state: str = Field(description="Human review state (REAL/FAKE/UNCERTAIN)")
    conflict_flag: bool = Field(description="Whether evidence conflicts with the classifier")
    decision_reason: str = Field(description="Human-readable explanation of decision")
    explanation: str = Field(description="Full explanation with token importance")
    important_tokens: list[TokenImportanceSchema] = Field(description="Token importance scores")
    evidence_found: bool = Field(description="Whether external evidence was retrieved")
    evidence_summary: str = Field(description="Human-readable evidence summary")
    claim: str = Field(description="Extracted claim used for retrieval")
    queries: list[str] = Field(description="Search queries used for evidence retrieval")
    support_score: float = Field(ge=0.0, le=1.0, description="Evidence support score")
    contradiction_score: float = Field(ge=0.0, le=1.0, description="Evidence contradiction score")
    source_credibility_score: float = Field(ge=0.0, le=1.0, description="Average source credibility score")
    evidence_quality_score: float = Field(ge=0.0, le=1.0, description="Overall evidence quality score")
    sources: list[dict[str, Any]] = Field(description="Retrieved evidence sources")
    trace: dict[str, Any] | None = Field(default=None, description="Serialized agent execution trace")



class VersionResponse(BaseModel):
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["healthy"]


class APIErrorDetail(BaseModel):
    code: str
    message: str
    details: dict[str, Any] | None = None


class APIErrorResponse(BaseModel):
    error: APIErrorDetail
    request_id: str
