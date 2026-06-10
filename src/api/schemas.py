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
