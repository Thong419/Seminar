"""Prediction and analysis routes for the FastAPI backend."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.api.dependencies import get_agent_controller, get_monitoring_service, get_predictor, get_workflow
from src.api.schemas import (
    AgentResponse,
    AnalyzeRequest,
    AnalyzeResponse,
    EvidenceItemSchema,
    PredictRequest,
    PredictResponse,
    TokenImportanceSchema,
)
from src.privacy.pii_masker import mask_pii

if TYPE_CHECKING:
    from src.agent.controller import AgentController
    from src.agents.workflow import AgenticWorkflow
    from src.inference.predictor import Predictor
    from src.monitoring.monitor import MonitoringService


logger = logging.getLogger("src.api.routes")

router = APIRouter(tags=["prediction"])


def _request_id(request: Request) -> str:
    return str(getattr(request.state, "request_id", "unknown"))


def _structured_http_error(code: str, message: str, details: dict[str, object] | None = None) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_502_BAD_GATEWAY,
        detail={"code": code, "message": message, "details": details or {}},
    )


def _model_to_dict(item: object) -> dict[str, object]:
    dump = getattr(item, "model_dump", None)
    if callable(dump):
        return dict(dump())
    if isinstance(item, dict):
        return dict(item)
    raise TypeError(f"Unsupported evidence item type: {type(item)!r}")


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict fake or real news",
    description="Uses the transformer predictor interface only and returns the label plus confidence.",
)
def predict(
    payload: PredictRequest,
    request: Request,
    predictor: Predictor = Depends(get_predictor),
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
) -> PredictResponse:
    masked = mask_pii(payload.text)
    prediction = predictor.predict(masked.text)
    logger.info(
        "request_id=%s prediction=%s confidence=%.4f",
        _request_id(request),
        prediction.label,
        prediction.confidence,
    )
    try:
        monitoring_service.log_prediction(
            prediction=prediction.label,
            confidence=prediction.confidence,
            trust_score=0,
            article_text=masked.text,
            request_id=_request_id(request),
            endpoint="predict",
        )
    except Exception:  # pragma: no cover - monitoring must not block inference
        logger.exception("request_id=%s monitoring_log_failed endpoint=predict", _request_id(request))
    return PredictResponse(prediction=prediction.label, confidence=prediction.confidence)


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    summary="Run the full analysis workflow",
    description=(
        "Runs classification, real evidence retrieval, evidence analysis, decision making, "
        "and explainability in one call."
    ),
)
def analyze(
    payload: AnalyzeRequest,
    request: Request,
    workflow: AgenticWorkflow = Depends(get_workflow),
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
) -> AnalyzeResponse:
    masked = mask_pii(payload.text)
    try:
        result = workflow.run(masked.text)
    except HTTPException:
        raise
    except ValueError as exc:
        raise _structured_http_error(
            code="retrieval_failure",
            message="Evidence retrieval failed during analysis.",
            details={"error": str(exc)},
        ) from exc
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "missing_model",
                "message": "The model required for analysis is not available.",
                "details": {"error": str(exc)},
            },
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive boundary
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "unexpected_error",
                "message": "An unexpected error occurred while analyzing the article.",
                "details": {"error": str(exc)},
            },
        ) from exc

    evidence_items = [EvidenceItemSchema.model_validate(_model_to_dict(item)) for item in result.get("retrieved_evidence", [])]
    important_tokens = [TokenImportanceSchema.model_validate(item) for item in result.get("important_tokens", [])]

    logger.info(
        "request_id=%s prediction=%s confidence=%.4f",
        _request_id(request),
        result.get("predicted_label", "uncertain"),
        float(result.get("confidence", 0.0)),
    )

    try:
        monitoring_service.log_prediction(
            prediction=str(result.get("predicted_label", "uncertain")),
            confidence=float(result.get("confidence", 0.0)),
            trust_score=int(result.get("trust_score", 0)),
            article_text=masked.text,
            request_id=_request_id(request),
            endpoint="analyze",
        )
    except Exception:  # pragma: no cover - monitoring must not block workflow
        logger.exception("request_id=%s monitoring_log_failed endpoint=analyze", _request_id(request))

    return AnalyzeResponse(
        prediction=str(result.get("predicted_label", "uncertain")),
        confidence=float(result.get("confidence", 0.0)),
        trust_score=int(result.get("trust_score", 0)),
        important_tokens=important_tokens,
        evidence=evidence_items,
        evidence_summary=str(result.get("evidence_summary", "No supporting evidence was retrieved.")),
        final_explanation=str(result.get("explanation", "")),
        explanation_details=result.get("explanation_details"),
    )


@router.post(
    "/agent",
    response_model=AgentResponse,
    tags=["agent"],
    summary="Run the agentic workflow with conditional evidence retrieval",
    description=(
        "Runs the lightweight Agent Controller workflow: classification → confidence check → "
        "conditional evidence retrieval → decision → explanation. Returns detailed agent trace and decision reasoning."
    ),
)
def agent(
    payload: AnalyzeRequest,
    request: Request,
    agent_controller: AgentController = Depends(get_agent_controller),
    monitoring_service: MonitoringService = Depends(get_monitoring_service),
) -> AgentResponse:
    """Agent endpoint - orchestrates tools with conditional routing and tracing."""
    masked = mask_pii(payload.text)
    try:
        result = agent_controller.run(
            article_text=masked.text,
            request_id=_request_id(request),
        )
    except HTTPException:
        raise
    except Exception as exc:  # pragma: no cover - defensive boundary
        logger.exception("request_id=%s agent_workflow_failed", _request_id(request))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "agent_error",
                "message": "Agent workflow failed.",
                "details": {"error": str(exc)},
            },
        ) from exc

    logger.info(
        "request_id=%s agent_prediction=%s confidence=%.4f trust_score=%.4f risk_level=%s",
        _request_id(request),
        result.label,
        result.confidence,
        result.trust_score,
        result.risk_level,
    )

    try:
        monitoring_service.log_prediction(
            prediction=result.label,
            confidence=result.confidence,
            trust_score=int(result.trust_score * 100),
            article_text=masked.text,
            request_id=_request_id(request),
            endpoint="agent",
        )
    except Exception:  # pragma: no cover - monitoring must not block workflow
        logger.exception("request_id=%s monitoring_log_failed endpoint=agent", _request_id(request))

    return AgentResponse(
        prediction=result.label,
        confidence=result.confidence,
        trust_score=result.trust_score,
        risk_level=result.risk_level,
        human_review_state=result.human_review_state,
        conflict_flag=result.conflict_flag,
        decision_reason=result.decision_reason,
        explanation=result.explanation,
        important_tokens=[TokenImportanceSchema.model_validate(t) for t in result.important_tokens],
        evidence_found=result.evidence_found,
        evidence_summary=result.evidence_summary,
        claim=result.claim,
        queries=result.queries,
        support_score=result.support_score,
        contradiction_score=result.contradiction_score,
        source_credibility_score=result.source_credibility_score,
        evidence_quality_score=result.evidence_quality_score,
        sources=result.sources,
        trace=result.trace.to_dict() if result.trace else None,
    )
