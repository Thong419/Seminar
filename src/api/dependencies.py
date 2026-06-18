"""Dependency providers for the FastAPI backend."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import HTTPException, status

from src.config.pipeline import ModelConfig
from src.config.settings import AppSettings, get_settings

if TYPE_CHECKING:
    from src.agent.controller import AgentController
    from src.agent.workflow import AgenticWorkflow
    from src.inference.predictor import Predictor
    from src.monitoring.monitor import MonitoringService


def _missing_model_http_exception(settings: AppSettings, exc: Exception) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail={
            "code": "missing_model",
            "message": "The prediction model is not available.",
            "details": {"model_dir": str(settings.model_dir), "error": str(exc)},
        },
    )


@lru_cache(maxsize=1)
def get_model_config() -> ModelConfig:
    settings = get_settings()
    return ModelConfig(
        name=settings.model_name,
        model_output_dir=settings.model_dir,
        tokenizer_name=settings.model_name,
    )


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    settings = get_settings()
    model_config = get_model_config()
    try:
        from src.inference.predictor import Predictor

        return Predictor(model_dir=settings.model_dir, model_config=model_config)
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise _missing_model_http_exception(settings, exc) from exc


@lru_cache(maxsize=1)
def get_workflow() -> AgenticWorkflow:
    settings = get_settings()
    model_config = get_model_config()
    try:
        from src.agent.analysis.evidence_analysis_agent import EvidenceAnalysisAgent
        from src.agent.classification.classification_agent import ClassificationAgent
        from src.agent.decision.decision_agent import DecisionAgent
        from src.agent.explanation.explanation_agent import ExplanationAgent
        from src.agent.retrieval.retrieval_agent import RetrievalAgent
        from src.agent.state import AgentConfig
        from src.agent.workflow import AgenticWorkflow

        predictor = get_predictor()
        classifier = ClassificationAgent(predictor=predictor, model_config=model_config)
        retriever = RetrievalAgent(retrieval_config_path=Path("configs/retrieval.yaml"))
        analyzer = EvidenceAnalysisAgent()
        decider = DecisionAgent()
        explainer = ExplanationAgent(model_config=model_config)
        return AgenticWorkflow(
            classifier=classifier,
            retriever=retriever,
            analyzer=analyzer,
            decider=decider,
            explainer=explainer,
            config=AgentConfig(),
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "workflow_unavailable",
                "message": "The analysis workflow could not be initialized.",
                "details": {"error": str(exc), "model_dir": str(settings.model_dir)},
            },
        ) from exc


@lru_cache(maxsize=1)
def get_monitoring_service() -> MonitoringService:
    try:
        from src.monitoring.monitor import MonitoringService

        return MonitoringService.from_config_path(Path("configs/monitoring.yaml"))
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "monitoring_unavailable",
                "message": "The monitoring service could not be initialized.",
                "details": {"error": str(exc)},
            },
        ) from exc


@lru_cache(maxsize=1)
def get_agent_controller() -> AgentController:
    """Inject the Agent Controller with all dependencies."""
    settings = get_settings()
    model_config = get_model_config()
    try:
        from src.agent.controller import AgentController, AgentControllerConfig

        predictor = get_predictor()
        agent_config = AgentControllerConfig(
            confidence_threshold=0.85,
            enable_evidence_retrieval=True,
            enable_tracing=True,
        )
        return AgentController(
            predictor=predictor,
            model_config=model_config,
            explainability_config_path=Path("configs/explainability.yaml"),
            agent_config=agent_config,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "agent_controller_unavailable",
                "message": "The agent controller could not be initialized.",
                "details": {"error": str(exc)},
            },
        ) from exc
