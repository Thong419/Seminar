"""Human-readable explanation generator for the agentic workflow."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.agents.state import EvidenceItem, Verdict
from src.config.pipeline import ModelConfig
from src.explainability import (
    SHAPExplainer,
    UnifiedExplanation,
    extract_token_importance,
    format_explanation,
    load_explainability_config,
    save_misclassified_samples,
    save_token_importance,
)
from src.explainability.shap_explainer import SHAPExplanationResult, save_shap_artifact


@dataclass(frozen=True, slots=True)
class ExplanationContext:
    article_text: str
    true_label: str | None
    predicted_label: str
    confidence: float
    evidence: list[EvidenceItem]
    evidence_score: float
    final_decision: Verdict


@dataclass(frozen=True, slots=True)
class ExplanationResult:
    report: UnifiedExplanation
    shap_values_path: Path
    token_importance_path: Path
    misclassified_samples_path: Path | None = None


class ExplanationAgent:
    def __init__(
        self,
        model_config: ModelConfig | None = None,
        explainability_config_path: Path = Path("configs/explainability.yaml"),
        shap_explainer: SHAPExplainer | None = None,
    ) -> None:
        self.explainability_config = load_explainability_config(explainability_config_path)
        self.model_config = model_config or ModelConfig()
        self.shap_explainer = shap_explainer or SHAPExplainer(self.model_config)

    def generate(self, context: ExplanationContext) -> ExplanationResult:
        shap_result = self.shap_explainer.explain(
            context.article_text,
            top_k_tokens=self.explainability_config.top_k_tokens,
        )
        important_tokens = extract_token_importance(
            shap_result.tokens,
            shap_result.values,
            top_k=self.explainability_config.top_k_tokens,
        )
        source_trust = self._calculate_source_trust(context.evidence)
        report = format_explanation(
            prediction=context.predicted_label,
            confidence=context.confidence,
            evidence_score=context.evidence_score,
            source_trust=source_trust,
            important_tokens=important_tokens,
            evidence=context.evidence,
            weights=self.explainability_config.weights,
        )
        shap_path = save_shap_artifact(
            shap_result,
            self.explainability_config.artifact_dir / "shap_values.json",
        )
        token_path = save_token_importance(
            important_tokens,
            self.explainability_config.artifact_dir / "token_importance.json",
        )

        misclassified_samples_path = None
        if context.true_label and context.true_label != context.predicted_label:
            misclassified_samples_path = save_misclassified_samples(
                [
                    {
                        "article": context.article_text,
                        "true_label": context.true_label,
                        "predicted_label": context.predicted_label,
                        "confidence": context.confidence,
                        "important_tokens": important_tokens,
                    }
                ],
                self.explainability_config.artifact_dir / "error_analysis.csv",
                max_rows=self.explainability_config.max_misclassified_samples,
            )

        return ExplanationResult(
            report=report,
            shap_values_path=shap_path,
            token_importance_path=token_path,
            misclassified_samples_path=misclassified_samples_path,
        )

    def _calculate_source_trust(self, evidence: list[EvidenceItem]) -> float:
        if not evidence:
            return 0.0

        trust_scores = self.explainability_config.source_trust_scores or {}
        weighted_sum = 0.0
        weight_total = 0.0
        for item in evidence:
            relevance = max(item.relevance_score, 0.01)
            weighted_sum += trust_scores.get(item.source, 0.5) * relevance
            weight_total += relevance
        return weighted_sum / max(weight_total, 1e-9)
