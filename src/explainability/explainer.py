"""High-level explainability service wrapper."""

from __future__ import annotations

from pathlib import Path

from src.agent.state import EvidenceItem
from src.config.pipeline import ModelConfig
from src.explainability import (
    SHAPExplainer,
    UnifiedExplanation,
    extract_token_importance,
    format_explanation,
    load_explainability_config,
    save_shap_artifact,
    save_token_importance,
)


class ExplainabilityService:
    def __init__(
        self,
        model_config: ModelConfig | None = None,
        config_path: Path = Path("configs/explainability.yaml"),
    ) -> None:
        self.config = load_explainability_config(config_path)
        self.model_config = model_config or ModelConfig()
        self.shap_explainer = SHAPExplainer(self.model_config)

    def explain(
        self,
        article_text: str,
        prediction: str,
        confidence: float,
        evidence: list[EvidenceItem],
        evidence_score: float,
        trust_score: float | None = None,
    ) -> UnifiedExplanation:
        shap_result = self.shap_explainer.explain(article_text, top_k_tokens=self.config.top_k_tokens)
        important_tokens = extract_token_importance(
            shap_result.tokens,
            shap_result.values,
            top_k=self.config.top_k_tokens,
        )
        source_trust = self._calculate_source_trust(evidence)
        report = format_explanation(
            prediction=prediction,
            confidence=confidence,
            evidence_score=evidence_score,
            source_trust=source_trust,
            important_tokens=important_tokens,
            evidence=evidence,
            trust_score=trust_score,
            weights=self.config.weights,
        )
        save_shap_artifact(shap_result, self.config.artifact_dir / "shap_values.json")
        save_token_importance(important_tokens, self.config.artifact_dir / "token_importance.json")
        return report

    def _calculate_source_trust(self, evidence: list[EvidenceItem]) -> float:
        if not evidence:
            return 0.0

        trust_scores = self.config.source_trust_scores or {}
        weighted_sum = 0.0
        weight_total = 0.0
        for item in evidence:
            relevance = max(item.relevance_score, 0.01)
            base_trust = getattr(item, "source_credibility", None)
            if base_trust is None:
                base_trust = trust_scores.get(item.source, 0.5)
            weighted_sum += float(base_trust) * relevance
            weight_total += relevance
        return weighted_sum / max(weight_total, 1e-9)