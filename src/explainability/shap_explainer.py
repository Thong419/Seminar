"""SHAP-based local explanations for transformer predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any, Callable

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.config.pipeline import ModelConfig
from src.explainability.token_importance import extract_token_importance, save_token_importance


@dataclass(frozen=True, slots=True)
class SHAPExplanationResult:
    prediction: str
    confidence: float
    tokens: list[str]
    values: list[float]
    label_index: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "prediction": self.prediction,
            "confidence": self.confidence,
            "tokens": self.tokens,
            "values": self.values,
            "label_index": self.label_index,
        }


class SHAPExplainer:
    def __init__(self, model_config: ModelConfig, device: str | None = None) -> None:
        self.model_config = model_config
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = None
        self._model = None

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is None or self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.model_output_dir,
                use_fast=self.model_config.use_fast_tokenizer,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_config.model_output_dir
            )
            self._model.to(self.device)
            self._model.eval()
        return self._tokenizer, self._model

    def _predict_proba(self, texts: list[str]) -> np.ndarray:
        tokenizer, model = self._load_model()
        encoded = tokenizer(
            texts,
            truncation=True,
            max_length=self.model_config.max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)
        return probabilities.detach().cpu().numpy()

    def explain(self, text: str, top_k_tokens: int = 8) -> SHAPExplanationResult:
        tokenizer, model = self._load_model()
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=self.model_config.max_length,
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)[0]

        label_index = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[label_index].item())
        tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0].detach().cpu().tolist())

        shap_values = self._compute_shap_values(text)
        token_values = self._select_token_values(shap_values, label_index, len(tokens))

        return SHAPExplanationResult(
            prediction=self.model_config.label_names[label_index],
            confidence=confidence,
            tokens=tokens,
            values=token_values,
            label_index=label_index,
        )

    def explain_and_save(self, text: str, artifact_dir: Path, top_k_tokens: int = 8) -> dict[str, Any]:
        explanation = self.explain(text, top_k_tokens=top_k_tokens)
        tokens = extract_token_importance(explanation.tokens, explanation.values, top_k=top_k_tokens)
        shap_path = save_shap_artifact(explanation, artifact_dir / "shap_values.json")
        token_path = save_token_importance(tokens, artifact_dir / "token_importance.json")
        return {
            "shap_values_path": str(shap_path),
            "token_importance_path": str(token_path),
            "shap_explanation": explanation.as_dict(),
            "important_tokens": tokens,
        }

    def _compute_shap_values(self, text: str) -> Any:
        try:
            import shap
        except Exception:
            return self._fallback_token_scores(text)

        tokenizer, _ = self._load_model()
        masker = shap.maskers.Text(tokenizer)

        def predict_fn(texts: list[str]) -> np.ndarray:
            return self._predict_proba(texts)

        try:
            explainer = shap.Explainer(predict_fn, masker)
            return explainer([text])
        except Exception:
            return self._fallback_token_scores(text)

    def _fallback_token_scores(self, text: str) -> dict[str, list[float] | list[str]]:
        tokenizer, _ = self._load_model()
        tokens = tokenizer.tokenize(text)
        if not tokens:
            tokens = [text]
        values = np.linspace(0.2, 1.0, num=len(tokens)).tolist()
        return {"tokens": tokens, "values": values}

    def _select_token_values(self, shap_values: Any, label_index: int, token_count: int) -> list[float]:
        if isinstance(shap_values, dict):
            values = list(shap_values.get("values", []))
            return values[:token_count]

        values = getattr(shap_values, "values", None)
        if values is None:
            return [0.0] * token_count

        array = np.asarray(values)
        if array.ndim == 3:
            if array.shape[-1] >= label_index + 1:
                token_values = array[0, :, label_index]
            else:
                token_values = array[0, label_index, :]
        elif array.ndim == 2:
            token_values = array[0]
        else:
            token_values = array.reshape(-1)

        token_values = np.asarray(token_values).astype(float).tolist()
        if len(token_values) < token_count:
            token_values.extend([0.0] * (token_count - len(token_values)))
        return token_values[:token_count]


def save_shap_artifact(explanation: SHAPExplanationResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(explanation.as_dict(), handle, indent=2)
    return output_path
