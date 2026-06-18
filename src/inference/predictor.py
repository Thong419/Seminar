"""Inference predictor interface for transformer models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer, TextClassificationPipeline, pipeline

from src.config.pipeline import ModelConfig


@dataclass(frozen=True, slots=True)
class Prediction:
    label: str
    confidence: float


class Predictor:
    def __init__(self, model_dir: str | Path, model_config: ModelConfig) -> None:
        self.model_reference = str(model_dir)
        self.model_config = model_config
        self._pipeline = self._load_pipeline()

    def _load_pipeline(self) -> TextClassificationPipeline:
        tokenizer = AutoTokenizer.from_pretrained(self.model_reference)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_reference)
        return pipeline(
            task="text-classification",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
        )

    def predict(self, text: str) -> Prediction:
        result = self._pipeline(text)[0]
        label = str(result["label"]).lower()
        score = float(result["score"])
        return Prediction(label=label, confidence=score)