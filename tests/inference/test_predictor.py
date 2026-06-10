from pathlib import Path

from src.config.pipeline import ModelConfig


def test_predictor_module_imports() -> None:
    from src.inference.predictor import Prediction

    prediction = Prediction(label="fake", confidence=0.9)
    assert prediction.label == "fake"
    assert prediction.confidence == 0.9
