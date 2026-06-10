from pathlib import Path

import pandas as pd

from src.config.pipeline import EvaluationConfig, ModelConfig, TokenizationConfig, TransformerTrainingConfig
from src.training.transformer_trainer import DatasetLabelEncoder, NewsDataset


def test_news_dataset_encodes_labels() -> None:
    frame = pd.DataFrame({"text": ["hello"], "label": ["fake"]})
    class DummyTokenizer:
        def __call__(self, text, truncation, max_length, padding):
            return {"input_ids": [1, 2], "attention_mask": [1, 1]}

    dataset = NewsDataset(
        frame,
        "text",
        "label",
        DatasetLabelEncoder(("fake", "real")),
        DummyTokenizer(),
        16,
        True,
    )
    item = dataset[0]
    assert item["labels"] == 0
    assert item["input_ids"] == [1, 2]
