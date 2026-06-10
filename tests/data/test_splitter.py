import pandas as pd

from src.config.pipeline import SplitConfig
from src.data.splitter import split_dataset


def test_split_dataset_respects_ratios_and_disjoint_sets() -> None:
    frame = pd.DataFrame(
        {
            "text": [f"sample {index}" for index in range(20)],
            "label": ["fake"] * 10 + ["real"] * 10,
        }
    )
    train, validation, test, summary = split_dataset(
        frame,
        SplitConfig(train_size=0.7, validation_size=0.15, test_size=0.15, random_state=42),
        label_column="label",
    )

    assert summary.train_rows == 14
    assert summary.validation_rows == 3
    assert summary.test_rows == 3
    assert set(train.index).isdisjoint(validation.index)
    assert set(train.index).isdisjoint(test.index)
    assert set(validation.index).isdisjoint(test.index)