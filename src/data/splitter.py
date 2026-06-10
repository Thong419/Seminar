"""Train/validation/test splitting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.pipeline import SplitConfig


@dataclass(frozen=True, slots=True)
class SplitSummary:
    train_rows: int
    validation_rows: int
    test_rows: int


def _can_stratify(frame: pd.DataFrame, label_column: str) -> bool:
    if label_column not in frame.columns:
        return False
    value_counts = frame[label_column].value_counts(dropna=False)
    return len(value_counts) > 1 and int(value_counts.min()) >= 2


def split_dataset(
    frame: pd.DataFrame,
    split_config: SplitConfig,
    label_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitSummary]:
    if not 0.99 <= (split_config.train_size + split_config.validation_size + split_config.test_size) <= 1.01:
        raise ValueError("Train, validation, and test ratios must sum to 1.0.")
    if frame.empty:
        raise ValueError("Cannot split an empty dataset.")

    stratify_series = frame[label_column] if split_config.stratify and _can_stratify(frame, label_column) else None
    train_val_frame, test_frame = train_test_split(
        frame,
        test_size=split_config.test_size,
        random_state=split_config.random_state,
        stratify=stratify_series,
    )

    validation_ratio_within_train_val = split_config.validation_size / (
        split_config.train_size + split_config.validation_size
    )
    train_stratify = (
        train_val_frame[label_column]
        if split_config.stratify and _can_stratify(train_val_frame, label_column)
        else None
    )
    train_frame, validation_frame = train_test_split(
        train_val_frame,
        test_size=validation_ratio_within_train_val,
        random_state=split_config.random_state,
        stratify=train_stratify,
    )

    return (
        train_frame.reset_index(drop=True),
        validation_frame.reset_index(drop=True),
        test_frame.reset_index(drop=True),
        SplitSummary(
            train_rows=int(train_frame.shape[0]),
            validation_rows=int(validation_frame.shape[0]),
            test_rows=int(test_frame.shape[0]),
        ),
    )