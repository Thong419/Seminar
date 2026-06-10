"""Dataset loading and preparation facade for article classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config.pipeline import DatasetConfig
from src.data.preprocessing import clean_dataframe
from src.data.validator import ValidationReport, ValidationRules, validate_dataset


@dataclass(frozen=True, slots=True)
class DatasetSummary:
    rows: int
    columns: tuple[str, ...]
    validation: ValidationReport


def load_raw_dataset(dataset_config: DatasetConfig) -> pd.DataFrame:
    source_path = dataset_config.source_path
    if not source_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {source_path}")

    suffix = source_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source_path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(source_path, lines=suffix == ".jsonl")
    if suffix == ".parquet":
        return pd.read_parquet(source_path)

    raise ValueError(f"Unsupported dataset format: {suffix}")


def prepare_dataset(dataset_config: DatasetConfig) -> tuple[pd.DataFrame, DatasetSummary]:
    raw_frame = load_raw_dataset(dataset_config)
    validation = validate_dataset(
        raw_frame,
        ValidationRules(
            text_column=dataset_config.text_column,
            label_column=dataset_config.label_column,
            allowed_labels=dataset_config.allowed_labels,
        ),
    )
    if not validation.is_valid:
        raise ValueError("Raw dataset validation failed: " + "; ".join(validation.errors))

    cleaned_frame, _summary = clean_dataframe(
        raw_frame,
        text_column=dataset_config.text_column,
        label_column=dataset_config.label_column,
        config=dataset_config.preprocessing,
    )

    return cleaned_frame, DatasetSummary(
        rows=int(cleaned_frame.shape[0]),
        columns=tuple(cleaned_frame.columns.tolist()),
        validation=validation,
    )


def load_and_prepare_from_path(path: Path, dataset_config: DatasetConfig) -> tuple[pd.DataFrame, DatasetSummary]:
    del path
    return prepare_dataset(dataset_config)