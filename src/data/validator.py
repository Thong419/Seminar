"""Dataset validation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd


@dataclass(frozen=True, slots=True)
class ValidationRules:
    text_column: str = "text"
    label_column: str = "label"
    allowed_labels: tuple[str, ...] = ("fake", "real")
    min_rows: int = 1


@dataclass(frozen=True, slots=True)
class ValidationReport:
    is_valid: bool
    row_count: int
    duplicate_count: int
    missing_text_count: int
    missing_label_count: int
    invalid_label_count: int
    errors: tuple[str, ...] = field(default_factory=tuple)


def validate_dataset(frame: pd.DataFrame, rules: ValidationRules) -> ValidationReport:
    errors: list[str] = []
    if frame.empty:
        errors.append("Dataset is empty.")

    missing_columns = [column for column in (rules.text_column, rules.label_column) if column not in frame.columns]
    if missing_columns:
        errors.append(f"Missing required columns: {', '.join(missing_columns)}")

    text_series = frame[rules.text_column] if rules.text_column in frame.columns else pd.Series(dtype="object")
    label_series = frame[rules.label_column] if rules.label_column in frame.columns else pd.Series(dtype="object")

    missing_text_count = int(text_series.isna().sum())
    missing_label_count = int(label_series.isna().sum())
    duplicate_count = int(frame.duplicated().sum())

    invalid_label_count = 0
    if not label_series.empty:
        allowed = {label.lower() for label in rules.allowed_labels}
        invalid_label_count = int(
            label_series.dropna().astype(str).str.lower().map(lambda value: value not in allowed).sum()
        )
        if invalid_label_count:
            errors.append("Dataset contains labels outside the configured label set.")

    if frame.shape[0] < rules.min_rows:
        errors.append(f"Dataset must contain at least {rules.min_rows} rows.")

    return ValidationReport(
        is_valid=not errors,
        row_count=int(frame.shape[0]),
        duplicate_count=duplicate_count,
        missing_text_count=missing_text_count,
        missing_label_count=missing_label_count,
        invalid_label_count=invalid_label_count,
        errors=tuple(errors),
    )


def ensure_required_columns(frame: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")