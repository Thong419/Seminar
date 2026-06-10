"""Text and dataframe preprocessing for the news classification pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import html
import re

import pandas as pd

from src.config.pipeline import PreprocessingConfig


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
HTML_TAG_PATTERN = re.compile(r"<[^>]+>")
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class CleaningSummary:
    rows_before: int
    rows_after: int
    duplicates_removed: int


def lowercase_text(text: str) -> str:
    return text.lower()


def remove_urls(text: str) -> str:
    return URL_PATTERN.sub(" ", text)


def remove_html(text: str) -> str:
    return HTML_TAG_PATTERN.sub(" ", html.unescape(text))


def remove_special_characters(text: str, pattern: str) -> str:
    return re.sub(pattern, " ", text)


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def preprocess_text(text: str | float | None, config: PreprocessingConfig) -> str:
    value = "" if text is None or (isinstance(text, float) and pd.isna(text)) else str(text)
    if config.remove_html:
        value = remove_html(value)
    if config.remove_urls:
        value = remove_urls(value)
    if config.lowercase:
        value = lowercase_text(value)
    if config.remove_special_characters:
        value = remove_special_characters(value, config.special_characters_pattern)
    if config.remove_whitespace:
        value = normalize_whitespace(value)
    return value


def clean_dataframe(
    frame: pd.DataFrame,
    text_column: str,
    label_column: str,
    config: PreprocessingConfig,
) -> tuple[pd.DataFrame, CleaningSummary]:
    if text_column not in frame.columns:
        raise ValueError(f"Missing text column: {text_column}")
    if label_column not in frame.columns:
        raise ValueError(f"Missing label column: {label_column}")

    cleaned = frame.copy()
    rows_before = int(cleaned.shape[0])

    cleaned = cleaned.dropna(subset=[label_column])
    cleaned[text_column] = cleaned[text_column].map(lambda value: preprocess_text(value, config))
    cleaned[label_column] = cleaned[label_column].astype(str).str.lower().str.strip()
    cleaned = cleaned[cleaned[text_column].str.len() > 0]
    cleaned = cleaned.drop_duplicates(subset=[text_column, label_column]).reset_index(drop=True)

    return cleaned, CleaningSummary(
        rows_before=rows_before,
        rows_after=int(cleaned.shape[0]),
        duplicates_removed=rows_before - int(cleaned.shape[0]),
    )