"""Prepare ISOT Fake News Dataset into the project training schema.

Usage:
    python -m src.data.prepare_isot_dataset
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PrepareStats:
    fake_rows: int
    real_rows: int
    total_rows: int
    duplicate_rows: int
    missing_text_rows: int
    output_path: Path


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _load_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    LOGGER.info("Loading %s", path)
    frame = pd.read_csv(path)

    if "text" not in frame.columns:
        raise ValueError(f"Required column 'text' not found in {path}")

    subset = frame[["text"]].copy()
    subset["label"] = label
    return subset


def _validate_output_schema(frame: pd.DataFrame) -> None:
    expected_columns = ["text", "label"]
    if frame.columns.tolist() != expected_columns:
        raise ValueError(
            f"Output schema mismatch. Expected {expected_columns}, got {frame.columns.tolist()}"
        )

    if frame.empty:
        raise ValueError("Prepared dataset is empty.")

    missing_text_rows = int(frame["text"].isna().sum())
    if missing_text_rows > 0:
        raise ValueError(f"Prepared dataset contains {missing_text_rows} missing text values.")

    allowed_labels = {"fake", "real"}
    unique_labels = set(frame["label"].astype(str).str.lower().unique().tolist())
    if not unique_labels.issubset(allowed_labels):
        raise ValueError(f"Invalid labels detected: {sorted(unique_labels - allowed_labels)}")


def prepare_isot_dataset(
    fake_path: Path = Path("data/raw/Fake.csv"),
    true_path: Path = Path("data/raw/True.csv"),
    output_path: Path = Path("data/raw/fakenewsnet.csv"),
    random_state: int = 42,
) -> PrepareStats:
    fake_frame = _load_csv(fake_path, label="fake")
    real_frame = _load_csv(true_path, label="real")

    combined = pd.concat([fake_frame, real_frame], ignore_index=True)
    combined = combined[["text", "label"]]

    missing_text_rows = int(combined["text"].isna().sum())
    duplicate_rows = int(combined.duplicated().sum())

    combined["text"] = combined["text"].astype(str)

    # Keep all rows as requested; only normalize and validate, then shuffle deterministically.
    combined = combined.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    _validate_output_schema(combined)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    LOGGER.info("Saved prepared dataset to %s", output_path)

    return PrepareStats(
        fake_rows=int(fake_frame.shape[0]),
        real_rows=int(real_frame.shape[0]),
        total_rows=int(combined.shape[0]),
        duplicate_rows=duplicate_rows,
        missing_text_rows=missing_text_rows,
        output_path=output_path,
    )


def _print_stats(stats: PrepareStats) -> None:
    message = (
        "ISOT dataset prepared successfully\n"
        f"- Fake rows: {stats.fake_rows}\n"
        f"- Real rows: {stats.real_rows}\n"
        f"- Total rows: {stats.total_rows}\n"
        f"- Duplicate rows: {stats.duplicate_rows}\n"
        f"- Missing text rows: {stats.missing_text_rows}\n"
        f"- Output: {stats.output_path}"
    )
    print(message)


def main() -> None:
    _setup_logging()
    stats = prepare_isot_dataset()
    _print_stats(stats)


if __name__ == "__main__":
    main()
