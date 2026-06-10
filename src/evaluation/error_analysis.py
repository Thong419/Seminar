"""Error analysis utilities for transformer predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class ErrorAnalysisConfig:
    output_path: Path = Path("artifacts/evaluation/error_analysis.csv")


def build_error_analysis_frame(
    texts: list[str],
    y_true: list[str],
    y_pred: list[str],
    confidences: list[float],
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "text": texts,
            "true_label": y_true,
            "predicted_label": y_pred,
            "confidence": confidences,
        }
    )
    frame["error_type"] = "correct"
    frame.loc[(frame["true_label"] != frame["predicted_label"]) & (frame["predicted_label"] == "fake"), "error_type"] = "false_positive"
    frame.loc[(frame["true_label"] != frame["predicted_label"]) & (frame["predicted_label"] == "real"), "error_type"] = "false_negative"
    return frame


def save_error_analysis(frame: pd.DataFrame, output_path: Path = Path("artifacts/evaluation/error_analysis.csv")) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return output_path