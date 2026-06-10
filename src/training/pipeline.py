"""Training pipeline skeleton for baseline and transformer models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    raw_data_path: Path
    processed_data_path: Path
    model_output_path: Path
    experiment_name: str


class TrainingPipeline:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def load_data(self) -> pd.DataFrame:
        raise NotImplementedError("Data loading is not implemented yet.")

    def clean_data(self, frame: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError("Data cleaning is not implemented yet.")

    def split_data(
        self, frame: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError("Data splitting is not implemented yet.")

    def train(self) -> None:
        raise NotImplementedError("Model training is not implemented yet.")

    def evaluate(self) -> dict[str, float]:
        raise NotImplementedError("Model evaluation is not implemented yet.")

    def run(self) -> None:
        raise NotImplementedError(
            "End-to-end pipeline orchestration is not implemented yet."
        )