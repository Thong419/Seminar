"""Prediction logging for production inference traffic."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
import csv


@dataclass(frozen=True, slots=True)
class PredictionLogEntry:
    timestamp: str
    prediction: str
    confidence: float
    trust_score: int
    article_length: int
    request_id: str | None = None
    endpoint: str = "predict"

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class PredictionLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_prediction(
        self,
        prediction: str,
        confidence: float,
        trust_score: int,
        article_length: int,
        request_id: str | None = None,
        endpoint: str = "predict",
        timestamp: datetime | None = None,
    ) -> PredictionLogEntry:
        entry = PredictionLogEntry(
            timestamp=(timestamp or datetime.now(UTC)).isoformat(),
            prediction=prediction,
            confidence=float(confidence),
            trust_score=int(trust_score),
            article_length=int(article_length),
            request_id=request_id,
            endpoint=endpoint,
        )
        self._append(entry)
        return entry

    def _append(self, entry: PredictionLogEntry) -> None:
        write_header = not self.log_path.exists() or self.log_path.stat().st_size == 0
        with self.log_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(entry.as_dict().keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(entry.as_dict())
