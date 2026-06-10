"""Token importance ranking utilities."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from pathlib import Path
from typing import Iterable, Sequence
import json


SPECIAL_TOKEN_PATTERN = re.compile(r"^\s*$|^<[^>]+>$|^\[.*\]$")


@dataclass(frozen=True, slots=True)
class TokenImportance:
    token: str
    importance: float

    def as_dict(self) -> dict[str, str | float]:
        return asdict(self)


def extract_token_importance(
    tokens: Sequence[str],
    values: Sequence[float],
    top_k: int = 8,
) -> list[dict[str, str | float]]:
    if len(tokens) != len(values):
        raise ValueError("Tokens and values must have the same length.")

    ranked: list[TokenImportance] = []
    for token, value in zip(tokens, values, strict=True):
        normalized = _normalize_token(token)
        if not normalized:
            continue
        ranked.append(TokenImportance(token=normalized, importance=round(abs(float(value)), 6)))

    ranked.sort(key=lambda item: item.importance, reverse=True)
    return [item.as_dict() for item in ranked[:top_k]]


def save_token_importance(token_importance: list[dict[str, str | float]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(token_importance, handle, indent=2)
    return output_path


def _normalize_token(token: str) -> str:
    cleaned = token.strip()
    if SPECIAL_TOKEN_PATTERN.match(cleaned):
        return ""
    cleaned = cleaned.replace("Ġ", "").replace("▁", "").replace("##", "")
    cleaned = cleaned.strip(".,;:!?()[]{}\"'")
    return cleaned
