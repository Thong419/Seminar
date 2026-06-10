"""Synthetic adversarial perturbations for robustness evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random
import re


@dataclass(frozen=True, slots=True)
class AdversarialCase:
    name: str
    text: str

    def as_dict(self) -> dict[str, str]:
        return asdict(self)


def build_adversarial_cases(article_text: str) -> list[AdversarialCase]:
    """Create a small adversarial suite covering common noisy-text failure modes."""

    return [
        AdversarialCase(name="original", text=article_text),
        AdversarialCase(name="spelling_noise", text=_spelling_noise(article_text)),
        AdversarialCase(name="uppercase", text=article_text.upper()),
        AdversarialCase(name="repeated_punctuation", text=_repeat_punctuation(article_text)),
        AdversarialCase(name="shortened", text=_shorten(article_text)),
        AdversarialCase(name="out_of_domain", text="Recipe: mix ingredients and bake at 350 degrees until golden."),
        AdversarialCase(name="noisy_spacing", text=_insert_extra_spacing(article_text)),
    ]


def _spelling_noise(text: str) -> str:
    replacements = {"the": "teh", "with": "wth", "claim": "cliam", "article": "artcle", "evidence": "evedince"}
    words = [replacements.get(word.lower(), word) for word in text.split()]
    return " ".join(words)


def _repeat_punctuation(text: str) -> str:
    return re.sub(r"([.!?])", r"\1\1\1", text)


def _shorten(text: str) -> str:
    words = text.split()
    return " ".join(words[: max(5, len(words) // 3)])


def _insert_extra_spacing(text: str) -> str:
    return re.sub(r"\s+", "  ", text)
