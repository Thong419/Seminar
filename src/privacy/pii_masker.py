"""PII masking utilities used before model inference and logging."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from src.privacy.pii_detector import PIIEntity, detect_pii


@dataclass(frozen=True, slots=True)
class MaskedText:
    text: str
    entities: list[PIIEntity]

    def as_dict(self) -> dict[str, object]:
        return {"text": self.text, "entities": [entity.as_dict() for entity in self.entities]}


MASK_TOKENS = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "url": "[URL]",
    "address": "[ADDRESS]",
}


def mask_pii(text: str) -> MaskedText:
    """Replace detected PII with stable placeholders while keeping the original only in memory."""

    entities = detect_pii(text)
    if not entities:
        return MaskedText(text=text, entities=[])

    masked_parts: list[str] = []
    cursor = 0
    for entity in entities:
        masked_parts.append(text[cursor:entity.start])
        masked_parts.append(MASK_TOKENS.get(entity.entity_type, "[PII]"))
        cursor = entity.end
    masked_parts.append(text[cursor:])
    return MaskedText(text="".join(masked_parts), entities=entities)
