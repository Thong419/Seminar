"""Basic PII detection utilities for privacy protection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re


@dataclass(frozen=True, slots=True)
class PIIEntity:
    entity_type: str
    value: str
    start: int
    end: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


EMAIL_PATTERN = re.compile(r"(?P<email>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})")
PHONE_PATTERN = re.compile(
    r"(?P<phone>(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?){1,2}\d{3,4}(?:[\s.-]?\d{3,4})?)"
)
URL_PATTERN = re.compile(r"(?P<url>https?://[^\s]+|www\.[^\s]+)", re.IGNORECASE)
ADDRESS_PATTERN = re.compile(
    r"(?P<address>\b\d{1,5}\s+[A-Za-z0-9.'-]+(?:\s+[A-Za-z0-9.'-]+){0,4}\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way)\b)",
    re.IGNORECASE,
)


def detect_pii(text: str) -> list[PIIEntity]:
    """Detect common privacy-sensitive entities in free text."""

    entities: list[PIIEntity] = []
    for entity_type, pattern in (
        ("email", EMAIL_PATTERN),
        ("phone", PHONE_PATTERN),
        ("url", URL_PATTERN),
        ("address", ADDRESS_PATTERN),
    ):
        for match in pattern.finditer(text):
            value = match.group(entity_type)
            if value:
                entities.append(
                    PIIEntity(
                        entity_type=entity_type,
                        value=value,
                        start=match.start(entity_type),
                        end=match.end(entity_type),
                    )
                )

    entities.sort(key=lambda item: (item.start, item.end))
    return entities
