"""Classify the type of a claim to enable domain-aware evidence retrieval.

Supported claim types:
    medical_claim    – health, disease, treatment, cure, drug
    science_claim    – scientific findings, experiments, discoveries
    political_claim  – government, elections, political figures
    technology_claim – tech products, AI, internet, cyber
    economic_claim   – economy, markets, trade, finance
    general_claim    – fallback when no domain matches
"""

from __future__ import annotations

import re
from typing import Final

CLAIM_TYPES: Final = (
    "medical_claim",
    "science_claim",
    "political_claim",
    "technology_claim",
    "economic_claim",
    "general_claim",
)

_MEDICAL = re.compile(
    r"\b(cure[sd]?|curing|treatment|treat(s|ed|ing)?|disease|cancer|diabetes|covid|"
    r"vaccine|vaccin(at|e)|virus|bacteria|infection|symptom|pill|drug|medicine|medical|"
    r"hospital|doctor|patient|therapy|surgery|immune|heart|blood|brain|lung|organ|"
    r"heal(s|ed|ing)?|remedy|clinical|trial|dose|side.?effect|syndrome|chronic|"
    r"pharmaceutical|antibiotic|prescription|diagnosis|epidemic|pandemic|pathogen|"
    r"WHO|NIH|CDC|pubmed|oncol|tumor|tumour|carcinoma|chemotherapy|radiation therapy|"
    r"caffeine|supplement|vitamin|mineral|nutrient|detox|inflammation)\b",
    re.IGNORECASE,
)

_SCIENCE = re.compile(
    r"\b(scientist[s]?|researcher[s]?|stud(y|ies)|research|discover(y|ed|s)|"
    r"finding|experiment|laborator(y|ies)|nasa|physics|chemistry|biology|"
    r"gen(e|es|etic)|dna|rna|species|evolution|climate|planet|space|asteroid|"
    r"quantum|radiation|protein|cell|molecule|atom|theory|hypothesis|proof|"
    r"peer.?review|journal|nature|science|scientif|empirical|observation|"
    r"telescope|photon|particle|fossil|geology|oceanograph|meteorolog)\b",
    re.IGNORECASE,
)

_POLITICAL = re.compile(
    r"\b(president|senator|congress(man|woman)?|minister|parliament|government|"
    r"election|vote[sd]?|policy|democrat|republican|party|white.?house|legislation|"
    r"bill|law|treaty|diplomat(ic)?|politician|trump|biden|obama|clinton|bush|"
    r"administration|scandal|corruption|executive.?order|sanction[s]?|"
    r"secretary.?of.?state|supreme.?court|federal|governor|mayor|referendum)\b",
    re.IGNORECASE,
)

_TECHNOLOGY = re.compile(
    r"\b(smartphone|iphone|android|computer|software|hardware|artificial.?intelligence|"
    r"algorithm|internet|social.?media|facebook|twitter|google|amazon|microsoft|apple|"
    r"app(lication)?|cyber|robot(ic[s]?)?|automation|blockchain|cryptocurrency|bitcoin|"
    r"chip|processor|5g|wifi|cloud|tech(nolog)?|machine.?learning|deep.?learning|"
    r"hack(er[s]?|ing)?|data.?breach|surveillance|encryption|quantum.?computing|"
    r"semiconductor|autonomous.?vehicle|drone)\b",
    re.IGNORECASE,
)

_ECONOMIC = re.compile(
    r"\b(economy|gdp|inflation|recession|stock|market|trade|tariff|deficit|debt|"
    r"budget|unemployment|job[s]?|wage[s]?|salary|bank(ing)?|federal.?reserve|"
    r"interest.?rate|currency|dollar|tax(es)?|fiscal|monetary|profit|loss|revenue|"
    r"investment|financ(e|ial)|economic|imf|world.?bank|treasury|export|import|"
    r"supply.?chain|consumer.?price|poverty|inequality|subsid(y|ies))\b",
    re.IGNORECASE,
)

_PATTERNS = (
    ("medical_claim", _MEDICAL),
    ("science_claim", _SCIENCE),
    ("political_claim", _POLITICAL),
    ("technology_claim", _TECHNOLOGY),
    ("economic_claim", _ECONOMIC),
)


def classify_claim_type(text: str) -> str:
    """Return the most likely claim type for the given text.

    Args:
        text: Raw article text or claim sentence.

    Returns:
        One of CLAIM_TYPES. Defaults to 'general_claim' when no domain matches.
    """
    scores: dict[str, int] = {}
    for claim_type, pattern in _PATTERNS:
        matches = pattern.findall(text)
        scores[claim_type] = len(matches)

    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        return "general_claim"
    return best


def get_priority_domains(claim_type: str) -> list[str]:
    """Return a list of high-credibility domains for the given claim type.

    These are used to bias evidence retrieval toward authoritative sources.
    """
    _DOMAIN_MAP: dict[str, list[str]] = {
        "medical_claim": [
            "nih.gov",
            "who.int",
            "cdc.gov",
            "cancer.gov",
            "pubmed.ncbi.nlm.nih.gov",
            "mayoclinic.org",
            "nejm.org",
            "thelancet.com",
            "bmj.com",
            "healthline.com",
        ],
        "science_claim": [
            "nasa.gov",
            "nature.com",
            "science.org",
            "sciencedaily.com",
            "newscientist.com",
            "phys.org",
            "scientificamerican.com",
            "pnas.org",
            "esa.int",
            "noaa.gov",
        ],
        "political_claim": [
            "reuters.com",
            "apnews.com",
            "bbc.com",
            "politifact.com",
            "factcheck.org",
            "snopes.com",
            "washingtonpost.com",
            "congress.gov",
            "whitehouse.gov",
        ],
        "technology_claim": [
            "wired.com",
            "arstechnica.com",
            "techcrunch.com",
            "ieee.org",
            "acm.org",
            "technologyreview.com",
            "zdnet.com",
            "theverge.com",
        ],
        "economic_claim": [
            "imf.org",
            "worldbank.org",
            "bls.gov",
            "federalreserve.gov",
            "wsj.com",
            "ft.com",
            "bloomberg.com",
            "economist.com",
        ],
        "general_claim": [
            "reuters.com",
            "apnews.com",
            "bbc.com",
            "snopes.com",
            "factcheck.org",
            "politifact.com",
        ],
    }
    return _DOMAIN_MAP.get(claim_type, _DOMAIN_MAP["general_claim"])
