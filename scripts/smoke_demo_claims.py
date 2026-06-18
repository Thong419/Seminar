from __future__ import annotations

import json
from dataclasses import dataclass

import requests


API_URL = "http://localhost:8000/agent"
TIMEOUT_SECONDS = 90


@dataclass(frozen=True)
class DemoClaim:
    name: str
    text: str


CLAIMS = [
    DemoClaim(
        name="TRUMP_BANNON",
        text=(
            "WASHINGTON (Reuters) - U.S. President Donald Trump removed his chief strategist Steve Bannon "
            "from the National Security Council on Wednesday. The White House said the change was part of a "
            "routine restructuring to improve coordination of foreign policy and national security decisions."
        ),
    ),
    DemoClaim(
        name="PERSEVERANCE",
        text="NASA's Perseverance rover successfully collected rock samples from Jezero Crater on Mars.",
    ),
    DemoClaim(
        name="CANCER_HOAX",
        text=(
            "Scientists confirm that drinking warm salt water every morning can permanently cure all types of "
            "cancer within 3 days without any medical treatment."
        ),
    ),
    DemoClaim(
        name="SMARTPHONE_HOAX",
        text=(
            "The government secretly announced that smartphones will be banned worldwide starting next month to "
            "reduce human dependency on technology."
        ),
    ),
]


def run_smoke_test() -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for claim in CLAIMS:
        try:
            response = requests.post(API_URL, json={"text": claim.text}, timeout=TIMEOUT_SECONDS)
            payload = response.json()
            if response.status_code != 200:
                results.append({
                    "claim": claim.name,
                    "status_code": response.status_code,
                    "error": payload,
                })
                continue

            results.append(
                {
                    "claim": claim.name,
                    "prediction": payload.get("prediction"),
                    "human_review_state": payload.get("human_review_state"),
                    "evidence_found": payload.get("evidence_found"),
                    "support_score": round(float(payload.get("support_score", 0.0)), 3),
                    "contradiction_score": round(float(payload.get("contradiction_score", 0.0)), 3),
                    "evidence_quality_score": round(float(payload.get("evidence_quality_score", 0.0)), 3),
                    "top_titles": [item.get("title") for item in payload.get("sources", [])[:3]],
                    "rejected_titles": [item.get("title") for item in payload.get("rejected_evidence", [])[:5]],
                    "queries": payload.get("queries", [])[:4],
                }
            )
        except Exception as exc:
            results.append({"claim": claim.name, "error": repr(exc)})
    return results


if __name__ == "__main__":
    print(json.dumps(run_smoke_test(), indent=2, ensure_ascii=False))
