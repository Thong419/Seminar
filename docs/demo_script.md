# Demo Script

## Startup Steps

1. Run `docker compose up --build`.
2. Open the Streamlit frontend at `http://localhost:8501`.
3. Verify the FastAPI health endpoint at `http://localhost:8000/health`.

## Sample Articles

- Verified news article: should trend toward real or likely real.
- Sensational claim with no evidence: should trend toward fake or likely fake.
- Short or ambiguous text: should surface uncertainty.

## Expected Outputs

- Prediction badge.
- Confidence score.
- Trust score with progress bar.
- Important token table.
- Evidence cards with clickable links.
- Final explanation and evidence summary.

## Explanation Demo

Show how the explanation combines predicted label, trust score, evidence summary, and token importance.

## Monitoring Demo

Show entries under `artifacts/monitoring/predictions.csv` and the drift, confidence, and health reports in `artifacts/monitoring/`.
