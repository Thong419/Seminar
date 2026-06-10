# Project Structure

## Major Directories

- `src/api/`: FastAPI app, routers, schemas, and dependency providers.
- `src/agents/`: agent workflow, decision logic, retrieval, analysis, and explanation.
- `src/config/`: typed configuration loading.
- `src/data/`: dataset loading, preprocessing, and splitting.
- `src/evaluation/`: metric computation and reports.
- `src/explainability/`: SHAP integration, token importance, and formatting.
- `src/fairness/`: bias analysis utilities.
- `src/features/`: feature engineering helpers.
- `src/inference/`: prediction interfaces.
- `src/monitoring/`: logging, drift detection, health scoring, and retraining recommendations.
- `src/privacy/`: PII detection and masking.
- `src/robustness/`: adversarial testing and robustness reports.
- `src/training/`: baseline and transformer training pipelines.
- `src/utils/`: shared helpers.
- `frontend/`: Streamlit frontend and reusable UI pieces.
- `configs/`: YAML configuration files.
- `docs/`: deployment, architecture, demo, and responsible AI documentation.
- `artifacts/`: generated logs, reports, and charts.
- `tests/`: automated tests for the main application slices.
