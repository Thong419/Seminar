# Deployment Architecture

```mermaid
flowchart TD
    U[Browser] --> F[Streamlit Container]
    F --> B[FastAPI Container]
    B --> M[Model Artifacts]
    B --> R[Retrieval / Explainability / Monitoring]
    B --> A[Artifacts and Reports]
```
