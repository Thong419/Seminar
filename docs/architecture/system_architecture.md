# System Architecture

```mermaid
flowchart TD
    U[User] --> S[Streamlit Frontend]
    S --> A[FastAPI Backend]
    A --> W[Agent Workflow]
    W --> R[RoBERTa]
    W --> E[Evidence Retrieval]
    W --> X[Explainability]
```
