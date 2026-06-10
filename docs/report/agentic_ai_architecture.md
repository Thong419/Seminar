# Agentic AI Architecture

The agentic layer coordinates classification, confidence checking, evidence retrieval, evidence analysis, decision making, and explanation generation.

```mermaid
flowchart TD
    A[Article] --> C[Classification]
    C --> Q{Confidence Check}
    Q -->|High confidence| D[Decision]
    Q -->|Low confidence| R[Retrieve Evidence]
    R --> E[Evidence Analysis]
    E --> D
    D --> X[Explainability]
```

## Report Angle

- Emphasize modular reasoning.
- Explain why retrieval is only triggered when uncertainty is high.
- Describe how the final explanation integrates model, evidence, and trust signals.
