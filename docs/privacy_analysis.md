# Privacy Analysis

## Data Protection Approach

- User-submitted article text is scanned for common PII before it reaches the predictor or agent workflow.
- Detected emails, phone numbers, URLs, and basic address patterns are replaced with neutral placeholders.
- Only masked text is passed to downstream components and monitoring logs.
- Original text is kept in memory only for the current request and is not persisted by the privacy layer.

## Expected Benefits

- Reduces the chance of storing or forwarding personal data in logs, metrics, or analysis outputs.
- Makes local debugging safer because logs and monitoring summaries do not contain direct PII values.

## Residual Risk

- The masking logic is intentionally lightweight and may miss unusual address formats or embedded identifiers.
- Human review is still required for highly sensitive or regulated deployments.
