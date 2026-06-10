# Robustness Analysis

## Evaluation Scope

- Spelling mistakes
- Noisy text
- Uppercase text
- Repeated punctuation
- Shortened articles
- Out-of-domain text

## Reporting

- The robustness report stores the original and adversarial predictions, confidence changes, and an aggregate robustness score.
- Reports are written to `artifacts/robustness/robustness_report.json`.

## Failure Cases

- Satire news may be classified as fake even when the article is intentionally non-literal.
- Ambiguous articles can produce unstable confidence under perturbations.
- Shortened or out-of-domain text may not contain enough evidence for a stable decision.
