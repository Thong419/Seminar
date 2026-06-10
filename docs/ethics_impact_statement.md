# Ethics Impact Statement

## Who Benefits

- Readers seeking quick credibility checks for news articles.
- Analysts and moderators who need evidence-backed triage.
- Teams that need safer logging and better visibility into model behavior.

## Who May Be Harmed

- Writers of satire, opinion pieces, and ambiguous reporting may be misclassified.
- Communities represented in underbalanced training data may experience uneven performance.

## Risks of False Positives

- Legitimate articles may be labeled fake, reducing trust in real reporting.
- Users may over-rely on the system and incorrectly dismiss valid information.

## Risks of False Negatives

- Misinformation may be treated as credible and spread further.
- Weak confidence in the model may delay human review of harmful content.

## Explainability Benefits

- Token-level signals, evidence summaries, and trust scores support human interpretation.
- Explanations make failures easier to diagnose and review.

## Potential Misuse

- The system could be used to suppress inconvenient but legitimate viewpoints.
- Attackers may learn how to evade the detector by manipulating surface features.

## Human Oversight Recommendations

- Treat the model as a decision-support tool, not an authority.
- Escalate low-confidence or conflicting-evidence cases to human reviewers.
- Review fairness and robustness reports on a recurring schedule.
