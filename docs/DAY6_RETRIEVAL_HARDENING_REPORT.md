# DAY 6 - Retrieval, Evidence and Decision Calibration Hardening

## Scope and Constraints
- Goal: harden retrieval/evidence/decision quality for claim-centric correctness.
- Constraints honored:
  - no model retraining
  - no UI changes
  - existing API behavior preserved

## Root Causes Identified
1. Topic retrieval instead of claim retrieval:
- Previous query generation could over-index on entities (e.g., Water, Google, White House) and pull generic pages.

2. Weak evidence acceptance gate:
- Retrieved pages were ranked without an explicit assertion-level coverage check (subject/action/object).

3. Duplicate evidence inflation:
- Same or near-identical pages could appear multiple times and distort evidence quality.

4. Decision calibration gap:
- Classifier/evidence disagreement was handled, but explicit high-risk rules (A/B) were not enforced as first-class logic.

5. Limited audit trace of rejected evidence:
- Explanations did not clearly expose accepted vs rejected evidence counts for downstream debugging.

## Implemented Changes

### 1) Claim-Centric Retrieval Hardening
- File: `src/retrieval/query_generator.py`
- Changes:
  - Added claim assertion-focused token path (`_claim_focus_tokens`).
  - Prioritized claim-focused queries over entity-only queries.
  - De-prioritized generic entity queries that attract concept pages.

### 2) Evidence Relevance Filter and Rejection Logging
- File: `src/retrieval/relevance_filter.py` (new)
- Added:
  - `dedupe_documents(...)` for URL/title similarity dedup.
  - `score_claim_relevance(...)` with:
    - `entity_overlap_score`
    - `action_overlap_score`
    - `claim_coverage_score`
    - generic-page penalty and rejection reason.
  - Generic page penalties for concept-style pages (e.g., Water, Google, Smartphone, White House, UFO).

- File: `src/retrieval/pipeline.py`
- Changes:
  - Applied dedupe before ranking.
  - Applied claim coverage acceptance gate before ranking.
  - Stored `accepted_evidence` and `rejected_evidence` audit records in retrieval bundle.

### 3) Evidence Quality Penalty
- File: `src/agent/evidence_tool.py`
- Changes:
  - Added quality penalty for generic concept sources in `_evidence_quality_score(...)`.
  - Returned `accepted_evidence` and `rejected_evidence` in evidence output.

### 4) Decision Calibration Rules A/B/C
- File: `src/agent/decision_tool.py`
- Added explicit rules:
  - Rule A: classifier=fake + strong support + weak contradiction -> conflict -> UNCERTAIN.
  - Rule B: classifier=real + strong contradiction + weak support -> conflict -> UNCERTAIN.
  - Rule C: REAL/FAKE only when classifier direction aligns with evidence direction.
- Also expanded decision reason text to explicitly mention Rule A/Rule B when triggered.

### 5) Explanation and API Audit Trail
- File: `src/agent/controller.py`
- Changes:
  - Explanation text now includes accepted/rejected evidence counts.
  - Agent result now carries `accepted_evidence` and `rejected_evidence`.

- File: `src/api/schemas.py`, `src/api/routes.py`
- Changes:
  - Added `accepted_evidence` and `rejected_evidence` to `/agent` response schema and payload.
  - Kept backward compatibility using default empty lists.

## Benchmarks and Tests

### New Day 6 Tests
- File: `tests/retrieval/test_day6_hardening.py`
- Coverage:
  - dedup by URL/title similarity
  - reject generic Water page for specific cancer cure claim
  - accept assertion-focused refute evidence
  - decision Rule A conflict behavior
  - decision Rule B conflict behavior
  - aligned fake decision path

### Regression and Compatibility Tests Run
1. `pytest tests/retrieval/test_day6_hardening.py tests/retrieval/test_query_generator.py tests/retrieval/test_pipeline.py tests/api/test_agent_endpoint.py -q`
- Result: all passed.

2. `pytest tests/retrieval/test_retrieval_quality.py tests/agent/test_controller.py -q`
- Result: all passed.

3. `pytest tests/retrieval/test_day6_hardening.py tests/retrieval/test_retrieval_quality.py -q`
- Result: all passed.

## Before vs After (Behavioral Summary)
1. "Warm salt water cures all cancers in 3 days"
- Before: generic pages like "Water" could survive retrieval and influence support.
- After: generic pages are penalized/rejected by claim coverage + generic penalties; assertion-focused refute evidence is prioritized.

2. "Earth is flat" type claim with mixed scientific context
- Before: incidental support language in broader science text could blur stance.
- After: decision rules enforce UNCERTAIN when classifier/evidence conflict strongly; explicit Rule A/B conflict handling.

3. Generic entity pages (Google, Smartphone, White House)
- Before: entity-only retrieval could pass and inflate quality.
- After: entity-only query path de-prioritized and generic-page quality penalties applied.

## Limitations
1. Rule-based relevance and generic penalties can still miss edge cases with unusual wording.
2. Similar-title dedup uses heuristic similarity, not semantic clustering.
3. External web retrieval remains sensitive to search provider noise and snippet quality.
4. Claim coverage scoring currently uses lightweight token overlap, not full semantic entailment.

## Next Hardening Candidates (Optional)
1. Add semantic claim-evidence entailment score as a secondary gate.
2. Add domain-specific action lexicons per claim type.
3. Log rejected evidence examples in monitoring artifacts for drift analysis.
