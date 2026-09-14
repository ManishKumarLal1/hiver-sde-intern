# Judge–human agreement

**Judge:** rule-based rubric (`src/judge_rules.py`)
**Sample:** 20 random rows from the 202 agent predictions
**Method:** rule-based `overall` vs. author's manual `overall`

## Headline numbers

| Metric | Value |
|--------|-------|
| Rows compared | 20 |
| Judge mean | 3.99 |
| Human mean | 2.55 |
| Mean absolute difference | 1.52 |
| Spearman ρ | 0.140 (p = 0.556) |
| Exact agreement | 0% |
| Within-1 agreement | 35% |

## Interpretation

The rule-based judge is **systematically generous** compared to human
evaluation — a mean gap of 1.44 points on a 5-point scale. It also has
**negligible rank agreement** with human ratings (ρ = 0.14; note the sample
is small, n = 20, and p > 0.5, so treat as directional evidence rather than
a precise estimate).

The most severe disagreements reveal *why* the rule-based judge fails:

| id | intent | judge | human | what went wrong |
|----|--------|-------|-------|-----------------|
| 154 | refund_return | 4.4 | 1.0 | reply answers a different question (delivery vs refund) |
| 57 | cancellation | 3.8 | 1.0 | "kindly wait" — no cancellation action |
| 46 | complaint | 3.4 | 1.0 | joke reply ("round of appaws 😁🐶❤️") |
| 160 | refund_return | 4.4 | 2.0 | off-topic reply (delivery vs refund) |
| 83 | other | 4.2 | 2.0 | generic escalation reply for a thanks message |

**Root cause:** the judge checks four surface-level criteria
(relevance = topic-word overlap; groundedness = vocabulary overlap with the
historical reply; tone = presence of polite words; actionability = presence
of "DM"/"link"/"check"; safety = absence of fabricated specifics).
None of these detect when the reply is **on-topic text that answers the
wrong question** — the dominant failure mode of retrieval-based generation.

## Implications for the report

1. **The headline reply-quality number (judge mean ≈ 4.0) is misleading.**
   It reflects surface-level fluency, not customer-perceived quality.
   Human evaluation puts the agent at ~2.6/5.

2. **A rule-based judge should not be trusted for final quality decisions.**
   It is useful as a fast first-pass signal (it's right that the agent's
   replies *look* professional) but it cannot distinguish between a good
   reply and a well-written reply to the wrong question.

3. **Future work should replace this judge with an LLM judge** that can
   verify intent alignment. The current rule-based judge remains as a
   lightweight fallback for environments where no LLM is available.

## Caveats

- Sample size n = 20 is small; the ρ estimate is noisy. The mean absolute
  difference (1.52) is a more stable signal.
- Human scores are from a single annotator (the author). Cross-annotator
  agreement on the human side is not measured here.
- Rule-based scoring thresholds were not tuned; a tuned rubric might
  achieve higher agreement, but likely still misses the "wrong question"
  failure mode.
  