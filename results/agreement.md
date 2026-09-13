# Inter-annotator agreement — golden set

**Method:** Same annotator (intra-annotator), re-labelled 30 rows blindly ~24h after first pass, stratified across 9 intents.

| Metric | Raw agreement | Cohen's κ | Interpretation |
|--------|---------------|-----------|----------------|
| Intent | 66.7% | 0.613 | Substantial |
| Escalation | 70.0% | 0.435 | Moderate |

## Intent disagreements (10)

| id | pass 1 | pass 2 | category |
|----|--------|--------|----------|
| 17 | info_request | billing_payment | ambiguous (charge policy Q) |
| 43 | refund_return | complaint | pass 2 better (no ask) |
| 62 | complaint | order_status | ambiguous (emotional delivery failure) |
| 82 | complaint | info_request | pass 1 better |
| 97 | complaint | order_status | pass 1 better |
| 112 | info_request | other | pass 1 better |
| 126 | product_issue | other | pass 1 better |
| 142 | complaint | refund_return | pass 1 better |
| 169 | refund_return | order_status | ambiguous (return tracking) |
| 194 | cancellation | info_request | pass 1 better |

## Interpretation

- Intent κ = 0.613 falls in the "substantial agreement" band (Landis & Koch 1977).
- ~4 of 10 disagreements are genuine taxonomy ambiguity (charge policy Q, emotional delivery, return vs. delivery tracking).
- ~6 of 10 are drift in the second pass (fragments and short messages are the hardest).
- Escalation κ (0.435) is depressed by the intent disagreements — a wrong intent almost always drags escalation with it.

## Implications

- The taxonomy has known-hard boundaries: (a) complaints that reference delivery context, (b) info-requests about billing mechanics, (c) return-vs-delivery tracking.
- These are documented as limitations in the main report.
- The classifier's accuracy ceiling is bounded by this ambiguity — even a perfect model cannot exceed human agreement.