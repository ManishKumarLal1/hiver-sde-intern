# Golden Evaluation Set — AmazonHelp

## Size
202 hand-labelled examples.

## Sampling
- Started from 6,893 one-turn pairs (AmazonHelp, 100k subsample of the
  Customer Support on Twitter dataset).
- LLM pre-labelled 500 candidates with the intent taxonomy.
- Human-reviewed a stratified sample: 155 rows distributed across 9 intents
  with rare intents over-sampled.
- Added ~50 hand-picked edge cases: multi-intent, sarcasm/anger, short
  fragments, very long messages, image references, non-English.

## Labelling
- Single annotator (the author) following `docs/intent_taxonomy.md` and
  `docs/escalation_policy.md`.
- LLM was used only to pre-select candidates and provide a hint; every row
  was human-reviewed and often corrected (~15% override rate).
- Automated QA audit (`src/audit_golden_set.py`) flagged 22 real errors,
  which were corrected before evaluation.

## Agreement
- Intra-annotator check: 30 rows re-labelled blindly 24h later.
- Intent κ = 0.613, escalation κ = 0.435.
- Full report: `results/agreement.md`.

## Files
- `golden_set_to_review.csv` — the final 202-row set
- `golden_set_second_pass.csv` — 30-row blind re-label for agreement

## Known limitations
- Single annotator; no inter-annotator agreement.
- ~15% non-English; multilingual handling not designed.
- No multi-turn context — each example is a single message.
- Zero examples for `legal_threat`, `high_value`, `safety_issue`.
- `cancellation` under-represented (4%).
- `other` is a residual bucket, not a coherent intent.