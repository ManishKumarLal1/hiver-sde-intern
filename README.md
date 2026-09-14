# Hiver SDE Intern — AmazonHelp Support Agent

An AI support agent for AmazonHelp built from the Customer Support on Twitter
dataset. Includes a 202-row hand-labelled golden evaluation set, three baselines
(trivial, simple, hybrid), a rule-based reply judge with human-agreement
validation, and a full evaluation harness.

**Full write-up:** [report.md](report.md)
**Decision log:** [docs/decision_log.md](docs/decision_log.md)

## Quick start — reproduce headline results in under 2 minutes

    pip install -r requirements.txt
    make evaluate

This runs the hybrid agent on the 202-row golden set, applies the reason
post-processor, and evaluates all three systems. Results are written to
`results/eval_*.json`.

## Headline numbers

| System | Intent Acc | Macro F1 | Esc F1 | Reason Agr | Judge Q | Human Q (n=20) |
|--------|-----------|----------|--------|-----------|---------|----------------|
| Trivial | 0.322 | 0.054 | 0.401 | 0.287 | — | — |
| Simple | 0.391 | 0.382 | 0.561 | 0.411 | — | — |
| Hybrid agent | 0.391 | 0.382 | 0.561 | 0.312 | 3.99 | 2.55 |

## Data setup

1. Download `tweets.csv` from
   [Kaggle: Customer Support on Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter).
2. Save the first 100k rows as `data/raw/tweets_100k.csv`.
3. Run:

       python src/data_loader.py
       python src/preprocess.py

   This produces `data/processed/pairs_cleaned_5k.csv`, which the agent uses
   for retrieval.

## Repo structure

    report.md                    — final 9-section report
    docs/
        intent_taxonomy.md       — 9-intent definitions and rules
        escalation_policy.md     — frozen auto/escalate policy
        decision_log.md          — 20 non-obvious decisions
    golden_set/
        golden_set_to_review.csv — 202 hand-labelled examples
        golden_set_second_pass.csv — 30-row blind re-label for agreement
        README.md                — sampling and labelling note
    src/
        agent_hybrid.py          — deployed agent
        fix_agent_reasons.py     — escalation/reason post-processor
        baseline_trivial.py      — majority-class baseline
        baseline_simple.py       — TF-IDF + LogReg baseline
        eval_harness.py          — automated metrics
        judge_rules.py           — rule-based reply quality judge
        audit_golden_set.py      — QA rules for the golden set
    predictions/                 — outputs of each system
    results/                     — all metrics as JSON + Markdown

## Reproduce the baselines

    python src/baseline_trivial.py
    python src/eval_harness.py predictions/baseline_trivial.csv

    python src/baseline_simple.py
    python src/eval_harness.py predictions/baseline_simple.csv

## Evaluate reply quality

    python src/judge_rules.py predictions/agent.csv

Writes `results/judge_agent.csv`. Human agreement analysis is in
`results/judge_human_agreement.md`.

## Known limitations

- Single-annotator golden set; intra-annotator κ = 0.613 (intent) / 0.435 (escalation).
- LLM-based agent attempts failed on infrastructure limits (documented in §6 of the report).
- Three escalation reasons have zero golden-set examples: `legal_threat`,
  `high_value`, `safety_issue`.
- Deployed agent is not multilingual-aware.