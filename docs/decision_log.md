## Golden set limitations
- 202 rows, all 9 intents represented
- 0 rows for `legal_threat`, `high_value`, `safety_issue` — rare in source data
- 67/33 escalate vs. auto_handle split — reflects complaint-heavy traffic
- ~7% error rate corrected across 2 QA passes (see audit script in src/)