# Baselines — Day 4

| System | Intent Acc | Intent Macro F1 | Esc Macro F1 | Reason Agr |
|--------|-----------|-----------------|--------------|------------|
| Trivial (majority intent + always escalate) | 0.322 | 0.054 | 0.401 | 0.287 |
| Simple (TF-IDF + LogReg, 5-fold OOF + rule escalation) | 0.391 | 0.382 | 0.561 | 0.411 |

## Notes
- Simple baseline uses `cross_val_predict(cv=5)` to avoid training-set leakage.
  In-sample accuracy would be 0.90 but is not reported.
- Confusion is concentrated between complaint / order_status / refund_return,
  which is where the LLM-based agent must improve.
- Reply quality not yet measured — that's the LLM-judge stage on Day 5.
EOF