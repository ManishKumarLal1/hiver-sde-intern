import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / 'golden_set' / 'golden_set_to_review.csv'
OUT = ROOT / 'predictions' / 'baseline_trivial.csv'

df = pd.read_csv(GOLDEN, keep_default_na=False, na_values=[''])
df = df[df['intent'] != ''].copy()

majority = df['intent'].value_counts().idxmax()
print(f"Majority intent: {majority}")

out = df[['id']].copy()
out['predicted_intent'] = majority
out['predicted_escalation'] = 'escalate'
out['predicted_reason'] = 'high_emotion'
out['predicted_reply'] = ("We're sorry to hear about your issue. "
                          "Please DM us your order number and we'll look into it.")

OUT.parent.mkdir(exist_ok=True)
out.to_csv(OUT, index=False)
print(f"Saved {len(out)} predictions to {OUT}")
