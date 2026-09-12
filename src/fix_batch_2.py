import pandas as pd
from pathlib import Path

p = Path(__file__).resolve().parent.parent / 'golden_set' / 'golden_set_to_review.csv'
df = pd.read_csv(p, keep_default_na=False, na_values=[''])

COLS = ['intent','escalation_ground_truth','escalation_reason','difficulty','notes']
for c in COLS:
    df[c] = df[c].fillna('').astype('object')

FIXES = [
    (88,  'complaint',      'escalate',    'high_emotion',       'medium', 'Sarcastic delivery complaint'),
    (180, 'billing_payment','escalate',    'financial_risk',     'medium', 'Specific false charge, not info'),
    (26,  'billing_payment','escalate',    'financial_risk',     'medium', 'Charged but not shipped; billing substance'),
    (201, 'product_issue',  'auto_handle', 'deterministic_action','medium','App showing raw error string; bug report'),
]

for rid, intent, esc, reason, diff, note in FIXES:
    df.loc[df['id'] == rid, COLS] = [intent, esc, reason, diff, note]

df.to_csv(p, index=False)
print(f"Applied {len(FIXES)} fixes")