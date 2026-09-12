import pandas as pd
from pathlib import Path
import sys

ROW_ID = int(sys.argv[1]) if len(sys.argv) > 1 else 34
NEW_INTENT = sys.argv[2] if len(sys.argv) > 2 else 'order_status'
NEW_ESC    = sys.argv[3] if len(sys.argv) > 3 else 'auto_handle'
NEW_REASON = sys.argv[4] if len(sys.argv) > 4 else 'low_risk'
NEW_DIFF   = sys.argv[5] if len(sys.argv) > 5 else 'easy'
NEW_NOTE   = sys.argv[6] if len(sys.argv) > 6 else ''

p = Path(__file__).resolve().parent.parent / 'golden_set' / 'golden_set_topup_to_review.csv'
df = pd.read_csv(p, keep_default_na=False, na_values=[''])

for c in ['intent', 'escalation_ground_truth', 'escalation_reason',
          'difficulty', 'golden_reply', 'notes']:
    df[c] = df[c].fillna('').astype('object')

df.loc[df['id'] == ROW_ID, ['intent', 'escalation_ground_truth',
                            'escalation_reason', 'difficulty', 'notes']] = \
    [NEW_INTENT, NEW_ESC, NEW_REASON, NEW_DIFF, NEW_NOTE]

df.to_csv(p, index=False)
print(f"Fixed id={ROW_ID}: {NEW_INTENT} / {NEW_ESC} / {NEW_REASON} / {NEW_DIFF}")
