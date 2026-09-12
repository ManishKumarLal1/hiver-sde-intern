"""
One-time fix: recast review columns to object dtype and reset row 35.
Run from project root: python src/fix_csv.py
"""
import pandas as pd
from pathlib import Path

p = Path('golden_set/golden_set_to_review.csv')
df = pd.read_csv(p)

REVIEW_COLS = ['intent', 'escalation_ground_truth', 'escalation_reason',
               'difficulty', 'golden_reply', 'notes']

# Force all review columns to plain object/string dtype
for c in REVIEW_COLS:
    df[c] = df[c].astype('object').where(df[c].notna(), '')

# Reset row 35 (in case a partial save happened)
df.loc[df['id'] == 35, REVIEW_COLS] = ''

df.to_csv(p, index=False)

print("Fix complete.")
print("dtypes:")
print(df[REVIEW_COLS].dtypes)
print()
print(f"Labelled: {(df['intent'] != '').sum()} / {len(df)}")