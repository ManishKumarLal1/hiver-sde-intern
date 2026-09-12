import pandas as pd
from pathlib import Path

p = Path(__file__).resolve().parent.parent / 'golden_set' / 'golden_set_to_review.csv'
df = pd.read_csv(p)

cols = ['intent', 'escalation_ground_truth', 'escalation_reason',
        'difficulty', 'golden_reply', 'notes']

for c in cols:
    df[c] = df[c].fillna('').astype('object')

df.to_csv(p, index=False)
print("Recast done:", p)
print(df.dtypes)
print(f"Labelled: {(df['intent'].fillna('') != '').sum()} / {len(df)}")