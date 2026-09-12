import pandas as pd
from pathlib import Path

ROW_ID = 35   # change this to whichever id you want to reset

p = Path('golden_set/golden_set_to_review.csv')
df = pd.read_csv(p)

cols = ['intent', 'escalation_ground_truth', 'escalation_reason',
        'difficulty', 'golden_reply', 'notes']

# Force these columns to string type first (they may currently be float64 due to all-NaN)
for c in cols:
    df[c] = df[c].astype('object')

df.loc[df['id'] == ROW_ID, cols] = ''
df.to_csv(p, index=False)

done = (df['intent'].fillna('') != '').sum()
print(f"Reset row id={ROW_ID}")
print(f"Labelled now: {done} / {len(df)}")