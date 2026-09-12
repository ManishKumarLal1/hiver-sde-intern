"""
Downgrade difficulty from 'easy' to 'medium' for rows where the final intent
differs from the LLM hint. If you overrode the hint, the row wasn't easy.

Run from project root: python src/downgrade_difficulty.py
"""
import pandas as pd
from pathlib import Path

p = Path(__file__).resolve().parent.parent / 'golden_set' / 'golden_set_to_review.csv'
df = pd.read_csv(p, keep_default_na=False, na_values=[''])

COLS = ['intent','escalation_ground_truth','escalation_reason',
        'difficulty','golden_reply','notes']
for c in COLS:
    df[c] = df[c].fillna('').astype('object')

# Build a mask of rows to downgrade
to_downgrade = []
for i, row in df.iterrows():
    if row['difficulty'] != 'easy':
        continue
    if row['intent'] == '':
        continue  # unlabelled, skip

    hint_v2 = row.get('intent_llm_guess_v2', '')
    hint_v1 = row.get('intent_llm_guess', '')
    hint = ''
    if isinstance(hint_v2, str) and hint_v2.strip():
        hint = hint_v2
    elif isinstance(hint_v1, str) and hint_v1.strip():
        hint = hint_v1

    if hint and hint != row['intent']:
        to_downgrade.append(i)

print(f"Rows to downgrade (easy -> medium): {len(to_downgrade)}")
if to_downgrade:
    print("\nSample:")
    for i in to_downgrade[:10]:
        row = df.loc[i]
        hint_v2 = row.get('intent_llm_guess_v2', '') or row.get('intent_llm_guess', '')
        print(f"  id={row['id']:>4}  hint={hint_v2:>15}  final={row['intent']:>15}  | {str(row['customer_msg_clean'])[:70]}")

    df.loc[to_downgrade, 'difficulty'] = 'medium'
    df.to_csv(p, index=False)
    print(f"\nDowngraded {len(to_downgrade)} rows.")
else:
    print("No rows to downgrade. Nothing to do.")

print("\nFinal difficulty distribution:")
print(df[df['intent'] != '']['difficulty'].value_counts())
