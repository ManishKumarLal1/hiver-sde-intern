import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
p = ROOT / 'golden_set' / 'golden_set_to_review.csv'
out = ROOT / 'golden_set' / 'golden_set_second_pass.csv'

df = pd.read_csv(p, keep_default_na=False, na_values=[''])
df = df[df['intent'] != ''].copy()

parts = []
for intent in df['intent'].unique():
    sub = df[df['intent'] == intent]
    n = max(2, round(30 * len(sub) / len(df)))
    parts.append(sub.sample(min(n, len(sub)), random_state=99))

sample = pd.concat(parts)
if len(sample) > 30:
    sample = sample.sample(30, random_state=99)
elif len(sample) < 30:
    extra = df[~df['id'].isin(sample['id'])].sample(30 - len(sample), random_state=99)
    sample = pd.concat([sample, extra])

# Shuffle so you don't see intent ordering
sample = sample.sample(frac=1, random_state=42).reset_index(drop=True)

second = sample[['id', 'customer_msg_clean', 'brand_reply_clean']].copy()
second['intent'] = ''
second['escalation_ground_truth'] = ''

second.to_csv(out, index=False)
print(f"Saved {len(second)} rows to {out}")
print(f"ids: {second['id'].tolist()}")
print(f"\nSample composition (do NOT look at this while labelling):")
print(sample['intent'].value_counts())