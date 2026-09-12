import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / 'data' / 'processed' / 'pairs_cleaned_full.csv'
GOLDEN = ROOT / 'golden_set' / 'golden_set_to_review.csv'

# Load existing golden set
golden = pd.read_csv(GOLDEN, keep_default_na=False, na_values=[''])
used_msgs = set(golden['customer_msg_clean'])

# Load pool, remove used
pool = pd.read_csv(POOL)
pool = pool[~pool['customer_msg_clean'].isin(used_msgs)].copy()
print(f"Pool available: {len(pool)}")

# Keyword patterns for under-represented intents
PATTERNS = {
    'order_status':    r'\b(?:where is my|tracking|when will|delivery date|hasn\'?t arrived|not delivered|still waiting for (?:my )?package|expected delivery)\b',
    'billing_payment': r'\b(?:charge[ds]?|billing|overcharg|double.{0,5}charg|unauthori[sz]ed|deduct|payment.{0,10}(?:fail|declin)|extra.{0,10}(?:amount|charg)|wrong.{0,10}charg)\b',
    'cancellation':    r'\b(?:cancel|close.{0,10}account|unsubscribe|stop.{0,10}subscription|delete.{0,10}account)\b',
    'account_access':  r'\b(?:locked out|cannot (?:log|sign) ?in|password reset|hacked|forgot password)\b',
}

TARGETS = {
    'order_status': 25,
    'billing_payment': 10,
    'cancellation': 8,
    'account_access': 5,
}

rows = []
for intent, pattern in PATTERNS.items():
    hits = pool[pool['customer_msg_clean'].str.contains(pattern, case=False, na=False, regex=True)]
    hits = hits[hits['customer_msg_clean'].str.len() >= 15]
    n = min(TARGETS[intent], len(hits))
    hits = hits.sample(n, random_state=42).copy()
    hits['intent_llm_guess'] = intent
    rows.append(hits)
    print(f"{intent}: {n} candidates")

extra = pd.concat(rows).drop_duplicates(subset='customer_msg_clean').reset_index(drop=True)
extra['intent'] = ''
extra['escalation_ground_truth'] = ''
extra['escalation_reason'] = ''
extra['golden_reply'] = ''
extra['difficulty'] = ''
extra['is_edge_case'] = False
extra['notes'] = ''

# Match columns of golden set
for c in golden.columns:
    if c not in extra.columns:
        extra[c] = ''
extra = extra[golden.columns]

extra.to_csv(ROOT / 'golden_set' / 'golden_set_topup_to_review.csv', index=False)
print(f"\nSaved {len(extra)} rows to golden_set_topup_to_review.csv")
print(extra['intent_llm_guess'].value_counts())