"""
Mine rare intents from the full 5k pool using keyword search.
Outputs extra candidates for manual review.
"""
import pandas as pd

df = pd.read_csv('data/processed/pairs_cleaned_full.csv')

# Exclude messages already in the 500 pre-labelled pool
used = set(pd.read_csv('golden_set/candidates_500_prelabelled.csv')['customer_msg_clean'])
pool = df[~df['customer_msg_clean'].isin(used)].copy()
print(f"Pool after removing used: {len(pool)}")

PATTERNS = {
    'refund_return':  r'\b(refund|return|reimburse|money back|exchange|replacement)\b',
    'account_access': r'\b(login|log in|locked|password|sign in|unauthori[sz]ed|hack|access my account)\b',
    'cancellation':   r'\b(cancel|close.{0,10}account|unsubscribe|terminate|stop.{0,10}subscription|delete.{0,10}account)\b',
    'billing_payment':r'\b(charge[d]?|billing|payment|invoice|deduct|overcharg|double.{0,5}charg)\b',
}

rows = []
for intent, pattern in PATTERNS.items():
    hits = pool[pool['customer_msg_clean'].str.contains(pattern, case=False, na=False, regex=True)]
    hits = hits[hits['customer_msg_clean'].str.len() >= 15]
    hits = hits.sample(min(50, len(hits)), random_state=42).copy()
    hits['keyword_intent'] = intent
    rows.append(hits)
    print(f"{intent}: {len(hits)} candidates")

extra = pd.concat(rows).drop_duplicates(subset='customer_msg_clean').reset_index(drop=True)
extra.to_csv('golden_set/candidates_rare_extra.csv', index=False)
print(f"\nSaved {len(extra)} extra candidates to golden_set/candidates_rare_extra.csv")