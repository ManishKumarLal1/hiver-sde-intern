"""
Combine pre-labelled candidates + rare-intent extras into a stratified
review workbook. The `intent` column starts empty — the LLM guess is kept
in `intent_llm_guess` for reference during human review.
"""
import pandas as pd

TARGETS = {
    "order_status": 30, "refund_return": 22, "product_issue": 22,
    "complaint": 25, "account_access": 15, "billing_payment": 15,
    "cancellation": 10, "info_request": 8, "other": 8,
}

main = pd.read_csv('golden_set/candidates_500_prelabelled.csv')
main = main.rename(columns={'llm_intent': 'intent'})[
    ['customer_msg_clean', 'brand_reply_clean', 'intent']
]

extra = pd.read_csv('golden_set/candidates_rare_extra.csv')
extra = extra.rename(columns={'keyword_intent': 'intent'})[
    ['customer_msg_clean', 'brand_reply_clean', 'intent']
]

pool = pd.concat([main, extra]).drop_duplicates(subset='customer_msg_clean').reset_index(drop=True)

parts = []
for intent, n in TARGETS.items():
    subset = pool[pool['intent'] == intent]
    if len(subset) >= n:
        parts.append(subset.sample(n, random_state=42))
    else:
        parts.append(subset)
        print(f"⚠️  only {len(subset)} for {intent} (wanted {n})")

out = pd.concat(parts).sample(frac=1, random_state=42).reset_index(drop=True)
out['id'] = range(1, len(out) + 1)
out = out.rename(columns={'intent': 'intent_llm_guess'})

# Empty columns for human review
out['intent'] = ""
out['escalation_ground_truth'] = ""
out['escalation_reason'] = ""
out['golden_reply'] = ""
out['difficulty'] = ""
out['is_edge_case'] = False
out['notes'] = ""

out = out[['id', 'customer_msg_clean', 'brand_reply_clean',
           'intent_llm_guess', 'intent',
           'escalation_ground_truth', 'escalation_reason',
           'golden_reply', 'difficulty', 'is_edge_case', 'notes']]

out.to_csv('golden_set/golden_set_to_review.csv', index=False)
print(f"\nSaved {len(out)} rows to golden_set/golden_set_to_review.csv")
print(out['intent_llm_guess'].value_counts())