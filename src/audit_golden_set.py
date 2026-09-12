"""
Automated QA audit for the golden set.
Flags rows that violate consistency rules or common mislabeling patterns.
Run from project root: python src/audit_golden_set.py
"""
import pandas as pd
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / 'golden_set' / 'golden_set_to_review.csv'
OUT = ROOT / 'golden_set' / 'audit_needs_review.csv'

df = pd.read_csv(GOLDEN, keep_default_na=False, na_values=[''])
df = df[df['intent'] != ''].copy()   # only labelled rows
print(f"Auditing {len(df)} labelled rows\n")

flags = []

def flag(idx, rule, msg):
    flags.append({'id': df.at[idx, 'id'], 'intent': df.at[idx, 'intent'],
                  'escalation': df.at[idx, 'escalation_ground_truth'],
                  'reason': df.at[idx, 'escalation_reason'],
                  'difficulty': df.at[idx, 'difficulty'],
                  'message': str(df.at[idx, 'customer_msg_clean'])[:150],
                  'rule': rule, 'why': msg})

# ---------- Rule 1: escalation × reason consistency ----------
AUTO_REASONS = {'low_risk', 'deterministic_action'}
ESC_REASONS = {'security_sensitive', 'financial_risk', 'high_emotion',
               'high_value', 'repeat_failure', 'legal_threat',
               'safety_issue', 'kb_unavailable'}

for idx, row in df.iterrows():
    esc, reason = row['escalation_ground_truth'], row['escalation_reason']
    if esc == 'auto_handle' and reason in ESC_REASONS:
        flag(idx, 'R1-escalation-mismatch',
             f'auto_handle with escalate reason "{reason}"')
    if esc == 'escalate' and reason in AUTO_REASONS:
        flag(idx, 'R1-escalation-mismatch',
             f'escalate with auto reason "{reason}"')

# ---------- Rule 2: complaint needs no topic keywords ----------
TOPIC_WORDS = r'\b(refund|return|cancel|charged?|billing|payment|order|tracking|deliver|delivery|shipment|package|password|login|sign in|account)\b'
for idx, row in df[df['intent'] == 'complaint'].iterrows():
    m = str(row['customer_msg_clean'])
    if re.search(TOPIC_WORDS, m, re.IGNORECASE):
        flag(idx, 'R2-complaint-topic',
             'complaint but message mentions a topic word — could be a different intent')

# ---------- Rule 3: info_request with active problem words ----------
for idx, row in df[df['intent'] == 'info_request'].iterrows():
    m = str(row['customer_msg_clean'])
    if re.search(r'\b(not working|broken|locked|hacked|charged|late|missing|stuck|failed|error)\b',
                 m, re.IGNORECASE):
        flag(idx, 'R3-info-but-problem',
             'info_request but message describes an active problem')

# ---------- Rule 4: product_issue with money/dispute signals ----------
for idx, row in df[df['intent'] == 'product_issue'].iterrows():
    m = str(row['customer_msg_clean'])
    if re.search(r'\b(refund|money back|charge|billing|credit card)\b', m, re.IGNORECASE):
        flag(idx, 'R4-product-but-money',
             'product_issue but message mentions money — could be refund/billing')

# ---------- Rule 5: order_status with tracking-dispute signals ----------
for idx, row in df[df['intent'] == 'order_status'].iterrows():
    m = str(row['customer_msg_clean'])
    if re.search(r'\b(said delivered|claimed|but i didn.?t|isn.?t right|lying|false)\b',
                 m, re.IGNORECASE):
        flag(idx, 'R5-order-status-dispute',
             'order_status but seems to dispute tracking — check complaint')

# ---------- Rule 6: short fragments labelled as specific intents ----------
for idx, row in df.iterrows():
    m = str(row['customer_msg_clean'])
    if len(m) < 20 and row['intent'] not in ('other', 'complaint'):
        flag(idx, 'R6-short-fragment',
             f'{row["intent"]} but message is only {len(m)} chars — fragment risk')

# ---------- Rule 7: account_access should always escalate ----------
for idx, row in df[df['intent'] == 'account_access'].iterrows():
    if row['escalation_ground_truth'] != 'escalate':
        flag(idx, 'R7-account-access-auto',
             'account_access should always be escalate')

# ---------- Rule 8: complaint should always escalate ----------
for idx, row in df[df['intent'] == 'complaint'].iterrows():
    if row['escalation_ground_truth'] != 'escalate':
        flag(idx, 'R8-complaint-auto',
             'complaint should always be escalate')

# ---------- Rule 9: billing_payment should always escalate ----------
for idx, row in df[df['intent'] == 'billing_payment'].iterrows():
    if row['escalation_ground_truth'] != 'escalate':
        flag(idx, 'R9-billing-auto',
             'billing_payment should always be escalate')

# ---------- Rule 10: difficulty vs LLM disagreement ----------
for idx, row in df.iterrows():
    if row.get('difficulty') != 'easy':
        continue
    if row['intent'] == '':
        continue

    # Safe hint extraction that skips NaN/float values
    hint_v2 = row.get('intent_llm_guess_v2', '')
    hint_v1 = row.get('intent_llm_guess', '')
    hint = ''
    if isinstance(hint_v2, str) and hint_v2.strip():
        hint = hint_v2
    elif isinstance(hint_v1, str) and hint_v1.strip():
        hint = hint_v1

    if hint and hint != row['intent']:
        flag(idx, 'R10-easy-but-override',
             f'marked easy but overrode LLM hint ("{hint}" -> "{row["intent"]}")')

# ---------- Rule 11: non-English labelled as complaint ----------
def looks_non_english(s):
    return bool(re.search(r'[^\x00-\x7F]', s)) and not re.search(
        r'^[\x00-\x7F]+$', s)

for idx, row in df[df['intent'] == 'complaint'].iterrows():
    m = str(row['customer_msg_clean'])
    if looks_non_english(m):
        flag(idx, 'R11-non-english-complaint',
             'non-English labelled complaint — verify not other/info_request')

# ---------- Rule 12: reasons marked kb_unavailable that shouldn't be ----------
for idx, row in df[df['escalation_reason'] == 'kb_unavailable'].iterrows():
    m = str(row['customer_msg_clean'])
    if not re.search(r'\b(unusual|strange|weird|specific|particular|complex|how does|why does)\b',
                     m, re.IGNORECASE) and len(m) < 60:
        flag(idx, 'R12-suspicious-kb',
             'kb_unavailable on a short/simple message — verify')

# ---------- Write output ----------
audit = pd.DataFrame(flags).drop_duplicates(subset=['id', 'rule'])
audit.to_csv(OUT, index=False)

print(f"Flagged {audit['id'].nunique()} unique rows across {len(audit)} flags\n")
print("Flag counts by rule:")
print(audit['rule'].value_counts())
print(f"\nSaved to {OUT}")
print("\nTop 20 rows to review (sorted by id):")
cols = ['id', 'intent', 'escalation', 'reason', 'difficulty', 'rule']
print(audit.sort_values('id')[cols].head(20).to_string(index=False))