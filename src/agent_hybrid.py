"""
Hybrid agent:
  - Intent: TF-IDF + LogisticRegression (trained on golden set via out-of-fold)
  - Escalation: rule-based on intent + keyword triggers
  - Reason: rule-based
  - Reply: retrieved top-1 historical brand reply from the 5k pool

No LLM calls. No quotas. No local model. Runnable in seconds.
"""
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / 'golden_set' / 'golden_set_to_review.csv'
POOL = ROOT / 'data' / 'processed' / 'pairs_cleaned_full.csv'
OUT = ROOT / 'predictions' / 'agent.csv'

ALWAYS_ESC = {'account_access', 'billing_payment', 'complaint'}


def escalation_for(intent, msg):
    m = str(msg).lower()
    if intent in ALWAYS_ESC:
        return 'escalate'
    if re.search(r'\b(delivered but|not received|hacked|fraud|unauthori|legal|lawsuit|safety|burn|fire)\b', m):
        return 'escalate'
    if re.search(r'\b(again|still|third time|repeat)\b', m) and intent in {'order_status', 'refund_return'}:
        return 'escalate'
    return 'auto_handle'


def reason_for(intent, msg):
    m = str(msg).lower()
    if re.search(r'\b(hacked|unauthori|locked|password|security)\b', m): return 'security_sensitive'
    if re.search(r'\b(legal|lawsuit|court|chargeback)\b', m): return 'legal_threat'
    if re.search(r'\b(safety|burn|fire|injur)\b', m): return 'safety_issue'
    if re.search(r'\b(again|still|third time|repeat)\b', m): return 'repeat_failure'
    if re.search(r'\b(charged|charge|billing|payment|refund|money)\b', m): return 'financial_risk'
    if intent == 'complaint': return 'high_emotion'
    if intent == 'account_access': return 'security_sensitive'
    if intent == 'billing_payment': return 'financial_risk'
    if intent in ('info_request', 'other', 'cancellation'): return 'low_risk'
    return 'deterministic_action'


def main():
    golden = pd.read_csv(GOLDEN, keep_default_na=False, na_values=[''])
    golden = golden[golden['intent'] != ''].copy()

    # Intent classifier — OOF predictions
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(golden['customer_msg_clean'])
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    pred_intents = cross_val_predict(clf, X, golden['intent'], cv=5)

    # Escalation & reason — rules
    pred_esc = [escalation_for(i, m) for i, m in zip(pred_intents, golden['customer_msg_clean'])]
    pred_reason = [reason_for(i, m) for i, m in zip(pred_intents, golden['customer_msg_clean'])]

    # Retrieval replies
    pool = pd.read_csv(POOL).dropna(subset=['customer_msg_clean', 'brand_reply_clean'])
    pool = pool[pool['brand_reply_clean'].astype(str).str.strip() != ''].reset_index(drop=True)
    pool_vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    pool_X = pool_vec.fit_transform(pool['customer_msg_clean'].astype(str))

    def retrieve(msg):
        if not isinstance(msg, str) or not msg.strip():
            return ''
        q = pool_vec.transform([msg])
        sims = (pool_X @ q.T).toarray().ravel()
        idx = int(np.argmax(sims))
        return str(pool.iloc[idx]['brand_reply_clean'])

    pred_reply = [retrieve(m) for m in golden['customer_msg_clean']]

    out = pd.DataFrame({
        'id': golden['id'].values,
        'predicted_intent': pred_intents,
        'predicted_escalation': pred_esc,
        'predicted_reason': pred_reason,
        'predicted_reply': pred_reply,
    })
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Saved {len(out)} predictions to {OUT}")
    print("\nIntent distribution:")
    print(out['predicted_intent'].value_counts())


if __name__ == '__main__':
    main()