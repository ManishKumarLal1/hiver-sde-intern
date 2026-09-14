"""
Rule-based reply quality judge.
Scores each reply on 5 criteria using deterministic rules.

Fast, reproducible, no LLM required.
Run: python src/judge_rules.py predictions/agent.csv
"""
import re
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / 'golden_set' / 'golden_set_to_review.csv'


def score_relevance(reply, customer, intent):
    """Does the reply address the customer's topic?"""
    if not reply or len(reply) < 15:
        return 1
    r = reply.lower()
    c = customer.lower()
    # Topic keywords per intent
    topics = {
        'order_status':   ['order', 'deliver', 'track', 'ship', 'package', 'arriv'],
        'refund_return':  ['refund', 'return', 'money', 'reimburse'],
        'product_issue':  ['issue', 'problem', 'device', 'item', 'broken', 'work'],
        'account_access': ['account', 'access', 'log', 'password', 'unlock'],
        'billing_payment':['charge', 'billing', 'payment', 'amount'],
        'cancellation':   ['cancel', 'subscription'],
        'complaint':      ['sorry', 'apolog', 'understand', 'frustrat'],
        'info_request':   ['help', 'information', 'question', 'detail'],
        'other':          ['thank', 'appreciate', 'glad'],
    }
    hits = sum(1 for kw in topics.get(intent, []) if kw in r)
    if hits >= 2: return 5
    if hits == 1: return 4
    # Partial credit if any topic words appear
    if any(kw in r for kw in ['order', 'account', 'deliver', 'refund', 'charge']):
        return 3
    return 2


def score_groundedness(reply, reference):
    """Does the reply share vocabulary with the historical brand reply?"""
    if not reply or not reference:
        return 3
    r_words = set(re.findall(r'\w{4,}', reply.lower()))
    ref_words = set(re.findall(r'\w{4,}', str(reference).lower()))
    if not r_words:
        return 2
    overlap = len(r_words & ref_words) / len(r_words)
    if overlap >= 0.3: return 5
    if overlap >= 0.15: return 4
    if overlap >= 0.05: return 3
    return 2


def score_tone(reply):
    """Is the tone professional?"""
    if not reply or len(reply) < 15:
        return 1
    r = reply.lower()
    score = 3  # neutral baseline
    # Positive signals
    if any(w in r for w in ['sorry', 'apolog', 'understand', 'appreciate']):
        score += 1
    if any(w in r for w in ['please', 'thank', 'help', 'assist']):
        score += 0.5
    # Negative signals
    if any(w in r for w in ['stupid', 'idiot', 'fuck', 'shut up']):
        score -= 3
    if reply.isupper() and len(reply) > 30:
        score -= 1
    return max(1, min(5, round(score)))


def score_actionability(reply, escalation):
    """Does the reply move toward resolution?"""
    if not reply or len(reply) < 15:
        return 1
    r = reply.lower()
    # Escalation reply should invite to DM/phone
    if escalation == 'escalate':
        if any(w in r for w in ['dm', 'direct message', 'phone', 'chat', 'contact', 'reach']):
            return 5
        if any(w in r for w in ['look into', 'investigate', 'follow up', 'get back']):
            return 4
        return 2
    # Auto-handle reply should give info/steps
    if any(w in r for w in ['here', 'link', 'click', 'follow', 'step', 'visit']):
        return 5
    if any(w in r for w in ['will', 'help', 'assist', 'check']):
        return 4
    return 3


def score_safety(reply):
    """Does the reply avoid making up specifics?"""
    if not reply:
        return 1
    r = reply
    # Penalize fabricated specifics
    if re.search(r'\b[A-Z]{2,}\d{5,}\b', r):  # fake order numbers
        return 2
    if re.search(r'\$\d+|₹\d+|£\d+', r) and 'refund' not in r.lower():  # random money amounts
        return 3
    # Penalize absolute promises
    if re.search(r'\bguarantee\b|\bwill definitely\b|\b100% ', r, re.IGNORECASE):
        return 3
    return 5


def main(pred_path):
    golden = pd.read_csv(GOLDEN, keep_default_na=False, na_values=[''])
    golden = golden[golden['intent'] != ''][['id', 'customer_msg_clean', 'brand_reply_clean']]
    preds = pd.read_csv(pred_path, keep_default_na=False, na_values=[''])

    merged = golden.merge(preds, on='id', how='inner')
    print(f"Judging {len(merged)} replies (rule-based)")

    rows = []
    for _, row in merged.iterrows():
        reply = str(row.get('predicted_reply', ''))
        customer = str(row['customer_msg_clean'])
        reference = str(row['brand_reply_clean'])
        intent = row['predicted_intent']
        esc = row['predicted_escalation']

        r_rel = score_relevance(reply, customer, intent)
        r_grd = score_groundedness(reply, reference)
        r_tone = score_tone(reply)
        r_act = score_actionability(reply, esc)
        r_safe = score_safety(reply)
        overall = round((r_rel + r_grd + r_tone + r_act + r_safe) / 5, 1)

        rows.append({
            'id': row['id'],
            'intent': intent,
            'relevance': r_rel,
            'groundedness': r_grd,
            'tone': r_tone,
            'actionability': r_act,
            'safety': r_safe,
            'overall': overall,
        })

    df = pd.DataFrame(rows)
    name = Path(pred_path).stem
    out_path = ROOT / 'results' / f'judge_{name}.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    print("\nMean scores (1-5):")
    for c in ['relevance', 'groundedness', 'tone', 'actionability', 'safety', 'overall']:
        print(f"  {c:<15} {df[c].mean():.2f}")

    # Per-intent breakdown
    print("\nPer-intent overall score:")
    print(df.groupby('intent')['overall'].agg(['mean', 'count']).round(2))


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/judge_rules.py predictions/agent.csv")
        sys.exit(1)
    main(sys.argv[1])