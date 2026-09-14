"""
AI support agent using Google Gemini (generous free tier).
Pipeline: retrieve examples → single LLM call → intent + escalation + reason + reply.

Run from project root:
    python src/agent_gemini.py          # all rows
    python src/agent_gemini.py 5        # test 5 rows

Resumes from predictions/agent.csv if it exists (skips already-done ids).
"""
import os
import re
import json
import time
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
import google.generativeai as genai

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / 'golden_set' / 'golden_set_to_review.csv'
POOL = ROOT / 'data' / 'processed' / 'pairs_cleaned_full.csv'
OUT = ROOT / 'predictions' / 'agent.csv'

MODEL_NAME = "gemini-3.6-flash"
SLEEP = 1.5
MAX_RETRIES = 4

INTENTS = ['order_status', 'refund_return', 'product_issue', 'account_access',
           'billing_payment', 'cancellation', 'complaint', 'info_request', 'other']
REASONS = ['low_risk', 'deterministic_action', 'security_sensitive', 'financial_risk',
           'high_emotion', 'high_value', 'repeat_failure', 'legal_threat',
           'safety_issue', 'kb_unavailable']

PROMPT = """You are an AmazonHelp support agent. Given a customer tweet, classify intent, decide escalation, and draft a reply.

Customer message: {message}

Retrieved similar historical messages and brand replies (for grounding):
{examples}

Intent definitions (pick exactly one):
- order_status: delivery/tracking/late/missing package
- refund_return: refund/return/replacement requested
- product_issue: broken item or malfunctioning device/service
- account_access: login, password, locked, hacked (with an access ask)
- billing_payment: wrong charge, duplicate charge, payment failure
- cancellation: cancel order/sub/account
- complaint: venting, no actionable ask
- info_request: pre-purchase or general question
- other: thanks, fragments, off-topic

Escalation policy:
- account_access, billing_payment, complaint -> ALWAYS escalate
- order_status -> auto unless "delivered but not received" or high value
- refund_return -> auto unless disputed or >1000
- product_issue -> auto unless safety/repeat/high value
- cancellation -> auto unless retention/legal
- info_request, other -> auto

Reason tags (pick one):
low_risk, deterministic_action, security_sensitive, financial_risk, high_emotion,
high_value, repeat_failure, legal_threat, safety_issue, kb_unavailable

Write a 2-4 sentence professional reply in AmazonHelp's tone: brief, empathetic, action-oriented. Do not invent order numbers or policies.

Return ONLY valid JSON with these exact keys:
{{"intent": "<intent>", "escalation": "auto_handle" or "escalate", "reason": "<tag>", "reply": "<reply text>"}}
"""


def build_retrieval():
    pool = pd.read_csv(POOL)
    pool = pool.dropna(subset=['customer_msg_clean', 'brand_reply_clean'])
    pool = pool[pool['brand_reply_clean'].astype(str).str.strip() != '']
    pool = pool[pool['customer_msg_clean'].astype(str).str.strip() != ''].reset_index(drop=True)
    vec = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), min_df=2)
    X = vec.fit_transform(pool['customer_msg_clean'].astype(str))
    return pool, vec, X


def retrieve(pool, vec, X, msg, k=3):
    if not isinstance(msg, str) or not msg.strip():
        return []
    q = vec.transform([msg])
    sims = (X @ q.T).toarray().ravel()
    idx = np.argsort(-sims)[:k]
    return [{'customer': str(pool.iloc[i]['customer_msg_clean']),
             'reply': str(pool.iloc[i]['brand_reply_clean'])}
            for i in idx]


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    return text


def call_gemini(model, prompt):
    for attempt in range(MAX_RETRIES):
        try:
            resp = model.generate_content(
                prompt,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0,
                }
            )
            return json.loads(extract_json(resp.text))
        except Exception as e:
            err = str(e)
            print(f"  err: {err[:140]}")
            low = err.lower()
            if "429" in err or "quota" in low or "rate" in low:
                time.sleep(10 * (attempt + 1))
            else:
                time.sleep(2)
    raise RuntimeError("Gemini call failed after all retries")


def run_agent(limit=None):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY not set in .env")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)

    golden = pd.read_csv(GOLDEN, keep_default_na=False, na_values=[''])
    golden = golden[golden['intent'] != ''].copy()
    if limit:
        golden = golden.head(limit)

    # Resume from previous predictions if any
    done_ids = set()
    existing_rows = []
    if OUT.exists():
        try:
            prev = pd.read_csv(OUT, keep_default_na=False, na_values=[''])
            prev = prev[prev['predicted_intent'].astype(str).str.strip() != '']
            done_ids = set(prev['id'].tolist())
            existing_rows = prev.to_dict('records')
            print(f"Resuming — {len(done_ids)} rows already done")
        except Exception as e:
            print(f"Could not load previous predictions ({e}); starting fresh")

    remaining = golden[~golden['id'].isin(done_ids)].copy()
    print(f"Running on {len(remaining)} remaining rows (of {len(golden)})")
    if len(remaining) == 0:
        print("Nothing to do. All rows already predicted.")
        return

    pool, vec, X = build_retrieval()
    rows = list(existing_rows)

    for i, (_, row) in enumerate(remaining.iterrows(), 1):
        msg = str(row['customer_msg_clean'])
        examples = retrieve(pool, vec, X, msg, k=3)
        examples_str = "\n".join(
            f"- Customer: {e['customer'][:120]}\n  Brand reply: {e['reply'][:180]}"
            for e in examples
        ) or "(no close examples)"

        try:
            out = call_gemini(model, PROMPT.format(message=msg, examples=examples_str))
        except RuntimeError as e:
            print(f"  ⚠️  SKIPPING id={row['id']}: {e}")
            time.sleep(5)
            continue
        except KeyboardInterrupt:
            print(f"\nInterrupted. Saving {len(rows)} rows.")
            pd.DataFrame(rows).to_csv(OUT, index=False)
            return

        intent = out.get('intent', 'other')
        if intent not in INTENTS:
            intent = 'other'
        esc = out.get('escalation', 'auto_handle')
        if esc not in ('auto_handle', 'escalate'):
            esc = 'auto_handle'
        reason = out.get('reason', 'low_risk')
        if reason not in REASONS:
            reason = 'low_risk'
        reply = out.get('reply', '')

        rows.append({
            'id': row['id'],
            'predicted_intent': intent,
            'predicted_escalation': esc,
            'predicted_reason': reason,
            'predicted_reply': reply,
        })
        print(f"  [{i}/{len(remaining)}] id={row['id']} → {intent} / {esc} / {reason}")
        pd.DataFrame(rows).to_csv(OUT, index=False)

        time.sleep(SLEEP)

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\nDone. {len(rows)} predictions saved to {OUT}")


if __name__ == '__main__':
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_agent(limit=limit)