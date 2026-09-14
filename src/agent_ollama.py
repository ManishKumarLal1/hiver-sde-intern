"""
AI support agent for AmazonHelp.

Pipeline:
  1. Classify intent using LLM (few-shot, 9 intents)
  2. Retrieve top-3 similar historical messages + brand replies
  3. Decide escalation + reason using LLM + retrieval context
  4. Generate grounded reply using LLM + retrieved examples

Writes predictions to predictions/agent.csv in the standard schema.
"""
import os
import re
import json
import time
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / 'golden_set' / 'golden_set_to_review.csv'
POOL = ROOT / 'data' / 'processed' / 'pairs_cleaned_full.csv'
OUT = ROOT / 'predictions' / 'agent.csv'

MODEL = "llama3.2:3b"
SLEEP = 2.0
MAX_RETRIES = 5

INTENTS = ['order_status', 'refund_return', 'product_issue', 'account_access',
           'billing_payment', 'cancellation', 'complaint', 'info_request', 'other']
REASONS = ['low_risk', 'deterministic_action', 'security_sensitive', 'financial_risk',
           'high_emotion', 'high_value', 'repeat_failure', 'legal_threat',
           'safety_issue', 'kb_unavailable']

INTENT_PROMPT = """You classify AmazonHelp customer support tweets.

Pick EXACTLY ONE intent from:
{intents}

Priority order — stop at the first match:
1. Fragment/thanks/off-topic, no ask → other
2. Venting without an actionable ask → complaint
3. Pre-purchase or general question, no active issue → info_request
4. Cancel order/sub/account → cancellation
5. Wants money back / return / replacement → refund_return
6. Wrong charge / duplicate charge / payment failure / unauthorised TRANSACTION → billing_payment
7. Item broken / wrong / device/service malfunctioning → product_issue
8. Cannot log in / locked / hacked / password reset trouble (with an access ask) → account_access
9. Delivery status / tracking / late / missing package → order_status

CRITICAL:
- Do NOT use account_access just because the message says "login", "account", "password"
- Do NOT use order_status just because the message mentions delivery — check if there's a delivery ask
- Sarcasm is a complaint signal
- If a specific product is mentioned with a delivery issue → order_status
- If a message has multiple issues, pick the highest priority (see the order above)

Return only JSON: {{"intent": "<intent>", "confidence": <0-1>, "reasoning": "<short>"}}"""

ESCALATION_PROMPT = """You are deciding whether an AmazonHelp support tweet should be auto-handled or escalated to a human.

Intent: {intent}
Customer message: {message}
Retrieved similar examples (from historical brand responses):
{examples}

Escalation policy:
- account_access → ALWAYS escalate (security-sensitive)
- billing_payment → ALWAYS escalate (financial risk)
- complaint → ALWAYS escalate (high emotion)
- order_status → auto_handle unless "delivered but not received" or high value
- refund_return → auto_handle unless >1000 or repeat failure
- product_issue → auto_handle unless safety/repeat/high value
- cancellation → auto_handle unless retention/legal
- info_request → auto_handle
- other → auto_handle

Reason tags (pick one):
low_risk, deterministic_action, security_sensitive, financial_risk, high_emotion,
high_value, repeat_failure, legal_threat, safety_issue, kb_unavailable

Return only JSON: {{"escalation": "auto_handle"|"escalate", "reason": "<tag>"}}"""

REPLY_PROMPT = """You are a support agent for AmazonHelp, drafting a reply to a customer tweet.

Customer message: {message}
Detected intent: {intent}
Escalation: {escalation} ({reason})

Here are similar historical customer messages and the brand's replies, to ground your tone and content:
{examples}

Write a professional, empathetic reply (2-4 sentences, Twitter-appropriate).
- Do not make up order numbers or promises.
- If escalation is needed, invite the customer to a private channel (DM/phone).
- Match AmazonHelp's tone: brief, apologetic when warranted, action-oriented.
- Don't repeat the customer's insult back. Don't over-apologize.

Return only JSON: {{"reply": "<reply text>"}}"""

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
    return [{
        'customer': str(pool.iloc[i]['customer_msg_clean']),
        'reply': str(pool.iloc[i]['brand_reply_clean']),
        'score': float(sims[i]),
    } for i in idx]


def extract_json(text):
    """Extract JSON from model output, handling fenced blocks and double-encoding."""
    text = text.strip()
    # Strip ```json fences
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    return text


def parse_json_response(raw):
    """Try hard to get a dict from the raw model output."""
    text = extract_json(raw)
    # Attempt 1: direct parse
    try:
        out = json.loads(text)
        # Handle double-encoding: if we got a string, parse again
        if isinstance(out, str):
            out = json.loads(extract_json(out))
        if isinstance(out, dict):
            return out
    except Exception:
        pass
    # Attempt 2: find first {...} block via regex
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            out = json.loads(m.group(0))
            if isinstance(out, str):
                out = json.loads(extract_json(out))
            if isinstance(out, dict):
                return out
        except Exception:
            pass
    return {}


def llm_json(client, prompt, model=MODEL):
    for attempt in range(MAX_RETRIES):
        try:
            r = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            raw = r.choices[0].message.content
            out = parse_json_response(raw)
            if out:
                return out
            # Parse failed — print raw for debugging
            print(f"  parse failed, raw: {raw[:200]}")
        except Exception as e:
            print(f"  err: {str(e)[:150]}")
        time.sleep(1)
    raise RuntimeError("LLM call failed after all retries")
COMBINED_PROMPT = """Classify this AmazonHelp customer tweet.

Tweet: "{message}"

Pick ONE intent:
order_status | refund_return | product_issue | account_access | billing_payment | cancellation | complaint | info_request | other

Pick escalation: auto_handle | escalate
Pick reason: low_risk | security_sensitive | financial_risk | high_emotion | repeat_failure | kb_unavailable | deterministic_action | high_value | legal_threat | safety_issue

Write a short reply (2 sentences max).

Return JSON only:
{"intent": "...", "escalation": "...", "reason": "...", "reply": "..."}"""


def run_agent(limit=None):
    client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")

    golden = pd.read_csv(GOLDEN, keep_default_na=False, na_values=[''])
    golden = golden[golden['intent'] != ''].copy()
    if limit:
        golden = golden.head(limit)

    # Resume
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
            print(f"Could not load previous ({e}); starting fresh")

    remaining = golden[~golden['id'].isin(done_ids)].copy()
    print(f"Running agent on {len(remaining)} remaining rows (of {len(golden)})")

    rows = list(existing_rows)
    for i, (_, row) in enumerate(remaining.iterrows(), 1):
        msg = str(row['customer_msg_clean'])
        try:
            # Minimal prompt — no retrieval (small model can't handle it)
            out = llm_json(client, COMBINED_PROMPT.format(message=msg))
            if not isinstance(out, dict):
                print(f"  raw type: {type(out)}, value: {repr(out)[:200]}")
                out = out if isinstance(out, dict) else {}

            intent = out.get('intent', 'other')
            if intent not in INTENTS:
                intent = 'other'
            escalation = out.get('escalation', 'auto_handle')
            if escalation not in ('auto_handle', 'escalate'):
                escalation = 'auto_handle'
            reason = out.get('reason', 'low_risk')
            if reason not in REASONS:
                reason = 'low_risk'
            reply = out.get('reply', '')

            rows.append({
                'id': row['id'],
                'predicted_intent': intent,
                'predicted_escalation': escalation,
                'predicted_reason': reason,
                'predicted_reply': reply,
            })
            print(f"  [{i}/{len(remaining)}] id={row['id']} → {intent} / {escalation} / {reason}")
            pd.DataFrame(rows).to_csv(OUT, index=False)

            time.sleep(0.5)

        except KeyboardInterrupt:
            print(f"\nInterrupted. Saved {len(rows)} rows.")
            pd.DataFrame(rows).to_csv(OUT, index=False)
            return
        except Exception as e:
            print(f"  ⚠️  id={row['id']}: {e}")
            time.sleep(2)
            continue

    out = pd.DataFrame(rows)
    OUT.parent.mkdir(exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"\nDone. Saved {len(out)} predictions to {OUT}")

if __name__ == '__main__':
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_agent(limit=limit)