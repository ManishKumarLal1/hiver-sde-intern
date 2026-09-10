"""
Pre-label candidates using an LLM with the intent taxonomy.
Uses Groq (OpenAI-compatible) as the provider.

The LLM output is a FIRST PASS and MUST be human-reviewed.
"""
import os
import json
import time
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Groq model — free tier, fast, OpenAI-compatible
MODEL = "openai/gpt-oss-120b"

# Free-tier limits (Groq): ~30 requests/min. Sleep to stay safely under.
SLEEP_BETWEEN_CALLS = 2.2   # seconds
MAX_RETRIES = 5

INTENTS = [
    "order_status", "refund_return", "product_issue", "account_access",
    "billing_payment", "cancellation", "complaint", "info_request", "other"
]

SYSTEM_PROMPT = f"""You are labelling AmazonHelp customer support tweets.

Assign exactly ONE intent from this list:
{INTENTS}

Rules:
- order_status: delivery status, tracking, late/missing package
- refund_return: refund, return, replacement requested
- product_issue: broken item, malfunctioning device/service
- account_access: login, password, locked account, security
- billing_payment: wrong charge, duplicate charge, payment failure
- cancellation: cancel order, subscription, Prime
- complaint: frustration with service/agent, NO specific ask
- info_request: pre-purchase or informational question
- other: fragments, thanks/praise, unclassifiable

Return ONLY valid JSON in this exact shape:
{{"intent": "<intent>", "confidence": <0-1>}}"""


def call_llm(client, message, model=MODEL):
    """Call LLM with retry on rate limits."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": message}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            out = json.loads(resp.choices[0].message.content)
            intent = out.get("intent", "other")
            if intent not in INTENTS:
                intent = "other"
            return intent, out.get("confidence", 0.0)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower():
                wait = SLEEP_BETWEEN_CALLS * (attempt + 2)
                print(f"  rate-limited, waiting {wait:.1f}s ...")
                time.sleep(wait)
            else:
                print(f"  error: {msg}")
                time.sleep(SLEEP_BETWEEN_CALLS)
    return "other", 0.0


def prelabel(df, client, model=MODEL):
    intents, confidences = [], []
    total = len(df)
    for i, row in df.iterrows():
        msg = str(row["customer_msg_clean"])
        intent, conf = call_llm(client, msg, model)
        intents.append(intent)
        confidences.append(conf)
        if (i + 1) % 25 == 0 or (i + 1) == total:
            print(f"  labelled {i+1}/{total}")
        time.sleep(SLEEP_BETWEEN_CALLS)
    return intents, confidences


if __name__ == "__main__":
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise SystemExit("GROQ_API_KEY not set in environment/.env")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1"
    )

    df = pd.read_csv("golden_set/candidates_500.csv")
    print(f"Pre-labelling {len(df)} candidates with {MODEL} ...")

    df["llm_intent"], df["llm_confidence"] = prelabel(df, client)
    df.to_csv("golden_set/candidates_500_prelabelled.csv", index=False)

    print("\nDone. Distribution:")
    print(df["llm_intent"].value_counts())
    print(f"\nLow-confidence (<0.6): {(df['llm_confidence'] < 0.6).sum()}")