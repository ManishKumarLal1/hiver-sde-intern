"""
Re-label the golden set candidates with a stronger prompt that avoids
the account_access over-triggering problem.
Run from project root: python src/relabel_golden.py
"""
import os, json, time, re
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "openai/gpt-oss-120b"
SLEEP = 2.2
MAX_RETRIES = 5

INTENTS = [
    "order_status", "refund_return", "product_issue", "account_access",
    "billing_payment", "cancellation", "complaint", "info_request", "other"
]

SYSTEM_PROMPT = """You label AmazonHelp customer support tweets.

Pick EXACTLY ONE intent from: {intents}

Follow this decision order — stop at the first match:

1. If the message is a fragment, thanks, joke, off-topic, or contains no
   actionable request → "other"

2. If the customer is venting/frustrated WITHOUT asking for a specific action
   (no request to fix, refund, cancel, unlock, etc.) → "complaint"

3. If the customer asks a pre-purchase or general question with NO active
   problem (e.g. "is Prime monthly?") → "info_request"

4. If the customer wants to cancel an order, subscription, or account → "cancellation"

5. If the customer wants money back, a return, or a replacement → "refund_return"

6. If the customer reports a wrong charge, duplicate charge, payment failure,
   or unauthorised TRANSACTION (money-related) → "billing_payment"

7. If the customer reports a product/service is broken, wrong, or malfunctioning
   (device, app, Prime Video, Fire TV, Kindle) → "product_issue"

8. If the customer reports CANNOT LOG IN, is LOCKED OUT, has a hacked account,
   has FORGOTTEN their password, or their account credentials are compromised
   AND they are explicitly asking for help accessing the account → "account_access"

9. If the message is about delivery status, tracking, late package, or a
   delivered-but-missing package → "order_status"

CRITICAL: Do NOT use "account_access" just because the message mentions
"login", "account", "password", or "sign in".

Examples:
- "I cannot sign in, reset password 3 times, help!" → account_access (explicit access ask)
- "Your new login UI sucks" → complaint (venting, no ask)
- "Unauthorised transaction on my account" → billing_payment (money issue)
- "Thanks for trying to help" → other (no ask)
- "I can't DM you, have to login, so annoying" → complaint (venting)

Return ONLY valid JSON: {{"intent": "<intent>", "reasoning": "<short>"}}""".format(intents=INTENTS)


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    return text


def call(client, msg):
    for attempt in range(MAX_RETRIES):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": msg}
                ],
                response_format={"type": "json_object"},
                temperature=0
            )
            out = json.loads(extract_json(r.choices[0].message.content))
            intent = out.get("intent", "other")
            if intent not in INTENTS:
                intent = "other"
            return intent, out.get("reasoning", "")
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(SLEEP * (attempt + 2))
            else:
                print(f"  err: {e}")
                time.sleep(SLEEP)
    return "other", ""


def main():
    client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
    p = Path("golden_set/golden_set_to_review.csv")
    df = pd.read_csv(p)
    print(f"Re-labelling {len(df)} rows ...")

    new_intents, reasons = [], []
    for i, row in df.iterrows():
        intent, why = call(client, str(row["customer_msg_clean"]))
        new_intents.append(intent)
        reasons.append(why)
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(df)}")
        time.sleep(SLEEP)

    df["intent_llm_guess_v2"] = new_intents
    df["llm_reason"] = reasons
    df.to_csv(p, index=False)
    print("\nNew distribution:")
    print(df["intent_llm_guess_v2"].value_counts())


if __name__ == "__main__":
    main()