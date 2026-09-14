"""
LLM-as-judge for reply quality.
Uses local Ollama (no API keys, no rate limits).

Run: python src/judge.py predictions/agent.csv
"""
import os
import re
import json
import time
import sys
import pandas as pd
from pathlib import Path
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / 'golden_set' / 'golden_set_to_review.csv'

MODEL = "llama3.2:3b"       # <-- local Ollama model (was gpt-oss-120b)
SLEEP = 0.5                 # <-- Ollama is local, no rate limit
MAX_RETRIES = 3

JUDGE_PROMPT = """You are an expert evaluator for customer support replies.

Customer message: {customer}
Historical brand reply (grounding reference): {reference}
Agent reply (to evaluate): {reply}
Detected intent: {intent}

Score the agent reply on each criterion from 1 (poor) to 5 (excellent):

1. relevance — Does it address the customer's actual issue?
2. groundedness — Is it consistent with what the brand actually said?
3. tone — Is it professional and appropriate for AmazonHelp?
4. actionability — Does it move the customer toward resolution?
5. safety — Does it avoid making things up (fake order numbers, fake policies)?

Return only JSON:
{{
  "relevance": <1-5>,
  "groundedness": <1-5>,
  "tone": <1-5>,
  "actionability": <1-5>,
  "safety": <1-5>,
  "overall": <1-5>,
  "note": "<one sentence>"
}}"""


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    return text


def llm_json(client, prompt):
    for attempt in range(MAX_RETRIES):
        try:
            r = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
                stream=False,
            )
            return json.loads(extract_json(r.choices[0].message.content))
        except Exception as e:
            print(f"  err: {str(e)[:150]}")
            time.sleep(2 * (attempt + 1))
    return {}


def main(pred_path):
    client = OpenAI(
        api_key="ollama",                         # dummy value
        base_url="http://localhost:11434/v1"      # local Ollama server
    )

    golden = pd.read_csv(GOLDEN, keep_default_na=False, na_values=[''])
    golden = golden[golden['intent'] != ''][['id', 'customer_msg_clean', 'brand_reply_clean']]
    preds = pd.read_csv(pred_path, keep_default_na=False, na_values=[''])

    merged = golden.merge(preds, on='id', how='inner')
    print(f"Judging {len(merged)} replies")

    out = []
    for i, row in merged.iterrows():
        judge = llm_json(client, JUDGE_PROMPT.format(
            customer=row['customer_msg_clean'],
            reference=str(row['brand_reply_clean'])[:300],
            reply=str(row['predicted_reply'])[:500],
            intent=row['predicted_intent'],
        ))
        out.append({
            'id': row['id'],
            'intent': row['predicted_intent'],
            **{k: judge.get(k, 0) for k in
               ['relevance', 'groundedness', 'tone', 'actionability', 'safety', 'overall']},
            'note': judge.get('note', ''),
        })
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(merged)}")
        time.sleep(SLEEP)

    df = pd.DataFrame(out)
    name = Path(pred_path).stem
    out_path = ROOT / 'results' / f'judge_{name}.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved {out_path}")

    print("\nMean scores:")
    for c in ['relevance', 'groundedness', 'tone', 'actionability', 'safety', 'overall']:
        print(f"  {c:<15} {df[c].mean():.2f}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python src/judge.py predictions/agent.csv")
        sys.exit(1)
    main(sys.argv[1])