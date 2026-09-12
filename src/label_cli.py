"""
Interactive labelling CLI for the golden set.
Run from project root: python src/label_cli.py

Controls:
  1-9     pick intent
  a / e   auto_handle / escalate
  1-10    pick reason tag
  1-3     difficulty (easy/medium/hard)
  Enter   skip golden reply or note
  Ctrl+C  save and quit (resume anytime)
"""
import os
import sys
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / 'golden_set' / 'golden_set_to_review.csv'

INTENTS = [
    'order_status', 'refund_return', 'product_issue', 'account_access',
    'billing_payment', 'cancellation', 'complaint', 'info_request', 'other',
]
INTENT_KEYS = {str(i + 1): name for i, name in enumerate(INTENTS)}

REASONS = [
    'low_risk', 'deterministic_action', 'security_sensitive', 'financial_risk',
    'high_emotion', 'high_value', 'repeat_failure', 'legal_threat',
    'safety_issue', 'kb_unavailable',
]
REASON_KEYS = {str(i + 1): name for i, name in enumerate(REASONS)}

DIFFICULTIES = ['easy', 'medium', 'hard']
DIFF_KEYS = {str(i + 1): name for i, name in enumerate(DIFFICULTIES)}

REVIEW_COLS = ['intent', 'escalation_ground_truth', 'escalation_reason',
               'difficulty', 'golden_reply', 'notes']


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def header(idx, total, row):
    llm_hint = row.get('intent_llm_guess_v2') or row.get('intent_llm_guess') or ''
    print("=" * 78)
    print(f"Row {idx + 1} of {total}  |  id={row['id']}  |  "
          f"edge={bool(row.get('is_edge_case', False))}  |  LLM: {llm_hint}")
    print("-" * 78)
    print(f"CUSTOMER: {row['customer_msg_clean']}")
    print(f"BRAND:    {str(row['brand_reply_clean'])[:350]}")
    print("-" * 78)
    why = row.get('llm_reason', '')
    if isinstance(why, str) and why:
        print(f"LLM why:  {why[:180]}")
        print("-" * 78)


def prompt_menu(title, mapping):
    print(f"\n{title}")
    for k, v in mapping.items():
        print(f"  {k}) {v}")
    while True:
        ans = input("> ").strip()
        if ans in mapping:
            return mapping[ans]
        print(f"  invalid — enter one of {list(mapping.keys())}")


def prompt_escalation():
    print("\nEscalation:  a=auto_handle  e=escalate")
    while True:
        ans = input("> ").strip().lower()
        if ans == 'a':
            return 'auto_handle'
        if ans == 'e':
            return 'escalate'
        print("  invalid — enter 'a' or 'e'")


def label_one(row, df, idx):
    clear()
    header(idx, len(df), row)

    # Collect answers WITHOUT writing to df yet
    intent = prompt_menu("Intent:", INTENT_KEYS)
    esc = prompt_escalation()
    reason = prompt_menu("Reason:", REASON_KEYS)
    diff = prompt_menu("Difficulty:", DIFF_KEYS)

    print("\nGolden reply (Enter to skip):")
    reply = input("> ").strip()

    print("\nNote (Enter to skip):")
    note = input("> ").strip()

    # Commit all at once
    df.at[idx, 'intent'] = intent
    df.at[idx, 'escalation_ground_truth'] = esc
    df.at[idx, 'escalation_reason'] = reason
    df.at[idx, 'difficulty'] = diff
    df.at[idx, 'golden_reply'] = reply
    df.at[idx, 'notes'] = note

    df.to_csv(CSV, index=False)
    print("\n✓ saved.")
def label_one(row, df, idx):
    clear()
    header(idx, len(df), row)

    try:
        intent = prompt_menu("Intent:", INTENT_KEYS)
        esc = prompt_escalation()
        reason = prompt_menu("Reason:", REASON_KEYS)
        diff = prompt_menu("Difficulty:", DIFF_KEYS)

        print("\nGolden reply (Enter to skip):")
        reply = input("> ").strip()

        print("\nNote (Enter to skip):")
        note = input("> ").strip()
    except (KeyboardInterrupt, EOFError):
        # row not written → picked up next run
        raise

    df.at[idx, 'intent'] = intent
    df.at[idx, 'escalation_ground_truth'] = esc
    df.at[idx, 'escalation_reason'] = reason
    df.at[idx, 'difficulty'] = diff
    df.at[idx, 'golden_reply'] = reply
    df.at[idx, 'notes'] = note

    df.to_csv(CSV, index=False)
    print("\n✓ saved.")

def main():
    if not CSV.exists():
        print(f"File not found: {CSV}")
        sys.exit(1)

    df = pd.read_csv(CSV, keep_default_na=False, na_values=[''])
    for c in REVIEW_COLS:
        df[c] = df[c].fillna('').astype('object')

    remaining = df.index[df['intent'] == ''].tolist()
    if not remaining:
        print(f"All rows labelled. {(df['intent'] != '').sum()} / {len(df)}")
        return

    print(f"Remaining: {len(remaining)} of {len(df)}")
    print("Press Enter to begin (or Ctrl+C to quit).")
    try:
        input()
    except (KeyboardInterrupt, EOFError):
        print("\nNothing done.")
        return

    for idx in remaining:
        try:
            label_one(df.loc[idx], df, idx)
        except (KeyboardInterrupt, EOFError):
            print("\n\nInterrupted — saving progress...")
            try:
                df.to_csv(CSV, index=False)
                done = (df['intent'] != '').sum()
                print(f"Saved. Progress: {done}/{len(df)}")
            except KeyboardInterrupt:
                # user hit Ctrl+C again during save — write once more, ignore
                print("\nSaving again (please wait)...")
                df.to_csv(CSV, index=False)
                done = (df['intent'] != '').sum()
                print(f"Saved. Progress: {done}/{len(df)}")
            return
        except Exception as e:
            try:
                df.to_csv(CSV, index=False)
            except KeyboardInterrupt:
                df.to_csv(CSV, index=False)
            print(f"\nUnexpected error: {e}")
            print("Saved progress.")
            return

    done = (df['intent'] != '').sum()
    print(f"\nAll done. Labelled {done}/{len(df)} rows.")


if __name__ == '__main__':
    main()