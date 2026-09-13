"""
Lightweight labelling CLI for the second-pass agreement check.

Only asks for intent + escalation_ground_truth (no reason/difficulty/reply).

Run from project root: python src/label_second_pass_cli.py

Controls:
  1-9     pick intent
  a / e   auto_handle / escalate
  Ctrl+C  save and quit (resume anytime)
"""
import os
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / 'golden_set' / 'golden_set_second_pass.csv'

INTENTS = [
    'order_status', 'refund_return', 'product_issue', 'account_access',
    'billing_payment', 'cancellation', 'complaint', 'info_request', 'other',
]
INTENT_KEYS = {str(i + 1): name for i, name in enumerate(INTENTS)}


def clear():
    os.system('cls' if os.name == 'nt' else 'clear')


def main():
    if not CSV.exists():
        print(f"File not found: {CSV}")
        print("Run `python src/make_second_pass.py` first.")
        return

    df = pd.read_csv(CSV, keep_default_na=False, na_values=[''])
    df['intent'] = df['intent'].fillna('').astype('object')
    df['escalation_ground_truth'] = df['escalation_ground_truth'].fillna('').astype('object')

    remaining = df.index[df['intent'] == ''].tolist()
    if not remaining:
        done = (df['intent'] != '').sum()
        print(f"All rows labelled. {done}/{len(df)}")
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
            clear()
            row = df.loc[idx]

            print("=" * 78)
            print(f"Row {idx + 1} of {len(df)}  |  id={row['id']}")
            print("-" * 78)
            print(f"CUSTOMER: {row['customer_msg_clean']}")
            print(f"BRAND:    {str(row['brand_reply_clean'])[:350]}")
            print("-" * 78)

            # Intent
            print("\nIntent:")
            for k, v in INTENT_KEYS.items():
                print(f"  {k}) {v}")

            while True:
                ans = input("> ").strip()
                if ans in INTENT_KEYS:
                    intent = INTENT_KEYS[ans]
                    break
                print("  invalid — enter 1-9")

            # Escalation
            print("\nEscalation:  a=auto_handle  e=escalate")
            while True:
                ans = input("> ").strip().lower()
                if ans == 'a':
                    esc = 'auto_handle'
                    break
                if ans == 'e':
                    esc = 'escalate'
                    break
                print("  invalid — enter 'a' or 'e'")

            df.at[idx, 'intent'] = intent
            df.at[idx, 'escalation_ground_truth'] = esc
            df.to_csv(CSV, index=False)
            print("\n✓ saved.")

        except (KeyboardInterrupt, EOFError):
            print("\n\nInterrupted — saving...")
            try:
                df.to_csv(CSV, index=False)
            except KeyboardInterrupt:
                df.to_csv(CSV, index=False)
            done = (df['intent'] != '').sum()
            print(f"Saved. Progress: {done}/{len(df)}")
            return

    done = (df['intent'] != '').sum()
    print(f"\nAll done. Labelled {done}/{len(df)} rows.")


if __name__ == '__main__':
    main()