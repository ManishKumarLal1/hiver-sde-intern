"""
Merge the labelled top-up rows into the main golden set.
Safe to run multiple times — deduplicates by customer message.

Run from project root: python src/merge_topup.py
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / 'golden_set' / 'golden_set_to_review.csv'
TOPUP = ROOT / 'golden_set' / 'golden_set_topup_to_review.csv'
BACKUP = ROOT / 'golden_set' / 'golden_set_to_review.bak.csv'

# Load both
main = pd.read_csv(MAIN, keep_default_na=False, na_values=[''])
topup = pd.read_csv(TOPUP, keep_default_na=False, na_values=[''])

# Force review columns to string (avoid the float64 dtype bug)
COLS = ['intent', 'escalation_ground_truth', 'escalation_reason',
        'difficulty', 'golden_reply', 'notes']
for c in COLS:
    main[c] = main[c].fillna('').astype('object')
    topup[c] = topup[c].fillna('').astype('object')

print(f"Main:  {len(main)} rows, labelled: {(main['intent'] != '').sum()}")
print(f"Topup: {len(topup)} rows, labelled: {(topup['intent'] != '').sum()}")

# Only keep labelled topup rows
topup_labelled = topup[topup['intent'] != ''].copy()
if len(topup_labelled) == 0:
    print("No labelled top-up rows found. Nothing to merge.")
    raise SystemExit(0)

# Back up the current main file
main.to_csv(BACKUP, index=False)
print(f"\nBackup saved to {BACKUP.name}")

# Align columns: keep topup columns that exist in main, add missing ones
for c in main.columns:
    if c not in topup_labelled.columns:
        topup_labelled[c] = ''

# Keep only columns that exist in main (drop extras like intent_llm_guess_v2
# that might not be in main)
topup_labelled = topup_labelled[[c for c in main.columns if c in topup_labelled.columns]]
for c in main.columns:
    if c not in topup_labelled.columns:
        topup_labelled[c] = ''

topup_labelled = topup_labelled[main.columns]

# Merge, dedupe by customer message
merged = pd.concat([main, topup_labelled], ignore_index=True)
before = len(merged)
merged = merged.drop_duplicates(subset='customer_msg_clean', keep='first').reset_index(drop=True)
after = len(merged)
print(f"\nMerged: {before} rows before dedup, {after} after ({before - after} duplicates removed)")

# Reassign sequential ids
merged['id'] = range(1, len(merged) + 1)

# Save
merged.to_csv(MAIN, index=False)

print(f"\nFinal total: {len(merged)} rows")
print(f"Labelled:    {(merged['intent'] != '').sum()}")
print("\nIntent distribution:")
print(merged['intent'].value_counts())
print("\nEscalation distribution:")
print(merged['escalation_ground_truth'].value_counts())