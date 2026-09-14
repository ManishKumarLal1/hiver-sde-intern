"""Print the 20 sampled replies for manual scoring."""
import pandas as pd
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
preds = pd.read_csv(ROOT / 'predictions' / 'agent.csv', keep_default_na=False, na_values=[''])

ids = [96, 16, 31, 160, 187, 116, 70, 173, 162, 46, 67, 151, 167, 79, 196, 201, 57, 154, 83, 69]

sub = preds[preds['id'].isin(ids)]
for _, r in sub.iterrows():
    print(f"--- id={r['id']} ({r['predicted_intent']}) ---")
    reply = str(r['predicted_reply'])[:500]
    print(textwrap.fill(reply, 100))
    print()