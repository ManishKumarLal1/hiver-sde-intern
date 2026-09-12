import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
topup_p = ROOT / 'golden_set' / 'golden_set_topup_to_review.csv'
golden_p = ROOT / 'golden_set' / 'golden_set_to_review.csv'

topup = pd.read_csv(topup_p, keep_default_na=False, na_values=[''])
golden = pd.read_csv(golden_p, keep_default_na=False, na_values=[''])

next_id = int(golden['id'].max()) + 1
topup['id'] = range(next_id, next_id + len(topup))
topup.to_csv(topup_p, index=False)
print(f"Assigned ids {next_id} to {next_id + len(topup) - 1}")