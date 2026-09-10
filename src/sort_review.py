import pandas as pd
from pathlib import Path

p = Path('golden_set/golden_set_to_review.csv')
df = pd.read_csv(p)
df = df.sort_values(['is_edge_case', 'intent_llm_guess', 'id']).reset_index(drop=True)
df.to_csv(p, index=False)
print(f"Sorted {len(df)} rows. Edge cases at the bottom.")
print(df['is_edge_case'].sum(), "edge cases")