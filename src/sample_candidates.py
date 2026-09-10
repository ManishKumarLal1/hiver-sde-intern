"""
Sample 500 candidates from the cleaned pairs for LLM pre-labelling.
Stratified loosely by message length and to include multilingual messages.
"""
import pandas as pd
import numpy as np

def sample_candidates(input_path, output_path, n=500, seed=42):
    df = pd.read_csv(input_path)
    df = df.dropna(subset=['customer_msg_clean'])
    df = df[df['customer_msg_clean'].str.strip() != '']

    # Stratify loosely: 60% short, 25% medium, 15% long
    df['length'] = df['customer_msg_clean'].str.len()
    short = df[df['length'] < 80]
    medium = df[(df['length'] >= 80) & (df['length'] < 160)]
    long = df[df['length'] >= 160]

    n_short = int(n * 0.60)
    n_medium = int(n * 0.25)
    n_long = n - n_short - n_medium

    sampled = pd.concat([
        short.sample(min(n_short, len(short)), random_state=seed),
        medium.sample(min(n_medium, len(medium)), random_state=seed),
        long.sample(min(n_long, len(long)), random_state=seed),
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    sampled['id'] = range(1, len(sampled) + 1)
    sampled = sampled[['id', 'customer_msg', 'customer_msg_clean',
                       'brand_reply', 'brand_reply_clean']]
    sampled.to_csv(output_path, index=False)
    print(f"Saved {len(sampled)} candidates to {output_path}")
    return sampled

if __name__ == '__main__':
    sample_candidates(
        'data/processed/pairs_cleaned_full.csv',
        'golden_set/candidates_500.csv',
        n=500
    )