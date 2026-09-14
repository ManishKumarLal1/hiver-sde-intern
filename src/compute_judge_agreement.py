"""
Compute agreement between the rule-based judge and human scores.
Handles headers with spaces and trailing commas.
"""
import sys
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / 'results' / 'judge_human_sample.csv'


def main():
    if not SAMPLE.exists():
        print(f"File not found: {SAMPLE}")
        sys.exit(1)

    # skipinitialspace=True strips leading spaces after commas in headers AND data
    df = pd.read_csv(SAMPLE, skipinitialspace=True, keep_default_na=False, na_values=[''])

    # Strip any residual whitespace from column names
    df.columns = [c.strip() for c in df.columns]

    print("Columns detected:", df.columns.tolist())

    if 'human_overall' not in df.columns:
        print("\n'human_overall' column still missing after stripping.")
        print("Run this to add it:")
        print("  python src/fix_judge_sample.py")
        sys.exit(1)

    # Coerce to numeric, dropping bad values
    df['overall'] = pd.to_numeric(df['overall'], errors='coerce')
    df['human_overall'] = pd.to_numeric(df['human_overall'], errors='coerce')

    valid = df.dropna(subset=['overall', 'human_overall']).copy()

    if len(valid) == 0:
        print("No rows have both judge and human scores.")
        sys.exit(1)

    print(f"\nRows compared: {len(valid)} of {len(df)}\n")

    valid['diff'] = (valid['overall'] - valid['human_overall']).abs()

    print(f"{'id':>5} {'intent':<16} {'judge':>6} {'human':>6} {'diff':>5}")
    print("-" * 42)
    for _, r in valid.sort_values('id').iterrows():
        print(f"{int(r['id']):>5} {str(r['intent']):<16} "
              f"{r['overall']:>6.1f} {r['human_overall']:>6.1f} {r['diff']:>5.1f}")

    rho, pval = spearmanr(valid['overall'], valid['human_overall'])
    mae = valid['diff'].mean()
    exact = (valid['overall'] == valid['human_overall']).mean()
    within1 = (valid['diff'] <= 1).mean()

    print("\n" + "=" * 42)
    print("Judge-Human Agreement")
    print("=" * 42)
    print(f"Judge mean:            {valid['overall'].mean():.2f}")
    print(f"Human mean:            {valid['human_overall'].mean():.2f}")
    print(f"Spearman rho:          {rho:.3f}  (p={pval:.3f})")
    print(f"Mean absolute diff:    {mae:.2f}")
    print(f"Exact agreement:       {exact:.1%}")
    print(f"Within-1 agreement:    {within1:.1%}")

    if rho >= 0.8:
        level = "almost perfect"
    elif rho >= 0.6:
        level = "substantial"
    elif rho >= 0.4:
        level = "moderate"
    elif rho >= 0.2:
        level = "weak"
    else:
        level = "negligible"
    print(f"\nInterpretation: rho={rho:.3f} -> {level} rank agreement")
    if valid['overall'].mean() > valid['human_overall'].mean() + 0.3:
        print("Judge is GENEROUS vs human ratings")
    elif valid['overall'].mean() < valid['human_overall'].mean() - 0.3:
        print("Judge is HARSH vs human ratings")
    else:
        print("Judge and human are close on average")


if __name__ == '__main__':
    main()