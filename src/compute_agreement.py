import pandas as pd
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parent.parent
main = pd.read_csv(ROOT / 'golden_set' / 'golden_set_to_review.csv',
                   keep_default_na=False, na_values=['']).set_index('id')
second = pd.read_csv(ROOT / 'golden_set' / 'golden_set_second_pass.csv',
                     keep_default_na=False, na_values=['']).set_index('id')

common = [i for i in main.index.intersection(second.index)
          if second.loc[i, 'intent'] != '']

if len(common) == 0:
    print("No second-pass labels found yet. Fill in the CSV first.")
    raise SystemExit(0)

a_i = main.loc[common, 'intent']
b_i = second.loc[common, 'intent']
a_e = main.loc[common, 'escalation_ground_truth']
b_e = second.loc[common, 'escalation_ground_truth']

k_i = cohen_kappa_score(a_i, b_i)
k_e = cohen_kappa_score(a_e, b_e)

print(f"Rows compared: {len(common)}")
print(f"Intent agreement:     {(a_i == b_i).mean():.1%}  κ = {k_i:.3f}")
print(f"Escalation agreement: {(a_e == b_e).mean():.1%}  κ = {k_e:.3f}")

dis = a_i != b_i
if dis.sum() > 0:
    print(f"\n{dis.sum()} intent disagreements:")
    for i in a_i.index[dis]:
        print(f"  id={i}  pass1={a_i[i]}  pass2={b_i[i]}")
        print(f"      {str(main.loc[i,'customer_msg_clean'])[:100]}")