"""
Evaluation harness: compare a predictions CSV against the golden set.

Usage:
    python src/eval_harness.py predictions/baseline_trivial.csv
    python src/eval_harness.py predictions/baseline_simple.csv
    python src/eval_harness.py predictions/agent.csv
"""
import sys
import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, cohen_kappa_score
)

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / 'golden_set' / 'golden_set_to_review.csv'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

INTENTS = ['order_status', 'refund_return', 'product_issue', 'account_access',
           'billing_payment', 'cancellation', 'complaint', 'info_request', 'other']


def load_golden():
    df = pd.read_csv(GOLDEN, keep_default_na=False, na_values=[''])
    df = df[df['intent'] != ''].copy()
    return df


def evaluate(pred_path):
    golden = load_golden()
    preds = pd.read_csv(pred_path, keep_default_na=False, na_values=[''])

    # Merge on id
    merged = golden.merge(preds, on='id', how='inner', suffixes=('', '_pred'))
    if len(merged) == 0:
        print("ERROR: no overlapping ids between golden and predictions")
        return

    missing = len(golden) - len(merged)
    if missing > 0:
        print(f"⚠️  {missing} golden rows have no prediction (will be ignored)")

    y_true_i = merged['intent']
    y_pred_i = merged['predicted_intent'].fillna('other')

    # Intent metrics
    acc_i = accuracy_score(y_true_i, y_pred_i)
    p_i, r_i, f1_i, _ = precision_recall_fscore_support(
        y_true_i, y_pred_i, labels=INTENTS, average='macro', zero_division=0
    )
    weighted_f1_i = precision_recall_fscore_support(
        y_true_i, y_pred_i, average='weighted', zero_division=0
    )[2]

    # Per-class
    per_class = precision_recall_fscore_support(
        y_true_i, y_pred_i, labels=INTENTS, average=None, zero_division=0
    )

    # Escalation metrics
    y_true_e = merged['escalation_ground_truth']
    y_pred_e = merged['predicted_escalation'].fillna('auto_handle')
    acc_e = accuracy_score(y_true_e, y_pred_e)
    p_e, r_e, f1_e, _ = precision_recall_fscore_support(
        y_true_e, y_pred_e, labels=['auto_handle', 'escalate'],
        average='macro', zero_division=0
    )

    # Reason (if present)
    has_reason = 'predicted_reason' in merged.columns and merged['predicted_reason'].notna().any()
    acc_r = None
    if has_reason:
        mask = merged['predicted_reason'].fillna('') != ''
        if mask.sum() > 0:
            acc_r = (merged.loc[mask, 'escalation_reason'] ==
                     merged.loc[mask, 'predicted_reason']).mean()

    # Report
    name = Path(pred_path).stem
    print(f"\n{'=' * 60}\nEvaluation: {name}\n{'=' * 60}")
    print(f"Rows evaluated: {len(merged)}")
    print(f"\nIntent:")
    print(f"  Accuracy:      {acc_i:.3f}")
    print(f"  Macro F1:      {f1_i:.3f}")
    print(f"  Weighted F1:   {weighted_f1_i:.3f}")
    print(f"\nEscalation (auto_handle vs escalate):")
    print(f"  Accuracy:      {acc_e:.3f}")
    print(f"  Macro F1:      {f1_e:.3f}")
    if acc_r is not None:
        print(f"\nEscalation reason agreement: {acc_r:.3f}")

    print(f"\nPer-intent performance:")
    print(f"  {'intent':<16} {'prec':>6} {'rec':>6} {'f1':>6} {'support':>8}")
    for i, intent in enumerate(INTENTS):
        sup = (y_true_i == intent).sum()
        if sup == 0:
            continue
        print(f"  {intent:<16} {per_class[0][i]:>6.2f} {per_class[1][i]:>6.2f} "
              f"{per_class[2][i]:>6.2f} {sup:>8}")

    # Confusion matrix
    cm = confusion_matrix(y_true_i, y_pred_i, labels=INTENTS)
    print(f"\nConfusion matrix (rows=true, cols=pred):")
    header = " " * 18 + " ".join(f"{i[:4]:>5}" for i in INTENTS)
    print(header)
    for i, intent in enumerate(INTENTS):
        row = " ".join(f"{cm[i][j]:>5}" for j in range(len(INTENTS)))
        print(f"  {intent:<16}{row}")

    # Save JSON
    out = {
        'system': name,
        'n_evaluated': len(merged),
        'intent': {'accuracy': acc_i, 'macro_f1': f1_i, 'weighted_f1': weighted_f1_i},
        'escalation': {'accuracy': acc_e, 'macro_f1': f1_e},
        'reason_agreement': acc_r,
        'per_intent': {
            intent: {
                'precision': float(per_class[0][i]),
                'recall': float(per_class[1][i]),
                'f1': float(per_class[2][i]),
                'support': int((y_true_i == intent).sum()),
            } for i, intent in enumerate(INTENTS)
        }
    }
    out_path = RESULTS / f'eval_{name}.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python src/eval_harness.py <predictions.csv>")
        sys.exit(1)
    evaluate(sys.argv[1])