"""
Post-process agent predictions to enforce escalation/reason consistency.
Run after the agent finishes:
    python src/fix_agent_reasons.py
"""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / 'predictions' / 'agent.csv'

AUTO_REASONS = {'low_risk', 'deterministic_action'}
ESC_REASONS = {'security_sensitive', 'financial_risk', 'high_emotion',
               'high_value', 'repeat_failure', 'legal_threat',
               'safety_issue', 'kb_unavailable'}

# If auto_handle was chosen but reason is an escalate reason, what should the
# reason default to? Depends on intent.
AUTO_FALLBACK = {
    'order_status':    'deterministic_action',
    'refund_return':   'deterministic_action',
    'product_issue':   'deterministic_action',
    'cancellation':    'deterministic_action',
    'info_request':    'low_risk',
    'other':           'low_risk',
}

# If escalate was chosen but reason is an auto reason, what should the reason
# default to? Depends on intent.
ESC_FALLBACK = {
    'account_access':  'security_sensitive',
    'billing_payment': 'financial_risk',
    'complaint':       'high_emotion',
    'order_status':    'kb_unavailable',
    'refund_return':   'financial_risk',
    'product_issue':   'repeat_failure',
    'cancellation':    'high_emotion',
}

df = pd.read_csv(P, keep_default_na=False, na_values=[''])

fixed_auto = 0
fixed_esc = 0

for i, row in df.iterrows():
    esc = row['predicted_escalation']
    reason = row['predicted_reason']
    intent = row['predicted_intent']

    if esc == 'auto_handle' and reason in ESC_REASONS:
        df.at[i, 'predicted_reason'] = AUTO_FALLBACK.get(intent, 'low_risk')
        fixed_auto += 1

    elif esc == 'escalate' and reason in AUTO_REASONS:
        df.at[i, 'predicted_reason'] = ESC_FALLBACK.get(intent, 'high_emotion')
        fixed_esc += 1

df.to_csv(P, index=False)
print(f"Fixed {fixed_auto} auto/reason mismatches")
print(f"Fixed {fixed_esc} escalate/reason mismatches")
print(f"Total rows: {len(df)}")
print("\nEscalation distribution:")
print(df['predicted_escalation'].value_counts())
print("\nReason distribution:")
print(df['predicted_reason'].value_counts())