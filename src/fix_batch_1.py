import pandas as pd
from pathlib import Path

p = Path(__file__).resolve().parent.parent / 'golden_set' / 'golden_set_to_review.csv'
df = pd.read_csv(p, keep_default_na=False, na_values=[''])

COLS = ['intent','escalation_ground_truth','escalation_reason','difficulty','notes']
for c in COLS:
    df[c] = df[c].fillna('').astype('object')

# (id, intent, escalation, reason, difficulty, note)
FIXES = [
    # Escalation mismatches
    (31,  'order_status',   'escalate',     'high_emotion',       'medium', 'Angry delivery case'),
    (35,  'refund_return',  'escalate',     'financial_risk',     'medium', 'Refund dispute'),
    (37,  'cancellation',   'escalate',     'high_emotion',       'medium', 'Angry cancel'),
    (43,  'refund_return',  'escalate',     'high_emotion',       'medium', 'Angry refund case'),
    (54,  'complaint',      'escalate',     'high_emotion',       'medium', 'Complaint; reason corrected'),
    (69,  'info_request',   'auto_handle',  'low_risk',           'medium', 'KB covers this usage question'),
    (103, 'complaint',      'escalate',     'high_emotion',       'hard',   'Impatient nudge; not other'),
    (177, 'order_status',   'escalate',     'kb_unavailable',     'medium', 'Missed pickup; escalate'),
    (186, 'product_issue',  'auto_handle',  'deterministic_action','medium','Stuck error; troubleshoot; not safety'),
    (190, 'account_access', 'escalate',     'security_sensitive', 'medium', 'Access blocker; not cancellation'),
    (199, 'account_access', 'escalate',     'security_sensitive', 'easy',   'Access ask; reason corrected'),
]

for rid, intent, esc, reason, diff, note in FIXES:
    df.loc[df['id'] == rid, COLS] = [intent, esc, reason, diff, note]

df.to_csv(p, index=False)
print(f"Applied {len(FIXES)} fixes")