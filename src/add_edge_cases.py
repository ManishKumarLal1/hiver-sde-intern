"""
Hand-pick edge cases from the 5k pool and append to the review workbook.
Categories:
  - Multi-intent (multiple asks in one message)
  - Sarcastic / angry / all-caps
  - Very short fragments
  - Non-English messages
  - Very long messages
  - Messages referencing images/screenshots
"""
import pandas as pd
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL = ROOT / 'data' / 'processed' / 'pairs_cleaned_full.csv'
OUT  = ROOT / 'golden_set' / 'golden_set_to_review.csv'

df = pd.read_csv(POOL)
used = set(pd.read_csv(ROOT / 'golden_set' / 'golden_set_to_review.csv')['customer_msg_clean'])
pool = df[~df['customer_msg_clean'].isin(used)].copy()

def find(pattern, min_len=0, max_len=10_000, n=8, seed=42, flags=re.IGNORECASE):
    """Return up to n rows matching the regex, respecting length bounds."""
    sub = pool[pool['customer_msg_clean'].str.contains(pattern, na=False, regex=True, flags=flags)]
    sub = sub[(sub['customer_msg_clean'].str.len() >= min_len) &
              (sub['customer_msg_clean'].str.len() <= max_len)]
    return sub.sample(min(n, len(sub)), random_state=seed)

rows = []

# 1. Multi-intent: heuristic — has "and" or "but" plus a second ask
multi = pool[
    pool['customer_msg_clean'].str.contains(r'\band\b|\bbut\b', case=False, na=False, regex=True) &
    pool['customer_msg_clean'].str.contains(
        r'\b(refund|return|cancel|deliver|order|account|charge|help)\b',
        case=False, na=False, regex=True
    ) &
    (pool['customer_msg_clean'].str.len() > 80)
].sample(min(12, len(pool)), random_state=1)
rows.append(('multi_intent', multi))

# 2. Sarcasm / anger / all-caps
anger = pool[pool['customer_msg_clean'].str.contains(
    r'\bunacceptable\b|\bdisgusting\b|\bfurious\b|\bridiculous\b|'
    r'\bnever again\b|\bworst\b|\bterrible\b|\bAPPALLING\b|!!+',
    case=False, na=False, regex=True)]
rows.append(('anger_sarcasm', anger.sample(min(10, len(anger)), random_state=2)))

# 3. Very short fragments
short = pool[pool['customer_msg_clean'].str.len().between(2, 10)]
rows.append(('short_fragment', short.sample(min(8, len(short)), random_state=3)))

# 4. Very long messages
long_msgs = pool[pool['customer_msg_clean'].str.len() > 240]
rows.append(('very_long', long_msgs.sample(min(6, len(long_msgs)), random_state=4)))

# 5. Image / screenshot references
images = pool[pool['customer_msg_clean'].str.contains(
    r'\bscreenshot\b|\battached\b|\bphoto\b|\bpicture\b|\bsee below\b|\bDM me\b',
    case=False, na=False, regex=True)]
rows.append(('image_reference', images.sample(min(6, len(images)), random_state=5)))

# 6. Non-English — crude: contains non-ASCII letters
noneng = pool[pool['customer_msg_clean'].str.contains(
    r'[^\x00-\x7F]', na=False, regex=True)]
# Drop pure emoji / punctuation
noneng = noneng[noneng['customer_msg_clean'].str.contains(r'[A-Za-z\u00C0-\u024F\u3040-\u30FF\u4E00-\u9FFF]', na=False, regex=True)]
rows.append(('non_english', noneng.sample(min(12, len(noneng)), random_state=6)))

# Combine, tag, dedupe
parts = []
for tag, sub in rows:
    sub = sub.copy()
    sub['edge_tag'] = tag
    parts.append(sub)

edge = pd.concat(parts).drop_duplicates(subset='customer_msg_clean')
print("Edge case counts:")
print(edge['edge_tag'].value_counts())
print(f"Total unique edge cases: {len(edge)}")

# Build in the same shape as the review workbook
edge = edge[['customer_msg_clean', 'brand_reply_clean', 'edge_tag']]
edge = edge.rename(columns={'edge_tag': 'intent_llm_guess'})   # placeholder guess
edge['intent'] = ""
edge['escalation_ground_truth'] = ""
edge['escalation_reason'] = ""
edge['golden_reply'] = ""
edge['difficulty'] = "hard"
edge['is_edge_case'] = True
edge['notes'] = edge['intent_llm_guess']   # temporary: tag will be overwritten
edge['intent_llm_guess'] = ""

# Combine with existing review workbook
existing = pd.read_csv(ROOT / 'golden_set' / 'golden_set_to_review.csv')
existing['edge_note'] = ""
combined = pd.concat([existing, edge], ignore_index=True)
combined['id'] = range(1, len(combined) + 1)

combined.to_csv(ROOT / 'golden_set' / 'golden_set_to_review.csv', index=False)
print(f"\nTotal rows now: {len(combined)}")
print(f"Edge cases: {combined['is_edge_case'].sum()}")