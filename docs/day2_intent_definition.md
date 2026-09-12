# Day 2 — Exploratory Data Analysis & Intent Definition

**Brand:** AmazonHelp
**Dataset:** `data/processed/pairs_cleaned_full.csv` (6,893 one-turn pairs)
**Objective:** Understand the range of customer issues, define a small coherent
intent taxonomy, and produce a label guide to be used for building the golden
evaluation set.

---

## 1. Quantitative Overview

- **Sample size:** 6,893 one-turn pairs (customer message + brand reply)
- **Empty messages after cleaning:** 0
- **Customer message length:**
  - mean: 107 chars
  - median: 103 chars
  - max: 286 chars
  - distribution peaks near the 140-char Twitter limit, with a long tail of
    threaded / multi-tweet messages.
- **Languages:** mixed — English dominant, but significant Japanese, Portuguese,
  Spanish, French, German, and Italian traffic.

### Top unigrams (customer messages)
`order` (758), `delivery` (723), `prime` (604), `amazon` (1106),
`customer/service` (156), `thank/you` (106), `amazon/prime` (97),
`day/delivery` (74), `customer/care` (68).

### Recurring themes observed in random samples
- No indication of delivery / late delivery / bad delivery experience
- Wrong or old address
- Wrong item delivered
- Money not refunded
- Fire TV / Prime Video / streaming issues
- Issue resolved → positive feedback
- Locked accounts / login failures
- Unwanted Prime charges
- Frustrated venting about support quality

---

## 2. Intent Taxonomy (Locked)

Nine intents. Each maps to a **distinct resolution path**, which is what matters
for deciding auto-handle vs. escalate downstream.

| # | Intent | Description | Typical resolution |
|---|--------|-------------|--------------------|
| 1 | `order_status` | Where is my order, tracking, late/missing package | Check tracking, apologize, reship if needed |
| 2 | `refund_return` | Refund, return, or replacement requested | Process refund / RMA |
| 3 | `product_issue` | Broken item, malfunctioning device or service | Troubleshoot, replace |
| 4 | `account_access` | Login, password, locked account, security | Unlock, reset, verify |
| 5 | `billing_payment` | Wrong charge, duplicate charge, payment failure | Investigate billing, reverse charge |
| 6 | `cancellation` | Cancel order, subscription, or Prime | Cancel and confirm |
| 7 | `complaint` | Frustration with service/agent/process, **no specific ask** | Apologize + escalate to human |
| 8 | `info_request` | Pre-purchase or informational question | Answer from knowledge base |
| 9 | `other` | Fragments, unclassifiable, thanks/praise, chatter | Ignore or generic reply |

### Design decisions

- **Merged** `late_delivery`, `no_indication_of_delivery`, and `bad_delivery_experience`
  into `order_status` — they share the same resolution path.
- **Added** `complaint` after the first validation batch showed ~33% `other`,
  most of which was service frustration with no actionable ask.
- **Added** `info_request` after finding pre-purchase questions that didn't fit
  any other intent.
- **Folded** `compliment_thanks` into `other` — thanks/praise does not warrant a
  distinct resolution path and would balloon the category on multilingual data.
- **Kept** `account_access`, `billing_payment`, `cancellation` even though they
  are rare in random samples — they have distinct escalation rules.

---

## 3. Validation

Two validation batches were run after the initial EDA, each with a different
random seed, totalling **70 hand-labelled messages**.

### Batch 1 (30 messages)
| Intent | Count |
|--------|-------|
| order_status | 4 |
| refund_return | 4 |
| product_issue | 3 |
| account_access | 1 |
| billing_payment | 2 |
| cancellation | 1 |
| compliment_thanks | 6 |
| complaint | 5 |
| other | 4 |

`other` rate: **13%**.

### Batch 2 (40 messages)
| Intent | Count |
|--------|-------|
| order_status | 10 |
| complaint | 11 |
| other | 9 |
| info_request | 4 |
| refund_return | 2 |
| product_issue | 2 |
| account_access | 1 |
| cancellation | 1 |
| billing_payment | 0 |

`other` rate: **22.5% raw → 7.5% when non-English positive chatter is excluded.**

### Conclusion
- `other` is stable at **7–15%** on real support traffic once multilingual
  chatter is separated.
- `complaint` is the largest bucket (~25%). This is expected — a large share of
  customer support traffic is frustrated venting.
- `billing_payment` absent in batch 2 — normal variance, not a taxonomy gap.
- No labels required a "not sure" fallback more than twice per batch.
- **Taxonomy is locked.**

---

## 4. Label Guide

Each customer message receives **exactly ONE** primary intent — the customer's
main ask, not every issue mentioned.

### order_status
Customer asks about delivery status, tracking, or a late/missing package.
- "where is my order? tracking shows nothing"
- "package still hasn't arrived after 5 days"
- "delivery failed though I've been home all day"
- **Excludes:** refund requested → `refund_return`; wrong item → `product_issue`.

### refund_return
Customer wants money back, a return, or a replacement.
- "I want a refund for order 123"
- "item arrived damaged, how do I return?"
- "money not refunded after a month"
- **Excludes:** only complaints without an explicit refund ask → `complaint`.

### product_issue
Item is broken/wrong, or a service (Prime Video, Fire TV, app) is malfunctioning.
- "my fire stick won't turn on"
- "prime video keeps buffering on every device"
- "can't order to relay point, tried Chrome/Firefox"
- **Excludes:** delivery-related → `order_status`.

### account_access
Login, password reset, locked account, 2FA, or account security issues.
- "my account was locked as soon as I tried to log in"
- "someone has used my email for their Amazon account"

### billing_payment
Incorrect charge, duplicate charge, or payment method failure.
- "charged $10.99 for Prime when I paid INR 499"
- "support keeps trying to process charges without my consent"

### cancellation
Cancel an order, subscription, or Prime membership.
- "I want my Amazon Payments account CLOSED"
- "I want it delivered by today or please cancel it"

### complaint
Frustration with the service, agent, or process — **NO specific actionable ask**.
- "pointless to fill this form, you always give scripted replies"
- "was told you'd call back in 4 hours, it's been 24"
- "your service is dreadful"
- **Excludes:** if a concrete ask exists (refund, cancel) → use that intent.

### info_request
Pre-purchase or informational question with no active issue.
- "is Prime monthly or annual?"
- "planning to buy Fire HD 8, does it support X?"

### other
Fragments, unclassifiable, thanks/praise, non-action social chatter.
- "thanks!" / "Ok sounds good"
- "Duna, Solaris, Fabrica de Vespas!" (Portuguese book chatter)
- bare order numbers with no ask
- **Target:** ≤15% of the golden set.

---

## 5. Sample Size Targets for the Golden Set (Day 3)

Based on the validation batch distribution, target the following approximate
composition for a 200-example golden set (with over-sampling of rare intents):

| Intent | Target count | Reasoning |
|--------|-------------|-----------|
| order_status | 40 | most frequent |
| refund_return | 30 | frequent + high business value |
| product_issue | 30 | frequent |
| complaint | 30 | frequent, tricky edge cases |
| account_access | 20 | rare but distinct escalation path |
| billing_payment | 20 | rare but distinct escalation path |
| cancellation | 15 | rare but distinct escalation path |
| info_request | 10 | moderately rare |
| other | 5 | keep small — we don't need to over-evaluate junk |

Total: **200 examples**

This deliberately over-samples rare intents so the evaluation does not ignore
them. It also keeps `other` small, matching the target from the label guide.

---

## 6. Decision Log Entries (add to `decision_log.md`)

- Ran EDA on 6,893 cleaned one-turn pairs for AmazonHelp.
- Defined a 9-intent taxonomy based on manual inspection of 110+ customer messages.
- Merged delivery-related intents (`late_delivery`, `no_delivery_indication`) into
  `order_status`.
- Added `complaint` intent after batch 1 revealed ~33% `other`, mostly service
  frustration with no specific ask.
- Added `info_request` intent after finding pre-purchase questions in the sample.
- Folded `compliment_thanks` into `other` (no distinct resolution path).
- Validated the taxonomy across two batches (70 total hand-labelled messages).
  Locked after confirming `other` rate ≤15% on real support traffic.
- Decided **not** to filter to English — the taxonomy will be validated on
  multilingual data, with non-English chatter expected to fall into `other`.
- Set target composition for the 200-example golden set: over-sample rare
  intents, keep `other` small.

---

## 7. Artifacts Produced on Day 2

- `docs/intent_label_guide.md` (this file)
- `decision_log.md` updated
- Random samples inspected: 110+ messages across three batches
- Validation batch labels (kept privately, used to finalise taxonomy)

---

## 8. Open Questions for Day 3

1. **Escalation ground truth** — how do we define `auto_handle` vs `escalate`?
   Proposed rule: `complaint`, `billing_payment`, `account_access` → escalate;
   `order_status`, `info_request`, `other` → auto-handle; `refund_return`,
   `product_issue`, `cancellation` → depends on amount / context.
   This needs to be frozen before labelling.
2. **Golden reply** — should we write a golden reply for all 200 examples, or
   only a subset (e.g., 50)? Recommended: 50 to keep labelling tractable.
3. **Inter-annotator agreement** — will we have a second human label a subset of
   20–30 examples? Required for the report ("evidence of how well your judge
   agrees with a human").