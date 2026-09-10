# Intent Taxonomy — AmazonHelp

**Version:** 1.0 (locked)
**Brand:** AmazonHelp
**Built from:** 110+ hand-labelled messages sampled from 6,893 cleaned one-turn pairs
**Status:** Locked — will not be revised during golden-set construction unless a
systematic classification failure emerges.

---

## Overview

Nine intents. Each maps to a **distinct resolution path**, which is what matters
for deciding auto-handle vs. escalate downstream. The taxonomy is intentionally
small enough to train a classifier on a few hundred labels, but broad enough to
cover >85% of real support traffic.

| # | Intent | Frequency (random sample) | Default resolution |
|---|--------|--------------------------|--------------------|
| 1 | `order_status` | ~25% | Auto-handle if tracking available; else escalate |
| 2 | `complaint` | ~25% | Escalate to human |
| 3 | `refund_return` | ~10% | Escalate if amount > threshold; else auto |
| 4 | `product_issue` | ~8% | Auto-troubleshoot; escalate if unresolved |
| 5 | `billing_payment` | ~5% | Escalate to human |
| 6 | `account_access` | ~3% | Escalate to human (security-sensitive) |
| 7 | `cancellation` | ~3% | Auto-confirm |
| 8 | `info_request` | ~8% | Auto-answer from knowledge base |
| 9 | `other` | ~13% raw / ~7% on support traffic | Ignore or generic reply |

*Frequency estimates are approximate and will be re-measured against the golden
set. Rare intents are deliberately over-sampled in the golden set.*

---

## Intent Definitions

### 1. `order_status`
**Definition:** Customer asks about delivery status, tracking, or a late, missing,
or misdelivered package.

**Examples:**
- "where is my order? tracking shows nothing"
- "package still hasn't arrived after 5 days"
- "delivery failed though I've been home all day"
- "supposed to be delivered today, now rescheduled due to customer request"

**Excludes:**
- Refund explicitly requested → `refund_return`
- Wrong item delivered → `product_issue`

---

### 2. `refund_return`
**Definition:** Customer wants money back, a return, a replacement, or reports
a refund that hasn't been processed.

**Examples:**
- "I want a refund for order 123"
- "item arrived damaged, how do I return?"
- "money not refunded after a month"
- "WHERE IS MY MONEY? GIVE ME MY MONEY"

**Excludes:**
- General complaint without an explicit ask → `complaint`
- Refund mentioned but primary issue is delivery delay → `order_status`

---

### 3. `product_issue`
**Definition:** Item is broken/wrong, or a service (Prime Video, Fire TV, Kindle,
app) is malfunctioning.

**Examples:**
- "my fire stick won't turn on"
- "prime video keeps buffering on every device"
- "just received the projector and the volume does not work"
- "can't order to relay point, tried Chrome/Firefox"

**Excludes:**
- Delivery-related complaint → `order_status`
- Wrong charge for a product → `billing_payment`

---

### 4. `account_access`
**Definition:** Login failures, password resets, locked accounts, 2FA issues,
or account security concerns (unauthorized access, identity used).

**Examples:**
- "my account was locked as soon as I tried to log in on desktop"
- "someone has used my email address for their Amazon account"
- "haven't received the unlock email"

**Excludes:**
- Billing issues on the account → `billing_payment`
- Cancel-account requests → `cancellation`

---

### 5. `billing_payment`
**Definition:** Incorrect charge, duplicate charge, unwanted subscription charge,
or payment method failure.

**Examples:**
- "charged $10.99 for Prime when I paid INR 499"
- "support keeps trying to process charges without my consent"
- "Prime been charged with my brother's card"

**Excludes:**
- Customer wants the subscription cancelled → `cancellation`
- Refund requested for a physical product → `refund_return`

---

### 6. `cancellation`
**Definition:** Customer wants to cancel an order, subscription, membership, or
account.

**Examples:**
- "I want my Amazon Payments account CLOSED"
- "I want it delivered by today or please cancel it"
- "please cancel my Prime subscription"

**Excludes:**
- Customer wants a refund on an already-delivered item → `refund_return`
- Simply stops using the service (no explicit cancel) → `complaint` or `other`

---

### 7. `complaint`
**Definition:** Customer expresses frustration with the service, agent, or
process itself. **No specific actionable ask** — the customer is venting about
how they've been treated, not requesting a concrete action.

**Examples:**
- "pointless to fill this form, you always give scripted replies"
- "was told you'd call back in 4 hours, it's been 24"
- "your service is dreadful"
- "I contacted the couriers, they said you have to sort it... they hang up on me"
- "how have I got prime but can never get my item on time!!"

**Excludes:**
- If a concrete ask exists (refund, cancel, delivery status) → use that intent.
- Only use `complaint` when the message is purely frustration with no actionable
  request.

**Tie-breaking rule:** If in doubt between `complaint` and another intent, prefer
the other intent if a concrete ask is recoverable; otherwise `complaint`.

---

### 8. `info_request`
**Definition:** Pre-purchase or informational question with no active issue.
The customer has not yet placed an order or is asking a general question.

**Examples:**
- "is Prime monthly or annual?"
- "planning to buy Fire HD 8, does it support X?"
- "what does this email mean?"
- "can't find their contact details"

**Excludes:**
- Post-purchase status question → `order_status`
- Reporting a bug in a service → `product_issue`

---

### 9. `other`
**Definition:** Fragments, unclassifiable messages, thanks/praise, or non-action
social chatter. Also includes non-English positive chatter that has no support
content.

**Examples:**
- "Ok sounds good thanks. Done."
- "Duna, Solaris, Fabrica de Vespas!" (Portuguese book chatter)
- bare order numbers with no ask ("Your amazon.in Order #402-...")
- "Acabei de comprar o Kindle 😍" (Portuguese: "Just bought the Kindle 😍")

**Target rate:** ≤15% of the golden set. If it exceeds that, the taxonomy needs
revision.

---

## Design Principles

1. **One intent per message.** If a message contains multiple asks, label the
   **primary** ask (the first concrete action requested, or the most actionable).
2. **Resolution-path based.** Two intents should never have the same default
   resolution. If they do, they should be merged.
3. **Actionable vs. non-actionable.** `complaint` and `other` are the two
   non-actionable categories; the rest imply a resolution step.
4. **Small but complete.** 9 intents balances classifier trainability against
   coverage. Adding a 10th intent requires strong evidence of a distinct
   resolution path.
5. **Language-agnostic labels.** Intents apply to the customer's ask regardless
   of the language of the message. Non-English support traffic is labelled the
   same as English.

---

## Escalation Policy (draft — for Day 3)

The default resolution column above is a **starting hypothesis**. It will be
frozen during golden-set construction. Proposed rule:

| Intent | Default decision | Reason |
|--------|------------------|--------|
| `order_status` | auto_handle if tracking available, else escalate | tracking = safe auto-reply |
| `refund_return` | escalate if amount > ₹5000 / $50, else auto | high-value refunds need human review |
| `product_issue` | auto_troubleshoot first, escalate if unresolved | saves agent time on trivial issues |
| `account_access` | escalate | security-sensitive |
| `billing_payment` | escalate | financial risk |
| `cancellation` | auto_confirm | low risk, deterministic |
| `complaint` | escalate | requires empathy + authority |
| `info_request` | auto_handle | knowledge base covers this |
| `other` | auto_handle (generic ack) | no action needed |

This policy is the **ground truth** you will record in the golden set as
`escalation_ground_truth`. It is distinct from what the model predicts — the
model must learn to replicate this policy from the message content and retrieved
context.

---

## Version History

| Version | Date | Change |
|---------|------|--------|
| 1.0 | (today) | Initial locked taxonomy after two validation batches |