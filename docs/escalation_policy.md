# Escalation Policy — AmazonHelp

Version: 1.1
This policy defines the ground-truth decision for every golden-set example:
`auto_handle` or `escalate`. It is frozen before labelling to keep the
ground truth consistent. The model is expected to learn to replicate this
policy from message content + retrieved context.

## Principles

- **auto_handle** = the agent can draft a complete, safe reply without a human.
- **escalate** = a human must review or take over.
- Escalate when: financial risk, security risk, legal risk, or high-emotion
  cases requiring empathy and authority.
- When in doubt, **escalate** (safer for the brand).

## Escalation Rules

| Intent | Default | Escalate if... |
|--------|---------|----------------|
| order_status | auto_handle | package marked "delivered" but not received; order value > ₹5000; customer threatens legal action |
| refund_return | auto_handle | amount > ₹1000; customer reports multiple failed refunds |
| product_issue | auto_handle | safety issue (fire, injury, electrical); repeat complaint; high-value item > ₹5000 |
| account_access | escalate | always (security-sensitive) |
| billing_payment | escalate | always (financial risk) |
| cancellation | auto_handle | customer asks for retention offer; contract/legal issue |
| complaint | escalate | always (needs human empathy) |
| info_request | auto_handle | complex policy question with no KB answer |
| other | auto_handle | generic ack; no action required |

---

## Reason Codes — Complete Reference

Each row in the golden set gets exactly ONE `escalation_reason` tag. The tag
explains *why* the escalation policy made the decision it did — not what the
customer wants (that's the intent).

### 1. `low_risk`
**Use when:** the case is routine, the answer is obvious, and no money,
security, or emotion is involved.
**Applies to:** `info_request`, `cancellation`, `other`, and simple
`order_status` / `product_issue` auto-handled cases.
**Examples:**
- "Is Prime monthly or annual?" → info_request, auto_handle, low_risk
- "Cancel my order #123" → cancellation, auto_handle, low_risk
**Do NOT use for:** anything with money, anger, or security implications.

---

### 2. `deterministic_action`
**Use when:** the correct reply follows a fixed, unambiguous rule — the agent
has no discretion, so auto-handling is safe.
**Applies to:** `order_status` (tracking lookup), `cancellation` (cancel order),
`info_request` (documented policy).
**Examples:**
- "When will my order arrive?" (tracking available) → order_status, auto_handle, deterministic_action
- "How do I return an item?" → info_request, auto_handle, deterministic_action
**Do NOT use for:** questions that require interpretation or judgement.

---

### 3. `security_sensitive`
**Use when:** the case involves account access, credentials, identity,
unauthorised access, or anything where a wrong reply could compromise the
customer.
**Applies to:** `account_access` — always.
**Examples:**
- "I can't log in, password reset 4x failed" → account_access, escalate, security_sensitive
- "Someone hacked my account and changed my email" → account_access, escalate, security_sensitive
- "I clicked a phishing link, what do I do?" → account_access, escalate, security_sensitive
**Do NOT use for:** routine "how do I change my password" *info_request* questions
(those are auto_handle, low_risk).

---

### 4. `financial_risk`
**Use when:** money is involved and the wrong reply could cost the brand or the
customer — charges, refunds, payments, disputed amounts.
**Applies to:** `billing_payment` — always. Also for `refund_return` above
₹1000 and for any disputed charge.
**Examples:**
- "I was charged twice for order 123" → billing_payment, escalate, financial_risk
- "Unauthorised transaction on my account" → billing_payment, escalate, financial_risk
- "Refund of ₹5000 not received after a month" → refund_return, escalate, financial_risk
**Do NOT use for:** simple "how do refunds work?" info_request (auto_handle).

---

### 5. `high_emotion`
**Use when:** the customer is clearly angry, distressed, or venting — and the
situation calls for human empathy rather than a templated reply.
**Applies to:** `complaint` — always. Also for `order_status` / `refund_return`
where the tone is very heated.
**Examples:**
- "Your service is a joke, no one replies" → complaint, escalate, high_emotion
- "I'm FURIOUS, this is the third time" → complaint, escalate, high_emotion
- "I'm going to post about this everywhere" → complaint, escalate, high_emotion
**Do NOT use for:** neutral messages, even if the underlying issue is serious —
use `financial_risk` or `security_sensitive` instead.

---

### 6. `high_value`
**Use when:** the order or refund amount exceeds ₹5000 (approximately $60),
making the case worth human attention even if the issue itself is routine.
**Applies to:** `order_status`, `product_issue`, `refund_return` above threshold.
**Examples:**
- "My ₹12,000 TV arrived broken" → product_issue, escalate, high_value
- "Refund for order #999 (value ₹8,500) still not received" → refund_return, escalate, high_value
**Do NOT use for:** small-value cases even if they're escalated for other reasons.

---

### 7. `repeat_failure`
**Use when:** the customer explicitly says this is the 2nd, 3rd, or Nth time
they've reported the same issue, OR the thread context shows a prior
unresolved attempt.
**Applies to:** any intent, but most commonly `order_status`, `refund_return`,
`product_issue`.
**Examples:**
- "This is the third time I'm contacting you about this" → complaint or order_status, escalate, repeat_failure
- "You did it AGAIN — same problem as last time" → complaint, escalate, repeat_failure
**Do NOT use for:** first-time reports, even if severe.

---

### 8. `legal_threat`
**Use when:** the customer mentions lawyers, consumer court, legal action,
chargebacks via bank, or regulatory complaints.
**Applies to:** any intent.
**Examples:**
- "If this isn't resolved I'll take this to consumer court" → escalate, legal_threat
- "I'm filing a chargeback with my bank" → billing_payment, escalate, legal_threat
**Do NOT use for:** milder threats like "I'll never shop here again" — that's
`high_emotion`.

---

### 9. `safety_issue`
**Use when:** the product could cause harm — fire, electrical fault, injury,
health risk, or a defective item that has already caused damage.
**Applies to:** `product_issue`.
**Examples:**
- "My Fire TV adapter sparked and burned the cable" → product_issue, escalate, safety_issue
- "The pressure cooker exploded and injured my hand" → product_issue, escalate, safety_issue
**Do NOT use for:** ordinary "the item is broken" reports (product_issue,
auto_handle, low_risk) unless there's a risk of harm.

---

### 10. `kb_unavailable`
**Use when:** the customer's question is legitimate, but no approved Knowledge
Base article or historical reply covers it — a human must research the answer.
**Applies to:** rare `info_request` cases, unusual policy questions, cross-cutting
scenarios.
**Examples:**
- "Why did the Prime Video price change from ₹499 to ₹599 for my account only?" → info_request, escalate, kb_unavailable
- "What's the exact refund timeline for a Kindle purchased with a gift card in Europe?" → info_request, escalate, kb_unavailable
**Do NOT use for:** questions that ARE covered by a KB article, even if you
personally don't know the answer.

---

## Decision Priority (when two tags both seem to fit)

Use this order — the first matching tag wins:

1. `legal_threat` — always trumps other reasons
2. `safety_issue` — physical harm beats everything except legal
3. `security_sensitive` — account compromise is high priority
4. `financial_risk` — money disputes
5. `high_value` — amount threshold
6. `repeat_failure` — pattern of unresolved attempts
7. `high_emotion` — venting / anger
8. `kb_unavailable` — no answer available
9. `deterministic_action` — routine, rule-based
10. `low_risk` — fallback for auto-handled cases

## Reason × Escalation Consistency

| Reason | Always paired with |
|--------|-------------------|
| `low_risk` | auto_handle |
| `deterministic_action` | auto_handle |
| `security_sensitive` | escalate |
| `financial_risk` | escalate |
| `high_emotion` | escalate |
| `high_value` | escalate |
| `repeat_failure` | escalate |
| `legal_threat` | escalate |
| `safety_issue` | escalate |
| `kb_unavailable` | escalate |

If you ever find yourself pairing a reason with the "wrong" escalation
(e.g., `financial_risk` + `auto_handle`), stop and re-check — either the intent
or the reason is mislabelled.

## Version History

| Version | Change |
|---------|--------|
| 1.0 | Initial frozen policy |
| 1.1 | Added full reason-code reference with usage examples and decision priority |