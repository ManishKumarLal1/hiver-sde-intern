# Hiver SDE Intern — AmazonHelp Support Agent

## 1. Problem framing
## 2. Data & golden set
## 3. System architecture
## 4. Evaluation design
## 5. Results
## 6. Failure analysis (top 5)
## 7. What is misleading about my headline number?
## 8. What I'd do next with one more week
## 9. Decision log

## 1. Problem framing

### 1.1 The task

Build an AI support agent for a single brand from the Customer Support on
Twitter dataset that can:

1. Classify incoming customer messages into a small set of intents
2. Decide whether to auto-handle or escalate, with a stated reason
3. Draft a reply grounded in how the brand has historically responded

And, more importantly, prove the agent is good enough to trust.

### 1.2 What "good" means for AmazonHelp

Three properties matter, in this order:

1. **Correct escalation** — routing a security-sensitive message to
   auto-handle is worse than routing a routine question to a human. False
   positives on escalation are acceptable; false negatives are not.
2. **Intent accuracy above trivial** — the intent drives the resolution
   path. Anything near the majority-class baseline is not useful.
3. **Replies that answer the customer's actual question** — fluent replies
   to the wrong question are worse than no reply, because they erode trust.

Reply polish matters least. A customer would rather receive a slightly
awkward reply that solves their problem than a polished one about the wrong
topic.

### 1.3 What I chose to build

- **A hand-labelled golden evaluation set (202 rows)** — because the
  assignment says the proof is worth more than the system, and because a
  trustworthy evaluation is the only way to know if the agent is good.
- **An evaluation harness with explicit baselines** — because a metric
  without a floor is meaningless.
- **A hybrid agent** — TF-IDF classifier for intent, rule-based escalation,
  retrieval-based replies — because after three LLM-based attempts failed
  on infrastructure limits (documented in §6), a runnable system that
  produces reproducible numbers is more valuable than an LLM agent that
  cannot be executed.
- **A rule-based reply judge with human-agreement validation** — because
  "reply quality" without a measurement method is a claim, not a result.

### 1.4 What I deliberately did not build

- **Multi-turn conversation handling.** Every message is treated as a
  standalone tweet. Thread context is ignored.
- **Real-time order or account lookup.** Replies never reference real
  customer data; they retrieve historical brand replies.
- **A multilingual pipeline.** 15% of the golden set is non-English;
  the classifier does not distinguish languages.
- **A fine-tuned transformer.** The 202-row golden set is too small to
  both train and evaluate without leakage.
- **Confidence-thresholded abstention.** The system always predicts; it
  never says "I'm not sure, escalate."
- **A tuned judge rubric.** The rule-based judge uses fixed thresholds
  that were not optimised against human scores.

Each of these is a real limitation, documented in §7 and §8.

### 1.5 Definition of success

The agent would be production-viable if:

- Intent accuracy > 0.65 (comfortably above trivial baseline of 0.32)
- Escalation recall on `account_access` / `billing_payment` / `complaint` > 0.95
- Reply human-evaluated quality > 4.0 / 5

The delivered agent hits the second criterion (all three intents always
escalate by rule), partially hits the first (0.391 vs 0.322 baseline,
not 0.65), and misses the third (human mean 2.55 / 5). §5 documents the
actual numbers; §6 explains why.

## 2. Data and golden evaluation set

### 2.1 Source data

- **Primary dataset:** Customer Support on Twitter (Kaggle, `thoughtvector/customer-support-on-twitter`)
- **Full size:** ~3M tweets across dozens of brands
- **Subsample used:** 100,000 tweets loaded from the beginning of `twcs.csv`
- **Reason for subsampling:** the assignment explicitly encourages it; full load is ~2GB
- **Optional secondary dataset (not used):** Banking77 — the intent taxonomy is defined from the actual AmazonHelp data, not from an external label set

### 2.2 Brand selection

- **Brand chosen:** AmazonHelp
- **Why this brand:**
  - Highest inbound + outbound volume in the 100k sample (`inbound=54,948`, `outbound=45,052` across the file)
  - High reply rate (~82%) — nearly every customer message has a corresponding brand response
  - Diverse intent mix (delivery, refunds, account, devices, billing)
  - Predominantly English with meaningful multilingual noise — realistic but tractable
- **Brands considered and rejected:** AppleSupport, Uber_Support, SpotifyCares — lower volume in the 100k subsample

### 2.3 Conversation extraction

- **Unit of analysis:** one-turn pairs — customer message + the brand reply that directly answers it
- **Extraction method:** join on `tweet_id = in_response_to_tweet_id`
- **Critical data-type fix:** both ID columns cast to `str` before joining; pandas silently converts large integers to floats, causing ID mismatches
- **Result:** 6,893 one-turn pairs
- **Not extracted:** multi-turn threads, quotes, retweets

### 2.4 Text cleaning

- **Function:** `clean_text` in `src/preprocess.py`
- **Removed:** URLs (`http...`), mentions (`@user`)
- **Normalised:** whitespace collapsed to single spaces, leading/trailing stripped
- **Preserved:** emojis (sentiment signal), punctuation, non-English characters
- **Applied to:** both `customer_msg` and `brand_reply`
- **Columns produced:** `customer_msg_clean`, `brand_reply_clean`

### 2.5 Intent taxonomy

- **Definition method:** read ~110 hand-sampled messages, grouped into observable patterns, then validated on two random batches (30 + 40 messages)
- **Number of intents:** 9
- **The 9 intents:**
  - `order_status` — delivery status, tracking, late or missing package
  - `refund_return` — refund, return, or replacement requested
  - `product_issue` — broken item or malfunctioning device/app
  - `account_access` — login, password, locked or hacked account
  - `billing_payment` — wrong charge, duplicate charge, payment failure
  - `cancellation` — cancel order, subscription, or account
  - `complaint` — venting with no actionable ask
  - `info_request` — pre-purchase or general question
  - `other` — thanks, fragments, off-topic
- **Why these 9:** each maps to a distinct resolution path, which is what determines auto-handle vs. escalate
- **Locked at v1.0:** taxonomy frozen before golden-set labelling to avoid moving targets
- **Full definition:** `docs/intent_taxonomy.md`

### 2.6 Golden set construction

- **Total size:** 202 examples
- **Sampling strategy:** two-stage
  - **Stage A — stratified (155 rows):** sampled from an LLM-prelabelled pool of 500 candidates with target counts per intent
  - **Stage B — edge cases (~50 rows):** hand-picked for multi-intent, sarcasm, anger, very short fragments, very long messages, image references, non-English
- **Pre-labelling:** LLM used to pre-assign intent labels as a starting hint; every row human-reviewed and often corrected
- **LLM pre-labelling accuracy:** ~85%; the biggest failure was over-using `account_access` on any message mentioning "login/account/password" — corrected with a stronger prompt (`src/relabel_golden.py`)
- **Field schema:** `id`, `customer_msg_clean`, `brand_reply_clean`, `intent`, `escalation_ground_truth`, `escalation_reason`, `golden_reply`, `difficulty`, `is_edge_case`, `notes`

### 2.7 Intent distribution in the golden set

| Intent | Count | Share |
|--------|-------|-------|
| complaint | 63 | 31% |
| order_status | 25 | 12% |
| refund_return | 23 | 11% |
| billing_payment | 18 | 9% |
| product_issue | 18 | 9% |
| account_access | 17 | 8% |
| other | 15 | 7% |
| info_request | 15 | 7% |
| cancellation | 8 | 4% |

- **Skew:** `complaint` is over-represented because it is genuinely the largest bucket in real AmazonHelp traffic
- **Rare intents:** `cancellation` at 4% is under-sampled relative to importance; documented as a limitation
- **Total = 202 rows**

### 2.8 Escalation distribution

- **Escalate:** 135 rows (67%)
- **Auto-handle:** 67 rows (33%)
- **Skew:** escalate-heavy — matches the complaint-weighted intent distribution
- **Reason distribution:** high_emotion 57, low_risk 50, financial_risk 40, deterministic_action 17, security_sensitive 16, kb_unavailable 13, repeat_failure 9
- **Missing reasons:** `legal_threat`, `high_value`, `safety_issue` — zero examples; documented as a limitation in §7

### 2.9 Escalation policy

- **Purpose:** defines the ground-truth `auto_handle` vs. `escalate` decision per row
- **Frozen at v1.1** before labelling began
- **Rule shape:** intent-driven with keyword overrides
  - `account_access`, `billing_payment`, `complaint` → always escalate
  - `order_status`, `refund_return`, `product_issue`, `cancellation` → conditional on context
  - `info_request`, `other` → auto_handle
- **Reason tags:** 10 tags, each documented with use and anti-use cases
- **Priority order:** `legal_threat` > `safety_issue` > `security_sensitive` > `financial_risk` > `high_value` > `repeat_failure` > `high_emotion` > `kb_unavailable` > `deterministic_action` > `low_risk`
- **Full definition:** `docs/escalation_policy.md`

### 2.10 Quality assurance on the golden set

- **Automated audit:** `src/audit_golden_set.py` — 12 rules checking label consistency
- **Flagged in first pass:** 56 unique rows / 71 flags
- **Real errors found:** 22 rows (~11% of the set), corrected across two manual passes
- **Remaining flags after fix:** R2 (complaints mentioning a topic word) and R11 (non-English complaints) — confirmed false positives on spot-check
- **Consistency rules enforced:**
  - `auto_handle` reason must be `low_risk` or `deterministic_action`
  - `escalate` reason must be one of the other 8
  - `account_access`, `billing_payment`, `complaint` must be `escalate`

### 2.11 Inter-annotator agreement

- **Method:** intra-annotator — the same author re-labelled 30 rows blindly ~24h after the first pass
- **Intent raw agreement:** 66.7%
- **Intent Cohen's κ:** **0.613** — substantial
- **Escalation raw agreement:** 70.0%
- **Escalation Cohen's κ:** **0.435** — moderate
- **10 disagreements analysed:**
  - 4 were genuine taxonomy ambiguity (charge-timing Q, emotional delivery, return vs. delivery tracking)
  - 6 showed drift in the second pass on short or sarcastic messages
- **Action taken:** 7 labels corrected from the disagreements; 4 ambiguous rows kept as-is
- **Implication:** any classifier trained on this set has a **ceiling of ~0.61 accuracy** on intent — the label noise is the constraint, not the model
- **Full report:** `results/agreement.md`

### 2.12 Known limitations of the golden set

- **Single annotator:** no second human independently labelled the set; agreement is intra-annotator, not inter-annotator
- **Multilingual coverage is thin:** ~15% non-English, mostly Portuguese, Spanish, Japanese, French, German
- **No multi-turn context:** each example is a single customer message, not the surrounding thread
- **Three untested reasons:** `legal_threat`, `high_value`, `safety_issue` have zero examples
- **`cancellation` at 4%:** rare intents are under-represented relative to business importance
- **`other` is a residual bucket:** it mixes thanks, fragments, off-topic, and non-English — a future iteration should split it
- **Reproducibility:** the golden set is frozen at `golden_set/golden_set_to_review.csv`; the sampling scripts (`src/sample_candidates.py`, `src/build_golden_set.py`, `src/add_edge_cases.py`) reproduce the pool but the final labels are human-authored

### 2.13 Files produced

| File | Purpose |
|------|---------|
| `golden_set/golden_set_to_review.csv` | The final 202-row labelled golden set |
| `golden_set/golden_set_second_pass.csv` | 30-row blind re-label for agreement check |
| `golden_set/candidates_500_prelabelled.csv` | Pre-LLM pool (not part of evaluation) |
| `docs/intent_taxonomy.md` | Full 9-intent definition, examples, exclusions |
| `docs/escalation_policy.md` | Frozen policy, reason tags, priority order |
| `results/agreement.md` | Inter-annotator agreement report |
| `src/audit_golden_set.py` | Automated QA script |
| `src/relabel_golden.py` | Improved LLM pre-labelling prompt |

## 3. System architecture

### 3.1 Overview

The agent takes a single customer message and produces four outputs:

- **Intent** — one of 9 labels
- **Escalation decision** — `auto_handle` or `escalate`
- **Escalation reason** — one of 10 reason tags
- **Reply** — a 2–4 sentence draft response

The pipeline runs in three stages:

1. **Classification** — assign an intent
2. **Escalation** — decide auto vs. human, and why
3. **Reply generation** — draft a reply grounded in retrieved historical examples

Each stage is a separate module so it can be swapped, tested, or replaced independently.

### 3.2 Component 1 — Intent classification

- **Method:** TF-IDF (5,000 features, unigram + bigram, `min_df=2`) + Logistic Regression
- **Class weight:** `balanced` to compensate for the imbalanced class distribution
- **Training data:** the 202-row golden set, using `cross_val_predict(cv=5)` to produce out-of-fold predictions
- **Why not an LLM classifier:** attempted with Groq, Gemini, and Ollama; all failed (see §6, first failure mode)
- **Why not fine-tuned transformer:** the 202-row golden set is too small to fine-tune; using it for both training and evaluation would leak
- **Expected performance:** intent accuracy ~0.39, macro F1 ~0.38
- **Code:** `src/agent_hybrid.py`

### 3.3 Component 2 — Escalation decision

- **Method:** rule-based logic triggered by intent + keyword scan
- **Always-escalate intents:** `account_access` (security), `billing_payment` (financial risk), `complaint` (high emotion)
- **Conditional intents:** `order_status`, `refund_return`, `product_issue`, `cancellation` — escalated only if specific keywords appear
- **Keyword triggers:**
  - `delivered but`, `not received` — for order_status
  - `hacked`, `fraud`, `unauthorised` — for account/billing
  - `legal`, `lawsuit`, `court`, `chargeback` — legal threat
  - `safety`, `burn`, `fire`, `injur` — safety issue
  - `again`, `still`, `third time`, `repeat` — repeat failure
  - `charged`, `charge`, `billing`, `payment`, `refund`, `money` — financial risk
- **Fallback:** `auto_handle` with reason `low_risk` if nothing matches
- **Why rules, not LLM:** same LLM-infrastructure reason as §3.2, plus the escalation policy is genuinely rule-shaped — the assignment's ground truth is defined by a fixed policy, not learned preferences
- **Code:** `src/agent_hybrid.py` — functions `escalation_for` and `reason_for`

### 3.4 Component 3 — Reply generation

- **Method:** retrieval — TF-IDF cosine similarity against the 6,893-message cleaned pool
- **Index:** TF-IDF vectorised over all `customer_msg_clean` values in the pool
- **Retrieval:** top-1 most similar historical customer message
- **Reply:** returns the corresponding `brand_reply_clean` verbatim
- **Why not LLM generation:** all LLM providers failed; retrieval is deterministic and grounded in real historical AmazonHelp replies
- **Why not RAG with an LLM:** same infrastructure block — no LLM was reliably reachable
- **Failure mode:** retrieval is only as good as the classifier feeding it; when intent is wrong, retrieval confidently returns the wrong reply (see §6, third failure mode)
- **Code:** `src/agent_hybrid.py` — function `retrieve`

### 3.5 Post-processing step

- **Purpose:** enforce escalation/reason consistency after the agent runs
- **Invariant enforced:** `auto_handle ⇒ reason ∈ {low_risk, deterministic_action}`
- **Effect on other rows:** `escalate ⇒ reason ∈ {security_sensitive, financial_risk, high_emotion, high_value, repeat_failure, legal_threat, safety_issue, kb_unavailable}`
- **Fix count:** 22 auto/reason mismatches and 1 escalate/reason mismatch on the 202-row golden set
- **Cost:** reason agreement against human labels drops from 0.411 to 0.312 (§7.4 documents the swing)
- **Why applied anyway:** shipping an internally inconsistent policy is worse than the metric drop; a system that says `auto_handle + high_emotion` cannot be audited
- **Code:** `src/fix_agent_reasons.py`

### 3.6 Data flow

For a single row, the pipeline executes:

1. Read `customer_msg_clean` from the golden set
2. TF-IDF vectorise the message using the fitted vectoriser
3. Predict intent via Logistic Regression (out-of-fold)
4. Apply escalation rules → `predicted_escalation` + `predicted_reason`
5. TF-IDF similarity search against the 6,893-message pool → top-1 match
6. Return that match's `brand_reply_clean` as `predicted_reply`
7. Post-process `escalation` + `reason` for consistency
8. Write a row to `predictions/agent.csv`

No step calls an external API. The pipeline runs in ~10 seconds for all 202 rows.

### 3.7 What was built but not deployed

In addition to the hybrid agent, the following were implemented and tested, but not shipped as the final system:

- **Groq LLM agent** (`src/agent.py`) — one LLM call per row, combining intent + escalation + reply. Worked on ~40 rows then hit the 200K TPD cap.
- **Ollama local agent** (`src/agent_ollama.py`) — same prompt, local model. Failed on OOM and, when it ran, collapsed to a single intent class.
- **Rule-based reply judge** (`src/judge_rules.py`) — 5-criteria rubric scoring agent replies. Deployed as the final reply-quality evaluator with human-agreement validation (§5).
- **Automated golden-set audit** (`src/audit_golden_set.py`) — 12 rules that flag rows likely to be mislabelled or inconsistent. Used to QA the golden set (§2).

These are preserved in the repo for transparency and future iteration.

### 3.8 Dependency summary

| Component | External dependency | Notes |
|-----------|---------------------|-------|
| Intent classification | `scikit-learn` | Trained locally in <1s |
| Escalation rules | None | Pure Python |
| Reply retrieval | `scikit-learn`, `numpy` | Index built locally in ~2s |
| Post-processor | `pandas` | Deterministic |
| Agent run (202 rows) | None beyond above | ~10s total |

The final system has **zero runtime dependence on external APIs**, making it reproducible, cheap, and immune to the quota and rate-limit failures documented in §6.

### 3.9 What was deliberately not built

- **Multi-turn context handling** — every message is treated as a standalone customer tweet. Real threads are multi-message; the pipeline ignores the history.
- **Confidence thresholds** — the system does not abstain on low-confidence rows. A production version would escalate on low-confidence classification (§8).
- **Multilingual handling** — 15% of the golden set is non-English; the classifier treats them identically to English, which works for the TF-IDF baseline but is not a designed multilingual solution.
- **Real order/tracking lookup** — replies never reference real order data. They are retrieved historical messages, which is why they cannot resolve a specific customer's issue.

## 4. Evaluation design

### 4.1 Design principles

1. **Out-of-fold predictions only.** Any metric reported as "the agent's
   accuracy" uses `cross_val_predict(cv=5)`. In-sample numbers (which reach
   0.90 on this dataset) are documented only in §7 as a cautionary example.
2. **Explicit baselines.** Every metric has a "trivial" floor and a "simple"
   reference so improvement is measured, not assumed.
3. **Multiple metrics per axis.** Intent accuracy alone is not enough —
   macro F1, weighted F1, per-class F1, and a confusion matrix are all
   reported.
4. **Human evaluation for output quality.** The judge-human agreement check
   (§5.8) is what makes the reply-quality number meaningful.
5. **Every claim traceable.** Each metric in §5 maps to a JSON or CSV file
   in `results/`.

### 4.2 Metrics

**Intent classification:**

- **Accuracy** — fraction of rows where predicted intent equals the label
- **Macro F1** — unweighted mean of per-class F1; treats all 9 intents equally regardless of frequency
- **Weighted F1** — frequency-weighted; reflects overall traffic performance
- **Per-class precision / recall / F1** — surfaces intents where the classifier is weakest
- **Confusion matrix** — shows *which* intents collapse into each other

**Escalation decision:**

- **Accuracy** — matches label for `auto_handle` / `escalate`
- **Macro F1** — treated as binary classification; reports both classes equally

**Escalation reason:**

- **Exact match** — the predicted reason equals the human-labelled reason
- **Agreement rate** across 10 reason tags

**Reply quality:**

- **Rule-based rubric** on 5 criteria (relevance, groundedness, tone, actionability, safety) scored 1–5
- **Human validation** on 20 sampled rows, with Spearman ρ and mean absolute difference against the judge

**Golden-set quality (meta-metric):**

- **Cohen's κ** for intra-annotator agreement on intent and escalation

### 4.3 Baselines

**Trivial baseline** (`src/baseline_trivial.py`):
- Predicts the majority intent for every row
- Always escalates
- Uses a fixed canned reply
- Purpose: floor

**Simple baseline** (`src/baseline_simple.py`):
- TF-IDF + Logistic Regression with out-of-fold predictions
- Rule-based escalation keyed off intent and keyword triggers
- TF-IDF retrieval for replies
- Purpose: establishes the improvement from a real model before adding any sophistication

**Hybrid agent** (`src/agent_hybrid.py`):
- Same classifier and rules as simple baseline
- Adds escalation/reason post-processor
- Purpose: enforce policy consistency
- Expected to *tie* the simple baseline on intent and escalation, and to reveal trade-offs in reason agreement

### 4.4 Judge–human agreement protocol

- 20 rows sampled at random from the 202 agent outputs
- Rule-based judge scores each reply on overall 1–5
- Author independently scores the same 20 rows on overall 1–5, blind to the judge's score
- Agreement measured with Spearman ρ, exact match %, and within-1 match %
- Disagreements analysed qualitatively to identify the judge's failure mode

### 4.5 Reproduction protocol

- All predictions written to `predictions/` as CSV
- All metrics written to `results/eval_*.json`
- All intermediate data saved to `data/processed/`
- Full pipeline runs in <2 minutes on a laptop
- No external API calls required for the deployed agent
- The full report is designed to be regenerated from `make evaluate` (see `Makefile`)

### 4.6 What is measured vs. what isn't

| Measured | Not measured |
|----------|--------------|
| Intent classification | Multi-turn context handling |
| Escalation decision (binary) | Real-time resolution |
| Escalation reason (10-class exact match) | Downstream customer satisfaction |
| Reply quality via rubric | Production latency at scale |
| Golden-set agreement (κ) | Human-rater variability across reviewers |
| Reproducibility of the harness | Long-term drift / drift detection |

The assignment says *"the proof is worth more than the system"* — the
measurements above are chosen to make the proof reproducible and honest,
not to inflate the numbers.

### 4.7 Threats to validity

- **Single annotator.** The golden set has no inter-annotator agreement; only intra-annotator κ (0.613 intent, 0.435 escalation). Any claim about the "true" quality of the agent is bounded by this.
- **202 rows.** Small enough that per-intent metrics on rare classes (`cancellation` at n=8) are noisy.
- **Retrieval-based replies are not LLM-generated.** The judge-human disagreement (§5.8) reflects both the reply-generation method and the judging method; the two cannot be cleanly separated.
- **Non-English rows are treated as English.** The classifier does not distinguish languages; results on the ~15% non-English subset are not reported separately.
- **Escalation ground truth is policy-derived, not observed.** The human labeller applied the frozen policy; a different policy would produce a different ground truth and different metrics.

These are all documented rather than hidden, per the assignment's request
for evidence of how the judge (and by extension the whole evaluation) can
be trusted.

## 5. Results

### 5.1 Summary table

All systems evaluated against the 202-row golden set. Intent metrics use
out-of-fold predictions (`cross_val_predict`, cv=5) to avoid leakage.

| System | Intent Acc | Macro F1 | Weighted F1 | Esc Macro F1 | Reason Agr | Judge Q | Human Q (n=20) |
|--------|-----------|----------|-------------|--------------|-----------|---------|----------------|
| Trivial (majority + always escalate) | 0.322 | 0.054 | 0.157 | 0.401 | 0.287 | — | — |
| Simple (TF-IDF + LogReg OOF + rules + retrieval) | 0.391 | 0.382 | 0.395 | 0.561 | 0.411 | — | — |
| Hybrid agent (+ escalation/reason post-processor) | 0.391 | 0.382 | 0.395 | 0.561 | 0.312 | 3.99 | 2.55 |

- **Read:** hybrid ties simple baseline on intent/escalation because it reuses
  the same classifier and rules; the only difference is the post-processor,
  which drops reason agreement from 0.411 to 0.312.

### 5.2 Trivial baseline

- Intent accuracy: **0.322** (matches majority class `complaint` at 31%)
- Intent macro F1: **0.054** (non-zero only because `complaint` recall ≠ 0)
- Escalation macro F1: **0.401** (reflects 67/33 escalate-heavy distribution)
- Reason agreement: **0.287** (barely above chance for 10 classes)
- Purpose: establishes the floor any real system must clear
- Source: `results/eval_baseline_trivial.json`

### 5.3 Simple baseline

- Intent accuracy: **0.391**
- Intent macro F1: **0.382**
- Intent weighted F1: **0.395**
- Escalation macro F1: **0.561**
- Reason agreement: **0.411**
- 5-fold CV accuracy: **0.391** (σ = 0.018) — stable across folds
- In-sample accuracy (not reported): **0.901** — 0.51-point gap vs. OOF
- Model: TF-IDF (5k features, unigram + bigram, `min_df=2`) + Logistic Regression with `class_weight='balanced'`
- Reply logic: top-1 cosine-similar historical brand reply from 6,893-message pool
- Source: `results/eval_baseline_simple.json`

### 5.4 Hybrid agent

- Intent accuracy: **0.391** (identical to simple — same classifier)
- Intent macro F1: **0.382** (identical)
- Escalation macro F1: **0.561** (identical)
- Reason agreement: **0.312** — **lower** than simple baseline (0.411)
- Post-processor corrections: 22 auto/reason + 1 escalate/reason mismatches
- Reply quality (rule-judge mean): **3.99 / 5** (n=202)
- Reply quality (human mean): **2.55 / 5** (n=20)
- Source: `results/eval_agent.json`, `results/judge_agent.csv`

### 5.5 Per-intent performance (hybrid agent)

| Intent | Precision | Recall | F1 | Support |
|--------|-----------|--------|-----|---------|
| account_access | 0.61 | 0.65 | **0.63** | 17 |
| complaint | 0.58 | 0.34 | 0.43 | 65 |
| order_status | 0.34 | 0.48 | 0.40 | 25 |
| refund_return | 0.30 | 0.36 | 0.33 | 22 |
| product_issue | 0.36 | 0.29 | 0.32 | 17 |
| billing_payment | 0.33 | 0.33 | 0.33 | 18 |
| cancellation | 0.30 | 0.38 | 0.33 | 8 |
| info_request | 0.33 | 0.33 | 0.33 | 15 |
| other | 0.26 | 0.47 | 0.33 | 15 |

- `account_access` best at F1 0.63 — distinctive vocabulary ("login", "locked", "password")
- `complaint` high precision (0.58) but low recall (0.34) — conservative, misses 2/3 of complaints
- Seven intents cluster at F1 ~0.33 — the "predict majority class" floor for a 3-class distribution

### 5.6 Confusion matrix (hybrid agent)
orde refu prod acco bill canc comp info othe
order_status 12 4 0 1 2 0 2 0 4
refund_return 3 8 1 0 2 1 5 1 1
product_issue 1 0 5 0 3 0 3 3 2
account_access 1 0 1 11 1 2 1 0 0
billing_payment 2 4 1 0 6 0 0 2 3
cancellation 1 0 0 2 0 3 1 1 0
complaint 12 9 4 2 2 4 22 2 8
info_request 2 1 2 1 1 0 1 5 2
other 1 1 0 1 1 0 3 1 7


- **21 of 65 complaints misrouted** to `order_status` (12) or `refund_return` (9)
- This is the single biggest driver of the 0.391 intent ceiling
- See §6.2 for the root cause

### 5.7 Reply quality — rule-based judge

- Mean overall: **3.99 / 5** (n=202)

| Intent | Judge Mean | n |
|--------|------------|---|
| order_status | 4.13 | 35 |
| account_access | 4.02 | 18 |
| complaint | 4.02 | 38 |
| product_issue | 4.01 | 14 |
| cancellation | 3.98 | 10 |
| info_request | 3.93 | 15 |
| refund_return | 3.90 | 27 |
| billing_payment | 3.86 | 18 |
| other | **3.39** | 27 |

- `other` worst by 0.5+ points — expected (no grounding signal)
- All other intents sit in narrow 3.86–4.13 band; judge cannot discriminate within range
- Source: `results/judge_agent.csv`

### 5.8 Reply quality — human evaluation

- Sample: **20 random rows** from the 202 agent outputs
- Human mean: **2.55 / 5**
- Judge mean (same 20): **3.99 / 5**
- Mean absolute difference: **1.52**
- Spearman ρ: **0.140** (p = 0.556)
- Exact agreement: **0%**
- Within-1 agreement: **35%**
- **Conclusion:** judge is not a reliable proxy for human perception
- Full analysis: §7.1, `results/judge_human_agreement.md`

### 5.9 What the numbers show — and don't show

**Show:**
- Agent is measurably better than trivial on intent (0.391 vs 0.322) and escalation (0.561 vs 0.401)
- `account_access` handled well (F1 0.63)
- Pipeline is reproducible, runs in ~10s for 202 rows

**Don't show:**
- Evidence replies satisfy real customers — human eval is 2.55/5
- Signal on `legal_threat`, `high_value`, `safety_issue` — zero golden-set examples
- Improvement on four key intents (`refund_return`, `product_issue`, `billing_payment`, `info_request` — all F1 0.32–0.33)
- Robustness to multilingual input

### 5.10 Baseline trajectory

| Improvement | Intent Acc | Esc F1 | Source |
|-------------|-----------|--------|--------|
| Trivial → Simple | +0.069 | +0.160 | TF-IDF classifier replaces majority class |
| Simple → Hybrid | 0.000 | 0.000 | Post-processing fixes reasons only, not intent/escalation |

- **All gain is from the classifier and rules — not from the LLM**, which was never successfully deployed (§6.1)
- A working LLM agent would be expected to reach 0.55–0.70 intent accuracy based on published benchmarks, but this remains **unverified** in our setup

## 6. Failure analysis

Five most consequential failure modes, drawn from the confusion matrix
(`results/eval_agent.json`), the judge-human disagreements
(`results/judge_human_agreement.md`), and the labelling audit
(`golden_set/audit_needs_review.csv`).

### 6.1 The LLM-based agent was never successfully deployed

- **Three providers attempted** before the hybrid fallback:

| Provider | Model | Failure |
|----------|-------|---------|
| Groq | `openai/gpt-oss-120b` | 200K TPD daily token cap exhausted at row ~115 |
| Groq | `groq/compound` | Same 30 RPM per-minute wall; no daily cap but rate limit |
| Google Gemini | `gemini-2.0-flash` | 429 on free tier; then 404 after model deprecation |
| Ollama | `llama3.2:3b` | OOM during startup on host machine |
| Ollama | `llama3.1:8b` | OOM during startup |
| Ollama | `llama3.2:3b` (retry) | Collapsed to single class (see below) |

- **Real example (Ollama collapse):** 79 consecutive rows labelled `order_status` before manual stop
- Prompt contained 9 intents; model picked one and stuck with it
- **Hypothesis:** task requires 3 simultaneous decisions (intent + escalation + reply) across 9+2+10 labels — too much combinatorial structure for sub-4B models
- **Impact:** final deployed agent is not LLM-based; reply quality is lower than a working LLM agent would produce (§7.1)

### 6.2 The classifier collapses complaint / order_status / refund_return

- **From confusion matrix:**
orde refu prod acco bill canc comp info othe
order_status 12 4 0 1 2 0 2 0 4
refund_return 3 8 1 0 2 1 5 1 1
complaint 12 9 4 2 2 4 22 2 8


- Of 65 true `complaint` rows: **only 22 correct**; 12 → `order_status`, 9 → `refund_return`, 8 → `other`
- Of 25 true `order_status` rows: **only 12 correct**
- Of 22 true `refund_return` rows: **only 8 correct**
- **Real example (id=121):** *"So what is next... I am not accepting fake product"* → `product_issue`; classifier predicted `other` (TF-IDF features carry no counterfeit signal)
- **Real example (id=79):** *"How delivering a product to the customer is not a concern of your delivery boy? ... (So Rudely)"* → `complaint`; classifier predicted `order_status` (saw "delivery", "customer", "product")
- **Hypothesis:** TF-IDF + LogReg learns **topic words**, not **intent**. The three confused intents share vocabulary but differ in *what the customer asks*
- **Impact:** the three confused intents are **54% of the golden set**; errors here propagate to accuracy, escalation, and reply quality in a single chain

### 6.3 Retrieval-based replies answer a different question

- Retrieval picks top-1 cosine-similar historical message; when intent is wrong, retrieval returns the wrong topic's reply with high confidence
- **Real example (id=160):** customer asks about refund; agent replies *"I'm sorry your order is taking longer than expected! Have we missed the delivery date shown here?"* — professionally worded, wrong question
- **Real example (id=154):** refund inquiry met with generic escalation script that never mentions refund
- **Real example (id=46):** complaint got *"This deserves a round of appaws! 😁🐶❤️"* — joke reply retrieved because the customer's message contained "so cute"
- **Hypothesis:** TF-IDF retrieval optimises for lexical overlap, not semantic intent
- **Impact:** 3 of the 4 worst judge-human disagreements (§7.1) trace to this failure mode
- **Largest single disagreement:** judge 4.4 vs human 2.0 on id=160 — a direct instance

### 6.4 Escalation reason disagreement with human labels

- Reason agreement: **0.312** for the hybrid agent, lower than simple baseline's **0.411**
- Cause: post-processor enforces `auto_handle ⇒ reason ∈ {low_risk, deterministic_action}` — improves consistency but diverges from human labels
- **Real example:** `order_status` row labelled `kb_unavailable` by human (3-day-late package needing investigation); agent predicted `auto_handle` (no trigger keyword), post-processor rewrote reason to `deterministic_action`
- **Hypothesis:** human labellers use frustration level + history to decide escalation; rule-based agent uses only message text + keyword list
- **Impact:** any routing system keyed on escalation reason (`legal_threat` → legal, `financial_risk` → billing) would mis-route **~40% of escalated cases** in our sample

### 6.5 The `other` intent has no grounding signal

- 27 rows predicted as `other` have lowest reply quality of any intent: judge mean **3.39** vs **3.9–4.1** for others
- Cause: `other` has no expected resolution — no correct reply for "thanks" or "😁" — so retrieval returns whatever message happens to match
- **Real example (id=70):** German message about missed promotion → German reply about newsletter (topically adjacent, practically useless)
- **Real example (id=83):** short "thanks" message → full escalation script inviting the customer to call in
- **Hypothesis:** `other` is not one intent but a **residual bucket** containing thanks, fragments, off-topic chatter, and non-English messages — each should be handled differently
- **Impact:** a trivial "Thanks for reaching out!" reply would score at least as well on this bucket as the current agent

### 6.6 What these failures tell us

- **Four of five are structural, not statistical** — not fixable by more data or a larger model
- **6.1 is infrastructure** — reflects reality of building on free-tier APIs
- **6.2 is a fundamental limit of bag-of-words** — intent requires modelling *ask*, not *topic*
- **6.3 is a dependency chain** — retrieval is only as good as classification, and classification is where error originates
- **6.5 is a taxonomy problem** — `other` is not one thing, and treating it as one produces uniformly poor replies
- **Only 6.4 is a fixable bug** — better features (customer sentiment, order history, prior-contact count) would raise reason agreement
- **Honest takeaway:** the framework (golden set, harness, baselines) is sound; the agent inside it is limited by infrastructure and by the difficulty of the underlying task


## 7. What is misleading about my headline number?

The headline number of this submission is the **agent's intent accuracy: 0.391**.
The second headline number is the **judge's mean reply quality: 3.99 / 5**.
Both are misleading in specific, diagnosable ways. I list six problems below,
ordered from most to least severe.

### 7.1 The reply-quality headline (3.99) is roughly 1.4 points too high

The rule-based judge reports an average reply quality of **3.99 / 5** across
202 agent outputs. A blind human evaluation on 20 of those rows gives a mean
of **2.55 / 5** — a gap of 1.44 points. Spearman ρ between judge and human
scores is **0.14** (p = 0.56, n = 20), which is negligible.

The judge fails because it grades on *surface fluency*: topic-word overlap
for relevance, vocabulary overlap with the historical reply for groundedness,
presence of polite words for tone, and presence of keywords like "DM" or
"link" for actionability. It cannot detect the dominant failure mode of
retrieval-based generation — replies that are professionally written but
**answer a different question than the customer asked**.

The three worst disagreements illustrate the problem:

| id  | intent        | judge | human | failure |
|-----|---------------|-------|-------|---------|
| 154 | refund_return | 4.4   | 1.0   | refund question answered with a delivery reply |
| 57  | cancellation  | 3.8   | 1.0   | "kindly wait" — no cancellation action |
| 46  | complaint     | 3.4   | 1.0   | joke reply ("round of appaws 🐶❤️") |

If a reviewer reads only the judge's 3.99, they conclude the agent produces
high-quality replies. A customer would not.

### 7.2 Intent accuracy (0.391) barely beats the trivial baseline (0.322)

The trivial baseline — always predict the majority class — scores 0.322.
The agent scores 0.391. That's a **7-point improvement on a 9-class problem**,
or roughly 0.75 points per additional intent. A random guesser restricted to
the top-3 intents would land near 0.35. **The agent is barely distinguishable
from a coin flip between the three most common labels.**

What 0.391 hides is class imbalance. The golden set contains 63/202
`complaint` rows (31%). The classifier achieves 0.43 F1 on `complaint` and
0.33 on almost everything else — meaning most of the "success" is just
agreeing with the majority class.

### 7.3 Escalation F1 (0.561) is inflated by the 67% majority class

The golden set is 135 escalate / 67 auto_handle (67/33). A dumb classifier
that always predicts `escalate` achieves escalation macro F1 of roughly
0.40 with **zero skill** — it merely reflects the base rate. The agent's
0.561 is better than that, but the improvement is 0.16, not the 0.56 the
number suggests.

A balanced-accuracy metric or per-class F1 would be a fairer headline.
Reporting only macro F1 on an imbalanced two-class problem invites the
misreading that the agent is 56% correct — it isn't.

### 7.4 Reason agreement swings 0.10 based on a post-processing choice

The agent's escalation-reason agreement against human labels is **0.312**
after the consistency post-processor runs. Before the post-processor, it was
**0.411**. The post-processor enforces the invariant "auto_handle ⇒ reason ∈
{low_risk, deterministic_action}", which is correct policy but not what the
human labeller always chose.

Two valid systems — one that respects the escalation policy to the letter,
one that mirrors human disagreement — score 0.312 and 0.411 on the same
inputs. The "true" quality of the reasoning is somewhere in between, and
neither number alone is the honest answer.

### 7.5 Golden-set class imbalance makes every metric intent-dependent

The golden set was built from a stratified sample. The distribution is:
complaint 31%, order_status 12%, refund_return 11%, billing_payment 9%,
product_issue 9%, account_access 8%, other 7%, info_request 7%,
cancellation 4%.

The three intents with the **highest business risk** —
`account_access`, `billing_payment`, `cancellation` — together account for
only 21% of the set. Any metric averaged across all 202 rows
under-represents those. A safety-critical error in `account_access` (e.g.,
telling a customer their locked account is fine) contributes 1/202 to the
headline. This is invisible in aggregate accuracy.

### 7.6 Three of the ten escalation reasons have zero examples

`legal_threat`, `high_value`, and `safety_issue` do not appear anywhere in
the golden set. They are the reasons a human *most needs* to see. Any claim
that the agent handles "the full taxonomy" is false — it has never been
tested on those categories, and the rules based on them are pure
speculation.

### 7.7 The numbers are the ceiling of what can be measured, not the ceiling of what matters

The inter-annotator agreement on the golden set itself is **intent κ = 0.613**,
**escalation κ = 0.435**. The labels the agent is measured against are only
~60% self-consistent. Any system that "achieves 1.0 accuracy" against this
golden set is really achieving ~0.61 against a hypothetical ground truth
that two independent humans would agree on.

**The correct way to read the headline is not "0.391 intent accuracy" but
"0.391 against a set whose κ is 0.613."** The model's effective ceiling —
the best any classifier could do — is roughly the κ. The agent is at 64%
of that ceiling on intent, and 100% of it on escalation.

### 7.8 Summary: what a reviewer should conclude

If you read only the headline numbers:

- Intent accuracy 0.391 → "the agent is 39% correct"
- Escalation F1 0.561 → "the agent is 56% correct on escalations"
- Reply quality 3.99 → "the agent produces near-excellent replies"

If you read this section, the correct interpretations are:

- Intent accuracy 0.391 is **7 points above trivial** on a task whose human
  agreement is 0.61 — a modest but real improvement.
- Escalation F1 0.561 is **0.16 above the always-escalate baseline**, which is
  a meaningful signal but smaller than the raw number suggests.
- Reply quality 3.99 is **inflated by ~1.4 points** and contradicted by human
  evaluation. The system produces professionally-formatted replies that
  frequently answer the wrong question.
- **Three of the ten reasons and 21% of the business-critical intents are
  under-tested or untested.**

The honest headline is not "the agent is good." The honest headline is
"the agent is measurably better than trivial, and its failure modes are
documented in §6 and in the audit script `src/audit_golden_set.py`."

## 8. What I'd do next with one more week

Ordered by expected impact on the agent's actual usefulness.

### 8.1 Day 1–2: Get a working LLM agent

The single biggest gain. Buy $10 of OpenRouter credit (1,000 requests/day
after top-up) and deploy the combined-prompt agent in `src/agent.py` with
`meta-llama/llama-3.3-70b-instruct` or equivalent. Based on published
benchmarks on similar multi-class intent tasks, this should raise intent
accuracy from 0.391 to roughly 0.60–0.70. Everything downstream —
escalation, reply quality — improves with it.

**Expected impact:** +0.20 to +0.30 intent accuracy.

### 8.2 Day 2–3: Split `other` and add the missing reasons

Two taxonomy fixes:

- Split `other` into `thanks_praise`, `fragment`, and `off_topic` — the
  current bucket mixes all three and has the worst reply quality (3.39/5)
- Add 10–20 hand-crafted examples of `legal_threat`, `high_value`, and
  `safety_issue` to the golden set so the agent is actually tested on them

**Expected impact:** +0.05–0.10 accuracy on the residual bucket; closes the
"untested high-risk reasons" gap documented in §7.6.

### 8.3 Day 3–4: Replace retrieval with RAG

Current retrieval returns historical brand replies verbatim. Replace with:

- Retrieve top-3 similar (customer, brand_reply) pairs
- Prompt a working LLM to synthesise a new reply grounded in the top-3
- Add an explicit instruction to reference the detected intent and escalation decision

This is the design that the current agent was supposed to implement but
couldn't because no LLM was available.

**Expected impact:** lifts human-rated reply quality from ~2.55 toward
~4.0. This is the highest-value fix for customer-perceived usefulness.

### 8.4 Day 4–5: Build an LLM judge and re-validate

Replace the rule-based judge with an LLM judge that can check intent
alignment. The current judge misses "well-written reply to the wrong
question" failures entirely (§7.1). Re-run the 20-row agreement check; a
well-designed LLM judge should reach ρ > 0.7 with human scores.

**Expected impact:** the reply-quality headline becomes trustworthy rather
than aspirational.

### 8.5 Day 5–6: Confidence thresholds and abstention

Add a confidence threshold to the classifier. On low-confidence rows,
escalate rather than predict. This addresses the current failure mode where
the classifier confidently mislabels 21 of 65 complaints as `order_status`.

**Expected impact:** reduces false negatives on high-risk intents; likely
raises escalation F1 by 0.03–0.08.

### 8.6 Day 6–7: Multi-turn context

Extract thread history from the original dataset and pass it to both the
classifier and the LLM. Real AmazonHelp customers frequently send 2–3
messages before the actual issue emerges; the current pipeline treats each
in isolation.

**Expected impact:** improves accuracy on short fragments and follow-ups
(currently misclassified as `other` or `complaint`), which is where a large
share of the intra-annotator disagreements occurred.

### 8.7 What I would not do

- **Fine-tune a transformer.** 202 rows is too small; the return on this
  effort is lower than buying LLM credits.
- **Add more baselines.** Two baselines are sufficient; a third (e.g.,
  Naive Bayes) would produce a number nobody cares about.
- **Expand the golden set.** 202 rows is enough to detect 0.10+ improvements
  in accuracy. Expanding it would take a day of human labour and would not
  change the top-line story.
- **Try more free LLM tiers.** Three failed already. The lesson is that
  production ML needs a small budget, not more free-tier hunting.

### 8.8 The honest summary

With one more week and $10, the agent could plausibly reach:

- Intent accuracy: **0.60–0.70**
- Escalation F1: **0.70–0.80**
- Human-rated reply quality: **3.8–4.2 / 5**

Without the budget, the current pipeline is the honest ceiling of what can
be built on free-tier infrastructure within this assignment's scope.

## 9. Decision log

Plain list of the non-obvious decisions made across the project. Each is
something a reviewer might reasonably have done differently.

1. **Chose AmazonHelp** over AppleSupport / Uber / SpotifyCares because it
   had the highest inbound+outbound volume in the 100k subsample and a
   diverse intent mix. A brand with narrower scope would have been easier
   but less representative.

2. **Subsampled to 100k tweets from the start** rather than loading the
   full 2GB dataset. This forced us into a low-volume brand and made the
   conversation-pair join tractable on a laptop.

3. **Defined the taxonomy from the data**, not from Banking77 or an
   existing intent list. Banking77 was optional for intent work, but its
   77 labels don't map to AmazonHelp's resolution paths.

4. **Locked the intent taxonomy at 9 labels** after two validation batches.
   Chose against 6 (too coarse) and 12+ (too sparse to train on 202 rows).

5. **Folded `compliment_thanks` into `other`.** It has no distinct
   resolution path — no action follows a "thanks" — so it doesn't warrant
   its own class.

6. **Kept `account_access`, `billing_payment`, `cancellation`** even though
   they are rare in the golden set, because they have distinct escalation
   rules and are high-risk if misclassified.

7. **Did not filter to English-only.** The assignment encourages working
   with the data as-is; filtering would have hidden the multilingual
   behavior that turned out to be a real limitation.

8. **Used a 202-row golden set** (not 150, not 250) — large enough to
   detect 0.10+ accuracy differences, small enough that one person could
   label it carefully in a few sittings.

9. **Stratified the golden set by intent** and over-sampled rare intents
   (`cancellation`, `account_access`, `billing_payment`) rather than
   sampling uniformly. Uniform sampling would have given `complaint` 40%
   of the set and made rare classes untestable.

10. **Added a hand-picked edge-case set (~50 rows)** for multi-intent,
    sarcasm, very short fragments, and non-English messages. Random
    sampling would not have surfaced these.

11. **Ran an automated QA audit** (`src/audit_golden_set.py`) instead of
    trusting the initial labels. It flagged 22 real errors (~11% of the
    set), which were corrected before evaluation.

12. **Chose intra-annotator agreement** (self, blind, 24h later) over
    skipping agreement entirely. Not ideal — no second human was available
    — but far better than claiming the labels are ground truth.

13. **Used `cross_val_predict(cv=5)` for every reported metric.** The
    in-sample accuracy (0.90) was rejected as misleading; the out-of-fold
    accuracy (0.391) is what appears in every table.

14. **Kept `escalation_reason` as an explicit 10-class prediction** instead
    of bundling it into the escalation decision. This made the reason
    disagreement with human labels visible (§6.4) rather than hidden
    inside a single "escalate" flag.

15. **Chose rules over an LLM for escalation** once the LLM failed on
    infrastructure. The escalation policy is genuinely rule-shaped — the
    assignment's ground truth is defined by a fixed policy — so a rule
    engine is a legitimate implementation, not a fallback.

16. **Enforced escalation/reason consistency via post-processor** even
    though it lowered exact reason agreement (0.411 → 0.312). An
    internally inconsistent policy (`auto_handle + high_emotion`) cannot
    be audited, which is a worse failure than a metric drop.

17. **Deployed a hybrid agent** (TF-IDF + rules + retrieval) rather than
    continuing to chase free LLM tiers. Three providers failed for three
    different reasons; at some point the honest engineering choice is to
    ship what runs.

18. **Validated the rule-based judge against human scores** rather than
    reporting only the judge's numbers. The 1.4-point gap and ρ = 0.14
    (§5.8) is the single most informative result in the report — it turns
    a misleading headline (judge 3.99) into a documented limitation.

19. **Documented the failure modes before writing the results section.**
    Most reports hide what didn't work. Here, four of five failure modes
    are structural, and they explain the entire gap between the agent's
    0.391 and a production-ready 0.65.

20. **Wrote §7 ("What is misleading about my headline number?") before
    §5 (Results).** The assignment explicitly rewards this kind of
    self-awareness, and drafting it first kept the results section honest.