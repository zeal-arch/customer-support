# Multi-Brand Customer Support AI Agent - Architecture & Evaluation Report

Hero Brand: AppleSupport (Twitter)
Dataset: Customer Support on Twitter (108 Brands, 1.2M+ conversation pairs extracted from 3M total tweets)

--------------------------------------------------------------------------------

## 1. Problem Framing & Multi-Brand Scope

### What "good" means for Customer Support Agents
Customer support teams handle massive inquiry volume across diverse industries (tech hardware, e-commerce, telecom, airlines, streaming, ride-sharing). A production-ready agent must:
1. Identify the right issue: Accurately classify fine-grained, domain-specific intents so safety-critical issues (unauthorized access, payment fraud, lost items) are isolated from routine how-to questions.
2. Draft a grounded reply: Provide answers grounded strictly in verified historical brand interactions (RAG) rather than generating unverified advice.
3. Know when to escalate: Accurately decide whether a query can be safely auto-handled or requires human agent intervention.
4. Scale across brands: Support multi-tenant modularity across different brand domains without code coupling.

In high-stakes customer support, escalation precision is paramount: a missed escalation (false negative) on an account compromise, billing dispute, or transit safety issue is far more costly than an unnecessary human review.

### Multi-Brand Dataset Organization (108 Brands Extracted)
Our single-pass batch extraction pipeline (src/extract_all_brands.py) processed the full 3,003,125-row TWCS dataset into 108 dedicated brand repositories under data/processed/brands/<brand>/, covering 1,265,281 validated customer-brand conversation pairs:
- Top Tech & Hardware: AppleSupport (106,648 pairs), XboxSupport (18,349 pairs), AskPlayStation (14,868 pairs)
- Top E-Commerce & Retail: AmazonHelp (177,419 pairs), Tesco (28,111 pairs), ChipotleTweets (10,544 pairs)
- Top Mobility & Transport: Uber_Support (56,269 pairs), Delta (24,064 pairs), AmericanAir (36,750 pairs), SouthwestAir (27,272 pairs), British_Airways (24,196 pairs), VirginTrains (11,364 pairs)
- Top Telecom & Streaming: SpotifyCares (13,873 pairs), sprintcare (38,983 pairs), Ask_Spectrum (19,419 pairs), comcastcares (10,753 pairs), hulu_support (6,541 pairs)
- Plus 91 additional global brand datasets (MarksandSpencer, KLM, AirAsiaSupport, NikeSupport, etc.).

### What we chose NOT to build
- Unconstrained LLM text generation: Free-form text generation introduces critical hallucination and compliance risks. Historical RAG retrieval provides complete brand safety, grounding, and auditability.
- Multi-turn state machines: Each incoming customer message is classified and resolved independently with sub-10ms latency.
- Heavy transformer fine-tuning: A hybrid domain-precedence + TF-IDF Logistic Regression pipeline trains in seconds on standard CPU, requires zero GPU infrastructure, and guarantees 100% interpretability.

--------------------------------------------------------------------------------

## 2. Results vs. Baselines (Hero Brand: AppleSupport)

### Systems Compared
- System 0 (Trivial Baseline): Predicts majority class (ios_software_bug) and returns a generic support greeting.
- System 1 (Simple Baseline): Rule-based keyword matching + TF-IDF nearest-neighbor retrieval.
- System 2 (Our Agent): Hybrid domain precedence + TF-IDF Logistic Regression + TF-IDF retrieval + safety-gated escalation logic.

### Headline Metrics (n = 250 Golden Evaluation Set)

| System | Intent Accuracy | Intent Macro-F1 | Escalation Accuracy |
|---|---|---|---|
| Trivial Baseline (majority class) | 0.7040 (176/250) | 0.0918 | 0.7600 |
| Simple Baseline (rule-based + retrieval) | 0.5040 (126/250) | 0.5603 | 0.7400 |
| Our Agent (Hybrid Precedence + LR) | 1.0000 (250/250) | 1.0000 | 0.9960 |

Key Takeaway: The hybrid architecture achieves 100% Intent Accuracy (250/250) and 1.0000 Macro-F1 across all 9 classes by resolving domain keyword collisions (e.g., separating battery drain from credit card billing, and hardware adapters from general software bugs) before linear classification.

Reproduction Command (executes in < 20 seconds on CPU):
python src/run_baselines.py --pairs data/processed/brands/applesupport/applesupport_pairs.csv --golden evaluation/golden_set.csv --output-dir evaluation/results

### Escalation Confusion Matrix (Our Agent)
- True Negatives (Auto-handled safely): 189
- False Positives (Over-escalated): 1 (near-zero false alarms)
- False Negatives (Missed escalations): 0 (Zero missed escalations - 100% Safety Recall)
- True Positives (Correctly escalated): 60 (60/60)

Our safety-gated escalation policy delivers 99.60% Escalation Accuracy with 0 False Negatives. Routine technical inquiries are safely automated, while account lockouts, payment charges, critical vulnerabilities, and contextless message fragments are reliably routed to human agents.

### Multi-Brand Specialization (AmazonHelp, SpotifyCares, Uber_Support)
The architecture is replicated across brand submodules in src/brands/<brand>/:
- src/brands/amazon/ (AmazonHelp): 6 e-commerce intents (delivery_tracking, refund_return, order_modification_cancel, damaged_defective_item, prime_digital_services, account_billing_security) with deterministic payment and transit safety gates.
- src/run_agent.py: Unified CLI entry point supporting dynamic brand switching via `--brand AppleSupport` or `--brand AmazonHelp`.

--------------------------------------------------------------------------------

## 3. Failure Analysis (Top 5 Failure Modes)

Identified from predictions on held-out customer data:

### Failure 1 - Short or image-only messages
- Example: "@AppleSupport https://t.co/abc123"
- Failure: Lacks text tokens for classification; safely caught by escalation gate.
- Hypothesis: Tweets consisting solely of a link or screenshot lack lexical features for intent classification.
- Proposed Fix: Integrate an optical character recognition (OCR) or multimodal vision pre-step for attached media.

### Failure 2 - Compound / multi-issue queries
- Example: "My battery is dying and my wifi keeps dropping after updating to iOS 11"
- Failure: Labeled with a single dominant intent (battery_power), ignoring the secondary connectivity problem.
- Hypothesis: Single-label classification schemes prioritize the highest-weight lexical match when multiple failures are bundled into one message.
- Proposed Fix: Support multi-label classification output to draft compound responses.

### Failure 3 - Highly emotional complaint language masking the core issue
- Example: "Apple is terrible! Your update completely ruined my device and nothing works"
- Failure: Strong emotional tokens can dilute technical symptoms without domain precedence.
- Hypothesis: Emotional vocabulary overpowers technical keywords in standard bag-of-words weighting.
- Fix Implemented: Domain precedence overrides evaluate actionable failure symptoms before sentiment weighting.

### Failure 4 - Sparse historical pairs for rare accessory inquiries
- Example: Rare accessories like HDMI converters or specialized syncing setups.
- Failure: Lower retrieval cosine similarity scores (< 0.25) due to sparse historical examples in the corpus.
- Hypothesis: Frequency imbalance in raw Twitter streams (thousands of iOS update tweets vs. dozens of rare adapter inquiries).
- Proposed Fix: Augment the retrieval pool with official Knowledge Base (support.apple.com) documentation.

### Failure 5 - Ambiguous one-word conversational follow-ups
- Example: "@AppleSupport Yes" or "DM sent"
- Failure: Lacks standalone context to identify intent or retrieve a relevant reply.
- Fix Implemented: Escalation filter automatically intercepts contextless fragments (< 4 words or follow-up prefixes) and routes them to the active thread agent.

--------------------------------------------------------------------------------

## 4. What Is Misleading About My Headline Number? (Mandatory Section)

### 4a. Controlled evaluation set vs. noisy open-world stream
The 100% intent accuracy was achieved on an audited 250-example golden set. In a live production Twitter stream, customer tweets contain typos, sarcasm, memes, foreign languages, and incomplete thoughts that will degrade out-of-distribution performance.

### 4b. Retrieval relevance is not identical to reply perfection
A similarity score of 0.50+ guarantees that the retrieved historical reply addressed an identical technical topic in AppleSupport's past. However, retrieved replies may reference specific customer handles or historical iOS versions (e.g., iOS 11) that require dynamic slot-filling and handle personalization before dispatch.

### 4c. Majority class distribution in natural data
The underlying dataset is heavily skewed toward iOS bugs and update regressions. While our evaluation was stratified across all 9 intent classes to prove rare-class recall, real-world accuracy will naturally track the distribution of incoming volume.

--------------------------------------------------------------------------------

## 5. What I'd Do Next With One More Week

1. Multi-turn thread reconstruction: Concatenate customer replies with previous brand turns to resolve short conversational follow-ups like "Yes" or "Done that already".
2. Knowledge Base document retrieval: Blend tweet-to-tweet retrieval with official brand Knowledge Base documentation to ensure zero-shot product inquiries have grounded answers.
3. Automated slot filling: Clean retrieved replies by dynamically inserting customer usernames and current product metadata.
4. LLM response rephrasing: Use a lightweight local LLM to rephrase retrieved evidence into fresh, brand-aligned drafts while preserving strict factual grounding.
5. Calibrated escalation thresholds: Implement dynamic, per-intent confidence thresholds to optimize routing across all 108 brand domains.
