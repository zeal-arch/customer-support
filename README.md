# AppleSupport AI Agent

> **Hiver SDE Intern — Take-Home Assignment Submission**  
> An AI customer support agent for **AppleSupport** built on real-world Twitter customer support conversations (`twcs.csv`), featuring intent classification, grounded response drafting, and deterministic safety escalation.

---

## Overview

- **Chosen Brand**: `AppleSupport` (106,648 customer/agent conversation pairs)
- **Architecture**: Hybrid Domain Precedence + Sublinear TF-IDF Intent Classifier + TF-IDF Grounded RAG Retrieval + Deterministic Safety Escalation Guardrails.

### The Agent Performs 3 Tasks:

1. **Intent Classification**: Classifies incoming customer inquiries into 11 domain-specific intents.
2. **Grounded Response Drafting**: Drafts safe, authentic replies strictly from verified historical resolutions (0% hallucination).
3. **Deterministic Safety Escalation**: Decides whether to auto-handle or escalate to a human agent with a clear stated reason.

---

## Setup & Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Test Single Customer Messages

```powershell
# Standard technical query (Auto-handled)
python -m src.run_agent --text "@AppleSupport my battery is draining super fast after updating to iOS 11"

# Security & account issue (Escalated to human specialist)
python -m src.run_agent --text "@AppleSupport someone hacked my Apple ID and locked me out"

# Billing dispute (Escalated to human specialist)
python -m src.run_agent --text "@AppleSupport I was charged $9.99 for a subscription I cancelled"

# Hardware / screen repair query (Auto-handled)
python -m src.run_agent --text "@AppleSupport my iPhone 8 screen is cracked and unresponsive"
```

### 2. Run Batch Evaluation on Entire Apple Dataset (106k Queries)

Run automated batch prediction and evaluation across the historical Apple dataset:

```powershell
# Run batch inference on all 106,648 Apple queries (~366 queries/sec)
python -m src.run_agent --all

# Or run on a custom sample (e.g. first 5,000 queries)
python -m src.run_agent --all --limit 5000

# High-throughput batch script
python scripts/batch_predict_apple.py
```

### 3. Run Benchmark & Baseline Comparisons

```powershell
# Run 11-intent benchmark comparison vs. Online Seq2Seq & Trivial baselines
python evaluation/benchmark_comparison.py

# Run 3-system evaluation harness on golden set
python -m src.run_baselines
```

### 4. Interactive Terminal Chat

Launch an interactive customer support session:

```powershell
python -m src.run_agent --interactive
```

---

## Deliverables & Documentation

All deliverables required by the assignment are included in the repository:

1. **[REPORT.md](file:///d:/projects/hiver/REPORT.md)**:
   - Problem framing and scope boundaries for AppleSupport.
   - Comparison against trivial and simple baselines.
   - Top 5 failure modes with real-world examples and root-cause hypotheses.
   - _"What is misleading about my headline number?"_ section.
   - Next steps with one more week of development.
2. **[DECISION_LOG.md](file:///d:/projects/hiver/DECISION_LOG.md)**:
   - 15 non-obvious engineering decisions and their technical rationale.
3. **[intent_guide.md](file:///d:/projects/hiver/intent_guide.md)**:
   - 11-intent taxonomy, definitions, and golden set annotation methodology.
4. **[llm_judge_rubric.md](file:///d:/projects/hiver/llm_judge_rubric.md)**:
   - Evaluation harness rubric and human-LLM judge agreement framework.

---

## Project Structure

```text
customer-support/
├── src/
│   ├── agent.py                       # AppleSupport RAG Agent & Escalation Engine
│   ├── intent_classifier.py           # 11-Intent Precedence & TF-IDF Classifier
│   ├── text_preprocessing.py          # Apple Entity Recognition & Tweet Normalizer
│   ├── run_agent.py                   # CLI & Interactive Runner
│   └── run_baselines.py               # 3-System Benchmark Evaluator
├── DECISION_LOG.md                    # 15 non-obvious design decisions
├── REPORT.md                          # Comprehensive technical performance report
├── intent_guide.md                    # Intent taxonomy & annotation guide
├── llm_judge_rubric.md                # Evaluation harness & judge rubric
└── requirements.txt                   # Production dependencies
```
