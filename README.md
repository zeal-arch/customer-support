# AI Customer Support Agent

> **Hiver SDE Intern — Take-Home Assignment Submission**  
> An AI customer support agent built on real-world Twitter customer support conversations (`twcs.csv`), featuring intent classification, grounded response drafting, and deterministic safety escalation.

---

## Overview

- **Primary Brand**: `AppleSupport` (106,648 historical customer/agent conversation pairs)
- **Universal Multi-Brand Engine**: Also supports `Uber_Support`, `AmazonHelp`, `SpotifyCares`, and 100+ other brands with dynamic auto-routing.

### The Agent Performs 3 Tasks:

1. **Intent Classification**: Classifies incoming customer inquiries into domain-specific intents.
2. **Grounded Response Drafting**: Drafts safe, authentic replies strictly from verified historical resolutions (0% hallucination).
3. **Deterministic Safety Escalation**: Decides whether to auto-handle or escalate to a human agent with a clear reason.

---

## Headline Evaluation Results

Evaluated against a 250-example hand-labelled golden dataset across three systems:

| System                                       | Intent Accuracy | Intent Macro-F1 | Escalation Accuracy | Hallucination Rate |
| -------------------------------------------- | --------------- | --------------- | ------------------- | ------------------ |
| **Trivial Baseline** (Majority Class)        | 70.40%          | 0.0918          | 76.00%              | N/A                |
| **Simple Baseline** (Rule-based + TF-IDF)    | 50.40%          | 0.5603          | 71.60%              | 0.0%               |
| **Our Agent** (Hybrid Precedence + LR + RAG) | **100.00%**     | **1.0000**      | **93.60%**          | **0.0%**           |

- **Escalation Breakdown**: True Negatives (Auto-handled safely) = 189, False Positives (Over-escalated) = 1, False Negatives = 15, True Positives (Correctly escalated) = 45.

---

## Setup & Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset Setup

Download the Kaggle dataset (_Customer Support on Twitter_):

1. Download from [Kaggle: thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (or run `kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/twcs/ --unzip`).
2. Place the CSV file at:
   ```text
   data/twcs/twcs.csv
   ```

---

## How to Run

### Run the Agent on Any Customer Message

```powershell
# Standard technical query (Auto-handled)
python -m src.universal_agent --text "@AppleSupport my battery is draining super fast after the latest iOS update"

# Security & account issue (Escalated to human)
python -m src.universal_agent --text "@AppleSupport someone hacked my Apple ID and locked me out"

# Multi-brand query (Auto-routed to Uber)
python -m src.universal_agent --text "@Uber_Support my driver was driving recklessly and threatened me!"
```

### 2. Interactive Terminal Chat

```powershell
python -m src.universal_agent --interactive
```

### 3. Isolated Sub-Brand Test Modules

```powershell
# AppleSupport
python -m src.brands.apple.run --text "My AirPods won't connect to my MacBook Pro"

# Uber Support
python -m src.brands.uber.run --text "I was charged twice for my trip yesterday @Uber_Support"

# Amazon Help
python -m src.brands.amazon.run --text "My package was marked delivered but never arrived @AmazonHelp"

# Spotify Cares
python -m src.brands.spotify.run --text "Spotify Family plan charged me twice this month @SpotifyCares"
```

---

## Deliverables & Documentation

All deliverables required by the assignment are included in the repository:

1. **[REPORT.md](file:///d:/projects/hiver/REPORT.md)**:
   - Problem framing and scope boundaries.
   - Comparison against trivial and simple baselines.
   - Top 5 failure modes with real-world examples and root-cause hypotheses.
   - _"What is misleading about my headline number?"_ section.
   - Next steps with one more week of development.
2. **[DECISION_LOG.md](file:///d:/projects/hiver/DECISION_LOG.md)**:
   - 15 non-obvious engineering decisions and their technical rationale.
3. **[intent_guide.md](file:///d:/projects/hiver/intent_guide.md)**:
   - Intent taxonomy, definitions, and golden set annotation methodology.
4. **[llm_judge_rubric.md](file:///d:/projects/hiver/llm_judge_rubric.md)**:
   - Evaluation harness rubric and human-LLM judge agreement framework.

---

## Project Structure

```text
customer-support/
├── src/
│   ├── universal_agent.py             # Universal Master Agent (108+ Brands Auto-Routing)
│   ├── text_preprocessing.py          # Central Text Normalization & Entity Extractor
│   └── brands/                        # Self-contained Sub-Brand Test Modules
│       ├── apple/                     # AppleSupport RAG Agent & Intent Classifier
│       ├── uber/                      # Uber Support RAG Agent & Safety Classifier
│       ├── amazon/                    # Amazon Help RAG Agent & Order Classifier
│       └── spotify/                   # Spotify Cares RAG Agent & Billing Classifier
├── DECISION_LOG.md                    # 15 non-obvious design decisions
├── REPORT.md                          # Comprehensive technical performance report
├── intent_guide.md                    # Intent taxonomy & annotation guide
├── llm_judge_rubric.md                # Evaluation harness & judge rubric
└── requirements.txt                   # Production dependencies
```
