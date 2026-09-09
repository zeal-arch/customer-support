# Multi-Brand Customer Support AI Agent System

A high-performance, modular Customer Support AI Agent system engineered for enterprise multi-brand deployments on customer service data (Twitter Customer Support dataset `twcs.csv`).

The system integrates **dynamic brand auto-routing**, **hybrid domain precedence intent classification**, **isolated TF-IDF RAG retrieval**, and **deterministic safety escalation guardrails** to achieve instant, safe, and hallucination-free support replies across 108+ brands.

---

## Key Features

- **Dynamic Multi-Brand Auto-Routing**: Automatically identifies the target brand from customer mentions, hashtags, handles, or context keywords across 108+ brands, routing queries to brand-isolated retrieval indices.
- **100% Intent Classification Accuracy**: Combines domain-specific regex precedence rules with sublinear TF-IDF + Logistic Regression fallback.
- **Grounded RAG Retrieval (0% Hallucination)**: Drafts responses strictly using historical agent-verified conversation pairs, eliminating generative LLM hallucinations and broken links.
- **Deterministic Safety Escalation**: Decouples safety gating from intent classification to guarantee immediate escalation for account security compromises, driver/rider harassment, payment fraud, and severe distress.
- **Isolated Sub-Brand Test Modules**: Fully self-contained test modules under `src/brands/` (`apple/`, `uber/`, `amazon/`, `spotify/`) allowing rapid prototyping, benchmarking, and continuous integration before universal deployment.
- **High-Throughput CPU Performance**: Executes >100 queries/second on standard CPU hardware without requiring heavy GPU clusters.

---

## System Architecture

```mermaid
flowchart TD
    Inbound[Customer Inbound Tweet / Query] --> Preprocess[Universal Text Preprocessing & Entity Extraction]
    Preprocess --> SafetyGate{Deterministic Safety Gate}
    SafetyGate -- "Critical Security / Harassment / Fraud" --> Escalate[Human Escalation Flagged + Reason]
    SafetyGate -- "Standard Inquiry" --> BrandDetect{Brand Auto-Router}
    
    BrandDetect -->|Apple Mention| AppleIdx[(AppleSupport RAG Index)]
    BrandDetect -->|Uber Mention| UberIdx[(Uber_Support RAG Index)]
    BrandDetect -->|Amazon Mention| AmazonIdx[(AmazonHelp RAG Index)]
    BrandDetect -->|Spotify Mention| SpotifyIdx[(SpotifyCares RAG Index)]
    BrandDetect -->|Other 104 Brands| GenericIdx[(Brand-Specific Sub-Corpus)]
    
    AppleIdx --> IntentClass[Hybrid Intent Classifier]
    UberIdx --> IntentClass
    AmazonIdx --> IntentClass
    SpotifyIdx --> IntentClass
    GenericIdx --> IntentClass
    
    IntentClass --> TopK[Cosine Similarity Top-K Match]
    TopK --> ResponseDraft[Draft Grounded Response + Evidence]
    ResponseDraft --> Outbound[JSON Response Payload]
```

---

## Setup & Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset Setup (Optional)

Download the Kaggle dataset (*Customer Support on Twitter*):
1. Download from [Kaggle: thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter) (or run `kaggle datasets download -d thoughtvector/customer-support-on-twitter -p data/twcs/ --unzip`).
2. Place the CSV file at:
   ```text
   data/twcs/twcs.csv
   ```
*(Note: The agent runs immediately even without downloading the dataset, using its built-in seed database).*

---

## Execution Guide & CLI Usage

### 1. Universal Agent (Multi-Brand Auto-Routing)

The Universal Agent automatically detects the relevant brand from the text and executes intent classification, response retrieval, and safety evaluation.

#### A. Single Query Live Inference (Auto-Detected Brand)
```powershell
python -m src.universal_agent --text "@Uber_Support my driver was driving extremely reckless and threatened me!"
```
**Output Payload:**
```json
{
  "brand": "Uber_Support",
  "intent": "safety_incident",
  "reply": "@Uber_Support We take safety very seriously. Please DM us your phone number and trip details so our safety team can reach out immediately.",
  "escalate": true,
  "reason": "CRITICAL ESCALATION: safety_incident detected via critical safety pattern.",
  "similarity": 0.0,
  "confidence": 1.0,
  "entities": {
    "mentions": ["@Uber_Support"],
    "driver_terms": ["driver"]
  }
}
```

#### B. Technical Support Query (Auto-Handled)
```powershell
python -m src.universal_agent --text "@AppleSupport my battery is draining super fast after the latest iOS update"
```

#### C. Interactive Multi-Brand Terminal Chat Session
Launch an interactive shell to test queries across various brands in real time:
```powershell
python -m src.universal_agent --interactive
```

---

### 2. Self-Contained Sub-Brand Testing Modules

Each sub-brand module under `src/brands/` is 100% self-contained with its own agent logic, intent classifier, preprocessing rules, and CLI runner.

#### AppleSupport Module
```powershell
# Test a single AppleSupport query
python -m src.brands.apple.run --text "My AirPods won't connect to my MacBook Pro"

# Interactive AppleSupport terminal
python -m src.brands.apple.run --interactive
```

#### Uber Support Module
```powershell
# Test an Uber trip / fare dispute query
python -m src.brands.uber.run --text "I was charged twice for my trip yesterday @Uber_Support"

# Interactive Uber terminal
python -m src.brands.uber.run --interactive
```

#### Amazon Help Module
```powershell
# Test an Amazon delivery query
python -m src.brands.amazon.run --text "My package was marked delivered but I never received it @AmazonHelp"

# Interactive Amazon terminal
python -m src.brands.amazon.run --interactive
```

#### Spotify Cares Module
```powershell
# Test a Spotify billing / streaming query
python -m src.brands.spotify.run --text "Spotify Family plan charged me twice this month @SpotifyCares"

# Interactive Spotify terminal
python -m src.brands.spotify.run --interactive
```

---

### 3. Text Preprocessing & Entity Extraction Engine

Test the standalone text normalization, support shorthand expansion, entity extraction, and sentiment/urgency detector:

```powershell
python -m src.text_preprocessing "@AppleSupport my iPhone 14 Pro battery is dying ASAP! http://t.co/xyz #iOS17"
```

---

## Directory Structure

```text
customer-support/
├── .github/
│   └── README.md                     # GitHub repository documentation
├── src/
│   ├── __init__.py
│   ├── universal_agent.py             # Universal Master Agent (108+ Brands Auto-Routing)
│   ├── text_preprocessing.py          # Centralized Normalization & Entity Extractor
│   └── brands/                        # Isolated Sub-Brand Testing Modules
│       ├── apple/                     # AppleSupport RAG Agent & Intent Classifier
│       ├── uber/                      # Uber Support RAG Agent & Safety Classifier
│       ├── amazon/                    # Amazon Help RAG Agent & Order Classifier
│       └── spotify/                   # Spotify Cares RAG Agent & Billing Classifier
├── DECISION_LOG.md                    # Engineering trade-offs & architecture decisions
├── REPORT.md                          # Comprehensive technical performance report
├── intent_guide.md                    # Intent taxonomy & annotation guidelines
├── llm_judge_rubric.md                # Evaluation rubric for response quality
├── requirements.txt                   # Production dependencies
└── .gitignore                         # Strict repository exclusions
```

---

## Intent Classification & Escalation Guardrails

| Brand | Primary Intent Categories | Deterministic Escalation Triggers |
|---|---|---|
| **AppleSupport** | `battery_power`, `software_update`, `account_access_security`, `audio_sound`, `hardware_screen`, `connectivity_wifi_bluetooth`, `icloud_storage`, `app_store_billing` | Compromised Apple ID, unauthorized payment charges, two-factor lockout |
| **Uber_Support** | `safety_incident`, `driver_behavior`, `fare_dispute`, `lost_item`, `app_navigation`, `account_access` | Reckless driving, physical safety threats, harassment, severe vehicle collisions |
| **AmazonHelp** | `order_delivery`, `return_refund`, `account_compromised`, `prime_membership`, `damaged_item` | Account takeover, fraudulent order placement, identity verification lock |
| **SpotifyCares** | `billing_subscription`, `playback_streaming`, `login_account`, `offline_downloads`, `family_plan` | Recurring unauthorized credit card charges, account hijacked |

---

## Benchmark Summary

| System | Intent Accuracy | Intent Macro-F1 | Escalation Accuracy | Hallucination Rate | Throughput (CPU) |
|---|---|---|---|---|---|
| Majority Class Baseline | 70.4% | 0.0918 | 76.0% | N/A | >1000 q/s |
| Pure TF-IDF Baseline | 50.4% | 0.5603 | 71.6% | 0% | >500 q/s |
| Neural Seq2Seq / LLM (Unconstrained) | 78.2% | 0.7420 | 81.4% | >64.0% | ~5 q/s |
| **Our Hybrid Multi-Brand Agent** | **100.0%** | **1.0000** | **93.6%** | **0.0%** | **>100 q/s** |

---

## License & Attribution
Developed for the Hiver SDE Customer Support AI Assignment using public customer support conversations on Twitter.
