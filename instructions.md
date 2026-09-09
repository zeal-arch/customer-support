# Instructions & Execution Guide

This document provides clear, step-by-step instructions to set up the environment, run the baseline evaluation benchmarks, execute live customer support agent inference, and reproduce all reported metrics.

---

## 1. Environment Setup

### Prerequisites
- Python 3.9+ (tested on Python 3.10, 3.11, and 3.13)
- PowerShell, Bash, or Command Prompt
- Git

### Installation
Clone the repository and install the dependencies:

```bash
git clone https://github.com/zeal-arch/customer-support.git
cd customer-support

# (Optional) Create and activate a virtual environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

## 2. Reproducing the Benchmark (Single Command)

To run all 3 systems (Trivial Baseline, Simple Baseline, and Our Agent) side-by-side against the 250-example golden evaluation set:

```bash
python -m src.run_baselines --pairs data/processed/applesupport_pairs.csv --golden evaluation/golden_set.csv --output-dir evaluation/results
```

### Benchmark Results
The command executes in under 20 seconds and outputs:

| System | Intent Accuracy | Intent Macro-F1 | Escalation Accuracy |
|---|---|---|---|
| **System 0: Trivial Baseline** (Majority class) | 0.7040 | 0.0918 | 0.7600 |
| **System 1: Simple Baseline** (Rule-based + TF-IDF) | 0.5040 | 0.5603 | 0.7160 |
| **System 2: Our Agent** (Hybrid Precedence + LR + TF-IDF) | **1.0000** | **1.0000** | **0.9360** |

#### Escalation Matrix (Our Agent)
- **TN (Auto-handled safely)**: 189
- **FP (Over-escalated)**: 1 *(near-zero false alarms)*
- **FN (Missed escalations)**: 15
- **TP (Correct escalations)**: 45

### Generated Artifacts in `evaluation/results/`
- `comparison_table.csv`: Side-by-side metric comparison.
- `metrics.json`: Detailed accuracy, macro-F1, and confusion matrices.
- `intent_reports.json`: Per-class precision, recall, and F1 reports.
- `predictions.csv`: Row-by-row predictions from all 3 systems on all 250 evaluation examples.
- `reply_review_template.csv`: Review sheet for qualitative LLM or human evaluation.

---

## 3. Running Single-Turn Live Agent Inference

To test how the agent handles a customer tweet in real time:

### Example A: Standard Technical Inquiry (Auto-handled)
```bash
python -m src.run_agent --pairs data/processed/applesupport_pairs.csv --text "my battery is dying super fast after updating to iOS 11"
```

**Output:**
```json
{
  "intent": "battery_power",
  "reply": "@593887 Let's look into that. Send us a DM and we'll go from there. https://t.co/GDrqU22YpT",
  "escalate": false,
  "reason": "The intent and historical evidence are sufficiently clear for a draft.",
  "similarity": 0.573,
  "evidence": [
    "Has anyone else updated their @115858 iPhone to iOS 11 and noticed their battery has been dying super fast...",
    "@342218 @43925 @115858 Wow, my battery drains super fast after updating and I thought it was just me",
    "#Apple @115858 why is my iPhone battery dying so fast after updating to #iOS11... #fixit please"
  ]
}
```

### Example B: Sensitive Security / Account Issue (Escalated to Human)
```bash
python -m src.run_agent --pairs data/processed/applesupport_pairs.csv --text "I think someone hacked my Apple ID and locked me out"
```

**Output:**
```json
{
  "intent": "account_access_security",
  "reply": "@AppleSupport DM us using the link below. We'll look into this with you from there. https://t.co/GDrqU22YpT",
  "escalate": true,
  "reason": "Sensitive account or payment issue requires human review.",
  "similarity": 0.542,
  "evidence": [...]
}
```

---

## 4. Pipeline Modules & Architecture

- **`src/intent_classifier.py`**:
  Hybrid architecture combining high-precision domain precedence rules with sublinear TF-IDF + Logistic Regression fallback.
- **`src/agent.py`**:
  `SupportAgent` retrieval engine: vectorizes customer text, searches historical AppleSupport corpus, and gates responses with safety escalation logic.
- **`src/run_baselines.py`**:
  Evaluator that compares the 3 systems with leak-free separation (eval examples held out from the retrieval index).
- **`src/text_preprocessing.py`**:
  Fast Twitter-specific normalizer: expands support shorthand, handles punctuation/mentions, preserves domain keywords (`iOS`, `macOS`).

---

## 5. Comparative Experiments & Verification

To run additional empirical verification experiments:

1. **Compare Text Preprocessing Pipelines**:
   ```bash
   python -m src.compare_preprocessing --input data/processed/applesupport_pairs.csv --output evaluation/preprocessing_comparison.csv
   ```

2. **Benchmark on Banking77 Intent Dataset**:
   ```bash
   python -m src.banking77_experiment --max-train 1000 --max-test 500
   ```

3. **Run 3-System Baseline Comparison**:
   ```bash
   python -m src.run_baselines
   ```
