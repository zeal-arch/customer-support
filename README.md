# Brand Support Agent

This repository builds and evaluates a customer-support agent for one brand from the Customer Support on Twitter dataset.

The agent will:

1. classify an incoming customer message into an intent,
2. draft a response using similar historical conversations, and
3. decide whether to answer automatically or escalate to a human.

## Current status

The pipeline is complete and submission-ready. Headline results on 250 labeled examples:

| System | Intent Accuracy | Intent Macro-F1 | Escalation Accuracy |
|---|---|---|---|
| Trivial baseline | 0.732 | 0.094 | 0.748 |
| Simple baseline (rule-based + TF-IDF) | 0.556 | 0.712 | 0.760 |
| **Our Agent (Safety-Gated LR + TF-IDF)** | **0.924** | **0.783** | **0.864** |

*Safety highlights*: **FN = 2** (only 2 missed escalations out of 63, 96.8% recall) with Intent Macro-F1 of **0.783**. An optional cost-optimized mode (`--conf-threshold 0.45`) yields Escalation Accuracy 0.932.

Brand: **AppleSupport** — 106,648 customer/reply pairs extracted from the Kaggle dataset.


## Setup

Create a virtual environment and install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download the Kaggle dataset `thoughtvector/customer-support-on-twitter`. The CSV may be placed at:

```text
data/raw/twcs/twcs.csv
```

The expected columns are the standard dataset columns:

```text
tweet_id, author_id, inbound, created_at, text,
response_tweet_id, in_response_to_tweet_id
```

## First command

After placing the dataset in `data/raw/twcs/twcs.csv`, run:

```powershell
python -m src.inspect_data --input data/raw/twcs/twcs.csv
```

This streams the full file, reports inbound/outbound balance, and ranks likely brand support accounts. The raw file does not contain a dependable brand column, so brand selection is based on support-account handles and conversation quality.

## One-command reproduction (< 15 minutes)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Step 1 — finalize golden set labels
python -m src.label_golden_set

# Step 2 — run all three systems and produce metrics
python -m src.run_baselines `
  --pairs data/processed/applesupport_pairs.csv `
  --golden evaluation/golden_set.csv `
  --output-dir evaluation/results

# Step 3 — smoke-test the agent on a single message
python -m src.run_agent `
  --pairs data/processed/applesupport_pairs.csv `
  --text "My iPhone battery drains after the latest update"
```

Results land in `evaluation/results/`:
- `metrics.json` — intent accuracy, macro-F1, escalation accuracy per system
- `comparison_table.csv` — one-line summary of all three systems
- `predictions.csv` — per-row predictions for all 250 examples
- `reply_review_template.csv` — ready for LLM-as-judge scoring

## Milestones

- [x] Inspect dataset and select brand (AppleSupport, 106k replies)
- [x] Build clean brand-specific conversation table
- [x] Define 11 intents from development sample
- [x] Create 250-example golden evaluation set
- [x] Implement intent classifier (LR), retrieval, response drafting, and escalation
- [x] Compare against trivial and simple baselines
- [ ] Run human/LLM judge agreement checks (see `evaluation/llm_judge_rubric.md`)
- [x] Write report and decision log

## Current brand choice

The initial candidate is `AppleSupport`. The raw-data scan found 106,860 outbound replies for this account. This is a working choice, not a claim that AppleSupport is the only valid option.

To extract customer/reply pairs:

```powershell
python -m src.extract_brand `
  --input data/raw/twcs/twcs.csv `
  --brand AppleSupport `
  --output data/processed/applesupport_pairs.csv
```

To explore recurring topics before defining intents:

```powershell
python -m src.profile_brand --input data/processed/applesupport_pairs.csv
```

The initial intent definitions are documented in `evaluation/intent_guide.md`. Create a human-labeling sheet with:

```powershell
python -m src.create_labeling_sample `
  --input data/processed/applesupport_pairs.csv `
  --output evaluation/golden_set.csv `
  --size 250
```

## How the supplied notebooks are used

The supplied notebooks are references rather than the final pipeline:

- The chatbot notebook contributes the TF-IDF + Logistic Regression idea for a simple intent baseline. Its original `author_id` target is not used because author identity is not customer intent.
- The text-preprocessing notebook contributes normalization ideas. The implementation in `src/text_preprocessing.py` is intentionally conservative: it replaces URLs and mentions but preserves emojis, punctuation, product names, and complaint language.
- The starter notebook contributes basic schema and exploratory-analysis patterns.

Run the current transparent retrieval baseline on a single message:

```powershell
python -m src.run_agent `
  --pairs data/processed/applesupport_pairs.csv `
  --text "My iPhone battery drains after the latest update"
```

The optional semantic backend uses `sentence-transformers/all-MiniLM-L6-v2`:

```powershell
python -m src.run_agent `
  --pairs data/processed/applesupport_pairs.csv `
  --backend minilm `
  --max-reference 30000 `
  --text "My iPhone battery drains after the latest update"
```

Create draft labels to speed up golden-set review:

```powershell
python -m src.suggest_labels `
  --input evaluation/golden_set.csv `
  --output evaluation/golden_set_draft.csv
```

After reviewing and filling the official gold columns, run the leakage-safe evaluation:

```powershell
python -m src.evaluate_agent `
  --pairs data/processed/applesupport_pairs.csv `
  --golden evaluation/golden_set.csv `
  --output-dir evaluation/results
```

Compare raw text, our conservative preprocessing, and the notebook-style pipeline:

```powershell
python -m src.compare_preprocessing `
  --input data/processed/applesupport_pairs.csv `
  --output evaluation/preprocessing_comparison.csv
```

This comparison measures how close the retrieved historical messages are. It is a smoke test for retrieval, not the final assignment metric; final intent and reply-quality scores require the labeled golden set.

## Optional Banking77 experiment

Banking77 is a separate, banking-domain intent dataset. It is useful for comparing preprocessing and classifier choices because it already includes intent labels, but it must not be presented as AppleSupport performance.

Install the optional Hugging Face dependencies if needed:

```powershell
python -m pip install -r requirements-optional.txt
```

Run the intent benchmark:

```powershell
python -m src.banking77_experiment `
  --output evaluation/banking77_preprocessing_comparison.csv
```

Recommended Hugging Face components:

- `sentence-transformers/all-MiniLM-L6-v2` for lightweight semantic retrieval.
- A small DistilBERT classifier fine-tuned on our AppleSupport labels for the final intent model.
- A Banking77-fine-tuned classifier only for the optional Banking77 benchmark, because its 77 labels are banking-specific.

## Reproducibility target

The final README will include one command that reproduces the headline evaluation results on a checked-in sample or a documented downloaded subset in under 15 minutes.
