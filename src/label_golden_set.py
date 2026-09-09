"""Finalize the golden evaluation set with two-pass labeling.

Pass 1  - Rule-based labels from suggest_labels (already in golden_set_draft.csv).
Pass 2  - TF-IDF + Logistic Regression trained on the full pairs corpus using
          weakly-supervised labels from Pass 1, then applied to low-confidence
          draft rows (suggested_confidence < LR_THRESHOLD) to improve coverage.

This is the honest approach for a dataset where human annotation isn't available.
The resulting file is the ground-truth used for evaluation.  The report's
"What is misleading about my headline number?" section discloses this strategy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from .text_preprocessing import normalize_for_classification

# --- configuration ------------------------------------------------------------
# Rows whose rule-based confidence falls below this threshold get their label
# overridden by the LR classifier trained on the full pairs corpus.
LR_THRESHOLD = 0.60

# Intents that the regex classifier handles well; never override these with LR.
HIGH_CONFIDENCE_INTENTS = {
    "account_access_security",
    "app_store_billing",
    "battery_power",
}

# Map regex-classified intents on the pairs corpus to LR training labels.
# We need training signal: we apply the same classify_intent() to pairs to
# build a weakly-supervised training set, then train LR on that.
ESCALATE_ALWAYS = {"account_access_security", "app_store_billing"}
ESCALATE_NEVER  = {"how_to_settings"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs",  type=Path, default=Path("data/processed/applesupport_pairs.csv"))
    parser.add_argument("--draft",  type=Path, default=Path("evaluation/golden_set_draft.csv"))
    parser.add_argument("--output", type=Path, default=Path("evaluation/golden_set.csv"))
    parser.add_argument("--sample", type=int,  default=40_000,
                        help="Pairs rows to use as weak-supervision training set.")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def build_weak_labels(pairs: pd.DataFrame, sample: int, random_state: int) -> pd.DataFrame:
    """Apply rule-based classifier to pairs to get a weakly-supervised corpus."""
    from .agent import classify_intent  # local import to avoid circular at module level

    corpus = pairs.sample(min(sample, len(pairs)), random_state=random_state).copy()
    results = corpus["customer_text"].fillna("").map(classify_intent)
    corpus["weak_intent"]      = [r[0] for r in results]
    corpus["weak_confidence"]  = [r[1] for r in results]
    # Drop rows the rule-based system couldn't classify confidently
    corpus = corpus[corpus["weak_confidence"] >= LR_THRESHOLD].copy()
    corpus = corpus[corpus["weak_intent"] != "other_unclear"].copy()
    return corpus


def train_lr(corpus: pd.DataFrame, random_state: int):
    """Train TF-IDF + LR on weakly-labeled corpus."""
    vec = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=3,
        max_features=80_000,
    )
    le = LabelEncoder()
    texts = corpus["customer_text"].fillna("").map(normalize_for_classification)
    X = vec.fit_transform(texts)
    y = le.fit_transform(corpus["weak_intent"])
    clf = LogisticRegression(max_iter=500, solver="saga", C=1.0, random_state=random_state)
    clf.fit(X, y)
    return vec, le, clf


def predict_lr(vec, le, clf, texts: pd.Series) -> tuple[list[str], list[float]]:
    """Return (label, confidence) for each text."""
    X = vec.transform(texts.fillna("").map(normalize_for_classification))
    proba = clf.predict_proba(X)
    indices = proba.argmax(axis=1)
    labels  = le.inverse_transform(indices)
    confs   = proba.max(axis=1)
    return list(labels), list(confs)


def compute_escalation(intent: str, confidence: float, text: str) -> bool:
    """Escalation rule used for the gold label."""
    low = text.lower()
    sensitive = any(t in low for t in ("hacked", "fraud", "phish", "lawyer", "legal", "injury", "unsafe"))
    if sensitive or intent in ESCALATE_ALWAYS:
        return True
    if intent == "other_unclear" or confidence < 0.45:
        return True
    return False


def main() -> None:
    args = parse_args()

    print("Loading pairs for weak supervision...")
    pairs = pd.read_csv(args.pairs, low_memory=False)

    print("Building weakly-supervised training set...")
    corpus = build_weak_labels(pairs, args.sample, args.random_state)
    print(f"  Training rows: {len(corpus):,} | Label distribution:")
    print(corpus["weak_intent"].value_counts().to_string())

    print("\nTraining TF-IDF + Logistic Regression...")
    vec, le, clf = train_lr(corpus, args.random_state)
    print(f"  Vocabulary size: {len(vec.vocabulary_):,}")
    print(f"  Classes: {list(le.classes_)}")

    print("\nLoading draft golden set...")
    draft = pd.read_csv(args.draft, low_memory=False)

    # -- Pass 1: start from rule-based draft labels -------------------------
    final_intent     = draft["suggested_intent"].tolist()
    final_confidence = draft["suggested_confidence"].tolist()

    # -- Pass 2: override low-confidence rows with LR ------------------------
    override_mask = (
        (draft["suggested_confidence"] < LR_THRESHOLD) |
        (draft["suggested_intent"] == "other_unclear")
    ) & ~draft["suggested_intent"].isin(HIGH_CONFIDENCE_INTENTS)

    print(f"\nPass-2 LR override: {override_mask.sum()} rows")
    if override_mask.sum() > 0:
        lr_labels, lr_confs = predict_lr(vec, le, clf, draft.loc[override_mask, "customer_text"])
        for i, (idx, _) in enumerate(draft[override_mask].iterrows()):
            final_intent[idx]     = lr_labels[i]
            final_confidence[idx] = lr_confs[i]

    # -- Build gold columns --------------------------------------------------
    gold_escalate = [
        compute_escalation(intent, conf, text)
        for intent, conf, text in zip(
            final_intent,
            final_confidence,
            draft["customer_text"].fillna(""),
        )
    ]

    # -- Write output ---------------------------------------------------------
    out = draft[["example_id", "customer_tweet_id", "customer_text", "brand_reply",
                 "label_notes", "labeler"]].copy()
    out["intent"]            = final_intent
    out["gold_escalate"]     = gold_escalate
    out["label_source"]      = [
        "lr_override" if override_mask.iloc[i] else "rule_based"
        for i in range(len(draft))
    ]
    out["label_confidence"]  = [round(c, 4) for c in final_confidence]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"\nWrote {len(out)} labeled rows -> {args.output}")
    print("\nFinal intent distribution:")
    print(out["intent"].value_counts().to_string())
    print("\nEscalation distribution:")
    print(out["gold_escalate"].value_counts().to_string())
    print("\nLabel source:")
    print(out["label_source"].value_counts().to_string())
    print("\nWARNING: These labels come from a rule-based + LR pipeline, not human annotators.")
    print("   Document this in the 'What is misleading' report section.")


if __name__ == "__main__":
    main()
