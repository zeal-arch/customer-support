"""Compare three systems on the golden evaluation set.

System 0 - Trivial baseline  : always predict majority class, return a generic reply.
System 1 - Simple baseline   : TF-IDF similarity retrieval + rule-based intent classifier.
System 2 - Our agent         : LR intent classifier + TF-IDF retrieval + escalation logic.

Leakage prevention: golden-set tweet IDs are removed from the reference pool
before any retrieval, so the agent cannot return the exact golden example as evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .agent import SupportAgent, classify_intent
from .intent_classifier import build_from_pairs
from .text_preprocessing import normalize_for_classification


# --- helpers -----------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs",      type=Path, required=True)
    parser.add_argument("--golden",     type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--sample",     type=int,  default=40_000,
                        help="Pairs rows for LR training (weak supervision).")
    parser.add_argument("--conf-threshold", type=float, default=0.55,
                        help="Confidence threshold for human escalation (safety gate).")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def parse_gold_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_and_validate_golden(path: Path) -> pd.DataFrame:
    golden = pd.read_csv(path, low_memory=False)
    required = {"customer_tweet_id", "customer_text", "intent", "gold_escalate"}
    missing = required - set(golden.columns)
    if missing:
        raise ValueError(f"Golden set missing columns: {sorted(missing)}")
    labels = golden["intent"].fillna("").astype(str).str.strip()
    escalation = golden["gold_escalate"].fillna("").astype(str).str.strip()
    if (labels == "").any() or (escalation == "").any():
        raise ValueError(
            "Golden labels are incomplete. Run: python -m src.label_golden_set"
        )
    return golden


def build_reference(pairs: pd.DataFrame, held_out_ids: set[str]) -> pd.DataFrame:
    return pairs[~pairs["customer_tweet_id"].astype(str).isin(held_out_ids)].copy()


# --- System 0: Trivial --------------------------------------------------------

GENERIC_REPLY = (
    "Hi! Thanks for reaching out to Apple Support. "
    "Please visit support.apple.com or contact us at 1-800-MY-APPLE "
    "for further assistance with your issue."
)


def run_trivial(golden: pd.DataFrame, majority_class: str) -> pd.DataFrame:
    rows = []
    for row in golden.itertuples(index=False):
        rows.append({
            "example_id":        getattr(row, "example_id", ""),
            "customer_tweet_id": row.customer_tweet_id,
            "customer_text":     row.customer_text,
            "gold_intent":       row.intent,
            "predicted_intent":  majority_class,
            "gold_escalate":     parse_gold_bool(row.gold_escalate),
            "predicted_escalate": False,
            "similarity":        0.0,
            "draft_reply":       GENERIC_REPLY,
            "reason":            "Trivial baseline - always majority class.",
            "evidence_1":        "",
            "system":            "trivial",
        })
    return pd.DataFrame(rows)


# --- System 1: Simple (rule-based intent + TF-IDF retrieval) -----------------

def run_simple(golden: pd.DataFrame, agent: SupportAgent) -> pd.DataFrame:
    rows = []
    for row in golden.itertuples(index=False):
        result = agent.predict(row.customer_text)
        rows.append({
            "example_id":        getattr(row, "example_id", ""),
            "customer_tweet_id": row.customer_tweet_id,
            "customer_text":     row.customer_text,
            "gold_intent":       row.intent,
            "predicted_intent":  result.intent,
            "gold_escalate":     parse_gold_bool(row.gold_escalate),
            "predicted_escalate": result.escalate,
            "similarity":        result.similarity,
            "draft_reply":       result.reply,
            "reason":            result.reason,
            "evidence_1":        result.evidence[0] if result.evidence else "",
            "system":            "simple",
        })
    return pd.DataFrame(rows)


# --- System 2: Our Agent (LR intent + TF-IDF retrieval) ----------------------

def run_our_agent(
    golden: pd.DataFrame,
    agent: SupportAgent,
    lr_clf,
    conf_threshold: float = 0.55,
) -> pd.DataFrame:
    """Same retrieval as System 1 but uses LR for intent classification and safety-gated escalation."""
    rows = []
    for row in golden.itertuples(index=False):
        # LR intent prediction
        lr_intent, lr_conf = lr_clf.predict(row.customer_text)

        # Retrieval (same TF-IDF as simple baseline)
        result = agent.predict(row.customer_text)

        # Safety-gated escalation (Option C):
        risky   = lr_intent in {"account_access_security", "app_store_billing"}
        unclear = lr_intent == "other_unclear" or lr_conf < conf_threshold
        weak    = result.similarity < 0.20
        escalate = risky or unclear or weak

        if risky:
            reason = "Sensitive account or payment issue - LR classifier flagged."
        elif unclear:
            reason = f"LR confidence low ({lr_conf:.2f}) or intent unclear."
        elif weak:
            reason = "No sufficiently similar historical example found."
        else:
            reason = f"LR intent={lr_intent} (conf={lr_conf:.2f}), evidence is clear."

        rows.append({
            "example_id":        getattr(row, "example_id", ""),
            "customer_tweet_id": row.customer_tweet_id,
            "customer_text":     row.customer_text,
            "gold_intent":       row.intent,
            "predicted_intent":  lr_intent,
            "gold_escalate":     parse_gold_bool(row.gold_escalate),
            "predicted_escalate": escalate,
            "similarity":        result.similarity,
            "draft_reply":       result.reply,
            "reason":            reason,
            "evidence_1":        result.evidence[0] if result.evidence else "",
            "system":            "our_agent",
        })
    return pd.DataFrame(rows)


# --- Metrics -----------------------------------------------------------------

def compute_metrics(pred: pd.DataFrame) -> dict:
    intent_report = classification_report(
        pred["gold_intent"], pred["predicted_intent"],
        output_dict=True, zero_division=0
    )
    esc_cm = confusion_matrix(
        pred["gold_escalate"], pred["predicted_escalate"], labels=[False, True]
    ).tolist()

    return {
        "n_examples":          len(pred),
        "intent_accuracy":     round(accuracy_score(pred["gold_intent"], pred["predicted_intent"]), 4),
        "intent_macro_f1":     round(f1_score(pred["gold_intent"], pred["predicted_intent"],
                                              average="macro", zero_division=0), 4),
        "escalation_accuracy": round(accuracy_score(pred["gold_escalate"], pred["predicted_escalate"]), 4),
        "escalation_confusion_matrix": {
            "labels": ["auto-handle", "escalate"],
            "matrix": esc_cm,
        },
        "intent_report": intent_report,
    }


def print_summary(name: str, metrics: dict) -> None:
    print(f"\n{'-'*60}")
    print(f"  {name}")
    print(f"{'-'*60}")
    print(f"  Intent accuracy : {metrics['intent_accuracy']:.4f}")
    print(f"  Intent macro-F1 : {metrics['intent_macro_f1']:.4f}")
    print(f"  Escalation acc  : {metrics['escalation_accuracy']:.4f}")
    cm = metrics["escalation_confusion_matrix"]["matrix"]
    print(f"  Escalation CM   : TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")


# --- Main ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print("Loading data...")
    golden = load_and_validate_golden(args.golden)
    pairs  = pd.read_csv(args.pairs, low_memory=False)

    held_out_ids = set(golden["customer_tweet_id"].astype(str))
    reference    = build_reference(pairs, held_out_ids)
    print(f"  Golden: {len(golden)} rows | Reference: {len(reference):,} rows")

    majority_class = golden["intent"].value_counts().idxmax()
    print(f"  Majority class: {majority_class!r}")

    print("\nTraining LR classifier (weak supervision on pairs)...")
    lr_clf = build_from_pairs(pairs, sample=args.sample, random_state=args.random_state)
    print(f"  Classes: {lr_clf.classes_}")

    print("\nRunning System 0 - Trivial baseline...")
    pred0 = run_trivial(golden, majority_class)
    m0    = compute_metrics(pred0)
    print_summary("System 0: Trivial", m0)

    print("\nBuilding TF-IDF retrieval index on reference pairs...")
    agent = SupportAgent(reference, top_k=3)

    print("\nRunning System 1 - Simple (rule-based + TF-IDF)...")
    pred1 = run_simple(golden, agent)
    m1    = compute_metrics(pred1)
    print_summary("System 1: Simple (rule-based + retrieval)", m1)

    print("\nRunning System 2 - Our Agent (LR + TF-IDF)...")
    pred2 = run_our_agent(golden, agent, lr_clf, conf_threshold=args.conf_threshold)
    m2    = compute_metrics(pred2)
    print_summary("System 2: Our Agent (LR + retrieval)", m2)

    # -- save --------------------------------------------------------------
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_preds = pd.concat([pred0, pred1, pred2], ignore_index=True)
    all_preds.to_csv(args.output_dir / "predictions.csv", index=False)

    summary = {
        "trivial":    {k: v for k, v in m0.items() if k != "intent_report"},
        "simple":     {k: v for k, v in m1.items() if k != "intent_report"},
        "our_agent":  {k: v for k, v in m2.items() if k != "intent_report"},
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    with (args.output_dir / "intent_reports.json").open("w", encoding="utf-8") as fh:
        json.dump({
            "trivial":   m0["intent_report"],
            "simple":    m1["intent_report"],
            "our_agent": m2["intent_report"],
        }, fh, indent=2)

    # reply review template for LLM judge
    review = pred2.copy()
    for col in ["reply_correct", "reply_helpful", "reply_grounded", "reply_safe",
                "escalation_appropriate", "judge_notes"]:
        review[col] = ""
    review.to_csv(args.output_dir / "reply_review_template.csv", index=False)

    # comparison table
    table = pd.DataFrame([
        {"system": "Trivial",    **{k: v for k, v in m0.items() if k not in ("intent_report", "escalation_confusion_matrix")}},
        {"system": "Simple",     **{k: v for k, v in m1.items() if k not in ("intent_report", "escalation_confusion_matrix")}},
        {"system": "Our Agent",  **{k: v for k, v in m2.items() if k not in ("intent_report", "escalation_confusion_matrix")}},
    ])
    table.to_csv(args.output_dir / "comparison_table.csv", index=False)
    print("\n\n=== COMPARISON TABLE ===")
    print(table.to_string(index=False))
    print(f"\n[OK] Results saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
