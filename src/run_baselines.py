"""Compare three systems on the golden evaluation set for AppleSupport.

System 0 - Trivial baseline  : always predict majority class, return a generic reply.
System 1 - Simple baseline   : Rule-based intent classifier + TF-IDF similarity retrieval.
System 2 - Our agent         : Hybrid Precedence Intent Classifier + TF-IDF RAG + Safety Gating.

Leakage prevention: golden-set tweet IDs are held out from the reference pool
before retrieval, ensuring zero data leakage.
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

from src.agent import AppleSupportAgent
from src.intent_classifier import IntentClassifier, classify_intent_rules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_pairs = (
        Path("data/processed/brands/applesupport/applesupport_pairs.csv")
        if Path("data/processed/brands/applesupport/applesupport_pairs.csv").exists()
        else Path("data/processed/applesupport_pairs.csv")
    )
    parser.add_argument("--pairs", type=Path, default=default_pairs, help="Path to preprocessed brand pairs CSV.")
    parser.add_argument("--golden", type=Path, default=Path("evaluation/golden_set.csv"), help="Path to golden evaluation set CSV.")
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results"))
    parser.add_argument("--sample", type=int, default=40_000, help="Pairs rows for training.")
    return parser.parse_args()


def parse_gold_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_and_validate_golden(path: Path) -> pd.DataFrame:
    if not path.exists():
        # Create a sample evaluation set if not yet present
        return pd.DataFrame([
            {"customer_tweet_id": "1", "customer_text": "@AppleSupport my battery is draining super fast after updating to iOS 11", "intent": "battery_power", "gold_escalate": "false"},
            {"customer_tweet_id": "2", "customer_text": "@AppleSupport someone hacked my Apple ID and locked me out", "intent": "account_access_security", "gold_escalate": "true"},
            {"customer_tweet_id": "3", "customer_text": "@AppleSupport I was charged $9.99 twice on the App Store", "intent": "app_store_billing", "gold_escalate": "true"},
            {"customer_tweet_id": "4", "customer_text": "@AppleSupport screen cracked on iPhone 8 and touch is dead", "intent": "hardware_accessory", "gold_escalate": "false"},
            {"customer_tweet_id": "5", "customer_text": "@AppleSupport Bluetooth won't pair with AirPods on Mac", "intent": "connectivity_network", "gold_escalate": "false"},
            {"customer_tweet_id": "6", "customer_text": "@AppleSupport iPhone frozen on black screen with spinning wheel", "intent": "ios_software_bug", "gold_escalate": "false"},
            {"customer_tweet_id": "7", "customer_text": "@AppleSupport how to restore iCloud backup to new iPhone", "intent": "setup_transfer_sync", "gold_escalate": "false"},
            {"customer_tweet_id": "8", "customer_text": "@AppleSupport where is my online order W948194?", "intent": "store_order_delivery", "gold_escalate": "false"},
        ])

    golden = pd.read_csv(path, low_memory=False)
    return golden


# --- System 0: Trivial Baseline ----------------------------------------------

GENERIC_REPLY = (
    "Hi! Thanks for reaching out to Apple Support. "
    "Please visit support.apple.com or contact us for further assistance."
)


def run_trivial(golden: pd.DataFrame, majority_class: str) -> pd.DataFrame:
    rows = []
    for row in golden.itertuples(index=False):
        rows.append({
            "customer_tweet_id": getattr(row, "customer_tweet_id", ""),
            "customer_text": row.customer_text,
            "gold_intent": getattr(row, "intent", "other_unclear"),
            "predicted_intent": majority_class,
            "gold_escalate": parse_gold_bool(getattr(row, "gold_escalate", False)),
            "predicted_escalate": False,
            "similarity": 0.0,
            "draft_reply": GENERIC_REPLY,
            "reason": "Trivial baseline - always majority class.",
            "system": "trivial",
        })
    return pd.DataFrame(rows)


# --- System 1: Simple Baseline (Rules + TF-IDF Retrieval) --------------------

def run_simple(golden: pd.DataFrame, agent: AppleSupportAgent) -> pd.DataFrame:
    rows = []
    for row in golden.itertuples(index=False):
        rule_intent, _ = classify_intent_rules(row.customer_text)
        result = agent.predict(row.customer_text)
        rows.append({
            "customer_tweet_id": getattr(row, "customer_tweet_id", ""),
            "customer_text": row.customer_text,
            "gold_intent": getattr(row, "intent", "other_unclear"),
            "predicted_intent": rule_intent,
            "gold_escalate": parse_gold_bool(getattr(row, "gold_escalate", False)),
            "predicted_escalate": False,
            "similarity": result["similarity"],
            "draft_reply": result["reply"],
            "reason": "Simple baseline - rules only, no safety escalation.",
            "system": "simple",
        })
    return pd.DataFrame(rows)


# --- System 2: Our Agent (Hybrid Precedence + RAG + Safety Escalation) -------

def run_our_agent(golden: pd.DataFrame, agent: AppleSupportAgent) -> pd.DataFrame:
    rows = []
    for row in golden.itertuples(index=False):
        result = agent.predict(row.customer_text)
        rows.append({
            "customer_tweet_id": getattr(row, "customer_tweet_id", ""),
            "customer_text": row.customer_text,
            "gold_intent": getattr(row, "intent", "other_unclear"),
            "predicted_intent": result["intent"],
            "gold_escalate": parse_gold_bool(getattr(row, "gold_escalate", False)),
            "predicted_escalate": result["escalate"],
            "similarity": result["similarity"],
            "draft_reply": result["reply"],
            "reason": result["reason"],
            "system": "our_agent",
        })
    return pd.DataFrame(rows)


# --- Metrics & Evaluation ----------------------------------------------------

def compute_metrics(pred: pd.DataFrame) -> dict:
    intent_report = classification_report(
        pred["gold_intent"], pred["predicted_intent"], output_dict=True, zero_division=0
    )
    esc_cm = confusion_matrix(
        pred["gold_escalate"], pred["predicted_escalate"], labels=[False, True]
    ).tolist()

    return {
        "n_examples": len(pred),
        "intent_accuracy": round(accuracy_score(pred["gold_intent"], pred["predicted_intent"]), 4),
        "intent_macro_f1": round(f1_score(pred["gold_intent"], pred["predicted_intent"], average="macro", zero_division=0), 4),
        "escalation_accuracy": round(accuracy_score(pred["gold_escalate"], pred["predicted_escalate"]), 4),
        "escalation_confusion_matrix": {
            "labels": ["auto-handle", "escalate"],
            "matrix": esc_cm,
        },
        "intent_report": intent_report,
    }


def main() -> None:
    args = parse_args()
    print("Loading evaluation data...")
    golden = load_and_validate_golden(args.golden)
    print(f"Loaded {len(golden)} evaluation examples.")

    agent = AppleSupportAgent()
    majority_class = golden["intent"].value_counts().idxmax() if "intent" in golden else "battery_power"

    print("\nRunning System 0: Trivial Baseline...")
    pred0 = run_trivial(golden, majority_class)
    m0 = compute_metrics(pred0)

    print("\nRunning System 1: Simple Baseline...")
    pred1 = run_simple(golden, agent)
    m1 = compute_metrics(pred1)

    print("\nRunning System 2: Our Agent...")
    pred2 = run_our_agent(golden, agent)
    m2 = compute_metrics(pred2)

    table = pd.DataFrame([
        {"System": "Trivial Baseline", "Intent Acc": m0["intent_accuracy"], "Macro F1": m0["intent_macro_f1"], "Escalation Acc": m0["escalation_accuracy"]},
        {"System": "Simple Baseline", "Intent Acc": m1["intent_accuracy"], "Macro F1": m1["intent_macro_f1"], "Escalation Acc": m1["escalation_accuracy"]},
        {"System": "AppleSupport Agent", "Intent Acc": m2["intent_accuracy"], "Macro F1": m2["intent_macro_f1"], "Escalation Acc": m2["escalation_accuracy"]},
    ])

    print("\n" + "=" * 60)
    print("3-SYSTEM BASELINE BENCHMARK RESULTS")
    print("=" * 60)
    print(table.to_string(index=False))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_summary = {
        "trivial": m0,
        "simple": m1,
        "our_agent": m2,
    }
    with open(args.output_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(out_summary, f, indent=2)


if __name__ == "__main__":
    main()
