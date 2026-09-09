"""Evaluate the retrieval baseline against a reviewed golden set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .agent import SupportAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parse_gold_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def main() -> None:
    args = parse_args()
    golden = pd.read_csv(args.golden, low_memory=False)
    required = {"customer_tweet_id", "customer_text", "intent", "gold_escalate"}
    missing = required - set(golden.columns)
    if missing:
        raise ValueError(f"Golden set is missing columns: {sorted(missing)}")

    labels = golden["intent"].fillna("").astype(str).str.strip()
    escalation = golden["gold_escalate"].fillna("").astype(str).str.strip()
    if (labels == "").any() or (escalation == "").any():
        raise ValueError(
            "Golden labels are incomplete. Review evaluation/golden_set.csv before running headline metrics."
        )

    pairs = pd.read_csv(args.pairs, low_memory=False)
    held_out_ids = set(golden["customer_tweet_id"].astype(str))
    reference = pairs[~pairs["customer_tweet_id"].astype(str).isin(held_out_ids)].copy()
    agent = SupportAgent(reference)

    predictions = []
    for row in golden.itertuples(index=False):
        result = agent.predict(row.customer_text)
        predictions.append(
            {
                "example_id": getattr(row, "example_id", ""),
                "customer_tweet_id": row.customer_tweet_id,
                "customer_text": row.customer_text,
                "gold_intent": row.intent,
                "predicted_intent": result.intent,
                "gold_escalate": parse_gold_bool(row.gold_escalate),
                "predicted_escalate": result.escalate,
                "similarity": result.similarity,
                "draft_reply": result.reply,
                "reason": result.reason,
                "evidence_1": result.evidence[0] if result.evidence else "",
            }
        )

    pred = pd.DataFrame(predictions)
    intent_report = classification_report(
        pred["gold_intent"], pred["predicted_intent"], output_dict=True, zero_division=0
    )
    metrics = {
        "n_examples": len(pred),
        "intent_accuracy": accuracy_score(pred["gold_intent"], pred["predicted_intent"]),
        "intent_macro_f1": f1_score(
            pred["gold_intent"], pred["predicted_intent"], average="macro", zero_division=0
        ),
        "escalation_accuracy": accuracy_score(pred["gold_escalate"], pred["predicted_escalate"]),
        "intent_report": intent_report,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(args.output_dir / "predictions.csv", index=False)
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    review = pred.copy()
    for column in ["reply_correct", "reply_helpful", "reply_grounded", "reply_safe", "judge_notes"]:
        review[column] = ""
    review.to_csv(args.output_dir / "reply_review_template.csv", index=False)
    print(json.dumps({key: value for key, value in metrics.items() if key != "intent_report"}, indent=2))
    print(f"Saved evaluation outputs to {args.output_dir}")


if __name__ == "__main__":
    main()
