"""Create draft labels to accelerate review of the golden set."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .agent import classify_intent


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "1", "yes", "y"}


def suggested_escalation(text: str, intent: str, confidence: float) -> bool:
    lowered = text.lower()
    sensitive = any(
        token in lowered
        for token in ("hacked", "fraud", "phishing", "lawyer", "legal", "injury", "unsafe")
    )
    return sensitive or intent in {"account_access_security", "app_store_billing", "other_unclear"} or confidence < 0.5


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    suggestions = df["customer_text"].fillna("").map(classify_intent)
    df["suggested_intent"] = [item[0] for item in suggestions]
    df["suggested_confidence"] = [item[1] for item in suggestions]
    df["suggested_escalate"] = [
        suggested_escalation(text, intent, confidence)
        for text, (intent, confidence) in zip(df["customer_text"].fillna(""), suggestions)
    ]
    df["review_status"] = "needs_human_review"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Created draft labels at {args.output}")
    print("These are suggestions only; review them before using the file as ground truth.")


if __name__ == "__main__":
    main()
