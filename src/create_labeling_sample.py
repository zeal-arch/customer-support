"""Create a deterministic human-labeling sheet from brand pairs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=250)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, low_memory=False)
    df["text_length"] = df["customer_text"].fillna("").astype(str).str.len()
    df["length_bucket"] = pd.qcut(
        df["text_length"].rank(method="first"), q=5, labels=False
    )

    per_bucket = max(1, args.size // 5)
    sample = (
        df.groupby("length_bucket", group_keys=False)
        .apply(
            lambda group: group.sample(
                min(per_bucket, len(group)), random_state=args.random_state
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )

    if len(sample) < args.size:
        remaining = df[~df["customer_tweet_id"].isin(sample["customer_tweet_id"])]
        sample = pd.concat(
            [
                sample,
                remaining.sample(args.size - len(sample), random_state=args.random_state),
            ],
            ignore_index=True,
        )

    sample = sample.sample(frac=1, random_state=args.random_state).head(args.size)
    sample.insert(0, "example_id", range(1, len(sample) + 1))
    sample["intent"] = ""
    sample["gold_escalate"] = ""
    sample["label_notes"] = ""
    sample["labeler"] = ""
    sample = sample[
        [
            "example_id",
            "customer_tweet_id",
            "customer_text",
            "brand_reply",
            "intent",
            "gold_escalate",
            "label_notes",
            "labeler",
        ]
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(args.output, index=False)
    print(f"Created {len(sample):,} labeling rows at {args.output}")


if __name__ == "__main__":
    main()
