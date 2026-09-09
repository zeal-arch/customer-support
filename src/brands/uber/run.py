"""Uber Support customer support agent CLI runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from .agent import UberSupportAgent
except (ImportError, ValueError):
    from agent import UberSupportAgent

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PAIRS = (
    ROOT_DIR / "data" / "processed" / "brands" / "uber_support" / "uber_support_pairs.csv"
    if (ROOT_DIR / "data" / "processed" / "brands" / "uber_support" / "uber_support_pairs.csv").exists()
    else ROOT_DIR / "data" / "processed" / "uber_support_pairs.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS,
                        help="Path to preprocessed Uber Support pairs CSV.")
    parser.add_argument("--text", required=True, help="Customer tweet text.")
    parser.add_argument("--handle", default="", help="Optional customer Twitter handle to personalize reply.")
    parser.add_argument("--sample-size", type=int, default=35_000,
                        help="Pairs rows for training intent classifier.")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pairs.exists():
        raise FileNotFoundError(f"Pairs file not found at {args.pairs}.")

    pairs = pd.read_csv(args.pairs, low_memory=False)
    agent = UberSupportAgent(pairs, sample_size=args.sample_size, top_k=args.top_k)
    result = agent.predict(args.text, handle=args.handle)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
