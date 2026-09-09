"""Run the transparent retrieval baseline on one customer message."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .agent import SupportAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("data/processed/applesupport_pairs.csv"),
                        help="Path to preprocessed brand pairs CSV.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--backend", choices=["tfidf", "minilm"], default="tfidf")
    parser.add_argument("--max-reference", type=int, default=30_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = pd.read_csv(args.pairs, low_memory=False)
    if args.backend == "minilm":
        from .semantic_agent import SemanticSupportAgent

        agent = SemanticSupportAgent(pairs, top_k=args.top_k, max_reference=args.max_reference)
    else:
        agent = SupportAgent(pairs, top_k=args.top_k)
    result = agent.predict(args.text)
    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
