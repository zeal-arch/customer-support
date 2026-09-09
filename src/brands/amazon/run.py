"""Amazon customer support agent CLI runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from ...text_preprocessing import normalize_for_classification
    from .intent_classifier import classify_amazon_intent, decide_amazon_escalation
except (ImportError, ValueError):
    try:
        from src.text_preprocessing import normalize_for_classification
        from intent_classifier import classify_amazon_intent, decide_amazon_escalation
    except (ImportError, ValueError):
        from src.text_preprocessing import normalize_for_classification
        from src.brands.amazon.intent_classifier import classify_amazon_intent, decide_amazon_escalation

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PAIRS = (
    ROOT_DIR / "data" / "processed" / "brands" / "amazonhelp" / "amazonhelp_pairs.csv"
    if (ROOT_DIR / "data" / "processed" / "brands" / "amazonhelp" / "amazonhelp_pairs.csv").exists()
    else ROOT_DIR / "data" / "processed" / "amazonhelp_pairs.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS,
                        help="Path to preprocessed AmazonHelp pairs CSV.")
    parser.add_argument("--text", required=True, help="Customer message text.")
    parser.add_argument("--sample-size", type=int, default=40_000,
                        help="Number of reference pairs to load for fast retrieval.")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


class AmazonSupportAgent:
    def __init__(self, pairs: pd.DataFrame, top_k: int = 3) -> None:
        self.pairs = pairs.reset_index(drop=True)
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=3,
            max_features=60_000,
            strip_accents="unicode",
        )
        documents = self.pairs["customer_text"].fillna("").map(normalize_for_classification)
        self.matrix = self.vectorizer.fit_transform(documents)

    def predict(self, text: str) -> dict:
        query = self.vectorizer.transform([normalize_for_classification(text)])
        similarities = (self.matrix @ query.T).toarray().ravel()
        nearest = similarities.argsort()[::-1][: self.top_k]
        best = int(nearest[0])
        similarity = float(similarities[best])

        intent, confidence = classify_amazon_intent(text)
        escalate, reason = decide_amazon_escalation(intent, confidence, similarity)

        evidence = [str(self.pairs.iloc[idx]["customer_text"]) for idx in nearest]
        best_reply = str(self.pairs.iloc[best]["brand_reply"])

        return {
            "brand": "AmazonHelp",
            "intent": intent,
            "confidence": round(confidence, 4),
            "reply": best_reply,
            "escalate": escalate,
            "reason": reason,
            "similarity": round(similarity, 4),
            "evidence": evidence,
        }


def main() -> None:
    args = parse_args()
    if not args.pairs.exists():
        raise FileNotFoundError(f"Pairs file not found at {args.pairs}. Run src/extract_brand.py first.")

    pairs = pd.read_csv(args.pairs, low_memory=False)
    if len(pairs) > args.sample_size:
        pairs = pairs.sample(args.sample_size, random_state=42)

    agent = AmazonSupportAgent(pairs, top_k=args.top_k)
    result = agent.predict(args.text)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
