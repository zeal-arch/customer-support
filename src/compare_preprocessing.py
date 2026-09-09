"""Compare conservative preprocessing with the popular notebook-style pipeline."""

from __future__ import annotations

import argparse
import html
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

from .text_preprocessing import normalize_for_classification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=500)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def raw_text(text: object) -> str:
    return "" if text is None else str(text).lower()


def notebook_style_text(text: object) -> str:
    """Approximate the supplied notebook's URL/mention/punctuation/stopword path."""
    value = html.unescape("" if text is None else str(text)).lower()
    value = re.sub(r"(?:https?://|www\.)\S+", " ", value)
    value = re.sub(r"@[A-Za-z0-9_]+", " ", value)
    value = value.translate(str.maketrans("", "", string.punctuation))
    tokens = [token for token in value.split() if token not in ENGLISH_STOP_WORDS]
    return " ".join(tokens)


def evaluate_variant(
    name: str, train_text: pd.Series, query_text: pd.Series
) -> tuple[dict[str, float], np.ndarray]:
    functions = {
        "raw": raw_text,
        "conservative": normalize_for_classification,
        "notebook_style": notebook_style_text,
    }
    vectorizer = TfidfVectorizer(
        preprocessor=functions[name],
        tokenizer=str.split,
        token_pattern=None,
        ngram_range=(1, 2),
        min_df=2,
        max_features=50_000,
        norm="l2",
    )
    train_matrix = vectorizer.fit_transform(train_text)
    query_matrix = vectorizer.transform(query_text)
    similarities = query_matrix @ train_matrix.T
    best = similarities.max(axis=1).toarray().ravel()
    stats = {
        "mean_top1_similarity": float(best.mean()),
        "median_top1_similarity": float(np.median(best)),
        "p10_top1_similarity": float(np.quantile(best, 0.10)),
        "share_at_least_0_20": float((best >= 0.20).mean()),
        "share_at_least_0_40": float((best >= 0.40).mean()),
    }
    return stats, best


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, low_memory=False)
    queries = df.sample(min(args.sample_size, len(df)), random_state=args.random_state)
    train = df.drop(index=queries.index)

    rows = []
    all_scores = {}
    for name in ("raw", "conservative", "notebook_style"):
        stats, scores = evaluate_variant(name, train["customer_text"], queries["customer_text"])
        rows.append({"variant": name, **stats})
        all_scores[name] = scores

    results = pd.DataFrame(rows).sort_values("mean_top1_similarity", ascending=False)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nInterpretation: this is a retrieval-evidence smoke test, not the final reply-quality score.")
    print("Higher similarity means the variant finds closer historical messages; it does not prove the reply is correct.")

    comparison = pd.DataFrame(all_scores)
    comparison["customer_text"] = queries["customer_text"].to_numpy()
    best_variant = comparison[["raw", "conservative", "notebook_style"]].idxmax(axis=1)
    comparison["best_variant"] = best_variant
    print("\nExamples where preprocessing changes the strongest match:")
    for _, row in comparison[comparison.best_variant != "conservative"].head(5).iterrows():
        print(f"- {str(row['customer_text'])[:180]}")
        print(f"  raw={row['raw']:.3f}, conservative={row['conservative']:.3f}, notebook_style={row['notebook_style']:.3f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
