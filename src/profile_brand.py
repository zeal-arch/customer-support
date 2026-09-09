"""Profile a brand-specific customer/reply table to guide intent design."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=30_000)
    parser.add_argument("--clusters", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def clean_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.replace(r"https?://\S+", " ", regex=True)
        .str.replace(r"@[A-Za-z0-9_]+", " ", regex=True)
        .str.replace(r"\d+", " ", regex=True)
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def safe_console_text(value: str) -> str:
    return value.encode("ascii", errors="replace").decode("ascii")


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input, low_memory=False)
    if len(df) > args.sample_size:
        df = df.sample(args.sample_size, random_state=args.random_state)
    df = df.reset_index(drop=True)
    text = clean_text(df["customer_text"])

    print(f"Rows profiled: {len(df):,}")
    print(f"Unique customer messages: {df['customer_tweet_id'].nunique():,}")
    print(f"Median customer length: {text.str.len().median():.0f} characters")
    print(f"95th percentile customer length: {text.str.len().quantile(.95):.0f} characters")

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=10,
        max_features=15_000,
    )
    matrix = vectorizer.fit_transform(text)
    terms = vectorizer.get_feature_names_out()
    mean_scores = matrix.mean(axis=0).A1
    top_terms = terms[mean_scores.argsort()[-40:][::-1]]
    print("\nCommon discriminative terms:")
    print(", ".join(top_terms))

    model = KMeans(n_clusters=args.clusters, random_state=args.random_state, n_init=10)
    labels = model.fit_predict(matrix)
    print("\nExploratory clusters:")
    for cluster_id in range(args.clusters):
        row_scores = model.transform(matrix)[:, cluster_id]
        nearest = row_scores.argsort()[:3]
        center_terms = terms[model.cluster_centers_[cluster_id].argsort()[-10:][::-1]]
        print(f"\nCluster {cluster_id} ({(labels == cluster_id).sum():,} rows)")
        print(f"Terms: {', '.join(center_terms)}")
        for row_id in nearest:
            customer = re.sub(r"\s+", " ", str(df.loc[row_id, "customer_text"])).strip()
            reply = re.sub(r"\s+", " ", str(df.loc[row_id, "brand_reply"])).strip()
            print(f"  Customer: {safe_console_text(customer[:180])}")
            print(f"  Reply:    {safe_console_text(reply[:180])}")


if __name__ == "__main__":
    main()
