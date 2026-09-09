"""Extract customer messages and replies for one support account."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


USECOLS = [
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
]
DTYPES = {
    "tweet_id": "string",
    "author_id": "string",
    "response_tweet_id": "string",
    "in_response_to_tweet_id": "string",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--brand", required=True, help="Support account handle, e.g. AppleSupport")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    return parser.parse_args()


def read_chunks(path: Path, chunk_size: int):
    return pd.read_csv(
        path,
        usecols=USECOLS,
        dtype=DTYPES,
        chunksize=chunk_size,
        low_memory=False,
    )


def is_outbound(series: pd.Series) -> pd.Series:
    return series.astype("string").str.lower().eq("false")


def collect_brand_replies(path: Path, brand: str, chunk_size: int) -> pd.DataFrame:
    replies: list[pd.DataFrame] = []
    for chunk in tqdm(read_chunks(path, chunk_size), desc="Finding brand replies"):
        mask = chunk["author_id"].eq(brand) & is_outbound(chunk["inbound"])
        selected = chunk.loc[mask].copy()
        if not selected.empty:
            replies.append(selected)

    if not replies:
        raise ValueError(f"No outbound tweets found for brand account {brand!r}.")

    result = pd.concat(replies, ignore_index=True)
    result = result[result["in_response_to_tweet_id"].notna()].copy()
    return result


def collect_customer_tweets(
    path: Path, parent_ids: set[str], chunk_size: int
) -> pd.DataFrame:
    customers: list[pd.DataFrame] = []
    for chunk in tqdm(read_chunks(path, chunk_size), desc="Finding customer messages"):
        selected = chunk[chunk["tweet_id"].isin(parent_ids)].copy()
        if not selected.empty:
            customers.append(selected)

    if not customers:
        raise ValueError("No customer tweets matched the brand replies' parent IDs.")

    return pd.concat(customers, ignore_index=True)


def build_pairs(replies: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    replies = replies.rename(
        columns={
            "tweet_id": "brand_tweet_id",
            "author_id": "brand_account",
            "created_at": "brand_created_at",
            "text": "brand_reply",
            "response_tweet_id": "brand_response_tweet_id",
            "in_response_to_tweet_id": "customer_tweet_id",
        }
    )
    customers = customers.rename(
        columns={
            "tweet_id": "customer_tweet_id",
            "author_id": "customer_author_id",
            "created_at": "customer_created_at",
            "text": "customer_text",
            "response_tweet_id": "customer_response_tweet_id",
        }
    )
    pairs = replies.merge(customers, on="customer_tweet_id", how="inner")
    pairs = pairs[
        [
            "customer_tweet_id",
            "customer_author_id",
            "customer_created_at",
            "customer_text",
            "brand_tweet_id",
            "brand_account",
            "brand_created_at",
            "brand_reply",
            "brand_response_tweet_id",
        ]
    ]
    pairs = pairs.dropna(subset=["customer_text", "brand_reply"])
    pairs = pairs.drop_duplicates(subset=["brand_tweet_id"])
    return pairs.sort_values("customer_tweet_id").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(args.input)

    replies = collect_brand_replies(args.input, args.brand, args.chunk_size)
    parent_ids = set(replies["in_response_to_tweet_id"].dropna().astype(str))
    customers = collect_customer_tweets(args.input, parent_ids, args.chunk_size)
    pairs = build_pairs(replies, customers)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.output, index=False)
    print(f"Brand: {args.brand}")
    print(f"Brand replies with a parent tweet: {len(replies):,}")
    print(f"Matched customer/reply pairs: {len(pairs):,}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
