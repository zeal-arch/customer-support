"""Inspect the Customer Support on Twitter CSV before choosing a brand."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


EXPECTED_COLUMNS = {
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to twcs.csv")
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=0,
        help="Read only this many rows; 0 reads the full CSV.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Rows to process at a time when reading the full CSV.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of likely support accounts to display.",
    )
    return parser.parse_args()


def load_csv(path: Path, sample_rows: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download twcs.csv and place it there."
        )

    kwargs = {"nrows": sample_rows} if sample_rows > 0 else {}
    return pd.read_csv(path, low_memory=False, **kwargs)


def print_basic_report(df: pd.DataFrame, path: Path) -> None:
    print(f"Input: {path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {', '.join(df.columns)}")

    missing_columns = sorted(EXPECTED_COLUMNS - set(df.columns))
    if missing_columns:
        print(f"Missing expected columns: {', '.join(missing_columns)}")
    else:
        print("Schema: standard Customer Support on Twitter columns detected")

    if "inbound" in df:
        print("\nInbound values:")
        print(df["inbound"].value_counts(dropna=False).to_string())

    if "text" in df:
        text = df["text"].fillna("").astype(str)
        print("\nText quality:")
        print(f"  Empty text: {(text.str.strip() == '').sum():,}")
        print(f"  Median characters: {text.str.len().median():.0f}")
        print(f"  95th percentile characters: {text.str.len().quantile(0.95):.0f}")

    print("\nMissing values:")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing[missing > 0].head(10).to_string() or "None")

    print(
        "\nNote: the raw dataset does not always provide a clean brand column. "
        "Use author/conversation inspection before selecting a brand."
    )


def is_brand_account(value: object) -> bool:
    """Support handles are generally strings; customer IDs are numeric strings."""
    text = str(value).strip()
    return bool(text) and not text.isdigit()


def safe_console_text(value: str) -> str:
    """Keep reports printable on Windows consoles using legacy encodings."""
    return value.encode("ascii", errors="replace").decode("ascii")


def inspect_full_csv(path: Path, chunk_size: int, top: int) -> None:
    """Stream the full file and rank likely brand support accounts."""
    support_tweet_counts: Counter[str] = Counter()
    support_examples: dict[str, str] = {}
    row_count = 0
    inbound_counts: Counter[str] = Counter()

    for chunk in pd.read_csv(path, low_memory=False, chunksize=chunk_size):
        row_count += len(chunk)
        if "inbound" in chunk:
            inbound_counts.update(chunk["inbound"].astype(str).str.lower())

        if not {"author_id", "inbound", "text"}.issubset(chunk.columns):
            continue

        outbound = chunk[chunk["inbound"].astype(str).str.lower().eq("false")]
        for author, group in outbound.groupby("author_id", dropna=True):
            author = str(author)
            if not is_brand_account(author):
                continue
            support_tweet_counts[author] += len(group)
            if author not in support_examples:
                text = group["text"].dropna().astype(str)
                if not text.empty:
                    support_examples[author] = text.iloc[0]

    print(f"Rows scanned: {row_count:,}")
    print("Inbound values:")
    for value, count in inbound_counts.most_common():
        print(f"  {value}: {count:,}")

    print(f"\nLikely brand support accounts (top {top} by outbound tweets):")
    print("rank | account | outbound tweets | example reply")
    print("-----|---------|-----------------|---------------")
    for rank, (author, count) in enumerate(support_tweet_counts.most_common(top), start=1):
        example = support_examples.get(author, "").replace("\n", " ")
        if len(example) > 120:
            example = example[:117] + "..."
        author = safe_console_text(author)
        example = safe_console_text(example)
        print(f"{rank:>4} | {author:<30} | {count:>15,} | {example}")


def main() -> None:
    args = parse_args()
    if args.sample_rows > 0:
        df = load_csv(args.input, args.sample_rows)
        print_basic_report(df, args.input)
    else:
        inspect_full_csv(args.input, args.chunk_size, args.top)


if __name__ == "__main__":
    main()
