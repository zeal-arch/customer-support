"""Benchmark text preprocessing on the optional Banking77 intent dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from .compare_preprocessing import notebook_style_text, raw_text
from .text_preprocessing import normalize_for_classification


BANKING77_FILES = {
    "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
    "test": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # The current `datasets` package no longer executes the legacy
    # banking77.py loader, so load the official CSV files directly.
    dataset = load_dataset("csv", data_files=BANKING77_FILES)
    train = pd.DataFrame(dataset["train"])
    test = pd.DataFrame(dataset["test"])
    labels = sorted(set(train["category"]) | set(test["category"]))
    label_to_id = {label: index for index, label in enumerate(labels)}
    train["label"] = train["category"].map(label_to_id)
    test["label"] = test["category"].map(label_to_id)
    if args.max_train:
        train = train.head(args.max_train)
    if args.max_test:
        test = test.head(args.max_test)

    variants = {
        "raw": raw_text,
        "conservative": normalize_for_classification,
        "notebook_style": notebook_style_text,
    }
    rows = []
    for name, preprocess in variants.items():
        vectorizer = TfidfVectorizer(
            preprocessor=preprocess,
            tokenizer=str.split,
            token_pattern=None,
            ngram_range=(1, 2),
            min_df=2,
            max_features=100_000,
        )
        x_train = vectorizer.fit_transform(train["text"])
        x_test = vectorizer.transform(test["text"])
        model = LogisticRegression(max_iter=300, solver="saga")
        model.fit(x_train, train["label"])
        predicted = model.predict(x_test)
        rows.append(
            {
                "variant": name,
                "accuracy": accuracy_score(test["label"], predicted),
                "macro_f1": f1_score(test["label"], predicted, average="macro"),
                "features": len(vectorizer.vocabulary_),
            }
        )

    results = pd.DataFrame(rows).sort_values("macro_f1", ascending=False)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print("\nThis benchmark tests intent classification on Banking77 only; it is not an AppleSupport result.")
    print("The AppleSupport headline result must still come from our manually labeled AppleSupport set.")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
