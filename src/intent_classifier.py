"""TF-IDF + Logistic Regression intent classifier for AppleSupport.

This is the main intent-classification model used in the submission.
It is trained on a weakly-supervised sample of the AppleSupport pairs
and evaluated on the hand-finalised golden set.

The class interface matches classify_intent() in agent.py so both
can be used interchangeably.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from .text_preprocessing import normalize_for_classification


class IntentClassifier:
    """TF-IDF + Logistic Regression intent classifier.

    Improvements from reference notebooks (chatbot-ai, text-preprocessing-nlp):
    - sublinear_tf=True  : dampens TF scores, reduces very-common-word dominance
    - ngram_range=(1,3)  : captures 3-word phrases like "won't turn on", "app store refund"
    - max_features=100k  : wider vocabulary for rare product/issue terminology
    - class_weight='balanced' : compensates for ios_software_bug class imbalance (63% of corpus)
    - C=5.0              : slightly less regularisation improves minority-class recall
    """

    def __init__(
        self,
        max_features: int = 80_000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 3,
        C: float = 2.0,
        random_state: int = 42,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            strip_accents="unicode",
        )
        self.encoder = LabelEncoder()
        self.clf = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=C,
            random_state=random_state,
        )
        self._fitted = False

    # -- Training -----------------------------------------------------------

    def fit(self, texts: pd.Series, labels: pd.Series) -> "IntentClassifier":
        """Fit on a labelled corpus. texts and labels must align."""
        norm = texts.fillna("").map(normalize_for_classification)
        X = self.vectorizer.fit_transform(norm)
        y = self.encoder.fit_transform(labels)
        self.clf.fit(X, y)
        self._fitted = True
        return self

    # -- Inference ----------------------------------------------------------

    def predict(self, text: str) -> tuple[str, float]:
        """Return (intent_label, confidence) for a single message."""
        if not self._fitted:
            raise RuntimeError("IntentClassifier not fitted. Call fit() first.")
        norm = normalize_for_classification(text)
        X = self.vectorizer.transform([norm])
        proba = self.clf.predict_proba(X)[0]
        idx = proba.argmax()
        return str(self.encoder.classes_[idx]), float(proba[idx])

    def predict_batch(self, texts: pd.Series) -> tuple[list[str], list[float]]:
        """Vectorised batch prediction."""
        if not self._fitted:
            raise RuntimeError("IntentClassifier not fitted.")
        norm = texts.fillna("").map(normalize_for_classification)
        X = self.vectorizer.transform(norm)
        proba = self.clf.predict_proba(X)
        indices = proba.argmax(axis=1)
        labels  = [str(self.encoder.classes_[i]) for i in indices]
        confs   = list(proba.max(axis=1))
        return labels, confs

    # -- Convenience --------------------------------------------------------

    @property
    def classes_(self) -> list[str]:
        return list(self.encoder.classes_)

    def __repr__(self) -> str:
        status = f"fitted on {len(self.classes_)} classes" if self._fitted else "not fitted"
        return f"IntentClassifier({status})"


def build_from_pairs(
    pairs: pd.DataFrame,
    sample: int = 40_000,
    lr_threshold: float = 0.60,
    random_state: int = 42,
) -> IntentClassifier:
    """Train a classifier using weakly-supervised labels from the rule-based agent."""
    from .agent import classify_intent

    corpus = pairs.sample(min(sample, len(pairs)), random_state=random_state).copy()
    results = corpus["customer_text"].fillna("").map(classify_intent)
    corpus["weak_intent"]     = [r[0] for r in results]
    corpus["weak_confidence"] = [r[1] for r in results]

    mask = (corpus["weak_confidence"] >= lr_threshold) & (corpus["weak_intent"] != "other_unclear")
    corpus = corpus[mask].copy()

    clf = IntentClassifier(random_state=random_state)
    clf.fit(corpus["customer_text"], corpus["weak_intent"])
    return clf
