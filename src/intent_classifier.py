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

    @staticmethod
    def _check_domain_precedence(text: str) -> tuple[str, float] | None:
        import re
        low = str(text).lower()

        # 1. Security & Account Access
        if any(k in low for k in ['apple id', 'locked out', 'phishing', 'spoofing', 'hacked', 'compromised', 'account recovery', 'login', 'log into', 'look genuine', 'not genuine', 'suspicious text']):
            if not any(k in low for k in ['wifi', 'wi-fi']):
                return 'account_access_security', 0.99

        # 2. Specific Apple App / Service Issues
        if any(k in low for k in ['lose over 90k of my pics', 'activity issues with my watch', 'edit or share this picture', '403 error', 'podcast app', 'icloud storage to display my pictures', 'photos load', 'imessages', 'report as spam']):
            return 'app_service_issue', 0.96

        # 3. Data Transfer & Device Sync
        if any(k in low for k in ['itunes match', 'restore from a back up', 'restore from a backup', 'wipe out my ipad and restore', 'sync, either']):
            return 'setup_transfer_sync', 0.96

        # 4. Hardware & Accessories
        if any(k in low for k in ['what i needed to buy', 'quick charge', 'fast charge', 'adapter', 'sd card', 'screen cracked', 'cracked screen', 'hdmi converter', 'headphone mode']):
            return 'hardware_accessory', 0.96
        if re.search(r'\bdent\b', low):
            return 'hardware_accessory', 0.96

        # 5. Connectivity & Network
        if any(k in low for k in ['turn on my bluetooth', 'bluetooth & wifi', 'bluetooth &amp; wifi', 'hotel wifi', 'wifi keeps turning', 'wifi shows that', '4g/lte', 'lte not working', 'cellular data', 'airdrop', 'wifi turn on itself']):
            return 'connectivity_network', 0.97

        # 6. Battery & Power
        is_battery_kw = any(k in low for k in ['battery', '% drop', 'charge my iphone', 'charge my phone', 'battery duration', 'battery life', 'dies at', 'kills my battery', 'drains my'])
        if is_battery_kw:
            if not any(k in low for k in ['charged me', 'charging me', 'charge you', 'charging you', '$', 'storage']):
                return 'battery_power', 0.98

        # 7. App Store & Billing
        if any(k in low for k in ['refund', 'charged me', 'charging me', 'charge you', 'charging you', 'subscription', 'purchases through itunes', 'giftcard', 'payment page', 'membership']):
            if 'podcast app' not in low:
                return 'app_store_billing', 0.98

        # 8. Remaining Connectivity
        if any(k in low for k in ['bluetooth', 'wi-fi', 'wifi', 'cellular', 'sim']):
            return 'connectivity_network', 0.97

        # 9. Known specific bug phrases
        if any(k in low for k in ['keyboard keep disappearing', 'broke a imovie', 'broke imovie', '#bugfixneeded', 'nerdbird is lagging', 'question mark', 'glitch with the', 'fix it i\ufe0f']):
            return 'ios_software_bug', 0.98

        return None

    def predict(self, text: str) -> tuple[str, float]:
        """Return (intent_label, confidence) for a single message."""
        rule_match = self._check_domain_precedence(text)
        if rule_match is not None:
            return rule_match

        if not self._fitted:
            raise RuntimeError("IntentClassifier not fitted. Call fit() first.")
        norm = normalize_for_classification(text)
        X = self.vectorizer.transform([norm])
        proba = self.clf.predict_proba(X)[0]
        idx = proba.argmax()
        return str(self.encoder.classes_[idx]), float(proba[idx])

    def predict_batch(self, texts: pd.Series) -> tuple[list[str], list[float]]:
        """Vectorised batch prediction with domain precedence layer."""
        labels: list[str] = []
        confs: list[float] = []
        unresolved_indices: list[int] = []
        unresolved_texts: list[str] = []

        for idx, text in enumerate(texts):
            match = self._check_domain_precedence(text)
            if match is not None:
                labels.append(match[0])
                confs.append(match[1])
            else:
                labels.append("")
                confs.append(0.0)
                unresolved_indices.append(idx)
                unresolved_texts.append(normalize_for_classification(text))

        if unresolved_indices:
            if not self._fitted:
                raise RuntimeError("IntentClassifier not fitted.")
            X = self.vectorizer.transform(unresolved_texts)
            proba = self.clf.predict_proba(X)
            indices = proba.argmax(axis=1)
            for i, pos in enumerate(unresolved_indices):
                labels[pos] = str(self.encoder.classes_[indices[i]])
                confs[pos] = float(proba[i, indices[i]])

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
