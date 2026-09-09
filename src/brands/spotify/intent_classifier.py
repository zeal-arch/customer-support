"""Intent Classifier for SpotifyCares (Rule Precedence + TF-IDF Logistic Regression)."""

from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

try:
    from .text_preprocessing import normalize_for_classification
except (ImportError, ValueError):
    from src.text_preprocessing import normalize_for_classification

SPOTIFY_INTENTS = [
    "playback_music_streaming",
    "subscription_billing",
    "account_login_security",
    "device_app_compatibility",
    "artist_content_podcast",
]

INTENT_RULES = {
    "subscription_billing": [
        r"\b(?:charge|charged|billing|refund|receipt|subscription|premium|family plan|duo|student discount|renew|cancel subscription|trial)\b",
        r"\b(?:payment method|card declined|double charge|cost)\b",
    ],
    "account_login_security": [
        r"\b(?:login|log in|password|reset password|locked out|hacked|unauthorized|username|email change|verify)\b",
        r"\b(?:can't log into|account compromised|recover account)\b",
    ],
    "device_app_compatibility": [
        r"\b(?:bluetooth|carplay|android auto|chromecast|alexa|google home|sonos|smart tv|watch|airplay)\b",
        r"\b(?:app crash|app freezing|won't open|black screen|update app|ios|android)\b",
    ],
    "playback_music_streaming": [
        r"\b(?:stop playing|keeps pausing|skipping|buffering|offline|download|greyed out|won't play|can't play)\b",
        r"\b(?:playlist|queue|shuffle|repeat|audio quality|sound)\b",
    ],
    "artist_content_podcast": [
        r"\b(?:song missing|album|artist|podcast|episode|lyrics|canvas|track|explicit)\b",
    ],
}


def classify_spotify_intent_rules(text: str) -> tuple[str, float] | None:
    low = str(text).lower()

    if any(re.search(pat, low) for pat in INTENT_RULES["account_login_security"]):
        return "account_login_security", 0.99

    if any(re.search(pat, low) for pat in INTENT_RULES["subscription_billing"]):
        return "subscription_billing", 0.98

    if any(re.search(pat, low) for pat in INTENT_RULES["device_app_compatibility"]):
        return "device_app_compatibility", 0.96

    if any(re.search(pat, low) for pat in INTENT_RULES["playback_music_streaming"]):
        return "playback_music_streaming", 0.96

    if any(re.search(pat, low) for pat in INTENT_RULES["artist_content_podcast"]):
        return "artist_content_podcast", 0.95

    return None


class SpotifyIntentClassifier:
    def __init__(self, random_state: int = 42) -> None:
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=3,
            max_features=50_000,
            strip_accents="unicode",
        )
        self.encoder = LabelEncoder()
        self.clf = LogisticRegression(max_iter=1000, C=3.0, class_weight="balanced", random_state=random_state)
        self._fitted = False

    def fit(self, texts: pd.Series, labels: pd.Series) -> "SpotifyIntentClassifier":
        norm = texts.fillna("").map(normalize_for_classification)
        X = self.vectorizer.fit_transform(norm)
        y = self.encoder.fit_transform(labels)
        self.clf.fit(X, y)
        self._fitted = True
        return self

    def predict(self, text: str) -> tuple[str, float]:
        rule_match = classify_spotify_intent_rules(text)
        if rule_match is not None:
            return rule_match

        if not self._fitted:
            return "playback_music_streaming", 0.50

        norm = normalize_for_classification(text)
        X = self.vectorizer.transform([norm])
        proba = self.clf.predict_proba(X)[0]
        idx = proba.argmax()
        return str(self.encoder.classes_[idx]), float(proba[idx])


def build_spotify_from_pairs(
    pairs: pd.DataFrame,
    sample: int = 35_000,
    random_state: int = 42,
) -> SpotifyIntentClassifier:
    corpus = pairs.sample(min(sample, len(pairs)), random_state=random_state).copy()
    results = [classify_spotify_intent_rules(t) for t in corpus["customer_text"].fillna("")]
    corpus["intent"] = [r[0] if r else "playback_music_streaming" for r in results]
    corpus["conf"] = [r[1] if r else 0.40 for r in results]

    valid = corpus[corpus["conf"] >= 0.60].copy()
    clf = SpotifyIntentClassifier(random_state=random_state)
    clf.fit(valid["customer_text"], valid["intent"])
    return clf
