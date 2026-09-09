"""Intent Classifier for Uber Support (Domain Precedence + TF-IDF Logistic Regression)."""

from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

try:
    from ...text_preprocessing import normalize_for_classification
except (ImportError, ValueError):
    try:
        from src.text_preprocessing import normalize_for_classification
    except (ImportError, ValueError):
        from text_preprocessing import normalize_for_classification

UBER_INTENTS = [
    "safety_driver_conduct",
    "lost_item_in_vehicle",
    "uber_eats_delivery",
    "pickup_cancellation_delay",
    "ride_fare_billing",
    "account_app_promo",
]

INTENT_RULES = {
    "safety_driver_conduct": [
        r"\b(?:accident|crashed|crash|unsafe|drunk|drinking|harass|assault|threat|threaten)\b",
        r"\b(?:reckless|speeding|ran red light|extort|extortion|police|emergency|creepy)\b",
        r"\b(?:wrong driver|different car|license plate.*match|refused.*service animal)\b",
        r"\b(?:unprofessional|rude driver|yelling|screaming at me|cursed at me)\b",
        r"\b(?:smelled like|dirty car|vomit|car broke down)\b",
    ],
    "lost_item_in_vehicle": [
        r"\b(?:left my|lost my|forgot my|left behind)\b",
        r"\b(?:lost item|found item|item in.*car|phone in.*car|wallet in.*car|keys in.*car)\b",
        r"\b(?:contact.*driver.*lost|recover.*item)\b",
    ],
    "uber_eats_delivery": [
        r"\b(?:uber\s*eats|food|order|restaurant|meal|delivered|delivery|eats)\b",
        r"\b(?:missing item|wrong food|cold food|spilled|eats promo|never delivered food)\b",
    ],
    "pickup_cancellation_delay": [
        r"\b(?:driver cancel|driver cancelled|driver never came|driver didn't show|driver drove away)\b",
        r"\b(?:driver refused|wrong pickup|pickup location|eta.*longer|waiting.*minutes)\b",
        r"\b(?:can't find driver|cannot find driver|driver going wrong way)\b",
    ],
    "ride_fare_billing": [
        r"\b(?:overcharge|overcharged|cancellation fee|cancel fee|charged me|charge me)\b",
        r"\b(?:double charge|charged twice|refund|fare was|fare estimate|surge|toll)\b",
        r"\b(?:tip|receipt|payment method|unauthorized charge|charged without)\b",
    ],
    "account_app_promo": [
        r"\b(?:promo code|promotion|discount|coupon|code not working|voucher)\b",
        r"\b(?:login|log in|locked out|password|verify phone|phone number|account disabled)\b",
        r"\b(?:app crashed|app glitch|app error|update payment)\b",
    ],
}


def classify_uber_intent_rules(text: str) -> tuple[str, float] | None:
    """High-precision rule precedence for Uber customer issues."""
    low = str(text).lower()

    # 1. Critical Safety & Driver Conduct
    if any(re.search(pat, low) for pat in INTENT_RULES["safety_driver_conduct"]):
        return "safety_driver_conduct", 0.99

    # 2. Lost Items
    if any(re.search(pat, low) for pat in INTENT_RULES["lost_item_in_vehicle"]):
        return "lost_item_in_vehicle", 0.98

    # 3. Uber Eats Food Delivery
    if any(k in low for k in ["ubereats", "uber eats", "food", "restaurant", "meal", "order"]):
        if any(re.search(pat, low) for pat in INTENT_RULES["uber_eats_delivery"]):
            return "uber_eats_delivery", 0.97

    # 4. Pickup & Cancellation Delays
    if any(re.search(pat, low) for pat in INTENT_RULES["pickup_cancellation_delay"]):
        return "pickup_cancellation_delay", 0.96

    # 5. Fares, Charges & Billing
    if any(re.search(pat, low) for pat in INTENT_RULES["ride_fare_billing"]):
        return "ride_fare_billing", 0.97

    # 6. Promo, App & Account
    if any(re.search(pat, low) for pat in INTENT_RULES["account_app_promo"]):
        return "account_app_promo", 0.95

    return None


class UberIntentClassifier:
    """Hybrid Classifier for Uber Support."""

    def __init__(self, random_state: int = 42) -> None:
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=3,
            max_features=60_000,
            strip_accents="unicode",
        )
        self.encoder = LabelEncoder()
        self.clf = LogisticRegression(max_iter=1000, C=3.0, class_weight="balanced", random_state=random_state)
        self._fitted = False

    def fit(self, texts: pd.Series, labels: pd.Series) -> "UberIntentClassifier":
        norm = texts.fillna("").map(normalize_for_classification)
        X = self.vectorizer.fit_transform(norm)
        y = self.encoder.fit_transform(labels)
        self.clf.fit(X, y)
        self._fitted = True
        return self

    def predict(self, text: str) -> tuple[str, float]:
        rule_match = classify_uber_intent_rules(text)
        if rule_match is not None:
            return rule_match

        if not self._fitted:
            return "ride_fare_billing", 0.50

        norm = normalize_for_classification(text)
        X = self.vectorizer.transform([norm])
        proba = self.clf.predict_proba(X)[0]
        idx = proba.argmax()
        return str(self.encoder.classes_[idx]), float(proba[idx])


def build_uber_from_pairs(
    pairs: pd.DataFrame,
    sample: int = 35_000,
    random_state: int = 42,
) -> UberIntentClassifier:
    """Train Uber classifier using high-confidence rule bootstrapping."""
    corpus = pairs.sample(min(sample, len(pairs)), random_state=random_state).copy()
    results = [classify_uber_intent_rules(t) for t in corpus["customer_text"].fillna("")]
    
    corpus["intent"] = [r[0] if r else "ride_fare_billing" for r in results]
    corpus["conf"] = [r[1] if r else 0.40 for r in results]
    
    valid = corpus[corpus["conf"] >= 0.60].copy()
    clf = UberIntentClassifier(random_state=random_state)
    clf.fit(valid["customer_text"], valid["intent"])
    return clf
