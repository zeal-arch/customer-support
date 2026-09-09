"""TF-IDF + Logistic Regression and Rule-Based Intent Classifier for AppleSupport.

Classifies incoming customer messages into 11 well-defined domain intents:
1. account_access_security
2. app_store_billing
3. battery_power
4. connectivity_network
5. setup_transfer_sync
6. hardware_accessory
7. ios_software_bug
8. app_service_issue
9. store_order_delivery
10. feedback_complaint
11. other_unclear
"""

from __future__ import annotations

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from src.text_preprocessing import normalize_for_classification

INTENT_PATTERNS = {
    "account_access_security": [
        r"apple id",
        r"password",
        r"hacked",
        r"fraud",
        r"phish",
        r"compromis",
        r"sign in",
        r"login",
        r"disabled",
        r"locked out",
        r"two factor",
        r"2fa",
    ],
    "app_store_billing": [
        r"refund",
        r"charged",
        r"charge",
        r"subscription",
        r"purchase",
        r"payment",
        r"app store",
        r"bill",
        r"unauthorized charge",
    ],
    "battery_power": [
        r"battery",
        r"drain",
        r"draining",
        r"charge",
        r"charging",
        r"overheat",
        r"won't turn on",
        r"won t turn on",
        r"percentage",
    ],
    "connectivity_network": [
        r"wi[- ]?fi",
        r"bluetooth",
        r"airdrop",
        r"handoff",
        r"cellular",
        r"mobile data",
        r"sim",
        r"connect",
        r"hotspot",
    ],
    "setup_transfer_sync": [
        r"backup",
        r"restore",
        r"transfer",
        r"sync",
        r"new iphone",
        r"new phone",
        r"move .* data",
        r"migration",
    ],
    "hardware_accessory": [
        r"screen",
        r"display",
        r"camera",
        r"headphone",
        r"charger",
        r"adapter",
        r"crack",
        r"replace",
        r"repair",
        r"speaker",
        r"microphone",
        r"airpods",
    ],
    "ios_software_bug": [
        r"bug",
        r"glitch",
        r"freeze",
        r"frozen",
        r"crash",
        r"stuck",
        r"boot loop",
        r"loop",
        r"restart",
        r"apple logo",
        r"ios\s*\d*",
        r"update",
        r"question mark",
        r"autocorrect",
        r"keyboard",
        r"not working",
        r"won't work",
        r"won t work",
    ],
    "app_service_issue": [
        r"icloud",
        r"imessage",
        r"message",
        r"photos",
        r"music",
        r"maps",
        r"safari",
        r"facetime",
        r"mail app",
    ],
    "store_order_delivery": [
        r"order",
        r"delivery",
        r"shipping",
        r"tracking",
        r"apple store",
        r"pick up",
        r"package",
    ],
    "feedback_complaint": [
        r"hate",
        r"worst",
        r"terrible",
        r"useless",
        r"disappointed",
        r"ridiculous",
        r"unacceptable",
        r"never buy",
    ],
}


def classify_intent_rules(text: str) -> tuple[str, float]:
    """Domain-precedence regex rule-based classifier."""
    low = normalize_for_classification(text).lower()

    # Precedence order: high-risk / specific services first
    precedence = [
        "account_access_security",
        "app_store_billing",
        "battery_power",
        "connectivity_network",
        "setup_transfer_sync",
        "hardware_accessory",
        "app_service_issue",
        "store_order_delivery",
        "feedback_complaint",
        "ios_software_bug",
    ]

    for intent in precedence:
        patterns = INTENT_PATTERNS[intent]
        for pat in patterns:
            if re.search(pat, low):
                return intent, 0.98

    return "other_unclear", 0.40


class IntentClassifier:
    """TF-IDF + Logistic Regression intent classifier with rule fallback."""

    def __init__(
        self,
        max_features: int = 80_000,
        ngram_range: tuple[int, int] = (1, 2),
        min_df: int = 2,
        C: float = 2.0,
        random_state: int = 42,
    ) -> None:
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            strip_accents="unicode",
            sublinear_tf=True,
        )
        self.encoder = LabelEncoder()
        self.clf = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            C=C,
            random_state=random_state,
            class_weight="balanced",
        )
        self.is_fitted = False

    def fit(self, texts: list[str], labels: list[str]) -> IntentClassifier:
        clean_texts = [normalize_for_classification(t) for t in texts]
        X = self.vectorizer.fit_transform(clean_texts)
        y = self.encoder.fit_transform(labels)
        self.clf.fit(X, y)
        self.is_fitted = True
        return self

    def predict_one(self, text: str) -> tuple[str, float]:
        # Always check high-confidence rule precedence first
        rule_intent, rule_conf = classify_intent_rules(text)
        if rule_conf >= 0.90:
            return rule_intent, rule_conf

        if not self.is_fitted:
            return rule_intent, rule_conf

        clean = normalize_for_classification(text)
        X = self.vectorizer.transform([clean])
        probs = self.clf.predict_proba(X)[0]
        best_idx = probs.argmax()
        intent = str(self.encoder.inverse_transform([best_idx])[0])
        confidence = float(probs[best_idx])

        if confidence < 0.35 and rule_intent != "other_unclear":
            return rule_intent, rule_conf

        return intent, confidence
