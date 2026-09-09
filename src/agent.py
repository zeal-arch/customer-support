"""A transparent retrieval baseline for the AppleSupport agent."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .text_preprocessing import normalize_for_classification


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
    ],
    "app_store_billing": [
        r"refund",
        r"charged",
        r"charge",
        r"subscription",
        r"purchase",
        r"payment",
        r"app store",
    ],
    "battery_power": [
        r"battery",
        r"charge",
        r"charging",
        r"overheat",
        r"won't turn on",
        r"won t turn on",
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
    ],
    "setup_transfer_sync": [
        r"backup",
        r"restore",
        r"transfer",
        r"sync",
        r"new iphone",
        r"new phone",
        r"move .* data",
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
    ],
    "ios_software_bug": [
        r"bug",
        r"glitch",
        r"freeze",
        r"crash",
        r"ios",
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
        r"apple pay",
    ],
    "how_to_settings": [
        r"how do i",
        r"how can i",
        r"where do i",
        r"how to",
        r"enable",
        r"turn on",
        r"settings",
    ],
}


@dataclass
class AgentResult:
    intent: str
    reply: str
    escalate: bool
    reason: str
    similarity: float
    evidence: list[str]


def classify_intent(text: str) -> tuple[str, float]:
    normalized = normalize_for_classification(text)
    scores = {
        label: sum(bool(re.search(pattern, normalized)) for pattern in patterns)
        for label, patterns in INTENT_PATTERNS.items()
    }
    label, score = max(scores.items(), key=lambda item: item[1])
    if score == 0:
        return "other_unclear", 0.0
    total = sum(scores.values())
    return label, score / max(total, 1)


class SupportAgent:
    def __init__(self, pairs: pd.DataFrame, top_k: int = 3) -> None:
        self.pairs = pairs.reset_index(drop=True)
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=3,
            max_features=80_000,
            strip_accents="unicode",
        )
        documents = self.pairs["customer_text"].map(normalize_for_classification)
        self.matrix = self.vectorizer.fit_transform(documents)

    def predict(self, text: str) -> AgentResult:
        query = self.vectorizer.transform([normalize_for_classification(text)])
        similarities = (self.matrix @ query.T).toarray().ravel()
        nearest = similarities.argsort()[::-1][: self.top_k]
        best = int(nearest[0])
        similarity = float(similarities[best])
        intent, intent_confidence = classify_intent(text)

        risky = intent in {"account_access_security", "app_store_billing"}
        unclear = intent == "other_unclear" or intent_confidence < 0.5
        weak_evidence = similarity < 0.20
        escalate = risky or unclear or weak_evidence

        if risky:
            reason = "Sensitive account or payment issue requires human review."
        elif unclear:
            reason = "The message is ambiguous or the intent confidence is low."
        elif weak_evidence:
            reason = "No sufficiently similar historical support example was found."
        else:
            reason = "The intent and historical evidence are sufficiently clear for a draft."

        evidence = [str(self.pairs.iloc[index]["customer_text"]) for index in nearest]
        return AgentResult(
            intent=intent,
            reply=str(self.pairs.iloc[best]["brand_reply"]),
            escalate=escalate,
            reason=reason,
            similarity=similarity,
            evidence=evidence,
        )
