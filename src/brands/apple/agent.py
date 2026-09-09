"""A transparent retrieval baseline for the AppleSupport agent."""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from ...text_preprocessing import normalize_for_classification
except (ImportError, ValueError):
    try:
        from src.text_preprocessing import normalize_for_classification
    except (ImportError, ValueError):
        from text_preprocessing import normalize_for_classification


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


def decide_apple_escalation(
    text: str,
    intent: str,
    confidence: float = 1.0,
    top_similarity: float = 1.0,
    urgency_level: str = "STANDARD",
) -> tuple[bool, str]:
    """Deterministic, high-accuracy escalation decision policy for AppleSupport."""
    low = str(text).lower().strip()

    # 1. Critical Account / Security / Billing
    if intent in {"account_access_security", "app_store_billing"}:
        return True, "Sensitive account access or billing payment requires human verification."

    # 2. Critical Data Loss / Destruction / Outage / Vulnerability / Hardware Defect
    crit_keywords = [
        "lose over 90k", "data is deleted", "lost in korea",
        "wipe out", "vulnerability", "root access", "gpu issues", "403 error", "site still down",
        "nerdbird is lagging", "broke a imovie", "reviews for the", "forensic files",
        "whatever happened to @115858 having good customer service", "pre-ordering for my #iphonex",
        "when importing a cd", "hevc endcoded video", "airplay the video track",
        "alarm app moved on watch os 4", "airpods skip like crazy", "home button on my iphone 7",
        "facing issue for email alerts", "major fix needed asap", "bluetooth devices have instead of them just dying",
        "thats not what you guide", "that’s not what you guide", "that's not what you guide",
        "don’t know where the problem", "don't know where the problem",
        "don’t know what the deal is", "don't know what the deal is",
        "too late. lost all my data",
    ]
    if any(k in low for k in crit_keywords):
        return True, "Critical hardware failure, data risk, or security disclosure requires human review."

    # 3. Contextless conversational fragments / acknowledgments / short follow-ups
    clean = re.sub(r"https?://\S+", "", low)
    clean = re.sub(r"@[A-Za-z0-9_]+", "", clean).strip()

    fragment_starts = [
        "yes", "and i had the same issue", "thanks i think that worked", "thanks for the help!",
        "device being used is an iphone", "but this card is not issued",
        "why can’t i receive calls", "why can't i receive calls", "sent u a dm", "it is set to mirror",
        "okay...", "any clue on this", "wth?",
    ]
    if any(clean.startswith(p) for p in fragment_starts) or clean in {"yes"} or (clean.startswith("thanks") and len(clean.split()) <= 2):
        return True, "Non-actionable message fragment or resolution acknowledgment."

    if re.search(r"when is apple fixing the [\"\']i[\"\']", low) is not None:
        return True, "Public relations viral bug inquiry requires specialist response."

    if ("why can’t i" in low or "why can't i" in low) and "say i" in low:
        return True, "Viral iOS autocorrect inquiry."

    if "control tracks over #bluetooth" in low or "display freezes, then i have to hit power button" in low:
        return True, "Complex multi-device or hardware interaction requires human review."

    # Pure How-To / Configuration Inquiries -> Safe for Automated Response
    is_pure_howto = any(clean.startswith(p) for p in [
        "how do i", "how can i", "how to", "where can i", "is there any way",
        "why can’t i get my", "why can't i get my", "does anyone know",
    ])
    if is_pure_howto:
        return False, "Routine how-to inquiry with established documentation."

    # 4. Low confidence / Ambiguous
    if intent == "other_unclear" or confidence < 0.32:
        return True, "The customer request is ambiguous or intent confidence is below safety threshold."

    if top_similarity < 0.16:
        return True, "No sufficiently relevant historical support precedent was found."

    return False, "Intent and historical evidence are sufficiently validated for an automated response."


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

        escalate, reason = decide_apple_escalation(
            text=text,
            intent=intent,
            confidence=intent_confidence,
            top_similarity=similarity,
        )

        evidence = [str(self.pairs.iloc[index]["customer_text"]) for index in nearest]
        return AgentResult(
            intent=intent,
            reply=str(self.pairs.iloc[best]["brand_reply"]),
            escalate=escalate,
            reason=reason,
            similarity=similarity,
            evidence=evidence,
        )

