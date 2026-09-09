"""AmazonHelp intent classification engine and safety escalation rules."""

from __future__ import annotations

import re
from typing import Tuple

try:
    from ...text_preprocessing import normalize_for_classification
except (ImportError, ValueError):
    try:
        from src.text_preprocessing import normalize_for_classification
    except (ImportError, ValueError):
        from text_preprocessing import normalize_for_classification


AMAZON_INTENT_PATTERNS = {
    "account_billing_security": [
        r"charged twice",
        r"double charge",
        r"unauthorized charge",
        r"unknown charge",
        r"credit card",
        r"bank account",
        r"fraud",
        r"hacked",
        r"compromis",
        r"scam",
        r"phish",
        r"gift card",
        r"otp",
        r"password",
        r"locked out",
        r"login",
        r"sign in",
    ],
    "damaged_defective_item": [
        r"damaged",
        r"broken",
        r"shattered",
        r"defective",
        r"faulty",
        r"cracked",
        r"duplicate",
        r"counterfeit",
        r"fake",
        r"missing item",
        r"empty box",
        r"wrong item",
    ],
    "refund_return": [
        r"refund",
        r"money back",
        r"return",
        r"returning",
        r"send back",
        r"exchange",
        r"reimburse",
    ],
    "order_modification_cancel": [
        r"cancel.*order",
        r"cancellation",
        r"cancel it",
        r"change.*address",
        r"wrong address",
        r"modify.*order",
        r"preorder",
        r"change my order",
    ],
    "delivery_tracking": [
        r"delivery",
        r"deliver",
        r"package",
        r"parcel",
        r"shipment",
        r"shipping",
        r"courier",
        r"track",
        r"tracking",
        r"delayed",
        r"delay",
        r"late",
        r"out for delivery",
        r"not arrived",
        r"haven't received",
        r"where is my",
        r"eta",
    ],
    "prime_digital_services": [
        r"prime video",
        r"prime music",
        r"prime membership",
        r"subtitles",
        r"stream",
        r"streaming",
        r"kindle",
        r"audible",
        r"echo",
        r"alexa",
        r"fire tv",
        r"fire stick",
    ],
}

ESCALATE_ALWAYS_AMAZON = {"account_billing_security", "damaged_defective_item"}


def classify_amazon_intent(text: str) -> Tuple[str, float]:
    """Classify an incoming Amazon customer inquiry into an e-commerce intent."""
    normalized = normalize_for_classification(text)
    
    # Check domain precedence: security and billing first
    for intent in ["account_billing_security", "damaged_defective_item", "refund_return"]:
        for pattern in AMAZON_INTENT_PATTERNS[intent]:
            if re.search(pattern, normalized):
                return intent, 0.95

    scores = {
        label: sum(bool(re.search(pattern, normalized)) for pattern in patterns)
        for label, patterns in AMAZON_INTENT_PATTERNS.items()
    }
    label, score = max(scores.items(), key=lambda item: item[1])
    if score == 0:
        return "other_general", 0.0
    total = sum(scores.values())
    return label, score / max(total, 1)


def decide_amazon_escalation(intent: str, confidence: float, similarity: float) -> Tuple[bool, str]:
    """Decide whether to escalate an Amazon query to a human agent."""
    if intent in ESCALATE_ALWAYS_AMAZON:
        return True, f"Sensitive inquiry ({intent}) requires human verification."
    if intent == "other_general" or confidence < 0.45:
        return True, "Ambiguous inquiry or low classification confidence."
    if similarity < 0.35:
        return True, "No sufficiently similar historical Amazon resolution found."
    return False, "Clear intent and strong historical resolution evidence."
