"""Uber-specific text preprocessing and entity extraction."""

from __future__ import annotations

import html
import re

URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)
REPEAT_PUNCT_PATTERN = re.compile(r"([!?.]){2,}")
WHITESPACE_PATTERN = re.compile(r"\s+")

_CHAT_ABBREVS: dict[str, str] = {
    "brb": "be right back",
    "wtf": "what the",
    "omg": "oh my",
    "lol": "laughing",
    "imho": "in my opinion",
    "imo": "in my opinion",
    "fyi": "for your information",
    "asap": "as soon as possible",
    "thx": "thank you",
    "ty": "thank you",
    "pls": "please",
    "plz": "please",
    "ur": "your",
    "u": "you",
    "r": "are",
    "w/": "with",
    "w/o": "without",
    "idk": "i do not know",
    "dm": "direct message",
    "dms": "direct messages",
    "eta": "estimated arrival time",
}

UBER_SERVICES = [
    (r"\buber\s*eats\b", "Uber Eats"),
    (r"\buber\s*x(?:l)?\b", "UberX / XL"),
    (r"\buber\s*black\b", "Uber Black"),
    (r"\buber\s*pool\b", "Uber Pool"),
    (r"\buber\s*comfort\b", "Uber Comfort"),
    (r"\buber\s*auto\b", "Uber Auto"),
]

UBER_LOST_ITEMS = [
    (r"\b(?:phone|iphone|android|cell\s*phone|mobile)\b", "Phone / Device"),
    (r"\b(?:wallet|purse|cardholder|money\s*clip)\b", "Wallet / Purse"),
    (r"\b(?:keys?|house\s*keys?|car\s*keys?)\b", "Keys"),
    (r"\b(?:bag|backpack|luggage|suitcase|duffel)\b", "Bag / Luggage"),
    (r"\b(?:jacket|coat|sweater|hoodie|umbrella|glasses)\b", "Clothing / Accessory"),
    (r"\b(?:passport|id|driver\s*license|documents?)\b", "ID / Documents"),
]


def normalize_for_classification(text: object) -> str:
    """Normalization pipeline for Uber customer tweets."""
    value = html.unescape("" if text is None else str(text))
    value = URL_PATTERN.sub(" URL ", value)
    value = MENTION_PATTERN.sub(" USER ", value)
    value = EMOJI_PATTERN.sub(" ", value)
    value = REPEAT_PUNCT_PATTERN.sub(r"\1", value)
    value = value.lower().strip()
    words = value.split()
    value = " ".join(_CHAT_ABBREVS.get(w, w) for w in words)
    return WHITESPACE_PATTERN.sub(" ", value).strip()


def extract_uber_entities(text: str) -> dict:
    """Extract ride-sharing services, lost items, and trip elements."""
    low = str(text).lower()
    entities: dict[str, list[str]] = {"services": [], "lost_items": [], "trip_elements": []}

    for pat, label in UBER_SERVICES:
        if re.search(pat, low):
            entities["services"].append(label)

    for pat, label in UBER_LOST_ITEMS:
        if re.search(pat, low):
            entities["lost_items"].append(label)

    if any(k in low for k in ["driver", "chauffeur", "captain"]):
        entities["trip_elements"].append("Driver")

    return entities


def detect_uber_urgency(text: str) -> dict:
    """Detect critical driver misconduct, accidents, or harassment."""
    low = text.lower()
    safety_triggers = ["accident", "crashed", "unsafe", "drunk", "harass", "assault", "police", "threaten", "emergency"]
    if any(k in low for k in safety_triggers):
        return {"level": "CRITICAL_SAFETY", "score": 1.0, "triggers": ["Safety trigger"]}
    return {"level": "STANDARD", "score": 0.0, "triggers": []}


def clean_uber_reply(reply: str, customer_handle: str = "") -> str:
    """Sanitize and personalize Uber customer reply."""
    cleaned = re.sub(r"^@\w+\s*", "", reply.strip())
    if customer_handle:
        handle = customer_handle if customer_handle.startswith("@") else f"@{customer_handle}"
        cleaned = f"{handle} {cleaned}"
    return cleaned
