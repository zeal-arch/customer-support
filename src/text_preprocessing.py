"""Unified Text Preprocessing and Domain Entity Extraction for Multi-Brand Customer Support.

This single module serves the entire TWCS dataset and all brand agents:
- Universal Normalization (HTML unescaping, URL/Mention masking, Emoji stripping, Chat expansion, Lemmatization)
- Domain Entity Extractors (Apple, Uber, Amazon, Spotify, Airlines)
- Urgency & Critical Safety Detectors
- Handle Personalization & Reply Sanitizers
"""

from __future__ import annotations

import html
import re
from typing import Dict, List, Any

# ── Optional NLTK lemmatizer ─────────────────────────────────────────────────
try:
    from nltk.stem import WordNetLemmatizer
    import nltk

    nltk.data.find("corpora/wordnet")
    _LEMMATIZER = WordNetLemmatizer()
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False
    _LEMMATIZER = None  # type: ignore


# ── Compiled Core Regex Patterns ─────────────────────────────────────────────
URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
MENTION_PATTERN = re.compile(r"@[A-Za-z0-9_]+")
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport & map
    "\U0001F1E0-\U0001F1FF"   # flags
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


# ── Universal Normalization Functions ────────────────────────────────────────

def normalize_for_classification(text: object, lemmatize: bool = False) -> str:
    """Standardized normalization across all TWCS customer support queries."""
    value = html.unescape("" if text is None else str(text))
    value = URL_PATTERN.sub(" URL ", value)
    value = MENTION_PATTERN.sub(" USER ", value)
    value = EMOJI_PATTERN.sub(" ", value)
    value = REPEAT_PUNCT_PATTERN.sub(r"\1", value)
    value = value.lower().strip()

    words = value.split()
    words = [_CHAT_ABBREVS.get(w, w) for w in words]

    if lemmatize and _NLTK_AVAILABLE and _LEMMATIZER is not None:
        words = [_LEMMATIZER.lemmatize(w) for w in words]

    value = " ".join(words)
    return WHITESPACE_PATTERN.sub(" ", value).strip()


def normalize_for_retrieval(text: object) -> str:
    """Normalization pipeline for TF-IDF cosine similarity search."""
    return normalize_for_classification(text, lemmatize=False)


def clean_tweet_text(text: str) -> str:
    """Fast cleaner that strips URLs, user handles, and excessive whitespace."""
    t = str(text).lower()
    t = URL_PATTERN.sub("", t)
    t = MENTION_PATTERN.sub("", t)
    return WHITESPACE_PATTERN.sub(" ", t).strip()


def original_text(text: object) -> str:
    """Return a safe string representation without modifying characters."""
    return "" if text is None else str(text).strip()


# ── Apple Entity & Urgency Extraction ────────────────────────────────────────

APPLE_DEVICES = [
    (r"\biphone\s*(?:x[sr]?|1[1-5](?:\s*pro(?:\s*max)?)?|[6-8](?:\s*plus)?|se|5[sc]?)\b", "iPhone"),
    (r"\bipad\s*(?:pro|air|mini)?\b", "iPad"),
    (r"\bmacbook\s*(?:pro|air)?\b", "MacBook"),
    (r"\bapple\s*watch(?:\s*series\s*[1-9])?\b", "Apple Watch"),
    (r"\bairpods\s*(?:pro|max)?\b", "AirPods"),
    (r"\bimac(?:\s*pro)?\b", "iMac"),
    (r"\bmac\s*(?:mini|studio|pro)\b", "Mac Desktop"),
]

APPLE_OS = [
    (r"\bios\s*(?:1[0-7](?:\.[0-9]+)*|[6-9](?:\.[0-9]+)*)\b", "iOS"),
    (r"\bmacos\s*(?:high\s*sierra|sierra|mojave|catalina|big\s*sur|monterey|ventura|sonoma)?\b", "macOS"),
    (r"\bwatchos\s*[0-9.]*\b", "watchOS"),
]


def extract_apple_entities(text: str) -> dict:
    low = str(text).lower()
    entities: dict[str, list[str]] = {"devices": [], "os_versions": [], "components": []}

    for pat, label in APPLE_DEVICES:
        if re.search(pat, low):
            entities["devices"].append(label)

    for pat, label in APPLE_OS:
        if re.search(pat, low):
            entities["os_versions"].append(label)

    for comp in ["battery", "screen", "camera", "bluetooth", "wifi", "icloud", "apple id", "sim"]:
        if comp in low:
            entities["components"].append(comp)

    return entities


def detect_urgency(text: str) -> dict:
    low = text.lower()
    high_urgency = ["urgent", "asap", "emergency", "immediately", "locked out", "stolen", "hacked", "fraud", "compromised"]
    
    if any(k in low for k in high_urgency):
        return {"level": "CRITICAL", "signals": ["Urgent keyword"]}
    
    caps = len(re.findall(r"[A-Z]{3,}", text))
    if caps >= 2 or "!!!" in text:
        return {"level": "HIGH", "signals": ["Frustration/Caps"]}

    return {"level": "STANDARD", "signals": []}


def clean_apple_reply(reply: str, customer_handle: str = "") -> str:
    cleaned = re.sub(r"^@\w+\s*", "", reply.strip())
    if customer_handle:
        handle = customer_handle if customer_handle.startswith("@") else f"@{customer_handle}"
        cleaned = f"{handle} {cleaned}"
    return cleaned


# ── Uber Entity & Urgency Extraction ─────────────────────────────────────────

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


def extract_uber_entities(text: str) -> dict:
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
    low = text.lower()
    safety_triggers = ["accident", "crashed", "unsafe", "drunk", "harass", "assault", "police", "threaten", "emergency"]
    if any(k in low for k in safety_triggers):
        return {"level": "CRITICAL_SAFETY", "score": 1.0, "triggers": ["Safety trigger"]}
    return {"level": "STANDARD", "score": 0.0, "triggers": []}


def clean_uber_reply(reply: str, customer_handle: str = "") -> str:
    cleaned = re.sub(r"^@\w+\s*", "", reply.strip())
    if customer_handle:
        handle = customer_handle if customer_handle.startswith("@") else f"@{customer_handle}"
        cleaned = f"{handle} {cleaned}"
    return cleaned
