"""Text Preprocessing and Entity Extraction for AppleSupport AI Customer Support.

Normalizes incoming tweets and extracts domain entities (devices, OS versions,
hardware components) and detects customer urgency.
"""

from __future__ import annotations

import html
import re

# Optional NLTK lemmatizer (falls back to clean tokenization if not installed)
try:
    import nltk
    from nltk.stem import WordNetLemmatizer

    nltk.data.find("corpora/wordnet")
    _LEMMATIZER = WordNetLemmatizer()
    _NLTK_AVAILABLE = True
except Exception:
    _NLTK_AVAILABLE = False
    _LEMMATIZER = None

# Core Regex Patterns
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

# Apple Hardware & OS Entity Dictionaries
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


def normalize_for_classification(text: object, lemmatize: bool = False) -> str:
    """Standardized text normalization for AppleSupport intent classification."""
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
    """Fast cleaner that strips URLs, handles, and excessive whitespace."""
    t = str(text).lower()
    t = URL_PATTERN.sub("", t)
    t = MENTION_PATTERN.sub("", t)
    return WHITESPACE_PATTERN.sub(" ", t).strip()


def extract_apple_entities(text: str) -> dict:
    """Extract Apple hardware devices, operating system versions, and components."""
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
    """Detect urgency, frustration, or critical safety keywords in customer tweets."""
    low = text.lower()
    high_urgency = [
        "urgent", "asap", "emergency", "immediately", "locked out",
        "stolen", "hacked", "fraud", "compromised", "unauthorized",
    ]

    if any(k in low for k in high_urgency):
        return {"level": "CRITICAL", "signals": ["Urgent keyword"]}

    caps = len(re.findall(r"[A-Z]{3,}", text))
    if caps >= 2 or "!!!" in text:
        return {"level": "HIGH", "signals": ["Frustration/Caps"]}

    return {"level": "STANDARD", "signals": []}


def clean_apple_reply(reply: str, customer_handle: str = "") -> str:
    """Sanitize historical reply and attach customer handle."""
    cleaned = re.sub(r"^@\w+\s*", "", reply.strip())
    if customer_handle:
        handle = customer_handle if customer_handle.startswith("@") else f"@{customer_handle}"
        cleaned = f"{handle} {cleaned}"
    return cleaned
