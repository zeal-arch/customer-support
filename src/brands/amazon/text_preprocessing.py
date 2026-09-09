"""AmazonHelp-specific text preprocessing."""

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


def normalize_for_classification(text: object) -> str:
    value = html.unescape("" if text is None else str(text))
    value = URL_PATTERN.sub(" URL ", value)
    value = MENTION_PATTERN.sub(" USER ", value)
    value = EMOJI_PATTERN.sub(" ", value)
    value = REPEAT_PUNCT_PATTERN.sub(r"\1", value)
    return WHITESPACE_PATTERN.sub(" ", value.lower()).strip()
