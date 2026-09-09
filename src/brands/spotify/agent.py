"""Spotify Support Agent with RAG and deterministic escalation."""

from __future__ import annotations

import re
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from .intent_classifier import SpotifyIntentClassifier, build_spotify_from_pairs
    from ...text_preprocessing import normalize_for_classification
except (ImportError, ValueError):
    from src.brands.spotify.intent_classifier import SpotifyIntentClassifier, build_spotify_from_pairs
    from src.text_preprocessing import normalize_for_classification


def decide_spotify_escalation(
    text: str,
    intent: str,
    confidence: float = 1.0,
    top_similarity: float = 1.0,
) -> tuple[bool, str]:
    low = str(text).lower().strip()

    # 1. Security & Compromised accounts
    if intent == "account_login_security" and any(k in low for k in ["hacked", "stolen", "unauthorized", "compromised", "locked out"]):
        return True, "Account takeover or security compromise requires direct authentication review."

    # 2. Billing & Unauthorized charges
    if intent == "subscription_billing" and any(k in low for k in ["fraud", "unauthorized charge", "stolen card", "dispute", "charged twice"]):
        return True, "Billing dispute or potential unauthorized charge requires payment verification."

    # 3. Contextless fragments
    clean = re.sub(r"https?://\S+", "", low)
    clean = re.sub(r"@[A-Za-z0-9_]+", "", clean).strip()
    if clean in {"yes", "thanks", "done", "dm sent", "sent", "ok"} or len(clean.split()) <= 2:
        return True, "Non-actionable message fragment or resolution acknowledgement."

    # 4. Low confidence
    if confidence < 0.38:
        return True, "Customer query intent is ambiguous or below confidence threshold."

    if top_similarity < 0.16:
        return True, "No sufficiently similar historical support precedent found."

    return False, "Intent and historical evidence are sufficiently validated for an automated draft."


def clean_spotify_reply(reply: str, customer_handle: str = "") -> str:
    cleaned = re.sub(r"^@\w+\s*", "", reply.strip())
    if customer_handle:
        handle = customer_handle if customer_handle.startswith("@") else f"@{customer_handle}"
        cleaned = f"{handle} {cleaned}"
    return cleaned


class SpotifySupportAgent:
    def __init__(self, pairs: pd.DataFrame, sample_size: int = 35_000, top_k: int = 3) -> None:
        self.pairs = pairs.reset_index(drop=True)
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=3,
            max_features=50_000,
            strip_accents="unicode",
        )
        documents = self.pairs["customer_text"].fillna("").map(normalize_for_classification)
        self.matrix = self.vectorizer.fit_transform(documents)
        self.classifier = build_spotify_from_pairs(self.pairs, sample=sample_size)

    def predict(self, text: str, handle: str = "") -> dict:
        intent, confidence = self.classifier.predict(text)

        norm_query = normalize_for_classification(text)
        query_vec = self.vectorizer.transform([norm_query])
        raw_similarities = (self.matrix @ query_vec.T).toarray().ravel()

        candidate_pool_size = min(40, len(self.pairs))
        candidate_indices = raw_similarities.argsort()[::-1][:candidate_pool_size]

        scored_candidates: list[tuple[int, float, float]] = []
        for idx in candidate_indices:
            base_sim = float(raw_similarities[idx])
            cand_text = str(self.pairs.iloc[idx]["customer_text"]).lower()

            boost = 0.0
            if "spotify" in cand_text or "music" in cand_text or "app" in cand_text:
                boost += 0.05
            scored_candidates.append((idx, base_sim + boost, base_sim))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        best_idx, best_adj_score, best_raw_sim = scored_candidates[0]

        top_indices = [c[0] for c in scored_candidates[:self.top_k]]
        raw_reply = str(self.pairs.iloc[best_idx]["brand_reply"])
        cleaned_reply = clean_spotify_reply(raw_reply, customer_handle=handle)
        evidence = [str(self.pairs.iloc[i]["customer_text"]) for i in top_indices]

        escalate, reason = decide_spotify_escalation(
            text=text,
            intent=intent,
            confidence=confidence,
            top_similarity=best_raw_sim,
        )

        return {
            "brand": "SpotifyCares",
            "intent": intent,
            "confidence": round(confidence, 4),
            "reply": cleaned_reply,
            "escalate": escalate,
            "reason": reason,
            "similarity": round(best_raw_sim, 4),
            "evidence": evidence,
        }
