"""Uber Support Agent with intent-conditioned RAG and deterministic safety escalation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from ...text_preprocessing import (
        normalize_for_classification,
        extract_uber_entities,
        detect_uber_urgency,
        clean_uber_reply,
    )
    from .intent_classifier import UberIntentClassifier, build_uber_from_pairs
except (ImportError, ValueError):
    try:
        from src.text_preprocessing import (
            normalize_for_classification,
            extract_uber_entities,
            detect_uber_urgency,
            clean_uber_reply,
        )
        from intent_classifier import UberIntentClassifier, build_uber_from_pairs
    except (ImportError, ValueError):
        from src.text_preprocessing import (
            normalize_for_classification,
            extract_uber_entities,
            detect_uber_urgency,
            clean_uber_reply,
        )
        from src.brands.uber.intent_classifier import UberIntentClassifier, build_uber_from_pairs


def decide_uber_escalation(
    text: str,
    intent: str,
    confidence: float = 1.0,
    top_similarity: float = 1.0,
    urgency_level: str = "STANDARD",
) -> tuple[bool, str]:
    """Safety and escalation decision logic for Uber Support."""
    low = str(text).lower().strip()

    # 1. Critical Safety & Driver Harassment -> Immediate Mandatory Escalation
    if intent == "safety_driver_conduct" or urgency_level == "CRITICAL_SAFETY":
        return True, "Critical driver safety, misconduct, or accident report requires immediate human agent response."

    # 2. Urgent Lost Valuables
    if intent == "lost_item_in_vehicle" and any(k in low for k in ["phone", "iphone", "wallet", "keys", "passport", "medication", "stranded"]):
        return True, "Urgent lost valuable or critical item in vehicle requires direct driver liaison."

    # 3. Unauthorized billing / Fraud
    if intent == "ride_fare_billing" and any(k in low for k in ["fraud", "unauthorized", "stolen card", "hacked", "extort"]):
        return True, "Potential fraudulent charge or account security dispute requires financial review."

    # 4. Contextless message fragments
    clean = re.sub(r"https?://\S+", "", low)
    clean = re.sub(r"@[A-Za-z0-9_]+", "", clean).strip()
    if clean in {"yes", "thanks", "done", "dm sent"} or len(clean.split()) <= 2:
        return True, "Non-actionable message fragment or resolution acknowledgement."

    # 5. Low confidence or weak historical precedent
    if confidence < 0.38:
        return True, "Customer query intent is ambiguous or below confidence threshold."

    if top_similarity < 0.16:
        return True, "No sufficiently similar historical support precedent found."

    return False, "Intent and historical evidence are sufficiently validated for an automated draft."


class UberSupportAgent:
    """High-performance Uber Support Agent."""

    def __init__(self, pairs: pd.DataFrame, sample_size: int = 35_000, top_k: int = 3) -> None:
        self.pairs = pairs.reset_index(drop=True)
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=3,
            max_features=60_000,
            strip_accents="unicode",
        )
        documents = self.pairs["customer_text"].fillna("").map(normalize_for_classification)
        self.matrix = self.vectorizer.fit_transform(documents)
        self.classifier = build_uber_from_pairs(self.pairs, sample=sample_size)

    def predict(self, text: str, handle: str = "") -> dict:
        entities = extract_uber_entities(text)
        urgency = detect_uber_urgency(text)
        intent, confidence = self.classifier.predict(text)

        norm_query = normalize_for_classification(text)
        query_vec = self.vectorizer.transform([norm_query])
        raw_similarities = (self.matrix @ query_vec.T).toarray().ravel()

        candidate_pool_size = min(50, len(self.pairs))
        candidate_indices = raw_similarities.argsort()[::-1][:candidate_pool_size]

        target_services = set(s.lower() for s in entities["services"])
        target_lost = set(item.lower() for item in entities["lost_items"])

        scored_candidates: list[tuple[int, float, float]] = []
        for idx in candidate_indices:
            base_sim = float(raw_similarities[idx])
            cand_text = str(self.pairs.iloc[idx]["customer_text"]).lower()

            boost = 0.0
            if any(srv in cand_text for srv in target_services):
                boost += 0.10
            if any(item in cand_text for item in target_lost):
                boost += 0.12

            intent_kw_map = {
                "safety_driver_conduct": ["unsafe", "driver", "rude", "accident", "crash"],
                "lost_item_in_vehicle": ["left", "lost", "phone", "wallet", "keys"],
                "uber_eats_delivery": ["eats", "food", "order", "delivery", "meal"],
                "pickup_cancellation_delay": ["cancel", "waiting", "eta", "pickup", "location"],
                "ride_fare_billing": ["charge", "fare", "overcharge", "refund", "toll", "surge"],
                "account_app_promo": ["promo", "code", "login", "account", "discount"],
            }
            if any(kw in cand_text for kw in intent_kw_map.get(intent, [])):
                boost += 0.08

            scored_candidates.append((idx, base_sim + boost, base_sim))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        best_idx, best_adj_score, best_raw_sim = scored_candidates[0]

        top_indices = [c[0] for c in scored_candidates[:self.top_k]]
        raw_reply = str(self.pairs.iloc[best_idx]["brand_reply"])
        cleaned_reply = clean_uber_reply(raw_reply, customer_handle=handle)
        evidence = [str(self.pairs.iloc[i]["customer_text"]) for i in top_indices]

        escalate, reason = decide_uber_escalation(
            text=text,
            intent=intent,
            confidence=confidence,
            top_similarity=best_raw_sim,
            urgency_level=urgency["level"],
        )

        return {
            "brand": "Uber_Support",
            "intent": intent,
            "confidence": round(confidence, 4),
            "entities": entities,
            "urgency": urgency,
            "reply": cleaned_reply,
            "escalate": escalate,
            "reason": reason,
            "similarity": round(best_raw_sim, 4),
            "adjusted_score": round(best_adj_score, 4),
            "evidence": evidence,
        }
