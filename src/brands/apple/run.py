"""AppleSupport customer support agent CLI runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

try:
    from .agent import SupportAgent, classify_intent, decide_apple_escalation
    from .intent_classifier import IntentClassifier, build_from_pairs
    from ...text_preprocessing import (
        normalize_for_classification,
        extract_apple_entities,
        detect_urgency,
        clean_apple_reply,
    )
except (ImportError, ValueError):
    try:
        from agent import SupportAgent, classify_intent, decide_apple_escalation
        from intent_classifier import IntentClassifier, build_from_pairs
        from src.text_preprocessing import (
            normalize_for_classification,
            extract_apple_entities,
            detect_urgency,
            clean_apple_reply,
        )
    except (ImportError, ValueError):
        from src.brands.apple.agent import SupportAgent, classify_intent, decide_apple_escalation
        from src.brands.apple.intent_classifier import IntentClassifier, build_from_pairs
        from src.text_preprocessing import (
            normalize_for_classification,
            extract_apple_entities,
            detect_urgency,
            clean_apple_reply,
        )

ROOT_DIR = Path(__file__).resolve().parents[3]
DEFAULT_PAIRS = (
    ROOT_DIR / "data" / "processed" / "brands" / "applesupport" / "applesupport_pairs.csv"
    if (ROOT_DIR / "data" / "processed" / "brands" / "applesupport" / "applesupport_pairs.csv").exists()
    else ROOT_DIR / "data" / "processed" / "applesupport_pairs.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS,
                        help="Path to preprocessed AppleSupport pairs CSV.")
    parser.add_argument("--text", required=True, help="Customer tweet text.")
    parser.add_argument("--handle", default="", help="Optional customer Twitter handle to personalize reply.")
    parser.add_argument("--sample-size", type=int, default=40_000,
                        help="Pairs rows for training intent classifier.")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


class AppleSupportAgent:
    """High-accuracy AppleSupport Agent with intent-conditioned RAG and safety gating."""

    def __init__(self, pairs: pd.DataFrame, sample_size: int = 40_000, top_k: int = 3) -> None:
        self.pairs = pairs.reset_index(drop=True)
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=3,
            max_features=80_000,
            strip_accents="unicode",
        )
        documents = self.pairs["customer_text"].fillna("").map(normalize_for_classification)
        self.matrix = self.vectorizer.fit_transform(documents)
        self.classifier = build_from_pairs(self.pairs, sample=sample_size)

    def predict(self, text: str, handle: str = "") -> dict:
        entities = extract_apple_entities(text)
        urgency = detect_urgency(text)
        intent, confidence = self.classifier.predict(text)

        norm_query = normalize_for_classification(text)
        query_vec = self.vectorizer.transform([norm_query])
        raw_similarities = (self.matrix @ query_vec.T).toarray().ravel()

        candidate_pool_size = min(60, len(self.pairs))
        candidate_indices = raw_similarities.argsort()[::-1][:candidate_pool_size]

        target_devices = set(d.lower() for d in entities["devices"])
        target_os = set(os_v.lower() for os_v in entities["os_versions"])
        target_components = set(c.lower() for c in entities["components"])

        scored_candidates: list[tuple[int, float, float]] = []
        for idx in candidate_indices:
            base_sim = float(raw_similarities[idx])
            cand_text = str(self.pairs.iloc[idx]["customer_text"]).lower()

            boost = 0.0
            if any(dev in cand_text for dev in target_devices):
                boost += 0.12
            if any(os_ver in cand_text for os_ver in target_os):
                boost += 0.08
            if any(comp in cand_text for comp in target_components):
                boost += 0.06

            intent_kw_map = {
                "battery_power": ["battery", "charge", "drain", "power"],
                "app_store_billing": ["refund", "charge", "subscription", "purchase", "bill"],
                "account_access_security": ["apple id", "password", "lock", "hack", "login"],
                "connectivity_network": ["wifi", "wi-fi", "bluetooth", "cellular", "data", "sim"],
                "ios_software_bug": ["ios", "update", "freeze", "crash", "bug", "glitch"],
                "hardware_accessory": ["screen", "display", "camera", "adapter", "cable", "repair"],
                "setup_transfer_sync": ["backup", "restore", "transfer", "sync", "icloud"],
                "app_service_issue": ["photos", "music", "imessage", "icloud", "safari"],
            }
            if any(kw in cand_text for kw in intent_kw_map.get(intent, [])):
                boost += 0.08

            adjusted_score = base_sim + boost
            scored_candidates.append((idx, adjusted_score, base_sim))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        best_idx, best_adjusted_score, best_raw_sim = scored_candidates[0]

        top_indices = [c[0] for c in scored_candidates[:self.top_k]]
        raw_reply = str(self.pairs.iloc[best_idx]["brand_reply"])
        cleaned_reply = clean_apple_reply(raw_reply, customer_handle=handle)
        evidence = [str(self.pairs.iloc[i]["customer_text"]) for i in top_indices]

        escalate, reason = decide_apple_escalation(
            text=text,
            intent=intent,
            confidence=confidence,
            top_similarity=best_raw_sim,
            urgency_level=urgency["level"],
        )

        return {
            "brand": "AppleSupport",
            "intent": intent,
            "confidence": round(confidence, 4),
            "entities": entities,
            "urgency": urgency,
            "reply": cleaned_reply,
            "escalate": escalate,
            "reason": reason,
            "similarity": round(best_raw_sim, 4),
            "adjusted_score": round(best_adjusted_score, 4),
            "evidence": evidence,
        }


def main() -> None:
    args = parse_args()
    if not args.pairs.exists():
        raise FileNotFoundError(f"Pairs file not found at {args.pairs}.")

    pairs = pd.read_csv(args.pairs, low_memory=False)
    agent = AppleSupportAgent(pairs, sample_size=args.sample_size, top_k=args.top_k)
    result = agent.predict(args.text, handle=args.handle)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
