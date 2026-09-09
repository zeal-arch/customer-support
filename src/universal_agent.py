"""Universal Multi-Brand Customer Support Agent.

Automatically identifies target brand from tweet text, handles, or context,
and routes to the optimal brand-specific agent or cross-brand semantic index.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

BRANDS_DATA_DIR = ROOT_DIR / "data" / "processed" / "brands"

from src.text_preprocessing import normalize_for_classification
from src.brands.apple.run import AppleSupportAgent
from src.brands.uber.run import UberSupportAgent
from src.brands.amazon.run import AmazonSupportAgent
from src.brands.spotify.run import SpotifySupportAgent

BRAND_PATTERNS = [
    (r"(?i)@?applesupport|iphone|ipad|macbook|ios\s*\d|apple\s*watch|apple\s*id|airpods", "AppleSupport"),
    (r"(?i)@?uber_support|@?uber\b|uber\s*eats|ubereats|uber\s*driver", "Uber_Support"),
    (r"(?i)@?amazonhelp|@?amazon\b|amazon\s*prime|kindle|fire\s*tv|amazon\s*delivery", "AmazonHelp"),
    (r"(?i)@?spotifycares|@?spotify\b|spotify\s*premium|playlist|stream\s*music", "SpotifyCares"),
    (r"(?i)@?delta|delta\s*flight|delta\s*air", "Delta"),
    (r"(?i)@?americanair|american\s*airlines", "AmericanAir"),
    (r"(?i)@?british_airways|british\s*airways", "British_Airways"),
    (r"(?i)@?tmobilehelp|t-mobile|tmobile", "TMobileHelp"),
    (r"(?i)@?ask_spectrum|spectrum\s*internet|charter\s*spectrum", "Ask_Spectrum"),
    (r"(?i)@?xboxsupport|xbox\s*live|xbox\s*one|game\s*pass", "XboxSupport"),
    (r"(?i)@?askplaystation|playstation|ps4|ps5|psn\s*network", "AskPlayStation"),
    (r"(?i)@?upshelp|ups\s*tracking|ups\s*delivery|ups\s*package", "UPSHelp"),
    (r"(?i)@?askpaypal|paypal\s*transfer|paypal\s*credit", "AskPayPal"),
    (r"(?i)@?tesco|tesco\s*grocery|tesco\s*delivery", "Tesco"),
]


class GenericBrandSupportAgent:
    """Dynamic fallback RAG agent for any of the 108 brands in TWCS."""

    def __init__(self, brand_name: str, pairs_path: Path, top_k: int = 3) -> None:
        self.brand_name = brand_name
        self.top_k = top_k
        self.pairs = pd.read_csv(pairs_path, low_memory=False).reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_features=40_000,
            strip_accents="unicode",
        )
        documents = self.pairs["customer_text"].fillna("").map(normalize_for_classification)
        self.matrix = self.vectorizer.fit_transform(documents)

    def predict(self, text: str, handle: str = "") -> dict:
        query_vec = self.vectorizer.transform([normalize_for_classification(text)])
        sims = (self.matrix @ query_vec.T).toarray().ravel()
        best_idx = int(sims.argmax())
        top_sim = float(sims[best_idx])

        reply = str(self.pairs.iloc[best_idx]["brand_reply"])
        cleaned_reply = re.sub(r"^@\w+\s*", "", reply.strip())
        if handle:
            h = handle if handle.startswith("@") else f"@{handle}"
            cleaned_reply = f"{h} {cleaned_reply}"

        low = str(text).lower()
        escalate = False
        reason = "Automated high-relevance support draft."

        if any(k in low for k in ["fraud", "hacked", "stolen", "emergency", "police", "legal", "court", "threat"]):
            escalate = True
            reason = "Security, fraud, or legal dispute requires immediate human attention."
        elif top_sim < 0.18:
            escalate = True
            reason = "Confidence below threshold; routing to human agent."

        return {
            "brand": self.brand_name,
            "intent": "general_inquiry",
            "confidence": round(top_sim, 4),
            "reply": cleaned_reply,
            "escalate": escalate,
            "reason": reason,
            "similarity": round(top_sim, 4),
            "evidence": [str(self.pairs.iloc[best_idx]["customer_text"])],
        }


class UniversalSupportAgent:
    """Universal Enterprise Agent capable of answering queries across the entire TWCS dataset."""

    def __init__(self, cache_agents: bool = True) -> None:
        self.cache_agents = cache_agents
        self._cached_agents: Dict[str, Any] = {}

    def detect_brand(self, text: str, hint_brand: str = "") -> str:
        if hint_brand:
            return hint_brand

        for pattern, brand in BRAND_PATTERNS:
            if re.search(pattern, text):
                return brand

        return "GeneralSupport"

    def get_agent(self, brand: str) -> Any:
        if brand in self._cached_agents:
            return self._cached_agents[brand]

        agent = None
        brand_low = brand.lower()

        if brand_low in {"applesupport", "apple"}:
            pairs_p = BRANDS_DATA_DIR / "applesupport" / "applesupport_pairs.csv"
            if pairs_p.exists():
                pairs = pd.read_csv(pairs_p, low_memory=False)
                agent = AppleSupportAgent(pairs, sample_size=35_000)

        elif brand_low in {"uber_support", "uber"}:
            pairs_p = BRANDS_DATA_DIR / "uber_support" / "uber_support_pairs.csv"
            if pairs_p.exists():
                pairs = pd.read_csv(pairs_p, low_memory=False)
                agent = UberSupportAgent(pairs, sample_size=35_000)

        elif brand_low in {"amazonhelp", "amazon"}:
            pairs_p = BRANDS_DATA_DIR / "amazonhelp" / "amazonhelp_pairs.csv"
            if pairs_p.exists():
                pairs = pd.read_csv(pairs_p, low_memory=False)
                agent = AmazonSupportAgent(pairs)

        elif brand_low in {"spotifycares", "spotify"}:
            pairs_p = BRANDS_DATA_DIR / "spotifycares" / "spotifycares_pairs.csv"
            if pairs_p.exists():
                pairs = pd.read_csv(pairs_p, low_memory=False)
                agent = SpotifySupportAgent(pairs, sample_size=30_000)

        else:
            # Check dynamic dataset
            for folder in BRANDS_DATA_DIR.iterdir():
                if folder.is_dir() and folder.name.lower() == brand_low:
                    pairs_files = list(folder.glob("*_pairs.csv"))
                    if pairs_files:
                        agent = GenericBrandSupportAgent(brand, pairs_files[0])
                        break

        if agent is None:
            # Fallback to AppleSupport as reference baseline
            pairs_p = BRANDS_DATA_DIR / "applesupport" / "applesupport_pairs.csv"
            pairs = pd.read_csv(pairs_p, low_memory=False)
            agent = AppleSupportAgent(pairs, sample_size=20_000)

        if self.cache_agents:
            self._cached_agents[brand] = agent

        return agent

    def process_query(self, text: str, brand_hint: str = "", handle: str = "") -> dict:
        detected_brand = self.detect_brand(text, hint_brand=brand_hint)
        agent = self.get_agent(detected_brand)

        if hasattr(agent, "predict"):
            try:
                result = agent.predict(text, handle=handle)
            except TypeError:
                result = agent.predict(text)
        else:
            raise RuntimeError(f"Agent for {detected_brand} does not have a predict method.")

        result["detected_brand"] = detected_brand
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", required=True, help="Customer tweet text.")
    parser.add_argument("--brand", default="", help="Optional brand override.")
    parser.add_argument("--handle", default="", help="Customer handle.")
    args = parser.parse_args()

    agent = UniversalSupportAgent()
    res = agent.process_query(args.text, brand_hint=args.brand, handle=args.handle)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
