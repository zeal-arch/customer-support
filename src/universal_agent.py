"""Universal Multi-Brand Customer Support Agent.

Automatically identifies target brand from tweet text, handles, or context,
and routes to the optimal brand-specific agent or cross-brand semantic index.
Handles on-the-fly extraction from raw twcs.csv automatically if needed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

DATA_DIR = ROOT_DIR / "data"
BRANDS_DATA_DIR = DATA_DIR / "processed" / "brands"
RAW_DATA_CANDIDATES = [
    DATA_DIR / "twcs" / "twcs.csv",
    DATA_DIR / "raw" / "twcs" / "twcs.csv",
    DATA_DIR / "twcs.csv",
    DATA_DIR / "raw" / "twcs.csv",
]

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

# Embedded high-quality seed conversation pairs for instant zero-config fallback
EMBEDDED_SEED_PAIRS: Dict[str, list[dict[str, str]]] = {
    "applesupport": [
        {"customer_text": "My iPhone battery drains super fast after updating to iOS 17", "brand_reply": "Let's look into this with you. Send us a DM with your device model and current iOS build."},
        {"customer_text": "Someone hacked my Apple ID account and changed my password", "brand_reply": "We take account security very seriously. Please DM us immediately so we can secure your Apple ID."},
        {"customer_text": "My AirPods won't connect to my MacBook Pro bluetooth", "brand_reply": "Let's get your AirPods connected. Check your Bluetooth settings or reset your AirPods in their case."},
        {"customer_text": "I was charged twice for a subscription in the App Store", "brand_reply": "We can help check your App Store billing history. Please DM us with your Apple ID email."},
    ],
    "uber_support": [
        {"customer_text": "My driver was driving recklessly and threatened me", "brand_reply": "We take safety reports very seriously. Please DM us your phone number and trip details immediately."},
        {"customer_text": "I was charged twice for my Uber ride yesterday", "brand_reply": "We'd be glad to look into your fare. Please DM us your account email and trip receipt."},
        {"customer_text": "I left my phone and wallet in the back of the driver's car", "brand_reply": "Let's help you recover your lost item. Go to 'Your Trips' > 'Find lost item' in the Uber app or DM us."},
    ],
    "amazonhelp": [
        {"customer_text": "My package says delivered but I never received it", "brand_reply": "We're sorry for the trouble! Please DM us your 17-digit order number and delivery address."},
        {"customer_text": "Someone accessed my Amazon account and placed unauthorized orders", "brand_reply": "Please reach out to our security team immediately or DM us so we can lock unauthorized activity."},
        {"customer_text": "How do I return a damaged item for a refund?", "brand_reply": "You can initiate a return through 'Your Orders' > 'Return or Replace Items'. DM us if you need help."},
    ],
    "spotifycares": [
        {"customer_text": "Spotify charged my credit card twice this month for Premium", "brand_reply": "We'll be happy to review your charges. Please DM us your account email and receipt details."},
        {"customer_text": "My downloaded offline songs keep disappearing on iPhone", "brand_reply": "Let's troubleshoot your offline downloads. Try clearing the app cache and ensuring you are logged in."},
    ],
}


def get_raw_twcs_path() -> Optional[Path]:
    """Find local twcs.csv file if present."""
    for p in RAW_DATA_CANDIDATES:
        if p.exists() and p.stat().st_size > 1000:
            return p
    return None


def get_or_extract_brand_pairs(brand_name: str, sample_size: int = 35_000) -> pd.DataFrame:
    """Retrieve brand pairs from processed cache, or extract dynamically from raw twcs.csv."""
    brand_slug = brand_name.lower().replace("@", "")
    target_dir = BRANDS_DATA_DIR / brand_slug
    target_file = target_dir / f"{brand_slug}_pairs.csv"

    # 1. Load from processed cache if exists
    if target_file.exists():
        try:
            return pd.read_csv(target_file, low_memory=False)
        except Exception:
            pass

    # 2. Extract dynamically from raw twcs.csv if available
    raw_path = get_raw_twcs_path()
    if raw_path is not None:
        try:
            print(f"[*] Extracting brand '{brand_name}' from raw dataset: {raw_path.name}...")
            df = pd.read_csv(
                raw_path,
                usecols=["tweet_id", "author_id", "inbound", "text", "in_response_to_tweet_id"],
                low_memory=False,
            )
            brand_replies = df[df["author_id"].astype(str).str.lower() == brand_slug]
            inbound_tweets = df[df["inbound"] == True].set_index("tweet_id")

            merged = brand_replies.join(
                inbound_tweets, on="in_response_to_tweet_id", lsuffix="_reply", rsuffix="_inbound"
            )
            pairs = pd.DataFrame(
                {"customer_text": merged["text_inbound"], "brand_reply": merged["text_reply"]}
            ).dropna().reset_index(drop=True)

            if len(pairs) > 0:
                target_dir.mkdir(parents=True, exist_ok=True)
                pairs.to_csv(target_file, index=False)
                print(f"[+] Cached {len(pairs):,} pairs for '{brand_name}' at {target_file.name}")
                return pairs
        except Exception as err:
            print(f"[!] Dynamic extraction note: {err}")

    # 3. Fallback to high-quality seed dataset
    seed_key = brand_slug if brand_slug in EMBEDDED_SEED_PAIRS else "applesupport"
    seed_data = EMBEDDED_SEED_PAIRS.get(seed_key, EMBEDDED_SEED_PAIRS["applesupport"])
    return pd.DataFrame(seed_data)


class GenericBrandSupportAgent:
    """Dynamic fallback RAG agent for any brand."""

    def __init__(self, brand_name: str, pairs: pd.DataFrame, top_k: int = 3) -> None:
        self.brand_name = brand_name
        self.top_k = top_k
        self.pairs = pairs.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_features=40_000,
            strip_accents="unicode",
        )
        documents = self.pairs["customer_text"].fillna("").map(normalize_for_classification)
        self.matrix = self.vectorizer.fit_transform(documents)

    def predict(self, text: str, handle: str = "") -> dict:
        query_vec = self.vectorizer.transform([normalize_for_classification(text)])
        sims = (self.matrix @ query_vec.T).toarray().ravel()
        best_idx = int(sims.argmax()) if len(sims) > 0 else 0
        top_sim = float(sims[best_idx]) if len(sims) > 0 else 0.0

        reply = str(self.pairs.iloc[best_idx]["brand_reply"])
        cleaned_reply = re.sub(r"^@\w+\s*", "", reply.strip())
        if handle:
            h = handle if handle.startswith("@") else f"@{handle}"
            cleaned_reply = f"{h} {cleaned_reply}"

        low = str(text).lower()
        escalate = False
        reason = "Automated high-relevance support draft."

        if any(k in low for k in ["fraud", "hacked", "stolen", "emergency", "police", "legal", "threat", "reckless"]):
            escalate = True
            reason = "Security, safety, or legal dispute requires immediate human attention."
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

        return "AppleSupport"

    def get_agent(self, brand: str) -> Any:
        if brand in self._cached_agents:
            return self._cached_agents[brand]

        brand_low = brand.lower().replace("@", "")
        pairs = get_or_extract_brand_pairs(brand_low)

        if brand_low in {"applesupport", "apple"}:
            agent = AppleSupportAgent(pairs, sample_size=min(len(pairs), 35_000))
        elif brand_low in {"uber_support", "uber"}:
            agent = UberSupportAgent(pairs, sample_size=min(len(pairs), 35_000))
        elif brand_low in {"amazonhelp", "amazon"}:
            agent = AmazonSupportAgent(pairs)
        elif brand_low in {"spotifycares", "spotify"}:
            agent = SpotifySupportAgent(pairs, sample_size=min(len(pairs), 30_000))
        else:
            agent = GenericBrandSupportAgent(brand, pairs)

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
    parser.add_argument("--text", default="", help="Customer tweet text.")
    parser.add_argument("--brand", default="", help="Optional brand override.")
    parser.add_argument("--handle", default="", help="Customer handle.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive shell.")
    args = parser.parse_args()

    agent = UniversalSupportAgent()

    if args.interactive:
        print("=" * 60)
        print("Universal Multi-Brand Customer Support AI Agent")
        print("Type a customer query or 'exit' to quit.")
        print("=" * 60)
        while True:
            try:
                user_input = input("\nCustomer Tweet > ").strip()
                if not user_input or user_input.lower() in {"exit", "quit", "q"}:
                    break
                res = agent.process_query(user_input, brand_hint=args.brand)
                print(json.dumps(res, indent=2))
            except (KeyboardInterrupt, EOFError):
                break
        return

    if not args.text:
        parser.error("Either --text or --interactive must be provided.")

    res = agent.process_query(args.text, brand_hint=args.brand, handle=args.handle)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
