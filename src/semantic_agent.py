"""Semantic retrieval agent using a lightweight Hugging Face encoder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer

from .agent import AgentResult, classify_intent
from .text_preprocessing import normalize_for_classification


class SemanticSupportAgent:
    def __init__(
        self,
        pairs: pd.DataFrame,
        top_k: int = 3,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_reference: int = 30_000,
        random_state: int = 42,
    ) -> None:
        if len(pairs) > max_reference:
            pairs = pairs.sample(max_reference, random_state=random_state)
        self.pairs = pairs.reset_index(drop=True)
        self.top_k = top_k
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        documents = self.pairs["customer_text"].map(normalize_for_classification).tolist()
        self.embeddings = self.model.encode(
            documents,
            batch_size=64,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    def predict(self, text: str) -> AgentResult:
        query = self.model.encode(
            [normalize_for_classification(text)], normalize_embeddings=True
        )[0]
        similarities = self.embeddings @ query
        nearest = similarities.argsort()[::-1][: self.top_k]
        best = int(nearest[0])
        similarity = float(similarities[best])
        intent, confidence = classify_intent(text)
        risky = intent in {"account_access_security", "app_store_billing"}
        unclear = intent == "other_unclear" or confidence < 0.5
        weak_evidence = similarity < 0.45
        escalate = risky or unclear or weak_evidence

        if risky:
            reason = "Sensitive account or payment issue requires human review."
        elif unclear:
            reason = "The message is ambiguous or the intent confidence is low."
        elif weak_evidence:
            reason = "No sufficiently similar semantic support example was found."
        else:
            reason = "The intent and semantic evidence are sufficiently clear for a draft."

        return AgentResult(
            intent=intent,
            reply=str(self.pairs.iloc[best]["brand_reply"]),
            escalate=escalate,
            reason=reason,
            similarity=similarity,
            evidence=[str(self.pairs.iloc[index]["customer_text"]) for index in nearest],
        )
