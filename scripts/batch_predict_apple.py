"""High-throughput Batch Prediction Engine for 106,648 AppleSupport Queries.

Runs all 106k historical customer queries through:
- Intent Classification (11 Domain Categories)
- Grounded RAG Retrieval (TF-IDF Cosine Similarity)
- Deterministic Safety Escalation Guardrails

Saves comprehensive predictions to:
`data/processed/brands/applesupport/applesupport_predictions.csv`
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent import AppleSupportAgent, load_or_extract_applesupport_pairs
from src.intent_classifier import classify_intent_rules
from src.text_preprocessing import (
    clean_apple_reply,
    detect_urgency,
    extract_apple_entities,
    normalize_for_retrieval,
)

BATCH_SIZE = 2500


def run_full_batch_inference() -> Path:
    print("=" * 80)
    print("STARTING FULL BATCH PREDICTION FOR 106,648 APPLESUPPORT QUERIES")
    print("=" * 80)

    start_init = time.time()
    
    # 1. Load full dataset
    data_path = ROOT_DIR / "data" / "processed" / "brands" / "applesupport" / "applesupport_pairs.csv"
    if data_path.exists():
        pairs_df = pd.read_csv(data_path, low_memory=False)
    else:
        pairs_df = load_or_extract_applesupport_pairs(sample_size=120_000)

    total_queries = len(pairs_df)
    print(f"Loaded {total_queries:,} AppleSupport customer queries in {time.time() - start_init:.2f}s.")

    # 2. Initialize Agent & TF-IDF Retrieval Matrix
    print("Building full TF-IDF retrieval index...")
    agent = AppleSupportAgent(pairs=pairs_df.sample(min(total_queries, 35_000), random_state=42))
    print(f"TF-IDF index ready ({agent.matrix.shape[0]:,} reference documents).")

    # 3. Batch Inference Execution
    customer_texts = pairs_df["customer_text"].fillna("").astype(str).tolist()
    
    results = []
    start_inference = time.time()
    n_batches = int(np.ceil(total_queries / BATCH_SIZE))

    print(f"\nProcessing {total_queries:,} queries in {n_batches} chunks (batch size = {BATCH_SIZE})...\n")

    for b_idx in range(n_batches):
        b_start = b_idx * BATCH_SIZE
        b_end = min((b_idx + 1) * BATCH_SIZE, total_queries)
        batch_texts = customer_texts[b_start:b_end]

        # A. Batch Intent Classification
        batch_intents = [classify_intent_rules(t) for t in batch_texts]

        # B. Batch TF-IDF Retrieval Matrix Multiplication
        norm_queries = [normalize_for_retrieval(t) for t in batch_texts]
        query_vecs = agent.vectorizer.transform(norm_queries)
        sim_matrix = (agent.matrix @ query_vecs.T).toarray()  # Shape: (n_ref, batch_size)

        best_indices = sim_matrix.argmax(axis=0)  # Shape: (batch_size,)
        best_sims = sim_matrix.max(axis=0)

        # C. Formulate Output Predictions & Safety Gating
        for i, (text, (intent, conf), idx, sim) in enumerate(
            zip(batch_texts, batch_intents, best_indices, best_sims)
        ):
            best_row = agent.pairs.iloc[idx]
            reply = clean_apple_reply(str(best_row["brand_reply"]))

            # Safety escalation policy
            escalate = False
            reason = "Auto-handled: verified historical resolution match."
            low = text.lower()
            if intent in {"account_access_security", "app_store_billing"}:
                escalate = True
                reason = f"Deterministic safety rule: {intent} requires human agent verification."
            elif any(k in low for k in ["hacked", "fraud", "scam", "lawyer", "police", "stolen", "unauthorized"]):
                escalate = True
                reason = "Sensitive security/legal keyword triggered human escalation."
            elif sim < 0.20:
                escalate = True
                reason = f"Retrieval similarity ({sim:.3f}) below confidence threshold (0.20)."

            results.append({
                "customer_text": text,
                "predicted_intent": intent,
                "confidence": conf,
                "escalate": escalate,
                "escalation_reason": reason,
                "similarity_score": round(float(sim), 4),
                "grounded_reply": reply,
            })

        elapsed = time.time() - start_inference
        rate = b_end / elapsed
        percent = (b_end / total_queries) * 100
        print(f"[{b_idx+1:02d}/{n_batches:02d}] Processed {b_end:,}/{total_queries:,} queries ({percent:5.1f}%) | Speed: {rate:6.1f} q/s | Elapsed: {elapsed:.1f}s")

    total_time = time.time() - start_inference
    print("\n" + "=" * 80)
    print(f"BATCH INFERENCE COMPLETED IN {total_time:.2f}s ({total_time/60:.2f} min)")
    print(f"Average Throughput: {total_queries / total_time:.1f} queries/sec")
    print("=" * 80)

    # 4. Save results to CSV
    output_df = pd.DataFrame(results)
    out_dir = ROOT_DIR / "data" / "processed" / "brands" / "applesupport"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "applesupport_predictions.csv"
    output_df.to_csv(out_path, index=False)

    print(f"\n[SAVED] {len(output_df):,} predictions saved to: {out_path}")

    # Summary Statistics
    intent_dist = output_df["predicted_intent"].value_counts().to_dict()
    escalate_count = int(output_df["escalate"].sum())
    auto_count = len(output_df) - escalate_count

    print("\n" + "-" * 50)
    print("SUMMARY STATISTICS")
    print("-" * 50)
    print(f"Total Processed       : {len(output_df):,}")
    print(f"Auto-Handled Replies  : {auto_count:,} ({auto_count/len(output_df)*100:.1f}%)")
    print(f"Escalated to Human    : {escalate_count:,} ({escalate_count/len(output_df)*100:.1f}%)")
    print(f"Intent Distribution   :\n{pd.Series(intent_dist).to_string()}")

    return out_path


if __name__ == "__main__":
    run_full_batch_inference()
