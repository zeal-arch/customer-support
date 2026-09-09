"""CLI and batch runner for AppleSupport AI Customer Support Agent."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent import AppleSupportAgent


def run_batch_eval(agent: AppleSupportAgent, csv_path: Path, limit: int | None = None) -> None:
    """Process a batch of customer queries from CSV with progress and metrics."""
    print("=" * 80)
    print(f"PROCESSING BATCH QUERIES FROM: {csv_path.name}")
    print("=" * 80)

    if not csv_path.exists():
        print(f"Error: File not found at {csv_path}")
        return

    df = pd.read_csv(csv_path, low_memory=False)
    if limit and limit > 0:
        df = df.head(limit)

    total = len(df)
    print(f"Loaded {total:,} queries to evaluate...")

    results = []
    start_t = time.time()
    auto_count = 0
    escalate_count = 0

    for idx, row in df.iterrows():
        text = str(row.get("customer_text", row.get("text", "")))
        handle = str(row.get("customer_twitter_handle", row.get("author_id", "")))

        res = agent.predict(text, handle=handle)
        if res["escalate"]:
            escalate_count += 1
        else:
            auto_count += 1

        results.append({
            "customer_text": text,
            "predicted_intent": res["intent"],
            "confidence": res["confidence"],
            "escalate": res["escalate"],
            "reason": res["reason"],
            "similarity": res["similarity"],
            "reply": res["reply"],
        })

        if (idx + 1) % 5000 == 0 or (idx + 1) == total:
            elapsed = time.time() - start_t
            rate = (idx + 1) / elapsed
            print(f"Progress: {idx + 1:,}/{total:,} ({(idx + 1)/total*100:5.1f}%) | Speed: {rate:6.1f} q/s | Elapsed: {elapsed:.1f}s")

    total_time = time.time() - start_t
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETED")
    print("=" * 80)
    print(f"Total Evaluated       : {total:,}")
    print(f"Auto-Handled Replies  : {auto_count:,} ({auto_count/total*100:.1f}%)")
    print(f"Escalated to Human    : {escalate_count:,} ({escalate_count/total*100:.1f}%)")
    print(f"Total Run Time        : {total_time:.2f} seconds ({total_time/60:.2f} min)")
    print(f"Average Throughput    : {total / total_time:.1f} queries / sec")

    # Save output
    out_dir = ROOT_DIR / "data" / "processed" / "brands" / "applesupport"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "batch_run_results.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"\n[OUTPUT] Results saved to: {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="", help="Single customer tweet text to evaluate.")
    parser.add_argument("--handle", default="", help="Customer handle.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive chat session.")
    parser.add_argument("--all", action="store_true", help="Run batch evaluation on all 106k AppleSupport queries.")
    parser.add_argument("--batch", type=Path, default=None, help="Path to custom CSV containing customer queries.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows for batch run.")
    args = parser.parse_args()

    agent = AppleSupportAgent()

    if args.all:
        default_csv = ROOT_DIR / "data" / "processed" / "brands" / "applesupport" / "applesupport_pairs.csv"
        run_batch_eval(agent, default_csv, limit=args.limit)
        return

    if args.batch:
        run_batch_eval(agent, args.batch, limit=args.limit)
        return

    if args.interactive:
        print("=" * 60)
        print("AppleSupport AI Agent — Interactive Session")
        print("Type your customer message or 'exit' to quit.")
        print("=" * 60)
        while True:
            try:
                user_input = input("\nCustomer Tweet > ").strip()
                if not user_input or user_input.lower() in {"exit", "quit", "q"}:
                    break
                result = agent.predict(user_input, handle=args.handle)
                print(json.dumps(result, indent=2))
            except (KeyboardInterrupt, EOFError):
                break
        return

    if not args.text:
        parser.error("Specify --text, --all, --batch <file.csv>, or --interactive.")

    result = agent.predict(args.text, handle=args.handle)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
