"""CLI runner for AppleSupport AI Customer Support Agent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.agent import AppleSupportAgent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="", help="Customer tweet text to evaluate.")
    parser.add_argument("--handle", default="", help="Customer handle.")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive chat session.")
    args = parser.parse_args()

    agent = AppleSupportAgent()

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
        parser.error("Either --text or --interactive must be specified.")

    result = agent.predict(args.text, handle=args.handle)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
