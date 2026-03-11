"""Run the coordinator with real prompts and print full results (routing + agent output).
Usage: from project root, run: python scripts/run_coordinator_demo.py
Requires .env with OPENAI_API_KEY or ANTHROPIC_API_KEY."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from src.coordinator import run as coordinator_run


def _pp(obj):
    if isinstance(obj, dict):
        return json.dumps(obj, indent=2)
    return str(obj)


def main():
    prompts = [
        "Optimize cooling rate for a nickel superalloy to get good yield strength.",
        "What is the capital of France?",
        "I have a material: Inconel 718, process at 1100 C for 2 hours. Check schema and constraints.",
    ]
    for i, prompt in enumerate(prompts, 1):
        print("\n" + "=" * 70)
        print(f"PROMPT {i}: {prompt}")
        print("=" * 70)
        try:
            out = coordinator_run(prompt)
            print("ROUTING:", out.get("agent"), "| mode:", out.get("mode"))
            if "error" in out:
                print("ERROR:", _pp(out["error"]))
            else:
                result = out.get("result")
                if isinstance(result, dict):
                    if "output" in result:
                        print("\nAGENT OUTPUT (simulation report):")
                        print(result["output"])
                    elif "summary" in result:
                        print("\nAGENT OUTPUT (processor summary):")
                        print(result["summary"])
                    else:
                        print("\nAGENT RESULT:", _pp(result))
                else:
                    print("\nAGENT OUTPUT:")
                    print(result)
        except Exception as e:
            print("EXCEPTION:", type(e).__name__, str(e))
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
