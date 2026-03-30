"""CLI entrypoint for sandbox coordinator."""

from __future__ import annotations

import argparse
import json

from sandbox.coordinator import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run sandbox coordinator queue orchestration.")
    parser.add_argument("--queue", default="sandbox/prompts/coordinator_queue.txt", help="Queue file path")
    parser.add_argument("--config", default="sandbox/config.json", help="Config file path")
    args = parser.parse_args()
    print(json.dumps(run(queue_path=args.queue, config_path=args.config), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
