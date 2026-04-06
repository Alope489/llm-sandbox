"""Delete all ``kb-bench-*`` prefixed vector stores from the OpenAI account.

Use this script after:
* A benchmark run that crashed before completing inline VS deletions.
* A run executed with ``--keep-vector-stores``.

The script lists all vector stores in the account, filters by the name prefix
``kb-bench-``, prints a summary table, prompts for confirmation (unless
``--dry-run``), and then deletes all matching stores.

Usage::

    # Preview without deleting
    python new_kb_sandbox/cleanup_vector_stores.py --dry-run

    # Delete all kb-bench-* stores (prompts for confirmation)
    python new_kb_sandbox/cleanup_vector_stores.py

Dependencies:
    openai, python-dotenv.

Environment variables:
    OPENAI_API_KEY: Required.

Pillar compliance:
    - Pillar 1: Only deletes stores whose name starts with ``kb-bench-``; no
      other stores are touched.
    - Pillar 4: Prefix configurable via ``--prefix``; no hardcoded values.
    - Pillar 5: OPENAI_API_KEY from .env; no secrets printed.
    - Pillar 7: ``--dry-run`` safety flag; requires explicit confirmation.
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

_DEFAULT_PREFIX = "kb-bench-"


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Populated ``argparse.Namespace``.
    """
    parser = argparse.ArgumentParser(
        description="List and delete all kb-bench-* vector stores from the OpenAI account.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=_DEFAULT_PREFIX,
        help="Name prefix of vector stores to target.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching stores without deleting them.",
    )
    return parser.parse_args()


def _list_matching_stores(client: object, prefix: str) -> list[dict]:
    """Retrieve all vector stores from the account matching *prefix*.

    Paginates through all results using the OpenAI SDK's auto-pagination.

    Args:
        client: Authenticated ``openai.OpenAI`` client.
        prefix: Name prefix to filter on.

    Returns:
        List of dicts with keys ``id``, ``name``, ``file_count``,
        ``status``, ``created_at``.

    Complexity:
        O(N) where N = total vector stores in the account.
    """
    matching = []
    for vs in client.vector_stores.list():
        name = getattr(vs, "name", "") or ""
        if name.startswith(prefix):
            matching.append(
                {
                    "id": vs.id,
                    "name": name,
                    "file_count": getattr(vs, "file_counts", None),
                    "status": getattr(vs, "status", "unknown"),
                    "created_at": getattr(vs, "created_at", ""),
                }
            )
    return matching


def _print_table(stores: list[dict]) -> None:
    """Print a summary table of *stores* to stdout.

    Args:
        stores: List of store dicts as returned by ``_list_matching_stores``.

    Returns:
        None
    """
    if not stores:
        print("No matching vector stores found.")
        return
    col_w = [max(len(str(r[k])) for r in stores) for k in ("id", "name", "status", "created_at")]
    col_w = [max(cw, len(h)) for cw, h in zip(col_w, ("ID", "Name", "Status", "Created"))]
    header = (
        f"{'ID':<{col_w[0]}}  {'Name':<{col_w[1]}}  "
        f"{'Status':<{col_w[2]}}  {'Created':<{col_w[3]}}"
    )
    print(header)
    print("-" * len(header))
    for r in stores:
        print(
            f"{r['id']:<{col_w[0]}}  {r['name']:<{col_w[1]}}  "
            f"{r['status']!s:<{col_w[2]}}  {r['created_at']!s:<{col_w[3]}}"
        )
    print(f"\nTotal: {len(stores)} store(s).")


def main() -> None:
    """Entry point for the VS cleanup utility.

    Lists matching stores, prints a table, and optionally deletes them after
    a single confirmation prompt.

    Returns:
        None

    Raises:
        SystemExit: If OPENAI_API_KEY is absent or the user cancels.
    """
    args = _parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required.  Set it in .env or the environment.")

    from src.llm_pipeline_telemetry import get_openai_client  # noqa: PLC0415

    client = get_openai_client()

    print(f"Listing vector stores with prefix '{args.prefix}' …")
    stores = _list_matching_stores(client, args.prefix)
    _print_table(stores)

    if not stores:
        return

    if args.dry_run:
        print("\n[dry-run] No stores were deleted.")
        return

    answer = input(f"\nDelete all {len(stores)} store(s)? [y/N]: ").strip()
    if answer.lower() != "y":
        print("Aborted.  No stores were deleted.")
        return

    deleted = 0
    errors = 0
    for store in stores:
        try:
            client.vector_stores.delete(store["id"])
            print(f"  Deleted {store['id']} ({store['name']})")
            deleted += 1
        except Exception as exc:
            print(f"  ERROR deleting {store['id']}: {exc}", file=sys.stderr)
            errors += 1

    print(f"\nDone.  {deleted} deleted, {errors} error(s).")


if __name__ == "__main__":
    main()
