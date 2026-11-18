#!/usr/bin/env python3
"""Read exactly one change page from a fresh process.

Prints one JSON object: the page payload plus ``has_more`` and the cursor a
consumer must pass to the next invocation.  Any domain error exits 2 with a
locatable message on stderr.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytekey.collection import open_collection  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--page-size", type=int, required=True)
    ap.add_argument("--cursor", default=None)
    args = ap.parse_args()
    try:
        with open_collection(args.collection) as coll:
            page = coll.read_changes_page(
                args.name, args.page_size, cursor_token=args.cursor
            )
            payload = page.to_dict()
            payload["collection_id"] = page.collection_id
            payload["head"] = page.head
    except Exception as exc:  # explicit locatable failure
        print(f"error: {exc}", file=sys.stderr)
        return 2
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
