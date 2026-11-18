#!/usr/bin/env python3
"""Drain one subscription page from a fresh process (change-stream tests)."""
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
    ap.add_argument("--page-size", type=int, default=100)
    ap.add_argument("--cursor", default=None)
    args = ap.parse_args()
    with open_collection(args.collection) as coll:
        page = coll.read_changes_page(
            args.name, args.page_size, cursor_token=args.cursor
        )
    json.dump(page.to_dict(), sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
