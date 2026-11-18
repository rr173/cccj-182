#!/usr/bin/env python3
"""Standalone process: read exactly one page through a ready index.

Used by tests to prove that a brand-new process needs only the collection
path, snapshot token, index name, query and cursor to continue paging.  It
never imports the test suite and holds no in-memory state from the caller.
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
    ap.add_argument("--token", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--page-size", type=int, required=True)
    ap.add_argument("--cursor", default=None)
    args = ap.parse_args()
    with open(args.query, "r", encoding="utf-8") as fh:
        query = json.load(fh)
    with open_collection(args.collection) as coll:
        page, stats, plan = coll.scan_index_page(
            args.token, args.index, query, args.page_size,
            cursor_token=args.cursor,
        )
        payload = {
            "gen": page.gen,
            "sid": page.sid,
            "index": page.index_name,
            "instance_id": page.instance_id,
            "rows": [
                {"values": list(values),
                 coll.row_id_name: {"__hex__": rid.hex()}}
                for values, rid in page.rows
            ],
            "next_cursor": page.next_cursor,
            "stats": stats.to_dict(),
        }
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
