#!/usr/bin/env python3
"""Standalone process helper: read one snapshot page and print JSON.

Used by cross-process snapshot/cursor tests.  It opens the collection
fresh, performs exactly one paged read, prints the page, and exits -- no
connection, transaction or Python object of the calling process survives.
Run with ``python -I`` so the workspace is only reachable via sys.path.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytekey.cli import _coerce_query  # noqa: E402
from bytekey.collection import open_collection  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--token", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--page-size", type=int, required=True)
    ap.add_argument("--cursor", default=None)
    args = ap.parse_args()
    with open(args.query, "r", encoding="utf-8") as fh:
        query = json.load(fh)
    with open_collection(args.collection) as coll:
        query = _coerce_query(coll.schema, query)
        page, stats, _plan = coll.scan_snapshot_page(
            args.token, query, args.page_size, cursor_token=args.cursor
        )
        payload = {
            "gen": page.gen,
            "rows": [
                {"values": [
                    {"__hex__": v.hex()} if isinstance(v, (bytes, bytearray)) else v
                    for v in values
                ],
                 coll.row_id_name: {"__hex__": rid.hex()}}
                for values, rid in page.rows
            ],
            "next_cursor": page.next_cursor,
            "candidates_examined": stats.candidates_examined,
        }
    json.dump(payload, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
