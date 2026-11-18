#!/usr/bin/env python3
"""Apply one small batch from a fresh process, with a controllable hard kill.

Operations are puts of the ``(a text, b int64)`` schema used by the change
tests; ``--crash pre_commit`` aborts before the commit (nothing published) and
``--crash post_commit`` aborts after it (published, reply lost).
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
    ap.add_argument("--batch-id", required=True, help="hex string")
    ap.add_argument("--expected-head", type=int, required=True)
    ap.add_argument("--rows", required=True,
                    help='JSON list of [hex row_id, text a, int b]')
    ap.add_argument("--crash", choices=["pre_commit", "post_commit"],
                    default=None)
    args = ap.parse_args()

    with open(args.rows, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    operations = [
        {"op": "put", "row_id": bytes.fromhex(rid), "a": a, "b": b}
        for rid, a, b in rows
    ]

    def hook(phase, _conn):
        if phase == args.crash:
            os._exit(87)   # hard kill, no connection shutdown

    with open_collection(args.collection) as coll:
        try:
            receipt = coll.apply(
                bytes.fromhex(args.batch_id), args.expected_head,
                iter(operations),
                crash_hook=hook if args.crash else None,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    json.dump({"status": "applied", **receipt.to_dict()}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
