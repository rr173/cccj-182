#!/usr/bin/env python3
"""Standalone writer process for concurrency and crash tests.

Opens the collection fresh, applies one batch, optionally kills the process
at a controlled point (``--crash pre_commit|post_commit``) via os._exit so
there is no normal connection shutdown -- simulating a real hard failure --
and prints the receipt JSON on success.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytekey.collection import open_collection  # noqa: E402


def make_hook(phase_wanted):
    def hook(phase, _conn):
        # Only hard-kill at the requested commit boundary; no finally blocks,
        # no atexit, no connection close: a true hard kill.
        if phase == phase_wanted:
            os._exit(87)
    return hook


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--batch-id", required=True, help="hex string")
    ap.add_argument("--expected-head", type=int, required=True)
    ap.add_argument("--source", required=True,
                    help="JSON file with a list of hex byte strings -> puts")
    ap.add_argument("--crash", choices=["pre_commit", "post_commit"], default=None)
    args = ap.parse_args()

    with open(args.source, "r", encoding="utf-8") as fh:
        items = json.load(fh)
    operations = [
        {
            "op": "put",
            "row_id": bytes.fromhex(item["__hex__"]),
            "region": "r-" + item["__hex__"],
            "ts": int(item["__hex__"], 16) % 1000,
            "blob": bytes.fromhex(item["__hex__"]),
            "flag": True,
        }
        for item in items
    ]
    with open_collection(args.collection) as coll:
        try:
            receipt = coll.apply(
                bytes.fromhex(args.batch_id),
                args.expected_head,
                iter(operations),
                crash_hook=make_hook(args.crash) if args.crash else None,
            )
        except Exception as exc:  # explicit non-zero conflict/failure exit
            print(f"error: {exc}", file=sys.stderr)
            return 2
    json.dump({"status": "applied", **receipt.to_dict()}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
