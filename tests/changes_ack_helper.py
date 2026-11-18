#!/usr/bin/env python3
"""Acknowledge one subscription position from a fresh process."""
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
    ap.add_argument("--gen", type=int, required=True)
    ap.add_argument("--seq", type=int, required=True)
    ap.add_argument("--expected-gen", type=int, required=True)
    ap.add_argument("--expected-seq", type=int, required=True)
    args = ap.parse_args()
    try:
        with open_collection(args.collection) as coll:
            result = coll.acknowledge_changes(
                args.name, gen=args.gen, seq=args.seq,
                expected_gen=args.expected_gen,
                expected_seq=args.expected_seq,
            )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
