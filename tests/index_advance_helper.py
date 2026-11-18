#!/usr/bin/env python3
"""Standalone process that advances one secondary-index build by a budget.

Optionally hard-kills the process (``os._exit``) at a controlled commit
boundary so tests can prove that progress survives a real process exit.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytekey.collection import open_collection  # noqa: E402


def make_hook(phase_wanted):
    def hook(phase, _conn):
        if phase == phase_wanted:
            os._exit(88)
    return hook


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--build-id", default=None)
    ap.add_argument("--crash", choices=["pre_commit", "post_commit"], default=None)
    args = ap.parse_args()
    expected = None if args.build_id is None else bytes.fromhex(args.build_id)
    with open_collection(args.collection) as coll:
        try:
            status = coll.advance_index(
                args.name, args.budget, expected_build_id=expected,
                crash_hook=make_hook(args.crash) if args.crash else None,
            )
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
    json.dump(status.to_dict(), sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
