#!/usr/bin/env python3
"""Standalone join-prepare process for crash-recovery tests.

Reads a JSON job file describing one ``prepare_join`` call and optionally
hard-kills itself (``os._exit`` -- no cleanup, no atexit) at a named prepare
phase, simulating a real process termination mid-prepare.  Prints the
prepare result JSON on success.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bytekey.join import JoinSource, prepare_join  # noqa: E402

PHASES = ("after_meta", "after_left", "after_right", "after_matched",
          "pre_commit", "post_commit")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, help="JSON job description")
    args = ap.parse_args()
    with open(args.job, "r", encoding="utf-8") as fh:
        job = json.load(fh)

    crash = job.get("crash")

    def hook(phase, _conn):
        if phase == crash:
            os._exit(87)

    def source(spec):
        return JoinSource(
            collection=spec["collection"],
            token=spec["token"],
            query=spec["query"],
            index=spec.get("index"),
        )

    try:
        result = prepare_join(
            job["work"],
            left=source(job["left"]),
            right=source(job["right"]),
            mapping=job["mapping"],
            mode=job["mode"],
            nulls_equal=bool(job.get("nulls_equal", False)),
            page_size=int(job["page_size"]),
            crash_hook=hook if crash else None,
        )
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
