"""Runnable end-to-end demonstration of persistent change subscriptions.

Replays the mandated fixture from real files on disk, reads every page with
page size 1 (each step reopens the collection, so process exit is real),
acknowledges page by page, then runs the interleaved two-subscriber scenario,
history reclamation with a slow subscriber, and a deleted/re-created name.

Run (isolated):  python3 -I demo/demo_changes.py [--dir OUTDIR]
"""
import argparse
import json
import os
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from bytekey import Column, Schema, open_collection  # noqa: E402
from bytekey.collection import create_collection  # noqa: E402

SCHEMA = Schema(columns=(
    Column("kind", "text", asc=True, nulls_first=True),
    Column("n", "int64", asc=False, nulls_first=True),
    Column("note", "text", asc=True, nulls_first=False),
))

BATCHES = [
    ("seed", [("put", b"r1", ("a", 10, "old")),
              ("put", b"r2", ("a", 10, "keep"))]),
    ("g1", [("put", b"r1", ("c", 20, "new")),
            ("delete", b"r2", None)]),
    ("g2", [("put", b"r3", ("a", 7, "late"))]),
    ("g3", []),
    ("g4", [("delete", b"missing", None)]),
    ("g5", [("put", b"r2", ("b", 8, "reborn"))]),
]


def ops(items):
    for kind, rid, vals in items:
        if kind == "delete":
            yield {"op": "delete", "row_id": rid}
        else:
            yield {"op": "put", "row_id": rid, "kind": vals[0],
                   "n": vals[1], "note": vals[2]}


def build(path):
    create_collection(path, SCHEMA)
    with open_collection(path) as coll:
        head = 0
        for batch_id, items in BATCHES:
            coll.apply(batch_id.encode(), head, iter(ops(items)))
            head += 1


def image(value):
    return {"present": False} if value is None else {
        "present": True, "values": list(value)
    }


def drain(path, name, page_size):
    """Read pages one process-open at a time, acknowledging each page."""
    pages = []
    cursor = None
    while True:
        with open_collection(path) as coll:
            page = coll.read_changes_page(name, page_size, cursor_token=cursor)
            info = coll.subscription_info(name)
            ack = coll.acknowledge_changes(
                name, gen=page.gen, seq=page.seq,
                expected_gen=info.ack_gen, expected_seq=info.ack_seq,
            )
        pages.append({
            "position": [page.gen, page.seq],
            "ack": [ack["ack_gen"], ack["ack_seq"]],
            "events": [
                {"gen": e.gen, "seq": e.seq, "row_id": e.row_id.hex(),
                 "op": e.kind, "before": image(e.before),
                 "after": image(e.after)}
                for e in page.events
            ],
            "batches": [{"gen": b.gen, "events": b.event_count}
                        for b in page.batches],
            "has_more": page.has_more,
        })
        if not page.has_more:
            return pages
        cursor = page.next_cursor


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=None)
    ap.add_argument("--report", default=None)
    args = ap.parse_args()
    work = args.dir
    if work is None:
        import tempfile
        work = tempfile.mkdtemp(prefix="bytekey-changes-")
    else:
        os.makedirs(work, exist_ok=True)
    path = os.path.join(work, "changes.db")
    if os.path.exists(path):
        os.remove(path)
    build(path)

    report = {}
    with open_collection(path) as coll:
        coll.begin_subscription("downstream", start_gen=1)
        coll.begin_subscription("tabular", start_gen=3)
    # 1. Page size 1 across many process opens.
    report["primary_pages"] = drain(path, "downstream", 1)
    report["from_gen3_pages"] = drain(path, "tabular", 2)

    # 2. Retention with an active (now fully acknowledged) subscriber, then a
    #    slow subscriber that stops after one page.
    with open_collection(path) as coll:
        coll.begin_subscription("slow", start_gen=1)
        page = coll.read_changes_page("slow", 1)
        info = coll.subscription_info("slow")
        coll.acknowledge_changes(
            "slow", gen=page.gen, seq=page.seq,
            expected_gen=info.ack_gen, expected_seq=info.ack_seq,
        )
        head = coll.status().head
        report["collect_while_slow"] = coll.collect_history(head).to_dict()
        report["retain_while_slow"] = coll.change_retention()["retain_from"]
        slow_rest = []
        cursor = page.next_cursor
        while True:
            page = coll.read_changes_page("slow", 2, cursor_token=cursor)
            slow_rest.extend(
                {"gen": e.gen, "seq": e.seq, "op": e.kind,
                 "before": image(e.before)}
                for e in page.events
            )
            if not page.has_more:
                break
            cursor = page.next_cursor
        report["slow_still_reads"] = slow_rest
        coll.release_subscription("slow")
        report["collect_after_release"] = coll.collect_history(head).to_dict()
        report["retain_after_release"] = coll.change_retention()["retain_from"]

    # 3. A deleted/re-created name rejects the old cursor.  (The fully
    #    acknowledged stream was reclaimed, so publish one more generation to
    #    have a readable page.)
    with open_collection(path) as coll:
        coll.apply(b"g6", head, iter(ops([("put", b"r9", ("z", 1, "new"))])))
    with open_collection(path) as coll:
        coll.begin_subscription("reuse", start_gen=head + 1)
        page = coll.read_changes_page("reuse", 1)
        old_cursor = page.next_cursor
        coll.release_subscription("reuse")
        coll.begin_subscription("reuse", start_gen=head + 1)
        try:
            coll.read_changes_page("reuse", 1, cursor_token=old_cursor)
            report["old_cursor_rejected"] = False
        except Exception as exc:
            report["old_cursor_rejected"] = type(exc).__name__
            report["old_cursor_error"] = str(exc)

    text = json.dumps(report, indent=2, default=lambda b: b.hex())
    if args.report:
        with open(args.report, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    if args.dir is None:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
